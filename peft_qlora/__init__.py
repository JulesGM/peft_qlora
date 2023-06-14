# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Dict, Optional, Sequence

import bitsandbytes as bnb
import torch
import transformers
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from peft.tuners.lora import LoraLayer

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

def _find_all_linear_names(bits, model):
    cls = (
        bnb.nn.Linear4bit
        if bits == 4
        else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")

    return list(lora_module_names)


def _check_is_causal(model_name_or_path: str):
    try:
        config = transformers.AutoConfig.from_pretrained(
            model_name_or_path)
    except OSError as e:
        return

    if vars(config).get("is_encoder_decoder", False):
        raise ValueError(
            "We haven't tested the code with encoder-decoder models yet. "
            f"Pass ignore_is_causal_check=True to `peft_qlora.from_pretrained` to ignore this error, "
            "but do so at your own risk."
        )


def from_pretrained(
    model_name_or_path: str,
    fp16: bool = False,
    bf16: bool = True,
    max_memory_MB: Optional[int] = None,
    cache_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    full_finetune=False,
    gradient_checkpointing=True,
    bits: int = 4,
    quant_type: str = "nf4",
    double_quant: bool = True,
    trust_remote_code: bool = False,
    use_auth_token: bool = False,
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    ignore_is_causal_check: bool = False,
):
    """
    Main function of this library.

    Creates your model with QLora. You can
    then use it like a normal HuggingFace Peft Model.

    Very slightly modified from the original 
    qlora/qlora.get_accelerate_model to add the arguments and the defaults.
    
    Args:
        model_name_or_path: Huggingface auto model from_pretrained name or path argument.
        bf16: Whether to use bf16.
        fp16: Whether to use fp16.
        cache_dir: Huggingface caching dir.
        checkpoint_dir: Huggingface checkpoint dir.
        max_memory_MB: Max gpu memory to use in Megabytes.
        full_finetune: Finetune the entire model without adapters.
        gradient_checkpointing: Use gradient checkpointing. You want to use this.
        bits: How many bits to use.
        quant_type: Quantization data type to use. Should be one of `fp4` or `nf4`.
        double_quant: Compress the quantization statistics through double quantization.
        trust_remote_code: Enable unpickling of arbitrary code in AutoModelForCausalLM.from_pretrained.
        use_auth_token: Enables using Huggingface auth token from Git Credentials.
        lora_r: Lora R dimension.
        lora_alpha: Lora alpha.
        lora_dropout: Lora dropout.
        ignore_is_causal_check: We added this. This is if you want to try using an encoder decoder. It's untested.
    """

    cls = transformers.AutoModelForCausalLM
    if ignore_is_causal_check:
        config = transformers.AutoConfig.from_pretrained(model_name_or_path)
        if vars(config).get("is_encoder_decoder", False):
            cls = transformers.AutoModelForSeq2SeqLM
    else:
        _check_is_causal(model_name_or_path)

    if fp16 and bf16:
        raise ValueError("Can't use both fp16 and bf16")

    assert bits in [4, 8, 16, 32], (
        f"bits must be one of 4, 8, 16, 32, got {bits}")

    n_gpus = torch.cuda.device_count()
    
    if max_memory_MB:
        max_memory = f"{max_memory_MB}MB"
        max_memory = {i: max_memory for i in range(n_gpus)}
    else:
        max_memory is None

    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}
        max_memory = {"": max_memory[local_rank]}

    if full_finetune:
        assert bits in [16, 32]

    print(f"loading base model {model_name_or_path}...")
    compute_dtype = (
        torch.float16 if fp16 else (torch.bfloat16 if bf16 else torch.float32)
    )
    model = cls.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        load_in_4bit=bits == 4,
        load_in_8bit=bits == 8,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=transformers.BitsAndBytesConfig(
            load_in_4bit=bits == 4,
            load_in_8bit=bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=double_quant,
            bnb_4bit_quant_type=quant_type,
        ),
        torch_dtype=(
            torch.float32 if fp16 else (torch.bfloat16 if bf16 else torch.float32)
        ),
        trust_remote_code=trust_remote_code,
        use_auth_token=use_auth_token,
    )
    if compute_dtype == torch.float16 and bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print(
                "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
            )
            print("=" * 80)

    setattr(model, "model_parallel", True)
    setattr(model, "is_parallelizable", True)

    model.config.torch_dtype = (
        torch.float32 if fp16 else (torch.bfloat16 if bf16 else torch.float32)
    )

    if not full_finetune:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if not full_finetune:
        if checkpoint_dir is not None:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(
                model, os.path.join(checkpoint_dir, "adapter_model"), is_trainable=True
            )
        else:
            print(f"adding LoRA modules...")
            modules = _find_all_linear_names(bits, model)
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    return model
