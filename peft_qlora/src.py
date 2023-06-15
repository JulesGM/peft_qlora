# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Dict, Optional, Sequence

import bitsandbytes as bnb
import torch
import transformers
import peft
from peft.tuners.lora import LoraLayer


logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
_DEFAULT_VAL_LOCAL_RANK = (
    int(os.environ["LOCAL_RANK"]) 
    if "LOCAL_RANK" in os.environ else None
)

def _find_all_linear_names(bits: int, model):
    """ Finds all the linear layer names in the model.

    This is to pass them as targets for LORA.

    Node that this doesn't work at all with GPT2 as it 
    uses 1D convs instead of linear layers.
    
    Model can possibly quantized, but it's not necessary.
    The lora targets need to be found, whether the model 
    is quantized or not.

    Args:
        bits:
            How many bits to use. 4, 8, 16, 32
        model:
            The possibly quantized huggingface model.

    """


    cls = (
        bnb.nn.Linear4bit
        if bits == 4
        else (
            bnb.nn.Linear8bitLt 
            if bits == 8 
            else torch.nn.Linear
        )
    )

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(
                names[0] 
                if len(names) == 1 
                else names[-1]
            )

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")

    return list(lora_module_names)


def _check_is_causal(model_name_or_path: str):
    """ Ensures that the model is causal.

    The QLora code is only with causal models.

    """
    try:
        config = transformers.AutoConfig.from_pretrained(
            model_name_or_path)
    except OSError as e:
        return

    if vars(config).get("is_encoder_decoder", False):
        raise ValueError(
            "We haven't tested the code with encoder-decoder models yet. "
            f"Pass ignore_is_causal_check=True to "
            "`peft_qlora.from_pretrained` to ignore this error, "
            "but do so at your own risk."
        )


def from_pretrained(
    model_name_or_path: str,
    fp16: bool = False,
    bf16: bool = True,
    max_memory_MB: Optional[int] = None,
    cache_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    full_finetune: bool = False,
    gradient_checkpointing: bool =True,
    bits: int = 4,
    quant_type: str = "nf4",
    double_quant: bool = True,
    trust_remote_code: bool = False,
    use_auth_token: bool = False,
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    ignore_is_causal_check: bool = False,
    local_rank: Optional[int] = _DEFAULT_VAL_LOCAL_RANK,
):
    """Only public function of this library.

    Creates your model with QLora, using Peft and Bitsandbytes, 
    but also finding the all the possible linear layers instead of
    just k and v like in the regular Lora code.

    You can then use the model like you would a normal HuggingFace Peft Model.

    Very slightly modified from the original 
    qlora/qlora.get_accelerate_model to add the arguments and the defaults.
    
    Args:
        model_name_or_path: 
            Huggingface auto model from_pretrained name or path argument.
            No default.
        bf16: 
            Whether to use bf16.
            Default: True.
        fp16: 
            Whether to use fp16.
            Default: False.
        cache_dir: 
            Huggingface caching dir.
            Default: None.
        checkpoint_dir: 
            Huggingface checkpoint dir.
            Default: None.
        max_memory_MB: 
            Max gpu memory to use in Megabytes.
            Default: None.
        full_finetune: 
            Finetune the entire model without adapters.
            Default: False.
        gradient_checkpointing: 
            Use gradient checkpointing. You want to use this.
            Default: True.
        bits: 
            How many bits to use.
            Default: 4.
        quant_type: 
            Quantization data type to use. Should be one of `fp4` or `nf4`.
            Default: `nf4`.
        double_quant: 
            Compress the quantization statistics through double quantization.
            Default: True.
        trust_remote_code: 
            Enable unpickling of arbitrary code in AutoModelForCausalLM.from_pretrained.
            Default: False.
        use_auth_token: 
            Enables using Huggingface auth token from Git Credentials.
            Default: False.
        lora_r: 
            Lora R dimension.
            Default: 64.
        lora_alpha: 
            Lora alpha.
            Default: 16.
        lora_dropout: 
            Lora dropout.
            Default: 0.0.
        ignore_is_causal_check: 
            We added this. This is if you want to try using an encoder decoder. It's untested.
            Default: False.
        local_rank: 
            Local rank for distributed training. 
            Default: int(os.environ["LOCAL_RANK"]) if it exists.
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
        max_memory = None

    device_map = "auto"

    # if we are in a distributed setting, 
    # we need to set the device map and max memory per device
    if local_rank is not None:
        device_map = {"": local_rank}
        max_memory = (
            {"": max_memory[local_rank]} 
            if max_memory else None
        )

    if full_finetune:
        assert bits in [16, 32]

    logger.info(f"loading base model {model_name_or_path}...")
    compute_dtype = (
        torch.float16 
        if fp16 else (
            torch.bfloat16 
            if bf16 else torch.float32
        )
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
            torch.float32 
            if fp16 else (
                torch.bfloat16 
                if bf16 else torch.float32
            )
        ),
        trust_remote_code=trust_remote_code,
        use_auth_token=use_auth_token,
    )
    if compute_dtype == torch.float16 and bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print(
                "Your GPU supports bfloat16, "
                "you can accelerate training with "
                "the argument --bf16"
            )
            print("=" * 80)

    setattr(model, "model_parallel", True)
    setattr(model, "is_parallelizable", True)

    model.config.torch_dtype = (
        torch.float32 
        if fp16 else (
            torch.bfloat16 
            if bf16 else torch.float32
        )
    )

    if not full_finetune:
        model = peft.prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if not full_finetune:
        if checkpoint_dir is not None:
            logger.info("Loading adapters from checkpoint.")
            model = peft.PeftModel.from_pretrained(
                model, 
                os.path.join(checkpoint_dir, "adapter_model"), 
                is_trainable=True,
            )
        else:
            logger.info(f"Adding LoRA modules.")
            modules = _find_all_linear_names(
                bits=bits, model=model)
            
            assert modules, f"{modules = }, {bits = }"
            
            config = peft.LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",)
            
            model = peft.get_peft_model(model, config)

    fp32_weights = []
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if bf16:
                module = module.to(torch.bfloat16)

        if (
            "norm" in name or 
            isinstance(
                module, 
                torch.nn.modules.normalization.LayerNorm
            )):
            
            # <JULESGM FIX>
            # The original code doesn't cast the layer norm to bfloat16,
            # Afaik this just doesn't run for me. It should be ok like
            # this.
            if bf16:
                module = module.to(torch.bfloat16)
            # </JULESGM FIX>

            else:
                module = module.to(torch.float32)

        if (
            "lm_head" in name or 
            "embed" in name or
            isinstance(module, torch.nn.Embedding)
        ):
            if hasattr(module, "weight"):
                if bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

        if bf16 or fp16:
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    fp32_weights.append((name, module.weight.dtype, type(module)))

    assert not bf16 or not fp32_weights, (
        f"Found fp32 weights in {fp32_weights}. "
    )

    return model
