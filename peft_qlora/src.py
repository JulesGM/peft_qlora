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

def _find_all_linear_names(bits: int, model, lm_head_name: str):
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

    if lm_head_name in lora_module_names:  # needed for 16-bit
        lora_module_names.remove(lm_head_name)

    return list(lora_module_names)


def _check_is_causal(model_name_or_path, trust_remote_code):
    """ Ensures that the model is causal.

    The QLora code is only with causal models.

    """
    try:
        config = transformers.AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code)
    except OSError:
        logger.warning(
            "The model doesn't have a config.json. "
            "We assume that it's causal. Use --force_seq2seq to override."
        )
        return

    if vars(config).get("is_encoder_decoder", False):
        raise ValueError(
            "We haven't tested the code with encoder-decoder models yet. "
            f"Pass ignore_is_causal_check=True to "
            "`peft_qlora.from_pretrained` to ignore this error, "
            "but do so at your own risk."
        )


def _find_lm_head(model):
    """
    The original code tries to detect the lm head by checking for the presence
    of "lm_head" in the name of the module, which is again very flimsy. We try
    to be more general. We find the lm_head by assuming that objects created by 
    AutoModelForCausalLM have two modules as children, one 
    transformer.modeling_utils.PreTrainedModel, and the other is the lm_head.
    We make sure that lm_head is of a reasonable type for a lm_head.
    This is a lot more general.
    """
    children = [
        dict(name=name, module=module) 
        for name, module in model.named_children()
    ]
    assert len(children) == 2, len(children)

    pretrained_model = children[0]
    lm_head = children[1]

    assert isinstance(
        pretrained_model["module"],
        transformers.modeling_utils.PreTrainedModel), (
        type(pretrained_model))
    
    assert isinstance(lm_head["module"], torch.nn.Linear), type(lm_head)

    return lm_head



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
    local_rank: Optional[int] = _DEFAULT_VAL_LOCAL_RANK,
    ignore_is_causal_check: bool = False,
    force_seq2seq: bool = False,
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

    # -------------------------------------------------------------------------
    # JulesGM: We added those checks, as well as `experimental`
    # support for encoder-decoder models. It should work out of the box,
    # but we added warnings & requiring turning of "ignore_is_causal_check"
    # because it's not in the original code.
    # -------------------------------------------------------------------------
    cls = transformers.AutoModelForCausalLM
    if force_seq2seq:
        cls = transformers.AutoModelForSeq2SeqLM
        logger.warning(
            "Seq2SeqLMs support is experimental. Use at your own risk."
        )
    elif ignore_is_causal_check:
        try:
            config = transformers.AutoConfig.from_pretrained(
                model_name_or_path, 
                trust_remote_code=trust_remote_code,
            )
            if vars(config).get("is_encoder_decoder", False):
                logger.warning(
                    "Encoder-decoder models are untested with this library.")
                cls = transformers.AutoModelForSeq2SeqLM
        except OSError:
            # This model doesn't have a config.json file, so we can't check
            # if it's an encoder-decoder model. 
            logger.warning(
                    "This model doesn't have a config.json file, "
                    "so we can't check if it's an encoder-decoder model. "
                    "Defaulting to causal. Use --force_seq2seq if you wanted "
                    "a causal model."
                )
    else:
        _check_is_causal(model_name_or_path, trust_remote_code)

    if fp16 and bf16:
        raise ValueError("Can't use both fp16 and bf16")

    assert bits in [4, 8, 16, 32], (
        f"bits must be one of 4, 8, 16, 32, got {bits = }")
    

    # -------------------------------------------------------------------------
    # JulesGM: We added support for max_memory = None, so it doesn't
    # automatically overflow to cpu offloading, which is slow and should not
    # happen silently. 
    # -------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------


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

    # JulesGM: This is identical to the original code.
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


    # -------------------------------------------------------------------------
    # JulesGM:
    # The original code modifies the lm_head by the name specific to type of
    # model they were fine-tuning, which is really very flimsy.
    #
    # We try to be more general.
    # -------------------------------------------------------------------------
    lm_head = _find_lm_head(model)
    # -------------------------------------------------------------------------

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
                bits=bits, 
                model=model, 
                lm_head_name=lm_head["name"],
            )
            
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

        # -------------------------------------------------------------------------
        # JulesGM:
        # The original code finds layer norms by the presence of "name"
        # in the names of the modules, which is again very flimsy.
        # We try to be more general by checking the type.
        # -------------------------------------------------------------------------
        if isinstance(
                module, 
                torch.nn.modules.normalization.LayerNorm
        ):    
            # -------------------------------------------------------------------------
            # JulesGM - FIX
            # The original code doesn't cast the layer norms to bfloat16, but to float32,
            # but that just didn't run for me at all.
            #
            # The idea from the rest of the code is to cast
            # non low bytes layers to bfloat16 in bf16 mode, and to float32 in fp16 mode.
            # So we changed it to cast layer norm layers to bfloat16 in bf16 mode, and left 
            # it to float32 in in other modes, and it works.
            #
            # This is the only somewhat significant change to the original code, but feels pretty
            # reasonable, and the model trains perfectly fine, and doesn't work otherwise.
            if bf16:
                module = module.to(torch.bfloat16)
            # -------------------------------------------------------------------------

            else:
                module = module.to(torch.float32)

        # -------------------------------------------------------------------------
        # JulesGM:
        # The original code tries to find the embedding by looking at "embed"
        # in the name of the module, which is again very flimsy. We try to be more
        # general by checking the type.
        # 
        # We also use our more general method of detecting the lm_head
        # -------------------------------------------------------------------------
        if (
            module is lm_head["module"] or
            isinstance(module, torch.nn.Embedding)  
        ):
            if hasattr(module, "weight"):
                if bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

        if bf16:
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    fp32_weights.append((name, module.weight.dtype, type(module)))

    assert not bf16 or not fp32_weights, (
        f"Found fp32 weights in {fp32_weights}. "
    )

    return model
