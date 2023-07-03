# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Dict, Optional, Union

import bitsandbytes as bnb
import torch
import transformers
import transformers.models.llama.modeling_llama
import peft
from peft.tuners.lora import LoraLayer


LOGGER = logging.getLogger(__name__)
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
_DEFAULT_VAL_LOCAL_RANK = (
    int(os.environ["LOCAL_RANK"])
    if "LOCAL_RANK" in os.environ else None)


def _find_all_linear_names(bits: int, model, head_name: Optional[str]):
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

    if head_name and head_name in lora_module_names:  # needed for 16-bit
        lora_module_names.remove(head_name)

    return list(lora_module_names)


def _check_is_causal(model_name_or_path, trust_remote_code):
    """ Ensures that the model is causal.

    The QLora code is only with causal models.

    """
    try:
        config = transformers.AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code)
    except OSError:
        LOGGER.warning(
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


def _find_head(model):
    """
    The original code tries to detect the head by checking for the presence
    of "lm_head" in the name of the module, which is again very flimsy. We try
    to be more general. We find the head by assuming that objects created by 
    AutoModelFor* have two modules as children, one 
    transformer.modeling_utils.PreTrainedModel, and the other is the lm_head.
    We make sure that lm_head is of a reasonable type for a lm_head.
    This is a lot more general.
    """

    children = [
        dict(name=name, module=module) 
        for name, module in model.named_children()
    ]

    if len(children) != 2:
        LOGGER.warning(f"Couldn't find the head. Got children: {[x['name'] for x in children]}")
        return None

    assert len(children) == 2, len(children)

    pretrained_model = children[0]
    head = children[1]

    assert isinstance(
        pretrained_model["module"],
        transformers.modeling_utils.PreTrainedModel), (
        type(pretrained_model))
    
    assert isinstance(head["module"], torch.nn.Linear), type(head)

    return head


PEFT_TASK_REGISTRY = {
    transformers.AutoModelForSeq2SeqLM: peft.utils.config.TaskType.SEQ_2_SEQ_LM,
    transformers.AutoModelForCausalLM: peft.utils.config.TaskType.CAUSAL_LM,
    transformers.AutoModelForSequenceClassification: peft.utils.config.TaskType.SEQ_CLS,
    transformers.AutoModelForTokenClassification: peft.utils.config.TaskType.TOKEN_CLS,   
}

def _resolve_hf_model_cls_and_peft_task(
        model_name_or_path: str,
        trust_remote_code: bool, 
        hf_model_cls: Optional[type],
        peft_task: Optional[peft.utils.config.TaskType],
    ) -> tuple[type, peft.utils.config.TaskType]:

    if hf_model_cls is None:
        config = None
        try:
            config = transformers.AutoConfig.from_pretrained(
                model_name_or_path, 
                trust_remote_code=trust_remote_code,)
            
        except OSError:
            LOGGER.warning(
                "The Hugging Face model doesn't have a config.json. "
                "Please supply the `hf_model_cls` explicitly. "
                "We default to `transformers.AutoModelForCausalLM`.")

        if config and vars(config).get("is_encoder_decoder", False):
            hf_model_cls = transformers.AutoModelForSeq2SeqLM
        else:
            hf_model_cls = transformers.AutoModelForCausalLM
    
    if peft_task is None:
        peft_task = PEFT_TASK_REGISTRY.get(hf_model_cls, None)

    assert peft_task, (
        "`peft_task` is None and `hf_model_cls` is not in the default registry, so "
        "we can't infer the task type. Please pass `peft_task` explicitly. "
        f"`peft_task` registry:\n{PEFT_TASK_REGISTRY}"
    )

    return hf_model_cls, peft_task


def from_pretrained(
    model_name_or_path: str,
    hf_model_cls: Optional[type],
    peft_task: Optional[peft.utils.config.TaskType],
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
            No default, needs to be specified.
        hf_model_cls:
            The Hugging Face class to call from_pretrained on, like AutoModelForCausalLM. 
            
            If it is None, it will default to AutModelForCausalLM for a causal model, 
            and AutoModelForSeq2SeqLM for a seq2seq model.

            default: None
        peft_task:
            The peft-config task. Defined at peft.utils.config.TaskType:

                class TaskType(str, enum.Enum):
                    SEQ_CLS = "SEQ_CLS"
                    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
                    CAUSAL_LM = "CAUSAL_LM"
                    TOKEN_CLS = "TOKEN_CLS"
            
            If `hf_model_cls` is AutoModelForCausalLM, we set it to CAUSAL_LM.
            If `hf_model_cls` is AutoModelForSeq2SeqLM, we set it to SEQ_2_SEQ_LM.
            If `hf_model_cls` is AutoModelForSequenceClassification, we set it to SEQ_CLS.
            If `hf_model_cls` is AutoModelForTokenClassification, we set it to TOKEN_CLS.

            Needs to be defined by the user otherwise.
            
            default: None
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
        local_rank: 
            Local rank for distributed training. 
            Default: int(os.environ["LOCAL_RANK"]) if it exists.
    """
    
    hf_model_cls, peft_task = _resolve_hf_model_cls_and_peft_task(
        model_name_or_path=model_name_or_path, 
        trust_remote_code=trust_remote_code,
        hf_model_cls=hf_model_cls,
        peft_task=peft_task,
    )

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

    LOGGER.info(f"loading base model {model_name_or_path}...")
    compute_dtype = (
        torch.float16 
        if fp16 else (
            torch.bfloat16 
            if bf16 else torch.float32
        )
    )

    # JulesGM: This is identical to the original code.
    model: transformers.PreTrainedModel = hf_model_cls.from_pretrained(  # type: ignore
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
    head = _find_head(model)
    # -------------------------------------------------------------------------

    if not full_finetune:
        if checkpoint_dir is not None:
            LOGGER.info("Loading adapters from checkpoint.")
            model = peft.PeftModel.from_pretrained( # type: ignore
                model, 
                os.path.join(checkpoint_dir, "adapter_model"), 
                is_trainable=True,
            )
        else:
            LOGGER.info(f"Adding LoRA modules.")
            modules = _find_all_linear_names(
                bits=bits, 
                model=model, 
                head_name=head["name"] if head else None,
            )
            
            assert modules, f"{modules = }, {bits = }"
            
            config = peft.LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=peft_task,
            )
            
            model = peft.get_peft_model(model, config) # type: ignore

    model: Union[transformers.PreTrainedModel]

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
                (torch.nn.modules.normalization.LayerNorm,
                 transformers.models.llama.modeling_llama.LlamaRMSNorm)
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
        is_head = (head and module is head["module"])
        if (is_head or
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
