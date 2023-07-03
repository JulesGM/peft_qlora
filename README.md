# Only HuggingFace PEFT-QLoRA

QLora from https://github.com/artidoro/qlora/ is currently only available as part of a training script.

We split it from the script & made it its own library with the same defaults to allow for easier use in other projects.

# How to use:

Create your model using `peft_lora.from_pretrained`, then use it like a normal [Peft](https://github.com/huggingface/peft)/Huggingface model, like [in this example](https://github.com/huggingface/peft/blob/main/examples/causal_language_modeling/peft_lora_clm_accelerate_big_model_inference.ipynb).

Very slightly modified from the original `qlora/qlora.get_accelerate_model` to add the arguments explicitly and the defaults from the arg config.

**Note:** Only tested on causal models. There is a way to test it on seq2seq models, however: pass `ignore_is_causal_check=True` to `peft_qlora.from_pretrained`, and the `model_name_or_path` of an encoder-decoder, like `google/flan-t5-xxl`.

## Example:

```python

model = peft_lora.from_pretrained("EleutherAI/pythia-12b", bf16=True)

# < .. use the model like you would with any other peft model: https://github.com/huggingface/peft >

```

## peft_lora.from_pretrained:

**model_name_or_path:**

Huggingface auto model from_pretrained name or path argument.

No default, needs to be specified.

**hf_model_cls:**
The Hugging Face class to call from_pretrained on, like AutoModelForCausalLM. 

If it is None, it will default to AutModelForCausalLM for a causal model, 
and AutoModelForSeq2SeqLM for a seq2seq model.

default: `None`

**peft_task:**
The peft-config task. Defined at peft.utils.config.TaskType:

```python
        class TaskType(str, enum.Enum):
            SEQ_CLS = "SEQ_CLS"
            SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
            CAUSAL_LM = "CAUSAL_LM"
            TOKEN_CLS = "TOKEN_CLS"
```

If `hf_model_cls` is `AutoModelForCausalLM`, we set it to `CAUSAL_LM`.
If `hf_model_cls` is `AutoModelForSeq2SeqLM`, we set it to `SEQ_2_SEQ_LM`.
If `hf_model_cls` is `AutoModelForTokenClassification`, we set it to `TOKEN_CLS`.
If `hf_model_cls` is `AutoModelForSequenceClassification`, we set it to `SEQ_CLS`.

Needs to be defined by the user otherwise.

Default: None

**bf16:** Whether to use bf16.

**fp16:** Whether to use fp16.

**cache_dir:** Huggingface caching dir.

**checkpoint_dir:** Huggingface checkpoint dir.

**max_memory_MB:** Max gpu memory to use in Megabytes.

**full_finetune:** Finetune the entire model without adapters.

**gradient_checkpointing:** Use gradient checkpointing. You want to use this.

**bits:** How many bits to use.

**quant_type:** Quantization data type to use. Should be one of `fp4` or `nf4`.

**double_quant:** Compress the quantization statistics through double quantization.

**trust_remote_code:** Enable unpickling of arbitrary code in AutoModelForCausalLM.from_pretrained.

**use_auth_token:** Enables using Huggingface auth token from Git Credentials.

**lora_r:** Lora R dimension.

**lora_alpha:** Lora alpha.

**lora_dropout:** Lora dropout.
