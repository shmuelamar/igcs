import logging
from functools import cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from igcs import configs
from igcs.utils.diskcache import disk_cache

logger = logging.getLogger(__name__)

LLAMA3_MODEL_IDS = {
    "instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "base": "meta-llama/Meta-Llama-3-8B",
}


@cache
def get_llama3_model(
    model_name_or_path: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    logger.info(f"Loading model from {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="left",
    )

    # TODO: currently gemma-3 has known bug which sometimes crash
    if "gemma-3" in model_name_or_path.lower():
        from transformers import Gemma3ForCausalLM

        model = Gemma3ForCausalLM(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=bool("phi" in model_name_or_path.lower()),
        )
    logger.info(f"Done Loading model")
    return model, tokenizer


def completion(
    prompt: str,
    system: str | None,
    model: str,
    skip_special_tokens=False,
    chat_history: list[dict] | None = None,
) -> str:
    if system is not None and chat_history:
        raise ValueError("cannot have system and chat_history set together")

    conv = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    if system is None:
        conv.pop(0)

    if chat_history:
        conv = chat_history + conv

    # do not break existing cache
    if skip_special_tokens:
        return generate(conversation=conv, model_name=model)
    return generate(conversation=conv, model_name=model, skip_special_tokens=False)


@disk_cache(cache_dir=configs.CACHE_DIR / "llama_generate")
def generate(
    conversation: list[dict[str, str]],
    model_name: str,
    max_new_tokens: int = 2048,
    skip_special_tokens=True,
) -> str:
    model, tokenizer = get_llama3_model(model_name)
    if "phi" in model_name.lower() or "mistral" in model_name.lower():
        terminators = [tokenizer.eos_token_id]
    elif "gemma-3" in model_name.lower():
        terminators = [tokenizer.encode("<end_of_turn>", add_special_tokens=False)[0]]
    elif "qwen2.5" in model_name.lower():
        terminators = [tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]]
    elif "llama-3" in model_name.lower():
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
    elif "smollm2" in model_name.lower():
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        ]
    else:
        raise ValueError(f"{model_name} is not supported")
    logger.info(f"predicting conversation")

    logger.debug(f"conversation: {conversation}")

    # tokenize
    input_ids = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    # generate
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=False,
    )
    response = outputs[0][input_ids.shape[-1] :]
    if not skip_special_tokens and len(response) and response[-1] in terminators:
        response = response[:-1]

    response_text = tokenizer.decode(response, skip_special_tokens=skip_special_tokens)
    logger.debug(f"model completion: {response_text}")
    return response_text
