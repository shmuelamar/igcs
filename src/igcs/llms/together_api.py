import functools
import logging
import time
from functools import partial
from typing import Dict, List

from together import Together
from together.error import InvalidRequestError

from igcs import configs
from igcs.utils.diskcache import disk_cache

LLAMA_70B = "meta-llama/Llama-3-70b-chat-hf"
LLAMA_405B_FP8 = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"

logger = logging.getLogger(__name__)


@functools.cache
def get_client():
    # api key env var is TOGETHER_API_KEY
    return Together()


def prompt2msgs(prompt: str, system_prompt: str | None) -> List[Dict]:
    msgs = []
    if system_prompt is not None:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def completion(prompt: str, system: str | None, model: str) -> str:
    msgs = prompt2msgs(prompt, system)
    logger.info(f"calling {model} with msgs: {repr(msgs)[:100]}...")
    resp = chat(msgs=msgs, model=model)
    return resp["choices"][0]["message"]["content"]


completion_llama_70b = partial(completion, model=LLAMA_70B)
completion_llama_405b = partial(completion, model=LLAMA_405B_FP8)


@disk_cache(cache_dir=configs.CACHE_DIR / "together-chat")
def chat(
    *,
    msgs: List[Dict],
    model: str,
):
    client = get_client()
    time.sleep(5.0)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=msgs,
            top_p=0,
            top_k=0,
            temperature=0,
        )
    except InvalidRequestError as exc:
        if exc.http_status == 400:
            logger.warning(f"error: {exc}")
            return {"choices": [{"message": {"content": f"<error {exc}>"}}]}
        raise
    return resp.model_dump()


if __name__ == "__main__":
    print(chat(msgs=prompt2msgs("say hi", system_prompt=None)))
    print(completion_llama_70b("say hi", None))
