import logging
import time
from functools import cache
from typing import Dict, List

import openai
from openai import NOT_GIVEN

from igcs import configs
from igcs.utils.diskcache import disk_cache

last_call = 0

GPT4_MODEL = "gpt-4-turbo-2024-04-09"

logger = logging.getLogger(__name__)


@cache
def get_client():
    return openai.OpenAI()


def prompt2msgs(prompt: str, system_prompt: str) -> List[Dict[str, str]]:
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    if system_prompt is None:
        msgs.pop(0)
    return msgs


def completion(
    prompt: str,
    system: str | None,
    chat_history: list[dict[str, str]] | None = None,
    **kwds,
) -> str | list[str]:
    msgs = prompt2msgs(prompt, system)
    if chat_history:
        msgs = chat_history + msgs
    resp = chat(msgs=msgs, **kwds)

    if len(resp["choices"]) == 1:
        return resp["choices"][0]["message"]["content"]
    return [x["message"]["content"] for x in resp["choices"]]


@disk_cache(cache_dir=configs.CACHE_DIR / "open-ai-chat")
def chat(
    *,
    msgs: List[dict],
    model=GPT4_MODEL,
    temperature: float = 0,
    n: int = NOT_GIVEN,
    max_tokens: int = NOT_GIVEN,
):
    global last_call
    client = get_client()

    # sleep 0.05 secs between requests
    sleep = last_call - time.time() + 0.05
    if sleep > 0:
        print(f"sleeping {sleep:.2f} secs")
        time.sleep(sleep)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
        )
    # simple retry
    except Exception as exc:
        logger.warning(f"got error from OpenAI api - {repr(exc)}")
        time.sleep(60)
        resp = client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=0,
        )
    last_call = time.time()
    return resp.to_dict(mode="json")
