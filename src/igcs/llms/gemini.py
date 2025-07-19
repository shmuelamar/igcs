import logging
import os
import time
from functools import cache
from typing import Dict, List

import google.generativeai as genai
from google import generativeai

from igcs import configs
from igcs.utils import log
from igcs.utils.diskcache import disk_cache

GEMINI_MODEL = "gemini-1.5-pro-latest"
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

logger = logging.getLogger(__name__)


@cache
def get_model(model_name: str, system_prompt: str) -> genai.GenerativeModel:
    return genai.GenerativeModel(
        model_name,
        system_instruction=system_prompt,
        safety_settings=[
            {
                "category": cat,
                "threshold": "BLOCK_NONE",
            }
            for cat in [
                "HARM_CATEGORY_SEXUAL",
                "HARM_CATEGORY_DANGEROUS",
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            ]
        ],
        generation_config={
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 8192,
        },
    )


def prompt2msgs(prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "user", "parts": [prompt]},
    ]


def completion(prompt: str, system: str) -> str:
    msgs = prompt2msgs(prompt)
    return chat(msgs=msgs, system=system)


@disk_cache(cache_dir=configs.CACHE_DIR / "gemini")
def chat(*, msgs: List[dict], system: str, model=GEMINI_MODEL) -> str:
    api = get_model(model, system)
    history = msgs[:-1]
    convo = api.start_chat(history=history)

    exc = None
    retries = 10000
    retry = 0
    while retry < retries:
        retry += 1
        try:
            resp = convo.send_message(msgs[-1], stream=True)
            parts = []
            for chunk in resp:
                parts.append(chunk.text)
            return "".join(parts)
        except generativeai.types.generation_types.BlockedPromptException:
            return f"<error {exc!r} retries={retries}>"
        except Exception as e:
            logger.warning(f"error in gemini retry #{retry}", exc_info=True)
            time.sleep(10)
            exc = e
            if "429 Resource has been exhausted" in repr(e):
                if retries == 0:
                    raise RuntimeError(f"Too many Retries - {retries}")
                retry -= 1

            if "Can not build a coherent char history after a" in str(
                exc
            ) or "BlockedPromptException: block_reason: OTHER" in str(exc):
                return f"<error {exc!r} retries={retries}>"

    if "Can not build a coherent char history after a" not in str(
        exc
    ) and "BlockedPromptException: block_reason: OTHER" not in str(exc):
        raise exc
    return f"<error {exc!r} retries={retries}>"


if __name__ == "__main__":  # pragma: nocover
    log.init()
    msgs = prompt2msgs("say hi to me and welcome")
    res = chat(msgs=msgs, system="you are a mighty pirate that talks in pirate style")
    print(res)
