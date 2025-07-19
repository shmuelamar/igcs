import logging
import time
from typing import Dict, List

import anthropic
from anthropic import NOT_GIVEN

from igcs import configs
from igcs.utils.diskcache import disk_cache

logger = logging.getLogger(__name__)

# defaults to os.environ.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic()

CLAUDE_MODEL = "claude-3-opus-20240229"


def prompt2msgs(prompt: str) -> List[Dict]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ],
        }
    ]


def completion(prompt: str, system: str) -> str:
    msgs = prompt2msgs(prompt)
    resp = chat(msgs=msgs, system=system)
    if resp["stop_reason"] != "end_turn":
        logger.warning(f'claude - got stop_reason {resp["stop_reason"]}')

    return resp["content"][0]["text"]


@disk_cache(cache_dir=configs.CACHE_DIR / "claude-chat")
def chat(
    *,
    msgs: List[Dict],
    system: str | None = None,
    model: str = CLAUDE_MODEL,
    max_retries: int = 5,
):
    error = None
    for retry in range(max_retries):
        try:
            message = client.messages.create(
                model=model,
                system=system or NOT_GIVEN,
                max_tokens=4096,
                temperature=0,
                messages=msgs,
            )
            return message.to_dict(mode="json")
        except anthropic.BadRequestError as exc:
            error = exc
            if (
                exc.status_code == 400
                and exc.response.json()["error"]["message"]
                == "Output blocked by content filtering policy"
            ):
                return {
                    "content": [{"text": f"<error {exc!r}>"}],
                    "stop_reason": "error",
                }
        except anthropic.InternalServerError as exc:
            error = exc
            if exc.status_code == 529 and exc.response.json()["error"]["message"] == "Overloaded":
                logger.warning(f"claude - got 529 Overloaded - retry #{retry}/5")
                time.sleep(10 + 2**retry)
            else:
                raise
    raise ValueError("Max retries exceeded") from error


if __name__ == "__main__":  # pragma: nocover
    print(chat(msgs=prompt2msgs("say hi")))
