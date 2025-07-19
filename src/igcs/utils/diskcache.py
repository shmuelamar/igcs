import base64
import functools
import hashlib
import json
import logging
import os
from glob import glob
from typing import Literal

import numpy as np

from igcs import configs
from igcs.utils import log

logger = logging.getLogger(__name__)


def disk_cache(cache_dir: str, ext: Literal["json", "npy"] = "json"):
    """Caches function output to disk"""
    os.makedirs(cache_dir, exist_ok=True)

    def _get_key(kwargs: dict) -> str:
        kwargs_cache = hashlib.sha1(json.dumps(kwargs, sort_keys=True).encode()).digest()
        name = base64.b64encode(kwargs_cache).decode()
        name = name.strip("=")
        name = name.replace("+", "-").replace("/", "_")
        return os.path.join(cache_dir, f"{name}.{ext}")

    def wrapper(f):
        @functools.wraps(f)
        def wrapped(**kwargs):
            cache_fname = _get_key(kwargs)
            if os.path.exists(cache_fname):
                logger.info(f"reading from cache {cache_fname}")

                if ext == "json":
                    with open(cache_fname) as fp:
                        return json.load(fp)
                elif ext == "npy":
                    np.load(cache_fname)
                else:
                    raise ValueError(ext)

            result = f(**kwargs)

            if ext == "json":
                with open(cache_fname, "w") as fp:
                    json.dump(result, fp)
            elif ext == "npy":
                np.save(cache_fname, result)
            else:
                raise ValueError(ext)

            return result

        return wrapped

    return wrapper


def clear_cache(cache_dir, pattern):
    for fname in glob(os.path.join(cache_dir, "*.json")):
        with open(fname) as fp:
            data = json.load(fp)

        if isinstance(data, str) and data.startswith("<error"):
            logger.info(f"error data - {data!r}")
        elif isinstance(data, dict) and data.get("content", [{}])[0].get("text", "").startswith(
            "<error"
        ):
            logger.info(f"error data - {data!r}")
        if data == pattern:
            logger.info(f"clearing cache of file {fname} with data={data!r}")
            os.remove(fname)


if __name__ == "__main__":
    log.init()
    clear_cache(
        configs.CACHE_DIR / "gemini",
        # """<error BrokenResponseError('Can not build a coherent char history after a broken streaming response (See the previous Exception fro details). To inspect the last response object, use `chat.last`.To remove the last request/response `Content` objects from the chat call `last_send, last_received = chat.rewind()` and continue without it.') retries=3>""",
        "<error ResourceExhausted('Resource has been exhausted (e.g. check quota).') retries=3>",
    )
