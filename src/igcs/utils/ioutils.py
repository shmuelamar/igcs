import gzip
import json
from pathlib import Path
from typing import Iterable, Literal


def jsonl_load(fname: str | Path) -> Iterable:
    with _get_opener(fname) as fp:
        for line in fp:
            yield json.loads(line)


def jsonl_dump(data: Iterable, fname: str | Path):
    with _get_opener(fname, "w") as fp:
        for obj in data:
            json.dump(obj, fp)
            fp.write("\n")


def _get_opener(fname: str | Path, mode: Literal["w", "r"] = "r"):
    if str(fname).endswith(".gz"):
        return gzip.open(fname, mode=mode + "t")
    return open(fname, mode=mode)
