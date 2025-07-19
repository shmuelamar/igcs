from pathlib import Path

import pandas as pd
from datasets import load_dataset

from igcs.configs import WIKI_FNAME

WIKIPEDIA_DUMP_VERSION = "20240420"


def prepare_wikipedia_pages(sample_size: int):
    ds = load_dataset(
        "wikipedia",
        language="en",
        date=WIKIPEDIA_DUMP_VERSION,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    return list(ds.take(sample_size))


def main(sample_size: int = 100_000, outfile: Path = WIKI_FNAME):
    data = prepare_wikipedia_pages(sample_size)
    pd.DataFrame(data).to_json(outfile, lines=True, compression="gzip", orient="records")


if __name__ == "__main__":  # pragma: nocover
    main()
