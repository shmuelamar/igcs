from pathlib import Path

from datasets import load_dataset

from igcs.configs import PUBMED_FNAME
from igcs.utils import ioutils


def prepare_pubmed_abstracts(sample_size: int):
    ds = load_dataset("pubmed", trust_remote_code=True)
    return ds["train"].shuffle(seed=42).select(range(sample_size))


def main(sample_size: int = 100_000, outfile: Path = PUBMED_FNAME):
    samples = prepare_pubmed_abstracts(sample_size)
    ioutils.jsonl_dump(samples, outfile)


if __name__ == "__main__":  # pragma: nocover
    main()
