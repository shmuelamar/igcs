import concurrent
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from pathlib import Path

import pandas as pd
from nltk import word_tokenize
from tqdm import tqdm

from igcs import configs

logger = logging.getLogger("igcs.create_gencs_dataset.pg19_books")


def load_sample(fname: str) -> dict:
    with open(fname) as fp:
        content = fp.read()
        sample_id = int(os.path.split(fname)[-1].removesuffix(".txt"))

    return {
        "id": sample_id,
        "content": content,
        "word_count": len(word_tokenize(content)),
    }


def prepare_pg19_books(pg19_books_dir: Path):
    fnames = list(glob(os.path.join(pg19_books_dir, "*.txt")))

    data = []

    with ProcessPoolExecutor(max_workers=None) as pool:
        future_to_fname = {pool.submit(load_sample, fname): fname for fname in fnames}
        for future in tqdm(concurrent.futures.as_completed(future_to_fname), total=len(fnames)):
            data.append(future.result())

    df = pd.DataFrame(data)
    df.sort_values(["id"], inplace=True)
    return df


def main(
    train_dir=configs.RAW_TRAIN_DATASETS_DIR / "deepmind-gutenberg" / "train",
    outfile=configs.RAW_TRAIN_DATASETS_DIR / "pg19_books.jsonl.gz",
):
    df = prepare_pg19_books(train_dir)
    logger.info(f"saving to {outfile}")
    df.to_json(outfile, lines=True, compression="gzip", orient="records")


if __name__ == "__main__":  # pragma: nocover
    main()
