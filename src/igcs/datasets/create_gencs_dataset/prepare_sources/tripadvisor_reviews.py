import gzip
import json
import logging
import os
from glob import glob
from pathlib import Path
from typing import List

from tqdm import tqdm

from igcs import configs

logger = logging.getLogger("igcs.create_gencs_dataset.tripadvisor_reviews")


def prepare_tripadvisor_reviews(trip_advisor_dir: Path) -> List[dict]:
    fnames = list(glob(os.path.join(trip_advisor_dir, "*.json")))
    data = []
    logger.info(f"found {len(fnames)} tripadvisor reviews")

    for fname in tqdm(fnames):
        with open(fname) as fp:
            data.append(json.load(fp))

    return data


def main(
    train_dir=configs.RAW_TRAIN_DATASETS_DIR / "TripAdvisor",
    outfile=configs.RAW_TRAIN_DATASETS_DIR / "trip_advisor.jsonl.gz",
):
    # parse Enron emails into a file later to be sampled as required
    data = prepare_tripadvisor_reviews(train_dir)
    logger.info(f"saving to {outfile}")

    with gzip.open(outfile, "wt") as fp:
        for row in tqdm(data):
            fp.write(json.dumps(row) + "\n")
    logger.info("done")
