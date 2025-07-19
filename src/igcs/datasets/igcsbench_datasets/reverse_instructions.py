import logging
import pathlib
from typing import Dict, Iterable

from igcs.configs import RI_DATASET_DIR
from igcs.datasets.igcsbench_datasets.base import SourceDataset
from igcs.entities import SelectionSample
from igcs.utils import ioutils, log

logger = logging.getLogger(__name__)


class ReverseInstructionsDataset(SourceDataset):
    name = "ReverseInstructions"
    splits_fnames: Dict[str, str] = {
        split: f"{split}.jsonl.gz" for split in ["train", "dev", "test"]
    }
    dirname: pathlib.Path = RI_DATASET_DIR

    def prepare_file(self, fname: pathlib.Path, split: str) -> Iterable[SelectionSample]:
        for row in ioutils.jsonl_load(fname):
            sample = SelectionSample(**row)
            sample.id = f"ri.{sample.id}"  # add ri prefix for clarity
            yield sample


if __name__ == "__main__":
    log.init()
    dataset = ReverseInstructionsDataset().prepare_dataset()
    s = dataset["dev"][0]
    print(s)
    print({split: len(data) for split, data in dataset.items()})
