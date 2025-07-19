import numpy as np

from igcs.datasets import load_dataset
from igcs.datasets.create_gencs_dataset.create_reverse_instructions_dataset_splits import (
    get_sample_ila,
)


def main():
    aspectnews = [
        x for split in ["train", "dev", "test"] for x in load_dataset("AspectNews", split)
    ]
    per_sample_iaa = [get_sample_ila(sample.selections, sample.docs)[0] for sample in aspectnews]
    print(np.mean(per_sample_iaa))


if __name__ == "__main__":
    main()
