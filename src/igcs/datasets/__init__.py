from pathlib import Path

from igcs import configs
from igcs.datasets.igcsbench_datasets import DATASETS
from igcs.entities import SelectionSample
from igcs.utils import ioutils


def load_dataset(dataset: str, split_name: str) -> list[SelectionSample]:
    dataset_fname = get_dataset_file(f"{dataset}/{split_name}")[0]
    return [SelectionSample(**d) for d in ioutils.jsonl_load(dataset_fname)]


def get_dataset_file(
    dataset: str | Path, is_prompt: bool = False, verify_exist: bool = True
) -> tuple[Path, str | None, str | None]:
    dataset = str(dataset)
    if is_prompt:
        basedir = configs.PROMPTS_DIR
        prompts_prefix = "prompts-"
    else:
        basedir = configs.IGCS_DATA_DIR
        prompts_prefix = ""

    # check for predefined dataset
    valid_splits = ["train", "dev", "test"]
    if any(dataset.endswith(f"/{split}") for split in valid_splits) and dataset.count("/") == 1:
        dataset, split = dataset.split("/")

        # check for all dataset
        if dataset.lower() == "all":
            return basedir / f"{prompts_prefix}{split}-all.jsonl.gz", dataset, split

        ds2name = {ds.name.lower(): ds.name for ds in DATASETS}
        ds_name = ds2name.get(dataset.lower())

        # check for formal dataset name
        if ds_name:
            return (
                basedir / split / f"{prompts_prefix}{split}-{ds_name}.jsonl.gz",
                dataset,
                split,
            )

    # check for existing file path
    filename = Path(dataset)
    if verify_exist and not filename.exists():
        raise ValueError(f"filename {filename} doesnt exist")
    return filename, None, None
