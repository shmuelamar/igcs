import os

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from igcs import configs

# from - https://huggingface.co/datasets/codeparrot/github-code#languages
# Idea - select programming languages (not config or markup langs) with permissive license.
SELECTED_LANGS = frozenset(
    {
        "Assembly",
        "C",
        "C#",
        "C++",
        "GO",
        "Java",
        "JavaScript",
        "PHP",
        "Perl",
        "Python",
        "Ruby",
        "Rust",
        "Scala",
        "Shell",
        "TypeScript",
    }
)
SELECTED_LICENSES = frozenset({"mit", "apache-2.0"})


def load_raw_data(sample_size: int):
    data = []
    ds = load_dataset(
        "codeparrot/github-code-clean",
        streaming=True,
        split="train",
        trust_remote_code=True,
    )

    progress = tqdm(total=sample_size)
    while True:
        for batch in ds.iter(100_000):
            batch_df = pd.DataFrame(batch)
            batch_df = batch_df[
                batch_df["language"].isin(SELECTED_LANGS)
                & batch_df["license"].isin(SELECTED_LICENSES)
            ]
            for _, x in batch_df.iterrows():
                progress.update()
                data.append(x.to_dict())
                if len(data) >= sample_size:
                    return data


def prepare_github_code(sample_size: int):
    data = load_raw_data(1_000_000)
    df = pd.DataFrame(data)

    # split into clusters by common folder name
    df["cluster_id"] = df.apply(lambda x: x["repo_name"] + "/" + os.path.dirname(x["path"]), axis=1)

    cluster_cnt = df["cluster_id"].value_counts().sort_values()
    multidoc_cluster_ids = cluster_cnt[cluster_cnt > 1].index
    singledoc_cluster_ids = cluster_cnt[cluster_cnt == 1].index

    # sample stratified for both multi and single docs
    multidoc_sampled_ids = pd.Series(multidoc_cluster_ids).sample(
        n=sample_size // 2, random_state=42
    )
    singledoc_sampled_ids = pd.Series(singledoc_cluster_ids).sample(
        n=sample_size // 2, random_state=42
    )
    sampled_df = df[
        df["cluster_id"].isin(multidoc_sampled_ids) | df["cluster_id"].isin(singledoc_sampled_ids)
    ]

    # prepare samples by grouping on cluster id
    clustered_data = []
    for cluster_id, grp in sampled_df.groupby("cluster_id"):
        clustered_data.append(
            {
                "repo_name": grp.iloc[0]["repo_name"],
                "cluster_id": cluster_id,
                "licenses": sorted(set(grp["license"])),
                "languages": sorted(set(grp["language"])),
                "size": int(grp["size"].sum()),
                "documents": [
                    {
                        "filename": x["path"],
                        "content": x["code"],
                        "language": x["language"],
                        "size": x["size"],
                    }
                    for _, x in grp.iterrows()
                ],
            }
        )
    return pd.DataFrame(clustered_data)


def main(outfile=configs.RAW_TRAIN_DATASETS_DIR / "github_code_clustered.jsonl.gz"):
    clustered_df = prepare_github_code(sample_size=5000)
    clustered_df.to_json(outfile, compression="gzip", orient="records", lines=True)


if __name__ == "__main__":  # pragma: nocover
    main()
