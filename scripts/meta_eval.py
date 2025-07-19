import json
import logging

import pandas as pd
import scipy.stats

from eval_on_all import NAME_TO_MODEL
from igcs.configs import ROOT_DIR
from igcs.utils import log

logger = logging.getLogger(__name__)


COMMON_MODEL_PREFIX = "trainer_models_llama-3_"
MODELS_LIST = [x for x in NAME_TO_MODEL.keys()]


def main():
    with open(ROOT_DIR / "agg_results", "models_results_concat.json") as fp:
        data = json.load(fp)

    dataset_data = []
    for row in data:
        ds = row["dataset_name"]
        if ds in ["ReverseInstructions", "EvidenceDetection", "SaliencyDetection"]:
            continue

        if row["model"] not in MODELS_LIST:
            print("skipping", row["model_codename"])
            continue
        if (
            "-norm-" in row["model_codename"]
            or "LOO-RI-" in row["model_codename"]
            or row["model_codename"] in ["LOO-ASPSEL-fuzz-md", "LOO-ASPSEL-fuzz-sd"]
        ):
            continue

        main_metric = (row["orig_score"], row["ext-f1"])
        precision = ((row.get("precision") or row["rouge2-p"]), row["ext-precision"])
        recall = ((row.get("recall") or row["rouge2-r"]), row["ext-recall"])
        dataset_data.append(
            {
                "model": row["model_codename"],
                "task": ds,
                "main_metric_orig": main_metric[0],
                "main_metric_ext": main_metric[1],
                "precision_orig": precision[0],
                "precision_ext": precision[1],
                "recall_orig": recall[0],
                "recall_ext": recall[1],
            }
        )

    df = pd.DataFrame(dataset_data)
    total_size = len(df)

    # only use models that benchmarked on all datasets
    models_with_full_predictions = []
    for _, row in df.iterrows():
        models_with_full_predictions.append(row.copy().to_dict())
    df = pd.DataFrame(models_with_full_predictions)

    print(f"found {len(df)}/{total_size} full models")
    print("all models:")
    print(sorted(df["model"].unique()))

    results = []

    for task in sorted(df["task"].unique()):
        for metric in ["main", "precision", "recall"]:
            scores = df[df["task"] == task][["main_metric_orig", "main_metric_ext"]].dropna().copy()
            original_score = scores["main_metric_orig"].values
            generic_score = scores["main_metric_ext"].values

            pearson = scipy.stats.pearsonr(original_score, generic_score)
            spearman = scipy.stats.spearmanr(original_score, generic_score)
            kendall = scipy.stats.kendalltau(original_score, generic_score)
            print(f"{metric} scores for {task}:")
            print("pearson:", pearson)
            print("spearman:", spearman)
            print("kendall:", kendall)

            results.append(
                {
                    "dataset_metric": (f"{task}-{metric[0]}" if metric != "main" else task),
                    "pearson": round(pearson.statistic, 4),
                    "pearson-p-value": round(pearson.pvalue, 6),
                    "spearman": round(spearman.statistic, 4),
                    "spearman-p-value": round(spearman.pvalue, 6),
                    "kendall": round(kendall.statistic, 4),
                    "kendall-p-value": round(kendall.pvalue, 6),
                    "num_systems": len(original_score),
                }
            )

    results_df = pd.DataFrame(results)
    print(results_df.to_markdown(index=False))

    results_summary_df = results_df[
        ["dataset_metric", "pearson", "spearman", "kendall"]  # , "num_systems"]
    ].copy()
    results_summary_df = results_summary_df[
        results_summary_df["dataset_metric"].isin(["AspectNews", "DebateSum", "OpenAsp", "SciFact"])
    ].round(3)
    results_summary_df.to_csv(ROOT_DIR / "agg_results" / "meta_eval2.csv", index=False)


if __name__ == "__main__":
    log.init()
    main()
