import re
from collections import Counter
from functools import cache
from itertools import product

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from scripts.analyze_manual_annotation import load_ri_dataset


def get_annotation_df():
    annotators = ["benny", "eviatar"]
    manual_annotations = [
        pd.read_excel(
            f"../tagging/IGCS Annotations V2.1.xlsx",
            sheet_name=f"manual_annotations_{name}",
        ).set_index("Unnamed: 0")
        for name in annotators
    ]

    assert all(
        ann.index.tolist() == manual_annotations[0].index.tolist() for ann in manual_annotations
    )

    ann_df = manual_annotations[0].join(
        manual_annotations[1], lsuffix="_a0", rsuffix="_a1", how="inner"
    )
    assert len(ann_df) == len(manual_annotations[0]), len(ann_df)
    assert ann_df.index.tolist() == manual_annotations[0].index.tolist()

    for col in ["id", "topic_id", "instructions"]:
        assert (ann_df[col + "_a0"] == ann_df[col + "_a1"]).all()

    metadata_columns = ["id_a0", "topic_id_a0", "instructions_a0"]
    annotated_columns = [
        "naturalness (1-5)",
        "specificity (1-5)",
        "salience (1-5)",
        "comment",
    ]
    all_columns = metadata_columns[:]
    all_columns += [col + "_a0" for col in annotated_columns]
    all_columns += [col + "_a1" for col in annotated_columns]
    ann_df = (
        ann_df[all_columns]
        .rename(columns=lambda x: re.sub(re.escape(" (1-5)"), "", x))
        .rename(columns={col: col.removesuffix("_a0") for col in metadata_columns})
    )
    return ann_df.set_index("id", drop=True)


def custom_matrix_cohen_kappa(y1, y2, *, labels=None, weights=None):
    confusion = confusion_matrix(y1, y2, labels=labels)
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    if weights is None:
        w_mat = np.ones([n_classes, n_classes], dtype=int)
        w_mat.flat[:: n_classes + 1] = 0
    else:
        w_mat = np.zeros([n_classes, n_classes], dtype=int)
        w_mat += np.arange(n_classes)
        if weights == "linear":
            w_mat = np.abs(w_mat - w_mat.T)
        else:
            w_mat = (w_mat - w_mat.T) ** 2

    for i in range(w_mat.shape[0]):
        w_mat[i, min(i + 1, w_mat.shape[0] - 1)] = 0
        w_mat[i, max(i - 1, 0)] = 0
    w_mat = np.maximum(w_mat - 1, 0)

    k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
    return 1 - k


@cache
def get_ri_dataset():
    return load_ri_dataset()


def main(close_is_same: bool, positive_only: bool, three_or_five, weights: str | None):
    ann_df = get_annotation_df()
    ri_dataset = get_ri_dataset()
    id2sample = {x.id: x for x in ri_dataset}

    metrics_kappa = {}
    for metric in ["naturalness", "specificity", "salience"]:

        scores = []
        ids = []
        metric_a = metric + "_a0"
        metric_b = metric + "_a1"
        valid_values = [1, 2, 3, 4, 5, "1", "2", "3", "4", "5"]  # , 'inf', '?'

        for idx, row in ann_df.iterrows():
            # skip negatives from calculations
            if id2sample[idx].is_negative:
                if metric == "salience":
                    continue
                elif positive_only:
                    continue

            if row[metric_a] not in valid_values or row[metric_b] not in valid_values:
                # print(f'Skipping {idx}: {row[metric_a]} and {row[metric_b]}')
                continue
            if row[metric_a] == "inf":
                score_a = 1
            elif row[metric_a] == "?":
                score_a = 3
            else:
                score_a = row[metric_a]

            if three_or_five == 5:
                scores.append((int(score_a), int(row[metric_b])))
            elif three_or_five == 3:
                five_to_three = [None, 1, 1, 2, 3, 3]  # map 1,2 -> 1; 3 -> 2; 4,5 -> 3
                scores.append(
                    (
                        five_to_three[int(row[metric_a])],
                        five_to_three[int(row[metric_b])],
                    )
                )
            else:
                raise ValueError("invalid 3or5")

            ids.append(idx)

        scores = np.asarray(scores)

        if close_is_same:
            k = custom_matrix_cohen_kappa(
                scores[:, 0], scores[:, 1], weights=weights  # , labels=[1, 2, 3, 4, 5]
            )
        else:
            k = metrics.cohen_kappa_score(
                scores[:, 0], scores[:, 1], weights=weights  # , labels=[1, 2, 3, 4, 5]
            )

        metrics_kappa[metric] = k

        consolidated_scores = (scores[:, 0] + scores[:, 1]) / 2
        print("-------")
        print(f"{metric} (n={len(scores)}): κ={k:.3}")
        print(
            f"histogram annotator0 (mean={scores.T[0].mean():.2} ± {scores.T[0].std():.3f})",
            Counter(scores.T[0]),
        )
        print(
            f"histogram annotator1 (mean={scores.T[1].mean():.2} ± {scores.T[1].std():.3f})",
            Counter(scores.T[1]),
        )
        print(
            f"consolidate n={consolidated_scores.size} mean={consolidated_scores.mean():.3f} std={consolidated_scores.std():.3f} var={consolidated_scores.var()}",
            Counter(consolidated_scores),
        )
        print("diff between annotators:")
        print(scores.T[0] - scores.T[1])
        # print(ann_df[metric_a].value_counts())
        print("-------")

    return metrics_kappa


if __name__ == "__main__":
    selected_key = {
        "close_is_same": False,
        "positive_only": False,
        "three_or_five": 5,
        "weights": "linear",
    }
    res = []
    for close_is_same, positive_only, three_or_five, weights in product(
        [True, False], [True, False], [5, 3], [None, "linear", "quadratic"]
    ):

        key = {
            "close_is_same": close_is_same,
            "positive_only": positive_only,
            "three_or_five": three_or_five,
            "weights": weights,
        }
        if key != selected_key:
            continue

        print(", ".join([f"{k}:{v}" for k, v in key.items()]))
        kappas = main(
            close_is_same=close_is_same,
            positive_only=positive_only,
            three_or_five=three_or_five,
            weights=weights,
        )
        res.append(
            {
                **key,
                **kappas,
            }
        )
