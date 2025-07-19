import collections
from typing import Callable, Dict, Sequence, TypeVar

import numpy as np
import scipy
from rouge_score import rouge_scorer

from igcs.entities import Selection

T = TypeVar("T")


def unigram_f1(target: Sequence, prediction: Sequence) -> float:
    return unigram_score(target, prediction)["f1"]


ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def score_rouge(target: str, prediction: str, detailed: bool = False) -> Dict[str, float]:
    scores = ROUGE_SCORER.score(target, prediction)
    scores_dict = {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
    }

    if detailed:
        scores_dict.update(
            {
                "rouge2-p": scores["rouge2"].precision,
                "rouge2-r": scores["rouge2"].recall,
            }
        )
    scores_dict["rougeL"] = scores["rougeL"].fmeasure
    return scores_dict


def pairwise_similarity(
    ref: Selection,
    pred: Selection,
    groupby_doc: bool,
    tokenizer: Callable[[str], Sequence] | None,
    metric: Callable[[Sequence, Sequence], float],
) -> float:
    if groupby_doc and ref.doc_id != pred.doc_id:
        return 0.0

    if tokenizer:
        ref_tokens = tokenizer(ref.content)
        pred_tokens = tokenizer(pred.content)
    else:
        ref_tokens = ref.content
        pred_tokens = pred.content
    return metric(ref_tokens, pred_tokens)


def unigram_score(target: Sequence, prediction: Sequence) -> dict:
    """Computes token F1 score for a single target and prediction."""
    common = collections.Counter(target) & collections.Counter(prediction)
    num_same = sum(common.values())
    if len(target) == 0 and len(prediction) == 0:
        return {"f1": 1.0, "recall": 1.0, "precision": 1.0}
    elif len(target) == 0 and len(prediction) > 0:
        return {"f1": 0.0, "recall": 0.0, "precision": 0.0}
    elif len(target) > 0 and len(prediction) == 0:
        return {"f1": 0.0, "recall": 0.0, "precision": 0.0}
    elif num_same == 0:
        return {"f1": 0.0, "recall": 0.0, "precision": 0.0}
    else:
        precision = 1.0 * num_same / len(prediction)
        recall = 1.0 * num_same / len(target)
        f1 = (2 * precision * recall) / (precision + recall)
        return {"f1": f1, "recall": recall, "precision": precision}


def iou_score(target: Sequence, prediction: Sequence) -> float:
    target_tokens = set(target)
    pred_tokens = set(prediction)

    intersection = len(target_tokens & pred_tokens)
    union = len(target_tokens) + len(pred_tokens) - intersection

    if union == 0:
        if len(target_tokens) == 0 and len(pred_tokens) == 0:
            return 1.0
        else:
            return 0.0
    return intersection / union


def mean_confidence_interval(data, confidence: float):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h
