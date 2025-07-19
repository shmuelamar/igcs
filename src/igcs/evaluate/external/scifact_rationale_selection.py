"""
Computes the rationale selection F1 as in the paper. In order for a sentence to
be correctly identified as a rationale sentence, all other sentences in its gold
rationale must be identified as well.
"""

import argparse
import jsonlines
from collections import Counter


def safe_divide(num, denom):
    if denom == 0:
        return 0
    else:
        return num / denom


def compute_f1(counts, difficulty=None):
    correct_key = "correct" if difficulty is None else f"correct_{difficulty}"
    precision = safe_divide(counts[correct_key], counts["retrieved"])
    recall = safe_divide(counts[correct_key], counts["relevant"])
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def is_correct(pred_sentence, pred_sentences, gold_sets):
    """
    A predicted sentence is correctly identified if it is part of a gold
    rationale, and all other sentences in the gold rationale are also
    predicted rationale sentences.
    """
    for gold_set in gold_sets:
        gold_sents = gold_set["sentences"]
        if pred_sentence in gold_sents:
            if all([x in pred_sentences for x in gold_sents]):
                return True
            else:
                return False

    return False


def eval_predictions(corpus, dataset, rationale_selection):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--corpus', type=str, required=True)
    # parser.add_argument('--dataset', type=str, required=True)
    # parser.add_argument('--rationale-selection', type=str, required=True)
    # args = parser.parse_args()

    corpus = {doc["doc_id"]: doc for doc in jsonlines.open(corpus)}
    dataset = jsonlines.open(dataset)
    rationale_selection = jsonlines.open(rationale_selection)

    counts = Counter()

    for data, retrieval in zip(dataset, rationale_selection):
        assert data["id"] == retrieval["claim_id"]

        # Count all the gold evidence sentences.
        for doc_key, gold_rationales in data["evidence"].items():
            for entry in gold_rationales:
                counts["relevant"] += len(entry["sentences"])

        claim_id = retrieval["claim_id"]

        for doc_id, pred_sentences in retrieval["evidence"].items():
            true_evidence_sets = data["evidence"].get(doc_id) or []

            for pred_sentence in pred_sentences:
                counts["retrieved"] += 1
                if is_correct(pred_sentence, pred_sentences, true_evidence_sets):
                    counts["correct"] += 1

    f1 = compute_f1(counts)
    return f1
