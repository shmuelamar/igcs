from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from igcs import configs
from igcs.datasets.create_gencs_dataset.create_reverse_instructions_dataset_splits import (
    filter_ri_dataset,
    get_sample_ila,
)
from igcs.entities import SelectionSample
from igcs.utils import log


def main(infile: str, min_selection_votes):
    ri_df = pd.read_json(infile, lines=True)
    raw_samples = filter_ri_dataset(
        ri_df,
        min_selection_votes=min_selection_votes,
        skip_empty=False,
        skip_all_hallucination=True,
    )["filtered_samples"]
    samples = [SelectionSample(**x) for batch in raw_samples for x in batch]

    ila_scores = defaultdict(float)
    freqs = defaultdict(int)
    for sample in tqdm(samples):
        if sample.is_negative:
            # we dont tag them
            continue

        models_selections = [
            s for s in sample.selections if s.id in ["Claude3-Opus", "GPT4", "Gemini-1.5"]
        ]
        assert len(models_selections) == 3, sample.model_dump()
        ila_score, pair_scores = get_sample_ila(models_selections, sample.docs)
        ila_scores["overall"] += ila_score
        freqs["overall"] += 1
        for (model_a, model_b), pair_score in pair_scores.items():
            ila_scores[model_a] += pair_score
            ila_scores[model_b] += pair_score
            freqs[model_a] += 1
            freqs[model_b] += 1
            ila_scores[f"{model_a}:{model_b}"] += pair_score
            freqs[f"{model_a}:{model_b}"] += 1

    for model, score in ila_scores.items():
        score /= freqs[model]
        print(f"{model}: {score:.3f}")


if __name__ == "__main__":
    log.init("WARN")
    main(configs.ANNOTATED_RI_FNAME, min_selection_votes=0)
    # Must be the same
    main(configs.ANNOTATED_RI_FNAME, min_selection_votes=2)
