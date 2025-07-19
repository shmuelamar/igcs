import json
import os
import re
from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd

from igcs.cseval import score_selection
from igcs.entities import Selection, SelectionGroup, SelectionSample
from scripts.generate_datasets_table import load_all_datasets


def load_ri_dataset(is_majority: bool = False):
    if is_majority:
        dataset_name = "ReverseInstructions-Majority"
    else:
        dataset_name = "ReverseInstructions"
    src_df = load_all_datasets()
    return src_df[(src_df["source_dataset"] == dataset_name) & (src_df["split"] == "test")][
        "sample"
    ].tolist()


def get_annotation(row: dict, id2sample: dict[str, SelectionSample]):
    assert len(row["annotations"]) == 1

    instructions = row["data"]["instructions"]
    selections = {inst: [] for inst in instructions}

    sample_id = row["data"]["id"]
    sample_selection = id2sample[sample_id]

    docs_text = row["data"]["text"]
    md = docs_text.strip().startswith("-~-~-~--~- Document #0 -~--~-~-~-")

    for annotation in row["annotations"][0]["result"]:
        start = annotation["value"]["start"]
        end = annotation["value"]["end"]
        # labeled_text = annotation['value']['text']
        extracted_text = docs_text[start:end]

        assert len(annotation["value"]["labels"]) == 1
        labeled_inst = annotation["value"]["labels"][0]
        inst = labeled_inst.split(": ", 1)[1]
        assert inst in instructions

        # in case of a single document (i.e. we dont have document header) we use doc_id = 0
        if not md:
            doc_id = 0

        # in case of multi document, we need to adjust the start & end pos to be relative to the document
        # rather than the absolute start and end.
        else:
            matches = re.findall(r"(\n?-~-~-~--~- Document #(\d+) -~--~-~-~-\n\n)", docs_text[:end])
            assert matches
            doc_id = int(matches[-1][1])  # last match doc_id
            # header start position + header length
            doc_start_pos = docs_text.find(matches[-1][0]) + len(matches[-1][0])
            start -= doc_start_pos
            end -= doc_start_pos
            assert start >= 0
            assert end > start

        doc_text = sample_selection.docs[doc_id].text[start:end]

        assert doc_id >= 0
        assert doc_text == extracted_text, "cannot find the extracted text in the original document"

        tagged_selection = Selection(
            doc_id=doc_id,
            start_pos=start,
            end_pos=end,
            content=doc_text,
        )
        selections[inst].append(tagged_selection)

    return {
        inst: sorted(sels, key=lambda s: (s.doc_id, s.start_pos, s.end_pos))
        for inst, sels in selections.items()
    }


def get_matching_samples(
    matching_samples: Sequence[SelectionSample] | None,
    inst2selections: dict[str, list[Selection]],
    id2sample: dict[str, SelectionSample],
    id2sample_majority: dict[str, SelectionSample],
    topic_id: str,
    annotator_id: int,
):
    if not matching_samples:
        matching_samples = {
            x.instruction_context["instruction"]: x.model_copy(deep=True)
            for sid, x in id2sample.items()
            if sid.startswith(topic_id)
        }
    assert len(matching_samples) == 5
    assert matching_samples.keys() == inst2selections.keys()

    matching_samples_majority = {
        x.instruction_context["instruction"]: x.model_copy(deep=True)
        for sid, x in id2sample_majority.items()
        if sid.startswith(topic_id)
    }
    assert len(matching_samples_majority) == 5
    assert matching_samples_majority.keys() == inst2selections.keys()

    for inst, sample in matching_samples.items():
        annotated_selections = inst2selections[inst]
        sample.selections.append(
            SelectionGroup(selections=annotated_selections, id=f"annotator_{annotator_id}")
        )

        # add the majority selections
        if sample.is_negative or any([s.id == "Majority" for s in sample.selections]):
            continue

        majority_sels = [
            m.model_copy(deep=True)
            for m in matching_samples_majority[inst].selections
            if m.id == "Merged"
        ]
        assert len(majority_sels) == 1
        majority_sels = majority_sels[0].selections
        sample.selections.append(SelectionGroup(selections=majority_sels, id="Majority"))
    return matching_samples


def main(dirname="../tagging", metric="f1"):
    # load annotators data
    ri_dataset_union = load_ri_dataset(is_majority=False)
    ri_dataset_majority = load_ri_dataset(is_majority=True)
    id2sample_union = {x.id: x for x in ri_dataset_union}
    id2sample_majority = {x.id: x for x in ri_dataset_majority}
    assert (
        len(ri_dataset_union)
        == len(id2sample_majority)
        == len(ri_dataset_majority)
        == len(id2sample_union)
    )

    data = {}

    all_matching_samples = {}
    for rater in [0, 1]:
        with open(os.path.join(dirname, f"annotator_0.json")) as fp:  # both should tag same data
            data[rater] = json.load(fp)
        expected_ids = sorted([x["id"] for x in data[rater]])

        with open(os.path.join(dirname, f"annotator_{rater}_results.json")) as fp:
            tagged_data = json.load(fp)
        tagged_ids = sorted([x["data"]["id"] for x in tagged_data])

        for row in tagged_data:
            inst2selections = get_annotation(row, id2sample_union)
            topic_id = row["data"]["topic_id"]
            matching_samples = all_matching_samples.get(row["id"])
            all_matching_samples[row["id"]] = get_matching_samples(
                matching_samples,
                inst2selections,
                id2sample_union,
                id2sample_majority,
                topic_id,
                rater,
            )

        assert expected_ids == tagged_ids

    all_matching_samples = sorted(
        [sample for samples in all_matching_samples.values() for sample in samples.values()],
        key=lambda s: s.id,
    )

    # calc F1 for each model
    model2scores = defaultdict(list)
    annotator2negatives_scores = defaultdict(list)

    all_annotator_ids = [f"annotator_0", f"annotator_1"]
    all_model_ids = [
        "Claude3-Opus",
        "GPT4",
        "Gemini-1.5",
    ]
    all_dataset_ids = [
        "Merged",
        "Majority",
    ]
    all_rater_ids = all_annotator_ids + all_model_ids + all_dataset_ids
    agreement_matrix = np.zeros((len(all_rater_ids), len(all_rater_ids), len(all_matching_samples)))

    for sid, sample in enumerate(all_matching_samples):
        for i, rater in enumerate(all_rater_ids):
            if sample.is_negative and rater in all_model_ids + all_dataset_ids:
                agreement_matrix[i, :, sid] = np.nan
                continue

            annotator_selection = [s for s in sample.selections if s.id == rater]
            assert len(annotator_selection) == 1
            annotator_selection = annotator_selection[0].selections

            if sample.is_negative:
                annotator2negatives_scores[rater].append(
                    score_selection(annotator_selection, [], sample.docs)[metric]
                )
                agreement_matrix[i, :, sid] = np.nan
                continue

            for j, other_rater in enumerate(all_rater_ids):
                model_selection = [s for s in sample.selections if s.id == other_rater]
                assert len(model_selection) == 1
                model_selection = model_selection[0].selections

                k_score = score_selection(annotator_selection, model_selection, sample.docs)[metric]
                model2scores[f"{rater}_{other_rater}"].append(k_score)
                agreement_matrix[i, j, sid] = k_score

    print((np.nanmean(agreement_matrix, axis=2) * 100).round(1))

    human_human_agreement = np.nanmean(agreement_matrix[0, 1])
    human_llm_agreement = sum(
        [np.nanmean(agreement_matrix[i, j]) for i in range(2) for j in range(2, 5)]
    ) / (2 * 3)
    llm_llm_agreement = (
        sum([np.nanmean(agreement_matrix[i, j]) for i in range(2, 5) for j in range(i + 1, 5)]) / 3
    )
    print(f"human-human agreement = ({human_human_agreement:.3f})")
    print(f"llm_llm_agreement ({llm_llm_agreement:.3f})")
    print(f"human_llm_agreement ({human_llm_agreement:.3f})")
    print(np.min(agreement_matrix))

    for rater in [0, 1]:
        print(f"------- annotator {rater}")
        print(
            pd.DataFrame(model2scores)[f"annotator_{1-rater}_annotator_{rater}"].round(3).tolist()
        )
        print("non-empty selections scores:")
        print(pd.DataFrame(model2scores).mean())
        print("empty selections scores:")
        print(pd.DataFrame(annotator2negatives_scores).mean())


if __name__ == "__main__":
    main(metric="f1")

#     Bn    Ev    Cld    G4   gem    Uni   Maj
# B [100.   58.6  59.7  57.9  59.   62.4  63. ]
# E [ 58.6 100.   63.7  63.3  64.8  64.5  67.4]
# C [ 59.7  63.7 100.   68.9  73.9  83.6  87.4]
# G [ 57.9  63.3  68.9 100.   68.2  71.5  82.1]
# g [ 59.   64.8  73.9  68.2 100.   89.3  85. ]
# U [ 62.4  64.5  83.6  71.5  89.3 100.   82.5]
# m [ 63.   67.4  87.4  82.1  85.   82.5 100. ]

# empty selections scores:
# annotator_0 (Bn)    0.657143
# annotator_1 (Ev)    0.971429
