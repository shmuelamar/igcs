import concurrent
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import pandas as pd

import igcs.datasets.create_gencs_dataset.response_parser
from igcs import grounding, predict
from igcs.configs import ANNOTATED_RI_FNAME, SAMPLED_RI_RAW_DATA_FNAME
from igcs.datasets.igcsbench_datasets import ReverseInstructionsDataset
from igcs.entities import Doc, Selection, SelectionGroup, SelectionSample, SelectionType
from igcs.evaluate.evaluation_functions import unigram_score
from igcs.evaluate.helpers import normalized_unigram
from igcs.predict import LLM_MODEL
from igcs.utils import log
from igcs.utils.prompting import SYSTEM_ROLE

logger = logging.getLogger("igcs.create_gencs_dataset.annotate_dataset")

ANNOTATION_INSTRUCTIONS = """
You are a manager of a {software|publishing} firm and you are required to train the best students on how to perform {code|content} selection from given sources.
Write 5 short instructions for selecting {code|content} from the given {file|document}(s) to challenge students and train them on how to select relevant {code|content} based on diverse instructions.

Guidelines:
  1. You must keep the 5 instructions short and concise - as a single sentence.
  2. Instructions must start with the words "Select {code|content}" as they are always for selecting {code|content} from the {file|document}(s) and never for writing a new {code|text} nor paraphrasing the original content.
  3. Instructions should not be too specific that hint on the answer and not too vague that cannot be fulfilled.
  4. Write the instructions as a numbered list.

The {source-code|document}(s):
""".strip()


ANNOTATION_NEG_INSTRUCTIONS = """
You are a manager of a {software|publishing} firm and you are required to train the best students on how to perform {code|content} selection from given sources.
Write 5 short instructions that have no matching {code|content} from the given {file|document}(s) to challenge students and train them to avoid selecting {code|content} when none matches the instruction.

Guidelines:
  1. You must keep the 5 instructions short and concise - as a single sentence.
  2. Instructions must start with the words "Select {code|content}" as they are always for selecting {code|content} from the {file|document}(s) and never for writing a new {code|text} nor paraphrasing the original content.
  3. Instructions should not be too specific that hint on the answer and not too vague that cannot be fulfilled.
  4. Write the instructions as a numbered list.

The {source-code|document}(s):
""".strip()


SELECTION_INSTRUCTIONS = """
For every instruction listed below, select {code|content} from the below {source file|document}(s) that matches the instruction.

Guidelines:
  1. Output the exact verbatim {code|text phrases} from the {source file|document}(s). Do not change spaces or punctuation, do not fix typos and avoid any other changes to the {code|content} you select.
  2. Follow the instructions as closely as possible and pay careful attention to them.
  3. Output format must be a two level nested list, the first level is the instruction and the second level is the multiple {code|content} selections copied from the original {source file|document}(s).

Instructions:
""".strip()


def format_annotation_guidelines(guidelines: str, is_code: bool) -> str:
    repl_group = r"\1" if is_code else r"\2"
    formatted_guidelines = re.sub(r"\{([\w -]+)\|([\w -]+)\}", repl_group, guidelines)
    assert "{" not in formatted_guidelines and "}" not in formatted_guidelines
    return formatted_guidelines


SYSTEM_ROLE_MANAGER = "You are a manager of a {software|publishing} firm"


def format_doc(docs: List[Doc], is_code: bool):
    prompt = []
    file_type = "Source File" if is_code else "Document"

    for doc_id, doc in enumerate(docs, start=1):
        if not doc.filename:
            prompt.append(f"\n\n{file_type} #{doc_id}:\n")
        else:
            prompt.append(f'\n\n{file_type} #{doc_id} - "{doc.filename.strip()}":\n')
        prompt.append(doc.text.strip())

    return prompt


def sample2tagging_prompt(is_code: bool, is_negative: bool, docs: list[Doc]) -> tuple[str, str]:
    if is_negative:
        guidelines_template = ANNOTATION_NEG_INSTRUCTIONS
    else:
        guidelines_template = ANNOTATION_INSTRUCTIONS

    prompt = [format_annotation_guidelines(guidelines_template, is_code)]
    prompt += format_doc(docs, is_code)
    system_role = format_annotation_guidelines(SYSTEM_ROLE_MANAGER, is_code)
    return "".join(prompt), system_role


def sample2selection_prompt(
    sample_type: str, generated_instructions: list[str], docs: list[Doc]
) -> tuple[str, str]:
    is_code = bool(sample_type == "code")
    file_type = "Source File" if is_code else "Document"

    prompt = [format_annotation_guidelines(SELECTION_INSTRUCTIONS, is_code), "\n"]

    for i, instruction in enumerate(generated_instructions, start=1):
        prompt.append(f"{i}. {instruction}.\n")
    prompt.append("\n")

    prompt.append(f"{file_type}(s):")
    prompt += format_doc(docs, is_code)
    return SYSTEM_ROLE, "".join(prompt)


def load_df(input_file: Path | str) -> pd.DataFrame:
    df = pd.read_json(input_file, lines=True)
    df["documents"] = df["documents"].apply(lambda docs: [Doc(**doc) for doc in docs])
    logger.info(f"loaded {len(df)} samples")
    return df


def parse_generated_instructions(raw_instructions: str, uid: str) -> list[str]:
    """Parse a numbered list of items in a strict manner into a list."""

    try:
        parsed_instructions = list(
            igcs.datasets.create_gencs_dataset.response_parser.parse_markdown_nested_list(
                raw_instructions
            )
        )
        parsed_instructions = [
            igcs.datasets.create_gencs_dataset.response_parser.clean_text(i)
            for i in parsed_instructions
        ]
    except Exception as exc:
        raise exc from ValueError(f"cannot parse {uid} - {raw_instructions!r}")

    return parsed_instructions


def ground_selections(instruction2selections: dict, docs: List[Doc]) -> Dict[str, List[Selection]]:
    instruction2gs = {}
    for instruction, selections in instruction2selections.items():
        grounded_selections = []
        for selection in selections:
            if not isinstance(selection, str):
                logger.warning(
                    f"cannot ground selection {selection!r} from {instruction2selections!r}"
                )
                continue
            gsel = grounding.ground_selection(selection, docs)
            grounded_selections.append(gsel)
        instruction2gs[instruction] = grounded_selections

    return instruction2gs


def translate_instructions(
    instructions: list[str],
    model_instructions: dict[str, list[Selection]],
    uid: str,
    model: str,
) -> tuple[dict, list[dict]]:
    trans_instructions = {}
    differ_instructions = []
    # verify instructions are of same length (i.e. 5)
    for instruction_i, model_instruction_i in zip(instructions, model_instructions, strict=True):
        trans_instructions[instruction_i] = model_instructions[model_instruction_i]

        # anonymized instructions - only numbering
        if model_instruction_i in list("012345"):
            continue

        # verify instructions are related
        tokens_a = normalized_unigram(instruction_i)
        tokens_b = normalized_unigram(model_instruction_i)
        scores = unigram_score(tokens_a, tokens_b)
        if scores["f1"] < 0.5 and scores["recall"] < 0.5:
            logger.warning(
                f"{uid} - model {model} has too different instructions: {instruction_i} != {model_instruction_i}. scores: {scores}"
            )
            differ_instructions.append(
                {
                    "instruction": instruction_i,
                    "model_instruction": model_instruction_i,
                    "similarity": scores,
                }
            )

    return trans_instructions, differ_instructions


def make_selection_samples(sample: dict) -> List[SelectionSample] | None:
    samples = []

    model2selections = sample["selection_results"]
    instructions = sample["generated_instructions"]
    sample_type = sample["type"]
    is_negative = sample["is_negative"]

    if model2selections is None:
        return None

    if not model2selections and not is_negative:
        return []

    docs = sample["documents"]

    for i, instruction in enumerate(instructions):
        model2selections_i = {}
        for model, inst2selections in model2selections.items():
            if inst2selections and instruction in inst2selections:
                model2selections_i[model] = inst2selections[instruction]

        sample_selections = [
            SelectionGroup(id=model, selections=sels) for model, sels in model2selections_i.items()
        ]

        # in case of negative selections - we only alert if model did select anything
        # but we don't use the selections.
        if is_negative:
            for sels in sample_selections:
                if sels.selections:
                    logging.info(
                        f"model {sels.id} returned {len(sels.selections)} for negative selections"
                    )
            sample_selections = [SelectionGroup(id="Negative", selections=[])]

        samples.append(
            SelectionSample(
                id=sample["id"] + f".inst-{i}",
                source_dataset=ReverseInstructionsDataset.name,
                selections=sample_selections,
                docs=docs,
                instruction_context={
                    "instruction": instructions[i],
                    "doc_type": "source file" if sample_type == "code" else "document",
                    "is_negative": is_negative,
                },
                source_sample=None,
                type=SelectionType.span,
            )
        )

    return samples


def step1_generate_instruction(sample: dict, model: LLM_MODEL) -> tuple[list[str], dict]:
    uid = sample["id"]
    is_code = bool(sample["type"] == "code")
    is_negative = bool(sample["is_negative"])
    docs = sample["documents"]

    # create prompt chat
    prompt, system = sample2tagging_prompt(is_code, is_negative, docs)

    # model answer
    raw_instructions = predict.completion(prompt=prompt, system=system, model=model)
    instructions = parse_generated_instructions(raw_instructions, uid)

    # detailed debug info
    debug_info = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "completion": raw_instructions,
    }

    if not len(instructions) == 5:
        logger.warning(
            f"{uid} - got response with wrong number of instructions - {raw_instructions!r}"
        )

    return instructions, debug_info


def step2_perform_selection(
    sample: dict, generated_instructions: list[str], model: LLM_MODEL
) -> tuple[dict[str, list[Selection]] | None, dict]:
    logger.info(f'performing content selection for {sample["id"]} with {model}')

    uid = sample["id"]
    docs = sample["documents"]
    sample_type = sample["type"]

    # create prompt chat
    system, prompt = sample2selection_prompt(sample_type, generated_instructions, docs)

    # model answer
    raw_selections = predict.completion(prompt=prompt, system=system, model=model)

    # parse the model output
    is_error = False
    differ_instructions = None
    instruction2selections = None
    try:
        instruction2raw_selections = (
            igcs.datasets.create_gencs_dataset.response_parser.parse_multi_selections(
                raw_selections
            )
        )
    except (ValueError, TypeError, AttributeError):
        is_error = True
        instruction2raw_selections = None

    if instruction2raw_selections is not None:
        # ground selections to the documents
        instruction2selections = ground_selections(instruction2raw_selections, docs)

        # verify and map back instructions if model output slightly different wording for the
        # original instruction.
        instruction2selections, differ_instructions = translate_instructions(
            generated_instructions, instruction2selections, uid, model
        )
    elif not is_error:
        logger.warning(f"{uid} - cannot parse selection of {model}: {raw_selections}")
    else:
        logger.warning(f"{uid} - error response of {model}")

    # detailed debug info
    debug_info = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "completion": raw_selections,
        "instruction2raw_selections": instruction2raw_selections,
        "differ_instructions": differ_instructions,
        "is_error": is_error,
    }

    return instruction2selections, debug_info


def serialize_df(df: pd.DataFrame):
    df["documents"] = df["documents"].apply(lambda docs: [d.dict() for d in docs])
    df["samples"] = df["samples"].apply(lambda samples: [s.dict() for s in samples])
    df["selection_results"] = df["selection_results"].apply(
        lambda res: {
            model: (
                {inst: [s.dict() for s in sels] for inst, sels in mres.items()} if mres else mres
            )
            for model, mres in res.items()
        }
    )


def main(
    input_file: str | Path,
    output_fname: str | Path,
    debug: bool = True,
):
    input_file = Path(input_file)
    input_file.parent.mkdir(parents=True, exist_ok=True)

    # step1 - load raw, un-annotated data
    df = load_df(input_file)

    instruction_model: LLM_MODEL = "GPT4"
    selection_models: list[LLM_MODEL] = ["GPT4", "Claude3-Opus", "Gemini-1.5"]

    all_instructions_results = []
    all_selections_results = []
    all_debug_infos = []
    for _, sample in df.iterrows():
        logger.info(f'processing sample {sample["id"]}')
        debug_info = {}

        # Step1 - generate instructions using the selected model
        logger.info(f'generating instructions for {sample["id"]} with {instruction_model}')
        instructions, instructions_debug_info = step1_generate_instruction(
            sample, model=instruction_model
        )
        all_instructions_results.append(instructions)
        debug_info["generate_instructions"] = instructions_debug_info

        # Step2 - perform content selection with relevant models
        selection_results = {}
        debug_info["selections"] = []

        with ThreadPoolExecutor(max_workers=len(selection_models)) as pool:
            future2model = {
                pool.submit(step2_perform_selection, sample, instructions, model=model): model
                for model in selection_models
            }
            for future in concurrent.futures.as_completed(future2model):
                selection_model = future2model[future]
                instruction2selections, selections_debug_info = future.result()

                selection_results[selection_model] = instruction2selections
                debug_info["selections"].append(selections_debug_info)

        if debug:
            all_debug_infos.append(debug_info)
        all_selections_results.append(selection_results)

    df["generated_instructions"] = all_instructions_results
    df["selection_results"] = all_selections_results

    # step 4 - output debug information
    if debug:
        df["debug_info"] = all_debug_infos

    df["samples"] = df.apply(make_selection_samples, axis=1)

    # serialize and save
    logger.info(f"saving {len(df)} records")
    df.sort_values(["id"], inplace=True)
    serialize_df(df)
    df.to_json(output_fname, lines=True, orient="records")
    logger.info(f"saved annotated data to {output_fname}")


if __name__ == "__main__":  # pragma: nocover
    log.init()
    main(input_file=SAMPLED_RI_RAW_DATA_FNAME, output_fname=ANNOTATED_RI_FNAME)
