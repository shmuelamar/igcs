import argparse
import json
import logging
import random
from pathlib import Path
from typing import Callable, Literal

from igcs import configs, cseval
from igcs.datasets import get_dataset_file
from igcs.entities import Doc, SelectionGroup, SelectionSample
from igcs.llms import claude, gemini, gpt4, llama3_igcs, together_api
from igcs.utils import (
    ioutils,
    log,
    prompting,
)

logger = logging.getLogger(__name__)

LLM_MODEL = Literal["GPT4", "Claude3-Opus", "Gemini-1.5", "llama3"]
API_MODEL2COMPLETION: dict[str, Callable[[str, str], str]] = {
    "GPT4": gpt4.completion,
    "Claude3-Opus": claude.completion,
    "Gemini-1.5": gemini.completion,
    "llama3-70B": together_api.completion_llama_70b,
    "llama3-405B": together_api.completion_llama_405b,
}


def get_prompt_and_system_from_conversations(msgs: dict) -> tuple[str, str]:
    assert 2 <= len(msgs) <= 3
    if len(msgs) == 2:
        assert msgs[2]["role"] == "assistant"
    assert msgs[0]["role"] == "system"
    system = msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    prompt = msgs[1]["content"]
    return prompt, system


def completion(
    *,
    prompt: str,
    system: str | None,
    model: str,
    chat_history: list[dict] | None = None,
) -> str:
    model_name, model_type = get_model_id(model)
    if model_type == "api":
        if chat_history is not None:
            return API_MODEL2COMPLETION[model](
                prompt=prompt, system=system, chat_history=chat_history
            )
        return API_MODEL2COMPLETION[model](prompt=prompt, system=system)

    params = {
        "prompt": prompt,
        "system": system,
        "model": model_name,
        "skip_special_tokens": True,
        "chat_history": chat_history,
    }

    if model_type == "llama3":
        logger.info(f"predicting llama3 - {model}")
        return llama3_igcs.completion(**params)

    assert model_type in ["mistral", "phi"], model_type
    # mistral and phi accepts no system prompt
    params["system"] = None
    logger.info(f"predicting {model_type} - {model}")
    return llama3_igcs.completion(**params)


def get_canonic_fname(split_name: str, dataset_name: str, model: str) -> str:
    return f"{split_name}-{dataset_name}-{model.replace('/', '_')}.jsonl.gz"


def get_model_id(model_or_path: str) -> tuple[str, str]:
    name = model_or_path.lower()
    model2name = {name.lower(): name for name in API_MODEL2COMPLETION}

    # predefined model
    model2name.get(name)
    if name in model2name:
        return model2name[name], "api"

    if name in llama3_igcs.LLAMA3_MODEL_IDS:
        return llama3_igcs.LLAMA3_MODEL_IDS[name], "llama3"

    if "mistral" in name:
        return model_or_path, "mistral"

    if "phi" in name:
        return model_or_path, "phi"

    return model_or_path, "llama3"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        help="One of zeroshot (default), icl (in-context learning), single_doc (one source document at a time), or icl_single_doc for the last two modes combined.",
        type=str,
        required=False,
        default="zeroshot",
    )
    parser.add_argument(
        "-i",
        "--infile",
        help="Input prompts file in JSON-Lines format for prediction."
        "file can be also a predefined dataset such as OpenAsp/test",
        nargs="+",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Model to predict results on",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--outfile",
        help="Output predictions file in JSON-Lines format. The scripts adds `selection` key to every row "
        "in the input file, keeping other keys intact.",
        type=Path,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        help="Predict only on the first n samples specified. Defaults to all samples.",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--skip-eval",
        help="If set, disable evaluation step at end.",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--shuffle",
        help="If set, shuffles predicted samples. Can be combined with --num-samples.",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--dry-run",
        help="If set, does not predict but only prints prompts.",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--icl-num-samples",
        help="Number of samples from the train set to include in in-context learning mode. Only relevant if mode is icl",
        type=int,
        required=False,
        default=2,
    )
    parser.add_argument(
        "--icl-samples-from-eval",
        help="When set uses ICL samples from the eval set (required for datasets without train set)",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--randomize-icl-samples",
        help="Whether to randomize per eval sample the ICL samples or use the same samples for all the eval set",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--prompt-variant",
        help="The index of the prompt template variant to use",
        required=False,
        default=0,
        type=int,
    )

    args = parser.parse_args(argv)
    if len(args.infile) > 1 and args.outfile:
        raise ValueError(
            "Cannot define multiple input files and a single output file - call this script multiple times instead."
        )
    return args


def run_predict(
    infile: Path,
    model: LLM_MODEL,
    do_eval: bool,
    outfile: Path | None = None,
    num_samples: int | None = None,
    shuffle: bool = False,
    dry_run: bool = False,
    prompt_variant: int = 0,
):
    if prompt_variant != 0:
        model_save_name = f"P{prompt_variant}-{model}"
    else:
        model_save_name = model

    resolved_infile, dataset_name, split_name = get_dataset_file(str(infile), is_prompt=True)
    if not outfile:
        if not dataset_name or not split_name:
            raise ValueError(
                "Cannot infer output filename - please specify output file or predefined input dataset"
            )
        outfile = configs.PREDICTIONS_DIR / get_canonic_fname(
            split_name, dataset_name, model_save_name
        )
        logger.info(f"Setting output file to {outfile}")

    logger.info(f"Loading {infile} data from {resolved_infile}")
    outfile.parent.mkdir(parents=True, exist_ok=True)

    data = list(ioutils.jsonl_load(resolved_infile))
    samples = [
        SelectionSample(**s)
        for s in ioutils.jsonl_load(get_dataset_file(str(infile), is_prompt=False)[0])
    ]
    if shuffle:
        logger.info("Shuffling data")
        random.Random(42).shuffle(data)
        random.Random(42).shuffle(samples)

    if num_samples is not None:
        logger.info(f"Predicting only the first {num_samples} samples from {resolved_infile}")
        data = data[:num_samples]
        samples = samples[:num_samples]

    for i, (row, sample) in enumerate(zip(data, samples, strict=True)):
        sample_id = row.get("sample_id", i)
        assert sample_id == sample.id
        if prompt_variant == 0:
            prompt, system = get_prompt_and_system_from_conversations(row["conversations"])
        else:
            prompt = prompting.format_prompt(sample, instruction_variant_id=prompt_variant)
            system = prompting.SYSTEM_ROLE

        logger.info(f"predicting for sample {sample_id} - {i+1}/{len(data)}")
        logger.debug(f"prompt: {prompt}")

        if not dry_run:
            model_selection = completion(prompt=prompt, system=system, model=model)

            row["selection"] = model_selection
            logger.debug(f"Selection: {model_selection}")
            logger.debug(
                f"Answer: {json.dumps([t.content for t in sample.selections[0].selections], indent=4)}"
            )
        else:
            logger.info(f"DRYRUN:\nsystem: {system}\nprompt:\n{prompt}")

    if dry_run:
        logger.info("DRYRUN: finished")
        return

    logger.info(f"writing results to {outfile}")
    ioutils.jsonl_dump(data, outfile)

    if do_eval and dataset_name and split_name:
        logger.info("performing evaluation.")
        eval_outfile = configs.EVAL_RESULTS_DIR / get_canonic_fname(
            split_name, dataset_name, model_save_name
        )
        cseval.run_cs_eval(
            references=f"{dataset_name}/{split_name}",
            predictions=outfile,
            output=eval_outfile,
            allow_partial=num_samples is not None,
        )


def get_icl_train_samples(
    sample_id: str,
    samples_pool: list[SelectionSample],
    icl_num_samples: int,
    single_doc: bool,
    seed=42,
) -> list[SelectionSample]:
    from igcs.datasets.prepare_igcs_datasets import get_merged_selection_merged

    # we sample +1 samples so that if any sample is the same as the currently
    # evaluated - we can skip it and take another one

    icl_samples = random.Random(seed).sample(samples_pool, k=icl_num_samples + 1)
    icl_samples = [icl_sample for icl_sample in icl_samples if icl_sample.id != sample_id][
        :icl_num_samples
    ]
    assert len(icl_samples) == icl_num_samples

    logger.info(
        f"selected {len(icl_samples)} samples for ICL prompt: {', '.join([s.id for s in icl_samples])}"
    )

    # prepare training samples as examples, in case of multi refs we merge selections similar
    # to how IGCS datasets are derived.
    # ensure the ICL samples do not include the sample_id that is currently evaluated
    icl_samples = [
        icl_sample.model_copy(
            deep=True,
            update={
                "selections": [SelectionGroup(selections=get_merged_selection_merged(icl_sample))]
            },
        )
        for icl_sample in icl_samples
    ]

    if single_doc:
        rng = random.Random(seed)
        for icl_sample in icl_samples:
            selected_doc: Doc = rng.choice(icl_sample.docs)
            # choose randomly only a single icl sample per doc
            icl_sample.selections[0].selections = [
                sel.model_copy(deep=True, update={"doc_id": 0})
                for sel in icl_sample.selections[0].selections
                if sel.doc_id == selected_doc.id
            ]
            icl_sample.docs = [selected_doc.model_copy(deep=True, update={"id": 0})]

    return icl_samples


def run_predict_icl(
    infile: Path,
    model: LLM_MODEL,
    do_eval: bool,
    single_doc: bool,
    outfile: Path | None = None,
    num_samples: int | None = None,
    icl_samples_from_eval: bool = False,
    randomize_icl_samples: bool = False,
    shuffle: bool = False,
    dry_run: bool = False,
    icl_num_samples: int = 2,
    prompt_variant: int = 0,
):
    if randomize_icl_samples:
        model_save_name = f"RND-{model}"
    else:
        model_save_name = model

    if icl_num_samples != 2:
        model_save_name = f"N{icl_num_samples}-{model_save_name}"

    if prompt_variant != 0:
        model_save_name = f"P{prompt_variant}-{model_save_name}"

    if single_doc:
        model_save_name = f"SD-ICL-{model_save_name}"
    else:
        model_save_name = f"ICL-{model_save_name}"

    eval_infile, dataset_name, split_name = get_dataset_file(str(infile), is_prompt=False)
    if not dataset_name or not split_name:
        raise ValueError("ICL only supports datasets defined on `igcs.igcsbench_datasets` module")
    train_infile = get_dataset_file(f"{dataset_name}/train", is_prompt=False)[0]

    if not outfile:
        outfile = configs.PREDICTIONS_DIR / get_canonic_fname(
            split_name, dataset_name, model_save_name
        )
        logger.info(f"Setting output file to {outfile}")

    logger.info(f"Loading {dataset_name}/{split_name} data from {eval_infile}")
    outfile.parent.mkdir(parents=True, exist_ok=True)

    eval_data = [SelectionSample(**s) for s in ioutils.jsonl_load(eval_infile)]

    if icl_samples_from_eval:
        logger.info("use eval samples for ICL samples")
        icl_samples_pool = [SelectionSample(**s) for s in ioutils.jsonl_load(eval_infile)]
    else:
        icl_samples_pool = [SelectionSample(**s) for s in ioutils.jsonl_load(train_infile)]

    if shuffle:
        logger.info("Shuffling data")
        random.Random(42).shuffle(eval_data)

    if num_samples is not None:
        logger.info(f"Predicting only the first {num_samples} samples from {eval_infile}")
        eval_data = eval_data[:num_samples]

    predictions = []
    for i, sample in enumerate(eval_data):
        icl_samples = get_icl_train_samples(
            sample_id=sample.id,
            samples_pool=icl_samples_pool,
            icl_num_samples=icl_num_samples,
            single_doc=single_doc,
            seed=sample.id if randomize_icl_samples else 42,
        )

        system = prompting.SYSTEM_ROLE
        logger.info(f"predicting for sample {sample.id} - {i+1}/{len(eval_data)}")

        if not single_doc:
            prompt = prompting.get_icl_prompt_for_sample(
                sample, train_samples=icl_samples, prompt_variant=prompt_variant
            )
            if not dry_run:
                model_selection = completion(prompt=prompt, system=system, model=model)
                row = {
                    "sample_id": sample.id,
                    "conversations": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "selection": model_selection,
                }
                predictions.append(row)
            else:
                logger.info(f"DRYRUN:\nsystem: {system}\nprompt:\n{prompt}")

        else:
            model_selection, conversations = exec_single_doc(
                sample,
                model,
                system,
                dry_run,
                icl_samples=icl_samples,
                prompt_variant=prompt_variant,
            )
            row = {
                "sample_id": sample.id,
                "conversations": conversations,
                "selection": model_selection,
            }
            predictions.append(row)

    if dry_run:
        logger.info("DRYRUN: finished")
        return

    logger.info(f"writing results to {outfile}")
    ioutils.jsonl_dump(predictions, outfile)

    if do_eval:
        logger.info("performing evaluation.")
        eval_outfile = configs.EVAL_RESULTS_DIR / get_canonic_fname(
            split_name, dataset_name, model_save_name
        )
        cseval.run_cs_eval(
            references=eval_infile,
            predictions=outfile,
            output=eval_outfile,
            allow_partial=num_samples is not None,
            multi_selections=single_doc,
        )


def exec_single_doc(
    sample: SelectionSample,
    model: str,
    system: str,
    dry_run: bool = False,
    icl_samples: list[SelectionSample] | None = None,
    prompt_variant: int = 0,
):
    selections = []
    conversations = []
    for doc_id, doc in enumerate(sample.docs):
        logger.info(f"predicting doc {doc_id + 1}/{len(sample.docs)}")

        doc_sample = sample.model_copy(deep=True)
        doc_sample.docs = [Doc(id=0, text=doc.text)]

        if not icl_samples:
            prompt = prompting.format_prompt(doc_sample, instruction_variant_id=prompt_variant)
        else:
            prompt = prompting.get_icl_prompt_for_sample(
                doc_sample, train_samples=icl_samples, prompt_variant=prompt_variant
            )

        conversations.append(
            [
                {"role": "system", "content": system},
                {"role": "system", "content": prompt},
            ]
        )

        if not dry_run:
            doc_selection = completion(prompt=prompt, system=system, model=model)
            selections.append(doc_selection)
        else:
            logger.info(f"DRYRUN:\nsystem: {system}\nprompt:\n{prompt}")

    return selections, conversations


def run_predict_single_doc(
    infile: Path,
    model: LLM_MODEL,
    do_eval: bool,
    outfile: Path | None = None,
    num_samples: int | None = None,
    shuffle: bool = False,
    dry_run: bool = False,
    prompt_variant: int = 0,
):
    if prompt_variant != 0:
        model_save_name = f"SD-P{prompt_variant}-{model}"
    else:
        model_save_name = f"SD-{model}"
    infile, dataset_name, split_name = get_dataset_file(str(infile), is_prompt=False)
    if not dataset_name or not split_name:
        raise ValueError(
            "Single doc only supports datasets defined on `igcs.igcsbench_datasets` module"
        )

    if not outfile:
        outfile = configs.PREDICTIONS_DIR / get_canonic_fname(
            split_name, dataset_name, model_save_name
        )
        logger.info(f"Setting output file to {outfile}")

    logger.info(f"Loading {dataset_name}/{split_name} data from {infile}")
    outfile.parent.mkdir(parents=True, exist_ok=True)

    data = [SelectionSample(**s) for s in ioutils.jsonl_load(infile)]

    if shuffle:
        logger.info("Shuffling data")
        random.Random(42).shuffle(data)

    if num_samples is not None:
        logger.info(f"Predicting only the first {num_samples} samples from {infile}")
        data = data[:num_samples]

    predictions = []
    system = prompting.SYSTEM_ROLE
    for i, sample in enumerate(data):
        logger.info(f"predicting for sample {sample.id} - {i+1}/{len(data)}")
        selections, conversations = exec_single_doc(
            sample, model, system, dry_run, prompt_variant=prompt_variant
        )

        row = {
            "sample_id": sample.id,
            "conversations": conversations,
            "selection": selections,
        }
        predictions.append(row)

    if dry_run:
        logger.info("DRYRUN: finished")
        return

    logger.info(f"writing results to {outfile}")
    ioutils.jsonl_dump(predictions, outfile)

    if do_eval:
        logger.info("performing evaluation.")
        eval_outfile = configs.EVAL_RESULTS_DIR / get_canonic_fname(
            split_name, dataset_name, model_save_name
        )
        cseval.run_cs_eval(
            references=infile,
            predictions=outfile,
            output=eval_outfile,
            allow_partial=num_samples is not None,
            multi_selections=True,
        )


def main(argv=None):
    args = parse_args(argv)
    mode = args.mode.lower()

    for infile in args.infile:
        if mode == "zeroshot":
            run_predict(
                infile=infile,
                model=args.model,
                do_eval=not args.skip_eval,
                outfile=args.outfile,
                num_samples=args.num_samples,
                shuffle=args.shuffle,
                dry_run=args.dry_run,
                prompt_variant=args.prompt_variant,
            )
        elif mode in ["icl", "icl_single_doc"]:
            run_predict_icl(
                infile=infile,
                model=args.model,
                do_eval=not args.skip_eval,
                single_doc=mode.endswith("_single_doc"),
                outfile=args.outfile,
                num_samples=args.num_samples,
                icl_samples_from_eval=args.icl_samples_from_eval,
                randomize_icl_samples=args.randomize_icl_samples,
                shuffle=args.shuffle,
                dry_run=args.dry_run,
                icl_num_samples=args.icl_num_samples,
                prompt_variant=args.prompt_variant,
            )
        elif mode == "single_doc":
            run_predict_single_doc(
                infile=infile,
                model=args.model,
                do_eval=not args.skip_eval,
                outfile=args.outfile,
                num_samples=args.num_samples,
                shuffle=args.shuffle,
                dry_run=args.dry_run,
                prompt_variant=args.prompt_variant,
            )
        else:
            raise ValueError(f"Unknown mode {mode}")


if __name__ == "__main__":  # pragma: nocover
    log.init()
    main()
