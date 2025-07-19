# This is a modified version of TRL's `SFTTrainer` example (https://github.com/huggingface/trl/blob/main/examples/scripts/sft_trainer.py),
# adapted to run with DeepSpeed ZeRO-3 and Mistral-7B-V1.0. The settings below were run on 1 node of 8 x A100 (80GB) GPUs.
#
# Usage:
#   - Install the latest transformers & accelerate versions: `pip install -U transformers accelerate`
#   - Install deepspeed: `pip install deepspeed==0.9.5`
#   - Install TRL from main: pip install git+https://github.com/huggingface/trl.git
#   - Clone the repo: git clone github.com/huggingface/trl.git
#   - Copy this Gist into trl/examples/scripts
#   - Run from root of trl repo with: accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero3.yaml --gradient_accumulation_steps 8 examples/scripts/sft_trainer.py
import gzip
import json
import multiprocessing
import os
import random
from copy import deepcopy
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, Optional

import datasets
import pandas as pd
import torch
from peft import LoraConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    PreTrainedTokenizer,
    TrainingArguments,
)
from transformers.trainer_pt_utils import LabelSmoother
from trl import ORPOConfig, ORPOTrainer, SFTTrainer

from igcs.train_model import train_helpers

tqdm.pandas()

BASE_MODELS = {
    # Llama-3
    "llama-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama-8b": "meta-llama/Meta-Llama-3-8B",
    # Gemma-3
    "gemma-3-1b": "google/gemma-3-1b-it",
    "gemma-3-4b": "google/gemma-3-4b-it",
    "gemma-3-12b": "google/gemma-3-12b-it",
    "gemma-3-27b": "google/gemma-3-27b-it",
    # Qwen
    "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B": "Qwen/Qwen2.5-3B-Instruct",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-14B": "Qwen/Qwen2.5-14B-Instruct",
    "Qwen2.5-32B": "Qwen/Qwen2.5-32B-Instruct",
    "Qwen2.5-72B": "Qwen/Qwen2.5-72B-Instruct",
    # SmolLM2
    "SmolLM2-135M": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "SmolLM2-360M": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "SmolLM2-1.7B": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    # "mistral-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    # "mistral": "mistralai/Mistral-7B-v0.1",
    # "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # "mixtral-instruct": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # "llama-7b": "huggyllama/llama-7b",
    # "phi-3-instruct": "microsoft/Phi-3-mini-128k-instruct",
}


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    # required args
    output_dir: Optional[str] = field(metadata={"help": "the output directory"})
    model_name: Optional[str] = field(
        default=BASE_MODELS["llama-8b-instruct"], metadata={"help": "the model name"}
    )
    train_dataset: Optional[str] = field(
        default="all", metadata={"help": "the dataset to train on"}
    )
    train_sample: Optional[float] = field(
        default=None, metadata={"help": "How many fraction of the dataset to train on"}
    )
    orpo: Optional[bool] = field(default=False, metadata={"help": "enable ORBO training"})
    igcs_answer_format: Optional[bool] = field(
        default=False,
        metadata={"help": "enable specialized IGCS encoding answer format"},
    )

    # main args
    dataset_dir: Optional[str] = field(
        default="./igcs-dataset/prompts", metadata={"help": "the dataset local dir"}
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "Checkpoint path"}
    )

    # Training Hyper-Parameters
    num_train_epochs: Optional[int] = field(
        default=3, metadata={"help": "the number of training epochs"}
    )
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "the learning rate"})
    seq_length: Optional[int] = field(
        default=4096, metadata={"help": "Input sequence length"}  # 32768
    )
    batch_size: Optional[int] = field(default=1, metadata={"help": "the batch size"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "enable gradient checkpointing"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=3, metadata={"help": "the number of gradient accumulation steps"}
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata={"help": "learning rate scheduler type"}
    )
    warmup_ratio: Optional[float] = field(default=0.06, metadata={"help": "the warmup ratio"})
    weight_decay: Optional[float] = field(default=0.1, metadata={"help": "the weight decay"})
    neftune_noise_alpha: Optional[float] = field(
        default=None, metadata={"help": "neftune embeddings noise alpha"}
    )

    # Training / GPU Optimizations
    bf16: Optional[bool] = field(default=True, metadata={"help": "enable bf16 support"})
    tf32: Optional[bool] = field(default=True, metadata={"help": "enable tf32 support"})
    dataloader_num_workers: Optional[int] = field(
        default=8, metadata={"help": "number of dataloader workers"}
    )
    fsdp: Optional[str] = field(default="full_shard auto_wrap", metadata={"help": "fsdp mode"})
    attn_implementation: Optional[str] = "flash_attention_2"

    # PEFT / LoRA / Quantization
    load_in_8bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 8 bits precision"}
    )
    load_in_4bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 4 bits precision"}
    )
    use_peft: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use PEFT or not to train adapters"}
    )
    peft_lora_r: Optional[int] = field(
        default=64, metadata={"help": "the r parameter of the LoRA adapters"}
    )
    peft_lora_alpha: Optional[int] = field(
        default=16, metadata={"help": "the alpha parameter of the LoRA adapters"}
    )

    # Evaluation / Logging
    evaluation_strategy: Optional[str] = field(default="epoch", metadata={"help": "eval strategy"})
    log_with: Optional[str] = field(
        default="wandb", metadata={"help": "use 'wandb' to log with wandb"}  # "wandb"
    )
    logging_steps: Optional[int] = field(
        default=5, metadata={"help": "the number of logging steps"}
    )

    # Other
    trust_remote_code: Optional[bool] = field(
        default=False, metadata={"help": "Enable `trust_remote_code`"}
    )

    use_cache: Optional[bool] = field(default=False, metadata={"help": "use cache for model"})
    dataset_text_field: Optional[str] = field(
        default="text", metadata={"help": "the text field of the dataset"}
    )

    # Save
    save_strategy: Optional[str] = field(
        default="steps", metadata={"help": "save strategy to save checkpoints"}
    )
    overwrite_output_dir: Optional[bool] = field(
        default=True, metadata={"help": "Overwrite output model"}
    )
    save_steps: Optional[int] = field(
        default=1000,
        metadata={"help": "Number of updates steps before two checkpoint saves"},
    )
    save_total_limit: Optional[int] = field(
        default=3, metadata={"help": "Limits total number of checkpoints."}
    )
    push_to_hub: Optional[bool] = field(
        default=False, metadata={"help": "Push the model to HF Hub"}
    )
    hub_model_id: Optional[str] = field(
        default="igcs", metadata={"help": "The name of the model on HF Hub"}
    )


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def get_separator_token(tokenizer: PreTrainedTokenizer) -> str:
    model_name = tokenizer.name_or_path.lower()
    if "llama-3" in model_name:
        return "<|eot_id|>"
    elif "gemma-3" in model_name:
        return "<end_of_turn>"
    elif "qwen2.5" in model_name or "smollm2" in model_name:
        return "<|im_end|>"
    # elif "mistral" in model_name:
    #     return "[/INST]"
    # elif "phi-3" in model_name:
    #     return "<|end|>"
    raise ValueError(f"unknown model {model_name}")


def preprocess(sample: dict, tokenizer: PreTrainedTokenizer, seq_length: int):
    model_name = tokenizer.name_or_path.lower()
    # mistral and phi-3 do not accept system prompts
    if "mistral" in model_name or "phi-3" in model_name:
        conv = sample["conversations"][1:]
    elif "llama-3" in model_name or "qwen2.5" in model_name or "smollm2" in model_name:
        conv = sample["conversations"]
    elif "gemma-3" in model_name:
        conv = deepcopy(sample["conversations"])
        # Merge system and user prompt in Gemma-3, similar to the way the chat template merges them
        conv[1]["content"] = conv[0]["content"] + "\n\n" + conv[1]["content"]
        conv = conv[1:]
    else:
        raise ValueError(f"unknown model {model_name}")

    text = tokenizer.apply_chat_template(conv, tokenize=False)
    output = tokenizer(
        text,
        truncation=True,
        max_length=seq_length,
        add_special_tokens=False,
        padding=False,
        return_tensors=None,
    )

    sep_token_text = get_separator_token(tokenizer)
    instruct_token_ids = tokenizer.encode(sep_token_text, add_special_tokens=False)

    roles = [turn["role"] for turn in conv]
    assert set(roles) <= {"system", "user", "assistant"}, roles

    role_idx = 0
    roles_end_indices = []
    for i in range(len(output["input_ids"])):
        if output["input_ids"][i : i + len(instruct_token_ids)] == instruct_token_ids:
            roles_end_indices.append((roles[role_idx], i))
            role_idx += 1

    if len(roles) > len(roles_end_indices):
        print(f"WARN: ignoring sample - too long {text[:256]}...{text[-256:]}")
    else:
        assert len(roles) == len(roles_end_indices)

    # convert to tensor
    output["input_ids"] = torch.tensor(output["input_ids"])
    output["labels"] = output["input_ids"].clone()

    # ignore system tokens
    start_idx = 0
    for role, end_idx in roles_end_indices:
        if role != "assistant":
            output["labels"][start_idx : end_idx + len(instruct_token_ids)] = IGNORE_TOKEN_ID
        start_idx = end_idx

    output["attention_mask"] = output["labels"].ne(tokenizer.pad_token_id)
    return output


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, seq_length: int, tokenizer: PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.seq_length = seq_length
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess(self.raw_data[i], self.tokenizer, self.seq_length)
        ret = dict(
            input_ids=ret["input_ids"],
            labels=ret["labels"],
            attention_mask=ret["attention_mask"],
        )
        self.cached_data_dict[i] = ret

        return ret


def load_dataset_data(fnames: str | list[str]):
    print(f"Loading dataset from {fnames}")
    if isinstance(fnames, str):
        fnames = [fnames]

    data = []
    for fname in fnames:
        with gzip.open(fname, "rt") as fp:
            for line in fp:
                data.append(json.loads(line))

    random.seed(42)
    random.shuffle(data)
    print(f"found {len(data)} training samples")
    return data


def load_raw_datasets(script_args):
    if script_args.igcs_answer_format:
        prefix_name = ""
    elif script_args.orpo:
        prefix_name = "orpo-"
    else:
        prefix_name = "prompts-"
    if script_args.train_dataset == "all":
        train_data = load_dataset_data(
            os.path.join(script_args.dataset_dir, f"{prefix_name}train-all.jsonl.gz")
        )

        if script_args.train_sample is not None and script_args.train_sample < 1:
            all_train_df = pd.DataFrame(train_data)
            sampled_train_data_df, _ = train_test_split(
                all_train_df,
                train_size=script_args.train_sample,
                random_state=42,
                stratify=all_train_df["sample_id"].apply(
                    lambda x: x.split(".")[0] if "." in x else "NEG"
                ),
            )
            train_data = sampled_train_data_df.to_dict(orient="records")
            print(f"using {len(train_data)}/{len(all_train_df)} samples")

    else:
        if script_args.train_sample:
            print("WARNING: cannot set train_sample without all - ignored")
        train_data = load_dataset_data(
            fnames=[
                os.path.join(
                    script_args.dataset_dir,
                    "train",
                    f"{prefix_name}train-{name}.jsonl.gz",
                )
                for name in script_args.train_dataset.split(",")
            ]
        )

    eval_data = {
        os.path.basename(ds_fname).split(".")[0]: load_dataset_data(ds_fname)
        for ds_fname in glob(os.path.join(script_args.dataset_dir, "dev", "*.jsonl.gz"))
    }

    return train_data, eval_data


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    print(script_args)

    # Step 1: Load the dataset
    if "llama-3" in script_args.model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
    elif "phi-3" in script_args.model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    elif "mistral" in script_args.model_name.lower():
        # tokenizer should have pad_token, mistral doesn't have one, so we find one
        tokenizer = AutoTokenizer.from_pretrained(
            script_args.model_name,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
        )
        tokenizer.pad_token = tokenizer.unk_token
    else:
        # Gemma3 and Qwen2.5 and Smollm2
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)

    train_data, eval_data = load_raw_datasets(script_args)

    if script_args.igcs_answer_format:
        train_data = [
            train_helpers.igcs_format_sample(s, model_name=script_args.model_name)
            for s in train_data
        ]
        eval_data = {
            name: [
                train_helpers.igcs_format_sample(s, model_name=script_args.model_name)
                for s in edata
            ]
            for name, edata in eval_data.items()
        }
        print(train_data[0])
        print(list(eval_data.values())[0][0])

    if script_args.orpo:

        def orpo_process(row):
            row["prompt"] = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}], tokenize=False
            )
            row["chosen"] = tokenizer.apply_chat_template(
                [{"role": "assistant", "content": row["chosen"]}], tokenize=False
            ).removeprefix("<|begin_of_text|>")
            row["rejected"] = tokenizer.apply_chat_template(
                [{"role": "assistant", "content": row["rejected"]}], tokenize=False
            ).removeprefix("<|begin_of_text|>")
            return row

        train_dataset = datasets.Dataset.from_list(train_data).map(
            orpo_process, num_proc=multiprocessing.cpu_count()
        )
        eval_datasets = datasets.DatasetDict(
            {
                name: datasets.Dataset.from_list(edata).map(
                    orpo_process, num_proc=multiprocessing.cpu_count()
                )
                for name, edata in eval_data.items()
            }
        )
    else:

        def make_dataset(data):
            return LazySupervisedDataset(
                raw_data=data,
                seq_length=script_args.seq_length,
                tokenizer=tokenizer,
            )

        train_dataset = make_dataset(train_data)
        eval_datasets = {name: make_dataset(edata) for name, edata in eval_data.items()}

    print(f"training dataset: {train_dataset}")
    print(f"eval datasets: {eval_datasets}")

    # Step 2: Load the model
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        print("enabling quantization")
        from accelerate import Accelerator
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = torch.bfloat16 if script_args.bf16 else None

    if "gemma-3" in script_args.model_name.lower():
        from transformers import Gemma3ForCausalLM

        model = Gemma3ForCausalLM(
            script_args.model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=script_args.trust_remote_code,
            torch_dtype=torch_dtype,
            attn_implementation=script_args.attn_implementation,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=script_args.trust_remote_code,
            torch_dtype=torch_dtype,
            use_cache=script_args.use_cache,
            attn_implementation=script_args.attn_implementation,
        )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Step 3: Define the training arguments
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        resume_from_checkpoint=script_args.resume_from_checkpoint,
        overwrite_output_dir=script_args.overwrite_output_dir,
        per_device_train_batch_size=script_args.batch_size,
        per_device_eval_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if script_args.gradient_checkpointing else None
        ),
        learning_rate=script_args.learning_rate,
        weight_decay=script_args.weight_decay,
        warmup_ratio=script_args.warmup_ratio,
        lr_scheduler_type=script_args.lr_scheduler_type,
        # fsdp_config=get_fsdp_config(script_args.fsdp, script_args.model_name),
        # fsdp=script_args.fsdp,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        report_to=script_args.log_with,
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        bf16=script_args.bf16,
        tf32=script_args.tf32,
        evaluation_strategy=script_args.evaluation_strategy,
        eval_accumulation_steps=1,
        logging_first_step=True,
        log_level="info",
        neftune_noise_alpha=script_args.neftune_noise_alpha,
    )

    # Step 4: Define the LoraConfig
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.peft_lora_r,
            lora_alpha=script_args.peft_lora_alpha,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    # Step 5: Define the Trainer
    if script_args.orpo:
        orpo_config = ORPOConfig(
            beta=0.1,
            **training_args.to_dict(),
            max_length=(script_args.seq_length + 2048) * script_args.batch_size,
            max_prompt_length=script_args.seq_length,
            max_completion_length=2048,
        )
        trainer = ORPOTrainer(
            model=model,
            tokenizer=tokenizer,
            args=orpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_datasets,
            # data_collator=DataCollatorForSeq2Seq(
            #     tokenizer=tokenizer, max_length=script_args.seq_length
            # ),
            peft_config=peft_config,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=train_dataset,
            eval_dataset=eval_datasets,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=tokenizer, max_length=script_args.seq_length
            ),
            dataset_text_field=script_args.dataset_text_field,
            peft_config=peft_config,
            dataset_kwargs={"skip_prepare_dataset": True},
            # packing=False,
        )

    trainer.train()

    # Step 6: Save the model
    trainer.save_model(script_args.output_dir)


if __name__ == "__main__":  # pragma: nocover
    main()
