"""Stage-specific LoRA fine-tuning."""

import argparse
import random

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

from common import (
    KTAS_LEVELS, MAX_SEQ_LENGTH, MODEL_CONFIGS, load_cases, load_label_hierarchy, load_model, load_tokenizer,
)
from prompts import build_prompt

SEED = 42
BATCH_SIZE = 32
STAGE_HPARAMS = {
    "major": {"epochs": 3, "learning_rate": 5e-5, "rank": 16, "alpha": 32, "dropout": 0.1, "max_grad_norm": 1.0},
    "minor": {"epochs": 3, "learning_rate": 5e-5, "rank": 16, "alpha": 32, "dropout": 0.1, "max_grad_norm": 1.0},
    "ktas": {"epochs": 6, "learning_rate": 8e-5, "rank": 32, "alpha": 64, "dropout": 0.05, "max_grad_norm": 0.5},
}
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_examples(stage: str, cases: list[dict], hierarchy: dict[str, list[str]], tokenizer) -> Dataset:
    rng = random.Random(SEED)
    examples = []
    for case in cases:
        if stage == "major":
            candidates, target = list(hierarchy), case["major"]
            rng.shuffle(candidates)
        elif stage == "minor":
            candidates, target = list(hierarchy[case["major"]]), case["minor"]
            rng.shuffle(candidates)
        else:
            candidates, target = [str(level) for level in KTAS_LEVELS], str(case["ktas"])
        prompt = build_prompt(stage, case["text"], candidates, case["major"], case["minor"])

        prompt_ids = tokenizer(prompt)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        if len(prompt_ids) + len(target_ids) > MAX_SEQ_LENGTH:
            raise ValueError(f"Training example exceeds {MAX_SEQ_LENGTH} tokens")
        examples.append({
            "input_ids": prompt_ids + target_ids,
            "attention_mask": [1] * (len(prompt_ids) + len(target_ids)),
            "labels": [-100] * len(prompt_ids) + target_ids,
        })
    return Dataset.from_list(examples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=list(STAGE_HPARAMS))
    parser.add_argument("--model_type", required=True, choices=list(MODEL_CONFIGS))
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--label_file", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    hp = STAGE_HPARAMS[args.stage]

    set_seed(SEED)
    tokenizer = load_tokenizer(args.model_type)
    hierarchy = load_label_hierarchy(args.label_file)
    dataset = build_examples(args.stage, load_cases(args.train_file, hierarchy), hierarchy, tokenizer)
    if args.model_type == "medgemma-4b":
        dataset = dataset.map(lambda example: {"token_type_ids": [0] * len(example["input_ids"])})

    model = load_model(args.model_type)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=hp["rank"],
        lora_alpha=hp["alpha"],
        lora_dropout=hp["dropout"],
        target_modules=LORA_TARGET_MODULES,
    ))
    model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=hp["epochs"],
        learning_rate=hp["learning_rate"],
        max_grad_norm=hp["max_grad_norm"],
        per_device_train_batch_size=BATCH_SIZE // 2,
        gradient_accumulation_steps=2,
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        weight_decay=0.01,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no",
        logging_steps=10,
        seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
