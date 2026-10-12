"""Hierarchical inference with constrained candidate scoring."""

import argparse
import os

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm

from common import (
    KTAS_LEVELS, MAX_SEQ_LENGTH, MODEL_CONFIGS, load_cases, load_label_hierarchy, load_model, load_tokenizer,
)
from prompts import build_prompt

TEMPERATURE = 0.01
TOP_K = {"major": 3, "minor": 3, "ktas": 2}
STAGES = ("major", "minor", "ktas")


class CandidateScorer:
    def __init__(self, model, tokenizer, use_adapters: bool) -> None:
        self.model, self.tokenizer, self.use_adapters = model, tokenizer, use_adapters

    @torch.inference_mode()
    def rank(self, stage: str, prompt: str, candidates: list[str], batch_size: int = 32) -> list[str]:
        if self.use_adapters:
            self.model.set_adapter(stage)
        prompt_ids = self.tokenizer(prompt)["input_ids"]
        candidate_ids = [self.tokenizer(c, add_special_tokens=False)["input_ids"] for c in candidates]
        if len(prompt_ids) + max(map(len, candidate_ids)) > MAX_SEQ_LENGTH:
            raise ValueError(f"Prompt and candidate exceed {MAX_SEQ_LENGTH} tokens")

        start = len(prompt_ids) - 1
        scores = []
        for i in range(0, len(candidates), batch_size):
            chunk = candidate_ids[i:i + batch_size]
            sequences = [prompt_ids + ids for ids in chunk]
            batch = self.tokenizer.pad(
                [{"input_ids": seq, "attention_mask": [1] * len(seq)} for seq in sequences], return_tensors="pt"
            ).to(next(self.model.parameters()).device)
            logits = self.model(**batch).logits
            for row, ids in enumerate(chunk):
                log_probs = torch.log_softmax(logits[row, start:start + len(ids)] / TEMPERATURE, dim=-1)
                positions = torch.arange(len(ids), device=log_probs.device)
                token_scores = log_probs[positions, torch.tensor(ids, device=log_probs.device)]
                scores.append(token_scores.mean().item())
        order = sorted(range(len(candidates)), key=lambda j: scores[j], reverse=True)
        return [candidates[j] for j in order]


def predict(scorer: CandidateScorer, case: dict, hierarchy: dict[str, list[str]]) -> dict:
    text = case["text"]
    majors = scorer.rank("major", build_prompt("major", text, list(hierarchy)), list(hierarchy))
    branch = hierarchy[majors[0]]
    minors = scorer.rank("minor", build_prompt("minor", text, branch, majors[0]), branch)
    levels = [str(level) for level in KTAS_LEVELS]
    ktas = scorer.rank("ktas", build_prompt("ktas", text, levels, majors[0], minors[0]), levels)

    row = {"true_major": case["major"], "true_minor": case["minor"], "true_ktas": case["ktas"]}
    for stage, ranked in zip(STAGES, (majors, minors, ktas)):
        for k in range(TOP_K[stage]):
            row[f"pred_{stage}_{k + 1}"] = ranked[k] if k < len(ranked) else ""
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", required=True, choices=list(MODEL_CONFIGS))
    parser.add_argument("--adapter_dir")
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--label_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_type)
    model = load_model(args.model_type, device_map="auto")
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, os.path.join(args.adapter_dir, "major"), adapter_name="major")
        for stage in ("minor", "ktas"):
            model.load_adapter(os.path.join(args.adapter_dir, stage), adapter_name=stage)
    model.eval()

    scorer = CandidateScorer(model, tokenizer, use_adapters=bool(args.adapter_dir))
    hierarchy = load_label_hierarchy(args.label_file)
    cases = load_cases(args.test_file, hierarchy)
    rows = [predict(scorer, case, hierarchy) for case in tqdm(cases, desc="Inference")]

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    pd.DataFrame(rows, index=pd.RangeIndex(1, len(rows) + 1, name="case_index")).to_csv(
        args.output_file, encoding="utf-8-sig"
    )


if __name__ == "__main__":
    main()
