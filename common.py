"""Model and data loading."""

import json

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

MODEL_CONFIGS = {
    "medgemma-4b": ("google/medgemma-4b-it", AutoModelForImageTextToText),
    "qwen3-8b": ("Qwen/Qwen3-8B", AutoModelForCausalLM),
    "hari-q3-8b": ("snuh/hari-q3-8b", AutoModelForCausalLM),
    "llama-3.1-8b": ("meta-llama/Llama-3.1-8B", AutoModelForCausalLM),
}
MAX_SEQ_LENGTH = 2048
KTAS_LEVELS = (1, 2, 3, 4, 5)
SPEAKER_ORDER = {"I": 0, "CHATGPT": 1}


def load_tokenizer(model_type: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS[model_type][0], padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_type: str, **kwargs):
    path, model_class = MODEL_CONFIGS[model_type]
    return model_class.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa", **kwargs)


def normalize_category(value) -> str:
    return str(value).replace("/", "_").replace(",", "_").strip()


def load_label_hierarchy(path: str) -> dict[str, list[str]]:
    df = pd.read_excel(path)
    pairs = {(normalize_category(a), normalize_category(b)) for a, b in zip(df["대분류"], df["중분류"])}
    return {major: sorted(m for a, m in pairs if a == major) for major in sorted({a for a, _ in pairs})}


def conversation_to_text(conversation: str) -> str:
    turns: dict[int, list[tuple[int, int, str]]] = {}
    for position, item in enumerate(json.loads(conversation)):
        entry = (SPEAKER_ORDER[item["speaker"]], position, str(item["utterance"]).strip())
        turns.setdefault(int(item["turn"]), []).append(entry)
    return "\n\n".join("\n".join(text for *_, text in sorted(turns[t])) for t in sorted(turns))


def load_cases(path: str, hierarchy: dict[str, list[str]]) -> list[dict]:
    if path.lower().endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="cp949")

    if "환자 상태 설명" in df:
        texts = [str(text).strip() for text in df["환자 상태 설명"]]
    else:
        column = next((name for name in ("챗GPT와 대화한 내용", "챗GPT와의 대화 내용") if name in df), None)
        if column is None:
            raise ValueError(f"{path}: missing patient description or conversation column")
        texts = [conversation_to_text(text) for text in df[column]]

    cases = []
    for row, (text, major, minor, level) in enumerate(zip(texts, df["대분류"], df["중분류"], df["KTAS 레벨"]), 2):
        major, minor, level = normalize_category(major), normalize_category(minor), int(float(level))
        if minor not in hierarchy.get(major, []) or level not in KTAS_LEVELS:
            raise ValueError(f"{path}, row {row}: invalid label {major}, {minor}, {level}")
        cases.append({"text": text, "major": major, "minor": minor, "ktas": level})
    return cases
