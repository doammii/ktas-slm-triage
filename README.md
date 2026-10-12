# KTAS Triage Prediction with Small Language Models

Stage-specific LoRA fine-tuning, hierarchical inference, and standalone evaluation for
Major Category → Minor Category → KTAS Level prediction. For research use only.

| File | Purpose |
|---|---|
| `train.py` | Stage-specific LoRA fine-tuning |
| `inference.py` | Hierarchical inference with constrained candidate scoring |
| `evaluate.py` | Point estimates and bootstrap 95% CIs |
| `prompts.py` | Stage-specific Korean prompt templates |
| `common.py` | Model and data loading |

## Setup

Reference environment: NVIDIA H100 SXM 80GB, CUDA 12.4, Python 3.11.

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

Some checkpoints are gated on Hugging Face. Run `hf auth login` or set `HF_TOKEN`.

| `--model_type` | Checkpoint |
|---|---|
| `medgemma-4b` | `google/medgemma-4b-it` |
| `qwen3-8b` | `Qwen/Qwen3-8B` |
| `hari-q3-8b` | `snuh/hari-q3-8b` |
| `llama-3.1-8b` | `meta-llama/Llama-3.1-8B` |

## Data

The datasets are not included.

| Input | Format | Columns |
|---|---|---|
| Label hierarchy | `.xlsx`, one row per Major–Minor pair | `대분류`, `중분류` |
| Description dataset | `.csv` or `.xlsx` | `환자 상태 설명`, `대분류`, `중분류`, `KTAS 레벨` |
| Conversation dataset | `.csv` or `.xlsx` | `챗GPT와 대화한 내용`, `대분류`, `중분류`, `KTAS 레벨` |

`챗GPT와 대화한 내용` (also accepted: `챗GPT와의 대화 내용`) contains a JSON array of
`{turn, speaker, utterance}` objects, with `speaker` set to `I` (triage clinician) or `CHATGPT` (informant).

## Fine-tuning

```bash
for stage in major minor ktas; do
  python train.py --stage $stage --model_type hari-q3-8b \
    --train_file data/description.xlsx --label_file data/label_hierarchy.xlsx \
    --output_dir outputs/hari-q3-8b/$stage
done
```

## Inference

```bash
for run in 1 2 3; do
  python inference.py --model_type hari-q3-8b --adapter_dir outputs/hari-q3-8b \
    --test_file data/conversation.csv --label_file data/label_hierarchy.xlsx \
    --output_file results/hari-q3-8b_finetuned_run$run.csv
  python inference.py --model_type hari-q3-8b \
    --test_file data/conversation.csv --label_file data/label_hierarchy.xlsx \
    --output_file results/hari-q3-8b_base_run$run.csv
done
```

Omitting `--adapter_dir` runs the base model.

## Evaluation

```bash
python evaluate.py --runs results/hari-q3-8b_finetuned_run{1,2,3}.csv \
  --baseline_runs results/hari-q3-8b_base_run{1,2,3}.csv \
  --output_file results/hari-q3-8b_summary.csv
```

`--baseline_runs` is optional and adds the paired difference (runs minus baseline runs).

## Data availability

The fine-tuned `hari-q3-8b` LoRA adapters (`major`, `minor`, and `ktas`) and the label hierarchy are available from the
corresponding author on reasonable request. The description and conversation datasets are available from the
corresponding author on reasonable request, subject to approval by the Institutional Review Board of Soonchunhyang
University Hospital Seoul.

## Citation

If you use this code, please cite the release you used. Citation metadata, including authors, title, version, and
release date, are maintained in `CITATION.cff` and are also available through **Cite this repository** on GitHub.

## License

BSD 2-Clause License. See `LICENSE`.
