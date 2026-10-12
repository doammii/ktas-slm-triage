"""Point estimates and bootstrap 95% CIs for the standalone evaluation."""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

N_BOOTSTRAP = 5000
SEED = 42
KTAS_LEVELS = [1, 2, 3, 4, 5]
STAGES = ("major", "minor", "ktas")
TRUE_COLUMNS = ["case_index", "true_major", "true_minor", "true_ktas"]
METRICS = [
    "major_top1", "major_top3", "major_macro_f1", "major_weighted_f1",
    "minor_top1", "minor_top3", "minor_macro_f1", "minor_weighted_f1",
    "joint_top1",
    "ktas_top1", "ktas_top2", "ktas_macro_f1", "ktas_weighted_f1", "ktas_qwk", "over_triage", "under_triage",
    "exact_match",
]


class Run:
    def __init__(self, path: str) -> None:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        self.truth = df[TRUE_COLUMNS]
        self.true = {stage: df[f"true_{stage}"].to_numpy() for stage in STAGES}
        self.pred = {stage: df[f"pred_{stage}_1"].to_numpy() for stage in STAGES}
        self.true["ktas"], self.pred["ktas"] = self.true["ktas"].astype(int), self.pred["ktas"].astype(int)
        self.labels = {stage: sorted(set(self.true[stage])) for stage in STAGES}

        def in_top(stage: str, k: int) -> np.ndarray:
            preds = df[[f"pred_{stage}_{i}" for i in range(1, k + 1)]].to_numpy()
            return (preds == df[f"true_{stage}"].to_numpy()[:, None]).any(axis=1)

        major_ok, minor_ok, ktas_ok = in_top("major", 1), in_top("minor", 1), in_top("ktas", 1)
        self.indicators = {
            "major_top1": major_ok,
            "major_top3": in_top("major", 3),
            "minor_top1": minor_ok,
            "minor_top3": in_top("minor", 3),
            "joint_top1": major_ok & minor_ok,
            "ktas_top1": ktas_ok,
            "ktas_top2": in_top("ktas", 2),
            "over_triage": self.pred["ktas"] < self.true["ktas"],
            "under_triage": self.pred["ktas"] > self.true["ktas"],
            "exact_match": major_ok & minor_ok & ktas_ok,
        }

    def _f1(self, stage: str, idx: np.ndarray) -> tuple[float, float]:
        _, _, f1, support = precision_recall_fscore_support(
            self.true[stage][idx], self.pred[stage][idx], labels=self.labels[stage], zero_division=0
        )
        return float(f1.mean()), float(np.average(f1, weights=support))

    def metrics(self, idx: np.ndarray, minor_strata_idx: np.ndarray) -> dict[str, float]:
        out = {name: float(values[idx].mean()) for name, values in self.indicators.items()}
        out["major_macro_f1"], out["major_weighted_f1"] = self._f1("major", idx)
        _, out["minor_weighted_f1"] = self._f1("minor", idx)
        out["minor_macro_f1"], _ = self._f1("minor", minor_strata_idx)
        out["ktas_macro_f1"], out["ktas_weighted_f1"] = self._f1("ktas", idx)
        out["ktas_qwk"] = float(cohen_kappa_score(
            self.true["ktas"][idx], self.pred["ktas"][idx], labels=KTAS_LEVELS, weights="quadratic"
        ))
        return out


def mean_over_runs(runs: list[Run], idx: np.ndarray, minor_strata_idx: np.ndarray) -> dict[str, float]:
    per_run = [run.metrics(idx, minor_strata_idx) for run in runs]
    return {name: float(np.mean([m[name] for m in per_run])) for name in per_run[0]}


def load_runs(paths: list[str]) -> list[Run]:
    runs = [Run(path) for path in paths]
    if any(not run.truth.equals(runs[0].truth) for run in runs):
        raise ValueError("All runs must contain the same cases in the same order")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--baseline_runs", nargs="+")
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    configs = {"estimate": load_runs(args.runs)}
    if args.baseline_runs:
        configs["baseline"] = load_runs(args.baseline_runs)
        if not configs["baseline"][0].truth.equals(configs["estimate"][0].truth):
            raise ValueError("Paired configurations must contain the same cases in the same order")

    n = len(configs["estimate"][0].truth)
    reference = configs["estimate"][0]
    minor_strata = [np.flatnonzero(reference.true["minor"] == label) for label in reference.labels["minor"]]
    full = np.arange(n)
    point = {name: mean_over_runs(runs, full, full) for name, runs in configs.items()}

    rng = np.random.default_rng(SEED)
    replicates = {name: [] for name in configs}
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        minor_strata_idx = np.concatenate([m[rng.integers(0, len(m), len(m))] for m in minor_strata])
        for name, runs in configs.items():
            replicates[name].append(mean_over_runs(runs, idx, minor_strata_idx))

    table = []
    for metric in METRICS:
        row = {"metric": metric}
        draws = {name: np.array([r[metric] for r in replicates[name]]) for name in configs}
        for name in configs:
            row[name] = point[name][metric]
            row[f"{name}_ci_lower"], row[f"{name}_ci_upper"] = np.percentile(draws[name], [2.5, 97.5])
        if "baseline" in configs:
            row["difference"] = point["estimate"][metric] - point["baseline"][metric]
            differences = draws["estimate"] - draws["baseline"]
            row["difference_ci_lower"], row["difference_ci_upper"] = np.percentile(differences, [2.5, 97.5])
        table.append(row)

    result = pd.DataFrame(table)
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    result.to_csv(args.output_file, index=False)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
