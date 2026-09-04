#!/usr/bin/env python3
# Run: python -u run_structure_baselines.py --config config_structure.json
"""Compare DeepSPY with zero-shot Chai-1 and AlphaFold3 structure scores."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)



METRICS = [
    "mcc", "accuracy", "balanced_accuracy", "roc_auc", "auprc", "average_precision",
    "precision", "recall", "specificity", "f1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_structure.json")))
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def canonical_pair(protein1: str, protein2: str) -> str:
    return f"{protein1.strip()}_{protein2.strip()}".lower()


def load_deepspy(prediction_file: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for row in read_tsv(prediction_file):
        pair = canonical_pair(row["protein1_id"], row["protein2_id"])
        if pair in records:
            raise ValueError(f"Duplicate DeepSPY OOF pair: {pair}")
        records[pair] = {
            "fold": int(row["fold"]),
            "y_true": int(row["true_label"]),
            "score": float(row["predicted_probability"]),
            "prediction": int(row["predicted_label"]),
        }
    if len(records) != 267:
        raise ValueError(f"Expected 267 unique DeepSPY OOF pairs, found {len(records)}")
    return records


def load_structure_scores(path: Path, threshold: float) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for row in read_tsv(path):
        pair = row["pair"].strip().lower()
        if pair in records:
            raise ValueError(f"Duplicate structure pair in {path}: {pair}")
        iptm = float(row["iptm"])
        prediction = int(iptm >= threshold)
        if "Binary_Label" in row and int(row["Binary_Label"]) != prediction:
            raise ValueError(f"{path}/{pair}: Binary_Label is inconsistent with ipTM threshold {threshold}")
        records[pair] = {
            "iptm": iptm,
            "ptm": float(row["ptm"]),
            "ranking_score": float(row["ranking_score"]),
            "prediction": prediction,
        }
    return records


def metrics(y_true: np.ndarray, score: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    ap = float(average_precision_score(y_true, score))
    return {
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "auprc": ap,
        "average_precision": ap,
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def aligned_arrays(
    pairs: Sequence[str], deep: dict[str, dict[str, Any]], source: dict[str, dict[str, Any]], score: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([deep[pair]["y_true"] for pair in pairs], dtype=np.int64),
        np.asarray([source[pair][score] for pair in pairs], dtype=np.float64),
        np.asarray([source[pair]["prediction"] for pair in pairs], dtype=np.int64),
    )


def evaluate_model(
    model_name: str,
    deep: dict[str, dict[str, Any]],
    source: dict[str, dict[str, Any]],
    score_field: str,
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fold_rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        pairs = sorted(pair for pair, row in deep.items() if row["fold"] == fold)
        y_true, scores, predictions = aligned_arrays(pairs, deep, source, score_field)
        fold_rows.append({
            "model": model_name, "score": score_field, "threshold": threshold,
            "fold": fold, "n": len(pairs), "n_positive": int(y_true.sum()),
            **metrics(y_true, scores, predictions),
        })

    pairs = sorted(deep)
    y_true, scores, predictions = aligned_arrays(pairs, deep, source, score_field)
    pooled = metrics(y_true, scores, predictions)
    summary: dict[str, Any] = {
        "model": model_name, "score": score_field, "threshold": threshold,
        "n": len(pairs), "n_positive": int(y_true.sum()), "n_folds": 5,
    }
    for metric_name in METRICS:
        values = np.asarray([row[metric_name] for row in fold_rows], dtype=np.float64)
        summary[f"{metric_name}_mean"] = float(values.mean())
        summary[f"{metric_name}_sd"] = float(values.std(ddof=1))
        summary[f"{metric_name}_pooled"] = pooled[metric_name]
    for field in ["tn", "fp", "fn", "tp"]:
        summary[f"{field}_pooled"] = pooled[field]
    return fold_rows, summary


def evaluate_deepspy(deep: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = {
        pair: {"score": row["score"], "prediction": row["prediction"]}
        for pair, row in deep.items()
    }
    return evaluate_model("DeepSPY", deep, source, "score", 0.5)


def score_sensitivity(
    name: str, deep: dict[str, dict[str, Any]], source: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for score_field in ["iptm", "ranking_score", "ptm"]:
        fold_auc: list[float] = []
        fold_ap: list[float] = []
        for fold in range(1, 6):
            pairs = sorted(pair for pair, row in deep.items() if row["fold"] == fold)
            y_true = np.asarray([deep[pair]["y_true"] for pair in pairs])
            scores = np.asarray([source[pair][score_field] for pair in pairs])
            fold_auc.append(float(roc_auc_score(y_true, scores)))
            fold_ap.append(float(average_precision_score(y_true, scores)))
        pairs = sorted(deep)
        y_true = np.asarray([deep[pair]["y_true"] for pair in pairs])
        scores = np.asarray([source[pair][score_field] for pair in pairs])
        rows.append({
            "model": name, "score": score_field,
            "roc_auc_mean": float(np.mean(fold_auc)),
            "roc_auc_sd": float(np.std(fold_auc, ddof=1)),
            "roc_auc_pooled": float(roc_auc_score(y_true, scores)),
            "average_precision_mean": float(np.mean(fold_ap)),
            "average_precision_sd": float(np.std(fold_ap, ddof=1)),
            "average_precision_pooled": float(average_precision_score(y_true, scores)),
        })
    return rows


def make_aligned_rows(
    deep: dict[str, dict[str, Any]],
    chai: dict[str, dict[str, Any]],
    af3: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in sorted(deep, key=lambda item: (deep[item]["fold"], item)):
        rows.append({
            "pair": pair, "fold": deep[pair]["fold"], "y_true": deep[pair]["y_true"],
            "deepspy_score": deep[pair]["score"], "deepspy_prediction": deep[pair]["prediction"],
            "chai1_iptm": chai[pair]["iptm"], "chai1_ptm": chai[pair]["ptm"],
            "chai1_ranking_score": chai[pair]["ranking_score"],
            "chai1_prediction": chai[pair]["prediction"],
            "alphafold3_iptm": af3[pair]["iptm"], "alphafold3_ptm": af3[pair]["ptm"],
            "alphafold3_ranking_score": af3[pair]["ranking_score"],
            "alphafold3_prediction": af3[pair]["prediction"],
        })
    return rows


def make_plots(run_dir: Path, aligned: Sequence[dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
    except Exception as exc:
        print(f"Plots skipped: {exc}")
        return

    y_true = np.asarray([row["y_true"] for row in aligned])
    series = [
        ("DeepSPY", np.asarray([row["deepspy_score"] for row in aligned])),
        ("Chai-1 ipTM", np.asarray([row["chai1_iptm"] for row in aligned])),
        ("AlphaFold3 ipTM", np.asarray([row["alphafold3_iptm"] for row in aligned])),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, scores in series:
        RocCurveDisplay.from_predictions(y_true, scores, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_true, scores, name=name, ax=axes[1])
    axes[0].set_title("Pooled OOF ROC")
    axes[1].set_title("Pooled OOF precision-recall")
    fig.tight_layout()
    fig.savefig(run_dir / "deepspy_chai1_alphafold3_curves.png", dpi=220)
    plt.close(fig)


def write_report(run_dir: Path, summaries: Sequence[dict[str, Any]], threshold: float) -> None:
    by_name = {row["model"]: row for row in summaries}
    lines = [
        "# Matched-pair comparison of DeepSPY, Chai-1, and AlphaFold 3", "",
        "## Comparison protocol", "",
        "- Data: 267 Gold pairs (53 positive and 214 negative).",
        "- Chai-1 and AlphaFold 3 cover all 267 pairs without missing or duplicate entries.",
        f"- Structural baseline score: ipTM; fixed classification threshold: ipTM >= {threshold:g}.",
        "- Results from the older 366-pair dataset were not used.",
        "- Chai-1 uses the maximum ipTM among five candidate models per pair; AlphaFold 3 uses its selected top model.",
        "- The table reports fivefold means; pooled and per-fold metrics are provided as TSV files.",
        "", "## Main results", "",
        "| Model | MCC | Accuracy | AUROC | AP | Precision | Recall | Specificity | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["DeepSPY", "Chai-1", "AlphaFold3"]:
        row = by_name[name]
        lines.append(
            f"| {name} | {row['mcc_mean']:.3f} | {row['accuracy_mean']:.3f} | "
            f"{row['roc_auc_mean']:.3f} | {row['average_precision_mean']:.3f} | "
            f"{row['precision_mean']:.3f} | {row['recall_mean']:.3f} | "
            f"{row['specificity_mean']:.3f} | {row['f1_mean']:.3f} |"
        )
    deep = by_name["DeepSPY"]
    chai = by_name["Chai-1"]
    af3 = by_name["AlphaFold3"]
    lines.extend([
        "", "## Interpretation", "",
        f"DeepSPY achieved a mean MCC of {deep['mcc_mean']:.3f}, compared with {chai['mcc_mean']:.3f} for Chai-1 and {af3['mcc_mean']:.3f} for AlphaFold 3.",
        f"At the fixed threshold, Chai-1 yielded pooled TP={chai['tp_pooled']} and FN={chai['fn_pooled']}; AlphaFold 3 yielded TP={af3['tp_pooled']} and FN={af3['fn_pooled']}.",
        "ipTM is a structural confidence score and not a recognition probability calibrated on this dataset. These methods are therefore presented as zero-shot structural baselines.",
        "pTM mainly reflects global structural confidence; score_sensitivity.tsv reports sensitivity analyses for pTM and ranking score.",
    ])
    (run_dir / "STRUCTURE_BASELINE_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> Path:
    args = parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.output_dir:
        config["output_dir"] = args.output_dir
    threshold = float(config["iptm_threshold"])

    deep = load_deepspy(Path(config["deep_predictions"]))
    chai_all = load_structure_scores(Path(config["chai_summary"]), threshold)
    af3_all = load_structure_scores(Path(config["alphafold3_summary"]), threshold)
    required = set(deep)
    for name, source in [("Chai-1", chai_all), ("AlphaFold3", af3_all)]:
        missing = required.difference(source)
        if missing:
            raise ValueError(f"{name} is missing {len(missing)} Gold pairs: {sorted(missing)[:10]}")
    chai = {pair: chai_all[pair] for pair in deep}
    af3 = {pair: af3_all[pair] for pair in deep}

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["output_dir"]) / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    all_fold_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    deep_folds, deep_summary = evaluate_deepspy(deep)
    all_fold_rows.extend(deep_folds)
    summaries.append(deep_summary)
    for name, source in [("Chai-1", chai), ("AlphaFold3", af3)]:
        fold_rows, summary = evaluate_model(name, deep, source, "iptm", threshold)
        all_fold_rows.extend(fold_rows)
        summaries.append(summary)

    aligned = make_aligned_rows(deep, chai, af3)
    sensitivity = score_sensitivity("Chai-1", deep, chai) + score_sensitivity("AlphaFold3", deep, af3)
    write_tsv(run_dir / "comparison_summary.tsv", summaries)
    write_tsv(run_dir / "per_fold_metrics.tsv", all_fold_rows)
    write_tsv(run_dir / "aligned_predictions.tsv", aligned)
    write_tsv(run_dir / "score_sensitivity.tsv", sensitivity)
    make_plots(run_dir, aligned)
    write_report(run_dir, summaries, threshold)

    audit = {
        "n_gold_pairs": len(deep),
        "n_positive": sum(row["y_true"] for row in deep.values()),
        "chai_total_pairs": len(chai_all), "chai_gold_overlap": len(chai),
        "alphafold3_total_pairs": len(af3_all), "alphafold3_gold_overlap": len(af3),
        "primary_structure_score": "iptm", "iptm_threshold": threshold,
        "config": config,
    }
    (run_dir / "comparison_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    source_dir = run_dir / "source_code"
    source_dir.mkdir()
    for filename in [
        "run_structure_baselines.py",
        "config_ml.json", "config_structure.json",
    ]:
        source = Path(__file__).with_name(filename)
        if source.exists():
            shutil.copy2(source, source_dir / filename)
    print(f"Completed: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
