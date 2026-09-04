#!/usr/bin/env python3
# Run: python -u run_ml_baselines.py --config config_ml.json
"""Leakage-controlled classical ML baselines for the 267 Gold-only pairs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import FixedFeatureStore, PairRecord, read_pairs_tsv


SPLIT_MODES = {"existing_folds", "pair_stratified", "protein1_grouped", "protein2_grouped"}
METRIC_NAMES = [
    "mcc", "accuracy", "balanced_accuracy", "roc_auc", "auprc", "average_precision",
    "precision", "recall", "specificity", "f1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_ml.json")))
    parser.add_argument("--split-mode", choices=sorted(SPLIT_MODES), default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--feature-sets", nargs="+", choices=sorted(FixedFeatureStore.VALID_FEATURE_SETS),
        default=None, help="Optional subset of configured fixed feature sets",
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=["dummy_prior", "logistic_regression", "linear_svm", "random_forest"],
        default=None, help="Optional subset of configured classifiers",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs/features/splits without fitting models")
    return parser.parse_args()


def load_config(path: str | Path, args: argparse.Namespace) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if args.split_mode:
        config["experiment"]["split_mode"] = args.split_mode
    if args.output_dir:
        config["experiment"]["output_dir"] = args.output_dir
    if args.feature_sets:
        config["feature_sets"] = args.feature_sets
    if args.models:
        config["models"] = args.models
    split_mode = config["experiment"]["split_mode"]
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"Unsupported split_mode={split_mode}; choose from {sorted(SPLIT_MODES)}")
    return config


def validate_binary_splits(splits: Sequence[tuple[np.ndarray, np.ndarray]], y: np.ndarray, context: str) -> None:
    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        train_classes = np.unique(y[train_idx])
        test_classes = np.unique(y[test_idx])
        if len(train_classes) != 2 or len(test_classes) != 2:
            raise ValueError(
                f"{context} fold {fold} does not contain both classes: "
                f"train={train_classes.tolist()}, test={test_classes.tolist()}"
            )
        if np.intersect1d(train_idx, test_idx).size:
            raise ValueError(f"{context} fold {fold}: train/test index overlap")


def make_outer_splits(records: Sequence[PairRecord], config: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    exp = config["experiment"]
    y = np.asarray([pair.label for pair in records], dtype=np.int64)
    pair_to_index = {pair.pair_id: idx for idx, pair in enumerate(records)}
    mode = exp["split_mode"]
    n_splits = int(exp["outer_folds"])

    if mode == "existing_folds":
        assignment_path = Path(config["data"]["fold_assignments_tsv"])
        with assignment_path.open("r", encoding="utf-8", newline="") as handle:
            assignment_rows = list(csv.DictReader(handle, delimiter="\t"))
        expected_all = set(pair_to_index)
        test_counts = {pair_id: 0 for pair_id in expected_all}
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for fold in range(1, n_splits + 1):
            train_map = {
                row["pair_id"]: int(row["label"])
                for row in assignment_rows
                if int(row["fold"]) == fold and row["split"] == "train"
            }
            test_map = {
                row["pair_id"]: int(row["label"])
                for row in assignment_rows
                if int(row["fold"]) == fold and row["split"] == "test"
            }
            if set(train_map) & set(test_map):
                raise ValueError(f"Reference fold {fold}: train/test pair overlap")
            if set(train_map) | set(test_map) != expected_all:
                missing = expected_all.difference(set(train_map) | set(test_map))
                extra = (set(train_map) | set(test_map)).difference(expected_all)
                raise ValueError(f"Reference fold {fold} mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
            for pair_id, label in {**train_map, **test_map}.items():
                if records[pair_to_index[pair_id]].label != label:
                    raise ValueError(f"Reference label mismatch for {pair_id}")
            for pair_id in test_map:
                test_counts[pair_id] += 1
            splits.append((
                np.asarray([pair_to_index[pair_id] for pair_id in train_map], dtype=np.int64),
                np.asarray([pair_to_index[pair_id] for pair_id in test_map], dtype=np.int64),
            ))
        bad_counts = {pair_id: count for pair_id, count in test_counts.items() if count != 1}
        if bad_counts:
            raise ValueError(f"Every pair must be outer-test exactly once: {bad_counts}")
    elif mode == "pair_stratified":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(exp["random_seed"]))
        splits = list(splitter.split(np.zeros(len(records)), y))
    else:
        groups = np.asarray([
            pair.protein1 if mode == "protein1_grouped" else pair.protein2
            for pair in records
        ])
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=int(exp["random_seed"])
        )
        splits = list(splitter.split(np.zeros(len(records)), y, groups))
        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            overlap = set(groups[train_idx]) & set(groups[test_idx])
            if overlap:
                raise ValueError(f"{mode} fold {fold}: group leakage: {sorted(overlap)}")

    validate_binary_splits(splits, y, f"outer/{mode}")
    return splits


def build_estimator(
    model_name: str, seed: int, fixed_parameters: dict[str, dict[str, Any]]
) -> tuple[Any, str]:
    if model_name == "dummy_prior":
        return DummyClassifier(strategy="prior"), "predict_proba"
    if model_name == "logistic_regression":
        parameters = {
            "C": 1.0, "solver": "liblinear", "class_weight": "balanced", "max_iter": 5000,
            **fixed_parameters.get(model_name, {}),
        }
        classifier = LogisticRegression(random_state=seed, **parameters)
        return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)]), "predict_proba"
    if model_name == "linear_svm":
        # Use raw margins: probability=True would run a hidden, non-group-aware
        # Platt-calibration CV inside libsvm, weakening grouped-CV guarantees.
        parameters = {
            "C": 1.0, "kernel": "linear", "probability": False, "class_weight": "balanced",
            **fixed_parameters.get(model_name, {}),
        }
        if parameters["probability"]:
            raise ValueError("linear_svm probability must remain false; use decision_function scores")
        classifier = SVC(random_state=seed, **parameters)
        return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)]), "decision_function"
    if model_name == "random_forest":
        parameters = {
            "n_estimators": 300, "max_depth": None, "min_samples_leaf": 1,
            "max_features": "sqrt", "class_weight": "balanced_subsample", "n_jobs": 1,
            **fixed_parameters.get(model_name, {}),
        }
        classifier = RandomForestClassifier(random_state=seed, **parameters)
        return Pipeline([("classifier", classifier)]), "predict_proba"
    raise ValueError(f"Unknown model: {model_name}")


def extract_scores(estimator: Any, x: np.ndarray, score_method: str) -> np.ndarray:
    raw = getattr(estimator, score_method)(x)
    if score_method == "predict_proba":
        raw = raw[:, 1]
    scores = np.asarray(raw, dtype=np.float64).reshape(-1)
    if not np.isfinite(scores).all():
        raise ValueError("Estimator returned non-finite scores")
    return scores
def calculate_metrics(y_true: np.ndarray, scores: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    ap = float(average_precision_score(y_true, scores))
    return {
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "auprc": ap,
        "average_precision": ap,
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def write_tsv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_provenance(run_dir: Path, config: dict, config_path: Path) -> None:
    source_dir = run_dir / "source_code"
    source_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "run_ml_baselines.py", "features.py",
        "config_ml.json", "config_structure.json",
    ]:
        source = Path(__file__).with_name(filename)
        if source.exists():
            shutil.copy2(source, source_dir / filename)
    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    environment = {
        "timestamp": dt.datetime.now().astimezone().isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "command": sys.argv,
        "config_source": str(config_path.resolve()),
    }
    try:
        environment["pip_freeze"] = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], check=True,
            capture_output=True, text=True,
        ).stdout.splitlines()
    except Exception as exc:  # provenance failure must not abort a completed experiment
        environment["pip_freeze_error"] = repr(exc)
    with (run_dir / "runtime_environment.json").open("w", encoding="utf-8") as handle:
        json.dump(environment, handle, indent=2, ensure_ascii=False)


def save_fold_assignments(
    run_dir: Path, records: Sequence[PairRecord], splits: Sequence[tuple[np.ndarray, np.ndarray]], mode: str
) -> None:
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        for subset, indices in (("train", train_idx), ("test", test_idx)):
            for idx in indices:
                pair = records[int(idx)]
                rows.append({
                    "fold": fold, "subset": subset, "pair_id": pair.pair_id,
                    "protein1": pair.protein1, "protein2": pair.protein2,
                    "label": pair.label, "split_mode": mode,
                })
    write_tsv(run_dir / "outer_fold_assignments.tsv", rows)


def make_summary(per_fold: Sequence[dict], predictions: Sequence[dict]) -> list[dict]:
    summary: list[dict] = []
    keys = sorted({(row["feature_set"], row["model"]) for row in per_fold})
    for feature_set, model_name in keys:
        fold_rows = [row for row in per_fold if row["feature_set"] == feature_set and row["model"] == model_name]
        pred_rows = [row for row in predictions if row["feature_set"] == feature_set and row["model"] == model_name]
        y_true = np.asarray([row["y_true"] for row in pred_rows], dtype=np.int64)
        scores = np.asarray([row["y_score"] for row in pred_rows], dtype=np.float64)
        y_pred = np.asarray([row["y_pred"] for row in pred_rows], dtype=np.int64)
        pooled = calculate_metrics(y_true, scores, y_pred)
        result: dict[str, Any] = {
            "feature_set": feature_set,
            "model": model_name,
            "n_oof": len(pred_rows),
            "n_folds": len(fold_rows),
        }
        for metric in METRIC_NAMES:
            values = np.asarray([row[metric] for row in fold_rows], dtype=np.float64)
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            result[f"{metric}_pooled_oof"] = pooled[metric]
        summary.append(result)
    return summary


def make_plots(run_dir: Path, predictions: Sequence[dict], feature_sets: Sequence[str]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
    except Exception as exc:
        print(f"Warning: plots skipped because matplotlib is unavailable: {exc}", file=sys.stderr)
        return
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    for feature_set in feature_sets:
        subset = [row for row in predictions if row["feature_set"] == feature_set]
        if not subset:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for model_name in sorted({row["model"] for row in subset}):
            rows = [row for row in subset if row["model"] == model_name]
            y_true = np.asarray([row["y_true"] for row in rows], dtype=np.int64)
            y_score = np.asarray([row["y_score"] for row in rows], dtype=np.float64)
            RocCurveDisplay.from_predictions(y_true, y_score, name=model_name, ax=axes[0])
            PrecisionRecallDisplay.from_predictions(y_true, y_score, name=model_name, ax=axes[1])
        axes[0].set_title(f"Pooled out-of-fold ROC: {feature_set}")
        axes[1].set_title(f"Pooled out-of-fold PR: {feature_set}")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{feature_set}_pooled_curves.png", dpi=220)
        plt.close(fig)


def main() -> Path:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path, args)
    exp = config["experiment"]
    records = read_pairs_tsv(config["data"]["pairs_tsv"])
    y = np.asarray([pair.label for pair in records], dtype=np.int64)
    splits = make_outer_splits(records, config)
    feature_store = FixedFeatureStore(config["data"], records)
    audit = feature_store.audit(records, config["feature_sets"])
    audit["split_mode"] = exp["split_mode"]
    audit["model_selection"] = "prespecified fixed parameters and fixed thresholds; no inner CV"
    audit["outer_fold_sizes"] = []
    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        train_protein1 = {records[int(index)].protein1 for index in train_idx}
        train_protein2 = {records[int(index)].protein2 for index in train_idx}
        test_records = [records[int(index)] for index in test_idx]
        protein1_seen = np.asarray([pair.protein1 in train_protein1 for pair in test_records])
        protein2_seen = np.asarray([pair.protein2 in train_protein2 for pair in test_records])
        audit["outer_fold_sizes"].append({
            "fold": fold,
            "train": int(len(train_idx)), "test": int(len(test_idx)),
            "train_positive": int(y[train_idx].sum()), "test_positive": int(y[test_idx].sum()),
            "test_pairs_protein1_seen_in_train": int(protein1_seen.sum()),
            "test_pairs_protein2_seen_in_train": int(protein2_seen.sum()),
            "test_pairs_both_proteins_seen_in_train": int((protein1_seen & protein2_seen).sum()),
            "test_pairs_neither_protein_seen_in_train": int((~protein1_seen & ~protein2_seen).sum()),
        })

    if args.dry_run:
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return Path(exp["output_dir"])

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(exp["output_dir"]) / f"{timestamp}_{exp['split_mode']}"
    run_dir.mkdir(parents=True, exist_ok=False)
    save_provenance(run_dir, config, config_path)
    with (run_dir / "data_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    save_fold_assignments(run_dir, records, splits, exp["split_mode"])

    per_fold_rows: list[dict] = []
    prediction_rows: list[dict] = []
    parameter_rows: list[dict] = []
    feature_matrices = {
        feature_set: feature_store.make_matrix(records, feature_set)
        for feature_set in config["feature_sets"]
    }

    for feature_set, x in feature_matrices.items():
        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            for model_name in config["models"]:
                print(f"[{feature_set}] fold {fold}/{len(splits)}: {model_name}", flush=True)
                estimator, score_method = build_estimator(
                    model_name,
                    int(exp["random_seed"]) + fold,
                    config.get("fixed_parameters", {}),
                )
                fitted = clone(estimator).fit(x_train, y_train)
                fixed_params = config.get("fixed_parameters", {}).get(model_name, {})
                threshold = 0.5 if score_method == "predict_proba" else 0.0

                test_scores = extract_scores(fitted, x_test, score_method)
                test_predictions = (test_scores >= threshold).astype(np.int64)
                metrics = calculate_metrics(y_test, test_scores, test_predictions)
                per_fold_rows.append({
                    "feature_set": feature_set, "model": model_name, "fold": fold,
                    "n_train": len(train_idx), "n_test": len(test_idx),
                    "n_train_positive": int(y_train.sum()), "n_test_positive": int(y_test.sum()),
                    "fixed_threshold": float(threshold),
                    **metrics,
                })
                parameter_rows.append({
                    "feature_set": feature_set, "model": model_name, "fold": fold,
                    "score_method": score_method,
                    "fixed_threshold": float(threshold),
                    "fixed_params_json": json.dumps(fixed_params, sort_keys=True),
                })
                for local_idx, score, prediction in zip(test_idx, test_scores, test_predictions):
                    pair = records[int(local_idx)]
                    prediction_rows.append({
                        "feature_set": feature_set, "model": model_name, "fold": fold,
                        "pair_id": pair.pair_id, "protein1": pair.protein1, "protein2": pair.protein2,
                        "y_true": pair.label, "y_score": float(score), "y_pred": int(prediction),
                        "threshold": float(threshold), "score_type": score_method,
                    })
                if bool(exp["save_models"]):
                    model_dir = run_dir / "models" / feature_set / model_name
                    model_dir.mkdir(parents=True, exist_ok=True)
                    joblib.dump(fitted, model_dir / f"fold_{fold}.joblib")

    summary_rows = make_summary(per_fold_rows, prediction_rows)
    write_tsv(run_dir / "per_fold_metrics.tsv", per_fold_rows)
    write_tsv(run_dir / "oof_predictions.tsv", prediction_rows)
    write_tsv(run_dir / "fixed_parameters.tsv", parameter_rows)
    write_tsv(run_dir / "summary_metrics.tsv", summary_rows)
    if bool(exp["make_plots"]):
        make_plots(run_dir, prediction_rows, config["feature_sets"])
    print(f"Completed. Results: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
