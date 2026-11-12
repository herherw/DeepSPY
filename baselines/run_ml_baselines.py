"""Gold-data ML baselines using exactly the DeepSPY outer folds.

This is an independent runner and does not modify the historical baseline code.
It evaluates AAC+DPC, mean ESM-C, and their concatenation with LR, DT and
linear SVM. Probability models use threshold 0.5; linear SVM uses score 0.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


AA = "ACDEFGHIKLMNPQRSTVWY"


def cfg_get(cfg, *names, default=None):
    for name in names:
        node = cfg
        ok = True
        for part in name.split("."):
            if not isinstance(node, dict) or part not in node:
                ok = False
                break
            node = node[part]
        if ok:
            return node
    return default


def read_sequences(path):
    """Read sequence CSVs with or without headers and in either column order.

    Historical DeepSPY exports are not fully consistent: some files are
    ``protein,sequence``, some are ``sequence,protein``, and some have no
    header. Identify fields from their contents instead of relying on a fixed
    column position.
    """
    raw = pd.read_csv(path, sep="\t" if Path(path).suffix == ".tsv" else ",", header=None, dtype=str, keep_default_na=False)
    result = {}
    sequence_letters = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ*-.")
    for row_no, row in raw.iterrows():
        values = [str(v).strip() for v in row.tolist() if str(v).strip()]
        if not values:
            continue
        lower = {v.lower() for v in values}
        # Header variants used by the project include prot1_type/prot1_seq/
        # length and prot2_type/prot2_seq/length.
        if (lower & {"sequence", "seq", "protein", "protein_id", "id", "name", "length"}
                or any(v.endswith("_seq") or v.endswith("_type") for v in lower)):
            continue

        seq_candidates = [
            v.upper().replace(" ", "") for v in values
            if len(v.strip()) >= 10
            and set(v.upper().replace(" ", "")).issubset(sequence_letters)
        ]
        if not seq_candidates:
            raise ValueError(f"{path}: row {row_no + 1}: cannot identify an amino-acid sequence from {values}")
        sequence = max(seq_candidates, key=len).replace("*", "").replace("-", "").replace(".", "")

        id_candidates = [
            v for v in values
            if v.upper().replace(" ", "") not in seq_candidates
            and not v.isdigit()
        ]
        preferred = [v for v in id_candidates if v.lower().startswith("phs")]
        if preferred:
            protein_id = preferred[0]
        elif id_candidates:
            protein_id = min(id_candidates, key=len)
        else:
            raise ValueError(f"{path}: row {row_no + 1}: cannot identify protein ID from {values}")
        if protein_id in result and result[protein_id] != sequence:
            raise ValueError(f"{path}: conflicting sequences for {protein_id}")
        result[protein_id] = sequence
    if not result:
        raise ValueError(f"No protein sequences loaded from {path}")
    return result


def read_pairs(path):
    path = Path(path)
    # DeepSPY's authoritative gold list is a headerless, whitespace-delimited
    # two-column file: "pair_id label". Keep support for the historical TSV.
    first = ""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                first = line.strip()
                break
    fields = first.split()
    has_header = any(x.lower() in {"pair_id", "protein1", "protein2", "label"} for x in fields)
    if not has_header and len(fields) == 2:
        df = pd.read_csv(path, sep=r"\s+", header=None, names=["pair_id", "label"], dtype={"pair_id": str})
        split = df["pair_id"].str.split("_", n=1, expand=True)
        if split.shape[1] != 2 or split.isna().any().any():
            raise ValueError(f"Cannot derive protein1/protein2 from pair IDs in {path}")
        df["protein1"] = split[0]
        df["protein2"] = split[1]
    else:
        df = pd.read_csv(path, sep=r"\s+" if path.suffix.lower() == ".txt" else "\t")
        df = df.rename(columns={"protein1_id":"protein1","protein2_id":"protein2"})
        required = {"protein1", "protein2", "label"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"{path} must be either a two-column 'pair_id label' file or contain {sorted(required)}"
            )
        if "pair_id" not in df.columns:
            df.insert(0, "pair_id", df.protein1.astype(str) + "_" + df.protein2.astype(str))
    df = df[["pair_id", "protein1", "protein2", "label"]].copy()
    df["label"] = df["label"].astype(int)
    if df.pair_id.duplicated().any():
        raise ValueError("Duplicate pair_id values in canonical pair table")
    return df


def normalize_pair_id(p1, p2):
    p1 = str(p1)
    return f"{p1[:-6] if p1.endswith('-RNase') else p1}_{p2}"


def locate_fold_root(reference):
    reference = Path(reference)
    candidates = [reference, reference / "shared_training"]
    for candidate in candidates:
        if all((candidate / f"fold_{i}" / "train_pairs.txt").is_file() and
               (candidate / f"fold_{i}" / "test_pairs.txt").is_file()
               for i in range(1, 6)):
            return candidate
    raise FileNotFoundError(f"Cannot find fold_1..fold_5 train/test files below {reference}")


def parse_fold_file(path, canonical):
    by_id = canonical.set_index("pair_id")
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) == 2:  # pair_id label
                pair_id, label = fields
            elif len(fields) == 3:  # protein1 protein2 label
                pair_id, label = normalize_pair_id(fields[0], fields[1]), fields[2]
            else:
                raise ValueError(f"{path}:{line_no}: expected 2 or 3 fields")
            if pair_id not in by_id.index:
                raise ValueError(f"{path}:{line_no}: unknown pair {pair_id}")
            if int(label) != int(by_id.loc[pair_id, "label"]):
                raise ValueError(f"{path}:{line_no}: label mismatch for {pair_id}")
            rows.append(pair_id)
    if len(rows) != len(set(rows)):
        raise ValueError(f"Duplicate pairs in {path}")
    return rows


def load_folds(reference, canonical):
    root = locate_fold_root(reference)
    folds, test_seen = [], []
    all_ids = set(canonical.pair_id)
    for i in range(1, 6):
        folder = root / f"fold_{i}"
        train = parse_fold_file(folder / "train_pairs.txt", canonical)
        test = parse_fold_file(folder / "test_pairs.txt", canonical)
        if set(train) & set(test):
            raise ValueError(f"Fold {i}: train/test overlap")
        if set(train) | set(test) != all_ids:
            raise ValueError(f"Fold {i}: train/test union differs from canonical pairs")
        folds.append((train, test))
        test_seen.extend(test)
    if len(test_seen) != len(all_ids) or set(test_seen) != all_ids or len(test_seen) != len(set(test_seen)):
        raise ValueError("Five test folds do not cover every canonical pair exactly once")
    return root, folds


def load_npy_dict(path):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dictionary-like npy at {path}")
    return obj


def aac(seq):
    seq = "".join(x for x in seq if x in AA)
    n = max(len(seq), 1)
    return np.asarray([seq.count(x) / n for x in AA], dtype=np.float32)


def dpc(seq):
    seq = "".join(x for x in seq if x in AA)
    counts = {a + b: 0 for a in AA for b in AA}
    for a, b in zip(seq, seq[1:]):
        counts[a + b] += 1
    n = max(len(seq) - 1, 1)
    return np.asarray([counts[a + b] / n for a in AA for b in AA], dtype=np.float32)


def mean_esmc(array, seq_len, protein):
    x = np.asarray(array)
    if x.ndim == 1:
        return x.astype(np.float32)
    if x.ndim != 2:
        raise ValueError(f"{protein}: ESM-C array must be 1D or 2D, got {x.shape}")
    if x.shape[0] == seq_len + 2:
        x = x[1:-1]
    elif x.shape[0] != seq_len:
        raise ValueError(f"{protein}: ESM-C length {x.shape[0]} != sequence length {seq_len} (+ optional BOS/EOS)")
    return x.mean(axis=0).astype(np.float32)


def build_features(pairs, seq1, seq2, emb1, emb2):
    hand, esmc = {}, {}
    for row in pairs.itertuples(index=False):
        p1, p2 = str(row.protein1), str(row.protein2)
        p1_key = p1[:-6] if p1.endswith("-RNase") and p1[:-6] in seq1 else p1
        if p1_key not in seq1 or p2 not in seq2:
            raise KeyError(f"Missing sequence for {p1}/{p2}")
        if p1_key not in emb1 or p2 not in emb2:
            raise KeyError(f"Missing ESM-C feature for {p1}/{p2}")
        s1, s2 = seq1[p1_key], seq2[p2]
        hand[row.pair_id] = np.concatenate([aac(s1), dpc(s1), aac(s2), dpc(s2)])
        esmc[row.pair_id] = np.concatenate([
            mean_esmc(emb1[p1_key], len(s1), p1_key),
            mean_esmc(emb2[p2], len(s2), p2),
        ])
    return {
        "aac_dpc": hand,
        "esmc_mean": esmc,
        "aac_dpc_esmc_mean": {k: np.concatenate([hand[k], esmc[k]]) for k in hand},
    }


def metrics(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "auroc": roc_auc_score(y, score),
        "average_precision": average_precision_score(y, score),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def estimator(name, seed, fixed_parameters=None):
    fixed_parameters = fixed_parameters or {}
    if name == "lr":
        model = LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=5000, random_state=seed)
        return Pipeline([("scale", StandardScaler()), ("model", model)])
    if name == "svm":
        params = {"C": 1.0, "class_weight": "balanced", "dual": False, "max_iter": 50000}
        overrides = dict(fixed_parameters.get("linear_svm", {}))
        if overrides.pop("kernel", "linear") != "linear" or overrides.pop("probability", False):
            raise ValueError("LinearSVC requires kernel=linear and probability=false")
        params.update(overrides)
        params["random_state"] = seed
        model = LinearSVC(**params)
        return Pipeline([("scale", StandardScaler()), ("model", model)])
    if name == "dt":
        return DecisionTreeClassifier()
    raise KeyError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_ml.json")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate canonical pairs and the five DeepSPY folds, then exit")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    for key, value in cfg["data"].items():
        cfg["data"][key] = str((config_path.parent / value).resolve())
    for key in ("output_dir", "standard_output_dir"):
        if key in cfg.get("experiment", {}):
            cfg["experiment"][key] = str((config_path.parent / cfg["experiment"][key]).resolve())

    pairs_path = cfg_get(
        cfg,
        "data.pairs_tsv",
        "data.pair_path",
        "data.pairs_path",
        "data.pair_tsv",
        "data.pairs_file",
        "pairs_tsv",
        "pair_path",
        "pairs_path",
    )
    # Some project configs use a descriptive key such as gold_pairs_path.
    # Accept it only when exactly one unambiguous TSV pair-table candidate exists.
    if not pairs_path and isinstance(cfg.get("data"), dict):
        candidates = [
            value for key, value in cfg["data"].items()
            if isinstance(value, str)
            and "pair" in key.lower()
            and "fold" not in key.lower()
            and value.lower().endswith((".tsv", ".csv", ".txt"))
        ]
        if len(candidates) == 1:
            pairs_path = candidates[0]
    reference = cfg_get(cfg, "data.reference_fold_dir", "reference_fold_dir", "deep_result_dir")
    seq1_path = cfg_get(cfg, "data.pro1_sequence_csv", "pro1_sequence_csv")
    seq2_path = cfg_get(cfg, "data.pro2_sequence_csv", "pro2_sequence_csv")
    emb1_path = cfg_get(cfg, "data.pro1_esmc_npy", "pro1_esmc_npy")
    emb2_path = cfg_get(cfg, "data.pro2_esmc_npy", "pro2_esmc_npy")
    output = cfg_get(cfg, "experiment.output_dir", "output_dir", default=str(config_path.parent / "results" / "comparison_ml"))
    seed = int(cfg_get(cfg, "experiment.seed", "seed", default=124))
    missing = [n for n, v in {"pairs_tsv": pairs_path, "reference_fold_dir": reference,
                              "pro1_sequence_csv": seq1_path, "pro2_sequence_csv": seq2_path,
                              "pro1_esmc_npy": emb1_path, "pro2_esmc_npy": emb2_path}.items() if not v]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")

    pairs = read_pairs(pairs_path)
    fold_root, folds = load_folds(reference, pairs)
    if args.validate_only:
        print(json.dumps({
            "canonical_pairs": len(pairs),
            "positive": int(pairs.label.sum()),
            "negative": int((pairs.label == 0).sum()),
            "fold_root": str(fold_root),
            "test_fold_sizes": [len(test) for _, test in folds],
            "test_coverage_exactly_once": True,
        }, ensure_ascii=False, indent=2))
        return
    seq1, seq2 = read_sequences(seq1_path), read_sequences(seq2_path)
    features = build_features(pairs, seq1, seq2, load_npy_dict(emb1_path), load_npy_dict(emb2_path))
    by_id = pairs.set_index("pair_id")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "resolved_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    fixed_parameters = cfg.get("fixed_parameters", {})
    assignment_rows = []
    all_summary = []
    for fold_no, (_, test_ids) in enumerate(folds, 1):
        assignment_rows.extend({"pair_id": x, "fold": fold_no} for x in test_ids)
    pd.DataFrame(assignment_rows).to_csv(out / "outer_fold_assignments.tsv", sep="\t", index=False)

    for feature_name, feature_map in features.items():
        for model_name in ("lr", "svm", "dt"):
            run_dir = out / feature_name / model_name
            run_dir.mkdir(parents=True)
            fold_metrics, oof = [], []
            for fold_no, (train_ids, test_ids) in enumerate(folds, 1):
                x_train = np.stack([feature_map[x] for x in train_ids])
                x_test = np.stack([feature_map[x] for x in test_ids])
                y_train = by_id.loc[train_ids, "label"].to_numpy(dtype=int)
                y_test = by_id.loc[test_ids, "label"].to_numpy(dtype=int)
                clf = estimator(model_name, seed + fold_no, fixed_parameters)
                clf.fit(x_train, y_train)
                if model_name == "svm":
                    score = clf.decision_function(x_test)
                    pred = (score >= 0.0).astype(int)
                    threshold = 0.0
                else:
                    score = clf.predict_proba(x_test)[:, 1]
                    pred = (score >= 0.5).astype(int)
                    threshold = 0.5
                fm = {"fold": fold_no, "feature_set": feature_name, "model": model_name,
                      "threshold": threshold, **metrics(y_test, pred, score)}
                fold_metrics.append(fm)
                for pid, y, s, p in zip(test_ids, y_test, score, pred):
                    row = by_id.loc[pid]
                    oof.append({"pair_id": pid, "protein1": row.protein1, "protein2": row.protein2,
                                "label": int(y), "fold": fold_no, "score": float(s),
                                "prediction": int(p), "threshold": threshold})
            fold_df, oof_df = pd.DataFrame(fold_metrics), pd.DataFrame(oof)
            fold_df.to_csv(run_dir / "per_fold_metrics.tsv", sep="\t", index=False)
            oof_df.to_csv(run_dir / "outer_oof_predictions.tsv", sep="\t", index=False)
            pooled = metrics(oof_df.label.to_numpy(), oof_df.prediction.to_numpy(), oof_df.score.to_numpy())
            summary = {"feature_set": feature_name, "model": model_name,
                       "threshold": float(oof_df.threshold.iloc[0]), "n": len(oof_df),
                       **{f"mean_{k}": fold_df[k].mean() for k in ("accuracy", "precision", "recall", "specificity", "f1", "mcc", "auroc", "average_precision")},
                       **{f"sd_{k}": fold_df[k].std(ddof=1) for k in ("accuracy", "precision", "recall", "specificity", "f1", "mcc", "auroc", "average_precision")},
                       **{f"pooled_{k}": v for k, v in pooled.items()}}
            all_summary.append(summary)
            pd.DataFrame([summary]).to_csv(run_dir / "summary.tsv", sep="\t", index=False)

    pd.DataFrame(all_summary).to_csv(out / "all_baselines_summary.tsv", sep="\t", index=False)
    audit = {
        "canonical_pairs": len(pairs), "positive": int(pairs.label.sum()),
        "negative": int((pairs.label == 0).sum()), "fold_root": str(fold_root),
        "test_fold_sizes": [len(x[1]) for x in folds],
        "test_coverage_exactly_once": True,
        "feature_sets": list(features),
        "thresholds": {"lr": 0.5, "dt": 0.5, "svm": 0.0},
    }
    (out / "data_and_split_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Completed 9 baselines. Results: {out}")


if __name__ == "__main__":
    main()
