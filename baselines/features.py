#!/usr/bin/env python3
# Run (syntax check): python -m py_compile baseline_features.py
"""Fixed, label-independent protein-pair feature construction."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    protein1: str
    protein2: str
    label: int


def read_pairs_tsv(path: str | Path) -> list[PairRecord]:
    """Read the canonical Gold-only pair table and validate pair identifiers."""
    path = Path(path)
    records: list[PairRecord] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"pair_id", "protein1_id", "protein2_id", "label"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            pair_id = row["pair_id"].strip()
            if pair_id in seen:
                raise ValueError(f"Duplicate pair_id in {path}: {pair_id}")
            seen.add(pair_id)

            # Feature-cache identifiers use e.g. PhS3 rather than PhS3-RNase.
            # pair_id is the authoritative cache key and split only once because
            # protein2 names contain hyphens but not underscores.
            if "_" not in pair_id:
                raise ValueError(f"Invalid pair_id (missing underscore): {pair_id}")
            cache_protein1, cache_protein2 = pair_id.split("_", 1)
            table_protein1 = row["protein1_id"].strip()
            table_protein2 = row["protein2_id"].strip()
            if table_protein1.removesuffix("-RNase") != cache_protein1:
                raise ValueError(
                    f"{pair_id}: protein1 column ({table_protein1}) does not match pair_id"
                )
            if table_protein2 != cache_protein2:
                raise ValueError(
                    f"{pair_id}: protein2 column ({table_protein2}) does not match pair_id"
                )
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"Non-binary label for {pair_id}: {label}")
            records.append(PairRecord(pair_id, cache_protein1, cache_protein2, label))
    if not records:
        raise ValueError(f"No pairs found in {path}")
    return records


def _load_npy_dict(path: str | Path) -> Dict[str, np.ndarray]:
    array = np.load(path, allow_pickle=True)
    if array.shape != () or array.dtype != object:
        raise ValueError(f"Expected a scalar object-dict cache in {path}, got {array.shape}/{array.dtype}")
    value = array.item()
    if not isinstance(value, dict):
        raise ValueError(f"Expected dict in {path}, got {type(value).__name__}")
    return {str(key): np.asarray(item, dtype=np.float32) for key, item in value.items()}


def _load_sequences(path: str | Path, id_column: str, sequence_column: str) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    delimiter = "\t" if Path(path).suffix.lower() in {".tsv", ".txt"} else ","
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            protein_id = str(row[id_column]).strip()
            sequence = str(row[sequence_column]).strip()
            if not sequence:
                raise ValueError(f"Empty sequence for {protein_id} in {path}")
            sequences[protein_id] = sequence
    return sequences


def _residue_only_embedding(
    protein_id: str,
    embedding: np.ndarray,
    sequence: str,
    source: str,
) -> np.ndarray:
    """Remove ESM-C BOS/EOS exactly once using biological length as truth."""
    if embedding.ndim != 2:
        raise ValueError(f"{source}/{protein_id}: expected 2-D embedding, got {embedding.shape}")
    sequence_length = len(sequence)
    if embedding.shape[0] == sequence_length + 2:
        embedding = embedding[1:-1]
    elif embedding.shape[0] != sequence_length:
        raise ValueError(
            f"{source}/{protein_id}: embedding rows={embedding.shape[0]}, "
            f"sequence length={sequence_length}; expected L or L+2"
        )
    if embedding.shape[0] == 0 or not np.isfinite(embedding).all():
        raise ValueError(f"{source}/{protein_id}: empty or non-finite residue embedding")
    return embedding


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DIPEPTIDES = tuple(first + second for first in AMINO_ACIDS for second in AMINO_ACIDS)


def sequence_composition(sequence: str, protein_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Calculate AAC and DPC directly from an unaligned protein sequence."""
    invalid = sorted(set(sequence).difference(AMINO_ACIDS))
    if invalid:
        raise ValueError(f"{protein_id}: unsupported amino-acid symbols: {invalid}")
    if len(sequence) < 2:
        raise ValueError(f"{protein_id}: sequence must contain at least two residues")

    residue_counts = Counter(sequence)
    aac = np.asarray(
        [residue_counts.get(amino_acid, 0) / len(sequence) for amino_acid in AMINO_ACIDS],
        dtype=np.float32,
    )
    dipeptide_counts = Counter(sequence[index:index + 2] for index in range(len(sequence) - 1))
    dpc = np.asarray(
        [dipeptide_counts.get(dipeptide, 0) / (len(sequence) - 1) for dipeptide in DIPEPTIDES],
        dtype=np.float32,
    )
    return aac, dpc


class FixedFeatureStore:
    """Loads only sequence-derived descriptors; no labels or DeepSPY weights."""

    VALID_FEATURE_SETS = {"aac_dpc", "esmc_mean", "aac_dpc_esmc_mean"}

    def __init__(self, data_config: Mapping[str, str], required_pairs: Sequence[PairRecord]):
        self.esmc1 = _load_npy_dict(data_config["pro1_esmc_npy"])
        self.esmc2 = _load_npy_dict(data_config["pro2_esmc_npy"])
        self.seq1 = _load_sequences(data_config["pro1_sequence_tsv"], "protein_id", "sequence")
        self.seq2 = _load_sequences(data_config["pro2_sequence_tsv"], "protein_id", "sequence")

        self.required_protein1 = sorted({pair.protein1 for pair in required_pairs})
        self.required_protein2 = sorted({pair.protein2 for pair in required_pairs})
        self._validate_keys()
        self.aac1, self.dpc1 = self._composition_side(self.required_protein1, self.seq1)
        self.aac2, self.dpc2 = self._composition_side(self.required_protein2, self.seq2)
        self.esmc_mean1 = self._mean_pool_side(self.required_protein1, self.esmc1, self.seq1, "protein1")
        self.esmc_mean2 = self._mean_pool_side(self.required_protein2, self.esmc2, self.seq2, "protein2")

    def _validate_keys(self) -> None:
        side_specs = [
            ("protein1", self.required_protein1, self.esmc1, self.seq1),
            ("protein2", self.required_protein2, self.esmc2, self.seq2),
        ]
        for side, required, esmc, sequence in side_specs:
            for name, cache in [("ESM-C", esmc), ("sequence", sequence)]:
                missing = sorted(set(required).difference(cache))
                if missing:
                    raise KeyError(f"{side} {name} cache is missing: {missing}")

    @staticmethod
    def _composition_side(
        protein_ids: Iterable[str], sequences: Mapping[str, str]
    ) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        aac: Dict[str, np.ndarray] = {}
        dpc: Dict[str, np.ndarray] = {}
        for protein_id in protein_ids:
            aac[protein_id], dpc[protein_id] = sequence_composition(sequences[protein_id], protein_id)
        return aac, dpc

    @staticmethod
    def _mean_pool_side(
        protein_ids: Iterable[str],
        embeddings: Mapping[str, np.ndarray],
        sequences: Mapping[str, str],
        source: str,
    ) -> Dict[str, np.ndarray]:
        means: Dict[str, np.ndarray] = {}
        embedding_dim: int | None = None
        for protein_id in protein_ids:
            residue_embedding = _residue_only_embedding(
                protein_id, embeddings[protein_id], sequences[protein_id], source
            )
            if embedding_dim is None:
                embedding_dim = residue_embedding.shape[1]
            elif residue_embedding.shape[1] != embedding_dim:
                raise ValueError(
                    f"{source}/{protein_id}: inconsistent ESM-C dimension "
                    f"{residue_embedding.shape[1]} != {embedding_dim}"
                )
            means[protein_id] = residue_embedding.mean(axis=0, dtype=np.float64).astype(np.float32)
        return means

    def make_matrix(self, records: Sequence[PairRecord], feature_set: str) -> np.ndarray:
        if feature_set not in self.VALID_FEATURE_SETS:
            raise ValueError(f"Unknown feature set: {feature_set}")
        rows: list[np.ndarray] = []
        for pair in records:
            parts: list[np.ndarray] = []
            if feature_set in {"aac_dpc", "aac_dpc_esmc_mean"}:
                parts.extend([
                    self.aac1[pair.protein1], self.dpc1[pair.protein1],
                    self.aac2[pair.protein2], self.dpc2[pair.protein2],
                ])
            if feature_set in {"esmc_mean", "aac_dpc_esmc_mean"}:
                parts.extend([self.esmc_mean1[pair.protein1], self.esmc_mean2[pair.protein2]])
            rows.append(np.concatenate(parts).astype(np.float32, copy=False))
        matrix = np.stack(rows)
        if not np.isfinite(matrix).all():
            raise ValueError(f"Non-finite values in {feature_set} matrix")
        return matrix

    def audit(self, records: Sequence[PairRecord], feature_sets: Sequence[str]) -> dict:
        return {
            "n_pairs": len(records),
            "n_positive": sum(pair.label for pair in records),
            "n_negative": len(records) - sum(pair.label for pair in records),
            "n_unique_protein1": len(self.required_protein1),
            "n_unique_protein2": len(self.required_protein2),
            "feature_dimensions": {
                feature_set: int(self.make_matrix(records[:1], feature_set).shape[1])
                for feature_set in feature_sets
            },
            "esmc_special_tokens": "BOS/EOS removed when cache length equals biological length + 2",
            "aac_dpc_source": "calculated directly from raw sequences at runtime",
            "feature_supervision": "none; raw-sequence descriptors and frozen pretrained ESM-C only",
        }
