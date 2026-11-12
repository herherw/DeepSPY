#!/usr/bin/env python3
"""Create residue-aligned ESM-C and CKSAAP-ESMC caches.

The ESM-C API returns [BOS, residue_1, ..., residue_L, EOS].  Cross-attention
and residue numbering must instead consume exactly L rows.  This script strips
the two special-token rows and rebuilds CKSAAP-ESMC from the aligned rows.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}


from .utils import load_config


def load_sequences(path, id_column, sequence_column):
    delimiter = "\t" if Path(path).suffix.lower() in {".tsv", ".txt"} else ","
    dataframe = pd.read_csv(path, encoding="utf-8-sig", sep=delimiter)
    required = {id_column, sequence_column}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return {
        str(row[id_column]): str(row[sequence_column]).strip().upper()
        for _, row in dataframe.iterrows()
    }


def residue_only_embedding(raw_embedding, sequence, protein_id):
    embedding = np.asarray(raw_embedding)
    if embedding.ndim != 2:
        raise ValueError(
            f"{protein_id}: expected 2-D ESM-C embedding, got {embedding.shape}"
        )
    sequence_length = len(sequence)
    if embedding.shape[0] == sequence_length + 2:
        return np.ascontiguousarray(embedding[1:-1]), "stripped_BOS_EOS"
    if embedding.shape[0] == sequence_length:
        return np.ascontiguousarray(embedding), "already_residue_only"
    raise ValueError(
        f"{protein_id}: ESM-C length {embedding.shape[0]} is neither "
        f"L={sequence_length} nor L+2={sequence_length + 2}"
    )


def build_cksaap_esmc(sequence, embedding, max_gap=4):
    """Reproduce the original k=4 CKSAAP-ESMC layout after token removal."""
    sequence_length, embedding_dim = embedding.shape
    if sequence_length != len(sequence):
        raise ValueError(
            f"CKSAAP input mismatch: embedding={sequence_length}, "
            f"sequence={len(sequence)}"
        )

    output = np.zeros(
        (max_gap + 1, embedding_dim, 20, 20), dtype=np.float32
    )
    embedding = embedding.astype(np.float32, copy=False)
    for gap in range(max_gap + 1):
        pair_count = sequence_length - gap - 1
        if pair_count <= 0:
            raise ValueError(
                f"Sequence length {sequence_length} is too short for gap {gap}"
            )
        for left_position in range(pair_count):
            right_position = left_position + gap + 1
            left_index = AA_TO_INDEX.get(sequence[left_position])
            right_index = AA_TO_INDEX.get(sequence[right_position])
            # Match the original implementation: unknown pairs do not enter one
            # of the 400 standard cells, but remain in the normalization count.
            if left_index is None or right_index is None:
                continue
            output[gap, :, left_index, right_index] += 0.5 * (
                embedding[left_position] + embedding[right_position]
            )
        output[gap] /= float(pair_count)
    return output.reshape((max_gap + 1) * embedding_dim, 20, 20)


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with open(temporary_path, "wb") as handle:
        np.save(handle, value, allow_pickle=True)
    os.replace(temporary_path, path)


def process_side(raw_path, sequence_path, id_column, sequence_column,
                 esmc_output_path, ckesmc_output_path, audit_only=False,
                 overwrite=False):
    raw_embeddings = np.load(raw_path, allow_pickle=True).item()
    sequences = load_sequences(sequence_path, id_column, sequence_column)
    missing_sequences = sorted(set(raw_embeddings).difference(sequences))
    if missing_sequences:
        raise KeyError(
            f"No sequence for {len(missing_sequences)} embeddings; "
            f"first IDs: {missing_sequences[:5]}"
        )

    aligned_embeddings = {}
    status_counts = {"stripped_BOS_EOS": 0, "already_residue_only": 0}
    for protein_id, raw_embedding in raw_embeddings.items():
        aligned, status = residue_only_embedding(
            raw_embedding, sequences[protein_id], protein_id
        )
        aligned_embeddings[protein_id] = aligned
        status_counts[status] += 1

    print(
        f"{raw_path}: {len(aligned_embeddings)} proteins; "
        f"stripped BOS/EOS={status_counts['stripped_BOS_EOS']}; "
        f"already residue-only={status_counts['already_residue_only']}"
    )
    if audit_only:
        return status_counts

    for output_path in (esmc_output_path, ckesmc_output_path):
        if Path(output_path).exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {output_path}; use --overwrite explicitly"
            )

    ckesmc = {}
    for index, (protein_id, embedding) in enumerate(aligned_embeddings.items(), 1):
        ckesmc[protein_id] = build_cksaap_esmc(
            sequences[protein_id], embedding, max_gap=4
        )
        print(
            f"Built CKSAAP-ESMC {index}/{len(aligned_embeddings)}: {protein_id}",
            flush=True,
        )

    atomic_save(esmc_output_path, aligned_embeddings)
    atomic_save(ckesmc_output_path, ckesmc)
    print(f"Saved residue-only ESM-C: {esmc_output_path}")
    print(f"Saved rebuilt CKSAAP-ESMC: {ckesmc_output_path}")
    return status_counts


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark_450ep_t050.yml")
    parser.add_argument(
        "--audit-only", action="store_true",
        help="validate raw ESM-C lengths without writing output caches",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace existing residue-only output caches",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)["data"]
    process_side(
        config["pro1_esmc_raw_path"], config["pro1_seq_path"],
        "protein_id", "sequence", config["pro1_esmc_path"],
        config["pro1_ckesmc_path"], args.audit_only, args.overwrite,
    )
    process_side(
        config["pro2_esmc_raw_path"], config["pro2_seq_path"],
        "protein_id", "sequence", config["pro2_esmc_path"],
        config["pro2_ckesmc_path"], args.audit_only, args.overwrite,
    )


if __name__ == "__main__":
    main()
