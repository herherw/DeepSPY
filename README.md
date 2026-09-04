# DeepSPY

DeepSPY predicts S-RNase–SLF recognition specificity from protein sequences and
prioritizes candidate specificity residues. It is intended as a screening and
hypothesis-generation tool: predicted interactions and high-attention residues
require independent experimental validation.

## Repository structure

```text
DeepSPY/
├── deepspy/           # Model, data loader, training, prediction, and utilities
├── configs/           # Full model and four single-component ablations
├── data/              # Curated dataset, sequences, fixed folds, and examples
├── baselines/         # ML and structure-baseline implementations
├── results/           # Numerical results supporting the article
├── scripts/           # Commands for training, ablations, and baselines
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

Large checkpoints and precomputed features are deposited separately on Zenodo
and are not stored in GitHub.

## Dataset

`data/gold_pairs.tsv` contains 267 experimentally curated Gold pairs: 53
positive and 214 negative pairs. Energy-derived Silver labels are not used.
`data/splits/fold_assignments.tsv` records the exact five pair-stratified folds;
each pair appears in an outer test fold once.

## Installation

Python 3.12 and a CUDA-capable GPU are recommended.

```bash
git clone https://github.com/OWNER/DeepSPY.git
cd DeepSPY
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Install Zenodo artifacts

Download these three files from the associated Zenodo record:

```text
deepspy_checkpoints_v1.0.0.tar.gz
deepspy_features_v1.0.0.tar.gz
deepspy_source_data_v1.0.0.tar.gz  # optional for source-data analyses
```

Extract them into `artifacts/`:

```bash
mkdir -p artifacts
tar -xzf /path/to/deepspy_checkpoints_v1.0.0.tar.gz -C artifacts
tar -xzf /path/to/deepspy_features_v1.0.0.tar.gz -C artifacts
tar -xzf /path/to/deepspy_source_data_v1.0.0.tar.gz -C artifacts
```

The resulting paths used by the code are
`artifacts/checkpoints/fold_1_best_model.pkl` through fold 5 and
`artifacts/features/`.

## Training and ablation experiments

Run all commands from the repository root.

```bash
# Full fivefold model
bash scripts/run_cv.sh

# Four ablations: AAC/DPC, CKSAAP-ESMC, contrastive module, and attention module
bash scripts/run_ablations.sh
```

The equivalent full-model command is:

```bash
python -u -m deepspy.train --config configs/full_model.yml
```

## Baselines

```bash
bash scripts/run_baselines.sh
```

The machine-learning baselines use AAC/DPC plus mean-pooled ESM-C and the same
fixed outer folds, without inner cross-validation. Chai-1 and AlphaFold 3 are
evaluated as zero-shot structural baselines using ipTM and a fixed threshold of
0.6. AUPRC is reported as average precision (AP).

## Prediction

For a pair included in the released feature dictionaries:

```bash
python -m deepspy.predict \
  --model_path artifacts/checkpoints/fold_1_best_model.pkl \
  --protein1 PhS3 \
  --protein2 PhS9-SLF1 \
  --details \
  --out_dir outputs/predictions/PhS3_PhS9-SLF1
```

Use all five checkpoints to assess fold-wise prediction stability. Novel
sequences require AAC, DPC, residue-aligned ESM-C, and CKSAAP-ESMC features in
the same formats used during training.

## Reported results retained in GitHub

- `results/oof_predictions.tsv`: all 267 DeepSPY out-of-fold calls.
- `results/ablations/`: fold-level and mean ± SD ablation metrics.
- `results/baselines/`: input structure scores and fold-level baseline
  metrics.
- `results/residue_prioritization/`: representative candidate residues.

## Interpretation and limitations

Cross-attention enters classification through a learned gated interaction
logit, but attention scores remain exploratory rankings rather than causal
effects. The dataset is small and imbalanced, and pair-level splitting allows
individual proteins to recur across training and test pairs.

The archived paper implementation evaluates the outer test fold after each
epoch and retains the checkpoint with the highest test-fold MCC. This may
optimistically bias the reported performance and should be disclosed when the
legacy results are described. Future confirmatory evaluation should select the
epoch using training-only information and evaluate an untouched test set once.

## Citation and license

Before release, replace the placeholder authors and GitHub URL in
`CITATION.cff` and `.zenodo.json`. An article DOI is not required before the
article is published. Code is provided under the [Apache License 2.0](LICENSE).
