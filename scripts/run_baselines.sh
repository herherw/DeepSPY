#!/usr/bin/env bash
# Run all reported baseline families from the repository root.
set -euo pipefail

python -u baselines/run_ml_baselines.py --config baselines/config_ml.json
python -u baselines/run_structure_baselines.py --config baselines/config_structure.json

