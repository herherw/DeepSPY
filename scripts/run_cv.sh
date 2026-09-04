#!/usr/bin/env bash
# Reproduce the full fivefold DeepSPY experiment from the repository root.
set -euo pipefail

python -u -m deepspy.train --config configs/full_model.yml
