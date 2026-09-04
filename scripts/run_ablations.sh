#!/usr/bin/env bash
# Run the four single-component ablations from the repository root.
set -euo pipefail

for config in \
  configs/ablations/no_aac_dpc.yml \
  configs/ablations/no_cksaap_esmc.yml \
  configs/ablations/no_contrastive.yml \
  configs/ablations/no_cross_attention.yml
do
  python -u -m deepspy.train --config "$config"
done

