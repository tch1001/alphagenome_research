#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="/home/degen2/alphafold-stuff/alphagenome_research"
readonly OPEN_SPLICE="${REPO_ROOT}/experiments/interpretability/opensplice"
readonly RUN_DIR="${OPEN_SPLICE}/results/v3_3_2_development_ood_sidecar_one_shot"
readonly ANALYSIS_DIR="${OPEN_SPLICE}/results/v3_3_2_development_ood_sidecar_analysis_v3_3_2_2"
readonly ANALYZER="${OPEN_SPLICE}/analyze_encoder_skip_ood_sidecar_v3_3_2_2.py"

exec /usr/bin/python3 "${ANALYZER}" \
  --run-dir "${RUN_DIR}" \
  --bundle-root "${REPO_ROOT}" \
  --output-json "${ANALYSIS_DIR}/ANALYSIS.json" \
  --output-markdown "${ANALYSIS_DIR}/RESULT.md"
