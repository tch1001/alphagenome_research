#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

if [[ "$#" -ne 1 || "$1" != "--acknowledge-structural-only-v3-3-4-5" ]]; then
  echo "usage: $0 --acknowledge-structural-only-v3-3-4-5" >&2
  exit 2
fi

exec python "${SCRIPT_DIR}/analyze_encoder_skip_ood_sidecar_v3_3_4_6.py" \
  --acknowledge-structural-only-v3-3-4-5
