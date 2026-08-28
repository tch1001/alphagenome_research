#!/usr/bin/env bash
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

if [[ "$#" -ne 7 \
      || "$1" != "--acknowledge-structural-only-v3-3-4-5-1" \
      || "$2" != "--authorized-git-head" \
      || ! "$3" =~ ^[0-9a-f]{40}$ \
      || "$4" != "--authorized-freeze-sha256" \
      || ! "$5" =~ ^[0-9a-f]{64}$ \
      || "$6" != "--authorized-freeze-size-bytes" \
      || ! "$7" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "usage: $0 --acknowledge-structural-only-v3-3-4-5-1 --authorized-git-head <40hex> --authorized-freeze-sha256 <64hex> --authorized-freeze-size-bytes <integer>" >&2
  exit 2
fi

exec /usr/bin/python3 \
  "${SCRIPT_DIR}/analyze_encoder_skip_ood_sidecar_v3_3_4_5_1.py" \
  "$1" "$2" "$3" "$4" "$5" "$6" "$7"
