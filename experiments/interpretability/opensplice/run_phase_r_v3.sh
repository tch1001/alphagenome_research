#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
workspace_dir="$(cd "$repo_dir/.." && pwd)"

unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false

exec "$workspace_dir/agvenv/bin/python" \
  "$repo_dir/experiments/interpretability/opensplice/run_phase_r_v3.py" "$@"
