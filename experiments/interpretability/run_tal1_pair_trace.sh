#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
workspace_dir="$(cd "$repo_dir/.." && pwd)"

# The host's LD_LIBRARY_PATH shadows the CUDA libraries bundled with JAX.
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false

exec "$workspace_dir/agvenv/bin/python" \
  "$repo_dir/experiments/interpretability/run_tal1_pair_trace.py" "$@"
