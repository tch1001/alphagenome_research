#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON_DEFAULT="$(dirname -- "$REPO_ROOT")/agvenv/bin/python"
PYTHON_BIN="${ALPHAGENOME_PYTHON:-$PYTHON_DEFAULT}"

unset LD_LIBRARY_PATH XLA_FLAGS JAX_COMPILATION_CACHE_DIR
while IFS= read -r variable_name; do
  upper_name="${variable_name^^}"
  if [[ "$variable_name" == JAX_PERSISTENT_CACHE_* ]] || \
     { [[ "$upper_name" == *AUTOTUNE* ]] && \
       [[ "$upper_name" =~ (LOAD|DUMP|CACHE) ]]; }; then
    unset "$variable_name"
  fi
done < <(compgen -e)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_COMPILATION_CACHE=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO_ROOT/src:$(dirname -- "$REPO_ROOT")/alphagenome/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "AlphaGenome Python is not executable: $PYTHON_BIN" >&2
  exit 64
fi

DRY_RUN=false
for argument in "$@"; do
  if [[ "$argument" == '--successful-preflight' ]] || \
     [[ "$argument" == --successful-preflight=* ]]; then
    echo "--successful-preflight is reserved for this wrapper" >&2
    exit 64
  fi
  if [[ "$argument" == '--dry-run' ]]; then
    DRY_RUN=true
  fi
done

"$PYTHON_BIN" \
  "$SCRIPT_DIR/validate_encoder_skip_ood_sidecar_bootstrap_v3_3_2.py" \
  --check

if [[ "$DRY_RUN" == true ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/run_device_preflight_v3_3_2.py" --dry-run
  exec "$PYTHON_BIN" \
    "$SCRIPT_DIR/launch_encoder_skip_ood_sidecar_v3_3_2.py" "$@"
fi

SUCCESSFUL_PREFLIGHT="$(
  "$PYTHON_BIN" "$SCRIPT_DIR/run_device_preflight_v3_3_2.py" --run
)"
exec "$PYTHON_BIN" \
  "$SCRIPT_DIR/launch_encoder_skip_ood_sidecar_v3_3_2.py" \
  --successful-preflight "$SUCCESSFUL_PREFLIGHT" "$@"
