#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON_DEFAULT="$(dirname -- "$REPO_ROOT")/agvenv/bin/python"
PYTHON_BIN="${ALPHAGENOME_PYTHON:-$PYTHON_DEFAULT}"

unset LD_LIBRARY_PATH XLA_FLAGS JAX_COMPILATION_CACHE_DIR \
  CUDA_CACHE_PATH CUDA_CACHE_MAXSIZE \
  TRITON_DUMP_DIR TRITON_OVERRIDE_DIR \
  ALPHAGENOME_V3_3_3_CACHE_ROLE ALPHAGENOME_V3_3_3_CACHE_ROOT \
  TRITON_CACHE_DIR XDG_CACHE_HOME
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
export CUDA_CACHE_DISABLE=1
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

prepare_cache_root() {
  local role="$1"
  local root="$2"
  if ! mkdir -- "$root"; then
    echo "Fresh v3.3.3 cache root reservation failed: $root" >&2
    exit 65
  fi
  mkdir -- "$root/triton" "$root/xdg"
  export ALPHAGENOME_V3_3_3_CACHE_ROLE="$role"
  export ALPHAGENOME_V3_3_3_CACHE_ROOT="$root"
  export TRITON_CACHE_DIR="$root/triton"
  export XDG_CACHE_HOME="$root/xdg"
}

if [[ "$DRY_RUN" == true ]]; then
  DRY_CACHE_ROOT="$(mktemp -d /tmp/alphagenome-v3.3.3-dry-cache.XXXXXX)"
  mkdir -p -- "$DRY_CACHE_ROOT/triton" "$DRY_CACHE_ROOT/xdg"
  export ALPHAGENOME_V3_3_3_CACHE_ROLE=dry_run
  export ALPHAGENOME_V3_3_3_CACHE_ROOT="$DRY_CACHE_ROOT"
  export TRITON_CACHE_DIR="$DRY_CACHE_ROOT/triton"
  export XDG_CACHE_HOME="$DRY_CACHE_ROOT/xdg"
else
  prepare_cache_root external_preflight \
    "$SCRIPT_DIR/results/v3_3_3_preflight_kernel_cache"
fi

"$PYTHON_BIN" \
  "$SCRIPT_DIR/validate_encoder_skip_ood_sidecar_bootstrap_v3_3_3.py" \
  --check

if [[ "$DRY_RUN" == true ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/run_device_preflight_v3_3_3.py" --dry-run
  exec "$PYTHON_BIN" \
    "$SCRIPT_DIR/launch_encoder_skip_ood_sidecar_v3_3_3.py" "$@"
fi

SUCCESSFUL_PREFLIGHT="$(
  "$PYTHON_BIN" "$SCRIPT_DIR/run_device_preflight_v3_3_3.py" --run
)"
prepare_cache_root model "$SCRIPT_DIR/results/v3_3_3_model_kernel_cache"
exec "$PYTHON_BIN" \
  "$SCRIPT_DIR/launch_encoder_skip_ood_sidecar_v3_3_3.py" \
  --successful-preflight "$SUCCESSFUL_PREFLIGHT" "$@"
