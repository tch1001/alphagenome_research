#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON_DEFAULT="$(dirname -- "$REPO_ROOT")/agvenv/bin/python"
PYTHON_BIN="${ALPHAGENOME_PYTHON:-$PYTHON_DEFAULT}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "AlphaGenome Python is not executable: $PYTHON_BIN" >&2
  exit 64
fi

if (( $# > 1 )) || { (( $# == 1 )) && [[ "$1" != '--dry-run' ]]; }; then
  echo "v3.3.4.4 accepts no production options; --dry-run is the sole option." >&2
  exit 64
fi

required_authorization=(
  V3344_AUTHORIZED_GIT_HEAD
  V3344_AUTHORIZED_FREEZE_SHA256
  V3344_AUTHORIZED_FREEZE_SIZE_BYTES
)
for name in "${required_authorization[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing external post-commit authorization: $name" >&2
    exit 64
  fi
done
while IFS= read -r name; do
  case "$name" in
    V3344_AUTHORIZED_GIT_HEAD|V3344_AUTHORIZED_FREEZE_SHA256|V3344_AUTHORIZED_FREEZE_SIZE_BYTES) ;;
    *)
      echo "Unapproved v3.3.4.4 authorization variable: $name" >&2
      exit 64
      ;;
  esac
done < <(compgen -e | grep '^V3344_AUTHORIZED_' || true)

unset LD_LIBRARY_PATH XLA_FLAGS JAX_COMPILATION_CACHE_DIR \
  CUDA_CACHE_PATH CUDA_CACHE_MAXSIZE TRITON_DUMP_DIR TRITON_OVERRIDE_DIR \
  ALPHAGENOME_V3_3_4_3_CACHE_ROLE ALPHAGENOME_V3_3_4_3_CACHE_ROOT \
  ALPHAGENOME_V3_3_4_4_CACHE_ROLE ALPHAGENOME_V3_3_4_4_CACHE_ROOT \
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

# The stdlib launcher owns Gate A and every later allocation in one process.
# This shell deliberately creates none of the six frozen production paths.
exec "$PYTHON_BIN" \
  "$SCRIPT_DIR/launch_encoder_skip_ood_sidecar_v3_3_4_4.py" "$@"
