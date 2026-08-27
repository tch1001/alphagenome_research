#!/usr/bin/env bash
set -euo pipefail

LOCK_COMMIT='fd4dc6913335a6966420d60ef04bc4643b751a27'
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
LOCKED_CHECKOUT_DEFAULT="$(dirname -- "$REPO_ROOT")/alphagenome_research_exact_fd4dc6913335"
LOCKED_CHECKOUT="${ALPHAGENOME_EXACT_SOURCE_CHECKOUT:-$LOCKED_CHECKOUT_DEFAULT}"
PYTHON_DEFAULT="$(dirname -- "$REPO_ROOT")/agvenv/bin/python"
PYTHON_BIN="${ALPHAGENOME_PYTHON:-$PYTHON_DEFAULT}"

# This correction precedes every Python/JAX process, including dry-run helper
# discovery.  An empty LD_LIBRARY_PATH is not equivalent to an absent one.
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Exact-source Python is not executable: $PYTHON_BIN" >&2
  exit 64
fi

DRY_RUN=false
for argument in "$@"; do
  if [[ "$argument" == '--dry-run' ]]; then
    DRY_RUN=true
  fi
done

SUCCESSFUL_PREFLIGHT=''
if [[ "$DRY_RUN" == false ]]; then
  # This subprocess imports only JAX/JAXLIB and creates no scientific output.
  # Failed infrastructure records remain durable and may be followed by a new
  # preflight only after the environment is repaired.
  SUCCESSFUL_PREFLIGHT="$(
    "$PYTHON_BIN" "$SCRIPT_DIR/run_device_preflight_v3_1_2.py" --run
  )"
fi

if [[ ! -e "$LOCKED_CHECKOUT/.git" ]]; then
  if [[ -e "$LOCKED_CHECKOUT" ]]; then
    echo "Refusing non-worktree path: $LOCKED_CHECKOUT" >&2
    exit 64
  fi
  git -C "$REPO_ROOT" worktree add --detach "$LOCKED_CHECKOUT" "$LOCK_COMMIT"
fi

OBSERVED_HEAD="$(git -C "$LOCKED_CHECKOUT" rev-parse HEAD)"
if [[ "$OBSERVED_HEAD" != "$LOCK_COMMIT" ]]; then
  echo "Exact-source worktree is at $OBSERVED_HEAD, expected $LOCK_COMMIT" >&2
  exit 65
fi
if [[ -n "$(git -C "$LOCKED_CHECKOUT" status --porcelain --untracked-files=no)" ]]; then
  echo "Exact-source worktree has tracked changes" >&2
  exit 65
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$LOCKED_CHECKOUT/src${PYTHONPATH:+:$PYTHONPATH}"
export ALPHAGENOME_PROTOC_BIN="${ALPHAGENOME_PROTOC:-$(command -v protoc || true)}"
if [[ -z "$ALPHAGENOME_PROTOC_BIN" || ! -x "$ALPHAGENOME_PROTOC_BIN" ]]; then
  echo "protoc is required for the declared repository build artifact" >&2
  exit 66
fi
export ALPHAGENOME_PROTO_ROOT="$(
  "$PYTHON_BIN" -c \
    'import importlib.resources; print(importlib.resources.files("alphagenome").parent)'
)"

RUNNER_ARGS=(--locked-checkout "$LOCKED_CHECKOUT")
if [[ "$DRY_RUN" == false ]]; then
  RUNNER_ARGS+=(--successful-preflight "$SUCCESSFUL_PREFLIGHT")
fi
exec "$PYTHON_BIN" \
  "$SCRIPT_DIR/run_exact_source_gate0_v3_1_2.py" \
  "${RUNNER_ARGS[@]}" \
  "$@"
