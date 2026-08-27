#!/usr/bin/env bash
set -euo pipefail

LOCK_COMMIT='fd4dc6913335a6966420d60ef04bc4643b751a27'
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
LOCKED_CHECKOUT_DEFAULT="$(dirname -- "$REPO_ROOT")/alphagenome_research_exact_fd4dc6913335"
LOCKED_CHECKOUT="${ALPHAGENOME_EXACT_SOURCE_CHECKOUT:-$LOCKED_CHECKOUT_DEFAULT}"
PYTHON_DEFAULT="$(dirname -- "$REPO_ROOT")/agvenv/bin/python"
PYTHON_BIN="${ALPHAGENOME_PYTHON:-$PYTHON_DEFAULT}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Exact-source Python is not executable: $PYTHON_BIN" >&2
  exit 64
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
  echo "protoc is required for the declared repository build-hook outputs" >&2
  exit 66
fi
export ALPHAGENOME_PROTO_ROOT="$(
  "$PYTHON_BIN" -c \
    'import importlib.resources; print(importlib.resources.files("alphagenome").parent)'
)"
exec "$PYTHON_BIN" \
  "$SCRIPT_DIR/run_exact_source_gate0_v3_1_1.py" \
  --locked-checkout "$LOCKED_CHECKOUT" \
  "$@"
