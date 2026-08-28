#!/usr/bin/env python3
"""Validate v3.3 frozen bytes, then load the model runner in-process."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import runpy
import sys
import time
import types


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))
# pylint: disable=g-import-not-at-top
import validate_encoder_skip_bootstrap_v3_3 as bootstrap


def _sanitize_environment() -> dict[str, str]:
  for name in ('LD_LIBRARY_PATH', 'XLA_FLAGS', 'JAX_COMPILATION_CACHE_DIR'):
    os.environ.pop(name, None)
  for name in tuple(os.environ):
    upper = name.upper()
    if name.startswith('JAX_PERSISTENT_CACHE_') or (
        'AUTOTUNE' in upper
        and any(term in upper for term in ('LOAD', 'DUMP', 'CACHE'))
    ):
      os.environ.pop(name, None)
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
  os.environ['JAX_ENABLE_COMPILATION_CACHE'] = 'false'
  os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
  return {
      'LD_LIBRARY_PATH': 'absent',
      'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
      'JAX_ENABLE_COMPILATION_CACHE': 'false',
  }


def main() -> None:
  environment = _sanitize_environment()
  generated = bootstrap.proto_gate.validate_generated_bindings_before_import()
  frozen = bootstrap.validate_freeze()
  attestation = types.ModuleType('_opensplice_v3_3_bootstrap_attestation')
  attestation.record = {
      'pid': __import__('os').getpid(),
      'created_at_unix_s': time.time(),
      'generated_bindings': generated,
      'sanitized_environment': environment,
      'freeze': frozen,
      'launcher_path': str(Path(__file__).resolve()),
      'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'bootstrap_path': str(Path(bootstrap.__file__).resolve()),
      'bootstrap_sha256': hashlib.sha256(
          Path(bootstrap.__file__).read_bytes()
      ).hexdigest(),
  }
  sys.modules[attestation.__name__] = attestation
  runpy.run_path(
      str(_HERE / 'run_encoder_skip_factorial_v3_3.py'), run_name='__main__'
  )


if __name__ == '__main__':
  main()
