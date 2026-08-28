#!/usr/bin/env python3
"""Validate v3.3.3 frozen bytes, then load the sidecar in-process."""

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
import validate_encoder_skip_ood_sidecar_bootstrap_v3_3_3 as bootstrap


ATTESTATION_MODULE = '_opensplice_v3_3_3_ood_sidecar_bootstrap_attestation'


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _sanitize_environment() -> dict[str, str]:
  for name in ('LD_LIBRARY_PATH', *bootstrap.DENIED_CACHE_ENVIRONMENT_NAMES):
    os.environ.pop(name, None)
  for name in tuple(os.environ):
    upper = name.upper()
    if any(
        name.startswith(prefix)
        for prefix in bootstrap.DENIED_CACHE_ENVIRONMENT_PREFIXES
    ) or (
        'AUTOTUNE' in upper
        and any(term in upper for term in ('LOAD', 'DUMP', 'CACHE'))
    ):
      os.environ.pop(name, None)
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
  os.environ['JAX_ENABLE_COMPILATION_CACHE'] = 'false'
  os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
  cache_environment = bootstrap.assert_cache_environment_sanitized()
  return {
      'LD_LIBRARY_PATH': 'absent',
      'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
      'JAX_ENABLE_COMPILATION_CACHE': 'false',
      'cache_environment': cache_environment,
  }


def main() -> None:
  environment = _sanitize_environment()
  generated = (
      bootstrap.v33_bootstrap.proto_gate
      .validate_generated_bindings_before_import()
  )
  frozen = bootstrap.validate_freeze()
  attestation = types.ModuleType(ATTESTATION_MODULE)
  attestation.record = {
      'pid': os.getpid(),
      'created_at_unix_s': time.time(),
      'generated_bindings': generated,
      'sanitized_environment': environment,
      'freeze': frozen,
      'launcher_path': str(Path(__file__).resolve()),
      'launcher_sha256': _sha256(Path(__file__)),
      'bootstrap_path': str(Path(bootstrap.__file__).resolve()),
      'bootstrap_sha256': _sha256(Path(bootstrap.__file__)),
  }
  sys.modules[ATTESTATION_MODULE] = attestation
  runpy.run_path(
      str(_HERE / 'run_encoder_skip_ood_sidecar_v3_3_3.py'),
      run_name='__main__',
  )


if __name__ == '__main__':
  main()
