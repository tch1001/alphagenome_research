#!/usr/bin/env python3
"""Append-only JAX-only device preflight for the v3.3.2 OOD sidecar."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any
import warnings


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))
# pylint: disable=g-import-not-at-top
import run_device_preflight_v3_3 as v33_preflight
import validate_encoder_skip_ood_sidecar_bootstrap_v3_3_2 as bootstrap


SCRIPT_VERSION = 'opensplice-device-preflight-v3.3.2'
FREEZE_PATH = bootstrap.FREEZE_PATH
PREFLIGHT_DIR = bootstrap.PREFLIGHT_DIR
PREFIXES = ('XLA', 'JAX', 'CUDA', 'CUDNN', 'CUBLAS', 'TRITON')


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('--run', action='store_true')
  group.add_argument('--dry-run', action='store_true')
  return parser.parse_args()


def _json_bytes(value: Any) -> bytes:
  return (json.dumps(
      value, indent=2, sort_keys=True, allow_nan=False, default=str
  ) + '\n').encode('utf-8')


def assert_environment() -> dict[str, str]:
  environment = v33_preflight.assert_environment()
  if os.environ.get('JAX_ENABLE_COMPILATION_CACHE') != 'false':
    raise ValueError('JAX_ENABLE_COMPILATION_CACHE must be literal false.')
  return environment


def validate_freeze_and_bundle() -> dict[str, Any]:
  generated = (
      bootstrap.v33_bootstrap.proto_gate
      .validate_generated_bindings_before_import()
  )
  binding = bootstrap.validate_freeze()
  frozen = json.loads(FREEZE_PATH.read_text(encoding='utf-8'))
  expected = {
      'script_version': 'opensplice-encoder-skip-ood-sidecar-v3.3.2',
      'attempt_id': (
          'opensplice-v3.3.2-development-ood-sidecar-one-shot'
      ),
      'amendment_sha256': bootstrap.AMENDMENT_SHA256,
      'preflight_script_version': SCRIPT_VERSION,
      'preflight_dir': str(PREFLIGHT_DIR.resolve()),
  }
  for name, value in expected.items():
    if frozen.get(name) != value:
      raise ValueError(f'v3.3.2 preflight freeze mismatch: {name}.')
  return {
      **binding,
      'generated_bindings_pre_import': generated,
  }


def collect_observation() -> dict[str, Any]:
  environment = assert_environment()
  observation = v33_preflight.collect_observation()
  observation['v3_3_2_runtime_environment'] = environment
  observation['jax_enable_compilation_cache'] = False
  return observation


def _reserve_number() -> int:
  PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
  with (PREFLIGHT_DIR / '.allocation.lock').open('a+') as handle:
    fcntl.flock(handle, fcntl.LOCK_EX)
    numbers = []
    for path in PREFLIGHT_DIR.glob('preflight_*.json'):
      try:
        numbers.append(int(path.stem.removeprefix('preflight_')))
      except ValueError:
        pass
    number = max(numbers, default=-1) + 1
    reservation = PREFLIGHT_DIR / f'.preflight_{number:04d}.reserved'
    descriptor = os.open(
        reservation, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
    )
    os.close(descriptor)
  return number


def run_preflight() -> tuple[Path, bool]:
  frozen = validate_freeze_and_bundle()
  number = _reserve_number()
  prefix = f'preflight_{number:04d}'
  stdout_temp = PREFLIGHT_DIR / f'.{prefix}.{os.getpid()}.stdout.tmp'
  stderr_temp = PREFLIGHT_DIR / f'.{prefix}.{os.getpid()}.stderr.tmp'
  observation = None
  failure = None
  warning_records = []
  with v33_preflight.base._capture_file_descriptors(  # pylint: disable=protected-access
      stdout_temp, stderr_temp
  ):
    try:
      with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        observation = collect_observation()
      warning_records = [{
          'category': item.category.__name__,
          'message': str(item.message),
          'filename': item.filename,
          'lineno': item.lineno,
      } for item in caught]
    except Exception as error:  # pylint: disable=broad-exception-caught
      failure = {
          'type': type(error).__name__,
          'message': str(error),
          'traceback': ''.join(traceback.format_exception(error)),
      }
      traceback.print_exception(error, file=sys.stderr)
  stdout_path = PREFLIGHT_DIR / f'{prefix}.stdout.log'
  stderr_path = PREFLIGHT_DIR / f'{prefix}.stderr.log'
  stdout_sha = v33_preflight.base._write_new(  # pylint: disable=protected-access
      stdout_path, stdout_temp.read_bytes()
  )
  stderr_sha = v33_preflight.base._write_new(  # pylint: disable=protected-access
      stderr_path, stderr_temp.read_bytes()
  )
  stdout_temp.unlink(missing_ok=True)
  stderr_temp.unlink(missing_ok=True)
  passed = failure is None
  record = {
      'script_version': SCRIPT_VERSION,
      'status': 'pass' if passed else 'failure',
      'preflight_attempt_number': number,
      'amendment_sha256': bootstrap.AMENDMENT_SHA256,
      'original_protocol_sha256': bootstrap.ORIGINAL_PROTOCOL_SHA256,
      'freeze_sha256': frozen['sha256'],
      'freeze': frozen,
      'observation': observation,
      'failure': failure,
      'warnings': warning_records,
      'logs': {
          'stdout': {'path': str(stdout_path), 'sha256': stdout_sha},
          'stderr': {'path': str(stderr_path), 'sha256': stderr_sha},
      },
      'no_model_or_biological_access': True,
      'no_jit_or_array_kernel': True,
      'created_at_unix_s': time.time(),
  }
  record_path = PREFLIGHT_DIR / f'{prefix}.json'
  v33_preflight.base._write_new(  # pylint: disable=protected-access
      record_path, _json_bytes(record)
  )
  return record_path, passed


def build_dry_run_plan() -> dict[str, Any]:
  return {
      'script_version': SCRIPT_VERSION,
      'amendment_sha256': bootstrap.AMENDMENT_SHA256,
      'original_protocol_sha256': bootstrap.ORIGINAL_PROTOCOL_SHA256,
      'dry_run': True,
      'preflight_dir': str(PREFLIGHT_DIR),
      'scientific_output_created': False,
      'model_calls': 0,
      'jit_calls': 0,
      'required_device': {
          'default_backend': 'gpu',
          'visible_gpu_count': 1,
          'device_kind': v33_preflight.base.EXPECTED_DEVICE_KIND,
          'uuid': v33_preflight.base.EXPECTED_GPU_UUID,
          'compute_capability': v33_preflight.base.EXPECTED_COMPUTE_CAPABILITY,
      },
  }


def main() -> None:
  args = _parse_args()
  if args.dry_run:
    print(json.dumps(build_dry_run_plan(), indent=2))
    return
  path, passed = run_preflight()
  print(path.resolve())
  if not passed:
    raise SystemExit(2)


if __name__ == '__main__':
  main()
