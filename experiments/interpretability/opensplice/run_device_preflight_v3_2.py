#!/usr/bin/env python3
"""Append-only JAX-only device/environment preflight for OpenSplice v3.2."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any
import warnings


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))
import run_device_preflight_v3_1_2 as base  # pylint: disable=g-import-not-at-top
import validate_superset_graph_bootstrap_v3_2 as bootstrap  # pylint: disable=g-import-not-at-top


SCRIPT_VERSION = 'opensplice-device-preflight-v3.2.0'
PROTOCOL_PATH = (
    _HERE / 'v3_wider_mechanism' / 'superset_graph_protocol_v3_2.md'
)
PROTOCOL_SHA256 = (
    '1e87839250e838d1d9aa95162e76a3acb6b1eaab59b481c60dab943b62f6caea'
)
FREEZE_PATH = _HERE / 'superset_graph_v3_2_freeze.json'
PREFLIGHT_DIR = _HERE / 'results' / 'v3_2_device_preflight'
PREFIXES = ('XLA', 'JAX', 'CUDA', 'CUDNN', 'CUBLAS', 'TRITON')


def _reject_confirmation_path(path: Path) -> None:
  if any('confirm' in part.lower() for part in path.resolve().parts):
    raise ValueError(f'Confirmation-named path is forbidden: {path}.')


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('--run', action='store_true')
  group.add_argument('--dry-run', action='store_true')
  return parser.parse_args()


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
  return (json.dumps(
      value, indent=2, sort_keys=True, allow_nan=False, default=str
  ) + '\n').encode('utf-8')


def assert_environment() -> dict[str, str]:
  base.assert_sanitized_environment()
  if os.environ.get('JAX_ENABLE_COMPILATION_CACHE') != 'false':
    raise ValueError('JAX_ENABLE_COMPILATION_CACHE must be literal false.')
  forbidden = [
      name for name in ('XLA_FLAGS', 'JAX_COMPILATION_CACHE_DIR')
      if name in os.environ
  ]
  forbidden.extend(
      name for name in os.environ if name.startswith('JAX_PERSISTENT_CACHE_')
  )
  forbidden.extend(
      name for name in os.environ
      if 'AUTOTUNE' in name.upper()
      and any(term in name.upper() for term in ('LOAD', 'DUMP', 'CACHE'))
  )
  if forbidden:
    raise ValueError(f'Forbidden cache/autotune environment: {forbidden}.')
  return {
      name: value for name, value in sorted(os.environ.items())
      if name.startswith(PREFIXES)
  }


def validate_freeze_and_bundle() -> dict[str, Any]:
  repo = _HERE.parents[2]
  generated_bindings = bootstrap.validate_generated_bindings_before_import()
  for path in (FREEZE_PATH, PROTOCOL_PATH, PREFLIGHT_DIR):
    _reject_confirmation_path(path)
  if _sha256(PROTOCOL_PATH) != PROTOCOL_SHA256:
    raise ValueError('v3.2 protocol hash mismatch.')
  frozen = json.loads(FREEZE_PATH.read_text(encoding='utf-8'))
  expected = {
      'script_version': 'opensplice-superset-graph-v3.2.0',
      'attempt_id': 'opensplice-v3.2-development-superset-graph-one-shot',
      'protocol_sha256': PROTOCOL_SHA256,
      'preflight_script_version': SCRIPT_VERSION,
      'preflight_dir': str(PREFLIGHT_DIR.resolve()),
  }
  for name, value in expected.items():
    if frozen.get(name) != value:
      raise ValueError(f'v3.2 preflight freeze mismatch: {name}.')
  paths = [FREEZE_PATH]
  for relative, digest in frozen['file_sha256'].items():
    path = repo / relative
    _reject_confirmation_path(path)
    if _sha256(path) != digest:
      raise ValueError(f'v3.2 frozen file changed: {relative}.')
    paths.append(path)
  relative_paths = tuple(str(path.relative_to(repo)) for path in paths)
  for relative in relative_paths:
    subprocess.run(
        ('git', '-C', str(repo), 'ls-files', '--error-unmatch', relative),
        check=True, capture_output=True,
    )
  if subprocess.check_output(
      ('git', '-C', str(repo), 'diff', '--binary', 'HEAD', '--',
       *relative_paths)
  ):
    raise ValueError('Committed v3.2 bundle differs from working tree.')
  return {
      'path': str(FREEZE_PATH.resolve()),
      'sha256': _sha256(FREEZE_PATH),
      'git_head': subprocess.check_output(
          ('git', '-C', str(repo), 'rev-parse', 'HEAD'), text=True
      ).strip(),
      'tracked_clean': True,
      'generated_bindings_pre_import': generated_bindings,
  }


def collect_observation() -> dict[str, Any]:
  environment = assert_environment()
  observation = base.collect_device_observation()
  import jax  # pylint: disable=g-import-not-at-top
  if bool(jax.config.jax_enable_compilation_cache):
    raise ValueError('JAX runtime compilation cache is enabled.')
  base.validate_device_observation(observation)
  observation['v3_2_runtime_environment'] = environment
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
  with base._capture_file_descriptors(  # pylint: disable=protected-access
      stdout_temp, stderr_temp
  ):
    try:
      with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        observation = collect_observation()
      warning_records = [{
          'category': item.category.__name__, 'message': str(item.message),
          'filename': item.filename, 'lineno': item.lineno,
      } for item in caught]
    except Exception as error:  # durable infrastructure failures
      failure = {
          'type': type(error).__name__, 'message': str(error),
          'traceback': ''.join(traceback.format_exception(error)),
      }
      traceback.print_exception(error, file=sys.stderr)
  stdout_path = PREFLIGHT_DIR / f'{prefix}.stdout.log'
  stderr_path = PREFLIGHT_DIR / f'{prefix}.stderr.log'
  stdout_sha = base._write_new(  # pylint: disable=protected-access
      stdout_path, stdout_temp.read_bytes()
  )
  stderr_sha = base._write_new(  # pylint: disable=protected-access
      stderr_path, stderr_temp.read_bytes()
  )
  stdout_temp.unlink(missing_ok=True)
  stderr_temp.unlink(missing_ok=True)
  passed = failure is None
  record = {
      'script_version': SCRIPT_VERSION,
      'status': 'pass' if passed else 'failure',
      'preflight_attempt_number': number,
      'protocol_sha256': PROTOCOL_SHA256,
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
  base._write_new(record_path, _json_bytes(record))  # pylint: disable=protected-access
  return record_path, passed


def build_dry_run_plan() -> dict[str, Any]:
  return {
      'script_version': SCRIPT_VERSION,
      'protocol_sha256': PROTOCOL_SHA256,
      'dry_run': True,
      'preflight_dir': str(PREFLIGHT_DIR),
      'scientific_output_created': False,
      'imports_if_run': ['jax', 'jaxlib'],
      'model_calls': 0,
      'jit_calls': 0,
      'required_environment': {
          'LD_LIBRARY_PATH': 'absent',
          'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
          'JAX_ENABLE_COMPILATION_CACHE': 'false',
          'compiler_and_autotune_cache_inputs': 'absent',
      },
      'required_device': {
          'default_backend': 'gpu',
          'visible_gpu_count': 1,
          'device_kind': base.EXPECTED_DEVICE_KIND,
          'uuid': base.EXPECTED_GPU_UUID,
          'compute_capability': base.EXPECTED_COMPUTE_CAPABILITY,
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
