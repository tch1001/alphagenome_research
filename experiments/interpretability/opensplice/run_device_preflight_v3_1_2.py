#!/usr/bin/env python3
"""Append-only JAX-only GPU preflight for exact-source diagnostic v3.1.2.

This process imports no AlphaGenome, AlphaGenome Research, OpenSplice helper,
checkpoint, FASTA, manifest, or variant module.  It performs no JIT or array
operation.  Failed infrastructure checks are durable and repeatable; they do
not create or consume the scientific one-shot attempt.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence
import warnings


SCRIPT_VERSION = 'opensplice-device-preflight-v3.1.2'
EXPECTED_DEVICE_KIND = 'NVIDIA GeForce RTX 3090'
EXPECTED_GPU_UUID = 'GPU-64111645-1e42-a96d-f192-4abbec4b8090'
EXPECTED_COMPUTE_CAPABILITY = '8.6'
_HERE = Path(__file__).resolve().parent
PREFLIGHT_DIR = _HERE / 'results' / 'v3_1_2_device_preflight'
FREEZE_PATH = _HERE / 'exact_source_gate0_v3_1_2_freeze.json'
PROTOCOL_PATH = (
    _HERE
    / 'v3_wider_mechanism'
    / 'device_preflight_amendment_v3_1_2.md'
)
PROTOCOL_SHA256 = (
    '5134bda6a004f101c948b0aabfa3a992849398e4a768fd24f329736fa03d2822'
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run', action='store_true')
  parser.add_argument('--dry-run', action='store_true')
  args = parser.parse_args()
  if args.run == args.dry_run:
    parser.error('Choose exactly one of --run or --dry-run.')
  return args


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
  return (
      json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str)
      + '\n'
  ).encode('utf-8')


def _write_new(path: Path, data: bytes) -> str:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
  descriptor = os.open(
      temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
  )
  try:
    with os.fdopen(descriptor, 'wb') as handle:
      handle.write(data)
      handle.flush()
      os.fsync(handle.fileno())
    try:
      os.link(temporary, path)
    except FileExistsError as error:
      raise FileExistsError(f'Append-only preflight exists: {path}.') from error
  finally:
    temporary.unlink(missing_ok=True)
  return hashlib.sha256(data).hexdigest()


def assert_sanitized_environment() -> dict[str, Any]:
  library_present = 'LD_LIBRARY_PATH' in os.environ
  preallocate = os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')
  if library_present:
    raise ValueError('LD_LIBRARY_PATH must be absent, not empty.')
  if preallocate != 'false':
    raise ValueError(
        'XLA_PYTHON_CLIENT_PREALLOCATE must equal lowercase "false".'
    )
  return {
      'LD_LIBRARY_PATH': {'present': False, 'value': None},
      'XLA_PYTHON_CLIENT_PREALLOCATE': preallocate,
  }


def _package_version(name: str) -> str | None:
  try:
    return importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:
    return None


def _runtime_environment() -> dict[str, str | None]:
  prefixes = ('JAX_', 'CUDA_', 'NVIDIA_')
  exact = ('XLA_FLAGS', 'TF_XLA_FLAGS', 'LD_LIBRARY_PATH', 'PATH')
  names = set(exact)
  names.update(
      name for name in os.environ if name.startswith(prefixes)
  )
  return {name: os.environ.get(name) for name in sorted(names)}


def _nvidia_smi() -> dict[str, Any]:
  command = (
      'nvidia-smi',
      '--query-gpu=index,name,uuid,compute_cap,vbios_version,driver_version',
      '--format=csv,noheader',
  )
  result = subprocess.run(
      command, check=False, capture_output=True, text=True, timeout=10
  )
  lines = tuple(
      line.strip() for line in result.stdout.splitlines() if line.strip()
  )
  parsed = None
  if result.returncode == 0 and len(lines) == 1:
    fields = tuple(part.strip() for part in lines[0].split(','))
    if len(fields) == 6:
      parsed = dict(zip(
          ('index', 'name', 'uuid', 'compute_capability', 'vbios', 'driver'),
          fields,
          strict=True,
      ))
  return {
      'command': command,
      'returncode': result.returncode,
      'stdout': result.stdout,
      'stderr': result.stderr,
      'lines': lines,
      'parsed_single_gpu': parsed,
  }


def validate_device_observation(observation: Mapping[str, Any]) -> None:
  if observation['environment']['LD_LIBRARY_PATH']['present']:
    raise ValueError('Preflight observed LD_LIBRARY_PATH.')
  if observation['environment']['XLA_PYTHON_CLIENT_PREALLOCATE'] != 'false':
    raise ValueError('Preflight observed invalid JAX preallocation setting.')
  if observation['jax_default_backend'] != 'gpu':
    raise ValueError(
        f'JAX default backend is {observation["jax_default_backend"]!r}, not gpu.'
    )
  devices = observation['jax_gpu_devices']
  if len(devices) != 1:
    raise ValueError(f'Expected exactly one JAX GPU, observed {len(devices)}.')
  device = devices[0]
  if device['platform'] != 'gpu':
    raise ValueError(f'JAX device platform is {device["platform"]!r}.')
  if device['device_kind'] != EXPECTED_DEVICE_KIND:
    raise ValueError(
        f'JAX device kind is {device["device_kind"]!r}, expected '
        f'{EXPECTED_DEVICE_KIND!r}.'
    )
  smi = observation['nvidia_smi']
  if smi['returncode'] != 0 or len(smi['lines']) != 1:
    raise ValueError('nvidia-smi did not return exactly one visible GPU.')
  physical = smi['parsed_single_gpu']
  if physical is None:
    raise ValueError('nvidia-smi result could not be parsed exactly.')
  expected = {
      'index': '0',
      'name': EXPECTED_DEVICE_KIND,
      'uuid': EXPECTED_GPU_UUID,
      'compute_capability': EXPECTED_COMPUTE_CAPABILITY,
  }
  for name, value in expected.items():
    if physical[name] != value:
      raise ValueError(
          f'nvidia-smi {name} is {physical[name]!r}, expected {value!r}.'
      )


def collect_device_observation() -> dict[str, Any]:
  """Initializes only the JAX backend and performs no array/JIT operation."""
  environment = assert_sanitized_environment()
  jax = importlib.import_module('jax')
  jaxlib = importlib.import_module('jaxlib')
  gpu_devices = jax.devices('gpu')
  observation = {
      'environment': environment,
      'runtime_environment': _runtime_environment(),
      'python_executable': sys.executable,
      'python_version': sys.version,
      'platform': platform.platform(),
      'kernel': platform.release(),
      'hostname': socket.gethostname(),
      'pid': os.getpid(),
      'packages': {
          name: _package_version(name) for name in (
              'jax', 'jaxlib', 'jax-cuda12-plugin', 'jax-cuda12-pjrt',
              'nvidia-cuda-runtime-cu12', 'nvidia-cudnn-cu12',
              'nvidia-cublas-cu12', 'nvidia-cusparse-cu12',
          )
      },
      'jax_module_version': getattr(jax, '__version__', None),
      'jaxlib_module_version': getattr(jaxlib, '__version__', None),
      'jax_default_backend': jax.default_backend(),
      'jax_gpu_devices': tuple({
          'id': getattr(device, 'id', None),
          'platform': getattr(device, 'platform', None),
          'device_kind': getattr(device, 'device_kind', None),
          'client_platform': getattr(
              getattr(device, 'client', None), 'platform', None
          ),
      } for device in gpu_devices),
      'nvidia_smi': _nvidia_smi(),
      'no_jit_no_array_no_model': True,
  }
  validate_device_observation(observation)
  return observation


def _bundle_paths() -> tuple[Path, ...]:
  return (
      Path(__file__).resolve(),
      _HERE / 'run_exact_source_gate0_v3_1_2.py',
      _HERE / 'run_exact_source_gate0_v3_1_2.sh',
      _HERE / 'run_exact_source_gate0_v3_1_2_test.py',
      FREEZE_PATH,
      PROTOCOL_PATH,
      _HERE / 'results' / 'v3_1_1_exact_source_gate0_identity_one_shot'
      / 'ATTEMPT_STARTED.json',
      _HERE / 'results' / 'v3_1_1_exact_source_gate0_identity_one_shot'
      / 'PARTIAL_FAILURE.md',
  )


def validate_committed_bundle() -> dict[str, Any]:
  repo = _HERE.parents[2]
  paths = _bundle_paths()
  relative = tuple(str(path.relative_to(repo)) for path in paths)
  for path in paths:
    subprocess.run(
        ('git', '-C', str(repo), 'ls-files', '--error-unmatch',
         str(path.relative_to(repo))),
        check=True,
        capture_output=True,
    )
  diff = subprocess.check_output(
      ('git', '-C', str(repo), 'diff', '--binary', 'HEAD', '--', *relative)
  )
  if diff:
    raise ValueError('v3.1.2 preflight bundle differs from committed HEAD.')
  return {
      'git_head': subprocess.check_output(
          ('git', '-C', str(repo), 'rev-parse', 'HEAD'), text=True
      ).strip(),
      'paths': relative,
      'file_sha256': {
          str(path.relative_to(repo)): _sha256(path) for path in paths
      },
      'tracked_and_clean': True,
  }


def validate_freeze() -> dict[str, Any]:
  frozen = json.loads(FREEZE_PATH.read_text(encoding='utf-8'))
  expected = {
      'preflight_script_version': SCRIPT_VERSION,
      'runner_script_version': 'opensplice-exact-source-gate0-v3.1.2',
      'protocol_sha256': PROTOCOL_SHA256,
      'preflight_sha256': _sha256(Path(__file__).resolve()),
      'runner_sha256': _sha256(_HERE / 'run_exact_source_gate0_v3_1_2.py'),
      'wrapper_sha256': _sha256(_HERE / 'run_exact_source_gate0_v3_1_2.sh'),
      'test_sha256': _sha256(
          _HERE / 'run_exact_source_gate0_v3_1_2_test.py'
      ),
      'expected_device_kind': EXPECTED_DEVICE_KIND,
      'expected_gpu_uuid': EXPECTED_GPU_UUID,
      'expected_compute_capability': EXPECTED_COMPUTE_CAPABILITY,
      'preflight_dir': str(PREFLIGHT_DIR.resolve()),
  }
  for name, value in expected.items():
    if frozen.get(name) != value:
      raise ValueError(f'v3.1.2 freeze mismatch: {name}.')
  return {**frozen, 'path': str(FREEZE_PATH), 'sha256': _sha256(FREEZE_PATH)}


def _reserve_attempt_number() -> int:
  PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
  lock_path = PREFLIGHT_DIR / '.allocation.lock'
  with lock_path.open('a+', encoding='utf-8') as handle:
    fcntl.flock(handle, fcntl.LOCK_EX)
    numbers = []
    for pattern in ('preflight_*.json', '.preflight_*.reserved'):
      for path in PREFLIGHT_DIR.glob(pattern):
        text = path.name.removeprefix('.').removeprefix('preflight_')
        text = text.removesuffix('.reserved').removesuffix('.json')
        try:
          numbers.append(int(text))
        except ValueError:
          continue
    number = max(numbers, default=-1) + 1
    reservation = PREFLIGHT_DIR / f'.preflight_{number:04d}.reserved'
    descriptor = os.open(
        reservation, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
    )
    os.close(descriptor)
    return number


@contextlib.contextmanager
def _capture_file_descriptors(stdout_path: Path, stderr_path: Path):
  saved_stdout = os.dup(1)
  saved_stderr = os.dup(2)
  with stdout_path.open('wb') as stdout, stderr_path.open('wb') as stderr:
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(stdout.fileno(), 1)
    os.dup2(stderr.fileno(), 2)
    try:
      yield
    finally:
      sys.stdout.flush()
      sys.stderr.flush()
      os.dup2(saved_stdout, 1)
      os.dup2(saved_stderr, 2)
      os.close(saved_stdout)
      os.close(saved_stderr)


def run_external_preflight() -> tuple[Path, bool]:
  # Commit/freeze validity are prerequisites to beginning a preflight.  Device
  # and environment failures after allocation are captured durably below.
  bundle = validate_committed_bundle()
  frozen = validate_freeze()
  number = _reserve_attempt_number()
  prefix = f'preflight_{number:04d}'
  stdout_temp = PREFLIGHT_DIR / f'.{prefix}.{os.getpid()}.stdout.tmp'
  stderr_temp = PREFLIGHT_DIR / f'.{prefix}.{os.getpid()}.stderr.tmp'
  observation = None
  error_record = None
  warning_records = []
  with _capture_file_descriptors(stdout_temp, stderr_temp):
    try:
      with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        observation = collect_device_observation()
      warning_records = tuple({
          'category': item.category.__name__,
          'message': str(item.message),
          'filename': item.filename,
          'lineno': item.lineno,
      } for item in caught)
    except Exception as error:  # all infrastructure failures must be durable
      error_record = {
          'exception_type': type(error).__name__,
          'message': str(error),
          'traceback': ''.join(traceback.format_exception(error)),
      }
      traceback.print_exception(error, file=sys.stderr)
  stdout_path = PREFLIGHT_DIR / f'{prefix}.stdout.log'
  stderr_path = PREFLIGHT_DIR / f'{prefix}.stderr.log'
  stdout_sha = _write_new(stdout_path, stdout_temp.read_bytes())
  stderr_sha = _write_new(stderr_path, stderr_temp.read_bytes())
  stdout_temp.unlink(missing_ok=True)
  stderr_temp.unlink(missing_ok=True)
  passed = error_record is None
  record = {
      'script_version': SCRIPT_VERSION,
      'preflight_attempt_number': number,
      'status': 'pass' if passed else 'failure',
      'created_at_unix_s': time.time(),
      'bundle': bundle,
      'freeze': frozen,
      'environment_contract': {
          'LD_LIBRARY_PATH': 'absent',
          'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
      },
      'observation': observation,
      'warnings': warning_records,
      'failure': error_record,
      'logs': {
          'stdout': {'path': str(stdout_path), 'sha256': stdout_sha},
          'stderr': {'path': str(stderr_path), 'sha256': stderr_sha},
      },
      'no_model_or_biological_access': True,
      'no_jit_or_array_kernel': True,
  }
  record_path = PREFLIGHT_DIR / f'{prefix}.json'
  _write_new(record_path, _json_bytes(record))
  return record_path, passed


def build_dry_run_plan() -> dict[str, Any]:
  return {
      'script_version': SCRIPT_VERSION,
      'dry_run': True,
      'protocol_sha256': PROTOCOL_SHA256,
      'preflight_dir': str(PREFLIGHT_DIR),
      'scientific_output_created': False,
      'imports_if_run': ('jax', 'jaxlib'),
      'forbidden_imports': (
          'alphagenome', 'alphagenome_research', 'run_inference_trace',
      ),
      'jit_calls': 0,
      'array_kernel_calls': 0,
      'model_calls': 0,
      'required_device': {
          'default_backend': 'gpu',
          'visible_gpu_count': 1,
          'device_kind': EXPECTED_DEVICE_KIND,
          'uuid': EXPECTED_GPU_UUID,
          'compute_capability': EXPECTED_COMPUTE_CAPABILITY,
      },
  }


def main() -> None:
  args = _parse_args()
  if args.dry_run:
    print(json.dumps(build_dry_run_plan(), indent=2))
    return
  record_path, passed = run_external_preflight()
  print(record_path.resolve())
  if not passed:
    raise SystemExit(2)


if __name__ == '__main__':
  main()
