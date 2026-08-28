#!/usr/bin/env python3
"""Sole append-only, JAX-only v3.3.4.4 external device preflight."""

from __future__ import annotations

import argparse
import errno
import hashlib
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
# This module is standard-library-only.  JAX is imported lazily after the
# formal publication probe succeeds.
import validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_4 as bootstrap  # pylint: disable=g-import-not-at-top


PREFLIGHT_SCRIPT_VERSION = 'opensplice-device-preflight-v3.3.4.4'
SCRIPT_VERSION = PREFLIGHT_SCRIPT_VERSION
FREEZE_PATH = bootstrap.FREEZE_PATH
PREFLIGHT_DIR = bootstrap.PREFLIGHT_DIR
_SUCCESS_PAYLOAD = b'opensplice-v3.3.4.4-renameat2-noreplace-probe-v1\n'
_COLLISION_PAYLOAD = b'opensplice-v3.3.4.4-collision-probe-v1\n'


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('--run', action='store_true')
  group.add_argument('--dry-run', action='store_true')
  return parser.parse_args()


def _json_bytes(value: Any) -> bytes:
  return (json.dumps(
      value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
  ) + '\n').encode('utf-8')


def _simple_binding(success: dict[str, Any], *, absolute: bool) -> dict[str, Any]:
  path = success['final_relative_path']
  if absolute:
    path = str((PREFLIGHT_DIR / path).resolve())
  return {
      'path': path, 'sha256': success['sha256'],
      'size_bytes': success['size_bytes'],
  }


def _publication_binding(path: str, source: dict[str, Any]) -> dict[str, Any]:
  return {
      'path': path, 'sha256': source['sha256'],
      'size_bytes': source['size_bytes'], 'mode': source['mode'],
      'st_dev': source['st_dev'], 'st_ino': source['st_ino'],
      'st_nlink': source['st_nlink'],
  }


def _register_external_cache() -> dict[str, Any]:
  for relative in ('.', 'triton', 'xdg'):
    bootstrap.allocate_publication_directory(
        'external_cache', relative, register_existing=True
    )
  attestation = bootstrap.assert_cache_environment_sanitized()
  if (
      attestation['cache_role'] != 'external_preflight'
      or attestation['pre_import_file_count'] != 0
      or attestation['pre_import_tree_sha256'] != hashlib.sha256(b'').hexdigest()
  ):
    raise RuntimeError('External cache was not empty before the probe.')
  return attestation


def _formal_publication_probe() -> dict[str, Any]:
  contract = bootstrap.PUBLICATION_CONTRACT_V3_3_4_1[
      'external_preflight_probe_contract'
  ]
  if (
      hashlib.sha256(_SUCCESS_PAYLOAD).hexdigest() != contract['final_sha256']
      or len(_SUCCESS_PAYLOAD) != contract['final_size_bytes']
      or hashlib.sha256(_COLLISION_PAYLOAD).hexdigest()
      != contract['collision_sha256']
      or len(_COLLISION_PAYLOAD) != contract['collision_size_bytes']
  ):
    raise RuntimeError('Frozen publication-probe payload contract changed.')
  final_name = contract['final_basename']
  success = bootstrap.publish_bytes(
      'external_cache', final_name, _SUCCESS_PAYLOAD,
      artifact_role='atomic_publication_probe_success',
  )
  try:
    bootstrap.publish_bytes(
        'external_cache', final_name, _COLLISION_PAYLOAD,
        artifact_role='atomic_publication_probe_collision',
        allow_existing_final_for_probe=True,
    )
  except bootstrap.PublicationError as error:
    failure = error.publication_failure
  else:
    raise RuntimeError('Collision publication unexpectedly replaced a final.')
  if (
      failure['failure_stage'] != 'rename_noreplace'
      or failure['errno'] != errno.EEXIST
      or failure['rename_noreplace_attempted'] is not True
      or failure['rename_noreplace_succeeded'] is not False
  ):
    raise RuntimeError('Collision probe did not fail exactly with EEXIST.')
  parent_fd = os.open(
      bootstrap.PREFLIGHT_KERNEL_CACHE_DIR,
      os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
  )
  try:
    os.fsync(parent_fd)
  finally:
    os.close(parent_fd)
  final_state = bootstrap._entry_state_path(  # pylint: disable=protected-access
      bootstrap.PREFLIGHT_KERNEL_CACHE_DIR / final_name
  )
  temp_relative = failure['temp_relative_path']
  temp_state = bootstrap._entry_state_path(  # pylint: disable=protected-access
      bootstrap.PREFLIGHT_KERNEL_CACHE_DIR / temp_relative
  )
  final_binding = _publication_binding(final_name, success)
  collision_binding = _publication_binding(temp_relative, temp_state)
  if (
      _publication_binding(final_name, final_state) != final_binding
      or _publication_binding(temp_relative, temp_state) != collision_binding
      or final_state['state'] != 'present'
      or final_state['entry_type'] != 'regular'
      or temp_state['state'] != 'present'
      or temp_state['entry_type'] != 'regular'
  ):
    raise RuntimeError('Publication probe entries changed after collision fsync.')
  probe = {
      'schema_version': bootstrap.PUBLICATION_SCHEMA_VERSION,
      'method': bootstrap.PUBLICATION_METHOD,
      'supported': True,
      'successful_final_binding': final_binding,
      'collision_errno': errno.EEXIST,
      'collision_no_replace_exact': True,
      'collision_temp_binding': collision_binding,
      'destination_unchanged': True,
      'temp_orphan_preserved': True,
      'parent_fsync_exact': True,
  }
  bootstrap.validate_publication_probe_live_bindings(probe)
  return probe


def _allocate_preflight_root() -> None:
  bootstrap.allocate_publication_directory('external_preflight')
  bootstrap.create_empty_lifecycle_file(
      'external_preflight', '.allocation.lock', mode=0o600
  )
  bootstrap.create_empty_lifecycle_file(
      'external_preflight', '.preflight_0000.reserved', mode=0o400
  )


def _validate_authorized_freeze() -> dict[str, Any]:
  authorization = bootstrap.validate_external_freeze_authorization()
  frozen = json.loads(FREEZE_PATH.read_text(encoding='utf-8'))
  version_proof = bootstrap.validate_preflight_version_contract(frozen)
  prior_prefix = (
      bootstrap.validate_prior_v3_3_4_3_consumed_preflight_prefix()
  )
  expected = {
      'script_version': 'v3.3.4.4',
      'attempt_id': 'v3.3.4.4-development-ood-sidecar-one-shot',
      'amendment_sha256': bootstrap.AMENDMENT_SHA256,
      'preflight_script_version': SCRIPT_VERSION,
      'preflight_dir': str(PREFLIGHT_DIR.resolve()),
      'publication_contract_v3_3_4_1': (
          bootstrap.PUBLICATION_CONTRACT_V3_3_4_1
      ),
      'external_freeze_authorization_contract': (
          bootstrap.EXTERNAL_FREEZE_AUTHORIZATION_CONTRACT
      ),
  }
  for name, value in expected.items():
    if frozen.get(name) != value:
      raise ValueError(f'v3.3.4.4 preflight freeze mismatch: {name}.')
  return {
      'path': str(FREEZE_PATH.resolve()),
      'sha256': authorization['freeze_sha256'],
      'size_bytes': authorization['freeze_size_bytes'],
      'external_freeze_authorization': authorization,
      'preflight_version_proof': version_proof,
      'prior_v3_3_4_3_consumed_preflight_prefix': prior_prefix,
      'prior_v3_3_4_3_consumed_preflight_prefix_content_binding': (
          bootstrap.canonical_content_binding(prior_prefix)
      ),
  }


def _cache_hit_evidence(pre_import: dict[str, Any]) -> dict[str, Any]:
  files_present = pre_import['pre_import_file_count'] != 0
  evidence = {
      'pre_import_files_present': files_present,
      'default_user_cache_path_eligible': False,
      'persistent_compilation_cache_hit_reported': False,
      'executable_deserialized': False,
      'compile_skipped': None,
      'compile_stage_not_applicable': True,
      'old_cache_input_opened': False,
      'routing_exact': True,
      'cache_hit': files_present,
  }
  if evidence['cache_hit']:
    raise RuntimeError('External preflight cache input was detected.')
  return evidence


def _collect_observation(
    pre_import: dict[str, Any]
) -> dict[str, Any]:
  # The import is deliberately below the publication probe.
  import run_device_preflight_v3_3 as v33_preflight  # pylint: disable=g-import-not-at-top
  environment = v33_preflight.assert_environment()
  live_cache = bootstrap.assert_live_cache_environment_matches(pre_import)
  observation = v33_preflight.collect_observation()
  return {
      'atomic_publication_supported': True,
      'environment': observation['v3_3_runtime_environment'],
      'hostname': observation['hostname'],
      'jax_default_backend': observation['jax_default_backend'],
      'jax_enable_compilation_cache': observation['jax_enable_compilation_cache'],
      'jax_gpu_devices': observation['jax_gpu_devices'],
      'jax_module_version': observation['jax_module_version'],
      'jaxlib_module_version': observation['jaxlib_module_version'],
      'kernel': observation['kernel'],
      'no_jit_no_array_no_model': True,
      'nvidia_smi': observation['nvidia_smi'],
      'packages': observation['packages'],
      'pid': os.getpid(),
      'platform': observation['platform'],
      'python_executable': observation['python_executable'],
      'python_version': observation['python_version'],
      'runtime_environment': environment,
      'v3_3_4_4_runtime_environment': {
          **environment, 'cache_environment': pre_import,
          'live_cache_environment': live_cache,
      },
  }


def _log_binding(success: dict[str, Any]) -> dict[str, Any]:
  return _simple_binding(success, absolute=True)


def run_preflight() -> tuple[Path, bool]:
  freeze = _validate_authorized_freeze()
  pre_import = _register_external_cache()
  probe = _formal_publication_probe()
  _allocate_preflight_root()
  stdout_handle = bootstrap.open_publication_stream(
      'external_preflight', 'preflight_0000.stdout.log',
      artifact_role='preflight_stdout',
  )
  stderr_handle = bootstrap.open_publication_stream(
      'external_preflight', 'preflight_0000.stderr.log',
      artifact_role='preflight_stderr',
  )
  observation = None
  failure = None
  warning_records: list[str] = []
  saved_stdout = os.dup(1)
  saved_stderr = os.dup(2)
  try:
    os.dup2(stdout_handle.temp_fd, 1)
    os.dup2(stderr_handle.temp_fd, 2)
    try:
      with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        observation = _collect_observation(pre_import)
      warning_records = [str(item.message) for item in caught]
    except Exception as error:  # pylint: disable=broad-exception-caught
      failure = {
          'type': type(error).__name__, 'message': str(error),
          'traceback': ''.join(traceback.format_exception(error)),
      }
      traceback.print_exception(error, file=sys.stderr)
  finally:
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(saved_stdout, 1)
    os.dup2(saved_stderr, 2)
    os.close(saved_stdout)
    os.close(saved_stderr)
  stdout_success = stdout_handle.finalize()
  stderr_success = stderr_handle.finalize()
  post_cache = bootstrap.cache_output_tree_binding(
      bootstrap.PREFLIGHT_KERNEL_CACHE_DIR
  )
  hit = _cache_hit_evidence(pre_import)
  passed = failure is None
  record = {
      'amendment_sha256': bootstrap.AMENDMENT_SHA256,
      'atomic_publication_probe': probe,
      'created_at_unix_s': time.time(),
      'external_freeze_authorization': freeze['external_freeze_authorization'],
      'external_cache_post_observation': post_cache,
      'external_cache_hit_evidence': hit,
      'failure': failure,
      'freeze': freeze,
      'freeze_sha256': freeze['sha256'],
      'logs': {
          'stdout': _log_binding(stdout_success),
          'stderr': _log_binding(stderr_success),
      },
      'no_jit_or_array_kernel': True,
      'no_model_or_biological_access': True,
      'observation': observation,
      'original_protocol_sha256': bootstrap.ORIGINAL_PROTOCOL_SHA256,
      'preflight_attempt_number': 0,
      'script_version': SCRIPT_VERSION,
      'status': 'pass' if passed else 'fail',
      'warnings': warning_records,
      'prior_v3_3_4_3_consumed_preflight_prefix': freeze[
          'prior_v3_3_4_3_consumed_preflight_prefix'
      ],
      'prior_v3_3_4_3_consumed_preflight_prefix_content_binding': freeze[
          'prior_v3_3_4_3_consumed_preflight_prefix_content_binding'
      ],
  }
  result = bootstrap.publish_bytes(
      'external_preflight', 'preflight_0000.json', _json_bytes(record),
      artifact_role='preflight_record',
  )
  return PREFLIGHT_DIR / result['final_relative_path'], passed


def build_dry_run_plan() -> dict[str, Any]:
  return {
      'script_version': SCRIPT_VERSION,
      'amendment_sha256': bootstrap.AMENDMENT_SHA256,
      'dry_run': True,
      'preflight_dir': str(PREFLIGHT_DIR),
      'scientific_output_created': False,
      'model_calls': 0,
      'jit_calls': 0,
      'sole_attempt_number': 0,
      'atomic_publication_probe_required': True,
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
