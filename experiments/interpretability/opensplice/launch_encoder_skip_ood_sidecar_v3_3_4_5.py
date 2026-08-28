#!/usr/bin/env python3
"""Own Gate A, preflight, START, Gate B, then load v3.3.4.5 in-process."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
import traceback
import types
from typing import Any


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))
import validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_5 as bootstrap  # pylint: disable=g-import-not-at-top


ATTESTATION_MODULE = '_opensplice_v3_3_4_5_ood_sidecar_bootstrap_attestation'
ATTEMPT_ID = 'v3.3.4.5-development-ood-sidecar-one-shot'
SCRIPT_VERSION = 'v3.3.4.5'
DISCLOSURE = (
    'Later-exon metadata/labels were exposed after protocol freeze; no '
    'later-exon model outputs, activations, or interventions are used.'
)
def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--dry-run', action='store_true')
  args = parser.parse_args()
  if len(sys.argv) != (2 if args.dry_run else 1):
    parser.error('--dry-run is the only accepted option.')
  return args


def _canonical_bytes(value: Any) -> bytes:
  return json.dumps(
      value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
      allow_nan=False,
  ).encode('utf-8')


def _content_binding(value: Any) -> dict[str, Any]:
  payload = _canonical_bytes(value)
  return {'sha256': hashlib.sha256(payload).hexdigest(),
          'size_bytes': len(payload)}


def _json_bytes(value: Any) -> bytes:
  return (json.dumps(
      value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
  ) + '\n').encode('utf-8')


def _publish_new(path: Path, value: Any) -> dict[str, Any]:
  payload = _json_bytes(value)
  relative = path.relative_to(bootstrap.OUTPUT_DIR).as_posix()
  bootstrap.ensure_publication_directory(
      'model_run', Path(relative).parent.as_posix()
  )
  return bootstrap.publish_bytes(
      'model_run', relative, payload, artifact_role=path.name
  )


def _publish_sibling_new(path: Path, value: Any) -> dict[str, Any]:
  """Publishes beside START without recreating its existing parent."""
  return _publish_new(path, value)


def _sanitize(role: str, root: Path) -> dict[str, Any]:
  for name in ('LD_LIBRARY_PATH', *bootstrap.DENIED_CACHE_ENVIRONMENT_NAMES):
    os.environ.pop(name, None)
  for name in tuple(os.environ):
    upper = name.upper()
    if (
        any(name.startswith(prefix) for prefix in
            bootstrap.DENIED_CACHE_ENVIRONMENT_PREFIXES)
        or ('AUTOTUNE' in upper and any(
            token in upper for token in ('LOAD', 'DUMP', 'CACHE')
        ))
    ):
      os.environ.pop(name, None)
  os.environ.update({
      'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
      'JAX_ENABLE_COMPILATION_CACHE': 'false',
      'CUDA_CACHE_DISABLE': '1',
      'PYTHONDONTWRITEBYTECODE': '1',
      bootstrap.CACHE_ROLE_ENVIRONMENT: role,
      bootstrap.CACHE_ROOT_ENVIRONMENT: str(root),
      'TRITON_CACHE_DIR': str(root / 'triton'),
      'XDG_CACHE_HOME': str(root / 'xdg'),
  })
  return bootstrap.assert_cache_environment_sanitized()


def _allocate_cache(root: Path, role: str) -> dict[str, Any]:
  root_role = {
      'external_preflight': 'external_cache', 'model': 'model_cache'
  }[role]
  if root != bootstrap.PUBLICATION_ROOTS[root_role]:
    raise ValueError('Cache root/role mismatch.')
  bootstrap.ensure_publication_directory(root_role)
  bootstrap.ensure_publication_directory(root_role, 'triton')
  bootstrap.ensure_publication_directory(root_role, 'xdg')
  return _sanitize(role, root)


def _temporary_gate_cache() -> Path:
  root = Path(tempfile.mkdtemp(prefix='alphagenome-v3.3.4.5-dry-cache.'))
  os.mkdir(root / 'triton', 0o700)
  os.mkdir(root / 'xdg', 0o700)
  return root


def _remove_empty_gate_cache(root: Path) -> None:
  os.rmdir(root / 'triton')
  os.rmdir(root / 'xdg')
  os.rmdir(root)


def _validate_preflight_record(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
  if path.resolve() != (bootstrap.PREFLIGHT_DIR / 'preflight_0000.json').resolve():
    raise ValueError('External preflight path is not the sole frozen record.')
  state = bootstrap.validate_completed_external_preflight_state()
  record = json.loads(path.read_text(encoding='utf-8'))
  if record['status'] != 'pass' or record['failure'] is not None:
    raise ValueError('External preflight did not pass.')
  live_authorization = bootstrap.validate_external_freeze_authorization()
  if record.get('external_freeze_authorization') != live_authorization:
    raise ValueError('External preflight authorization differs from Gate A.')
  prior_prefix = (
      bootstrap.validate_recorded_prior_v3_3_4_3_consumed_preflight_prefix(
          record.get('prior_v3_3_4_3_consumed_preflight_prefix'),
          record.get(
              'prior_v3_3_4_3_consumed_preflight_prefix_content_binding'
          ),
      )
  )
  prior_binding = bootstrap.canonical_content_binding(prior_prefix)
  prior_v3_3_4_4_prefix = (
      bootstrap.validate_recorded_prior_v3_3_4_4_consumed_preflight_prefix(
          record.get('prior_v3_3_4_4_consumed_preflight_prefix'),
          record.get(
              'prior_v3_3_4_4_consumed_preflight_prefix_content_binding'
          ),
      )
  )
  binding = {
      'path': str(path.resolve()),
      'sha256': bootstrap._sha256(path),  # pylint: disable=protected-access
      'size_bytes': path.stat().st_size,
  }
  if {
      'sha256': binding['sha256'], 'size_bytes': binding['size_bytes'],
  } != state['file_sha256']['preflight_0000.json']:
    raise ValueError('External preflight record changed after phase validation.')
  successful = {
      'artifact_binding': binding,
      'root_file_count': state['file_count'],
      'root_file_tree_sha256': state['tree_sha256'],
      'external_pid': record['observation']['pid'],
      'status': 'pass',
      'external_freeze_authorization': record[
          'external_freeze_authorization'
      ],
      'external_cache_post_observation': record[
          'external_cache_post_observation'
      ],
      'external_cache_hit_evidence': record[
          'external_cache_hit_evidence'
      ],
      'prior_v3_3_4_3_consumed_preflight_prefix': prior_prefix,
      'prior_v3_3_4_3_consumed_preflight_prefix_content_binding': (
          prior_binding
      ),
      'prior_v3_3_4_4_consumed_preflight_prefix': prior_v3_3_4_4_prefix,
      'prior_v3_3_4_4_consumed_preflight_prefix_content_binding': dict(
          bootstrap.V3_3_4_4_CONSUMED_PREFIX_BINDING
      ),
  }
  if len(successful) != 12 or set(successful) != {
      'artifact_binding', 'root_file_count', 'root_file_tree_sha256',
      'external_pid', 'status', 'external_freeze_authorization',
      'external_cache_post_observation', 'external_cache_hit_evidence',
      'prior_v3_3_4_3_consumed_preflight_prefix',
      'prior_v3_3_4_3_consumed_preflight_prefix_content_binding',
      'prior_v3_3_4_4_consumed_preflight_prefix',
      'prior_v3_3_4_4_consumed_preflight_prefix_content_binding',
  }:
    raise RuntimeError('Successful-preflight record is not exact12.')
  if bootstrap.validate_completed_external_preflight_state() != state:
    raise ValueError('Completed preflight changed during parent validation.')
  return record, successful


def _same_process_observation(
    *, gate_a: dict[str, Any], successful_preflight: dict[str, Any],
    model_cache_environment: dict[str, Any],
    model_cache_binding: dict[str, Any], frozen: dict[str, Any]
) -> dict[str, Any]:
  # Importing this module initializes only the JAX backend; no array/JIT/model.
  import run_device_preflight_v3_3 as v33_preflight  # pylint: disable=g-import-not-at-top
  observation = v33_preflight.collect_observation()
  live_model_cache = bootstrap.assert_cache_environment_sanitized()
  if live_model_cache != model_cache_environment:
    # The input attestation is the exact pre-JAX object; any new file before
    # START consumes the pre-START state and no model run is created.
    raise RuntimeError('Model cache changed during same-process pre-START gate.')
  external_pid = successful_preflight['external_pid']
  if external_pid == os.getpid():
    raise RuntimeError('External and same-process preflight PIDs must differ.')
  return {
      'pid': os.getpid(),
      'parent_pid': os.getppid(),
      'external_preflight_pid': external_pid,
      'default_backend': observation['jax_default_backend'],
      'jax_gpu_devices': observation['jax_gpu_devices'],
      'nvidia_smi': observation['nvidia_smi'],
      'runtime_environment': observation['runtime_environment'],
      'runtime_versions': {
          name: importlib.metadata.version(name)
          for name in frozen['runtime_version_manifest'][
              'packages'
          ]
      },
      'freeze_sha256': gate_a['sha256'],
      'external_freeze_authorization': gate_a[
          'external_freeze_authorization'
      ],
      'external_preflight_binding': successful_preflight[
          'artifact_binding'
      ],
      'external_preflight_tree_sha256': successful_preflight[
          'root_file_tree_sha256'
      ],
      'model_cache_pre_import': model_cache_binding,
      'current_source_inventory_exact': True,
      'prior_artifacts_exact': True,
      'no_model_constructed': True,
      'no_jit_or_array_kernel': True,
      'created_at_unix_s': time.time(),
  }


def _start_record(
    *, gate_a: dict[str, Any], frozen: dict[str, Any],
    successful_preflight: dict[str, Any], same_process: dict[str, Any]
) -> dict[str, Any]:
  source_audit = dict(zip(
      bootstrap.SOURCE_INPUT_AUDIT_KEYS,
      bootstrap.SOURCE_INPUT_AUDIT_CONTRACT['start_values'],
      strict=True,
  ))
  if source_audit != {
      name: value for name, value in zip(
          bootstrap.SOURCE_INPUT_AUDIT_KEYS,
          (True, True, True, True, None, None, None, None), strict=True
      )
  }:
    raise RuntimeError('START source-input audit contract changed.')
  execution_contract = frozen['terminal_contract']['execution_contract']
  frozen_inventory = frozen['source_inventory_contract']
  inventory_digest = hashlib.sha256()
  for row in frozen_inventory['rows']:
    inventory_digest.update(row['path'].encode('utf-8'))
    inventory_digest.update(b'\0')
    inventory_digest.update(bytes.fromhex(row['sha256']))
  source_inventory_attestation = {
      'row_count': frozen_inventory['source_row_count'],
      'rows': frozen_inventory['rows'],
      'tree_sha256': inventory_digest.hexdigest(),
      'git_head': gate_a['git_head'],
      'tracked_clean': True,
      'live_equals_head': True,
  }
  record = {
      'status': 'attempt_started',
      'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION,
      'amendment_sha256': bootstrap.AMENDMENT_SHA256,
      'amendment_commit': bootstrap.AMENDMENT_COMMIT,
      'original_protocol_sha256': bootstrap.ORIGINAL_PROTOCOL_SHA256,
      'freeze_path': str(bootstrap.FREEZE_PATH.resolve()),
      'freeze_sha256': gate_a['sha256'],
      'git_head': gate_a['git_head'],
      'external_freeze_authorization': gate_a[
          'external_freeze_authorization'
      ],
      'runner_pid': os.getpid(),
      'parent_pid': os.getppid(),
      'started_at_unix_s': time.time(),
      'successful_preflight': successful_preflight,
      'same_process_preflight': same_process,
      'same_process_preflight_content_binding': _content_binding(same_process),
      'fresh_paths': {
          'device_preflight': str(bootstrap.PREFLIGHT_DIR.resolve()),
          'preflight_kernel_cache': str(
              bootstrap.PREFLIGHT_KERNEL_CACHE_DIR.resolve()
          ),
          'model_kernel_cache': str(
              bootstrap.MODEL_KERNEL_CACHE_DIR.resolve()
          ),
          'model_run': str(bootstrap.OUTPUT_DIR.resolve()),
          'analysis_attempt': str(bootstrap.ANALYSIS_ATTEMPT_DIR.resolve()),
          'analysis_output': str(bootstrap.ANALYSIS_DIR.resolve()),
      },
      'budgets': {
          'max_wall_time_seconds': 7200,
          'max_output_bytes': 1073741824,
          'expected_records': 80,
          'expected_model_applies': 320,
          'lower_attempt_budget': 1,
          'compile_attempt_budget': 1,
          'run_complete_size_cap_bytes': 16777216,
      },
      'execution_contract': execution_contract,
      'source_inventory_attestation': source_inventory_attestation,
      'prior_v3_3_3_binding': gate_a['v3_3_3_run'],
      'prior_v3_3_3_1_archive_binding': gate_a['v3_3_3_1_archive'],
      'source_input_audit': source_audit,
      'source_input_audit_content_binding': _content_binding(source_audit),
      'program_signature_contract': frozen[
          'program_signature_attestation_contract'
      ],
      'cache_isolation_contract': frozen['cache_isolation_contract'],
      'confirmation_scope_disclosure': DISCLOSURE,
      'confirmation_model_calls': 0,
      'scientific_summary_computed': False,
      'donor_normalization_computed': False,
      'shapley_or_nomination_computed': False,
      'interaction_or_resolution_computed': False,
      'nomination_performed': False,
      'combined_analysis_permitted': False,
      'prior_v3_3_4_3_consumed_preflight_prefix': gate_a[
          'prior_v3_3_4_3_consumed_preflight_prefix'
      ],
      'prior_v3_3_4_3_consumed_preflight_prefix_content_binding': gate_a[
          'prior_v3_3_4_3_consumed_preflight_prefix_content_binding'
      ],
      'prior_v3_3_4_4_consumed_preflight_prefix': gate_a[
          'prior_v3_3_4_4_consumed_preflight_prefix'
      ],
      'prior_v3_3_4_4_consumed_preflight_prefix_content_binding': gate_a[
          'prior_v3_3_4_4_consumed_preflight_prefix_content_binding'
      ],
  }
  if set(record) != set(bootstrap.START_RECORD_KEYS) or len(record) != 38:
    raise RuntimeError('START record is not the exact 38-key schema.')
  return record


def _post_start_failure(error: Exception, start: dict[str, Any]) -> None:
  audit = dict(start['source_input_audit'])
  audit['bootstrap_sources_and_prior_trees_exact'] = False
  record = {
      'status': 'controlled_stop_post_start_provenance_failure',
      'stop_reason': 'post_start_provenance_failure',
      'message': 'Gate B failed before any scientific import or model use.',
      'failure': {
          'type': type(error).__name__,
          'message': str(error),
          'traceback': ''.join(traceback.format_exception(error)),
      },
      'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION,
      'amendment_sha256': bootstrap.AMENDMENT_SHA256,
      'freeze_sha256': start['freeze_sha256'],
      'git_head': start['git_head'],
      'external_freeze_authorization': start[
          'external_freeze_authorization'
      ],
      'runner_pid': os.getpid(),
      'source_inventory_failure': str(error),
      'model_constructed': False,
      'model_apply_count': 0,
      'source_input_audit': audit,
      'source_input_audit_content_binding': _content_binding(audit),
      'confirmation_model_calls': 0,
      'scientific_summary_computed': False,
      'combined_analysis_permitted': False,
      'failed_at_unix_s': time.time(),
      'prior_v3_3_4_3_consumed_preflight_prefix': start[
          'prior_v3_3_4_3_consumed_preflight_prefix'
      ],
      'prior_v3_3_4_3_consumed_preflight_prefix_content_binding': start[
          'prior_v3_3_4_3_consumed_preflight_prefix_content_binding'
      ],
      'prior_v3_3_4_4_consumed_preflight_prefix': start[
          'prior_v3_3_4_4_consumed_preflight_prefix'
      ],
      'prior_v3_3_4_4_consumed_preflight_prefix_content_binding': start[
          'prior_v3_3_4_4_consumed_preflight_prefix_content_binding'
      ],
  }
  if (
      set(record) != set(bootstrap.POST_START_PROVENANCE_FAILURE_KEYS)
      or len(record) != 24
  ):
    raise RuntimeError('POST_START failure is not the exact 24-key schema.')
  _publish_sibling_new(
      bootstrap.OUTPUT_DIR / 'POST_START_PROVENANCE_FAILURE.json', record
  )


def _run_binding_map() -> dict[str, dict[str, Any]]:
  result = {}
  for path in bootstrap._strict_file_tree(  # pylint: disable=protected-access
      bootstrap.OUTPUT_DIR
  ):
    result[path.relative_to(bootstrap.OUTPUT_DIR).as_posix()] = {
        'sha256': bootstrap._sha256(path),  # pylint: disable=protected-access
        'size_bytes': path.stat().st_size,
    }
  return dict(sorted(result.items()))


def _binding_tree_sha256(bindings: dict[str, dict[str, Any]]) -> str:
  digest = hashlib.sha256()
  for relative, binding in sorted(bindings.items()):
    digest.update(relative.encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(binding['sha256']))
  return digest.hexdigest()


def _preterminal_tree_binding() -> dict[str, Any]:
  bindings = _run_binding_map()
  directories = ['.']
  return {
      'file_count': len(bindings),
      'directory_count': 1,
      'file_bindings': bindings,
      'file_tree_sha256': _binding_tree_sha256(bindings),
      'directory_paths': directories,
      'directory_tree_sha256': hashlib.sha256(b'D\0.\0').hexdigest(),
  }


def _model_cache_hit_evidence(binding: dict[str, Any]) -> dict[str, Any]:
  result = {
      'pre_import_files_present': binding['file_count'] != 0,
      'default_user_cache_path_eligible': False,
      'persistent_compilation_cache_hit_reported': False,
      'executable_deserialized': False,
      'compile_skipped': None,
      'compile_stage_not_applicable': True,
      'old_cache_input_opened': False,
      'routing_exact': binding['cache_root'] == str(
          bootstrap.MODEL_KERNEL_CACHE_DIR.resolve()
      ),
      'cache_hit': False,
  }
  result['cache_hit'] = bool(
      result['pre_import_files_present']
      or result['default_user_cache_path_eligible']
      or result['persistent_compilation_cache_hit_reported']
      or result['executable_deserialized']
      or result['old_cache_input_opened']
      or not result['routing_exact']
  )
  return result


def _publish_model_cache_pre_import_stop(
    *, start: dict[str, Any], live_cache: dict[str, Any],
    evidence: dict[str, Any], error: Exception,
) -> None:
  """Publishes the exact four-file pre-scientific-import terminal."""
  source_audit = dict(start['source_input_audit'])
  failure = {
      'type': type(error).__name__, 'message': str(error),
      'traceback': ''.join(traceback.format_exception(error)),
  }
  artifact = {
      'status': 'model_cache_pre_import_hit', 'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION,
      'external_freeze_authorization': start['external_freeze_authorization'],
      'model_cache_pre_import': live_cache,
      'cache_hit_evidence': evidence,
      'source_input_audit': source_audit,
      'source_input_audit_content_binding': _content_binding(source_audit),
      'failure': failure, 'model_constructed': False,
      'model_apply_count': 0, 'created_at_unix_s': time.time(),
  }
  _publish_new(
      bootstrap.OUTPUT_DIR / 'MODEL_CACHE_PRE_IMPORT_HIT.json', artifact
  )
  empty_sha = hashlib.sha256(b'').hexdigest()
  raw = {
      'schema_version': 'v3.3.4.5-raw-manifest-v1',
      'status': 'empty_controlled_stop', 'attempt_id': ATTEMPT_ID,
      'external_freeze_authorization': start['external_freeze_authorization'],
      'valid_artifact_count': 0, 'artifact_bindings': {},
      'artifact_tree_sha256': empty_sha,
      'valid_recipient_anchor_pairs': [], 'failed_current_binding': None,
      'dispatch_started_count': 0, 'dispatch_completed_count': 0,
      'dispatch_started_bindings': {},
      'dispatch_started_tree_sha256': empty_sha,
      'dispatch_completed_bindings': {},
      'dispatch_completed_tree_sha256': empty_sha,
      'source_input_audit_content_binding': _content_binding(source_audit),
      'same_object_attestation_content_binding': None,
      'created_at_unix_s': time.time(),
  }
  _publish_new(bootstrap.OUTPUT_DIR / 'RAW_MANIFEST.json', raw)
  preterminal = _preterminal_tree_binding()
  elapsed = time.time() - start['started_at_unix_s']
  phase = {
      name: name in {
          'preflight_passed', 'start_persisted',
          'post_start_source_gate_passed',
      }
      for name in bootstrap.TERMINAL_CONTRACT['phase_state_keys']
  }
  journal = {
      'started_count': 0, 'completed_count': 0,
      'started_bindings': {}, 'completed_bindings': {},
      'started_tree_sha256': empty_sha,
      'completed_tree_sha256': empty_sha,
      'started_prefix_exact': True, 'completed_prefix_exact': True,
  }
  cache_final = {
      'pre_import': start['same_process_preflight']['model_cache_pre_import'],
      'historical_stage': None, 'historical_binding': None,
      'terminal': live_cache, 'cache_hit_evidence': evidence,
      'historical_to_terminal_tree_exact': None,
      'historical_to_terminal_equality_is_a_gate': False,
      'historical_snapshot_not_reauthenticated_as_live_files': True,
      'default_user_cache_paths_eligible': False,
      'cache_outputs_are_diagnostic_only': True,
  }
  record = {
      'status': 'controlled_stop_cache_hit',
      'stop_reason': 'model_cache_pre_import_hit',
      'message': 'Model cache input appeared after START; no scientific import.',
      'failure': failure, 'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION,
      'amendment_sha256': bootstrap.AMENDMENT_SHA256,
      'amendment_commit': bootstrap.AMENDMENT_COMMIT,
      'original_protocol_sha256': bootstrap.ORIGINAL_PROTOCOL_SHA256,
      'freeze_sha256': start['freeze_sha256'], 'git_head': start['git_head'],
      'external_freeze_authorization': start['external_freeze_authorization'],
      'runner_pid': os.getpid(),
      'started_at_unix_s': start['started_at_unix_s'],
      'completed_at_unix_s': time.time(), 'phase_state': phase,
      'terminal_detail': {
          'k_valid_records': 0, 'd_completed': 0,
          'failed_execution_index': None, 'failed_call_role': None,
          'failure_phase': 'cache_pre_import', 'forbidden_operation': None,
          'provenance_artifact_role': None,
      },
      'budgets': {
          'max_wall_time_seconds': 7200,
          'elapsed_wall_time_seconds': elapsed,
          'wall_time_within_budget': elapsed <= 7200,
          'max_output_bytes': 1073741824,
          'preterminal_output_bytes': sum(
              row['size_bytes']
              for row in preterminal['file_bindings'].values()
          ),
          'run_complete_size_cap_bytes': 16777216,
          'preterminal_plus_terminal_cap_within_budget': True,
      },
      'source_input_audit': source_audit,
      'source_input_audit_content_binding': _content_binding(source_audit),
      'same_object_attestation': None,
      'same_object_attestation_content_binding': None,
      'program_signature_attestation_binding': None,
      'source_program_gate': None, 'compiler_binding': None,
      'compiler_artifact_bindings': {}, 'attempt_budget_audit': None,
      'diagnostic_provenance_complete': None,
      'compiled_backend_diagnostic_only': None,
      'backend_diagnostics': None, 'diagnostic_comparisons': None,
      'dispatch_journal': journal, 'raw_manifest': raw,
      'preterminal_tree_binding': preterminal, 'valid_record_count': 0,
      'failed_current_binding': None, 'model_apply_attempt_count': 0,
      'model_apply_success_count': 0, 'expected_model_apply_count': 320,
      'eight_row_lower_attempt_count': 0,
      'eight_row_compile_attempt_count': 0,
      'eight_row_successful_compile_count': 0,
      'six_row_compile_count': 0, 'identity_rerun_count': 0,
      'main_cube_rerun_count': 0, 'old_ood_records_reused': 0,
      'confirmation_model_calls': 0,
      'all_80_recipient_anchors_complete': False,
      'id0_all20': False, 'id255_all20': False,
      'import_provenance_phases': {
          'pre_model': None, 'post_model_precompile': None, 'terminal': None,
      },
      'protobuf_provenance_sha256': None,
      'model_kernel_cache_final': cache_final,
      'prior_v3_3_3_binding': start['prior_v3_3_3_binding'],
      'prior_v3_3_3_1_archive_binding': start[
          'prior_v3_3_3_1_archive_binding'
      ],
      'publication_audit': bootstrap.publication_audit('model_run'),
      'confirmation_scope_disclosure': DISCLOSURE,
      'scientific_summary_computed': False,
      'donor_normalization_computed': False,
      'shapley_or_nomination_computed': False,
      'interaction_or_resolution_computed': False,
      'nomination_performed': False, 'combined_analysis_permitted': False,
      'no_retry': True,
      'prior_v3_3_4_3_consumed_preflight_prefix': start[
          'prior_v3_3_4_3_consumed_preflight_prefix'
      ],
      'prior_v3_3_4_3_consumed_preflight_prefix_content_binding': start[
          'prior_v3_3_4_3_consumed_preflight_prefix_content_binding'
      ],
      'prior_v3_3_4_4_consumed_preflight_prefix': start[
          'prior_v3_3_4_4_consumed_preflight_prefix'
      ],
      'prior_v3_3_4_4_consumed_preflight_prefix_content_binding': start[
          'prior_v3_3_4_4_consumed_preflight_prefix_content_binding'
      ],
  }
  if set(record) != set(bootstrap.TERMINAL_CONTRACT['run_complete_keys']):
    raise RuntimeError('Launcher cache-stop RUN_COMPLETE schema changed.')
  _publish_new(bootstrap.OUTPUT_DIR / 'RUN_COMPLETE.json', record)


def _dry_run(gate_a: dict[str, Any]) -> None:
  print(json.dumps({
      'script_version': SCRIPT_VERSION,
      'amendment_sha256': bootstrap.AMENDMENT_SHA256,
      'dry_run': True,
      'gate_a_passed': True,
      'authorized_freeze_sha256': gate_a['sha256'],
      'recipient_count': 20,
      'anchor_ids': [0, 127, 128, 255],
      'record_count': 80,
      'model_apply_count': 320,
      'eight_row_compile_count': 1,
      'six_row_compile_count': 0,
      'production_paths_created': False,
      'confirmation_model_calls': 0,
  }, indent=2))


def main() -> None:
  args = _parse_args()
  path_absence = bootstrap.validate_gate_a_path_absence()
  gate_cache = _temporary_gate_cache()
  try:
    _sanitize('dry_run', gate_cache)
    gate_a = bootstrap.validate_freeze()
  finally:
    _remove_empty_gate_cache(gate_cache)
  gate_a['fresh_path_absence'] = path_absence
  if args.dry_run:
    _dry_run(gate_a)
    return

  _allocate_cache(
      bootstrap.PREFLIGHT_KERNEL_CACHE_DIR, 'external_preflight'
  )
  completed = subprocess.run(
      [sys.executable, str(_HERE / 'run_device_preflight_v3_3_4_5.py'), '--run'],
      check=False, capture_output=True, text=True,
  )
  if completed.returncode != 0:
    sys.stderr.write(completed.stderr)
    raise SystemExit(completed.returncode)
  path = Path(completed.stdout.strip().splitlines()[-1])
  preflight_record, successful_preflight = _validate_preflight_record(path)
  del preflight_record
  # The external preflight and cache are independently validated before the
  # distinct model-cache root is allocated.
  model_cache_environment = _allocate_cache(
      bootstrap.MODEL_KERNEL_CACHE_DIR, 'model'
  )
  model_cache = bootstrap.cache_output_tree_binding(
      bootstrap.MODEL_KERNEL_CACHE_DIR
  )
  frozen = json.loads(bootstrap.FREEZE_PATH.read_text(encoding='utf-8'))
  same_process = _same_process_observation(
      gate_a=gate_a,
      successful_preflight=successful_preflight,
      model_cache_environment=model_cache_environment,
      model_cache_binding=model_cache,
      frozen=frozen,
  )
  start = _start_record(
      gate_a=gate_a,
      frozen=frozen,
      successful_preflight=successful_preflight,
      same_process=same_process,
  )
  bootstrap.allocate_publication_directory('model_run')
  _publish_new(bootstrap.OUTPUT_DIR / 'ATTEMPT_STARTED.json', start)
  try:
    gate_b = bootstrap.validate_freeze(allow_started_output=True)
  except Exception as error:  # pylint: disable=broad-exception-caught
    _post_start_failure(error, start)
    raise SystemExit(2) from error

  # Repeat the isolated model-cache input gate after START/Gate B and before
  # importing the runner (which imports JAX/model/scientific modules).
  live_model_cache = bootstrap.cache_output_tree_binding(
      bootstrap.MODEL_KERNEL_CACHE_DIR
  )
  cache_evidence = _model_cache_hit_evidence(live_model_cache)
  if cache_evidence['cache_hit']:
    cache_error = RuntimeError(
        'Adverse model-cache input detected after START and Gate B.'
    )
    _publish_model_cache_pre_import_stop(
        start=start, live_cache=live_model_cache,
        evidence=cache_evidence, error=cache_error,
    )
    return

  attestation = types.ModuleType(ATTESTATION_MODULE)
  attestation.record = {
      'pid': os.getpid(),
      'created_at_unix_s': time.time(),
      'gate_a': gate_a,
      'gate_b': gate_b,
      'start': start,
      'start_binding': {
          'path': str((bootstrap.OUTPUT_DIR / 'ATTEMPT_STARTED.json').resolve()),
          'sha256': bootstrap._sha256(  # pylint: disable=protected-access
              bootstrap.OUTPUT_DIR / 'ATTEMPT_STARTED.json'
          ),
          'size_bytes': (
              bootstrap.OUTPUT_DIR / 'ATTEMPT_STARTED.json'
          ).stat().st_size,
      },
      'external_freeze_authorization': gate_a[
          'external_freeze_authorization'
      ],
      'same_process_preflight': same_process,
      'successful_preflight': successful_preflight,
      'v3_3_3_run': gate_a['v3_3_3_run'],
      'v3_3_3_1_archive': gate_a['v3_3_3_1_archive'],
  }
  sys.modules[ATTESTATION_MODULE] = attestation
  runpy.run_path(
      str(_HERE / 'run_encoder_skip_ood_sidecar_v3_3_4_5.py'),
      run_name='__main__',
  )


if __name__ == '__main__':
  main()
