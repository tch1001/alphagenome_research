#!/usr/bin/env python3
"""One-shot exact-source Gate-0 diagnostic v3.1.2 with GPU preflight.

The scientific identity logic is reused unchanged from committed v3.1.1.
This successor adds only a sanitized, fail-closed external and same-process
RTX 3090 gate before the new one-shot boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping, Sequence


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))
import run_device_preflight_v3_1_2 as device_preflight  # pylint: disable=g-import-not-at-top
import run_exact_source_gate0_v3_1_1 as base  # pylint: disable=g-import-not-at-top


SCRIPT_VERSION = 'opensplice-exact-source-gate0-v3.1.2'
ATTEMPT_ID = 'opensplice-v3.1.2-exact-source-gate0-one-shot'
LOCK_COMMIT = base.LOCK_COMMIT
PROTOCOL_PATH = (
    _HERE
    / 'v3_wider_mechanism'
    / 'device_preflight_amendment_v3_1_2.md'
)
PROTOCOL_SHA256 = device_preflight.PROTOCOL_SHA256
FREEZE_PATH = _HERE / 'exact_source_gate0_v3_1_2_freeze.json'
OUTPUT_DIR = (
    _HERE / 'results' / 'v3_1_2_exact_source_gate0_identity_one_shot'
)
START_PATH = OUTPUT_DIR / 'ATTEMPT_STARTED.json'
V3_1_1_RESULT_DIR = (
    _HERE / 'results' / 'v3_1_1_exact_source_gate0_identity_one_shot'
)
V3_1_1_START_SHA256 = (
    '1c1d24219e49e089806a02956bf8d7a44bdbd103e28adad6217eb3f94b424587'
)
V3_1_1_START_ONLY_TREE_SHA256 = (
    'a310ecddcace66bc9362e18249b08506c8e84ed7217a594a17e464ffb130d510'
)
V3_1_1_PARTIAL_FAILURE_SHA256 = (
    'c3e09b9503ef4e84875719bcc2f85131bfca428c11a3458ccfc9389bc7bc0ab7'
)
V3_1_1_BUNDLE_COMMIT = 'c95c7c284487227fc6bcc2d1ae05a00088e37b17'
V3_1_1_ATTEMPT_ID = 'opensplice-v3.1.1-exact-source-gate0-one-shot'

def _configure_base_scientific_output() -> None:
  """Version the reused scientific helpers without import-time side effects."""
  # No threshold, target, batch, identity, persistence, or HLO logic is
  # reimplemented or modified here.
  base.SCRIPT_VERSION = SCRIPT_VERSION
  base.ATTEMPT_ID = ATTEMPT_ID
  base.OUTPUT_DIR = OUTPUT_DIR
  base.START_PATH = START_PATH


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--locked-checkout', required=True, type=Path)
  parser.add_argument('--checkpoint', type=Path)
  parser.add_argument('--successful-preflight', type=Path)
  parser.add_argument('--dry-run', action='store_true')
  return parser.parse_args()


def validate_v3_1_1_partial_failure() -> dict[str, Any]:
  start = V3_1_1_RESULT_DIR / 'ATTEMPT_STARTED.json'
  disclosure = V3_1_1_RESULT_DIR / 'PARTIAL_FAILURE.md'
  if base._sha256(start) != V3_1_1_START_SHA256:  # pylint: disable=protected-access
    raise ValueError('v3.1.1 append-only start record changed.')
  if base._sha256(disclosure) != V3_1_1_PARTIAL_FAILURE_SHA256:  # pylint: disable=protected-access
    raise ValueError('v3.1.1 partial-failure disclosure changed.')
  start_record = json.loads(start.read_text(encoding='utf-8'))
  if start_record.get('attempt_id') != V3_1_1_ATTEMPT_ID:
    raise ValueError('Unexpected v3.1.1 attempt ID.')
  if start_record['runtime_before_model_creation']['devices'] != [{
      'client_platform': 'cpu',
      'device_kind': 'cpu',
      'id': 0,
      'platform': 'cpu',
      'visible_order': 0,
  }]:
    raise ValueError('v3.1.1 partial failure no longer records CPU fallback.')
  return {
      'status': 'partial_infrastructure_failure_never_resume_or_retry',
      'bundle_commit': V3_1_1_BUNDLE_COMMIT,
      'start_path': str(start),
      'start_sha256': V3_1_1_START_SHA256,
      'start_only_tree_sha256': V3_1_1_START_ONLY_TREE_SHA256,
      'partial_failure_path': str(disclosure),
      'partial_failure_sha256': V3_1_1_PARTIAL_FAILURE_SHA256,
      'compiled_model_executables': 0,
      'identity_calls': 0,
      'active_interventions': 0,
      'confirmation_calls': 0,
  }


def validate_external_preflight(path: Path) -> dict[str, Any]:
  path = path.resolve()
  try:
    path.relative_to(device_preflight.PREFLIGHT_DIR.resolve())
  except ValueError as error:
    raise ValueError('External preflight is outside its append-only directory.') from error
  if not path.is_file() or path.suffix != '.json':
    raise ValueError('Successful external preflight record is missing.')
  record = json.loads(path.read_text(encoding='utf-8'))
  if record.get('script_version') != device_preflight.SCRIPT_VERSION:
    raise ValueError('External preflight has the wrong script version.')
  if record.get('status') != 'pass':
    raise ValueError('External preflight did not pass.')
  if not record.get('no_model_or_biological_access'):
    raise ValueError('External preflight model-access declaration is invalid.')
  if not record.get('no_jit_or_array_kernel'):
    raise ValueError('External preflight JIT declaration is invalid.')
  frozen = record.get('freeze', {})
  if frozen.get('sha256') != base._sha256(FREEZE_PATH):  # pylint: disable=protected-access
    raise ValueError('External preflight used a different freeze manifest.')
  device_preflight.validate_device_observation(record['observation'])
  log_bindings = {}
  for stream in ('stdout', 'stderr'):
    binding = record['logs'][stream]
    log_path = Path(binding['path']).resolve()
    try:
      log_path.relative_to(device_preflight.PREFLIGHT_DIR.resolve())
    except ValueError as error:
      raise ValueError(f'External {stream} log is outside preflight dir.') from error
    observed = base._sha256(log_path)  # pylint: disable=protected-access
    if observed != binding['sha256']:
      raise ValueError(f'External preflight {stream} log changed.')
    log_bindings[stream] = {
        'path': str(log_path), 'sha256': observed,
    }
  return {
      'passed': True,
      'path': str(path),
      'sha256': base._sha256(path),  # pylint: disable=protected-access
      'attempt_number': record['preflight_attempt_number'],
      'observation': record['observation'],
      'logs': log_bindings,
      'bundle_git_head': record['bundle']['git_head'],
      'freeze_sha256': frozen['sha256'],
  }


def validate_freeze(
    *,
    proto_build: Mapping[str, Any],
    import_provenance: Mapping[str, Any],
    mixed_precision: Mapping[str, Any],
    v3_1_1_failure: Mapping[str, Any],
) -> dict[str, Any]:
  frozen = json.loads(FREEZE_PATH.read_text(encoding='utf-8'))
  expected = {
      'preflight_script_version': device_preflight.SCRIPT_VERSION,
      'runner_script_version': SCRIPT_VERSION,
      'attempt_id': ATTEMPT_ID,
      'lock_commit': LOCK_COMMIT,
      'protocol_sha256': PROTOCOL_SHA256,
      'runner_sha256': base._sha256(Path(__file__).resolve()),  # pylint: disable=protected-access
      'wrapper_sha256': base._sha256(  # pylint: disable=protected-access
          Path(__file__).with_suffix('.sh')
      ),
      'preflight_sha256': base._sha256(  # pylint: disable=protected-access
          Path(device_preflight.__file__).resolve()
      ),
      'test_sha256': base._sha256(  # pylint: disable=protected-access
          _HERE / 'run_exact_source_gate0_v3_1_2_test.py'
      ),
      'v3_1_1_base_runner_sha256': base._sha256(  # pylint: disable=protected-access
          Path(base.__file__).resolve()
      ),
      'v3_1_1_freeze_sha256': base._sha256(  # pylint: disable=protected-access
          _HERE / 'exact_source_gate0_v3_1_1_freeze.json'
      ),
      'v3_1_1_failure': dict(v3_1_1_failure),
      'proto_build_manifest_sha256': proto_build['manifest_sha256'],
      'initial_transitive_import_tree_sha256': import_provenance[
          'module_tree_sha256'
      ],
      'mixed_precision': dict(mixed_precision),
      'expected_device_kind': device_preflight.EXPECTED_DEVICE_KIND,
      'expected_gpu_uuid': device_preflight.EXPECTED_GPU_UUID,
      'expected_compute_capability': (
          device_preflight.EXPECTED_COMPUTE_CAPABILITY
      ),
      'output_dir': str(OUTPUT_DIR.resolve()),
      'preflight_dir': str(device_preflight.PREFLIGHT_DIR.resolve()),
      'environment_contract': {
          'LD_LIBRARY_PATH': 'absent',
          'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
      },
  }
  for name, value in expected.items():
    if base._json_normalized(frozen.get(name)) != base._json_normalized(value):  # pylint: disable=protected-access
      raise ValueError(f'v3.1.2 freeze mismatch: {name}.')
  return {**frozen, 'path': str(FREEZE_PATH),
          'sha256': base._sha256(FREEZE_PATH)}  # pylint: disable=protected-access


def build_dry_run_plan(
    *,
    records: Sequence[Mapping[str, Any]],
    worktree: Mapping[str, Any],
    sources: Mapping[str, Any],
    module_sources: Mapping[str, Any],
    mixed_precision: Mapping[str, Any],
    proto_build: Mapping[str, Any],
    launch_freeze: Mapping[str, Any],
    v3_1_1_failure: Mapping[str, Any],
) -> dict[str, Any]:
  return {
      'script_version': SCRIPT_VERSION,
      'attempt_id': ATTEMPT_ID,
      'dry_run': True,
      'protocol_sha256': PROTOCOL_SHA256,
      'output_dir': str(OUTPUT_DIR),
      'output_dir_must_be_absent': True,
      'v3_1_1_preserved': dict(v3_1_1_failure),
      'sanitized_environment': {
          'LD_LIBRARY_PATH': 'absent',
          'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
      },
      'external_preflight': device_preflight.build_dry_run_plan(),
      'same_process_preflight_before_attempt_and_model': True,
      'locked_worktree': dict(worktree),
      'locked_sources': dict(sources),
      'initial_transitive_import_tree_sha256': module_sources[
          'module_tree_sha256'
      ],
      'mixed_precision': dict(mixed_precision),
      'proto_build_manifest_sha256': proto_build['manifest_sha256'],
      'launch_freeze': dict(launch_freeze),
      'identity_units': len(records),
      'all_false_apply_calls': len(records) * 2,
      'active_intervention_calls': 0,
      'confirmation_model_calls': 0,
      'post_start_any_failure_consumes_v3_1_2': True,
  }


def _error_record(error: BaseException, stage: str) -> dict[str, Any]:
  return {
      'stage': stage,
      'exception_type': type(error).__name__,
      'message': str(error),
      'traceback': ''.join(traceback.format_exception(error)),
  }


def main() -> None:
  args = _parse_args()
  _configure_base_scientific_output()
  device_preflight.assert_sanitized_environment()
  if base._sha256(PROTOCOL_PATH) != PROTOCOL_SHA256:  # pylint: disable=protected-access
    raise ValueError('Frozen v3.1.2 amendment hash mismatch.')
  if OUTPUT_DIR.exists():
    raise FileExistsError(
        'v3.1.2 scientific output already exists and cannot resume or retry.'
    )

  locked_root = args.locked_checkout.resolve()
  worktree = base.validate_locked_worktree(locked_root)
  protoc_text = os.environ.get('ALPHAGENOME_PROTOC_BIN')
  proto_root_text = os.environ.get('ALPHAGENOME_PROTO_ROOT')
  if not protoc_text or not proto_root_text:
    raise ValueError('Exact-source wrapper did not declare protobuf tools.')
  proto_build = base.prepare_generated_proto_binding(
      locked_root,
      protoc_bin=Path(protoc_text).resolve(),
      alphagenome_proto_root=Path(proto_root_text).resolve(),
  )
  proto_build = base.disclose_proto_generation(proto_build)
  base.validate_generated_proto_binding(locked_root)
  records = base.load_locked_identity_records()
  sources = base.validate_locked_sources(locked_root, records)
  modules = base.load_locked_modules(locked_root)
  module_sources = base.imported_module_provenance(modules, locked_root)
  mixed_precision = base.mixed_precision_provenance(locked_root)
  v3_1_1_failure = validate_v3_1_1_partial_failure()
  launch_freeze = validate_freeze(
      proto_build=proto_build,
      import_provenance=module_sources,
      mixed_precision=mixed_precision,
      v3_1_1_failure=v3_1_1_failure,
  )
  cases = base.validate_static_cases(modules, records)
  checkpoint = modules.v2._checkpoint_path(  # pylint: disable=protected-access
      args.checkpoint
  )
  if checkpoint.name != modules.route_v3.CHECKPOINT_SNAPSHOT:
    raise ValueError('Checkpoint snapshot differs from frozen Phase-R lock.')
  if str(checkpoint) != records[0]['configuration']['checkpoint_path']:
    raise ValueError('Checkpoint path differs from frozen Phase-R lock.')

  if args.dry_run:
    print(json.dumps(build_dry_run_plan(
        records=records,
        worktree=worktree,
        sources=sources,
        module_sources=module_sources,
        mixed_precision=mixed_precision,
        proto_build=proto_build,
        launch_freeze=launch_freeze,
        v3_1_1_failure=v3_1_1_failure,
    ), indent=2, default=str))
    return

  if args.successful_preflight is None:
    raise ValueError('Actual v3.1.2 run requires an external preflight record.')
  bundle = device_preflight.validate_committed_bundle()
  external_preflight = validate_external_preflight(args.successful_preflight)
  if external_preflight['bundle_git_head'] != bundle['git_head']:
    raise ValueError('External preflight used a different committed bundle.')
  same_process_preflight = device_preflight.collect_device_observation()
  # This second call explicitly asserts the dtypes before any lowering.
  runtime_before_model = base.runtime_provenance(modules)

  start_record = {
      'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION,
      'status': 'started_append_only_one_shot',
      'started_at_unix_s': time.time(),
      'bundle': bundle,
      'protocol_path': str(PROTOCOL_PATH),
      'protocol_sha256': PROTOCOL_SHA256,
      'launch_freeze': launch_freeze,
      'v3_1_1_partial_failure': v3_1_1_failure,
      'environment_contract': device_preflight.assert_sanitized_environment(),
      'external_device_preflight': external_preflight,
      'same_process_device_preflight': same_process_preflight,
      'worktree': worktree,
      'generated_proto_build_provenance': proto_build,
      'locked_sources': sources,
      'import_resolution': module_sources,
      'mixed_precision': mixed_precision,
      'checkpoint_path': str(checkpoint),
      'checkpoint_snapshot': checkpoint.name,
      'locked_phase_r_identity_tree_sha256': base.LOCKED_PHASE_R_TREE_SHA256,
      'runtime_before_model_creation': runtime_before_model,
      'one_shot_policy': {
          'identity_units': base.EXPECTED_CASES,
          'calls_per_identity': 2,
          'active_interventions': 0,
          'resume': False,
          'retry': False,
          'overwrite': False,
          'post_start_any_failure_consumes_attempt': True,
      },
  }
  base._ensure_fresh_attempt(start_record)  # pylint: disable=protected-access

  statuses = []
  terminal = False
  compiled_apply = None
  compile_seconds = 0.0
  try:
    model_instance = modules.dna_model.create(
        checkpoint,
        model_settings=modules.dna_model.ModelSettings(
            attention_backend=modules.route_v3.ATTENTION_BACKEND
        ),
    )
    base._write_new(  # pylint: disable=protected-access
        OUTPUT_DIR / 'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
        {
            'attempt_id': ATTEMPT_ID,
            'stage': 'post_model_precompile',
            'provenance': base.imported_module_provenance(modules, locked_root),
            'created_at_unix_s': time.time(),
        },
    )
    raw_apply = (
        modules.dna_model
        .create_splice_classification_logit_margin_route_census_apply(
            model_instance._metadata,  # pylint: disable=protected-access
            attention_backend=modules.route_v3.ATTENTION_BACKEND,
        )
    )
    jitted_apply = modules.jax.jit(raw_apply)
    for index, (case, record) in enumerate(zip(cases, records, strict=True)):
      if compiled_apply is None:
        interval = modules.v2.centered_interval(
            case, modules.route_v3.CONTEXT_BP
        )
        positions = modules.v2.trace_position_sets(case, interval)
        selection = modules.phase_r_v3.phase_r_trace_selection(positions)
        metadata = model_instance._metadata[  # pylint: disable=protected-access
            modules.public_dna_model.Organism.HOMO_SAPIENS
        ].splice_sites
        target, resolved = modules.route_v3.target_selection(
            metadata, case, interval
        )
        dna_batch, sequence_sha = modules.route_v3._build_six_row_batch(  # pylint: disable=protected-access
            model_instance, case, interval
        )
        base.validate_live_case_linkage(
            modules, case, interval, resolved, sequence_sha, record
        )
        interventions = modules.phase_r_v3.group_interventions(selection, None)
        compile_args = (
            model_instance._params,  # pylint: disable=protected-access
            model_instance._state,  # pylint: disable=protected-access
            dna_batch,
            modules.jnp.zeros((6,), modules.jnp.int32),
            selection,
            interventions,
            target,
        )
        compiled_apply, _, compile_seconds = base.compile_with_provenance(
            jitted_apply, compile_args
        )
      artifact, case_terminal = base.run_identity_unit(
          modules,
          model_instance,
          compiled_apply,
          case,
          record,
          compile_seconds=compile_seconds if index == 0 else 0.0,
      )
      path = base._case_path(case)  # pylint: disable=protected-access
      base._write_new(path, artifact)  # pylint: disable=protected-access
      statuses.append({
          'order': case.order,
          'variant_id': case.variant_id,
          'gene': case.gene,
          'status': artifact['status'],
          'artifact': str(path.relative_to(OUTPUT_DIR)),
      })
      if case_terminal:
        terminal = True
        break
  except Exception as error:
    terminal = True
    base._write_new(  # pylint: disable=protected-access
        OUTPUT_DIR / 'TERMINAL_FAILURE.json',
        {
            'attempt_id': ATTEMPT_ID,
            'status': 'runtime_failure',
            'failure': _error_record(error, 'model_load_or_compile'),
            'created_at_unix_s': time.time(),
        },
    )

  base._write_new(  # pylint: disable=protected-access
      OUTPUT_DIR / 'COMPLETION_PROVENANCE.json',
      {
          'attempt_id': ATTEMPT_ID,
          'runtime_after_cohort': base.runtime_provenance(modules),
          'transitive_import_provenance_at_completion': (
              base.imported_module_provenance(modules, locked_root)
          ),
          'autotune': base.autotune_provenance(),
          'created_at_unix_s': time.time(),
      },
  )
  published = sorted(
      path for path in OUTPUT_DIR.rglob('*')
      if path.is_file() and path.name != 'SUMMARY.json'
  )
  summary = base._summary(published, statuses, terminal=terminal)  # pylint: disable=protected-access
  summary_path = OUTPUT_DIR / 'SUMMARY.json'
  base._write_new(summary_path, summary)  # pylint: disable=protected-access
  print(summary_path.resolve())
  if not summary['cohort_passed']:
    raise SystemExit(2)


if __name__ == '__main__':
  main()
