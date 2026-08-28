#!/usr/bin/env python3
"""One-shot development-only OpenSplice v3.3.2 OOD sidecar runner."""

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


_PREIMPORT_ATTESTATION_MODULE = (
    '_opensplice_v3_3_2_ood_sidecar_bootstrap_attestation'
)
if (
    __name__ == '__main__'
    and _PREIMPORT_ATTESTATION_MODULE not in sys.modules
):
  raise RuntimeError(
      'Direct v3.3.2 execution is forbidden before pre-import bootstrap; '
      'use launch_encoder_skip_ood_sidecar_v3_3_2.py.'
  )

import jax
import jax.numpy as jnp
import numpy as np

from alphagenome_research.model import dna_model
from alphagenome_research.model import interpretability


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))
# pylint: disable=g-import-not-at-top
import run_encoder_skip_factorial_v3_3 as v33
import run_superset_graph_v3_2 as v32
import validate_encoder_skip_ood_sidecar_bootstrap_v3_3_2 as bootstrap


SCRIPT_VERSION = 'opensplice-encoder-skip-ood-sidecar-v3.3.2'
ATTEMPT_ID = 'opensplice-v3.3.2-development-ood-sidecar-one-shot'
PREFLIGHT_SCRIPT_VERSION = 'opensplice-device-preflight-v3.3.2'
ATTESTATION_MODULE = _PREIMPORT_ATTESTATION_MODULE
AMENDMENT_PATH = bootstrap.AMENDMENT_PATH
AMENDMENT_SHA256 = bootstrap.AMENDMENT_SHA256
AMENDMENT_COMMIT = bootstrap.AMENDMENT_COMMIT
ORIGINAL_PROTOCOL_SHA256 = bootstrap.ORIGINAL_PROTOCOL_SHA256
ORIGINAL_FREEZE_PATH = bootstrap.ORIGINAL_FREEZE_PATH
ORIGINAL_RUN_DIR = bootstrap.ORIGINAL_RUN_DIR
FREEZE_PATH = bootstrap.FREEZE_PATH
OUTPUT_DIR = bootstrap.OUTPUT_DIR
ANALYSIS_DIR = bootstrap.ANALYSIS_DIR
PREFLIGHT_DIR = bootstrap.PREFLIGHT_DIR
START_PATH = OUTPUT_DIR / 'ATTEMPT_STARTED.json'
ANCHOR_IDS = (0, 127, 128, 255)
RECIPIENT_ORDERS = tuple(range(20))
INVARIANT_ROWS = (0, 1, 3, 5, 6, 7)
ACTIVE_RECIPIENT_ROWS = (2, 4)
EXPECTED_RECORD_COUNT = 80
EXPECTED_APPLY_COUNT = 320
MAX_WALL_TIME_SECONDS = 2 * 60 * 60
MAX_OUTPUT_BYTES = 1024 * 1024 * 1024


def _sha256(path: Path) -> str:
  return v32._sha256(path)  # pylint: disable=protected-access


def _write_new(path: Path, value: Any) -> str:
  return v32._write_new(path, value)  # pylint: disable=protected-access


def _write_new_text(path: Path, value: str) -> str:
  return v32._write_new_text(path, value)  # pylint: disable=protected-access


def _reject_confirmation_path(path: Path) -> None:
  v32._reject_confirmation_path(path)  # pylint: disable=protected-access


def _slug(value: str) -> str:
  return v32._slug(value)  # pylint: disable=protected-access


def sidecar_execution_order(
    cases: Sequence[Any],
) -> tuple[tuple[int, int], ...]:
  """Returns the frozen recipient-major, anchor-minor 80-record order."""
  if tuple(case.order for case in cases) != RECIPIENT_ORDERS:
    raise ValueError('v3.3.2 requires development recipient orders 0--19.')
  result = tuple(
      (order, anchor_id)
      for order in RECIPIENT_ORDERS
      for anchor_id in ANCHOR_IDS
  )
  if len(result) != EXPECTED_RECORD_COUNT or len(set(result)) != len(result):
    raise ValueError('v3.3.2 sidecar order is incomplete or duplicated.')
  return result


def build_dry_run_plan(
    cases: Sequence[Any], *, max_variants: int, max_anchors: int
) -> dict[str, Any]:
  sidecar_execution_order(cases)
  return {
      'script_version': SCRIPT_VERSION,
      'amendment_sha256': AMENDMENT_SHA256,
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'dry_run': True,
      'development_case_count': 20,
      'recipient_orders': list(RECIPIENT_ORDERS),
      'anchor_ids': list(ANCHOR_IDS),
      'execution_order': 'recipient-major, anchor-minor',
      'ood_record_count': EXPECTED_RECORD_COUNT,
      'applies_per_record': 4,
      'model_apply_count': EXPECTED_APPLY_COUNT,
      'eight_row_compile_count': 1,
      'six_row_compile_count': 0,
      'identity_rerun_count': 0,
      'main_cube_rerun_count': 0,
      'old_ood_records_reused': 0,
      'displayed_variants': min(20, max_variants or 20),
      'displayed_anchors': min(4, max_anchors or 4),
      'confirmation_model_calls': 0,
      'output_dir': str(OUTPUT_DIR),
      'analysis_dir': str(ANALYSIS_DIR),
  }


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', type=Path)
  parser.add_argument('--successful-preflight', type=Path)
  parser.add_argument('--dry-run', action='store_true')
  parser.add_argument('--max-variants', type=int, default=0)
  parser.add_argument('--max-anchors', type=int, default=0)
  args = parser.parse_args()
  if not args.dry_run and (args.max_variants or args.max_anchors):
    parser.error('Bounded flags are dry-run-only; v3.3.2 is all-or-none.')
  if args.max_variants < 0 or args.max_anchors < 0:
    parser.error('Dry-run bounds must be non-negative.')
  return args


def _artifact_path(case: Any, anchor_id: int) -> Path:
  case_key = f'{case.order:03d}_{_slug(case.variant_id)}'
  return (
      OUTPUT_DIR / 'raw' / 'ood_anchors' / case_key
      / f'{anchor_id:03d}.json'
  )


def _old_raw_relative(case: Any, family: str, anchor_id: int | None) -> str:
  case_key = f'{case.order:03d}_{_slug(case.variant_id)}'
  if family == 'identity':
    if anchor_id is not None:
      raise ValueError('Identity binding must not include an anchor ID.')
    return f'raw/identity/{case_key}.json'
  if family == 'coalition' and anchor_id is not None:
    return f'raw/coalitions/{case_key}/{anchor_id:03d}.json'
  raise ValueError('Unknown original artifact binding family.')


def original_artifact_binding(
    original_manifest: Mapping[str, Any],
    case: Any,
    family: str,
    anchor_id: int | None = None,
) -> dict[str, str]:
  relative = _old_raw_relative(case, family, anchor_id)
  expected_sha = original_manifest['artifact_sha256'].get(relative)
  if not isinstance(expected_sha, str):
    raise ValueError(f'Original linked artifact is absent: {relative}.')
  path = (ORIGINAL_RUN_DIR / relative).resolve()
  if not path.is_relative_to(ORIGINAL_RUN_DIR.resolve()):
    raise ValueError('Original linked artifact escaped the frozen run.')
  if _sha256(path) != expected_sha:
    raise ValueError(f'Original linked artifact changed: {relative}.')
  return {
      'path': relative,
      'sha256': expected_sha,
  }


def compact_rowwise_fingerprint(values: Any) -> dict[str, Any]:
  """Hashes each batch row's exact host bytes with explicit shape/dtype.

  SHA-256 is compact collision-resistant audit evidence, not a substitute for
  the direct ``np.array_equal`` gates performed while the arrays are live.
  """
  array = np.asarray(values)
  if array.ndim < 1 or array.shape[0] != 8:
    raise ValueError('Rowwise sidecar fingerprints require batch axis 8.')
  rows = []
  for index in range(8):
    row = np.ascontiguousarray(array[index])
    rows.append({
        'row': index,
        'shape': list(row.shape),
        'dtype': str(row.dtype),
        'size_bytes': int(row.nbytes),
        'sha256': hashlib.sha256(row.tobytes(order='C')).hexdigest(),
    })
  return {
      'full_shape': list(array.shape),
      'dtype': str(array.dtype),
      'row_count': 8,
      'rows': rows,
      'collision_semantics': (
          'SHA-256 per exact row byte string; direct live equality is the gate.'
      ),
  }


def trace_rowwise_fingerprints(
    trace: interpretability.SupersetGraphTrace,
) -> dict[str, Any]:
  branch = trace.stage_a
  return {
      'natural_final_embeddings': compact_rowwise_fingerprint(
          branch.natural_final_embeddings
      ),
      'effective_final_embeddings': compact_rowwise_fingerprint(
          branch.effective_final_embeddings
      ),
      'transformer_output_natural_fingerprint': {
          'shape': list(np.asarray(
              branch.transformer_output_natural_fingerprint
          ).shape),
          'dtype': str(np.asarray(
              branch.transformer_output_natural_fingerprint
          ).dtype),
          'values': np.asarray(
              branch.transformer_output_natural_fingerprint
          ).tolist(),
      },
      'encoder_skips_natural_fingerprints': {
          'shape': list(np.asarray(
              branch.encoder_skips_natural_fingerprints
          ).shape),
          'dtype': str(np.asarray(
              branch.encoder_skips_natural_fingerprints
          ).dtype),
          'values': np.asarray(
              branch.encoder_skips_natural_fingerprints
          ).tolist(),
      },
  }


def _assert_rows_equal(left: Any, right: Any, rows: Sequence[int]) -> None:
  left_array = np.asarray(left)
  right_array = np.asarray(right)
  if left_array.shape != right_array.shape or left_array.shape[0] != 8:
    raise ValueError('Eight-row tensor shapes differ between sidecar calls.')
  for row in rows:
    if not np.array_equal(left_array[row], right_array[row]):
      raise ValueError(
          f'Eight-row downstream natural route differs at invariant row {row}.'
      )


def validate_ood_sidecar_anchor(
    intended: tuple[Any, interpretability.SupersetGraphTrace],
    intended_repeated: tuple[Any, interpretability.SupersetGraphTrace],
    unrelated: tuple[Any, interpretability.SupersetGraphTrace],
    unrelated_repeated: tuple[Any, interpretability.SupersetGraphTrace],
    anchor_id: int,
    intended_interventions: interpretability.SupersetGraphInterventions,
    unrelated_interventions: interpretability.SupersetGraphInterventions,
) -> dict[str, Any]:
  """Applies the sole v3.3.2 repair while retaining all stronger gates."""
  if anchor_id not in ANCHOR_IDS:
    raise ValueError('v3.3.2 received an unfrozen OOD anchor.')
  intended_evidence, intended_trace = intended
  intended_repeat_evidence, intended_repeat_trace = intended_repeated
  unrelated_evidence, unrelated_trace = unrelated
  unrelated_repeat_evidence, unrelated_repeat_trace = unrelated_repeated
  v32._assert_evidence_repeat(  # pylint: disable=protected-access
      intended_evidence, intended_repeat_evidence
  )
  v32._assert_trace_repeat(  # pylint: disable=protected-access
      intended_trace, intended_repeat_trace
  )
  v32._assert_evidence_repeat(  # pylint: disable=protected-access
      unrelated_evidence, unrelated_repeat_evidence
  )
  v32._assert_trace_repeat(  # pylint: disable=protected-access
      unrelated_trace, unrelated_repeat_trace
  )
  v33._assert_runtime_transfer_contract(  # pylint: disable=protected-access
      intended_interventions,
      anchor_id,
      batch_size=8,
      donor_rows=v33.EIGHT_INTENDED_DONOR_ROWS,
      identity_rows=v33.EIGHT_IDENTITY_ROWS,
  )
  v33._assert_runtime_transfer_contract(  # pylint: disable=protected-access
      unrelated_interventions,
      anchor_id,
      batch_size=8,
      donor_rows=v33.EIGHT_UNRELATED_DONOR_ROWS,
      identity_rows=v33.EIGHT_IDENTITY_ROWS,
  )
  v33._validate_eight_route_trace(  # pylint: disable=protected-access
      intended_trace, anchor_id, unrelated=False
  )
  v33._validate_eight_route_trace(  # pylint: disable=protected-access
      unrelated_trace, anchor_id, unrelated=True
  )

  for natural_name, _ in v32._TRANSFORMER_PAIRS:  # pylint: disable=protected-access
    if not np.array_equal(
        getattr(intended_trace.transformer, natural_name),
        getattr(unrelated_trace.transformer, natural_name),
    ):
      raise ValueError(
          f'Eight-row upstream transformer seam differs: {natural_name}.'
      )
  intended_branch = intended_trace.stage_a
  unrelated_branch = unrelated_trace.stage_a
  for field in (
      'transformer_output_natural_fingerprint',
      'encoder_skips_natural_fingerprints',
  ):
    if not np.array_equal(
        getattr(intended_branch, field), getattr(unrelated_branch, field)
    ):
      raise ValueError(f'Eight-row upstream natural route differs: {field}.')
  _assert_rows_equal(
      intended_branch.natural_final_embeddings,
      unrelated_branch.natural_final_embeddings,
      INVARIANT_ROWS,
  )
  if anchor_id == 0:
    _assert_rows_equal(
        intended_branch.natural_final_embeddings,
        unrelated_branch.natural_final_embeddings,
        tuple(range(8)),
    )
    for branch in (intended_branch, unrelated_branch):
      natural_final = np.asarray(branch.natural_final_embeddings)
      if (
          not np.array_equal(natural_final[2], natural_final[1])
          or not np.array_equal(natural_final[4], natural_final[0])
      ):
        raise ValueError(
            'ID0 natural-final recipient rows are not same-allele no-ops.'
        )

  intended_readout = v33.target_readout(intended_evidence, batch_size=8)
  unrelated_readout = v33.target_readout(unrelated_evidence, batch_size=8)
  for readout in (intended_readout, unrelated_readout):
    v33._assert_readout_rows_equal(  # pylint: disable=protected-access
        readout, 3, readout, 1
    )
    v33._assert_readout_rows_equal(  # pylint: disable=protected-access
        readout, 5, readout, 0
    )
  for row in INVARIANT_ROWS:
    v33._assert_readout_rows_equal(  # pylint: disable=protected-access
        intended_readout, row, unrelated_readout, row
    )
  if anchor_id == 0:
    for readout in (intended_readout, unrelated_readout):
      v33._assert_readout_rows_equal(  # pylint: disable=protected-access
          readout, 2, readout, 1
      )
      v33._assert_readout_rows_equal(  # pylint: disable=protected-access
          readout, 4, readout, 0
      )
    for row in range(8):
      v33._assert_readout_rows_equal(  # pylint: disable=protected-access
          intended_readout, row, unrelated_readout, row
      )
  if anchor_id == 255:
    v33._assert_readout_rows_equal(  # pylint: disable=protected-access
        intended_readout, 2, intended_readout, 0
    )
    v33._assert_readout_rows_equal(  # pylint: disable=protected-access
        intended_readout, 4, intended_readout, 1
    )
    v33._assert_readout_rows_equal(  # pylint: disable=protected-access
        unrelated_readout, 2, unrelated_readout, 6
    )
    v33._assert_readout_rows_equal(  # pylint: disable=protected-access
        unrelated_readout, 4, unrelated_readout, 7
    )
  return {
      'passed': True,
      'corrected_host_assertion_version': 'v3.3.2',
      'upstream_transformer_natural_tensors_all8_exact_between_calls': True,
      'upstream_T_E_natural_fingerprints_all8_exact_between_calls': True,
      'natural_final_invariant_rows_exact_between_calls': True,
      'natural_final_invariant_rows': list(INVARIANT_ROWS),
      'active_rows_cross_call_equality_not_required': list(
          ACTIVE_RECIPIENT_ROWS
      ),
      'active_rows_forced_difference_not_required': True,
      'full_within_call_natural_effective_final_exact': True,
      'endpoint_invariant_rows_exact_between_calls': True,
      'self_rows_exact_within_each_call': True,
      'id0_all8_natural_final_exact_between_calls': anchor_id == 0,
      'id0_within_call_natural_final_recipient_noop_exact': anchor_id == 0,
      'id0_all8_endpoint_exact_between_calls': anchor_id == 0,
      'id0_recipient_noop_exact': anchor_id == 0,
      'id255_intended_endpoint_closure_exact': anchor_id == 255,
      'id255_unrelated_endpoint_closure_exact': anchor_id == 255,
      'intended_route_tensor_donor_exact': True,
      'unrelated_route_tensor_donor_exact': True,
      'enabled_disabled_T_E_exact': True,
      'runtime_route_masks_and_maps_exact': True,
      'intended_target_repeat_exact': True,
      'intended_trace_repeat_exact': True,
      'unrelated_target_repeat_exact': True,
      'unrelated_trace_repeat_exact': True,
      'transformer_internal_seams_disabled_exact': True,
      'final_embedding_disabled_exact': True,
      'normalization_computed': False,
  }


def consume_bootstrap_attestation() -> dict[str, Any]:
  module = sys.modules.pop(ATTESTATION_MODULE, None)
  record = getattr(module, 'record', None)
  if not isinstance(record, dict):
    raise RuntimeError(
        'Direct v3.3.2 runner invocation is forbidden; use its launcher.'
    )
  if record.get('pid') != os.getpid():
    raise RuntimeError('v3.3.2 bootstrap attestation came from another process.')
  freeze = record.get('freeze', {})
  if (
      freeze.get('sha256') != _sha256(FREEZE_PATH)
      or freeze.get('tracked_head_clean') is not True
  ):
    raise RuntimeError('v3.3.2 bootstrap freeze attestation changed.')
  original = freeze.get('original_run', {})
  for name, value in bootstrap.EXPECTED_ORIGINAL_BINDING.items():
    if original.get(name) != value:
      raise RuntimeError(f'Original-v3.3 bootstrap binding changed: {name}.')
  if original.get('status_predicates') != bootstrap.EXPECTED_ORIGINAL_STATUS:
    raise RuntimeError('Original-v3.3 structural predicates changed.')
  if record.get('sanitized_environment') != {
      'LD_LIBRARY_PATH': 'absent',
      'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
      'JAX_ENABLE_COMPILATION_CACHE': 'false',
  }:
    raise RuntimeError('v3.3.2 launcher environment attestation changed.')
  for field, path in (
      ('launcher', _HERE / 'launch_encoder_skip_ood_sidecar_v3_3_2.py'),
      ('bootstrap', _HERE / 'validate_encoder_skip_ood_sidecar_bootstrap_v3_3_2.py'),
  ):
    if record.get(f'{field}_path') != str(path.resolve()):
      raise RuntimeError(f'v3.3.2 bootstrap {field} path changed.')
    if record.get(f'{field}_sha256') != _sha256(path):
      raise RuntimeError(f'v3.3.2 bootstrap {field} hash changed.')
  return {
      'pid': record['pid'],
      'created_at_unix_s': record['created_at_unix_s'],
      'sanitized_environment': record['sanitized_environment'],
      'generated_bindings': record['generated_bindings'],
      'launcher_path': record['launcher_path'],
      'launcher_sha256': record['launcher_sha256'],
      'bootstrap_path': record['bootstrap_path'],
      'bootstrap_sha256': record['bootstrap_sha256'],
      'freeze_path': freeze['path'],
      'freeze_sha256': freeze['sha256'],
      'git_head': freeze['git_head'],
      'tracked_head_clean': True,
      'original_run': {
          key: original[key]
          for key in bootstrap.EXPECTED_ORIGINAL_BINDING
      },
      'original_eight_row_compiler': original['eight_row_compiler'],
      'original_run_all_5158_files_rehashed': True,
      'v3_3_1_status': freeze['v3_3_1_status'],
      'original_bundle': freeze['original_bundle'],
  }


def validate_external_preflight(
    path: Path, frozen: Mapping[str, Any]
) -> dict[str, Any]:
  path = path.resolve()
  _reject_confirmation_path(path)
  if not path.is_relative_to(PREFLIGHT_DIR.resolve()):
    raise ValueError('v3.3.2 preflight is outside its append-only directory.')
  record = json.loads(path.read_text(encoding='utf-8'))
  expected = {
      'script_version': PREFLIGHT_SCRIPT_VERSION,
      'status': 'pass',
      'amendment_sha256': AMENDMENT_SHA256,
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'freeze_sha256': _sha256(FREEZE_PATH),
      'no_model_or_biological_access': True,
      'no_jit_or_array_kernel': True,
  }
  for name, value in expected.items():
    if record.get(name) != value:
      raise ValueError(f'External v3.3.2 preflight changed: {name}.')
  if record.get('failure') is not None:
    raise ValueError('External v3.3.2 preflight contains a failure.')
  logs = record.get('logs', {})
  validated_logs = {}
  for stream in ('stdout', 'stderr'):
    binding = logs.get(stream, {})
    log_path = Path(str(binding.get('path', ''))).resolve()
    if not log_path.is_relative_to(PREFLIGHT_DIR.resolve()):
      raise ValueError(f'External preflight {stream} log escaped directory.')
    if _sha256(log_path) != binding.get('sha256'):
      raise ValueError(f'External preflight {stream} log changed.')
    validated_logs[stream] = {
        'path': str(log_path), 'sha256': binding['sha256']
    }
  observation = record.get('observation', {})
  v33.v32.device_gate.validate_device_observation(observation)
  v33.validate_device_version_manifest(
      observation, json.loads(ORIGINAL_FREEZE_PATH.read_text(encoding='utf-8'))
  )
  return {
      'path': str(path),
      'sha256': _sha256(path),
      **record,
      'validated_logs': validated_logs,
  }


def import_provenance(original_frozen: Mapping[str, Any]) -> dict[str, Any]:
  record = v33.import_provenance(original_frozen)
  scientific_paths = (
      Path(__file__).resolve(),
      _HERE / 'launch_encoder_skip_ood_sidecar_v3_3_2.py',
      _HERE / 'validate_encoder_skip_ood_sidecar_bootstrap_v3_3_2.py',
      _HERE / 'run_device_preflight_v3_3_2.py',
      AMENDMENT_PATH,
      FREEZE_PATH,
      _HERE / 'run_encoder_skip_factorial_v3_3.py',
  )
  record['v3_3_2_sidecar_sources'] = {
      str(path.resolve()): {
          'sha256': _sha256(path.resolve()),
          'size_bytes': path.resolve().stat().st_size,
      }
      for path in scientific_paths
  }
  return record


def _compiler_artifacts(
    lowered: Any,
    compiled: Any,
    seconds: float,
    original_compiler: Mapping[str, Any],
    program_signatures: Mapping[str, Any],
) -> dict[str, Any]:
  directory = OUTPUT_DIR / 'compiler' / 'eight_row'
  stable = str(lowered.compiler_ir(dialect='stablehlo'))
  hlo_object = lowered.compiler_ir(dialect='hlo')
  hlo = (
      hlo_object.as_hlo_text()
      if hasattr(hlo_object, 'as_hlo_text') else str(hlo_object)
  )
  compiled_hlo = compiled.as_text()
  artifacts = {}
  for name, filename, content in (
      ('stablehlo', 'graph.stablehlo.mlir', stable),
      ('hlo', 'graph.pre_backend.hlo.txt', hlo),
      ('compiled_hlo', 'graph.compiled.hlo.txt', compiled_hlo),
  ):
    path = directory / filename
    digest = _write_new_text(path, content)
    artifacts[name] = {
        'path': str(path),
        'sha256': digest,
        'size_bytes': len(content.encode('utf-8')),
    }
  fingerprint = hashlib.sha256(
      bytes.fromhex(artifacts['compiled_hlo']['sha256'])
  ).hexdigest()
  comparison = {
      name: {
          'sha256_exact': artifacts[name]['sha256']
          == original_compiler['artifacts'][name]['sha256'],
          'size_exact': artifacts[name]['size_bytes']
          == original_compiler['artifacts'][name]['size_bytes'],
      }
      for name in ('stablehlo', 'hlo', 'compiled_hlo')
  }
  graph_exact = all(
      check['sha256_exact'] and check['size_exact']
      for check in comparison.values()
  ) and fingerprint == original_compiler['executable_fingerprint']
  record = {
      'executable_name': 'eight_row',
      'compile_count': 1,
      'compile_seconds': seconds,
      'executable_fingerprint': fingerprint,
      'artifacts': artifacts,
      'program_signatures': dict(program_signatures),
      'original_v3_3_compiler_binding': dict(original_compiler),
      'original_graph_comparison': comparison,
      'original_executable_fingerprint_exact': (
          fingerprint == original_compiler['executable_fingerprint']
      ),
      'graph_and_hlo_exact_to_original_v3_3': graph_exact,
  }
  _write_new(directory / 'COMPILER_PROVENANCE.json', record)
  return record


def _eight_row_batch(
    recipient_common: tuple[Any, ...], donor_common: tuple[Any, ...]
) -> np.ndarray:
  return v33._eight_row_batch(  # pylint: disable=protected-access
      recipient_common, donor_common
  )


def _case_record(case: Any) -> dict[str, Any]:
  return v32._case_record(case)  # pylint: disable=protected-access


def _counted_apply(
    compiled: Any, args: Sequence[Any], apply_counter: list[int]
) -> tuple[Any, float]:
  """Records every invocation before dispatch, including failed calls."""
  if len(apply_counter) != 1 or apply_counter[0] < 0:
    raise ValueError('Apply counter must be a one-element non-negative list.')
  apply_counter[0] += 1
  return v32._timed_apply(compiled, args)  # pylint: disable=protected-access


def _run_anchor(
    compiled: Any,
    recipient: Any,
    donor: Any,
    recipient_common: tuple[Any, ...],
    donor_common: tuple[Any, ...],
    params: Any,
    state: Any,
    anchor_id: int,
    signatures: Mapping[str, Any],
    freeze_sha256: str,
    executable_fingerprint: str,
    execution_index: int,
    original_manifest: Mapping[str, Any],
    apply_counter: list[int],
) -> dict[str, Any]:
  _, selection, target, recipient_sequence_sha = recipient_common
  donor_sequence_sha = donor_common[3]
  dna = _eight_row_batch(recipient_common, donor_common)
  intended_interventions = v33.eight_row_interventions(
      selection, anchor_id, unrelated=False
  )
  unrelated_interventions = v33.eight_row_interventions(
      selection, anchor_id, unrelated=True
  )
  for interventions in (intended_interventions, unrelated_interventions):
    v32.assert_same_program_signature(
        signatures['eight_interventions'], interventions
    )
  v32.assert_same_program_signature(signatures['selection'], selection)
  v32.assert_same_program_signature(signatures['target'], target)
  common_args = (
      params, state, dna, jnp.zeros((8,), jnp.int32), selection,
  )
  intended, intended_seconds = _counted_apply(
      compiled, (*common_args, intended_interventions, target), apply_counter
  )
  intended_repeat, intended_repeat_seconds = _counted_apply(
      compiled, (*common_args, intended_interventions, target), apply_counter
  )
  unrelated, unrelated_seconds = _counted_apply(
      compiled, (*common_args, unrelated_interventions, target), apply_counter
  )
  unrelated_repeat, unrelated_repeat_seconds = _counted_apply(
      compiled, (*common_args, unrelated_interventions, target), apply_counter
  )
  intended_readout = None
  intended_repeat_readout = None
  unrelated_readout = None
  unrelated_repeat_readout = None
  rowwise = None
  trace_fingerprints = None
  original_bindings = None
  raw_movement = None
  try:
    intended_readout = v33.target_readout(intended[0], batch_size=8)
    intended_repeat_readout = v33.target_readout(
        intended_repeat[0], batch_size=8
    )
    unrelated_readout = v33.target_readout(unrelated[0], batch_size=8)
    unrelated_repeat_readout = v33.target_readout(
        unrelated_repeat[0], batch_size=8
    )
    rowwise = {
        'intended': trace_rowwise_fingerprints(intended[1]),
        'intended_repeat': trace_rowwise_fingerprints(intended_repeat[1]),
        'unrelated': trace_rowwise_fingerprints(unrelated[1]),
        'unrelated_repeat': trace_rowwise_fingerprints(unrelated_repeat[1]),
    }
    trace_fingerprints = {
        'intended': v32.trace_fingerprint(intended[1]),
        'intended_repeat': v32.trace_fingerprint(intended_repeat[1]),
        'unrelated': v32.trace_fingerprint(unrelated[1]),
        'unrelated_repeat': v32.trace_fingerprint(unrelated_repeat[1]),
    }
    original_bindings = {
        'recipient_identity': original_artifact_binding(
            original_manifest, recipient, 'identity'
        ),
        'donor_identity': original_artifact_binding(
            original_manifest, donor, 'identity'
        ),
        'recipient_six_row_coalition': original_artifact_binding(
            original_manifest, recipient, 'coalition', anchor_id
        ),
    }
    raw_movement = {
        'intended': v33.raw_bidirectional_movement(intended_readout),
        'unrelated': v33.raw_bidirectional_movement(unrelated_readout),
    }
    checks = validate_ood_sidecar_anchor(
        intended,
        intended_repeat,
        unrelated,
        unrelated_repeat,
        anchor_id,
        intended_interventions,
        unrelated_interventions,
    )
    status, failure = 'complete', None
  except Exception as error:  # pylint: disable=broad-exception-caught
    checks = None
    status = 'invalid'
    failure = {'type': type(error).__name__, 'message': str(error)}
  artifact = {
      'status': status,
      'family': 'v3_3_2_unrelated_donor_sidecar_anchor',
      'script_version': SCRIPT_VERSION,
      'amendment_sha256': AMENDMENT_SHA256,
      'amendment_commit': AMENDMENT_COMMIT,
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'freeze_sha256': freeze_sha256,
      'execution_index': execution_index,
      'sidecar_execution_index': execution_index,
      'execution_order': 'recipient-major, anchor-minor',
      'eight_row_executable_fingerprint': executable_fingerprint,
      'same_eight_row_compiled_executable': True,
      'six_row_executable_used': False,
      'recipient_case': _case_record(recipient),
      'donor_case': _case_record(donor),
      'coalition': v33.coalition_metadata(anchor_id),
      'batch_roles': list(v33.EIGHT_ROLES),
      'natural_identity_rows': list(v33.EIGHT_IDENTITY_ROWS),
      'intended_donor_rows': list(v33.EIGHT_INTENDED_DONOR_ROWS),
      'unrelated_donor_rows': list(v33.EIGHT_UNRELATED_DONOR_ROWS),
      'invariant_rows_between_calls': list(INVARIANT_ROWS),
      'active_recipient_rows': list(ACTIVE_RECIPIENT_ROWS),
      'active_recipient_cross_call_equality_gate': False,
      'active_recipient_cross_call_inequality_gate': False,
      'original_artifact_bindings': original_bindings,
      'original_ood_records_used_as_data': False,
      'recipient_sequence_sha256': recipient_sequence_sha,
      'donor_sequence_sha256': donor_sequence_sha,
      'runtime_interventions': {
          'intended': v33.intervention_record(intended_interventions),
          'unrelated': v33.intervention_record(unrelated_interventions),
      },
      'intended_target_readout': intended_readout,
      'intended_repeat_target_readout': intended_repeat_readout,
      'unrelated_target_readout': unrelated_readout,
      'unrelated_repeat_target_readout': unrelated_repeat_readout,
      'intended_trace_fingerprint': (
          None if trace_fingerprints is None else trace_fingerprints['intended']
      ),
      'intended_repeat_trace_fingerprint': (
          None if trace_fingerprints is None
          else trace_fingerprints['intended_repeat']
      ),
      'unrelated_trace_fingerprint': (
          None if trace_fingerprints is None else trace_fingerprints['unrelated']
      ),
      'unrelated_repeat_trace_fingerprint': (
          None if trace_fingerprints is None
          else trace_fingerprints['unrelated_repeat']
      ),
      'rowwise_trace_fingerprints': rowwise,
      'raw_movement': raw_movement,
      'model_apply_count_through_record': apply_counter[0],
      'checks': checks,
      'failure': failure,
      'seconds': {
          'intended': intended_seconds,
          'intended_repeat': intended_repeat_seconds,
          'unrelated': unrelated_seconds,
          'unrelated_repeat': unrelated_repeat_seconds,
      },
      'created_at_unix_s': time.time(),
  }
  path = _artifact_path(recipient, anchor_id)
  digest = _write_new(path, artifact)
  return {
      'status': status,
      'recipient_order': recipient.order,
      'anchor_id': anchor_id,
      'checks': checks,
      'failure': failure,
      'artifact_binding': {
          'path': str(path.relative_to(OUTPUT_DIR)),
          'sha256': digest,
      },
  }


def _tree_digest(paths: Sequence[Path], root: Path) -> str:
  digest = hashlib.sha256()
  for path in sorted(paths):
    digest.update(str(path.relative_to(root)).encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(_sha256(path)))
  return digest.hexdigest()


def _raw_manifest() -> dict[str, Any]:
  raw_dir = OUTPUT_DIR / 'raw'
  paths = bootstrap._strict_file_tree(raw_dir) if raw_dir.exists() else []  # pylint: disable=protected-access
  if any(path.suffix != '.json' for path in paths):
    raise RuntimeError('Sidecar raw tree contains a non-JSON file.')
  return {
      'artifact_count': len(paths),
      'artifact_sha256': {
          str(path.relative_to(OUTPUT_DIR)): _sha256(path) for path in paths
      },
      'artifact_tree_sha256': _tree_digest(paths, OUTPUT_DIR),
  }


def _assert_attempt_budget(started_monotonic: float) -> None:
  if time.monotonic() - started_monotonic > MAX_WALL_TIME_SECONDS:
    raise RuntimeError('v3.3.2 frozen wall-time budget was exceeded.')
  output_size = sum(
      path.stat().st_size for path in OUTPUT_DIR.rglob('*') if path.is_file()
  )
  if output_size > MAX_OUTPUT_BYTES:
    raise RuntimeError('v3.3.2 frozen output-storage budget was exceeded.')


def _write_completion(
    *,
    stop_reason: str | None,
    message: str,
    results: Sequence[Mapping[str, Any]],
    apply_count: int,
    compiler: Mapping[str, Any],
    original_run_binding: Mapping[str, Any],
    v3_3_1_status: Mapping[str, Any],
) -> dict[str, Any]:
  raw = _raw_manifest()
  if raw['artifact_count'] != len(results):
    raise RuntimeError('Raw sidecar artifact count differs from result prefix.')
  if apply_count != 4 * len(results):
    raise RuntimeError('Controlled sidecar prefix does not have four applies/record.')
  _write_new(OUTPUT_DIR / 'RAW_MANIFEST.json', raw)
  import_phases = {
      'pre_model': _sha256(OUTPUT_DIR / 'IMPORT_PROVENANCE_PRE_MODEL.json'),
      'post_model_precompile': _sha256(
          OUTPUT_DIR / 'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json'
      ),
      'postcompile': _sha256(OUTPUT_DIR / 'IMPORT_PROVENANCE.json'),
  }
  complete_pairs = {
      (item['recipient_order'], item['anchor_id'])
      for item in results if item['status'] == 'complete'
  }
  expected_pairs = {
      (order, anchor) for order in RECIPIENT_ORDERS for anchor in ANCHOR_IDS
  }
  record = {
      'status': 'complete' if stop_reason is None else 'controlled_stop',
      'stop_reason': stop_reason,
      'message': message,
      'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION,
      'amendment_sha256': AMENDMENT_SHA256,
      'amendment_commit': AMENDMENT_COMMIT,
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'freeze_sha256': _sha256(FREEZE_PATH),
      'ood_anchor_record_count': len(results),
      'ood_invalid_count': sum(item['status'] != 'complete' for item in results),
      'unique_recipient_anchor_count': len({
          (item['recipient_order'], item['anchor_id']) for item in results
      }),
      'all_80_recipient_anchors_complete': complete_pairs == expected_pairs,
      'model_apply_count': apply_count,
      'expected_model_apply_count': EXPECTED_APPLY_COUNT,
      'eight_row_compile_count': 1,
      'six_row_compile_count': 0,
      'identity_rerun_count': 0,
      'main_cube_rerun_count': 0,
      'old_ood_records_reused': 0,
      'one_fixed_eight_row_executable': True,
      'eight_row_compiler': dict(compiler),
      'eight_row_executable_fingerprint': compiler[
          'executable_fingerprint'
      ],
      'graph_and_hlo_exact_to_original_v3_3': compiler[
          'graph_and_hlo_exact_to_original_v3_3'
      ],
      'id0_all20': sum(
          item['anchor_id'] == 0 and item['status'] == 'complete'
          and item['checks']['id0_all8_endpoint_exact_between_calls']
          for item in results
      ) == 20,
      'id255_all20': sum(
          item['anchor_id'] == 255 and item['status'] == 'complete'
          and item['checks']['id255_intended_endpoint_closure_exact']
          and item['checks']['id255_unrelated_endpoint_closure_exact']
          for item in results
      ) == 20,
      'invariant_rows_between_calls': list(INVARIANT_ROWS),
      'active_rows_have_no_forced_cross_call_predicate': True,
      'original_run_binding': dict(original_run_binding),
      'original_run_revalidated_in_full': True,
      'original_ood_records_provenance_only': True,
      'v3_3_1_status': dict(v3_3_1_status),
      'import_provenance_phases': import_phases,
      'import_provenance_sha256': import_phases['postcompile'],
      'protobuf_provenance_sha256': _sha256(
          OUTPUT_DIR / 'PROTOBUF_PROVENANCE.json'
      ),
      'raw_manifest': raw,
      'confirmation_model_calls': 0,
      'confirmation_scope_disclosure': (
          'Later-exon metadata/labels were exposed after protocol freeze; '
          'no later-exon model outputs, activations, or interventions are used.'
      ),
      'scientific_summary_computed': False,
      'shapley_or_nomination_computed': False,
      'completed_at_unix_s': time.time(),
  }
  _write_new(OUTPUT_DIR / 'RUN_COMPLETE.json', record)
  return record


def _controlled_stop(
    *,
    reason: str,
    message: str,
    results: Sequence[Mapping[str, Any]],
    apply_count: int,
    compiler: Mapping[str, Any],
    original_run_binding: Mapping[str, Any],
    v3_3_1_status: Mapping[str, Any],
) -> None:
  _write_completion(
      stop_reason=reason,
      message=message,
      results=results,
      apply_count=apply_count,
      compiler=compiler,
      original_run_binding=original_run_binding,
      v3_3_1_status=v3_3_1_status,
  )


def _compact_original_binding(
    original: Mapping[str, Any]
) -> dict[str, Any]:
  return {
      key: original[key] for key in bootstrap.EXPECTED_ORIGINAL_BINDING
  }


def _write_terminal_failure(
    error: Exception,
    *,
    completed_record_count: int,
    apply_count: int,
    compiler_created: bool,
) -> None:
  _write_new(OUTPUT_DIR / 'TERMINAL_FAILURE.json', {
      'status': 'terminal_failure',
      'type': type(error).__name__,
      'message': str(error),
      'traceback': ''.join(traceback.format_exception(error)),
      'completed_record_count': completed_record_count,
      'model_apply_count': apply_count,
      'eight_row_compile_count': int(compiler_created),
      'six_row_compile_count': 0,
      'identity_rerun_count': 0,
      'main_cube_rerun_count': 0,
      'confirmation_model_calls': 0,
      'created_at_unix_s': time.time(),
  })


def main() -> None:
  args = _parse_args()
  bootstrap_attestation = consume_bootstrap_attestation()
  for path in (
      OUTPUT_DIR,
      ANALYSIS_DIR,
      PREFLIGHT_DIR,
      AMENDMENT_PATH,
      FREEZE_PATH,
      ORIGINAL_FREEZE_PATH,
      ORIGINAL_RUN_DIR,
      v32.DEVELOPMENT_VARIANTS_PATH,
      v32.DEVELOPMENT_EXONS_PATH,
      v32.CHECKPOINT_MANIFEST_PATH,
      v32.REFERENCE_BINDINGS_PATH,
  ):
    _reject_confirmation_path(path)
  if args.successful_preflight is not None:
    _reject_confirmation_path(args.successful_preflight)
  if args.checkpoint is not None:
    _reject_confirmation_path(args.checkpoint)
  if _sha256(AMENDMENT_PATH) != AMENDMENT_SHA256:
    raise ValueError('v3.3.2 amendment hash changed.')
  if _sha256(ORIGINAL_FREEZE_PATH) != bootstrap.ORIGINAL_FREEZE_SHA256:
    raise ValueError('Original v3.3 freeze changed.')
  cases = v32.load_development_cases()
  execution_order = sidecar_execution_order(cases)
  if args.dry_run:
    print(json.dumps(build_dry_run_plan(
        cases,
        max_variants=args.max_variants,
        max_anchors=args.max_anchors,
    ), indent=2))
    return

  environment = v32.assert_v3_2_environment()
  frozen = json.loads(FREEZE_PATH.read_text(encoding='utf-8'))
  original_frozen = json.loads(
      ORIGINAL_FREEZE_PATH.read_text(encoding='utf-8')
  )
  original_run = bootstrap_attestation['original_run']
  # Re-read only the original raw hash map. The launcher already rehashed all
  # 5,158 files before any model import; no endpoint value is opened here.
  original_manifest = json.loads(
      (ORIGINAL_RUN_DIR / 'RAW_MANIFEST.json').read_text(encoding='utf-8')
  )
  if (
      original_manifest.get('artifact_count')
      != bootstrap.EXPECTED_ORIGINAL_BINDING['raw_artifact_count']
      or original_manifest.get('artifact_tree_sha256')
      != bootstrap.EXPECTED_ORIGINAL_BINDING['raw_artifact_tree_sha256']
  ):
    raise ValueError('Original raw-manifest linkage changed after bootstrap.')
  if args.successful_preflight is None:
    raise ValueError('v3.3.2 requires its fresh successful external preflight.')
  external = validate_external_preflight(args.successful_preflight, frozen)
  same_process = v33.v32.device_gate.collect_device_observation()
  same_process['packages'].update(v33.runtime_version_binding()['packages'])
  v33.v32.device_gate.validate_device_observation(same_process)
  v33.validate_device_version_manifest(same_process, original_frozen)
  if OUTPUT_DIR.exists() or ANALYSIS_DIR.exists():
    raise FileExistsError('v3.3.2 output/analysis exists; never resume or retry.')
  checkpoint = v32.v2._checkpoint_path(args.checkpoint)  # pylint: disable=protected-access
  _reject_confirmation_path(checkpoint)
  if checkpoint.name != v32.route_v3.CHECKPOINT_SNAPSHOT:
    raise ValueError('v3.3.2 checkpoint snapshot changed.')
  checkpoint_binding = v32.validate_checkpoint(checkpoint)
  reference_object_binding = v32.validate_reference_object()
  start = {
      'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION,
      'status': 'started_append_only_one_shot',
      'amendment': {
          'path': str(AMENDMENT_PATH.resolve()),
          'sha256': AMENDMENT_SHA256,
          'commit': AMENDMENT_COMMIT,
      },
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'freeze': frozen,
      'freeze_sha256': _sha256(FREEZE_PATH),
      'bootstrap': bootstrap_attestation,
      'external_preflight': external,
      'same_process_preflight': same_process,
      'runtime_environment': environment,
      'runtime_version_binding': v33.runtime_version_binding(),
      'checkpoint_path': str(checkpoint),
      'checkpoint_binding': checkpoint_binding,
      'reference_object_binding': reference_object_binding,
      'reference_sequence_bindings': {
          'path': str(v32.REFERENCE_BINDINGS_PATH.resolve()),
          'sha256': v32.REFERENCE_BINDINGS_SHA256,
      },
      'original_run_binding': _compact_original_binding(original_run),
      'original_run_revalidated_in_full': True,
      'v3_3_1_status': bootstrap_attestation['v3_3_1_status'],
      'record_count_contract': EXPECTED_RECORD_COUNT,
      'model_apply_count_contract': EXPECTED_APPLY_COUNT,
      'compile_count_contract': {'eight_row': 1, 'six_row': 0},
      'rerun_count_contract': {'identity': 0, 'main_cube': 0},
      'execution_order_contract': {
          'recipient_orders': list(RECIPIENT_ORDERS),
          'anchor_ids': list(ANCHOR_IDS),
          'major': 'recipient',
          'minor': 'anchor',
          'indices': [0, 79],
      },
      'invariant_rows_between_calls': list(INVARIANT_ROWS),
      'active_recipient_rows_without_cross_call_predicate': list(
          ACTIVE_RECIPIENT_ROWS
      ),
      'max_wall_time_seconds': MAX_WALL_TIME_SECONDS,
      'max_output_bytes': MAX_OUTPUT_BYTES,
      'confirmation_model_calls': 0,
      'confirmation_scope_disclosure': (
          'Later-exon metadata/labels were exposed after protocol freeze; '
          'no later-exon model outputs, activations, or interventions are used.'
      ),
      'started_at_unix_s': time.time(),
  }
  _write_new(START_PATH, start)
  results: list[dict[str, Any]] = []
  apply_counter = [0]
  attempt_started_monotonic = time.monotonic()
  compiler: dict[str, Any] | None = None
  try:
    _write_new(OUTPUT_DIR / 'PROTOBUF_PROVENANCE.json', original_frozen[
        'protobuf_binding'
    ])
    imports_pre_model = import_provenance(original_frozen)
    _write_new(
        OUTPUT_DIR / 'IMPORT_PROVENANCE_PRE_MODEL.json', imports_pre_model
    )
    model_instance = dna_model.create(
        checkpoint,
        model_settings=dna_model.ModelSettings(
            attention_backend=v32.route_v3.ATTENTION_BACKEND
        ),
    )
    params = model_instance._params  # pylint: disable=protected-access
    state = model_instance._state  # pylint: disable=protected-access
    common_by_order = {}
    for case in cases:
      interval, position_sets, selection, target, resolved, dna, sequence_sha = (
          v32._case_inputs(model_instance, case)  # pylint: disable=protected-access
      )
      del interval, position_sets, resolved
      common_by_order[case.order] = (dna, selection, target, sequence_sha)
    prototype_common = common_by_order[0]
    donor_prototype_common = common_by_order[v33.OOD_DONOR_ORDER[0]]
    prototype_selection = prototype_common[1]
    prototype_target = prototype_common[2]
    prototype_dna = _eight_row_batch(
        prototype_common, donor_prototype_common
    )
    prototype_interventions = v33.eight_row_interventions(
        prototype_selection, 0, unrelated=False
    )
    unrelated_prototype_interventions = v33.eight_row_interventions(
        prototype_selection, 0, unrelated=True
    )
    signatures = {
        'selection': v32.pytree_signature(prototype_selection),
        'target': v32.pytree_signature(prototype_target),
        'eight_interventions': v32.pytree_signature(
            prototype_interventions
        ),
    }
    v32.assert_same_program_signature(
        signatures['eight_interventions'], unrelated_prototype_interventions
    )
    raw_apply = (
        dna_model
        .create_splice_classification_logit_margin_eight_row_superset_graph_apply(
            model_instance._metadata,  # pylint: disable=protected-access
            attention_backend=v32.route_v3.ATTENTION_BACKEND,
        )
    )
    imports_post_model = import_provenance(original_frozen)
    v32.assert_import_provenance_stable(
        imports_pre_model, imports_post_model
    )
    _write_new(
        OUTPUT_DIR / 'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
        imports_post_model,
    )
    prototype_args = (
        params,
        state,
        prototype_dna,
        jnp.zeros((8,), jnp.int32),
        prototype_selection,
        prototype_interventions,
        prototype_target,
    )
    compile_start = time.perf_counter()
    lowered = jax.jit(raw_apply).lower(*prototype_args)
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - compile_start
    original_compiler = bootstrap_attestation['original_eight_row_compiler']
    compiler = _compiler_artifacts(
        lowered,
        compiled,
        compile_seconds,
        original_compiler,
        signatures,
    )
    imports = import_provenance(original_frozen)
    v32.assert_import_provenance_stable(imports_post_model, imports)
    _write_new(OUTPUT_DIR / 'IMPORT_PROVENANCE.json', imports)
    if not compiler['graph_and_hlo_exact_to_original_v3_3']:
      _controlled_stop(
          reason='compiler_graph_mismatch',
          message='New eight-row graph/HLO differs from frozen v3.3.',
          results=results,
          apply_count=apply_counter[0],
          compiler=compiler,
          original_run_binding=_compact_original_binding(original_run),
          v3_3_1_status=bootstrap_attestation['v3_3_1_status'],
      )
      return

    cases_by_order = {case.order: case for case in cases}
    for execution_index, (order, anchor_id) in enumerate(execution_order):
      recipient = cases_by_order[order]
      donor = cases_by_order[v33.OOD_DONOR_ORDER[order]]
      result = _run_anchor(
          compiled,
          recipient,
          donor,
          common_by_order[order],
          common_by_order[donor.order],
          params,
          state,
          anchor_id,
          signatures,
          _sha256(FREEZE_PATH),
          compiler['executable_fingerprint'],
          execution_index,
          original_manifest,
          apply_counter,
      )
      results.append(result)
      _assert_attempt_budget(attempt_started_monotonic)
      if result['status'] != 'complete':
        _controlled_stop(
            reason='ood_tooling_failure',
            message=(
                f'OOD sidecar audit failed at order={order}, '
                f'anchor_id={anchor_id}.'
            ),
            results=results,
            apply_count=apply_counter[0],
            compiler=compiler,
            original_run_binding=_compact_original_binding(original_run),
            v3_3_1_status=bootstrap_attestation['v3_3_1_status'],
        )
        return

    if len(results) != EXPECTED_RECORD_COUNT:
      raise RuntimeError('v3.3.2 did not execute exactly 80 records.')
    _write_completion(
        stop_reason=None,
        message='All 80 frozen v3.3.2 OOD sidecar records completed.',
        results=results,
        apply_count=apply_counter[0],
        compiler=compiler,
        original_run_binding=_compact_original_binding(original_run),
        v3_3_1_status=bootstrap_attestation['v3_3_1_status'],
    )
  except Exception as error:
    _write_terminal_failure(
        error,
        completed_record_count=len(results),
        apply_count=apply_counter[0],
        compiler_created=compiler is not None,
    )
    raise


if __name__ == '__main__':
  main()
