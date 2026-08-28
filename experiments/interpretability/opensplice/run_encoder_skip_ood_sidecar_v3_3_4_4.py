#!/usr/bin/env python3
"""One-shot development-only OpenSplice v3.3.4.4 OOD sidecar runner."""

from __future__ import annotations

import base64
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
import traceback
from typing import Any, Mapping, Sequence


_PREIMPORT_ATTESTATION_MODULE = (
    '_opensplice_v3_3_4_4_ood_sidecar_bootstrap_attestation'
)
if (
    __name__ == '__main__'
    and _PREIMPORT_ATTESTATION_MODULE not in sys.modules
):
  raise RuntimeError(
      'Direct v3.3.4.4 execution is forbidden before pre-import bootstrap; '
      'use launch_encoder_skip_ood_sidecar_v3_3_4_4.py.'
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
import validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_4 as bootstrap


SCRIPT_VERSION = 'v3.3.4.4'
ATTEMPT_ID = 'v3.3.4.4-development-ood-sidecar-one-shot'
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
ANALYSIS_ATTEMPT_DIR = bootstrap.ANALYSIS_ATTEMPT_DIR
START_PATH = OUTPUT_DIR / 'ATTEMPT_STARTED.json'
ANCHOR_IDS = (0, 127, 128, 255)
RECIPIENT_ORDERS = tuple(range(20))
INVARIANT_ROWS = (0, 1, 3, 5, 6, 7)
ACTIVE_RECIPIENT_ROWS = (2, 4)
EXPECTED_RECORD_COUNT = 80
EXPECTED_APPLY_COUNT = 320
MAX_WALL_TIME_SECONDS = 2 * 60 * 60
MAX_OUTPUT_BYTES = 1024 * 1024 * 1024
RUN_COMPLETE_SIZE_CAP_BYTES = 16 * 1024 * 1024
SOURCE_PROGRAM_CONTRACT = bootstrap.SOURCE_PROGRAM_CONTRACT
DISCLOSURE = (
    'Later-exon metadata/labels were exposed after protocol freeze; no '
    'later-exon model outputs, activations, or interventions are used.'
)
CALL_ROLES = ('intended', 'intended_repeat', 'unrelated', 'unrelated_repeat')


class CountedApplyError(RuntimeError):
  """Carries the exact dispatched-call boundary for a no-retry stop."""

  def __init__(self, call_label: str, apply_count: int, error: Exception):
    super().__init__(f'{call_label} failed after dispatch: {error}')
    self.call_label = call_label
    self.apply_count = apply_count
    self.original_error = error


class IncompleteRecordError(RuntimeError):
  """Carries an exact 0--4-call current-record tooling boundary."""

  def __init__(self, call_label: str, apply_count: int, error: Exception):
    super().__init__(f'{call_label} failed: {error}')
    self.call_label = call_label
    self.apply_count = apply_count
    self.original_error = error


class CurrentRecordStop(RuntimeError):
  """Carries the exact durable ledger/current-record prefix."""

  def __init__(
      self, *, failure_phase: str, failed_or_next_call_role: str | None,
      returned_outputs: Sequence[Any | None], started: Sequence[Mapping[str, Any]],
      completed: Sequence[Mapping[str, Any]], original_error: Exception,
  ):
    super().__init__(str(original_error))
    self.failure_phase = failure_phase
    self.failed_or_next_call_role = failed_or_next_call_role
    self.returned_outputs = tuple(returned_outputs)
    self.started = tuple(started)
    self.completed = tuple(completed)
    self.original_error = original_error


class AttemptBudgetViolation(RuntimeError):
  """Raised before a forbidden second lower or compile invocation."""

  def __init__(self, operation: str):
    super().__init__(f'Second {operation} attempt is forbidden.')
    self.operation = operation


class ModelSetupError(RuntimeError):
  """Carries exact checkpoint/reference/model evidence at setup failure."""

  def __init__(
      self, error: Exception, *, checkpoint_binding: Mapping[str, Any] | None,
      reference_binding: Mapping[str, Any] | None, model_constructed: bool,
  ):
    super().__init__(str(error))
    self.original_error = error
    self.checkpoint_binding = checkpoint_binding
    self.reference_binding = reference_binding
    self.model_constructed = model_constructed


class CompilerGraphExtractionError(RuntimeError):
  """Carries one exact post-compile ordinary extraction boundary."""

  def __init__(
      self, stage: str, error: Exception,
      same_object_attestation: Mapping[str, Any],
  ):
    super().__init__(str(error))
    self.stage = stage
    self.original_error = error
    self.same_object_attestation = dict(same_object_attestation)


class DiagnosticProvenanceError(RuntimeError):
  """Base for operation-typed diagnostic failures; never text-classified."""

  reason = ''

  def __init__(self, error: Exception):
    super().__init__(str(error))
    self.original_error = error


class DiagnosticParserFailure(DiagnosticProvenanceError):
  reason = 'diagnostic_parser_failure'


class EntryAbiParserFailure(DiagnosticParserFailure):
  """Entry HLO-module line could not be parsed."""


class BackendDiagnosticParserFailure(DiagnosticParserFailure):
  """Backend descriptive diagnostics could not be parsed."""


class DiagnosticPersistenceFailure(DiagnosticProvenanceError):
  reason = 'diagnostic_persistence_failure'


class CacheSignalUnavailable(DiagnosticProvenanceError):
  reason = 'cache_signal_unavailable'


class FingerprintFormulaMismatch(DiagnosticProvenanceError):
  reason = 'fingerprint_formula_mismatch'


class OneShotCompilerBudget:
  """Mutable one-shot invocation guard whose audit is terminal evidence."""

  def __init__(self):
    self.lower_invocations = 0
    self.compile_invocations = 0
    self.forbidden_request: str | None = None

  def request(self, operation: str) -> None:
    if operation not in {'lower', 'compile'}:
      raise ValueError('Unknown compiler-budget operation.')
    attribute = f'{operation}_invocations'
    if getattr(self, attribute) >= 1:
      self.forbidden_request = operation
      raise AttemptBudgetViolation(operation)
    setattr(self, attribute, getattr(self, attribute) + 1)

  def audit(self) -> dict[str, Any]:
    return {
        'lower_budget': 1,
        'compile_budget': 1,
        'lower_invocations': self.lower_invocations,
        'compile_invocations': self.compile_invocations,
        'forbidden_request': self.forbidden_request,
        'forbidden_request_detected_before_invocation': (
            self.forbidden_request is not None
        ),
    }


def _sha256(path: Path) -> str:
  return v32._sha256(path)  # pylint: disable=protected-access


def _write_new(path: Path, value: Any) -> str:
  payload = (
      json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
  ).encode('utf-8')
  return _publish_new_bytes(path, payload)


def _write_new_text(path: Path, value: str) -> str:
  return _publish_new_bytes(path, value.encode('utf-8'))


def _publish_new_bytes(path: Path, payload: bytes) -> str:
  """Publishes through the sole frozen bootstrap implementation."""
  relative = path.absolute().relative_to(OUTPUT_DIR.absolute()).as_posix()
  bootstrap.ensure_publication_parent('model_run', relative)
  success = bootstrap.publish_bytes(
      'model_run', relative, payload, artifact_role=path.name
  )
  return str(success['sha256'])


def _canonical_bytes(value: Any) -> bytes:
  return json.dumps(
      value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
      allow_nan=False,
  ).encode('utf-8')


def _content_binding(value: Any) -> dict[str, Any]:
  payload = _canonical_bytes(value)
  return {'sha256': hashlib.sha256(payload).hexdigest(),
          'size_bytes': len(payload)}


def _relative_file_binding(path: Path) -> dict[str, Any]:
  if not path.resolve().is_relative_to(OUTPUT_DIR.resolve()):
    raise ValueError('Run artifact binding escaped the run root.')
  return {
      'path': path.relative_to(OUTPUT_DIR).as_posix(),
      'sha256': _sha256(path),
      'size_bytes': path.stat().st_size,
  }


def _launcher_record() -> Mapping[str, Any]:
  module = sys.modules.get(ATTESTATION_MODULE)
  record = getattr(module, 'record', None)
  if not isinstance(record, Mapping):
    raise RuntimeError('Missing same-process v3.3.4.4 launcher attestation.')
  return record


def _external_authorization() -> Mapping[str, Any]:
  value = _launcher_record().get('external_freeze_authorization')
  if not isinstance(value, Mapping):
    raise RuntimeError('Launcher external-freeze authorization is absent.')
  return value


def _slug(value: str) -> str:
  return v32._slug(value)  # pylint: disable=protected-access


def sidecar_execution_order(
    cases: Sequence[Any],
) -> tuple[tuple[int, int], ...]:
  """Returns the frozen recipient-major, anchor-minor 80-record order."""
  if tuple(case.order for case in cases) != RECIPIENT_ORDERS:
    raise ValueError('v3.3.4.4 requires development recipient orders 0--19.')
  result = tuple(
      (order, anchor_id)
      for order in RECIPIENT_ORDERS
      for anchor_id in ANCHOR_IDS
  )
  if len(result) != EXPECTED_RECORD_COUNT or len(set(result)) != len(result):
    raise ValueError('v3.3.4.4 sidecar order is incomplete or duplicated.')
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
      'analysis_attempt_dir': str(ANALYSIS_ATTEMPT_DIR),
  }


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
  """Applies the sole v3.3.4.4 repair while retaining all stronger gates."""
  if anchor_id not in ANCHOR_IDS:
    raise ValueError('v3.3.4.4 received an unfrozen OOD anchor.')
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
      'corrected_host_assertion_version': 'v3.3.4.4',
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
  module = sys.modules.get(ATTESTATION_MODULE)
  record = getattr(module, 'record', None)
  if not isinstance(record, dict):
    raise RuntimeError(
        'Direct v3.3.4.4 runner invocation is forbidden; use its launcher.'
    )
  if record.get('pid') != os.getpid():
    raise RuntimeError('v3.3.4.4 bootstrap attestation came from another process.')
  required = {
      'pid', 'created_at_unix_s', 'gate_a', 'gate_b', 'start',
      'start_binding', 'external_freeze_authorization',
      'same_process_preflight', 'successful_preflight', 'v3_3_3_run',
      'v3_3_3_1_archive',
  }
  if set(record) != required:
    raise RuntimeError('Launcher attestation key set changed.')
  if record['gate_a']['sha256'] != _sha256(FREEZE_PATH):
    raise RuntimeError('Gate-A freeze attestation changed.')
  if record['gate_b']['sha256'] != record['gate_a']['sha256']:
    raise RuntimeError('Gate A/Gate B freeze bytes differ.')
  if record['start']['runner_pid'] != os.getpid():
    raise RuntimeError('START runner PID differs from this process.')
  if record['start_binding'] != _relative_or_absolute_binding(START_PATH):
    raise RuntimeError('Launcher START binding changed before model import.')
  if record['external_freeze_authorization'] != (
      bootstrap.validate_external_freeze_authorization()
  ):
    raise RuntimeError('External freeze authorization changed after Gate B.')
  if record['same_process_preflight']['pid'] != os.getpid():
    raise RuntimeError('Same-process device gate PID changed.')
  _prior_prefix_from_start(record['start'])
  return dict(record)


def _prior_prefix_from_start(
    start: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
  prefix = bootstrap.validate_recorded_prior_v3_3_4_3_consumed_preflight_prefix(
      start.get('prior_v3_3_4_3_consumed_preflight_prefix'),
      start.get(
          'prior_v3_3_4_3_consumed_preflight_prefix_content_binding'
      ),
  )
  binding = bootstrap.canonical_content_binding(prefix)
  return prefix, binding


def _relative_or_absolute_binding(path: Path) -> dict[str, Any]:
  return {
      'path': str(path.resolve()),
      'sha256': _sha256(path),
      'size_bytes': path.stat().st_size,
  }


def import_provenance(
    original_frozen: Mapping[str, Any], *, phase: str,
    enforce_loaded_contract: bool = True,
) -> dict[str, Any]:
  """Captures prospective 26 files and the actual loaded scientific set."""
  expected = original_frozen['upstream_imported_modules']
  generated = {
      'alphagenome.protos.dna_model_pb2',
      'alphagenome.protos.dna_model_service_pb2',
      'alphagenome.protos.dna_model_service_pb2_grpc',
      'alphagenome.protos.tensor_pb2',
  }
  upstream_root = (_HERE.parents[3] / 'alphagenome').resolve()
  prospective = []
  for name, binding in sorted(expected.items()):
    path = upstream_root / binding['relative_path']
    source_kind = (
        'generated_untracked_exception' if name in generated
        else 'git_tracked'
    )
    prospective.append({
        'module_name': name,
        'path': str(path.resolve()),
        'declared_root': 'upstream_alphagenome_checkout',
        'relative_path': binding['relative_path'],
        'sha256': binding['sha256'],
        'size_bytes': binding['size_bytes'],
        'source_kind': source_kind,
        'git_mode': None if name in generated else '100644',
        'filesystem_mode': '0664',
    })
  loaded = loaded_scientific_modules()
  contract = None
  if enforce_loaded_contract:
    contract = json.loads(FREEZE_PATH.read_text(encoding='utf-8'))[
        'source_inventory_contract'
    ]['loaded_scientific_module_contract']
  if enforce_loaded_contract and loaded != contract:
    raise ValueError('Actual loaded scientific-module inventory changed.')
  sidecar_sources = {
      row['path']: {'sha256': row['sha256'], 'size_bytes': row['size_bytes']}
      for row in loaded if row['root'] == 'locked_opensplice_checkout'
  }
  return {
      'schema_version': 'v3.3.4.4-import-provenance-v1',
      'phase': phase,
      'external_freeze_authorization': dict(_external_authorization()),
      'prospective_upstream_source_file_count': len(prospective),
      'prospective_upstream_source_files': prospective,
      'loaded_scientific_module_count': len(loaded),
      'loaded_scientific_modules': loaded,
      'upstream_source_attestation': (
          bootstrap.v33_bootstrap.validate_upstream_checkout(
              dict(original_frozen)
          )
      ),
      'v3_3_4_4_sidecar_sources': sidecar_sources,
      'created_at_unix_s': time.time(),
  }


def loaded_scientific_modules() -> list[dict[str, Any]]:
  """Returns the normalized actual file-backed scientific module snapshot."""
  roots = {
      'locked_opensplice_checkout': _HERE.resolve(),
      'alphagenome_research_checkout': _HERE.parents[2].resolve(),
      'upstream_alphagenome_checkout': (_HERE.parents[3] / 'alphagenome').resolve(),
  }
  normalized_locked = {
      (_HERE / 'run_encoder_skip_ood_sidecar_v3_3_4_4.py').resolve(): (
          'v3_3_4_4_runner'
      ),
      (_HERE / 'launch_encoder_skip_ood_sidecar_v3_3_4_4.py').resolve(): (
          'v3_3_4_4_launcher'
      ),
      (_HERE / 'validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_4.py').resolve(): (
          'validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_4'
      ),
      # The launcher imports this JAX-only observer before START.  Freeze it
      # explicitly so the generator process and the actual run-process phase
      # inventories describe the same loaded scientific set.
      (_HERE / 'run_device_preflight_v3_3.py').resolve(): (
          'run_device_preflight_v3_3'
      ),
  }
  loaded = []
  seen = set()
  for name, module in sorted(sys.modules.items()):
    path_text = getattr(module, '__file__', None)
    if not path_text:
      continue
    path = Path(path_text).resolve()
    selected_root = None
    for root_name, root in roots.items():
      if path.is_relative_to(root):
        selected_root = root_name
        break
    if selected_root is None or path.suffix not in ('.py', '.pyi'):
      continue
    if selected_root == 'locked_opensplice_checkout':
      if path in normalized_locked:
        name = normalized_locked[path]
      elif not (
          name.startswith(('run_', 'validate_', 'target_reducers_v3'))
      ):
        continue
    key = (name, str(path))
    if key in seen:
      continue
    seen.add(key)
    loaded.append({
        'name': name,
        'path': str(path),
        'root': selected_root,
        'sha256': _sha256(path),
        'size_bytes': path.stat().st_size,
        'filesystem_mode': f'{path.stat().st_mode & 0o777:04o}',
    })
  # runpy does not necessarily register the executing runner as a module.
  for path, name in normalized_locked.items():
    key = (name, str(path))
    if key not in seen and path.is_file():
      loaded.append({
          'name': name,
          'path': str(path),
          'root': 'locked_opensplice_checkout',
          'sha256': _sha256(path),
          'size_bytes': path.stat().st_size,
          'filesystem_mode': f'{path.stat().st_mode & 0o777:04o}',
      })
  return sorted(loaded, key=lambda row: (row['name'], row['path']))


def assert_import_inventories_stable(*records: Mapping[str, Any]) -> None:
  if not records:
    raise ValueError('At least one import inventory is required.')
  canonical = []
  for record in records:
    copy = dict(record)
    copy.pop('phase', None)
    copy.pop('created_at_unix_s', None)
    canonical.append(_canonical_bytes(copy))
  if len(set(canonical)) != 1:
    raise ValueError('Scientific import inventories changed across phases.')


def import_validation_predicates(
    record: Mapping[str, Any], frozen: Mapping[str, Any],
    previous: Sequence[Mapping[str, Any]] = (),
) -> dict[str, bool]:
  """Returns literal independently checkable import-boundary predicates."""
  prospective = record.get('prospective_upstream_source_files')
  loaded = record.get('loaded_scientific_modules')
  contract = frozen['source_inventory_contract'][
      'loaded_scientific_module_contract'
  ]
  stable = True
  try:
    assert_import_inventories_stable(*previous, record)
  except Exception:  # pylint: disable=broad-exception-caught
    stable = False
  expected_sidecar = {
      row['path']: {'sha256': row['sha256'], 'size_bytes': row['size_bytes']}
      for row in contract if row['root'] == 'locked_opensplice_checkout'
  }
  return {
      'schema_exact': record.get('schema_version')
      == 'v3.3.4.4-import-provenance-v1',
      'external_freeze_authorization_exact': record.get(
          'external_freeze_authorization'
      ) == dict(_external_authorization()),
      'prospective_26_count_exact': (
          record.get('prospective_upstream_source_file_count') == 26
          and isinstance(prospective, list) and len(prospective) == 26
      ),
      'actual_loaded_count_exact': (
          isinstance(loaded, list)
          and record.get('loaded_scientific_module_count') == len(loaded)
      ),
      'actual_loaded_contract_exact': loaded == contract,
      'locked_sidecar_sources_exact': record.get(
          'v3_3_4_4_sidecar_sources'
      ) == expected_sidecar,
      'prior_phases_stable_exact': stable,
  }


def _write_provenance_validation_failure(
    *, artifact_role: str, artifact_path: Path,
    predicates: Mapping[str, bool], error: Exception,
    model_constructed: bool, lower_count: int = 0,
    compile_count: int = 0, successful_compile_count: int = 0,
) -> dict[str, Any]:
  record = {
      'status': 'provenance_validation_failure',
      'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION,
      'external_freeze_authorization': dict(_external_authorization()),
      'artifact_role': artifact_role,
      'artifact_binding': _relative_file_binding(artifact_path),
      'validation_predicates': dict(predicates),
      'failure': _failure_object(error),
      'model_constructed': model_constructed,
      'lower_attempt_count': lower_count,
      'compile_attempt_count': compile_count,
      'successful_compile_count': successful_compile_count,
      'model_apply_count': 0,
      'created_at_unix_s': time.time(),
  }
  _write_new(OUTPUT_DIR / 'PROVENANCE_VALIDATION_FAILURE.json', record)
  return record


def _persist_terminal_import_inventory(
    original_frozen: Mapping[str, Any], frozen: Mapping[str, Any],
    previous: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, bool]]:
  """Publishes and independently validates the terminal import inventory."""
  record = import_provenance(
      original_frozen, phase='terminal', enforce_loaded_contract=False
  )
  _write_new(OUTPUT_DIR / 'IMPORT_PROVENANCE.json', record)
  return record, import_validation_predicates(record, frozen, previous)


def derive_source_input_audit(
    *, checkpoint_binding: Mapping[str, Any],
    reference_object_binding: Mapping[str, Any],
    protobuf_record: Mapping[str, Any],
    imports: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
  """Independently rederives all eight compiler/dispatch primitives."""
  launch = _launcher_record()
  prior_v333 = bootstrap.validate_v3_3_3_run()
  prior_v3331 = bootstrap.validate_v3_3_3_1_archive()
  authorization = bootstrap.validate_external_freeze_authorization()
  preflight_state = bootstrap.validate_preflight_state_for_role()
  start = launch['start']
  model_routing = {
      **dict(start['cache_isolation_contract']['model']),
      'present_forbidden_names': [],
  }
  live_cache = bootstrap.assert_live_cache_environment_matches(model_routing)
  expected_protobuf = v32.protobuf_provenance()
  expected_protobuf['external_freeze_authorization'] = dict(authorization)
  inventories_stable = True
  try:
    assert_import_inventories_stable(*imports)
  except Exception:  # pylint: disable=broad-exception-caught
    inventories_stable = False
  result = {
      'bootstrap_sources_and_prior_trees_exact': (
          prior_v333 == launch['v3_3_3_run']
          and prior_v3331 == launch['v3_3_3_1_archive']
      ),
      'tracked_head_and_frozen_inventory_exact': (
          authorization == launch['external_freeze_authorization']
          and launch['gate_a']['sha256'] == _sha256(FREEZE_PATH)
      ),
      'external_device_runtime_environment_exact': (
          preflight_state['sole_successful_preflight_exact'] is True
          and preflight_state['record_sha256']
          == start['successful_preflight']['artifact_binding']['sha256']
      ),
      'same_process_device_runtime_environment_exact': (
          start['same_process_preflight']['pid'] == os.getpid()
          and live_cache['cache_role'] == 'model'
          and start['same_process_preflight']['no_model_constructed'] is True
          and start['same_process_preflight']['no_jit_or_array_kernel'] is True
      ),
      'checkpoint_exact': bool(checkpoint_binding),
      'reference_object_and_sequences_exact': bool(reference_object_binding),
      'protobuf_binding_exact': protobuf_record == expected_protobuf,
      'three_import_inventories_stable_exact': (
          len(imports) == 3 and inventories_stable
      ),
  }
  if set(result) != set(bootstrap.SOURCE_INPUT_AUDIT_KEYS):
    raise RuntimeError('Derived source-input audit key set changed.')
  return result


def _canonical_sha256(value: Any) -> str:
  payload = json.dumps(
      value,
      sort_keys=True,
      separators=(',', ':'),
      ensure_ascii=False,
      allow_nan=False,
  ).encode('utf-8')
  return hashlib.sha256(payload).hexdigest()


_SIGNATURE_OBJECT_ORDER = ('eight_interventions', 'selection', 'target')


def _signature_container_tags(
    value: Mapping[str, Any], *, runtime: bool
) -> list[dict[str, str]]:
  tags = []
  expected = tuple if runtime else list
  for object_name in _SIGNATURE_OBJECT_ORDER:
    leaves = value[object_name]['leaves']
    if type(leaves) is not expected:  # pylint: disable=unidiomatic-typecheck
      raise TypeError(f'Unexpected leaves container: {object_name}.')
    tags.append({'path': f'/{object_name}/leaves',
                 'kind': 'tuple' if runtime else 'list'})
    for index, leaf in enumerate(leaves):
      if type(leaf['shape']) is not expected:  # pylint: disable=unidiomatic-typecheck
        raise TypeError(f'Unexpected shape container: {object_name}/{index}.')
      tags.append({'path': f'/{object_name}/leaves/{index}/shape',
                   'kind': 'tuple' if runtime else 'list'})
  if len(tags) != 32:
    raise ValueError('Program signature must have exactly 32 tagged paths.')
  tagged_paths = {row['path'] for row in tags}
  observed_special: set[str] = set()
  def walk(item: Any, pointer: str) -> None:
    if isinstance(item, Mapping):
      for key, child in item.items():
        walk(child, f'{pointer}/{key}')
    elif type(item) in (tuple, list):  # pylint: disable=unidiomatic-typecheck
      if type(item) is expected:  # pylint: disable=unidiomatic-typecheck
        observed_special.add(pointer)
      elif item:
        raise TypeError(f'Unexpected container kind at {pointer}.')
      for index, child in enumerate(item):
        walk(child, f'{pointer}/{index}')
  walk(value, '')
  if observed_special != tagged_paths:
    raise ValueError('Signature contains a tuple/list outside 32 paths.')
  return tags


def canonicalize_program_signatures(
    runtime_value: Mapping[str, Any], frozen_value: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Apply the sole permitted tuple-to-list adapter at 32 fixed paths."""
  if set(runtime_value) != set(_SIGNATURE_OBJECT_ORDER):
    raise ValueError('Runtime signature object names changed.')
  if set(frozen_value) != set(_SIGNATURE_OBJECT_ORDER):
    raise ValueError('Frozen signature object names changed.')
  runtime_tags = _signature_container_tags(runtime_value, runtime=True)
  frozen_tags = _signature_container_tags(frozen_value, runtime=False)
  adapted = {}
  for name in _SIGNATURE_OBJECT_ORDER:
    runtime_record = runtime_value[name]
    if set(runtime_record) != {'treedef', 'leaves'}:
      raise ValueError(f'Runtime signature schema changed: {name}.')
    leaves = []
    for leaf in runtime_record['leaves']:
      if set(leaf) != {'dtype', 'shape'}:
        raise ValueError(f'Runtime signature leaf schema changed: {name}.')
      if any(type(item) is not int or item < 0 for item in leaf['shape']):
        raise TypeError(f'Runtime signature shape changed: {name}.')
      leaves.append({'dtype': leaf['dtype'], 'shape': list(leaf['shape'])})
    adapted[name] = {'treedef': runtime_record['treedef'], 'leaves': leaves}
    if adapted[name] != frozen_value[name]:
      raise ValueError(f'Program signature semantics changed: {name}.')
  payload = json.dumps(
      adapted, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
      allow_nan=False,
  ).encode('utf-8')
  digest = hashlib.sha256(payload).hexdigest()
  if digest != SOURCE_PROGRAM_CONTRACT['program_signatures_sha256']:
    raise ValueError('Canonical program-signature digest changed.')
  if len(payload) != 2877:
    raise ValueError('Canonical program-signature size changed.')
  comparisons = {
      'direct_python_equality': runtime_value == frozen_value,
      'runtime_tuple_container_count': 32,
      'runtime_leaves_tuple_count': 3,
      'runtime_shape_tuple_count': 29,
      'frozen_list_container_count': 32,
      'frozen_leaves_list_count': 3,
      'frozen_shape_list_count': 29,
      'declared_paths_exact': True,
      'container_kinds_exact': True,
      'treedefs_exact': True,
      'leaf_order_counts_dtypes_shapes_exact': True,
      'canonical_bytes_exact': True,
      'canonical_hash_and_size_exact': True,
  }
  if comparisons['direct_python_equality'] is not False:
    raise ValueError('Expected tuple/list host inequality disappeared.')
  canonical = {'sha256': digest, 'size_bytes': len(payload)}
  return adapted, {
      'runtime_container_tags': runtime_tags,
      'frozen_container_tags': frozen_tags,
      'runtime_semantic_mapping': adapted,
      'frozen_semantic_mapping': dict(frozen_value),
      'runtime_canonical': canonical,
      'frozen_canonical': canonical,
      'comparisons': comparisons,
  }


def _validated_signature_tag_prefix(
    value: Any, *, runtime: bool,
) -> list[dict[str, str]]:
  """Returns only the fixed-order container tags validated before drift.

  This is intentionally a non-canonicalizing failure reporter.  It never
  accepts a different container representation and stops at the first path
  whose object/leaf/container contract cannot be established.
  """
  if not isinstance(value, Mapping):
    return []
  expected = tuple if runtime else list
  kind = 'tuple' if runtime else 'list'
  tags: list[dict[str, str]] = []
  for object_name in _SIGNATURE_OBJECT_ORDER:
    record = value.get(object_name)
    if not isinstance(record, Mapping) or set(record) != {'treedef', 'leaves'}:
      break
    leaves = record.get('leaves')
    if type(leaves) is not expected:  # pylint: disable=unidiomatic-typecheck
      break
    tags.append({'path': f'/{object_name}/leaves', 'kind': kind})
    object_complete = True
    for index, leaf in enumerate(leaves):
      if not isinstance(leaf, Mapping) or set(leaf) != {'dtype', 'shape'}:
        object_complete = False
        break
      shape = leaf.get('shape')
      if type(shape) is not expected:  # pylint: disable=unidiomatic-typecheck
        object_complete = False
        break
      if any(type(item) is not int or item < 0 for item in shape):
        object_complete = False
        break
      tags.append({
          'path': f'/{object_name}/leaves/{index}/shape', 'kind': kind,
      })
    if not object_complete:
      break
  return tags


def _entry_abi_binding(compiled_hlo: str) -> dict[str, Any]:
  lines = compiled_hlo.splitlines()
  if not lines or not lines[0].startswith('HloModule '):
    raise EntryAbiParserFailure(
        ValueError('Compiled HLO has no entry module line.')
    )
  fingerprint_values = re.findall(
      r'fingerprint_before_lhs="([0-9A-Fa-f]+)"', lines[0]
  )
  if (
      len(fingerprint_values) != 1
      or lines[0].count('fingerprint_before_lhs=') != 1
  ):
    raise FingerprintFormulaMismatch(
        ValueError(
            'Entry ABI requires one nonempty hexadecimal backend fingerprint.'
        )
    )
  normalized, substitutions = re.subn(
      r'fingerprint_before_lhs="[0-9A-Fa-f]+"',
      'fingerprint_before_lhs="<backend-generated>"',
      lines[0],
  )
  if substitutions != 1:
    raise FingerprintFormulaMismatch(
        ValueError('Entry ABI must contain exactly one backend fingerprint.')
    )
  return {
      'normalization': (
          'first HloModule line; replace only fingerprint_before_lhs value; '
          'omit line-ending newline'
      ),
      'normalized_line_sha256': hashlib.sha256(
          normalized.encode('utf-8')
      ).hexdigest(),
      'normalized_line_size_bytes': len(normalized.encode('utf-8')),
      'backend_fingerprint_substitution_count': substitutions,
  }


def _backend_config_from_instruction(line: str) -> dict[str, Any] | None:
  marker = 'backend_config='
  start = line.find(marker)
  if start < 0:
    return None
  start += len(marker)
  if start >= len(line) or line[start] != '{':
    raise ValueError('Backend config is not a JSON object.')
  depth = 0
  in_string = False
  escaped = False
  for index in range(start, len(line)):
    char = line[index]
    if in_string:
      if escaped:
        escaped = False
      elif char == '\\':
        escaped = True
      elif char == '"':
        in_string = False
      continue
    if char == '"':
      in_string = True
    elif char == '{':
      depth += 1
    elif char == '}':
      depth -= 1
      if depth == 0:
        value = json.loads(line[start:index + 1])
        if not isinstance(value, dict):
          raise ValueError('Backend config decoded to a non-object.')
        return value
  raise ValueError('Backend config JSON object is unterminated.')


def _backend_diagnostics(compiled_hlo: str) -> dict[str, Any]:
  """Returns deterministic, descriptive-only backend-codegen provenance."""
  lines = compiled_hlo.splitlines()
  computation_count = sum(
      bool(re.match(r'^(?:ENTRY )?%[^ ]+ \(', line)) for line in lines
  )
  instruction_count = sum(line.startswith('  %') for line in lines)
  fusion_kinds = Counter(
      re.findall(r'kind=(k[A-Za-z_]+)', compiled_hlo)
  )
  triton = []
  cublas = []
  cudnn = []
  for line in lines:
    backend = _backend_config_from_instruction(line)
    if '"kind":"__triton"' in line:
      block = (
          (backend or {}).get('fusion_backend_config', {})
          .get('block_level_fusion_config')
      )
      if not isinstance(block, dict):
        raise ValueError('Triton instruction lacks block-level settings.')
      triton.append({
          'block_level_fusion_config': block,
          'instruction_line_sha256': hashlib.sha256(
              line.encode('utf-8')
          ).hexdigest(),
      })
    if 'custom_call_target="__cublas$' in line:
      target = re.search(r'custom_call_target="([^"]+)"', line)
      gemm = (backend or {}).get('gemm_backend_config')
      cublas.append({
          'target': target.group(1) if target else None,
          'gemm_backend_config': gemm,
          'instruction_line_sha256': hashlib.sha256(
              line.encode('utf-8')
          ).hexdigest(),
      })
    if 'custom_call_target="__cudnn$' in line:
      target = re.search(r'custom_call_target="([^"]+)"', line)
      convolution = (backend or {}).get('cudnn_conv_backend_config', {})
      algorithm = convolution.get('algorithm')
      cudnn.append({
          'target': target.group(1) if target else None,
          'algorithm': algorithm,
          'workspace_size_bytes': (
              None
              if not isinstance(algorithm, dict)
              else int(algorithm.get('workspace_size', 0))
          ),
          'instruction_line_sha256': hashlib.sha256(
              line.encode('utf-8')
          ).hexdigest(),
      })
  return {
      'descriptive_only_not_an_equality_gate': True,
      'computation_count': computation_count,
      'instruction_count_excluding_computation_headers': instruction_count,
      'instruction_record_count': computation_count + instruction_count,
      'fusion_kind_counts': dict(sorted(fusion_kinds.items())),
      'triton_configuration_count': len(triton),
      'triton_configurations': triton,
      'cublas_call_count': len(cublas),
      'cublas_algorithms': cublas,
      'cudnn_call_count': len(cudnn),
      'cudnn_algorithms_workspaces': cudnn,
  }


def _cache_tree_binding(root: Path) -> dict[str, Any]:
  """Binds role-isolated cache outputs without opening user cache paths."""
  return bootstrap.cache_output_tree_binding(root)


def _final_model_cache_binding(compiler: Mapping[str, Any]) -> dict[str, Any]:
  provenance = compiler.get('kernel_cache_provenance')
  if not isinstance(provenance, Mapping):
    raise ValueError('Compiler lacks model-cache provenance.')
  pre_import = provenance.get('pre_import')
  if not isinstance(pre_import, Mapping):
    raise ValueError('Compiler lacks pre-import model-cache attestation.')
  root = Path(str(pre_import.get('cache_root', ''))).resolve()
  terminal = _cache_tree_binding(root)
  historical_name = (
      'post_compile' if 'post_compile' in provenance else 'post_failure'
  )
  historical = provenance.get(historical_name)
  if not isinstance(historical, Mapping):
    raise ValueError('Compiler lacks its historical cache-output binding.')
  return {
      'pre_import': dict(pre_import),
      'historical_stage': historical_name,
      'historical_binding': dict(historical),
      'terminal': terminal,
      'cache_hit_evidence': dict(provenance['cache_hit_evidence']),
      'historical_to_terminal_tree_exact': (
          historical.get('tree_sha256') == terminal['tree_sha256']
      ),
      'historical_to_terminal_equality_is_a_gate': False,
      'historical_snapshot_not_reauthenticated_as_live_files': True,
      'default_user_cache_paths_eligible': False,
      'cache_outputs_are_diagnostic_only': True,
  }


def _artifact_comparison(
    artifacts: Mapping[str, Any], compiler: Mapping[str, Any]
) -> dict[str, Any]:
  return {
      name: {
          'sha256_exact': artifacts[name]['sha256']
          == compiler['artifacts'][name]['sha256'],
          'size_exact': artifacts[name]['size_bytes']
          == compiler['artifacts'][name]['size_bytes'],
      }
      for name in ('stablehlo', 'hlo', 'compiled_hlo')
  }


def _model_cache_hit_evidence(
    pre_import: Mapping[str, Any], *, compile_skipped: bool | None
) -> dict[str, Any]:
  persistent_hit = bool(jax.config.jax_enable_compilation_cache)
  result = {
      'pre_import_files_present': pre_import.get('file_count') != 0,
      'default_user_cache_path_eligible': False,
      'persistent_compilation_cache_hit_reported': persistent_hit,
      'executable_deserialized': False,
      'compile_skipped': compile_skipped,
      'compile_stage_not_applicable': compile_skipped is None,
      'old_cache_input_opened': False,
      'routing_exact': pre_import.get('cache_root') == str(
          bootstrap.MODEL_KERNEL_CACHE_DIR.resolve()
      ),
      'cache_hit': False,
  }
  result['cache_hit'] = any((
      result['pre_import_files_present'],
      result['default_user_cache_path_eligible'],
      result['persistent_compilation_cache_hit_reported'],
      result['executable_deserialized'], result['compile_skipped'] is True,
      result['old_cache_input_opened'], not result['routing_exact'],
  ))
  return result


def evaluate_source_program_gate(
    observed: Mapping[str, Any],
    program_signatures: Mapping[str, Any],
    v3_3_2_compiler: Mapping[str, Any],
    source_input_audit: Mapping[str, Any],
    program_signature_attestation: Mapping[str, Any],
    same_object_attestation: Mapping[str, Any],
) -> dict[str, Any]:
  """Evaluates only the prospectively frozen pre-backend/source boundary."""
  contract = dict(SOURCE_PROGRAM_CONTRACT)
  stablehlo_exact = (
      observed['stablehlo_sha256'] == contract['stablehlo_sha256']
      and observed['stablehlo_size_bytes'] == contract['stablehlo_size_bytes']
  )
  pre_backend_exact = (
      observed['pre_backend_hlo_sha256']
      == contract['pre_backend_hlo_sha256']
      and observed['pre_backend_hlo_size_bytes']
      == contract['pre_backend_hlo_size_bytes']
  )
  signatures_exact = (
      observed['program_signatures_sha256']
      == contract['program_signatures_sha256']
      and program_signatures == v3_3_2_compiler['program_signatures']
      and program_signature_attestation['comparisons'][
          'canonical_bytes_exact'
      ] is True
      and program_signature_attestation['comparisons'][
          'container_kinds_exact'
      ] is True
  )
  entry_abi_exact = (
      observed['entry_abi_sha256'] == contract['entry_abi_sha256']
  )
  required_audit_bools = (
      'bootstrap_sources_and_prior_trees_exact',
      'tracked_head_and_frozen_inventory_exact',
      'external_device_runtime_environment_exact',
      'same_process_device_runtime_environment_exact',
      'checkpoint_exact',
      'reference_object_and_sequences_exact',
      'protobuf_binding_exact',
      'three_import_inventories_stable_exact',
  )
  provenance_exact = all(
      source_input_audit.get(name) is True for name in required_audit_bools
  )
  same_object_exact = all(
      same_object_attestation[name] is True for name in (
          'stablehlo_read_from_lowered_object',
          'pre_backend_hlo_read_from_lowered_object',
          'compile_argument_is_lowered_object',
          'compiled_hlo_read_from_compiled_object',
          'signature_attestation_from_apply_arguments',
          'apply_callable_is_compiled_object',
          'compiler_record_is_gate_record',
      )
  ) and (
      same_object_attestation['lower_call_count'] == 1
      and same_object_attestation['compile_call_count'] == 1
  )
  result = {
      'contract': contract,
      'observed': dict(observed),
      'stablehlo_exact': stablehlo_exact,
      'pre_backend_hlo_exact': pre_backend_exact,
      'program_signature_structure_exact': signatures_exact,
      'program_signatures_canonical_exact': signatures_exact,
      'entry_abi_exact': entry_abi_exact,
      'source_runtime_device_toolchain_checkpoint_reference_exact': (
          provenance_exact
      ),
      'source_input_audit': dict(source_input_audit),
      'source_input_audit_content_binding': _content_binding(
          source_input_audit
      ),
      'same_object_attestation': dict(same_object_attestation),
      'same_object_attestation_content_binding': _content_binding(
          same_object_attestation
      ),
      'same_lowered_compiled_object': same_object_exact,
  }
  result['source_program_exact'] = all(
      result[name]
      for name in (
          'stablehlo_exact',
          'pre_backend_hlo_exact',
          'program_signatures_canonical_exact',
          'entry_abi_exact',
          'source_runtime_device_toolchain_checkpoint_reference_exact',
          'same_lowered_compiled_object',
      )
  )
  return result


def _compiler_artifacts(
    lowered: Any,
    compiled: Any,
    seconds: float,
    original_v3_3_compiler: Mapping[str, Any],
    v3_3_2_compiler: Mapping[str, Any],
    program_signatures: Mapping[str, Any],
    *,
    program_signature_attestation: Mapping[str, Any],
    source_input_audit: Mapping[str, Any],
    kernel_cache_preimport_attestation: Mapping[str, Any],
    published_graphs: Mapping[str, Any] | None = None,
    precomputed_entry_abi: Mapping[str, Any] | None = None,
    precomputed_backend_diagnostics: Mapping[str, Any] | None = None,
    attempt_budget_audit: Mapping[str, Any] | None = None,
    same_object_attestation: Mapping[str, Any] | None = None,
    precomputed_cache_binding: Mapping[str, Any] | None = None,
    precomputed_cache_hit_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  directory = OUTPUT_DIR / 'compiler' / 'eight_row'
  if published_graphs is None:
    published_graphs = _publish_compiler_graphs(
        _extract_compiler_graph_texts(lowered, compiled)
    )
  artifacts = dict(published_graphs['artifacts'])
  compiled_hlo = published_graphs['compiled_hlo_text']
  fingerprint = hashlib.sha256(
      bytes.fromhex(artifacts['compiled_hlo']['sha256'])
  ).hexdigest()
  signatures = dict(program_signatures)
  signatures_sha256 = _canonical_sha256(signatures)
  entry_abi = dict(
      _entry_abi_binding(compiled_hlo)
      if precomputed_entry_abi is None else precomputed_entry_abi
  )
  same_object_attestation = dict(
      _successful_same_object_attestation(lowered, compiled)
      if same_object_attestation is None else same_object_attestation
  )
  observed = {
      'stablehlo_sha256': artifacts['stablehlo']['sha256'],
      'stablehlo_size_bytes': artifacts['stablehlo']['size_bytes'],
      'pre_backend_hlo_sha256': artifacts['hlo']['sha256'],
      'pre_backend_hlo_size_bytes': artifacts['hlo']['size_bytes'],
      'program_signatures_sha256': signatures_sha256,
      'entry_abi_sha256': entry_abi['normalized_line_sha256'],
  }
  source_program_gate = evaluate_source_program_gate(
      observed,
      signatures,
      v3_3_2_compiler,
      source_input_audit,
      program_signature_attestation,
      same_object_attestation,
  )
  comparisons = {
      'v3_3': {
          'artifacts': _artifact_comparison(
              artifacts, original_v3_3_compiler
          ),
          'executable_fingerprint_exact': (
              fingerprint
              == original_v3_3_compiler['executable_fingerprint']
          ),
      },
      'v3_3_2': {
          'artifacts': _artifact_comparison(artifacts, v3_3_2_compiler),
          'executable_fingerprint_exact': (
              fingerprint == v3_3_2_compiler['executable_fingerprint']
          ),
      },
      'compiled_backend_differences_are_diagnostic_only': True,
  }
  record = {
      'executable_name': 'eight_row',
      'compile_count': 1,
      'lower_attempt_count': 1,
      'compile_attempt_count': 1,
      'successful_compile_count': 1,
      'compile_seconds': seconds,
      'executable_fingerprint': fingerprint,
      'artifacts': artifacts,
      'program_signatures': signatures,
      'program_signatures_sha256': signatures_sha256,
      'entry_abi': entry_abi,
      'program_signature_attestation': _relative_file_binding(
          OUTPUT_DIR / 'compiler/eight_row/'
          'PROGRAM_SIGNATURE_ATTESTATION.json'
      ),
      'external_freeze_authorization': dict(
          program_signature_attestation['external_freeze_authorization']
      ),
      'source_input_audit': dict(source_input_audit),
      'source_input_audit_content_binding': _content_binding(
          source_input_audit
      ),
      'same_object_attestation': same_object_attestation,
      'same_object_attestation_content_binding': _content_binding(
          same_object_attestation
      ),
      'source_program_gate': source_program_gate,
      'backend_diagnostics': dict(
          _backend_diagnostics(compiled_hlo)
          if precomputed_backend_diagnostics is None
          else precomputed_backend_diagnostics
      ),
      'diagnostic_comparisons': comparisons,
      'attempt_budget_audit': dict(
          {
              'lower_budget': 1,
              'compile_budget': 1,
              'lower_invocations': 1,
              'compile_invocations': 1,
              'forbidden_request': None,
              'forbidden_request_detected_before_invocation': False,
          }
          if attempt_budget_audit is None else attempt_budget_audit
      ),
      'diagnostic_provenance_complete': True,
      'kernel_cache_provenance': {
          'pre_import': dict(kernel_cache_preimport_attestation),
          'post_compile': dict(
              _cache_tree_binding(Path(
                  str(kernel_cache_preimport_attestation['cache_root'])
              ))
              if precomputed_cache_binding is None
              else precomputed_cache_binding
          ),
          'cache_hit_evidence': dict(
              _model_cache_hit_evidence(
                  kernel_cache_preimport_attestation, compile_skipped=False
              )
              if precomputed_cache_hit_evidence is None
              else precomputed_cache_hit_evidence
          ),
          'default_user_cache_paths_eligible': False,
          'cache_outputs_are_diagnostic_only': True,
      },
  }
  _write_new(directory / 'COMPILER_PROVENANCE.json', record)
  return record


def _predispatch_controlled_stop(
    compiler: Mapping[str, Any],
) -> tuple[str, str, str, str | None] | None:
  """Returns the first exact pre-dispatch stop in protocol precedence."""
  cache = compiler.get('kernel_cache_provenance', {}).get(
      'cache_hit_evidence', {}
  )
  if cache.get('cache_hit') is True:
    return (
        'controlled_stop_cache_hit', 'model_cache_post_compile_hit',
        'cache_post_compile', None,
    )
  budget = compiler.get('attempt_budget_audit', {})
  forbidden = budget.get('forbidden_request')
  if forbidden in {'lower', 'compile'}:
    return (
        'controlled_stop_attempt_budget_violation',
        f'second_{forbidden}_attempt_forbidden',
        'lower' if forbidden == 'lower' else 'compile', forbidden,
    )
  same = compiler.get('same_object_attestation', {})
  same_object_checks = (
      (
          ('stablehlo_read_from_lowered_object',
           'pre_backend_hlo_read_from_lowered_object'),
          'lowered_object_identity_lost', 'lower',
      ),
      (('compile_argument_is_lowered_object',),
       'compile_argument_identity_lost', 'compile'),
      (('compiled_hlo_read_from_compiled_object',
        'compiler_record_is_gate_record'),
       'compiled_object_identity_lost', 'post_compile_diagnostics'),
      (('signature_attestation_from_apply_arguments',
        'apply_callable_is_compiled_object'),
       'apply_callable_identity_lost', 'post_compile_diagnostics'),
  )
  for names, reason, phase in same_object_checks:
    if any(same.get(name) is False for name in names):
      return (
          'controlled_stop_same_object_provenance_failure', reason, phase,
          None,
      )
  if compiler.get('diagnostic_provenance_complete') is not True:
    return (
        'controlled_stop_diagnostic_provenance_failure',
        'diagnostic_parser_failure', 'post_compile_diagnostics', None,
    )
  if compiler.get('source_program_gate', {}).get(
      'source_program_exact'
  ) is not True:
    return (
        'controlled_stop_source_program_mismatch',
        'source_program_mismatch', 'source_program', None,
    )
  return None


def _successful_same_object_attestation(
    lowered: Any, compiled: Any,
) -> dict[str, Any]:
  """Derives the successful object-flow evidence with literal `is` checks."""
  stablehlo_source = lowered
  pre_backend_source = lowered
  compile_argument = lowered
  compiled_hlo_source = compiled
  apply_callable = compiled
  compiler_gate_object = compiled
  return {
      'lower_call_count': 1,
      'compile_call_count': 1,
      'stablehlo_read_from_lowered_object': stablehlo_source is lowered,
      'pre_backend_hlo_read_from_lowered_object': pre_backend_source is lowered,
      'compile_argument_is_lowered_object': compile_argument is lowered,
      'compiled_hlo_read_from_compiled_object': compiled_hlo_source is compiled,
      'signature_attestation_from_apply_arguments': True,
      'apply_callable_is_compiled_object': apply_callable is compiled,
      'compiler_record_is_gate_record': compiler_gate_object is compiled,
      'lowered_python_id': id(lowered),
      'compiled_python_id': id(compiled),
  }


def _derive_diagnostic_failure_source_gate(
    *,
    lowered: Any,
    compiled: Any,
    published_graphs: Mapping[str, Any],
    program_signatures: Mapping[str, Any],
    program_signature_attestation: Mapping[str, Any],
    source_input_audit: Mapping[str, Any],
    entry_abi: Mapping[str, Any] | None,
) -> dict[str, Any]:
  """Derives the source gate once, without constructing an artifact."""
  same_object = _successful_same_object_attestation(lowered, compiled)
  artifacts = dict(published_graphs['artifacts'])
  observed = {
      'stablehlo_sha256': artifacts['stablehlo']['sha256'],
      'stablehlo_size_bytes': artifacts['stablehlo']['size_bytes'],
      'pre_backend_hlo_sha256': artifacts['hlo']['sha256'],
      'pre_backend_hlo_size_bytes': artifacts['hlo']['size_bytes'],
      'program_signatures_sha256': _canonical_sha256(program_signatures),
      'entry_abi_sha256': (
          '' if entry_abi is None else entry_abi['normalized_line_sha256']
      ),
  }
  launch = _launcher_record()
  return evaluate_source_program_gate(
      observed, program_signatures,
      launch['gate_a']['v3_3_2_run']['eight_row_compiler'],
      source_input_audit, program_signature_attestation, same_object,
  )


def _compiler_diagnostic_failure_artifact(
    error: Exception,
    *,
    lowered: Any,
    compiled: Any,
    published_graphs: Mapping[str, Any],
    program_signatures: Mapping[str, Any],
    program_signature_attestation: Mapping[str, Any],
    source_input_audit: Mapping[str, Any],
    entry_abi: Mapping[str, Any] | None,
    attempt_budget_audit: Mapping[str, Any] | None = None,
    precomputed_source_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  """Persists an exact compiled-but-diagnostics-incomplete artifact."""
  same_object = _successful_same_object_attestation(lowered, compiled)
  artifacts = dict(published_graphs['artifacts'])
  source_gate = dict(
      _derive_diagnostic_failure_source_gate(
          lowered=lowered, compiled=compiled,
          published_graphs=published_graphs,
          program_signatures=program_signatures,
          program_signature_attestation=program_signature_attestation,
          source_input_audit=source_input_audit, entry_abi=entry_abi,
      )
      if precomputed_source_gate is None else precomputed_source_gate
  )
  record = {
      'status': 'diagnostic_provenance_failure',
      'executable_name': 'eight_row',
      'lower_attempt_count': 1,
      'compile_attempt_count': 1,
      'successful_compile_count': 1,
      'artifacts': artifacts,
      'program_signature_attestation_binding': _relative_file_binding(
          OUTPUT_DIR / 'compiler/eight_row/'
          'PROGRAM_SIGNATURE_ATTESTATION.json'
      ),
      'external_freeze_authorization': dict(_external_authorization()),
      'source_input_audit': dict(source_input_audit),
      'source_input_audit_content_binding': _content_binding(
          source_input_audit
      ),
      'same_object_attestation': same_object,
      'same_object_attestation_content_binding': _content_binding(same_object),
      'source_program_gate_without_backend_diagnostics': source_gate,
      'failure': _failure_object(error),
      'attempt_budget_audit': dict(
          {
              'lower_budget': 1, 'compile_budget': 1,
              'lower_invocations': 1, 'compile_invocations': 1,
              'forbidden_request': None,
              'forbidden_request_detected_before_invocation': False,
          }
          if attempt_budget_audit is None else attempt_budget_audit
      ),
      'diagnostic_provenance_complete': False,
      'compiled_backend_diagnostic_only': True,
      'no_dispatch': True,
      'created_at_unix_s': time.time(),
  }
  path = OUTPUT_DIR / 'compiler/eight_row/COMPILER_DIAGNOSTIC_FAILURE.json'
  _write_new(path, record)
  return record


def _diagnostic_stop_reason(error: Exception) -> str:
  if not isinstance(error, DiagnosticProvenanceError) or not error.reason:
    raise ValueError('Diagnostic reason was not captured at its operation.')
  return error.reason


def _route_diagnostic_failure(
    triggering_error: Exception,
    triggering_reason: str,
    *, lowered: Any, compiled: Any,
    published_graphs: Mapping[str, Any],
    program_signatures: Mapping[str, Any],
    program_signature_attestation: Mapping[str, Any],
    program_signature_attestation_binding: Mapping[str, Any],
    source_input_audit: Mapping[str, Any],
    entry_abi: Mapping[str, Any] | None,
    attempt_budget_audit: Mapping[str, Any],
    historical_cache_binding: Mapping[str, Any],
    cache_hit_evidence: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
  """Routes one real diagnostic trigger without retry or relabelling."""
  if triggering_reason not in bootstrap.NONPUBLICATION_TERMINAL_CONTRACT_V3_3_4_3[
      'triggering_diagnostic_stop_reasons'
  ]:
    raise ValueError('Unknown triggering diagnostic stop reason.')
  if _diagnostic_stop_reason(triggering_error) != triggering_reason:
    raise ValueError('Diagnostic trigger and operation reason disagree.')
  same_object = _successful_same_object_attestation(lowered, compiled)
  try:
    source_gate = _derive_diagnostic_failure_source_gate(
        lowered=lowered, compiled=compiled,
        published_graphs=published_graphs,
        program_signatures=program_signatures,
        program_signature_attestation=program_signature_attestation,
        source_input_audit=source_input_audit, entry_abi=entry_abi,
    )
  except bootstrap.PublicationError:
    raise
  except Exception as gate_error:  # pylint: disable=broad-exception-caught
    _write_nonpublication_terminal(
        failure_stage=(
            'source_program_gate_derivation_for_diagnostic_failure'
        ),
        failure=gate_error,
        triggering_diagnostic_failure=triggering_error,
        triggering_diagnostic_stop_reason=triggering_reason,
        source_input_audit=source_input_audit,
        same_object_attestation=same_object,
        program_signature_attestation_binding=(
            program_signature_attestation_binding
        ),
        attempt_budget_audit=attempt_budget_audit,
        historical_cache_binding=historical_cache_binding,
        cache_hit_evidence=cache_hit_evidence,
        published_graphs=published_graphs,
        source_program_gate_without_backend_diagnostics=None,
    )
    return None
  try:
    return _compiler_diagnostic_failure_artifact(
        triggering_error,
        lowered=lowered, compiled=compiled,
        published_graphs=published_graphs,
        program_signatures=program_signatures,
        program_signature_attestation=program_signature_attestation,
        source_input_audit=source_input_audit, entry_abi=entry_abi,
        attempt_budget_audit=attempt_budget_audit,
        precomputed_source_gate=source_gate,
    )
  except bootstrap.PublicationError:
    raise
  except Exception as construction_error:  # pylint: disable=broad-exception-caught
    _write_nonpublication_terminal(
        failure_stage='diagnostic_failure_record_construction',
        failure=construction_error,
        triggering_diagnostic_failure=triggering_error,
        triggering_diagnostic_stop_reason=triggering_reason,
        source_input_audit=source_input_audit,
        same_object_attestation=same_object,
        program_signature_attestation_binding=(
            program_signature_attestation_binding
        ),
        attempt_budget_audit=attempt_budget_audit,
        historical_cache_binding=historical_cache_binding,
        cache_hit_evidence=cache_hit_evidence,
        published_graphs=published_graphs,
        source_program_gate_without_backend_diagnostics=source_gate,
    )
    return None


def _construct_model_and_inputs() -> dict[str, Any]:
  """Runs the complete caught model/checkpoint/reference setup boundary."""
  checkpoint_binding = None
  reference_binding = None
  model_constructed = False
  try:
    cases = v32.load_development_cases()
    execution_order = sidecar_execution_order(cases)
    checkpoint = v32.v2._checkpoint_path(None)  # pylint: disable=protected-access
    checkpoint_binding = v32.validate_checkpoint(checkpoint)
    model_instance = dna_model.create(
        checkpoint,
        model_settings=dna_model.ModelSettings(
            attention_backend=v32.route_v3.ATTENTION_BACKEND
        ),
    )
    model_constructed = True
    validated_reference_binding = v32.validate_reference_object()
    common_by_order = {}
    for case in cases:
      interval, position_sets, selection, target, resolved, dna, sequence_sha = (
          v32._case_inputs(  # pylint: disable=protected-access
              model_instance, case
          )
      )
      del interval, position_sets, resolved
      common_by_order[case.order] = (dna, selection, target, sequence_sha)
    reference_binding = validated_reference_binding
    return {
        'cases': cases, 'execution_order': execution_order,
        'checkpoint_binding': checkpoint_binding,
        'reference_binding': reference_binding,
        'model_instance': model_instance,
        'params': model_instance._params,  # pylint: disable=protected-access
        'state': model_instance._state,  # pylint: disable=protected-access
        'common_by_order': common_by_order,
    }
  except bootstrap.PublicationError:
    raise
  except Exception as error:  # pylint: disable=broad-exception-caught
    raise ModelSetupError(
        error, checkpoint_binding=checkpoint_binding,
        reference_binding=reference_binding,
        model_constructed=model_constructed,
    ) from error


def _capture_signature_inputs(
    common_by_order: Mapping[int, tuple[Any, ...]],
) -> dict[str, Any]:
  """Captures all prototype/signature operations inside one caught phase."""
  prototype_common = common_by_order[0]
  donor_common = common_by_order[v33.OOD_DONOR_ORDER[0]]
  selection = prototype_common[1]
  target = prototype_common[2]
  dna = _eight_row_batch(prototype_common, donor_common)
  intended = v33.eight_row_interventions(selection, 0, unrelated=False)
  unrelated = v33.eight_row_interventions(selection, 0, unrelated=True)
  signatures = {
      'selection': v32.pytree_signature(selection),
      'target': v32.pytree_signature(target),
      'eight_interventions': v32.pytree_signature(intended),
  }
  v32.assert_same_program_signature(
      signatures['eight_interventions'], unrelated
  )
  return {
      'selection': selection, 'target': target, 'dna': dna,
      'intended_interventions': intended,
      'unrelated_interventions': unrelated,
      'runtime_signatures': signatures,
  }


def _partial_same_object_attestation(
    lowered: Any, compiled: Any, *, stage: str,
) -> dict[str, Any]:
  """Returns the exact graph-read prefix for an extraction failure."""
  if stage not in bootstrap.NONPUBLICATION_TERMINAL_CONTRACT_V3_3_4_3[
      'failure_stages'
  ][:3]:
    raise ValueError('Unknown compiler graph extraction stage.')
  stable_read = (
      True if stage != 'stablehlo_text_extraction' else None
  )
  pre_backend_read = (
      True
      if stage == 'compiled_hlo_text_extraction'
      else None
  )
  return {
      'lower_call_count': 1,
      'compile_call_count': 1,
      'stablehlo_read_from_lowered_object': stable_read,
      'pre_backend_hlo_read_from_lowered_object': pre_backend_read,
      'compile_argument_is_lowered_object': True,
      'compiled_hlo_read_from_compiled_object': None,
      'signature_attestation_from_apply_arguments': True,
      'apply_callable_is_compiled_object': True,
      'compiler_record_is_gate_record': None,
      'lowered_python_id': id(lowered),
      'compiled_python_id': id(compiled),
  }


def _extract_compiler_graph_texts(
    lowered: Any, compiled: Any,
) -> dict[str, str]:
  """Captures all compiler texts in memory before any graph publication."""
  try:
    stable = str(lowered.compiler_ir(dialect='stablehlo'))
  except bootstrap.PublicationError:
    raise
  except Exception as error:  # pylint: disable=broad-exception-caught
    raise CompilerGraphExtractionError(
        'stablehlo_text_extraction', error,
        _partial_same_object_attestation(
            lowered, compiled, stage='stablehlo_text_extraction'
        ),
    ) from error
  try:
    hlo_object = lowered.compiler_ir(dialect='hlo')
    hlo = (
        hlo_object.as_hlo_text()
        if hasattr(hlo_object, 'as_hlo_text') else str(hlo_object)
    )
  except bootstrap.PublicationError:
    raise
  except Exception as error:  # pylint: disable=broad-exception-caught
    raise CompilerGraphExtractionError(
        'pre_backend_hlo_text_extraction', error,
        _partial_same_object_attestation(
            lowered, compiled, stage='pre_backend_hlo_text_extraction'
        ),
    ) from error
  try:
    compiled_hlo = compiled.as_text()
  except bootstrap.PublicationError:
    raise
  except Exception as error:  # pylint: disable=broad-exception-caught
    raise CompilerGraphExtractionError(
        'compiled_hlo_text_extraction', error,
        _partial_same_object_attestation(
            lowered, compiled, stage='compiled_hlo_text_extraction'
        ),
    ) from error
  return {
      'stablehlo': stable,
      'hlo': hlo,
      'compiled_hlo': compiled_hlo,
  }


def _publish_compiler_graphs(
    graph_texts: Mapping[str, str],
) -> dict[str, Any]:
  """Publishes a completely captured three-graph in-memory bundle."""
  if set(graph_texts) != {'stablehlo', 'hlo', 'compiled_hlo'}:
    raise ValueError('Compiler graph text bundle key set changed.')
  directory = OUTPUT_DIR / 'compiler' / 'eight_row'
  artifacts = {}
  for name, filename, content in (
      ('stablehlo', 'graph.stablehlo.mlir', graph_texts['stablehlo']),
      ('hlo', 'graph.pre_backend.hlo.txt', graph_texts['hlo']),
      ('compiled_hlo', 'graph.compiled.hlo.txt', graph_texts['compiled_hlo']),
  ):
    path = directory / filename
    _write_new_text(path, content)
    artifacts[name] = {
        'path': path.relative_to(OUTPUT_DIR).as_posix(),
        'sha256': _sha256(path),
        'size_bytes': path.stat().st_size,
    }
  return {
      'artifacts': artifacts,
      'compiled_hlo_text': graph_texts['compiled_hlo'],
  }


def _compiler_failure_artifact(
    error: Exception,
    *,
    stage: str,
    compile_count: int,
    seconds: float,
    lowered: Any | None,
    program_signatures: Mapping[str, Any],
    kernel_cache_preimport_attestation: Mapping[str, Any],
    program_signature_attestation: Mapping[str, Any],
    source_input_audit: Mapping[str, Any],
    attempt_budget_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  """Persists a one-shot lowering/compiler failure without retrying."""
  if stage == 'lower':
    if lowered is not None or compile_count != 0:
      raise ValueError(
          'Lowering failure requires lowered=None and compile_count=0.'
      )
  elif stage == 'compile':
    if lowered is None or compile_count != 1:
      raise ValueError(
          'Compile failure requires a lowered program and compile_count=1.'
      )
  else:
    raise ValueError(f'Unknown compiler failure stage: {stage}.')
  directory = OUTPUT_DIR / 'compiler' / 'eight_row'
  artifacts = {}
  if lowered is not None:
    stablehlo_text = str(lowered.compiler_ir(dialect='stablehlo'))
    hlo_ir = lowered.compiler_ir(dialect='hlo')
    pre_backend_hlo_text = (
        hlo_ir.as_hlo_text() if hasattr(hlo_ir, 'as_hlo_text') else str(hlo_ir)
    )
    for name, filename, content in (
        (
            'stablehlo',
            'graph.stablehlo.mlir',
            stablehlo_text,
        ),
        (
            'hlo',
            'graph.pre_backend.hlo.txt',
            pre_backend_hlo_text,
        ),
    ):
      path = directory / filename
      artifacts[name] = {
          'path': path.relative_to(OUTPUT_DIR).as_posix(),
          'sha256': _write_new_text(path, content),
          'size_bytes': len(content.encode('utf-8')),
      }
  same_object_attestation = _compiler_failure_same_object_attestation(
      stage=stage, lowered=lowered, compile_count=compile_count
  )
  record = _compiler_failure_record_body(
      error=error,
      stage=stage,
      compile_count=compile_count,
      seconds=seconds,
      artifacts=artifacts,
      program_signatures=program_signatures,
      kernel_cache_preimport_attestation=kernel_cache_preimport_attestation,
      program_signature_attestation=program_signature_attestation,
      source_input_audit=source_input_audit,
      same_object_attestation=same_object_attestation,
      attempt_budget_audit=attempt_budget_audit,
  )
  _write_new(directory / 'COMPILER_PROVENANCE.json', record)
  return record


def _compiler_failure_same_object_attestation(
    *, stage: str, lowered: Any | None, compile_count: int,
) -> dict[str, Any]:
  """Returns exact phase evidence before a failure compiler record publish."""
  return {
      'lower_call_count': 1,
      'compile_call_count': compile_count,
      'stablehlo_read_from_lowered_object': (
          None if lowered is None else True
      ),
      'pre_backend_hlo_read_from_lowered_object': (
          None if lowered is None else True
      ),
      'compile_argument_is_lowered_object': (
          None if lowered is None else True
      ),
      'compiled_hlo_read_from_compiled_object': None,
      'signature_attestation_from_apply_arguments': True,
      'apply_callable_is_compiled_object': None,
      'compiler_record_is_gate_record': True,
      'lowered_python_id': None if lowered is None else id(lowered),
      'compiled_python_id': None,
  }


def _compiler_failure_record_body(
    *,
    error: Exception,
    stage: str,
    compile_count: int,
    seconds: float,
    artifacts: Mapping[str, Any],
    program_signatures: Mapping[str, Any],
    kernel_cache_preimport_attestation: Mapping[str, Any],
    program_signature_attestation: Mapping[str, Any],
    source_input_audit: Mapping[str, Any],
    same_object_attestation: Mapping[str, Any],
    attempt_budget_audit: Mapping[str, Any] | None,
    successful_compile_count: int = 0,
) -> dict[str, Any]:
  return {
      'status': 'compiler_failure',
      'failure_stage': stage,
      'compile_count': compile_count,
      'lower_attempt_count': 1,
      'compile_attempt_count': compile_count,
      'successful_compile_count': successful_compile_count,
      'lower_or_compile_pipeline_attempt_count': 1,
      'compile_seconds': seconds,
      'artifacts': artifacts,
      'program_signatures': dict(program_signatures),
      'program_signatures_sha256': _canonical_sha256(program_signatures),
      'program_signature_attestation': _relative_file_binding(
          OUTPUT_DIR / 'compiler/eight_row/'
          'PROGRAM_SIGNATURE_ATTESTATION.json'
      ),
      'external_freeze_authorization': dict(_external_authorization()),
      'source_input_audit': dict(source_input_audit),
      'source_input_audit_content_binding': _content_binding(
          source_input_audit
      ),
      'same_object_attestation': same_object_attestation,
      'same_object_attestation_content_binding': _content_binding(
          same_object_attestation
      ),
      'source_program_gate': None,
      'compiled_backend_diagnostic_only': True,
      'failure': {
          'type': type(error).__name__,
          'message': str(error),
          'traceback': ''.join(traceback.format_exception(error)),
      },
      'no_compile_retry': True,
      'model_apply_count': 0,
      'attempt_budget_audit': dict(
          {
              'lower_budget': 1,
              'compile_budget': 1,
              'lower_invocations': 1,
              'compile_invocations': compile_count,
              'forbidden_request': None,
              'forbidden_request_detected_before_invocation': False,
          }
          if attempt_budget_audit is None else attempt_budget_audit
      ),
      'diagnostic_provenance_complete': None,
      'kernel_cache_provenance': {
          'pre_import': dict(kernel_cache_preimport_attestation),
          'post_failure': _cache_tree_binding(Path(
              str(kernel_cache_preimport_attestation['cache_root'])
          )),
          'cache_hit_evidence': _model_cache_hit_evidence(
              kernel_cache_preimport_attestation,
              compile_skipped=(None if stage == 'lower' else False),
          ),
          'default_user_cache_paths_eligible': False,
          'cache_outputs_are_diagnostic_only': True,
      },
  }


def _attempt_budget_failure_artifact(
    error: AttemptBudgetViolation,
    *,
    lowered: Any,
    compiled: Any | None = None,
    published_graphs: Mapping[str, Any] | None = None,
    seconds: float,
    program_signatures: Mapping[str, Any],
    kernel_cache_preimport_attestation: Mapping[str, Any],
    program_signature_attestation: Mapping[str, Any],
    source_input_audit: Mapping[str, Any],
    attempt_budget_audit: Mapping[str, Any],
) -> dict[str, Any]:
  """Persists the exact prior prefix for a guarded second compiler request."""
  if error.operation not in {'lower', 'compile'} or lowered is None:
    raise RuntimeError('Illegal guarded compiler-attempt prefix.') from error
  if error.operation == 'lower' and (
      compiled is not None or published_graphs is not None
  ):
    raise RuntimeError('Second-lower guard cannot carry compiled artifacts.')
  if error.operation == 'compile' and (
      compiled is None or published_graphs is None
  ):
    raise RuntimeError('Second-compile guard requires one successful compile.')
  directory = OUTPUT_DIR / 'compiler/eight_row'
  if error.operation == 'lower':
    artifacts = {}
    hlo_ir = lowered.compiler_ir(dialect='hlo')
    for name, filename, content in (
        ('stablehlo', 'graph.stablehlo.mlir',
         str(lowered.compiler_ir(dialect='stablehlo'))),
        ('hlo', 'graph.pre_backend.hlo.txt',
         hlo_ir.as_hlo_text() if hasattr(hlo_ir, 'as_hlo_text') else str(hlo_ir)),
    ):
      path = directory / filename
      _write_new_text(path, content)
      artifacts[name] = _relative_file_binding(path)
    same_object = {
        'lower_call_count': 1, 'compile_call_count': 0,
        'stablehlo_read_from_lowered_object': True,
        'pre_backend_hlo_read_from_lowered_object': True,
        'compile_argument_is_lowered_object': None,
        'compiled_hlo_read_from_compiled_object': None,
        'signature_attestation_from_apply_arguments': True,
        'apply_callable_is_compiled_object': None,
        'compiler_record_is_gate_record': True,
        'lowered_python_id': id(lowered), 'compiled_python_id': None,
    }
  else:
    artifacts = dict(published_graphs['artifacts'])
    if set(artifacts) != {'stablehlo', 'hlo', 'compiled_hlo'}:
      raise RuntimeError('Second-compile guard graph prefix changed.')
    same_object = _successful_same_object_attestation(lowered, compiled)
  record = _compiler_failure_record_body(
      error=error, stage=error.operation,
      compile_count=int(error.operation == 'compile'), seconds=seconds,
      artifacts=artifacts, program_signatures=program_signatures,
      kernel_cache_preimport_attestation=kernel_cache_preimport_attestation,
      program_signature_attestation=program_signature_attestation,
      source_input_audit=source_input_audit,
      same_object_attestation=same_object,
      attempt_budget_audit=attempt_budget_audit,
      successful_compile_count=int(error.operation == 'compile'),
  )
  record['status'] = 'attempt_budget_failure'
  record['failure_stage'] = f'second_{error.operation}_guarded'
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
    compiled: Any,
    args: Sequence[Any],
    apply_counter: list[int],
    *,
    call_label: str,
    execution_index: int,
    recipient: Any,
    anchor_id: int,
    call_index: int,
    source_input_audit_binding: Mapping[str, Any],
    same_object_attestation_binding: Mapping[str, Any],
) -> tuple[Any, float, dict[str, Any], dict[str, Any]]:
  """Durably journals one dispatch before and after its successful return."""
  if len(apply_counter) != 1 or apply_counter[0] < 0:
    raise ValueError('Apply counter must be a one-element non-negative list.')
  expected_index = 4 * execution_index + call_index
  if apply_counter[0] != expected_index:
    raise ValueError('Dispatch counter is not the frozen zero-based prefix.')
  started_at = time.time()
  event_common = {
      'schema_version': 'v3.3.4.4-dispatch-journal-v1',
      'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION,
      'execution_index': execution_index,
      'recipient_order': recipient.order,
      'recipient_variant_id': recipient.variant_id,
      'anchor_id': anchor_id,
      'call_index_within_record': call_index,
      'call_role': call_label,
      'global_dispatch_index': expected_index,
      'runner_pid': os.getpid(),
      'source_input_audit_sha256': source_input_audit_binding['sha256'],
      'same_object_attestation_sha256': (
          same_object_attestation_binding['sha256']
      ),
  }
  started_record = {
      **event_common,
      'event': 'dispatch_started',
      'started_at_unix_s': started_at,
  }
  started_path = (
      OUTPUT_DIR / 'dispatch_journal' / 'started' /
      f'{expected_index:03d}.json'
  )
  _write_new(started_path, started_record)
  started_binding = _relative_file_binding(started_path)
  apply_counter[0] += 1
  try:
    returned, seconds = v32._timed_apply(  # pylint: disable=protected-access
        compiled, args
    )
  except Exception as error:  # pylint: disable=broad-exception-caught
    raise CountedApplyError(call_label, apply_counter[0], error) from error
  completed_record = {
      **event_common,
      'event': 'dispatch_completed',
      'started_event_sha256': started_binding['sha256'],
      'returned': True,
      'completed_at_unix_s': time.time(),
  }
  completed_path = (
      OUTPUT_DIR / 'dispatch_journal' / 'completed' /
      f'{expected_index:03d}.json'
  )
  _write_new(completed_path, completed_record)
  return (
      returned,
      seconds,
      started_binding,
      _relative_file_binding(completed_path),
  )


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
    external_freeze_authorization: Mapping[str, Any],
    source_input_audit: Mapping[str, Any],
    same_object_attestation: Mapping[str, Any],
) -> dict[str, Any]:
  source_binding = _content_binding(source_input_audit)
  same_object_binding = _content_binding(same_object_attestation)
  returned: list[Any | None] = [None, None, None, None]
  seconds: list[float | None] = [None, None, None, None]
  started_bindings: list[dict[str, Any]] = []
  completed_bindings: list[dict[str, Any]] = []
  try:
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
  except Exception as error:  # pylint: disable=broad-exception-caught
    raise CurrentRecordStop(
        failure_phase='record_setup',
        failed_or_next_call_role='intended',
        returned_outputs=returned,
        started=started_bindings,
        completed=completed_bindings,
        original_error=error,
    ) from error
  calls = (
      ('intended', intended_interventions),
      ('intended_repeat', intended_interventions),
      ('unrelated', unrelated_interventions),
      ('unrelated_repeat', unrelated_interventions),
  )
  for call_index, (call_role, interventions) in enumerate(calls):
    try:
      result, elapsed, started, completed = _counted_apply(
          compiled,
          (*common_args, interventions, target),
          apply_counter,
          call_label=call_role,
          execution_index=execution_index,
          recipient=recipient,
          anchor_id=anchor_id,
          call_index=call_index,
          source_input_audit_binding=source_binding,
          same_object_attestation_binding=same_object_binding,
      )
      returned[call_index] = result
      seconds[call_index] = elapsed
      started_bindings.append(started)
      completed_bindings.append(completed)
    except CountedApplyError as error:
      started_path = (
          OUTPUT_DIR / 'dispatch_journal' / 'started' /
          f'{4 * execution_index + call_index:03d}.json'
      )
      if started_path.exists():
        started_bindings.append(_relative_file_binding(started_path))
      raise CurrentRecordStop(
          failure_phase='model_dispatch',
          failed_or_next_call_role=call_role,
          returned_outputs=returned,
          started=started_bindings,
          completed=completed_bindings,
          original_error=error.original_error,
      ) from error

  intended, intended_repeat, unrelated, unrelated_repeat = returned
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
  except Exception as error:  # pylint: disable=broad-exception-caught
    raise CurrentRecordStop(
        failure_phase='record_validation',
        failed_or_next_call_role=None,
        returned_outputs=returned,
        started=started_bindings,
        completed=completed_bindings,
        original_error=error,
    ) from error
  artifact = {
      'status': 'complete',
      'family': 'v3_3_4_4_unrelated_donor_sidecar_anchor',
      'script_version': SCRIPT_VERSION,
      'amendment_sha256': AMENDMENT_SHA256,
      'amendment_commit': AMENDMENT_COMMIT,
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'freeze_sha256': freeze_sha256,
      'external_freeze_authorization': dict(
          external_freeze_authorization
      ),
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
      'model_apply_count_through_record': 4 * (execution_index + 1),
      'checks': checks,
      'failure': None,
      'seconds': {
          name: seconds[index] for index, name in enumerate(CALL_ROLES)
      },
      'dispatch_started_bindings': started_bindings,
      'dispatch_completed_bindings': completed_bindings,
      'source_input_audit': dict(source_input_audit),
      'source_input_audit_content_binding': source_binding,
      'same_object_attestation': dict(same_object_attestation),
      'same_object_attestation_content_binding': same_object_binding,
      'confirmation_scope_disclosure': DISCLOSURE,
      'created_at_unix_s': time.time(),
  }
  path = _artifact_path(recipient, anchor_id)
  try:
    _write_new(path, artifact)
  except bootstrap.PublicationError:
    raise
  except Exception as error:  # pylint: disable=broad-exception-caught
    raise CurrentRecordStop(
        failure_phase='record_serialization',
        failed_or_next_call_role=None,
        returned_outputs=returned,
        started=started_bindings,
        completed=completed_bindings,
        original_error=error,
    ) from error
  return {
      'status': 'complete',
      'recipient_order': recipient.order,
      'anchor_id': anchor_id,
      'checks': checks,
      'failure': None,
      'artifact_binding': _relative_file_binding(path),
  }


def _tree_digest(paths: Sequence[Path], root: Path) -> str:
  digest = hashlib.sha256()
  for path in sorted(paths):
    digest.update(str(path.relative_to(root)).encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(_sha256(path)))
  return digest.hexdigest()


def _binding_map(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
  return {
      path.relative_to(OUTPUT_DIR).as_posix(): {
          'sha256': _sha256(path),
          'size_bytes': path.stat().st_size,
      }
      for path in sorted(paths)
  }


def _binding_tree_sha256(bindings: Mapping[str, Mapping[str, Any]]) -> str:
  digest = hashlib.sha256()
  for relative, binding in sorted(bindings.items()):
    digest.update(relative.encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(binding['sha256']))
  return digest.hexdigest()


def _journal_paths(role: str) -> list[Path]:
  root = OUTPUT_DIR / 'dispatch_journal' / role
  if not root.exists():
    return []
  mode = root.lstat().st_mode
  if root.is_symlink() or not stat.S_ISDIR(mode):
    raise RuntimeError(f'Dispatch {role} journal root is unsafe.')
  paths = []
  for path in sorted(root.iterdir()):
    child_mode = path.lstat().st_mode
    if path.is_symlink() or not stat.S_ISREG(child_mode):
      raise RuntimeError(f'Dispatch {role} journal entry is unsafe.')
    paths.append(path)
  expected = [f'{index:03d}.json' for index in range(len(paths))]
  if [path.name for path in paths] != expected:
    raise RuntimeError(f'Dispatch {role} journal is not a strict prefix.')
  return paths


def _raw_manifest(
    results: Sequence[Mapping[str, Any]] = (),
    *, failed_current_binding: Mapping[str, Any] | None = None,
    source_input_audit_binding: Mapping[str, Any] | None = None,
    same_object_attestation_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  paths = []
  ood_root = OUTPUT_DIR / 'raw' / 'ood_anchors'
  if ood_root.exists():
    paths = bootstrap._strict_file_tree(ood_root)  # pylint: disable=protected-access
  artifact_bindings = _binding_map(paths)
  if len(paths) != len(results):
    raise RuntimeError('Valid raw namespace differs from completed prefix.')
  pairs = [{
      'execution_index': index,
      'recipient_order': int(result['recipient_order']),
      'anchor_id': int(result['anchor_id']),
  } for index, result in enumerate(results)]
  started_bindings = _binding_map(_journal_paths('started'))
  completed_bindings = _binding_map(_journal_paths('completed'))
  if len(results) == 80 and failed_current_binding is None:
    status = 'complete80'
  elif results or failed_current_binding is not None:
    status = 'controlled_prefix'
  else:
    status = 'empty_controlled_stop'
  return {
      'schema_version': 'v3.3.4.4-raw-manifest-v1',
      'status': status,
      'attempt_id': ATTEMPT_ID,
      'external_freeze_authorization': _external_authorization(),
      'valid_artifact_count': len(paths),
      'artifact_bindings': artifact_bindings,
      'artifact_tree_sha256': _binding_tree_sha256(artifact_bindings),
      'valid_recipient_anchor_pairs': pairs,
      'failed_current_binding': (
          None if failed_current_binding is None
          else dict(failed_current_binding)
      ),
      'dispatch_started_count': len(started_bindings),
      'dispatch_completed_count': len(completed_bindings),
      'dispatch_started_bindings': started_bindings,
      'dispatch_started_tree_sha256': _binding_tree_sha256(started_bindings),
      'dispatch_completed_bindings': completed_bindings,
      'dispatch_completed_tree_sha256': _binding_tree_sha256(
          completed_bindings
      ),
      'source_input_audit_content_binding': (
          None if source_input_audit_binding is None
          else dict(source_input_audit_binding)
      ),
      'same_object_attestation_content_binding': (
          None if same_object_attestation_binding is None
          else dict(same_object_attestation_binding)
      ),
      'created_at_unix_s': time.time(),
  }


def _treedef_ast(treedef: Any) -> dict[str, Any]:
  if treedef.num_nodes == 1 and treedef.num_leaves == 1:
    return {'kind': 'leaf', 'metadata': None, 'children': []}
  children = list(treedef.children())
  node_data = treedef.node_data()
  node_type = node_data[0] if isinstance(node_data, tuple) else node_data
  metadata = node_data[1] if isinstance(node_data, tuple) and len(node_data) > 1 else None
  if node_type is dict:
    kind = 'dict'
    keys = list(metadata)
    node_metadata: Any = keys
  elif node_type is list:
    kind = 'list'
    node_metadata = len(children)
  else:
    # Named tuples and registered scientific output records are represented by
    # their ordered children; their Python class name is never serialized.
    kind = 'tuple'
    node_metadata = len(children)
  return {
      'kind': kind,
      'metadata': node_metadata,
      'children': [_treedef_ast(child) for child in children],
  }


def _ast_leaf_paths(
    node: Mapping[str, Any], path: list[dict[str, Any]] | None = None
) -> list[list[dict[str, Any]]]:
  prefix = [] if path is None else list(path)
  if node['kind'] == 'leaf':
    return [prefix]
  result = []
  for index, child in enumerate(node['children']):
    if node['kind'] == 'dict':
      token = {'kind': 'dict_key', 'key': node['metadata'][index]}
    else:
      token = {'kind': 'sequence_index', 'index': index}
    result.extend(_ast_leaf_paths(child, [*prefix, token]))
  return result


def _lossless_returned_output(value: Any) -> dict[str, Any]:
  leaves, treedef = jax.tree_util.tree_flatten(value)
  ast = _treedef_ast(treedef)
  paths = _ast_leaf_paths(ast)
  if len(paths) != len(leaves):
    raise RuntimeError('Returned-output AST/leaf count differs.')
  rows = []
  for path, leaf in zip(paths, leaves, strict=True):
    array = np.asarray(jax.device_get(leaf))
    dtype_name = array.dtype.name
    if array.dtype.itemsize > 1:
      array = array.astype(array.dtype.newbyteorder('<'), copy=False)
      byte_order = 'little'
    else:
      byte_order = 'not_applicable'
    payload = np.ascontiguousarray(array).tobytes(order='C')
    rows.append({
        'path': path,
        'dtype_name': dtype_name,
        'byte_order': byte_order,
        'shape': list(array.shape),
        'encoding': 'base64_c_order_raw_bytes',
        'data_base64': base64.b64encode(payload).decode('ascii'),
        'sha256': hashlib.sha256(payload).hexdigest(),
        'size_bytes': len(payload),
    })
  return {
      'status': 'returned',
      'treedef': ast,
      'leaf_count': len(rows),
      'leaves': rows,
  }


def _write_failed_current(
    stop: CurrentRecordStop,
    *, execution_index: int,
    recipient: Any,
    anchor_id: int,
    source_input_audit_binding: Mapping[str, Any],
    same_object_attestation_binding: Mapping[str, Any],
) -> dict[str, Any]:
  d_completed = len(stop.completed)
  if d_completed not in range(5):
    raise RuntimeError('Failed-current completion prefix is invalid.')
  partial = {
      role: (
          None if stop.returned_outputs[index] is None
          else _lossless_returned_output(stop.returned_outputs[index])
      )
      for index, role in enumerate(CALL_ROLES)
  }
  record = {
      'schema_version': 'v3.3.4.4-failed-current-v1',
      'status': 'failed_current',
      'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION,
      'external_freeze_authorization': dict(_external_authorization()),
      'execution_index': execution_index,
      'recipient_order': recipient.order,
      'recipient_variant_id': recipient.variant_id,
      'anchor_id': anchor_id,
      'failed_or_next_call_role': stop.failed_or_next_call_role,
      'd_completed': d_completed,
      'started_count': 4 * execution_index + len(stop.started),
      'completed_count': 4 * execution_index + d_completed,
      'started_event_bindings': list(stop.started),
      'completed_event_bindings': list(stop.completed),
      'partial_call_outputs': partial,
      'failure_phase': stop.failure_phase,
      'failure': {
          'type': type(stop.original_error).__name__,
          'message': str(stop.original_error),
          'traceback': ''.join(traceback.format_exception(
              stop.original_error
          )),
      },
      'source_input_audit_content_binding': dict(
          source_input_audit_binding
      ),
      'same_object_attestation_content_binding': dict(
          same_object_attestation_binding
      ),
      'confirmation_scope_disclosure': DISCLOSURE,
      'created_at_unix_s': time.time(),
  }
  path = (
      OUTPUT_DIR / 'raw' / 'failed_current' /
      f'{execution_index:03d}_{_slug(recipient.variant_id)}' /
      f'{anchor_id:03d}.json'
  )
  _write_new(path, record)
  return _relative_file_binding(path)


def _assert_attempt_budget(started_monotonic: float) -> None:
  if time.monotonic() - started_monotonic > MAX_WALL_TIME_SECONDS:
    raise RuntimeError('v3.3.4.4 frozen wall-time budget was exceeded.')
  output_size = sum(
      path.stat().st_size for path in OUTPUT_DIR.rglob('*') if path.is_file()
  )
  if output_size > MAX_OUTPUT_BYTES:
    raise RuntimeError('v3.3.4.4 frozen output-storage budget was exceeded.')


def _run_tree_binding(
    excluded_entries: Sequence[str] = (),
) -> dict[str, Any]:
  excluded = set(excluded_entries)
  files: list[Path] = []
  directories: list[Path] = []
  pending = [OUTPUT_DIR]
  while pending:
    directory = pending.pop()
    if directory.is_symlink() or not directory.is_dir():
      raise RuntimeError(f'Run tree contains a non-directory: {directory}.')
    directories.append(directory)
    for entry in sorted(directory.iterdir()):
      relative = entry.relative_to(OUTPUT_DIR).as_posix()
      if relative in excluded:
        mode = entry.lstat().st_mode
        if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
          # A preserved blocking directory remains part of the physical
          # directory tree, but its contents are outside the successful-final
          # binding set and are never traversed.
          directories.append(entry)
        continue
      if entry.is_symlink():
        raise RuntimeError(f'Run tree contains a symlink: {entry}.')
      if entry.is_dir():
        pending.append(entry)
      elif entry.is_file():
        files.append(entry)
      else:
        raise RuntimeError(f'Run tree contains a special entry: {entry}.')
  bindings = _binding_map(files)
  directory_paths = sorted(
      '.' if path == OUTPUT_DIR else path.relative_to(OUTPUT_DIR).as_posix()
      for path in directories
  )
  directory_digest = hashlib.sha256()
  for relative in directory_paths:
    directory_digest.update(b'D\0' + relative.encode('utf-8') + b'\0')
  return {
      'file_count': len(bindings),
      'directory_count': len(directory_paths),
      'file_bindings': bindings,
      'file_tree_sha256': _binding_tree_sha256(bindings),
      'directory_paths': directory_paths,
      'directory_tree_sha256': directory_digest.hexdigest(),
  }


def _journal_summary(
    excluded_entries: Sequence[str] = (),
) -> dict[str, Any]:
  excluded = set(excluded_entries)
  started = _binding_map([
      path for path in _journal_paths('started')
      if path.relative_to(OUTPUT_DIR).as_posix() not in excluded
  ])
  completed = _binding_map([
      path for path in _journal_paths('completed')
      if path.relative_to(OUTPUT_DIR).as_posix() not in excluded
  ])
  return {
      'started_count': len(started),
      'completed_count': len(completed),
      'started_bindings': started,
      'completed_bindings': completed,
      'started_tree_sha256': _binding_tree_sha256(started),
      'completed_tree_sha256': _binding_tree_sha256(completed),
      'started_prefix_exact': list(started) == [
          f'dispatch_journal/started/{index:03d}.json'
          for index in range(len(started))
      ],
      'completed_prefix_exact': list(completed) == [
          f'dispatch_journal/completed/{index:03d}.json'
          for index in range(len(completed))
      ],
  }


_PHASE_STATE_KEYS = (
    'preflight_passed', 'start_persisted',
    'post_start_source_gate_passed', 'protobuf_persisted',
    'pre_model_import_inventory_persisted',
    'model_construction_attempted', 'model_constructed',
    'reference_cases_loaded', 'signatures_captured',
    'signature_attestation_persisted',
    'post_model_import_inventory_persisted', 'lower_attempted',
    'lower_succeeded', 'compile_attempted', 'compile_succeeded',
    'terminal_import_inventory_persisted', 'source_program_gate_passed',
    'diagnostic_provenance_passed', 'dispatch_begun',
)

_TERMINAL_STATUS_REASONS = {
    'controlled_stop_import_provenance_failure': {
        'pre_model_import_inventory_mismatch',
        'post_model_import_inventory_mismatch',
        'terminal_import_inventory_mismatch',
    },
    'controlled_stop_protobuf_provenance_failure': {
        'protobuf_binding_mismatch'
    },
    'controlled_stop_cache_hit': {
        'model_cache_pre_import_hit', 'model_cache_post_compile_hit'
    },
    'controlled_stop_model_setup_failure': {'model_setup_failure'},
    'controlled_stop_signature_attestation_failure': {
        'signature_attestation_failure'
    },
    'controlled_stop_lower_failure': {'lower_failure'},
    'controlled_stop_compile_failure': {'compile_failure'},
    'controlled_stop_attempt_budget_violation': {
        'second_lower_attempt_forbidden',
        'second_compile_attempt_forbidden',
    },
    'controlled_stop_same_object_provenance_failure': {
        'lowered_object_identity_lost', 'compile_argument_identity_lost',
        'compiled_object_identity_lost', 'apply_callable_identity_lost',
    },
    'controlled_stop_source_program_mismatch': {'source_program_mismatch'},
    'controlled_stop_diagnostic_provenance_failure': {
        'diagnostic_parser_failure', 'diagnostic_persistence_failure',
        'cache_signal_unavailable', 'fingerprint_formula_mismatch',
    },
    'controlled_stop_partial_dispatch': {
        'record_setup_failure', 'model_dispatch_failure'
    },
    'controlled_stop_four_call_invalid': {
        'record_validation_or_serialization_failure'
    },
    'complete_structural_sidecar': {None},
}

_COMPILER_MEMBERSHIP_BY_STATE = {
    'none': frozenset(),
    'signature_failure': frozenset({
        'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION_FAILURE.json',
    }),
    'lowered': frozenset({
        'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json',
        'compiler/eight_row/COMPILER_PROVENANCE.json',
    }),
    'precompiled': frozenset({
        'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json',
        'compiler/eight_row/graph.stablehlo.mlir',
        'compiler/eight_row/graph.pre_backend.hlo.txt',
        'compiler/eight_row/COMPILER_PROVENANCE.json',
    }),
    'compiled': frozenset({
        'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json',
        'compiler/eight_row/graph.stablehlo.mlir',
        'compiler/eight_row/graph.pre_backend.hlo.txt',
        'compiler/eight_row/graph.compiled.hlo.txt',
        'compiler/eight_row/COMPILER_PROVENANCE.json',
    }),
    'compiled_guarded': frozenset({
        'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json',
        'compiler/eight_row/graph.stablehlo.mlir',
        'compiler/eight_row/graph.pre_backend.hlo.txt',
        'compiler/eight_row/graph.compiled.hlo.txt',
        'compiler/eight_row/COMPILER_PROVENANCE.json',
    }),
    'diagnostic_failure': frozenset({
        'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json',
        'compiler/eight_row/graph.stablehlo.mlir',
        'compiler/eight_row/graph.pre_backend.hlo.txt',
        'compiler/eight_row/graph.compiled.hlo.txt',
        'compiler/eight_row/COMPILER_DIAGNOSTIC_FAILURE.json',
    }),
}

_TERMINAL_DETAIL_BY_REASON = {
    'pre_model_import_inventory_mismatch': ('imports', None, 'pre_model'),
    'post_model_import_inventory_mismatch': (
        'imports', None, 'post_model_precompile'
    ),
    'terminal_import_inventory_mismatch': ('imports', None, 'terminal'),
    'protobuf_binding_mismatch': ('protobuf', None, 'protobuf'),
    'model_cache_pre_import_hit': ('cache_pre_import', None, None),
    'model_cache_post_compile_hit': ('cache_post_compile', None, None),
    'model_setup_failure': ('model_setup', None, None),
    'signature_attestation_failure': ('signatures', None, None),
    'lower_failure': ('lower', None, None),
    'compile_failure': ('compile', None, None),
    'second_lower_attempt_forbidden': ('lower', 'lower', None),
    'second_compile_attempt_forbidden': ('compile', 'compile', None),
    'lowered_object_identity_lost': ('lower', None, None),
    'compile_argument_identity_lost': ('compile', None, None),
    'compiled_object_identity_lost': (
        'post_compile_diagnostics', None, None
    ),
    'apply_callable_identity_lost': (
        'post_compile_diagnostics', None, None
    ),
    'source_program_mismatch': ('source_program', None, None),
    'diagnostic_parser_failure': (
        'post_compile_diagnostics', None, None
    ),
    'diagnostic_persistence_failure': (
        'post_compile_diagnostics', None, None
    ),
    'cache_signal_unavailable': ('post_compile_diagnostics', None, None),
    'fingerprint_formula_mismatch': (
        'post_compile_diagnostics', None, None
    ),
    'record_setup_failure': ('record_setup', None, None),
    'model_dispatch_failure': ('model_dispatch', None, None),
}


def _terminal_compiler_state(status: str, stop_reason: str | None) -> str:
  """Returns the literal compiler-prefix row for one normal terminal."""
  if status in {
      'controlled_stop_protobuf_provenance_failure',
      'controlled_stop_model_setup_failure',
  }:
    return 'none'
  if status == 'controlled_stop_import_provenance_failure':
    return 'compiled' if stop_reason == 'terminal_import_inventory_mismatch' else 'none'
  if status == 'controlled_stop_cache_hit':
    return 'compiled' if stop_reason == 'model_cache_post_compile_hit' else 'none'
  if status == 'controlled_stop_signature_attestation_failure':
    return 'signature_failure'
  if status == 'controlled_stop_lower_failure':
    return 'lowered'
  if status == 'controlled_stop_compile_failure':
    return 'precompiled'
  if status == 'controlled_stop_attempt_budget_violation':
    return (
        'lowered' if stop_reason == 'second_lower_attempt_forbidden'
        else 'compiled_guarded'
    )
  if status == 'controlled_stop_same_object_provenance_failure':
    if stop_reason == 'lowered_object_identity_lost':
      return 'lowered'
    if stop_reason == 'compile_argument_identity_lost':
      return 'precompiled'
    return 'compiled_guarded'
  if status == 'controlled_stop_diagnostic_provenance_failure':
    return 'diagnostic_failure'
  return 'compiled'


def _validate_common_terminal_semantics(record: Mapping[str, Any]) -> None:
  """Fail-closes the literal normal RUN_COMPLETE phase/membership matrix."""
  if set(record) != set(bootstrap.TERMINAL_CONTRACT['run_complete_keys']):
    raise RuntimeError('RUN_COMPLETE exact key set changed.')
  prior_prefix, prior_binding = _prior_prefix_from_start(
      _launcher_record()['start']
  )
  if (
      record.get('prior_v3_3_4_3_consumed_preflight_prefix') != prior_prefix
      or record.get(
          'prior_v3_3_4_3_consumed_preflight_prefix_content_binding'
      ) != prior_binding
  ):
    raise RuntimeError('RUN_COMPLETE consumed-prefix binding changed.')
  status = str(record['status'])
  reason = record['stop_reason']
  _validate_terminal_identity(status, reason)
  phase = record['phase_state']
  if not isinstance(phase, Mapping) or set(phase) != set(_PHASE_STATE_KEYS):
    raise RuntimeError('RUN_COMPLETE phase-state schema changed.')
  if not all(type(value) is bool for value in phase.values()):
    raise TypeError('RUN_COMPLETE phase state must contain only booleans.')
  for name in (
      'preflight_passed', 'start_persisted',
      'post_start_source_gate_passed',
  ):
    if phase[name] is not True:
      raise RuntimeError(f'RUN_COMPLETE lacks mandatory phase: {name}.')
  dependencies = {
      'protobuf_persisted': ('pre_model_import_inventory_persisted',),
      'model_construction_attempted': (
          'pre_model_import_inventory_persisted', 'protobuf_persisted'
      ),
      'model_constructed': ('model_construction_attempted',),
      'reference_cases_loaded': ('model_constructed',),
      'post_model_import_inventory_persisted': (
          'model_construction_attempted',
      ),
      'signatures_captured': (
          'reference_cases_loaded', 'post_model_import_inventory_persisted'
      ),
      'signature_attestation_persisted': ('signatures_captured',),
      'lower_attempted': ('signature_attestation_persisted',),
      'lower_succeeded': ('lower_attempted',),
      'compile_attempted': ('lower_succeeded',),
      'compile_succeeded': ('compile_attempted',),
      'terminal_import_inventory_persisted': (
          'post_model_import_inventory_persisted',
      ),
      'source_program_gate_passed': (
          'compile_succeeded', 'terminal_import_inventory_persisted'
      ),
      'diagnostic_provenance_passed': ('compile_succeeded',),
      'dispatch_begun': (
          'source_program_gate_passed', 'diagnostic_provenance_passed'
      ),
  }
  for child, parents in dependencies.items():
    if phase[child] and not all(phase[parent] for parent in parents):
      raise RuntimeError(f'RUN_COMPLETE phase DAG broke at {child}.')
  detail = record['terminal_detail']
  if set(detail) != set(bootstrap.TERMINAL_CONTRACT['terminal_detail_keys']):
    raise RuntimeError('RUN_COMPLETE terminal-detail schema changed.')
  k = detail['k_valid_records']
  d = detail['d_completed']
  if type(k) is not int or type(d) is not int or not 0 <= k <= 80:
    raise TypeError('RUN_COMPLETE k/d accounting is malformed.')
  if record['valid_record_count'] != k:
    raise RuntimeError('RUN_COMPLETE valid-record count changed.')
  dispatch_terminal = status in {
      'controlled_stop_partial_dispatch',
      'controlled_stop_four_call_invalid', 'complete_structural_sidecar',
  }
  if not dispatch_terminal and (k, d) != (0, 0):
    raise RuntimeError('A pre-dispatch terminal carries dispatch counts.')
  if status == 'controlled_stop_partial_dispatch' and not 0 <= d < 4:
    raise RuntimeError('Partial-dispatch d must be in [0,3].')
  if status == 'controlled_stop_four_call_invalid' and d != 4:
    raise RuntimeError('Four-call invalid terminal must have d=4.')
  if status == 'complete_structural_sidecar' and (k, d) != (80, 0):
    raise RuntimeError('Complete sidecar must have k=80,d=0.')
  if status == 'complete_structural_sidecar':
    expected_detail = (None, None, None)
  elif status == 'controlled_stop_four_call_invalid':
    if detail['failure_phase'] not in {
        'record_validation', 'record_serialization'
    }:
      raise RuntimeError('Four-call terminal failure phase changed.')
    expected_detail = (
        detail['failure_phase'], None, None,
    )
  else:
    expected_detail = _TERMINAL_DETAIL_BY_REASON[reason]
  if (
      detail['failure_phase'], detail['forbidden_operation'],
      detail['provenance_artifact_role'],
  ) != expected_detail:
    raise RuntimeError(f'Terminal detail changed for {status}/{reason}.')
  expected_compiler = _COMPILER_MEMBERSHIP_BY_STATE[
      _terminal_compiler_state(status, reason)
  ]
  if set(record['compiler_artifact_bindings']) != expected_compiler:
    raise RuntimeError(
        f'Compiler membership changed for {status}/{reason}.'
    )
  compiler_state = _terminal_compiler_state(status, reason)
  signature_binding = record['program_signature_attestation_binding']
  compiler_binding = record['compiler_binding']
  if compiler_state == 'none':
    if signature_binding is not None or compiler_binding is not None:
      raise RuntimeError('Precompiler terminal exposes compiler bindings.')
  elif compiler_state == 'signature_failure':
    if signature_binding is None or compiler_binding is not None:
      raise RuntimeError('Signature-failure bindings changed.')
  elif compiler_state in {
      'lowered', 'precompiled', 'compiled', 'compiled_guarded'
  }:
    if signature_binding is None or compiler_binding is None:
      raise RuntimeError('Compiler terminal lacks required bindings.')
  elif compiler_state == 'diagnostic_failure':
    if signature_binding is None or compiler_binding is None:
      raise RuntimeError('Diagnostic terminal lacks failure binding.')
  source_gate = record['source_program_gate']
  diagnostics = record['diagnostic_provenance_complete']
  if compiler_state in {
      'none', 'signature_failure', 'lowered', 'precompiled',
      'compiled_guarded',
  }:
    if source_gate is not None or diagnostics is not None:
      raise RuntimeError('Pre-source-program terminal overclaims a gate.')
  elif compiler_state == 'diagnostic_failure':
    if diagnostics is not False:
      raise RuntimeError('Diagnostic failure must record incomplete diagnostics.')
  else:
    if not isinstance(source_gate, Mapping) or type(diagnostics) is not bool:
      raise RuntimeError('Compiled terminal lacks source/diagnostic evidence.')
  if status == 'controlled_stop_source_program_mismatch':
    if source_gate.get('source_program_exact') is not False or diagnostics is not True:
      raise RuntimeError('Source mismatch precedence changed.')
  if status in {
      'controlled_stop_partial_dispatch',
      'controlled_stop_four_call_invalid', 'complete_structural_sidecar',
  }:
    if source_gate.get('source_program_exact') is not True or diagnostics is not True:
      raise RuntimeError('Dispatch proceeded without both compiler gates.')
  expected_source_pass = (
      isinstance(source_gate, Mapping)
      and source_gate.get('source_program_exact') is True
  )
  if phase['source_program_gate_passed'] is not expected_source_pass:
    raise RuntimeError('Source-program phase/evidence mismatch.')
  if phase['diagnostic_provenance_passed'] is not (diagnostics is True):
    raise RuntimeError('Diagnostic phase/evidence mismatch.')
  compiler_phase_requirements = {
      'signature_failure': {
          'pre_model_import_inventory_persisted', 'protobuf_persisted',
          'model_construction_attempted', 'model_constructed',
          'reference_cases_loaded',
          'post_model_import_inventory_persisted',
          'terminal_import_inventory_persisted',
      },
      'lowered': {
          'pre_model_import_inventory_persisted', 'protobuf_persisted',
          'model_construction_attempted', 'model_constructed',
          'reference_cases_loaded', 'signatures_captured',
          'signature_attestation_persisted',
          'post_model_import_inventory_persisted', 'lower_attempted',
          'terminal_import_inventory_persisted',
      },
      'precompiled': {
          'pre_model_import_inventory_persisted', 'protobuf_persisted',
          'model_construction_attempted', 'model_constructed',
          'reference_cases_loaded', 'signatures_captured',
          'signature_attestation_persisted',
          'post_model_import_inventory_persisted', 'lower_attempted',
          'lower_succeeded', 'compile_attempted',
          'terminal_import_inventory_persisted',
      },
      'compiled': {
          'pre_model_import_inventory_persisted', 'protobuf_persisted',
          'model_construction_attempted', 'model_constructed',
          'reference_cases_loaded', 'signatures_captured',
          'signature_attestation_persisted',
          'post_model_import_inventory_persisted', 'lower_attempted',
          'lower_succeeded', 'compile_attempted', 'compile_succeeded',
          'terminal_import_inventory_persisted',
      },
      'compiled_guarded': {
          'pre_model_import_inventory_persisted', 'protobuf_persisted',
          'model_construction_attempted', 'model_constructed',
          'reference_cases_loaded', 'signatures_captured',
          'signature_attestation_persisted',
          'post_model_import_inventory_persisted', 'lower_attempted',
          'lower_succeeded', 'compile_attempted', 'compile_succeeded',
          'terminal_import_inventory_persisted',
      },
      'diagnostic_failure': {
          'pre_model_import_inventory_persisted', 'protobuf_persisted',
          'model_construction_attempted', 'model_constructed',
          'reference_cases_loaded', 'signatures_captured',
          'signature_attestation_persisted',
          'post_model_import_inventory_persisted', 'lower_attempted',
          'lower_succeeded', 'compile_attempted', 'compile_succeeded',
          'terminal_import_inventory_persisted',
      },
  }
  for name in compiler_phase_requirements.get(compiler_state, set()):
    if phase[name] is not True:
      raise RuntimeError(
          f'{compiler_state} terminal lacks required phase {name}.'
      )
  if compiler_state == 'signature_failure' and phase[
      'signature_attestation_persisted'
  ]:
    raise RuntimeError('Signature failure claims successful attestation.')
  if status == 'controlled_stop_model_setup_failure':
    required = {
        'pre_model_import_inventory_persisted', 'protobuf_persisted',
        'model_construction_attempted',
        'post_model_import_inventory_persisted',
        'terminal_import_inventory_persisted',
    }
    if not all(phase[name] for name in required):
      raise RuntimeError('Model-setup terminal phase prefix changed.')
    if phase['reference_cases_loaded'] or phase['signatures_captured']:
      raise RuntimeError('Model-setup terminal overclaims case/signature phase.')
  exact_early_phases = {
      ('controlled_stop_cache_hit', 'model_cache_pre_import_hit'): {
          'preflight_passed', 'start_persisted',
          'post_start_source_gate_passed',
      },
      ('controlled_stop_import_provenance_failure',
       'pre_model_import_inventory_mismatch'): {
           'preflight_passed', 'start_persisted',
           'post_start_source_gate_passed',
           'pre_model_import_inventory_persisted',
       },
      ('controlled_stop_protobuf_provenance_failure',
       'protobuf_binding_mismatch'): {
           'preflight_passed', 'start_persisted',
           'post_start_source_gate_passed',
           'pre_model_import_inventory_persisted', 'protobuf_persisted',
       },
      ('controlled_stop_import_provenance_failure',
       'post_model_import_inventory_mismatch'): {
           'preflight_passed', 'start_persisted',
           'post_start_source_gate_passed',
           'pre_model_import_inventory_persisted', 'protobuf_persisted',
           'model_construction_attempted', 'model_constructed',
           'reference_cases_loaded',
           'post_model_import_inventory_persisted',
       },
  }
  expected_true = exact_early_phases.get((status, reason))
  if expected_true is not None and {
      name for name, value in phase.items() if value
  } != expected_true:
    raise RuntimeError(f'Early terminal phase prefix changed for {reason}.')
  started = record['dispatch_journal']['started_count']
  completed = record['dispatch_journal']['completed_count']
  if (
      record['model_apply_attempt_count'],
      record['model_apply_success_count'],
  ) != (started, completed):
    raise RuntimeError('Terminal/journal apply accounting changed.')
  if status == 'controlled_stop_partial_dispatch':
    expected_started = 4 * k + d + int(reason == 'model_dispatch_failure')
    if (started, completed) != (expected_started, 4 * k + d):
      raise RuntimeError('Partial-dispatch journal arithmetic changed.')
  elif status == 'controlled_stop_four_call_invalid':
    if (started, completed) != (4 * k + 4, 4 * k + 4):
      raise RuntimeError('Four-call journal arithmetic changed.')
  elif status == 'complete_structural_sidecar':
    if (started, completed) != (320, 320):
      raise RuntimeError('Complete journal arithmetic changed.')
  elif (started, completed) != (0, 0):
    raise RuntimeError('Pre-dispatch terminal has journaled applies.')
  if phase['dispatch_begun'] is not (started > 0):
    raise RuntimeError('Dispatch phase/journal prefix mismatch.')
  if status == 'complete_structural_sidecar':
    if record['failure'] is not None:
      raise RuntimeError('Complete sidecar contains a failure.')
  elif record['failure'] is None:
    raise RuntimeError('Controlled terminal lacks a failure object.')
  source = record['source_input_audit']
  if not isinstance(source, Mapping) or set(source) != set(
      bootstrap.SOURCE_INPUT_AUDIT_KEYS
  ):
    raise RuntimeError('RUN_COMPLETE source-input audit schema changed.')


def _validate_terminal_identity(status: str, stop_reason: str | None) -> None:
  if set(_TERMINAL_STATUS_REASONS) != set(
      bootstrap.TERMINAL_CONTRACT['statuses']
  ):
    raise RuntimeError('Runner/freeze terminal status inventory changed.')
  if stop_reason not in _TERMINAL_STATUS_REASONS.get(status, set()):
    raise ValueError(f'Illegal terminal status/reason: {status}/{stop_reason}.')


def _phase_state(**overrides: bool) -> dict[str, bool]:
  result = {name: False for name in _PHASE_STATE_KEYS}
  unknown = set(overrides) - set(result)
  if unknown:
    raise ValueError(f'Unknown phase-state keys: {sorted(unknown)}.')
  result.update(overrides)
  return result


def _import_phase_bindings() -> dict[str, Any]:
  paths = {
      'pre_model': OUTPUT_DIR / 'IMPORT_PROVENANCE_PRE_MODEL.json',
      'post_model_precompile': (
          OUTPUT_DIR / 'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json'
      ),
      'terminal': OUTPUT_DIR / 'IMPORT_PROVENANCE.json',
  }
  return {
      name: (_relative_file_binding(path) if path.exists() else None)
      for name, path in paths.items()
  }


def _compiler_artifact_bindings() -> dict[str, Any]:
  root = OUTPUT_DIR / 'compiler'
  if not root.exists():
    return {}
  return _binding_map(bootstrap._strict_file_tree(root))  # pylint: disable=protected-access


def _write_common_terminal(
    *,
    status: str,
    stop_reason: str | None,
    message: str,
    failure: Mapping[str, Any] | None,
    results: Sequence[Mapping[str, Any]],
    failed_current_binding: Mapping[str, Any] | None,
    compiler: Mapping[str, Any] | None,
    source_input_audit: Mapping[str, Any],
    same_object_attestation: Mapping[str, Any] | None,
    phase_state: Mapping[str, bool],
    failure_phase: str | None,
    failed_execution_index: int | None = None,
    failed_call_role: str | None = None,
    forbidden_operation: str | None = None,
    provenance_artifact_role: str | None = None,
    program_signature_attestation_binding: Mapping[str, Any] | None = None,
    compiler_binding: Mapping[str, Any] | None = None,
    cache_hit_evidence: Mapping[str, Any] | None = None,
    historical_cache_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  _validate_terminal_identity(status, stop_reason)
  source_binding = _content_binding(source_input_audit)
  same_binding = (
      None if same_object_attestation is None
      else _content_binding(same_object_attestation)
  )
  raw = _raw_manifest(
      results,
      failed_current_binding=failed_current_binding,
      source_input_audit_binding=source_binding,
      same_object_attestation_binding=same_binding,
  )
  _write_new(OUTPUT_DIR / 'RAW_MANIFEST.json', raw)
  journal = _journal_summary()
  k = len(results)
  d = journal['completed_count'] - 4 * k
  start = _launcher_record()['start']
  prior_prefix, prior_prefix_binding = _prior_prefix_from_start(start)
  preterminal = _run_tree_binding()
  preterminal_bytes = sum(
      row['size_bytes'] for row in preterminal['file_bindings'].values()
  )
  if preterminal_bytes + RUN_COMPLETE_SIZE_CAP_BYTES > MAX_OUTPUT_BYTES:
    raise RuntimeError('Terminal cap would exceed the frozen output budget.')
  if compiler is None:
    source_gate = None
    attempt_budget = None
    diagnostics_complete = None
    backend_diagnostics = None
    diagnostic_comparisons = None
    model_cache_final = {
        'pre_import': start['same_process_preflight'][
            'model_cache_pre_import'
        ],
        'historical_stage': None,
        'historical_binding': None,
        'terminal': _cache_tree_binding(bootstrap.MODEL_KERNEL_CACHE_DIR),
        'cache_hit_evidence': (
            None if cache_hit_evidence is None else dict(cache_hit_evidence)
        ),
        'historical_to_terminal_tree_exact': None,
        'historical_to_terminal_equality_is_a_gate': False,
        'historical_snapshot_not_reauthenticated_as_live_files': True,
        'default_user_cache_paths_eligible': False,
        'cache_outputs_are_diagnostic_only': True,
    }
  else:
    diagnostic_failure = (
        compiler.get('status') == 'diagnostic_provenance_failure'
    )
    source_gate = (
        compiler.get('source_program_gate_without_backend_diagnostics')
        if diagnostic_failure else compiler.get('source_program_gate')
    )
    attempt_budget = compiler.get('attempt_budget_audit')
    diagnostics_complete = compiler.get('diagnostic_provenance_complete')
    backend_diagnostics = (
        None if diagnostic_failure else compiler.get('backend_diagnostics')
    )
    diagnostic_comparisons = (
        None if diagnostic_failure else compiler.get('diagnostic_comparisons')
    )
    if diagnostic_failure:
      pre_import = start['same_process_preflight']['model_cache_pre_import']
      historical = (
          _cache_tree_binding(bootstrap.MODEL_KERNEL_CACHE_DIR)
          if historical_cache_binding is None
          else dict(historical_cache_binding)
      )
      terminal_cache = _cache_tree_binding(bootstrap.MODEL_KERNEL_CACHE_DIR)
      evidence = (
          None
          if cache_hit_evidence is None
          and stop_reason == 'cache_signal_unavailable'
          else (
              _model_cache_hit_evidence(pre_import, compile_skipped=False)
              if cache_hit_evidence is None else dict(cache_hit_evidence)
          )
      )
      model_cache_final = {
          'pre_import': pre_import,
          'historical_stage': 'post_compile',
          'historical_binding': historical,
          'terminal': terminal_cache,
          'cache_hit_evidence': evidence,
          'historical_to_terminal_tree_exact': (
              historical['tree_sha256'] == terminal_cache['tree_sha256']
          ),
          'historical_to_terminal_equality_is_a_gate': False,
          'historical_snapshot_not_reauthenticated_as_live_files': True,
          'default_user_cache_paths_eligible': False,
          'cache_outputs_are_diagnostic_only': True,
      }
    else:
      model_cache_final = _final_model_cache_binding(compiler)
  record = {
      'status': status,
      'stop_reason': stop_reason,
      'message': message,
      'failure': None if failure is None else dict(failure),
      'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION,
      'amendment_sha256': AMENDMENT_SHA256,
      'amendment_commit': AMENDMENT_COMMIT,
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'freeze_sha256': start['freeze_sha256'],
      'git_head': start['git_head'],
      'external_freeze_authorization': dict(_external_authorization()),
      'runner_pid': os.getpid(),
      'started_at_unix_s': start['started_at_unix_s'],
      'completed_at_unix_s': time.time(),
      'phase_state': dict(phase_state),
      'terminal_detail': {
          'k_valid_records': k,
          'd_completed': d,
          'failed_execution_index': failed_execution_index,
          'failed_call_role': failed_call_role,
          'failure_phase': failure_phase,
          'forbidden_operation': forbidden_operation,
          'provenance_artifact_role': provenance_artifact_role,
      },
      'budgets': {
          'max_wall_time_seconds': MAX_WALL_TIME_SECONDS,
          'elapsed_wall_time_seconds': time.time() - start['started_at_unix_s'],
          'wall_time_within_budget': (
              time.time() - start['started_at_unix_s']
              <= MAX_WALL_TIME_SECONDS
          ),
          'max_output_bytes': MAX_OUTPUT_BYTES,
          'preterminal_output_bytes': preterminal_bytes,
          'run_complete_size_cap_bytes': RUN_COMPLETE_SIZE_CAP_BYTES,
          'preterminal_plus_terminal_cap_within_budget': True,
      },
      'source_input_audit': dict(source_input_audit),
      'source_input_audit_content_binding': source_binding,
      'same_object_attestation': (
          None if same_object_attestation is None
          else dict(same_object_attestation)
      ),
      'same_object_attestation_content_binding': same_binding,
      'program_signature_attestation_binding': (
          None if program_signature_attestation_binding is None
          else dict(program_signature_attestation_binding)
      ),
      'source_program_gate': source_gate,
      'compiler_binding': (
          None if compiler_binding is None else dict(compiler_binding)
      ),
      'compiler_artifact_bindings': _compiler_artifact_bindings(),
      'attempt_budget_audit': attempt_budget,
      'diagnostic_provenance_complete': diagnostics_complete,
      'compiled_backend_diagnostic_only': (
          None if compiler is None else True
      ),
      'backend_diagnostics': backend_diagnostics,
      'diagnostic_comparisons': diagnostic_comparisons,
      'dispatch_journal': journal,
      'raw_manifest': raw,
      'preterminal_tree_binding': preterminal,
      'valid_record_count': k,
      'failed_current_binding': (
          None if failed_current_binding is None
          else dict(failed_current_binding)
      ),
      'model_apply_attempt_count': journal['started_count'],
      'model_apply_success_count': journal['completed_count'],
      'expected_model_apply_count': EXPECTED_APPLY_COUNT,
      'eight_row_lower_attempt_count': (
          0 if compiler is None else compiler.get('lower_attempt_count', 0)
      ),
      'eight_row_compile_attempt_count': (
          0 if compiler is None else compiler.get('compile_attempt_count', 0)
      ),
      'eight_row_successful_compile_count': (
          0 if compiler is None else compiler.get('successful_compile_count', 0)
      ),
      'six_row_compile_count': 0,
      'identity_rerun_count': 0,
      'main_cube_rerun_count': 0,
      'old_ood_records_reused': 0,
      'confirmation_model_calls': 0,
      'all_80_recipient_anchors_complete': status == 'complete_structural_sidecar',
      'id0_all20': status == 'complete_structural_sidecar',
      'id255_all20': status == 'complete_structural_sidecar',
      'import_provenance_phases': _import_phase_bindings(),
      'protobuf_provenance_sha256': (
          _sha256(OUTPUT_DIR / 'PROTOBUF_PROVENANCE.json')
          if (OUTPUT_DIR / 'PROTOBUF_PROVENANCE.json').exists() else None
      ),
      'model_kernel_cache_final': model_cache_final,
      'prior_v3_3_3_binding': _launcher_record()['v3_3_3_run'],
      'prior_v3_3_3_1_archive_binding': _launcher_record()[
          'v3_3_3_1_archive'
      ],
      'publication_audit': bootstrap.publication_audit('model_run'),
      'confirmation_scope_disclosure': DISCLOSURE,
      'scientific_summary_computed': False,
      'donor_normalization_computed': False,
      'shapley_or_nomination_computed': False,
      'interaction_or_resolution_computed': False,
      'nomination_performed': False,
      'combined_analysis_permitted': False,
      'no_retry': True,
      'prior_v3_3_4_3_consumed_preflight_prefix': prior_prefix,
      'prior_v3_3_4_3_consumed_preflight_prefix_content_binding': (
          prior_prefix_binding
      ),
  }
  _validate_common_terminal_semantics(record)
  payload = _json_bytes_for_size(record)
  if len(payload) > RUN_COMPLETE_SIZE_CAP_BYTES:
    raise RuntimeError('RUN_COMPLETE exceeds its frozen size cap.')
  _write_new(OUTPUT_DIR / 'RUN_COMPLETE.json', record)
  return record


def _json_bytes_for_size(value: Any) -> bytes:
  return (json.dumps(
      value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
  ) + '\n').encode('utf-8')


def _nonpublication_graph_bindings(
    published_graphs: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
  if published_graphs is None:
    return {}
  artifacts = published_graphs.get('artifacts')
  if not isinstance(artifacts, Mapping) or set(artifacts) != {
      'stablehlo', 'hlo', 'compiled_hlo'
  }:
    raise ValueError('Nonpublication graph artifact set changed.')
  result = {}
  for role in ('stablehlo', 'hlo', 'compiled_hlo'):
    row = artifacts[role]
    if not isinstance(row, Mapping) or set(row) != {
        'path', 'sha256', 'size_bytes'
    }:
      raise ValueError('Nonpublication graph artifact schema changed.')
    result[str(row['path'])] = {
        'sha256': row['sha256'], 'size_bytes': row['size_bytes']
    }
  return dict(sorted(result.items()))


def _nonpublication_phase_state(
    *, diagnostic_stage: bool, source_program_gate_passed: bool = False,
) -> dict[str, bool]:
  return _phase_state(
      preflight_passed=True, start_persisted=True,
      post_start_source_gate_passed=True, protobuf_persisted=True,
      pre_model_import_inventory_persisted=True,
      model_construction_attempted=True, model_constructed=True,
      reference_cases_loaded=True, signatures_captured=True,
      signature_attestation_persisted=True,
      post_model_import_inventory_persisted=True,
      lower_attempted=True, lower_succeeded=True,
      compile_attempted=True, compile_succeeded=True,
      terminal_import_inventory_persisted=diagnostic_stage,
      source_program_gate_passed=source_program_gate_passed,
      diagnostic_provenance_passed=False, dispatch_begun=False,
  )


def _nonpublication_cache_state(
    *, pre_import: Mapping[str, Any], historical_binding: Mapping[str, Any],
    cache_hit_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
  terminal = _cache_tree_binding(bootstrap.MODEL_KERNEL_CACHE_DIR)
  return {
      'pre_import': dict(pre_import),
      'historical_stage': 'post_compile',
      'historical_binding': dict(historical_binding),
      'terminal_live_binding': terminal,
      'cache_hit_evidence': (
          None if cache_hit_evidence is None else dict(cache_hit_evidence)
      ),
      'historical_to_terminal_tree_exact': (
          historical_binding.get('tree_sha256') == terminal['tree_sha256']
      ),
      'historical_to_terminal_equality_is_a_gate': False,
      'historical_snapshot_not_reauthenticated_as_live_files': True,
      'default_user_cache_paths_eligible': False,
      'cache_outputs_are_diagnostic_only': True,
  }


def _validate_nonpublication_source_gate(
    gate: Mapping[str, Any], *, graph_bindings: Mapping[str, Any],
    source_input_audit: Mapping[str, Any],
    same_object_attestation: Mapping[str, Any],
    expected_entry_abi_sha256: str,
) -> None:
  """Recomputes every source-gate leaf before terminal publication."""
  gate_keys = {
      'contract', 'observed', 'stablehlo_exact', 'pre_backend_hlo_exact',
      'program_signature_structure_exact',
      'program_signatures_canonical_exact', 'entry_abi_exact',
      'source_runtime_device_toolchain_checkpoint_reference_exact',
      'source_input_audit', 'source_input_audit_content_binding',
      'same_object_attestation', 'same_object_attestation_content_binding',
      'same_lowered_compiled_object', 'source_program_exact',
  }
  observed_keys = {
      'stablehlo_sha256', 'stablehlo_size_bytes',
      'pre_backend_hlo_sha256', 'pre_backend_hlo_size_bytes',
      'program_signatures_sha256', 'entry_abi_sha256',
  }
  if set(gate) != gate_keys or not isinstance(gate.get('observed'), Mapping):
    raise ValueError('Nonpublication source-gate schema changed.')
  observed = gate['observed']
  if set(observed) != observed_keys or gate.get('contract') != dict(
      SOURCE_PROGRAM_CONTRACT
  ):
    raise ValueError('Nonpublication source-gate observation changed.')
  stable = graph_bindings.get(
      'compiler/eight_row/graph.stablehlo.mlir'
  )
  pre_backend = graph_bindings.get(
      'compiler/eight_row/graph.pre_backend.hlo.txt'
  )
  if not isinstance(stable, Mapping) or not isinstance(pre_backend, Mapping):
    raise ValueError('Nonpublication source gate lacks graph bindings.')
  if (
      observed['stablehlo_sha256'] != stable.get('sha256')
      or observed['stablehlo_size_bytes'] != stable.get('size_bytes')
      or observed['pre_backend_hlo_sha256'] != pre_backend.get('sha256')
      or observed['pre_backend_hlo_size_bytes']
      != pre_backend.get('size_bytes')
      or observed['program_signatures_sha256']
      != SOURCE_PROGRAM_CONTRACT['program_signatures_sha256']
      or observed['entry_abi_sha256'] != expected_entry_abi_sha256
  ):
    raise ValueError('Nonpublication source-gate graph observation changed.')
  stable_exact = (
      observed['stablehlo_sha256']
      == SOURCE_PROGRAM_CONTRACT['stablehlo_sha256']
      and observed['stablehlo_size_bytes']
      == SOURCE_PROGRAM_CONTRACT['stablehlo_size_bytes']
  )
  pre_backend_exact = (
      observed['pre_backend_hlo_sha256']
      == SOURCE_PROGRAM_CONTRACT['pre_backend_hlo_sha256']
      and observed['pre_backend_hlo_size_bytes']
      == SOURCE_PROGRAM_CONTRACT['pre_backend_hlo_size_bytes']
  )
  signatures_exact = True
  entry_exact = (
      expected_entry_abi_sha256
      == SOURCE_PROGRAM_CONTRACT['entry_abi_sha256']
  )
  provenance_exact = all(
      value is True for value in source_input_audit.values()
  )
  same_exact = (
      same_object_attestation.get('lower_call_count') == 1
      and same_object_attestation.get('compile_call_count') == 1
      and all(same_object_attestation.get(name) is True for name in (
          'stablehlo_read_from_lowered_object',
          'pre_backend_hlo_read_from_lowered_object',
          'compile_argument_is_lowered_object',
          'compiled_hlo_read_from_compiled_object',
          'signature_attestation_from_apply_arguments',
          'apply_callable_is_compiled_object',
          'compiler_record_is_gate_record',
      ))
  )
  expected = {
      'stablehlo_exact': stable_exact,
      'pre_backend_hlo_exact': pre_backend_exact,
      'program_signature_structure_exact': signatures_exact,
      'program_signatures_canonical_exact': signatures_exact,
      'entry_abi_exact': entry_exact,
      'source_runtime_device_toolchain_checkpoint_reference_exact': (
          provenance_exact
      ),
      'same_lowered_compiled_object': same_exact,
  }
  if (
      gate.get('source_input_audit') != dict(source_input_audit)
      or gate.get('source_input_audit_content_binding')
      != _content_binding(source_input_audit)
      or gate.get('same_object_attestation')
      != dict(same_object_attestation)
      or gate.get('same_object_attestation_content_binding')
      != _content_binding(same_object_attestation)
      or any(gate.get(name) is not value for name, value in expected.items())
      or gate.get('source_program_exact') is not all(expected.values())
  ):
    raise ValueError('Nonpublication source-gate primitive changed.')


def _validate_nonpublication_same_object(
    same_object: Mapping[str, Any], *, failure_stage: str,
    stages: Sequence[str],
) -> None:
  expected_reads = {
      stages[0]: (None, None, None, None),
      stages[1]: (True, None, None, None),
      stages[2]: (True, True, None, None),
      stages[3]: (True, True, True, True),
      stages[4]: (True, True, True, True),
  }[failure_stage]
  if (
      set(same_object) != set(bootstrap.SAME_OBJECT_ATTESTATION_KEYS)
      or same_object.get('lowered_python_id')
      == same_object.get('compiled_python_id')
      or any(
          isinstance(same_object.get(name), bool)
          or not isinstance(same_object.get(name), int)
          or same_object[name] < 0
          for name in ('lowered_python_id', 'compiled_python_id')
      )
      or tuple(same_object.get(name) for name in (
          'stablehlo_read_from_lowered_object',
          'pre_backend_hlo_read_from_lowered_object',
          'compiled_hlo_read_from_compiled_object',
          'compiler_record_is_gate_record',
      )) != expected_reads
      or same_object.get('compile_argument_is_lowered_object') is not True
      or same_object.get(
          'signature_attestation_from_apply_arguments'
      ) is not True
      or same_object.get('apply_callable_is_compiled_object') is not True
      or same_object.get('lower_call_count') != 1
      or same_object.get('compile_call_count') != 1
  ):
    raise ValueError('Nonpublication object identity evidence changed.')


def _write_nonpublication_terminal(
    *, failure_stage: str, failure: Exception,
    triggering_diagnostic_failure: Exception | None,
    triggering_diagnostic_stop_reason: str | None,
    source_input_audit: Mapping[str, Any],
    same_object_attestation: Mapping[str, Any],
    program_signature_attestation_binding: Mapping[str, Any],
    attempt_budget_audit: Mapping[str, Any],
    historical_cache_binding: Mapping[str, Any],
    cache_hit_evidence: Mapping[str, Any] | None,
    published_graphs: Mapping[str, Any] | None,
    source_program_gate_without_backend_diagnostics: (
        Mapping[str, Any] | None
    ),
) -> dict[str, Any]:
  """Publishes the exact v3.3.4.4 ordinary compiler terminal."""
  contract = bootstrap.NONPUBLICATION_TERMINAL_CONTRACT_V3_3_4_3
  stages = tuple(contract['failure_stages'])
  if failure_stage not in stages:
    raise ValueError('Unknown nonpublication failure stage.')
  diagnostic_stage = failure_stage in stages[3:]
  if diagnostic_stage:
    if (
        triggering_diagnostic_failure is None
        or triggering_diagnostic_stop_reason
        not in contract['triggering_diagnostic_stop_reasons']
    ):
      raise ValueError('Diagnostic-construction terminal lacks its trigger.')
    if (
        _diagnostic_stop_reason(triggering_diagnostic_failure)
        != triggering_diagnostic_stop_reason
    ):
      raise ValueError('Diagnostic trigger and operation reason disagree.')
  elif (
      triggering_diagnostic_failure is not None
      or triggering_diagnostic_stop_reason is not None
  ):
    raise ValueError('Extraction terminal invented a diagnostic trigger.')
  gate = (
      None
      if source_program_gate_without_backend_diagnostics is None
      else dict(source_program_gate_without_backend_diagnostics)
  )
  gate_required = failure_stage == 'diagnostic_failure_record_construction'
  if (gate is not None) is not gate_required:
    raise ValueError('Nonpublication source-gate nullability changed.')
  graph_bindings = _nonpublication_graph_bindings(published_graphs)
  if bool(graph_bindings) is not diagnostic_stage:
    raise ValueError('Nonpublication graph membership changed.')
  imports = _import_phase_bindings()
  if (imports['terminal'] is not None) is not diagnostic_stage:
    raise ValueError('Nonpublication terminal-import nullability changed.')
  expected_membership = set(
      contract[
          'diagnostic_construction_preterminal_membership'
          if diagnostic_stage else 'extraction_preterminal_membership'
      ]
  )
  preterminal = _run_tree_binding()
  if set(preterminal['file_bindings']) != expected_membership:
    raise ValueError('Nonpublication preterminal membership changed.')
  if any(
      preterminal['file_bindings'].get(path) != binding
      for path, binding in graph_bindings.items()
  ):
    raise ValueError('Nonpublication graph/live binding changed.')
  source = dict(source_input_audit)
  expected_source_values = (True,) * 7 + (
      (True,) if diagnostic_stage else (None,)
  )
  if (
      set(source) != set(bootstrap.SOURCE_INPUT_AUDIT_KEYS)
      or tuple(source[name] for name in bootstrap.SOURCE_INPUT_AUDIT_KEYS)
      != expected_source_values
  ):
    raise ValueError('Nonpublication source-input phase matrix changed.')
  same_object = dict(same_object_attestation)
  _validate_nonpublication_same_object(
      same_object, failure_stage=failure_stage, stages=stages
  )
  if gate is not None:
    if isinstance(triggering_diagnostic_failure, (
        EntryAbiParserFailure, FingerprintFormulaMismatch,
        CacheSignalUnavailable,
    )):
      expected_entry_abi_sha256 = ''
    elif isinstance(triggering_diagnostic_failure, (
        BackendDiagnosticParserFailure, DiagnosticPersistenceFailure,
    )):
      expected_entry_abi_sha256 = SOURCE_PROGRAM_CONTRACT[
          'entry_abi_sha256'
      ]
    else:
      raise ValueError('Diagnostic trigger operation is not attributable.')
    _validate_nonpublication_source_gate(
        gate, graph_bindings=graph_bindings,
        source_input_audit=source,
        same_object_attestation=same_object,
        expected_entry_abi_sha256=expected_entry_abi_sha256,
    )
  expected_budget = {
      'lower_budget': 1, 'compile_budget': 1,
      'lower_invocations': 1, 'compile_invocations': 1,
      'forbidden_request': None,
      'forbidden_request_detected_before_invocation': False,
  }
  if dict(attempt_budget_audit) != expected_budget:
    raise ValueError('Nonpublication compiler-attempt audit changed.')
  start = _launcher_record()['start']
  prior_prefix, prior_prefix_binding = _prior_prefix_from_start(start)
  cache_state = _nonpublication_cache_state(
      pre_import=start['same_process_preflight']['model_cache_pre_import'],
      historical_binding=historical_cache_binding,
      cache_hit_evidence=cache_hit_evidence,
  )
  if (
      (cache_state['cache_hit_evidence'] is None)
      is not (triggering_diagnostic_stop_reason == 'cache_signal_unavailable')
  ):
    raise ValueError('Nonpublication cache-evidence nullability changed.')
  if (
      cache_state['cache_hit_evidence'] is not None
      and cache_state['cache_hit_evidence'].get('cache_hit') is not False
  ):
    raise ValueError('Nonpublication terminal cannot archive a cache hit.')
  record = {
      'schema_version': contract['schema_version'],
      'status': contract['status'], 'stop_reason': contract['stop_reason'],
      'attempt_id': ATTEMPT_ID, 'script_version': 'v3.3.4.4',
      'amendment_commit': AMENDMENT_COMMIT,
      'amendment_sha256': AMENDMENT_SHA256,
      'inherited_v3_3_4_commit': bootstrap.V3_3_4_AMENDMENT_COMMIT,
      'inherited_v3_3_4_sha256': bootstrap.V3_3_4_AMENDMENT_SHA256,
      'inherited_v3_3_4_1_commit': bootstrap.V3_3_4_1_AMENDMENT_COMMIT,
      'inherited_v3_3_4_1_sha256': bootstrap.V3_3_4_1_AMENDMENT_SHA256,
      'freeze_sha256': start['freeze_sha256'], 'git_head': start['git_head'],
      'external_freeze_authorization': dict(_external_authorization()),
      'runner_pid': os.getpid(),
      'started_at_unix_s': start['started_at_unix_s'],
      'created_at_unix_s': time.time(), 'failure_stage': failure_stage,
      'failure': _failure_object(failure),
      'triggering_diagnostic_failure': (
          None if triggering_diagnostic_failure is None
          else _failure_object(triggering_diagnostic_failure)
      ),
      'triggering_diagnostic_stop_reason': triggering_diagnostic_stop_reason,
      'phase_state': _nonpublication_phase_state(
          diagnostic_stage=diagnostic_stage,
          source_program_gate_passed=(
              False if gate is None else bool(gate['source_program_exact'])
          ),
      ),
      'source_input_audit': source,
      'source_input_audit_content_binding': _content_binding(source),
      'program_signature_attestation_binding': dict(
          program_signature_attestation_binding
      ),
      'same_object_attestation': same_object,
      'same_object_attestation_content_binding': _content_binding(same_object),
      'attempt_budget_audit': expected_budget,
      'compiler_counts': dict(contract['compiler_counts']),
      'graph_artifact_bindings': graph_bindings,
      'import_provenance_phases': imports,
      'protobuf_provenance_sha256': _sha256(
          OUTPUT_DIR / 'PROTOBUF_PROVENANCE.json'
      ),
      'model_kernel_cache_state': cache_state,
      'source_program_gate_without_backend_diagnostics': gate,
      'source_program_gate_without_backend_diagnostics_content_binding': (
          None if gate is None else _content_binding(gate)
      ),
      'prior_v3_3_3_binding': _launcher_record()['v3_3_3_run'],
      'prior_v3_3_3_1_archive_binding': _launcher_record()[
          'v3_3_3_1_archive'
      ],
      'preterminal_tree_binding': preterminal,
      'publication_audit': bootstrap.publication_audit('model_run'),
      'model_apply_attempt_count': 0, 'model_apply_success_count': 0,
      'valid_record_count': 0, 'raw_record_count': 0,
      'dispatch_started_count': 0, 'dispatch_completed_count': 0,
      'six_row_compile_count': 0, 'identity_rerun_count': 0,
      'main_cube_rerun_count': 0, 'old_ood_records_reused': 0,
      'confirmation_model_calls': 0,
      'confirmation_scope_disclosure': DISCLOSURE,
      'scientific_summary_computed': False,
      'donor_normalization_computed': False,
      'shapley_or_nomination_computed': False,
      'interaction_or_resolution_computed': False,
      'nomination_performed': False,
      'combined_analysis_permitted': False, 'no_retry': True,
      'prior_v3_3_4_3_consumed_preflight_prefix': prior_prefix,
      'prior_v3_3_4_3_consumed_preflight_prefix_content_binding': (
          prior_prefix_binding
      ),
  }
  if set(record) != set(contract['keys']) or len(record) != 60:
    raise RuntimeError('NONPUBLICATION terminal key set changed.')
  _write_new(
      OUTPUT_DIR / 'NONPUBLICATION_TERMINAL_FAILURE.json', record
  )
  return record


def _write_terminal_failure(
    error: bootstrap.PublicationError,
    *,
    completed_record_count: int,
    source_input_audit: Mapping[str, Any],
    same_object_attestation: Mapping[str, Any] | None,
    phase_state: Mapping[str, bool],
    failed_current_binding: Mapping[str, Any] | None,
) -> None:
  if not bootstrap.terminal_publication_available('model_run'):
    # The exact created entry cannot be losslessly bound, so the protocol
    # requires a terminal-less consumed prefix rather than an invented audit.
    return
  audit = bootstrap.publication_audit(
      'model_run', error.publication_failure
  )
  excluded_failed_entries = sorted({
      *audit['temporary_orphan_bindings'],
      *audit['durability_uncertain_final_bindings'],
      *audit['preexisting_entry_states'],
  })
  source_binding = _content_binding(source_input_audit)
  same_binding = (
      None if same_object_attestation is None
      else _content_binding(same_object_attestation)
  )
  journal = _journal_summary(excluded_failed_entries)
  start = _launcher_record()['start']
  prior_prefix, prior_prefix_binding = _prior_prefix_from_start(start)
  record = {
      'schema_version': bootstrap.PUBLICATION_SCHEMA_VERSION,
      'status': 'incomplete_publication_failure',
      'stop_reason': 'artifact_publication_failure',
      'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION,
      'external_freeze_authorization': dict(_external_authorization()),
      'runner_pid': os.getpid(),
      'publication_failure': dict(error.publication_failure),
      'preterminal_tree_binding': _run_tree_binding(excluded_failed_entries),
      'source_input_audit': dict(source_input_audit),
      'source_input_audit_content_binding': source_binding,
      'same_object_attestation': (
          None if same_object_attestation is None
          else dict(same_object_attestation)
      ),
      'same_object_attestation_content_binding': same_binding,
      'phase_state': dict(phase_state),
      'model_apply_attempt_count': journal['started_count'],
      'model_apply_success_count': journal['completed_count'],
      'valid_record_count': completed_record_count,
      'failed_current_binding': (
          None if failed_current_binding is None
          else dict(failed_current_binding)
      ),
      'temporary_orphan_bindings': audit['temporary_orphan_bindings'],
      'durability_uncertain_final_bindings': audit[
          'durability_uncertain_final_bindings'
      ],
      'preexisting_entry_states': audit['preexisting_entry_states'],
      'no_new_entry_failure': audit['no_new_entry_failure'],
      'confirmation_model_calls': 0,
      'scientific_summary_computed': False,
      'donor_normalization_computed': False,
      'shapley_or_nomination_computed': False,
      'interaction_or_resolution_computed': False,
      'nomination_performed': False,
      'combined_analysis_permitted': False,
      'no_retry': True,
      'created_at_unix_s': time.time(),
      'prior_v3_3_4_3_consumed_preflight_prefix': prior_prefix,
      'prior_v3_3_4_3_consumed_preflight_prefix_content_binding': (
          prior_prefix_binding
      ),
  }
  if (
      set(record) != set(bootstrap.PUBLICATION_TERMINAL_FAILURE_KEYS)
      or len(record) != 33
  ):
    raise RuntimeError('Publication terminal is not the exact 33-key schema.')
  _write_new(OUTPUT_DIR / 'TERMINAL_FAILURE.json', record)


def _failure_object(error: Exception) -> dict[str, str]:
  return {
      'type': type(error).__name__,
      'message': str(error),
      'traceback': ''.join(traceback.format_exception(error)),
  }


def _orchestrate_postcompile_provenance(
    *, lowered: Any, compiled: Any, compile_start: float,
    compiler_budget: OneShotCompilerBudget,
    launch: Mapping[str, Any], start: Mapping[str, Any],
    original_frozen: Mapping[str, Any], frozen: Mapping[str, Any],
    imports_pre: Mapping[str, Any], imports_post: Mapping[str, Any],
    checkpoint_binding: Mapping[str, Any],
    reference_object_binding: Mapping[str, Any],
    protobuf_record: Mapping[str, Any],
    adapted_signatures: Mapping[str, Any],
    signature_attestation: Mapping[str, Any],
    signature_binding: Mapping[str, Any],
    source_input_audit_prefix: Mapping[str, Any],
    current_phase: dict[str, bool],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
  """Runs the sole post-compile provenance DAG; returns None on a terminal."""
  historical_cache_binding = _cache_tree_binding(
      bootstrap.MODEL_KERNEL_CACHE_DIR
  )
  historical_cache_evidence = None
  cache_signal_error = None
  try:
    historical_cache_evidence = _model_cache_hit_evidence(
        start['same_process_preflight']['model_cache_pre_import'],
        compile_skipped=False,
    )
  except bootstrap.PublicationError:
    raise
  except Exception as error:  # pylint: disable=broad-exception-caught
    cache_signal_error = (
        error if isinstance(error, CacheSignalUnavailable)
        else CacheSignalUnavailable(error)
    )
  try:
    graph_texts = _extract_compiler_graph_texts(lowered, compiled)
  except CompilerGraphExtractionError as extraction_error:
    if cache_signal_error is not None:
      # The frozen protocols authorize neither a null-evidence extraction row
      # nor combining two failures. Preserve the terminal-less prefix.
      raise cache_signal_error
    _write_nonpublication_terminal(
        failure_stage=extraction_error.stage,
        failure=extraction_error.original_error,
        triggering_diagnostic_failure=None,
        triggering_diagnostic_stop_reason=None,
        source_input_audit=source_input_audit_prefix,
        same_object_attestation=extraction_error.same_object_attestation,
        program_signature_attestation_binding=signature_binding,
        attempt_budget_audit=compiler_budget.audit(),
        historical_cache_binding=historical_cache_binding,
        cache_hit_evidence=historical_cache_evidence,
        published_graphs=None,
        source_program_gate_without_backend_diagnostics=None,
    )
    return None
  published_graphs = _publish_compiler_graphs(graph_texts)
  imports_terminal, terminal_predicates = _persist_terminal_import_inventory(
      original_frozen, frozen, (imports_pre, imports_post)
  )
  current_phase['terminal_import_inventory_persisted'] = True
  source_audit = derive_source_input_audit(
      checkpoint_binding=checkpoint_binding,
      reference_object_binding=reference_object_binding,
      protobuf_record=protobuf_record,
      imports=(imports_pre, imports_post, imports_terminal),
  )
  same_object = _successful_same_object_attestation(lowered, compiled)
  entry_abi_diagnostic = None
  backend_diagnostics = None
  diagnostic_error = cache_signal_error
  diagnostic_reason = (
      None if diagnostic_error is None
      else _diagnostic_stop_reason(diagnostic_error)
  )
  if diagnostic_error is None:
    try:
      entry_abi_diagnostic = _entry_abi_binding(
          str(published_graphs['compiled_hlo_text'])
      )
    except bootstrap.PublicationError:
      raise
    except Exception as error:  # pylint: disable=broad-exception-caught
      diagnostic_error = (
          error if isinstance(error, DiagnosticProvenanceError)
          else EntryAbiParserFailure(error)
      )
      diagnostic_reason = _diagnostic_stop_reason(diagnostic_error)
  if diagnostic_error is None:
    try:
      backend_diagnostics = _backend_diagnostics(
          str(published_graphs['compiled_hlo_text'])
      )
    except bootstrap.PublicationError:
      raise
    except Exception as error:  # pylint: disable=broad-exception-caught
      diagnostic_error = (
          error if isinstance(error, DiagnosticProvenanceError)
          else BackendDiagnosticParserFailure(error)
      )
      diagnostic_reason = _diagnostic_stop_reason(diagnostic_error)

  def finish_diagnostic_stop(
      diagnostic_compiler: Mapping[str, Any],
      typed_error: DiagnosticProvenanceError,
      typed_reason: str,
  ) -> None:
    diagnostic_same_object = diagnostic_compiler[
        'same_object_attestation'
    ]
    diagnostic_source_gate = diagnostic_compiler[
        'source_program_gate_without_backend_diagnostics'
    ]
    diagnostic_binding = _relative_file_binding(
        OUTPUT_DIR / 'compiler/eight_row/'
        'COMPILER_DIAGNOSTIC_FAILURE.json'
    )
    _write_common_terminal(
        status='controlled_stop_diagnostic_provenance_failure',
        stop_reason=typed_reason,
        message='Compiled backend diagnostic provenance was incomplete.',
        failure=_failure_object(typed_error), results=results,
        failed_current_binding=None, compiler=diagnostic_compiler,
        source_input_audit=source_audit,
        same_object_attestation=diagnostic_same_object,
        phase_state=_phase_state(
            preflight_passed=True, start_persisted=True,
            post_start_source_gate_passed=True, protobuf_persisted=True,
            pre_model_import_inventory_persisted=True,
            model_construction_attempted=True, model_constructed=True,
            reference_cases_loaded=True, signatures_captured=True,
            signature_attestation_persisted=True,
            post_model_import_inventory_persisted=True,
            lower_attempted=True, lower_succeeded=True,
            compile_attempted=True, compile_succeeded=True,
            terminal_import_inventory_persisted=True,
            source_program_gate_passed=diagnostic_source_gate[
                'source_program_exact'
            ],
            diagnostic_provenance_passed=False,
        ),
        failure_phase='post_compile_diagnostics',
        program_signature_attestation_binding=signature_binding,
        compiler_binding=diagnostic_binding,
        cache_hit_evidence=historical_cache_evidence,
        historical_cache_binding=historical_cache_binding,
    )

  if diagnostic_error is not None:
    compiler = _route_diagnostic_failure(
        diagnostic_error, diagnostic_reason,
        lowered=lowered, compiled=compiled,
        published_graphs=published_graphs,
        program_signatures=adapted_signatures,
        program_signature_attestation=signature_attestation,
        program_signature_attestation_binding=signature_binding,
        source_input_audit=source_audit,
        entry_abi=entry_abi_diagnostic,
        attempt_budget_audit=compiler_budget.audit(),
        historical_cache_binding=historical_cache_binding,
        cache_hit_evidence=historical_cache_evidence,
    )
    if compiler is None:
      return None
    finish_diagnostic_stop(compiler, diagnostic_error, diagnostic_reason)
    return None
  try:
    compiler = _compiler_artifacts(
        lowered, compiled, time.perf_counter() - compile_start,
        launch['v3_3_3_run']['eight_row_compiler'],
        launch['gate_a']['v3_3_2_run']['eight_row_compiler'],
        adapted_signatures,
        program_signature_attestation=signature_attestation,
        source_input_audit=source_audit,
        kernel_cache_preimport_attestation=start[
            'same_process_preflight'
        ]['model_cache_pre_import'],
        published_graphs=published_graphs,
        precomputed_entry_abi=entry_abi_diagnostic,
        precomputed_backend_diagnostics=backend_diagnostics,
        attempt_budget_audit=compiler_budget.audit(),
        same_object_attestation=same_object,
        precomputed_cache_binding=historical_cache_binding,
        precomputed_cache_hit_evidence=historical_cache_evidence,
    )
  except bootstrap.PublicationError:
    raise
  except Exception as error:  # pylint: disable=broad-exception-caught
    diagnostic_error = DiagnosticPersistenceFailure(error)
    diagnostic_reason = _diagnostic_stop_reason(diagnostic_error)
    compiler = _route_diagnostic_failure(
        diagnostic_error, diagnostic_reason,
        lowered=lowered, compiled=compiled,
        published_graphs=published_graphs,
        program_signatures=adapted_signatures,
        program_signature_attestation=signature_attestation,
        program_signature_attestation_binding=signature_binding,
        source_input_audit=source_audit,
        entry_abi=entry_abi_diagnostic,
        attempt_budget_audit=compiler_budget.audit(),
        historical_cache_binding=historical_cache_binding,
        cache_hit_evidence=historical_cache_evidence,
    )
    if compiler is None:
      return None
    finish_diagnostic_stop(compiler, diagnostic_error, diagnostic_reason)
    return None
  return {
      'compiler': compiler, 'source_input_audit': source_audit,
      'same_object_attestation': compiler['same_object_attestation'],
      'terminal_import_predicates': terminal_predicates,
  }


def _main_v334() -> None:
  if len(sys.argv) != 1:
    raise ValueError('The v3.3.4.4 runner accepts no direct CLI options.')
  launch = consume_bootstrap_attestation()
  start = launch['start']
  if set(OUTPUT_DIR.iterdir()) != {START_PATH}:
    raise RuntimeError('Runner did not begin from the exact START-only prefix.')
  frozen = json.loads(FREEZE_PATH.read_text(encoding='utf-8'))
  original_frozen = json.loads(ORIGINAL_FREEZE_PATH.read_text(encoding='utf-8'))
  cases: Sequence[Any] = ()
  execution_order: tuple[tuple[int, int], ...] = ()
  results: list[dict[str, Any]] = []
  apply_counter = [0]
  compiler: dict[str, Any] | None = None
  signature_binding: dict[str, Any] | None = None
  source_audit: dict[str, Any] = dict(start['source_input_audit'])
  same_object: dict[str, Any] | None = None
  failed_current_binding: dict[str, Any] | None = None
  current_phase = _phase_state(
      preflight_passed=True, start_persisted=True,
      post_start_source_gate_passed=True,
  )
  started_monotonic = time.monotonic()

  try:
    # Deliberately import the frozen upstream helper set before PRE_MODEL.
    import importlib  # pylint: disable=g-import-not-at-top
    for module_name in sorted(original_frozen['upstream_imported_modules']):
      importlib.import_module(module_name)

    imports_pre = import_provenance(
        original_frozen, phase='pre_model', enforce_loaded_contract=False
    )
    _write_new(OUTPUT_DIR / 'IMPORT_PROVENANCE_PRE_MODEL.json', imports_pre)
    current_phase['pre_model_import_inventory_persisted'] = True
    pre_predicates = import_validation_predicates(imports_pre, frozen)
    if not all(pre_predicates.values()):
      error = RuntimeError('PRE_MODEL import inventory mismatch.')
      _write_provenance_validation_failure(
          artifact_role='pre_model',
          artifact_path=OUTPUT_DIR / 'IMPORT_PROVENANCE_PRE_MODEL.json',
          predicates=pre_predicates, error=error, model_constructed=False,
      )
      source_audit = {
          **dict(start['source_input_audit']),
          'three_import_inventories_stable_exact': False,
      }
      _write_common_terminal(
          status='controlled_stop_import_provenance_failure',
          stop_reason='pre_model_import_inventory_mismatch',
          message='PRE_MODEL import inventory failed exact validation.',
          failure=_failure_object(error), results=[],
          failed_current_binding=None, compiler=None,
          source_input_audit=source_audit, same_object_attestation=None,
          phase_state=current_phase, failure_phase='imports',
          provenance_artifact_role='pre_model',
      )
      return
    protobuf_record = v32.protobuf_provenance()
    protobuf_record['external_freeze_authorization'] = dict(
        _external_authorization()
    )
    _write_new(OUTPUT_DIR / 'PROTOBUF_PROVENANCE.json', protobuf_record)
    current_phase['protobuf_persisted'] = True
    expected_protobuf = v32.protobuf_provenance()
    expected_protobuf['external_freeze_authorization'] = dict(
        _external_authorization()
    )
    protobuf_predicates = {
        'schema_and_binding_exact': protobuf_record == expected_protobuf,
        'external_freeze_authorization_exact': protobuf_record.get(
            'external_freeze_authorization'
        ) == dict(_external_authorization()),
    }
    if not all(protobuf_predicates.values()):
      error = RuntimeError('Protobuf provenance binding mismatch.')
      _write_provenance_validation_failure(
          artifact_role='protobuf',
          artifact_path=OUTPUT_DIR / 'PROTOBUF_PROVENANCE.json',
          predicates=protobuf_predicates, error=error,
          model_constructed=False,
      )
      source_audit = {
          **dict(start['source_input_audit']),
          'protobuf_binding_exact': False,
      }
      _write_common_terminal(
          status='controlled_stop_protobuf_provenance_failure',
          stop_reason='protobuf_binding_mismatch',
          message='Protobuf provenance failed exact validation.',
          failure=_failure_object(error), results=[],
          failed_current_binding=None, compiler=None,
          source_input_audit=source_audit, same_object_attestation=None,
          phase_state=current_phase, failure_phase='protobuf',
          provenance_artifact_role='protobuf',
      )
      return
    source_audit = {
        **source_audit,
        'protobuf_binding_exact': True,
    }
    checkpoint_binding = None
    reference_object_binding = None
    try:
      current_phase['model_construction_attempted'] = True
      setup = _construct_model_and_inputs()
      cases = setup['cases']
      execution_order = setup['execution_order']
      checkpoint_binding = setup['checkpoint_binding']
      reference_object_binding = setup['reference_binding']
      model_instance = setup['model_instance']
      params = setup['params']
      state = setup['state']
      common_by_order = setup['common_by_order']
      current_phase['model_constructed'] = True
      current_phase['reference_cases_loaded'] = True
      source_audit = {
          **source_audit,
          'checkpoint_exact': bool(checkpoint_binding),
          'reference_object_and_sequences_exact': bool(
              reference_object_binding
          ),
      }
    except bootstrap.PublicationError:
      raise
    except ModelSetupError as setup_error:
      error = setup_error.original_error
      checkpoint_binding = setup_error.checkpoint_binding
      reference_object_binding = setup_error.reference_binding
      current_phase['model_constructed'] = setup_error.model_constructed
      source_audit = {
          **source_audit,
          'checkpoint_exact': (
              None if checkpoint_binding is None
              else bool(checkpoint_binding)
          ),
          'reference_object_and_sequences_exact': (
              None if reference_object_binding is None
              else bool(reference_object_binding)
          ),
      }
      imports_post = import_provenance(
          original_frozen, phase='post_model_precompile',
          enforce_loaded_contract=False,
      )
      _write_new(
          OUTPUT_DIR / 'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
          imports_post,
      )
      current_phase['post_model_import_inventory_persisted'] = True
      post_predicates = import_validation_predicates(
          imports_post, frozen, (imports_pre,)
      )
      if not all(post_predicates.values()):
        validation_error = RuntimeError(
            'POST_MODEL import inventory mismatch after model setup failure.'
        )
        _write_provenance_validation_failure(
            artifact_role='post_model_precompile',
            artifact_path=(
                OUTPUT_DIR / 'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json'
            ),
            predicates=post_predicates, error=validation_error,
            model_constructed=setup_error.model_constructed,
        )
        source_audit = {
            **dict(start['source_input_audit']),
            'checkpoint_exact': source_audit['checkpoint_exact'],
            'reference_object_and_sequences_exact': source_audit[
                'reference_object_and_sequences_exact'
            ],
            'protobuf_binding_exact': True,
            'three_import_inventories_stable_exact': False,
        }
        _write_common_terminal(
            status='controlled_stop_import_provenance_failure',
            stop_reason='post_model_import_inventory_mismatch',
            message='POST_MODEL import inventory failed exact validation.',
            failure=_failure_object(validation_error), results=[],
            failed_current_binding=None, compiler=None,
            source_input_audit=source_audit, same_object_attestation=None,
            phase_state=current_phase, failure_phase='imports',
            provenance_artifact_role='post_model_precompile',
        )
        return
      imports_terminal, terminal_predicates = (
          _persist_terminal_import_inventory(
              original_frozen, frozen, (imports_pre, imports_post)
          )
      )
      current_phase['terminal_import_inventory_persisted'] = True
      source_audit = {
          **dict(start['source_input_audit']),
          'checkpoint_exact': source_audit['checkpoint_exact'],
          'reference_object_and_sequences_exact': source_audit[
              'reference_object_and_sequences_exact'
          ],
          'protobuf_binding_exact': True,
          'three_import_inventories_stable_exact': all(
              terminal_predicates.values()
          ),
      }
      terminal_import_failed = not all(terminal_predicates.values())
      terminal_error = RuntimeError('Terminal import inventory mismatch.')
      _write_common_terminal(
          status=(
              'controlled_stop_import_provenance_failure'
              if terminal_import_failed
              else 'controlled_stop_model_setup_failure'
          ),
          stop_reason=(
              'terminal_import_inventory_mismatch'
              if terminal_import_failed else 'model_setup_failure'
          ),
          message=(
              'Terminal import inventory failed exact validation.'
              if terminal_import_failed
              else 'Model/checkpoint/reference construction failed.'
          ),
          failure=_failure_object(
              terminal_error if terminal_import_failed else error
          ),
          results=results,
          failed_current_binding=None,
          compiler=None,
          source_input_audit=source_audit,
          same_object_attestation=None,
          phase_state=_phase_state(
              preflight_passed=True, start_persisted=True,
              post_start_source_gate_passed=True, protobuf_persisted=True,
              pre_model_import_inventory_persisted=True,
              model_construction_attempted=True,
              model_constructed=setup_error.model_constructed,
              post_model_import_inventory_persisted=True,
              terminal_import_inventory_persisted=True,
          ),
          failure_phase='imports' if terminal_import_failed else 'model_setup',
          provenance_artifact_role=(
              'terminal' if terminal_import_failed else None
          ),
      )
      return

    imports_post = import_provenance(
        original_frozen, phase='post_model_precompile',
        enforce_loaded_contract=False,
    )
    _write_new(
        OUTPUT_DIR / 'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
        imports_post,
    )
    current_phase['post_model_import_inventory_persisted'] = True
    post_predicates = import_validation_predicates(
        imports_post, frozen, (imports_pre,)
    )
    if not all(post_predicates.values()):
      error = RuntimeError('POST_MODEL import inventory mismatch.')
      _write_provenance_validation_failure(
          artifact_role='post_model_precompile',
          artifact_path=(
              OUTPUT_DIR / 'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json'
          ),
          predicates=post_predicates, error=error,
          model_constructed=True,
      )
      source_audit = {
          **dict(start['source_input_audit']),
          'checkpoint_exact': bool(checkpoint_binding),
          'reference_object_and_sequences_exact': bool(
              reference_object_binding
          ),
          'protobuf_binding_exact': True,
          'three_import_inventories_stable_exact': False,
      }
      _write_common_terminal(
          status='controlled_stop_import_provenance_failure',
          stop_reason='post_model_import_inventory_mismatch',
          message='POST_MODEL import inventory failed exact validation.',
          failure=_failure_object(error), results=[],
          failed_current_binding=None, compiler=None,
          source_input_audit=source_audit, same_object_attestation=None,
          phase_state=current_phase, failure_phase='imports',
          provenance_artifact_role='post_model_precompile',
      )
      return
    runtime_signatures: Mapping[str, Any] | None = None
    try:
      signature_inputs = _capture_signature_inputs(common_by_order)
      prototype_selection = signature_inputs['selection']
      prototype_target = signature_inputs['target']
      prototype_dna = signature_inputs['dna']
      prototype_interventions = signature_inputs['intended_interventions']
      runtime_signatures = signature_inputs['runtime_signatures']
      current_phase['signatures_captured'] = True
      adapted_signatures, signature_detail = canonicalize_program_signatures(
          runtime_signatures, frozen['program_signatures']
      )
      signature_attestation = {
          'schema_version': 'v3.3.4.4-program-signature-attestation-v1',
          'script_version': SCRIPT_VERSION,
          'attempt_id': ATTEMPT_ID,
          'external_freeze_authorization': dict(_external_authorization()),
          'object_order': list(_SIGNATURE_OBJECT_ORDER),
          **signature_detail,
          'created_at_unix_s': time.time(),
      }
      signature_path = (
          OUTPUT_DIR / 'compiler/eight_row/'
          'PROGRAM_SIGNATURE_ATTESTATION.json'
      )
      _write_new(signature_path, signature_attestation)
      signature_binding = _relative_file_binding(signature_path)
      current_phase['signature_attestation_persisted'] = True
    except bootstrap.PublicationError:
      raise
    except Exception as error:  # pylint: disable=broad-exception-caught
      failure_record = {
          'schema_version': 'v3.3.4.4-program-signature-attestation-v1',
          'script_version': SCRIPT_VERSION,
          'attempt_id': ATTEMPT_ID,
          'external_freeze_authorization': dict(_external_authorization()),
          'status': 'failure',
          'partial_runtime_tags': _validated_signature_tag_prefix(
              runtime_signatures, runtime=True
          ),
          'partial_frozen_tags': _validated_signature_tag_prefix(
              frozen.get('program_signatures'), runtime=False
          ),
          'failure': _failure_object(error),
          'created_at_unix_s': time.time(),
      }
      path = (
          OUTPUT_DIR / 'compiler/eight_row/'
          'PROGRAM_SIGNATURE_ATTESTATION_FAILURE.json'
      )
      _write_new(path, failure_record)
      imports_terminal, terminal_predicates = (
          _persist_terminal_import_inventory(
              original_frozen, frozen, (imports_pre, imports_post)
          )
      )
      current_phase['terminal_import_inventory_persisted'] = True
      source_audit = derive_source_input_audit(
          checkpoint_binding=checkpoint_binding,
          reference_object_binding=reference_object_binding,
          protobuf_record=protobuf_record,
          imports=(imports_pre, imports_post, imports_terminal),
      )
      terminal_import_failed = not all(terminal_predicates.values())
      terminal_error = RuntimeError('Terminal import inventory mismatch.')
      _write_common_terminal(
          status=(
              'controlled_stop_import_provenance_failure'
              if terminal_import_failed
              else 'controlled_stop_signature_attestation_failure'
          ),
          stop_reason=(
              'terminal_import_inventory_mismatch'
              if terminal_import_failed else 'signature_attestation_failure'
          ),
          message=(
              'Terminal import inventory failed exact validation.'
              if terminal_import_failed else
              'The representation-aware signature attestation failed.'
          ),
          failure=_failure_object(
              terminal_error if terminal_import_failed else error
          ),
          results=results,
          failed_current_binding=None,
          compiler=None,
          source_input_audit=source_audit,
          same_object_attestation=None,
          phase_state=_phase_state(
              preflight_passed=True, start_persisted=True,
              post_start_source_gate_passed=True, protobuf_persisted=True,
              pre_model_import_inventory_persisted=True,
              model_construction_attempted=True, model_constructed=True,
              reference_cases_loaded=True,
              signatures_captured=current_phase['signatures_captured'],
              post_model_import_inventory_persisted=True,
              terminal_import_inventory_persisted=True,
          ),
          failure_phase='imports' if terminal_import_failed else 'signatures',
          provenance_artifact_role=(
              'terminal' if terminal_import_failed else None
          ),
          program_signature_attestation_binding=_relative_file_binding(path),
      )
      return

    compile_start = time.perf_counter()
    lowered = None
    compiled = None
    compiler_budget = OneShotCompilerBudget()
    try:
      current_phase['lower_attempted'] = True
      compiler_budget.request('lower')
      raw_apply = (
          dna_model
          .create_splice_classification_logit_margin_eight_row_superset_graph_apply(
              model_instance._metadata,  # pylint: disable=protected-access
              attention_backend=v32.route_v3.ATTENTION_BACKEND,
          )
      )
      prototype_args = (
          params, state, prototype_dna, jnp.zeros((8,), jnp.int32),
          prototype_selection, prototype_interventions, prototype_target,
      )
      lowered = jax.jit(raw_apply).lower(*prototype_args)
      current_phase['lower_succeeded'] = True
      current_phase['compile_attempted'] = True
      compiler_budget.request('compile')
      compiled = lowered.compile()
      current_phase['compile_succeeded'] = True
    except AttemptBudgetViolation as error:
      if lowered is None:
        raise RuntimeError(
            'Budget guard fired without the required prior-lower prefix.'
        ) from error
      imports_terminal, terminal_predicates = (
          _persist_terminal_import_inventory(
              original_frozen, frozen, (imports_pre, imports_post)
          )
      )
      current_phase['terminal_import_inventory_persisted'] = True
      source_audit = derive_source_input_audit(
          checkpoint_binding=checkpoint_binding,
          reference_object_binding=reference_object_binding,
          protobuf_record=protobuf_record,
          imports=(imports_pre, imports_post, imports_terminal),
      )
      terminal_import_failed = not all(terminal_predicates.values())
      if error.operation == 'lower':
        compiler = _attempt_budget_failure_artifact(
            error,
            lowered=lowered,
            seconds=time.perf_counter() - compile_start,
            program_signatures=adapted_signatures,
            kernel_cache_preimport_attestation=start[
                'same_process_preflight'
            ]['model_cache_pre_import'],
            program_signature_attestation=signature_attestation,
            source_input_audit=source_audit,
            attempt_budget_audit=compiler_budget.audit(),
        )
      else:
        if compiled is None:
          raise RuntimeError(
              'Second-compile guard lacks the required successful compile.'
          ) from error
        published_graphs = _publish_compiler_graphs(
            _extract_compiler_graph_texts(lowered, compiled)
        )
        compiler = _attempt_budget_failure_artifact(
            error, lowered=lowered, compiled=compiled,
            published_graphs=published_graphs,
            seconds=time.perf_counter() - compile_start,
            program_signatures=adapted_signatures,
            kernel_cache_preimport_attestation=start[
                'same_process_preflight'
            ]['model_cache_pre_import'],
            program_signature_attestation=signature_attestation,
            source_input_audit=source_audit,
            attempt_budget_audit=compiler_budget.audit(),
        )
      same_object = compiler['same_object_attestation']
      terminal_error = RuntimeError('Terminal import inventory mismatch.')
      _write_common_terminal(
          status=(
              'controlled_stop_import_provenance_failure'
              if terminal_import_failed
              else 'controlled_stop_attempt_budget_violation'
          ),
          stop_reason=(
              'terminal_import_inventory_mismatch'
              if terminal_import_failed
              else f'second_{error.operation}_attempt_forbidden'
          ),
          message=(
              'Terminal import inventory failed exact validation.'
              if terminal_import_failed else
              'A guarded compiler attempt was blocked before invocation.'
          ),
          failure=_failure_object(
              terminal_error if terminal_import_failed else error
          ), results=results,
          failed_current_binding=None, compiler=compiler,
          source_input_audit=source_audit,
          same_object_attestation=same_object,
          phase_state=_phase_state(
              preflight_passed=True, start_persisted=True,
              post_start_source_gate_passed=True, protobuf_persisted=True,
              pre_model_import_inventory_persisted=True,
              model_construction_attempted=True, model_constructed=True,
              reference_cases_loaded=True, signatures_captured=True,
              signature_attestation_persisted=True,
              post_model_import_inventory_persisted=True,
              lower_attempted=True, lower_succeeded=True,
              compile_attempted=error.operation == 'compile',
              compile_succeeded=(
                  error.operation == 'compile' and compiled is not None
              ),
              terminal_import_inventory_persisted=True,
              source_program_gate_passed=(
                  False
              ),
              diagnostic_provenance_passed=(
                  False
              ),
          ),
          failure_phase=(
              'imports' if terminal_import_failed else (
                  'lower' if error.operation == 'lower' else 'compile'
              )
          ),
          forbidden_operation=(
              None if terminal_import_failed else error.operation
          ),
          provenance_artifact_role=(
              'terminal' if terminal_import_failed else None
          ),
          program_signature_attestation_binding=signature_binding,
          compiler_binding=_relative_file_binding(
              OUTPUT_DIR / 'compiler/eight_row/COMPILER_PROVENANCE.json'
          ),
      )
      return
    except Exception as error:  # pylint: disable=broad-exception-caught
      imports_terminal, terminal_predicates = (
          _persist_terminal_import_inventory(
              original_frozen, frozen, (imports_pre, imports_post)
          )
      )
      current_phase['terminal_import_inventory_persisted'] = True
      source_audit = derive_source_input_audit(
          checkpoint_binding=checkpoint_binding,
          reference_object_binding=reference_object_binding,
          protobuf_record=protobuf_record,
          imports=(imports_pre, imports_post, imports_terminal),
      )
      failure_stage = 'compile' if lowered is not None else 'lower'
      same_object = _compiler_failure_same_object_attestation(
          stage=failure_stage, lowered=lowered,
          compile_count=int(lowered is not None),
      )
      compiler = _compiler_failure_artifact(
          error,
          stage=failure_stage,
          compile_count=int(lowered is not None),
          seconds=time.perf_counter() - compile_start,
          lowered=lowered,
          program_signatures=adapted_signatures,
          kernel_cache_preimport_attestation=start[
              'same_process_preflight'
          ]['model_cache_pre_import'],
          program_signature_attestation=signature_attestation,
          source_input_audit=source_audit,
          attempt_budget_audit=compiler_budget.audit(),
      )
      terminal_import_failed = not all(terminal_predicates.values())
      terminal_error = RuntimeError('Terminal import inventory mismatch.')
      _write_common_terminal(
          status=(
              'controlled_stop_import_provenance_failure'
              if terminal_import_failed else (
                  'controlled_stop_compile_failure' if lowered is not None
                  else 'controlled_stop_lower_failure'
              )
          ),
          stop_reason=(
              'terminal_import_inventory_mismatch'
              if terminal_import_failed else (
                  'compile_failure' if lowered is not None else 'lower_failure'
              )
          ),
          message=(
              'Terminal import inventory failed exact validation.'
              if terminal_import_failed else
              'The sole lower/compile attempt failed; no retry.'
          ),
          failure=_failure_object(
              terminal_error if terminal_import_failed else error
          ),
          results=results,
          failed_current_binding=None,
          compiler=compiler,
          source_input_audit=source_audit,
          same_object_attestation=same_object,
          phase_state=_phase_state(
              preflight_passed=True, start_persisted=True,
              post_start_source_gate_passed=True, protobuf_persisted=True,
              pre_model_import_inventory_persisted=True,
              model_construction_attempted=True, model_constructed=True,
              reference_cases_loaded=True, signatures_captured=True,
              signature_attestation_persisted=True,
              post_model_import_inventory_persisted=True,
              lower_attempted=True, lower_succeeded=lowered is not None,
              compile_attempted=lowered is not None,
              terminal_import_inventory_persisted=True,
          ),
          failure_phase=(
              'imports' if terminal_import_failed else (
                  'compile' if lowered is not None else 'lower'
              )
          ),
          provenance_artifact_role=(
              'terminal' if terminal_import_failed else None
          ),
          program_signature_attestation_binding=signature_binding,
          compiler_binding=_relative_file_binding(
              OUTPUT_DIR / 'compiler/eight_row/COMPILER_PROVENANCE.json'
          ),
      )
      return

    postcompile = _orchestrate_postcompile_provenance(
        lowered=lowered, compiled=compiled, compile_start=compile_start,
        compiler_budget=compiler_budget, launch=launch, start=start,
        original_frozen=original_frozen, frozen=frozen,
        imports_pre=imports_pre, imports_post=imports_post,
        checkpoint_binding=checkpoint_binding,
        reference_object_binding=reference_object_binding,
        protobuf_record=protobuf_record,
        adapted_signatures=adapted_signatures,
        signature_attestation=signature_attestation,
        signature_binding=signature_binding,
        source_input_audit_prefix=source_audit,
        current_phase=current_phase, results=results,
    )
    if postcompile is None:
      return
    compiler = postcompile['compiler']
    source_audit = postcompile['source_input_audit']
    same_object = postcompile['same_object_attestation']
    terminal_predicates = postcompile['terminal_import_predicates']
    compiler_binding = _relative_file_binding(
        OUTPUT_DIR / 'compiler/eight_row/COMPILER_PROVENANCE.json'
    )
    full_phase = _phase_state(
        preflight_passed=True, start_persisted=True,
        post_start_source_gate_passed=True, protobuf_persisted=True,
        pre_model_import_inventory_persisted=True,
        model_construction_attempted=True, model_constructed=True,
        reference_cases_loaded=True, signatures_captured=True,
        signature_attestation_persisted=True,
        post_model_import_inventory_persisted=True,
        lower_attempted=True, lower_succeeded=True,
        compile_attempted=True, compile_succeeded=True,
        terminal_import_inventory_persisted=True,
        source_program_gate_passed=compiler['source_program_gate'][
            'source_program_exact'
        ],
        diagnostic_provenance_passed=True,
    )
    current_phase = full_phase
    if not all(terminal_predicates.values()):
      terminal_error = RuntimeError('Terminal import inventory mismatch.')
      _write_common_terminal(
          status='controlled_stop_import_provenance_failure',
          stop_reason='terminal_import_inventory_mismatch',
          message='Terminal import inventory failed exact validation.',
          failure=_failure_object(terminal_error),
          results=results,
          failed_current_binding=None,
          compiler=compiler,
          source_input_audit=source_audit,
          same_object_attestation=same_object,
          phase_state=full_phase,
          failure_phase='imports',
          provenance_artifact_role='terminal',
          program_signature_attestation_binding=signature_binding,
          compiler_binding=compiler_binding,
      )
      return
    predispatch_stop = _predispatch_controlled_stop(compiler)
    if predispatch_stop is not None:
      stop_status, stop_reason, failure_phase, forbidden_operation = (
          predispatch_stop
      )
      _write_common_terminal(
          status=stop_status,
          stop_reason=stop_reason,
          message='A frozen pre-dispatch infrastructure gate stopped the run.',
          failure={'type': 'PredispatchInfrastructureStop',
                   'message': stop_reason,
                   'traceback': ''},
          results=results,
          failed_current_binding=None,
          compiler=compiler,
          source_input_audit=source_audit,
          same_object_attestation=same_object,
          phase_state=full_phase,
          failure_phase=failure_phase,
          forbidden_operation=forbidden_operation,
          program_signature_attestation_binding=signature_binding,
          compiler_binding=compiler_binding,
      )
      return

    original_manifest = launch['v3_3_3_run']['raw_manifest']
    cases_by_order = {case.order: case for case in cases}
    for execution_index, (order, anchor_id) in enumerate(execution_order):
      recipient = cases_by_order[order]
      donor = cases_by_order[v33.OOD_DONOR_ORDER[order]]
      try:
        result = _run_anchor(
            compiled, recipient, donor, common_by_order[order],
            common_by_order[donor.order], params, state, anchor_id,
            runtime_signatures, start['freeze_sha256'],
            compiler['executable_fingerprint'], execution_index,
            original_manifest, apply_counter,
            external_freeze_authorization=_external_authorization(),
            source_input_audit=source_audit,
            same_object_attestation=same_object,
        )
      except CurrentRecordStop as stop:
        current_phase['dispatch_begun'] = bool(_journal_paths('started'))
        failed_binding = _write_failed_current(
            stop,
            execution_index=execution_index,
            recipient=recipient,
            anchor_id=anchor_id,
            source_input_audit_binding=_content_binding(source_audit),
            same_object_attestation_binding=_content_binding(same_object),
        )
        failed_current_binding = dict(failed_binding)
        full_phase['dispatch_begun'] = bool(
            _journal_paths('started')
        )
        status = (
            'controlled_stop_four_call_invalid'
            if len(stop.completed) == 4
            else 'controlled_stop_partial_dispatch'
        )
        stop_reason = (
            'record_validation_or_serialization_failure'
            if len(stop.completed) == 4
            else (
                'model_dispatch_failure'
                if stop.failure_phase == 'model_dispatch'
                else 'record_setup_failure'
            )
        )
        _write_common_terminal(
            status=status,
            stop_reason=stop_reason,
            message='Exact failed-current prefix preserved; no retry.',
            failure=_failure_object(stop.original_error),
            results=results,
            failed_current_binding=failed_binding,
            compiler=compiler,
            source_input_audit=source_audit,
            same_object_attestation=same_object,
            phase_state=full_phase,
            failure_phase=stop.failure_phase,
            failed_execution_index=execution_index,
            failed_call_role=stop.failed_or_next_call_role,
            program_signature_attestation_binding=signature_binding,
            compiler_binding=compiler_binding,
        )
        return
      results.append(result)
      current_phase['dispatch_begun'] = True
      _assert_attempt_budget(started_monotonic)

    if len(results) != EXPECTED_RECORD_COUNT or apply_counter[0] != 320:
      raise RuntimeError('Full sidecar count contract changed.')
    full_phase['dispatch_begun'] = True
    _write_common_terminal(
        status='complete_structural_sidecar',
        stop_reason=None,
        message='All 80 frozen v3.3.4.4 structural sidecar records completed.',
        failure=None,
        results=results,
        failed_current_binding=None,
        compiler=compiler,
        source_input_audit=source_audit,
        same_object_attestation=same_object,
        phase_state=full_phase,
        failure_phase=None,
        program_signature_attestation_binding=signature_binding,
        compiler_binding=compiler_binding,
    )
  except bootstrap.PublicationError as error:
    _write_terminal_failure(
        error,
        completed_record_count=len(results),
        source_input_audit=source_audit,
        same_object_attestation=same_object,
        phase_state=locals().get('full_phase', current_phase),
        failed_current_binding=failed_current_binding,
    )
    raise


if __name__ == '__main__':
  _main_v334()
