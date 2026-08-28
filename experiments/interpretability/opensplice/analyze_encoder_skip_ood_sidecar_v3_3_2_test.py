"""CPU-only synthetic tests for the v3.3.2 structural sidecar analyzer."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


_MODULE_PATH = Path(__file__).with_name(
    'analyze_encoder_skip_ood_sidecar_v3_3_2.py'
)
_SPEC = importlib.util.spec_from_file_location(
    'analyze_encoder_skip_ood_sidecar_v3_3_2', _MODULE_PATH
)
analyzer = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = analyzer
_SPEC.loader.exec_module(analyzer)


def _write_json(path: Path, value) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
      json.dumps(value, sort_keys=True, allow_nan=False) + '\n',
      encoding='utf-8',
  )


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _readout(values: list[float]) -> dict:
  logits, margins, totals, means = [], [], [], []
  for raw in values:
    value = analyzer._v33._f32(raw)  # pylint: disable=protected-access
    logits.append([[value, 0.0], [value, 0.0]])
    margins.append([value, value])
    total = analyzer._v33._f32(value + value)  # pylint: disable=protected-access
    totals.append(total)
    means.append(analyzer._v33._f32(total / 2.0))  # pylint: disable=protected-access
  return {
      'endpoint_axis': ['acceptor', 'donor'],
      'selected_logit_axis': ['relevant_class', 'padding_class'],
      'selected_logits': logits,
      'endpoint_margins': margins,
      'totals': totals,
      'means': means,
      'num_values': 2,
  }


def _runtime(coalition_id: int, donor_rows: tuple[int, ...]) -> dict:
  t, e_mask = divmod(coalition_id, 128)
  active = [False, False, True, True, True, True, False, False]

  def whole(enabled):
    return {
        'donor_batch_indices': [list(donor_rows) for _ in enabled],
        'natural_identity_batch_indices': [
            list(analyzer.IDENTITY_ROWS) for _ in enabled
        ],
        'transfer_mask': [
            [bool(on and is_active) for is_active in active] for on in enabled
        ],
    }

  residual = {
      'donor_batch_indices': [
          [[row] * 24 for row in range(8)] for _ in range(9)
      ],
      'transfer_mask': [
          [[False] * 24 for _ in range(8)] for _ in range(9)
      ],
  }
  return {
      'transformer_output': whole([bool(t)]),
      'encoder_skips': whole([
          bool(e_mask & (1 << index)) for index in range(7)
      ]),
      'final_embedding': {
          'donor_batch_indices': [[[row, row] for row in range(8)]],
          'transfer_mask': [[[False, False] for _ in range(8)]],
      },
      'phase_r_residuals': {
          name: copy.deepcopy(residual) for name in (
              'pre_attention_residual_transfer',
              'post_attention_residual_transfer',
              'post_mlp_residual_transfer',
          )
      },
  }


def _row_fingerprint(shas: list[str]) -> dict:
  return {
      'full_shape': [8, 2],
      'dtype': 'float32',
      'row_count': 8,
      'rows': [
          {
              'row': row, 'shape': [2], 'dtype': 'float32',
              'size_bytes': 8, 'sha256': digest,
          }
          for row, digest in enumerate(shas)
      ],
      'collision_semantics': (
          'SHA-256 per exact row byte string; direct live equality is the gate.'
      ),
  }


def _upstream(shape: tuple[int, ...]) -> dict:
  def values(dimensions: tuple[int, ...]):
    if len(dimensions) == 1:
      return [0.0] * dimensions[0]
    return [values(dimensions[1:]) for _ in range(dimensions[0])]
  return {'shape': list(shape), 'dtype': 'float32', 'values': values(shape)}


def _rowwise(anchor: int, *, active_equal: bool = False) -> dict:
  intended = [hashlib.sha256(f'i-{row}'.encode()).hexdigest() for row in range(8)]
  unrelated = list(intended)
  if anchor != 0 and not active_equal:
    for row in analyzer.ACTIVE_ROWS:
      unrelated[row] = hashlib.sha256(f'u-{row}'.encode()).hexdigest()

  def call(rows: list[str]) -> dict:
    fingerprint = _row_fingerprint(rows)
    return {
        'natural_final_embeddings': copy.deepcopy(fingerprint),
        'effective_final_embeddings': copy.deepcopy(fingerprint),
        'transformer_output_natural_fingerprint': _upstream((8, 4)),
        'encoder_skips_natural_fingerprints': _upstream((7, 8, 4)),
    }
  intended_call = call(intended)
  unrelated_call = call(unrelated)
  return {
      'intended': intended_call,
      'intended_repeat': copy.deepcopy(intended_call),
      'unrelated': unrelated_call,
      'unrelated_repeat': copy.deepcopy(unrelated_call),
  }


def _checks(anchor: int) -> dict:
  result = {key: True for key in analyzer._CHECK_KEYS}
  result.update({
      'corrected_host_assertion_version': 'v3.3.2',
      'natural_final_invariant_rows': list(analyzer.INVARIANT_ROWS),
      'active_rows_cross_call_equality_not_required': list(analyzer.ACTIVE_ROWS),
      'normalization_computed': False,
      'id0_all8_natural_final_exact_between_calls': anchor == 0,
      'id0_within_call_natural_final_recipient_noop_exact': anchor == 0,
      'id0_all8_endpoint_exact_between_calls': anchor == 0,
      'id0_recipient_noop_exact': anchor == 0,
      'id255_intended_endpoint_closure_exact': anchor == 255,
      'id255_unrelated_endpoint_closure_exact': anchor == 255,
  })
  return result


def _values(anchor: int) -> tuple[list[float], list[float]]:
  if anchor == 0:
    shared = [10, 20, 20, 20, 10, 10, 30, 40]
    return shared, list(shared)
  if anchor == 255:
    return (
        [10, 20, 10, 20, 20, 10, 30, 40],
        [10, 20, 30, 20, 40, 10, 30, 40],
    )
  return (
      [10, 20, 15, 20, 15, 10, 30, 40],
      [10, 20, 30, 20, 40, 10, 30, 40],
  )


def _trace() -> dict:
  return {'sha256': 'f' * 64, 'leaves': [{'shape': [8, 2], 'dtype': 'float32'}]}


def _case(order: int) -> dict:
  return {'order': order, 'variant_id': f'variant-{order}'}


def _make_original_binding(
    root: Path, mapping: dict[str, str], relative: str,
) -> dict:
  path = root / relative
  if not path.exists():
    _write_json(path, {'structural_fixture': relative})
  mapping[relative] = _sha(path)
  return {'path': relative, 'sha256': mapping[relative]}


def _record(
    *, original_root: Path, original_manifest: dict[str, str],
    cases: dict[int, dict], sequence_bindings: dict[int, dict],
    order: int, anchor: int, execution_index: int,
    status: str = 'complete', active_equal: bool = False,
) -> dict:
  recipient, donor = cases[order], cases[analyzer._donor_order(order)]
  recipient_identity = analyzer._original_relative(recipient, 'identity', None)
  donor_identity = analyzer._original_relative(donor, 'identity', None)
  coalition = analyzer._original_relative(recipient, 'coalition', anchor)
  intended_values, unrelated_values = _values(anchor)
  intended = _readout(intended_values)
  unrelated = _readout(unrelated_values)
  parsed_intended = analyzer._v33._readout(  # pylint: disable=protected-access
      {'x': intended}, 'x', 'fixture', rows=8
  )
  parsed_unrelated = analyzer._v33._readout(  # pylint: disable=protected-access
      {'x': unrelated}, 'x', 'fixture', rows=8
  )
  intended_movement = analyzer._v33._raw_movements(  # pylint: disable=protected-access
      parsed_intended
  )['movements']
  unrelated_movement = analyzer._v33._raw_movements(  # pylint: disable=protected-access
      parsed_unrelated
  )['movements']
  trace = _trace()
  return {
      'status': status,
      'family': 'v3_3_2_unrelated_donor_sidecar_anchor',
      'script_version': analyzer.SCRIPT_VERSION,
      'amendment_sha256': analyzer.AMENDMENT_SHA256,
      'amendment_commit': analyzer.AMENDMENT_COMMIT,
      'original_protocol_sha256': analyzer.ORIGINAL_PROTOCOL_SHA256,
      'freeze_sha256': '9' * 64,
      'execution_index': execution_index,
      'sidecar_execution_index': execution_index,
      'execution_order': 'recipient-major, anchor-minor',
      'eight_row_executable_fingerprint': 'e' * 64,
      'same_eight_row_compiled_executable': True,
      'six_row_executable_used': False,
      'recipient_case': recipient,
      'donor_case': donor,
      'coalition': analyzer._expected_coalition(anchor),
      'batch_roles': list(analyzer.EIGHT_ROLES),
      'natural_identity_rows': list(analyzer.IDENTITY_ROWS),
      'intended_donor_rows': list(analyzer.INTENDED_DONOR_ROWS),
      'unrelated_donor_rows': list(analyzer.UNRELATED_DONOR_ROWS),
      'invariant_rows_between_calls': list(analyzer.INVARIANT_ROWS),
      'active_recipient_rows': list(analyzer.ACTIVE_ROWS),
      'active_recipient_cross_call_equality_gate': False,
      'active_recipient_cross_call_inequality_gate': False,
      'original_artifact_bindings': {
          'recipient_identity': _make_original_binding(
              original_root, original_manifest, recipient_identity
          ),
          'donor_identity': _make_original_binding(
              original_root, original_manifest, donor_identity
          ),
          'recipient_six_row_coalition': _make_original_binding(
              original_root, original_manifest, coalition
          ),
      },
      'original_ood_records_used_as_data': False,
      'recipient_sequence_sha256': copy.deepcopy(sequence_bindings[order]),
      'donor_sequence_sha256': copy.deepcopy(sequence_bindings[donor['order']]),
      'runtime_interventions': {
          'intended': _runtime(anchor, analyzer.INTENDED_DONOR_ROWS),
          'unrelated': _runtime(anchor, analyzer.UNRELATED_DONOR_ROWS),
      },
      'intended_target_readout': intended,
      'intended_repeat_target_readout': copy.deepcopy(intended),
      'unrelated_target_readout': unrelated,
      'unrelated_repeat_target_readout': copy.deepcopy(unrelated),
      'intended_trace_fingerprint': trace,
      'intended_repeat_trace_fingerprint': copy.deepcopy(trace),
      'unrelated_trace_fingerprint': copy.deepcopy(trace),
      'unrelated_repeat_trace_fingerprint': copy.deepcopy(trace),
      'rowwise_trace_fingerprints': _rowwise(
          anchor, active_equal=active_equal
      ),
      'raw_movement': {
          'intended': intended_movement,
          'unrelated': unrelated_movement,
      },
      'model_apply_count_through_record': 4 * (execution_index + 1),
      'checks': _checks(anchor) if status == 'complete' else None,
      'failure': None if status == 'complete' else {
          'type': 'ValueError', 'message': 'synthetic frozen live gate failed',
      },
      'seconds': {
          'intended': 1.0, 'intended_repeat': 1.0,
          'unrelated': 1.0, 'unrelated_repeat': 1.0,
      },
      'created_at_unix_s': 1.0,
  }


class RawRecordTest(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.original = self.root / 'original'
    self.cases = {order: _case(order) for order in range(20)}
    self.sequences = {
        order: {'reference': f'{order:064x}', 'alternate': f'{order + 100:064x}'}
        for order in range(20)
    }
    self.manifest = {}

  def tearDown(self):
    self.temporary.cleanup()

  def validate(self, record: dict, anchor: int, *, invalid: bool = False):
    with mock.patch.object(analyzer, '_ORIGINAL_RUN_DIR', self.original):
      return analyzer._validate_record(
          record, case=self.cases[0], donor_case=self.cases[10],
          anchor=anchor, execution_index=0, freeze_sha256='9' * 64,
          executable_fingerprint='e' * 64,
          original_manifest=self.manifest,
          sequence_bindings=self.sequences, allow_invalid=invalid,
      )

  def make(self, anchor: int, *, active_equal: bool = False) -> dict:
    return _record(
        original_root=self.original, original_manifest=self.manifest,
        cases=self.cases, sequence_bindings=self.sequences,
        order=0, anchor=anchor, execution_index=0,
        active_equal=active_equal,
    )

  def test_active_rows_may_differ(self):
    self.assertEqual(self.validate(self.make(127), 127)['status'], 'complete')

  def test_active_rows_may_also_match(self):
    self.assertEqual(
        self.validate(self.make(127, active_equal=True), 127)['status'],
        'complete',
    )

  def test_invariant_natural_final_tamper_fails(self):
    record = self.make(127)
    record['rowwise_trace_fingerprints']['unrelated'][
        'natural_final_embeddings'
    ]['rows'][3]['sha256'] = 'a' * 64
    record['rowwise_trace_fingerprints']['unrelated_repeat'][
        'natural_final_embeddings'
    ]['rows'][3]['sha256'] = 'a' * 64
    for call in ('unrelated', 'unrelated_repeat'):
      record['rowwise_trace_fingerprints'][call][
          'effective_final_embeddings'
      ]['rows'][3]['sha256'] = 'a' * 64
    with self.assertRaisesRegex(analyzer.AnalysisError, 'invariant rows'):
      self.validate(record, 127)

  def test_upstream_compact_tamper_fails(self):
    record = self.make(127)
    record['rowwise_trace_fingerprints']['unrelated'][
        'transformer_output_natural_fingerprint'
    ]['values'][2][1] = 1.0
    record['rowwise_trace_fingerprints']['unrelated_repeat'][
        'transformer_output_natural_fingerprint'
    ]['values'][2][1] = 1.0
    with self.assertRaisesRegex(analyzer.AnalysisError, 'upstream natural T'):
      self.validate(record, 127)

  def test_endpoint_margin_tamper_fails(self):
    record = self.make(127)
    record['unrelated_target_readout']['endpoint_margins'][2][0] = 99.0
    with self.assertRaisesRegex(analyzer._v33.AnalysisError, 'differs'):
      self.validate(record, 127)

  def test_runtime_donor_map_tamper_fails(self):
    record = self.make(127)
    record['runtime_interventions']['unrelated'][
        'encoder_skips'
    ]['donor_batch_indices'][0][2] = 0
    with self.assertRaisesRegex(analyzer._v33.AnalysisError, 'donor map'):
      self.validate(record, 127)

  def test_id0_and_id255_closures(self):
    self.validate(self.make(0), 0)
    self.validate(self.make(255), 255)

  def test_sequence_binding_tamper_fails(self):
    record = self.make(127)
    record['donor_sequence_sha256']['alternate'] = 'f' * 64
    with self.assertRaisesRegex(analyzer.AnalysisError, 'donor_sequence'):
      self.validate(record, 127)

  def test_invalid_record_accepts_exact_partial_readout_prefix(self):
    record = self.make(127)
    record['status'] = 'invalid'
    record['checks'] = None
    record['failure'] = {'type': 'ValueError', 'message': 'live gate failed'}
    self.assertEqual(self.validate(record, 127, invalid=True)['status'], 'invalid')
    record['unrelated_repeat_target_readout'] = None
    record.update({
        'rowwise_trace_fingerprints': None,
        'intended_trace_fingerprint': None,
        'intended_repeat_trace_fingerprint': None,
        'unrelated_trace_fingerprint': None,
        'unrelated_repeat_trace_fingerprint': None,
        'original_artifact_bindings': None,
        'raw_movement': None,
    })
    self.assertEqual(self.validate(record, 127, invalid=True)['status'], 'invalid')

  def test_invalid_early_readout_failure_audits_exact_empty_payload(self):
    record = self.make(127)
    record.update({
        'status': 'invalid', 'checks': None,
        'failure': {'type': 'ValueError', 'message': 'readout failed'},
        'intended_target_readout': None,
        'intended_repeat_target_readout': None,
        'unrelated_target_readout': None,
        'unrelated_repeat_target_readout': None,
        'rowwise_trace_fingerprints': None,
        'intended_trace_fingerprint': None,
        'intended_repeat_trace_fingerprint': None,
        'unrelated_trace_fingerprint': None,
        'unrelated_repeat_trace_fingerprint': None,
        'original_artifact_bindings': None,
        'raw_movement': None,
    })
    self.assertEqual(self.validate(record, 127, invalid=True)['status'], 'invalid')

  def test_invalid_binding_stage_failure_audits_prior_evidence(self):
    record = self.make(127)
    record.update({
        'status': 'invalid', 'checks': None,
        'failure': {'type': 'FileNotFoundError', 'message': 'binding failed'},
        'original_artifact_bindings': None,
        'raw_movement': None,
    })
    self.assertEqual(self.validate(record, 127, invalid=True)['status'], 'invalid')

  def test_invalid_payload_rejects_dependency_gap(self):
    record = self.make(127)
    record.update({
        'status': 'invalid', 'checks': None,
        'failure': {'type': 'ValueError', 'message': 'partial failure'},
        'intended_target_readout': None,
        'rowwise_trace_fingerprints': None,
        'intended_trace_fingerprint': None,
        'intended_repeat_trace_fingerprint': None,
        'unrelated_trace_fingerprint': None,
        'unrelated_repeat_trace_fingerprint': None,
        'original_artifact_bindings': None,
        'raw_movement': None,
    })
    with self.assertRaisesRegex(analyzer.AnalysisError, 'exact prefix'):
      self.validate(record, 127, invalid=True)


class CompletionPrefixTest(unittest.TestCase):

  def completion(self, *, reason=None, count=80) -> dict:
    full = reason is None
    return {
        'status': 'complete' if full else 'controlled_stop',
        'stop_reason': reason,
        'message': (
            'All 80 frozen v3.3.2 OOD sidecar records completed.' if full
            else 'New eight-row graph/HLO differs from frozen v3.3.'
        ),
        'attempt_id': analyzer.ATTEMPT_ID,
        'script_version': analyzer.SCRIPT_VERSION,
        'amendment_sha256': analyzer.AMENDMENT_SHA256,
        'amendment_commit': analyzer.AMENDMENT_COMMIT,
        'original_protocol_sha256': analyzer.ORIGINAL_PROTOCOL_SHA256,
        'freeze_sha256': '9' * 64,
        'ood_anchor_record_count': count,
        'ood_invalid_count': 0 if full or reason == 'compiler_graph_mismatch' else 1,
        'unique_recipient_anchor_count': count,
        'all_80_recipient_anchors_complete': full,
        'model_apply_count': 4 * count,
        'expected_model_apply_count': 320,
        'eight_row_compile_count': 1,
        'six_row_compile_count': 0,
        'identity_rerun_count': 0,
        'main_cube_rerun_count': 0,
        'old_ood_records_reused': 0,
        'one_fixed_eight_row_executable': True,
        'eight_row_compiler': {},
        'eight_row_executable_fingerprint': 'e' * 64,
        'graph_and_hlo_exact_to_original_v3_3': reason != 'compiler_graph_mismatch',
        'id0_all20': full,
        'id255_all20': full,
        'invariant_rows_between_calls': list(analyzer.INVARIANT_ROWS),
        'active_rows_have_no_forced_cross_call_predicate': True,
        'original_run_binding': dict(analyzer._ORIGINAL_BINDING),
        'original_run_revalidated_in_full': True,
        'original_ood_records_provenance_only': True,
        'v3_3_1_status': {'state': 'completed'},
        'import_provenance_phases': {},
        'import_provenance_sha256': 'a' * 64,
        'protobuf_provenance_sha256': 'b' * 64,
        'raw_manifest': {},
        'confirmation_model_calls': 0,
        'confirmation_scope_disclosure': analyzer.CONFIRMATION_DISCLOSURE,
        'scientific_summary_computed': False,
        'shapley_or_nomination_computed': False,
        'completed_at_unix_s': 1.0,
    }

  def test_complete_and_compiler_stop(self):
    prefix, full = analyzer._completion_prefix(
        self.completion(), freeze_sha='9' * 64,
        v331_status={'state': 'completed'},
    )
    self.assertTrue(full)
    self.assertEqual(len(prefix), 80)
    prefix, full = analyzer._completion_prefix(
        self.completion(reason='compiler_graph_mismatch', count=0),
        freeze_sha='9' * 64, v331_status={'state': 'completed'},
    )
    self.assertFalse(full)
    self.assertEqual(prefix, ())

  def test_ood_prefix_requires_exact_message_and_final_invalid(self):
    record = self.completion(reason='ood_tooling_failure', count=2)
    record['message'] = 'OOD sidecar audit failed at order=0, anchor_id=127.'
    prefix, full = analyzer._completion_prefix(
        record, freeze_sha='9' * 64, v331_status={'state': 'completed'},
    )
    self.assertFalse(full)
    self.assertEqual(prefix, ((0, 0), (0, 127)))
    record['ood_invalid_count'] = 0
    with self.assertRaisesRegex(analyzer.AnalysisError, 'allowed prefix'):
      analyzer._completion_prefix(
          record, freeze_sha='9' * 64,
          v331_status={'state': 'completed'},
      )


class ProvenanceValidatorTest(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)

  def tearDown(self):
    self.temporary.cleanup()

  def test_original_binding_is_literal_runner_bootstrap_contract(self):
    bootstrap_path = Path(analyzer.__file__).with_name(
        'validate_encoder_skip_ood_sidecar_bootstrap_v3_3_2.py'
    )
    specification = importlib.util.spec_from_file_location(
        '_v3_3_2_bootstrap_contract_fixture', bootstrap_path
    )
    self.assertIsNotNone(specification)
    assert specification is not None and specification.loader is not None
    bootstrap = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(bootstrap)
    self.assertEqual(
        analyzer._ORIGINAL_BINDING, bootstrap.EXPECTED_ORIGINAL_BINDING
    )
    for key, value in analyzer._ORIGINAL_BINDING.items():
      if key.endswith('_sha256'):
        self.assertTrue(analyzer._is_sha256(value), key)

  def test_compiler_rehashes_all_three_ir_files_and_comparison(self):
    run = self.root / 'run'
    directory = run / 'compiler/eight_row'
    directory.mkdir(parents=True)
    filenames = {
        'stablehlo': 'graph.stablehlo.mlir',
        'hlo': 'graph.pre_backend.hlo.txt',
        'compiled_hlo': 'graph.compiled.hlo.txt',
    }
    artifacts, original_artifacts = {}, {}
    for name, filename in filenames.items():
      path = directory / filename
      path.write_text(f'{name} fixture\n', encoding='utf-8')
      artifacts[name] = {
          'path': str(path.resolve()), 'sha256': _sha(path),
          'size_bytes': path.stat().st_size,
      }
      original_artifacts[name] = {
          'path': f'/frozen/original/{filename}',
          'sha256': _sha(path), 'size_bytes': path.stat().st_size,
      }
    fingerprint = hashlib.sha256(
        bytes.fromhex(artifacts['compiled_hlo']['sha256'])
    ).hexdigest()
    original = {
        'executable_name': 'eight_row', 'compile_count': 1,
        'compile_seconds': 1.0, 'executable_fingerprint': fingerprint,
        'artifacts': original_artifacts,
    }
    comparison = {
        name: {'sha256_exact': True, 'size_exact': True} for name in filenames
    }
    compiler = {
        'executable_name': 'eight_row', 'compile_count': 1,
        'compile_seconds': 1.0, 'executable_fingerprint': fingerprint,
        'artifacts': artifacts,
        'program_signatures': {
            'selection': {}, 'target': {}, 'eight_interventions': {},
        },
        'original_v3_3_compiler_binding': original,
        'original_graph_comparison': comparison,
        'original_executable_fingerprint_exact': True,
        'graph_and_hlo_exact_to_original_v3_3': True,
    }
    _write_json(directory / 'COMPILER_PROVENANCE.json', compiler)
    observed, audit = analyzer._validate_compiler(run, compiler, original)
    self.assertEqual(observed, fingerprint)
    self.assertTrue(audit['graph_and_hlo_exact_to_original_v3_3'])
    (directory / filenames['hlo']).write_text('tampered\n', encoding='utf-8')
    with self.assertRaisesRegex(analyzer.AnalysisError, 'bytes changed'):
      analyzer._validate_compiler(run, compiler, original)

  def test_import_provenance_requires_exact_seven_sidecar_sources(self):
    bundle = self.root / 'repo'
    upstream = self.root / 'alphagenome'
    bundle.mkdir()
    upstream.mkdir()
    sources = []
    for index in range(7):
      source = bundle / f'source-{index}.py'
      source.write_text(f'# {index}\n', encoding='utf-8')
      sources.append(source)
    generated = set(analyzer._v33.UPSTREAM_GENERATED_MODULE_NAMES)
    names = sorted(generated) + [f'alphagenome.fake_{index}' for index in range(22)]
    inventory, modules = {}, []
    for index, name in enumerate(names):
      path = upstream / f'module-{index}.py'
      path.write_text(f'# upstream {name}\n', encoding='utf-8')
      binding = {
          'relative_path': path.name, 'sha256': _sha(path),
          'size_bytes': path.stat().st_size,
      }
      inventory[name] = binding
      modules.append({
          'name': name, 'path': str(path.resolve()),
          'root': 'upstream_alphagenome_checkout',
          'sha256': binding['sha256'], 'size_bytes': binding['size_bytes'],
      })
    for name in (
        'alphagenome_research.model.model',
        'alphagenome_research.model.dna_model',
        'alphagenome_research.model.interpretability',
    ):
      path = bundle / f'{name.rsplit(".", 1)[-1]}.py'
      path.write_text(f'# {name}\n', encoding='utf-8')
      modules.append({
          'name': name, 'path': str(path.resolve()),
          'root': 'alphagenome_research_checkout', 'sha256': _sha(path),
          'size_bytes': path.stat().st_size,
      })
    alias = bundle / 'run_encoder_skip_ood_sidecar_v3_3_2.py'
    alias.write_text('# alias\n', encoding='utf-8')
    for name in ('__main__', '__mp_main__'):
      modules.append({
          'name': name, 'path': str(alias.resolve()),
          'root': 'alphagenome_research_checkout', 'sha256': _sha(alias),
          'size_bytes': alias.stat().st_size,
      })
    exception = {'fixture': True}
    attestation = {
        'git_head': '1' * 40, 'tracked_head_clean': True,
        'imported_module_count': 26,
        'imported_modules': {
            name: {
                **binding,
                'path': str((upstream / binding['relative_path']).resolve()),
                'source_kind': (
                    'generated_exact_byte_exception' if name in generated
                    else 'tracked'
                ),
            }
            for name, binding in inventory.items()
        },
        'tracked_imported_module_count': 22,
        'generated_imported_module_count': 4,
        'generated_binding_exception': exception,
    }
    source_binding = {
        str(path.resolve()): {
            'sha256': _sha(path), 'size_bytes': path.stat().st_size,
        } for path in sources
    }
    value = {
        'module_count': len(modules), 'modules': modules,
        'upstream_source_attestation': attestation,
        'v3_3_2_sidecar_sources': source_binding,
    }
    provenance = self.root / 'IMPORT_PROVENANCE.json'
    _write_json(provenance, value)
    freeze = {
        'upstream_imported_modules': inventory,
        'upstream_generated_binding_exception': exception,
        'upstream_alphagenome_git_head': '1' * 40,
    }
    with mock.patch.object(
        analyzer, '_sidecar_source_paths', return_value=tuple(sources)
    ):
      rows = analyzer._validate_import_file(
          provenance, _sha(provenance), bundle_root=bundle, freeze=freeze
      )
      self.assertIn('__mp_main__', rows)
      value['v3_3_2_sidecar_sources'].pop(str(sources[-1].resolve()))
      _write_json(provenance, value)
      with self.assertRaisesRegex(analyzer.AnalysisError, 'seven-source'):
        analyzer._validate_import_file(
            provenance, _sha(provenance), bundle_root=bundle, freeze=freeze
        )

  def test_protobuf_generated_output_path_keys_are_rehashed(self):
    run = self.root / 'run'
    run.mkdir()
    outputs = []
    for suffix in ('py', 'pyi'):
      path = self.root / f'calibration_scores_pb2.{suffix}'
      path.write_text(f'# {suffix}\n', encoding='utf-8')
      outputs.append(path)
    binding = {
        str(path.resolve()): {
            'sha256': _sha(path), 'size_bytes': path.stat().st_size,
        } for path in outputs
    }
    value = {'generated_outputs': binding}
    path = run / 'PROTOBUF_PROVENANCE.json'
    _write_json(path, value)
    completion = {'protobuf_provenance_sha256': _sha(path)}
    audit = analyzer._validate_protobuf(
        run, completion, {'protobuf_binding': value}
    )
    self.assertEqual(audit['generated_output_count'], 2)
    outputs[0].write_text('# tampered\n', encoding='utf-8')
    with self.assertRaisesRegex(analyzer.AnalysisError, 'bytes changed'):
      analyzer._validate_protobuf(run, completion, {'protobuf_binding': value})

  def test_exact_prerequisite_tree_rejects_extra_directory_and_symlink(self):
    directory = self.root / 'prior'
    directory.mkdir()
    for name in ('A.json', 'B.md'):
      (directory / name).write_text(name, encoding='utf-8')
    mapping = {
        path.name: {'sha256': _sha(path), 'size_bytes': path.stat().st_size}
        for path in directory.iterdir()
    }
    tree = analyzer._tree_digest(directory.iterdir(), directory)
    analyzer._validate_exact_flat_tree(directory, mapping, tree, 'prior')
    extra = directory / 'empty'
    extra.mkdir()
    with self.assertRaisesRegex(analyzer.AnalysisError, 'directory/symlink'):
      analyzer._validate_exact_flat_tree(directory, mapping, tree, 'prior')
    extra.rmdir()
    (directory / 'link').symlink_to(directory / 'A.json')
    with self.assertRaisesRegex(analyzer.AnalysisError, 'directory/symlink'):
      analyzer._validate_exact_flat_tree(directory, mapping, tree, 'prior')
    (directory / 'link').unlink()
    os.mkfifo(directory / 'fifo')
    with self.assertRaisesRegex(analyzer.AnalysisError, 'directory/symlink'):
      analyzer._validate_exact_flat_tree(directory, mapping, tree, 'prior')


class RunnerShapedIntegrationTest(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.run = self.root / 'run'
    self.original = self.root / 'original'
    self.run.mkdir()
    (self.run / 'compiler/eight_row').mkdir(parents=True)
    self.cases = {order: _case(order) for order in range(20)}
    self.sequences = {
        order: {'reference': f'{order:064x}', 'alternate': f'{order + 100:064x}'}
        for order in range(20)
    }
    self.original_manifest = {}
    _write_json(self.original / 'RUN_COMPLETE.json', {'eight_row_compiler': {}})
    for name in (
        'ATTEMPT_STARTED.json', 'IMPORT_PROVENANCE_PRE_MODEL.json',
        'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
        'IMPORT_PROVENANCE.json', 'PROTOBUF_PROVENANCE.json',
    ):
      _write_json(self.run / name, {'fixture': name})

  def tearDown(self):
    self.temporary.cleanup()

  def _completion(self, *, reason=None, count=80) -> dict:
    return CompletionPrefixTest().completion(reason=reason, count=count)

  def _write_tree(self, *, reason=None, count=80) -> None:
    completion = self._completion(reason=reason, count=count)
    mapping = {}
    for index, (order, anchor) in enumerate(analyzer._execution_order()[:count]):
      status = (
          'invalid' if reason == 'ood_tooling_failure' and index == count - 1
          else 'complete'
      )
      record = _record(
          original_root=self.original,
          original_manifest=self.original_manifest,
          cases=self.cases, sequence_bindings=self.sequences,
          order=order, anchor=anchor, execution_index=index, status=status,
      )
      relative = analyzer._artifact_relative(self.cases[order], anchor)
      path = self.run / relative
      _write_json(path, record)
      mapping[relative] = _sha(path)
    manifest = {
        'artifact_count': count,
        'artifact_sha256': mapping,
        'artifact_tree_sha256': analyzer._tree_digest(
            (self.run / relative for relative in mapping), self.run
        ),
    }
    completion['raw_manifest'] = manifest
    if reason == 'ood_tooling_failure':
      order, anchor = analyzer._execution_order()[count - 1]
      completion['message'] = (
          f'OOD sidecar audit failed at order={order}, anchor_id={anchor}.'
      )
      complete_rows = analyzer._execution_order()[:count - 1]
      completion['id0_all20'] = sum(anchor == 0 for _, anchor in complete_rows) == 20
      completion['id255_all20'] = sum(anchor == 255 for _, anchor in complete_rows) == 20
    _write_json(self.run / 'RAW_MANIFEST.json', manifest)
    _write_json(self.run / 'RUN_COMPLETE.json', completion)

  def _rewrite_final_invalid(self, *, stage: str) -> None:
    completion = json.loads((self.run / 'RUN_COMPLETE.json').read_text())
    count = completion['ood_anchor_record_count']
    order, anchor = analyzer._execution_order()[count - 1]
    relative = analyzer._artifact_relative(self.cases[order], anchor)
    path = self.run / relative
    record = json.loads(path.read_text())
    if stage == 'early_readout':
      record.update({
          'intended_target_readout': None,
          'intended_repeat_target_readout': None,
          'unrelated_target_readout': None,
          'unrelated_repeat_target_readout': None,
          'rowwise_trace_fingerprints': None,
          'intended_trace_fingerprint': None,
          'intended_repeat_trace_fingerprint': None,
          'unrelated_trace_fingerprint': None,
          'unrelated_repeat_trace_fingerprint': None,
          'original_artifact_bindings': None,
          'raw_movement': None,
          'failure': {'type': 'ValueError', 'message': 'readout failed'},
      })
    elif stage == 'binding':
      record.update({
          'original_artifact_bindings': None, 'raw_movement': None,
          'failure': {'type': 'OSError', 'message': 'binding failed'},
      })
    else:
      raise AssertionError(stage)
    _write_json(path, record)
    manifest = completion['raw_manifest']
    manifest['artifact_sha256'][relative] = _sha(path)
    manifest['artifact_tree_sha256'] = analyzer._tree_digest(
        (self.run / item for item in manifest['artifact_sha256']), self.run
    )
    completion['raw_manifest'] = manifest
    _write_json(self.run / 'RAW_MANIFEST.json', manifest)
    _write_json(self.run / 'RUN_COMPLETE.json', completion)

  def _analyze(self):
    start_audit = {'v3_3_1_status': {'state': 'completed'}}
    complete = json.loads((self.run / 'RUN_COMPLETE.json').read_text())
    compiler = {
        'executable_fingerprint': 'e' * 64,
        'graph_and_hlo_exact_to_original_v3_3': complete[
            'graph_and_hlo_exact_to_original_v3_3'
        ],
    }
    with (
        mock.patch.object(analyzer, '_ORIGINAL_RUN_DIR', self.original),
        mock.patch.object(analyzer._v33, '_load_cases', return_value=self.cases),
        mock.patch.object(
            analyzer, '_validate_freeze_and_start',
            return_value=(
                {}, '9' * 64, start_audit,
                self.original_manifest, self.sequences,
            ),
        ),
        mock.patch.object(
            analyzer, '_validate_compiler', return_value=('e' * 64, compiler)
        ),
        mock.patch.object(analyzer, '_validate_imports', return_value={'ok': True}),
        mock.patch.object(analyzer, '_validate_protobuf', return_value={'ok': True}),
    ):
      return analyzer.analyze(self.run, bundle_root=analyzer._REPO_ROOT)

  def test_complete_80_is_structurally_eligible_but_never_interprets(self):
    self._write_tree()
    result = self._analyze()
    self.assertEqual(result['decision'], 'sidecar_complete_structural_audit')
    self.assertTrue(result['combined_analysis_permitted'])
    self.assertFalse(result['scientific_summary_computed'])
    self.assertFalse(result['shapley_or_nomination_computed'])
    self.assertIsNone(result['nomination'])

  def test_first_invalid_ood_prefix_audits_without_interpretation(self):
    self._write_tree(reason='ood_tooling_failure', count=2)
    result = self._analyze()
    self.assertEqual(result['decision'], 'controlled_stop_ood_tooling_failure')
    self.assertFalse(result['combined_analysis_permitted'])
    self.assertFalse(result['nomination_performed'])
    self.assertIsNone(result['resolution_analysis'])

  def test_controlled_prefix_audits_early_readout_failure(self):
    self._write_tree(reason='ood_tooling_failure', count=2)
    self._rewrite_final_invalid(stage='early_readout')
    result = self._analyze()
    self.assertEqual(result['decision'], 'controlled_stop_ood_tooling_failure')
    self.assertEqual(result['sidecar_audit']['invalid_record_count'], 1)

  def test_controlled_prefix_audits_binding_stage_failure(self):
    self._write_tree(reason='ood_tooling_failure', count=2)
    self._rewrite_final_invalid(stage='binding')
    result = self._analyze()
    self.assertEqual(result['decision'], 'controlled_stop_ood_tooling_failure')
    self.assertFalse(result['combined_analysis_permitted'])

  def test_compiler_mismatch_zero_prefix_audits_without_interpretation(self):
    self._write_tree(reason='compiler_graph_mismatch', count=0)
    result = self._analyze()
    self.assertEqual(result['decision'], 'controlled_stop_compiler_graph_mismatch')
    self.assertEqual(result['sidecar_audit']['audited_record_count'], 0)
    self.assertFalse(result['combined_analysis_permitted'])
    self.assertFalse(result['scientific_summary_computed'])

  def test_missing_raw_fails_closed(self):
    self._write_tree(reason='ood_tooling_failure', count=2)
    path = self.run / analyzer._artifact_relative(self.cases[0], 127)
    path.unlink()
    with self.assertRaisesRegex(analyzer.AnalysisError, 'Raw tree'):
      self._analyze()


if __name__ == '__main__':
  unittest.main()
