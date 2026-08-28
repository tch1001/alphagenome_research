#!/usr/bin/env python3
"""CPU-only tests for the standalone v3.3.4.5 structural analyzer."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock


_HERE = Path(__file__).resolve().parent
_ANALYZER_PATH = _HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_4_5.py'
_SPEC = importlib.util.spec_from_file_location('_v334_analyzer_test_target', _ANALYZER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
analyzer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analyzer)


def _consumed_prefix() -> tuple[dict[str, object], dict[str, object]]:
  prefix = analyzer._expected_consumed_v3343_prefix()  # pylint: disable=protected-access
  return prefix, analyzer._content_binding(prefix)  # pylint: disable=protected-access


def _consumed_v3344_prefix() -> tuple[dict[str, object], dict[str, object]]:
  prefix = analyzer._expected_consumed_v3344_prefix()  # pylint: disable=protected-access
  return prefix, analyzer._content_binding(prefix)  # pylint: disable=protected-access


def _both_consumed_prefix_fields() -> dict[str, object]:
  v3343, v3343_binding = _consumed_prefix()
  v3344, v3344_binding = _consumed_v3344_prefix()
  return {
      'prior_v3_3_4_3_consumed_preflight_prefix': v3343,
      'prior_v3_3_4_3_consumed_preflight_prefix_content_binding': (
          v3343_binding
      ),
      'prior_v3_3_4_4_consumed_preflight_prefix': v3344,
      'prior_v3_3_4_4_consumed_preflight_prefix_content_binding': (
          v3344_binding
      ),
  }


def _source_audit(value: object = True) -> dict[str, object]:
  return {name: value for name in analyzer._SOURCE_AUDIT_KEYS}  # pylint: disable=protected-access


def _same_object() -> dict[str, object]:
  return {
      'lower_call_count': 1, 'compile_call_count': 1,
      'stablehlo_read_from_lowered_object': True,
      'pre_backend_hlo_read_from_lowered_object': True,
      'compile_argument_is_lowered_object': True,
      'compiled_hlo_read_from_compiled_object': True,
      'signature_attestation_from_apply_arguments': True,
      'apply_callable_is_compiled_object': True,
      'compiler_record_is_gate_record': True,
      'lowered_python_id': 7, 'compiled_python_id': 8,
  }


def _readout(seed: float = 0.0) -> dict[str, object]:
  logits = []
  margins = []
  totals = []
  means = []
  for row in range(8):
    row_logits = []
    row_margins = []
    for endpoint in range(2):
      relevant = analyzer._f32(seed + row + endpoint + 1.0, 'fixture')  # pylint: disable=protected-access
      padding = analyzer._f32(seed + row + endpoint, 'fixture')  # pylint: disable=protected-access
      margin = analyzer._f32(relevant - padding, 'fixture')  # pylint: disable=protected-access
      row_logits.append([relevant, padding])
      row_margins.append(margin)
    total = analyzer._f32(sum(row_margins), 'fixture')  # pylint: disable=protected-access
    logits.append(row_logits)
    margins.append(row_margins)
    totals.append(total)
    means.append(analyzer._f32(total / 2.0, 'fixture'))  # pylint: disable=protected-access
  return {
      'endpoint_axis': ['acceptor', 'donor'],
      'selected_logit_axis': ['relevant_class', 'padding_class'],
      'selected_logits': logits, 'endpoint_margins': margins,
      'means': means, 'totals': totals, 'num_values': 2,
  }


def _readout_rows(values: list[float]) -> dict[str, object]:
  value = _readout()
  for row, seed in enumerate(values):
    for endpoint in range(2):
      relevant = analyzer._f32(seed + endpoint + 1.0, 'fixture')  # pylint: disable=protected-access
      padding = analyzer._f32(seed + endpoint, 'fixture')  # pylint: disable=protected-access
      value['selected_logits'][row][endpoint] = [relevant, padding]
      value['endpoint_margins'][row][endpoint] = analyzer._f32(  # pylint: disable=protected-access
          relevant - padding, 'fixture'
      )
    value['totals'][row] = analyzer._f32(2.0, 'fixture')  # pylint: disable=protected-access
    value['means'][row] = analyzer._f32(1.0, 'fixture')  # pylint: disable=protected-access
  return value


def _readouts_for_anchor(anchor: int) -> tuple[dict[str, object], dict[str, object]]:
  intended_values = [0.0, 10.0, 20.0, 10.0, 40.0, 0.0, 60.0, 70.0]
  unrelated_values = [0.0, 10.0, 26.0, 10.0, 47.0, 0.0, 60.0, 70.0]
  if anchor == 0:
    intended_values[2], intended_values[4] = 10.0, 0.0
    unrelated_values[2], unrelated_values[4] = 10.0, 0.0
  elif anchor == 255:
    intended_values[2], intended_values[4] = 0.0, 10.0
    unrelated_values[2], unrelated_values[4] = 60.0, 70.0
  return _readout_rows(intended_values), _readout_rows(unrelated_values)


def _runtime_route(anchor: int, donor_rows: tuple[int, ...]) -> dict[str, object]:
  t, e_mask = divmod(anchor, 128)
  active = [False, False, True, True, True, True, False, False]
  def route(components: int, enabled: list[bool]) -> dict[str, object]:
    return {
        'donor_batch_indices': [list(donor_rows) for _ in range(components)],
        'natural_identity_batch_indices': [list(analyzer.IDENTITY_ROWS) for _ in range(components)],
        'transfer_mask': [
            [flag and row_active for row_active in active] for flag in enabled
        ],
    }
  residual = {
      'donor_batch_indices': [
          [[row] * 24 for row in range(8)] for _ in range(9)
      ],
      'transfer_mask': [[[False] * 24 for _ in range(8)] for _ in range(9)],
  }
  return {
      'transformer_output': route(1, [bool(t)]),
      'encoder_skips': route(7, [bool(e_mask & (1 << i)) for i in range(7)]),
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


def _fingerprint_rows() -> dict[str, object]:
  rows = [{
      'row': index, 'shape': [2], 'dtype': 'float32',
      'size_bytes': 8, 'sha256': hashlib.sha256(bytes([index])).hexdigest(),
  } for index in range(8)]
  return {
      'full_shape': [8, 2], 'dtype': 'float32', 'row_count': 8,
      'rows': rows,
      'collision_semantics': (
          'SHA-256 per exact row byte string; direct live equality is the gate.'
      ),
  }


def _rowwise() -> dict[str, object]:
  call = {
      'natural_final_embeddings': _fingerprint_rows(),
      'effective_final_embeddings': _fingerprint_rows(),
      'transformer_output_natural_fingerprint': {
          'shape': [8, 4], 'dtype': 'float32',
          'values': [[0.0] * 4 for _ in range(8)],
      },
      'encoder_skips_natural_fingerprints': {
          'shape': [7, 8, 4], 'dtype': 'float32',
          'values': [[[0.0] * 4 for _ in range(8)] for _ in range(7)],
      },
  }
  return {name: copy.deepcopy(call) for name in (
      'intended', 'intended_repeat', 'unrelated', 'unrelated_repeat'
  )}


def _checks(anchor: int) -> dict[str, object]:
  value = {name: True for name in analyzer._CHECK_KEYS}  # pylint: disable=protected-access
  value.update({
      'corrected_host_assertion_version': 'v3.3.4.5',
      'natural_final_invariant_rows': list(analyzer.INVARIANT_ROWS),
      'active_rows_cross_call_equality_not_required': list(analyzer.ACTIVE_ROWS),
      'normalization_computed': False,
  })
  for name in (
      'id0_all8_natural_final_exact_between_calls',
      'id0_within_call_natural_final_recipient_noop_exact',
      'id0_all8_endpoint_exact_between_calls', 'id0_recipient_noop_exact',
  ):
    value[name] = anchor == 0
  for name in (
      'id255_intended_endpoint_closure_exact',
      'id255_unrelated_endpoint_closure_exact',
  ):
    value[name] = anchor == 255
  return value


def _record_fixture(
    *, case: dict[str, object], donor: dict[str, object], anchor: int,
    execution_index: int, freeze_sha: str, executable: str,
    original_manifest: dict[str, str], sequences: dict[int, object],
    authorization: dict[str, object], source: dict[str, object],
    same_object: dict[str, object], started: dict[str, object],
    completed: dict[str, object],
) -> dict[str, object]:
  intended, unrelated = _readouts_for_anchor(anchor)
  e_players = ['E64', 'E32', 'E16', 'E8', 'E4', 'E2', 'E1']
  t, e_mask = divmod(anchor, 128)
  e_bits = [bool(e_mask & (1 << index)) for index in range(7)]
  links = {}
  for name, linked, family, linked_anchor in (
      ('recipient_identity', case, 'identity', None),
      ('donor_identity', donor, 'identity', None),
      ('recipient_six_row_coalition', case, 'coalition', anchor),
  ):
    relative = analyzer._original_relative(  # pylint: disable=protected-access
        linked, family, linked_anchor
    )
    links[name] = {'path': relative, 'sha256': original_manifest[relative]}
  trace = {'sha256': 'c' * 64, 'leaves': []}
  started_rows = []
  completed_rows = []
  for call_index in range(4):
    index = 4 * execution_index + call_index
    started_path = f'dispatch_journal/started/{index:03d}.json'
    completed_path = f'dispatch_journal/completed/{index:03d}.json'
    started_rows.append({'path': started_path, **started[started_path]})
    completed_rows.append({'path': completed_path, **completed[completed_path]})
  return {
      'status': 'complete',
      'family': 'v3_3_4_5_unrelated_donor_sidecar_anchor',
      'script_version': analyzer.SCRIPT_VERSION,
      'amendment_sha256': analyzer.AMENDMENT_SHA256,
      'amendment_commit': analyzer.AMENDMENT_COMMIT,
      'original_protocol_sha256': analyzer.ORIGINAL_PROTOCOL_SHA256,
      'freeze_sha256': freeze_sha,
      'external_freeze_authorization': authorization,
      'execution_index': execution_index,
      'sidecar_execution_index': execution_index,
      'execution_order': 'recipient-major, anchor-minor',
      'eight_row_executable_fingerprint': executable,
      'same_eight_row_compiled_executable': True,
      'six_row_executable_used': False,
      'recipient_case': case, 'donor_case': donor,
      'coalition': {
          'coalition_id': anchor, 't': t, 'e_mask': e_mask,
          'e_bits': e_bits, 'e_bits_binary': format(e_mask, '07b'),
          'enabled_players': (['T'] if t else []) + [
              player for player, enabled in zip(e_players, e_bits, strict=True)
              if enabled
          ],
          'coalition_bit_order': [*e_players, 'T'],
          'shapley_player_order': ['T', *e_players],
      },
      'batch_roles': list(analyzer.EIGHT_ROLES),
      'natural_identity_rows': list(analyzer.IDENTITY_ROWS),
      'intended_donor_rows': list(analyzer.INTENDED_DONOR_ROWS),
      'unrelated_donor_rows': list(analyzer.UNRELATED_DONOR_ROWS),
      'invariant_rows_between_calls': list(analyzer.INVARIANT_ROWS),
      'active_recipient_rows': list(analyzer.ACTIVE_ROWS),
      'active_recipient_cross_call_equality_gate': False,
      'active_recipient_cross_call_inequality_gate': False,
      'original_artifact_bindings': links,
      'original_ood_records_used_as_data': False,
      'recipient_sequence_sha256': sequences[int(case['order'])],
      'donor_sequence_sha256': sequences[int(donor['order'])],
      'runtime_interventions': {
          'intended': _runtime_route(anchor, analyzer.INTENDED_DONOR_ROWS),
          'unrelated': _runtime_route(anchor, analyzer.UNRELATED_DONOR_ROWS),
      },
      'intended_target_readout': intended,
      'intended_repeat_target_readout': copy.deepcopy(intended),
      'unrelated_target_readout': unrelated,
      'unrelated_repeat_target_readout': copy.deepcopy(unrelated),
      'intended_trace_fingerprint': trace,
      'intended_repeat_trace_fingerprint': copy.deepcopy(trace),
      'unrelated_trace_fingerprint': copy.deepcopy(trace),
      'unrelated_repeat_trace_fingerprint': copy.deepcopy(trace),
      'rowwise_trace_fingerprints': _rowwise(),
      'raw_movement': {
          call: {
              'reference_into_alternate': 0.0,
              'alternate_into_reference': 0.0,
          } for call in ('intended', 'unrelated')
      },
      'model_apply_count_through_record': 4 * (execution_index + 1),
      'checks': _checks(anchor), 'failure': None,
      'seconds': {call: 0.0 for call in (
          'intended', 'intended_repeat', 'unrelated', 'unrelated_repeat'
      )},
      'dispatch_started_bindings': started_rows,
      'dispatch_completed_bindings': completed_rows,
      'source_input_audit': source,
      'source_input_audit_content_binding': analyzer._content_binding(source),  # pylint: disable=protected-access
      'same_object_attestation': same_object,
      'same_object_attestation_content_binding': analyzer._content_binding(same_object),  # pylint: disable=protected-access
      'confirmation_scope_disclosure': analyzer.CONFIRMATION_DISCLOSURE,
      'created_at_unix_s': 1.0,
  }


def _phase(**updates: bool) -> dict[str, bool]:
  result = {name: False for name in analyzer._PHASE_STATE_KEYS}  # pylint: disable=protected-access
  result.update(updates)
  return result


def _publication_audit() -> dict[str, object]:
  return {
      'schema_version': analyzer.PUBLICATION_SCHEMA_VERSION,
      'method': analyzer.PUBLICATION_METHOD,
      'successful_final_count_before_terminal': 0,
      'successful_final_bindings_before_terminal': {},
      'temporary_orphan_count': 0, 'temporary_orphan_bindings': {},
      'durability_uncertain_final_count': 0,
      'durability_uncertain_final_bindings': {},
      'preexisting_entry_count': 0, 'preexisting_entry_states': {},
      'no_new_entry_failure': False, 'publication_failure': None,
      'no_published_final_deleted': True,
      'no_temp_or_final_reused': True, 'no_publication_retry': True,
  }


def _model_cache_evidence(
    *, pre_import_files_present: bool, compile_skipped: bool | None,
) -> dict[str, object]:
  return {
      'pre_import_files_present': pre_import_files_present,
      'default_user_cache_path_eligible': False,
      'persistent_compilation_cache_hit_reported': False,
      'executable_deserialized': False,
      'compile_skipped': compile_skipped,
      'compile_stage_not_applicable': compile_skipped is None,
      'old_cache_input_opened': False,
      'routing_exact': True,
      'cache_hit': pre_import_files_present,
  }


def _live_binding(path: Path) -> dict[str, object]:
  status = path.lstat()
  return {
      'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
      'size_bytes': status.st_size,
      'mode': f'{status.st_mode & 0o7777:04o}',
      'st_dev': status.st_dev, 'st_ino': status.st_ino,
      'st_nlink': status.st_nlink,
  }


def _entry_state(path: Path) -> dict[str, object]:
  return analyzer._observe_entry_state(path)  # pylint: disable=protected-access


def _write_json_0400(path: Path, value: object) -> None:
  path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
  path.write_text(
      json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n',
      encoding='utf-8',
  )
  path.chmod(0o400)


def _cases() -> dict[int, dict[str, object]]:
  return {
      order: {'order': order, 'variant_id': f'fixture-{order}'}
      for order in range(20)
  }


def _dispatch_event(
    *, index: int, completed: bool, runner_pid: int,
    source_sha: str, object_sha: str, started_sha: str | None = None,
) -> dict[str, object]:
  value: dict[str, object] = {
      'schema_version': 'v3.3.4.5-dispatch-event-v1',
      'event': 'dispatch_completed' if completed else 'dispatch_started',
      'attempt_id': analyzer.ATTEMPT_ID,
      'script_version': analyzer.SCRIPT_VERSION,
      **analyzer._event_identity(index, _cases()),  # pylint: disable=protected-access
      'runner_pid': runner_pid,
      'source_input_audit_sha256': source_sha,
      'same_object_attestation_sha256': object_sha,
  }
  if completed:
    value.update({
        'started_event_sha256': started_sha,
        'returned': True,
        'completed_at_unix_s': 2.0,
    })
  else:
    value['started_at_unix_s'] = 1.0
  return value


def _terminal(
    *, status: str, reason: str | None, k: int, d: int,
    attempted: int, completed: int, phase: dict[str, bool],
    failure_phase: str | None,
) -> tuple[dict[str, object], dict[str, object]]:
  authorization = {
      'git_head': 'a' * 40, 'freeze_path': '/fixed/freeze.json',
      'freeze_sha256': 'f' * 64, 'freeze_size_bytes': 1,
      'live_equals_git_show': True, 'tracked_clean': True,
      'authorization_source': 'external_post_commit_audit',
  }
  source = _source_audit()
  consumed_fields = _both_consumed_prefix_fields()
  full = status == 'complete_structural_sidecar'
  node = {name: None for name in analyzer._RUN_COMPLETE_KEYS}  # pylint: disable=protected-access
  node.update({
      'status': status, 'stop_reason': reason,
      'message': 'fixture',
      'failure': None if full else {
          'type': 'FixtureStop', 'message': 'fixture', 'traceback': 'fixture'
      },
      'attempt_id': analyzer.ATTEMPT_ID,
      'script_version': analyzer.SCRIPT_VERSION,
      'amendment_sha256': analyzer.AMENDMENT_SHA256,
      'amendment_commit': analyzer.AMENDMENT_COMMIT,
      'original_protocol_sha256': analyzer.ORIGINAL_PROTOCOL_SHA256,
      'freeze_sha256': 'f' * 64, 'git_head': 'a' * 40,
      'external_freeze_authorization': authorization, 'runner_pid': 42,
      'started_at_unix_s': 10.0, 'completed_at_unix_s': 11.0,
      'phase_state': phase,
      'terminal_detail': {
          'k_valid_records': k, 'd_completed': d,
          'failed_execution_index': None if full or k == 0 and d == 0 else k,
          'failed_call_role': None,
          'failure_phase': failure_phase,
          'forbidden_operation': None, 'provenance_artifact_role': None,
      },
      'budgets': {
          'max_wall_time_seconds': 7200, 'elapsed_wall_time_seconds': 1.0,
          'wall_time_within_budget': True,
          'max_output_bytes': 1_073_741_824,
          'preterminal_output_bytes': 100,
          'run_complete_size_cap_bytes': 16_777_216,
          'preterminal_plus_terminal_cap_within_budget': True,
      },
      'source_input_audit': source,
      'source_input_audit_content_binding': analyzer._content_binding(source),  # pylint: disable=protected-access
      'dispatch_journal': {
          'started_count': attempted, 'completed_count': completed,
          'started_bindings': {}, 'completed_bindings': {},
          'started_tree_sha256': analyzer.EMPTY_SHA256,
          'completed_tree_sha256': analyzer.EMPTY_SHA256,
          'started_prefix_exact': True, 'completed_prefix_exact': True,
      },
      'valid_record_count': k,
      'model_apply_attempt_count': attempted,
      'model_apply_success_count': completed,
      'expected_model_apply_count': 320,
      'eight_row_lower_attempt_count': int(phase['lower_attempted']),
      'eight_row_compile_attempt_count': int(phase['compile_attempted']),
      'eight_row_successful_compile_count': int(phase['compile_succeeded']),
      'six_row_compile_count': 0, 'identity_rerun_count': 0,
      'main_cube_rerun_count': 0, 'old_ood_records_reused': 0,
      'confirmation_model_calls': 0,
      'all_80_recipient_anchors_complete': full,
      'id0_all20': full, 'id255_all20': full,
      'prior_v3_3_3_binding': {}, 'prior_v3_3_3_1_archive_binding': {},
      'confirmation_scope_disclosure': analyzer.CONFIRMATION_DISCLOSURE,
      'scientific_summary_computed': False,
      'donor_normalization_computed': False,
      'shapley_or_nomination_computed': False,
      'interaction_or_resolution_computed': False,
      'nomination_performed': False, 'combined_analysis_permitted': False,
      'no_retry': True,
      'publication_audit': _publication_audit(),
      **copy.deepcopy(consumed_fields),
  })
  start = {
      'external_freeze_authorization': authorization, 'runner_pid': 42,
      'started_at_unix_s': 10.0,
      **copy.deepcopy(consumed_fields),
  }
  return node, start


class StandaloneAndPrimitiveTest(unittest.TestCase):

  def test_module_has_no_old_analyzer_or_model_import(self):
    tree = ast.parse(_ANALYZER_PATH.read_text(encoding='utf-8'))
    imports = []
    for node in ast.walk(tree):
      if isinstance(node, ast.Import):
        imports.extend(alias.name for alias in node.names)
      elif isinstance(node, ast.ImportFrom):
        imports.append(node.module or '')
    self.assertFalse(any('analyze_encoder_skip_ood_sidecar_v3_3' in name for name in imports))
    self.assertFalse(any(name.startswith(('jax', 'jaxlib', 'alphagenome')) for name in imports))

  def test_cpu_guard_rejects_all_forbidden_module_prefixes(self):
    for name in ('jax', 'jaxlib.xla_extension', 'alphagenome',
                 'alphagenome_research.model.dna_model'):
      with self.subTest(name=name), mock.patch.dict(sys.modules, {name: object()}):
        with self.assertRaisesRegex(analyzer.AnalysisError, 'forbidden'):
          analyzer._assert_cpu_only('fixture')  # pylint: disable=protected-access

  def test_canonical_hash_is_key_order_independent_and_list_order_sensitive(self):
    left = {'b': [1, 2], 'a': {'dtype': 'f32'}}
    right = {'a': {'dtype': 'f32'}, 'b': [1, 2]}
    self.assertEqual(analyzer._canonical_json_sha256(left), analyzer._canonical_json_sha256(right))  # pylint: disable=protected-access
    self.assertNotEqual(analyzer._canonical_json_sha256(left), analyzer._canonical_json_sha256({'b': [2, 1], 'a': {'dtype': 'f32'}}))  # pylint: disable=protected-access

  def test_content_binding_rejects_semantic_tamper(self):
    value = _source_audit()
    binding = analyzer._content_binding(value)  # pylint: disable=protected-access
    self.assertEqual(analyzer._validate_content_bound_object(value, binding, 'x'), value)  # pylint: disable=protected-access
    changed = dict(value)
    changed['checkpoint_exact'] = False
    with self.assertRaisesRegex(analyzer.AnalysisError, 'binding'):
      analyzer._validate_content_bound_object(changed, binding, 'x')  # pylint: disable=protected-access

  def test_same_object_success_is_exact_and_bound(self):
    value = _same_object()
    binding = analyzer._content_binding(value)  # pylint: disable=protected-access
    self.assertEqual(analyzer._validate_same_object_success(value, binding, 'x'), value)  # pylint: disable=protected-access
    value['compile_call_count'] = 2
    with self.assertRaisesRegex(analyzer.AnalysisError, 'successful object flow'):
      analyzer._validate_same_object_success(value, analyzer._content_binding(value), 'x')  # pylint: disable=protected-access

  def test_entry_abi_replaces_only_backend_fingerprint(self):
    line = 'HloModule x, fingerprint_before_lhs="1a2B", literal="keep"'
    normalized, digest = analyzer._normalized_entry_abi(line + '\nbody')  # pylint: disable=protected-access
    expected = line.replace('"1a2B"', '"<backend-generated>"')
    self.assertEqual(normalized, expected)
    self.assertEqual(digest, hashlib.sha256(expected.encode()).hexdigest())
    with self.assertRaises(analyzer.AnalysisError):
      analyzer._normalized_entry_abi('HloModule x')  # pylint: disable=protected-access
    malformed = (
        'HloModule x, fingerprint_before_lhs=""',
        'HloModule x, fingerprint_before_lhs="not-hex"',
        'HloModule x, fingerprint_before_lhs=abcd',
        (
            'HloModule x, fingerprint_before_lhs="abcd", '
            'fingerprint_before_lhs="ef01"'
        ),
        (
            'HloModule x, fingerprint_before_lhs="abcd", '
            'fingerprint_before_lhs="not-hex"'
        ),
    )
    for malformed_line in malformed:
      with self.subTest(malformed=malformed_line), self.assertRaisesRegex(
          analyzer.AnalysisError, 'one nonempty hexadecimal'
      ):
        analyzer._normalized_entry_abi(malformed_line)  # pylint: disable=protected-access

  def test_backend_parser_preserves_nested_tiles(self):
    config = {'fusion_backend_config': {'kind': '__triton',
              'block_level_fusion_config': {'output_tiles': [[1, 2], [3, 4]]}}}
    line = (
        '  %x = f32[] fusion(), backend_config='
        + json.dumps(config, separators=(',', ':'))
    )
    self.assertEqual(analyzer._backend_config_from_instruction(line), config)  # pylint: disable=protected-access
    diagnostics = analyzer._recompute_backend_diagnostics('HloModule x\n' + line)  # pylint: disable=protected-access
    self.assertEqual(diagnostics['triton_configurations'][0]['block_level_fusion_config']['output_tiles'], [[1, 2], [3, 4]])

  def test_runtime_route_checks_exact_maps_and_disabled_seams(self):
    value = _runtime_route(255, analyzer.INTENDED_DONOR_ROWS)
    analyzer._runtime_route(value, coalition_id=255, donor_rows=analyzer.INTENDED_DONOR_ROWS, label='x')  # pylint: disable=protected-access
    value['final_embedding']['donor_batch_indices'][0][0] = [1, 1]
    with self.assertRaisesRegex(analyzer.AnalysisError, 'final_embedding'):
      analyzer._runtime_route(value, coalition_id=255, donor_rows=analyzer.INTENDED_DONOR_ROWS, label='x')  # pylint: disable=protected-access

  def test_readout_recomputes_float32_margins(self):
    record = {'readout': _readout()}
    analyzer._readout(record, 'readout', 'x')  # pylint: disable=protected-access
    record['readout']['endpoint_margins'][0][0] = 2.0
    with self.assertRaisesRegex(analyzer.AnalysisError, 'margin'):
      analyzer._readout(record, 'readout', 'x')  # pylint: disable=protected-access

  def test_rowwise_and_checks_are_literal(self):
    analyzer._validate_rowwise(_rowwise(), 0, 'x')  # pylint: disable=protected-access
    analyzer._validate_checks(_checks(0), 0, 'x')  # pylint: disable=protected-access
    bad = _checks(0)
    bad['normalization_computed'] = True
    with self.assertRaisesRegex(analyzer.AnalysisError, 'normalization'):
      analyzer._validate_checks(bad, 0, 'x')  # pylint: disable=protected-access

  def test_cache_digest_distinguishes_directories_from_files(self):
    files = {'triton/a': {'sha256': 'a' * 64, 'size_bytes': 1}}
    first = analyzer._cache_binding_digest(['.', 'triton'], files)  # pylint: disable=protected-access
    second = analyzer._binding_map_digest(files)  # pylint: disable=protected-access
    self.assertNotEqual(first, second)

  def test_cache_binding_requires_exact_root_mode(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / 'cache'
      root.mkdir(mode=0o700)
      (root / 'triton').mkdir(mode=0o700)
      (root / 'xdg').mkdir(mode=0o700)
      analyzer._live_cache_binding(  # pylint: disable=protected-access
          root, 'external_preflight', 'cache fixture'
      )
      root.chmod(0o755)
      with self.assertRaisesRegex(analyzer.AnalysisError, 'root mode'):
        analyzer._live_cache_binding(  # pylint: disable=protected-access
            root, 'external_preflight', 'cache fixture'
        )

  def test_external_preflight_requires_exact_root_mode_before_record_read(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / 'preflight'
      root.mkdir(mode=0o755)
      root.chmod(0o755)
      with (
          mock.patch.object(analyzer, '_PREFLIGHT_DIR', root),
          mock.patch.object(
              analyzer, '_read_json',
              side_effect=AssertionError('record read before root mode gate'),
          ),
          self.assertRaisesRegex(analyzer.AnalysisError, 'root mode'),
      ):
        analyzer._validate_preflight_and_same_process({}, {})  # pylint: disable=protected-access

  def test_model_cache_evidence_is_exact_for_lower_and_compile_failures(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / 'model-cache'
      (root / 'triton').mkdir(parents=True, mode=0o700)
      (root / 'xdg').mkdir(mode=0o700)
      binding = analyzer._live_cache_binding(  # pylint: disable=protected-access
          root, 'model', 'fixture model cache'
      )
      for stage, compile_skipped in (('lower', None), ('compile', False)):
        with self.subTest(stage=stage), mock.patch.object(
            analyzer, '_MODEL_CACHE_DIR', root
        ):
          evidence = _model_cache_evidence(
              pre_import_files_present=False,
              compile_skipped=compile_skipped,
          )
          compiler = {
              'successful_compile_count': 0,
              'failure_stage': stage,
              'kernel_cache_provenance': {
                  'pre_import': binding,
                  'post_failure': binding,
                  'cache_hit_evidence': evidence,
                  'default_user_cache_paths_eligible': False,
                  'cache_outputs_are_diagnostic_only': True,
              },
          }
          terminal = {
              'pre_import': binding,
              'historical_stage': 'post_failure',
              'historical_binding': binding,
              'terminal': binding,
              'cache_hit_evidence': evidence,
              'historical_to_terminal_tree_exact': True,
              'historical_to_terminal_equality_is_a_gate': False,
              'historical_snapshot_not_reauthenticated_as_live_files': True,
              'default_user_cache_paths_eligible': False,
              'cache_outputs_are_diagnostic_only': True,
          }
          checked = analyzer._validate_model_cache_final(  # pylint: disable=protected-access
              terminal, compiler=compiler,
              status=f'controlled_stop_{stage}_failure',
              reason=f'{stage}_failure',
          )
          self.assertFalse(checked['cache_hit'])
          changed = copy.deepcopy(terminal)
          changed['cache_hit_evidence']['compile_skipped'] = False
          if stage == 'lower':
            changed_compiler = copy.deepcopy(compiler)
            changed_compiler['kernel_cache_provenance'][
                'cache_hit_evidence'
            ]['compile_skipped'] = False
            with self.assertRaisesRegex(
                analyzer.AnalysisError, 'cache-hit formula'
            ):
              analyzer._validate_model_cache_final(  # pylint: disable=protected-access
                  changed, compiler=changed_compiler,
                  status='controlled_stop_lower_failure',
                  reason='lower_failure',
              )

  def test_preimport_cache_hit_terminal_permits_exact_early_evidence(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / 'model-cache'
      (root / 'triton').mkdir(parents=True, mode=0o700)
      (root / 'xdg').mkdir(mode=0o700)
      pre_import = analyzer._live_cache_binding(  # pylint: disable=protected-access
          root, 'model', 'fixture model cache'
      )
      cache_file = root / 'xdg' / 'appeared-after-start'
      cache_file.write_bytes(b'cache input')
      cache_file.chmod(0o600)
      terminal_binding = analyzer._live_cache_binding(  # pylint: disable=protected-access
          root, 'model', 'fixture model cache'
      )
      evidence = _model_cache_evidence(
          pre_import_files_present=True, compile_skipped=None,
      )
      terminal = {
          'pre_import': pre_import,
          'historical_stage': None,
          'historical_binding': None,
          'terminal': terminal_binding,
          'cache_hit_evidence': evidence,
          'historical_to_terminal_tree_exact': None,
          'historical_to_terminal_equality_is_a_gate': False,
          'historical_snapshot_not_reauthenticated_as_live_files': True,
          'default_user_cache_paths_eligible': False,
          'cache_outputs_are_diagnostic_only': True,
      }
      with mock.patch.object(analyzer, '_MODEL_CACHE_DIR', root):
        checked = analyzer._validate_model_cache_final(  # pylint: disable=protected-access
            terminal, compiler=None, status='controlled_stop_cache_hit',
            reason='model_cache_pre_import_hit',
        )
      self.assertTrue(checked['cache_hit'])

  def test_diagnostic_compiler_artifact_accepts_all_four_exact_reasons(self):
    reasons = (
        'diagnostic_parser_failure', 'diagnostic_persistence_failure',
        'cache_signal_unavailable', 'fingerprint_formula_mismatch',
    )
    with tempfile.TemporaryDirectory() as directory:
      run = Path(directory) / 'run'
      compiler_dir = run / 'compiler/eight_row'
      compiler_dir.mkdir(parents=True, mode=0o700)
      payloads = {
          'stablehlo': (
              'compiler/eight_row/graph.stablehlo.mlir', b'stable\n'
          ),
          'hlo': (
              'compiler/eight_row/graph.pre_backend.hlo.txt', b'pre\n'
          ),
          'compiled_hlo': (
              'compiler/eight_row/graph.compiled.hlo.txt',
              b'HloModule fixture, fingerprint_before_lhs="abcd"\nENTRY x\n',
          ),
      }
      artifacts = {}
      for name, (relative, payload) in payloads.items():
        path = run / relative
        path.write_bytes(payload)
        path.chmod(0o400)
        artifacts[name] = {
            'path': relative, 'sha256': hashlib.sha256(payload).hexdigest(),
            'size_bytes': len(payload),
        }
      _, entry_sha = analyzer._normalized_entry_abi(  # pylint: disable=protected-access
          payloads['compiled_hlo'][1].decode()
      )
      signatures = {'fixture': {'leaves': []}}
      signatures_sha = analyzer._canonical_json_sha256(signatures)  # pylint: disable=protected-access
      source = _source_audit()
      same = _same_object()
      authorization = {'fixture': True}
      signature_binding = {
          'path': 'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json',
          'sha256': 'a' * 64, 'size_bytes': 1,
      }
      contract = {'fixture': True}
      for reason in reasons:
        with self.subTest(reason=reason):
          trigger_type = {
              'diagnostic_parser_failure': 'BackendDiagnosticParserFailure',
              'diagnostic_persistence_failure': 'DiagnosticPersistenceFailure',
              'cache_signal_unavailable': 'CacheSignalUnavailable',
              'fingerprint_formula_mismatch': 'FingerprintFormulaMismatch',
          }[reason]
          compiled_payload = {
              'diagnostic_parser_failure': (
                  b'HloModule fixture, fingerprint_before_lhs="abcd"\n'
                  b'  %x = f32[] custom-call(), backend_config={\n'
              ),
              'diagnostic_persistence_failure': (
                  b'HloModule fixture, fingerprint_before_lhs="abcd"\nENTRY x\n'
              ),
              'cache_signal_unavailable': (
                  b'HloModule fixture, fingerprint_before_lhs="abcd"\nENTRY x\n'
              ),
              'fingerprint_formula_mismatch': b'HloModule fixture\nENTRY x\n',
          }[reason]
          compiled_path = run / artifacts['compiled_hlo']['path']
          compiled_path.chmod(0o600)
          compiled_path.write_bytes(compiled_payload)
          compiled_path.chmod(0o400)
          artifacts['compiled_hlo'] = {
              'path': artifacts['compiled_hlo']['path'],
              'sha256': hashlib.sha256(compiled_payload).hexdigest(),
              'size_bytes': len(compiled_payload),
          }
          observed_entry = (
              entry_sha
              if reason in {
                  'diagnostic_parser_failure',
                  'diagnostic_persistence_failure',
              } else ''
          )
          primitive_entry = bool(observed_entry)
          gate = {
              'contract': contract,
              'observed': {
                  'stablehlo_sha256': artifacts['stablehlo']['sha256'],
                  'stablehlo_size_bytes': artifacts['stablehlo']['size_bytes'],
                  'pre_backend_hlo_sha256': artifacts['hlo']['sha256'],
                  'pre_backend_hlo_size_bytes': artifacts['hlo']['size_bytes'],
                  'program_signatures_sha256': signatures_sha,
                  'entry_abi_sha256': observed_entry,
              },
              'stablehlo_exact': True, 'pre_backend_hlo_exact': True,
              'program_signature_structure_exact': True,
              'program_signatures_canonical_exact': True,
              'entry_abi_exact': primitive_entry,
              'source_runtime_device_toolchain_checkpoint_reference_exact': True,
              'source_input_audit': source,
              'source_input_audit_content_binding': analyzer._content_binding(source),  # pylint: disable=protected-access
              'same_object_attestation': same,
              'same_object_attestation_content_binding': analyzer._content_binding(same),  # pylint: disable=protected-access
              'same_lowered_compiled_object': True,
              'source_program_exact': primitive_entry,
          }
          compiler = {
              'status': 'diagnostic_provenance_failure',
              'executable_name': 'eight_row', 'lower_attempt_count': 1,
              'compile_attempt_count': 1, 'successful_compile_count': 1,
              'artifacts': artifacts,
              'program_signature_attestation_binding': signature_binding,
              'external_freeze_authorization': authorization,
              'source_input_audit': source,
              'source_input_audit_content_binding': analyzer._content_binding(source),  # pylint: disable=protected-access
              'same_object_attestation': same,
              'same_object_attestation_content_binding': analyzer._content_binding(same),  # pylint: disable=protected-access
              'source_program_gate_without_backend_diagnostics': gate,
              'failure': {
                  'type': trigger_type,
                  'message': 'cache parser fingerprint publication adversary',
                  'traceback': 'fixture',
              },
              'attempt_budget_audit': {
                  'lower_budget': 1, 'compile_budget': 1,
                  'lower_invocations': 1, 'compile_invocations': 1,
                  'forbidden_request': None,
                  'forbidden_request_detected_before_invocation': False,
              },
              'diagnostic_provenance_complete': False,
              'compiled_backend_diagnostic_only': True,
              'no_dispatch': True, 'created_at_unix_s': 1.0,
          }
          compiler_path = (
              compiler_dir / 'COMPILER_DIAGNOSTIC_FAILURE.json'
          )
          if compiler_path.exists():
            compiler_path.chmod(0o600)
          compiler_path.write_text(
              json.dumps(compiler, sort_keys=True, separators=(',', ':')),
              encoding='utf-8',
          )
          compiler_path.chmod(0o400)
          compiler_binding = {
              'path': (
                  'compiler/eight_row/COMPILER_DIAGNOSTIC_FAILURE.json'
              ),
              'sha256': hashlib.sha256(
                  compiler_path.read_bytes()
              ).hexdigest(),
              'size_bytes': compiler_path.stat().st_size,
          }
          compiler_artifact_bindings = {
              signature_binding['path']: {
                  'sha256': signature_binding['sha256'],
                  'size_bytes': signature_binding['size_bytes'],
              },
              compiler_binding['path']: {
                  'sha256': compiler_binding['sha256'],
                  'size_bytes': compiler_binding['size_bytes'],
              },
              **{
                  row['path']: {
                      'sha256': row['sha256'],
                      'size_bytes': row['size_bytes'],
                  }
                  for row in artifacts.values()
              },
          }
          completion = {
              'status': 'controlled_stop_diagnostic_provenance_failure',
              'stop_reason': reason,
              'program_signature_attestation_binding': signature_binding,
              'same_object_attestation': same,
              'same_object_attestation_content_binding': analyzer._content_binding(same),  # pylint: disable=protected-access
              'source_program_gate': gate,
              'diagnostic_provenance_complete': False,
              'compiled_backend_diagnostic_only': True,
              'backend_diagnostics': None, 'diagnostic_comparisons': None,
              'compiler_binding': compiler_binding,
              'compiler_artifact_bindings': dict(sorted(
                  compiler_artifact_bindings.items()
              )),
          }
          with (
              mock.patch.object(
                  analyzer, 'PROGRAM_SIGNATURES_SHA256', signatures_sha
              ),
              mock.patch.object(analyzer, 'ENTRY_ABI_SHA256', entry_sha),
              mock.patch.object(analyzer, 'SOURCE_STABLEHLO', {
                  'sha256': artifacts['stablehlo']['sha256'],
                  'size_bytes': artifacts['stablehlo']['size_bytes'],
              }),
              mock.patch.object(analyzer, 'SOURCE_PRE_BACKEND_HLO', {
                  'sha256': artifacts['hlo']['sha256'],
                  'size_bytes': artifacts['hlo']['size_bytes'],
              }),
              mock.patch.object(
                  analyzer, '_validate_signature_attestation',
                  return_value={'canonical_sha256': signatures_sha},
              ),
          ):
            checked = analyzer._validate_diagnostic_failure_record(  # pylint: disable=protected-access
                compiler, run_dir=run,
                freeze={
                    'program_signatures': signatures,
                    'source_program_contract': contract,
                },
                start={'external_freeze_authorization': authorization},
                completion=completion,
            )
          self.assertEqual(checked['state'], 'diagnostic_provenance_failed')
          self.assertIs(checked['source_program_exact'], primitive_entry)

  def test_diagnostic_entry_parser_preserves_both_exact_failure_subcases(self):
    failures = (
        (
            'EntryAbiParserFailure', 'diagnostic_parser_failure',
            'not an HLO module\n',
        ),
        (
            'FingerprintFormulaMismatch', 'fingerprint_formula_mismatch',
            'HloModule no_fingerprint\n',
        ),
    )
    for trigger_type, reason, compiled_hlo in failures:
      with self.subTest(reason=reason):
        self.assertFalse(analyzer._diagnostic_entry_abi_exact(  # pylint: disable=protected-access
            compiled_hlo, reason=reason,
            failure={'type': trigger_type, 'message': 'adversarial cache text',
                     'traceback': 'fixture'},
            observed_sha256='',
        ))
        with self.assertRaisesRegex(
            analyzer.AnalysisError, 'type/reason'
        ):
          analyzer._diagnostic_entry_abi_exact(  # pylint: disable=protected-access
              compiled_hlo, reason=reason,
              failure={'type': 'CacheSignalUnavailable', 'message': 'irrelevant',
                       'traceback': 'fixture'},
              observed_sha256='',
          )

  def test_diagnostic_compiler_cache_is_historical_without_cache_provenance(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / 'model-cache'
      (root / 'triton').mkdir(parents=True, mode=0o700)
      (root / 'xdg').mkdir(mode=0o700)
      binding = analyzer._live_cache_binding(  # pylint: disable=protected-access
          root, 'model', 'fixture model cache'
      )
      evidence = _model_cache_evidence(
          pre_import_files_present=False, compile_skipped=False,
      )
      terminal = {
          'pre_import': binding, 'historical_stage': 'post_compile',
          'historical_binding': binding, 'terminal': binding,
          'cache_hit_evidence': evidence,
          'historical_to_terminal_tree_exact': True,
          'historical_to_terminal_equality_is_a_gate': False,
          'historical_snapshot_not_reauthenticated_as_live_files': True,
          'default_user_cache_paths_eligible': False,
          'cache_outputs_are_diagnostic_only': True,
      }
      compiler = {
          'status': 'diagnostic_provenance_failure',
          'successful_compile_count': 1,
      }
      with mock.patch.object(analyzer, '_MODEL_CACHE_DIR', root):
        checked = analyzer._validate_model_cache_final(  # pylint: disable=protected-access
            terminal, compiler=compiler,
            status='controlled_stop_diagnostic_provenance_failure',
            reason='diagnostic_parser_failure',
        )
      self.assertFalse(checked['cache_hit'])

  def test_partial_output_rejects_noncanonical_base64(self):
    payload = struct.pack('<f', 1.0)
    import base64
    leaf = {
        'path': [], 'dtype_name': 'float32', 'byte_order': 'little',
        'shape': [], 'encoding': 'base64_c_order_raw_bytes',
        'data_base64': base64.b64encode(payload).decode(),
        'sha256': hashlib.sha256(payload).hexdigest(), 'size_bytes': 4,
    }
    value = {'status': 'returned', 'treedef': {
        'kind': 'leaf', 'metadata': None, 'children': []
    }, 'leaf_count': 1, 'leaves': [leaf]}
    analyzer._validate_partial_output(value, 'x')  # pylint: disable=protected-access
    value['leaves'][0]['data_base64'] = '!!!!'
    with self.assertRaises(analyzer.AnalysisError):
      analyzer._validate_partial_output(value, 'x')  # pylint: disable=protected-access


class StructuralResultAndAppendOnlyTest(unittest.TestCase):

  def _binding(self) -> dict[str, object]:
    return {'path': '/tmp/fixed', 'sha256': 'a' * 64, 'size_bytes': 1}

  def test_structural_result_draft_has_literal_27_keys_and_no_science(self):
    result = analyzer._result_v3345(  # pylint: disable=protected-access
        status='complete_controlled_stop_structural_archive',
        decision='controlled_stop_source_program_mismatch',
        terminal_kind='run_complete', compiler_state='compiled_source_mismatch',
        k=0, d=0, started=0, completed=0, id0=False, id255=False,
        prior333={'exact': True}, prior331={'exact': True},
        start_binding=self._binding(), run_binding={'fixed': True},
        preflight_binding={'fixed': True},
        model_publication_audit=_publication_audit(),
    )
    self.assertEqual(len(result), 27)
    self.assertIsNone(result['publication_audit'])
    self.assertFalse(result['scientific_summary_computed'])
    self.assertFalse(result['donor_normalization_computed'])
    self.assertFalse(result['shapley_or_nomination_computed'])
    self.assertFalse(result['combined_analysis_permitted'])
    forbidden = {'score', 'shapley', 'nominee', 'recovery', 'rank'}
    self.assertFalse(forbidden & set(result))

  def test_terminal_full80_accounting_is_exact(self):
    phase = _phase(**{name: True for name in analyzer._PHASE_STATE_KEYS})  # pylint: disable=protected-access
    value, start = _terminal(
        status='complete_structural_sidecar', reason=None, k=80, d=0,
        attempted=320, completed=320, phase=phase, failure_phase=None,
    )
    checked, _ = analyzer._validate_terminal_common(  # pylint: disable=protected-access
        value, freeze_sha='f' * 64, start=start
    )
    self.assertEqual(checked['valid_record_count'], 80)
    changed = copy.deepcopy(value)
    changed['model_apply_success_count'] = 319
    with self.assertRaisesRegex(analyzer.AnalysisError, 'model-apply'):
      analyzer._validate_terminal_common(changed, freeze_sha='f' * 64, start=start)  # pylint: disable=protected-access

  def test_terminal_partial_dispatch_counts_started_failure(self):
    phase = _phase(**{
        name: True for name in analyzer._PHASE_STATE_KEYS  # pylint: disable=protected-access
    })
    value, start = _terminal(
        status='controlled_stop_partial_dispatch',
        reason='model_dispatch_failure', k=3, d=2,
        attempted=15, completed=14, phase=phase,
        failure_phase='model_dispatch',
    )
    value['all_80_recipient_anchors_complete'] = False
    value['id0_all20'] = False
    value['id255_all20'] = False
    value['terminal_detail']['failed_call_role'] = 'unrelated'
    analyzer._validate_terminal_common(value, freeze_sha='f' * 64, start=start)  # pylint: disable=protected-access
    value['model_apply_attempt_count'] = 14
    with self.assertRaisesRegex(analyzer.AnalysisError, 'model-apply'):
      analyzer._validate_terminal_common(value, freeze_sha='f' * 64, start=start)  # pylint: disable=protected-access

  def test_terminal_four_call_invalid_prefix_is_exact(self):
    phase = _phase(**{
        name: True for name in analyzer._PHASE_STATE_KEYS  # pylint: disable=protected-access
    })
    value, start = _terminal(
        status='controlled_stop_four_call_invalid',
        reason='record_validation_or_serialization_failure',
        k=7, d=4, attempted=32, completed=32, phase=phase,
        failure_phase='record_validation',
    )
    value['all_80_recipient_anchors_complete'] = False
    value['id0_all20'] = False
    value['id255_all20'] = False
    value['terminal_detail']['failed_call_role'] = None
    value['terminal_detail']['failed_execution_index'] = 7
    analyzer._validate_terminal_common(  # pylint: disable=protected-access
        value, freeze_sha='f' * 64, start=start
    )
    changed = copy.deepcopy(value)
    changed['terminal_detail']['d_completed'] = 3
    with self.assertRaisesRegex(analyzer.AnalysisError, 'Four-call'):
      analyzer._validate_terminal_common(  # pylint: disable=protected-access
          changed, freeze_sha='f' * 64, start=start
      )

  def test_all_80_runner_shaped_records_pass_structural_validator(self):
    cases = _cases()
    source = _source_audit(True)
    same_object = _same_object()
    authorization = {'fixture': True}
    freeze_sha = 'f' * 64
    executable = 'e' * 64
    sequences = {
        order: {'reference': f'{order:064x}', 'alternate': f'{order + 100:064x}'}
        for order in range(20)
    }
    started = {
        f'dispatch_journal/started/{index:03d}.json': {
            'sha256': f'{index + 1:064x}', 'size_bytes': index + 1,
        } for index in range(320)
    }
    completed = {
        f'dispatch_journal/completed/{index:03d}.json': {
            'sha256': f'{index + 1000:064x}', 'size_bytes': index + 2,
        } for index in range(320)
    }
    with tempfile.TemporaryDirectory() as directory:
      original = Path(directory) / 'original'
      original.mkdir()
      original_manifest = {}
      for case in cases.values():
        for family, anchor in (
            [('identity', None)]
            + [('coalition', value) for value in analyzer.ANCHOR_IDS]
        ):
          relative = analyzer._original_relative(  # pylint: disable=protected-access
              case, family, anchor
          )
          path = original / relative
          path.parent.mkdir(parents=True, exist_ok=True)
          path.write_text('{}\n', encoding='utf-8')
          original_manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
      audits = []
      with mock.patch.object(analyzer, '_ORIGINAL_CUBE_DIR', original):
        for execution_index, (order, anchor) in enumerate(
            analyzer._execution_order()  # pylint: disable=protected-access
        ):
          donor_order = analyzer._donor_order(order)  # pylint: disable=protected-access
          record = _record_fixture(
              case=cases[order], donor=cases[donor_order], anchor=anchor,
              execution_index=execution_index, freeze_sha=freeze_sha,
              executable=executable, original_manifest=original_manifest,
              sequences=sequences, authorization=authorization,
              source=source, same_object=same_object,
              started=started, completed=completed,
          )
          audits.append(analyzer._validate_record(  # pylint: disable=protected-access
              record, case=cases[order], donor_case=cases[donor_order],
              anchor=anchor, execution_index=execution_index,
              freeze_sha256=freeze_sha,
              executable_fingerprint=executable,
              original_manifest=original_manifest,
              sequence_bindings=sequences, authorization=authorization,
              source_audit=source, same_object=same_object,
              started_bindings=started, completed_bindings=completed,
              allow_invalid=False,
          ))
          if execution_index == 1:
            tampered = copy.deepcopy(record)
            tampered['runtime_interventions']['unrelated'][
                'transformer_output'
            ]['donor_batch_indices'][0][2] = 0
            with self.assertRaisesRegex(analyzer.AnalysisError, 'donor rows'):
              analyzer._validate_record(  # pylint: disable=protected-access
                  tampered, case=cases[order], donor_case=cases[donor_order],
                  anchor=anchor, execution_index=execution_index,
                  freeze_sha256=freeze_sha,
                  executable_fingerprint=executable,
                  original_manifest=original_manifest,
                  sequence_bindings=sequences, authorization=authorization,
                  source_audit=source, same_object=same_object,
                  started_bindings=started, completed_bindings=completed,
                  allow_invalid=False,
              )
      self.assertEqual(len(audits), 80)
      self.assertEqual(sum(row['anchor'] == 0 for row in audits), 20)
      self.assertEqual(sum(row['anchor'] == 255 for row in audits), 20)

  def test_output_state_accepts_only_append_only_publication_prefix(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'analysis'
      with mock.patch.object(analyzer, '_ANALYSIS_DIR', output):
        audit = _publication_audit()
        self.assertEqual(analyzer._analysis_output_state(audit)['state'], 'absent')  # pylint: disable=protected-access
        output.mkdir(mode=0o700)
        result = output / 'RESULT.md'
        result.write_text('structural only', encoding='utf-8')
        result.chmod(0o400)
        status = result.lstat()
        audit['successful_final_count_before_terminal'] = 1
        audit['successful_final_bindings_before_terminal'] = {
            'RESULT.md': {
                'sha256': analyzer._sha256(result),  # pylint: disable=protected-access
                'size_bytes': status.st_size, 'mode': '0400',
                'st_dev': status.st_dev, 'st_ino': status.st_ino,
                'st_nlink': status.st_nlink,
            },
        }
        state = analyzer._analysis_output_state(audit)  # pylint: disable=protected-access
        self.assertEqual(state['state'], 'published_prefix')
        self.assertEqual(state['published_prefix'], ['RESULT.md'])
        extra = output / 'extra'
        extra.write_text('x', encoding='utf-8')
        with self.assertRaises(analyzer.AnalysisError):
          analyzer._analysis_output_state(audit)  # pylint: disable=protected-access

  def test_direct_production_analyze_requires_active_attempt_token(self):
    with self.assertRaisesRegex(analyzer.AnalysisError, 'post-START'):
      analyzer.analyze(analyzer._RUN_DIR)  # pylint: disable=protected-access

  def test_freeze_revalidation_accepts_only_exact_active_start_singleton(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      attempt = root / 'attempt'
      output = root / 'output'
      with (
          mock.patch.object(analyzer, '_ANALYSIS_ATTEMPT_DIR', attempt),
          mock.patch.object(analyzer, '_ANALYSIS_DIR', output),
      ):
        analyzer._validate_analysis_destination_state(None)  # pylint: disable=protected-access
        attempt.mkdir(mode=0o700)
        started = attempt / 'ANALYSIS_ATTEMPT_STARTED.json'
        started.write_text('{}\n', encoding='utf-8')
        started.chmod(0o400)
        digest = hashlib.sha256(started.read_bytes()).hexdigest()
        analyzer._validate_analysis_destination_state(digest)  # pylint: disable=protected-access
        with self.assertRaisesRegex(analyzer.AnalysisError, 'not fresh'):
          analyzer._validate_analysis_destination_state(None)  # pylint: disable=protected-access
        extra = attempt / 'extra.json'
        extra.write_text('{}\n', encoding='utf-8')
        extra.chmod(0o400)
        with self.assertRaises(analyzer.AnalysisError):
          analyzer._validate_analysis_destination_state(digest)  # pylint: disable=protected-access

  def test_shell_accepts_only_literal_acknowledgement(self):
    shell = (_HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_4_5.sh').read_text(encoding='utf-8')
    self.assertIn('--acknowledge-structural-only-v3-3-4-5', shell)
    self.assertNotIn('--run-dir', shell)
    self.assertNotIn('--force', shell)

  def test_atomic_publish_refuses_overwrite(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'fixed.json'
      path.write_bytes(b'first')
      with self.assertRaises(FileExistsError):
        analyzer._publish_new_bytes(  # pylint: disable=protected-access
            path, b'second', root_role='analysis_output',
            root=Path(directory), artifact_role='fixture',
        )

  def test_confirmation_path_guard_rejects_only_paths(self):
    with self.assertRaisesRegex(analyzer.AnalysisError, 'confirmation'):
      analyzer._guard_path(Path('/tmp/confirmation/results'))  # pylint: disable=protected-access
    cases = analyzer._load_cases()  # pylint: disable=protected-access
    self.assertEqual(len(cases), 20)
    for order, case in cases.items():
      with self.subTest(order=order, variant=case['variant_id']):
        analyzer._guard_path(  # pylint: disable=protected-access
            Path('/tmp/v3343-development')
            / analyzer._artifact_relative(case, 255)  # pylint: disable=protected-access
        )
        analyzer._guard_path(  # pylint: disable=protected-access
            Path('/tmp/v3343-development')
            / analyzer._failed_current_relative(case, 255)  # pylint: disable=protected-access
        )
    # Larger development slugs are not confused with an exact forbidden gene
    # component, while a literal forbidden component remains rejected.
    analyzer._guard_path(  # pylint: disable=protected-access
        Path('/tmp/v3343-development/raw/000_ELN_e19_G60T/000.json')
    )
    with self.assertRaisesRegex(analyzer.AnalysisError, 'confirmation'):
      analyzer._guard_path(Path('/tmp/v3343-development/raw/ELN/000.json'))  # pylint: disable=protected-access
    # The mandatory disclosure text is not interpreted as a path.
    self.assertIn('model outputs', analyzer.CONFIRMATION_DISCLOSURE)


class ConsumedPrefixAndLifecycleTest(unittest.TestCase):

  def test_v3344_consumed_prefix_live_archive_and_canonical_binding(self):
    prefix, binding = _consumed_v3344_prefix()
    self.assertEqual(set(prefix), analyzer._CONSUMED_V3344_PREFIX_KEYS)  # pylint: disable=protected-access
    self.assertEqual(binding, {
        'sha256': analyzer.CONSUMED_V3344_PREFIX_SHA256,
        'size_bytes': 8653,
    })
    self.assertEqual(prefix['traceback_provenance'], {
        'storage': 'coordinator_captured_not_persisted',
        'sha256': analyzer.CONSUMED_V3344_TRACEBACK_SHA256,
        'size_bytes': 1168, 'session_id': None,
        'captured_at_unix_s': None,
        'wall_clock_timestamp_available': False,
    })
    checked, checked_binding = analyzer._validate_consumed_v3344_prefix(  # pylint: disable=protected-access
        {'prior_v3_3_4_4_consumed_preflight_prefix': prefix},
        label='test v3344 consumed prefix',
    )
    self.assertEqual(checked, prefix)
    self.assertEqual(checked_binding, binding)
    bootstrap_path = (
        _HERE / 'validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_5.py'
    )
    spec = importlib.util.spec_from_file_location(
        '_v3345_bootstrap_consumed_v3344_prefix', bootstrap_path
    )
    assert spec is not None and spec.loader is not None
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    self.assertEqual(
        bootstrap.validate_prior_v3_3_4_4_consumed_preflight_prefix(), prefix
    )
    self.assertEqual(bootstrap.canonical_content_binding(prefix), binding)

  def test_every_v3344_consumed_prefix_key_and_leaf_is_fail_closed(self):
    prefix, _ = _consumed_v3344_prefix()
    containers = []
    leaves = []
    def walk(value, path=()):
      if isinstance(value, dict):
        containers.append(path)
        for key, child in value.items():
          walk(child, (*path, key))
      elif isinstance(value, list):
        for index, child in enumerate(value):
          walk(child, (*path, index))
      else:
        leaves.append(path)
    def parent_at(value, path):
      node = value
      for part in path[:-1]:
        node = node[part]
      return node
    walk(prefix)
    self.assertGreater(len(leaves), 100)
    for path in leaves:
      with self.subTest(kind='leaf', path=path):
        changed = copy.deepcopy(prefix)
        parent_at(changed, path)[path[-1]] = {'tampered': True}
        with self.assertRaisesRegex(analyzer.AnalysisError, 'prefix changed'):
          analyzer._validate_consumed_v3344_prefix(  # pylint: disable=protected-access
              {'prior_v3_3_4_4_consumed_preflight_prefix': changed},
              label='tampered v3344 leaf',
          )
    for path in containers:
      if not path:
        continue
      with self.subTest(kind='missing-key', path=path):
        changed = copy.deepcopy(prefix)
        parent_at(changed, path).pop(path[-1])
        with self.assertRaises(analyzer.AnalysisError):
          analyzer._validate_consumed_v3344_prefix(  # pylint: disable=protected-access
              {'prior_v3_3_4_4_consumed_preflight_prefix': changed},
              label='tampered v3344 key',
          )
    changed = copy.deepcopy(prefix)
    changed['extra'] = True
    with self.assertRaisesRegex(analyzer.AnalysisError, 'key set'):
      analyzer._validate_consumed_v3344_prefix(  # pylint: disable=protected-access
          {'prior_v3_3_4_4_consumed_preflight_prefix': changed},
          label='extra v3344 key',
      )

  def test_consumed_archive_walkers_reject_missing_extra_symlink_and_special(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / 'archive'
      root.mkdir(mode=0o700)
      expected = root / 'expected'
      expected.write_bytes(b'expected')
      expected.chmod(0o400)
      analyzer._strict_tree(root, {'expected'}, 'archive fixture')  # pylint: disable=protected-access
      expected.unlink()
      with self.assertRaisesRegex(analyzer.AnalysisError, 'membership'):
        analyzer._strict_tree(root, {'expected'}, 'archive fixture')  # pylint: disable=protected-access
      expected.write_bytes(b'expected')
      expected.chmod(0o400)
      extra = root / 'extra'
      extra.write_bytes(b'extra')
      extra.chmod(0o400)
      with self.assertRaisesRegex(analyzer.AnalysisError, 'membership'):
        analyzer._strict_tree(root, {'expected'}, 'archive fixture')  # pylint: disable=protected-access
      extra.unlink()
      expected.unlink()
      expected.symlink_to(root / 'missing')
      with self.assertRaisesRegex(analyzer.AnalysisError, 'symlink'):
        analyzer._strict_tree(root, {'expected'}, 'archive fixture')  # pylint: disable=protected-access
      expected.unlink()
      expected.mkdir(mode=0o700)
      with self.assertRaises(analyzer.AnalysisError):
        analyzer._strict_tree(root, {'expected'}, 'archive fixture')  # pylint: disable=protected-access

  def test_consumed_prefix_live_tree_sources_and_canonical_bindings(self):
    prefix, binding = _consumed_prefix()
    self.assertEqual(set(prefix), analyzer._CONSUMED_PREFIX_KEYS)  # pylint: disable=protected-access
    self.assertEqual(binding, {
        'sha256': 'c42ce8bd47918daf90affba701eb1dc193c1ba2cbb38e3b0f836e8b81f306f88',
        'size_bytes': 6418,
    })
    checked, checked_binding = analyzer._validate_consumed_v3343_prefix(  # pylint: disable=protected-access
        {'prior_v3_3_4_3_consumed_preflight_prefix': prefix},
        label='test consumed prefix',
    )
    self.assertEqual(checked, prefix)
    self.assertEqual(checked_binding, binding)
    self.assertEqual(
        analyzer._content_binding(prefix['cache_tree_binding']),  # pylint: disable=protected-access
        {'sha256': analyzer.CONSUMED_V3343_CACHE_BINDING_SHA256,
         'size_bytes': 745},
    )
    bootstrap_path = (
        _HERE / 'validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_5.py'
    )
    spec = importlib.util.spec_from_file_location(
        '_v3345_bootstrap_consumed_prefix', bootstrap_path
    )
    assert spec is not None and spec.loader is not None
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    bootstrap_prefix = (
        bootstrap.validate_prior_v3_3_4_3_consumed_preflight_prefix()
    )
    self.assertEqual(bootstrap_prefix, prefix)
    self.assertEqual(bootstrap.canonical_content_binding(prefix), binding)

  def test_every_consumed_prefix_top_level_field_is_fail_closed(self):
    prefix, _ = _consumed_prefix()
    for key in sorted(prefix):
      with self.subTest(key=key):
        changed = copy.deepcopy(prefix)
        changed[key] = None
        with self.assertRaisesRegex(analyzer.AnalysisError, 'prefix changed'):
          analyzer._validate_consumed_v3343_prefix(  # pylint: disable=protected-access
              {'prior_v3_3_4_3_consumed_preflight_prefix': changed},
              label='tampered consumed prefix',
          )

  def test_embedded_prefix_requires_exact_object_and_canonical_binding(self):
    prefix, binding = _consumed_prefix()
    v3344, v3344_binding = _consumed_v3344_prefix()
    frozen = {
        'prior_v3_3_4_3_consumed_preflight_prefix': prefix,
        'prior_v3_3_4_4_consumed_preflight_prefix': v3344,
    }
    embedded = {
        'prior_v3_3_4_3_consumed_preflight_prefix': copy.deepcopy(prefix),
        'prior_v3_3_4_3_consumed_preflight_prefix_content_binding': dict(binding),
        'prior_v3_3_4_4_consumed_preflight_prefix': copy.deepcopy(v3344),
        'prior_v3_3_4_4_consumed_preflight_prefix_content_binding': dict(
            v3344_binding
        ),
    }
    analyzer._validate_embedded_consumed_prefix(  # pylint: disable=protected-access
        embedded, frozen, label='embedded fixture'
    )
    for key in tuple(embedded):
      with self.subTest(key=key):
        changed = copy.deepcopy(embedded)
        changed[key] = None
        with self.assertRaisesRegex(analyzer.AnalysisError, 'binding changed'):
          analyzer._validate_embedded_consumed_prefix(  # pylint: disable=protected-access
              changed, frozen, label='embedded fixture'
          )

  def test_consumed_prefix_gate_precedes_run_start_and_any_raw_read(self):
    events = []
    prefix, _ = _consumed_prefix()
    frozen = {'prior_v3_3_4_3_consumed_preflight_prefix': prefix}
    def provenance_gate(*_args, **_kwargs):
      analyzer._validate_consumed_v3343_prefix(  # pylint: disable=protected-access
          frozen, label='ordered prefix gate'
      )
      events.append('prefix_and_sources')
      return frozen, 'a' * 64, {}, {}, {}, {}
    def start_read(*_args, **_kwargs):
      events.append('run_start')
      raise analyzer.AnalysisError('intentional stop before run evidence')
    with tempfile.TemporaryDirectory() as directory:
      run = Path(directory) / 'run'
      run.mkdir(mode=0o700)
      with (
          mock.patch.object(
              analyzer, '_validate_freeze_v3345', side_effect=provenance_gate
          ),
          mock.patch.object(
              analyzer, '_validate_start_v3345', side_effect=start_read
          ),
          self.assertRaisesRegex(analyzer.AnalysisError, 'intentional stop'),
      ):
        analyzer.analyze(
            run, bundle_root=Path(directory),
            _raw_access_marker=lambda: events.append('raw'),
        )
    self.assertEqual(events, ['prefix_and_sources', 'run_start'])

  def test_both_consumed_prefixes_precede_toctou_run_rehash(self):
    events = []
    run_binding = {
        'file_bindings': {
            'RUN_COMPLETE.json': {
                'sha256': 'a' * 64, 'size_bytes': 1,
            },
        },
        'terminal_kind': 'run_complete',
    }
    def old_prefix(*_args, **_kwargs):
      events.append('v3343_prefix')
      return {}, {}
    def new_prefix(*_args, **_kwargs):
      events.append('v3344_prefix')
      raise analyzer.AnalysisError('intentional prefix stop')
    def forbidden_run_read(*_args, **_kwargs):
      events.append('run_read')
      raise AssertionError('run bytes were reached before both prefix gates')
    with (
        tempfile.TemporaryDirectory() as directory,
        mock.patch.object(analyzer, '_assert_cpu_only'),
        mock.patch.object(analyzer, '_assert_predecessor_v334_paths_absent'),
        mock.patch.object(analyzer, '_validate_active_analysis_attempt'),
        mock.patch.object(analyzer, '_read_json', return_value={}),
        mock.patch.object(
            analyzer, '_validate_consumed_v3343_prefix', side_effect=old_prefix
        ),
        mock.patch.object(
            analyzer, '_validate_consumed_v3344_prefix', side_effect=new_prefix
        ),
        mock.patch.object(analyzer, '_strict_regular', side_effect=forbidden_run_read),
        self.assertRaisesRegex(analyzer.AnalysisError, 'intentional prefix stop'),
    ):
      analyzer._analysis_toctou_check(  # pylint: disable=protected-access
          run_dir=Path(directory), started_sha256='b' * 64,
          result={'run_binding': run_binding}, label='before RESULT',
      )
    self.assertEqual(events, ['v3343_prefix', 'v3344_prefix'])

  def test_analysis_start_result_and_failure_have_revised_exact_schemas(self):
    prefix, binding = _consumed_prefix()
    v3344, v3344_binding = _consumed_v3344_prefix()
    precheck = {
        'git_head': 'a' * 40, 'freeze_sha256': 'b' * 64,
        'external_freeze_authorization': {'fixture': True},
        'analyzer_binding': {'path': '/analyzer', 'sha256': 'c' * 64,
                             'size_bytes': 1},
        'test_binding': {'path': '/test', 'sha256': 'd' * 64,
                         'size_bytes': 1},
        'run_terminal_binding': {'path': '/terminal', 'sha256': 'e' * 64,
                                 'size_bytes': 1},
        'prior_v3_3_4_3_consumed_preflight_prefix': prefix,
        'prior_v3_3_4_3_consumed_preflight_prefix_content_binding': binding,
        'prior_v3_3_4_4_consumed_preflight_prefix': v3344,
        'prior_v3_3_4_4_consumed_preflight_prefix_content_binding': (
            v3344_binding
        ),
    }
    started = analyzer._analysis_started_record(precheck)  # pylint: disable=protected-access
    self.assertEqual(set(started), analyzer._ANALYSIS_STARTED_KEYS)  # pylint: disable=protected-access
    self.assertEqual(len(started), 18)
    self.assertEqual(
        started['acknowledgement'],
        '--acknowledge-structural-only-v3-3-4-5',
    )

    result = analyzer._result_v3345(  # pylint: disable=protected-access
        status='complete_controlled_stop_structural_archive',
        decision='controlled_stop_source_program_mismatch',
        terminal_kind='run_complete', compiler_state='compiled_source_mismatch',
        k=0, d=0, started=0, completed=0, id0=False, id255=False,
        prior333={'exact': True}, prior331={'exact': True},
        start_binding={'path': '/start', 'sha256': 'f' * 64, 'size_bytes': 1},
        run_binding={'fixed': True}, preflight_binding={'fixed': True},
        model_publication_audit=_publication_audit(),
    )
    self.assertEqual(set(result), analyzer._ANALYSIS_KEYS)  # pylint: disable=protected-access
    self.assertEqual(len(result), 27)
    self.assertEqual(result['prior_v3_3_4_3_consumed_preflight_prefix'], prefix)
    self.assertEqual(
        result['source_and_prior_audit'],
        {
            'current_132_source_rows_exact': True,
            'historical_96_source_rows_exact': True,
            'git_head_exact': True, 'tracked_clean': True,
            'external_freeze_authorization_exact': True,
            'prior_v3_3_3_exact': True, 'prior_v3_3_3_1_exact': True,
            'old_analyzer_paths_absent': True, 'pre_start_exact': True,
            'post_start_exact': True, 'final_exact': True,
            'prior_v3_3_4_3_consumed_preflight_prefix_exact': True,
            'prior_v3_3_4_4_consumed_preflight_prefix_exact': True,
        },
    )

    with tempfile.TemporaryDirectory() as directory:
      attempt = Path(directory) / 'attempt'
      output = Path(directory) / 'output'
      attempt.mkdir(mode=0o700)
      started_path = attempt / 'ANALYSIS_ATTEMPT_STARTED.json'
      _write_json_0400(started_path, started)
      started_sha = hashlib.sha256(started_path.read_bytes()).hexdigest()
      with (
          mock.patch.object(analyzer, '_ANALYSIS_ATTEMPT_DIR', attempt),
          mock.patch.object(analyzer, '_ANALYSIS_DIR', output),
          mock.patch.object(
              analyzer, '_current_analysis_publication_audits',
              return_value=(_publication_audit(), _publication_audit()),
          ),
      ):
        failure = analyzer._analysis_failure_record(  # pylint: disable=protected-access
            RuntimeError('fixture failure'), started_sha, raw_reached=False,
        )
      self.assertEqual(len(failure), 19)
      self.assertEqual(
          set(failure), {
              'status', 'attempt_id', 'analysis_attempt_start_binding',
              'type', 'message', 'traceback', 'raw_values_read',
              'scientific_analysis_performed', 'output_dir_state',
              'publication_failure', 'temporary_orphan_bindings',
              'durability_uncertain_final_bindings',
              'preexisting_entry_states', 'no_new_entry_failure',
              'failed_at_unix_s',
              'prior_v3_3_4_3_consumed_preflight_prefix',
              'prior_v3_3_4_3_consumed_preflight_prefix_content_binding',
              'prior_v3_3_4_4_consumed_preflight_prefix',
              'prior_v3_3_4_4_consumed_preflight_prefix_content_binding',
          },
      )
      self.assertFalse(failure['raw_values_read'])
      self.assertFalse(failure['scientific_analysis_performed'])


class PublicationAmendmentTest(unittest.TestCase):

  def test_v3345_atomic_probe_live_bytes_and_names_are_exact(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / 'cache'
      root.mkdir(mode=0o700)
      external_pid = 12345
      final = root / 'atomic_publication_probe_v3_3_4_5.txt'
      collision = (
          root / f'.v3345.tmp.{external_pid}.000001.{"a" * 32}'
      )
      final.write_bytes(
          b'opensplice-v3.3.4.5-renameat2-noreplace-probe-v1\n'
      )
      collision.write_bytes(b'opensplice-v3.3.4.5-collision-probe-v1\n')
      final.chmod(0o400)
      collision.chmod(0o400)
      value = {
          'schema_version': analyzer.PUBLICATION_SCHEMA_VERSION,
          'method': analyzer.PUBLICATION_METHOD, 'supported': True,
          'successful_final_binding': {
              'path': final.name, **_live_binding(final),
          },
          'collision_errno': 17, 'collision_no_replace_exact': True,
          'collision_temp_binding': {
              'path': collision.name, **_live_binding(collision),
          },
          'destination_unchanged': True, 'temp_orphan_preserved': True,
          'parent_fsync_exact': True,
      }
      with mock.patch.object(analyzer, '_PREFLIGHT_CACHE_DIR', root):
        self.assertEqual(
            analyzer._validate_atomic_publication_probe(  # pylint: disable=protected-access
                value, external_pid=external_pid
            ),
            value,
        )
        for field in ('successful_final_binding', 'collision_temp_binding'):
          with self.subTest(field=field):
            changed = copy.deepcopy(value)
            changed[field]['sha256'] = '0' * 64
            with self.assertRaises(analyzer.AnalysisError):
              analyzer._validate_atomic_publication_probe(  # pylint: disable=protected-access
                  changed, external_pid=external_pid
              )

  def _publication_terminal_fixture(
      self, root: Path, failure_class: str,
  ) -> tuple[dict[str, object], dict[str, object]]:
    root.mkdir(mode=0o700)
    start_path = root / 'ATTEMPT_STARTED.json'
    _write_json_0400(start_path, {'fixture': True})
    runner_pid = 42
    final_relative = 'compiler/eight_row/final.json'
    temp_relative = (
        'compiler/eight_row/'
        f'.v3345.tmp.{runner_pid}.000000.{"a" * 32}'
    )
    final_path = root / final_relative
    temp_path = root / temp_relative
    final_path.parent.mkdir(parents=True, mode=0o700)
    (root / 'compiler').chmod(0o700)
    final_path.parent.chmod(0o700)
    no_new = failure_class in {'no_new', 'preexisting_regular', 'preexisting_directory'}
    rename_succeeded = failure_class == 'uncertain'
    if failure_class == 'preexisting_regular':
      final_path.write_text('preexisting\n', encoding='utf-8')
      final_path.chmod(0o400)
    elif failure_class == 'preexisting_directory':
      final_path.mkdir(mode=0o755)
      child = final_path / 'ignored'
      child.write_text('outside opaque state\n', encoding='utf-8')
    elif failure_class == 'temporary':
      temp_path.write_text('temporary\n', encoding='utf-8')
      temp_path.chmod(0o600)
    elif failure_class == 'uncertain':
      final_path.write_text('uncertain\n', encoding='utf-8')
      final_path.chmod(0o400)
    elif failure_class != 'no_new':
      raise AssertionError(f'unknown fixture class {failure_class}')
    temp_state = _entry_state(temp_path)
    final_state = _entry_state(final_path)
    publication_failure = {
        'schema_version': analyzer.PUBLICATION_SCHEMA_VERSION,
        'method': analyzer.PUBLICATION_METHOD,
        'root_role': 'model_run', 'artifact_role': 'fixture',
        'final_relative_path': final_relative,
        'temp_relative_path': temp_relative,
        'publication_ordinal': 0, 'runner_pid': runner_pid,
        'failure_stage': (
            'parent_fsync' if rename_succeeded else
            'write' if failure_class == 'temporary' else
            'final_preexistence' if failure_class.startswith('preexisting')
            else 'temp_open'
        ),
        'errno': 17 if failure_class.startswith('preexisting') else 5,
        'error_type': 'OSError', 'message': 'fixture',
        'rename_noreplace_attempted': rename_succeeded,
        'rename_noreplace_succeeded': rename_succeeded,
        'parent_fsync_attempted': rename_succeeded,
        'parent_fsync_succeeded': False,
        'temp_state': temp_state, 'final_state': final_state,
        'created_at_unix_s': 3.0,
    }
    temporary = {}
    uncertain = {}
    preexisting = {}
    if failure_class == 'temporary':
      temporary[temp_relative] = _live_binding(temp_path)
    elif failure_class == 'uncertain':
      uncertain[final_relative] = _live_binding(final_path)
    elif failure_class.startswith('preexisting'):
      preexisting[final_relative] = final_state
    opaque = {final_relative} if failure_class == 'preexisting_directory' else set()
    directories = analyzer._live_directory_paths(  # pylint: disable=protected-access
        root, 'fixture', opaque_directories=opaque
    )
    successful = {
        'ATTEMPT_STARTED.json': {
            'sha256': hashlib.sha256(start_path.read_bytes()).hexdigest(),
            'size_bytes': start_path.stat().st_size,
        },
    }
    source = _source_audit(None)
    for key in list(source)[:4]:
      source[key] = True
    authorization = {'fixture': True}
    consumed_fields = _both_consumed_prefix_fields()
    terminal = {
        'schema_version': analyzer.PUBLICATION_SCHEMA_VERSION,
        'status': 'incomplete_publication_failure',
        'stop_reason': 'artifact_publication_failure',
        'attempt_id': analyzer.ATTEMPT_ID,
        'script_version': analyzer.SCRIPT_VERSION,
        'external_freeze_authorization': authorization,
        'runner_pid': runner_pid,
        'publication_failure': publication_failure,
        'preterminal_tree_binding': {
            'file_count': 1, 'directory_count': len(directories),
            'file_bindings': successful,
            'file_tree_sha256': analyzer._binding_map_digest(successful),  # pylint: disable=protected-access
            'directory_paths': directories,
            'directory_tree_sha256': analyzer._directory_digest(directories),  # pylint: disable=protected-access
        },
        'source_input_audit': source,
        'source_input_audit_content_binding': analyzer._content_binding(source),  # pylint: disable=protected-access
        'same_object_attestation': None,
        'same_object_attestation_content_binding': None,
        'phase_state': _phase(
            preflight_passed=True, start_persisted=True,
            post_start_source_gate_passed=True,
        ),
        'model_apply_attempt_count': 0,
        'model_apply_success_count': 0,
        'valid_record_count': 0, 'failed_current_binding': None,
        'temporary_orphan_bindings': temporary,
        'durability_uncertain_final_bindings': uncertain,
        'preexisting_entry_states': preexisting,
        'no_new_entry_failure': no_new,
        'confirmation_model_calls': 0,
        'scientific_summary_computed': False,
        'donor_normalization_computed': False,
        'shapley_or_nomination_computed': False,
        'interaction_or_resolution_computed': False,
        'nomination_performed': False,
        'combined_analysis_permitted': False,
        'no_retry': True, 'created_at_unix_s': 4.0,
        **copy.deepcopy(consumed_fields),
    }
    _write_json_0400(root / 'TERMINAL_FAILURE.json', terminal)
    return terminal, {
        'external_freeze_authorization': authorization,
        'runner_pid': runner_pid,
        **copy.deepcopy(consumed_fields),
    }

  def test_publication_contract_is_literal_and_exact(self):
    contract = {
        'schema_version': analyzer.PUBLICATION_SCHEMA_VERSION,
        'method': analyzer.PUBLICATION_METHOD,
        'temp_name_regex': (
            r'^\.v3345\.tmp\.[1-9][0-9]*\.[0-9]{6}\.[0-9a-f]{32}$'
        ),
        'nonce_bytes': 16,
        'open_flags': [
            'O_RDWR', 'O_CREAT', 'O_EXCL', 'O_NOFOLLOW', 'O_CLOEXEC'
        ],
        'initial_mode': '0600', 'sealed_mode': '0400',
        'rename_flags': ['RENAME_NOREPLACE'],
        'same_directory_required': True,
        'keep_fd_open_through_rename': True,
        'file_fsync_count': 2, 'parent_fsync_required': True,
        'post_publish_inode_revalidation_required': True,
        'no_replace': True, 'no_fallback': True, 'no_retry': True,
        'temporary_orphan_preservation_required': True,
        'durability_uncertain_final_preservation_required': True,
        'successful_publication_object_keys': list(
            analyzer.PUBLICATION_SUCCESS_KEYS
        ),
        'publication_failure_object_keys': list(
            analyzer.PUBLICATION_FAILURE_KEYS
        ),
        'entry_state_object_keys': list(analyzer.ENTRY_STATE_KEYS),
        'external_preflight_probe_contract': {
            'final_basename': 'atomic_publication_probe_v3_3_4_5.txt',
            'final_sha256': (
                    '7ffb46419c01255944db76c4530e7943574212aa4c4595fa85254bc9d21d6bd1'
            ),
            'final_size_bytes': 49,
            'collision_sha256': (
                    'd7e55ae0ed0453b3d29f92731588b9626f10d5814b0f0ecd3198ced485940d44'
            ),
            'collision_size_bytes': 39, 'collision_errno': 17,
            'collision_temp_preserved': True,
            'parent_fsync_exact_required': True,
        },
    }
    self.assertEqual(
        analyzer._validate_publication_contract(contract), contract  # pylint: disable=protected-access
    )
    bootstrap_path = (
        _HERE / 'validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_5.py'
    )
    spec = importlib.util.spec_from_file_location(
        '_v3343_bootstrap_publication_contract', bootstrap_path
    )
    assert spec is not None and spec.loader is not None
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    self.assertEqual(
        analyzer._validate_publication_contract(  # pylint: disable=protected-access
            bootstrap.PUBLICATION_CONTRACT_V3_3_4_1
        ),
        bootstrap.PUBLICATION_CONTRACT_V3_3_4_1,
    )
    preflight_source = (
        _HERE / 'run_device_preflight_v3_3_4_5.py'
    ).read_text(encoding='utf-8')
    self.assertIn(
        'bootstrap.PUBLICATION_CONTRACT_V3_3_4_1', preflight_source
    )
    changed = copy.deepcopy(contract)
    changed['no_retry'] = False
    with self.assertRaises(analyzer.AnalysisError):
      analyzer._validate_publication_contract(changed)  # pylint: disable=protected-access

  def test_all_publication_terminal_classes_are_exact_structural_archives(self):
    expected = {
        'no_new': 'publication_failed_no_new_entry_no_scientific_analysis',
        'preexisting_regular': (
            'preexisting_entry_preserved_no_scientific_analysis'
        ),
        'preexisting_directory': (
            'preexisting_entry_preserved_no_scientific_analysis'
        ),
        'temporary': 'temporary_orphan_preserved_no_scientific_analysis',
        'uncertain': (
            'durability_uncertain_final_preserved_no_scientific_analysis'
        ),
    }
    for failure_class, decision in expected.items():
      with self.subTest(failure_class=failure_class):
        with tempfile.TemporaryDirectory() as directory:
          root = Path(directory) / 'run'
          terminal, start = self._publication_terminal_fixture(
              root, failure_class
          )
          with mock.patch.object(analyzer, '_load_cases', return_value=_cases()):
            checked, detail, audit = (
                analyzer._validate_terminal_failure_archive(  # pylint: disable=protected-access
                    root, terminal, start=start
                )
            )
          self.assertEqual(detail['decision'], decision)
          self.assertEqual(checked['valid_record_count'], 0)
          self.assertTrue(audit['no_publication_retry'])
          binding = analyzer._terminal_failure_run_binding(  # pylint: disable=protected-access
              root, checked
          )
          self.assertEqual(binding['terminal_kind'], 'terminal_failure')

  def test_analysis_publication_audit_has_exact_17_keys(self):
    with tempfile.TemporaryDirectory() as directory:
      base = Path(directory)
      attempt = base / 'attempt'
      output = base / 'output'
      attempt.mkdir(mode=0o700)
      output.mkdir(mode=0o700)
      start = attempt / 'ANALYSIS_ATTEMPT_STARTED.json'
      result = output / 'RESULT.md'
      start.write_text('{}\n', encoding='utf-8')
      result.write_text('structural\n', encoding='utf-8')
      start.chmod(0o400)
      result.chmod(0o400)
      attempt_tree = analyzer._publication_tree_binding(  # pylint: disable=protected-access
          attempt, role='analysis_attempt',
          expected_files={'ANALYSIS_ATTEMPT_STARTED.json'},
      )
      output_tree = analyzer._publication_tree_binding(  # pylint: disable=protected-access
          output, role='analysis_output', expected_files={'RESULT.md'},
      )
      audit = analyzer._analysis_publication_audit(  # pylint: disable=protected-access
          attempt_tree=attempt_tree, output_tree=output_tree,
      )
      self.assertEqual(len(audit), 17)
      self.assertEqual(audit['successful_final_count_before_terminal'], 2)
      self.assertEqual(
          set(audit['successful_final_bindings_before_terminal']),
          {'analysis_attempt', 'analysis_output'},
      )
      bad = copy.deepcopy(audit)
      bad['temporary_orphan_count'] = 1
      with self.assertRaises(analyzer.AnalysisError):
        analyzer._validate_analysis_publication_audit(bad)  # pylint: disable=protected-access

  def test_output_state_represents_all_publication_failure_classes(self):
    for case in ('temporary', 'uncertain', 'preexisting'):
      with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / 'analysis'
        output.mkdir(mode=0o700)
        audit = _publication_audit()
        if case == 'temporary':
          path = output / f'.v3345.tmp.12.000001.{"a" * 32}'
        else:
          path = output / 'ANALYSIS.json'
        path.write_bytes(case.encode())
        path.chmod(0o400)
        if case == 'temporary':
          audit['temporary_orphan_count'] = 1
          audit['temporary_orphan_bindings'] = {path.name: _live_binding(path)}
        elif case == 'uncertain':
          audit['durability_uncertain_final_count'] = 1
          audit['durability_uncertain_final_bindings'] = {
              path.name: _live_binding(path)
          }
        else:
          audit['preexisting_entry_count'] = 1
          audit['preexisting_entry_states'] = {path.name: _entry_state(path)}
        audit['publication_failure'] = {
            'fixture': True,
        }
        with mock.patch.object(analyzer, '_ANALYSIS_DIR', output):
          state = analyzer._analysis_output_state(audit)  # pylint: disable=protected-access
        self.assertEqual(state['state'], 'publication_failure_prefix')
        self.assertEqual(state['published_prefix'], [])

  def test_model_publication_walker_rejects_symlink_to_directory(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / 'run'
      root.mkdir(mode=0o700)
      terminal = root / 'POST_START_PROVENANCE_FAILURE.json'
      terminal.write_text('{}\n', encoding='utf-8')
      terminal.chmod(0o400)
      target = Path(directory) / 'target'
      target.mkdir()
      (root / 'linked').symlink_to(target, target_is_directory=True)
      with self.assertRaisesRegex(analyzer.AnalysisError, 'symlink'):
        analyzer._model_publication_audit_without_failure(  # pylint: disable=protected-access
            root, terminal_name=terminal.name,
        )

  def test_publisher_delegates_to_shared_helper_and_checks_live_inode(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      path = root / 'RESULT.md'
      nonce = 'b' * 32

      def publish_bytes(role, relative, payload, *, artifact_role):
        self.assertEqual((role, relative, artifact_role), (
            'analysis_output', 'RESULT.md', 'fixture_result'
        ))
        path.write_bytes(payload)
        path.chmod(0o400)
        binding = _live_binding(path)
        return {
            'schema_version': analyzer.PUBLICATION_SCHEMA_VERSION,
            'method': analyzer.PUBLICATION_METHOD, 'root_role': role,
            'final_relative_path': relative,
            'temp_basename': f'.v3345.tmp.{__import__("os").getpid()}.000000.{nonce}',
            'publication_ordinal': 0, 'runner_pid': __import__('os').getpid(),
            'nonce_hex': nonce, **binding,
            'file_fsync_before_rename': True,
            'file_fsync_after_fchmod': True,
            'rename_noreplace_succeeded': True,
            'parent_fsync_succeeded': True,
            'post_publish_revalidation_exact': True,
        }

      fake = types.SimpleNamespace(publish_bytes=publish_bytes)
      with mock.patch.dict(
          sys.modules,
          {'validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_5': fake},
      ):
        result = analyzer._publish_new_bytes(  # pylint: disable=protected-access
            path, b'fixed\n', root_role='analysis_output', root=root,
            artifact_role='fixture_result',
        )
      self.assertEqual(result['sha256'], hashlib.sha256(b'fixed\n').hexdigest())

  def test_source_has_no_forbidden_publication_fallback(self):
    source = _ANALYZER_PATH.read_text(encoding='utf-8')
    for forbidden in (
        'O_' + 'TMPFILE', '/proc/self/' + 'fd', 'link' + 'at(',
        'os.' + 'rename(', 'os.' + 'replace(',
    ):
      self.assertNotIn(forbidden, source)

  def test_terminal_directory_audit_preserves_empty_nested_directories(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / 'run'
      (root / 'empty/nested').mkdir(parents=True, mode=0o700)
      root.chmod(0o700)
      (root / 'empty').chmod(0o700)
      (root / 'empty/nested').chmod(0o700)
      self.assertEqual(
          analyzer._live_directory_paths(root, 'fixture'),  # pylint: disable=protected-access
          ['.', 'empty', 'empty/nested'],
      )
      (root / 'opaque/ignored-child').mkdir(parents=True, mode=0o700)
      (root / 'opaque').chmod(0o755)
      self.assertEqual(
          analyzer._live_directory_paths(  # pylint: disable=protected-access
              root, 'fixture', opaque_directories={'opaque'}
          ),
          ['.', 'empty', 'empty/nested', 'opaque'],
      )

  def test_publication_terminal_does_not_fabricate_failed_current(self):
    with tempfile.TemporaryDirectory() as directory:
      run = Path(directory) / 'run'
      run.mkdir(mode=0o700)
      source_binding = analyzer._content_binding(_source_audit())  # pylint: disable=protected-access
      object_binding = analyzer._content_binding(_same_object())  # pylint: disable=protected-access
      started_path = run / 'dispatch_journal/started/000.json'
      started = _dispatch_event(
          index=0, completed=False, runner_pid=42,
          source_sha=source_binding['sha256'],
          object_sha=object_binding['sha256'],
      )
      _write_json_0400(started_path, started)
      completed = _dispatch_event(
          index=0, completed=True, runner_pid=42,
          source_sha=source_binding['sha256'],
          object_sha=object_binding['sha256'],
          started_sha=hashlib.sha256(started_path.read_bytes()).hexdigest(),
      )
      _write_json_0400(
          run / 'dispatch_journal/completed/000.json', completed
      )
      terminal = {
          'valid_record_count': 0,
          'model_apply_attempt_count': 1,
          'model_apply_success_count': 1,
          'failed_current_binding': None,
          'runner_pid': 42,
          'external_freeze_authorization': {'fixture': True},
      }
      with mock.patch.object(analyzer, '_load_cases', return_value=_cases()):
        analyzer._validate_terminal_failure_prefix(  # pylint: disable=protected-access
            run, terminal, source_binding=source_binding,
            object_binding=object_binding,
        )

  def test_failed_current_uses_exact_nested_runner_path(self):
    case = {'order': 0, 'variant_id': 'chr1:2:A>G'}
    self.assertEqual(
        analyzer._failed_current_relative(case, 127),  # pylint: disable=protected-access
        'raw/failed_current/000_chr1_2_A_G/127.json',
    )


class NonpublicationTerminalAmendmentTest(unittest.TestCase):

  def _fixture(
      self, root: Path, stage: str,
  ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    diagnostic = stage in {
        'source_program_gate_derivation_for_diagnostic_failure',
        'diagnostic_failure_record_construction',
    }
    preterminal = {
        'ATTEMPT_STARTED.json', 'IMPORT_PROVENANCE_PRE_MODEL.json',
        'PROTOBUF_PROVENANCE.json',
        'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
        'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json',
    }
    if diagnostic:
      preterminal |= {
          'compiler/eight_row/graph.stablehlo.mlir',
          'compiler/eight_row/graph.pre_backend.hlo.txt',
          'compiler/eight_row/graph.compiled.hlo.txt',
          'IMPORT_PROVENANCE.json',
      }
    for relative in sorted(preterminal):
      path = root / relative
      path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
      for parent in [path.parent, *path.parent.parents]:
        if parent == root or root in parent.parents:
          parent.chmod(0o700)
      path.write_bytes((relative + '\n').encode())
      path.chmod(0o400)
    bindings = {
        relative: {
            'sha256': hashlib.sha256((root / relative).read_bytes()).hexdigest(),
            'size_bytes': (root / relative).stat().st_size,
        }
        for relative in sorted(preterminal)
    }
    directories = analyzer._parent_directories(preterminal)  # pylint: disable=protected-access
    source = _source_audit(True)
    if not diagnostic:
      source['three_import_inventories_stable_exact'] = None
    same = _same_object()
    same['compiler_record_is_gate_record'] = True if diagnostic else None
    read_matrix = {
        'stablehlo_text_extraction': (None, None, None),
        'pre_backend_hlo_text_extraction': (True, None, None),
        'compiled_hlo_text_extraction': (True, True, None),
        'source_program_gate_derivation_for_diagnostic_failure': (
            True, True, True
        ),
        'diagnostic_failure_record_construction': (True, True, True),
    }[stage]
    for name, value in zip((
        'stablehlo_read_from_lowered_object',
        'pre_backend_hlo_read_from_lowered_object',
        'compiled_hlo_read_from_compiled_object',
    ), read_matrix, strict=True):
      same[name] = value
    phase = {name: False for name in analyzer._PHASE_STATE_KEYS}  # pylint: disable=protected-access
    for name in (
        'preflight_passed', 'start_persisted',
        'post_start_source_gate_passed', 'protobuf_persisted',
        'pre_model_import_inventory_persisted',
        'model_construction_attempted', 'model_constructed',
        'reference_cases_loaded', 'signatures_captured',
        'signature_attestation_persisted',
        'post_model_import_inventory_persisted', 'lower_attempted',
        'lower_succeeded', 'compile_attempted', 'compile_succeeded',
    ):
      phase[name] = True
    if diagnostic:
      phase['terminal_import_inventory_persisted'] = True
    authorization = {'fixture': 'authorization'}
    consumed_fields = _both_consumed_prefix_fields()
    start = {
        'git_head': 'a' * 40, 'runner_pid': 42, 'started_at_unix_s': 1.0,
        'external_freeze_authorization': authorization,
        **copy.deepcopy(consumed_fields),
    }
    graph_paths = {
        name: binding for name, binding in bindings.items()
        if name.endswith(('.mlir', '.hlo.txt'))
    }
    import_names = {
        'pre_model': 'IMPORT_PROVENANCE_PRE_MODEL.json',
        'post_model_precompile': 'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
        'terminal': 'IMPORT_PROVENANCE.json',
    }
    import_bindings = {}
    for phase_name, relative in import_names.items():
      if relative not in bindings:
        import_bindings[phase_name] = None
      else:
        import_bindings[phase_name] = {'path': relative, **bindings[relative]}
    source_gate = (
        {'fixture': 'source gate'}
        if stage == 'diagnostic_failure_record_construction' else None
    )
    source_gate_binding = (
        analyzer._content_binding(source_gate)  # pylint: disable=protected-access
        if source_gate is not None else None
    )
    triggering_reason = 'diagnostic_parser_failure' if diagnostic else None
    triggering_failure = (
        {
            'type': 'EntryAbiParserFailure',
            'message': 'adversarial cache fingerprint publication words',
            'traceback': 'tb',
        }
        if diagnostic else None
    )
    terminal = {
        key: None for key in analyzer.NONPUBLICATION_TERMINAL_KEYS
    }
    terminal.update({
        'schema_version': 'v3.3.4.5-nonpublication-terminal-v1',
        'status': 'incomplete_nonpublication_infrastructure_failure',
        'stop_reason': 'post_compile_nonpublication_infrastructure_failure',
        'attempt_id': analyzer.ATTEMPT_ID,
        'script_version': analyzer.SCRIPT_VERSION,
        'amendment_commit': analyzer.AMENDMENT_COMMIT,
        'amendment_sha256': analyzer.AMENDMENT_SHA256,
        'inherited_v3_3_4_commit': analyzer.PREDECESSOR_AMENDMENT_COMMIT,
        'inherited_v3_3_4_sha256': analyzer.PREDECESSOR_AMENDMENT_SHA256,
        'inherited_v3_3_4_1_commit': analyzer.PUBLICATION_AMENDMENT_COMMIT,
        'inherited_v3_3_4_1_sha256': analyzer.PUBLICATION_AMENDMENT_SHA256,
        'freeze_sha256': 'b' * 64, 'git_head': start['git_head'],
        'external_freeze_authorization': authorization, 'runner_pid': 42,
        'started_at_unix_s': 1.0, 'created_at_unix_s': 2.0,
        'failure_stage': stage,
        'failure': {'type': 'RuntimeError', 'message': 'later', 'traceback': 'tb'},
        'triggering_diagnostic_failure': triggering_failure,
        'triggering_diagnostic_stop_reason': triggering_reason,
        'phase_state': phase, 'source_input_audit': source,
        'source_input_audit_content_binding': analyzer._content_binding(source),  # pylint: disable=protected-access
        'program_signature_attestation_binding': {
            'path': 'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json',
            **bindings['compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json'],
        },
        'same_object_attestation': same,
        'same_object_attestation_content_binding': analyzer._content_binding(same),  # pylint: disable=protected-access
        'attempt_budget_audit': {
            'lower_budget': 1, 'compile_budget': 1, 'lower_invocations': 1,
            'compile_invocations': 1, 'forbidden_request': None,
            'forbidden_request_detected_before_invocation': False,
        },
        'compiler_counts': {
            'lower_attempt_count': 1, 'compile_attempt_count': 1,
            'successful_compile_count': 1,
        },
        'graph_artifact_bindings': graph_paths,
        'import_provenance_phases': import_bindings,
        'protobuf_provenance_sha256': bindings['PROTOBUF_PROVENANCE.json']['sha256'],
        'model_kernel_cache_state': {'fixture': 'cache'},
        'source_program_gate_without_backend_diagnostics': source_gate,
        'source_program_gate_without_backend_diagnostics_content_binding': (
            source_gate_binding
        ),
        'prior_v3_3_3_binding': {'prior': 333},
        'prior_v3_3_3_1_archive_binding': {'prior': 331},
        'preterminal_tree_binding': {
            'file_count': len(bindings), 'directory_count': len(directories),
            'file_bindings': bindings,
            'file_tree_sha256': analyzer._binding_map_digest(bindings),  # pylint: disable=protected-access
            'directory_paths': directories,
            'directory_tree_sha256': analyzer._directory_digest(directories),  # pylint: disable=protected-access
        },
        'publication_audit': {'fixture': 'publication'},
        'model_apply_attempt_count': 0, 'model_apply_success_count': 0,
        'valid_record_count': 0, 'raw_record_count': 0,
        'dispatch_started_count': 0, 'dispatch_completed_count': 0,
        'six_row_compile_count': 0, 'identity_rerun_count': 0,
        'main_cube_rerun_count': 0, 'old_ood_records_reused': 0,
        'confirmation_model_calls': 0,
        'confirmation_scope_disclosure': analyzer.CONFIRMATION_DISCLOSURE,
        'scientific_summary_computed': False,
        'donor_normalization_computed': False,
        'shapley_or_nomination_computed': False,
        'interaction_or_resolution_computed': False,
        'nomination_performed': False, 'combined_analysis_permitted': False,
        'no_retry': True,
        **copy.deepcopy(consumed_fields),
    })
    _write_json_0400(root / 'NONPUBLICATION_TERMINAL_FAILURE.json', terminal)
    return terminal, start, {
        'program_signatures': {}, 'source_program_contract': {},
        'prior_v3_3_4_3_consumed_preflight_prefix': (
            consumed_fields['prior_v3_3_4_3_consumed_preflight_prefix']
        ),
        'prior_v3_3_4_4_consumed_preflight_prefix': (
            consumed_fields['prior_v3_3_4_4_consumed_preflight_prefix']
        ),
    }

  def test_exact_62_key_contract_is_literal(self):
    self.assertEqual(len(analyzer.NONPUBLICATION_TERMINAL_KEYS), 62)
    self.assertEqual(len(set(analyzer.NONPUBLICATION_TERMINAL_KEYS)), 62)
    self.assertIn(
        'triggering_diagnostic_stop_reason',
        analyzer.NONPUBLICATION_TERMINAL_KEYS,
    )
    bootstrap_path = (
        _HERE / 'validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_5.py'
    )
    spec = importlib.util.spec_from_file_location('_v3343_bootstrap_contract', bootstrap_path)
    assert spec is not None and spec.loader is not None
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    self.assertEqual(
        analyzer._expected_nonpublication_terminal_contract(),  # pylint: disable=protected-access
        bootstrap.NONPUBLICATION_TERMINAL_CONTRACT_V3_3_4_5,
    )
    self.assertEqual(
        bootstrap.PUBLICATION_SCHEMA_VERSION,
        analyzer.PUBLICATION_SCHEMA_VERSION,
    )
    self.assertEqual(
        bootstrap.PUBLICATION_CONTRACT_V3_3_4_1['temp_name_regex'],
        r'^\.v3345\.tmp\.[1-9][0-9]*\.[0-9]{6}\.[0-9a-f]{32}$',
    )

  def test_runner_identity_literals_match_committed_protocol(self):
    runner = ast.parse(
        (_HERE / 'run_encoder_skip_ood_sidecar_v3_3_4_5.py').read_text(
            encoding='utf-8'
        )
    )
    values = {}
    for node in runner.body:
      if (
          isinstance(node, ast.Assign) and len(node.targets) == 1
          and isinstance(node.targets[0], ast.Name)
          and node.targets[0].id in {'SCRIPT_VERSION', 'ATTEMPT_ID'}
      ):
        values[node.targets[0].id] = ast.literal_eval(node.value)
    self.assertEqual(values['SCRIPT_VERSION'], analyzer.SCRIPT_VERSION)
    self.assertEqual(values['ATTEMPT_ID'], analyzer.ATTEMPT_ID)
    preflight = ast.parse(
        (_HERE / 'run_device_preflight_v3_3_4_5.py').read_text(
            encoding='utf-8'
        )
    )
    preflight_versions = [
        ast.literal_eval(node.value) for node in preflight.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == 'PREFLIGHT_SCRIPT_VERSION'
            for target in node.targets
        )
    ]
    self.assertEqual(
        preflight_versions, [analyzer.PREFLIGHT_SCRIPT_VERSION]
    )
    self.assertNotEqual(analyzer.PREFLIGHT_SCRIPT_VERSION, analyzer.SCRIPT_VERSION)

  def test_runner_diagnostic_classifier_is_operation_typed(self):
    runner_path = _HERE / 'run_encoder_skip_ood_sidecar_v3_3_4_5.py'
    runner_source = runner_path.read_text(encoding='utf-8')
    runner = ast.parse(runner_source)
    classes = {
        node.name: node for node in runner.body if isinstance(node, ast.ClassDef)
    }
    expected_bases = {
        'EntryAbiParserFailure': 'DiagnosticParserFailure',
        'BackendDiagnosticParserFailure': 'DiagnosticParserFailure',
        'DiagnosticPersistenceFailure': 'DiagnosticProvenanceError',
        'CacheSignalUnavailable': 'DiagnosticProvenanceError',
        'FingerprintFormulaMismatch': 'DiagnosticProvenanceError',
    }
    for name, base in expected_bases.items():
      self.assertIn(name, classes)
      self.assertEqual(
          [item.id for item in classes[name].bases if isinstance(item, ast.Name)],
          [base],
      )
    parser_reason = next(
        ast.literal_eval(item.value)
        for item in classes['DiagnosticParserFailure'].body
        if (
            isinstance(item, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'reason'
                    for target in item.targets)
        )
    )
    self.assertEqual(parser_reason, 'diagnostic_parser_failure')
    self.assertNotIn('str(error).lower()', runner_source)
    self.assertIn(
        '_diagnostic_stop_reason(triggering_error) != triggering_reason',
        runner_source,
    )
    self.assertIn(
        'diagnostic_error = DiagnosticPersistenceFailure(error)',
        runner_source,
    )

  def test_machine_extracted_serializer_keysets_match_analyzer(self):
    def function(tree, name):
      return next(
          node for node in tree.body
          if isinstance(node, ast.FunctionDef) and node.name == name
      )

    def returned_dict_keys(node):
      returned = next(
          item.value for item in ast.walk(node)
          if isinstance(item, ast.Return)
      )
      if isinstance(returned, ast.Dict):
        value = returned
      elif isinstance(returned, ast.Name):
        value = next(
            item.value for item in ast.walk(node)
            if isinstance(item, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == returned.id for target in item.targets
            )
            and isinstance(item.value, ast.Dict)
        )
      else:
        self.fail('Serializer does not return a literal dict or named dict.')
      return {
        key.value for key in value.keys if isinstance(key, ast.Constant)
      }

    def assigned_dict_keys(node, target_name, expected_count):
      candidates = []
      for item in ast.walk(node):
        if not isinstance(item, ast.Assign) or not isinstance(item.value, ast.Dict):
          continue
        if not any(
            isinstance(target, ast.Name) and target.id == target_name
            for target in item.targets
        ):
          continue
        keys = {
            key.value for key in item.value.keys if isinstance(key, ast.Constant)
        }
        if len(keys) == expected_count:
          candidates.append(keys)
      self.assertEqual(len(candidates), 1)
      return candidates[0]

    launcher = ast.parse(
        (_HERE / 'launch_encoder_skip_ood_sidecar_v3_3_4_5.py').read_text(
            encoding='utf-8'
        )
    )
    runner = ast.parse(
        (_HERE / 'run_encoder_skip_ood_sidecar_v3_3_4_5.py').read_text(
            encoding='utf-8'
        )
    )
    preflight = ast.parse(
        (_HERE / 'run_device_preflight_v3_3_4_5.py').read_text(
            encoding='utf-8'
        )
    )
    self.assertEqual(
        assigned_dict_keys(function(preflight, 'run_preflight'), 'record', 22),
        analyzer._PREFLIGHT_RECORD_KEYS,  # pylint: disable=protected-access
    )
    self.assertEqual(
        returned_dict_keys(function(launcher, '_start_record')),
        analyzer._START_KEYS,  # pylint: disable=protected-access
    )
    self.assertEqual(
        assigned_dict_keys(
            function(launcher, '_validate_preflight_record'),
            'successful', 12,
        ),
        {
            'artifact_binding', 'root_file_count', 'root_file_tree_sha256',
            'external_pid', 'status', 'external_freeze_authorization',
            'external_cache_post_observation',
            'external_cache_hit_evidence',
            'prior_v3_3_4_3_consumed_preflight_prefix',
            'prior_v3_3_4_3_consumed_preflight_prefix_content_binding',
            'prior_v3_3_4_4_consumed_preflight_prefix',
            'prior_v3_3_4_4_consumed_preflight_prefix_content_binding',
        },
    )
    self.assertEqual(
        assigned_dict_keys(
            function(launcher, '_post_start_failure'), 'record', 24
        ),
        analyzer._POST_START_FAILURE_KEYS,  # pylint: disable=protected-access
    )
    self.assertEqual(
        assigned_dict_keys(
              function(runner, '_write_common_terminal'), 'record', 68
        ),
        analyzer._RUN_COMPLETE_KEYS,  # pylint: disable=protected-access
    )
    self.assertEqual(
        assigned_dict_keys(
              function(runner, '_write_terminal_failure'), 'record', 35
        ),
        analyzer.TERMINAL_FAILURE_KEYS,
    )
    self.assertEqual(
        assigned_dict_keys(
            function(runner, '_write_nonpublication_terminal'), 'record', 62
        ),
        analyzer.NONPUBLICATION_TERMINAL_KEYS,
    )
    self.assertEqual(len(analyzer._FREEZE_KEYS), 86)  # pylint: disable=protected-access
    self.assertEqual(len(analyzer._V3345_SOURCE_PATHS), 12)  # pylint: disable=protected-access
    self.assertEqual(len(analyzer._ANALYSIS_STARTED_KEYS), 18)  # pylint: disable=protected-access
    self.assertEqual(len(analyzer._ANALYSIS_KEYS), 27)  # pylint: disable=protected-access

  def test_all_five_runner_shaped_memberships_and_nullability(self):
    for stage in sorted(analyzer.NONPUBLICATION_FAILURE_STAGES):
      with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
        run = Path(directory) / 'run'
        terminal, start, freeze = self._fixture(run, stage)
        diagnostic = stage.startswith(('source_program_', 'diagnostic_'))
        with (
            mock.patch.object(
                analyzer, '_validate_signature_attestation',
                return_value={'canonical_sha256': analyzer.PROGRAM_SIGNATURES_SHA256},
            ),
            mock.patch.object(analyzer, '_validate_imports', return_value={'ok': True}),
            mock.patch.object(analyzer, '_validate_protobuf', return_value={'ok': True}),
            mock.patch.object(
                analyzer, '_validate_nonpublication_cache',
                return_value={'path': 'cache'},
            ),
            mock.patch.object(
                analyzer, '_validate_run_publication_audit',
                return_value={'publication': True},
            ),
            mock.patch.object(
                analyzer, '_validate_nonpublication_source_gate',
                return_value={
                    'entry_abi_exact': False, 'source_program_exact': False,
                },
            ) as gate,
        ):
          checked, audit, run_binding, auxiliary = (
              analyzer._validate_nonpublication_terminal_archive(  # pylint: disable=protected-access
                  run, terminal, start=start, freeze=freeze,
                  freeze_sha='b' * 64, bundle_root=Path(directory),
                  prior333={'prior': 333}, prior331={'prior': 331},
              )
          )
        self.assertEqual(checked['failure_stage'], stage)
        self.assertEqual(run_binding['file_count'], 10 if diagnostic else 6)
        self.assertFalse(audit['diagnostic_provenance_complete'])
        self.assertEqual(auxiliary['cache'], {'path': 'cache'})
        self.assertEqual(
            gate.call_count,
            int(stage == 'diagnostic_failure_record_construction'),
        )

  def test_trigger_source_and_same_object_tamper_fail_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      run = Path(directory) / 'run'
      terminal, start, freeze = self._fixture(
          run, 'stablehlo_text_extraction'
      )
      terminal['triggering_diagnostic_stop_reason'] = 'diagnostic_parser_failure'
      with self.assertRaisesRegex(analyzer.AnalysisError, 'invented'):
        analyzer._validate_nonpublication_terminal_archive(  # pylint: disable=protected-access
            run, terminal, start=start, freeze=freeze, freeze_sha='b' * 64,
            bundle_root=Path(directory), prior333={'prior': 333},
            prior331={'prior': 331},
        )
      terminal['triggering_diagnostic_stop_reason'] = None
      terminal['same_object_attestation']['compiled_python_id'] = 7
      terminal['same_object_attestation_content_binding'] = (
          analyzer._content_binding(terminal['same_object_attestation'])  # pylint: disable=protected-access
      )
      with self.assertRaisesRegex(analyzer.AnalysisError, 'same-object'):
        analyzer._validate_nonpublication_same_object(  # pylint: disable=protected-access
            terminal['same_object_attestation'],
            terminal['same_object_attestation_content_binding'],
            'stablehlo_text_extraction',
        )

  def test_operation_typed_diagnostic_reason_ignores_adversarial_message(self):
    valid = 'HloModule x, fingerprint_before_lhs="abc"\n'
    cases = (
        (
            'EntryAbiParserFailure', 'diagnostic_parser_failure',
            'not an HloModule line\n', 'entry_abi',
        ),
        (
            'BackendDiagnosticParserFailure', 'diagnostic_parser_failure',
            valid + '  %x = f32[] custom-call(), backend_config={\n',
            'backend_diagnostics',
        ),
        (
            'DiagnosticPersistenceFailure',
            'diagnostic_persistence_failure', valid,
            'diagnostic_persistence_residual',
        ),
        (
            'CacheSignalUnavailable', 'cache_signal_unavailable',
            'the cache operation precedes parser replay\n',
            'cache_evidence_nullability',
        ),
        (
            'FingerprintFormulaMismatch', 'fingerprint_formula_mismatch',
            'HloModule x\n', 'entry_abi',
        ),
    )
    for trigger_type, reason, compiled, operation in cases:
      with self.subTest(trigger_type=trigger_type):
        failure = {
            'type': trigger_type,
            'message': 'cache parser fingerprint publication adversary',
            'traceback': 'adversarial free-form traceback',
        }
        audit = analyzer._validate_triggering_diagnostic_operation(  # pylint: disable=protected-access
            failure, reason, compiled,
        )
        self.assertEqual(audit['operation_replayed'], operation)
        wrong_reason = next(
            item for item in analyzer.DIAGNOSTIC_STOP_REASONS
            if item != reason
        )
        with self.assertRaisesRegex(analyzer.AnalysisError, 'type/reason'):
          analyzer._validate_triggering_diagnostic_operation(  # pylint: disable=protected-access
              failure, wrong_reason, compiled,
          )
    evidence_mismatches = (
        ('EntryAbiParserFailure', 'diagnostic_parser_failure', valid),
        ('FingerprintFormulaMismatch', 'fingerprint_formula_mismatch', valid),
        (
            'BackendDiagnosticParserFailure', 'diagnostic_parser_failure',
            valid,
        ),
        (
            'DiagnosticPersistenceFailure',
            'diagnostic_persistence_failure',
            valid + '  %x = f32[] custom-call(), backend_config={\n',
        ),
    )
    for trigger_type, reason, compiled in evidence_mismatches:
      with self.subTest(mismatched_evidence=trigger_type):
        with self.assertRaises(analyzer.AnalysisError):
          analyzer._validate_triggering_diagnostic_operation(  # pylint: disable=protected-access
              {
                  'type': trigger_type, 'message': 'irrelevant',
                  'traceback': 'tb',
              },
              reason, compiled,
          )

  def test_record_construction_phase_equals_true_or_false_source_gate(self):
    for source_exact in (False, True):
      with self.subTest(source_exact=source_exact), tempfile.TemporaryDirectory() as directory:
        run = Path(directory) / 'run'
        terminal, start, freeze = self._fixture(
            run, 'diagnostic_failure_record_construction'
        )
        terminal['phase_state']['source_program_gate_passed'] = source_exact
        with (
            mock.patch.object(
                analyzer, '_validate_signature_attestation',
                return_value={'canonical_sha256': analyzer.PROGRAM_SIGNATURES_SHA256},
            ),
            mock.patch.object(analyzer, '_validate_imports', return_value={'ok': True}),
            mock.patch.object(analyzer, '_validate_protobuf', return_value={'ok': True}),
            mock.patch.object(
                analyzer, '_validate_nonpublication_cache',
                return_value={'path': 'cache'},
            ),
            mock.patch.object(
                analyzer, '_validate_run_publication_audit',
                return_value={'publication': True},
            ),
            mock.patch.object(
                analyzer, '_validate_nonpublication_source_gate',
                return_value={
                    'entry_abi_exact': source_exact,
                    'source_program_exact': source_exact,
                },
            ),
        ):
          checked, _, _, _ = analyzer._validate_nonpublication_terminal_archive(  # pylint: disable=protected-access
              run, terminal, start=start, freeze=freeze,
              freeze_sha='b' * 64, bundle_root=Path(directory),
              prior333={'prior': 333}, prior331={'prior': 331},
          )
        self.assertIs(
            checked['phase_state']['source_program_gate_passed'], source_exact
        )
        terminal['phase_state']['source_program_gate_passed'] = not source_exact
        with (
            mock.patch.object(
                analyzer, '_validate_signature_attestation',
                return_value={'canonical_sha256': analyzer.PROGRAM_SIGNATURES_SHA256},
            ),
            mock.patch.object(analyzer, '_validate_imports', return_value={'ok': True}),
            mock.patch.object(analyzer, '_validate_protobuf', return_value={'ok': True}),
            mock.patch.object(
                analyzer, '_validate_nonpublication_cache',
                return_value={'path': 'cache'},
            ),
            mock.patch.object(
                analyzer, '_validate_run_publication_audit',
                return_value={'publication': True},
            ),
            mock.patch.object(
                analyzer, '_validate_nonpublication_source_gate',
                return_value={
                    'entry_abi_exact': source_exact,
                    'source_program_exact': source_exact,
                },
            ),
            self.assertRaisesRegex(analyzer.AnalysisError, 'phase-state'),
        ):
          analyzer._validate_nonpublication_terminal_archive(  # pylint: disable=protected-access
              run, terminal, start=start, freeze=freeze,
              freeze_sha='b' * 64, bundle_root=Path(directory),
              prior333={'prior': 333}, prior331={'prior': 331},
          )

  def test_every_diagnostic_same_object_primitive_is_applicable_true(self):
    names = (
        'stablehlo_read_from_lowered_object',
        'pre_backend_hlo_read_from_lowered_object',
        'compile_argument_is_lowered_object',
        'compiled_hlo_read_from_compiled_object',
        'signature_attestation_from_apply_arguments',
        'apply_callable_is_compiled_object',
        'compiler_record_is_gate_record',
    )
    base = _same_object()
    for stage in (
        'source_program_gate_derivation_for_diagnostic_failure',
        'diagnostic_failure_record_construction',
    ):
      base['compiler_record_is_gate_record'] = True
      analyzer._validate_nonpublication_same_object(  # pylint: disable=protected-access
          base, analyzer._content_binding(base), stage  # pylint: disable=protected-access
      )
      for name in names:
        with self.subTest(stage=stage, name=name):
          changed = copy.deepcopy(base)
          changed[name] = False
          with self.assertRaisesRegex(analyzer.AnalysisError, 'same-object'):
            analyzer._validate_nonpublication_same_object(  # pylint: disable=protected-access
                changed, analyzer._content_binding(changed), stage  # pylint: disable=protected-access
            )

  def test_nonpublication_source_gate_recomputes_every_leaf(self):
    with tempfile.TemporaryDirectory() as directory:
      run = Path(directory)
      paths = {
          'compiler/eight_row/graph.stablehlo.mlir': b'stable',
          'compiler/eight_row/graph.pre_backend.hlo.txt': b'pre',
          'compiler/eight_row/graph.compiled.hlo.txt': b'compiled',
      }
      artifacts = {}
      for relative, payload in paths.items():
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        artifacts[relative] = {
            'sha256': hashlib.sha256(payload).hexdigest(),
            'size_bytes': len(payload),
        }
      source = _source_audit(True)
      same = _same_object()
      same['compiler_record_is_gate_record'] = True
      signatures = {'fixture': []}
      signature_sha = analyzer._canonical_json_sha256(signatures)  # pylint: disable=protected-access
      observed = {
          'stablehlo_sha256': artifacts[
              'compiler/eight_row/graph.stablehlo.mlir'
          ]['sha256'],
          'stablehlo_size_bytes': len(b'stable'),
          'pre_backend_hlo_sha256': artifacts[
              'compiler/eight_row/graph.pre_backend.hlo.txt'
          ]['sha256'],
          'pre_backend_hlo_size_bytes': len(b'pre'),
          'program_signatures_sha256': signature_sha,
          'entry_abi_sha256': 'e' * 64,
      }
      gate = {
          'contract': {'fixture': 'contract'}, 'observed': observed,
          'stablehlo_exact': True, 'pre_backend_hlo_exact': True,
          'program_signature_structure_exact': True,
          'program_signatures_canonical_exact': True,
          'entry_abi_exact': True,
          'source_runtime_device_toolchain_checkpoint_reference_exact': True,
          'source_input_audit': source,
          'source_input_audit_content_binding': analyzer._content_binding(source),  # pylint: disable=protected-access
          'same_object_attestation': same,
          'same_object_attestation_content_binding': analyzer._content_binding(same),  # pylint: disable=protected-access
          'same_lowered_compiled_object': True, 'source_program_exact': True,
      }
      with (
          mock.patch.object(
              analyzer, 'SOURCE_STABLEHLO',
              artifacts['compiler/eight_row/graph.stablehlo.mlir'],
          ),
          mock.patch.object(
              analyzer, 'SOURCE_PRE_BACKEND_HLO',
              artifacts['compiler/eight_row/graph.pre_backend.hlo.txt'],
          ),
          mock.patch.object(analyzer, 'PROGRAM_SIGNATURES_SHA256', signature_sha),
          mock.patch.object(
              analyzer, '_diagnostic_entry_abi_exact', return_value=True
          ),
      ):
        checked = analyzer._validate_nonpublication_source_gate(  # pylint: disable=protected-access
            gate, analyzer._content_binding(gate), artifacts=artifacts,  # pylint: disable=protected-access
            source_audit=source, same_object=same,
            freeze={
                'program_signatures': signatures,
                'source_program_contract': {'fixture': 'contract'},
            },
            run_dir=run, reason='diagnostic_parser_failure',
            triggering_failure={
                'type': 'BackendDiagnosticParserFailure',
                'message': 'adversarial cache text', 'traceback': 'tb'
            },
        )
        self.assertTrue(checked['source_program_exact'])
        for name in (
            'stablehlo_exact', 'pre_backend_hlo_exact',
            'program_signature_structure_exact',
            'program_signatures_canonical_exact', 'entry_abi_exact',
            'source_runtime_device_toolchain_checkpoint_reference_exact',
            'same_lowered_compiled_object',
        ):
          changed = copy.deepcopy(gate)
          changed[name] = False
          with self.subTest(name=name), self.assertRaisesRegex(
              analyzer.AnalysisError, 'evidence'
          ):
            analyzer._validate_nonpublication_source_gate(  # pylint: disable=protected-access
                changed, analyzer._content_binding(changed), artifacts=artifacts,  # pylint: disable=protected-access
                source_audit=source, same_object=same,
                freeze={
                    'program_signatures': signatures,
                    'source_program_contract': {'fixture': 'contract'},
                },
                run_dir=run, reason='diagnostic_parser_failure',
                triggering_failure={
                    'type': 'BackendDiagnosticParserFailure',
                    'message': 'adversarial cache text', 'traceback': 'tb'
                },
            )

  def test_cache_signal_source_gate_has_no_entry_evidence(self):
    with tempfile.TemporaryDirectory() as directory:
      run = Path(directory)
      payloads = {
          'compiler/eight_row/graph.stablehlo.mlir': b'stable',
          'compiler/eight_row/graph.pre_backend.hlo.txt': b'pre',
          'compiler/eight_row/graph.compiled.hlo.txt': (
              b'HloModule fixture, fingerprint_before_lhs="abcd"\nENTRY x\n'
          ),
      }
      artifacts = {}
      for relative, payload in payloads.items():
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        artifacts[relative] = {
            'sha256': hashlib.sha256(payload).hexdigest(),
            'size_bytes': len(payload),
        }
      source = _source_audit(True)
      same = _same_object()
      same['compiler_record_is_gate_record'] = True
      signatures = {'fixture': []}
      signature_sha = analyzer._canonical_json_sha256(signatures)  # pylint: disable=protected-access
      observed = {
          'stablehlo_sha256': artifacts[
              'compiler/eight_row/graph.stablehlo.mlir'
          ]['sha256'],
          'stablehlo_size_bytes': len(b'stable'),
          'pre_backend_hlo_sha256': artifacts[
              'compiler/eight_row/graph.pre_backend.hlo.txt'
          ]['sha256'],
          'pre_backend_hlo_size_bytes': len(b'pre'),
          'program_signatures_sha256': signature_sha,
          'entry_abi_sha256': '',
      }
      gate = {
          'contract': {'fixture': 'contract'}, 'observed': observed,
          'stablehlo_exact': True, 'pre_backend_hlo_exact': True,
          'program_signature_structure_exact': True,
          'program_signatures_canonical_exact': True,
          'entry_abi_exact': False,
          'source_runtime_device_toolchain_checkpoint_reference_exact': True,
          'source_input_audit': source,
          'source_input_audit_content_binding': analyzer._content_binding(source),  # pylint: disable=protected-access
          'same_object_attestation': same,
          'same_object_attestation_content_binding': analyzer._content_binding(same),  # pylint: disable=protected-access
          'same_lowered_compiled_object': True, 'source_program_exact': False,
      }
      kwargs = {
          'artifacts': artifacts, 'source_audit': source,
          'same_object': same,
          'freeze': {
              'program_signatures': signatures,
              'source_program_contract': {'fixture': 'contract'},
          },
          'run_dir': run, 'reason': 'cache_signal_unavailable',
          'triggering_failure': {
              'type': 'CacheSignalUnavailable',
              'message': 'parser fingerprint publication words are irrelevant',
              'traceback': 'tb',
          },
      }
      with (
          mock.patch.object(
              analyzer, 'SOURCE_STABLEHLO',
              artifacts['compiler/eight_row/graph.stablehlo.mlir'],
          ),
          mock.patch.object(
              analyzer, 'SOURCE_PRE_BACKEND_HLO',
              artifacts['compiler/eight_row/graph.pre_backend.hlo.txt'],
          ),
          mock.patch.object(analyzer, 'PROGRAM_SIGNATURES_SHA256', signature_sha),
      ):
        checked = analyzer._validate_nonpublication_source_gate(  # pylint: disable=protected-access
            gate, analyzer._content_binding(gate), **kwargs  # pylint: disable=protected-access
        )
        self.assertFalse(checked['entry_abi_exact'])
        self.assertFalse(checked['source_program_exact'])
        wrong_type = copy.deepcopy(kwargs)
        wrong_type['triggering_failure'] = {
            'type': 'BackendDiagnosticParserFailure',
            'message': 'cache', 'traceback': 'tb',
        }
        with self.assertRaisesRegex(analyzer.AnalysisError, 'type/reason'):
          analyzer._validate_nonpublication_source_gate(  # pylint: disable=protected-access
              gate, analyzer._content_binding(gate), **wrong_type  # pylint: disable=protected-access
          )
        fabricated = copy.deepcopy(gate)
        fabricated['observed']['entry_abi_sha256'] = 'e' * 64
        fabricated['entry_abi_exact'] = True
        fabricated['source_program_exact'] = True
        with self.assertRaisesRegex(analyzer.AnalysisError, 'entry evidence'):
          analyzer._validate_nonpublication_source_gate(  # pylint: disable=protected-access
              fabricated, analyzer._content_binding(fabricated), **kwargs  # pylint: disable=protected-access
          )

  def test_cache_signal_unavailable_is_the_only_null_evidence_case(self):
    empty_binding = {
        'directory_paths': ['.', 'triton', 'xdg'], 'file_count': 0,
        'tree_sha256': '1' * 64,
    }
    terminal_binding = {
        'directory_paths': ['.', 'triton', 'xdg'], 'file_count': 1,
        'tree_sha256': '2' * 64,
    }
    cache = {
        'pre_import': {}, 'historical_stage': 'post_compile',
        'historical_binding': {}, 'terminal_live_binding': {},
        'cache_hit_evidence': None,
        'historical_to_terminal_tree_exact': False,
        'historical_to_terminal_equality_is_a_gate': False,
        'historical_snapshot_not_reauthenticated_as_live_files': True,
        'default_user_cache_paths_eligible': False,
        'cache_outputs_are_diagnostic_only': True,
    }
    with mock.patch.object(
        analyzer, '_validate_cache_binding',
        side_effect=[empty_binding, empty_binding, terminal_binding],
    ):
      result = analyzer._validate_nonpublication_cache(  # pylint: disable=protected-access
          cache, triggering_reason='cache_signal_unavailable'
      )
    self.assertIsNone(result['cache_hit_evidence'])
    with (
        mock.patch.object(
            analyzer, '_validate_cache_binding',
            side_effect=[empty_binding, empty_binding, terminal_binding],
        ),
        self.assertRaises((analyzer.AnalysisError, TypeError)),
    ):
      analyzer._validate_nonpublication_cache(  # pylint: disable=protected-access
          cache, triggering_reason='diagnostic_parser_failure'
      )
    cache['cache_hit_evidence'] = _model_cache_evidence(
        pre_import_files_present=False, compile_skipped=False
    )
    with mock.patch.object(
        analyzer, '_validate_cache_binding',
        side_effect=[empty_binding, empty_binding, terminal_binding],
    ):
      normal = analyzer._validate_nonpublication_cache(  # pylint: disable=protected-access
          cache, triggering_reason='diagnostic_parser_failure'
      )
    self.assertFalse(normal['cache_hit_evidence']['cache_hit'])

  def test_nonpublication_result_is_exact_structural_only_outcome(self):
    result = analyzer._result_v3345(  # pylint: disable=protected-access
        status='complete_incomplete_nonpublication_infrastructure_archive',
        decision='post_compile_nonpublication_failure_no_scientific_analysis',
        terminal_kind='nonpublication_terminal_failure',
        compiler_state='compiled_without_legal_graph_gate_record',
        k=0, d=0, started=0, completed=0, id0=False, id255=False,
        prior333={'bound': True}, prior331={'bound': True},
        start_binding={'sha256': 'a' * 64}, run_binding={'terminal_kind': 'x'},
        preflight_binding={'bound': True},
        model_publication_audit=_publication_audit(),
    )
    self.assertEqual(
        result['analysis_version'], 'v3.3.4.5-structural-analyzer-v1'
    )
    self.assertFalse(result['control_audit']['control_state_eligible'])
    for key in (
        'scientific_summary_computed', 'donor_normalization_computed',
        'shapley_or_nomination_computed',
        'interaction_or_resolution_computed', 'nomination_performed',
        'combined_analysis_permitted',
    ):
      self.assertFalse(result[key])

  def test_analyze_routes_nonpublication_without_raw_access(self):
    with tempfile.TemporaryDirectory() as directory:
      run = Path(directory) / 'run'
      run.mkdir(mode=0o700)
      _write_json_0400(run / 'NONPUBLICATION_TERMINAL_FAILURE.json', {})
      start = {
          'git_head': 'a' * 40, 'runner_pid': 42,
          'started_at_unix_s': 1.0,
          'external_freeze_authorization': {'authorized': True},
      }
      terminal = {
          'status': 'incomplete_nonpublication_infrastructure_failure',
          'stop_reason': 'post_compile_nonpublication_infrastructure_failure',
          'failure_stage': 'compiled_hlo_text_extraction',
          'triggering_diagnostic_stop_reason': None,
      }
      cache = {
          'path': 'cache', 'pre_import_binding': None,
          'historical_binding': None, 'terminal_live_binding': None,
          'directory_paths_exact': True, 'cache_hit': False,
          'cache_hit_evidence': None,
          'historical_to_terminal_equality_is_a_gate': False,
      }
      reached = {'raw': False}
      with (
          mock.patch.object(
              analyzer, '_validate_freeze_v3345',
              return_value=({}, 'b' * 64, {'prior': 333}, {}, {}, {'prior': 331}),
          ),
          mock.patch.object(analyzer, '_validate_start_v3345', return_value=start),
          mock.patch.object(
              analyzer, '_validate_preflight_and_same_process',
              return_value={'preflight': True},
          ),
          mock.patch.object(
              analyzer, '_validate_nonpublication_terminal_archive',
              return_value=(
                  terminal, {'source_input_audit_exact': True},
                  {'terminal_kind': 'nonpublication_terminal_failure'},
                  {'publication': _publication_audit(), 'cache': cache},
              ),
          ),
          mock.patch.object(
              analyzer, '_absolute_binding', return_value={'sha256': 'c' * 64}
          ),
      ):
        result = analyzer.analyze(
            run, bundle_root=Path(directory),
            _raw_access_marker=lambda: reached.__setitem__('raw', True),
        )
      self.assertFalse(reached['raw'])
      self.assertEqual(
          result['status'],
          'complete_incomplete_nonpublication_infrastructure_archive',
      )
      self.assertEqual(
          result['decision'],
          'post_compile_nonpublication_failure_no_scientific_analysis',
      )


if __name__ == '__main__':
  unittest.main()
