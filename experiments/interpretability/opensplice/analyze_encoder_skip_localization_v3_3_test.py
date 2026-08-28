"""CPU-only synthetic tests for the v3.3 offline analyzer."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import sys
import unittest
from unittest import mock


_MODULE_PATH = Path(__file__).with_name(
    'analyze_encoder_skip_localization_v3_3.py'
)
_SPEC = importlib.util.spec_from_file_location(
    'analyze_encoder_skip_localization_v3_3', _MODULE_PATH
)
analyzer = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = analyzer
_SPEC.loader.exec_module(analyzer)


def _write_json(path: Path, value) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
      json.dumps(value, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8'
  )


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _readout(values: list[float]) -> dict:
  logits, margins, totals, means = [], [], [], []
  for raw in values:
    value = analyzer._f32(raw)
    logits.append([[value, 0.0], [value, 0.0]])
    margins.append([value, value])
    total = analyzer._f32(value + value)
    totals.append(total)
    means.append(analyzer._f32(total / 2.0))
  return {
      'endpoint_axis': ['acceptor', 'donor'],
      'selected_logit_axis': ['relevant_class', 'padding_class'],
      'selected_logits': logits,
      'endpoint_margins': margins,
      'totals': totals,
      'means': means,
      'num_values': 2,
  }


def _metric(b: float, movement_scale: float = 1.0) -> dict:
  movement = b * movement_scale
  return {
      'baseline_delta': 1.0,
      'movements': {
          'reference_into_alternate': -movement,
          'alternate_into_reference': movement,
      },
      'mean_absolute_movement': abs(movement),
      'recoveries': {
          'reference_into_alternate': b,
          'alternate_into_reference': b,
      },
      'B': b,
  }


def _runtime(rows: int, coalition_id: int, donor_rows: list[int]) -> dict:
  t, e_mask = divmod(coalition_id, 128)
  natural = [0, 1, 1, 1, 0, 0] + ([6, 7] if rows == 8 else [])
  active = [False, False, True, True, True, True] + (
      [False, False] if rows == 8 else []
  )

  def whole(enabled):
    return {
        'donor_batch_indices': [list(donor_rows) for _ in enabled],
        'natural_identity_batch_indices': [list(natural) for _ in enabled],
        'transfer_mask': [
            [bool(on and recipient) for recipient in active] for on in enabled
        ],
    }

  residual = {
      'donor_batch_indices': [
          [[row] * 24 for row in range(rows)] for _ in range(9)
      ],
      'transfer_mask': [
          [[False] * 24 for _ in range(rows)] for _ in range(9)
      ],
  }
  return {
      'transformer_output': whole([bool(t)]),
      'encoder_skips': whole([
          bool(e_mask & (1 << index)) for index in range(7)
      ]),
      'final_embedding': {
          'donor_batch_indices': [[[row, row] for row in range(rows)]],
          'transfer_mask': [[[False, False] for _ in range(rows)]],
      },
      'phase_r_residuals': {
          name: json.loads(json.dumps(residual)) for name in (
              'pre_attention_residual_transfer',
              'post_attention_residual_transfer',
              'post_mlp_residual_transfer',
          )
      },
  }


def _coalition_metadata(coalition_id: int) -> dict:
  t, e_mask = divmod(coalition_id, 128)
  e_bits = [bool(e_mask & (1 << index)) for index in range(7)]
  return {
      'coalition_id': coalition_id,
      't': t,
      'e_mask': e_mask,
      'e_bits': e_bits,
      'e_bits_binary': format(e_mask, '07b'),
      'enabled_players': (['T'] if t else []) + [
          player for player, enabled in zip(analyzer.PLAYERS_E, e_bits, strict=True)
          if enabled
      ],
      'coalition_bit_order': list(analyzer.PLAYERS_E) + ['T'],
      'shapley_player_order': list(analyzer.PLAYERS_8),
  }


class ReadoutTest(unittest.TestCase):

  def test_reconstructs_raw_endpoint_evidence(self):
    readout = _readout([0, 1, 0.5, 1, 0.5, 0])
    record = {'target_readout': readout, 'repeat_target_readout': readout}
    observed = analyzer._require_repeat(
        record, 'target_readout', 'repeat_target_readout', 'test', rows=6
    )
    self.assertEqual(observed['means'][2], 0.5)
    self.assertAlmostEqual(analyzer._metrics(observed)['B'], 0.5)

  def test_rejects_tampered_margin(self):
    readout = _readout([0, 1, 0.5, 1, 0.5, 0])
    readout['endpoint_margins'][2][0] = 0.25
    with self.assertRaisesRegex(analyzer.AnalysisError, 'differs'):
      analyzer._readout({'target_readout': readout}, 'target_readout', 'x', rows=6)


class RuntimeInterventionTest(unittest.TestCase):

  def test_exact_six_row_runtime_arrays(self):
    runtime = _runtime(6, 131, [0, 1, 0, 1, 1, 0])
    analyzer._runtime_route(
        runtime, rows=6, coalition_id=131,
        donor_rows=[0, 1, 0, 1, 1, 0], label='runtime',
    )

  def test_tampered_active_mask_fails(self):
    runtime = _runtime(6, 131, [0, 1, 0, 1, 1, 0])
    runtime['encoder_skips']['transfer_mask'][0][2] = False
    with self.assertRaisesRegex(analyzer.AnalysisError, 'transfer mask'):
      analyzer._runtime_route(
          runtime, rows=6, coalition_id=131,
          donor_rows=[0, 1, 0, 1, 1, 0], label='runtime',
      )

  def test_tampered_disabled_residual_self_map_fails(self):
    runtime = _runtime(8, 255, [0, 1, 6, 1, 7, 0, 6, 7])
    runtime['phase_r_residuals'][
        'post_mlp_residual_transfer'
    ]['donor_batch_indices'][2][3][4] = 2
    with self.assertRaisesRegex(analyzer.AnalysisError, 'disabled-route'):
      analyzer._runtime_route(
          runtime, rows=8, coalition_id=255,
          donor_rows=[0, 1, 6, 1, 7, 0, 6, 7], label='runtime',
      )

  def test_rejects_nonfinite(self):
    readout = _readout([0, 1, 0.5, 1, 0.5, 0])
    readout['selected_logits'][0][0][0] = float('nan')
    with self.assertRaisesRegex(analyzer.AnalysisError, 'non-finite'):
      analyzer._readout({'target_readout': readout}, 'target_readout', 'x', rows=6)


class OodValidationTest(unittest.TestCase):

  def _binding(self, run_dir: Path, relative: str) -> dict:
    path = run_dir / relative
    _write_json(path, {'bound': relative})
    return {'path': relative, 'sha256': _sha(path)}

  def _record(self, run_dir: Path, *, coalition_id: int = 127):
    cases = analyzer._load_cases()
    recipient, donor = cases[0], cases[10]
    identity_relative = analyzer._artifact_relative('identity', recipient, None)
    donor_identity_relative = analyzer._artifact_relative('identity', donor, None)
    linked_relative = analyzer._artifact_relative(
        'coalition', recipient, coalition_id
    )
    recipient_sequence = {'reference': 'a' * 64, 'alternate': 'b' * 64}
    donor_sequence = {'reference': 'c' * 64, 'alternate': 'd' * 64}
    # These eight-row baselines deliberately differ from the six-row identity
    # baselines. Only comparisons within the one eight-row executable are valid.
    intended = _readout([10, 20, 15, 20, 15, 10, 30, 40])
    unrelated = _readout([10, 20, 30, 20, 40, 10, 30, 40])
    checks = {
        key: True for key in (
            'passed', 'baseline_rows_exact_between_calls',
            'self_rows_exact_between_calls',
            'donor_source_rows_exact_between_calls', 'self_targets_exact',
            'intended_route_tensor_donor_exact',
            'unrelated_route_tensor_donor_exact',
            'enabled_disabled_T_E_exact',
            'runtime_route_masks_and_maps_exact',
            'natural_route_tensors_exact_between_calls',
            'transformer_internal_seams_disabled_exact',
            'final_embedding_disabled_exact',
            'intended_target_repeat_exact', 'intended_trace_repeat_exact',
            'unrelated_target_repeat_exact', 'unrelated_trace_repeat_exact',
        )
    }
    checks.update({
        'normalization_computed': False,
        'id0_all_recipient_rows_noop_exact': False,
        'id0_intended_unrelated_all_rows_exact': False,
        'id255_intended_endpoint_closure_exact': False,
        'id255_unrelated_endpoint_closure_exact': False,
    })
    intended_raw = analyzer._raw_movements(
        analyzer._readout({'x': intended}, 'x', 'fixture', rows=8)
    )['movements']
    unrelated_raw = analyzer._raw_movements(
        analyzer._readout({'x': unrelated}, 'x', 'fixture', rows=8)
    )['movements']
    fingerprint = 'e' * 64
    trace = {'sha256': 'f' * 64, 'leaves': []}
    record = {
        'status': 'complete', 'family': 'unrelated_donor_anchor',
        'script_version': analyzer.SOURCE_SCRIPT_VERSION,
        'protocol_sha256': analyzer.PROTOCOL_SHA256,
        'supersession_sha256': analyzer.SUPERSESSION_SHA256,
        'supersession_commit': 'c64def4', 'freeze_sha256': '9' * 64,
        'execution_index': 5140, 'eight_row_executable_fingerprint': fingerprint,
        'same_eight_row_compiled_executable': True,
        'recipient_case': recipient, 'donor_case': donor,
        'coalition': _coalition_metadata(coalition_id),
        'batch_roles': list(analyzer.TRACE_ROLES) + [
            'unrelated_reference_donor', 'unrelated_alternate_donor'
        ],
        'natural_identity_rows': [0, 1, 1, 1, 0, 0, 6, 7],
        'intended_donor_rows': [0, 1, 0, 1, 1, 0, 6, 7],
        'unrelated_donor_rows': [0, 1, 6, 1, 7, 0, 6, 7],
        'identity_binding': self._binding(run_dir, identity_relative),
        'donor_identity_binding': self._binding(run_dir, donor_identity_relative),
        'linked_six_row_coalition': self._binding(run_dir, linked_relative),
        'recipient_sequence_sha256': recipient_sequence,
        'donor_sequence_sha256': donor_sequence,
        'runtime_interventions': {
            'intended': _runtime(8, coalition_id, [0, 1, 0, 1, 1, 0, 6, 7]),
            'unrelated': _runtime(8, coalition_id, [0, 1, 6, 1, 7, 0, 6, 7]),
        },
        'intended_target_readout': intended,
        'intended_repeat_target_readout': intended,
        'unrelated_target_readout': unrelated,
        'unrelated_repeat_target_readout': unrelated,
        'intended_trace_fingerprint': trace,
        'intended_repeat_trace_fingerprint': trace,
        'unrelated_trace_fingerprint': trace,
        'unrelated_repeat_trace_fingerprint': trace,
        'raw_movement': {'intended': intended_raw, 'unrelated': unrelated_raw},
        'checks': checks, 'failure': None,
        'seconds': {
            'intended': 1.0, 'intended_repeat': 1.0,
            'unrelated': 1.0, 'unrelated_repeat': 1.0,
        },
        'created_at_unix_s': 2.0,
    }
    identities = (
        {'sequence_sha256': recipient_sequence, 'readout': _readout([0, 1, 1, 1, 0, 0])},
        {'sequence_sha256': donor_sequence, 'readout': _readout([2, 3, 3, 3, 2, 2])},
    )
    return record, recipient, donor, identities, (
        identity_relative, donor_identity_relative, linked_relative
    ), fingerprint

  def test_accepts_within_eight_controls_despite_cross_executable_drift(self):
    with tempfile.TemporaryDirectory() as temporary:
      run_dir = Path(temporary)
      record, recipient, donor, identities, paths, fingerprint = self._record(run_dir)
      result = analyzer._validate_ood_record(
          record, recipient, donor, 127, identity=identities[0],
          identity_relative=paths[0], donor_identity=identities[1],
          donor_identity_relative=paths[1], linked_relative=paths[2],
          run_dir=run_dir, freeze_sha='9' * 64, execution_index=5140,
          eight_fingerprint=fingerprint, label='ood',
      )
      self.assertFalse(result['donor_normalized_recovery_computed'])

  def test_donor_sequence_must_match_bound_donor_identity(self):
    with tempfile.TemporaryDirectory() as temporary:
      run_dir = Path(temporary)
      record, recipient, donor, identities, paths, fingerprint = self._record(run_dir)
      record['donor_sequence_sha256'] = {
          'reference': '1' * 64, 'alternate': '2' * 64
      }
      with self.assertRaisesRegex(analyzer.AnalysisError, 'donor identity'):
        analyzer._validate_ood_record(
            record, recipient, donor, 127, identity=identities[0],
            identity_relative=paths[0], donor_identity=identities[1],
            donor_identity_relative=paths[1], linked_relative=paths[2],
            run_dir=run_dir, freeze_sha='9' * 64, execution_index=5140,
            eight_fingerprint=fingerprint, label='ood',
        )


class ProvenanceSchemaTest(unittest.TestCase):

  def _device_observation(self, *, pid: int, runtime: dict) -> dict:
    return {
        'pid': pid,
        'jax_default_backend': 'gpu',
        'jax_gpu_devices': [{
            'device_kind': analyzer.EXPECTED_DEVICE_KIND,
            'platform': 'gpu', 'client_platform': 'gpu',
        }],
        'nvidia_smi': {'parsed_single_gpu': {
            'name': analyzer.EXPECTED_DEVICE_KIND,
            'uuid': analyzer.EXPECTED_GPU_UUID,
            'compute_capability': analyzer.EXPECTED_COMPUTE_CAPABILITY,
        }},
        'environment': {
            'LD_LIBRARY_PATH': {'present': False, 'value': None},
            'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
        },
        'no_jit_no_array_no_model': True,
        'jax_enable_compilation_cache': False,
        'v3_3_runtime_environment': runtime,
    }

  def _start_fixture(self, root: Path):
    run_dir = root / 'run'
    run_dir.mkdir()
    implementation = root / 'implementation.py'
    implementation.write_text('VALUE = 1\n', encoding='utf-8')
    generated_pb2 = root / 'calibration_scores_pb2.py'
    generated_pyi = root / 'calibration_scores_pb2.pyi'
    generated_pb2.write_text('# generated\n', encoding='utf-8')
    generated_pyi.write_text('# generated typing\n', encoding='utf-8')
    generated_outputs = {
        str(path.resolve()): {
            'sha256': _sha(path), 'size_bytes': path.stat().st_size,
        } for path in (generated_pb2, generated_pyi)
    }
    freeze_path = root / 'freeze.json'
    freeze = {
        'attempt_id': analyzer.ATTEMPT_ID,
        'script_version': analyzer.SOURCE_SCRIPT_VERSION,
        'protocol_sha256': analyzer.PROTOCOL_SHA256,
        'supersession_sha256': analyzer.SUPERSESSION_SHA256,
        'supersession_commit': 'c64def4',
        'scientific_record_count': 5220,
        'model_apply_count': 10600,
        'identity_record_count': 20,
        'coalition_record_count': 5120,
        'ood_anchor_record_count': 80,
        'six_row_compile_count': 1,
        'eight_row_compile_count': 1,
        'effect_order': list(analyzer.EFFECT_ORDERS),
        'neutral_order': list(analyzer.NEUTRAL_ORDERS),
        'gate0_anchor_ids': [0, 255],
        'ood_anchor_ids': list(analyzer.OOD_ANCHORS),
        'coalition_bit_order': list(analyzer.PLAYERS_E) + ['T'],
        'shapley_player_order': list(analyzer.PLAYERS_8),
        'shapley_absolute_tolerance': 1e-9,
        'shapley_relative_tolerance': 1e-9,
        'max_wall_time_seconds': 14400,
        'max_output_bytes': 8589934592,
        'protocol_path': str(analyzer._PROTOCOL_PATH.resolve()),
        'supersession_path': str(analyzer._SUPERSESSION_PATH.resolve()),
        'output_dir': str(run_dir.resolve()),
        'analysis_dir': str((root / 'analysis').resolve()),
        'file_sha256': {
            str(implementation.resolve().relative_to('/')): _sha(implementation)
        },
        'protobuf_binding': {
            'protobuf_runtime_version': '7.35.1',
            'embedded_generated_header': [],
            'generated_outputs': generated_outputs,
        },
    }
    _write_json(freeze_path, freeze)
    freeze_sha = _sha(freeze_path)
    start = {
        'attempt_id': analyzer.ATTEMPT_ID,
        'script_version': analyzer.SOURCE_SCRIPT_VERSION,
        'status': 'started_append_only_one_shot',
        'protocol_sha256': analyzer.PROTOCOL_SHA256,
        'supersession_sha256': analyzer.SUPERSESSION_SHA256,
        'supersession_commit': 'c64def4',
        'supersession': {
            'path': str(analyzer._SUPERSESSION_PATH.resolve()),
            'sha256': analyzer.SUPERSESSION_SHA256,
            'commit': 'c64def4',
        },
        'compile_count_contract': 2,
        'compile_count_by_executable': {'six_row': 1, 'eight_row': 1},
        'scientific_record_count_contract': 5220,
        'model_apply_count_contract': 10600,
        'max_wall_time_seconds': 14400,
        'max_output_bytes': 8589934592,
        'execution_order_contract': {
            'identities': 'manifest orders 0..19',
            'coalition_anchors': 'ID0 then ID255 for orders 0..19',
            'remaining_effects': 'orders 0..5,10..15; each IDs1..254 increasing',
            'remaining_neutrals': 'orders 6..9,16..19; each IDs1..254 increasing',
            'ood': 'orders0..19; IDs0,127,128,255',
        },
        'confirmation_model_calls': 0,
        'confirmation_scope_disclosure': analyzer.CONFIRMATION_SCOPE_DISCLOSURE,
        'freeze': {'path': str(freeze_path.resolve()), 'sha256': freeze_sha, **freeze},
        'bundle': {},
        'external_preflight': {},
        'same_process_preflight': {},
        'checkpoint_path': '',
        'checkpoint_binding': {},
        'reference_object_binding': {},
        'reference_sequence_bindings': {},
        'started_at_unix_s': 1.0,
        'same_process_pre_import_bootstrap': {
            'pid': 123,
            'created_at_unix_s': 1.0,
            'generated_bindings': {
                'pre_import_gate': True,
                'historical_generator_argv': 'unknown',
                'exact_regeneration_claim': False,
                'generated_artifact_exception': [],
                'artifacts': {
                    path.name: {
                        'path': str(path.resolve()), 'sha256': _sha(path),
                        'size_bytes': path.stat().st_size,
                    } for path in (generated_pb2, generated_pyi)
                },
                'embedded_header': [],
                'protobuf_runtime_version': '7.35.1',
            },
            'sanitized_environment': {
                'LD_LIBRARY_PATH': 'absent',
                'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
                'JAX_ENABLE_COMPILATION_CACHE': 'false',
            },
            'freeze': {
                'path': str(freeze_path.resolve()),
                'sha256': freeze_sha,
                'tracked_head_clean': True,
                'git_head': 'a' * 40,
                'tracked_paths': sorted({
                    str(freeze_path.resolve().relative_to('/')),
                    str(analyzer._PROTOCOL_PATH.resolve().relative_to('/')),
                    str(analyzer._SUPERSESSION_PATH.resolve().relative_to('/')),
                    str(implementation.resolve().relative_to('/')),
                }),
            },
            'launcher_path': str(
                analyzer._HERE.joinpath('launch_encoder_skip_factorial_v3_3.py').resolve()
            ),
            'launcher_sha256': _sha(
                analyzer._HERE / 'launch_encoder_skip_factorial_v3_3.py'
            ),
            'bootstrap_path': str(
                analyzer._HERE.joinpath('validate_encoder_skip_bootstrap_v3_3.py').resolve()
            ),
            'bootstrap_sha256': _sha(
                analyzer._HERE / 'validate_encoder_skip_bootstrap_v3_3.py'
            ),
        },
        'runtime_environment': {
            'JAX_ENABLE_COMPILATION_CACHE': 'false',
            'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
        },
        'runtime_version_binding': {},
        'prior_v3_2_evidence': analyzer.EXPECTED_PRIOR_V3_2_EVIDENCE,
    }
    start['same_process_pre_import_bootstrap']['freeze'].update({
        'upstream_checkout': {},
        'prior_v3_2_evidence': analyzer.EXPECTED_PRIOR_V3_2_EVIDENCE,
    })
    _write_json(run_dir / 'ATTEMPT_STARTED.json', start)
    return run_dir, start

  def test_runner_shaped_start_and_legitimate_disclosure(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      run_dir, _ = self._start_fixture(root)
      freeze, _, audit = analyzer._validate_start(
          run_dir, bundle_root=Path('/'), validate_external_inputs=False
      )
      self.assertEqual(freeze['model_apply_count'], 10600)
      self.assertTrue(audit['tracked_head_clean_at_pre_import'])

  def test_bootstrap_clean_gate_tamper_fails(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      run_dir, start = self._start_fixture(root)
      start['same_process_pre_import_bootstrap']['freeze'][
          'tracked_head_clean'
      ] = False
      _write_json(run_dir / 'ATTEMPT_STARTED.json', start)
      with self.assertRaisesRegex(analyzer.AnalysisError, 'bootstrap'):
        analyzer._validate_start(
            run_dir, bundle_root=Path('/'), validate_external_inputs=False
        )

  def test_same_process_pid_and_external_preflight_are_bound(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      preflight_dir = root / 'preflight'
      preflight_dir.mkdir()
      runtime = {
          'JAX_ENABLE_COMPILATION_CACHE': 'false',
          'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
      }
      stdout, stderr = preflight_dir / 'stdout.log', preflight_dir / 'stderr.log'
      stdout.write_text('ok\n', encoding='utf-8')
      stderr.write_text('', encoding='utf-8')
      observation = self._device_observation(pid=321, runtime=runtime)
      raw = {
          'script_version': 'opensplice-device-preflight-v3.3.0',
          'status': 'pass', 'protocol_sha256': analyzer.PROTOCOL_SHA256,
          'freeze_sha256': 'a' * 64, 'failure': None,
          'no_model_or_biological_access': True,
          'no_jit_or_array_kernel': True,
          'logs': {
              'stdout': {'path': str(stdout), 'sha256': _sha(stdout)},
              'stderr': {'path': str(stderr), 'sha256': _sha(stderr)},
          },
          'observation': observation,
      }
      preflight = preflight_dir / 'PREFLIGHT.json'
      _write_json(preflight, raw)
      validated_logs = raw['logs']
      start = {
          'runtime_environment': runtime,
          'same_process_pre_import_bootstrap': {
              'pid': 321, 'freeze': {'git_head': 'b' * 40}
          },
          'same_process_preflight': observation,
          'external_preflight': {
              'path': str(preflight), 'sha256': _sha(preflight),
              **raw, 'validated_logs': validated_logs,
          },
          'bundle': {
              'git_head': 'b' * 40, 'tracked_clean': True,
              'generated_artifact_exception': [
                  '?? src/alphagenome_research/protos/calibration_scores_pb2.py',
                  '?? src/alphagenome_research/protos/calibration_scores_pb2.pyi',
              ],
          },
      }
      audit = analyzer._validate_bundle_and_preflight(
          start, {'preflight_dir': str(preflight_dir)}, 'a' * 64,
          bundle_root=root,
      )
      self.assertTrue(audit['same_process_exact_rtx3090_uuid_gate'])
      start['same_process_preflight'] = self._device_observation(
          pid=999, runtime=runtime
      )
      with self.assertRaisesRegex(analyzer.AnalysisError, 'PID'):
        analyzer._validate_bundle_and_preflight(
            start, {'preflight_dir': str(preflight_dir)}, 'a' * 64,
            bundle_root=root,
        )

  def test_identity_sequence_must_match_frozen_reference(self):
    expected = {'reference': 'a' * 64, 'alternate': 'b' * 64}
    analyzer._validate_frozen_sequence_binding(
        {'sequence_sha256': expected}, expected, 'identity'
    )
    with self.assertRaisesRegex(analyzer.AnalysisError, 'frozen GRCh38'):
      analyzer._validate_frozen_sequence_binding(
          {'sequence_sha256': {
              'reference': 'a' * 64, 'alternate': 'c' * 64
          }}, expected, 'identity'
      )

  def test_exact_prior_v3_2_evidence_tree_is_current(self):
    audit = analyzer._validate_prior_v3_2_evidence(
        analyzer.EXPECTED_PRIOR_V3_2_EVIDENCE
    )
    self.assertEqual(audit['prior_v3_2_raw_artifact_count'], 2660)
    tampered = json.loads(json.dumps(analyzer.EXPECTED_PRIOR_V3_2_EVIDENCE))
    tampered['raw_manifest']['artifact_tree_sha256'] = '0' * 64
    with self.assertRaisesRegex(analyzer.AnalysisError, 'protocol section 1'):
      analyzer._validate_prior_v3_2_evidence(tampered)

  def test_runtime_manifest_binds_every_frozen_package(self):
    packages = {name: f'v-{index}' for index, name in enumerate(analyzer.RUNTIME_PACKAGES)}
    physical = {
        'index': '0', 'name': analyzer.EXPECTED_DEVICE_KIND,
        'uuid': analyzer.EXPECTED_GPU_UUID,
        'compute_capability': analyzer.EXPECTED_COMPUTE_CAPABILITY,
        'vbios': 'x', 'driver': 'y',
    }
    manifest = {
        'python_version': '3.11.0', 'platform': 'linux', 'kernel': 'kernel',
        'packages': packages, 'nvidia_smi': physical,
    }
    binding = {
        key: manifest[key] for key in ('python_version', 'platform', 'kernel', 'packages')
    }
    observation = {
        'python_version': '3.11.0 (main)', 'platform': 'linux',
        'kernel': 'kernel', 'packages': packages,
        'nvidia_smi': {'parsed_single_gpu': physical},
    }
    start = {
        'runtime_version_binding': binding,
        'same_process_preflight': observation,
        'external_preflight': {'observation': observation},
    }
    analyzer._validate_runtime_manifest(start, {'runtime_version_manifest': manifest})
    changed = json.loads(json.dumps(start))
    changed['same_process_preflight']['packages']['numpy'] = 'wrong'
    with self.assertRaisesRegex(analyzer.AnalysisError, 'package versions'):
      analyzer._validate_runtime_manifest(changed, {'runtime_version_manifest': manifest})

  def test_upstream_checkout_exact_22_tracked_plus_4_generated_split(self):
    provenance_path = (
        analyzer._PRIOR_V3_2_RUN_DIR / 'IMPORT_PROVENANCE.json'
    )
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    upstream_root = analyzer._HERE.parents[3] / 'alphagenome'
    inventory = {}
    for row in provenance['modules']:
      if row['root'] != 'upstream_alphagenome_checkout':
        continue
      inventory[row['name']] = {
          'relative_path': str(Path(row['path']).resolve().relative_to(upstream_root)),
          'sha256': row['sha256'], 'size_bytes': row['size_bytes'],
      }
    self.assertEqual(len(inventory), 26)
    exception = {
        'module_names': list(analyzer.UPSTREAM_GENERATED_MODULE_NAMES),
        'generated_outputs': analyzer.UPSTREAM_GENERATED_OUTPUTS,
        'source_inputs': analyzer.UPSTREAM_GENERATED_SOURCE_INPUTS,
        'embedded_headers': analyzer.UPSTREAM_GENERATED_HEADERS,
        'protobuf_runtime_version': '7.35.1',
        'grpcio_runtime_version': '1.83.0',
        'grpcio_tools': 'unavailable_not_used',
        'historical_generator': 'unknown',
        'historical_generator_argv': 'unknown',
        'exact_regeneration_claim': False,
    }
    head = subprocess.check_output(
        ('git', '-C', str(upstream_root), 'rev-parse', 'HEAD'), text=True
    ).strip()
    imported_modules = {
        name: {
            **binding,
            'path': str((upstream_root / binding['relative_path']).resolve()),
            'source_kind': (
                'generated_exact_byte_exception'
                if name in analyzer.UPSTREAM_GENERATED_MODULE_NAMES else 'tracked'
            ),
        }
        for name, binding in inventory.items()
    }
    freeze = {
        'upstream_imported_modules': inventory,
        'upstream_alphagenome_git_head': head,
        'upstream_generated_binding_exception': exception,
        'runtime_version_manifest': {
            'packages': {'protobuf': '7.35.1', 'grpcio': '1.83.0'}
        },
    }
    start = {'same_process_pre_import_bootstrap': {'freeze': {
        'upstream_checkout': {
            'git_head': head, 'tracked_head_clean': True,
            'imported_module_count': 26, 'imported_modules': imported_modules,
            'tracked_imported_module_count': 22,
            'generated_imported_module_count': 4,
            'generated_binding_exception': exception,
        }
    }}}
    _, audit = analyzer._validate_upstream_checkout(
        start, freeze, bundle_root=analyzer._HERE.parents[2]
    )
    self.assertEqual(audit['upstream_tracked_imported_module_count'], 22)
    self.assertTrue(audit['upstream_generated_exception_verified'])
    with tempfile.TemporaryDirectory() as temporary:
      import_path = Path(temporary) / 'IMPORT_PROVENANCE.json'
      upstream_rows = [
          row for row in provenance['modules']
          if row['root'] == 'upstream_alphagenome_checkout'
      ]
      import_record = {
          'module_count': len(upstream_rows), 'modules': upstream_rows,
          'upstream_source_attestation': start[
              'same_process_pre_import_bootstrap'
          ]['freeze']['upstream_checkout'],
      }
      _write_json(import_path, import_record)
      loaded = analyzer._validate_import_file(
          import_path, _sha(import_path), bundle_root=analyzer._HERE.parents[2],
          upstream_inventory=inventory, upstream_head=head,
          upstream_exception=exception,
      )
      self.assertEqual(len(loaded), 26)
      import_record['upstream_source_attestation']['imported_modules'][
          'alphagenome.protos.tensor_pb2'
      ]['source_kind'] = 'tracked'
      _write_json(import_path, import_record)
      with self.assertRaisesRegex(analyzer.AnalysisError, 'source attestation'):
        analyzer._validate_import_file(
            import_path, _sha(import_path),
            bundle_root=analyzer._HERE.parents[2],
            upstream_inventory=inventory, upstream_head=head,
            upstream_exception=exception,
        )
    bad = json.loads(json.dumps(freeze))
    bad['upstream_generated_binding_exception']['generated_outputs'][
        'alphagenome.protos.tensor_pb2'
    ]['sha256'] = '0' * 64
    with self.assertRaisesRegex(analyzer.AnalysisError, 'exception claims'):
      analyzer._validate_upstream_checkout(
          start, bad, bundle_root=analyzer._HERE.parents[2]
      )

  def _compiler(self, run_dir: Path, name: str):
    directory = run_dir / 'compiler' / name
    directory.mkdir(parents=True)
    artifacts = {}
    for key, filename in (
        ('stablehlo', 'graph.stablehlo.mlir'),
        ('hlo', 'graph.pre_backend.hlo.txt'),
        ('compiled_hlo', 'graph.compiled.hlo.txt'),
    ):
      path = directory / filename
      path.write_text(f'{name}:{key}\n', encoding='utf-8')
      artifacts[key] = {
          'path': str(path.resolve()), 'sha256': _sha(path),
          'size_bytes': path.stat().st_size,
      }
    fingerprint = hashlib.sha256(
        bytes.fromhex(artifacts['compiled_hlo']['sha256'])
    ).hexdigest()
    record = {
        'executable_name': name, 'compile_count': 1,
        'compile_seconds': 1.0, 'executable_fingerprint': fingerprint,
        'artifacts': artifacts,
    }
    _write_json(directory / 'COMPILER_PROVENANCE.json', record)
    return record

  def test_completion_binds_both_exact_compilers(self):
    with tempfile.TemporaryDirectory() as temporary:
      run_dir = Path(temporary)
      six = self._compiler(run_dir, 'six_row')
      eight = self._compiler(run_dir, 'eight_row')
      manifest = {
          'artifact_count': 5220, 'artifact_sha256': {},
          'artifact_tree_sha256': 'b' * 64,
      }
      complete = {
          'status': 'complete', 'stop_reason': None,
          'attempt_id': analyzer.ATTEMPT_ID,
          'script_version': analyzer.SOURCE_SCRIPT_VERSION,
          'protocol_sha256': analyzer.PROTOCOL_SHA256,
          'freeze_sha256': 'c' * 64,
          'supersession_sha256': analyzer.SUPERSESSION_SHA256,
          'supersession_commit': 'c64def4',
          'identity_count': 20, 'eligible_effect_count': 12,
          'identity_invalid_count': 0, 'coalition_record_count': 5120,
          'coalition_invalid_count': 0, 'ood_anchor_record_count': 80,
          'ood_invalid_count': 0, 'scientific_record_count': 5220,
          'model_apply_count': 10600, 'id0_noop_all20': True,
          'id255_closure_all20': True, 'all_effects_target_eligible': True,
          'all_neutrals_retained': True, 'compile_count': 2,
          'confirmation_model_calls': 0, 'raw_manifest': manifest,
          'confirmation_scope_disclosure': analyzer.CONFIRMATION_SCOPE_DISCLOSURE,
          'completed_at_unix_s': 2.0,
          'six_row_compiler': six, 'eight_row_compiler': eight,
          'six_row_executable_fingerprint': six['executable_fingerprint'],
          'eight_row_executable_fingerprint': eight['executable_fingerprint'],
      }
      observed = analyzer._validate_completion(
          complete, manifest, freeze_sha='c' * 64, run_dir=run_dir
      )
      self.assertEqual(observed[0], six['executable_fingerprint'])
      self.assertEqual(observed[1], eight['executable_fingerprint'])


class FullRawManifestTest(unittest.TestCase):
  """Exercises the literal 5,220-path contract without model-shaped payloads."""

  @classmethod
  def setUpClass(cls):
    cls._temporary = tempfile.TemporaryDirectory()
    cls.run_dir = Path(cls._temporary.name)
    cls.cases = analyzer._load_cases()
    mapping = {}
    for family, order, coalition_id in analyzer._expected_execution_order():
      relative = analyzer._artifact_relative(
          family, cls.cases[order], coalition_id
      )
      path = cls.run_dir / relative
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text('{}\n', encoding='utf-8')
      mapping[relative] = _sha(path)
    paths = [cls.run_dir / relative for relative in mapping]
    cls.manifest = {
        'artifact_count': 5220,
        'artifact_sha256': mapping,
        'artifact_tree_sha256': analyzer._tree_digest(paths, cls.run_dir),
    }
    _write_json(cls.run_dir / 'RAW_MANIFEST.json', cls.manifest)

  @classmethod
  def tearDownClass(cls):
    cls._temporary.cleanup()

  def test_exact_5220_tree_and_dependency_order(self):
    manifest, mapping = analyzer._validate_manifest(self.run_dir, self.cases)
    self.assertEqual(manifest['artifact_count'], 5220)
    self.assertEqual(len(mapping), 5220)
    order = analyzer._expected_execution_order()
    self.assertEqual(order[:3], [
        ('identity', 0, None), ('identity', 1, None), ('identity', 2, None)
    ])
    self.assertEqual(order[20:24], [
        ('coalition', 0, 0), ('coalition', 0, 255),
        ('coalition', 1, 0), ('coalition', 1, 255),
    ])
    self.assertEqual(order[-4:], [
        ('ood', 19, 0), ('ood', 19, 127),
        ('ood', 19, 128), ('ood', 19, 255),
    ])

  def test_missing_raw_file_fails_closed(self):
    relative = next(iter(self.manifest['artifact_sha256']))
    path = self.run_dir / relative
    original = path.read_bytes()
    path.unlink()
    try:
      with self.assertRaisesRegex(analyzer.AnalysisError, 'frozen execution prefix'):
        analyzer._validate_manifest(self.run_dir, self.cases)
    finally:
      path.write_bytes(original)

  def test_extra_raw_file_fails_closed(self):
    extra = self.run_dir / 'raw' / 'unexpected.json'
    extra.write_text('{}\n', encoding='utf-8')
    try:
      with self.assertRaisesRegex(analyzer.AnalysisError, 'frozen execution prefix'):
        analyzer._validate_manifest(self.run_dir, self.cases)
    finally:
      extra.unlink()

  def test_full_5220_record_orchestration_and_nomination(self):
    """Runs the analyzer's full dependency/order/statistics path CPU-only."""
    cases = self.cases
    (self.run_dir / 'RUN_COMPLETE.json').write_text('{}\n', encoding='utf-8')
    complete = {
        'status': 'complete', 'stop_reason': None, 'message': 'complete',
        'attempt_id': analyzer.ATTEMPT_ID,
        'script_version': analyzer.SOURCE_SCRIPT_VERSION,
        'protocol_sha256': analyzer.PROTOCOL_SHA256,
        'freeze_sha256': 'a' * 64,
        'supersession_sha256': analyzer.SUPERSESSION_SHA256,
        'supersession_commit': 'c64def4',
        'identity_count': 20, 'eligible_effect_count': 12,
        'all_effects_target_eligible': True, 'all_neutrals_retained': True,
        'identity_invalid_count': 0, 'coalition_record_count': 5120,
        'coalition_invalid_count': 0, 'ood_anchor_record_count': 80,
        'ood_invalid_count': 0, 'scientific_record_count': 5220,
        'model_apply_count': 10600, 'id0_noop_all20': True,
        'id255_closure_all20': True, 'six_row_compiler': {},
        'eight_row_compiler': {}, 'six_row_executable_fingerprint': '6' * 64,
        'eight_row_executable_fingerprint': '8' * 64, 'compile_count': 2,
        'import_provenance_sha256': '1' * 64,
        'import_provenance_phases': {
            'pre_model': '1' * 64, 'post_model_precompile': '2' * 64,
            'postcompile': '3' * 64,
        },
        'protobuf_provenance_sha256': '4' * 64,
        'raw_manifest': self.manifest, 'confirmation_model_calls': 0,
        'confirmation_scope_disclosure': analyzer.CONFIRMATION_SCOPE_DISCLOSURE,
        'completed_at_unix_s': 3.0,
    }
    eligibility = {
        'eligible_effects': [cases[o]['variant_id'] for o in analyzer.EFFECT_ORDERS],
        'ineligible_effects': [],
        'neutral_controls': [cases[o]['variant_id'] for o in analyzer.NEUTRAL_ORDERS],
    }
    clean_identity_readout = analyzer._readout(
        {'x': _readout([0, 1, 1, 1, 0, 0])}, 'x', 'fixture', rows=6
    )
    clean_coalition_readout = analyzer._readout(
        {'x': _readout([0, 1, 0.5, 1, 0.5, 0])}, 'x', 'fixture', rows=6
    )
    sequence_bindings = {
        order: {'reference': f'{order:064x}', 'alternate': f'{order + 20:064x}'}
        for order in analyzer.ALL_ORDERS
    }

    original_read_json = analyzer._read_json

    def fake_read_json(path):
      path = Path(path)
      if path.name == 'RUN_COMPLETE.json':
        return complete
      if path.name == 'TARGET_ELIGIBILITY.json':
        return eligibility
      if path.name == 'RAW_MANIFEST.json':
        return original_read_json(path)
      return {}

    def fake_identity(_record, case, **_kwargs):
      return {
          'readout': clean_identity_readout,
          'sequence_sha256': sequence_bindings[case['order']],
          'natural_route_fingerprints': {},
          'six_row_executable_fingerprint': '6' * 64,
          'predicted_delta': 1.0,
          'eligible': case['order'] in analyzer.EFFECT_ORDERS,
          'program_signatures': {'frozen': True},
      }

    def fake_coalition(_record, _case, coalition_id, **_kwargs):
      b = 1.0 if coalition_id in (127, 255) else 0.85 if coalition_id in (3, 131) else 0.1
      return _metric(b), clean_coalition_readout

    def fake_ood(_record, recipient, donor, coalition_id, **_kwargs):
      return {
          'recipient_order': recipient['order'], 'donor_order': donor['order'],
          'coalition_id': coalition_id,
          'intended_raw_movement': {
              'reference_into_alternate': -0.5,
              'alternate_into_reference': 0.5,
          },
          'unrelated_raw_movement': {
              'reference_into_alternate': -0.25,
              'alternate_into_reference': 0.25,
          },
          'intended_mean_absolute_movement': 0.5,
          'unrelated_mean_absolute_movement': 0.25,
          'donor_normalized_recovery_computed': False,
      }

    with mock.patch.object(analyzer, '_validate_start', return_value=(
        {'analysis_dir': str(self.run_dir.parent / 'analysis'),
         'upstream_imported_modules': {}, 'upstream_alphagenome_git_head': '0' * 40},
        'a' * 64,
        {'sequence_bindings': sequence_bindings},
    )), mock.patch.object(analyzer, '_validate_top_level_tree'), mock.patch.object(
        analyzer, '_validate_completion', return_value=(
            '6' * 64, '8' * 64, {'compile_count': 2}
        )
    ), mock.patch.object(analyzer, '_validate_imports', return_value={
        'stable_shared_module_bytes': True
    }), mock.patch.object(analyzer, '_validate_protobuf', return_value={
        'binding_verified': True
    }), mock.patch.object(analyzer, '_read_json', side_effect=fake_read_json), mock.patch.object(
        analyzer, '_validate_identity_record', side_effect=fake_identity
    ), mock.patch.object(
        analyzer, '_validate_coalition_record', side_effect=fake_coalition
    ), mock.patch.object(analyzer, '_validate_ood_record', side_effect=fake_ood):
      result = analyzer.analyze(self.run_dir, bundle_root=self.run_dir.parent)
    self.assertEqual(result['gate_0']['coalition_count'], 5120)
    self.assertEqual(result['gate_0']['ood_anchor_count'], 80)
    self.assertEqual(result['nomination']['coalition_id'], 3)
    self.assertEqual(
        result['resolution_analysis']['per_variant_decomposition']['0'][
            'reference_into_alternate'
        ]['joint_8_player']['player_order'],
        list(analyzer.PLAYERS_8),
    )


class ControlledStopIntegrationTest(unittest.TestCase):

  def _run_stop(self, reason: str):
    cases = analyzer._load_cases()
    full_order = analyzer._expected_execution_order()
    coalition_count = 1 if reason == 'gate0_closure_failure' else 0
    expected_order = full_order[:20 + coalition_count]
    with tempfile.TemporaryDirectory() as temporary:
      run_dir = Path(temporary)
      mapping = {}
      for family, order, coalition_id in expected_order:
        relative = analyzer._artifact_relative(family, cases[order], coalition_id)
        path = run_dir / relative
        _write_json(path, {})
        mapping[relative] = _sha(path)
      manifest = {
          'artifact_count': len(mapping), 'artifact_sha256': mapping,
          'artifact_tree_sha256': analyzer._tree_digest(
              [run_dir / relative for relative in mapping], run_dir
          ),
      }
      _write_json(run_dir / 'RAW_MANIFEST.json', manifest)
      (run_dir / 'RUN_COMPLETE.json').write_text('{}\n', encoding='utf-8')
      eligible_count = 11 if reason == 'target_comparability_failure' else 12
      complete = {
          'status': 'controlled_stop', 'stop_reason': reason,
          'message': 'controlled test', 'attempt_id': analyzer.ATTEMPT_ID,
          'script_version': analyzer.SOURCE_SCRIPT_VERSION,
          'protocol_sha256': analyzer.PROTOCOL_SHA256,
          'freeze_sha256': 'a' * 64,
          'supersession_sha256': analyzer.SUPERSESSION_SHA256,
          'supersession_commit': 'c64def4', 'identity_count': 20,
          'eligible_effect_count': eligible_count,
          'all_effects_target_eligible': eligible_count == 12,
          'all_neutrals_retained': True, 'identity_invalid_count': 0,
          'coalition_record_count': coalition_count,
          'coalition_invalid_count': coalition_count,
          'ood_anchor_record_count': 0, 'ood_invalid_count': 0,
          'scientific_record_count': 20 + coalition_count,
          'model_apply_count': 40 + 2 * coalition_count,
          'id0_noop_all20': False, 'id255_closure_all20': False,
          'six_row_compiler': {}, 'eight_row_compiler': {},
          'six_row_executable_fingerprint': '6' * 64,
          'eight_row_executable_fingerprint': '8' * 64,
          'compile_count': 2, 'import_provenance_sha256': '1' * 64,
          'import_provenance_phases': {
              'pre_model': '1' * 64, 'post_model_precompile': '2' * 64,
              'postcompile': '3' * 64,
          },
          'protobuf_provenance_sha256': '4' * 64,
          'raw_manifest': manifest, 'confirmation_model_calls': 0,
          'confirmation_scope_disclosure': analyzer.CONFIRMATION_SCOPE_DISCLOSURE,
          'completed_at_unix_s': 4.0,
      }
      excluded = analyzer.EFFECT_ORDERS[-1] if eligible_count == 11 else None
      eligibility = {
          'eligible_effects': [
              cases[order]['variant_id'] for order in analyzer.EFFECT_ORDERS
              if order != excluded
          ],
          'ineligible_effects': (
              [cases[excluded]['variant_id']] if excluded is not None else []
          ),
          'neutral_controls': [
              cases[order]['variant_id'] for order in analyzer.NEUTRAL_ORDERS
          ],
      }
      sequences = {
          order: {'reference': f'{order:064x}', 'alternate': f'{order + 20:064x}'}
          for order in analyzer.ALL_ORDERS
      }
      clean = analyzer._readout(
          {'x': _readout([0, 1, 1, 1, 0, 0])}, 'x', 'fixture', rows=6
      )
      original_read_json = analyzer._read_json
      terminal_relative = analyzer._artifact_relative(*(
          ('coalition', cases[0], 0) if coalition_count else
          ('identity', cases[0], None)
      )) if coalition_count else None

      def fake_read_json(path):
        path = Path(path)
        if path.name == 'RUN_COMPLETE.json':
          return complete
        if path.name == 'RAW_MANIFEST.json':
          return original_read_json(path)
        if path.name == 'TARGET_ELIGIBILITY.json':
          return eligibility
        relative = str(path.relative_to(run_dir))
        return {'status': 'invalid'} if relative == terminal_relative else {}

      def fake_identity(_record, case, **_kwargs):
        return {
            'readout': clean, 'sequence_sha256': sequences[case['order']],
            'natural_route_fingerprints': {},
            'six_row_executable_fingerprint': '6' * 64,
            'predicted_delta': 1.0,
            'eligible': (
                case['order'] in analyzer.EFFECT_ORDERS
                and case['order'] != excluded
            ),
            'program_signatures': {'frozen': True},
        }

      with mock.patch.object(analyzer, '_validate_start', return_value=(
          {'analysis_dir': str(run_dir.parent / 'analysis'),
           'upstream_imported_modules': {},
           'upstream_alphagenome_git_head': '0' * 40},
          'a' * 64, {'sequence_bindings': sequences},
      )), mock.patch.object(analyzer, '_validate_top_level_tree'), mock.patch.object(
          analyzer, '_validate_completion_bindings', return_value=(
              '6' * 64, '8' * 64, {'compile_count': 2}
          )
      ), mock.patch.object(analyzer, '_validate_imports', return_value={}), mock.patch.object(
          analyzer, '_validate_protobuf', return_value={}
      ), mock.patch.object(analyzer, '_read_json', side_effect=fake_read_json), mock.patch.object(
          analyzer, '_validate_identity_record', side_effect=fake_identity
      ), mock.patch.object(analyzer, '_validate_invalid_record'):
        result = analyzer.analyze(run_dir, bundle_root=run_dir.parent)
      return result

  def test_target_comparability_stop_refuses_shapley_and_nomination(self):
    result = self._run_stop('target_comparability_failure')
    self.assertEqual(result['controlled_stop']['eligible_effect_count'], 11)
    self.assertFalse(result['controlled_stop']['shapley_computed'])
    self.assertIsNone(result['nomination'])
    self.assertIsNone(result['resolution_analysis'])

  def test_first_invalid_gate0_prefix_refuses_shapley_and_nomination(self):
    result = self._run_stop('gate0_closure_failure')
    self.assertEqual(result['controlled_stop']['coalition_record_count'], 1)
    self.assertEqual(result['controlled_stop']['coalition_invalid_count'], 1)
    self.assertFalse(result['controlled_stop']['nomination_performed'])


class ExactGameTest(unittest.TestCase):

  def test_additive_shapley_interaction_and_harsanyi(self):
    weights = [1.0, -2.0, 3.5]
    values = {
        mask: 7.0 + sum(
            weight for index, weight in enumerate(weights) if mask & (1 << index)
        )
        for mask in range(8)
    }
    phi = analyzer.exact_shapley(values, ('a', 'b', 'c'))
    self.assertAlmostEqual(phi['a'], 1.0)
    self.assertAlmostEqual(phi['b'], -2.0)
    self.assertAlmostEqual(phi['c'], 3.5)
    interactions = analyzer.pairwise_shapley_interactions(
        values, ('a', 'b', 'c')
    )
    self.assertTrue(all(abs(value) < 1e-12 for value in interactions.values()))
    dividends, mass = analyzer.harsanyi_dividends(values, ('a', 'b', 'c'))
    self.assertEqual(dividends['000'], 7.0)
    self.assertAlmostEqual(mass['1'], 6.5)
    self.assertAlmostEqual(mass['2'], 0.0)
    self.assertAlmostEqual(mass['3'], 0.0)

  def test_pair_synergy(self):
    values = {
        mask: float(bool(mask & 1) and bool(mask & 2)) * 4.0
        for mask in range(4)
    }
    phi = analyzer.exact_shapley(values, ('a', 'b'))
    self.assertEqual(phi, {'a': 2.0, 'b': 2.0})
    interactions = analyzer.pairwise_shapley_interactions(values, ('a', 'b'))
    self.assertEqual(interactions['a:b'], 4.0)
    dividends, mass = analyzer.harsanyi_dividends(values, ('a', 'b'))
    self.assertEqual(dividends['11'], 4.0)
    self.assertEqual(mass, {'1': 0.0, '2': 4.0})

  def test_missing_cube_fails_closed(self):
    with self.assertRaisesRegex(analyzer.AnalysisError, 'incomplete'):
      analyzer.exact_shapley({0: 0.0, 1: 1.0}, ('a', 'b'))

  def test_eight_player_T_bit_is_remapped(self):
    cube = {}
    for coalition_id in range(256):
      t, e_mask = divmod(coalition_id, 128)
      movement = 10.0 * t + sum(
          index + 1 for index in range(7) if e_mask & (1 << index)
      )
      cube[coalition_id] = {
          'baseline_delta': 1.0,
          'movements': {
              'reference_into_alternate': movement,
              'alternate_into_reference': movement,
          },
      }
    result = analyzer.decompose_variant(cube)
    view = result['reference_into_alternate']['joint_8_player']
    self.assertEqual(view['player_order'], list(analyzer.PLAYERS_8))
    phi = view['raw_shapley']
    self.assertAlmostEqual(phi['T'], 10.0)
    for index, player in enumerate(analyzer.PLAYERS_E):
      self.assertAlmostEqual(phi[player], index + 1)

  def test_frozen_tolerance_on_nontrivial_float32_eight_player_game(self):
    values = {
        mask: analyzer._f32(
            math.sin(mask * 0.37) + math.cos(mask * 0.11) + mask / 257.0
        )
        for mask in range(256)
    }
    phi = analyzer.exact_shapley(values, analyzer.PLAYERS_8)
    residual = abs(
        math.fsum(phi.values()) - (values[255] - values[0])
    )
    self.assertLessEqual(
        residual,
        analyzer.SHAPLEY_ABS_TOLERANCE
        + analyzer.SHAPLEY_REL_TOLERANCE * abs(values[255] - values[0]),
    )


class NominationTest(unittest.TestCase):

  def test_reciprocal_movements_are_oriented_before_sign_comparison(self):
    metrics = _metric(0.75)
    self.assertEqual(analyzer._oriented_alt_movement(metrics), 0.75)

  def _fixture(self):
    cases, cubes = {}, {}
    for order in analyzer.ALL_ORDERS:
      gene = 'BRAF' if order < 10 else 'SLC25A48'
      cases[order] = {
          'order': order,
          'gene': gene,
          'variant_id': f'{gene}_{order}',
          'delta_logit': 1.0,
      }
      is_effect = order in analyzer.EFFECT_ORDERS
      scale = 1.0 if is_effect else 0.1
      cube = {}
      for coalition_id in range(256):
        b = 0.1
        if coalition_id in (127, 255):
          b = 1.0
        if coalition_id in (3, 131):
          b = 0.85
        cube[coalition_id] = _metric(b, scale)
      cubes[order] = cube
    return cases, cubes

  def test_minimal_skip_only_candidate_and_neutral_gate(self):
    cases, cubes = self._fixture()
    result = analyzer.summarize_cubes(cubes, cases)
    self.assertEqual(result['decision'], 'skip_only_route')
    self.assertEqual(result['nomination']['coalition_id'], 3)
    self.assertEqual(result['nomination']['enabled_skips'], ['E64', 'E32'])
    self.assertTrue(result['biological_alignment']['passes_both_exons'])
    self.assertFalse(result['T_dependent_family']['evaluated'])

  def test_skip_precedence_over_T_dependent(self):
    cases, cubes = self._fixture()
    for order in analyzer.EFFECT_ORDERS:
      cubes[order][129] = _metric(1.0)
    result = analyzer.summarize_cubes(cubes, cases)
    self.assertEqual(result['nomination']['coalition_id'], 3)

  def test_missing_neutral_cube_fails_closed(self):
    cases, cubes = self._fixture()
    del cubes[6][42]
    with self.assertRaisesRegex(analyzer.AnalysisError, 'complete 256'):
      analyzer.summarize_cubes(cubes, cases)

  def test_weak_all_e_stops_skip_family(self):
    cases, cubes = self._fixture()
    for order in range(0, 6):
      cubes[order][127] = _metric(0.0)
    result = analyzer.summarize_cubes(cubes, cases)
    self.assertIn('BRAF all-E denominator', result['skip_only_family']['failure'])
    self.assertTrue(result['T_dependent_family']['evaluated'])


class ScopeTest(unittest.TestCase):

  def test_confirmation_paths_rejected(self):
    with self.assertRaisesRegex(analyzer.AnalysisError, 'confirmation'):
      analyzer._guard_path(Path('/tmp/confirmation/results'))

  def test_forbidden_gene_in_nested_metadata_rejected(self):
    with self.assertRaisesRegex(analyzer.AnalysisError, 'confirmation'):
      analyzer._reject_forbidden_content({'case': {'gene': 'ELN'}}, 'x')

  def test_protocol_and_supersession_hashes(self):
    bindings = analyzer._validate_protocol_bindings()
    self.assertEqual(bindings['protocol_sha256'], analyzer.PROTOCOL_SHA256)
    self.assertEqual(bindings['upstream_provenance_amendment_commit'], '93227d4')
    self.assertTrue(bindings['superseded_plan_is_not_a_gate'])


if __name__ == '__main__':
  unittest.main()
