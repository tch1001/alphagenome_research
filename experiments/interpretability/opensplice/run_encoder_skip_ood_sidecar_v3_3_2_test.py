#!/usr/bin/env python3
"""CPU-only tests for the OpenSplice v3.3.2 OOD sidecar."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

import run_encoder_skip_factorial_v3_3_test as fixtures
import run_encoder_skip_ood_sidecar_v3_3_2 as runner


def _trace_with_active_rows(trace, *, row2: float, row4: float):
  natural = np.asarray(trace.stage_a.natural_final_embeddings).copy()
  natural[2] = row2
  natural[4] = row4
  natural = jnp.asarray(natural, jnp.bfloat16)
  return dataclasses.replace(
      trace,
      stage_a=dataclasses.replace(
          trace.stage_a,
          natural_final_embeddings=natural,
          effective_final_embeddings=natural,
      ),
  )


def _anchor_inputs(anchor_id: int, *, active_rows_differ: bool = True):
  if anchor_id == 0:
    intended_values = unrelated_values = (1, 3, 3, 3, 1, 1, 7, 9)
  elif anchor_id == 255:
    intended_values = (1, 3, 1, 3, 3, 1, 7, 9)
    unrelated_values = (1, 3, 7, 3, 9, 1, 7, 9)
  else:
    intended_values = (1, 3, 5, 3, 6, 1, 7, 9)
    unrelated_values = (
        (1, 3, 7, 3, 9, 1, 7, 9)
        if active_rows_differ else intended_values
    )
  intended_evidence = fixtures._evidence(intended_values)  # pylint: disable=protected-access
  unrelated_evidence = fixtures._evidence(unrelated_values)  # pylint: disable=protected-access
  intended_trace = fixtures._trace(8)  # pylint: disable=protected-access
  unrelated_trace = fixtures._trace(8)  # pylint: disable=protected-access
  if anchor_id != 0 and active_rows_differ:
    unrelated_trace = _trace_with_active_rows(
        unrelated_trace, row2=6.0, row4=7.0
    )
  selection = fixtures._selection()  # pylint: disable=protected-access
  intended_interventions = runner.v33.eight_row_interventions(
      selection, anchor_id, unrelated=False
  )
  unrelated_interventions = runner.v33.eight_row_interventions(
      selection, anchor_id, unrelated=True
  )
  return (
      (intended_evidence, intended_trace),
      (intended_evidence, intended_trace),
      (unrelated_evidence, unrelated_trace),
      (unrelated_evidence, unrelated_trace),
      anchor_id,
      intended_interventions,
      unrelated_interventions,
  )


class EncoderSkipOodSidecarV332Test(unittest.TestCase):

  def test_order_counts_and_dry_run_are_exact(self):
    cases = tuple(SimpleNamespace(order=index) for index in range(20))
    order = runner.sidecar_execution_order(cases)
    self.assertEqual(len(order), 80)
    self.assertEqual(len(set(order)), 80)
    self.assertEqual(order[:5], (
        (0, 0), (0, 127), (0, 128), (0, 255), (1, 0)
    ))
    self.assertEqual(order[-1], (19, 255))
    plan = runner.build_dry_run_plan(
        cases, max_variants=1, max_anchors=2
    )
    self.assertEqual(plan['ood_record_count'], 80)
    self.assertEqual(plan['model_apply_count'], 320)
    self.assertEqual(plan['eight_row_compile_count'], 1)
    self.assertEqual(plan['six_row_compile_count'], 0)
    self.assertEqual(plan['identity_rerun_count'], 0)
    self.assertEqual(plan['main_cube_rerun_count'], 0)
    self.assertEqual(plan['old_ood_records_reused'], 0)
    self.assertEqual(plan['confirmation_model_calls'], 0)

  def test_all_anchor_interventions_share_fixed_pytree_and_maps(self):
    selection = fixtures._selection()  # pylint: disable=protected-access
    reference = None
    for anchor_id in runner.ANCHOR_IDS:
      for unrelated, donors in (
          (False, runner.v33.EIGHT_INTENDED_DONOR_ROWS),
          (True, runner.v33.EIGHT_UNRELATED_DONOR_ROWS),
      ):
        interventions = runner.v33.eight_row_interventions(
            selection, anchor_id, unrelated=unrelated
        )
        tree = jax.tree_util.tree_structure(interventions)
        reference = reference or tree
        self.assertEqual(tree, reference)
        runner.v33._assert_runtime_transfer_contract(  # pylint: disable=protected-access
            interventions,
            anchor_id,
            batch_size=8,
            donor_rows=donors,
            identity_rows=runner.v33.EIGHT_IDENTITY_ROWS,
        )

  def test_corrected_validator_accepts_active_row_difference(self):
    checks = runner.validate_ood_sidecar_anchor(
        *_anchor_inputs(127, active_rows_differ=True)
    )
    self.assertTrue(checks[
        'natural_final_invariant_rows_exact_between_calls'
    ])
    self.assertEqual(
        checks['active_rows_cross_call_equality_not_required'], [2, 4]
    )
    self.assertTrue(checks['active_rows_forced_difference_not_required'])

  def test_corrected_validator_does_not_force_active_row_difference(self):
    checks = runner.validate_ood_sidecar_anchor(
        *_anchor_inputs(128, active_rows_differ=False)
    )
    self.assertTrue(checks['passed'])

  def test_invariant_row_and_upstream_drift_fail_closed(self):
    for invariant_row in runner.INVARIANT_ROWS:
      with self.subTest(invariant_row=invariant_row):
        args = list(_anchor_inputs(127, active_rows_differ=True))
        unrelated = args[2]
        natural = np.asarray(
            unrelated[1].stage_a.natural_final_embeddings
        ).copy()
        natural[invariant_row] = np.float32(100 + invariant_row)
        natural = jnp.asarray(natural, jnp.bfloat16)
        drifted_trace = dataclasses.replace(
            unrelated[1],
            stage_a=dataclasses.replace(
                unrelated[1].stage_a,
                natural_final_embeddings=natural,
                effective_final_embeddings=natural,
            ),
        )
        args[2] = (unrelated[0], drifted_trace)
        args[3] = (unrelated[0], drifted_trace)
        with self.assertRaisesRegex(
            ValueError, f'invariant row {invariant_row}'
        ):
          runner.validate_ood_sidecar_anchor(*args)

    for fingerprint_field in (
        'transformer_output_natural_fingerprint',
        'encoder_skips_natural_fingerprints',
    ):
      with self.subTest(fingerprint_field=fingerprint_field):
        args = list(_anchor_inputs(127, active_rows_differ=True))
        unrelated = args[2]
        branch = unrelated[1].stage_a
        value = getattr(branch, fingerprint_field)
        index = (0, 0) if value.ndim == 2 else (0, 0, 0)
        bad_fingerprint = value.at[index].set(1)
        drifted_trace = dataclasses.replace(
            unrelated[1],
            stage_a=dataclasses.replace(
                branch, **{fingerprint_field: bad_fingerprint}
            ),
        )
        args[2] = (unrelated[0], drifted_trace)
        args[3] = (unrelated[0], drifted_trace)
        with self.assertRaisesRegex(ValueError, 'upstream natural route'):
          runner.validate_ood_sidecar_anchor(*args)

    for natural_name, effective_name in runner.v32._TRANSFORMER_PAIRS:  # pylint: disable=protected-access
      with self.subTest(transformer_field=natural_name):
        args = list(_anchor_inputs(127, active_rows_differ=True))
        unrelated = args[2]
        transformer = unrelated[1].transformer
        natural = getattr(transformer, natural_name)
        changed = natural.at[(0,) * natural.ndim].set(1)
        drifted_trace = dataclasses.replace(
            unrelated[1],
            transformer=dataclasses.replace(
                transformer,
                **{natural_name: changed, effective_name: changed},
            ),
        )
        args[2] = (unrelated[0], drifted_trace)
        args[3] = (unrelated[0], drifted_trace)
        with self.assertRaisesRegex(ValueError, 'upstream transformer seam'):
          runner.validate_ood_sidecar_anchor(*args)

  def test_id0_and_id255_closures_remain_strong(self):
    id0 = runner.validate_ood_sidecar_anchor(*_anchor_inputs(0))
    self.assertTrue(id0['id0_all8_natural_final_exact_between_calls'])
    self.assertTrue(id0['id0_all8_endpoint_exact_between_calls'])
    id255 = runner.validate_ood_sidecar_anchor(*_anchor_inputs(255))
    self.assertTrue(id255['id255_intended_endpoint_closure_exact'])
    self.assertTrue(id255['id255_unrelated_endpoint_closure_exact'])

    args = list(_anchor_inputs(0))
    intended = args[0]
    unrelated = args[2]
    bad_intended = _trace_with_active_rows(
        intended[1], row2=5.0, row4=6.0
    )
    bad_unrelated = _trace_with_active_rows(
        unrelated[1], row2=5.0, row4=6.0
    )
    args[0] = (intended[0], bad_intended)
    args[1] = (intended[0], bad_intended)
    args[2] = (unrelated[0], bad_unrelated)
    args[3] = (unrelated[0], bad_unrelated)
    with self.assertRaisesRegex(ValueError, 'natural-final recipient'):
      runner.validate_ood_sidecar_anchor(*args)

    args = list(_anchor_inputs(255))
    intended_bad = fixtures._evidence((1, 3, 2, 3, 3, 1, 7, 9))  # pylint: disable=protected-access
    args[0] = (intended_bad, args[0][1])
    args[1] = (intended_bad, args[1][1])
    with self.assertRaisesRegex(ValueError, 'Endpoint readout differs'):
      runner.validate_ood_sidecar_anchor(*args)

    args = list(_anchor_inputs(255))
    unrelated_bad = fixtures._evidence((1, 3, 8, 3, 9, 1, 7, 9))  # pylint: disable=protected-access
    args[2] = (unrelated_bad, args[2][1])
    args[3] = (unrelated_bad, args[3][1])
    with self.assertRaisesRegex(ValueError, 'Endpoint readout differs'):
      runner.validate_ood_sidecar_anchor(*args)

  def test_repeat_donor_and_final_seam_tampering_fails(self):
    args = list(_anchor_inputs(255))
    changed_repeat = fixtures._evidence((1, 3, 1, 3, 3, 1, 7, 10))  # pylint: disable=protected-access
    args[1] = (changed_repeat, args[1][1])
    with self.assertRaisesRegex(ValueError, 'repeat'):
      runner.validate_ood_sidecar_anchor(*args)

    args = list(_anchor_inputs(255))
    corrupted = dataclasses.replace(
        args[5],
        stage_a=dataclasses.replace(
            args[5].stage_a,
            encoder_skips=dataclasses.replace(
                args[5].stage_a.encoder_skips,
                donor_batch_indices=(
                    args[5].stage_a.encoder_skips.donor_batch_indices
                    .at[0, 2].set(5)
                ),
            ),
        ),
    )
    args[5] = corrupted
    with self.assertRaisesRegex(ValueError, 'donor map'):
      runner.validate_ood_sidecar_anchor(*args)

    args = list(_anchor_inputs(255))
    intended = args[0]
    changed_effective = intended[1].stage_a.effective_final_embeddings.at[0, 0, 0].set(5)
    bad_trace = dataclasses.replace(
        intended[1],
        stage_a=dataclasses.replace(
            intended[1].stage_a,
            effective_final_embeddings=changed_effective,
        ),
    )
    args[0] = (intended[0], bad_trace)
    args[1] = (intended[0], bad_trace)
    with self.assertRaisesRegex(ValueError, 'final seam'):
      runner.validate_ood_sidecar_anchor(*args)

  def test_compact_rowwise_fingerprint_is_exact_and_row_local(self):
    values = np.arange(8 * 2 * 3, dtype=np.uint16).reshape(8, 2, 3)
    first = runner.compact_rowwise_fingerprint(values)
    second = runner.compact_rowwise_fingerprint(values.copy())
    self.assertEqual(first, second)
    changed = values.copy()
    changed[2, 0, 0] += 1
    third = runner.compact_rowwise_fingerprint(changed)
    for row in range(8):
      equal = first['rows'][row]['sha256'] == third['rows'][row]['sha256']
      self.assertEqual(equal, row != 2)

  def test_original_binding_uses_manifest_hash_without_opening_json(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      relative = 'raw/identity/000_variant.json'
      path = root / relative
      path.parent.mkdir(parents=True)
      path.write_bytes(b'opaque scientific bytes')
      digest = runner._sha256(path)  # pylint: disable=protected-access
      case = SimpleNamespace(order=0, variant_id='variant')
      with mock.patch.object(runner, 'ORIGINAL_RUN_DIR', root):
        binding = runner.original_artifact_binding(
            {'artifact_sha256': {relative: digest}}, case, 'identity'
        )
        self.assertEqual(binding, {'path': relative, 'sha256': digest})
        path.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'changed'):
          runner.original_artifact_binding(
              {'artifact_sha256': {relative: digest}}, case, 'identity'
          )

  def test_runner_source_has_no_six_row_or_cube_apply_path(self):
    text = Path(runner.__file__).read_text(encoding='utf-8')
    self.assertNotIn(
        'create_splice_classification_logit_margin_superset_graph_apply(', text
    )
    self.assertNotIn('def _run_identity(', text)
    self.assertNotIn('def _run_coalition(', text)
    self.assertEqual(
        text.count(
            'create_splice_classification_logit_margin_eight_row_superset_graph_apply('
        ),
        1,
    )
    start_index = text.index('_write_new(START_PATH, start)')
    try_index = text.index('  try:', start_index)
    protobuf_index = text.index("'PROTOBUF_PROVENANCE.json'", start_index)
    self.assertLess(start_index, try_index)
    self.assertLess(try_index, protobuf_index)

  def test_apply_counter_increments_before_failed_dispatch(self):
    counter = [0]
    with mock.patch.object(
        runner.v32, '_timed_apply', side_effect=RuntimeError('dispatch failed')
    ):
      with self.assertRaisesRegex(RuntimeError, 'dispatch failed'):
        runner._counted_apply(object(), (), counter)  # pylint: disable=protected-access
    self.assertEqual(counter, [1])

  def test_four_completed_calls_persist_postprocessing_failure(self):
    selection = fixtures._selection()  # pylint: disable=protected-access
    trace = fixtures._trace(8)  # pylint: disable=protected-access
    evidence = fixtures._evidence((1, 3, 3, 3, 1, 1, 7, 9))  # pylint: disable=protected-access
    calls = [((evidence, trace), 0.1)] * 4
    call_iterator = iter(calls)
    def counted_apply(_compiled, _args, apply_counter):
      apply_counter[0] += 1
      return next(call_iterator)
    recipient = SimpleNamespace(order=0, variant_id='recipient')
    donor = SimpleNamespace(order=10, variant_id='donor')
    common = (
        np.zeros((6, 2, 4), np.float32), selection, object(), 'a' * 64
    )
    donor_common = (
        np.zeros((6, 2, 4), np.float32), selection, object(), 'b' * 64
    )
    signatures = {
        'selection': runner.v32.pytree_signature(selection),
        'target': None,
        'eight_interventions': runner.v32.pytree_signature(
            runner.v33.eight_row_interventions(
                selection, 0, unrelated=False
            )
        ),
    }
    with tempfile.TemporaryDirectory() as directory:
      counter = [0]
      with (
          mock.patch.object(runner, 'OUTPUT_DIR', Path(directory)),
          mock.patch.object(runner, '_counted_apply', side_effect=counted_apply),
          mock.patch.object(
              runner.v32, 'assert_same_program_signature', return_value=None
          ),
          mock.patch.object(
              runner, 'original_artifact_binding',
              side_effect=ValueError('linked artifact tampered'),
          ),
          mock.patch.object(runner, '_case_record', return_value={'order': 0}),
      ):
        result = runner._run_anchor(  # pylint: disable=protected-access
            object(), recipient, donor, common, donor_common,
            object(), object(), 0, signatures, 'f' * 64,
            'e' * 64, 0, {'artifact_sha256': {}}, counter,
        )
        self.assertEqual(result['status'], 'invalid')
        self.assertEqual(result['failure']['message'], 'linked artifact tampered')
        self.assertEqual(counter, [4])
        self.assertEqual(len(list(Path(directory).rglob('*.json'))), 1)

  def test_wrapper_rejects_caller_preflight_override_before_bootstrap(self):
    wrapper = Path(runner.__file__).with_name(
        'run_encoder_skip_ood_sidecar_v3_3_2.sh'
    )
    result = subprocess.run(
        ('bash', str(wrapper), '--dry-run', '--successful-preflight=/tmp/x'),
        text=True,
        capture_output=True,
        check=False,
    )
    self.assertEqual(result.returncode, 64)
    self.assertIn('reserved', result.stderr)

  def test_controlled_prefix_requires_exact_incremental_apply_count(self):
    result = {
        'status': 'invalid',
        'recipient_order': 0,
        'anchor_id': 0,
        'checks': None,
    }
    compiler = {
        'executable_fingerprint': 'e' * 64,
        'graph_and_hlo_exact_to_original_v3_3': True,
    }
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      raw = root / 'raw/ood_anchors/000_case/000.json'
      raw.parent.mkdir(parents=True)
      raw.write_text('{}\n', encoding='utf-8')
      for name in (
          'IMPORT_PROVENANCE_PRE_MODEL.json',
          'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
          'IMPORT_PROVENANCE.json',
          'PROTOBUF_PROVENANCE.json',
      ):
        (root / name).write_text('{}\n', encoding='utf-8')
      freeze = root / 'freeze.json'
      freeze.write_text('{}\n', encoding='utf-8')
      with (
          mock.patch.object(runner, 'OUTPUT_DIR', root),
          mock.patch.object(runner, 'FREEZE_PATH', freeze),
      ):
        with self.assertRaisesRegex(RuntimeError, 'four applies'):
          runner._write_completion(  # pylint: disable=protected-access
              stop_reason='ood_tooling_failure',
              message='test',
              results=[result],
              apply_count=3,
              compiler=compiler,
              original_run_binding={},
              v3_3_1_status={},
          )
        completion = runner._write_completion(  # pylint: disable=protected-access
            stop_reason='ood_tooling_failure',
            message='test',
            results=[result],
            apply_count=4,
            compiler=compiler,
            original_run_binding={},
            v3_3_1_status={},
        )
      self.assertEqual(completion['model_apply_count'], 4)
      self.assertEqual(completion['ood_anchor_record_count'], 1)
      self.assertEqual(completion['status'], 'controlled_stop')

  def test_post_start_failure_persists_exact_zero_apply_count(self):
    with tempfile.TemporaryDirectory() as directory:
      with mock.patch.object(runner, 'OUTPUT_DIR', Path(directory)):
        runner._write_terminal_failure(  # pylint: disable=protected-access
            RuntimeError('pre-model provenance failed'),
            completed_record_count=0,
            apply_count=0,
            compiler_created=False,
        )
      record = json.loads(
          (Path(directory) / 'TERMINAL_FAILURE.json').read_text(
              encoding='utf-8'
          )
      )
    self.assertEqual(record['model_apply_count'], 0)
    self.assertEqual(record['completed_record_count'], 0)
    self.assertEqual(record['eight_row_compile_count'], 0)
    self.assertEqual(record['confirmation_model_calls'], 0)

  def test_direct_main_requires_same_process_attestation(self):
    sys.modules.pop(runner.ATTESTATION_MODULE, None)
    with self.assertRaisesRegex(RuntimeError, 'launcher'):
      runner.consume_bootstrap_attestation()

  def test_direct_script_stops_before_jax_import(self):
    result = subprocess.run(
        (sys.executable, runner.__file__, '--dry-run'),
        text=True,
        capture_output=True,
        check=False,
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn('before pre-import bootstrap', result.stderr)
    self.assertNotIn('Jax plugin configuration error', result.stderr)


if __name__ == '__main__':
  unittest.main()
