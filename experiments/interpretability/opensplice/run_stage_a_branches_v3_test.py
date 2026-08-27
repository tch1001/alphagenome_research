"""CPU tests for the Stage-A closure and whole-branch runner."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import jax.numpy as jnp
import numpy as np
import pandas as pd


_MODULE_PATH = Path(__file__).with_name('run_stage_a_branches_v3.py')
_SPEC = importlib.util.spec_from_file_location(
    'run_stage_a_branches_v3', _MODULE_PATH
)
runner = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)


def _target(values=(1, 2, 2, 2, 1, 1)):
  mean = jnp.asarray(values, jnp.float32)
  return runner.interpretability.TargetSummary(
      total=mean * 2,
      mean=mean,
      num_values=jnp.asarray(2, jnp.int32),
  )


def _trace(
    *,
    final_effective_rows=(1, 2, 2, 2, 1, 1),
    active_transformer=False,
    active_skips=False,
):
  natural_rows = jnp.asarray([1, 2, 2, 2, 1, 1], jnp.float32)
  effective_rows = jnp.asarray(final_effective_rows, jnp.float32)
  natural = jnp.broadcast_to(natural_rows[:, None, None], (6, 2, 3))
  effective = jnp.broadcast_to(effective_rows[:, None, None], (6, 2, 3))
  effective_matches_natural = jnp.asarray(
      [True, True, False, True, False, True]
      if active_transformer or active_skips else [True] * 6,
      jnp.bool,
  )
  inactive_intervention_donor_matches = jnp.asarray(
      [True, True, False, True, False, True], jnp.bool
  )
  fingerprints = jnp.asarray([
      [11, 12, 13, 14],
      [21, 22, 23, 24],
      [21, 22, 23, 24],
      [21, 22, 23, 24],
      [11, 12, 13, 14],
      [11, 12, 13, 14],
  ], jnp.uint32)
  return runner.interpretability.StageABranchTrace(
      transformer_output_natural_matches_identity=jnp.ones((6,), jnp.bool),
      transformer_output_effective_matches_natural=(
          effective_matches_natural
          if active_transformer else jnp.ones((6,), jnp.bool)
      ),
      transformer_output_effective_matches_intervention_donor=(
          jnp.ones((6,), jnp.bool)
          if active_transformer else inactive_intervention_donor_matches
      ),
      transformer_output_natural_fingerprint=fingerprints,
      encoder_skips_natural_match_identity=jnp.ones((7, 6), jnp.bool),
      encoder_skips_effective_match_natural=jnp.broadcast_to(
          effective_matches_natural
          if active_skips else jnp.ones((6,), jnp.bool), (7, 6)
      ),
      encoder_skips_effective_match_intervention_donor=jnp.broadcast_to(
          jnp.ones((6,), jnp.bool)
          if active_skips else inactive_intervention_donor_matches, (7, 6)
      ),
      encoder_skips_natural_fingerprints=jnp.broadcast_to(
          fingerprints[None, ...], (7, 6, 4)
      ),
      natural_final_embeddings=natural,
      effective_final_embeddings=effective,
  )


def _record(values):
  return {
      'fingerprint': 'test',
      'checks': {
          'target_means': dict(
              zip(runner.route_v3.TRACE_BATCH_ROLES, values, strict=True)
          )
      },
  }


class StageABranchesV3Test(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.cases = runner.route_v3.load_development_cases()
    self.case = self.cases[0]
    interval = runner.v2.centered_interval(
        self.case, runner.route_v3.CONTEXT_BP
    )
    metadata = pd.DataFrame({
        'name': ['donor', 'acceptor', 'donor', 'acceptor', 'padding'],
        'strand': ['+', '+', '-', '-', '.'],
    })
    _, self.resolved = runner.route_v3.target_selection(
        metadata, self.case, interval
    )
    self.selection = runner.branch_selection(self.resolved)

  def test_component_order_puts_both_closures_before_isolated_branches(self):
    components = runner.enumerate_components()
    self.assertEqual(
        tuple(component.name for component in components), runner.COMPONENTS
    )
    self.assertTrue(
        all(component.closure_required for component in components[:2])
    )
    self.assertFalse(
        any(component.closure_required for component in components[2:])
    )

  def test_mandatory_closures_do_not_use_isolated_branch_eligibility(self):
    identities = {
        case.variant_id: {
            'direction_gate': {
                'eligible_for_causal_census': case.is_effect
            }
        }
        for case in self.cases
    }
    first_effect = next(case for case in self.cases if case.is_effect)
    identities[first_effect.variant_id]['direction_gate'][
        'eligible_for_causal_census'
    ] = False
    closure_ids = {
        case.variant_id for case in runner.mandatory_closure_cases(self.cases)
    }
    branch_ids = {
        case.variant_id
        for case in runner.isolated_branch_cases(self.cases, identities)
    }
    self.assertIn(first_effect.variant_id, closure_ids)
    self.assertNotIn(first_effect.variant_id, branch_ids)
    self.assertEqual(len(closure_ids), 12)

  def test_negative_strand_selection_is_exact_acceptor_and_donor(self):
    self.assertEqual(
        [endpoint.role for endpoint in self.resolved.endpoints],
        ['acceptor', 'donor'],
    )
    np.testing.assert_array_equal(
        self.selection.final_embedding_positions,
        [endpoint.position_index for endpoint in self.resolved.endpoints],
    )
    self.assertTrue(np.asarray(self.selection.final_embedding_valid_mask).all())

  def test_component_masks_are_branch_isolated_and_use_frozen_donors(self):
    by_name = {
        component.name: component
        for component in runner.enumerate_components()
    }
    for name, expected in {
        'final_embedding_A_D_closure': (False, False, True),
        'joint_T_plus_E_closure': (True, True, False),
        'whole_T': (True, False, False),
        'whole_E': (False, True, False),
    }.items():
      interventions = runner.component_interventions(
          self.selection, by_name[name]
      )
      observed = (
          bool(np.asarray(
              interventions.transformer_output.transfer_mask
          ).any()),
          bool(np.asarray(interventions.encoder_skips.transfer_mask).any()),
          bool(np.asarray(interventions.final_embedding.transfer_mask).any()),
      )
      self.assertEqual(observed, expected)
      np.testing.assert_array_equal(
          interventions.transformer_output.donor_batch_indices[0],
          runner.route_v3.TRACE_BATCH_DONORS,
      )
      np.testing.assert_array_equal(
          interventions.transformer_output.natural_identity_batch_indices[0],
          [0, 1, 1, 1, 0, 0],
      )
      self.assertFalse(np.asarray(
          interventions.transformer_output.transfer_mask[:, :2]
      ).any())
      self.assertFalse(np.asarray(
          interventions.encoder_skips.transfer_mask[:, :2]
      ).any())
      self.assertFalse(np.asarray(
          interventions.final_embedding.transfer_mask[:, :2, :]
      ).any())

  def test_identity_requires_exact_repeat_duplicates_and_noop(self):
    target = _target()
    trace = _trace()
    result = runner.validate_identity_audit(target, trace, target, trace)
    self.assertTrue(result['passed'])
    self.assertTrue(result['natural_T_fingerprint_repeat_exact'])
    bad = dataclasses.replace(
        trace,
        effective_final_embeddings=trace.effective_final_embeddings.at[
            2, 0, 0
        ].set(9),
    )
    with self.assertRaisesRegex(ValueError, 'duplicate trace|no-op'):
      runner.validate_identity_audit(target, bad, target, bad)
    bad_repeat = dataclasses.replace(
        trace,
        transformer_output_natural_fingerprint=(
            trace.transformer_output_natural_fingerprint.at[0, 0].add(1)
        ),
    )
    with self.assertRaisesRegex(ValueError, 'repeat failed'):
      runner.validate_identity_audit(target, trace, target, bad_repeat)

  def test_final_embedding_and_joint_branch_closures_are_bit_exact(self):
    identity = (1, 2, 2, 2, 1, 1)
    closed_target = _target((1, 2, 1, 2, 2, 1))
    final_component, joint_component = runner.enumerate_components()[:2]
    final_trace = _trace(final_effective_rows=(1, 2, 1, 2, 2, 1))
    final = runner.validate_component_audit(
        closed_target, final_trace, final_component, identity
    )
    joint = runner.validate_component_audit(
        closed_target,
        _trace(active_transformer=True, active_skips=True),
        joint_component,
        identity,
    )
    self.assertTrue(final['closure']['passed'])
    self.assertTrue(joint['closure']['passed'])
    bad_target = _target((1, 2, 1.5, 2, 2, 1))
    with self.assertRaisesRegex(ValueError, 'closure failed'):
      runner.validate_component_audit(
          bad_target,
          _trace(active_transformer=True, active_skips=True),
          joint_component,
          identity,
      )

  def test_live_whole_transfer_requires_natural_self_and_effective_donor(self):
    component = runner.enumerate_components()[2]
    trace = _trace(active_transformer=True)
    result = runner.validate_component_audit(
        _target((1, 2, 1.5, 2, 1.5, 1)),
        trace,
        component,
        (1, 2, 2, 2, 1, 1),
    )
    self.assertTrue(result['transformer_natural_self_tensors_exact'])
    self.assertTrue(result['transformer_effective_donor_tensors_exact'])
    bad = dataclasses.replace(
        trace,
        transformer_output_natural_matches_identity=(
            trace.transformer_output_natural_matches_identity.at[3].set(False)
        ),
    )
    with self.assertRaisesRegex(ValueError, 'natural self'):
      runner.validate_component_audit(
          _target((1, 2, 1.5, 2, 1.5, 1)),
          bad,
          component,
          (1, 2, 2, 2, 1, 1),
      )

  def test_locked_phase_r_identity_tree_and_cross_graph_gate(self):
    records = runner.load_locked_phase_r_identities(self.cases)
    self.assertEqual(len(records), 20)
    record = records[self.case.variant_id]
    self.assertEqual(
        record['configuration']['phase_runner_sha256'],
        runner.v2._sha256(  # pylint: disable=protected-access
            runner.PHASE_R_RUNNER_PATH
        ),
    )
    self.assertEqual(
        record['configuration']['v2_runner_sha256'],
        runner.v2._sha256(  # pylint: disable=protected-access
            runner.V2_RUNNER_PATH
        ),
    )
    means = [
        record['checks']['target_means'][role]
        for role in runner.route_v3.TRACE_BATCH_ROLES
    ]
    result = runner.validate_locked_phase_r_identity(
        self.case, means, record
    )
    self.assertTrue(result['passed'])
    drifted = list(means)
    drifted[0] += runner.CROSS_EXECUTABLE_TARGET_TOLERANCE * 2
    with self.assertRaisesRegex(ValueError, 'differs from its locked'):
      runner.validate_locked_phase_r_identity(self.case, drifted, record)

  def test_current_phase_r_case_inputs_link_exactly_to_lock(self):
    record = runner.load_locked_phase_r_identities(self.cases)[
        self.case.variant_id
    ]
    interval = runner.v2.centered_interval(
        self.case, runner.route_v3.CONTEXT_BP
    )
    position_sets = runner.v2.trace_position_sets(self.case, interval)
    linkage = runner.validate_locked_phase_r_linkage(
        self.case,
        interval,
        self.resolved,
        record['configuration']['sequence_sha256'],
        position_sets,
        record,
    )
    self.assertTrue(linkage['passed'])
    with self.assertRaisesRegex(ValueError, 'resolved_position_sets'):
      runner.validate_locked_phase_r_linkage(
          self.case,
          interval,
          self.resolved,
          record['configuration']['sequence_sha256'],
          position_sets[:-1],
          record,
      )

  def test_frozen_configuration_binds_final_amendment_bytes(self):
    record = runner.load_locked_phase_r_identities(self.cases)[
        self.case.variant_id
    ]
    frozen = runner.frozen_configuration(Path(
        record['configuration']['checkpoint_path']
    ))
    self.assertEqual(
        frozen['dual_reference_amendment']['sha256'],
        runner.v2._sha256(  # pylint: disable=protected-access
            runner.DUAL_REFERENCE_AMENDMENT_PATH
        ),
    )
    self.assertEqual(
        frozen['phase_r_runner_sha256'],
        record['configuration']['phase_runner_sha256'],
    )
    self.assertEqual(
        frozen['v2_runner_sha256'],
        record['configuration']['v2_runner_sha256'],
    )

  def test_dual_reference_records_drift_without_relaxing_locked_gate(self):
    reference = (2.546875, 3.5869140625, 3.5869140625,
                 3.5869140625, 2.546875, 2.546875)
    stage = list(reference)
    for row in (1, 2, 3):
      stage[row] += 0.0048828125
    diagnostic = runner.compare_stage_a_to_phase_r_reference(stage, reference)
    self.assertTrue(diagnostic['diagnostic_only_not_a_gate'])
    self.assertEqual(
        diagnostic['maximum_absolute_difference'], 0.0048828125
    )
    records = runner.load_locked_phase_r_identities(self.cases)
    locked = records['BRAF_e14_T71A']
    runner.validate_locked_phase_r_identity(
        self.cases[1], reference, locked
    )
    bad_reference = list(reference)
    bad_reference[1] += 0.0048828125
    with self.assertRaisesRegex(ValueError, 'differs from its locked'):
      runner.validate_locked_phase_r_identity(
          self.cases[1], bad_reference, locked
      )

  def test_shapley_uses_raw_empty_T_E_and_joint_targets(self):
    identity = _record((1, 3, 3, 3, 1, 1))
    results = {
        'whole_T': _record((1, 3, 2.5, 3, 1.5, 1)),
        'whole_E': _record((1, 3, 2.0, 3, 2.0, 1)),
        'joint_T_plus_E_closure': _record((1, 3, 1.0, 3, 3.0, 1)),
    }
    partition = runner.shapley_partition(identity, results)
    forward = partition['reference_into_alternate']
    self.assertAlmostEqual(forward['raw_phi_T'], -0.75)
    self.assertAlmostEqual(forward['raw_phi_E'], -1.25)
    self.assertAlmostEqual(forward['raw_interaction'], -0.5)
    self.assertAlmostEqual(forward['normalized_phi_T'], 0.375)
    self.assertAlmostEqual(forward['normalized_phi_E'], 0.625)

  def test_dry_plan_is_development_only_and_declares_remaining_slice(self):
    plan = runner.build_dry_run_plan(self.cases, runner.enumerate_components())
    self.assertEqual(plan['variant_count'], 20)
    self.assertEqual(plan['effect_count'], 12)
    self.assertEqual(plan['component_calls_per_eligible_effect'], 4)
    self.assertEqual(plan['mandatory_closure_calls_per_effect'], 2)
    self.assertEqual(plan['isolated_branch_calls_per_eligible_effect'], 2)
    self.assertEqual(plan['stage_a_identity_calls_per_variant'], 2)
    self.assertEqual(plan['phase_r_semantic_reference_calls_per_variant'], 2)
    self.assertEqual(
        plan['execution_order'][:2],
        (
            'all_20_current_phase_r_references',
            'all_20_stage_a_identities',
        ),
    )
    self.assertEqual(
        plan['dual_reference_amendment_sha256'],
        runner.v2._sha256(  # pylint: disable=protected-access
            runner.DUAL_REFERENCE_AMENDMENT_PATH
        ),
    )
    self.assertEqual(plan['confirmation_access'], 'disabled')
    self.assertIn('receptive_field', plan['remaining_stage_a_work'])
    encoded = str(plan)
    for confirmation_gene in ('ELN', 'EIF4A2', 'DMD'):
      self.assertNotIn(confirmation_gene, encoded)

  def test_resume_fails_closed_on_configuration_mismatch(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'component.json'
      runner.v2._write_atomic(path, {  # pylint: disable=protected-access
          'status': 'complete', 'fingerprint': 'expected'
      })
      with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
        runner.v2._load_completed(  # pylint: disable=protected-access
            path, 'changed'
        )


if __name__ == '__main__':
  unittest.main()
