"""CPU tests for the OpenSplice v3 Phase-R residual-grid runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


_MODULE_PATH = Path(__file__).with_name('run_phase_r_v3.py')
_SPEC = importlib.util.spec_from_file_location('run_phase_r_v3', _MODULE_PATH)
runner = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)


class PhaseRV3Test(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.cases = runner.route_v3.load_development_cases()
    self.case = self.cases[4]
    self.interval = runner.v2.centered_interval(
        self.case, runner.route_v3.CONTEXT_BP
    )

  def test_grid_is_exact_v2_72_candidates_and_216_controlled_groups(self):
    groups = runner.enumerate_groups(self.case, self.interval)
    self.assertEqual(len(groups), 216)
    self.assertEqual(sum(group.is_candidate for group in groups), 72)
    self.assertEqual({group.layer for group in groups}, set(range(6)))
    self.assertEqual({group.stage for group in groups}, set(runner.STAGES))
    self.assertNotIn(6, {group.layer for group in groups})
    self.assertEqual(
        tuple(group.position_set.name for group in groups[:12]),
        runner.POSITION_SET_NAMES,
    )
    self.assertEqual(
        [(group.stage, group.layer) for group in groups[::12]],
        [
            (stage, layer)
            for stage in runner.STAGES
            for layer in runner.LAYERS
        ],
    )

  def test_position_controls_preserve_cardinality_offsets_and_nonoverlap(self):
    position_sets = runner.v2.trace_position_sets(self.case, self.interval)
    by_name = {position_set.name: position_set for position_set in position_sets}
    for candidate_name in runner.CANDIDATE_POSITION_SETS:
      candidate = by_name[candidate_name]
      for direction in ('upstream', 'downstream'):
        control = by_name[f'{candidate_name}_control_{direction}']
        self.assertEqual(len(candidate.tokens), len(control.tokens))
        offsets = tuple(
            observed - expected
            for observed, expected in zip(
                control.tokens, candidate.tokens, strict=True
            )
        )
        self.assertEqual(len(set(offsets)), 1)
        self.assertGreaterEqual(
            abs(offsets[0]), runner.v2.CONTROL_START_DISTANCE_TOKENS
        )
        self.assertFalse(set(control.tokens) & set(by_name['S'].tokens))

  def test_trace_selection_reuses_v2_slots_and_disables_every_route_slot(self):
    position_sets = runner.v2.trace_position_sets(self.case, self.interval)
    expected = runner.v2.transformer_trace_selection(position_sets)
    observed = runner.phase_r_trace_selection(position_sets)
    np.testing.assert_array_equal(
        observed.transformer.residual_positions.positions,
        expected.residual_positions.positions,
    )
    np.testing.assert_array_equal(
        observed.transformer.residual_positions.valid_mask,
        expected.residual_positions.valid_mask,
    )
    for mask in (
        observed.encoder_valid_mask,
        observed.decoder_skip_valid_mask,
        observed.decoder_output_valid_mask,
        observed.final_embedding_valid_mask,
    ):
      self.assertFalse(np.asarray(mask).any())

  def test_one_group_activates_only_requested_transformer_slots(self):
    position_sets = runner.v2.trace_position_sets(self.case, self.interval)
    selection = runner.phase_r_trace_selection(position_sets)
    group = next(
        group
        for group in runner.enumerate_groups(self.case, self.interval)
        if group.stage == 'post_attention'
        and group.layer == 5
        and group.position_set.name == 'S'
    )
    interventions = runner.group_interventions(selection, group)
    active = interventions.transformer.post_attention_residual_transfer
    expected = np.zeros((9, 6, runner.v2.NUM_TRACE_SLOTS), bool)
    expected[5, 2:, np.asarray(group.position_set.slots)] = True
    np.testing.assert_array_equal(active.transfer_mask, expected)
    self.assertFalse(np.asarray(
        interventions.transformer.pre_attention_residual_transfer.transfer_mask
    ).any())
    self.assertFalse(np.asarray(
        interventions.transformer.post_mlp_residual_transfer.transfer_mask
    ).any())
    for transfer in (
        interventions.encoder_outputs,
        interventions.decoder_skip_states,
        interventions.decoder_outputs,
        interventions.final_embeddings,
    ):
      self.assertFalse(np.asarray(transfer.transfer_mask).any())

  def test_identity_has_fixed_pytree_with_every_transfer_false(self):
    position_sets = runner.v2.trace_position_sets(self.case, self.interval)
    selection = runner.phase_r_trace_selection(position_sets)
    identity = runner.group_interventions(selection, None)
    for transfer in (
        identity.transformer.pre_attention_residual_transfer,
        identity.transformer.post_attention_residual_transfer,
        identity.transformer.post_mlp_residual_transfer,
        identity.encoder_outputs,
        identity.decoder_skip_states,
        identity.decoder_outputs,
        identity.final_embeddings,
    ):
      self.assertFalse(np.asarray(transfer.transfer_mask).any())

  def test_dry_plan_is_development_only_and_contains_no_route_census(self):
    plan = runner.build_dry_run_plan(self.cases, max_groups=1)
    self.assertEqual(plan['variant_count'], 20)
    self.assertEqual(plan['full_candidate_count'], 72)
    self.assertEqual(plan['full_group_count_per_eligible_effect'], 216)
    self.assertEqual(plan['bounded_group_count_per_eligible_effect'], 1)
    self.assertEqual(plan['non_transformer_route_components'], 0)
    self.assertEqual(plan['layers'], tuple(range(6)))
    encoded = str(plan)
    for confirmation_gene in ('ELN', 'EIF4A2', 'DMD'):
      self.assertNotIn(confirmation_gene, encoded)
    for forbidden in ('encoder_outputs', 'decoder_outputs', 'layer6'):
      self.assertNotIn(forbidden, encoded)

  def test_active_slots_include_all_joint_S_positions(self):
    group = next(
        group
        for group in runner.enumerate_groups(self.case, self.interval)
        if group.stage == 'pre_attention'
        and group.layer == 0
        and group.position_set.name == 'S'
    )
    mask = runner._active_slot_mask(group)  # pylint: disable=protected-access
    self.assertEqual(int(mask.sum()), len(group.position_set.slots))
    self.assertTrue(mask[np.asarray(group.position_set.slots)].all())

  def test_resume_fails_closed_on_configuration_mismatch(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'group.json'
      runner.v2._write_atomic(path, {  # pylint: disable=protected-access
          'status': 'complete', 'fingerprint': 'one'
      })
      with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
        runner.v2._load_completed(path, 'two')  # pylint: disable=protected-access


if __name__ == '__main__':
  unittest.main()
