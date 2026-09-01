#!/usr/bin/env python3
"""Tests for the spatial encoder-skip model runner."""

from __future__ import annotations

import dataclasses
from pathlib import Path
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import run_route_census_v3 as route_v3  # pylint: disable=g-import-not-at-top
import run_spatial_encoder_skip_experiment as runner  # pylint: disable=g-import-not-at-top
# pylint: enable=wrong-import-position


class SpatialEncoderSkipRunnerTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.plan, cls.plan_sha256 = runner.load_plan()
    cls.cases = route_v3.load_development_cases()
    cls.bindings = runner.bind_cases(cls.plan, cls.cases)

  def test_plan_binds_all_development_cases_without_confirmation(self):
    self.assertEqual(len(self.bindings), 20)
    self.assertFalse(self.plan['scope']['confirmation_access'])
    self.assertEqual(
        [case.variant_id for case, _ in self.bindings],
        [planned['variant_id'] for planned in self.plan['cases']],
    )

  def test_every_condition_has_one_fixed_selector_shape(self):
    for case, planned in self.bindings:
      for condition in planned['conditions']:
        selection = runner.spatial_selection(case, condition)
        positions = np.asarray(selection.decoder_skip_positions)
        valid = np.asarray(selection.decoder_skip_valid_mask)
        self.assertEqual(positions.shape, (7, 6))
        self.assertEqual(valid.shape, (7, 6))
        self.assertFalse(valid[0].any())
        self.assertFalse(valid[4].any())
        self.assertTrue(valid[1].any())
        self.assertTrue(valid[2].any())
        self.assertTrue(valid[3].any())
        self.assertTrue(valid[5].any())
        self.assertTrue(valid[6].any())

  def test_active_runtime_contract_is_only_positional_decoder_skips(self):
    case, planned = self.bindings[0]
    selection = runner.spatial_selection(case, planned['conditions'][0])
    interventions = runner.spatial_interventions(selection, active=True)
    runner.validate_runtime_contract(selection, interventions, active=True)
    expected = (
        np.asarray(selection.decoder_skip_valid_mask)[:, None, :]
        & runner.RECIPIENT_ROWS[None, :, None]
    )
    self.assertTrue(np.array_equal(
        np.asarray(interventions.decoder_skip_states.transfer_mask), expected
    ))
    self.assertFalse(
        np.asarray(interventions.encoder_outputs.transfer_mask).any()
    )
    self.assertFalse(
        np.asarray(interventions.decoder_outputs.transfer_mask).any()
    )
    self.assertFalse(
        np.asarray(interventions.final_embeddings.transfer_mask).any()
    )

  def test_identity_runtime_contract_is_all_false_with_same_shapes(self):
    case, planned = self.bindings[0]
    selection = runner.spatial_selection(case, planned['conditions'][0])
    active = runner.spatial_interventions(selection, active=True)
    identity = runner.spatial_interventions(selection, active=False)
    runner.validate_runtime_contract(selection, identity, active=False)
    self.assertEqual(
        active.decoder_skip_states.transfer_mask.shape,
        identity.decoder_skip_states.transfer_mask.shape,
    )
    self.assertFalse(
        np.asarray(identity.decoder_skip_states.transfer_mask).any()
    )

  def test_runtime_contract_rejects_a_non_spatial_route(self):
    case, planned = self.bindings[0]
    selection = runner.spatial_selection(case, planned['conditions'][0])
    interventions = runner.spatial_interventions(selection, active=True)
    bad_encoder = dataclasses.replace(
        interventions.encoder_outputs,
        transfer_mask=(
            interventions.encoder_outputs.transfer_mask.at[0, 2, 0].set(True)
        ),
    )
    interventions = dataclasses.replace(
        interventions, encoder_outputs=bad_encoder
    )
    with self.assertRaisesRegex(runner.SpatialRunError, 'encoder_outputs'):
      runner.validate_runtime_contract(selection, interventions, active=True)

  def test_recovery_uses_self_controls_in_both_directions(self):
    recovered = runner.recovery_from_means((4, 2, 3, 2, 3, 4))
    self.assertEqual(recovered['reference_into_alternate'], 0.5)
    self.assertEqual(recovered['alternate_into_reference'], 0.5)
    self.assertEqual(recovered['bidirectional_bottleneck'], 0.5)
    self.assertEqual(recovered['bidirectional_mean'], 0.5)

  def test_full_dry_run_is_the_frozen_520_apply_design(self):
    dry = runner.build_dry_run(self.plan, self.bindings, max_conditions=0)
    self.assertEqual(dry['variant_count'], 20)
    self.assertEqual(dry['condition_count_per_variant'], 12)
    self.assertEqual(dry['model_apply_count'], 520)
    self.assertEqual(dry['fixed_decoder_skip_shape'], [7, 6])
    self.assertFalse(dry['confirmation_access'])


if __name__ == '__main__':
  unittest.main()
