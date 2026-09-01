#!/usr/bin/env python3
"""Tests for the spatial encoder-skip experiment planner."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import prepare_spatial_encoder_skip_experiment as planner  # pylint: disable=g-import-not-at-top
# pylint: enable=wrong-import-position


class SpatialEncoderSkipPlanTest(unittest.TestCase):

  def test_guarded_support_adds_one_token_each_side(self):
    self.assertEqual(
        planner.guarded_support(
            [513], interval_start=0, resolution=8
        ),
        (63, 64, 65),
    )
    self.assertEqual(
        planner.guarded_support(
            [513, 521], interval_start=0, resolution=8
        ),
        (63, 64, 65, 66),
    )

  def test_guarded_support_preserves_disjoint_components(self):
    self.assertEqual(
        planner.guarded_support(
            [513, 593], interval_start=0, resolution=8
        ),
        (63, 64, 65, 73, 74, 75),
    )

  def test_translated_controls_are_equal_shape_distant_and_disjoint(self):
    support = (1000, 1001, 1002)
    forbidden = set(range(990, 1011))
    upstream = planner.translated_control(
        support, direction=-1, resolution=2, forbidden=forbidden
    )
    downstream = planner.translated_control(
        support, direction=1, resolution=2, forbidden=forbidden
    )
    self.assertEqual(len(upstream), len(support))
    self.assertEqual(len(downstream), len(support))
    self.assertEqual(
        {a - b for a, b in zip(support, upstream)}, {256}
    )
    self.assertEqual(
        {a - b for a, b in zip(downstream, support)}, {256}
    )
    self.assertFalse(set(upstream) & forbidden)
    self.assertFalse(set(downstream) & forbidden)

  def test_real_plan_is_exact_development_only_520_apply_design(self):
    plan = planner.build_plan()
    self.assertEqual(plan['scope'], {
        'development_only': True,
        'genes': ['BRAF', 'SLC25A48'],
        'variant_count': 20,
        'effect_count': 12,
        'neutral_count': 8,
        'confirmation_access': False,
    })
    design = plan['model_behavior_design']
    self.assertEqual(design['candidate_mask'], 110)
    self.assertEqual(
        design['candidate_players'], ['E32', 'E16', 'E8', 'E2', 'E1']
    )
    self.assertEqual(design['maximum_position_slots'], 6)
    self.assertEqual(design['condition_count_per_variant'], 12)
    self.assertEqual(design['planned_model_apply_count'], 520)
    self.assertEqual(design['planned_compile_count'], 1)
    self.assertFalse(plan['controls']['os_kernel_is_a_gate'])
    self.assertEqual(len(plan['cases']), 20)

  def test_real_conditions_have_fixed_stage_order_and_equal_shape_controls(
      self
  ):
    plan = planner.build_plan()
    stage_names = [stage[0] for stage in planner.STAGES]
    for case in plan['cases']:
      by_id = {
          condition['condition_id']: condition
          for condition in case['conditions']
      }
      self.assertEqual(len(by_id), 12)
      for support in planner.SUPPORT_NAMES:
        intended = by_id[f'{support}_intended']
        for location in ('upstream', 'downstream'):
          control = by_id[f'{support}_{location}']
          self.assertEqual(
              [stage['player'] for stage in control['stages']], stage_names
          )
          for intended_stage, control_stage in zip(
              intended['stages'], control['stages'], strict=True
          ):
            self.assertEqual(
                len(intended_stage['positions']),
                len(control_stage['positions']),
            )
            if intended_stage['enabled']:
              offsets = {
                  control_position - intended_position
                  for control_position, intended_position in zip(
                      control_stage['positions'], intended_stage['positions'],
                      strict=True,
                  )
              }
              self.assertEqual(len(offsets), 1)
              minimum = (
                  planner.MINIMUM_CONTROL_DISTANCE_BP
                  // intended_stage['resolution_bp']
              )
              self.assertGreaterEqual(abs(next(iter(offsets))), minimum)
            else:
              self.assertEqual(control_stage['positions'], [])


if __name__ == '__main__':
  unittest.main()
