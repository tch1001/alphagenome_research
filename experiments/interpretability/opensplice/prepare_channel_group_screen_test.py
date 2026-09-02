#!/usr/bin/env python3
"""Tests for the V-local grouped-channel screen plan."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import prepare_channel_group_screen as planner
# pylint: enable=wrong-import-position


class ChannelGroupScreenPlanTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.plan = planner.build_plan()

  def test_groups_partition_all_five_candidate_stage_widths(self):
    groups = self.plan['groups']
    self.assertEqual(len(groups), 172)
    self.assertEqual(sum(group['channel_count'] for group in groups), 5504)
    for stage, _, width, enabled in planner.STAGES:
      stage_groups = [group for group in groups if group['stage'] == stage]
      if not enabled:
        self.assertEqual(stage_groups, [])
        continue
      self.assertEqual(stage_groups[0]['channel_start_inclusive'], 0)
      self.assertEqual(stage_groups[-1]['channel_end_exclusive'], width)
      for left, right in zip(stage_groups, stage_groups[1:]):
        self.assertEqual(
            left['channel_end_exclusive'], right['channel_start_inclusive']
        )

  def test_plan_is_development_only_and_lean(self):
    self.assertEqual(self.plan['scope']['variant_count'], 20)
    self.assertEqual(self.plan['scope']['effect_count'], 12)
    self.assertEqual(self.plan['scope']['neutral_count'], 8)
    self.assertFalse(self.plan['scope']['confirmation_access'])
    self.assertEqual(self.plan['design']['planned_model_apply_count'], 3520)
    self.assertEqual(self.plan['design']['planned_compile_count'], 1)
    self.assertFalse(self.plan['controls']['os_kernel_is_a_gate'])

  def test_case_stage_bindings_are_fixed_shape_v_local_supports(self):
    self.assertEqual(len(self.plan['cases']), 20)
    expected_enabled = [False, True, True, True, False, True, True]
    for case in self.plan['cases']:
      self.assertEqual(len(case['stages']), 7)
      self.assertEqual(
          [stage['route_enabled'] for stage in case['stages']],
          expected_enabled,
      )
      for stage in case['stages']:
        if stage['route_enabled']:
          self.assertEqual(len(stage['positions']), 3)
        else:
          self.assertEqual(stage['positions'], [])

  def test_group_ids_and_indices_are_unique(self):
    groups = self.plan['groups']
    self.assertEqual(
        [group['group_index'] for group in groups], list(range(172))
    )
    self.assertEqual(len({group['group_id'] for group in groups}), 172)


if __name__ == '__main__':
  unittest.main()
