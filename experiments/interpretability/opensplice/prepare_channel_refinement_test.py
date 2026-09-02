#!/usr/bin/env python3
"""Tests for the deterministic 8-channel refinement plan."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import prepare_channel_refinement as planner
# pylint: enable=wrong-import-position


class ChannelRefinementPlanTest(unittest.TestCase):

  def test_parent_selection_is_frozen_from_analysis(self):
    plan = planner.build_plan()
    self.assertEqual([parent['group_id'] for parent in plan['parents']], [
        'E1_c0160_0191',
        'E32_c0000_0031',
        'E16_c0000_0031',
        'E16_c0512_0543',
        'E2_c0160_0191',
    ])
    self.assertFalse(plan['scope']['confirmation_access'])

  def test_children_exactly_partition_each_parent(self):
    plan = planner.build_plan()
    self.assertEqual(len(plan['groups']), 20)
    for parent in plan['parents']:
      children = [
          child for child in plan['groups']
          if child['parent_group_id'] == parent['group_id']
      ]
      self.assertEqual(len(children), 4)
      self.assertTrue(all(child['channel_count'] == 8 for child in children))
      covered = [
          channel
          for child in children
          for channel in range(
              child['channel_start_inclusive'],
              child['channel_end_exclusive'],
          )
      ]
      self.assertEqual(covered, list(range(
          parent['channel_start_inclusive'],
          parent['channel_end_exclusive'],
      )))

  def test_apply_count_is_lean(self):
    plan = planner.build_plan()
    self.assertEqual(plan['design']['planned_model_apply_count'], 480)


if __name__ == '__main__':
  unittest.main()
