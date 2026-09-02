#!/usr/bin/env python3
"""Tests for the individual-channel causal validation plan."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import prepare_individual_channel_validation as planner
# pylint: enable=wrong-import-position


class IndividualChannelValidationPlanTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.plan = planner.build_plan()

  def test_selected_children_are_frozen(self):
    self.assertEqual(
        [child['group_id'] for child in self.plan['selected_children']],
        list(planner.SELECTED_CHILD_IDS),
    )
    self.assertFalse(self.plan['scope']['confirmation_access'])

  def test_condition_partition_and_apply_count(self):
    necessity = [
        value for value in self.plan['conditions']
        if value['kind'] == 'individual_channel_necessity'
    ]
    sufficiency = [
        value for value in self.plan['conditions']
        if value['kind'] == 'eight_channel_only_sufficiency'
    ]
    self.assertEqual(len(necessity), 32)
    self.assertEqual(len(sufficiency), 12)
    self.assertEqual(
        self.plan['design']['planned_model_apply_count'], 960
    )

  def test_every_child_has_eight_channels_and_three_sufficiency_locations(self):
    for child_id in planner.SELECTED_CHILD_IDS:
      necessity = [
          value for value in self.plan['conditions']
          if value['kind'] == 'individual_channel_necessity'
          and value['feature']['parent_child_id'] == child_id
      ]
      sufficiency = [
          value for value in self.plan['conditions']
          if value['kind'] == 'eight_channel_only_sufficiency'
          and value['feature']['parent_child_id'] == child_id
      ]
      self.assertEqual(len(necessity), 8)
      self.assertTrue(all(
          value['feature']['channel_count'] == 1 for value in necessity
      ))
      self.assertEqual(
          [value['location'] for value in sufficiency],
          list(planner.LOCATIONS),
      )

  def test_shifted_positions_are_equal_shape_and_distinct(self):
    for case in self.plan['cases']:
      positions = case['positions_by_location']
      for stage_index in (1, 2, 3, 5, 6):
        intended = positions['intended'][stage_index]['positions']
        upstream = positions['upstream'][stage_index]['positions']
        downstream = positions['downstream'][stage_index]['positions']
        self.assertEqual(len(intended), len(upstream))
        self.assertEqual(len(intended), len(downstream))
        self.assertNotEqual(intended, upstream)
        self.assertNotEqual(intended, downstream)


if __name__ == '__main__':
  unittest.main()
