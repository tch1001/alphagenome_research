#!/usr/bin/env python3
"""Tests for the single-channel spatial sufficiency plan."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import prepare_single_channel_sufficiency as planner
# pylint: enable=wrong-import-position


class SingleChannelSufficiencyPlanTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.plan = planner.build_plan()

  def test_three_advancing_coordinates_are_frozen(self):
    self.assertEqual(
        [(value['selected_for_gene'], value['stage'],
          value['channel_start_inclusive'])
         for value in self.plan['channels']],
        [('BRAF', 'E16', 3), ('SLC25A48', 'E1', 175),
         ('SLC25A48', 'E2', 175)],
    )
    self.assertFalse(self.plan['scope']['confirmation_access'])

  def test_each_channel_has_three_locations(self):
    self.assertEqual(len(self.plan['conditions']), 9)
    for channel in self.plan['channels']:
      conditions = [
          value for value in self.plan['conditions']
          if value['feature']['source_condition_id']
          == channel['source_condition_id']
      ]
      self.assertEqual(
          [value['location'] for value in conditions],
          list(planner.LOCATIONS),
      )

  def test_full_design_has_260_applies(self):
    self.assertEqual(self.plan['design']['planned_model_apply_count'], 260)


if __name__ == '__main__':
  unittest.main()
