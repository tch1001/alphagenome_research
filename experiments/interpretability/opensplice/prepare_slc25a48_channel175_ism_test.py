#!/usr/bin/env python3
"""Tests for the SLC25A48 channel-175 ISM plan."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import prepare_slc25a48_channel175_ism as planner
# pylint: enable=wrong-import-position


class Slc25a48Channel175IsmPlanTest(unittest.TestCase):

  def test_frozen_design(self):
    plan = planner.build_plan()
    self.assertFalse(plan['scope']['confirmation_access'])
    self.assertEqual(plan['design']['candidate_edit_count'], 123)
    self.assertEqual(plan['design']['batch_count'], 25)
    self.assertEqual(plan['design']['planned_model_apply_count'], 50)
    self.assertEqual(plan['design']['traced_channel'], 175)


if __name__ == '__main__':
  unittest.main()
