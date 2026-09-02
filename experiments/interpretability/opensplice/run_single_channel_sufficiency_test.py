#!/usr/bin/env python3
"""Tests for the single-channel spatial sufficiency runner."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import run_single_channel_sufficiency as runner
# pylint: enable=wrong-import-position


class SingleChannelSufficiencyRunnerTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.plan, _ = runner.load_plan()
    cls.bindings = runner.screen.bind_cases(
        cls.plan, runner.route_v3.load_development_cases()
    )

  def test_complete_dry_run(self):
    dry = runner.build_dry_run(self.bindings, self.plan['conditions'])
    self.assertEqual(dry['variant_count'], 20)
    self.assertEqual(dry['selected_channel_count'], 3)
    self.assertEqual(dry['condition_count'], 9)
    self.assertEqual(dry['model_apply_count'], 260)
    self.assertFalse(dry['confirmation_access'])

  def test_every_condition_selects_exactly_one_channel(self):
    for condition in self.plan['conditions']:
      self.assertEqual(int(runner.single_channel_mask(condition).sum()), 1)


if __name__ == '__main__':
  unittest.main()
