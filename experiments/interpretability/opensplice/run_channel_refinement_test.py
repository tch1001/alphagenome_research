#!/usr/bin/env python3
"""Tests for the 8-channel refinement runner."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import run_channel_refinement as runner
# pylint: enable=wrong-import-position


class ChannelRefinementRunnerTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.plan, _ = runner.load_plan()
    cls.bindings = runner.screen.bind_cases(
        cls.plan, runner.route_v3.load_development_cases()
    )

  def test_complete_dry_run(self):
    dry = runner.build_dry_run(self.bindings, self.plan['groups'])
    self.assertEqual(dry['variant_count'], 20)
    self.assertEqual(dry['parent_group_count'], 5)
    self.assertEqual(dry['group_count'], 20)
    self.assertEqual(dry['model_apply_count'], 480)
    self.assertFalse(dry['confirmation_access'])

  def test_each_condition_withholds_exactly_eight_candidate_channels(self):
    full = runner.screen.channel_mask()
    for group in self.plan['groups']:
      child = runner.screen.channel_mask(group)
      self.assertEqual(int(full.sum() - child.sum()), 8)
      changed = np.argwhere(full != child)
      self.assertEqual(set(changed[:, 0]), {group['stage_index']})
      self.assertEqual(
          changed[:, 1].tolist(),
          list(range(
              group['channel_start_inclusive'],
              group['channel_end_exclusive'],
          )),
      )


if __name__ == '__main__':
  unittest.main()
