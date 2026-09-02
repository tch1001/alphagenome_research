#!/usr/bin/env python3
"""Tests for the SLC25A48 channel-175 ISM runner."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import run_slc25a48_channel175_ism as runner
# pylint: enable=wrong-import-position


class Slc25a48Channel175IsmRunnerTest(unittest.TestCase):

  def test_dry_run_and_batch_layout(self):
    plan, _ = runner.load_plan()
    dry = runner.build_dry_run(plan)
    self.assertFalse(dry['confirmation_access'])
    self.assertEqual(dry['candidate_edit_count'], 123)
    self.assertEqual(dry['batch_count'], 25)
    self.assertEqual(dry['model_apply_count'], 50)
    candidates = runner.build_candidates('A' * 100, 50)
    batches = runner.batch_layout(candidates)
    self.assertEqual(len(batches), 25)
    self.assertTrue(all(len(rows) == 6 for rows in batches))

  def test_feature_selection_is_fixed_shape(self):
    positions, valid = runner.feature_selection(1001, 500)
    self.assertEqual(positions.shape, (7, 49))
    self.assertEqual(valid.shape, (7, 49))


if __name__ == '__main__':
  unittest.main()
