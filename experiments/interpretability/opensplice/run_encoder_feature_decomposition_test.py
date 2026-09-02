#!/usr/bin/env python3
"""Tests for the natural encoder feature-decomposition runner."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import run_encoder_feature_decomposition as runner
# pylint: enable=wrong-import-position


class EncoderFeatureDecompositionRunnerTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.plan, _ = runner.load_plan()
    cls.bindings = runner.bind_cases(
        cls.plan, runner.route_v3.load_development_cases()
    )

  def test_complete_dry_run(self):
    dry = runner.build_dry_run(self.bindings)
    self.assertFalse(dry['confirmation_access'])
    self.assertEqual(dry['variant_count'], 20)
    self.assertEqual(dry['traced_channels'], [3, 175])
    self.assertEqual(dry['model_apply_count'], 40)

  def test_positions_match_route_selection(self):
    for case, planned in self.bindings:
      positions, valid = runner.trace_selection(case, planned)
      self.assertEqual(positions.shape, (7, 3))
      self.assertEqual(valid.shape, (7, 3))


if __name__ == '__main__':
  unittest.main()
