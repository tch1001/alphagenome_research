#!/usr/bin/env python3
"""Tests for the encoder-feature decomposition plan."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import prepare_encoder_feature_decomposition as planner
# pylint: enable=wrong-import-position


class EncoderFeatureDecompositionPlanTest(unittest.TestCase):

  def test_frozen_design(self):
    plan = planner.build_plan()
    self.assertFalse(plan['scope']['confirmation_access'])
    self.assertEqual(plan['scope']['variant_count'], 20)
    self.assertEqual(plan['design']['traced_channel_indices'], [3, 175])
    self.assertEqual(plan['design']['planned_model_apply_count'], 40)
    self.assertEqual(len(plan['cases']), 20)
    for case in plan['cases']:
      self.assertEqual(
          [stage['resolution_bp'] for stage in case['stages']],
          [1, 2, 4, 8, 16, 32, 64],
      )


if __name__ == '__main__':
  unittest.main()
