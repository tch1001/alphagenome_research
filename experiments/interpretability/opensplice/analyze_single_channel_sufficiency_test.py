#!/usr/bin/env python3
"""Tests for the single-channel spatial sufficiency analysis."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import analyze_single_channel_sufficiency as analysis
# pylint: enable=wrong-import-position


class SingleChannelSufficiencyAnalysisTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.loaded = analysis.load_results()
    cls.result = analysis.analyze(cls.loaded)

  def test_complete_result_is_confirmation_free(self):
    self.assertEqual(len(self.loaded['rows']), 180)
    self.assertEqual(len(self.loaded['full_rows']), 20)
    self.assertFalse(self.result['scope']['confirmation_access'])
    self.assertTrue(
        self.result['control_summary']['all_runtime_controls_passed']
    )

  def test_all_shifted_values_are_exactly_zero(self):
    controls = self.result['control_summary']
    self.assertEqual(controls['shifted_value_count'], 120)
    self.assertTrue(controls['all_shifted_B_exactly_zero'])
    self.assertEqual(controls['maximum_absolute_shifted_B'], 0)

  def test_e16_3_and_e2_175_pass_but_e1_175_does_not(self):
    self.assertEqual(
        self.result['channels_passing_frozen_sufficiency_rule'],
        ['E16_c0003', 'E2_c0175'],
    )

  def test_e2_175_is_positive_in_all_six_slc_effects(self):
    effect = self.result['per_channel']['E2_c0175']['per_gene'][
        'SLC25A48'
    ]['effect']
    self.assertEqual(effect['positive_spatial_contrast_count'], 6)
    self.assertAlmostEqual(
        effect['median_bidirectional_bottleneck_by_location']['intended'],
        0.015332479453516918,
    )


if __name__ == '__main__':
  unittest.main()
