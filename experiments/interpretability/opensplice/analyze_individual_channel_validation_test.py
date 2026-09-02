#!/usr/bin/env python3
"""Tests for the individual-channel causal validation analysis."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import analyze_individual_channel_validation as analysis
# pylint: enable=wrong-import-position


class IndividualChannelValidationAnalysisTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.loaded = analysis.load_results()
    cls.result = analysis.analyze(cls.loaded)

  def test_complete_validation_is_confirmation_free(self):
    self.assertEqual(len(self.loaded['rows']), 880)
    self.assertEqual(len(self.loaded['full_rows']), 20)
    self.assertFalse(self.result['scope']['confirmation_access'])
    self.assertTrue(
        self.result['control_summary']['all_runtime_controls_passed']
    )

  def test_all_shifted_sufficiency_controls_are_exactly_zero(self):
    controls = self.result['control_summary']
    self.assertEqual(controls['shifted_sufficiency_control_count'], 160)
    self.assertTrue(controls['all_shifted_sufficiency_B_exactly_zero'])
    self.assertEqual(controls['maximum_absolute_shifted_sufficiency_B'], 0)

  def test_three_individual_coordinates_advance(self):
    necessity = self.result['individual_necessity']
    self.assertEqual(necessity['advancing_channel_count'], 3)
    self.assertEqual(
        [(row['stage'], row['channel_start_inclusive'])
         for row in necessity['advancing_channels']],
        [('E16', 3), ('E1', 175), ('E2', 175)],
    )

  def test_channel_175_is_strong_at_two_slc_resolutions(self):
    rows = {
        (row['stage'], row['channel_start_inclusive']): row
        for row in self.result['individual_necessity']['advancing_channels']
    }
    self.assertAlmostEqual(
        rows[('E2', 175)]['per_gene']['SLC25A48']['effect']['median'],
        0.045123195533709276,
    )
    self.assertAlmostEqual(
        rows[('E1', 175)]['per_gene']['SLC25A48']['effect']['median'],
        0.03987051378089504,
    )

  def test_all_four_children_pass_localized_sufficiency(self):
    sufficiency = self.result['eight_channel_sufficiency']
    self.assertEqual(sufficiency['selected_child_count'], 4)
    self.assertTrue(
        sufficiency[
            'all_children_pass_selected_gene_localized_sufficiency_rule'
        ]
    )


if __name__ == '__main__':
  unittest.main()
