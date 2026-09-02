#!/usr/bin/env python3
"""Tests for the SLC25A48 channel-175 ISM analysis."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import analyze_slc25a48_channel175_ism as analyzer
# pylint: enable=wrong-import-position


class Slc25a48Channel175IsmAnalysisTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.analysis = analyzer.analyze()

  def test_frozen_core_and_candidate_count(self):
    candidate_count = self.analysis['source']['candidate_edit_count']
    self.assertEqual(candidate_count, 123)
    self.assertEqual(candidate_count, len(self.analysis['all_edits']))
    self.assertEqual(
        self.analysis['reference_core_minus3_through_0'], 'TAGG'
    )

  def test_mechanistic_predictions_pass(self):
    interpretation = self.analysis['interpretation']
    self.assertTrue(all(interpretation.values()))
    association = self.analysis[
        'channel_to_acceptor_output_associations'
    ]['e2_output_delta_at_acceptor']
    self.assertGreater(association['pearson'], 0.9)


if __name__ == '__main__':
  unittest.main()
