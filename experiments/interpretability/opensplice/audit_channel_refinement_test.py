#!/usr/bin/env python3
"""Tests for the independent 8-channel refinement audit."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import audit_channel_refinement as audit
# pylint: enable=wrong-import-position


class ChannelRefinementAuditTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.result = audit.audit(audit.DEFAULT_RAW, audit.DEFAULT_ANALYSIS)

  def test_independent_arithmetic_and_rankings_match(self):
    self.assertEqual(self.result['maximum_raw_recovery_difference'], 0)
    self.assertEqual(self.result['maximum_analysis_median_difference'], 0)
    self.assertTrue(self.result['all_rankings_match_exactly'])

  def test_independent_candidate_selection(self):
    self.assertEqual(self.result['selected_children'], {
        'BRAF': ['E32_c0000_0007', 'E16_c0000_0007'],
        'SLC25A48': ['E2_c0168_0175', 'E1_c0168_0175'],
    })


if __name__ == '__main__':
  unittest.main()
