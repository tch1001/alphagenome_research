#!/usr/bin/env python3
"""Tests for the independent individual-channel validation audit."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import audit_individual_channel_validation as audit
# pylint: enable=wrong-import-position


class IndividualChannelValidationAuditTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.result = audit.audit(audit.DEFAULT_RAW, audit.DEFAULT_ANALYSIS)

  def test_independent_arithmetic_and_rankings_match(self):
    self.assertEqual(self.result['maximum_raw_recovery_difference'], 0)
    self.assertEqual(
        self.result['maximum_analysis_aggregation_difference'], 0
    )
    self.assertTrue(self.result['all_rankings_match_exactly'])
    self.assertTrue(
        self.result['all_160_shifted_sufficiency_controls_exactly_zero']
    )

  def test_independent_advance_set(self):
    self.assertEqual(self.result['advancing_condition_ids'], [
        'necessity_E16_c0003',
        'necessity_E1_c0175',
        'necessity_E2_c0175',
    ])


if __name__ == '__main__':
  unittest.main()
