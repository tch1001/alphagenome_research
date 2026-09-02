#!/usr/bin/env python3
"""Tests for the independent single-channel sufficiency audit."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import audit_single_channel_sufficiency as audit
# pylint: enable=wrong-import-position


class SingleChannelSufficiencyAuditTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.result = audit.audit(audit.DEFAULT_RAW, audit.DEFAULT_ANALYSIS)

  def test_independent_arithmetic_and_controls_match(self):
    self.assertEqual(self.result['maximum_raw_recovery_difference'], 0)
    self.assertEqual(
        self.result['maximum_analysis_aggregation_difference'], 0
    )
    self.assertTrue(self.result['all_120_shifted_values_exactly_zero'])

  def test_independent_passing_set(self):
    self.assertEqual(
        self.result['passing_channels'], ['E16_c0003', 'E2_c0175']
    )


if __name__ == '__main__':
  unittest.main()
