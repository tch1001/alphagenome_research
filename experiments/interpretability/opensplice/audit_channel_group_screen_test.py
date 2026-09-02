#!/usr/bin/env python3
"""Tests for the independent channel-screen audit."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import audit_channel_group_screen as audit
# pylint: enable=wrong-import-position


class ChannelGroupScreenAuditTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.result = audit.audit(audit.DEFAULT_RAW, audit.DEFAULT_ANALYSIS)

  def test_independent_recovery_and_aggregation_are_exact(self):
    self.assertEqual(self.result['maximum_raw_recovery_difference'], 0)
    self.assertEqual(
        self.result['maximum_analysis_aggregation_difference'], 0
    )
    self.assertTrue(self.result['cross_gene_ranking_matches_exactly'])

  def test_independent_parent_selection(self):
    self.assertEqual(self.result['top_cross_gene_groups'], [
        'E1_c0160_0191', 'E32_c0000_0031', 'E16_c0000_0031'
    ])
    self.assertEqual(self.result['top_within_gene_group'], {
        'BRAF': 'E16_c0512_0543',
        'SLC25A48': 'E2_c0160_0191',
    })


if __name__ == '__main__':
  unittest.main()
