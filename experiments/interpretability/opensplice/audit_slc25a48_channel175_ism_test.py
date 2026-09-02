#!/usr/bin/env python3
"""Tests for the independent SLC25A48 channel-175 ISM audit."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import audit_slc25a48_channel175_ism as auditor
# pylint: enable=wrong-import-position


class Slc25a48Channel175IsmAuditTest(unittest.TestCase):

  def test_independent_metrics_match(self):
    result = auditor.audit()
    self.assertEqual(result['candidate_edit_count'], 123)
    self.assertEqual(result['invariant_ag_negative_count'], 6)
    self.assertEqual(result['maximum_analysis_difference'], 0)


if __name__ == '__main__':
  unittest.main()
