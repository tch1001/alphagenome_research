#!/usr/bin/env python3
"""Tests for the independent encoder feature-decomposition audit."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import audit_encoder_feature_decomposition as auditor
# pylint: enable=wrong-import-position


class EncoderFeatureDecompositionAuditTest(unittest.TestCase):

  def test_independent_metrics_match(self):
    result = auditor.audit()
    self.assertEqual(result['maximum_analysis_difference'], 0)
    self.assertEqual(result['e2_effect_amplified_count'], 6)
    self.assertLessEqual(
        result['maximum_direct_weight_activation_error'], 1e-3
    )


if __name__ == '__main__':
  unittest.main()
