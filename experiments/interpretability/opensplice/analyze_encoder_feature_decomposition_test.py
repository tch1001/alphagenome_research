#!/usr/bin/env python3
"""Tests for encoder-feature decomposition analysis."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import analyze_encoder_feature_decomposition as analyzer
# pylint: enable=wrong-import-position


class EncoderFeatureDecompositionAnalysisTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.analysis = analyzer.analyze()

  def test_weight_activation_closure_and_acceptor_core(self):
    weights = self.analysis['weight_analysis']
    self.assertEqual(weights['direct_kernel_preferred_core'], 'TAGG')
    self.assertLessEqual(
        weights['maximum_direct_weight_to_activation_absolute_difference'],
        1e-3,
    )

  def test_effect_specific_e2_amplification(self):
    interpretation = self.analysis['interpretation']
    self.assertTrue(
        interpretation[
            'e1_channel_175_is_nonlinear_composite_acceptor_detector'
        ]
    )
    self.assertTrue(interpretation['e2_amplifies_all_six_effect_vectors'])
    self.assertFalse(interpretation['e2_increases_cross_effect_alignment'])


if __name__ == '__main__':
  unittest.main()
