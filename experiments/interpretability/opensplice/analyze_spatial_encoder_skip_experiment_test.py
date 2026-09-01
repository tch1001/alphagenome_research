#!/usr/bin/env python3
"""Tests for the spatial encoder-skip model-behavior analysis."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import analyze_spatial_encoder_skip_experiment as analysis
# pylint: enable=wrong-import-position


class SpatialEncoderSkipAnalysisTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.loaded = analysis.load_results(analysis.DEFAULT_INPUT)
    cls.result = analysis.analyze(cls.loaded)

  def test_recovery_recomputes_both_reciprocal_directions(self):
    readout = {
        'row_roles': list(analysis.ROLES),
        'means': [4, 2, 3, 2, 3, 4],
        'totals': [8, 4, 6, 4, 6, 8],
        'num_values': 2,
    }
    self.assertEqual(analysis.recovery_from_readout(readout), {
        'reference_into_alternate': 0.5,
        'alternate_into_reference': 0.5,
        'bidirectional_bottleneck': 0.5,
        'bidirectional_mean': 0.5,
    })

  def test_full_cube_is_complete_and_confirmation_free(self):
    self.assertEqual(len(self.loaded['rows']), 240)
    self.assertEqual(
        self.loaded['source']['raw_identity_file_count'], 20
    )
    self.assertEqual(
        self.loaded['source']['raw_condition_file_count'], 240
    )
    self.assertFalse(self.result['scope']['confirmation_access'])

  def test_all_shifted_controls_are_exactly_zero(self):
    controls = self.result['control_summary']
    self.assertTrue(controls['all_160_shifted_control_B_exactly_zero'])
    self.assertEqual(controls['maximum_absolute_shifted_control_B'], 0.0)

  def test_frozen_rule_selects_v_a_and_s_but_not_d(self):
    self.assertEqual(
        self.result['supports_passing_frozen_rule_in_both_genes'],
        ['V', 'A', 'S'],
    )
    primary = self.result['effect_variants_primary']
    self.assertAlmostEqual(
        primary['V']['per_gene']['BRAF'][
            'median_bidirectional_bottleneck'
        ]['intended'],
        0.41409281454318736,
    )
    self.assertAlmostEqual(
        primary['V']['per_gene']['SLC25A48'][
            'median_bidirectional_bottleneck'
        ]['intended'],
        0.39648756761411597,
    )
    self.assertAlmostEqual(
        primary['D']['per_gene']['SLC25A48'][
            'median_bidirectional_bottleneck'
        ]['intended'],
        0.008933437880865,
    )

  def test_every_effect_variant_has_positive_spatial_contrast(self):
    primary = self.result['effect_variants_primary']
    for support in analysis.SUPPORTS:
      for gene in analysis.GENES:
        value = primary[support]['per_gene'][gene]
        self.assertEqual(value['variant_count'], 6)
        self.assertEqual(value['positive_spatial_contrast_count'], 6)

  def test_braf_neutrals_remain_a_specificity_warning(self):
    warning = self.result['specificity_warning']
    self.assertFalse(
        warning['passes_effect_exceeds_neutral_for_all_passing_supports']
    )
    for support in self.result[
        'supports_passing_frozen_rule_in_both_genes'
    ]:
      self.assertFalse(
          warning['per_support_per_gene'][support]['BRAF'][
              'effect_exceeds_neutral'
          ]
      )
      self.assertTrue(
          warning['per_support_per_gene'][support]['SLC25A48'][
              'effect_exceeds_neutral'
          ]
      )

  def test_overlap_audit_records_splice_site_collinearity(self):
    overlap = self.result['effect_support_overlap']
    self.assertGreater(
        overlap['SLC25A48']['V-A']['overlapping_case_stage_fraction'], 0.9
    )
    self.assertGreater(
        overlap['BRAF']['V-D']['overlapping_case_stage_count'], 0
    )


if __name__ == '__main__':
  unittest.main()
