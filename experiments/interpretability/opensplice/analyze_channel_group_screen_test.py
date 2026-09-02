#!/usr/bin/env python3
"""Tests for the V-local channel-group screen analysis."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import analyze_channel_group_screen as analysis
# pylint: enable=wrong-import-position


class ChannelGroupScreenAnalysisTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.loaded = analysis.load_results()
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

  def test_complete_screen_is_confirmation_free(self):
    self.assertEqual(len(self.loaded['rows']), 3440)
    self.assertEqual(len(self.loaded['full_rows']), 20)
    self.assertEqual(
        self.loaded['source']['raw_group_file_count'], 3440
    )
    self.assertFalse(self.result['scope']['confirmation_access'])
    self.assertTrue(
        self.result['control_summary']['all_runtime_controls_passed']
    )

  def test_cross_gene_ranking_and_losses(self):
    top = self.result['rankings']['top_10_cross_gene']
    self.assertEqual(
        [row['ranked_group_id'] for row in top[:3]],
        ['E1_c0160_0191', 'E32_c0000_0031', 'E16_c0000_0031'],
    )
    self.assertAlmostEqual(
        top[0]['per_gene']['BRAF']['effect']['median_necessity_loss'],
        0.02006971581742875,
    )
    self.assertAlmostEqual(
        top[0]['per_gene']['SLC25A48']['effect']['median_necessity_loss'],
        0.048143736529292686,
    )

  def test_refinement_includes_shared_and_gene_specific_parents(self):
    selected = self.result['recommended_refinement']
    self.assertEqual(selected['deterministic_parent_group_ids'], [
        'E1_c0160_0191',
        'E32_c0000_0031',
        'E16_c0000_0031',
        'E16_c0512_0543',
        'E2_c0160_0191',
    ])
    self.assertEqual(
        selected['top_within_gene_parent']['BRAF']['ranked_group_id'],
        'E16_c0512_0543',
    )
    self.assertEqual(
        selected['top_within_gene_parent']['SLC25A48']['ranked_group_id'],
        'E2_c0160_0191',
    )

  def test_persistent_band_is_present_at_every_candidate_resolution(self):
    band = self.result['persistent_channel_band_160_191']
    self.assertEqual([row['stage'] for row in band], list(analysis.ENABLED))
    self.assertTrue(all(
        row['per_gene']['SLC25A48']['effect']['median_necessity_loss'] > 0.03
        for row in band
    ))

  def test_screen_is_distributed_and_gene_profiles_differ(self):
    rankings = self.result['rankings']
    self.assertEqual(
        rankings['positive_effect_median_in_both_gene_count'], 55
    )
    self.assertEqual(
        rankings['positive_effect_median_group_count_by_gene'],
        {'BRAF': 60, 'SLC25A48': 166},
    )


if __name__ == '__main__':
  unittest.main()
