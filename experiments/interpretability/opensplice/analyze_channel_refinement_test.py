#!/usr/bin/env python3
"""Tests for the completed 8-channel refinement analysis."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import analyze_channel_refinement as analysis
# pylint: enable=wrong-import-position


class ChannelRefinementAnalysisTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.loaded = analysis.load_results()
    cls.result = analysis.analyze(cls.loaded)

  def test_complete_refinement_is_confirmation_free(self):
    self.assertEqual(len(self.loaded['rows']), 400)
    self.assertEqual(len(self.loaded['full_rows']), 20)
    self.assertFalse(self.result['scope']['confirmation_access'])
    self.assertTrue(
        self.result['control_summary']['all_runtime_controls_passed']
    )

  def test_shared_parents_have_different_dominant_children(self):
    divergence = self.result['shared_parent_divergence']
    self.assertEqual(divergence['parent_count'], 3)
    self.assertEqual(
        divergence['different_top_child_in_both_gene_count'], 3
    )

  def test_braf_and_slc_programs_are_offset(self):
    parents = self.result['parent_summaries']
    self.assertEqual(
        parents['E32_c0000_0031']['top_child_by_gene']['BRAF']['group_id'],
        'E32_c0000_0007',
    )
    self.assertEqual(
        parents['E32_c0000_0031']['top_child_by_gene']['SLC25A48'][
            'group_id'
        ],
        'E32_c0016_0023',
    )
    self.assertEqual(
        parents['E16_c0000_0031']['top_child_by_gene']['BRAF']['group_id'],
        'E16_c0000_0007',
    )
    self.assertEqual(
        parents['E16_c0000_0031']['top_child_by_gene']['SLC25A48'][
            'group_id'
        ],
        'E16_c0016_0023',
    )

  def test_individual_refinement_selection_is_specific(self):
    selected = self.result['recommended_individual_channel_refinement']
    self.assertEqual(selected['deduplicated_child_ids'], [
        'E32_c0000_0007',
        'E16_c0000_0007',
        'E2_c0168_0175',
        'E1_c0168_0175',
    ])
    self.assertEqual(selected['individual_channel_count'], 32)

  def test_strong_slc_children_match_across_e1_and_e2(self):
    selected = self.result['recommended_individual_channel_refinement'][
        'per_gene'
    ]['SLC25A48']
    self.assertEqual(
        [row['group_id'] for row in selected],
        ['E2_c0168_0175', 'E1_c0168_0175'],
    )
    self.assertAlmostEqual(
        selected[0]['per_gene']['SLC25A48']['effect'][
            'median_necessity_loss'
        ],
        0.04556050829739722,
    )


if __name__ == '__main__':
  unittest.main()
