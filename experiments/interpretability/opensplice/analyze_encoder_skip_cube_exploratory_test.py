#!/usr/bin/env python3
"""Focused tests for the exploratory encoder-skip cube analyzer."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

import analyze_encoder_skip_cube_exploratory as analyzer  # pylint: disable=g-import-not-at-top


class EncoderSkipCubeExploratoryTest(unittest.TestCase):

  def test_endpoint_values_are_recomputed_from_raw_logits(self):
    target = {
        'selected_logit_axis': ['relevant_class', 'padding_class'],
        'endpoint_axis': ['acceptor', 'donor'],
        'selected_logits': [
            [[float(row + 2), 1.0], [float(row + 4), 2.0]]
            for row in range(6)
        ],
    }
    target['endpoint_margins'] = [
        [float(row + 1), float(row + 2)] for row in range(6)
    ]
    target['totals'] = [float(2 * row + 3) for row in range(6)]
    target['means'] = [float(row) + 1.5 for row in range(6)]
    record = {'target_readout': target, 'repeat_target_readout': target.copy()}
    self.assertEqual(
        analyzer._endpoint_values(record),  # pylint: disable=protected-access
        target['means'],
    )

  def test_shapley_matches_independent_permutation_definition(self):
    for player_count in range(1, 7):
      values = []
      for mask in range(1 << player_count):
        additive = sum(
            (index + 1.25) for index in range(player_count)
            if mask & (1 << index)
        )
        interaction = 0.7 if player_count >= 2 and mask & 3 == 3 else 0.0
        values.append(additive + interaction)
      direct = analyzer.shapley_values(values, player_count)
      permutation = analyzer.shapley_values_by_permutations(
          values, player_count
      )
      for observed, expected in zip(direct, permutation):
        self.assertAlmostEqual(observed, expected, places=12)
      self.assertAlmostEqual(
          sum(direct), values[-1] - values[0], places=12
      )

  def test_harsanyi_dividends_reconstruct_every_coalition(self):
    player_count = 5
    values = [
        math.sin(mask + 0.25) + 0.1 * mask.bit_count()
        for mask in range(1 << player_count)
    ]
    dividends = analyzer.harsanyi_dividends(values, player_count)
    for coalition in range(1 << player_count):
      reconstruction = sum(
          dividends[subset]
          for subset in range(1 << player_count)
          if subset & coalition == subset
      )
      self.assertAlmostEqual(reconstruction, values[coalition], places=12)

  def test_eight_player_mask_mapping_is_bijective_and_exact(self):
    observed = {
        analyzer.eight_player_mask_to_coalition_id(mask)
        for mask in range(256)
    }
    self.assertEqual(observed, set(range(256)))
    self.assertEqual(analyzer.eight_player_mask_to_coalition_id(0), 0)
    self.assertEqual(analyzer.eight_player_mask_to_coalition_id(1), 128)
    self.assertEqual(analyzer.eight_player_mask_to_coalition_id(2), 1)
    self.assertEqual(analyzer.eight_player_mask_to_coalition_id(255), 255)

  def test_candidate_rule_prefers_smallest_then_maximin_then_mask(self):
    rows = {
        family: {
            str(mask): {'BRAF': 0.1, 'SLC25A48': 0.1}
            for mask in range(128)
        }
        for family in ('natural_T', 'donor_T')
    }
    rows['natural_T']['127'] = {'BRAF': 1.0, 'SLC25A48': 1.0}
    rows['natural_T']['3'] = {'BRAF': 0.81, 'SLC25A48': 0.82}
    rows['natural_T']['5'] = {'BRAF': 0.84, 'SLC25A48': 0.83}
    rows['natural_T']['6'] = {'BRAF': 0.84, 'SLC25A48': 0.83}
    candidate = analyzer.choose_resolution_candidate(rows)
    self.assertIsNotNone(candidate)
    self.assertEqual(candidate['family'], 'natural_T')
    self.assertEqual(candidate['e_mask'], 5)
    self.assertEqual(candidate['enabled_players'], ['E64', 'E16'])

  def test_additive_game_has_only_singleton_dividends(self):
    weights = (0.3, -0.2, 1.1, 0.7)
    values = [
        2.0 + sum(
            weight for index, weight in enumerate(weights)
            if mask & (1 << index)
        )
        for mask in range(16)
    ]
    dividends = analyzer.harsanyi_dividends(values, 4)
    self.assertAlmostEqual(dividends[0], 2.0)
    for index, weight in enumerate(weights):
      self.assertAlmostEqual(dividends[1 << index], weight)
    for mask in range(1, 16):
      if mask.bit_count() > 1:
        self.assertAlmostEqual(dividends[mask], 0.0)


if __name__ == '__main__':
  unittest.main()
