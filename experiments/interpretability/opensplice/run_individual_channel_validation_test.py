#!/usr/bin/env python3
"""Tests for individual-channel necessity and sufficiency runner."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import run_individual_channel_validation as runner
# pylint: enable=wrong-import-position


class IndividualChannelValidationRunnerTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.plan, _ = runner.load_plan()
    cls.bindings = runner.screen.bind_cases(
        cls.plan, runner.route_v3.load_development_cases()
    )

  def test_complete_dry_run(self):
    dry = runner.build_dry_run(self.bindings, self.plan['conditions'])
    self.assertEqual(dry['variant_count'], 20)
    self.assertEqual(dry['condition_count'], 44)
    self.assertEqual(dry['individual_necessity_condition_count'], 32)
    self.assertEqual(dry['sufficiency_condition_count'], 12)
    self.assertEqual(dry['model_apply_count'], 960)
    self.assertFalse(dry['confirmation_access'])

  def test_necessity_masks_withhold_one_channel(self):
    full = runner.screen.channel_mask()
    for condition in self.plan['conditions'][:32]:
      channels = runner.condition_channels(condition)
      self.assertEqual(int(full.sum() - channels.sum()), 1)

  def test_sufficiency_masks_select_eight_channels(self):
    for condition in self.plan['conditions'][32:]:
      channels = runner.condition_channels(condition)
      self.assertEqual(int(channels.sum()), 8)
      feature = condition['feature']
      selected = np.argwhere(channels)
      self.assertEqual(set(selected[:, 0]), {feature['stage_index']})

  def test_shifted_selectors_are_equal_shape_and_distinct(self):
    case, planned = self.bindings[0]
    intended = runner.condition_selection(case, planned, 'intended')
    upstream = runner.condition_selection(case, planned, 'upstream')
    downstream = runner.condition_selection(case, planned, 'downstream')
    intended_mask = np.asarray(intended.decoder_skip_valid_mask)
    self.assertTrue(np.array_equal(
        intended_mask, np.asarray(upstream.decoder_skip_valid_mask)
    ))
    self.assertTrue(np.array_equal(
        intended_mask, np.asarray(downstream.decoder_skip_valid_mask)
    ))
    self.assertFalse(np.array_equal(
        np.asarray(intended.decoder_skip_positions),
        np.asarray(upstream.decoder_skip_positions),
    ))
    self.assertFalse(np.array_equal(
        np.asarray(intended.decoder_skip_positions),
        np.asarray(downstream.decoder_skip_positions),
    ))


if __name__ == '__main__':
  unittest.main()
