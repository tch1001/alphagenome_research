#!/usr/bin/env python3
"""Tests for the V-local grouped-channel screen runner."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import run_channel_group_screen as runner
import run_route_census_v3 as route_v3
# pylint: enable=wrong-import-position


class ChannelGroupScreenRunnerTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.plan, cls.plan_sha256 = runner.load_plan()
    cls.bindings = runner.bind_cases(
        cls.plan, route_v3.load_development_cases()
    )

  def test_plan_binds_exact_development_partition(self):
    self.assertEqual(len(self.bindings), 20)
    self.assertFalse(self.plan['scope']['confirmation_access'])

  def test_full_channel_mask_covers_exact_candidate_widths(self):
    mask = runner.channel_mask()
    self.assertEqual(mask.shape, (7, 1536))
    self.assertEqual(np.count_nonzero(mask), 5504)
    for index, (_, _, width, enabled) in enumerate(runner.planner.STAGES):
      self.assertEqual(np.count_nonzero(mask[index]), width if enabled else 0)

  def test_every_group_withholds_only_its_contiguous_block(self):
    full = runner.channel_mask()
    for group in self.plan['groups']:
      without = runner.channel_mask(group)
      difference = full & ~without
      self.assertEqual(
          np.count_nonzero(difference), group['channel_count']
      )
      self.assertEqual(
          np.flatnonzero(difference.any(axis=1)).tolist(),
          [group['stage_index']],
      )
      selected = np.flatnonzero(difference[group['stage_index']])
      self.assertEqual(selected[0], group['channel_start_inclusive'])
      self.assertEqual(selected[-1] + 1, group['channel_end_exclusive'])

  def test_selection_and_interventions_keep_one_fixed_pytree_shape(self):
    case, planned = self.bindings[0]
    selection = runner.channel_selection(case, planned)
    full = runner.channel_mask()
    without = runner.channel_mask(self.plan['groups'][0])
    identity = runner.interventions(
        selection, full, active_positions=False
    )
    full_value = runner.interventions(
        selection, full, active_positions=True
    )
    group_value = runner.interventions(
        selection, without, active_positions=True
    )
    runner.validate_runtime_contract(
        selection, identity, full, active_positions=False
    )
    runner.validate_runtime_contract(
        selection, full_value, full, active_positions=True
    )
    runner.validate_runtime_contract(
        selection, group_value, without, active_positions=True
    )
    self.assertEqual(
        identity.decoder_skip_states.channel_mask.shape, (7, 1536)
    )
    self.assertEqual(
        full_value.decoder_skip_states.channel_mask.shape, (7, 1536)
    )
    self.assertEqual(
        group_value.decoder_skip_states.channel_mask.shape, (7, 1536)
    )
    self.assertEqual(
        selection.decoder_skip_positions.shape, (7, runner.spatial.SLOTS)
    )

  def test_full_dry_run_has_3520_applies(self):
    dry = runner.build_dry_run(self.bindings, self.plan['groups'])
    self.assertEqual(dry['variant_count'], 20)
    self.assertEqual(dry['group_count'], 172)
    self.assertEqual(dry['model_apply_count'], 3520)
    self.assertEqual(dry['fixed_channel_shape'], [7, 1536])
    self.assertFalse(dry['confirmation_access'])


if __name__ == '__main__':
  unittest.main()
