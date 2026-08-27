#!/usr/bin/env python3
"""Integrity checks for the canonical TAL1 causal-tracing artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


RESULTS = Path(__file__).with_name('results')
RESIDUAL_RESULT = RESULTS / (
    'tal1_residual_head_self_controls_v2_131kb_shift959.json'
)
PAIR_RESULT = RESULTS / 'tal1_pair_self_controls_131kb_shift959.json'


def _load(path: Path) -> dict:
  return json.loads(path.read_text())


def _corrected(
    cross_allele: float,
    recipient_self: float,
    source_baseline: float,
    recipient_baseline: float,
) -> float:
  """Returns movement from a self control toward the source allele."""
  return (cross_allele - recipient_self) / (
      source_baseline - recipient_baseline
  )


class ResultArtifactsTest(unittest.TestCase):

  def test_direction_schema_and_recovery_signs(self):
    for result in (_load(RESIDUAL_RESULT), _load(PAIR_RESULT)):
      self.assertEqual(
          result['patch_direction_schema']['reference_to_alternate'],
          {
              'source_allele': 'reference',
              'recipient_allele': 'alternate',
              'same_recipient_control': 'alternate_to_alternate',
          },
      )
      self.assertEqual(
          result['patch_direction_schema']['alternate_to_reference'],
          {
              'source_allele': 'alternate',
              'recipient_allele': 'reference',
              'same_recipient_control': 'reference_to_reference',
          },
      )

    # The formula must keep the same interpretation whichever allele is larger.
    self.assertAlmostEqual(_corrected(11, 20, 10, 20), 0.9)
    self.assertAlmostEqual(_corrected(19, 10, 20, 10), 0.9)

  def test_residual_and_local_head_metrics_recompute_exactly(self):
    result = _load(RESIDUAL_RESULT)
    ref = result['baseline']['reference']['total'][0]
    alt = result['baseline']['alternate']['total'][0]
    self.assertEqual(result['baseline']['reference_repeat_delta'], 0)
    self.assertEqual(result['baseline']['alternate_repeat_delta'], 0)

    records = (
        result['reference_to_alternate_residual_patches']
        + result['local_head_value_patches']
    )
    for record in records:
      self.assertAlmostEqual(
          record['corrected_recovery_toward_reference'],
          _corrected(
              record.get(
                  'reference_to_alternate_patched_target_total',
                  record.get('patched_target_total'),
              ),
              record['alternate_self_patched_target_total'],
              ref,
              alt,
          ),
      )
      self.assertAlmostEqual(
          record['corrected_recovery_toward_alternate'],
          _corrected(
              record['alternate_to_reference_patched_target_total'],
              record['reference_self_patched_target_total'],
              alt,
              ref,
          ),
      )

    enhancer = next(
        record
        for record in result['reference_to_alternate_residual_patches']
        if record['stage'] == 'pre_attention'
        and record['layer'] == 0
        and record['region'] == 'enhancer'
    )
    self.assertGreater(enhancer['corrected_recovery_toward_reference'], 0.9)
    self.assertGreater(enhancer['corrected_recovery_toward_alternate'], 0.9)
    max_distance = max(
        abs(record['corrected_recovery_toward_reference'])
        for record in result['reference_to_alternate_residual_patches']
        if record['region_role'] == 'distance_control'
    )
    self.assertLess(max_distance, 0.05)
    self.assertLess(
        max(
            abs(record['corrected_recovery_toward_reference'])
            for record in result['local_head_value_patches']
        ),
        0.05,
    )

  def test_pair_metrics_are_self_controlled_and_near_null(self):
    result = _load(PAIR_RESULT)
    ref = result['baseline']['reference']['total'][0]
    alt = result['baseline']['alternate']['total'][0]
    records = result['candidate_pair_patches']
    self.assertEqual(len(records), 432)
    for record in records:
      self.assertEqual(record['alternate_self_patch_delta_from_baseline'], 0)
      self.assertEqual(record['reference_self_patch_delta_from_baseline'], 0)
      self.assertAlmostEqual(
          record['corrected_recovery_toward_reference'],
          _corrected(
              record['patched_target_total'],
              record['alternate_self_patched_target_total'],
              ref,
              alt,
          ),
      )
      self.assertAlmostEqual(
          record['corrected_recovery_toward_alternate'],
          _corrected(
              record['alternate_to_reference_patched_target_total'],
              record['reference_self_patched_target_total'],
              alt,
              ref,
          ),
      )
    candidate = [
        record for record in records if record['edge_role'] == 'candidate'
    ]
    self.assertLess(
        max(abs(record['corrected_recovery_toward_reference']) for record in candidate),
        0.01,
    )
    self.assertLess(
        max(abs(record['corrected_recovery_toward_alternate']) for record in candidate),
        0.01,
    )

  def test_canonical_runs_use_identical_sequence_and_crop(self):
    residual = _load(RESIDUAL_RESULT)
    pair = _load(PAIR_RESULT)
    for key in (
        'interval',
        'variant',
        'variant_zero_based_offset',
        'variant_offset_in_128bp_token',
        'variant_offset_in_2048bp_bin',
        'sequence_provenance',
    ):
      self.assertEqual(
          residual['configuration'][key], pair['configuration'][key]
      )


if __name__ == '__main__':
  unittest.main()
