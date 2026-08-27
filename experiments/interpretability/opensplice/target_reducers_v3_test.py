"""Synthetic CPU tests for the isolated OpenSplice v3 target contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


_MODULE_PATH = Path(__file__).with_name('target_reducers_v3.py')
_SPEC = importlib.util.spec_from_file_location('target_reducers_v3', _MODULE_PATH)
target_reducers = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = target_reducers
_SPEC.loader.exec_module(target_reducers)


def _classification_metadata():
  return pd.DataFrame({
      'name': ['donor', 'acceptor', 'donor', 'acceptor', 'Padding'],
      'strand': ['+', '+', '-', '-', '.'],
  })


def _usage_metadata():
  return pd.DataFrame({
      'name': [
          'usage_CL:0002518 total RNA-seq',
          'usage_CL:0002518 total RNA-seq',
          'usage_CL:0000000 total RNA-seq',
      ],
      'strand': ['+', '-', '+'],
      'ontology_curie': ['CL:0002518', 'CL:0002518', 'CL:0000000'],
  })


class TargetReducersV3Test(unittest.TestCase):

  def test_positive_and_negative_strand_canonical_mapping(self):
    positive = target_reducers.classification_logit_target(
        _classification_metadata(),
        interval_start_0based=900,
        interval_width=300,
        exon_start_1based=1001,
        exon_end_1based=1100,
        strand='+',
    )
    negative = target_reducers.classification_logit_target(
        _classification_metadata(),
        interval_start_0based=900,
        interval_width=300,
        exon_start_1based=1001,
        exon_end_1based=1100,
        strand='-',
    )
    self.assertEqual(
        [
            (site.role, site.position_index, site.track_index)
            for site in positive.endpoints
        ],
        [('acceptor', 100, 1), ('donor', 199, 0)],
    )
    self.assertEqual(
        [
            (site.role, site.position_index, site.track_index)
            for site in negative.endpoints
        ],
        [('acceptor', 199, 3), ('donor', 100, 2)],
    )
    self.assertEqual(positive.padding_track_index, 4)

  def test_classification_margin_is_paired_and_common_shift_invariant(self):
    target = target_reducers.classification_logit_target(
        _classification_metadata(),
        interval_start_0based=0,
        interval_width=4,
        exon_start_1based=2,
        exon_end_1based=4,
        strand='+',
    )
    logits = np.zeros((2, 4, 5), dtype=np.float32)
    logits[:, 1, 1] = [4.0, 8.0]
    logits[:, 1, 4] = [1.0, 2.0]
    logits[:, 3, 0] = [7.0, 3.0]
    logits[:, 3, 4] = [2.0, 1.0]
    # Large cross terms at the wrong position must not enter the paired target.
    logits[:, 1, 0] = 1000
    logits[:, 3, 1] = 1000

    reduced = target_reducers.reduce_classification_logit_margin(logits, target)
    shifted = target_reducers.reduce_classification_logit_margin(
        logits + np.array([10, 10, 10, 10, 10], dtype=np.float32), target
    )

    np.testing.assert_allclose(reduced.acceptor, [3.0, 6.0])
    np.testing.assert_allclose(reduced.donor, [5.0, 2.0])
    np.testing.assert_allclose(reduced.mean, [4.0, 4.0])
    np.testing.assert_allclose(shifted.mean, reduced.mean)

  def test_classification_metadata_and_interval_fail_closed(self):
    duplicate = _classification_metadata()
    duplicate.loc[4] = ['donor', '+']
    with self.assertRaisesRegex(ValueError, 'Duplicate splice-classification'):
      target_reducers.classification_logit_target(
          duplicate,
          interval_start_0based=0,
          interval_width=100,
          exon_start_1based=10,
          exon_end_1based=20,
          strand='+',
      )
    with self.assertRaisesRegex(ValueError, 'outside'):
      target_reducers.classification_logit_target(
          _classification_metadata(),
          interval_start_0based=100,
          interval_width=10,
          exon_start_1based=10,
          exon_end_1based=20,
          strand='+',
      )

  def test_usage_track_is_exact_strand_specific_logit(self):
    target = target_reducers.usage_logit_target(
        _usage_metadata(),
        ontology_curie='CL:0002518',
        interval_start_0based=0,
        interval_width=4,
        exon_start_1based=2,
        exon_end_1based=4,
        strand='-',
    )
    self.assertTrue(all(site.track_index == 1 for site in target.endpoints))
    logits = np.zeros((4, 3), dtype=np.float32)
    logits[3, 1] = 3.0
    logits[1, 1] = -1.0
    reduced = target_reducers.reduce_usage_logits(logits, target)
    np.testing.assert_allclose(reduced.acceptor, [3.0])
    np.testing.assert_allclose(reduced.donor, [-1.0])
    np.testing.assert_allclose(reduced.mean, [1.0])

  def test_usage_track_ambiguity_fails_closed(self):
    metadata = pd.concat([_usage_metadata(), _usage_metadata().iloc[[0]]])
    with self.assertRaisesRegex(ValueError, 'found 2'):
      target_reducers.resolve_usage_track(
          metadata,
          ontology_curie='CL:0002518',
          strand='+',
      )

  def test_internal_logit_extraction_never_falls_back_to_probabilities(self):
    with self.assertRaisesRegex(ValueError, 'do not reconstruct'):
      target_reducers.extract_internal_logits(
          {'splice_sites_classification': {'predictions': np.ones((2, 3, 5))}},
          head_name='splice_sites_classification',
      )
    logits = target_reducers.extract_internal_logits(
        {'splice_sites_classification': {'logits': np.ones((2, 3, 5))}},
        head_name='splice_sites_classification',
    )
    self.assertEqual(logits.shape, (2, 3, 5))

  def test_cassette_junction_delta_matches_logit_psi_formula(self):
    j = target_reducers.JunctionCoordinate
    inclusion_upstream = j('chr1', 100, 200, '+')
    inclusion_downstream = j('chr1', 300, 400, '+')
    skipping = j('chr1', 100, 400, '+')
    target = target_reducers.CassetteJunctionTarget(
        inclusion_upstream=inclusion_upstream,
        inclusion_downstream=inclusion_downstream,
        skipping=skipping,
        track_index=1,
    )
    junctions = [skipping, inclusion_downstream, inclusion_upstream]
    reference = np.array([[0, 5], [0, 10], [0, 10]], dtype=np.float32)
    alternate = np.array([[0, 5], [0, 20], [0, 20]], dtype=np.float32)

    delta = target_reducers.delta_cassette_junction_logit_psi(
        junctions, reference, junctions, alternate, target
    )

    self.assertAlmostEqual(delta, np.log(2.0), places=7)

  def test_junction_reducer_rejects_missing_duplicate_and_negative_counts(self):
    j = target_reducers.JunctionCoordinate
    inclusion_upstream = j('chr1', 100, 200, '+')
    inclusion_downstream = j('chr1', 300, 400, '+')
    skipping = j('chr1', 100, 400, '+')
    target = target_reducers.CassetteJunctionTarget(
        inclusion_upstream=inclusion_upstream,
        inclusion_downstream=inclusion_downstream,
        skipping=skipping,
        track_index=0,
    )
    with self.assertRaisesRegex(ValueError, 'found 0'):
      target_reducers.reduce_cassette_junction_logit_psi(
          [inclusion_upstream, skipping], np.ones((2, 1)), target
      )
    with self.assertRaisesRegex(ValueError, 'found 2'):
      target_reducers.reduce_cassette_junction_logit_psi(
          [inclusion_upstream, inclusion_downstream, skipping, skipping],
          np.ones((4, 1)),
          target,
      )
    with self.assertRaisesRegex(ValueError, 'nonnegative'):
      target_reducers.reduce_cassette_junction_logit_psi(
          [inclusion_upstream, inclusion_downstream, skipping],
          np.array([[1], [1], [-1]]),
          target,
      )


if __name__ == '__main__':
  unittest.main()
