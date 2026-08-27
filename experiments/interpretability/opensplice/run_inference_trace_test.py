"""CPU tests for the resume-safe OpenSplice inference/tracing runner."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

import jax.numpy as jnp
import numpy as np
import pandas as pd


_MODULE_PATH = Path(__file__).with_name('run_inference_trace.py')
_SPEC = importlib.util.spec_from_file_location(
    'run_inference_trace', _MODULE_PATH
)
run_inference_trace = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = run_inference_trace
_SPEC.loader.exec_module(run_inference_trace)


class RunInferenceTraceTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.tempdir = tempfile.TemporaryDirectory()
    self.root = Path(self.tempdir.name)
    self.exons = self.root / 'exons.tsv'
    self.selected = self.root / 'selected.tsv'
    self.exons.write_text(
        'selection_order\tgene\texon_id\tensembl_exon_id\tchromosome\tstrand\t'
        'exon_start_1based\texon_end_1based\n'
        '1\tGENE\tGENE_e1\tENSETEST1\tchr1\t-\t1001\t1100\n',
        encoding='utf-8',
    )

  def tearDown(self):
    self.tempdir.cleanup()
    super().tearDown()

  def _write_selected(self, *, ref='A', alt='C', mut_type='sub'):
    columns = sorted(run_inference_trace.REQUIRED_SELECTED_COLUMNS)
    row = {column: '' for column in columns}
    row.update({
        'selection_version': 'opensplice-circuit-v2-snv-only',
        'exon_order': '1',
        'gene': 'GENE',
        'exon_id': 'GENE_e1',
        'ensembl_exon_id': 'ENSETEST1',
        'selection_class': 'significant_effect',
        'observed_effect_sign': 'positive',
        'chromosome': '1',
        'position_1based': '1050',
        'reference_bases': ref,
        'alternate_bases': alt,
        'variant_id': 'GENE_e1_A51G',
        'region': 'Exon',
        'mut_type': mut_type,
        'delta_psi': '12.5',
        'delta_logit': '2.0',
        'significant': 'yes',
        'measured': 'true',
    })
    with self.selected.open('w', encoding='utf-8', newline='') as handle:
      writer = csv.DictWriter(handle, fieldnames=columns, delimiter='\t')
      writer.writeheader()
      writer.writerow(row)

  def test_frozen_join_and_negative_strand_target_mapping(self):
    self._write_selected()

    (case,) = run_inference_trace.load_cases(self.selected, self.exons)
    interval = run_inference_trace.centered_interval(case, 16_384)
    selection = run_inference_trace.paired_target_selection(case, interval)

    self.assertEqual(case.chromosome, 'chr1')
    self.assertEqual(
        run_inference_trace.canonical_sites(case),
        (('acceptor', 1100, 3), ('donor', 1001, 2)),
    )
    np.testing.assert_array_equal(selection.track_indices, [3, 2])
    position_sets = run_inference_trace.trace_position_sets(case, interval)
    by_name = {
        position_set.name: position_set for position_set in position_sets
    }
    self.assertEqual(len(position_sets), 12)
    self.assertEqual(
        by_name['S'].tokens,
        tuple(
            dict.fromkeys(
                by_name['V'].tokens
                + by_name['A'].tokens
                + by_name['D'].tokens
            )
        ),
    )
    for name in ('V', 'A', 'D', 'S'):
      candidate = by_name[name]
      for suffix in ('upstream', 'downstream'):
        control = by_name[f'{name}_control_{suffix}']
        self.assertEqual(len(control.tokens), len(candidate.tokens))
        offsets = [
            observed - expected
            for observed, expected in zip(
                control.tokens, candidate.tokens, strict=True
            )
        ]
        self.assertEqual(len(set(offsets)), 1)
        self.assertGreaterEqual(abs(offsets[0]), 4)
        self.assertFalse(set(control.tokens) & set(by_name['S'].tokens))

  def test_non_snv_fails_closed(self):
    self._write_selected(ref='AT', alt='A', mut_type='del1')

    with self.assertRaisesRegex(ValueError, 'SNV-only'):
      run_inference_trace.load_cases(self.selected, self.exons)

  def test_public_score_uses_only_paired_canonical_channels(self):
    self._write_selected()
    (case,) = run_inference_trace.load_cases(self.selected, self.exons)
    interval = run_inference_trace.centered_interval(case, 16_384)
    metadata = pd.DataFrame({
        'name': ['donor', 'acceptor', 'donor', 'acceptor'],
        'strand': ['+', '+', '-', '-'],
    })
    reference = np.zeros((interval.width, 4), dtype=np.float32)
    alternate = np.zeros_like(reference)
    acceptor_index = case.exon_end_1based - 1 - interval.start
    donor_index = case.exon_start_1based - 1 - interval.start
    reference[acceptor_index, 3] = 0.2
    alternate[acceptor_index, 3] = 0.6
    reference[donor_index, 2] = 0.4
    alternate[donor_index, 2] = 0.8
    # Cross terms must not enter the paired score.
    alternate[acceptor_index, 2] = 99
    alternate[donor_index, 3] = 99
    output = types.SimpleNamespace(
        reference=types.SimpleNamespace(
            splice_sites=types.SimpleNamespace(
                values=reference, metadata=metadata
            )
        ),
        alternate=types.SimpleNamespace(
            splice_sites=types.SimpleNamespace(
                values=alternate, metadata=metadata.copy()
            )
        ),
    )

    score = run_inference_trace.score_splice_site_tracks(
        case, interval, output
    )

    self.assertAlmostEqual(score['delta_acceptor'], 0.4)
    self.assertAlmostEqual(score['delta_donor'], 0.4)
    self.assertAlmostEqual(score['mean_delta_splice'], 0.4)

  def test_direction_gate_excludes_neutral_and_handles_both_signs(self):
    self.assertTrue(
        run_inference_trace.direction_result(
            -0.2,
            -10,
            is_effect=True,
            predicted_effect_threshold=0.01,
        )['direction_correct']
    )
    self.assertFalse(
        run_inference_trace.direction_result(
            0.2,
            -10,
            is_effect=True,
            predicted_effect_threshold=0.01,
        )['direction_correct']
    )
    self.assertIsNone(
        run_inference_trace.direction_result(
            0.2,
            0.01,
            is_effect=False,
            predicted_effect_threshold=0.01,
        )['direction_correct']
    )
    below = run_inference_trace.direction_result(
        0.009,
        10,
        is_effect=True,
        predicted_effect_threshold=0.01,
    )
    self.assertEqual(below['predicted_sign'], 'below_threshold')
    self.assertIsNone(below['direction_correct'])

  def test_joint_residual_patch_masks_every_position_in_set(self):
    identity = (
        run_inference_trace.interpretability.no_transformer_interventions(
            batch_size=1, num_edges=1
        )
    )
    donor_trace = types.SimpleNamespace(
        pre_attention_residuals=jnp.ones((9, 1, 24, 4), jnp.float32)
    )

    intervention = run_inference_trace._residual_patch(  # pylint: disable=protected-access
        identity,
        donor_trace,
        stage='pre_attention',
        layer=2,
        slots=(1, 4, 7),
    )

    expected = np.zeros((9, 1, 24), dtype=bool)
    expected[2, 0, [1, 4, 7]] = True
    np.testing.assert_array_equal(
        intervention.pre_attention_residual.replace_mask, expected
    )

  def test_public_and_paired_target_must_match(self):
    public = {'reference_mean': 0.2, 'alternate_mean': 0.4}
    deltas = run_inference_trace.validate_public_paired_target(
        public, 0.2, 0.4
    )
    self.assertEqual(deltas['reference_delta_from_public'], 0)
    accepted = run_inference_trace.validate_public_paired_target(
        public,
        0.2 + run_inference_trace.PUBLIC_PAIRED_TARGET_TOLERANCE,
        0.4,
    )
    self.assertAlmostEqual(
        accepted['reference_delta_from_public'],
        run_inference_trace.PUBLIC_PAIRED_TARGET_TOLERANCE,
    )
    with self.assertRaisesRegex(ValueError, 'disagrees'):
      run_inference_trace.validate_public_paired_target(
          public, 0.21, 0.4
      )

  def test_resume_artifact_rejects_configuration_mismatch(self):
    path = self.root / 'result.json'
    run_inference_trace._write_atomic(  # pylint: disable=protected-access
        path,
        {'status': 'complete', 'fingerprint': 'expected'},
    )
    self.assertIsNotNone(
        run_inference_trace._load_completed(  # pylint: disable=protected-access
            path, 'expected'
        )
    )
    with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
      run_inference_trace._load_completed(  # pylint: disable=protected-access
          path, 'different'
      )


if __name__ == '__main__':
  unittest.main()
