#!/usr/bin/env python3
"""CPU-only tests for the OpenSplice v3.3 encoder-skip runner."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import run_encoder_skip_factorial_v3_3 as runner
from alphagenome_research.model import interpretability


def _selection() -> interpretability.SupersetGraphSelection:
  return interpretability.SupersetGraphSelection(
      transformer=interpretability.TransformerTraceSelection(
          pair_bias_edges=interpretability.PairBiasEdgeSelection(
              query_bins=jnp.zeros((1,), jnp.int32),
              key_bins=jnp.zeros((1,), jnp.int32),
              valid_mask=jnp.ones((1,), jnp.bool),
          ),
          head_output_positions=interpretability.HeadOutputSelection(
              positions=jnp.zeros((1,), jnp.int32),
              valid_mask=jnp.ones((1,), jnp.bool),
          ),
          residual_positions=interpretability.SequenceResidualSelection(
              positions=jnp.arange(24, dtype=jnp.int32),
              valid_mask=jnp.ones((24,), jnp.bool),
          ),
      ),
      stage_a=interpretability.StageABranchSelection(
          final_embedding_positions=jnp.asarray([10, 20], jnp.int32),
          final_embedding_valid_mask=jnp.ones((2,), jnp.bool),
      ),
  )


def _evidence(values):
  means = jnp.asarray(values, jnp.float32)
  margins = jnp.stack((means, means), axis=1)
  padding = jnp.zeros_like(margins)
  selected = jnp.stack((margins, padding), axis=-1)
  return interpretability.SpliceClassificationLogitMarginEvidence(
      selected_logits=selected,
      margins=margins,
      target=interpretability.TargetSummary(
          total=margins.sum(axis=1),
          mean=margins.mean(axis=1),
          num_values=jnp.asarray(2, jnp.int32),
      ),
  )


def _transformer_trace(batch_size):
  pair = jnp.zeros((9, batch_size, 1, 1), jnp.bfloat16)
  head = jnp.zeros((9, batch_size, 1, 1, 1), jnp.bfloat16)
  residual = jnp.zeros((9, batch_size, 24, 1), jnp.bfloat16)
  return interpretability.TransformerTrace(
      compact_pair_bias_edges=pair,
      effective_compact_pair_bias_edges=pair,
      head_value_outputs=head,
      effective_head_value_outputs=head,
      pre_attention_residuals=residual,
      effective_pre_attention_residuals=residual,
      post_attention_residuals=residual,
      effective_post_attention_residuals=residual,
      post_mlp_residuals=residual,
      effective_post_mlp_residuals=residual,
  )


def _trace(batch_size):
  identity_rows = (
      runner.SIX_IDENTITY_ROWS
      if batch_size == 6 else runner.EIGHT_IDENTITY_ROWS
  )
  natural_final = jnp.asarray(identity_rows, jnp.bfloat16)[:, None, None]
  natural_final = jnp.broadcast_to(natural_final, (batch_size, 2, 1))
  return interpretability.SupersetGraphTrace(
      transformer=_transformer_trace(batch_size),
      stage_a=interpretability.StageABranchTrace(
          transformer_output_natural_matches_identity=jnp.ones(
              (batch_size,), jnp.bool
          ),
          transformer_output_effective_matches_natural=jnp.ones(
              (batch_size,), jnp.bool
          ),
          transformer_output_effective_matches_intervention_donor=jnp.ones(
              (batch_size,), jnp.bool
          ),
          transformer_output_natural_fingerprint=jnp.zeros(
              (batch_size, 4), jnp.uint32
          ),
          encoder_skips_natural_match_identity=jnp.ones(
              (7, batch_size), jnp.bool
          ),
          encoder_skips_effective_match_natural=jnp.ones(
              (7, batch_size), jnp.bool
          ),
          encoder_skips_effective_match_intervention_donor=jnp.ones(
              (7, batch_size), jnp.bool
          ),
          encoder_skips_natural_fingerprints=jnp.zeros(
              (7, batch_size, 4), jnp.uint32
          ),
          natural_final_embeddings=natural_final,
          effective_final_embeddings=natural_final,
      ),
  )


class EncoderSkipFactorialV33Test(unittest.TestCase):

  def test_coalition_metadata_and_execution_order_are_exact(self):
    self.assertEqual(runner.coalition_metadata(0)['enabled_players'], [])
    self.assertEqual(
        runner.coalition_metadata(255)['enabled_players'],
        list(runner.SHAPLEY_PLAYER_ORDER),
    )
    self.assertEqual(
        runner.coalition_metadata(129)['e_bits'],
        [True, False, False, False, False, False, False],
    )
    cases = tuple(SimpleNamespace(order=index) for index in range(20))
    order = runner.coalition_execution_order(cases)
    self.assertEqual(len(order), 5120)
    self.assertEqual(len(set(order)), 5120)
    self.assertEqual(order[:4], ((0, 0), (0, 255), (1, 0), (1, 255)))
    self.assertEqual(order[40], (0, 1))
    self.assertEqual(order[40 + 12 * 254], (6, 1))

  def test_all_256_six_row_coalitions_share_one_pytree(self):
    selection = _selection()
    reference = jax.tree_util.tree_structure(
        runner.coalition_interventions(selection, 0)
    )
    for coalition_id in range(256):
      interventions = runner.coalition_interventions(selection, coalition_id)
      self.assertEqual(jax.tree_util.tree_structure(interventions), reference)
      runner._assert_runtime_transfer_contract(  # pylint: disable=protected-access
          interventions,
          coalition_id,
          batch_size=6,
          donor_rows=runner.SIX_DONOR_ROWS,
          identity_rows=runner.SIX_IDENTITY_ROWS,
      )
    identity = runner.coalition_interventions(selection, 0)
    bad_final = dataclasses.replace(
        identity,
        stage_a=dataclasses.replace(
            identity.stage_a,
            final_embedding=dataclasses.replace(
                identity.stage_a.final_embedding,
                donor_batch_indices=(
                    identity.stage_a.final_embedding.donor_batch_indices
                    .at[0, 0, 0].set(1)
                ),
            ),
        ),
    )
    with self.assertRaisesRegex(ValueError, 'Final-embedding'):
      runner._assert_runtime_transfer_contract(  # pylint: disable=protected-access
          bad_final,
          0,
          batch_size=6,
          donor_rows=runner.SIX_DONOR_ROWS,
          identity_rows=runner.SIX_IDENTITY_ROWS,
      )

  def test_eight_row_intended_and_ood_maps_are_exact_fixed_pytrees(self):
    selection = _selection()
    intended = runner.eight_row_interventions(
        selection, 255, unrelated=False
    )
    unrelated = runner.eight_row_interventions(
        selection, 255, unrelated=True
    )
    self.assertEqual(
        jax.tree_util.tree_structure(intended),
        jax.tree_util.tree_structure(unrelated),
    )
    for value, donors in (
        (intended, runner.EIGHT_INTENDED_DONOR_ROWS),
        (unrelated, runner.EIGHT_UNRELATED_DONOR_ROWS),
    ):
      runner._assert_runtime_transfer_contract(  # pylint: disable=protected-access
          value,
          255,
          batch_size=8,
          donor_rows=donors,
          identity_rows=runner.EIGHT_IDENTITY_ROWS,
      )
    np.testing.assert_array_equal(
        np.asarray(unrelated.stage_a.encoder_skips.donor_batch_indices)[0],
        np.asarray(runner.EIGHT_UNRELATED_DONOR_ROWS),
    )
    np.testing.assert_array_equal(
        np.asarray(unrelated.stage_a.encoder_skips.transfer_mask)[:, [0, 1, 6, 7]],
        False,
    )

  def test_coalition_validator_checks_id0_id255_and_runtime_arrays(self):
    identity_evidence = _evidence((1, 3, 3, 3, 1, 1))
    trace = _trace(6)
    identity = {
        'target_readout': runner.target_readout(identity_evidence, batch_size=6),
        # pylint: disable=protected-access
        'natural_route_fingerprints': (
            runner._natural_route_fingerprints(trace)
        ),
    }
    id0 = runner.coalition_interventions(_selection(), 0)
    checks = runner.validate_coalition(
        (identity_evidence, trace),
        (identity_evidence, trace),
        identity,
        0,
        id0,
    )
    self.assertTrue(checks['id0_identity_endpoint_exact'])
    closed = _evidence((1, 3, 1, 3, 3, 1))
    id255 = runner.coalition_interventions(_selection(), 255)
    checks = runner.validate_coalition(
        (closed, trace), (closed, trace), identity, 255, id255
    )
    self.assertTrue(checks['id255_endpoint_closure_exact'])
    corrupted = dataclasses.replace(
        id255,
        stage_a=dataclasses.replace(
            id255.stage_a,
            encoder_skips=dataclasses.replace(
                id255.stage_a.encoder_skips,
                donor_batch_indices=(
                    id255.stage_a.encoder_skips.donor_batch_indices.at[0, 2].set(5)
                ),
            ),
        ),
    )
    with self.assertRaisesRegex(ValueError, 'donor map'):
      runner.validate_coalition(
          (closed, trace), (closed, trace), identity, 255, corrupted
      )

  def test_ood_validator_requires_repeats_self_and_both_donor_maps(self):
    intended_evidence = _evidence((1, 3, 1, 3, 3, 1, 7, 9))
    unrelated_evidence = _evidence((1, 3, 7, 3, 9, 1, 7, 9))
    trace = _trace(8)
    intended_interventions = runner.eight_row_interventions(
        _selection(), 255, unrelated=False
    )
    unrelated_interventions = runner.eight_row_interventions(
        _selection(), 255, unrelated=True
    )
    checks = runner.validate_ood_anchor(
        (intended_evidence, trace),
        (intended_evidence, trace),
        (unrelated_evidence, trace),
        (unrelated_evidence, trace),
        255,
        intended_interventions,
        unrelated_interventions,
    )
    self.assertTrue(checks['unrelated_target_repeat_exact'])
    self.assertTrue(checks['id255_unrelated_endpoint_closure_exact'])
    drifted = _evidence((1, 3, 7, 4, 9, 1, 7, 9))
    with self.assertRaisesRegex(ValueError, 'row'):
      runner.validate_ood_anchor(
          (intended_evidence, trace),
          (intended_evidence, trace),
          (drifted, trace),
          (drifted, trace),
          255,
          intended_interventions,
          unrelated_interventions,
      )

  def test_readout_and_dry_plan_freeze_raw_schema_and_counts(self):
    evidence = _evidence((1, 3, 3, 3, 1, 1, 7, 9))
    readout = runner.target_readout(evidence, batch_size=8)
    self.assertEqual(np.asarray(readout['selected_logits']).shape, (8, 2, 2))
    self.assertEqual(np.asarray(readout['endpoint_margins']).shape, (8, 2))
    plan = runner.build_dry_run_plan(
        tuple(range(20)), max_variants=1, max_coalitions=2
    )
    self.assertEqual(plan['scientific_record_count'], 5220)
    self.assertEqual(plan['model_apply_count'], 10600)
    self.assertEqual(plan['total_compile_count'], 2)
    self.assertEqual(plan['confirmation_model_calls'], 0)
    movement = runner.raw_bidirectional_movement({
        'means': [1.0, 3.0, 7.0, 3.0, 9.0, 1.0, 7.0, 9.0]
    })
    self.assertEqual(movement, {
        'reference_into_alternate': 4.0,
        'alternate_into_reference': 8.0,
    })
    self.assertNotIn('recovery', movement)
    self.assertNotIn('bidirectional_bottleneck', movement)

  def test_runtime_manifest_is_fail_closed_for_critical_packages(self):
    current = runner.runtime_version_binding()
    physical = {
        'compute_capability': '8.6',
        'driver': '560.35.05',
        'index': '0',
        'name': 'NVIDIA GeForce RTX 3090',
        'uuid': runner.EXPECTED_GPU_UUID,
        'vbios': '94.02.42.C0.05',
    }
    frozen = {'runtime_version_manifest': {**current, 'nvidia_smi': physical}}
    observation = {
        'packages': {
            name: current['packages'][name]
            for name in runner.RUNTIME_PACKAGES
        },
        'jax_module_version': current['packages']['jax'],
        'jaxlib_module_version': current['packages']['jaxlib'],
        'python_version': current['python_version'],
        'platform': current['platform'],
        'kernel': current['kernel'],
        'nvidia_smi': {'parsed_single_gpu': physical},
    }
    runner.validate_device_version_manifest(observation, frozen)
    frozen['runtime_version_manifest']['packages']['numpy'] = 'wrong'
    with self.assertRaisesRegex(ValueError, 'critical numerical runtime'):
      runner.validate_device_version_manifest(observation, frozen)


if __name__ == '__main__':
  unittest.main()
