"""CPU tests for the development-only OpenSplice v3 route runner."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import jax.numpy as jnp
import numpy as np
import pandas as pd


_MODULE_PATH = Path(__file__).with_name('run_route_census_v3.py')
_SPEC = importlib.util.spec_from_file_location('run_route_census_v3', _MODULE_PATH)
runner = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)


def _synthetic_trace():
  """Makes compact traces with exact six-row duplicate/self semantics."""
  natural_rows = jnp.asarray([1, 2, 2, 2, 1, 1], jnp.float32)
  transformer_leaf = jnp.broadcast_to(
      natural_rows[None, :, None, None], (9, 6, 3, 1)
  )
  transformer = runner.interpretability.TransformerTrace(
      compact_pair_bias_edges=transformer_leaf,
      effective_compact_pair_bias_edges=transformer_leaf,
      head_value_outputs=transformer_leaf,
      effective_head_value_outputs=transformer_leaf,
      pre_attention_residuals=transformer_leaf,
      effective_pre_attention_residuals=transformer_leaf,
      post_attention_residuals=transformer_leaf,
      effective_post_attention_residuals=transformer_leaf,
      post_mlp_residuals=transformer_leaf,
      effective_post_mlp_residuals=transformer_leaf,
  )
  route_leaf = jnp.broadcast_to(natural_rows[:, None, None], (6, 3, 1))
  return runner.interpretability.CausalRouteTrace(
      transformer=transformer,
      encoder_outputs=(route_leaf,) * 8,
      effective_encoder_outputs=(route_leaf,) * 8,
      decoder_skip_states=(route_leaf,) * 7,
      effective_decoder_skip_states=(route_leaf,) * 7,
      decoder_outputs=(route_leaf,) * 7,
      effective_decoder_outputs=(route_leaf,) * 7,
      final_embeddings=(route_leaf,) * 2,
      effective_final_embeddings=(route_leaf,) * 2,
  )


def _target(values=(1, 2, 2, 2, 1, 1)):
  mean = jnp.asarray(values, jnp.float32)
  return runner.interpretability.TargetSummary(
      total=mean * 2, mean=mean, num_values=jnp.asarray(2, jnp.int32)
  )


class RouteCensusV3Test(unittest.TestCase):

  def test_loader_is_exact_development_allowlist(self):
    cases = runner.load_development_cases()
    self.assertEqual(len(cases), 20)
    self.assertEqual({case.gene for case in cases}, {'BRAF', 'SLC25A48'})
    self.assertEqual(sum(case.is_effect for case in cases), 12)
    self.assertEqual(sum(not case.is_effect for case in cases), 8)
    self.assertEqual({case.order for case in cases}, set(range(20)))

  def test_component_catalog_has_all_51_and_late_transformers(self):
    components = runner.enumerate_components()
    self.assertEqual(len(components), 51)
    self.assertEqual([component.order for component in components], list(range(51)))
    for family in (
        'transformer_pre_attention',
        'transformer_post_attention',
        'transformer_post_mlp',
    ):
      self.assertEqual(
          {component.stage for component in components if component.family == family},
          set(range(9)),
      )

  def test_route_positions_are_fixed_padded_at_every_resolution(self):
    case = runner.load_development_cases()[4]  # exonic site: normally 3 distinct bp.
    interval = runner.v2.centered_interval(case, runner.CONTEXT_BP)
    selection = runner.route_selection(case, interval)
    self.assertEqual(selection.encoder_positions.shape, (8, 3))
    self.assertEqual(selection.decoder_skip_positions.shape, (7, 3))
    self.assertEqual(selection.decoder_output_positions.shape, (7, 3))
    self.assertEqual(selection.final_embedding_positions.shape, (2, 3))
    self.assertEqual(
        selection.transformer.residual_positions.positions.shape, (3,)
    )
    self.assertTrue(np.asarray(selection.encoder_valid_mask[0]).all())
    self.assertLessEqual(
        np.asarray(selection.encoder_valid_mask[-1]).sum(), 3
    )

  def test_strand_aware_logit_target_is_acceptor_then_donor(self):
    metadata = pd.DataFrame({
        'name': ['donor', 'acceptor', 'donor', 'acceptor', 'padding'],
        'strand': ['+', '+', '-', '-', '.'],
    })
    braf = runner.load_development_cases()[0]
    interval = runner.v2.centered_interval(braf, runner.CONTEXT_BP)
    selection, resolved = runner.target_selection(metadata, braf, interval)
    self.assertEqual([endpoint.role for endpoint in resolved.endpoints], [
        'acceptor', 'donor'
    ])
    np.testing.assert_array_equal(selection.canonical_track_indices, [3, 2])
    self.assertEqual(int(selection.padding_track_index), 4)

  def test_intervention_activates_one_component_with_frozen_donors(self):
    case = runner.load_development_cases()[4]
    interval = runner.v2.centered_interval(case, runner.CONTEXT_BP)
    selection = runner.route_selection(case, interval)
    transformer_component = next(
        component for component in runner.enumerate_components()
        if component.family == 'transformer_post_mlp' and component.stage == 8
    )
    interventions = runner.component_interventions(
        selection, transformer_component
    )
    transfer = interventions.transformer.post_mlp_residual_transfer
    mask = np.asarray(transfer.transfer_mask)
    self.assertFalse(mask[:8].any())
    self.assertFalse(mask[8, :2].any())
    self.assertTrue(mask[8, 2:, np.asarray(
        selection.transformer.residual_positions.valid_mask
    )].all())
    np.testing.assert_array_equal(
        np.asarray(transfer.donor_batch_indices[8, :, 0]),
        runner.TRACE_BATCH_DONORS,
    )
    self.assertFalse(np.asarray(
        interventions.transformer.pre_attention_residual_transfer.transfer_mask
    ).any())

  def test_identity_gate_checks_target_and_all_route_duplicates(self):
    target = _target()
    trace = _synthetic_trace()
    checks = runner.validate_identity_audit(target, trace, target, trace)
    self.assertTrue(checks['passed'])
    bad_mean = target.mean.at[2].set(3)
    bad = dataclasses.replace(target, mean=bad_mean, total=bad_mean * 2)
    with self.assertRaisesRegex(ValueError, 'ALT duplicate'):
      runner.validate_identity_audit(bad, trace, bad, trace)

  def test_component_gate_checks_targets_and_all_live_donor_vectors(self):
    trace = _synthetic_trace()
    live_rows = jnp.asarray([1, 2, 1, 2, 2, 1], jnp.float32)
    live_leaf = jnp.broadcast_to(live_rows[:, None, None], (6, 3, 1))
    effective = list(trace.effective_encoder_outputs)
    effective[0] = live_leaf
    trace = dataclasses.replace(
        trace, effective_encoder_outputs=tuple(effective)
    )
    component = runner.RouteComponent(
        order=0, family='encoder_outputs', stage=0, resolution_bp=1,
        channel_width=1, seam='test',
    )
    checks = runner.validate_component_audit(
        _target(), trace, component, (1, 2, 2, 2, 1, 1),
        (True, True, False),
    )
    self.assertTrue(checks['passed'])
    self.assertEqual(
        checks['self_control_corrected_recovery']['bidirectional_bottleneck'],
        0,
    )
    bad_effective = list(trace.effective_encoder_outputs)
    bad_effective[0] = bad_effective[0].at[2, 0, 0].set(9)
    bad_trace = dataclasses.replace(
        trace, effective_encoder_outputs=tuple(bad_effective)
    )
    with self.assertRaisesRegex(ValueError, 'donor-vector'):
      runner.validate_component_audit(
          _target(), bad_trace, component, (1, 2, 2, 2, 1, 1),
          (True, True, False),
      )

  def test_direction_gate_uses_delta_logit_sign_and_point01_threshold(self):
    case = runner.load_development_cases()[0]
    self.assertTrue(runner.direction_gate(case, [0, 0.02])[
        'eligible_for_causal_census'
    ])
    self.assertFalse(runner.direction_gate(case, [0, -0.02])[
        'eligible_for_causal_census'
    ])
    self.assertFalse(runner.direction_gate(case, [0, 0.009])[
        'eligible_for_causal_census'
    ])

  def test_dry_plan_never_exposes_confirmation_partition(self):
    plan = runner.build_dry_run_plan(
        runner.load_development_cases(), runner.enumerate_components()
    )
    self.assertEqual(plan['variant_count'], 20)
    self.assertEqual(plan['component_count'], 51)
    encoded = str(plan)
    for confirmation_gene in ('ELN', 'EIF4A2', 'DMD'):
      self.assertNotIn(confirmation_gene, encoded)

  def test_resume_fails_closed_on_fingerprint_mismatch(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'artifact.json'
      runner.v2._write_atomic(path, {  # pylint: disable=protected-access
          'status': 'complete', 'fingerprint': 'expected'
      })
      with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
        runner.v2._load_completed(path, 'changed')  # pylint: disable=protected-access


if __name__ == '__main__':
  unittest.main()
