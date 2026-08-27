#!/usr/bin/env python3
"""CPU-only unit tests for the OpenSplice v3.2 superset runner."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

import run_superset_graph_v3_2 as runner
from alphagenome_research.model import interpretability


def _evidence(values=(1.0, 3.0, 3.0, 3.0, 1.0, 1.0)):
  means = jnp.asarray(values, jnp.float32)
  margins = jnp.stack((means - 0.25, means + 0.25), axis=1)
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


def _transformer_trace():
  natural_pair = jnp.zeros((9, 6, 1, 1), jnp.bfloat16)
  natural_head = jnp.zeros((9, 6, 1, 1, 1), jnp.bfloat16)
  base = jnp.asarray(runner.NATURAL_IDENTITY_ROWS, jnp.bfloat16)
  natural_residual = jnp.broadcast_to(
      base[None, :, None, None], (9, 6, runner.v2.NUM_TRACE_SLOTS, 1)
  )
  effective_pre = natural_residual.at[0, 2, 0].set(
      natural_residual[0, 0, 0]
  ).at[0, 4, 0].set(natural_residual[0, 1, 0])
  return interpretability.TransformerTrace(
      compact_pair_bias_edges=natural_pair,
      effective_compact_pair_bias_edges=natural_pair,
      head_value_outputs=natural_head,
      effective_head_value_outputs=natural_head,
      pre_attention_residuals=natural_residual,
      effective_pre_attention_residuals=effective_pre,
      post_attention_residuals=natural_residual,
      effective_post_attention_residuals=natural_residual,
      post_mlp_residuals=natural_residual,
      effective_post_mlp_residuals=natural_residual,
  )


def _stage_trace(*, final_live=False):
  base = jnp.asarray(runner.NATURAL_IDENTITY_ROWS, jnp.bfloat16)
  natural_final = jnp.broadcast_to(base[:, None, None], (6, 2, 1))
  effective_final = natural_final
  if final_live:
    effective_final = effective_final.at[2].set(natural_final[0])
    effective_final = effective_final.at[4].set(natural_final[1])
  return interpretability.StageABranchTrace(
      transformer_output_natural_matches_identity=jnp.ones((6,), jnp.bool),
      transformer_output_effective_matches_natural=jnp.ones((6,), jnp.bool),
      transformer_output_effective_matches_intervention_donor=(
          jnp.ones((6,), jnp.bool)
      ),
      transformer_output_natural_fingerprint=jnp.zeros((6, 4), jnp.uint32),
      encoder_skips_natural_match_identity=jnp.ones((7, 6), jnp.bool),
      encoder_skips_effective_match_natural=jnp.ones((7, 6), jnp.bool),
      encoder_skips_effective_match_intervention_donor=(
          jnp.ones((7, 6), jnp.bool)
      ),
      encoder_skips_natural_fingerprints=jnp.zeros((7, 6, 4), jnp.uint32),
      natural_final_embeddings=natural_final,
      effective_final_embeddings=effective_final,
  )


class SupersetGraphRunnerTest(unittest.TestCase):

  def test_development_projection_is_exact_and_does_not_call_full_loader(self):
    with mock.patch.object(
        runner.route_v3, 'load_development_cases',
        side_effect=AssertionError('full loader must not run'),
    ):
      cases = runner.load_development_cases()
    self.assertEqual(tuple(case.order for case in cases), tuple(range(20)))
    self.assertEqual({case.gene for case in cases}, {'BRAF', 'SLC25A48'})
    self.assertEqual(sum(case.is_effect for case in cases), 12)

  def test_target_readout_serializes_exact_endpoint_algebra(self):
    evidence = _evidence()
    readout = runner.target_readout(evidence)
    self.assertEqual(readout['endpoint_axis'], ['acceptor', 'donor'])
    self.assertEqual(
        readout['selected_logit_axis'],
        ['relevant_class', 'padding_class'],
    )
    self.assertEqual(np.asarray(readout['selected_logits']).shape, (6, 2, 2))
    self.assertEqual(np.asarray(readout['endpoint_margins']).shape, (6, 2))
    self.assertEqual(np.asarray(readout['means']).shape, (6,))
    self.assertEqual(readout['num_values'], 2)
    np.testing.assert_array_equal(
        np.asarray(readout['selected_logits'])[..., 0]
        - np.asarray(readout['selected_logits'])[..., 1],
        np.asarray(readout['endpoint_margins']),
    )

  def test_all_intervention_families_have_one_pytree_signature(self):
    case = runner.load_development_cases()[0]
    interval = runner.v2.centered_interval(case, runner.route_v3.CONTEXT_BP)
    positions = runner.v2.trace_position_sets(case, interval)
    resolved = SimpleNamespace(
        endpoints=(
            SimpleNamespace(role='acceptor', position_index=10),
            SimpleNamespace(role='donor', position_index=20),
        ),
        padding_track_index=4,
    )
    selection = runner.superset_selection(positions, resolved)
    identity = runner.identity_interventions(selection)
    signature = runner.pytree_signature(identity)
    for group in runner.phase_r.enumerate_groups(case, interval):
      runner.assert_same_program_signature(
          signature, runner.phase_r_interventions(selection, group)
      )
    for component in runner.stage_a.enumerate_components():
      runner.assert_same_program_signature(
          signature, runner.stage_a_interventions(selection, component)
      )
    self.assertIsNotNone(
        identity.transformer.pre_attention_residual_transfer
    )
    self.assertEqual(
        identity.transformer.pre_attention_residual_transfer.transfer_mask.shape,
        (9, 6, runner.v2.NUM_TRACE_SLOTS),
    )

  def test_dry_plan_freezes_one_compile_and_complete_grid(self):
    cases = runner.load_development_cases()
    plan = runner.build_dry_run_plan(cases, max_variants=1, max_groups=2)
    self.assertEqual(plan['development_case_count'], 20)
    self.assertEqual(plan['identity_calls'], 40)
    self.assertEqual(plan['phase_r_groups_per_eligible_effect'], 216)
    self.assertEqual(plan['phase_r_candidates'], 72)
    self.assertEqual(plan['compile_count'], 1)
    self.assertEqual(plan['confirmation_model_calls'], 0)

  def test_environment_gate_records_prefixes_and_rejects_cache_inputs(self):
    clean = {
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
        'JAX_ENABLE_COMPILATION_CACHE': 'false',
        'CUDA_VISIBLE_DEVICES': '0',
    }
    fake_jax = SimpleNamespace(
        config=SimpleNamespace(jax_enable_compilation_cache=False)
    )
    with mock.patch.dict(os.environ, clean, clear=True), mock.patch.object(
        runner.device_gate, 'assert_sanitized_environment'
    ), mock.patch.object(runner, 'jax', fake_jax):
      observed = runner.assert_v3_2_environment()
    self.assertEqual(observed['JAX_ENABLE_COMPILATION_CACHE'], 'false')
    with mock.patch.dict(
        os.environ, {**clean, 'XLA_FLAGS': '--bad'}, clear=True
    ), mock.patch.object(
        runner.device_gate, 'assert_sanitized_environment'
    ), mock.patch.object(runner, 'jax', fake_jax):
      with self.assertRaisesRegex(ValueError, 'Forbidden'):
        runner.assert_v3_2_environment()

  def test_confirmation_named_paths_fail_closed(self):
    with self.assertRaisesRegex(ValueError, 'Confirmation-named'):
      runner._reject_confirmation_path(  # pylint: disable=protected-access
          Path('/tmp/confirmation/data.json')
      )

  def test_recovery_uses_self_corrected_within_call_denominators(self):
    result = runner.recovery_statistics((1, 3, 1, 3, 3, 1))
    self.assertEqual(result['recovery']['reference_into_alternate'], 1.0)
    self.assertEqual(result['recovery']['alternate_into_reference'], 1.0)
    self.assertEqual(result['recovery']['bidirectional_bottleneck'], 1.0)

  def test_phase_r_requires_active_seam_natural_same_allele(self):
    case = runner.load_development_cases()[0]
    interval = runner.v2.centered_interval(case, runner.route_v3.CONTEXT_BP)
    group = runner.phase_r.enumerate_groups(case, interval)[0]
    evidence = _evidence()
    identity_readout = runner.target_readout(evidence)
    trace = interpretability.SupersetGraphTrace(
        transformer=_transformer_trace(), stage_a=_stage_trace()
    )
    checks = runner.validate_phase_r_group(
        evidence, trace, evidence, trace, identity_readout, group
    )
    self.assertTrue(checks['active_seam_natural_same_allele_exact'])
    natural = trace.transformer.pre_attention_residuals.at[
        group.layer, 2, jnp.asarray(group.position_set.slots)
    ].set(jnp.bfloat16(9))
    drifted = dataclasses.replace(
        trace,
        transformer=dataclasses.replace(
            trace.transformer, pre_attention_residuals=natural
        ),
    )
    with self.assertRaisesRegex(ValueError, 'natural same-allele'):
      runner.validate_phase_r_group(
          evidence, drifted, evidence, drifted, identity_readout, group
      )
    effective = trace.transformer.effective_pre_attention_residuals.at[
        group.layer, 0, jnp.asarray(group.position_set.slots)
    ].set(jnp.bfloat16(8))
    baseline_drift = dataclasses.replace(
        trace,
        transformer=dataclasses.replace(
            trace.transformer, effective_pre_attention_residuals=effective
        ),
    )
    with self.assertRaisesRegex(ValueError, 'baseline rows'):
      runner.validate_phase_r_group(
          evidence, baseline_drift, evidence, baseline_drift,
          identity_readout, group
      )

  def test_final_closure_requires_natural_route_and_endpoint_equality(self):
    identity = _evidence()
    active = _evidence((1, 3, 1, 3, 3, 1))
    trace = interpretability.SupersetGraphTrace(
        transformer=dataclasses.replace(
            _transformer_trace(),
            effective_pre_attention_residuals=(
                _transformer_trace().pre_attention_residuals
            ),
        ),
        stage_a=_stage_trace(final_live=True),
    )
    component = runner.stage_a.enumerate_components()[0]
    checks = runner.validate_stage_a_group(
        active, trace, active, trace, runner.target_readout(identity), component
    )
    self.assertTrue(checks['endpoint_level_closure_exact'])
    selected = active.selected_logits.at[2, 0, 0].add(0.5).at[
        2, 1, 0
    ].add(-0.5)
    altered = dataclasses.replace(
        active,
        selected_logits=selected,
        margins=selected[..., 0] - selected[..., 1],
    )
    with self.assertRaisesRegex(ValueError, 'raw endpoint logits'):
      runner.validate_stage_a_group(
          altered, trace, altered, trace,
          runner.target_readout(identity), component
      )
    effective_final = trace.stage_a.effective_final_embeddings.at[0].set(
        jnp.bfloat16(7)
    )
    baseline_drift = dataclasses.replace(
        trace,
        stage_a=dataclasses.replace(
            trace.stage_a, effective_final_embeddings=effective_final
        ),
    )
    with self.assertRaisesRegex(ValueError, 'baseline rows'):
      runner.validate_stage_a_group(
          active, baseline_drift, active, baseline_drift,
          runner.target_readout(identity), component
      )

  def test_all_four_stage_a_components_use_six_row_controls(self):
    identity = _evidence()
    closure = _evidence((1, 3, 1, 3, 3, 1))
    partial = _evidence((1, 3, 2, 3, 2, 1))
    transformer = dataclasses.replace(
        _transformer_trace(),
        effective_pre_attention_residuals=(
            _transformer_trace().pre_attention_residuals
        ),
    )
    for component in runner.stage_a.enumerate_components():
      evidence = closure if component.closure_required else partial
      trace = interpretability.SupersetGraphTrace(
          transformer=transformer,
          stage_a=_stage_trace(final_live=component.final_embedding),
      )
      checks = runner.validate_stage_a_group(
          evidence, trace, evidence, trace,
          runner.target_readout(identity), component
      )
      self.assertTrue(checks['passed'])
      self.assertTrue(checks['baseline_targets_exact_from_identity'])
      self.assertTrue(checks['self_targets_exact'])

  def test_component_order_and_controlled_stop_schema_are_frozen(self):
    components = runner.stage_a.enumerate_components()
    self.assertEqual(
        tuple(component.key for component in components),
        (
            '00_final_embedding_A_D_closure',
            '01_joint_T_plus_E_closure',
            '02_whole_T',
            '03_whole_E',
        ),
    )
    with tempfile.TemporaryDirectory() as temporary:
      output = Path(temporary) / 'run'
      with mock.patch.object(runner, 'OUTPUT_DIR', output):
        for filename in (
            'IMPORT_PROVENANCE_PRE_MODEL.json',
            'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
            'IMPORT_PROVENANCE.json',
        ):
          runner._write_new(  # pylint: disable=protected-access
              output / filename, {'module_count': 0, 'modules': []}
          )
        runner._write_new(  # pylint: disable=protected-access
            output / 'PROTOBUF_PROVENANCE.json', {'test': True}
        )
        result = runner._write_completion(  # pylint: disable=protected-access
            stop_reason='identity_tooling_failure',
            message='synthetic stop',
            identity_count=20,
            eligible_effect_count=0,
            phase_results=(),
            stage_results=(),
            closures_passed=None,
            compiler={'compile_count': 1, 'executable_fingerprint': 'abc'},
        )
      self.assertEqual(result['status'], 'controlled_stop')
      self.assertEqual(result['phase_r_group_count'], 0)
      self.assertEqual(result['stage_a_group_count'], 0)
      self.assertEqual(result['confirmation_model_calls'], 0)


if __name__ == '__main__':
  unittest.main()
