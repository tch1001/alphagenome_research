# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses

from absl.testing import absltest
from alphagenome.data import track_data
from alphagenome.models import dna_model as public_dna_model
from alphagenome_research.model import attention
from alphagenome_research.model import dna_model
from alphagenome_research.model import interpretability
from alphagenome_research.model.metadata import metadata as metadata_lib
from alphagenome_research.model import model
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


class InterpretabilityTest(absltest.TestCase):

  def _tower_selection(self):
    return interpretability.TransformerTraceSelection(
        pair_bias_edges=interpretability.PairBiasEdgeSelection(
            query_bins=jnp.array([0, 0], jnp.int32),
            key_bins=jnp.array([0, 0], jnp.int32),
            valid_mask=jnp.array([True, False]),
        ),
        head_output_positions=interpretability.HeadOutputSelection(
            positions=jnp.array([0, 7, 0], jnp.int32),
            valid_mask=jnp.array([True, True, False]),
        ),
        residual_positions=interpretability.SequenceResidualSelection(
            positions=jnp.array([0, 7, 0], jnp.int32),
            valid_mask=jnp.array([True, True, False]),
        ),
    )

  def test_attention_bias_trace_is_noop_and_preserves_parameter_tree(self):
    pair_x = jax.random.normal(jax.random.key(1), (1, 2, 2, 16))
    selection = interpretability.PairBiasEdgeSelection(
        query_bins=jnp.array([0, 1, 0], jnp.int32),
        key_bins=jnp.array([1, 0, 0], jnp.int32),
        valid_mask=jnp.array([True, True, False]),
    )

    def normal(x):
      return attention.AttentionBiasBlock()(x, is_training=False)

    def traced(x, edges):
      return attention.AttentionBiasBlock().forward_with_intermediates(
          x, is_training=False, edge_selection=edges
      )

    normal_fn = hk.transform_with_state(normal)
    traced_fn = hk.transform_with_state(traced)
    rng = jax.random.key(2)
    normal_params, normal_state = normal_fn.init(rng, pair_x)
    traced_params, traced_state = traced_fn.init(rng, pair_x, selection)

    jax.tree.map(
        np.testing.assert_array_equal, normal_params, traced_params
    )
    jax.tree.map(np.testing.assert_array_equal, normal_state, traced_state)
    normal_bias, _ = normal_fn.apply(
        normal_params, normal_state, None, pair_x
    )
    (traced_bias, trace), _ = traced_fn.apply(
        normal_params, normal_state, None, pair_x, selection
    )

    np.testing.assert_array_equal(normal_bias, traced_bias)
    chex.assert_shape(trace.compact_edges, (1, 3, 8))
    np.testing.assert_array_equal(
        trace.compact_edges, trace.effective_compact_edges
    )
    np.testing.assert_array_equal(trace.compact_edges[:, 2], 0)

  def test_pair_bias_replacement_is_dynamic_and_head_specific(self):
    pair_x = jnp.ones((1, 2, 2, 16), jnp.float32)
    selection = interpretability.PairBiasEdgeSelection(
        query_bins=jnp.array([0, 0], jnp.int32),
        key_bins=jnp.array([1, 0], jnp.int32),
        valid_mask=jnp.array([True, False]),
    )
    values = jnp.full((1, 2, 8), 7.0, jnp.float32)
    replace_mask = jnp.zeros((1, 2, 8), jnp.bool).at[0, 0, 3].set(True)

    def traced(x, edges, replacement_values, replacement_mask):
      replacement = interpretability.PairBiasReplacement(
          selection=edges,
          values=replacement_values,
          replace_mask=replacement_mask,
      )
      return attention.AttentionBiasBlock().forward_with_intermediates(
          x,
          is_training=False,
          edge_selection=edges,
          replacement=replacement,
      )

    traced_fn = hk.transform_with_state(traced)
    params, state = traced_fn.init(
        jax.random.key(3), pair_x, selection, values, replace_mask
    )
    apply = jax.jit(traced_fn.apply)
    (bias, trace), _ = apply(
        params, state, None, pair_x, selection, values, replace_mask
    )

    chex.assert_shape(bias, (1, 8, 32, 32))
    chex.assert_shape(trace.compact_edges, (1, 2, 8))
    np.testing.assert_array_equal(trace.compact_edges, 0)
    self.assertEqual(trace.effective_compact_edges[0, 0, 3], 7)
    np.testing.assert_array_equal(trace.effective_compact_edges[:, 1], 0)
    np.testing.assert_array_equal(bias[0, 3, :16, 16:], 7)
    np.testing.assert_array_equal(bias[0, :3, :16, 16:], 0)
    np.testing.assert_array_equal(bias[0, 4:, :16, 16:], 0)

  def test_mha_trace_is_noop_and_preserves_parameter_tree(self):
    x = jax.random.normal(jax.random.key(4), (1, 16, 32))
    bias = jax.random.normal(jax.random.key(5), (1, 8, 16, 16))
    selection = interpretability.HeadOutputSelection(
        positions=jnp.array([0, 3, 15, 0], jnp.int32),
        valid_mask=jnp.array([True, True, True, False]),
    )
    all_heads = jnp.ones((8,), jnp.float32)

    def normal(inputs, attention_bias):
      return attention.MHABlock()(
          inputs, attention_bias, is_training=False
      )

    def traced(inputs, attention_bias, positions, head_mask):
      return attention.MHABlock().forward_with_intermediates(
          inputs,
          attention_bias,
          is_training=False,
          head_output_selection=positions,
          head_mask=head_mask,
      )

    normal_fn = hk.transform_with_state(normal)
    traced_fn = hk.transform_with_state(traced)
    rng = jax.random.key(6)
    normal_params, normal_state = normal_fn.init(rng, x, bias)
    traced_params, traced_state = traced_fn.init(
        rng, x, bias, selection, all_heads
    )

    jax.tree.map(
        np.testing.assert_array_equal, normal_params, traced_params
    )
    jax.tree.map(np.testing.assert_array_equal, normal_state, traced_state)
    normal_output, _ = normal_fn.apply(
        normal_params, normal_state, None, x, bias
    )
    (traced_output, trace), _ = traced_fn.apply(
        normal_params,
        normal_state,
        None,
        x,
        bias,
        selection,
        all_heads,
    )

    np.testing.assert_array_equal(normal_output, traced_output)
    chex.assert_shape(trace.head_value_outputs, (1, 4, 8, 192))
    np.testing.assert_array_equal(
        trace.head_value_outputs, trace.effective_head_value_outputs
    )
    np.testing.assert_array_equal(trace.head_value_outputs[:, 3], 0)

  def test_head_mask_is_dynamic_and_changes_only_effective_trace(self):
    x = jax.random.normal(jax.random.key(7), (1, 16, 32))
    bias = jnp.zeros((1, 8, 16, 16), jnp.float32)
    selection = interpretability.HeadOutputSelection(
        positions=jnp.array([2, 11], jnp.int32),
        valid_mask=jnp.array([True, True]),
    )

    def traced(inputs, attention_bias, positions, head_mask):
      return attention.MHABlock().forward_with_intermediates(
          inputs,
          attention_bias,
          is_training=False,
          head_output_selection=positions,
          head_mask=head_mask,
      )

    traced_fn = hk.transform_with_state(traced)
    ones = jnp.ones((8,), jnp.float32)
    params, state = traced_fn.init(
        jax.random.key(8), x, bias, selection, ones
    )
    apply = jax.jit(traced_fn.apply)
    (baseline_output, baseline_trace), _ = apply(
        params, state, None, x, bias, selection, ones
    )
    mask = ones.at[5].set(0)
    (ablated_output, ablated_trace), _ = apply(
        params, state, None, x, bias, selection, mask
    )

    np.testing.assert_array_equal(
        baseline_trace.head_value_outputs, ablated_trace.head_value_outputs
    )
    np.testing.assert_array_equal(
        ablated_trace.effective_head_value_outputs[:, :, 5], 0
    )
    np.testing.assert_array_equal(
        ablated_trace.effective_head_value_outputs[:, :, :5],
        baseline_trace.head_value_outputs[:, :, :5],
    )
    np.testing.assert_array_equal(
        ablated_trace.effective_head_value_outputs[:, :, 6:],
        baseline_trace.head_value_outputs[:, :, 6:],
    )
    self.assertFalse(np.array_equal(baseline_output, ablated_output))

  def test_local_head_value_replacement_is_before_output_projection(self):
    x = jax.random.normal(jax.random.key(13), (1, 16, 32))
    bias = jnp.zeros((1, 8, 16, 16), jnp.float32)
    selection = interpretability.HeadOutputSelection(
        positions=jnp.array([2, 11], jnp.int32),
        valid_mask=jnp.array([True, True]),
    )

    def traced(
        inputs,
        attention_bias,
        positions,
        head_mask,
        replacement_values,
        replacement_mask,
    ):
      return attention.MHABlock().forward_with_intermediates(
          inputs,
          attention_bias,
          is_training=False,
          head_output_selection=positions,
          head_mask=head_mask,
          head_output_replacement_values=replacement_values,
          head_output_replace_mask=replacement_mask,
      )

    traced_fn = hk.transform_with_state(traced)
    all_heads = jnp.ones((8,), jnp.float32)
    empty_values = jnp.zeros((1, 2, 8, 192), jnp.float32)
    empty_mask = jnp.zeros((1, 2, 8), jnp.bool)
    params, state = traced_fn.init(
        jax.random.key(14),
        x,
        bias,
        selection,
        all_heads,
        empty_values,
        empty_mask,
    )
    apply = jax.jit(traced_fn.apply)
    (baseline_output, baseline_trace), _ = apply(
        params,
        state,
        None,
        x,
        bias,
        selection,
        all_heads,
        empty_values,
        empty_mask,
    )
    replacement_values = baseline_trace.head_value_outputs.at[
        0, 0, 5
    ].set(7)
    replacement_mask = empty_mask.at[0, 0, 5].set(True)
    (patched_output, patched_trace), _ = apply(
        params,
        state,
        None,
        x,
        bias,
        selection,
        all_heads,
        replacement_values,
        replacement_mask,
    )

    np.testing.assert_array_equal(
        patched_trace.head_value_outputs, baseline_trace.head_value_outputs
    )
    np.testing.assert_array_equal(
        patched_trace.effective_head_value_outputs[0, 0, 5], 7
    )
    np.testing.assert_array_equal(
        patched_trace.effective_head_value_outputs[0, 1],
        baseline_trace.head_value_outputs[0, 1],
    )
    self.assertFalse(np.array_equal(baseline_output, patched_output))

  def test_local_head_value_replacement_is_dynamic_and_padding_safe(self):
    head_outputs = jnp.arange(1 * 5 * 3 * 4, dtype=jnp.float32).reshape(
        1, 5, 3, 4
    )
    selection = interpretability.HeadOutputSelection(
        positions=jnp.array([1, 4, 99], jnp.int32),
        valid_mask=jnp.array([True, True, False]),
    )

    @jax.jit
    def replace(inputs, values, replace_mask):
      return interpretability.replace_head_value_outputs(
          inputs, selection, values, replace_mask
      )

    values = jnp.full((1, 3, 3, 4), 23.0)
    no_mask = jnp.zeros((1, 3, 3), jnp.bool)
    np.testing.assert_array_equal(
        replace(head_outputs, values, no_mask), head_outputs
    )
    mask = no_mask.at[0, 0, 2].set(True).at[0, 2, 0].set(True)
    replaced = replace(head_outputs, values, mask)
    np.testing.assert_array_equal(replaced[0, 1, 2], 23)
    np.testing.assert_array_equal(replaced[0, 1, :2], head_outputs[0, 1, :2])
    np.testing.assert_array_equal(replaced[0, 4], head_outputs[0, 4])
    np.testing.assert_array_equal(replaced[0, 0], head_outputs[0, 0])

  def test_sequence_residual_replacement_is_dynamic_and_padding_safe(self):
    residuals = jnp.arange(1 * 5 * 4, dtype=jnp.float32).reshape(1, 5, 4)
    selection = interpretability.SequenceResidualSelection(
        positions=jnp.array([1, 4, 99], jnp.int32),
        valid_mask=jnp.array([True, True, False]),
    )

    @jax.jit
    def replace(inputs, values, replace_mask):
      return interpretability.replace_sequence_residuals(
          inputs, selection, values, replace_mask
      )

    values = jnp.full((1, 3, 4), 17.0)
    no_mask = jnp.zeros((1, 3), jnp.bool)
    np.testing.assert_array_equal(
        replace(residuals, values, no_mask), residuals
    )
    mask = jnp.array([[True, False, True]])
    replaced = replace(residuals, values, mask)
    np.testing.assert_array_equal(replaced[:, 1], 17)
    np.testing.assert_array_equal(replaced[:, 4], residuals[:, 4])
    np.testing.assert_array_equal(replaced[:, 0], residuals[:, 0])
    captured = interpretability.gather_sequence_residuals(
        replaced, selection
    )
    chex.assert_shape(captured, (1, 3, 4))
    np.testing.assert_array_equal(captured[:, 2], 0)

  def test_live_batch_residual_transfer_is_exact_for_bf16_self_controls(self):
    row_a = jnp.arange(5 * 4, dtype=jnp.bfloat16).reshape(5, 4)
    row_b = (row_a + jnp.asarray(100, jnp.bfloat16)).astype(jnp.bfloat16)
    residuals = jnp.stack([row_a, row_a, row_b, row_b])
    selection = interpretability.SequenceResidualSelection(
        # The duplicate valid position exercises deterministic masked updates.
        positions=jnp.array([1, 3, 1, 99], jnp.int32),
        valid_mask=jnp.array([True, True, True, False]),
    )
    donors = jnp.broadcast_to(
        jnp.arange(4, dtype=jnp.int32)[:, None], (4, 4)
    )
    mask = jnp.zeros((4, 4), jnp.bool)
    # Same-input donor 0 -> recipient 1 must be an exact self no-op.
    donors = donors.at[1, 0].set(0)
    mask = mask.at[1, 0].set(True)
    # A real cross transfer copies row A into row B at position 3.
    donors = donors.at[3, 1].set(0)
    mask = mask.at[3, 1].set(True)
    # An enabled padded slot and an invalid donor both fail safely as no-ops.
    donors = donors.at[2, 3].set(0).at[2, 2].set(99)
    mask = mask.at[2, 3].set(True).at[2, 2].set(True)

    transferred = jax.jit(
        interpretability.transfer_sequence_residuals_within_batch
    )(residuals, selection, donors, mask)

    self.assertEqual(transferred.dtype, jnp.bfloat16)
    np.testing.assert_array_equal(transferred[1], residuals[1])
    np.testing.assert_array_equal(transferred[3, 3], residuals[0, 3])
    np.testing.assert_array_equal(transferred[3, 1], residuals[3, 1])
    np.testing.assert_array_equal(transferred[2], residuals[2])

  def test_transformer_tower_stacks_traces_and_preserves_parameter_tree(self):
    one_x = jax.random.normal(jax.random.key(9), (1, 16, 32))
    x = jnp.repeat(one_x, 2, axis=0)
    selection = self._tower_selection()
    no_interventions = interpretability.no_transformer_interventions(
        batch_size=2, num_edges=2
    )

    def normal(inputs):
      return model.TransformerTower()(inputs, is_training=False)

    def traced(inputs, trace_selection, interventions):
      return model.TransformerTower().forward_with_intermediates(
          inputs,
          is_training=False,
          trace_selection=trace_selection,
          interventions=interventions,
      )

    normal_fn = hk.transform_with_state(normal)
    traced_fn = hk.transform_with_state(traced)
    rng = jax.random.key(10)
    normal_params, normal_state = normal_fn.init(rng, x)
    traced_params, traced_state = traced_fn.init(
        rng, x, selection, no_interventions
    )
    jax.tree.map(
        np.testing.assert_array_equal, normal_params, traced_params
    )
    jax.tree.map(np.testing.assert_array_equal, normal_state, traced_state)

    (normal_x, normal_pair), _ = normal_fn.apply(
        normal_params, normal_state, None, x
    )
    (traced_x, traced_pair, trace), _ = traced_fn.apply(
        normal_params,
        normal_state,
        None,
        x,
        selection,
        no_interventions,
    )
    np.testing.assert_array_equal(normal_x, traced_x)
    np.testing.assert_array_equal(normal_pair, traced_pair)
    chex.assert_shape(trace.compact_pair_bias_edges, (9, 2, 2, 8))
    chex.assert_shape(trace.head_value_outputs, (9, 2, 3, 8, 192))
    chex.assert_shape(trace.pre_attention_residuals, (9, 2, 3, 32))
    chex.assert_shape(trace.post_attention_residuals, (9, 2, 3, 32))
    chex.assert_shape(trace.post_mlp_residuals, (9, 2, 3, 32))
    np.testing.assert_array_equal(
        trace.compact_pair_bias_edges,
        trace.effective_compact_pair_bias_edges,
    )
    np.testing.assert_array_equal(
        trace.head_value_outputs, trace.effective_head_value_outputs
    )
    np.testing.assert_array_equal(
        trace.pre_attention_residuals,
        trace.effective_pre_attention_residuals,
    )
    np.testing.assert_array_equal(
        trace.post_attention_residuals,
        trace.effective_post_attention_residuals,
    )
    np.testing.assert_array_equal(
        trace.post_mlp_residuals, trace.effective_post_mlp_residuals
    )
    np.testing.assert_array_equal(trace.compact_pair_bias_edges[:, :, 1], 0)
    np.testing.assert_array_equal(trace.head_value_outputs[:, :, 2], 0)
    np.testing.assert_array_equal(trace.pre_attention_residuals[:, :, 2], 0)

    donor_indices = jnp.broadcast_to(
        jnp.arange(2, dtype=jnp.int32)[None, :, None], (9, 2, 3)
    ).at[4, 1, 0].set(0)
    transfer_mask = jnp.zeros((9, 2, 3), jnp.bool).at[4, 1, 0].set(
        True
    )
    live_transfer = interpretability.SequenceResidualBatchTransfer(
        donor_batch_indices=donor_indices, transfer_mask=transfer_mask
    )
    transfer_interventions = dataclasses.replace(
        no_interventions, post_attention_residual_transfer=live_transfer
    )
    (self_transferred_x, _, self_transferred_trace), _ = traced_fn.apply(
        normal_params,
        normal_state,
        None,
        x,
        selection,
        transfer_interventions,
    )
    np.testing.assert_array_equal(self_transferred_x, traced_x)
    np.testing.assert_array_equal(
        self_transferred_trace.effective_post_attention_residuals,
        trace.effective_post_attention_residuals,
    )

    empty_residual = interpretability.no_sequence_residual_replacement(
        batch_size=2, num_positions=3, hidden_size=32
    )
    pre_attention_residual = interpretability.SequenceResidualReplacement(
        values=empty_residual.values.at[1, 0, 0].set(3),
        replace_mask=empty_residual.replace_mask.at[1, 0, 0].set(True),
    )
    post_attention_residual = interpretability.SequenceResidualReplacement(
        values=empty_residual.values.at[2, 0, 1].set(4),
        replace_mask=empty_residual.replace_mask.at[2, 0, 1].set(True),
    )
    post_mlp_residual = interpretability.SequenceResidualReplacement(
        values=empty_residual.values.at[3, 0, 0].set(5),
        replace_mask=empty_residual.replace_mask.at[3, 0, 0].set(True),
    )
    empty_head_output = interpretability.no_head_value_output_replacement(
        batch_size=2, num_positions=3
    )
    head_value_output_replacement = interpretability.HeadValueOutputReplacement(
        values=empty_head_output.values.at[5, 0, 0, 6].set(6),
        replace_mask=empty_head_output.replace_mask.at[5, 0, 0, 6].set(True),
    )

    interventions = interpretability.TransformerInterventions(
        head_masks=no_interventions.head_masks.at[4, 2].set(0),
        pair_bias_values=no_interventions.pair_bias_values.at[
            6, 0, 0, 3
        ].set(2),
        pair_bias_replace_mask=no_interventions.pair_bias_replace_mask.at[
            6, 0, 0, 3
        ].set(True),
        head_value_output_replacement=head_value_output_replacement,
        pre_attention_residual=pre_attention_residual,
        post_attention_residual=post_attention_residual,
        post_mlp_residual=post_mlp_residual,
    )
    (intervened_x, _, intervened_trace), _ = traced_fn.apply(
        normal_params,
        normal_state,
        None,
        x,
        selection,
        interventions,
    )
    np.testing.assert_array_equal(
        intervened_trace.effective_head_value_outputs[4, :, :, 2], 0
    )
    self.assertEqual(
        intervened_trace.effective_compact_pair_bias_edges[6, 0, 0, 3],
        2,
    )
    np.testing.assert_array_equal(
        intervened_trace.effective_pre_attention_residuals[1, 0, 0], 3
    )
    np.testing.assert_array_equal(
        intervened_trace.effective_post_attention_residuals[2, 0, 1], 4
    )
    np.testing.assert_array_equal(
        intervened_trace.effective_post_mlp_residuals[3, 0, 0], 5
    )
    np.testing.assert_array_equal(
        intervened_trace.effective_head_value_outputs[5, 0, 0, 6], 6
    )
    self.assertFalse(np.array_equal(traced_x, intervened_x))

  def test_experimental_factory_consumes_standard_tree_for_both_backends(self):
    init, _, normal_trunk_apply, _, _ = dna_model.create_model({})
    init_dna = jax.ShapeDtypeStruct((1, 2048, 4), jnp.float32)
    organism = jax.ShapeDtypeStruct((1,), jnp.int32)
    params, state = jax.eval_shape(
        init, jax.random.key(11), init_dna, organism
    )
    selection = self._tower_selection()
    interventions = interpretability.no_transformer_interventions(
        batch_size=1, num_edges=2
    )
    normal_embeddings = jax.eval_shape(
        normal_trunk_apply, params, state, init_dna, organism
    )

    dense_apply = dna_model.create_interpretability_apply({})
    dense_embeddings, dense_trace = jax.eval_shape(
        dense_apply,
        params,
        state,
        init_dna,
        organism,
        selection,
        interventions,
    )
    chex.assert_trees_all_equal_shapes_and_dtypes(
        normal_embeddings, dense_embeddings
    )
    chex.assert_shape(dense_trace.compact_pair_bias_edges, (9, 1, 2, 8))
    chex.assert_shape(dense_trace.head_value_outputs, (9, 1, 3, 8, 192))
    chex.assert_shape(dense_trace.pre_attention_residuals, (9, 1, 3, 1536))

    pallas_apply = dna_model.create_interpretability_apply(
        {}, attention_backend=attention.ATTENTION_BACKEND_PALLAS_TILED
    )
    pallas_dna = jax.ShapeDtypeStruct((1, 8192, 4), jnp.float32)
    pallas_head_replacement = (
        interpretability.no_head_value_output_replacement(
            batch_size=1, num_positions=3
        )
    )
    pallas_interventions = interpretability.TransformerInterventions(
        head_masks=interventions.head_masks,
        pair_bias_values=interventions.pair_bias_values,
        pair_bias_replace_mask=interventions.pair_bias_replace_mask,
        head_value_output_replacement=pallas_head_replacement,
    )
    _, pallas_trace = jax.eval_shape(
        pallas_apply,
        params,
        state,
        pallas_dna,
        organism,
        selection,
        pallas_interventions,
    )
    chex.assert_shape(pallas_trace.compact_pair_bias_edges, (9, 1, 2, 8))
    chex.assert_shape(pallas_trace.head_value_outputs, (9, 1, 3, 8, 192))
    chex.assert_shape(pallas_trace.pre_attention_residuals, (9, 1, 3, 1536))

  def test_target_reducer_and_targeted_factory_use_checkpoint(self):
    predictions = jnp.arange(2 * 5 * 4, dtype=jnp.float32).reshape(2, 5, 4)
    target_selection = interpretability.TargetSelection(
        position_indices=jnp.array([1, 4, 99], jnp.int32),
        position_valid_mask=jnp.array([True, True, False]),
        track_indices=jnp.array([0, 2, -5], jnp.int32),
        track_valid_mask=jnp.array([True, True, False]),
    )
    target = interpretability.reduce_target(predictions, target_selection)
    np.testing.assert_array_equal(target.total, jnp.array([44, 124]))
    np.testing.assert_array_equal(target.mean, jnp.array([11.0, 31.0]))
    self.assertEqual(target.num_values, 4)

    track_metadata = track_data.TrackMetadata(
        pd.DataFrame({
            'name': ['track_0', 'track_1'],
            'nonzero_mean': [1.0, 1.0],
        })
    )
    metadata = {
        public_dna_model.Organism.HOMO_SAPIENS: (
            metadata_lib.AlphaGenomeOutputMetadata(atac=track_metadata)
        )
    }
    init, _, _, _, _ = dna_model.create_model(metadata)
    dna = jax.ShapeDtypeStruct((1, 2048, 4), jnp.float32)
    organism = jax.ShapeDtypeStruct((1,), jnp.int32)
    params, state = jax.eval_shape(init, jax.random.key(12), dna, organism)
    targeted_apply = dna_model.create_targeted_interpretability_apply(
        metadata,
        interpretability.TargetSpec(
            head_name='atac', prediction_key='scaled_predictions_1bp'
        ),
    )
    summary, trace = jax.eval_shape(
        targeted_apply,
        params,
        state,
        dna,
        organism,
        self._tower_selection(),
        interpretability.no_transformer_interventions(
            batch_size=1, num_edges=2
        ),
        interpretability.TargetSelection(
            position_indices=jnp.array([0, 1], jnp.int32),
            position_valid_mask=jnp.array([True, True]),
            track_indices=jnp.array([0], jnp.int32),
            track_valid_mask=jnp.array([True]),
        ),
    )
    chex.assert_shape(summary.total, (1,))
    chex.assert_shape(summary.mean, (1,))
    chex.assert_shape(trace.compact_pair_bias_edges, (9, 1, 2, 8))

  def test_paired_target_reducer_does_not_include_cross_terms(self):
    predictions = jnp.arange(2 * 5 * 4, dtype=jnp.float32).reshape(2, 5, 4)
    selection = interpretability.PairedTargetSelection(
        position_indices=jnp.array([1, 4, 99], jnp.int32),
        track_indices=jnp.array([0, 2, -5], jnp.int32),
        valid_mask=jnp.array([True, True, False]),
    )

    target = interpretability.reduce_paired_target(predictions, selection)

    np.testing.assert_array_equal(target.total, jnp.array([22, 62]))
    np.testing.assert_array_equal(target.mean, jnp.array([11.0, 31.0]))
    self.assertEqual(target.num_values, 2)

  def test_paired_targeted_factory_uses_standard_checkpoint_tree(self):
    track_metadata = track_data.TrackMetadata(
        pd.DataFrame({
            'name': ['donor', 'acceptor', 'donor', 'acceptor', 'padding'],
            'strand': ['+', '+', '-', '-', '.'],
        })
    )
    metadata = {
        public_dna_model.Organism.HOMO_SAPIENS: (
            metadata_lib.AlphaGenomeOutputMetadata(splice_sites=track_metadata)
        )
    }
    init, _, _, _, _ = dna_model.create_model(metadata)
    dna = jax.ShapeDtypeStruct((1, 2048, 4), jnp.float32)
    organism = jax.ShapeDtypeStruct((1,), jnp.int32)
    params, state = jax.eval_shape(init, jax.random.key(15), dna, organism)
    paired_apply = dna_model.create_paired_targeted_interpretability_apply(
        metadata,
        interpretability.TargetSpec(
            head_name='splice_sites_classification',
            prediction_key='predictions',
        ),
    )

    summary, trace = jax.eval_shape(
        paired_apply,
        params,
        state,
        dna,
        organism,
        self._tower_selection(),
        interpretability.no_transformer_interventions(
            batch_size=1, num_edges=2
        ),
        interpretability.PairedTargetSelection(
            position_indices=jnp.array([100, 150], jnp.int32),
            track_indices=jnp.array([1, 0], jnp.int32),
            valid_mask=jnp.array([True, True]),
        ),
    )
    chex.assert_shape(summary.total, (1,))
    self.assertEqual(summary.num_values.shape, ())
    chex.assert_shape(trace.pre_attention_residuals, (9, 1, 3, 1536))


if __name__ == '__main__':
  absltest.main()
