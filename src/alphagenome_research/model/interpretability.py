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

"""Experimental primitives for causal tracing inside AlphaGenome attention.

The objects in this module are JAX pytrees, so selections and interventions can
be supplied as dynamic arguments to a jitted apply function.  Padded entries
allow callers to keep shapes fixed across loci: set ``valid_mask`` to false for
an unused entry and its captured value will be zero.

These APIs deliberately expose the compact learned attention bias and the raw
per-head weighted-value output.  Neither quantity is an attention probability.
"""

import dataclasses

import chex
import jax.numpy as jnp
from jaxtyping import (  # pylint: disable=g-multiple-import
    Array,
    Bool,
    Float,
    Int,
)


NUM_TRANSFORMER_LAYERS = 9


@dataclasses.dataclass(frozen=True, kw_only=True)
class TargetSpec:
  """Static description of a one-dimensional model-output target.

  ``head_name`` and ``prediction_key`` use the internal keys returned by
  :meth:`AlphaGenome.forward_heads`, for example ``rna_seq`` and
  ``scaled_predictions_1bp``.  This first implementation intentionally accepts
  only tensors shaped ``[batch, position, track]``.  Contact maps and splice
  junction matrices need dedicated reducers rather than ambiguous flattening.
  """

  head_name: str
  prediction_key: str


@chex.dataclass(frozen=True)
class TargetSelection:
  """Fixed-size position and track selection for a scalar biological target."""

  position_indices: Int[Array, 'Q']
  position_valid_mask: Bool[Array, 'Q']
  track_indices: Int[Array, 'T']
  track_valid_mask: Bool[Array, 'T']


@chex.dataclass(frozen=True)
class TargetSummary:
  """Per-example sum and mean over the selected model-output values."""

  total: Float[Array, 'B']
  mean: Float[Array, 'B']
  num_values: Int[Array, '']


@chex.dataclass(frozen=True)
class PairBiasEdgeSelection:
  """Directional coarse pair-bias edges to capture or replace.

  AlphaGenome's coarse pair bins each cover 16 sequence-transformer tokens, or
  2,048 bp.  ``query_bins[e]`` and ``key_bins[e]`` therefore identify a
  directional 2,048 bp x 2,048 bp edge, not a single base-to-base interaction.
  """

  query_bins: Int[Array, 'E']
  key_bins: Int[Array, 'E']
  valid_mask: Bool[Array, 'E']


@chex.dataclass(frozen=True)
class PairBiasReplacement:
  """Replacement values for selected compact pair-bias edges.

  ``replace_mask`` can select individual heads at each edge.  It has the same
  batch dimension as ``values`` so interventions can differ across examples.
  """

  selection: PairBiasEdgeSelection
  values: Float[Array, 'B E H']
  replace_mask: Bool[Array, 'B E H']


@chex.dataclass(frozen=True)
class HeadOutputSelection:
  """Sequence-token positions at which to capture every head output."""

  positions: Int[Array, 'P']
  valid_mask: Bool[Array, 'P']


@chex.dataclass(frozen=True)
class SequenceResidualSelection:
  """Fixed padded sequence-token positions for residual-stream tracing."""

  positions: Int[Array, 'R']
  valid_mask: Bool[Array, 'R']


@chex.dataclass(frozen=True)
class SequenceResidualReplacement:
  """Layer-stacked replacement vectors and whole-token replacement masks."""

  values: Float[Array, 'L B R D']
  replace_mask: Bool[Array, 'L B R']


@chex.dataclass(frozen=True)
class HeadValueOutputReplacement:
  """Layer-stacked local head-vector replacements at selected positions."""

  values: Float[Array, 'L B P H V']
  replace_mask: Bool[Array, 'L B P H']


@chex.dataclass(frozen=True)
class AttentionBiasTrace:
  """Selected compact bias values before and after an intervention."""

  compact_edges: Float[Array, 'B E H'] | None
  effective_compact_edges: Float[Array, 'B E H'] | None


@chex.dataclass(frozen=True)
class MHATrace:
  """Selected raw weighted-value head outputs before and after masking."""

  head_value_outputs: Float[Array, 'B P H V'] | None
  effective_head_value_outputs: Float[Array, 'B P H V'] | None


@chex.dataclass(frozen=True)
class TransformerTraceSelection:
  """Fixed-size selections returned from every sequence-attention layer."""

  pair_bias_edges: PairBiasEdgeSelection
  head_output_positions: HeadOutputSelection
  residual_positions: SequenceResidualSelection | None = None


@chex.dataclass(frozen=True)
class TransformerInterventions:
  """Dynamic interventions for all nine sequence-attention layers.

  ``head_masks`` multiplies the raw weighted-value output before each layer's
  output projection. ``pair_bias_values`` replaces entries selected by
  ``pair_bias_replace_mask`` at the shared directional edge selection.
  """

  head_masks: Float[Array, 'L H']
  pair_bias_values: Float[Array, 'L B E H']
  pair_bias_replace_mask: Bool[Array, 'L B E H']
  head_value_output_replacement: HeadValueOutputReplacement | None = None
  pre_attention_residual: SequenceResidualReplacement | None = None
  post_attention_residual: SequenceResidualReplacement | None = None
  post_mlp_residual: SequenceResidualReplacement | None = None


@chex.dataclass(frozen=True)
class TransformerTrace:
  """Selected attention internals stacked with layer as the leading axis."""

  compact_pair_bias_edges: Float[Array, 'L B E H']
  effective_compact_pair_bias_edges: Float[Array, 'L B E H']
  head_value_outputs: Float[Array, 'L B P H V']
  effective_head_value_outputs: Float[Array, 'L B P H V']
  pre_attention_residuals: Float[Array, 'L B R D']
  effective_pre_attention_residuals: Float[Array, 'L B R D']
  post_attention_residuals: Float[Array, 'L B R D']
  effective_post_attention_residuals: Float[Array, 'L B R D']
  post_mlp_residuals: Float[Array, 'L B R D']
  effective_post_mlp_residuals: Float[Array, 'L B R D']


def validate_transformer_interventions(
    interventions: TransformerInterventions,
    selection: TransformerTraceSelection,
    *,
    batch_size: int,
    num_heads: int,
    hidden_size: int,
    value_width: int,
) -> None:
  """Validates the static shapes needed by an instrumented tower call."""
  num_edges = selection.pair_bias_edges.query_bins.shape[0]
  expected_head_mask_shape = (NUM_TRANSFORMER_LAYERS, num_heads)
  expected_pair_shape = (
      NUM_TRANSFORMER_LAYERS,
      batch_size,
      num_edges,
      num_heads,
  )
  if interventions.head_masks.shape != expected_head_mask_shape:
    raise ValueError(
        f'head_masks must have shape {expected_head_mask_shape}, got '
        f'{interventions.head_masks.shape}.'
    )
  if interventions.pair_bias_values.shape != expected_pair_shape:
    raise ValueError(
        f'pair_bias_values must have shape {expected_pair_shape}, got '
        f'{interventions.pair_bias_values.shape}.'
    )
  if interventions.pair_bias_replace_mask.shape != expected_pair_shape:
    raise ValueError(
        f'pair_bias_replace_mask must have shape {expected_pair_shape}, got '
        f'{interventions.pair_bias_replace_mask.shape}.'
    )
  head_output_replacement = interventions.head_value_output_replacement
  if head_output_replacement is not None:
    num_head_positions = selection.head_output_positions.positions.shape[0]
    expected_head_values_shape = (
        NUM_TRANSFORMER_LAYERS,
        batch_size,
        num_head_positions,
        num_heads,
        value_width,
    )
    expected_head_replace_mask_shape = (
        NUM_TRANSFORMER_LAYERS,
        batch_size,
        num_head_positions,
        num_heads,
    )
    if head_output_replacement.values.shape != expected_head_values_shape:
      raise ValueError(
          'head_value_output_replacement.values must have shape '
          f'{expected_head_values_shape}, got '
          f'{head_output_replacement.values.shape}.'
      )
    if (
        head_output_replacement.replace_mask.shape
        != expected_head_replace_mask_shape
    ):
      raise ValueError(
          'head_value_output_replacement.replace_mask must have shape '
          f'{expected_head_replace_mask_shape}, got '
          f'{head_output_replacement.replace_mask.shape}.'
      )
  residual_selection = selection.residual_positions
  num_positions = (
      0 if residual_selection is None else residual_selection.positions.shape[0]
  )
  expected_residual_values_shape = (
      NUM_TRANSFORMER_LAYERS,
      batch_size,
      num_positions,
      hidden_size,
  )
  expected_residual_mask_shape = (
      NUM_TRANSFORMER_LAYERS,
      batch_size,
      num_positions,
  )
  for name, replacement in (
      ('pre_attention_residual', interventions.pre_attention_residual),
      ('post_attention_residual', interventions.post_attention_residual),
      ('post_mlp_residual', interventions.post_mlp_residual),
  ):
    if replacement is None:
      continue
    if replacement.values.shape != expected_residual_values_shape:
      raise ValueError(
          f'{name}.values must have shape {expected_residual_values_shape}, '
          f'got {replacement.values.shape}.'
      )
    if replacement.replace_mask.shape != expected_residual_mask_shape:
      raise ValueError(
          f'{name}.replace_mask must have shape '
          f'{expected_residual_mask_shape}, '
          f'got {replacement.replace_mask.shape}.'
      )


def no_transformer_interventions(
    *,
    batch_size: int,
    num_edges: int,
    num_heads: int = 8,
    dtype=jnp.float32,
) -> TransformerInterventions:
  """Builds fixed-shape identity interventions for an instrumented tower."""
  pair_shape = (
      NUM_TRANSFORMER_LAYERS,
      batch_size,
      num_edges,
      num_heads,
  )
  return TransformerInterventions(
      head_masks=jnp.ones(
          (NUM_TRANSFORMER_LAYERS, num_heads), dtype=dtype
      ),
      pair_bias_values=jnp.zeros(pair_shape, dtype=dtype),
      pair_bias_replace_mask=jnp.zeros(pair_shape, dtype=jnp.bool),
  )


def no_sequence_residual_replacement(
    *,
    batch_size: int,
    num_positions: int,
    hidden_size: int,
    dtype=jnp.float32,
) -> SequenceResidualReplacement:
  """Builds a fixed-shape, dynamically patchable identity replacement."""
  return SequenceResidualReplacement(
      values=jnp.zeros(
          (
              NUM_TRANSFORMER_LAYERS,
              batch_size,
              num_positions,
              hidden_size,
          ),
          dtype=dtype,
      ),
      replace_mask=jnp.zeros(
          (NUM_TRANSFORMER_LAYERS, batch_size, num_positions), dtype=jnp.bool
      ),
  )


def no_head_value_output_replacement(
    *,
    batch_size: int,
    num_positions: int,
    num_heads: int = 8,
    value_width: int = 192,
    dtype=jnp.float32,
) -> HeadValueOutputReplacement:
  """Builds fixed-shape identity storage for local head-output patching."""
  return HeadValueOutputReplacement(
      values=jnp.zeros(
          (
              NUM_TRANSFORMER_LAYERS,
              batch_size,
              num_positions,
              num_heads,
              value_width,
          ),
          dtype=dtype,
      ),
      replace_mask=jnp.zeros(
          (
              NUM_TRANSFORMER_LAYERS,
              batch_size,
              num_positions,
              num_heads,
          ),
          dtype=jnp.bool,
      ),
  )


def reduce_target(
    predictions: Float[Array, 'B S C'],
    selection: TargetSelection,
) -> TargetSummary:
  """Reduces selected positions and tracks without exporting dense outputs.

  Invalid padded selection entries contribute zero and are excluded from the
  denominator.  Indices are clipped before gathering so a padded index can be
  any integer without causing an accelerator bounds error.
  """
  if predictions.ndim != 3:
    raise ValueError(
        'Target predictions must have shape [batch, position, track].'
    )
  if (
      selection.position_indices.ndim != 1
      or selection.position_valid_mask.ndim != 1
      or selection.track_indices.ndim != 1
      or selection.track_valid_mask.ndim != 1
  ):
    raise ValueError('Target selection arrays must all have rank 1.')
  if selection.position_indices.shape != selection.position_valid_mask.shape:
    raise ValueError('Target position arrays must have the same shape.')
  if selection.track_indices.shape != selection.track_valid_mask.shape:
    raise ValueError('Target track arrays must have the same shape.')

  safe_positions = jnp.clip(
      selection.position_indices, 0, predictions.shape[1] - 1
  )
  safe_tracks = jnp.clip(selection.track_indices, 0, predictions.shape[2] - 1)
  values = predictions[:, safe_positions, :][:, :, safe_tracks].astype(
      jnp.float32
  )
  valid = (
      selection.position_valid_mask[:, None]
      & selection.track_valid_mask[None, :]
  )
  values = jnp.where(valid[None, :, :], values, 0.0)
  total = jnp.sum(values, axis=(1, 2), dtype=jnp.float32)
  num_values = jnp.sum(valid, dtype=jnp.int32)
  mean = total / jnp.maximum(num_values, 1)
  return TargetSummary(total=total, mean=mean, num_values=num_values)


def _validate_edge_selection(
    selection: PairBiasEdgeSelection, compact_bias: Array
) -> None:
  if compact_bias.ndim != 4:
    raise ValueError('compact_bias must have shape [batch, query, key, head].')
  if (
      selection.query_bins.ndim != 1
      or selection.key_bins.ndim != 1
      or selection.valid_mask.ndim != 1
  ):
    raise ValueError('Pair-bias selection arrays must all have rank 1.')
  if not (
      selection.query_bins.shape
      == selection.key_bins.shape
      == selection.valid_mask.shape
  ):
    raise ValueError('Pair-bias selection arrays must have the same shape.')


def gather_pair_bias_edges(
    compact_bias: Float[Array, 'B C C H'],
    selection: PairBiasEdgeSelection,
) -> Float[Array, 'B E H']:
  """Gathers selected directional edges, zeroing padded selection entries."""
  _validate_edge_selection(selection, compact_bias)
  coarse_length = compact_bias.shape[1]
  safe_query_bins = jnp.clip(selection.query_bins, 0, coarse_length - 1)
  safe_key_bins = jnp.clip(selection.key_bins, 0, coarse_length - 1)
  values = compact_bias[:, safe_query_bins, safe_key_bins, :]
  return jnp.where(selection.valid_mask[None, :, None], values, 0)


def replace_pair_bias_edges(
    compact_bias: Float[Array, 'B C C H'],
    replacement: PairBiasReplacement | None,
) -> Float[Array, 'B C C H']:
  """Returns compact bias with dynamically selected edge/head replacements."""
  if replacement is None:
    return compact_bias
  _validate_edge_selection(replacement.selection, compact_bias)
  batch_size, coarse_length, _, num_heads = compact_bias.shape
  num_edges = replacement.selection.query_bins.shape[0]
  expected_shape = (batch_size, num_edges, num_heads)
  if replacement.values.shape != expected_shape:
    raise ValueError(
        f'Pair-bias replacement values must have shape {expected_shape}.'
    )
  if replacement.replace_mask.shape != expected_shape:
    raise ValueError(
        f'Pair-bias replacement mask must have shape {expected_shape}.'
    )

  selection = replacement.selection
  clipped_query_bins = jnp.clip(selection.query_bins, 0, coarse_length - 1)
  clipped_key_bins = jnp.clip(selection.key_bins, 0, coarse_length - 1)
  current = compact_bias[:, clipped_query_bins, clipped_key_bins, :]
  replace_mask = replacement.replace_mask & selection.valid_mask[None, :, None]
  replacement_values = replacement.values.astype(compact_bias.dtype)
  updates = jnp.where(replace_mask, replacement_values, current)

  # Padded entries are moved one past the valid range and dropped by scatter.
  # This avoids an invalid padded edge overwriting a real edge at index zero.
  scatter_query_bins = jnp.where(
      selection.valid_mask, selection.query_bins, coarse_length
  )
  scatter_key_bins = jnp.where(
      selection.valid_mask, selection.key_bins, coarse_length
  )
  return compact_bias.at[
      :, scatter_query_bins, scatter_key_bins, :
  ].set(updates, mode='drop')


def _validate_head_selection(
    selection: HeadOutputSelection, head_outputs: Array
) -> None:
  if head_outputs.ndim != 4:
    raise ValueError(
        'head_outputs must have shape [batch, sequence, head, value].'
    )
  if selection.positions.ndim != 1 or selection.valid_mask.ndim != 1:
    raise ValueError('Head-output selection arrays must both have rank 1.')
  if selection.positions.shape != selection.valid_mask.shape:
    raise ValueError('Head-output selection arrays must have the same shape.')


def gather_head_outputs(
    head_outputs: Float[Array, 'B S H V'],
    selection: HeadOutputSelection,
) -> Float[Array, 'B P H V']:
  """Gathers selected sequence positions, zeroing padded selection entries."""
  _validate_head_selection(selection, head_outputs)
  safe_positions = jnp.clip(selection.positions, 0, head_outputs.shape[1] - 1)
  values = head_outputs[:, safe_positions, :, :]
  return jnp.where(selection.valid_mask[None, :, None, None], values, 0)


def replace_head_value_outputs(
    head_outputs: Float[Array, 'B S H V'],
    selection: HeadOutputSelection | None,
    values: Float[Array, 'B P H V'] | None,
    replace_mask: Bool[Array, 'B P H'] | None,
) -> Float[Array, 'B S H V']:
  """Locally replaces selected head vectors before the output projection."""
  if values is None and replace_mask is None:
    return head_outputs
  if selection is None:
    raise ValueError('A head-output selection is required for replacement.')
  if values is None or replace_mask is None:
    raise ValueError(
        'Head-output replacement values and mask are both required.'
    )
  _validate_head_selection(selection, head_outputs)
  batch_size, sequence_length, num_heads, value_width = head_outputs.shape
  num_positions = selection.positions.shape[0]
  expected_values_shape = (
      batch_size,
      num_positions,
      num_heads,
      value_width,
  )
  expected_mask_shape = (batch_size, num_positions, num_heads)
  if values.shape != expected_values_shape:
    raise ValueError(
        'Head-output replacement values must have shape '
        f'{expected_values_shape}.'
    )
  if replace_mask.shape != expected_mask_shape:
    raise ValueError(
        f'Head-output replacement mask must have shape {expected_mask_shape}.'
    )

  clipped_positions = jnp.clip(selection.positions, 0, sequence_length - 1)
  current = head_outputs[:, clipped_positions, :, :]
  effective_mask = replace_mask & selection.valid_mask[None, :, None]
  updates = jnp.where(
      effective_mask[..., None], values.astype(head_outputs.dtype), current
  )
  scatter_positions = jnp.where(
      selection.valid_mask, selection.positions, sequence_length
  )
  return head_outputs.at[:, scatter_positions, :, :].set(updates, mode='drop')


def apply_head_mask(
    head_outputs: Float[Array, 'B S H V'],
    head_mask: Float[Array, 'H'] | None,
) -> Float[Array, 'B S H V']:
  """Applies a dynamic multiplicative mask before the head-output projection."""
  if head_mask is None:
    return head_outputs
  if head_mask.ndim != 1 or head_mask.shape[0] != head_outputs.shape[2]:
    raise ValueError(
        f'head_mask must have shape ({head_outputs.shape[2]},), got '
        f'{head_mask.shape}.'
    )
  mask = head_mask.astype(head_outputs.dtype)[None, None, :, None]
  return head_outputs * mask


def _validate_residual_selection(
    selection: SequenceResidualSelection, residuals: Array
) -> None:
  if residuals.ndim != 3:
    raise ValueError('residuals must have shape [batch, sequence, hidden].')
  if selection.positions.ndim != 1 or selection.valid_mask.ndim != 1:
    raise ValueError('Residual selection arrays must both have rank 1.')
  if selection.positions.shape != selection.valid_mask.shape:
    raise ValueError('Residual selection arrays must have the same shape.')


def gather_sequence_residuals(
    residuals: Float[Array, 'B S D'],
    selection: SequenceResidualSelection | None,
) -> Float[Array, 'B R D']:
  """Gathers selected residual vectors and zeroes padded positions."""
  if selection is None:
    return jnp.zeros(
        (residuals.shape[0], 0, residuals.shape[2]), dtype=residuals.dtype
    )
  _validate_residual_selection(selection, residuals)
  safe_positions = jnp.clip(selection.positions, 0, residuals.shape[1] - 1)
  values = residuals[:, safe_positions, :]
  return jnp.where(selection.valid_mask[None, :, None], values, 0)


def replace_sequence_residuals(
    residuals: Float[Array, 'B S D'],
    selection: SequenceResidualSelection | None,
    values: Float[Array, 'B R D'] | None,
    replace_mask: Bool[Array, 'B R'] | None,
) -> Float[Array, 'B S D']:
  """Replaces selected whole-token residual vectors with dynamic values."""
  if values is None and replace_mask is None:
    return residuals
  if selection is None:
    raise ValueError('A residual selection is required for replacement.')
  if values is None or replace_mask is None:
    raise ValueError('Residual replacement values and mask are both required.')
  _validate_residual_selection(selection, residuals)
  batch_size, sequence_length, hidden_size = residuals.shape
  num_positions = selection.positions.shape[0]
  expected_values_shape = (batch_size, num_positions, hidden_size)
  expected_mask_shape = (batch_size, num_positions)
  if values.shape != expected_values_shape:
    raise ValueError(
        f'Residual replacement values must have shape {expected_values_shape}.'
    )
  if replace_mask.shape != expected_mask_shape:
    raise ValueError(
        f'Residual replacement mask must have shape {expected_mask_shape}.'
    )

  clipped_positions = jnp.clip(selection.positions, 0, sequence_length - 1)
  current = residuals[:, clipped_positions, :]
  effective_mask = replace_mask & selection.valid_mask[None, :]
  updates = jnp.where(
      effective_mask[..., None], values.astype(residuals.dtype), current
  )
  scatter_positions = jnp.where(
      selection.valid_mask, selection.positions, sequence_length
  )
  return residuals.at[:, scatter_positions, :].set(updates, mode='drop')
