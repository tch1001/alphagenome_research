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
import jax
import jax.numpy as jnp
from jaxtyping import (  # pylint: disable=g-multiple-import
    Array,
    Bool,
    Float,
    Int,
)


NUM_TRANSFORMER_LAYERS = 9
ENCODER_ROUTE_RESOLUTIONS = (1, 2, 4, 8, 16, 32, 64, 128)
DECODER_ROUTE_RESOLUTIONS = (64, 32, 16, 8, 4, 2, 1)
FINAL_EMBEDDING_ROUTE_RESOLUTIONS = (128, 1)
PAIRED_CAUSAL_BATCH_SIZE = 6
PAIRED_CAUSAL_DONOR_ROWS = (0, 1, 0, 1, 1, 0)
PAIRED_CAUSAL_IDENTITY_ROWS = (0, 1, 1, 1, 0, 0)


@dataclasses.dataclass(frozen=True)
class CausalRouteFamilyMetadata:
  """Static route-family layout for census enumeration and target reducers."""

  name: str
  resolutions_bp: tuple[int, ...]
  channel_widths: tuple[int, ...]
  seam: str


CAUSAL_ROUTE_FAMILIES = (
    CausalRouteFamilyMetadata(
        name='encoder_outputs',
        resolutions_bp=ENCODER_ROUTE_RESOLUTIONS,
        channel_widths=(768, 896, 1024, 1152, 1280, 1408, 1536, 1536),
        seam='after_encoder_stage_before_skip_storage_or_next_stage',
    ),
    CausalRouteFamilyMetadata(
        name='transformer_pre_attention',
        resolutions_bp=(128,) * NUM_TRANSFORMER_LAYERS,
        channel_widths=(1536,) * NUM_TRANSFORMER_LAYERS,
        seam='before_attention_update',
    ),
    CausalRouteFamilyMetadata(
        name='transformer_post_attention',
        resolutions_bp=(128,) * NUM_TRANSFORMER_LAYERS,
        channel_widths=(1536,) * NUM_TRANSFORMER_LAYERS,
        seam='after_attention_residual_before_mlp',
    ),
    CausalRouteFamilyMetadata(
        name='transformer_post_mlp',
        resolutions_bp=(128,) * NUM_TRANSFORMER_LAYERS,
        channel_widths=(1536,) * NUM_TRANSFORMER_LAYERS,
        seam='after_mlp_residual',
    ),
    CausalRouteFamilyMetadata(
        name='decoder_skip_states',
        resolutions_bp=DECODER_ROUTE_RESOLUTIONS,
        channel_widths=(1536, 1408, 1280, 1152, 1024, 896, 768),
        seam='after_skip_lookup_before_upres_block',
    ),
    CausalRouteFamilyMetadata(
        name='decoder_outputs',
        resolutions_bp=DECODER_ROUTE_RESOLUTIONS,
        channel_widths=(1536, 1408, 1280, 1152, 1024, 896, 768),
        seam='after_upres_block',
    ),
    CausalRouteFamilyMetadata(
        name='final_embeddings',
        resolutions_bp=FINAL_EMBEDDING_ROUTE_RESOLUTIONS,
        channel_widths=(3072, 1536),
        seam='after_output_embedder',
    ),
)


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
class PairedTargetSelection:
  """Fixed-size paired position/track values for a scalar output target.

  Unlike :class:`TargetSelection`, this selects ``(position_indices[q],
  track_indices[q])`` pairs rather than their Cartesian product.  This is
  required for targets such as strand-aware splice-site scores, where the
  acceptor and donor live at different positions and in different channels.
  """

  position_indices: Int[Array, 'Q']
  track_indices: Int[Array, 'Q']
  valid_mask: Bool[Array, 'Q']


@chex.dataclass(frozen=True)
class SpliceClassificationLogitMarginSelection:
  """The locked acceptor/donor class-vs-padding splice target.

  Both arrays have exactly two entries ordered ``[acceptor, donor]``.  The
  track indices must already be resolved for the exon strand.  There is no
  validity mask because both canonical endpoints are mandatory for this
  biological target.
  """

  canonical_position_indices: Int[Array, '2']
  canonical_track_indices: Int[Array, '2']
  padding_track_index: Int[Array, '']


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
class SequenceResidualBatchTransfer:
  """Layer-stacked live residual transfers between rows of one batch.

  For each enabled ``[layer, recipient, position]`` entry, the selected
  residual vector is gathered from ``donor_batch_indices`` at the same
  selected sequence position. Donors are read from the unmodified residual
  tensor at that seam, so transfers cannot cascade through other recipients.
  """

  donor_batch_indices: Int[Array, 'L B R']
  transfer_mask: Bool[Array, 'L B R']


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
  pre_attention_residual_transfer: SequenceResidualBatchTransfer | None = None
  post_attention_residual_transfer: SequenceResidualBatchTransfer | None = None
  post_mlp_residual_transfer: SequenceResidualBatchTransfer | None = None


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


@chex.dataclass(frozen=True)
class CausalRouteTraceSelection:
  """Fixed padded positions for every sequence route in the trunk.

  Stage axes follow the exported resolution constants above. Encoder output
  stage 128 is the post-pool trunk; the other encoder stages are also the U-Net
  skip tensors. ``decoder_skip`` selects those skip tensors at their isolated
  point of consumption, while ``decoder_output`` selects each UpResBlock output.
  """

  transformer: TransformerTraceSelection
  encoder_positions: Int[Array, '8 RE']
  encoder_valid_mask: Bool[Array, '8 RE']
  decoder_skip_positions: Int[Array, '7 RS']
  decoder_skip_valid_mask: Bool[Array, '7 RS']
  decoder_output_positions: Int[Array, '7 RD']
  decoder_output_valid_mask: Bool[Array, '7 RD']
  final_embedding_positions: Int[Array, '2 RF']
  final_embedding_valid_mask: Bool[Array, '2 RF']


@chex.dataclass(frozen=True)
class CausalRouteInterventions:
  """Opt-in live batch transfers spanning the complete sequence route."""

  transformer: TransformerInterventions
  encoder_outputs: SequenceResidualBatchTransfer
  decoder_skip_states: SequenceResidualBatchTransfer
  decoder_outputs: SequenceResidualBatchTransfer
  final_embeddings: SequenceResidualBatchTransfer


@chex.dataclass(frozen=True)
class CausalRouteTrace:
  """Compact selected values before/effectively after every route seam."""

  transformer: TransformerTrace
  encoder_outputs: tuple[Array, ...]
  effective_encoder_outputs: tuple[Array, ...]
  decoder_skip_states: tuple[Array, ...]
  effective_decoder_skip_states: tuple[Array, ...]
  decoder_outputs: tuple[Array, ...]
  effective_decoder_outputs: tuple[Array, ...]
  final_embeddings: tuple[Array, ...]
  effective_final_embeddings: tuple[Array, ...]


@chex.dataclass(frozen=True)
class WholeSequenceBatchTransfer:
  """Live whole-tensor transfers for route branches with differing lengths.

  The leading axis enumerates statically unrolled route stages. Unlike a
  selected residual transfer, this primitive replaces every sequence position
  and channel in an enabled tensor without materializing it in the returned
  trace.
  """

  donor_batch_indices: Int[Array, 'N B']
  natural_identity_batch_indices: Int[Array, 'N B']
  transfer_mask: Bool[Array, 'N B']


@chex.dataclass(frozen=True)
class StageABranchSelection:
  """Compact A/D positions used by the final post-GELU closure control."""

  final_embedding_positions: Int[Array, 'R']
  final_embedding_valid_mask: Bool[Array, 'R']


@chex.dataclass(frozen=True)
class StageABranchInterventions:
  """Whole T/E branch transfers and final post-GELU closure transfer."""

  transformer_output: WholeSequenceBatchTransfer
  encoder_skips: WholeSequenceBatchTransfer
  final_embedding: SequenceResidualBatchTransfer


@chex.dataclass(frozen=True)
class StageABranchTrace:
  """Compact exactness evidence for Stage-A closure and branch transfers."""

  transformer_output_natural_matches_identity: Bool[Array, 'B']
  transformer_output_effective_matches_natural: Bool[Array, 'B']
  transformer_output_effective_matches_intervention_donor: Bool[Array, 'B']
  transformer_output_natural_fingerprint: Int[Array, 'B 4']
  encoder_skips_natural_match_identity: Bool[Array, '7 B']
  encoder_skips_effective_match_natural: Bool[Array, '7 B']
  encoder_skips_effective_match_intervention_donor: Bool[Array, '7 B']
  encoder_skips_natural_fingerprints: Int[Array, '7 B 4']
  natural_final_embeddings: Float[Array, 'B R D']
  effective_final_embeddings: Float[Array, 'B R D']


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
  residual_interventions = (
      (
          'pre_attention_residual',
          interventions.pre_attention_residual,
          interventions.pre_attention_residual_transfer,
      ),
      (
          'post_attention_residual',
          interventions.post_attention_residual,
          interventions.post_attention_residual_transfer,
      ),
      (
          'post_mlp_residual',
          interventions.post_mlp_residual,
          interventions.post_mlp_residual_transfer,
      ),
  )
  for name, replacement, transfer in residual_interventions:
    if replacement is not None and transfer is not None:
      raise ValueError(
          f'{name} cannot use an external replacement and live batch '
          'transfer simultaneously.'
      )
    if replacement is not None:
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
    if transfer is None:
      continue
    if transfer.donor_batch_indices.shape != expected_residual_mask_shape:
      raise ValueError(
          f'{name}_transfer.donor_batch_indices must have shape '
          f'{expected_residual_mask_shape}, got '
          f'{transfer.donor_batch_indices.shape}.'
      )
    if transfer.transfer_mask.shape != expected_residual_mask_shape:
      raise ValueError(
          f'{name}_transfer.transfer_mask must have shape '
          f'{expected_residual_mask_shape}, got {transfer.transfer_mask.shape}.'
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


def no_sequence_route_batch_transfer(
    *,
    num_stages: int,
    batch_size: int,
    num_positions: int,
) -> SequenceResidualBatchTransfer:
  """Builds a fixed-shape all-false transfer for arbitrary sequence seams."""
  shape = (num_stages, batch_size, num_positions)
  donors = jnp.broadcast_to(
      jnp.arange(batch_size, dtype=jnp.int32)[None, :, None], shape
  )
  return SequenceResidualBatchTransfer(
      donor_batch_indices=donors,
      transfer_mask=jnp.zeros(shape, dtype=jnp.bool),
  )


def paired_six_row_batch_transfer(
    component_mask: Bool[Array, 'N R'],
) -> SequenceResidualBatchTransfer:
  """Expands a route-component mask into the frozen six-row donor protocol.

  Rows are REF baseline, ALT baseline, REF->ALT, ALT->ALT, ALT->REF, and
  REF->REF. Baseline rows are never replaced. ``component_mask`` remains a
  dynamic JAX argument, so route/stage/slot sweeps reuse one executable shape.
  """
  if component_mask.ndim != 2:
    raise ValueError('component_mask must have shape [stage, position].')
  num_stages, num_positions = component_mask.shape
  donor_rows = jnp.asarray(PAIRED_CAUSAL_DONOR_ROWS, dtype=jnp.int32)
  donors = jnp.broadcast_to(
      donor_rows[None, :, None],
      (num_stages, PAIRED_CAUSAL_BATCH_SIZE, num_positions),
  )
  recipient_rows = jnp.array(
      [False, False, True, True, True, True], dtype=jnp.bool
  )
  transfer_mask = (
      component_mask[:, None, :] & recipient_rows[None, :, None]
  )
  return SequenceResidualBatchTransfer(
      donor_batch_indices=donors, transfer_mask=transfer_mask
  )


def no_whole_sequence_batch_transfer(
    *, num_stages: int, batch_size: int
) -> WholeSequenceBatchTransfer:
  """Builds a fixed-shape all-false whole-tensor transfer."""
  donors = jnp.broadcast_to(
      jnp.arange(batch_size, dtype=jnp.int32)[None, :],
      (num_stages, batch_size),
  )
  return WholeSequenceBatchTransfer(
      donor_batch_indices=donors,
      natural_identity_batch_indices=donors,
      transfer_mask=jnp.zeros((num_stages, batch_size), jnp.bool),
  )


def paired_six_row_whole_sequence_transfer(
    component_mask: Bool[Array, 'N'],
) -> WholeSequenceBatchTransfer:
  """Expands dynamic stage masks into the frozen six-row whole transfer."""
  if component_mask.ndim != 1:
    raise ValueError('component_mask must have shape [stage].')
  num_stages = component_mask.shape[0]
  donor_rows = jnp.asarray(PAIRED_CAUSAL_DONOR_ROWS, jnp.int32)
  donors = jnp.broadcast_to(
      donor_rows[None, :], (num_stages, PAIRED_CAUSAL_BATCH_SIZE)
  )
  identity_rows = jnp.asarray(PAIRED_CAUSAL_IDENTITY_ROWS, jnp.int32)
  identity_donors = jnp.broadcast_to(
      identity_rows[None, :], (num_stages, PAIRED_CAUSAL_BATCH_SIZE)
  )
  recipient_rows = jnp.asarray(
      [False, False, True, True, True, True], jnp.bool
  )
  return WholeSequenceBatchTransfer(
      donor_batch_indices=donors,
      natural_identity_batch_indices=identity_donors,
      transfer_mask=component_mask[:, None] & recipient_rows[None, :],
  )


def transfer_whole_sequence_within_batch(
    values: Float[Array, 'B S D'],
    transfer: WholeSequenceBatchTransfer,
    stage: int,
) -> tuple[
    Float[Array, 'B S D'],
    Bool[Array, 'B'],
    Bool[Array, 'B'],
    Bool[Array, 'B'],
    Int[Array, 'B 4'],
]:
  """Transfers a complete live tensor and returns non-tautological audits.

  ``natural_matches_identity`` compares each unmodified row to its same-allele
  baseline. ``effective_matches_natural`` proves whether the transfer was a
  no-op. ``effective_matches_donor`` separately compares the post-transfer row
  to the requested intervention donor. The compact fingerprint hashes the
  natural tensor's exact BF16 bit patterns for cross-call repeat audits.
  """
  if transfer.donor_batch_indices.ndim != 2:
    raise ValueError('Whole-sequence donor indices must have shape [stage, B].')
  if transfer.transfer_mask.shape != transfer.donor_batch_indices.shape:
    raise ValueError('Whole-sequence transfer arrays must have equal shapes.')
  if (
      transfer.natural_identity_batch_indices.shape
      != transfer.donor_batch_indices.shape
  ):
    raise ValueError('Whole-sequence natural identity map has invalid shape.')
  batch_size = values.shape[0]
  if transfer.donor_batch_indices.shape[1] != batch_size:
    raise ValueError(
        'Whole-sequence transfer batch axis does not match values.'
    )
  donor_indices = transfer.donor_batch_indices[stage]
  mask = transfer.transfer_mask[stage]
  valid_donors = (donor_indices >= 0) & (donor_indices < batch_size)
  safe_donors = jnp.clip(donor_indices, 0, batch_size - 1)
  donor_values = values[safe_donors]
  identity_indices = transfer.natural_identity_batch_indices[stage]
  valid_identity = (identity_indices >= 0) & (identity_indices < batch_size)
  safe_identity = jnp.clip(identity_indices, 0, batch_size - 1)
  natural_matches_identity = valid_identity & jnp.all(
      values == values[safe_identity], axis=(1, 2)
  )
  effective = jnp.where(
      (mask & valid_donors)[:, None, None], donor_values, values
  )
  effective_matches_natural = jnp.all(effective == values, axis=(1, 2))
  effective_matches_donor = valid_donors & jnp.all(
      effective == donor_values, axis=(1, 2)
  )
  return (
      effective,
      natural_matches_identity,
      effective_matches_natural,
      effective_matches_donor,
      bitwise_tensor_fingerprint(values),
  )


def bitwise_tensor_fingerprint(
    values: Float[Array, 'B ...'],
) -> Int[Array, 'B 4']:
  """Returns four deterministic uint32 checksums of exact BF16 tensor bits.

  This is compact repeat evidence, not a collision-free equality proof. The
  four reductions are bitwise xor, modular sum, index-weighted modular sum and
  a second mixed-index modular sum. Shape is static in the compiled graph and
  audited separately by the trace schema.
  """
  if values.dtype != jnp.bfloat16:
    raise ValueError('Stage-A whole-tensor fingerprints require BF16 values.')
  bits = jax.lax.bitcast_convert_type(values, jnp.uint16)
  flat = bits.reshape((bits.shape[0], -1)).astype(jnp.uint32)
  indices = jnp.arange(1, flat.shape[1] + 1, dtype=jnp.uint32)
  mixed = indices ^ (indices >> jnp.uint32(16))
  mixed = mixed * jnp.uint32(0x7FEB352D)
  mixed = mixed ^ (mixed >> jnp.uint32(15))
  mixed = mixed * jnp.uint32(0x846CA68B)
  mixed = mixed ^ (mixed >> jnp.uint32(16))
  return jnp.stack(
      (
          jnp.bitwise_xor.reduce(flat, axis=1),
          jnp.sum(flat, axis=1, dtype=jnp.uint32),
          jnp.sum(flat * indices[None, :], axis=1, dtype=jnp.uint32),
          jnp.sum(flat * mixed[None, :], axis=1, dtype=jnp.uint32),
      ),
      axis=1,
  )


def no_stage_a_branch_interventions(
    selection: StageABranchSelection, *, batch_size: int
) -> StageABranchInterventions:
  """Builds an all-false fixed pytree for Stage-A branch experiments."""
  if batch_size != PAIRED_CAUSAL_BATCH_SIZE:
    raise ValueError('Stage-A branch experiments require exactly six rows.')
  num_positions = selection.final_embedding_positions.shape[0]
  return StageABranchInterventions(
      transformer_output=paired_six_row_whole_sequence_transfer(
          jnp.zeros((1,), jnp.bool)
      ),
      encoder_skips=paired_six_row_whole_sequence_transfer(
          jnp.zeros((len(DECODER_ROUTE_RESOLUTIONS),), jnp.bool)
      ),
      final_embedding=no_sequence_route_batch_transfer(
          num_stages=1,
          batch_size=batch_size,
          num_positions=num_positions,
      ),
  )


def validate_stage_a_branch_interventions(
    selection: StageABranchSelection,
    interventions: StageABranchInterventions,
    *,
    batch_size: int,
) -> None:
  """Validates fixed shapes for opt-in T/E/final-embedding transfers."""
  if selection.final_embedding_positions.ndim != 1:
    raise ValueError('Stage-A final positions must have rank one.')
  if (
      selection.final_embedding_valid_mask.shape
      != selection.final_embedding_positions.shape
  ):
    raise ValueError('Stage-A final position arrays must have equal shapes.')
  expected_whole_shapes = (
      ('transformer_output', interventions.transformer_output, (1, batch_size)),
      (
          'encoder_skips',
          interventions.encoder_skips,
          (len(DECODER_ROUTE_RESOLUTIONS), batch_size),
      ),
  )
  for name, transfer, expected_shape in expected_whole_shapes:
    if transfer.donor_batch_indices.shape != expected_shape:
      raise ValueError(f'{name} donors must have shape {expected_shape}.')
    if transfer.natural_identity_batch_indices.shape != expected_shape:
      raise ValueError(
          f'{name} natural identity map must have shape {expected_shape}.'
      )
    if transfer.transfer_mask.shape != expected_shape:
      raise ValueError(f'{name} mask must have shape {expected_shape}.')
  _validate_route_transfer(
      'final_embedding',
      interventions.final_embedding,
      num_stages=1,
      batch_size=batch_size,
      num_positions=selection.final_embedding_positions.shape[0],
  )


def _validate_route_selection_axis(
    name: str,
    positions: Array,
    valid_mask: Array,
    expected_stages: int,
) -> int:
  if positions.ndim != 2 or valid_mask.ndim != 2:
    raise ValueError(f'{name} selection arrays must have rank 2.')
  if positions.shape != valid_mask.shape:
    raise ValueError(f'{name} selection arrays must have the same shape.')
  if positions.shape[0] != expected_stages:
    raise ValueError(
        f'{name} must have {expected_stages} stages, got {positions.shape[0]}.'
    )
  return positions.shape[1]


def _validate_route_transfer(
    name: str,
    transfer: SequenceResidualBatchTransfer,
    *,
    num_stages: int,
    batch_size: int,
    num_positions: int,
) -> None:
  expected_shape = (num_stages, batch_size, num_positions)
  if transfer.donor_batch_indices.shape != expected_shape:
    raise ValueError(
        f'{name}.donor_batch_indices must have shape {expected_shape}, got '
        f'{transfer.donor_batch_indices.shape}.'
    )
  if transfer.transfer_mask.shape != expected_shape:
    raise ValueError(
        f'{name}.transfer_mask must have shape {expected_shape}, got '
        f'{transfer.transfer_mask.shape}.'
    )


def validate_causal_route_interventions(
    selection: CausalRouteTraceSelection,
    interventions: CausalRouteInterventions,
    *,
    batch_size: int,
) -> None:
  """Validates all non-transformer route axes and transfer shapes."""
  if selection.transformer.residual_positions is None:
    raise ValueError(
        'A complete causal route census requires transformer residual '
        'positions.'
    )
  route_specs = (
      (
          'encoder_outputs',
          selection.encoder_positions,
          selection.encoder_valid_mask,
          len(ENCODER_ROUTE_RESOLUTIONS),
          interventions.encoder_outputs,
      ),
      (
          'decoder_skip_states',
          selection.decoder_skip_positions,
          selection.decoder_skip_valid_mask,
          len(DECODER_ROUTE_RESOLUTIONS),
          interventions.decoder_skip_states,
      ),
      (
          'decoder_outputs',
          selection.decoder_output_positions,
          selection.decoder_output_valid_mask,
          len(DECODER_ROUTE_RESOLUTIONS),
          interventions.decoder_outputs,
      ),
      (
          'final_embeddings',
          selection.final_embedding_positions,
          selection.final_embedding_valid_mask,
          len(FINAL_EMBEDDING_ROUTE_RESOLUTIONS),
          interventions.final_embeddings,
      ),
  )
  for name, positions, valid_mask, num_stages, transfer in route_specs:
    num_positions = _validate_route_selection_axis(
        name, positions, valid_mask, num_stages
    )
    _validate_route_transfer(
        name,
        transfer,
        num_stages=num_stages,
        batch_size=batch_size,
        num_positions=num_positions,
    )


def no_causal_route_interventions(
    selection: CausalRouteTraceSelection,
    *,
    batch_size: int,
    num_edges: int,
    num_heads: int = 8,
    dtype=jnp.float32,
) -> CausalRouteInterventions:
  """Builds an all-false fixed pytree for a complete route-census apply."""
  if selection.transformer.residual_positions is None:
    raise ValueError(
        'A complete causal route census requires transformer residual '
        'positions.'
    )
  transformer = no_transformer_interventions(
      batch_size=batch_size,
      num_edges=num_edges,
      num_heads=num_heads,
      dtype=dtype,
  )
  num_transformer_positions = (
      selection.transformer.residual_positions.positions.shape[0]
  )
  transformer_transfer = no_sequence_route_batch_transfer(
      num_stages=NUM_TRANSFORMER_LAYERS,
      batch_size=batch_size,
      num_positions=num_transformer_positions,
  )
  transformer = dataclasses.replace(
      transformer,
      pre_attention_residual_transfer=transformer_transfer,
      post_attention_residual_transfer=transformer_transfer,
      post_mlp_residual_transfer=transformer_transfer,
  )
  return CausalRouteInterventions(
      transformer=transformer,
      encoder_outputs=no_sequence_route_batch_transfer(
          num_stages=len(ENCODER_ROUTE_RESOLUTIONS),
          batch_size=batch_size,
          num_positions=selection.encoder_positions.shape[1],
      ),
      decoder_skip_states=no_sequence_route_batch_transfer(
          num_stages=len(DECODER_ROUTE_RESOLUTIONS),
          batch_size=batch_size,
          num_positions=selection.decoder_skip_positions.shape[1],
      ),
      decoder_outputs=no_sequence_route_batch_transfer(
          num_stages=len(DECODER_ROUTE_RESOLUTIONS),
          batch_size=batch_size,
          num_positions=selection.decoder_output_positions.shape[1],
      ),
      final_embeddings=no_sequence_route_batch_transfer(
          num_stages=len(FINAL_EMBEDDING_ROUTE_RESOLUTIONS),
          batch_size=batch_size,
          num_positions=selection.final_embedding_positions.shape[1],
      ),
  )


def route_stage_selection(
    positions: Int[Array, 'N R'],
    valid_mask: Bool[Array, 'N R'],
    stage: int,
) -> SequenceResidualSelection:
  """Returns the sequence selection for one statically unrolled route stage."""
  return SequenceResidualSelection(
      positions=positions[stage], valid_mask=valid_mask[stage]
  )


def trace_and_transfer_route_stage(
    values: Float[Array, 'B S D'],
    positions: Int[Array, 'N R'],
    valid_mask: Bool[Array, 'N R'],
    transfer: SequenceResidualBatchTransfer,
    stage: int,
) -> tuple[
    Float[Array, 'B S D'],
    Float[Array, 'B R D'],
    Float[Array, 'B R D'],
]:
  """Captures and applies one transfer, returning effective route values."""
  selection = route_stage_selection(positions, valid_mask, stage)
  natural = gather_sequence_residuals(values, selection)
  effective_values = transfer_sequence_residuals_within_batch(
      values,
      selection,
      transfer.donor_batch_indices[stage],
      transfer.transfer_mask[stage],
  )
  effective = gather_sequence_residuals(effective_values, selection)
  return effective_values, natural, effective


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


def reduce_paired_target(
    predictions: Float[Array, 'B S C'], selection: PairedTargetSelection
) -> TargetSummary:
  """Reduces paired position/track values without forming a Cartesian box."""
  if predictions.ndim != 3:
    raise ValueError(
        'Target predictions must have shape [batch, position, track].'
    )
  if (
      selection.position_indices.ndim != 1
      or selection.track_indices.ndim != 1
      or selection.valid_mask.ndim != 1
  ):
    raise ValueError('Paired target selection arrays must all have rank 1.')
  if not (
      selection.position_indices.shape
      == selection.track_indices.shape
      == selection.valid_mask.shape
  ):
    raise ValueError('Paired target selection arrays must have the same shape.')
  safe_positions = jnp.clip(
      selection.position_indices, 0, predictions.shape[1] - 1
  )
  safe_tracks = jnp.clip(selection.track_indices, 0, predictions.shape[2] - 1)
  values = predictions[:, safe_positions, safe_tracks].astype(jnp.float32)
  values = jnp.where(selection.valid_mask[None, :], values, 0.0)
  total = jnp.sum(values, axis=1, dtype=jnp.float32)
  num_values = jnp.sum(selection.valid_mask, dtype=jnp.int32)
  mean = total / jnp.maximum(num_values, 1)
  return TargetSummary(total=total, mean=mean, num_values=num_values)


def reduce_splice_classification_logit_margin(
    logits: Float[Array, 'B S 5'],
    selection: SpliceClassificationLogitMarginSelection,
) -> TargetSummary:
  """Returns the mean canonical class-minus-padding pre-softmax margin.

  This reducer is intentionally separate from :func:`reduce_paired_target`.
  It consumes the classification head's internal logits and subtracts the
  padding/background logit at each exact canonical endpoint before taking the
  symmetric acceptor/donor mean.  A common shift of all five logits at either
  endpoint therefore cannot change the target.
  """
  if logits.ndim != 3:
    raise ValueError(
        'Splice-classification logits must have shape [batch, position, track].'
    )
  if logits.shape[2] != 5:
    raise ValueError(
        'Splice-classification logit margin requires exactly five tracks.'
    )
  if selection.canonical_position_indices.shape != (2,):
    raise ValueError('Canonical position indices must have shape [2].')
  if selection.canonical_track_indices.shape != (2,):
    raise ValueError('Canonical track indices must have shape [2].')
  if selection.padding_track_index.ndim != 0:
    raise ValueError('Padding track index must be a scalar.')

  safe_positions = jnp.clip(
      selection.canonical_position_indices, 0, logits.shape[1] - 1
  )
  safe_relevant_tracks = jnp.clip(
      selection.canonical_track_indices, 0, logits.shape[2] - 1
  )
  safe_padding_track = jnp.clip(
      selection.padding_track_index, 0, logits.shape[2] - 1
  )
  relevant_logits = logits[
      :, safe_positions, safe_relevant_tracks
  ].astype(jnp.float32)
  padding_logits = logits[:, safe_positions, safe_padding_track].astype(
      jnp.float32
  )
  margins = relevant_logits - padding_logits
  total = jnp.sum(margins, axis=1, dtype=jnp.float32)
  num_values = jnp.asarray(2, dtype=jnp.int32)
  return TargetSummary(
      total=total, mean=total / num_values, num_values=num_values
  )


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


def transfer_sequence_residuals_within_batch(
    residuals: Float[Array, 'B S D'],
    selection: SequenceResidualSelection | None,
    donor_batch_indices: Int[Array, 'B R'] | None,
    transfer_mask: Bool[Array, 'B R'] | None,
) -> Float[Array, 'B S D']:
  """Transfers live selected residuals between rows of the same batch.

  Donor values always come from the input ``residuals`` at this seam, rather
  than from a previously updated recipient. This makes the operation a true
  simultaneous transfer and prevents recipient-to-recipient cascades.
  """
  if donor_batch_indices is None and transfer_mask is None:
    return residuals
  if selection is None:
    raise ValueError('A residual selection is required for batch transfer.')
  if donor_batch_indices is None or transfer_mask is None:
    raise ValueError(
        'Residual donor indices and transfer mask are both required.'
    )
  _validate_residual_selection(selection, residuals)
  batch_size, sequence_length, _ = residuals.shape
  num_positions = selection.positions.shape[0]
  expected_shape = (batch_size, num_positions)
  if donor_batch_indices.shape != expected_shape:
    raise ValueError(
        f'Residual donor indices must have shape {expected_shape}.'
    )
  if transfer_mask.shape != expected_shape:
    raise ValueError(
        f'Residual transfer mask must have shape {expected_shape}.'
    )

  valid_donors = (donor_batch_indices >= 0) & (
      donor_batch_indices < batch_size
  )
  safe_donors = jnp.clip(donor_batch_indices, 0, batch_size - 1)
  safe_positions = jnp.clip(selection.positions, 0, sequence_length - 1)
  donor_values = residuals[
      safe_donors,
      jnp.broadcast_to(safe_positions[None, :], expected_shape),
      :,
  ]
  effective_mask = (
      transfer_mask
      & valid_donors
      & selection.valid_mask[None, :]
  )

  # Apply only enabled slots. This deterministic per-slot update is important
  # when several padded protocol labels map to the same 128-bp token: a later
  # disabled duplicate must not overwrite an earlier enabled transfer.
  output = residuals
  for position_slot in range(num_positions):
    position = safe_positions[position_slot]
    current = output[:, position, :]
    updates = jnp.where(
        effective_mask[:, position_slot, None],
        donor_values[:, position_slot, :].astype(residuals.dtype),
        current,
    )
    output = output.at[:, position, :].set(updates)
  return output
