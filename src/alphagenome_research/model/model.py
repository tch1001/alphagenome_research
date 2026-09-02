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

"""AlphaGenome model."""

from collections.abc import Mapping

from alphagenome import typing
from alphagenome.models import dna_model
from alphagenome_research.model import attention
from alphagenome_research.model import convolutions
from alphagenome_research.model import embeddings as embeddings_module
from alphagenome_research.model import heads as heads_module
from alphagenome_research.model import interpretability
from alphagenome_research.model import layers
from alphagenome_research.model import schemas
from alphagenome_research.model import splicing
from alphagenome_research.model.metadata import metadata as metadata_lib
import haiku as hk
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree, Shaped  # pylint: disable=g-importing-member, g-multiple-import


DEFAULT_NUM_SPLICE_SITES = 512
DEFAULT_SPLICE_SITE_THRESHOLD = 0.1


class SequenceEncoder(hk.Module):
  """Encodes a sequence of DNA into embeddings."""

  @typing.jaxtyped
  def __call__(
      self, dna_sequence: Float[Array, 'B S 4'], *, is_training: bool
  ) -> tuple[Float[Array, 'B S//128 D'], dict[str, Array]]:
    intermediates = {}
    x = convolutions.DnaEmbedder()(dna_sequence, is_training=is_training)
    intermediates['bin_size_1'] = x
    x = layers.pool(x)
    for block_idx, bin_size in enumerate([2, 4, 8, 16, 32, 64]):
      x = convolutions.DownResBlock(f'downres_block_{block_idx}')(
          x, is_training=is_training
      )
      intermediates[f'bin_size_{bin_size}'] = x
      x = layers.pool(x)
    return x, intermediates

  @hk.name_like('__call__')
  def forward_with_route_census(
      self,
      dna_sequence: Float[Array, 'B S 4'],
      *,
      is_training: bool,
      positions: Int[Array, '8 R'],
      valid_mask: Shaped[Array, '8 R'],
      transfer: interpretability.SequenceResidualBatchTransfer,
  ) -> tuple[Array, dict[str, Array], tuple[Array, ...], tuple[Array, ...]]:
    """Runs the encoder with compact live-transfer seams at every resolution."""
    intermediates = {}
    natural_traces = []
    effective_traces = []
    x = convolutions.DnaEmbedder()(dna_sequence, is_training=is_training)
    x, natural, effective = interpretability.trace_and_transfer_route_stage(
        x, positions, valid_mask, transfer, 0
    )
    natural_traces.append(natural)
    effective_traces.append(effective)
    intermediates['bin_size_1'] = x
    x = layers.pool(x)
    for stage, (block_idx, bin_size) in enumerate(
        zip(range(6), (2, 4, 8, 16, 32, 64), strict=True), start=1
    ):
      x = convolutions.DownResBlock(f'downres_block_{block_idx}')(
          x, is_training=is_training
      )
      x, natural, effective = interpretability.trace_and_transfer_route_stage(
          x, positions, valid_mask, transfer, stage
      )
      natural_traces.append(natural)
      effective_traces.append(effective)
      intermediates[f'bin_size_{bin_size}'] = x
      x = layers.pool(x)
    x, natural, effective = interpretability.trace_and_transfer_route_stage(
        x, positions, valid_mask, transfer, 7
    )
    natural_traces.append(natural)
    effective_traces.append(effective)
    return x, intermediates, tuple(natural_traces), tuple(effective_traces)

  @hk.name_like('__call__')
  def forward_with_block_decomposition(
      self,
      dna_sequence: Float[Array, 'B S 4'],
      *,
      positions: Int[Array, '7 R'],
      valid_mask: Shaped[Array, '7 R'],
      channel_indices: Int[Array, 'C'],
      is_training: bool,
  ) -> interpretability.EncoderBlockDecomposition:
    """Returns compact exact residual terms for inherited encoder channels."""
    carried_traces = []
    first_update_traces = []
    second_update_traces = []
    output_traces = []

    def gather(values: Array, stage: int) -> Array:
      selected = interpretability.gather_sequence_residuals(
          values,
          interpretability.route_stage_selection(
              positions, valid_mask, stage
          ),
      )
      return selected[:, :, channel_indices]

    x, carried, first_update = (
        convolutions.DnaEmbedder().forward_with_decomposition(
            dna_sequence, is_training=is_training
        )
    )
    carried_traces.append(gather(carried, 0))
    first_update_traces.append(gather(first_update, 0))
    second_update_traces.append(jnp.zeros_like(first_update_traces[-1]))
    output_traces.append(gather(x, 0))

    x = layers.pool(x)
    for stage, (block_idx, _) in enumerate(
        zip(range(6), (2, 4, 8, 16, 32, 64), strict=True), start=1
    ):
      x, carried, first_update, second_update = (
          convolutions.DownResBlock(
              f'downres_block_{block_idx}'
          ).forward_with_decomposition(x, is_training=is_training)
      )
      carried_traces.append(gather(carried, stage))
      first_update_traces.append(gather(first_update, stage))
      second_update_traces.append(gather(second_update, stage))
      output_traces.append(gather(x, stage))
      if stage < 6:
        x = layers.pool(x)
    return interpretability.EncoderBlockDecomposition(
        carried=jnp.stack(carried_traces),
        first_update=jnp.stack(first_update_traces),
        second_update=jnp.stack(second_update_traces),
        output=jnp.stack(output_traces),
    )


class SequenceDecoder(hk.Module):
  """Decodes a sequence of embeddings."""

  @typing.jaxtyped
  def __call__(
      self,
      x: Float[Array, 'B S D'],
      intermediates: dict[str, Array],
      *,
      is_training: bool,
  ) -> Float[Array, 'B S_final D_final']:
    for bin_size in [64, 32, 16, 8, 4, 2, 1]:
      x = convolutions.UpResBlock()(
          x, intermediates[f'bin_size_{bin_size}'], is_training=is_training
      )
    return x

  @hk.name_like('__call__')
  def forward_with_route_census(
      self,
      x: Float[Array, 'B S D'],
      intermediates: dict[str, Array],
      *,
      is_training: bool,
      skip_positions: Int[Array, '7 RS'],
      skip_valid_mask: Shaped[Array, '7 RS'],
      skip_transfer: interpretability.SequenceResidualBatchTransfer,
      output_positions: Int[Array, '7 RD'],
      output_valid_mask: Shaped[Array, '7 RD'],
      output_transfer: interpretability.SequenceResidualBatchTransfer,
  ) -> tuple[
      Array,
      tuple[Array, ...],
      tuple[Array, ...],
      tuple[Array, ...],
      tuple[Array, ...],
  ]:
    """Runs all U-Net skip-consumption and decoder-output route seams."""
    natural_skip_traces = []
    effective_skip_traces = []
    natural_output_traces = []
    effective_output_traces = []
    for stage, bin_size in enumerate(
        interpretability.DECODER_ROUTE_RESOLUTIONS
    ):
      unet_skip = intermediates[f'bin_size_{bin_size}']
      unet_skip, natural_skip, effective_skip = (
          interpretability.trace_and_transfer_route_stage(
              unet_skip,
              skip_positions,
              skip_valid_mask,
              skip_transfer,
              stage,
          )
      )
      natural_skip_traces.append(natural_skip)
      effective_skip_traces.append(effective_skip)
      x = convolutions.UpResBlock()(
          x, unet_skip, is_training=is_training
      )
      x, natural_output, effective_output = (
          interpretability.trace_and_transfer_route_stage(
              x,
              output_positions,
              output_valid_mask,
              output_transfer,
              stage,
          )
      )
      natural_output_traces.append(natural_output)
      effective_output_traces.append(effective_output)
    return (
        x,
        tuple(natural_skip_traces),
        tuple(effective_skip_traces),
        tuple(natural_output_traces),
        tuple(effective_output_traces),
    )

  @hk.name_like('__call__')
  def forward_with_whole_skip_transfers(
      self,
      x: Float[Array, 'B S D'],
      intermediates: dict[str, Array],
      *,
      is_training: bool,
      skip_transfer: interpretability.WholeSequenceBatchTransfer,
  ) -> tuple[Array, Array, Array, Array, Array]:
    """Decodes with branch-isolated whole live transfers of all U-Net skips."""
    natural_audits = []
    effective_natural_audits = []
    effective_audits = []
    natural_fingerprints = []
    for stage, bin_size in enumerate(
        interpretability.DECODER_ROUTE_RESOLUTIONS
    ):
      unet_skip = intermediates[f'bin_size_{bin_size}']
      (
          unet_skip,
          natural_matches,
          effective_matches_natural,
          effective_matches,
          natural_fingerprint,
      ) = interpretability.transfer_whole_sequence_within_batch(
          unet_skip, skip_transfer, stage
      )
      natural_audits.append(natural_matches)
      effective_natural_audits.append(effective_matches_natural)
      effective_audits.append(effective_matches)
      natural_fingerprints.append(natural_fingerprint)
      x = convolutions.UpResBlock()(
          x, unet_skip, is_training=is_training
      )
    return (
        x,
        jnp.stack(natural_audits),
        jnp.stack(effective_natural_audits),
        jnp.stack(effective_audits),
        jnp.stack(natural_fingerprints),
    )


class TransformerTower(hk.Module):
  """Transformer tower with interleaved pairwise updates."""

  def __init__(
      self,
      *,
      attention_backend: str = attention.ATTENTION_BACKEND_DENSE,
      name: str | None = None,
  ):
    super().__init__(name=name)
    self._attention_backend = attention.validate_attention_backend(
        attention_backend
    )

  @typing.jaxtyped
  def __call__(
      self, x: Float[Array, 'B S C'], *, is_training: bool
  ) -> tuple[Float[Array, 'B S C'], Float[Array, 'B S//16 S//16 F'] | None]:
    pair_x = None
    for i in range(9):
      if i % 2 == 0:
        pair_x = attention.PairUpdateBlock()(x, pair_x)
      mha_bias = attention.AttentionBiasBlock(
          attention_backend=self._attention_backend
      )(pair_x, is_training)
      x += attention.MHABlock(attention_backend=self._attention_backend)(
          x, mha_bias, is_training=is_training
      )
      x += attention.MLPBlock()(x, is_training=is_training)
    return x, pair_x

  @hk.name_like('__call__')
  def forward_with_intermediates(
      self,
      x: Float[Array, 'B S C'],
      *,
      is_training: bool,
      trace_selection: interpretability.TransformerTraceSelection,
      interventions: interpretability.TransformerInterventions,
  ) -> tuple[
      Float[Array, 'B S C'],
      Float[Array, 'B S//16 S//16 F'] | None,
      interpretability.TransformerTrace,
  ]:
    """Runs the tower with selected traces and dynamic causal interventions."""
    interpretability.validate_transformer_interventions(
        interventions,
        trace_selection,
        batch_size=x.shape[0],
        num_heads=8,
        hidden_size=x.shape[-1],
        value_width=192,
    )
    pair_x = None
    compact_bias_traces = []
    effective_compact_bias_traces = []
    head_output_traces = []
    effective_head_output_traces = []
    pre_attention_residual_traces = []
    effective_pre_attention_residual_traces = []
    post_attention_residual_traces = []
    effective_post_attention_residual_traces = []
    post_mlp_residual_traces = []
    effective_post_mlp_residual_traces = []
    for i in range(interpretability.NUM_TRANSFORMER_LAYERS):
      pre_attention_residual_traces.append(
          interpretability.gather_sequence_residuals(
              x, trace_selection.residual_positions
          )
      )
      pre_attention_replacement = interventions.pre_attention_residual
      x = interpretability.replace_sequence_residuals(
          x,
          trace_selection.residual_positions,
          None
          if pre_attention_replacement is None
          else pre_attention_replacement.values[i],
          None
          if pre_attention_replacement is None
          else pre_attention_replacement.replace_mask[i],
      )
      pre_attention_transfer = (
          interventions.pre_attention_residual_transfer
      )
      x = interpretability.transfer_sequence_residuals_within_batch(
          x,
          trace_selection.residual_positions,
          None
          if pre_attention_transfer is None
          else pre_attention_transfer.donor_batch_indices[i],
          None
          if pre_attention_transfer is None
          else pre_attention_transfer.transfer_mask[i],
          None
          if pre_attention_transfer is None
          or pre_attention_transfer.channel_mask is None
          else pre_attention_transfer.channel_mask[i, :x.shape[-1]],
      )
      effective_pre_attention_residual_traces.append(
          interpretability.gather_sequence_residuals(
              x, trace_selection.residual_positions
          )
      )
      if i % 2 == 0:
        pair_x = attention.PairUpdateBlock()(x, pair_x)
      replacement = interpretability.PairBiasReplacement(
          selection=trace_selection.pair_bias_edges,
          values=interventions.pair_bias_values[i],
          replace_mask=interventions.pair_bias_replace_mask[i],
      )
      mha_bias, bias_trace = attention.AttentionBiasBlock(
          attention_backend=self._attention_backend
      ).forward_with_intermediates(
          pair_x,
          is_training,
          edge_selection=trace_selection.pair_bias_edges,
          replacement=replacement,
      )
      mha_update, mha_trace = attention.MHABlock(
          attention_backend=self._attention_backend
      ).forward_with_intermediates(
          x,
          mha_bias,
          is_training=is_training,
          head_output_selection=trace_selection.head_output_positions,
          head_mask=interventions.head_masks[i],
          head_output_replacement_values=(
              None
              if interventions.head_value_output_replacement is None
              else interventions.head_value_output_replacement.values[i]
          ),
          head_output_replace_mask=(
              None
              if interventions.head_value_output_replacement is None
              else interventions.head_value_output_replacement.replace_mask[i]
          ),
      )
      x += mha_update
      post_attention_residual_traces.append(
          interpretability.gather_sequence_residuals(
              x, trace_selection.residual_positions
          )
      )
      post_attention_replacement = interventions.post_attention_residual
      x = interpretability.replace_sequence_residuals(
          x,
          trace_selection.residual_positions,
          None
          if post_attention_replacement is None
          else post_attention_replacement.values[i],
          None
          if post_attention_replacement is None
          else post_attention_replacement.replace_mask[i],
      )
      post_attention_transfer = (
          interventions.post_attention_residual_transfer
      )
      x = interpretability.transfer_sequence_residuals_within_batch(
          x,
          trace_selection.residual_positions,
          None
          if post_attention_transfer is None
          else post_attention_transfer.donor_batch_indices[i],
          None
          if post_attention_transfer is None
          else post_attention_transfer.transfer_mask[i],
          None
          if post_attention_transfer is None
          or post_attention_transfer.channel_mask is None
          else post_attention_transfer.channel_mask[i, :x.shape[-1]],
      )
      effective_post_attention_residual_traces.append(
          interpretability.gather_sequence_residuals(
              x, trace_selection.residual_positions
          )
      )
      x += attention.MLPBlock()(x, is_training=is_training)
      post_mlp_residual_traces.append(
          interpretability.gather_sequence_residuals(
              x, trace_selection.residual_positions
          )
      )
      post_mlp_replacement = interventions.post_mlp_residual
      x = interpretability.replace_sequence_residuals(
          x,
          trace_selection.residual_positions,
          None
          if post_mlp_replacement is None
          else post_mlp_replacement.values[i],
          None
          if post_mlp_replacement is None
          else post_mlp_replacement.replace_mask[i],
      )
      post_mlp_transfer = interventions.post_mlp_residual_transfer
      x = interpretability.transfer_sequence_residuals_within_batch(
          x,
          trace_selection.residual_positions,
          None
          if post_mlp_transfer is None
          else post_mlp_transfer.donor_batch_indices[i],
          None
          if post_mlp_transfer is None
          else post_mlp_transfer.transfer_mask[i],
          None
          if post_mlp_transfer is None
          or post_mlp_transfer.channel_mask is None
          else post_mlp_transfer.channel_mask[i, :x.shape[-1]],
      )
      effective_post_mlp_residual_traces.append(
          interpretability.gather_sequence_residuals(
              x, trace_selection.residual_positions
          )
      )
      assert bias_trace.compact_edges is not None
      assert bias_trace.effective_compact_edges is not None
      assert mha_trace.head_value_outputs is not None
      assert mha_trace.effective_head_value_outputs is not None
      compact_bias_traces.append(bias_trace.compact_edges)
      effective_compact_bias_traces.append(
          bias_trace.effective_compact_edges
      )
      head_output_traces.append(mha_trace.head_value_outputs)
      effective_head_output_traces.append(
          mha_trace.effective_head_value_outputs
      )
    return x, pair_x, interpretability.TransformerTrace(
        compact_pair_bias_edges=jnp.stack(compact_bias_traces),
        effective_compact_pair_bias_edges=jnp.stack(
            effective_compact_bias_traces
        ),
        head_value_outputs=jnp.stack(head_output_traces),
        effective_head_value_outputs=jnp.stack(
            effective_head_output_traces
        ),
        pre_attention_residuals=jnp.stack(pre_attention_residual_traces),
        effective_pre_attention_residuals=jnp.stack(
            effective_pre_attention_residual_traces
        ),
        post_attention_residuals=jnp.stack(post_attention_residual_traces),
        effective_post_attention_residuals=jnp.stack(
            effective_post_attention_residual_traces
        ),
        post_mlp_residuals=jnp.stack(post_mlp_residual_traces),
        effective_post_mlp_residuals=jnp.stack(
            effective_post_mlp_residual_traces
        ),
    )


class AlphaGenome(hk.Module):
  """Main AlphaGenome model.

  The model architecture consists of a sequence encoder, a transformer tower,
  and a sequence decoder. The output of the decoder is used to generate
  embeddings at 1bp resolution, while the output of the transformer tower
  is used to generate embeddings at 128bp resolution and pair embeddings.
  These embeddings are then passed to various heads to make predictions.
  """

  def __init__(
      self,
      output_metadata: Mapping[
          dna_model.Organism, metadata_lib.AlphaGenomeOutputMetadata
      ],
      *,
      num_splice_sites: int = DEFAULT_NUM_SPLICE_SITES,
      splice_site_threshold: float = DEFAULT_SPLICE_SITE_THRESHOLD,
      freeze_trunk_embeddings: bool = False,
      num_organisms: int = 2,
      attention_backend: str = attention.ATTENTION_BACKEND_DENSE,
      name: str | None = None,
  ):
    """Initializes the AlphaGenome model.

    Args:
      output_metadata: Metadata for the output tracks for each organism.
      num_splice_sites: The maximum number of splice sites that are extracted
        from the splice site classification predictions.
      splice_site_threshold: The threshold to use for splice site prediction.
      freeze_trunk_embeddings: Whether to stop the gradient to the embeddings.
        This is useful for training only the heads in fine-tuning.
      num_organisms: The number of organisms. This is used to initialize the
        organism embedding layer. Default is 2, for human and mouse. Leave at 2
        to load pre-trained weights.
      attention_backend: Sequence-attention implementation. ``dense`` keeps
        the original implementation; ``pallas_tiled`` enables the
        experimental forward-only, memory-efficient GPU kernel.
      name: The name of the module.
    """

    super().__init__(name=name or 'alphagenome')
    self._output_metadata = output_metadata
    self._num_splice_sites = num_splice_sites
    self._splice_site_threshold = splice_site_threshold
    self._freeze_trunk_embeddings = freeze_trunk_embeddings
    self._num_organisms = num_organisms
    self._attention_backend = attention.validate_attention_backend(
        attention_backend
    )
    self._heads: dict[heads_module.HeadName, heads_module.Head] = {}
    self._head_configs: dict[heads_module.HeadName, heads_module.HeadConfig] = (
        {}
    )
    for head in heads_module.HeadName:
      head_config = heads_module.get_head_config(head)
      output_type = head_config.output_type
      organisms_with_metadata = [
          organism
          for organism, metadata in output_metadata.items()
          if metadata.get(output_type) is not None
      ]
      if not organisms_with_metadata:
        # None of the organisms have metadata for this output type. Skip.
        continue
      missing_organisms = set(self._output_metadata.keys()) - set(
          organisms_with_metadata
      )
      if missing_organisms:
        raise ValueError(
            f'No metadata found for output type "{output_type.name}" for the'
            f' following organisms: {missing_organisms}. We expect the same set'
            ' of output types for all organisms. Use padding to account for'
            ' missing tracks.'
        )
      self._heads[head] = heads_module.create_head(
          head_config,
          self._output_metadata,
          num_organisms=num_organisms,
      )
      self._head_configs[head] = head_config

  @hk.name_like('__call__')
  def predict_junctions(
      self,
      trunk_embeddings: Float[Array, 'B S D'],
      splice_site_positions: Int[Array, 'B 4 K'],
      organism_index: Int[Array, 'B'],
  ) -> PyTree[Float[Array, 'B ...'] | None]:
    """Predicts splice site junctions from embeddings and splice site positions.

    Args:
      trunk_embeddings: The trunk embeddings to use for predictions.
      splice_site_positions: The splice site positions. Format: [batch, 4,
        num_splice_sites] with order: [donor_pos_idx, accept_pos_idx,
        donor_neg_idx, accept_neg_idx]
      organism_index: The organism index.

    Returns:
      The predictions for splice site junctions.
    """
    junction_head = self._heads.get(heads_module.HeadName.SPLICE_SITES_JUNCTION)
    if junction_head is None:
      raise ValueError('Junction head is not supported by this model.')
    with hk.name_scope('head'):
      return junction_head(
          embeddings_module.Embeddings(embeddings_1bp=trunk_embeddings),
          organism_index,
          splice_site_positions=splice_site_positions,
      )

  @hk.name_like('__call__')
  def forward_trunk(
      self,
      dna_sequence: Float[Array, 'B S 4'],
      organism_index: Int[Array, 'B'],
      *,
      is_training: bool = False,
  ) -> embeddings_module.Embeddings:
    """Encodes a sequence of DNA and makes predictions for various heads.

    Args:
      dna_sequence: The sequence of DNA to encode.
      organism_index: The organism index.
      is_training: Whether the model is in training mode.

    Returns:
      A tuple of (predictions, embeddings), where predictions is a dictionary
      of predictions for various heads.
    """
    trunk, intermediates = SequenceEncoder()(
        dna_sequence, is_training=is_training
    )
    if self._num_organisms >= 1:
      organism_embedding_trunk = embeddings_module.create_default_embedding(
          self._num_organisms, trunk.shape[-1]
      )(organism_index)
      trunk += organism_embedding_trunk[:, None, :]
    trunk, pair_activations = TransformerTower(
        attention_backend=self._attention_backend
    )(trunk, is_training=is_training)

    x = SequenceDecoder()(trunk, intermediates, is_training=is_training)

    embeddings_128bp = embeddings_module.OutputEmbedder(self._num_organisms)(
        trunk, organism_index, is_training=is_training
    )
    embeddings_1bp = embeddings_module.OutputEmbedder(self._num_organisms)(
        x, organism_index, is_training=is_training, skip_x=embeddings_128bp
    )
    embeddings_pair = embeddings_module.OutputPair(self._num_organisms)(
        pair_activations, organism_index
    )

    embeddings = embeddings_module.Embeddings(
        embeddings_1bp=embeddings_1bp,
        embeddings_128bp=embeddings_128bp,
        embeddings_pair=embeddings_pair,
    )
    if self._freeze_trunk_embeddings:
      embeddings = jax.lax.stop_gradient(embeddings)
    return embeddings

  @hk.name_like('__call__')
  def forward_encoder_block_decomposition(
      self,
      dna_sequence: Float[Array, 'B S 4'],
      *,
      positions: Int[Array, '7 R'],
      valid_mask: Shaped[Array, '7 R'],
      channel_indices: Int[Array, 'C'],
      is_training: bool = False,
  ) -> interpretability.EncoderBlockDecomposition:
    """Returns exact selected encoder residual terms without later towers."""
    return SequenceEncoder().forward_with_block_decomposition(
        dna_sequence,
        positions=positions,
        valid_mask=valid_mask,
        channel_indices=channel_indices,
        is_training=is_training,
    )

  @hk.name_like('__call__')
  def forward_trunk_with_intermediates(
      self,
      dna_sequence: Float[Array, 'B S 4'],
      organism_index: Int[Array, 'B'],
      *,
      trace_selection: interpretability.TransformerTraceSelection,
      interventions: interpretability.TransformerInterventions,
      is_training: bool = False,
  ) -> tuple[embeddings_module.Embeddings, interpretability.TransformerTrace]:
    """Runs the trunk and returns selected sequence-attention internals."""
    trunk, intermediates = SequenceEncoder()(
        dna_sequence, is_training=is_training
    )
    if self._num_organisms >= 1:
      organism_embedding_trunk = embeddings_module.create_default_embedding(
          self._num_organisms, trunk.shape[-1]
      )(organism_index)
      trunk += organism_embedding_trunk[:, None, :]
    trunk, pair_activations, trace = TransformerTower(
        attention_backend=self._attention_backend
    ).forward_with_intermediates(
        trunk,
        is_training=is_training,
        trace_selection=trace_selection,
        interventions=interventions,
    )

    x = SequenceDecoder()(trunk, intermediates, is_training=is_training)
    embeddings_128bp = embeddings_module.OutputEmbedder(self._num_organisms)(
        trunk, organism_index, is_training=is_training
    )
    embeddings_1bp = embeddings_module.OutputEmbedder(self._num_organisms)(
        x, organism_index, is_training=is_training, skip_x=embeddings_128bp
    )
    embeddings_pair = embeddings_module.OutputPair(self._num_organisms)(
        pair_activations, organism_index
    )
    embeddings = embeddings_module.Embeddings(
        embeddings_1bp=embeddings_1bp,
        embeddings_128bp=embeddings_128bp,
        embeddings_pair=embeddings_pair,
    )
    if self._freeze_trunk_embeddings:
      embeddings = jax.lax.stop_gradient(embeddings)
    return embeddings, trace

  @hk.name_like('__call__')
  def forward_trunk_with_route_census(
      self,
      dna_sequence: Float[Array, 'B S 4'],
      organism_index: Int[Array, 'B'],
      *,
      trace_selection: interpretability.CausalRouteTraceSelection,
      interventions: interpretability.CausalRouteInterventions,
      is_training: bool = False,
  ) -> tuple[
      embeddings_module.Embeddings, interpretability.CausalRouteTrace
  ]:
    """Runs the complete sequence route with compact live causal seams."""
    interpretability.validate_causal_route_interventions(
        trace_selection, interventions, batch_size=dna_sequence.shape[0]
    )
    (
        trunk,
        intermediates,
        encoder_traces,
        effective_encoder_traces,
    ) = SequenceEncoder().forward_with_route_census(
        dna_sequence,
        is_training=is_training,
        positions=trace_selection.encoder_positions,
        valid_mask=trace_selection.encoder_valid_mask,
        transfer=interventions.encoder_outputs,
    )
    if self._num_organisms >= 1:
      organism_embedding_trunk = embeddings_module.create_default_embedding(
          self._num_organisms, trunk.shape[-1]
      )(organism_index)
      trunk += organism_embedding_trunk[:, None, :]
    trunk, pair_activations, transformer_trace = TransformerTower(
        attention_backend=self._attention_backend
    ).forward_with_intermediates(
        trunk,
        is_training=is_training,
        trace_selection=trace_selection.transformer,
        interventions=interventions.transformer,
    )

    (
        x,
        decoder_skip_traces,
        effective_decoder_skip_traces,
        decoder_output_traces,
        effective_decoder_output_traces,
    ) = SequenceDecoder().forward_with_route_census(
        trunk,
        intermediates,
        is_training=is_training,
        skip_positions=trace_selection.decoder_skip_positions,
        skip_valid_mask=trace_selection.decoder_skip_valid_mask,
        skip_transfer=interventions.decoder_skip_states,
        output_positions=trace_selection.decoder_output_positions,
        output_valid_mask=trace_selection.decoder_output_valid_mask,
        output_transfer=interventions.decoder_outputs,
    )

    natural_final_embeddings = []
    effective_final_embeddings = []
    embeddings_128bp = embeddings_module.OutputEmbedder(self._num_organisms)(
        trunk, organism_index, is_training=is_training
    )
    embeddings_128bp, natural, effective = (
        interpretability.trace_and_transfer_route_stage(
            embeddings_128bp,
            trace_selection.final_embedding_positions,
            trace_selection.final_embedding_valid_mask,
            interventions.final_embeddings,
            0,
        )
    )
    natural_final_embeddings.append(natural)
    effective_final_embeddings.append(effective)
    embeddings_1bp = embeddings_module.OutputEmbedder(self._num_organisms)(
        x,
        organism_index,
        is_training=is_training,
        skip_x=embeddings_128bp,
    )
    embeddings_1bp, natural, effective = (
        interpretability.trace_and_transfer_route_stage(
            embeddings_1bp,
            trace_selection.final_embedding_positions,
            trace_selection.final_embedding_valid_mask,
            interventions.final_embeddings,
            1,
        )
    )
    natural_final_embeddings.append(natural)
    effective_final_embeddings.append(effective)
    embeddings_pair = embeddings_module.OutputPair(self._num_organisms)(
        pair_activations, organism_index
    )
    embeddings = embeddings_module.Embeddings(
        embeddings_1bp=embeddings_1bp,
        embeddings_128bp=embeddings_128bp,
        embeddings_pair=embeddings_pair,
    )
    if self._freeze_trunk_embeddings:
      embeddings = jax.lax.stop_gradient(embeddings)
    return embeddings, interpretability.CausalRouteTrace(
        transformer=transformer_trace,
        encoder_outputs=encoder_traces,
        effective_encoder_outputs=effective_encoder_traces,
        decoder_skip_states=decoder_skip_traces,
        effective_decoder_skip_states=effective_decoder_skip_traces,
        decoder_outputs=decoder_output_traces,
        effective_decoder_outputs=effective_decoder_output_traces,
        final_embeddings=tuple(natural_final_embeddings),
        effective_final_embeddings=tuple(effective_final_embeddings),
    )

  @hk.name_like('__call__')
  def forward_trunk_with_stage_a_branches(
      self,
      dna_sequence: Float[Array, 'B S 4'],
      organism_index: Int[Array, 'B'],
      *,
      selection: interpretability.StageABranchSelection,
      interventions: interpretability.StageABranchInterventions,
      is_training: bool = False,
  ) -> tuple[
      embeddings_module.Embeddings, interpretability.StageABranchTrace
  ]:
    """Runs opt-in whole T/E transfers and the final A/D closure seam."""
    interpretability.validate_stage_a_branch_interventions(
        selection, interventions, batch_size=dna_sequence.shape[0]
    )
    trunk, intermediates = SequenceEncoder()(
        dna_sequence, is_training=is_training
    )
    if self._num_organisms >= 1:
      organism_embedding_trunk = embeddings_module.create_default_embedding(
          self._num_organisms, trunk.shape[-1]
      )(organism_index)
      trunk += organism_embedding_trunk[:, None, :]
    trunk, pair_activations = TransformerTower(
        attention_backend=self._attention_backend
    )(trunk, is_training=is_training)
    (
        trunk,
        transformer_natural_matches,
        transformer_effective_matches_natural,
        transformer_effective_matches,
        transformer_fingerprint,
    ) = (
        interpretability.transfer_whole_sequence_within_batch(
            trunk, interventions.transformer_output, 0
        )
    )

    (
        x,
        skip_natural_matches,
        skip_effective_matches_natural,
        skip_effective_matches,
        skip_fingerprints,
    ) = SequenceDecoder().forward_with_whole_skip_transfers(
        trunk,
        intermediates,
        is_training=is_training,
        skip_transfer=interventions.encoder_skips,
    )
    embeddings_128bp = embeddings_module.OutputEmbedder(self._num_organisms)(
        trunk, organism_index, is_training=is_training
    )
    embeddings_1bp = embeddings_module.OutputEmbedder(self._num_organisms)(
        x,
        organism_index,
        is_training=is_training,
        skip_x=embeddings_128bp,
    )
    final_selection = interpretability.SequenceResidualSelection(
        positions=selection.final_embedding_positions,
        valid_mask=selection.final_embedding_valid_mask,
    )
    natural_final = interpretability.gather_sequence_residuals(
        embeddings_1bp, final_selection
    )
    embeddings_1bp = interpretability.transfer_sequence_residuals_within_batch(
        embeddings_1bp,
        final_selection,
        interventions.final_embedding.donor_batch_indices[0],
        interventions.final_embedding.transfer_mask[0],
        None
        if interventions.final_embedding.channel_mask is None
        else interventions.final_embedding.channel_mask[
            0, :embeddings_1bp.shape[-1]
        ],
    )
    effective_final = interpretability.gather_sequence_residuals(
        embeddings_1bp, final_selection
    )
    embeddings_pair = embeddings_module.OutputPair(self._num_organisms)(
        pair_activations, organism_index
    )
    embeddings = embeddings_module.Embeddings(
        embeddings_1bp=embeddings_1bp,
        embeddings_128bp=embeddings_128bp,
        embeddings_pair=embeddings_pair,
    )
    if self._freeze_trunk_embeddings:
      embeddings = jax.lax.stop_gradient(embeddings)
    return embeddings, interpretability.StageABranchTrace(
        transformer_output_natural_matches_identity=(
            transformer_natural_matches
        ),
        transformer_output_effective_matches_natural=(
            transformer_effective_matches_natural
        ),
        transformer_output_effective_matches_intervention_donor=(
            transformer_effective_matches
        ),
        transformer_output_natural_fingerprint=transformer_fingerprint,
        encoder_skips_natural_match_identity=skip_natural_matches,
        encoder_skips_effective_match_natural=(
            skip_effective_matches_natural
        ),
        encoder_skips_effective_match_intervention_donor=(
            skip_effective_matches
        ),
        encoder_skips_natural_fingerprints=skip_fingerprints,
        natural_final_embeddings=natural_final,
        effective_final_embeddings=effective_final,
    )

  @hk.name_like('__call__')
  def forward_trunk_with_superset_graph(
      self,
      dna_sequence: Float[Array, 'B S 4'],
      organism_index: Int[Array, 'B'],
      *,
      selection: interpretability.SupersetGraphSelection,
      interventions: interpretability.SupersetGraphInterventions,
      is_training: bool = False,
  ) -> tuple[
      embeddings_module.Embeddings, interpretability.SupersetGraphTrace
  ]:
    """Runs the integrated residual and whole-branch causal graph."""
    interpretability.validate_stage_a_branch_interventions(
        selection.stage_a,
        interventions.stage_a,
        batch_size=dna_sequence.shape[0],
    )
    trunk, intermediates = SequenceEncoder()(
        dna_sequence, is_training=is_training
    )
    if self._num_organisms >= 1:
      organism_embedding_trunk = embeddings_module.create_default_embedding(
          self._num_organisms, trunk.shape[-1]
      )(organism_index)
      trunk += organism_embedding_trunk[:, None, :]
    trunk, pair_activations, transformer_trace = TransformerTower(
        attention_backend=self._attention_backend
    ).forward_with_intermediates(
        trunk,
        is_training=is_training,
        trace_selection=selection.transformer,
        interventions=interventions.transformer,
    )
    (
        trunk,
        transformer_natural_matches,
        transformer_effective_matches_natural,
        transformer_effective_matches_donor,
        transformer_fingerprint,
    ) = interpretability.transfer_whole_sequence_within_batch(
        trunk, interventions.stage_a.transformer_output, 0
    )

    (
        x,
        skip_natural_matches,
        skip_effective_matches_natural,
        skip_effective_matches_donor,
        skip_fingerprints,
    ) = SequenceDecoder().forward_with_whole_skip_transfers(
        trunk,
        intermediates,
        is_training=is_training,
        skip_transfer=interventions.stage_a.encoder_skips,
    )
    embeddings_128bp = embeddings_module.OutputEmbedder(self._num_organisms)(
        trunk, organism_index, is_training=is_training
    )
    embeddings_1bp = embeddings_module.OutputEmbedder(self._num_organisms)(
        x,
        organism_index,
        is_training=is_training,
        skip_x=embeddings_128bp,
    )
    final_selection = interpretability.SequenceResidualSelection(
        positions=selection.stage_a.final_embedding_positions,
        valid_mask=selection.stage_a.final_embedding_valid_mask,
    )
    natural_final = interpretability.gather_sequence_residuals(
        embeddings_1bp, final_selection
    )
    embeddings_1bp = interpretability.transfer_sequence_residuals_within_batch(
        embeddings_1bp,
        final_selection,
        interventions.stage_a.final_embedding.donor_batch_indices[0],
        interventions.stage_a.final_embedding.transfer_mask[0],
        None
        if interventions.stage_a.final_embedding.channel_mask is None
        else interventions.stage_a.final_embedding.channel_mask[
            0, :embeddings_1bp.shape[-1]
        ],
    )
    effective_final = interpretability.gather_sequence_residuals(
        embeddings_1bp, final_selection
    )
    embeddings_pair = embeddings_module.OutputPair(self._num_organisms)(
        pair_activations, organism_index
    )
    embeddings = embeddings_module.Embeddings(
        embeddings_1bp=embeddings_1bp,
        embeddings_128bp=embeddings_128bp,
        embeddings_pair=embeddings_pair,
    )
    if self._freeze_trunk_embeddings:
      embeddings = jax.lax.stop_gradient(embeddings)
    stage_a_trace = interpretability.StageABranchTrace(
        transformer_output_natural_matches_identity=(
            transformer_natural_matches
        ),
        transformer_output_effective_matches_natural=(
            transformer_effective_matches_natural
        ),
        transformer_output_effective_matches_intervention_donor=(
            transformer_effective_matches_donor
        ),
        transformer_output_natural_fingerprint=transformer_fingerprint,
        encoder_skips_natural_match_identity=skip_natural_matches,
        encoder_skips_effective_match_natural=(
            skip_effective_matches_natural
        ),
        encoder_skips_effective_match_intervention_donor=(
            skip_effective_matches_donor
        ),
        encoder_skips_natural_fingerprints=skip_fingerprints,
        natural_final_embeddings=natural_final,
        effective_final_embeddings=effective_final,
    )
    return embeddings, interpretability.SupersetGraphTrace(
        transformer=transformer_trace, stage_a=stage_a_trace
    )

  @hk.name_like('__call__')
  def forward_heads(
      self,
      embeddings: embeddings_module.Embeddings,
      organism_index: Int[Array, 'B'],
  ) -> PyTree[Float[Array, 'B ...']]:
    """Computes predictions for various heads from embeddings."""
    predictions: PyTree[Float[Array, 'B ...']] = {
        'embeddings_1bp': embeddings.embeddings_1bp,
    }
    with hk.name_scope('head'):
      for head_name, head_fn in self._heads.items():
        if head_name == heads_module.HeadName.SPLICE_SITES_JUNCTION:
          # This head is handled separately (see below).
          continue
        predictions[head_name.value] = head_fn(
            embeddings,
            organism_index,
        )

    # Handle the splice junction head separately. It requires splice site
    # positions as input, which are derived from the splice site
    # classification predictions.
    if (
        junction_head := heads_module.HeadName.SPLICE_SITES_JUNCTION
    ) in self._heads:
      if heads_module.HeadName.SPLICE_SITES_CLASSIFICATION not in self._heads:
        raise ValueError(
            'SPLICE_SITES_CLASSIFICATION head is required for junctions'
            ' predictions.'
        )
      splice_sites_probabilities = predictions[
          heads_module.HeadName.SPLICE_SITES_CLASSIFICATION.value
      ]['predictions']
      splice_site_positions = splicing.generate_splice_site_positions(
          splice_sites_probabilities,
          alt=None,
          splice_sites=None,
          k=self._num_splice_sites,
          pad_to_length=self._num_splice_sites,
          threshold=self._splice_site_threshold,
      )
      assert isinstance(embeddings.embeddings_1bp, Array)
      predictions[junction_head.value] = self.predict_junctions(
          embeddings.embeddings_1bp, splice_site_positions, organism_index
      )
    return predictions

  @typing.jaxtyped
  def __call__(
      self,
      dna_sequence: Float[Array, 'B S 4'],
      organism_index: Int[Array, 'B'],
  ) -> tuple[PyTree[Shaped[Array, 'B ...']], embeddings_module.Embeddings]:
    """Encodes a sequence of DNA and makes predictions for various heads.

    Args:
      dna_sequence: The sequence of DNA to encode.
      organism_index: The organism index.

    Returns:
      A tuple of (predictions, embeddings), where predictions is a dictionary
      of predictions for various heads.
    """
    embeddings = self.forward_trunk(dna_sequence, organism_index)
    predictions = self.forward_heads(embeddings, organism_index)
    return predictions, embeddings

  @typing.jaxtyped
  def loss(
      self, batch: schemas.DataBatch
  ) -> tuple[
      Float[Array, ''], PyTree[Float[Array, '']], PyTree[Shaped[Array, 'B ...']]
  ]:
    """Returns the loss for the model."""
    predictions, _ = self(
        jnp.asarray(batch.dna_sequence, copy=False),
        jnp.asarray(batch.get_organism_index(), copy=False),
    )
    total_loss, all_scalars = 0.0, {}
    for head_name, head_fn in self._heads.items():
      scalars = head_fn.loss(predictions[head_name.value], batch)
      all_scalars.update(
          {f'{head_name.value}_{k}': v for k, v in scalars.items()}
      )
      total_loss += self._head_configs[head_name].loss_weight * scalars['loss']
    return jnp.asarray(total_loss), all_scalars, predictions
