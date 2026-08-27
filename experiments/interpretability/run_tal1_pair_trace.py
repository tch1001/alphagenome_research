#!/usr/bin/env python3
"""Causal pair-bias and attention-head tracing for the TAL1 tooling control.

This is an experimental research runner, not part of AlphaGenome's public API.
It intentionally uses the exact checkpoint arrays held by ``AlphaGenomeModel``
with the separate instrumented apply function.  The normal prediction API and
checkpoint tree remain unchanged.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from alphagenome.data import genome
from alphagenome.models import dna_model as public_dna_model
from alphagenome_research.io import genome as genome_io
from alphagenome_research.model import attention
from alphagenome_research.model import dna_model
from alphagenome_research.model import interpretability
import jax
import jax.numpy as jnp
import numpy as np


DEFAULT_MANIFEST = Path(__file__).with_name('manifests').joinpath(
    'tal1_patient_3_5_v1.json'
)
TAL1_GENE_START = 47_216_290
TAL1_CANONICAL_TSS = 47_232_225
PDZK1IP1_GENE_START = 47_183_582
PDZK1IP1_GENE_END = 47_191_398


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
  parser.add_argument('--checkpoint', type=Path)
  parser.add_argument('--context-bp', type=int, default=2**17)
  parser.add_argument(
      '--crop-shift-bp',
      type=int,
      default=959,
      help=(
          'Shift the variant-centred crop so the insertion lies inside both '
          'a 128-bp token and a 2,048-bp pair bin, rather than on a boundary.'
      ),
  )
  parser.add_argument('--output', type=Path)
  parser.add_argument(
      '--max-pair-interventions',
      type=int,
      default=432,
      help='Maximum edge/layer/head patches across candidate and controls.',
  )
  parser.add_argument(
      '--max-head-ablations',
      type=int,
      default=72,
      help='Maximum layer/head ablations; 0 skips the scan.',
  )
  parser.add_argument(
      '--max-residual-interventions',
      type=int,
      default=108,
      help='Maximum stage/layer/region residual patches; 0 skips the scan.',
  )
  parser.add_argument(
      '--max-local-head-patches',
      type=int,
      default=144,
      help='Maximum layer/token/head local patches; 0 skips the scan.',
  )
  parser.add_argument('--dry-run', action='store_true')
  return parser.parse_args()


def _checkpoint_path(explicit: Path | None) -> Path:
  if explicit is not None:
    return explicit.expanduser().resolve()
  snapshots = sorted(
      Path.home().glob(
          '.cache/huggingface/hub/models--google--alphagenome-all-folds/'
          'snapshots/*'
      )
  )
  if not snapshots:
    raise FileNotFoundError(
        'No cached all-folds checkpoint found; pass --checkpoint explicitly.'
    )
  return snapshots[-1]


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
  raw = path.read_bytes()
  manifest = json.loads(raw)
  if manifest.get('status') != 'ready':
    raise ValueError(f'Manifest is not runnable: {manifest.get("status")!r}.')
  if manifest.get('role') != 'tooling_control':
    raise ValueError('This runner accepts only an explicitly labelled tooling control.')
  return manifest, hashlib.sha256(raw).hexdigest()


def _variant_from_manifest(manifest: dict[str, Any]) -> genome.Variant:
  variant = manifest['perturbation']['variant']
  return genome.Variant(
      chromosome=variant['chromosome'],
      position=variant['position'],
      reference_bases=variant['reference_bases'],
      alternate_bases=variant['alternate_bases'],
  )


def _relative_index(
    genomic_position_zero_based: int, interval: genome.Interval, resolution: int
) -> int:
  if not interval.start <= genomic_position_zero_based < interval.end:
    raise ValueError(
        f'Position {genomic_position_zero_based} is outside {interval}.'
    )
  return (genomic_position_zero_based - interval.start) // resolution


def _track_index(metadata) -> tuple[int, dict[str, Any]]:
  rna = metadata.rna_seq
  mask = (
      (rna['ontology_curie'] == 'CL:0001059')
      & (rna['strand'] == '.')
      & (rna['Assay title'] == 'polyA plus RNA-seq')
      & ~rna['name'].str.lower().eq('padding')
  )
  matches = np.flatnonzero(mask.to_numpy())
  if len(matches) != 1:
    raise ValueError(f'Expected one CD34+ CMP RNA track, found {len(matches)}.')
  index = int(matches[0])
  return index, {
      key: value.item() if hasattr(value, 'item') else value
      for key, value in rna.iloc[index].to_dict().items()
  }


def _selections(
    interval: genome.Interval, variant: genome.Variant
) -> tuple[
    interpretability.TransformerTraceSelection,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
  # Variant positions are VCF-like 1-based. Model intervals are zero-based.
  variant_zero_based = variant.position - 1
  promoter_zero_based = TAL1_CANONICAL_TSS - 1
  enhancer_bin = _relative_index(variant_zero_based, interval, 2048)
  promoter_bin = _relative_index(promoter_zero_based, interval, 2048)
  promoter_token = _relative_index(promoter_zero_based, interval, 128)
  enhancer_token = _relative_index(variant_zero_based, interval, 128)

  distance_bins = enhancer_bin - promoter_bin
  edge_records = [
      ('promoter_to_enhancer', promoter_bin, enhancer_bin, 'candidate'),
      ('enhancer_to_promoter', enhancer_bin, promoter_bin, 'reverse_control'),
      ('promoter_self', promoter_bin, promoter_bin, 'self_control'),
      ('enhancer_self', enhancer_bin, enhancer_bin, 'self_control'),
      (
          'promoter_to_upstream_distance_match',
          promoter_bin,
          max(0, promoter_bin - distance_bins),
          'distance_control',
      ),
      (
          'enhancer_to_downstream_distance_match',
          enhancer_bin,
          min(interval.width // 2048 - 1, enhancer_bin + distance_bins),
          'distance_control',
      ),
  ]
  max_token = interval.width // 128 - 1
  distance_tokens = enhancer_token - promoter_token
  residual_regions = [
      ('promoter', promoter_token, 'candidate'),
      ('enhancer', enhancer_token, 'candidate'),
      (
          'upstream_distance_match',
          max(2, promoter_token - distance_tokens),
          'distance_control',
      ),
      (
          'downstream_distance_match',
          min(max_token - 2, enhancer_token + distance_tokens),
          'distance_control',
      ),
  ]
  residual_positions = []
  residual_records = []
  for name, center, role in residual_regions:
    selection_start = len(residual_positions)
    region_positions = list(range(center - 2, center + 3))
    residual_positions.extend(region_positions)
    residual_records.append({
        'name': name,
        'role': role,
        'center_token': center,
        'selection_indices': list(
            range(selection_start, selection_start + len(region_positions))
        ),
        'genomic_interval': [
            interval.start + 128 * region_positions[0],
            interval.start + 128 * (region_positions[-1] + 1),
        ],
    })
  selection = interpretability.TransformerTraceSelection(
      pair_bias_edges=interpretability.PairBiasEdgeSelection(
          query_bins=jnp.array([edge[1] for edge in edge_records], jnp.int32),
          key_bins=jnp.array([edge[2] for edge in edge_records], jnp.int32),
          valid_mask=jnp.ones((len(edge_records),), jnp.bool),
      ),
      head_output_positions=interpretability.HeadOutputSelection(
          positions=jnp.array([promoter_token, enhancer_token], jnp.int32),
          valid_mask=jnp.ones((2,), jnp.bool),
      ),
      residual_positions=interpretability.SequenceResidualSelection(
          positions=jnp.array(residual_positions, jnp.int32),
          valid_mask=jnp.ones((len(residual_positions),), jnp.bool),
      ),
  )
  records = [
      {
          'name': name,
          'query_bin': query,
          'key_bin': key,
          'role': role,
          'query_interval': [
              interval.start + 2048 * query,
              interval.start + 2048 * (query + 1),
          ],
          'key_interval': [
              interval.start + 2048 * key,
              interval.start + 2048 * (key + 1),
          ],
      }
      for name, query, key, role in edge_records
  ]
  return selection, records, residual_records


def _target_selection(
    interval: genome.Interval, track_index: int
) -> interpretability.TargetSelection:
  return _region_target_selection(
      interval,
      track_index,
      start_one_based=TAL1_GENE_START,
      end_one_based=TAL1_CANONICAL_TSS,
      padded_size=TAL1_CANONICAL_TSS - TAL1_GENE_START + 1,
  )


def _region_target_selection(
    interval: genome.Interval,
    track_index: int,
    *,
    start_one_based: int,
    end_one_based: int,
    padded_size: int,
) -> interpretability.TargetSelection:
  start = max(start_one_based - 1, interval.start) - interval.start
  end = min(end_one_based, interval.end) - interval.start
  if start >= end:
    raise ValueError('Target region is outside the selected interval.')
  num_valid = end - start
  if num_valid > padded_size:
    raise ValueError('Target region is larger than its fixed padded size.')
  positions = jnp.pad(
      jnp.arange(start, end, dtype=jnp.int32),
      (0, padded_size - num_valid),
  )
  return interpretability.TargetSelection(
      position_indices=positions,
      position_valid_mask=jnp.arange(padded_size) < num_valid,
      track_indices=jnp.array([track_index], jnp.int32),
      track_valid_mask=jnp.ones((1,), jnp.bool),
  )


def _to_json(value: Any) -> Any:
  if dataclasses.is_dataclass(value):
    return {
        field.name: _to_json(getattr(value, field.name))
        for field in dataclasses.fields(value)
    }
  if isinstance(value, (jax.Array, np.ndarray)):
    return np.asarray(value).tolist()
  if isinstance(value, (np.integer, np.floating)):
    return value.item()
  if isinstance(value, Path):
    return str(value)
  return value


def _timed_apply(apply, *args):
  start = time.perf_counter()
  output = apply(*args)
  jax.block_until_ready(output)
  return output, time.perf_counter() - start


def _patch_intervention(
    identity: interpretability.TransformerInterventions,
    donor_trace: interpretability.TransformerTrace,
    *,
    layer: int,
    edge: int,
    head: int,
) -> interpretability.TransformerInterventions:
  return interpretability.TransformerInterventions(
      head_masks=identity.head_masks,
      pair_bias_values=identity.pair_bias_values.at[layer, :, edge, head].set(
          donor_trace.compact_pair_bias_edges[layer, :, edge, head]
      ),
      pair_bias_replace_mask=identity.pair_bias_replace_mask.at[
          layer, :, edge, head
      ].set(True),
  )


def _patch_pair_edge_all_heads(
    identity: interpretability.TransformerInterventions,
    donor_trace: interpretability.TransformerTrace,
    *,
    edge: int,
    layer: int | None,
) -> interpretability.TransformerInterventions:
  """Replaces one compact pair edge across all heads in one/all layers."""
  layer_index = slice(None) if layer is None else layer
  return interpretability.TransformerInterventions(
      head_masks=identity.head_masks,
      pair_bias_values=identity.pair_bias_values.at[
          layer_index, :, edge, :
      ].set(donor_trace.compact_pair_bias_edges[layer_index, :, edge, :]),
      pair_bias_replace_mask=identity.pair_bias_replace_mask.at[
          layer_index, :, edge, :
      ].set(True),
  )


def _head_ablation(
    identity: interpretability.TransformerInterventions, layer: int, head: int
) -> interpretability.TransformerInterventions:
  return interpretability.TransformerInterventions(
      head_masks=identity.head_masks.at[layer, head].set(0),
      pair_bias_values=identity.pair_bias_values,
      pair_bias_replace_mask=identity.pair_bias_replace_mask,
  )


def _residual_patch(
    identity: interpretability.TransformerInterventions,
    donor_trace: interpretability.TransformerTrace,
    *,
    stage: str,
    layer: int,
    selection_indices: list[int],
) -> interpretability.TransformerInterventions:
  trace_field = f'{stage}_residuals'
  values = getattr(donor_trace, trace_field)
  replacement = interpretability.SequenceResidualReplacement(
      values=values,
      replace_mask=jnp.zeros(values.shape[:-1], jnp.bool).at[
          layer, :, jnp.array(selection_indices, jnp.int32)
      ].set(True),
  )
  kwargs = {
      'head_masks': identity.head_masks,
      'pair_bias_values': identity.pair_bias_values,
      'pair_bias_replace_mask': identity.pair_bias_replace_mask,
      'pre_attention_residual': identity.pre_attention_residual,
      'post_attention_residual': identity.post_attention_residual,
      'post_mlp_residual': identity.post_mlp_residual,
  }
  kwargs[f'{stage}_residual'] = replacement
  return interpretability.TransformerInterventions(**kwargs)


def _head_value_patch(
    identity: interpretability.TransformerInterventions,
    donor_trace: interpretability.TransformerTrace,
    *,
    layer: int,
    position_slot: int,
    head: int,
) -> interpretability.TransformerInterventions:
  values = donor_trace.head_value_outputs
  replacement = interpretability.HeadValueOutputReplacement(
      values=values,
      replace_mask=jnp.zeros(values.shape[:-1], jnp.bool).at[
          layer, :, position_slot, head
      ].set(True),
  )
  return interpretability.TransformerInterventions(
      head_masks=identity.head_masks,
      pair_bias_values=identity.pair_bias_values,
      pair_bias_replace_mask=identity.pair_bias_replace_mask,
      head_value_output_replacement=replacement,
      pre_attention_residual=identity.pre_attention_residual,
      post_attention_residual=identity.post_attention_residual,
      post_mlp_residual=identity.post_mlp_residual,
  )


def main() -> None:
  args = _parse_args()
  manifest, manifest_sha256 = _load_manifest(args.manifest)
  variant = _variant_from_manifest(manifest)
  interval = variant.reference_interval.resize(args.context_bp).shift(
      args.crop_shift_bp
  )
  selection, edge_records, residual_records = _selections(interval, variant)
  checkpoint = _checkpoint_path(args.checkpoint)
  output_path = args.output or Path(
      'experiments/interpretability/results/'
      f'tal1_pair_trace_{args.context_bp}_shift{args.crop_shift_bp}.json'
  )

  configuration = {
      'manifest': str(args.manifest.resolve()),
      'manifest_sha256': manifest_sha256,
      'checkpoint': str(checkpoint),
      'context_bp': args.context_bp,
      'crop_shift_bp': args.crop_shift_bp,
      'interval': str(interval),
      'variant': str(variant),
      'variant_zero_based_offset': variant.position - 1 - interval.start,
      'variant_offset_in_128bp_token': (
          variant.position - 1 - interval.start
      )
      % 128,
      'variant_offset_in_2048bp_bin': (
          variant.position - 1 - interval.start
      )
      % 2048,
      'attention_backend': attention.ATTENTION_BACKEND_PALLAS_TILED,
      'target': {
          'head_name': 'rna_seq',
          'prediction_key': 'scaled_predictions_1bp',
          'gene': 'TAL1',
          'gene_start_1_based': TAL1_GENE_START,
          'canonical_tss_1_based': TAL1_CANONICAL_TSS,
      },
      'edges': edge_records,
      'residual_regions': residual_records,
      'limitations': [
          'TAL1 is an official AlphaGenome tooling control, not independent '
          'validation.',
          'The internal ALT trace uses a fixed-length direct sequence insertion '
          'rather than the public indel-stitching prediction path.',
          'A compact pair-bias edge is a 2,048-bp block and is not an attention '
          'probability or biological contact.',
      ],
  }
  if args.dry_run:
    print(json.dumps(configuration, indent=2))
    return

  model_instance = dna_model.create(
      checkpoint,
      model_settings=dna_model.ModelSettings(
          attention_backend=attention.ATTENTION_BACKEND_PALLAS_TILED
      ),
  )
  organism = public_dna_model.Organism.HOMO_SAPIENS
  metadata = model_instance._metadata[organism]  # pylint: disable=protected-access
  track_index, track_metadata = _track_index(metadata)
  target_selection = _target_selection(interval, track_index)
  reference_sequence, alternate_sequence = genome_io.extract_variant_sequences(
      interval,
      variant,
      model_instance._get_fasta_extractor(organism),  # pylint: disable=protected-access
  )
  variant_offset = variant.start - interval.start
  observed_reference = reference_sequence[
      variant_offset : variant_offset + len(variant.reference_bases)
  ]
  if observed_reference != variant.reference_bases:
    raise ValueError(
        f'Reference mismatch: expected {variant.reference_bases!r}, '
        f'observed {observed_reference!r}.'
    )
  configuration['sequence_provenance'] = {
      'reference_sha256': hashlib.sha256(
          reference_sequence.encode('ascii')
      ).hexdigest(),
      'alternate_sha256': hashlib.sha256(
          alternate_sequence.encode('ascii')
      ).hexdigest(),
      'sequence_length': len(reference_sequence),
      'observed_reference_bases': observed_reference,
  }
  encoder = model_instance._one_hot_encoder  # pylint: disable=protected-access
  reference = jnp.asarray(encoder.encode(reference_sequence))[None]
  alternate = jnp.asarray(encoder.encode(alternate_sequence))[None]
  organism_index = jnp.array([0], jnp.int32)

  target_apply = dna_model.create_targeted_interpretability_apply(
      model_instance._metadata,  # pylint: disable=protected-access
      interpretability.TargetSpec(
          head_name='rna_seq', prediction_key='scaled_predictions_1bp'
      ),
      attention_backend=attention.ATTENTION_BACKEND_PALLAS_TILED,
  )
  target_apply = jax.jit(target_apply)
  identity = interpretability.no_transformer_interventions(
      batch_size=1,
      num_edges=len(edge_records),
      dtype=jnp.float32,
  )
  common_args = (
      model_instance._params,  # pylint: disable=protected-access
      model_instance._state,  # pylint: disable=protected-access
      organism_index,
      selection,
  )

  (reference_target, reference_trace), reference_seconds = _timed_apply(
      target_apply,
      common_args[0],
      common_args[1],
      reference,
      common_args[2],
      common_args[3],
      identity,
      target_selection,
  )
  (alternate_target, alternate_trace), alternate_seconds = _timed_apply(
      target_apply,
      common_args[0],
      common_args[1],
      alternate,
      common_args[2],
      common_args[3],
      identity,
      target_selection,
  )
  ref_value = float(reference_target.total[0])
  alt_value = float(alternate_target.total[0])
  denominator = ref_value - alt_value

  pair_patches = []
  for edge, edge_record in enumerate(edge_records):
    for layer in range(interpretability.NUM_TRANSFORMER_LAYERS):
      for head in range(8):
        if len(pair_patches) >= args.max_pair_interventions:
          break
        patched = _patch_intervention(
            identity,
            reference_trace,
            layer=layer,
            edge=edge,
            head=head,
        )
        (patched_target, _), seconds = _timed_apply(
            target_apply,
            common_args[0],
            common_args[1],
            alternate,
            common_args[2],
            common_args[3],
            patched,
            target_selection,
        )
        self_alt_patch = _patch_intervention(
            identity,
            alternate_trace,
            layer=layer,
            edge=edge,
            head=head,
        )
        (self_alt_target, _), self_alt_seconds = _timed_apply(
            target_apply,
            common_args[0],
            common_args[1],
            alternate,
            common_args[2],
            common_args[3],
            self_alt_patch,
            target_selection,
        )
        reciprocal_patch = _patch_intervention(
            identity,
            alternate_trace,
            layer=layer,
            edge=edge,
            head=head,
        )
        (reciprocal_target, _), reciprocal_seconds = _timed_apply(
            target_apply,
            common_args[0],
            common_args[1],
            reference,
            common_args[2],
            common_args[3],
            reciprocal_patch,
            target_selection,
        )
        self_ref_patch = _patch_intervention(
            identity,
            reference_trace,
            layer=layer,
            edge=edge,
            head=head,
        )
        (self_ref_target, _), self_ref_seconds = _timed_apply(
            target_apply,
            common_args[0],
            common_args[1],
            reference,
            common_args[2],
            common_args[3],
            self_ref_patch,
            target_selection,
        )
        patched_value = float(patched_target.total[0])
        self_alt_value = float(self_alt_target.total[0])
        reciprocal_value = float(reciprocal_target.total[0])
        self_ref_value = float(self_ref_target.total[0])
        recovery = (
            (patched_value - alt_value) / denominator
            if abs(denominator) > 1e-8
            else None
        )
        pair_patches.append({
            'layer': layer,
            'head': head,
            'edge': edge_record['name'],
            'edge_role': edge_record['role'],
            'patched_target_total': patched_value,
            'recovery_toward_reference': recovery,
            'seconds': seconds,
            'alternate_self_patched_target_total': self_alt_value,
            'alternate_self_patch_delta_from_baseline': (
                self_alt_value - alt_value
            ),
            'corrected_recovery_toward_reference': (
                (patched_value - self_alt_value) / denominator
                if abs(denominator) > 1e-8
                else None
            ),
            'alternate_self_patch_seconds': self_alt_seconds,
            'alternate_to_reference_patched_target_total': reciprocal_value,
            'recovery_toward_alternate': (
                (reciprocal_value - ref_value) / (alt_value - ref_value)
                if abs(alt_value - ref_value) > 1e-8
                else None
            ),
            'alternate_to_reference_seconds': reciprocal_seconds,
            'reference_self_patched_target_total': self_ref_value,
            'reference_self_patch_delta_from_baseline': (
                self_ref_value - ref_value
            ),
            'corrected_recovery_toward_alternate': (
                (reciprocal_value - self_ref_value)
                / (alt_value - ref_value)
                if abs(alt_value - ref_value) > 1e-8
                else None
            ),
            'reference_self_patch_seconds': self_ref_seconds,
        })
      if len(pair_patches) >= args.max_pair_interventions:
        break
    if len(pair_patches) >= args.max_pair_interventions:
      break

  candidate_edge = 0
  joint_pair_patches = []
  for layer in range(interpretability.NUM_TRANSFORMER_LAYERS):
    joint = _patch_pair_edge_all_heads(
        identity,
        reference_trace,
        edge=candidate_edge,
        layer=layer,
    )
    (joint_target, _), seconds = _timed_apply(
        target_apply,
        common_args[0],
        common_args[1],
        alternate,
        common_args[2],
        common_args[3],
        joint,
        target_selection,
    )
    self_alt_joint = _patch_pair_edge_all_heads(
        identity,
        alternate_trace,
        edge=candidate_edge,
        layer=layer,
    )
    (self_alt_joint_target, _), self_alt_seconds = _timed_apply(
        target_apply,
        common_args[0],
        common_args[1],
        alternate,
        common_args[2],
        common_args[3],
        self_alt_joint,
        target_selection,
    )
    reciprocal_joint = _patch_pair_edge_all_heads(
        identity,
        alternate_trace,
        edge=candidate_edge,
        layer=layer,
    )
    (reciprocal_joint_target, _), reciprocal_seconds = _timed_apply(
        target_apply,
        common_args[0],
        common_args[1],
        reference,
        common_args[2],
        common_args[3],
        reciprocal_joint,
        target_selection,
    )
    self_ref_joint = _patch_pair_edge_all_heads(
        identity,
        reference_trace,
        edge=candidate_edge,
        layer=layer,
    )
    (self_ref_joint_target, _), self_ref_seconds = _timed_apply(
        target_apply,
        common_args[0],
        common_args[1],
        reference,
        common_args[2],
        common_args[3],
        self_ref_joint,
        target_selection,
    )
    joint_value = float(joint_target.total[0])
    self_alt_value = float(self_alt_joint_target.total[0])
    reciprocal_value = float(reciprocal_joint_target.total[0])
    self_ref_value = float(self_ref_joint_target.total[0])
    joint_pair_patches.append({
        'layer': layer,
        'edge': edge_records[candidate_edge]['name'],
        'heads': 'all',
        'patched_target_total': joint_value,
        'recovery_toward_reference': (
            (joint_value - alt_value) / denominator
            if abs(denominator) > 1e-8
            else None
        ),
        'seconds': seconds,
        'alternate_self_patched_target_total': self_alt_value,
        'corrected_recovery_toward_reference': (
            (joint_value - self_alt_value) / denominator
            if abs(denominator) > 1e-8
            else None
        ),
        'alternate_self_patch_seconds': self_alt_seconds,
        'alternate_to_reference_patched_target_total': reciprocal_value,
        'corrected_recovery_toward_alternate': (
            (reciprocal_value - self_ref_value) / (alt_value - ref_value)
            if abs(alt_value - ref_value) > 1e-8
            else None
        ),
        'alternate_to_reference_seconds': reciprocal_seconds,
        'reference_self_patched_target_total': self_ref_value,
        'reference_self_patch_seconds': self_ref_seconds,
    })

  all_candidate = _patch_pair_edge_all_heads(
      identity,
      reference_trace,
      edge=candidate_edge,
      layer=None,
  )
  (all_candidate_target, _), all_candidate_seconds = _timed_apply(
      target_apply,
      common_args[0],
      common_args[1],
      alternate,
      common_args[2],
      common_args[3],
      all_candidate,
      target_selection,
  )
  all_candidate_value = float(all_candidate_target.total[0])
  all_self_alt = _patch_pair_edge_all_heads(
      identity,
      alternate_trace,
      edge=candidate_edge,
      layer=None,
  )
  (all_self_alt_target, _), all_self_alt_seconds = _timed_apply(
      target_apply,
      common_args[0],
      common_args[1],
      alternate,
      common_args[2],
      common_args[3],
      all_self_alt,
      target_selection,
  )
  (all_reciprocal_target, _), all_reciprocal_seconds = _timed_apply(
      target_apply,
      common_args[0],
      common_args[1],
      reference,
      common_args[2],
      common_args[3],
      all_self_alt,
      target_selection,
  )
  all_self_ref = _patch_pair_edge_all_heads(
      identity,
      reference_trace,
      edge=candidate_edge,
      layer=None,
  )
  (all_self_ref_target, _), all_self_ref_seconds = _timed_apply(
      target_apply,
      common_args[0],
      common_args[1],
      reference,
      common_args[2],
      common_args[3],
      all_self_ref,
      target_selection,
  )
  all_self_alt_value = float(all_self_alt_target.total[0])
  all_reciprocal_value = float(all_reciprocal_target.total[0])
  all_self_ref_value = float(all_self_ref_target.total[0])

  head_ablations = []
  for layer in range(interpretability.NUM_TRANSFORMER_LAYERS):
    for head in range(8):
      if len(head_ablations) >= args.max_head_ablations:
        break
      ablation = _head_ablation(identity, layer, head)
      (alt_ablated_target, _), alt_seconds = _timed_apply(
          target_apply,
          common_args[0],
          common_args[1],
          alternate,
          common_args[2],
          common_args[3],
          ablation,
          target_selection,
      )
      (ref_ablated_target, _), ref_seconds = _timed_apply(
          target_apply,
          common_args[0],
          common_args[1],
          reference,
          common_args[2],
          common_args[3],
          ablation,
          target_selection,
      )
      alt_ablated_value = float(alt_ablated_target.total[0])
      ref_ablated_value = float(ref_ablated_target.total[0])
      alt_effect = alt_ablated_value - alt_value
      ref_effect = ref_ablated_value - ref_value
      head_ablations.append({
          'layer': layer,
          'head': head,
          'alternate_ablated_target_total': alt_ablated_value,
          'reference_ablated_target_total': ref_ablated_value,
          'alternate_ablation_effect': alt_effect,
          'reference_ablation_effect': ref_effect,
          'allele_by_ablation_interaction': alt_effect - ref_effect,
          'fraction_of_variant_effect_removed': (
              -(alt_effect - ref_effect) / (alt_value - ref_value)
              if abs(alt_value - ref_value) > 1e-8
              else None
          ),
          'alternate_seconds': alt_seconds,
          'reference_seconds': ref_seconds,
      })
    if len(head_ablations) >= args.max_head_ablations:
      break

  (reference_repeat, _), reference_repeat_seconds = _timed_apply(
      target_apply,
      common_args[0],
      common_args[1],
      reference,
      common_args[2],
      common_args[3],
      identity,
      target_selection,
  )
  (alternate_repeat, _), alternate_repeat_seconds = _timed_apply(
      target_apply,
      common_args[0],
      common_args[1],
      alternate,
      common_args[2],
      common_args[3],
      identity,
      target_selection,
  )

  local_head_patches = []
  for position_slot, position_name in enumerate(('promoter', 'enhancer')):
    for layer in range(interpretability.NUM_TRANSFORMER_LAYERS):
      for head in range(8):
        if len(local_head_patches) >= args.max_local_head_patches:
          break
        denoising_patch = _head_value_patch(
            identity,
            reference_trace,
            layer=layer,
            position_slot=position_slot,
            head=head,
        )
        (denoised_target, _), denoising_seconds = _timed_apply(
            target_apply,
            common_args[0],
            common_args[1],
            alternate,
            common_args[2],
            common_args[3],
            denoising_patch,
            target_selection,
        )
        self_alt_patch = _head_value_patch(
            identity,
            alternate_trace,
            layer=layer,
            position_slot=position_slot,
            head=head,
        )
        (self_alt_target, _), self_alt_seconds = _timed_apply(
            target_apply,
            common_args[0],
            common_args[1],
            alternate,
            common_args[2],
            common_args[3],
            self_alt_patch,
            target_selection,
        )
        noising_patch = _head_value_patch(
            identity,
            alternate_trace,
            layer=layer,
            position_slot=position_slot,
            head=head,
        )
        (noised_target, _), noising_seconds = _timed_apply(
            target_apply,
            common_args[0],
            common_args[1],
            reference,
            common_args[2],
            common_args[3],
            noising_patch,
            target_selection,
        )
        self_ref_patch = _head_value_patch(
            identity,
            reference_trace,
            layer=layer,
            position_slot=position_slot,
            head=head,
        )
        (self_ref_target, _), self_ref_seconds = _timed_apply(
            target_apply,
            common_args[0],
            common_args[1],
            reference,
            common_args[2],
            common_args[3],
            self_ref_patch,
            target_selection,
        )
        denoised_value = float(denoised_target.total[0])
        self_alt_value = float(self_alt_target.total[0])
        noised_value = float(noised_target.total[0])
        self_ref_value = float(self_ref_target.total[0])
        local_head_patches.append({
            'position': position_name,
            'layer': layer,
            'head': head,
            'reference_to_alternate_patched_target_total': denoised_value,
            'recovery_toward_reference': (
                (denoised_value - alt_value) / denominator
                if abs(denominator) > 1e-8
                else None
            ),
            'reference_to_alternate_seconds': denoising_seconds,
            'alternate_self_patched_target_total': self_alt_value,
            'alternate_self_patch_delta_from_baseline': (
                self_alt_value - alt_value
            ),
            'corrected_recovery_toward_reference': (
                (denoised_value - self_alt_value) / denominator
                if abs(denominator) > 1e-8
                else None
            ),
            'alternate_self_patch_seconds': self_alt_seconds,
            'alternate_to_reference_patched_target_total': noised_value,
            'recovery_toward_alternate': (
                (noised_value - ref_value) / (alt_value - ref_value)
                if abs(alt_value - ref_value) > 1e-8
                else None
            ),
            'alternate_to_reference_seconds': noising_seconds,
            'reference_self_patched_target_total': self_ref_value,
            'reference_self_patch_delta_from_baseline': (
                self_ref_value - ref_value
            ),
            'corrected_recovery_toward_alternate': (
                (noised_value - self_ref_value) / (alt_value - ref_value)
                if abs(alt_value - ref_value) > 1e-8
                else None
            ),
            'reference_self_patch_seconds': self_ref_seconds,
        })
      if len(local_head_patches) >= args.max_local_head_patches:
        break
    if len(local_head_patches) >= args.max_local_head_patches:
      break

  residual_patches = []
  for stage in ('pre_attention', 'post_attention', 'post_mlp'):
    for layer in range(interpretability.NUM_TRANSFORMER_LAYERS):
      for region in residual_records:
        if len(residual_patches) >= args.max_residual_interventions:
          break
        patch = _residual_patch(
            identity,
            reference_trace,
            stage=stage,
            layer=layer,
            selection_indices=region['selection_indices'],
        )
        (patched_target, _), seconds = _timed_apply(
            target_apply,
            common_args[0],
            common_args[1],
            alternate,
            common_args[2],
            common_args[3],
            patch,
            target_selection,
        )
        self_alt_patch = _residual_patch(
            identity,
            alternate_trace,
            stage=stage,
            layer=layer,
            selection_indices=region['selection_indices'],
        )
        (self_alt_target, _), self_alt_seconds = _timed_apply(
            target_apply,
            common_args[0],
            common_args[1],
            alternate,
            common_args[2],
            common_args[3],
            self_alt_patch,
            target_selection,
        )
        reciprocal_patch = _residual_patch(
            identity,
            alternate_trace,
            stage=stage,
            layer=layer,
            selection_indices=region['selection_indices'],
        )
        (reciprocal_target, _), reciprocal_seconds = _timed_apply(
            target_apply,
            common_args[0],
            common_args[1],
            reference,
            common_args[2],
            common_args[3],
            reciprocal_patch,
            target_selection,
        )
        self_ref_patch = _residual_patch(
            identity,
            reference_trace,
            stage=stage,
            layer=layer,
            selection_indices=region['selection_indices'],
        )
        (self_ref_target, _), self_ref_seconds = _timed_apply(
            target_apply,
            common_args[0],
            common_args[1],
            reference,
            common_args[2],
            common_args[3],
            self_ref_patch,
            target_selection,
        )
        patched_value = float(patched_target.total[0])
        self_alt_value = float(self_alt_target.total[0])
        reciprocal_value = float(reciprocal_target.total[0])
        self_ref_value = float(self_ref_target.total[0])
        residual_patches.append({
            'stage': stage,
            'layer': layer,
            'region': region['name'],
            'region_role': region['role'],
            'patched_target_total': patched_value,
            'recovery_toward_reference': (
                (patched_value - alt_value) / denominator
                if abs(denominator) > 1e-8
                else None
            ),
            'reference_to_alternate_seconds': seconds,
            'alternate_self_patched_target_total': self_alt_value,
            'alternate_self_patch_delta_from_baseline': (
                self_alt_value - alt_value
            ),
            'corrected_recovery_toward_reference': (
                (patched_value - self_alt_value) / denominator
                if abs(denominator) > 1e-8
                else None
            ),
            'alternate_self_patch_seconds': self_alt_seconds,
            'alternate_to_reference_patched_target_total': reciprocal_value,
            'recovery_toward_alternate': (
                (reciprocal_value - ref_value) / (alt_value - ref_value)
                if abs(alt_value - ref_value) > 1e-8
                else None
            ),
            'alternate_to_reference_seconds': reciprocal_seconds,
            'reference_self_patched_target_total': self_ref_value,
            'reference_self_patch_delta_from_baseline': (
                self_ref_value - ref_value
            ),
            'corrected_recovery_toward_alternate': (
                (reciprocal_value - self_ref_value)
                / (alt_value - ref_value)
                if abs(alt_value - ref_value) > 1e-8
                else None
            ),
            'reference_self_patch_seconds': self_ref_seconds,
        })
      if len(residual_patches) >= args.max_residual_interventions:
        break
    if len(residual_patches) >= args.max_residual_interventions:
      break

  enhancer_region = next(
      region for region in residual_records if region['name'] == 'enhancer'
  )
  specificity_patch = _residual_patch(
      identity,
      reference_trace,
      stage='pre_attention',
      layer=0,
      selection_indices=enhancer_region['selection_indices'],
  )
  specificity_self_alt_patch = _residual_patch(
      identity,
      alternate_trace,
      stage='pre_attention',
      layer=0,
      selection_indices=enhancer_region['selection_indices'],
  )
  specificity_reciprocal_patch = specificity_self_alt_patch
  specificity_self_ref_patch = _residual_patch(
      identity,
      reference_trace,
      stage='pre_attention',
      layer=0,
      selection_indices=enhancer_region['selection_indices'],
  )
  specificity_targets = []
  for gene_name, gene_start, gene_end in (
      ('TAL1', TAL1_GENE_START, TAL1_CANONICAL_TSS),
      ('PDZK1IP1', PDZK1IP1_GENE_START, PDZK1IP1_GENE_END),
  ):
    gene_selection = _region_target_selection(
        interval,
        track_index,
        start_one_based=gene_start,
        end_one_based=gene_end,
        padded_size=TAL1_CANONICAL_TSS - TAL1_GENE_START + 1,
    )
    (gene_ref, _), _ = _timed_apply(
        target_apply,
        common_args[0],
        common_args[1],
        reference,
        common_args[2],
        common_args[3],
        identity,
        gene_selection,
    )
    (gene_alt, _), _ = _timed_apply(
        target_apply,
        common_args[0],
        common_args[1],
        alternate,
        common_args[2],
        common_args[3],
        identity,
        gene_selection,
    )
    (gene_patched, _), _ = _timed_apply(
        target_apply,
        common_args[0],
        common_args[1],
        alternate,
        common_args[2],
        common_args[3],
        specificity_patch,
        gene_selection,
    )
    (gene_self_alt, _), _ = _timed_apply(
        target_apply,
        common_args[0],
        common_args[1],
        alternate,
        common_args[2],
        common_args[3],
        specificity_self_alt_patch,
        gene_selection,
    )
    (gene_reciprocal, _), _ = _timed_apply(
        target_apply,
        common_args[0],
        common_args[1],
        reference,
        common_args[2],
        common_args[3],
        specificity_reciprocal_patch,
        gene_selection,
    )
    (gene_self_ref, _), _ = _timed_apply(
        target_apply,
        common_args[0],
        common_args[1],
        reference,
        common_args[2],
        common_args[3],
        specificity_self_ref_patch,
        gene_selection,
    )
    gene_ref_value = float(gene_ref.total[0])
    gene_alt_value = float(gene_alt.total[0])
    gene_patched_value = float(gene_patched.total[0])
    gene_self_alt_value = float(gene_self_alt.total[0])
    gene_reciprocal_value = float(gene_reciprocal.total[0])
    gene_self_ref_value = float(gene_self_ref.total[0])
    gene_effect = gene_alt_value - gene_ref_value
    specificity_targets.append({
        'gene': gene_name,
        'region_1_based': [gene_start, gene_end],
        'reference_total': gene_ref_value,
        'alternate_total': gene_alt_value,
        'alternate_minus_reference': gene_effect,
        'patched_alternate_total': gene_patched_value,
        'patch_delta_from_alternate': gene_patched_value - gene_alt_value,
        'patch_delta_normalized_by_tal1_variant_effect': (
            (gene_patched_value - gene_alt_value) / (alt_value - ref_value)
            if abs(alt_value - ref_value) > 1e-8
            else None
        ),
        'alternate_self_patched_target_total': gene_self_alt_value,
        'corrected_recovery_toward_reference_own_effect': (
            (gene_patched_value - gene_self_alt_value) / -gene_effect
            if abs(gene_effect) > 1e-8
            else None
        ),
        'alternate_to_reference_patched_target_total': gene_reciprocal_value,
        'reference_self_patched_target_total': gene_self_ref_value,
        'corrected_recovery_toward_alternate_own_effect': (
            (gene_reciprocal_value - gene_self_ref_value) / gene_effect
            if abs(gene_effect) > 1e-8
            else None
        ),
    })

  result = {
      'schema_version': '1.0.0',
      'created_at_unix_s': time.time(),
      'configuration': configuration,
      'environment': {
          'jax_version': jax.__version__,
          'devices': [str(device) for device in jax.local_devices()],
          'xla_python_client_preallocate': os.environ.get(
              'XLA_PYTHON_CLIENT_PREALLOCATE'
          ),
      },
      'patch_direction_schema': {
          'reference_to_alternate': {
              'source_allele': 'reference',
              'recipient_allele': 'alternate',
              'same_recipient_control': 'alternate_to_alternate',
          },
          'alternate_to_reference': {
              'source_allele': 'alternate',
              'recipient_allele': 'reference',
              'same_recipient_control': 'reference_to_reference',
          },
      },
      'target_track_index': track_index,
      'target_track_metadata': track_metadata,
      'baseline': {
          'reference': _to_json(reference_target),
          'alternate': _to_json(alternate_target),
          'alternate_minus_reference_total': alt_value - ref_value,
          'reference_seconds_compile_and_run': reference_seconds,
          'alternate_seconds_warm': alternate_seconds,
          'reference_repeat_total': float(reference_repeat.total[0]),
          'alternate_repeat_total': float(alternate_repeat.total[0]),
          'reference_repeat_delta': float(reference_repeat.total[0]) - ref_value,
          'alternate_repeat_delta': float(alternate_repeat.total[0]) - alt_value,
          'reference_repeat_seconds': reference_repeat_seconds,
          'alternate_repeat_seconds': alternate_repeat_seconds,
      },
      'traces': {
          'reference_compact_pair_bias_edges': _to_json(
              reference_trace.compact_pair_bias_edges
          ),
          'alternate_compact_pair_bias_edges': _to_json(
              alternate_trace.compact_pair_bias_edges
          ),
          'reference_head_value_outputs': _to_json(
              reference_trace.head_value_outputs
          ),
          'alternate_head_value_outputs': _to_json(
              alternate_trace.head_value_outputs
          ),
      },
      'candidate_pair_patches': pair_patches,
      'joint_candidate_pair_patches': joint_pair_patches,
      'all_layers_heads_candidate_pair_patch': {
          'patched_target_total': all_candidate_value,
          'recovery_toward_reference': (
              (all_candidate_value - alt_value) / denominator
              if abs(denominator) > 1e-8
              else None
          ),
          'seconds': all_candidate_seconds,
          'alternate_self_patched_target_total': all_self_alt_value,
          'corrected_recovery_toward_reference': (
              (all_candidate_value - all_self_alt_value) / denominator
              if abs(denominator) > 1e-8
              else None
          ),
          'alternate_self_patch_seconds': all_self_alt_seconds,
          'alternate_to_reference_patched_target_total': all_reciprocal_value,
          'corrected_recovery_toward_alternate': (
              (all_reciprocal_value - all_self_ref_value)
              / (alt_value - ref_value)
              if abs(alt_value - ref_value) > 1e-8
              else None
          ),
          'alternate_to_reference_seconds': all_reciprocal_seconds,
          'reference_self_patched_target_total': all_self_ref_value,
          'reference_self_patch_seconds': all_self_ref_seconds,
      },
      'head_ablations_with_allele_interaction': head_ablations,
      'local_head_value_patches': local_head_patches,
      'reference_to_alternate_residual_patches': residual_patches,
      'specificity_pre_attention_layer0_enhancer_patch': specificity_targets,
  }
  output_path.parent.mkdir(parents=True, exist_ok=True)
  temporary = output_path.with_suffix(output_path.suffix + '.tmp')
  temporary.write_text(json.dumps(result, indent=2) + '\n')
  temporary.replace(output_path)
  print(output_path.resolve())


if __name__ == '__main__':
  main()
