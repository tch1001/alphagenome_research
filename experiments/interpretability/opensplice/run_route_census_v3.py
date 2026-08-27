#!/usr/bin/env python3
"""Development-only OpenSplice v3 causal-route census.

This runner is deliberately locked to the 20 frozen BRAF/SLC25A48 development
SNVs, 16,384 bp context, dense attention, and the internal canonical
splice-classification logit margin.  It cannot select or infer on confirmation
exons.  Every causal component uses the frozen six-row live-transfer batch.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from alphagenome.models import dna_model as public_dna_model
from alphagenome_research.io import genome as genome_io
from alphagenome_research.model import attention
from alphagenome_research.model import dna_model
from alphagenome_research.model import interpretability


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))
import run_inference_trace as v2  # pylint: disable=g-import-not-at-top
import target_reducers_v3 as targets_v3  # pylint: disable=g-import-not-at-top


SCRIPT_VERSION = 'opensplice-route-census-v3.0.0'
SELECTED_PATH = _HERE / 'selected_variants_v2.tsv'
EXONS_PATH = _HERE / 'frozen_exons_v2.tsv'
SELECTED_SHA256 = (
    '09cf0003317d742dfa742481ff6a96896b679342717867b31c85283262a6fdf6'
)
EXONS_SHA256 = (
    'b95f8fc540f19222546322bebfb817a6c0f2147dd41325091086833584a09a75'
)
CHECKPOINT_SNAPSHOT = 'a8f293a76ee73d5b57f3bf2ae146510589fcf187'
CONTEXT_BP = 16_384
ATTENTION_BACKEND = attention.ATTENTION_BACKEND_DENSE
DEVELOPMENT_EXON_ORDERS = (1, 2)
DEVELOPMENT_GENES = ('BRAF', 'SLC25A48')
DEVELOPMENT_VARIANT_COUNT = 20
DEVELOPMENT_EFFECT_COUNT = 12
DEVELOPMENT_NEUTRAL_COUNT = 8
NUM_ROUTE_SLOTS = 3
NUM_DUMMY_EDGES = 1
NUM_DUMMY_HEAD_POSITIONS = 1
EFFECT_THRESHOLD = 0.01
TRACE_BATCH_ROLES = (
    'reference_baseline',
    'alternate_baseline',
    'reference_into_alternate',
    'alternate_into_alternate_self_control',
    'alternate_into_reference',
    'reference_into_reference_self_control',
)
TRACE_BATCH_DNA = ('REF', 'ALT', 'ALT', 'ALT', 'REF', 'REF')
TRACE_BATCH_DONORS = (0, 1, 0, 1, 1, 0)
DEFAULT_OUTPUT_DIR = _HERE / 'results' / 'v3_development_route_census'


@dataclasses.dataclass(frozen=True)
class RouteComponent:
  order: int
  family: str
  stage: int
  resolution_bp: int
  channel_width: int
  seam: str

  @property
  def key(self) -> str:
    return f'{self.order:02d}_{self.family}_stage{self.stage:02d}'


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', type=Path)
  parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument(
      '--max-variants', type=int, default=0,
      help='Development variants to audit; 0 means all 20.',
  )
  parser.add_argument(
      '--max-components', type=int, default=0,
      help='Route components per eligible effect; 0 means all 51.',
  )
  parser.add_argument('--dry-run', action='store_true')
  return parser.parse_args()


def _require_columns(
    fieldnames: Sequence[str] | None, required: Iterable[str], path: Path
) -> None:
  if fieldnames is None:
    raise ValueError(f'{path}: TSV has no header.')
  missing = sorted(set(required) - set(fieldnames))
  if missing:
    raise ValueError(f'{path}: missing columns: {", ".join(missing)}.')


def _verified_hash(path: Path, expected: str) -> str:
  observed = v2._sha256(path)  # pylint: disable=protected-access
  if observed != expected:
    raise ValueError(
        f'Frozen artifact hash mismatch for {path}: {observed} != {expected}.'
    )
  return observed


def load_development_cases() -> tuple[v2.Case, ...]:
  """Loads only the predeclared BRAF/SLC25A48 development partition."""
  _verified_hash(SELECTED_PATH, SELECTED_SHA256)
  _verified_hash(EXONS_PATH, EXONS_SHA256)
  exons: dict[str, v2.Exon] = {}
  with EXONS_PATH.open('r', encoding='utf-8', newline='') as handle:
    reader = csv.DictReader(handle, delimiter='\t')
    _require_columns(reader.fieldnames, v2.REQUIRED_EXON_COLUMNS, EXONS_PATH)
    for row in reader:
      order = int(row['selection_order'])
      if order not in DEVELOPMENT_EXON_ORDERS:
        continue
      exon = v2.Exon(
          order=order,
          gene=row['gene'],
          exon_id=row['exon_id'],
          ensembl_exon_id=row['ensembl_exon_id'],
          chromosome=v2._normalise_chromosome(row['chromosome']),
          strand=row['strand'],
          start_1based=int(row['exon_start_1based']),
          end_1based=int(row['exon_end_1based']),
      )
      if exon.gene != DEVELOPMENT_GENES[order - 1] or exon.strand not in '+-':
        raise ValueError(f'Unexpected frozen development exon: {exon}.')
      exons[exon.ensembl_exon_id] = exon
  if len(exons) != len(DEVELOPMENT_EXON_ORDERS):
    raise ValueError('Frozen development exon partition is incomplete.')

  cases = []
  seen = set()
  with SELECTED_PATH.open('r', encoding='utf-8', newline='') as handle:
    reader = csv.DictReader(handle, delimiter='\t')
    _require_columns(
        reader.fieldnames, v2.REQUIRED_SELECTED_COLUMNS, SELECTED_PATH
    )
    for source_order, row in enumerate(reader):
      exon_order = int(row['exon_order'])
      if exon_order not in DEVELOPMENT_EXON_ORDERS:
        continue
      variant_id = row['variant_id']
      if variant_id in seen:
        raise ValueError(f'Duplicate development variant {variant_id}.')
      seen.add(variant_id)
      try:
        exon = exons[row['ensembl_exon_id']]
      except KeyError as error:
        raise ValueError(f'{variant_id}: development exon mismatch.') from error
      for observed, expected, field in (
          (row['gene'], exon.gene, 'gene'),
          (row['exon_id'], exon.exon_id, 'exon_id'),
          (v2._normalise_chromosome(row['chromosome']), exon.chromosome,
           'chromosome'),
      ):
        if observed != expected:
          raise ValueError(f'{variant_id}: {field} mismatch.')
      ref = row['reference_bases'].upper()
      alt = row['alternate_bases'].upper()
      if (
          row['mut_type'].lower() != 'sub'
          or len(ref) != 1
          or len(alt) != 1
          or ref == alt
          or ref not in 'ACGT'
          or alt not in 'ACGT'
      ):
        raise ValueError(f'{variant_id}: v3 causal path is exact-SNV-only.')
      if not v2._is_true(row['measured']):  # pylint: disable=protected-access
        raise ValueError(f'{variant_id}: variant is not experimentally measured.')
      delta_psi = v2._finite_float(  # pylint: disable=protected-access
          row['delta_psi'], 'delta_psi', variant_id
      )
      delta_logit = v2._finite_float(  # pylint: disable=protected-access
          row['delta_logit'], 'delta_logit', variant_id
      )
      selection_class = row['selection_class']
      is_effect = 'neutral' not in selection_class.lower()
      observed_sign = row['observed_effect_sign'].strip().lower()
      expected_sign = (
          'positive' if delta_logit > 0 else 'negative' if delta_logit < 0
          else 'zero'
      )
      if is_effect:
        if observed_sign != expected_sign:
          raise ValueError(f'{variant_id}: experimental sign mismatch.')
      elif observed_sign not in {'neutral', 'neutral_control'}:
        raise ValueError(f'{variant_id}: invalid neutral sign label.')
      cases.append(v2.Case(
          order=source_order,
          selection_version=row['selection_version'],
          selection_class=selection_class,
          observed_effect_sign=observed_sign,
          gene=exon.gene,
          exon_id=exon.exon_id,
          ensembl_exon_id=exon.ensembl_exon_id,
          chromosome=exon.chromosome,
          strand=exon.strand,
          exon_start_1based=exon.start_1based,
          exon_end_1based=exon.end_1based,
          variant_id=variant_id,
          position_1based=int(row['position_1based']),
          reference_bases=ref,
          alternate_bases=alt,
          region=row['region'],
          mut_type=row['mut_type'],
          delta_psi=delta_psi,
          delta_logit=delta_logit,
      ))
  if len(cases) != DEVELOPMENT_VARIANT_COUNT:
    raise ValueError(
        f'Expected {DEVELOPMENT_VARIANT_COUNT} development variants, '
        f'observed {len(cases)}.'
    )
  if {case.gene for case in cases} != set(DEVELOPMENT_GENES):
    raise ValueError('Development gene allowlist mismatch.')
  if sum(case.is_effect for case in cases) != DEVELOPMENT_EFFECT_COUNT:
    raise ValueError('Development effect count mismatch.')
  if sum(not case.is_effect for case in cases) != DEVELOPMENT_NEUTRAL_COUNT:
    raise ValueError('Development neutral count mismatch.')
  if any(sum(case.gene == gene for case in cases) != 10
         for gene in DEVELOPMENT_GENES):
    raise ValueError('Expected ten development variants per gene.')
  return tuple(cases)


def enumerate_components() -> tuple[RouteComponent, ...]:
  components = []
  for family in interpretability.CAUSAL_ROUTE_FAMILIES:
    if len(family.resolutions_bp) != len(family.channel_widths):
      raise ValueError(f'Invalid route metadata for {family.name}.')
    for stage, (resolution, width) in enumerate(zip(
        family.resolutions_bp, family.channel_widths, strict=True
    )):
      components.append(RouteComponent(
          order=len(components), family=family.name, stage=stage,
          resolution_bp=resolution, channel_width=width, seam=family.seam,
      ))
  if len(components) != 51:
    raise ValueError(f'Expected 51 causal-route components, got {len(components)}.')
  return tuple(components)


def _positions_for_resolution(
    positions_1based: Sequence[int], interval_start_0based: int, resolution: int
) -> tuple[np.ndarray, np.ndarray]:
  tokens = tuple(dict.fromkeys(
      (position - 1 - interval_start_0based) // resolution
      for position in positions_1based
  ))
  if len(tokens) > NUM_ROUTE_SLOTS:
    raise ValueError('Route selection exceeds its fixed three slots.')
  sequence_length = CONTEXT_BP // resolution
  if any(token < 0 or token >= sequence_length for token in tokens):
    raise ValueError('Route position lies outside model interval.')
  padded = np.zeros((NUM_ROUTE_SLOTS,), np.int32)
  valid = np.zeros((NUM_ROUTE_SLOTS,), bool)
  padded[:len(tokens)] = tokens
  valid[:len(tokens)] = True
  return padded, valid


def route_selection(
    case: v2.Case, interval
) -> interpretability.CausalRouteTraceSelection:
  sites = v2.canonical_sites(case)
  positions_1based = tuple(dict.fromkeys((
      case.position_1based, sites[0][1], sites[1][1]
  )))

  def stack(resolutions):
    pairs = [
        _positions_for_resolution(positions_1based, interval.start, resolution)
        for resolution in resolutions
    ]
    return (
        jnp.asarray(np.stack([pair[0] for pair in pairs])),
        jnp.asarray(np.stack([pair[1] for pair in pairs])),
    )

  encoder_positions, encoder_valid = stack(
      interpretability.ENCODER_ROUTE_RESOLUTIONS
  )
  decoder_positions, decoder_valid = stack(
      interpretability.DECODER_ROUTE_RESOLUTIONS
  )
  final_positions, final_valid = stack(
      interpretability.FINAL_EMBEDDING_ROUTE_RESOLUTIONS
  )
  transformer_positions, transformer_valid = _positions_for_resolution(
      positions_1based, interval.start, 128
  )
  transformer = interpretability.TransformerTraceSelection(
      pair_bias_edges=interpretability.PairBiasEdgeSelection(
          query_bins=jnp.zeros((NUM_DUMMY_EDGES,), jnp.int32),
          key_bins=jnp.zeros((NUM_DUMMY_EDGES,), jnp.int32),
          valid_mask=jnp.zeros((NUM_DUMMY_EDGES,), jnp.bool),
      ),
      head_output_positions=interpretability.HeadOutputSelection(
          positions=jnp.zeros((NUM_DUMMY_HEAD_POSITIONS,), jnp.int32),
          valid_mask=jnp.zeros((NUM_DUMMY_HEAD_POSITIONS,), jnp.bool),
      ),
      residual_positions=interpretability.SequenceResidualSelection(
          positions=jnp.asarray(transformer_positions),
          valid_mask=jnp.asarray(transformer_valid),
      ),
  )
  return interpretability.CausalRouteTraceSelection(
      transformer=transformer,
      encoder_positions=encoder_positions,
      encoder_valid_mask=encoder_valid,
      decoder_skip_positions=decoder_positions,
      decoder_skip_valid_mask=decoder_valid,
      decoder_output_positions=decoder_positions,
      decoder_output_valid_mask=decoder_valid,
      final_embedding_positions=final_positions,
      final_embedding_valid_mask=final_valid,
  )


def target_selection(
    metadata, case: v2.Case, interval
) -> tuple[
    interpretability.SpliceClassificationLogitMarginSelection,
    targets_v3.ClassificationLogitTarget,
]:
  resolved = targets_v3.classification_logit_target(
      metadata,
      interval_start_0based=interval.start,
      interval_width=interval.width,
      exon_start_1based=case.exon_start_1based,
      exon_end_1based=case.exon_end_1based,
      strand=case.strand,
  )
  if tuple(endpoint.role for endpoint in resolved.endpoints) != (
      'acceptor', 'donor'
  ):
    raise ValueError('Canonical endpoints are not ordered [acceptor, donor].')
  selection = interpretability.SpliceClassificationLogitMarginSelection(
      canonical_position_indices=jnp.asarray(
          [endpoint.position_index for endpoint in resolved.endpoints],
          jnp.int32,
      ),
      canonical_track_indices=jnp.asarray(
          [endpoint.track_index for endpoint in resolved.endpoints], jnp.int32
      ),
      padding_track_index=jnp.asarray(resolved.padding_track_index, jnp.int32),
  )
  return selection, resolved


def component_interventions(
    selection: interpretability.CausalRouteTraceSelection,
    component: RouteComponent | None,
) -> interpretability.CausalRouteInterventions:
  identity = interpretability.no_causal_route_interventions(
      selection,
      batch_size=interpretability.PAIRED_CAUSAL_BATCH_SIZE,
      num_edges=NUM_DUMMY_EDGES,
  )
  if component is None:
    return identity

  if component.family.startswith('transformer_'):
    stage_name = component.family.removeprefix('transformer_')
    mask = jnp.zeros(
        (interpretability.NUM_TRANSFORMER_LAYERS, NUM_ROUTE_SLOTS), jnp.bool
    ).at[component.stage].set(
        selection.transformer.residual_positions.valid_mask
    )
    transfer = interpretability.paired_six_row_batch_transfer(mask)
    transformer = dataclasses.replace(
        identity.transformer,
        **{f'{stage_name}_residual_transfer': transfer},
    )
    return dataclasses.replace(identity, transformer=transformer)

  mapping = {
      'encoder_outputs': ('encoder_outputs', selection.encoder_valid_mask),
      'decoder_skip_states': (
          'decoder_skip_states', selection.decoder_skip_valid_mask
      ),
      'decoder_outputs': (
          'decoder_outputs', selection.decoder_output_valid_mask
      ),
      'final_embeddings': (
          'final_embeddings', selection.final_embedding_valid_mask
      ),
  }
  try:
    field, valid = mapping[component.family]
  except KeyError as error:
    raise ValueError(f'Unknown route family {component.family!r}.') from error
  mask = jnp.zeros_like(valid).at[component.stage].set(valid[component.stage])
  return dataclasses.replace(
      identity,
      **{field: interpretability.paired_six_row_batch_transfer(mask)},
  )


def _trace_arrays(
    trace: interpretability.CausalRouteTrace,
) -> list[tuple[str, np.ndarray, int]]:
  arrays = []
  for field in dataclasses.fields(trace.transformer):
    arrays.append((
        f'transformer.{field.name}',
        np.asarray(getattr(trace.transformer, field.name)), 1,
    ))
  for field_name in (
      'encoder_outputs', 'effective_encoder_outputs',
      'decoder_skip_states', 'effective_decoder_skip_states',
      'decoder_outputs', 'effective_decoder_outputs',
      'final_embeddings', 'effective_final_embeddings',
  ):
    for stage, value in enumerate(getattr(trace, field_name)):
      arrays.append((f'{field_name}.{stage}', np.asarray(value), 0))
  return arrays


def _take_batch(value: np.ndarray, axis: int, row: int) -> np.ndarray:
  return np.take(value, row, axis=axis)


def _target_values(target: interpretability.TargetSummary) -> np.ndarray:
  means = np.asarray(target.mean)
  totals = np.asarray(target.total)
  num_values = int(np.asarray(target.num_values))
  if means.shape != (6,) or totals.shape != (6,) or num_values != 2:
    raise ValueError(
        f'Invalid locked target shapes/count: {means.shape}, {totals.shape}, '
        f'{num_values}.'
    )
  if not np.isfinite(means).all() or not np.isfinite(totals).all():
    raise ValueError('Non-finite target summary.')
  if not np.array_equal(totals, means * 2):
    raise ValueError('Target total/mean/num_values audit failed.')
  return means


def validate_identity_audit(
    first_target: interpretability.TargetSummary,
    first_trace: interpretability.CausalRouteTrace,
    second_target: interpretability.TargetSummary,
    second_trace: interpretability.CausalRouteTrace,
) -> dict[str, Any]:
  first = _target_values(first_target)
  second = _target_values(second_target)
  if not np.array_equal(first, second):
    raise ValueError('Gate 0 target repeat is not bit-exact.')
  if not (first[0] == first[4] == first[5]):
    raise ValueError('Gate 0 REF duplicate targets differ.')
  if not (first[1] == first[2] == first[3]):
    raise ValueError('Gate 0 ALT duplicate targets differ.')
  first_arrays = _trace_arrays(first_trace)
  second_arrays = _trace_arrays(second_trace)
  if len(first_arrays) != len(second_arrays):
    raise ValueError('Gate 0 trace structure differs.')
  for (name, value, batch_axis), (other_name, other, other_axis) in zip(
      first_arrays, second_arrays, strict=True
  ):
    if name != other_name or batch_axis != other_axis or not np.array_equal(
        value, other
    ):
      raise ValueError(f'Gate 0 trace repeat failed at {name}.')
    for natural_row, duplicate_rows in ((0, (4, 5)), (1, (2, 3))):
      natural = _take_batch(value, batch_axis, natural_row)
      if any(not np.array_equal(
          natural, _take_batch(value, batch_axis, duplicate)
      ) for duplicate in duplicate_rows):
        raise ValueError(f'Gate 0 duplicate traces differ at {name}.')
  return {
      'passed': True,
      'target_repeat_exact': True,
      'target_duplicate_rows_exact': True,
      'trace_repeat_exact': True,
      'trace_duplicate_rows_exact': True,
      'target_total_equals_two_times_mean': True,
      'num_values': 2,
      'target_means': dict(zip(TRACE_BATCH_ROLES, first.tolist(), strict=True)),
  }


def _component_values(
    trace: interpretability.CausalRouteTrace, component: RouteComponent
) -> tuple[np.ndarray, np.ndarray]:
  if component.family.startswith('transformer_'):
    stage_name = component.family.removeprefix('transformer_')
    return (
        np.asarray(getattr(trace.transformer, f'{stage_name}_residuals'))[
            component.stage
        ],
        np.asarray(getattr(
            trace.transformer, f'effective_{stage_name}_residuals'
        ))[component.stage],
    )
  natural_field, effective_field = {
      'encoder_outputs': ('encoder_outputs', 'effective_encoder_outputs'),
      'decoder_skip_states': (
          'decoder_skip_states', 'effective_decoder_skip_states'
      ),
      'decoder_outputs': ('decoder_outputs', 'effective_decoder_outputs'),
      'final_embeddings': ('final_embeddings', 'effective_final_embeddings'),
  }[component.family]
  return (
      np.asarray(getattr(trace, natural_field)[component.stage]),
      np.asarray(getattr(trace, effective_field)[component.stage]),
  )


def validate_component_audit(
    target: interpretability.TargetSummary,
    trace: interpretability.CausalRouteTrace,
    component: RouteComponent,
    identity_target_means: Sequence[float],
    valid_slots: Sequence[bool],
) -> dict[str, Any]:
  values = _target_values(target)
  identity = np.asarray(identity_target_means)
  if not np.array_equal(values[:2], identity[:2]):
    raise ValueError('Live component baseline target rows drifted from Gate 0.')
  if values[3] != values[1] or values[5] != values[0]:
    raise ValueError('Live component self-control targets are not exact.')
  natural, effective = _component_values(trace, component)
  slots = np.flatnonzero(np.asarray(valid_slots, dtype=bool))
  if not len(slots):
    raise ValueError('Active component has no valid selected route slots.')
  vector_checks = {}
  for recipient, donor, label in (
      (2, 0, 'reference_into_alternate'),
      (3, 1, 'alternate_into_alternate_self_control'),
      (4, 1, 'alternate_into_reference'),
      (5, 0, 'reference_into_reference_self_control'),
  ):
    exact = np.array_equal(natural[donor, slots], effective[recipient, slots])
    vector_checks[label] = bool(exact)
    if not exact:
      raise ValueError(f'Live donor-vector audit failed for {label}.')
  ref, alt, ref_alt, alt_alt, alt_ref, ref_ref = values.tolist()

  def recovery(patched, self_control, donor, recipient):
    denominator = donor - recipient
    return None if denominator == 0 else (patched - self_control) / denominator

  r_ref_into_alt = recovery(ref_alt, alt_alt, ref, alt)
  r_alt_into_ref = recovery(alt_ref, ref_ref, alt, ref)
  bottleneck = (
      None if r_ref_into_alt is None or r_alt_into_ref is None
      else min(r_ref_into_alt, r_alt_into_ref)
  )
  return {
      'passed': True,
      'baseline_targets_exact_from_gate0': True,
      'self_targets_exact': True,
      'donor_vectors_exact': vector_checks,
      'target_means': dict(zip(TRACE_BATCH_ROLES, values.tolist(), strict=True)),
      'raw_movement': {
          'reference_into_alternate': ref_alt - alt_alt,
          'alternate_into_reference': alt_ref - ref_ref,
      },
      'self_control_corrected_recovery': {
          'reference_into_alternate': r_ref_into_alt,
          'alternate_into_reference': r_alt_into_ref,
          'bidirectional_bottleneck': bottleneck,
      },
  }


def _valid_mask_for_component(
    selection: interpretability.CausalRouteTraceSelection,
    component: RouteComponent,
) -> np.ndarray:
  if component.family.startswith('transformer_'):
    return np.asarray(selection.transformer.residual_positions.valid_mask)
  return np.asarray({
      'encoder_outputs': selection.encoder_valid_mask,
      'decoder_skip_states': selection.decoder_skip_valid_mask,
      'decoder_outputs': selection.decoder_output_valid_mask,
      'final_embeddings': selection.final_embedding_valid_mask,
  }[component.family][component.stage])


def direction_gate(case: v2.Case, target_means: Sequence[float]) -> dict[str, Any]:
  delta = float(target_means[1] - target_means[0])
  experimental_sign = int(np.sign(case.delta_logit))
  predicted_sign = int(np.sign(delta))
  eligible = bool(
      case.is_effect
      and abs(delta) >= EFFECT_THRESHOLD
      and predicted_sign == experimental_sign
  )
  return {
      'predicted_alt_minus_ref_logit_margin': delta,
      'experimental_delta_logit': case.delta_logit,
      'minimum_absolute_predicted_effect': EFFECT_THRESHOLD,
      'direction_matches_delta_logit': (
          predicted_sign == experimental_sign if case.is_effect else None
      ),
      'eligible_for_causal_census': eligible,
  }


def _build_six_row_batch(model_instance, case: v2.Case, interval):
  extractor = model_instance._get_fasta_extractor(  # pylint: disable=protected-access
      public_dna_model.Organism.HOMO_SAPIENS
  )
  raw = extractor.extract(interval)
  offset = case.position_1based - 1 - interval.start
  if raw[offset].upper() != case.reference_bases:
    raise ValueError(f'{case.variant_id}: FASTA reference allele mismatch.')
  reference_sequence, alternate_sequence = genome_io.extract_variant_sequences(
      interval, case.variant, extractor
  )
  if len(reference_sequence) != CONTEXT_BP or len(alternate_sequence) != CONTEXT_BP:
    raise ValueError('Causal route census requires aligned 16kb SNVs.')
  encoder = model_instance._one_hot_encoder  # pylint: disable=protected-access
  reference = jnp.asarray(encoder.encode(reference_sequence))[None]
  alternate = jnp.asarray(encoder.encode(alternate_sequence))[None]
  return jnp.concatenate(
      (reference, alternate, alternate, alternate, reference, reference), axis=0
  ), {
      'reference': hashlib.sha256(reference_sequence.encode('ascii')).hexdigest(),
      'alternate': hashlib.sha256(alternate_sequence.encode('ascii')).hexdigest(),
  }


def _timed_apply(apply_fn, *args):
  start = time.perf_counter()
  output = apply_fn(*args)
  jax.block_until_ready(output)
  return output, time.perf_counter() - start


def _git_output(repo: Path, *args: str) -> bytes:
  return subprocess.check_output(('git', '-C', str(repo), *args))


def code_fingerprint() -> dict[str, Any]:
  repo = _HERE.parents[2]
  head = _git_output(repo, 'rev-parse', 'HEAD').decode().strip()
  tracked_diff = _git_output(
      repo, 'diff', '--binary', 'HEAD', '--',
      'src/alphagenome_research/model/interpretability.py',
      'src/alphagenome_research/model/model.py',
      'src/alphagenome_research/model/dna_model.py',
  )
  files = (
      Path(__file__).resolve(),
      _HERE / 'target_reducers_v3.py',
      repo / 'src/alphagenome_research/model/interpretability.py',
      repo / 'src/alphagenome_research/model/model.py',
      repo / 'src/alphagenome_research/model/dna_model.py',
  )
  return {
      'git_head': head,
      'tracked_dirty_diff_sha256': hashlib.sha256(tracked_diff).hexdigest(),
      'file_sha256': {str(path.relative_to(repo)): v2._sha256(path) for path in files},
  }


def frozen_configuration(checkpoint: Path) -> dict[str, Any]:
  snapshot = checkpoint.name if checkpoint.name == CHECKPOINT_SNAPSHOT else None
  if snapshot != CHECKPOINT_SNAPSHOT:
    raise ValueError(
        f'Expected checkpoint snapshot {CHECKPOINT_SNAPSHOT}, got {checkpoint}.'
    )
  return {
      'script_version': SCRIPT_VERSION,
      'selected_variants_sha256': SELECTED_SHA256,
      'frozen_exons_sha256': EXONS_SHA256,
      'development_exon_orders': DEVELOPMENT_EXON_ORDERS,
      'development_genes': DEVELOPMENT_GENES,
      'checkpoint_path': str(checkpoint),
      'checkpoint_snapshot': CHECKPOINT_SNAPSHOT,
      'code': code_fingerprint(),
      'attention_backend': ATTENTION_BACKEND,
      'context_bp': CONTEXT_BP,
      'target': (
          'mean_of_acceptor_and_donor_classification_logit_minus_padding_logit'
      ),
      'target_head_key': 'splice_sites_classification/logits',
      'paired_batch_roles': TRACE_BATCH_ROLES,
      'paired_batch_dna': TRACE_BATCH_DNA,
      'paired_batch_donor_rows': TRACE_BATCH_DONORS,
      'selected_route_positions': 'S=unique(V_union_acceptor_union_donor)',
  }


def case_configuration(
    frozen: Mapping[str, Any], case: v2.Case, interval, resolved_target,
    sequence_sha256: Mapping[str, str] | None,
) -> dict[str, Any]:
  return {
      **frozen,
      'case': v2._case_record(case),  # pylint: disable=protected-access
      'interval': {
          'chromosome': interval.chromosome,
          'start_0based': interval.start,
          'end_0based_exclusive': interval.end,
      },
      'exon': {
          'start_1based': case.exon_start_1based,
          'end_1based': case.exon_end_1based,
          'strand': case.strand,
      },
      'canonical_target': {
          'endpoints': [dataclasses.asdict(x) for x in resolved_target.endpoints],
          'padding_track_index': resolved_target.padding_track_index,
      },
      'sequence_sha256': sequence_sha256,
  }


def _artifact_path(output_dir: Path, case: v2.Case, component=None) -> Path:
  slug = v2._slug(case.variant_id)  # pylint: disable=protected-access
  case_dir = f'{case.order:03d}_{slug}'
  if component is None:
    return output_dir / 'identity' / f'{case_dir}.json'
  return output_dir / 'components' / case_dir / f'{component.key}.json'


def _run_identity(
    model_instance, apply_fn, case: v2.Case, frozen: Mapping[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], Any, Any, Any, Any]:
  interval = v2.centered_interval(case, CONTEXT_BP)
  selection = route_selection(case, interval)
  metadata = model_instance._metadata[  # pylint: disable=protected-access
      public_dna_model.Organism.HOMO_SAPIENS
  ].splice_sites
  target, resolved = target_selection(metadata, case, interval)
  dna_batch, sequence_sha = _build_six_row_batch(model_instance, case, interval)
  configuration = {
      **case_configuration(frozen, case, interval, resolved, sequence_sha),
      'kind': 'gate0_all_false_identity_duplicate_repeat',
  }
  fingerprint = v2._fingerprint(configuration)  # pylint: disable=protected-access
  path = _artifact_path(output_dir, case)
  completed = v2._load_completed(path, fingerprint)  # pylint: disable=protected-access
  organism = jnp.zeros((6,), jnp.int32)
  common = (
      model_instance._params,  # pylint: disable=protected-access
      model_instance._state,  # pylint: disable=protected-access
      dna_batch,
      organism,
      selection,
  )
  if completed is None:
    intervention = component_interventions(selection, None)
    (first_target, first_trace), first_seconds = _timed_apply(
        apply_fn, *common, intervention, target
    )
    (second_target, second_trace), second_seconds = _timed_apply(
        apply_fn, *common, intervention, target
    )
    checks = validate_identity_audit(
        first_target, first_trace, second_target, second_trace
    )
    gate = direction_gate(case, np.asarray(first_target.mean))
    completed = {
        'status': 'complete', 'fingerprint': fingerprint,
        'configuration': configuration, 'checks': checks,
        'direction_gate': gate,
        'seconds': {'first_compile_and_run': first_seconds,
                    'exact_repeat': second_seconds},
        'created_at_unix_s': time.time(),
    }
    v2._write_atomic(path, completed)  # pylint: disable=protected-access
  return completed, common, target, selection, resolved


def _run_component(
    apply_fn, case: v2.Case, identity: Mapping[str, Any], common,
    target, selection, component: RouteComponent, frozen: Mapping[str, Any],
    resolved_target, output_dir: Path,
) -> dict[str, Any]:
  interval = v2.centered_interval(case, CONTEXT_BP)
  configuration = {
      **case_configuration(frozen, case, interval, resolved_target,
                           identity['configuration']['sequence_sha256']),
      'kind': 'six_row_live_causal_route_component',
      'gate0_fingerprint': identity['fingerprint'],
      'component': dataclasses.asdict(component),
  }
  fingerprint = v2._fingerprint(configuration)  # pylint: disable=protected-access
  path = _artifact_path(output_dir, case, component)
  completed = v2._load_completed(  # pylint: disable=protected-access
      path, fingerprint
  )
  if completed:
    return completed
  intervention = component_interventions(selection, component)
  (batch_target, batch_trace), seconds = _timed_apply(
      apply_fn, *common, intervention, target
  )
  checks = validate_component_audit(
      batch_target, batch_trace, component,
      tuple(identity['checks']['target_means'][role] for role in TRACE_BATCH_ROLES),
      _valid_mask_for_component(selection, component),
  )
  result = {
      'status': 'complete', 'fingerprint': fingerprint,
      'configuration': configuration, 'checks': checks,
      'seconds_compile_and_run': seconds, 'created_at_unix_s': time.time(),
  }
  v2._write_atomic(path, result)  # pylint: disable=protected-access
  return result


def build_dry_run_plan(
    cases: Sequence[v2.Case], components: Sequence[RouteComponent]
) -> dict[str, Any]:
  return {
      'script_version': SCRIPT_VERSION,
      'dry_run': True,
      'selected_variants_sha256': SELECTED_SHA256,
      'frozen_exons_sha256': EXONS_SHA256,
      'partition': 'development_only',
      'development_genes': DEVELOPMENT_GENES,
      'context_bp': CONTEXT_BP,
      'attention_backend': ATTENTION_BACKEND,
      'target_head_key': 'splice_sites_classification/logits',
      'target_scalar': 'TargetSummary.mean',
      'identity_calls_per_variant': 2,
      'component_calls_per_eligible_effect': len(components),
      'variant_count': len(cases),
      'effect_count': sum(case.is_effect for case in cases),
      'neutral_count': sum(not case.is_effect for case in cases),
      'component_count': len(components),
      'components': [dataclasses.asdict(component) for component in components],
      'variants': [{
          'order': case.order, 'gene': case.gene,
          'variant_id': case.variant_id, 'selection_class': case.selection_class,
      } for case in cases],
  }


def main() -> None:
  args = _parse_args()
  if args.max_variants < 0 or args.max_components < 0:
    raise ValueError('Limits must be nonnegative.')
  all_cases = load_development_cases()
  components = enumerate_components()
  cases = all_cases[:args.max_variants] if args.max_variants else all_cases
  components = (
      components[:args.max_components] if args.max_components else components
  )
  if args.dry_run:
    print(json.dumps(
        v2._to_json(  # pylint: disable=protected-access
            build_dry_run_plan(cases, components)
        ),
        indent=2, allow_nan=False,
    ))
    return

  checkpoint = v2._checkpoint_path(args.checkpoint)  # pylint: disable=protected-access
  frozen = frozen_configuration(checkpoint)
  model_instance = dna_model.create(
      checkpoint,
      model_settings=dna_model.ModelSettings(attention_backend=ATTENTION_BACKEND),
  )
  apply_fn = dna_model.create_splice_classification_logit_margin_route_census_apply(
      model_instance._metadata,  # pylint: disable=protected-access
      attention_backend=ATTENTION_BACKEND,
  )
  apply_fn = jax.jit(apply_fn)
  identities = {}
  live_inputs = {}
  for case in cases:
    identity, common, target, selection, resolved = _run_identity(
        model_instance, apply_fn, case, frozen, args.output_dir
    )
    identities[case.variant_id] = identity
    live_inputs[case.variant_id] = (common, target, selection, resolved)

  if not args.max_variants:
    eligible_by_gene = {
        gene: sum(
            case.gene == gene and case.is_effect and identities[case.variant_id][
                'direction_gate'
            ]['eligible_for_causal_census']
            for case in cases
        )
        for gene in DEVELOPMENT_GENES
    }
    if any(count < 3 for count in eligible_by_gene.values()):
      raise ValueError(
          'Full development Gate 0 requires at least three eligible effects '
          f'per gene; observed {eligible_by_gene}.'
      )

  completed_components = []
  for case in cases:
    identity = identities[case.variant_id]
    if not identity['direction_gate']['eligible_for_causal_census']:
      continue
    common, target, selection, resolved = live_inputs[case.variant_id]
    for component in components:
      completed_components.append(_run_component(
          apply_fn, case, identity, common, target, selection, component,
          frozen, resolved, args.output_dir,
      ))
  summary = {
      'status': 'complete', 'script_version': SCRIPT_VERSION,
      'partition': 'development_only', 'variant_count': len(cases),
      'eligible_effect_count': sum(
          identities[case.variant_id]['direction_gate'][
              'eligible_for_causal_census'
          ] for case in cases
      ),
      'completed_component_count': len(completed_components),
      'component_limit': len(components),
      'created_at_unix_s': time.time(),
  }
  v2._write_atomic(  # pylint: disable=protected-access
      args.output_dir / 'summary.json', summary
  )
  print((args.output_dir / 'summary.json').resolve())


if __name__ == '__main__':
  main()
