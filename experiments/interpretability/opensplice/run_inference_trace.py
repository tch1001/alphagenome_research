#!/usr/bin/env python3
"""Resume-safe AlphaGenome baseline and residual tracing for OpenSplice v2.

The primary score exactly follows OpenSplice's genome-mode processing: the mean
of the ALT-minus-REF probabilities at the strand-aware canonical acceptor and
donor.  Baselines use the normal public variant API and request only splice-site
classification.  The optional causal pilot uses an opt-in paired-target apply
function and exports only selected residuals and scalar target summaries.

The frozen v2 benchmark is SNV-only.  This runner deliberately fails closed on
indels because exact REF/ALT residual patching requires token-aligned sequences;
an indel-aware stitched tracing path is a separate experiment.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping, Sequence

from alphagenome.data import genome
from alphagenome.models import dna_model as public_dna_model
from alphagenome.models import dna_output
from alphagenome_research.io import genome as genome_io
from alphagenome_research.model import attention
from alphagenome_research.model import dna_model
from alphagenome_research.model import interpretability
import jax
import jax.numpy as jnp
import numpy as np


SCRIPT_VERSION = 'opensplice-inference-trace-v1.1.0'
DEFAULT_SELECTED = Path(__file__).with_name('selected_variants_v2.tsv')
DEFAULT_EXONS = Path(__file__).with_name('frozen_exons_v2.tsv')
DEFAULT_OUTPUT_DIR = Path(__file__).with_name('results').joinpath('v2')
DEVELOPMENT_CONTEXT_BP = 16_384
CONFIRMATION_CONTEXT_BP = 131_072
NUM_TRACE_SLOTS = 24
CONTROL_START_DISTANCE_TOKENS = 4
PAIR_PADDING_SIZE = 1
HEAD_PADDING_SIZE = 1
PREDICTED_EFFECT_THRESHOLD = 0.01
# Public prediction and instrumented tracing are separately compiled BF16
# graphs. Near probability 1.0, one BF16 ULP is 2**-8. This cross-graph guard is
# deliberately distinct from the protocol's 1e-6 same-executable repeat gate.
PUBLIC_PAIRED_TARGET_TOLERANCE = 2**-8

REQUIRED_SELECTED_COLUMNS = frozenset({
    'selection_version',
    'exon_order',
    'gene',
    'exon_id',
    'ensembl_exon_id',
    'selection_class',
    'observed_effect_sign',
    'chromosome',
    'position_1based',
    'reference_bases',
    'alternate_bases',
    'variant_id',
    'region',
    'mut_type',
    'delta_psi',
    'delta_logit',
    'significant',
    'measured',
})
REQUIRED_EXON_COLUMNS = frozenset({
    'selection_order',
    'gene',
    'exon_id',
    'ensembl_exon_id',
    'chromosome',
    'strand',
    'exon_start_1based',
    'exon_end_1based',
})


@dataclasses.dataclass(frozen=True)
class Exon:
  order: int
  gene: str
  exon_id: str
  ensembl_exon_id: str
  chromosome: str
  strand: str
  start_1based: int
  end_1based: int


@dataclasses.dataclass(frozen=True)
class Case:
  order: int
  selection_version: str
  selection_class: str
  observed_effect_sign: str
  gene: str
  exon_id: str
  ensembl_exon_id: str
  chromosome: str
  strand: str
  exon_start_1based: int
  exon_end_1based: int
  variant_id: str
  position_1based: int
  reference_bases: str
  alternate_bases: str
  region: str
  mut_type: str
  delta_psi: float
  delta_logit: float

  @property
  def is_effect(self) -> bool:
    return 'neutral' not in self.selection_class.lower()

  @property
  def variant(self) -> genome.Variant:
    return genome.Variant(
        chromosome=self.chromosome,
        position=self.position_1based,
        reference_bases=self.reference_bases,
        alternate_bases=self.alternate_bases,
        name=self.variant_id,
    )


@dataclasses.dataclass(frozen=True)
class TracePositionSet:
  name: str
  tokens: tuple[int, ...]
  slots: tuple[int, ...]
  role: str
  matched_candidate: str | None
  genomic_intervals: tuple[tuple[int, int], ...]


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--selected', type=Path, default=DEFAULT_SELECTED)
  parser.add_argument('--frozen-exons', type=Path, default=DEFAULT_EXONS)
  parser.add_argument('--checkpoint', type=Path)
  parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument(
      '--max-variants',
      type=int,
      default=0,
      help='Maximum frozen variants to run; 0 means all 50.',
  )
  parser.add_argument('--confirmation-131kb', action='store_true')
  parser.add_argument(
      '--trace-max-variants',
      type=int,
      default=0,
      help='Maximum direction-passing effect variants to trace; 0 disables.',
  )
  parser.add_argument(
      '--trace-max-groups-per-variant',
      type=int,
      default=12,
      help='Maximum stage/layer/region groups; each group runs four patches.',
  )
  parser.add_argument(
      '--trace-layers',
      default='0',
      help='Comma-separated transformer layers for the bounded pilot.',
  )
  parser.add_argument(
      '--trace-stages',
      default='pre_attention,post_attention,post_mlp',
      help='Comma-separated residual seams for the bounded pilot.',
  )
  parser.add_argument(
      '--trace-context-bp', type=int, default=DEVELOPMENT_CONTEXT_BP
  )
  parser.add_argument(
      '--prediction-effect-threshold',
      type=float,
      default=PREDICTED_EFFECT_THRESHOLD,
      help='Minimum absolute predicted mean delta required by the output gate.',
  )
  parser.add_argument(
      '--attention-backend',
      choices=sorted(attention.ATTENTION_BACKENDS),
      default=attention.ATTENTION_BACKEND_DENSE,
      help=(
          'Attention implementation. Dense is the causal-reference default; '
          'pallas_tiled is an optional numerical replication.'
      ),
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
    raise ValueError(f'{path}: missing required columns: {", ".join(missing)}')


def _normalise_chromosome(value: str) -> str:
  value = value.strip()
  return value if value.startswith('chr') else f'chr{value}'


def _finite_float(value: str, field: str, variant_id: str) -> float:
  try:
    result = float(value)
  except (TypeError, ValueError) as error:
    raise ValueError(f'{variant_id}: invalid {field}={value!r}.') from error
  if not math.isfinite(result):
    raise ValueError(f'{variant_id}: non-finite {field}={value!r}.')
  return result


def _is_true(value: str) -> bool:
  return value.strip().lower() in {'1', 'true', 'yes'}


def _read_exons(path: Path) -> dict[str, Exon]:
  with path.open('r', encoding='utf-8', newline='') as handle:
    reader = csv.DictReader(handle, delimiter='\t')
    _require_columns(reader.fieldnames, REQUIRED_EXON_COLUMNS, path)
    result = {}
    for row in reader:
      exon = Exon(
          order=int(row['selection_order']),
          gene=row['gene'],
          exon_id=row['exon_id'],
          ensembl_exon_id=row['ensembl_exon_id'],
          chromosome=_normalise_chromosome(row['chromosome']),
          strand=row['strand'],
          start_1based=int(row['exon_start_1based']),
          end_1based=int(row['exon_end_1based']),
      )
      if exon.strand not in {'+', '-'}:
        raise ValueError(f'{exon.exon_id}: invalid strand {exon.strand!r}.')
      if exon.start_1based > exon.end_1based:
        raise ValueError(f'{exon.exon_id}: reversed exon coordinates.')
      if exon.ensembl_exon_id in result:
        raise ValueError(f'Duplicate exon {exon.ensembl_exon_id}.')
      result[exon.ensembl_exon_id] = exon
  if not result:
    raise ValueError(f'{path}: no exons.')
  return result


def load_cases(selected_path: Path, exons_path: Path) -> tuple[Case, ...]:
  """Loads the frozen SNV selection and joins only tracked exon coordinates."""
  exons = _read_exons(exons_path)
  cases = []
  seen = set()
  with selected_path.open('r', encoding='utf-8', newline='') as handle:
    reader = csv.DictReader(handle, delimiter='\t')
    _require_columns(
        reader.fieldnames, REQUIRED_SELECTED_COLUMNS, selected_path
    )
    for index, row in enumerate(reader):
      variant_id = row['variant_id']
      if variant_id in seen:
        raise ValueError(f'Duplicate variant_id {variant_id}.')
      seen.add(variant_id)
      try:
        exon = exons[row['ensembl_exon_id']]
      except KeyError as error:
        raise ValueError(
            f'{variant_id}: exon {row["ensembl_exon_id"]} is not frozen.'
        ) from error
      chromosome = _normalise_chromosome(row['chromosome'])
      for field, observed, expected in (
          ('exon_order', int(row['exon_order']), exon.order),
          ('gene', row['gene'], exon.gene),
          ('exon_id', row['exon_id'], exon.exon_id),
          ('chromosome', chromosome, exon.chromosome),
      ):
        if observed != expected:
          raise ValueError(
              f'{variant_id}: {field} mismatch {observed!r} != {expected!r}.'
          )
      ref = row['reference_bases'].upper()
      alt = row['alternate_bases'].upper()
      if row['mut_type'].lower() != 'sub' or len(ref) != 1 or len(alt) != 1:
        raise ValueError(
            f'{variant_id}: v2 causal benchmark is SNV-only; observed '
            f'{ref}>{alt} '
            f'with mut_type={row["mut_type"]!r}.'
        )
      if ref == alt or ref not in 'ACGT' or alt not in 'ACGT':
        raise ValueError(f'{variant_id}: invalid exact SNV allele {ref}>{alt}.')
      if not _is_true(row['measured']):
        raise ValueError(f'{variant_id}: frozen variant is not measured.')
      observed_sign = row['observed_effect_sign'].strip().lower()
      delta_psi = _finite_float(row['delta_psi'], 'delta_psi', variant_id)
      delta_logit = _finite_float(
          row['delta_logit'], 'delta_logit', variant_id
      )
      is_effect = 'neutral' not in row['selection_class'].lower()
      expected_sign = (
          'positive'
          if delta_logit > 0
          else 'negative'
          if delta_logit < 0
          else 'zero'
      )
      sign_matches = (
          observed_sign == expected_sign
          if is_effect
          else observed_sign in {'neutral', 'neutral_control'}
      )
      if not sign_matches:
        raise ValueError(
            f'{variant_id}: observed_effect_sign={observed_sign!r} does not '
            f'match selection_class={row["selection_class"]!r} and '
            f'delta_logit={delta_logit}.'
        )
      cases.append(
          Case(
              order=index,
              selection_version=row['selection_version'],
              selection_class=row['selection_class'],
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
          )
      )
  if not cases:
    raise ValueError(f'{selected_path}: no selected variants.')
  versions = {case.selection_version for case in cases}
  if len(versions) != 1 or not next(iter(versions)).startswith(
      'opensplice-circuit-v2'
  ):
    raise ValueError(
        f'Expected one OpenSplice v2 selection, observed {versions}.'
    )
  return tuple(cases)


def centered_interval(case: Case, context_bp: int) -> genome.Interval:
  if context_bp <= 0 or context_bp % 2048:
    raise ValueError('Context length must be a positive multiple of 2,048 bp.')
  center_1based = (case.exon_start_1based + case.exon_end_1based) // 2
  start = center_1based - 1 - context_bp // 2
  interval = genome.Interval(case.chromosome, start, start + context_bp)
  required = (
      case.position_1based,
      case.exon_start_1based,
      case.exon_end_1based,
  )
  if not all(
      interval.start <= position - 1 < interval.end for position in required
  ):
    raise ValueError(
        f'{case.variant_id}: target positions do not fit {interval}.'
    )
  return interval


def canonical_sites(case: Case) -> tuple[tuple[str, int, int], ...]:
  """Returns (role, 1-based position, internal channel) pairs."""
  if case.strand == '+':
    return (
        ('acceptor', case.exon_start_1based, 1),
        ('donor', case.exon_end_1based, 0),
    )
  return (
      ('acceptor', case.exon_end_1based, 3),
      ('donor', case.exon_start_1based, 2),
  )


def paired_target_selection(
    case: Case, interval: genome.Interval
) -> interpretability.PairedTargetSelection:
  sites = canonical_sites(case)
  return interpretability.PairedTargetSelection(
      position_indices=jnp.array(
          [position - 1 - interval.start for _, position, _ in sites],
          jnp.int32,
      ),
      track_indices=jnp.array([channel for _, _, channel in sites], jnp.int32),
      valid_mask=jnp.ones((len(sites),), jnp.bool),
  )


def _metadata_channel_indices(metadata) -> dict[tuple[str, str], int]:
  channels = {}
  for column, (_, row) in enumerate(metadata.reset_index(drop=True).iterrows()):
    name = str(row['name']).strip().lower()
    strand = str(row['strand']).strip()
    if name == 'padding':
      continue
    key = (name, strand)
    if key in channels:
      raise ValueError(f'Duplicate splice-site metadata channel {key}.')
    channels[key] = column
  required = {
      ('donor', '+'),
      ('acceptor', '+'),
      ('donor', '-'),
      ('acceptor', '-'),
  }
  if set(channels) != required:
    raise ValueError(f'Unexpected splice-site metadata channels: {channels}.')
  return channels


def score_splice_site_tracks(
    case: Case, interval: genome.Interval, output
) -> dict[str, Any]:
  """Extracts canonical REF/ALT probabilities from a public VariantOutput."""
  reference = output.reference.splice_sites
  alternate = output.alternate.splice_sites
  if reference is None or alternate is None:
    raise ValueError('AlphaGenome did not return splice-site predictions.')
  if not reference.metadata.equals(alternate.metadata):
    raise ValueError('REF and ALT splice-site metadata differ.')
  channels = _metadata_channel_indices(reference.metadata)
  ref_values = np.asarray(reference.values)
  alt_values = np.asarray(alternate.values)
  if ref_values.shape != alt_values.shape or ref_values.ndim != 2:
    raise ValueError('Unexpected REF/ALT splice-site prediction shapes.')
  scores = {}
  for role, position, _ in canonical_sites(case):
    index = position - 1 - interval.start
    channel = channels[(role, case.strand)]
    if not 0 <= index < ref_values.shape[0]:
      raise ValueError(f'{role} position is outside returned predictions.')
    scores[f'{role}_position_1based'] = position
    scores[f'{role}_channel'] = channel
    scores[f'{role}_reference'] = float(ref_values[index, channel])
    scores[f'{role}_alternate'] = float(alt_values[index, channel])
    scores[f'delta_{role}'] = (
        scores[f'{role}_alternate'] - scores[f'{role}_reference']
    )
  scores['reference_mean'] = np.mean(
      [scores['acceptor_reference'], scores['donor_reference']],
      dtype=np.float64,
  ).item()
  scores['alternate_mean'] = np.mean(
      [scores['acceptor_alternate'], scores['donor_alternate']],
      dtype=np.float64,
  ).item()
  scores['mean_delta_splice'] = np.mean(
      [scores['delta_acceptor'], scores['delta_donor']], dtype=np.float64
  ).item()
  return scores


def direction_result(
    predicted_delta: float,
    experimental_delta_logit: float,
    *,
    is_effect: bool,
    predicted_effect_threshold: float,
) -> dict[str, Any]:
  predicted_sign = (
      'positive'
      if predicted_delta >= predicted_effect_threshold
      else 'negative'
      if predicted_delta <= -predicted_effect_threshold
      else 'below_threshold'
  )
  experimental_sign = (
      'positive'
      if experimental_delta_logit > 0
      else 'negative'
      if experimental_delta_logit < 0
      else 'zero'
  )
  correct = (
      predicted_sign == experimental_sign
      if is_effect and predicted_sign != 'below_threshold'
      else None
  )
  return {
      'predicted_sign': predicted_sign,
      'experimental_sign': experimental_sign,
      'predicted_effect_threshold': predicted_effect_threshold,
      'direction_correct': correct,
      'gated_for_tracing': correct is True,
  }


def trace_position_sets(
    case: Case, interval: genome.Interval
) -> tuple[TracePositionSet, ...]:
  """Builds protocol V/A/D/S sets and same-cardinality shifted controls."""
  token_count = interval.width // 128

  def token(position_1based: int) -> int:
    return (position_1based - 1 - interval.start) // 128

  variant_tokens = (token(case.position_1based),)
  acceptor_tokens = (token(canonical_sites(case)[0][1]),)
  donor_tokens = (token(canonical_sites(case)[1][1]),)
  splice_tokens = tuple(
      dict.fromkeys(variant_tokens + acceptor_tokens + donor_tokens)
  )
  candidates = (
      ('V', variant_tokens),
      ('A', acceptor_tokens),
      ('D', donor_tokens),
      ('S', splice_tokens),
  )
  occupied_candidates = set(splice_tokens)

  def shifted_control(
      tokens: tuple[int, ...], direction: int
  ) -> tuple[int, ...]:
    distance = CONTROL_START_DISTANCE_TOKENS
    while distance < token_count:
      shifted = tuple(value + direction * distance for value in tokens)
      if (
          min(shifted) >= 0
          and max(shifted) < token_count
          and occupied_candidates.isdisjoint(shifted)
      ):
        return shifted
      distance += 1
    raise ValueError(
        'No nonoverlapping in-bounds matched control is available.'
    )

  specifications = [
      (name, tokens, 'candidate', None) for name, tokens in candidates
  ]
  for name, tokens in candidates:
    specifications.append(
        (
            f'{name}_control_upstream',
            shifted_control(tokens, -1),
            'width_matched_control',
            name,
        )
    )
    specifications.append(
        (
            f'{name}_control_downstream',
            shifted_control(tokens, 1),
            'width_matched_control',
            name,
        )
    )

  unique_tokens = tuple(
      dict.fromkeys(
          token_index
          for _, tokens, _, _ in specifications
          for token_index in tokens
      )
  )
  if len(unique_tokens) > NUM_TRACE_SLOTS:
    raise ValueError('Trace selection exceeds its fixed padded slot count.')
  slot_by_token = {
      token_index: slot for slot, token_index in enumerate(unique_tokens)
  }
  position_sets = []
  for name, tokens, role, matched_candidate in specifications:
    position_sets.append(
        TracePositionSet(
            name=name,
            tokens=tokens,
            slots=tuple(slot_by_token[token_index] for token_index in tokens),
            role=role,
            matched_candidate=matched_candidate,
            genomic_intervals=tuple(
                (
                    interval.start + 128 * token_index,
                    interval.start + 128 * (token_index + 1),
                )
                for token_index in tokens
            ),
        )
    )
  return tuple(position_sets)


def transformer_trace_selection(
    position_sets: Sequence[TracePositionSet],
) -> interpretability.TransformerTraceSelection:
  positions = np.zeros((NUM_TRACE_SLOTS,), dtype=np.int32)
  valid = np.zeros((NUM_TRACE_SLOTS,), dtype=bool)
  for position_set in position_sets:
    for slot, token_index in zip(
        position_set.slots, position_set.tokens, strict=True
    ):
      if valid[slot] and positions[slot] != token_index:
        raise ValueError('Trace slot maps to more than one token.')
      positions[slot] = token_index
      valid[slot] = True
  return interpretability.TransformerTraceSelection(
      pair_bias_edges=interpretability.PairBiasEdgeSelection(
          query_bins=jnp.zeros((PAIR_PADDING_SIZE,), jnp.int32),
          key_bins=jnp.zeros((PAIR_PADDING_SIZE,), jnp.int32),
          valid_mask=jnp.zeros((PAIR_PADDING_SIZE,), jnp.bool),
      ),
      head_output_positions=interpretability.HeadOutputSelection(
          positions=jnp.zeros((HEAD_PADDING_SIZE,), jnp.int32),
          valid_mask=jnp.zeros((HEAD_PADDING_SIZE,), jnp.bool),
      ),
      residual_positions=interpretability.SequenceResidualSelection(
          positions=jnp.asarray(positions), valid_mask=jnp.asarray(valid)
      ),
  )


def _residual_patch(
    identity: interpretability.TransformerInterventions,
    donor_trace: interpretability.TransformerTrace,
    *,
    stage: str,
    layer: int,
    slots: Sequence[int],
) -> interpretability.TransformerInterventions:
  values = getattr(donor_trace, f'{stage}_residuals')
  replacement = interpretability.SequenceResidualReplacement(
      values=values,
      replace_mask=jnp.zeros(values.shape[:-1], jnp.bool).at[
          layer, :, jnp.array(slots, jnp.int32)
      ].set(True),
  )
  kwargs = {
      'head_masks': identity.head_masks,
      'pair_bias_values': identity.pair_bias_values,
      'pair_bias_replace_mask': identity.pair_bias_replace_mask,
      'head_value_output_replacement': identity.head_value_output_replacement,
      'pre_attention_residual': identity.pre_attention_residual,
      'post_attention_residual': identity.post_attention_residual,
      'post_mlp_residual': identity.post_mlp_residual,
  }
  kwargs[f'{stage}_residual'] = replacement
  return interpretability.TransformerInterventions(**kwargs)


def _checkpoint_path(explicit: Path | None) -> Path:
  if explicit is not None:
    result = explicit.expanduser().resolve()
    if not result.exists():
      raise FileNotFoundError(result)
    return result
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


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _slug(value: str) -> str:
  return re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('._')


def _fingerprint(configuration: Mapping[str, Any]) -> str:
  encoded = json.dumps(
      configuration, sort_keys=True, separators=(',', ':'), allow_nan=False
  ).encode('utf-8')
  return hashlib.sha256(encoded).hexdigest()


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
  if isinstance(value, tuple):
    return [_to_json(item) for item in value]
  return value


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(
      json.dumps(_to_json(value), indent=2, allow_nan=False) + '\n',
      encoding='utf-8',
  )
  temporary.replace(path)


def _load_completed(path: Path, fingerprint: str) -> dict[str, Any] | None:
  if not path.exists():
    return None
  try:
    value = json.loads(path.read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError) as error:
    raise ValueError(
        f'Incomplete or corrupt resume artifact: {path}.'
    ) from error
  if value.get('status') != 'complete':
    raise ValueError(f'Resume artifact is not complete: {path}.')
  if value.get('fingerprint') != fingerprint:
    raise ValueError(
        f'Resume artifact configuration mismatch: {path}; use a new output dir.'
    )
  return value


def _case_record(case: Case) -> dict[str, Any]:
  return dataclasses.asdict(case)


def baseline_configuration(
    case: Case,
    context_bp: int,
    *,
    selected_sha256: str,
    exons_sha256: str,
    checkpoint: Path,
    predicted_effect_threshold: float,
    attention_backend: str,
) -> dict[str, Any]:
  return {
      'script_version': SCRIPT_VERSION,
      'kind': 'baseline',
      'case': _case_record(case),
      'context_bp': context_bp,
      'selected_sha256': selected_sha256,
      'frozen_exons_sha256': exons_sha256,
      'checkpoint': str(checkpoint),
      'attention_backend': attention_backend,
      'score_protocol': 'mean_canonical_acceptor_and_donor_probability',
      'direction_gate_protocol': 'sign_delta_logit_and_min_abs_prediction',
      'predicted_effect_threshold': predicted_effect_threshold,
      'coordinate_alignment': 'normal_local_variant_api_ref_aligned',
  }


def _baseline_path(output_dir: Path, context_bp: int, case: Case) -> Path:
  return output_dir.joinpath(
      'baseline',
      f'{context_bp}bp',
      f'{case.order:03d}_{_slug(case.variant_id)}.json',
  )


def _run_baseline(
    model_instance: dna_model.AlphaGenomeModel,
    case: Case,
    context_bp: int,
    configuration: Mapping[str, Any],
    output_path: Path,
    *,
    predicted_effect_threshold: float,
) -> dict[str, Any]:
  fingerprint = _fingerprint(configuration)
  if completed := _load_completed(output_path, fingerprint):
    return completed
  interval = centered_interval(case, context_bp)
  start = time.perf_counter()
  output = model_instance.predict_variant(
      interval,
      case.variant,
      organism=public_dna_model.Organism.HOMO_SAPIENS,
      requested_outputs=[dna_output.OutputType.SPLICE_SITES],
      ontology_terms=None,
  )
  seconds = time.perf_counter() - start
  score = score_splice_site_tracks(case, interval, output)
  direction = direction_result(
      score['mean_delta_splice'],
      case.delta_logit,
      is_effect=case.is_effect,
      predicted_effect_threshold=predicted_effect_threshold,
  )
  result = {
      'status': 'complete',
      'fingerprint': fingerprint,
      'configuration': configuration,
      'interval': {
          'chromosome': interval.chromosome,
          'start_0based': interval.start,
          'end_0based_exclusive': interval.end,
      },
      'canonical_sites': [
          {
              'role': role,
              'position_1based': position,
              'internal_channel': channel,
          }
          for role, position, channel in canonical_sites(case)
      ],
      'prediction': score,
      'experimental': {
          'delta_psi': case.delta_psi,
          'delta_logit': case.delta_logit,
          'selection_class': case.selection_class,
          'observed_effect_sign': case.observed_effect_sign,
      },
      'direction_gate': direction,
      'seconds': seconds,
      'created_at_unix_s': time.time(),
  }
  _write_atomic(output_path, result)
  return result


def _timed_apply(apply, *args):
  start = time.perf_counter()
  output = apply(*args)
  jax.block_until_ready(output)
  return output, time.perf_counter() - start


def _trace_group_path(
    output_dir: Path,
    context_bp: int,
    case: Case,
    *,
    stage: str,
    layer: int,
    position_set: TracePositionSet,
) -> Path:
  return output_dir.joinpath(
      'trace',
      f'{context_bp}bp',
      f'{case.order:03d}_{_slug(case.variant_id)}',
      f'{stage}_layer{layer:02d}_{_slug(position_set.name)}.json',
  )


def _recovery(
    patched: float, self_recipient: float, donor: float, recipient: float
) -> float | None:
  denominator = donor - recipient
  if abs(denominator) <= 1e-8:
    return None
  return (patched - self_recipient) / denominator


def validate_public_paired_target(
    public_score: Mapping[str, Any],
    reference_value: float,
    alternate_value: float,
    *,
    tolerance: float = PUBLIC_PAIRED_TARGET_TOLERANCE,
) -> dict[str, float]:
  """Checks public and instrumented targets within one BF16 output ULP."""
  deltas = {
      'reference_delta_from_public': (
          reference_value - float(public_score['reference_mean'])
      ),
      'alternate_delta_from_public': (
          alternate_value - float(public_score['alternate_mean'])
      ),
  }
  if any(abs(value) > tolerance for value in deltas.values()):
    raise ValueError(
        'Paired interpretability target disagrees with the public splice-site '
        f'target beyond tolerance {tolerance}: {deltas}.'
    )
  return deltas


def _run_trace_variant(
    model_instance: dna_model.AlphaGenomeModel,
    paired_apply,
    case: Case,
    baseline: Mapping[str, Any],
    *,
    context_bp: int,
    output_dir: Path,
    base_configuration: Mapping[str, Any],
    stages: Sequence[str],
    layers: Sequence[int],
  max_groups: int,
) -> list[dict[str, Any]]:
  interval = centered_interval(case, context_bp)
  position_sets = trace_position_sets(case, interval)
  trace_selection = transformer_trace_selection(position_sets)
  target_selection = paired_target_selection(case, interval)
  identity = interpretability.no_transformer_interventions(
      batch_size=1, num_edges=PAIR_PADDING_SIZE, dtype=jnp.float32
  )
  extractor = model_instance._get_fasta_extractor(  # pylint: disable=protected-access
      public_dna_model.Organism.HOMO_SAPIENS
  )
  raw_sequence = extractor.extract(interval)
  variant_offset = case.position_1based - 1 - interval.start
  observed_ref = raw_sequence[
      variant_offset : variant_offset + len(case.reference_bases)
  ].upper()
  if observed_ref != case.reference_bases:
    raise ValueError(
        f'{case.variant_id}: FASTA REF mismatch {observed_ref!r} != '
        f'{case.reference_bases!r}.'
    )
  reference_sequence, alternate_sequence = genome_io.extract_variant_sequences(
      interval, case.variant, extractor
  )
  if not (
      len(reference_sequence) == len(alternate_sequence) == context_bp
  ):
    raise ValueError(
        f'{case.variant_id}: tracing requires aligned SNV sequences.'
    )
  encoder = model_instance._one_hot_encoder  # pylint: disable=protected-access
  reference = jnp.asarray(encoder.encode(reference_sequence))[None]
  alternate = jnp.asarray(encoder.encode(alternate_sequence))[None]
  organism_index = jnp.array([0], jnp.int32)
  common = (
      model_instance._params,  # pylint: disable=protected-access
      model_instance._state,  # pylint: disable=protected-access
      organism_index,
      trace_selection,
  )
  (reference_target, reference_trace), reference_seconds = _timed_apply(
      paired_apply,
      common[0],
      common[1],
      reference,
      common[2],
      common[3],
      identity,
      target_selection,
  )
  (alternate_target, alternate_trace), alternate_seconds = _timed_apply(
      paired_apply,
      common[0],
      common[1],
      alternate,
      common[2],
      common[3],
      identity,
      target_selection,
  )
  reference_value = float(reference_target.mean[0])
  alternate_value = float(alternate_target.mean[0])
  public_score = baseline['prediction']
  public_deltas = validate_public_paired_target(
      public_score, reference_value, alternate_value
  )
  direct_baseline = {
      'reference_mean': reference_value,
      'alternate_mean': alternate_value,
      'mean_delta_splice': alternate_value - reference_value,
      **public_deltas,
      'public_equivalence_tolerance': PUBLIC_PAIRED_TARGET_TOLERANCE,
      'reference_seconds_compile_and_run': reference_seconds,
      'alternate_seconds_warm': alternate_seconds,
  }

  planned = [
      (stage, layer, position_set)
      for stage in stages
      for layer in layers
      for position_set in position_sets
  ][:max_groups]
  outputs = []
  for stage, layer, position_set in planned:
    group_configuration = {
        **base_configuration,
        'kind': 'residual_trace_group',
        'context_bp': context_bp,
        'case': _case_record(case),
        'baseline_fingerprint': baseline['fingerprint'],
        'stage': stage,
        'layer': layer,
        'position_set': dataclasses.asdict(position_set),
        'direct_sequence_sha256': {
            'reference': hashlib.sha256(
                reference_sequence.encode('ascii')
            ).hexdigest(),
            'alternate': hashlib.sha256(
                alternate_sequence.encode('ascii')
            ).hexdigest(),
        },
    }
    fingerprint = _fingerprint(group_configuration)
    path = _trace_group_path(
        output_dir,
        context_bp,
        case,
        stage=stage,
        layer=layer,
        position_set=position_set,
    )
    if completed := _load_completed(path, fingerprint):
      outputs.append(completed)
      continue

    ref_into_alt = _residual_patch(
        identity,
        reference_trace,
        stage=stage,
        layer=layer,
        slots=position_set.slots,
    )
    (ref_alt_target, _), ref_alt_seconds = _timed_apply(
        paired_apply,
        common[0],
        common[1],
        alternate,
        common[2],
        common[3],
        ref_into_alt,
        target_selection,
    )
    alt_into_alt = _residual_patch(
        identity,
        alternate_trace,
        stage=stage,
        layer=layer,
        slots=position_set.slots,
    )
    (alt_alt_target, _), alt_alt_seconds = _timed_apply(
        paired_apply,
        common[0],
        common[1],
        alternate,
        common[2],
        common[3],
        alt_into_alt,
        target_selection,
    )
    alt_into_ref = _residual_patch(
        identity,
        alternate_trace,
        stage=stage,
        layer=layer,
        slots=position_set.slots,
    )
    (alt_ref_target, _), alt_ref_seconds = _timed_apply(
        paired_apply,
        common[0],
        common[1],
        reference,
        common[2],
        common[3],
        alt_into_ref,
        target_selection,
    )
    ref_into_ref = _residual_patch(
        identity,
        reference_trace,
        stage=stage,
        layer=layer,
        slots=position_set.slots,
    )
    (ref_ref_target, _), ref_ref_seconds = _timed_apply(
        paired_apply,
        common[0],
        common[1],
        reference,
        common[2],
        common[3],
        ref_into_ref,
        target_selection,
    )
    ref_alt_value = float(ref_alt_target.mean[0])
    alt_alt_value = float(alt_alt_target.mean[0])
    alt_ref_value = float(alt_ref_target.mean[0])
    ref_ref_value = float(ref_ref_target.mean[0])
    result = {
        'status': 'complete',
        'fingerprint': fingerprint,
        'configuration': group_configuration,
        'direct_baseline': direct_baseline,
        'patches': {
            'reference_into_alternate': {
                'donor_allele': 'reference',
                'recipient_allele': 'alternate',
                'target_mean': ref_alt_value,
                'seconds': ref_alt_seconds,
                'self_control_corrected_recovery': _recovery(
                    ref_alt_value,
                    alt_alt_value,
                    reference_value,
                    alternate_value,
                ),
            },
            'alternate_into_alternate_self_control': {
                'donor_allele': 'alternate',
                'recipient_allele': 'alternate',
                'target_mean': alt_alt_value,
                'delta_from_baseline': alt_alt_value - alternate_value,
                'seconds': alt_alt_seconds,
            },
            'alternate_into_reference': {
                'donor_allele': 'alternate',
                'recipient_allele': 'reference',
                'target_mean': alt_ref_value,
                'seconds': alt_ref_seconds,
                'self_control_corrected_recovery': _recovery(
                    alt_ref_value,
                    ref_ref_value,
                    alternate_value,
                    reference_value,
                ),
            },
            'reference_into_reference_self_control': {
                'donor_allele': 'reference',
                'recipient_allele': 'reference',
                'target_mean': ref_ref_value,
                'delta_from_baseline': ref_ref_value - reference_value,
                'seconds': ref_ref_seconds,
            },
        },
        'created_at_unix_s': time.time(),
    }
    _write_atomic(path, result)
    outputs.append(result)
  return outputs


def _parse_layers(value: str) -> tuple[int, ...]:
  try:
    layers = tuple(
        int(item.strip()) for item in value.split(',') if item.strip()
    )
  except ValueError as error:
    raise ValueError(
        '--trace-layers must contain comma-separated integers.'
    ) from error
  if not layers or any(
      layer < 0 or layer >= interpretability.NUM_TRANSFORMER_LAYERS
      for layer in layers
  ):
    raise ValueError('--trace-layers must be nonempty and within [0, 8].')
  return layers


def _parse_stages(value: str) -> tuple[str, ...]:
  stages = tuple(item.strip() for item in value.split(',') if item.strip())
  allowed = {'pre_attention', 'post_attention', 'post_mlp'}
  if not stages or any(stage not in allowed for stage in stages):
    raise ValueError(f'--trace-stages must contain only {sorted(allowed)}.')
  return stages


def build_dry_run_plan(
    cases: Sequence[Case],
    *,
    contexts: Sequence[int],
    trace_max_variants: int,
    trace_context_bp: int,
    trace_layers: Sequence[int],
    trace_stages: Sequence[str],
    trace_max_groups_per_variant: int,
    predicted_effect_threshold: float,
    attention_backend: str,
    selected_sha256: str,
    exons_sha256: str,
) -> dict[str, Any]:
  return {
      'script_version': SCRIPT_VERSION,
      'dry_run': True,
      'selected_sha256': selected_sha256,
      'frozen_exons_sha256': exons_sha256,
      'attention_backend': attention_backend,
      'variant_count': len(cases),
      'effect_count': sum(case.is_effect for case in cases),
      'neutral_count': sum(not case.is_effect for case in cases),
      'baseline_contexts_bp': list(contexts),
      'baseline_task_count': len(cases) * len(contexts),
      'trace': {
          'enabled': trace_max_variants > 0,
          'max_variants_after_direction_gate': trace_max_variants,
          'context_bp': trace_context_bp,
          'layers': list(trace_layers),
          'stages': list(trace_stages),
          'max_groups_per_variant': trace_max_groups_per_variant,
          'calls_per_group': 4,
          'indel_policy': 'fail_closed_snv_only',
          'direction_gate': {
              'experimental_sign': 'delta_logit',
              'minimum_absolute_predicted_mean_delta': (
                  predicted_effect_threshold
              ),
          },
          'candidate_sets': ['V', 'A', 'D', 'S=unique(V_union_A_union_D)'],
          'control_start_distance_tokens': CONTROL_START_DISTANCE_TOKENS,
      },
      'variants': [
          {
              'order': case.order,
              'variant_id': case.variant_id,
              'selection_class': case.selection_class,
              'observed_effect_sign': case.observed_effect_sign,
              'delta_psi': case.delta_psi,
              'delta_logit': case.delta_logit,
              'intervals': {
                  str(context): str(centered_interval(case, context))
                  for context in contexts
              },
              'canonical_sites': [
                  {
                      'role': role,
                      'position_1based': position,
                      'internal_channel': channel,
                  }
                  for role, position, channel in canonical_sites(case)
              ],
              'trace_position_sets': [
                  dataclasses.asdict(position_set)
                  for position_set in trace_position_sets(
                      case, centered_interval(case, trace_context_bp)
                  )
              ],
          }
          for case in cases
      ],
  }


def main() -> None:
  args = _parse_args()
  if args.max_variants < 0 or args.trace_max_variants < 0:
    raise ValueError('Variant limits must be nonnegative.')
  if args.trace_max_groups_per_variant <= 0:
    raise ValueError('--trace-max-groups-per-variant must be positive.')
  if args.prediction_effect_threshold <= 0:
    raise ValueError('--prediction-effect-threshold must be positive.')
  layers = _parse_layers(args.trace_layers)
  stages = _parse_stages(args.trace_stages)
  selected = args.selected.resolve()
  frozen_exons = args.frozen_exons.resolve()
  cases = load_cases(selected, frozen_exons)
  if args.max_variants:
    cases = cases[: args.max_variants]
  contexts = [DEVELOPMENT_CONTEXT_BP]
  if args.confirmation_131kb:
    contexts.append(CONFIRMATION_CONTEXT_BP)
  if args.trace_max_variants and args.trace_context_bp not in contexts:
    contexts.append(args.trace_context_bp)
  selected_sha256 = _sha256(selected)
  exons_sha256 = _sha256(frozen_exons)
  plan = build_dry_run_plan(
      cases,
      contexts=contexts,
      trace_max_variants=args.trace_max_variants,
      trace_context_bp=args.trace_context_bp,
      trace_layers=layers,
      trace_stages=stages,
      trace_max_groups_per_variant=args.trace_max_groups_per_variant,
      predicted_effect_threshold=args.prediction_effect_threshold,
      attention_backend=args.attention_backend,
      selected_sha256=selected_sha256,
      exons_sha256=exons_sha256,
  )
  if args.dry_run:
    print(json.dumps(_to_json(plan), indent=2, allow_nan=False))
    return

  checkpoint = _checkpoint_path(args.checkpoint)
  model_instance = dna_model.create(
      checkpoint,
      model_settings=dna_model.ModelSettings(
          attention_backend=args.attention_backend
      ),
  )
  baselines = {}
  for context_bp in contexts:
    for case in cases:
      configuration = baseline_configuration(
          case,
          context_bp,
          selected_sha256=selected_sha256,
          exons_sha256=exons_sha256,
          checkpoint=checkpoint,
          predicted_effect_threshold=args.prediction_effect_threshold,
          attention_backend=args.attention_backend,
      )
      path = _baseline_path(args.output_dir, context_bp, case)
      baselines[(case.variant_id, context_bp)] = _run_baseline(
          model_instance,
          case,
          context_bp,
          configuration,
          path,
          predicted_effect_threshold=args.prediction_effect_threshold,
      )

  trace_outputs = []
  if args.trace_max_variants:
    paired_apply = dna_model.create_paired_targeted_interpretability_apply(
        model_instance._metadata,  # pylint: disable=protected-access
        interpretability.TargetSpec(
            head_name='splice_sites_classification',
            prediction_key='predictions',
        ),
        attention_backend=args.attention_backend,
    )
    paired_apply = jax.jit(paired_apply)
    passing = [
        case
        for case in cases
        if case.is_effect
        and baselines[(case.variant_id, args.trace_context_bp)][
            'direction_gate'
        ]['gated_for_tracing']
    ][: args.trace_max_variants]
    trace_base_configuration = {
        'script_version': SCRIPT_VERSION,
        'selected_sha256': selected_sha256,
        'frozen_exons_sha256': exons_sha256,
        'checkpoint': str(checkpoint),
        'attention_backend': args.attention_backend,
        'score_protocol': 'paired_canonical_acceptor_and_donor_probability',
        'control_protocol': (
            'same_cardinality_relative_offsets_starting_'
            f'{CONTROL_START_DISTANCE_TOKENS}_tokens_away'
        ),
        'position_set_protocol': 'V_A_D_and_S_unique_union',
    }
    for case in passing:
      trace_outputs.extend(
          _run_trace_variant(
              model_instance,
              paired_apply,
              case,
              baselines[(case.variant_id, args.trace_context_bp)],
              context_bp=args.trace_context_bp,
              output_dir=args.output_dir,
              base_configuration=trace_base_configuration,
              stages=stages,
              layers=layers,
              max_groups=args.trace_max_groups_per_variant,
          )
      )

  development = [
      baselines[(case.variant_id, DEVELOPMENT_CONTEXT_BP)] for case in cases
  ]
  effect_results = [
      result
      for case, result in zip(cases, development, strict=True)
      if case.is_effect
  ]
  correct_count = sum(
      result['direction_gate']['direction_correct'] is True
      for result in effect_results
  )
  summary = {
      'status': 'complete',
      'script_version': SCRIPT_VERSION,
      'selected_sha256': selected_sha256,
      'frozen_exons_sha256': exons_sha256,
      'contexts_bp': contexts,
      'variant_count': len(cases),
      'effect_direction_gate': {
          'correct': correct_count,
          'total': len(effect_results),
          'accuracy': (
              correct_count / len(effect_results) if effect_results else None
          ),
      },
      'baseline_files': [
          str(_baseline_path(args.output_dir, context, case))
          for context in contexts
          for case in cases
      ],
      'trace_group_count': len(trace_outputs),
      'created_at_unix_s': time.time(),
  }
  _write_atomic(args.output_dir.joinpath('summary.json'), summary)
  print(args.output_dir.joinpath('summary.json').resolve())


if __name__ == '__main__':
  main()
