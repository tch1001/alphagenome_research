#!/usr/bin/env python3
"""Reference target contracts for the OpenSplice v3 development experiment.

This module is deliberately independent of the v2 route-census runner.  It
contains CPU reference reducers and fail-closed metadata/coordinate builders
for three readouts:

* the primary canonical splice-class logit margin;
* tissue-specific splice-site-usage logits; and
* a cassette-junction log-count ratio aligned to logit(PSI).

The classification and usage reducers consume internal pre-activation logits,
not public probability tracks.  The junction reducer consumes public junction
counts after exact junction coordinates have been frozen independently of
model predictions.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


CLASSIFICATION_HEAD_NAME = 'splice_sites_classification'
USAGE_HEAD_NAME = 'splice_sites_usage'
INTERNAL_LOGIT_KEY = 'logits'
PADDING_CLASS_NAME = 'padding'


@dataclasses.dataclass(frozen=True)
class CanonicalEndpoint:
  """One strand-aware canonical exon boundary in model-output coordinates."""

  role: str
  position_1based: int
  position_index: int
  track_index: int


@dataclasses.dataclass(frozen=True)
class ClassificationLogitTarget:
  """Two canonical classes and their shared no-splice/padding comparator."""

  endpoints: tuple[CanonicalEndpoint, CanonicalEndpoint]
  padding_track_index: int


@dataclasses.dataclass(frozen=True)
class PairedEndpointTarget:
  """Two canonical endpoints in one tissue- and strand-specific track."""

  endpoints: tuple[CanonicalEndpoint, CanonicalEndpoint]


@dataclasses.dataclass(frozen=True)
class EndpointReduction:
  """Per-batch acceptor, donor and symmetric mean scalar values."""

  acceptor: np.ndarray
  donor: np.ndarray
  mean: np.ndarray


@dataclasses.dataclass(frozen=True)
class JunctionCoordinate:
  """Exact public JunctionData key using 0-based half-open intron bounds."""

  chromosome: str
  start_0based: int
  end_0based: int
  strand: str

  def __post_init__(self):
    if self.strand not in {'+', '-'}:
      raise ValueError(f'Unexpected junction strand {self.strand!r}.')
    if self.start_0based < 0 or self.end_0based <= self.start_0based:
      raise ValueError(f'Invalid junction interval {self!r}.')


@dataclasses.dataclass(frozen=True)
class CassetteJunctionTarget:
  """Two inclusion junctions and one skipping junction for a cassette exon."""

  inclusion_upstream: JunctionCoordinate
  inclusion_downstream: JunctionCoordinate
  skipping: JunctionCoordinate
  track_index: int

  def __post_init__(self):
    junctions = (
        self.inclusion_upstream,
        self.inclusion_downstream,
        self.skipping,
    )
    if len(set(junctions)) != 3:
      raise ValueError('Cassette-junction target requires three distinct edges.')
    if len({junction.chromosome for junction in junctions}) != 1:
      raise ValueError('Cassette-junction edges must share one chromosome.')
    if len({junction.strand for junction in junctions}) != 1:
      raise ValueError('Cassette-junction edges must share one strand.')
    if self.track_index < 0:
      raise ValueError('Junction track index must be nonnegative.')


def _normalise_strand(value: object) -> str:
  strand = str(value).strip().lower()
  if strand in {'+', 'positive', 'strand_positive'}:
    return '+'
  if strand in {'-', 'negative', 'strand_negative'}:
    return '-'
  raise ValueError(f'Unexpected strand value {value!r}.')


def _require_columns(metadata: pd.DataFrame, columns: Sequence[str]) -> None:
  missing = sorted(set(columns) - set(metadata.columns))
  if missing:
    raise ValueError(f'Metadata is missing columns: {", ".join(missing)}.')


def _canonical_positions(
    *, exon_start_1based: int, exon_end_1based: int, strand: str
) -> tuple[tuple[str, int], tuple[str, int]]:
  strand = _normalise_strand(strand)
  if exon_start_1based <= 0 or exon_end_1based < exon_start_1based:
    raise ValueError('Invalid 1-based exon coordinates.')
  if strand == '+':
    return (
        ('acceptor', exon_start_1based),
        ('donor', exon_end_1based),
    )
  return (
      ('acceptor', exon_end_1based),
      ('donor', exon_start_1based),
  )


def _position_index(
    position_1based: int, interval_start_0based: int, interval_width: int
) -> int:
  if interval_start_0based < 0 or interval_width <= 0:
    raise ValueError('Invalid model interval.')
  index = position_1based - 1 - interval_start_0based
  if not 0 <= index < interval_width:
    raise ValueError(
        f'Position {position_1based} lies outside the model interval.'
    )
  return index


def classification_logit_target(
    metadata: pd.DataFrame,
    *,
    interval_start_0based: int,
    interval_width: int,
    exon_start_1based: int,
    exon_end_1based: int,
    strand: str,
) -> ClassificationLogitTarget:
  """Builds the exact two-boundary class-vs-padding logit target.

  The official classification head has exactly five mutually exclusive
  classes: donor/acceptor on each strand and a padding/no-splice class.  This
  builder verifies that complete contract instead of relying on column order.
  """
  _require_columns(metadata, ('name', 'strand'))
  if len(metadata) != 5:
    raise ValueError('Expected exactly five splice-classification tracks.')
  by_role_and_strand: dict[tuple[str, str], int] = {}
  padding_indices = []
  for track_index, (_, row) in enumerate(metadata.reset_index(drop=True).iterrows()):
    name = str(row['name']).strip().lower()
    if name == PADDING_CLASS_NAME:
      padding_indices.append(track_index)
      continue
    if name not in {'acceptor', 'donor'}:
      raise ValueError(f'Unexpected splice-classification track {name!r}.')
    key = (name, _normalise_strand(row['strand']))
    if key in by_role_and_strand:
      raise ValueError(f'Duplicate splice-classification track {key!r}.')
    by_role_and_strand[key] = track_index
  expected = {
      ('donor', '+'),
      ('acceptor', '+'),
      ('donor', '-'),
      ('acceptor', '-'),
  }
  if set(by_role_and_strand) != expected or len(padding_indices) != 1:
    raise ValueError('Splice-classification metadata contract does not match.')

  canonical = _canonical_positions(
      exon_start_1based=exon_start_1based,
      exon_end_1based=exon_end_1based,
      strand=strand,
  )
  normalised_strand = _normalise_strand(strand)
  endpoints = tuple(
      CanonicalEndpoint(
          role=role,
          position_1based=position,
          position_index=_position_index(
              position, interval_start_0based, interval_width
          ),
          track_index=by_role_and_strand[(role, normalised_strand)],
      )
      for role, position in canonical
  )
  return ClassificationLogitTarget(
      endpoints=endpoints,  # type: ignore[arg-type]
      padding_track_index=padding_indices[0],
  )


def resolve_usage_track(
    metadata: pd.DataFrame,
    *,
    ontology_curie: str,
    strand: str,
    assay: str = 'total RNA-seq',
) -> int:
  """Returns one exact usage track or fails on missing/ambiguous metadata."""
  _require_columns(metadata, ('name', 'strand', 'ontology_curie'))
  wanted_strand = _normalise_strand(strand)
  matches = []
  for track_index, (_, row) in enumerate(metadata.reset_index(drop=True).iterrows()):
    name = str(row['name']).strip()
    if (
        str(row['ontology_curie']).strip() == ontology_curie
        and _normalise_strand(row['strand']) == wanted_strand
        and name.lower().startswith('usage_')
        and assay.lower() in name.lower()
    ):
      matches.append(track_index)
  if len(matches) != 1:
    raise ValueError(
        f'Expected one {ontology_curie}/{wanted_strand}/{assay} usage track; '
        f'found {len(matches)}.'
    )
  return matches[0]


def usage_logit_target(
    metadata: pd.DataFrame,
    *,
    ontology_curie: str,
    interval_start_0based: int,
    interval_width: int,
    exon_start_1based: int,
    exon_end_1based: int,
    strand: str,
    assay: str = 'total RNA-seq',
) -> PairedEndpointTarget:
  """Builds the two-boundary target for one exact usage-logit track."""
  track_index = resolve_usage_track(
      metadata,
      ontology_curie=ontology_curie,
      strand=strand,
      assay=assay,
  )
  endpoints = tuple(
      CanonicalEndpoint(
          role=role,
          position_1based=position,
          position_index=_position_index(
              position, interval_start_0based, interval_width
          ),
          track_index=track_index,
      )
      for role, position in _canonical_positions(
          exon_start_1based=exon_start_1based,
          exon_end_1based=exon_end_1based,
          strand=strand,
      )
  )
  return PairedEndpointTarget(endpoints=endpoints)  # type: ignore[arg-type]


def _as_batched_tracks(values: np.ndarray, *, label: str) -> np.ndarray:
  values = np.asarray(values)
  if values.ndim == 2:
    values = values[None, ...]
  if values.ndim != 3:
    raise ValueError(f'{label} must have shape [batch, position, track].')
  values = values.astype(np.float64, copy=False)
  if not np.all(np.isfinite(values)):
    raise ValueError(f'{label} contains a non-finite value.')
  return values


def _validate_endpoint_bounds(
    values: np.ndarray, endpoints: Sequence[CanonicalEndpoint]
) -> None:
  for endpoint in endpoints:
    if not 0 <= endpoint.position_index < values.shape[1]:
      raise ValueError(f'{endpoint.role} position index is out of bounds.')
    if not 0 <= endpoint.track_index < values.shape[2]:
      raise ValueError(f'{endpoint.role} track index is out of bounds.')


def reduce_classification_logit_margin(
    logits: np.ndarray, target: ClassificationLogitTarget
) -> EndpointReduction:
  """Reduces relevant-class minus padding logits at canonical boundaries.

  Subtracting the padding logit makes the target invariant to a common shift
  of all five logits and equals log(p_relevant / p_padding) without evaluating
  or inverting a softmax.
  """
  values = _as_batched_tracks(logits, label='Classification logits')
  _validate_endpoint_bounds(values, target.endpoints)
  if not 0 <= target.padding_track_index < values.shape[2]:
    raise ValueError('Padding track index is out of bounds.')
  reduced = {}
  for endpoint in target.endpoints:
    reduced[endpoint.role] = (
        values[:, endpoint.position_index, endpoint.track_index]
        - values[:, endpoint.position_index, target.padding_track_index]
    )
  return EndpointReduction(
      acceptor=reduced['acceptor'],
      donor=reduced['donor'],
      mean=(reduced['acceptor'] + reduced['donor']) / 2.0,
  )


def reduce_usage_logits(
    logits: np.ndarray, target: PairedEndpointTarget
) -> EndpointReduction:
  """Reduces two canonical usage logits for one exact tissue/strand track."""
  values = _as_batched_tracks(logits, label='Splice-site-usage logits')
  _validate_endpoint_bounds(values, target.endpoints)
  reduced = {
      endpoint.role: values[
          :, endpoint.position_index, endpoint.track_index
      ]
      for endpoint in target.endpoints
  }
  return EndpointReduction(
      acceptor=reduced['acceptor'],
      donor=reduced['donor'],
      mean=(reduced['acceptor'] + reduced['donor']) / 2.0,
  )


def delta_endpoint_reduction(
    reference: EndpointReduction, alternate: EndpointReduction
) -> EndpointReduction:
  """Returns ALT-minus-REF endpoint and mean deltas."""
  if not (
      reference.acceptor.shape
      == reference.donor.shape
      == reference.mean.shape
      == alternate.acceptor.shape
      == alternate.donor.shape
      == alternate.mean.shape
  ):
    raise ValueError('REF and ALT endpoint reductions have different shapes.')
  return EndpointReduction(
      acceptor=alternate.acceptor - reference.acceptor,
      donor=alternate.donor - reference.donor,
      mean=alternate.mean - reference.mean,
  )


def extract_internal_logits(
    predictions: Mapping[str, Mapping[str, object]], *, head_name: str
) -> np.ndarray:
  """Extracts an internal logit tensor without accepting public probabilities."""
  if head_name not in {CLASSIFICATION_HEAD_NAME, USAGE_HEAD_NAME}:
    raise ValueError(f'Unsupported v3 logit head {head_name!r}.')
  try:
    logits = predictions[head_name][INTERNAL_LOGIT_KEY]
  except KeyError as error:
    raise ValueError(
        f'Missing internal target {head_name!r}/{INTERNAL_LOGIT_KEY!r}; '
        'do not reconstruct logits from public probabilities.'
    ) from error
  return _as_batched_tracks(np.asarray(logits), label=f'{head_name} logits')


def resolve_junction_track(
    metadata: pd.DataFrame,
    *,
    ontology_curie: str,
    assay: str = 'total RNA-seq',
) -> int:
  """Returns one exact strand-agnostic junction tissue track."""
  _require_columns(metadata, ('name', 'ontology_curie'))
  matches = []
  for track_index, (_, row) in enumerate(metadata.reset_index(drop=True).iterrows()):
    name = str(row['name']).strip()
    if (
        str(row['ontology_curie']).strip() == ontology_curie
        and name.lower().startswith('junction_')
        and assay.lower() in name.lower()
    ):
      matches.append(track_index)
  if len(matches) != 1:
    raise ValueError(
        f'Expected one {ontology_curie}/{assay} junction track; '
        f'found {len(matches)}.'
    )
  return matches[0]


def reduce_cassette_junction_logit_psi(
    junctions: Sequence[JunctionCoordinate],
    values: np.ndarray,
    target: CassetteJunctionTarget,
    *,
    epsilon: float = 1e-7,
) -> float:
  """Computes log(mean inclusion count / skipping count) for one allele.

  This is logit of `mean_inclusion / (mean_inclusion + skipping)`, matching the
  OpenSplice inclusion-versus-skipping logit scale.  Exact coordinate joins are
  mandatory; prediction-selected or nearest junctions are not accepted.
  """
  if not np.isfinite(epsilon) or epsilon <= 0:
    raise ValueError('epsilon must be finite and positive.')
  counts = np.asarray(values, dtype=np.float64)
  if counts.ndim != 2 or counts.shape[0] != len(junctions):
    raise ValueError('Junction values must have shape [junction, track].')
  if not np.all(np.isfinite(counts)) or np.any(counts < 0):
    raise ValueError('Junction counts must be finite and nonnegative.')
  if not 0 <= target.track_index < counts.shape[1]:
    raise ValueError('Junction track index is out of bounds.')
  index: dict[JunctionCoordinate, list[int]] = {}
  for row_index, junction in enumerate(junctions):
    index.setdefault(junction, []).append(row_index)

  def exact_count(junction: JunctionCoordinate) -> float:
    matches = index.get(junction, [])
    if len(matches) != 1:
      raise ValueError(
          f'Expected one exact junction row for {junction!r}; '
          f'found {len(matches)}.'
      )
    return float(counts[matches[0], target.track_index])

  inclusion = np.mean([
      exact_count(target.inclusion_upstream),
      exact_count(target.inclusion_downstream),
  ])
  skipping = exact_count(target.skipping)
  return float(np.log(inclusion + epsilon) - np.log(skipping + epsilon))


def delta_cassette_junction_logit_psi(
    reference_junctions: Sequence[JunctionCoordinate],
    reference_values: np.ndarray,
    alternate_junctions: Sequence[JunctionCoordinate],
    alternate_values: np.ndarray,
    target: CassetteJunctionTarget,
    *,
    epsilon: float = 1e-7,
) -> float:
  """Returns ALT-minus-REF cassette-junction logit(PSI)."""
  reference = reduce_cassette_junction_logit_psi(
      reference_junctions,
      reference_values,
      target,
      epsilon=epsilon,
  )
  alternate = reduce_cassette_junction_logit_psi(
      alternate_junctions,
      alternate_values,
      target,
      epsilon=epsilon,
  )
  return alternate - reference
