#!/usr/bin/env python3
"""Build the deterministic model-behavior plan for spatial skip patching."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
SELECTED_PATH = HERE / 'selected_variants_v2.tsv'
EXONS_PATH = HERE / 'frozen_exons_v2.tsv'
DEFAULT_OUTPUT = HERE / 'spatial_encoder_skip_plan_v1.json'
CONTEXT_BP = 16_384
MINIMUM_CONTROL_DISTANCE_BP = 512
STAGES = (
    ('E64', 64), ('E32', 32), ('E16', 16), ('E8', 8),
    ('E4', 4), ('E2', 2), ('E1', 1),
)
CANDIDATE_PLAYERS = ('E32', 'E16', 'E8', 'E2', 'E1')
SUPPORT_NAMES = ('V', 'A', 'D', 'S')
CONTROL_LOCATIONS = ('intended', 'upstream', 'downstream')
DEVELOPMENT_GENES = ('BRAF', 'SLC25A48')


class PlanError(RuntimeError):
  """Raised when the biological plan inputs or coordinates are invalid."""


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_tsv(path: Path) -> list[dict[str, str]]:
  with path.open(newline='', encoding='utf-8') as handle:
    return list(csv.DictReader(handle, delimiter='\t'))


def _components(tokens: Sequence[int]) -> list[list[int]]:
  result = []
  for token in sorted(set(tokens)):
    if not result or token != result[-1][-1] + 1:
      result.append([token])
    else:
      result[-1].append(token)
  return result


def guarded_support(
    positions_1based: Iterable[int], *, interval_start: int,
    resolution: int,
) -> tuple[int, ...]:
  """Maps bases to tokens and adds one guard token per component side."""
  sequence_length = CONTEXT_BP // resolution
  tokens = sorted(set(
      (int(position) - 1 - interval_start) // resolution
      for position in positions_1based
  ))
  if not tokens or any(
      token < 0 or token >= sequence_length for token in tokens
  ):
    raise PlanError('Biological support lies outside the model interval.')
  guarded = set(tokens)
  for component in _components(tokens):
    guarded.add(component[0] - 1)
    guarded.add(component[-1] + 1)
  if any(token < 0 or token >= sequence_length for token in guarded):
    raise PlanError('Guarded support lies outside the model interval.')
  return tuple(sorted(guarded))


def translated_control(
    support: Sequence[int], *, direction: int, resolution: int,
    forbidden: Iterable[int],
) -> tuple[int, ...]:
  """Translates a support by at least 512 bp until it is valid and disjoint."""
  if direction not in (-1, 1):
    raise ValueError('Control direction must be -1 or 1.')
  sequence_length = CONTEXT_BP // resolution
  forbidden = set(forbidden)
  minimum_offset = math.ceil(MINIMUM_CONTROL_DISTANCE_BP / resolution)
  for extra in range(sequence_length):
    offset = direction * (minimum_offset + extra)
    candidate = tuple(token + offset for token in support)
    if min(candidate) < 0 or max(candidate) >= sequence_length:
      break
    if not set(candidate) & forbidden:
      return candidate
  raise PlanError('Could not place a valid shifted spatial control.')


def _load_development_cases() -> list[dict[str, Any]]:
  exon_rows = {
      row['gene']: row for row in _read_tsv(EXONS_PATH)
      if row['gene'] in DEVELOPMENT_GENES
  }
  if tuple(exon_rows) != DEVELOPMENT_GENES:
    raise PlanError('Development exon metadata order changed.')
  selected = [
      row for row in _read_tsv(SELECTED_PATH)
      if row['gene'] in DEVELOPMENT_GENES
  ]
  if len(selected) != 20:
    raise PlanError('Expected exactly 20 development variants.')
  cases = []
  for order, row in enumerate(selected):
    exon = exon_rows[row['gene']]
    selection_class = row['selection_class']
    expected = (
        'significant_effect'
        if order in (*range(6), *range(10, 16))
        else 'neutral_control'
    )
    if selection_class != expected:
      raise PlanError(f'Unexpected selection class at order {order}.')
    exon_start = int(exon['exon_start_1based'])
    exon_end = int(exon['exon_end_1based'])
    center_1based = (exon_start + exon_end) // 2
    interval_start = center_1based - 1 - CONTEXT_BP // 2
    strand = exon['strand']
    acceptor = exon_start if strand == '+' else exon_end
    donor = exon_end if strand == '+' else exon_start
    variant = int(row['position_1based'])
    if not all(
        interval_start <= position - 1 < interval_start + CONTEXT_BP
        for position in (variant, acceptor, donor)
    ):
      raise PlanError(f'Case {order} does not fit the development context.')
    cases.append({
        'order': order,
        'gene': row['gene'],
        'variant_id': row['variant_id'],
        'selection_class': selection_class,
        'chromosome': row['chromosome'],
        'strand': strand,
        'interval_start_0based': interval_start,
        'interval_end_0based_exclusive': interval_start + CONTEXT_BP,
        'variant_position_1based': variant,
        'acceptor_position_1based': acceptor,
        'donor_position_1based': donor,
    })
  return cases


def _case_conditions(case: Mapping[str, Any]) -> list[dict[str, Any]]:
  bases = {
      'V': (case['variant_position_1based'],),
      'A': (case['acceptor_position_1based'],),
      'D': (case['donor_position_1based'],),
      'S': (
          case['acceptor_position_1based'],
          case['donor_position_1based'],
      ),
  }
  biological_by_stage = {}
  for player, resolution in STAGES:
    biological_by_stage[player] = set()
    for support in SUPPORT_NAMES:
      biological_by_stage[player].update(guarded_support(
          bases[support], interval_start=case['interval_start_0based'],
          resolution=resolution,
      ))

  conditions = []
  for support_name in SUPPORT_NAMES:
    intended_by_stage = {
        player: guarded_support(
            bases[support_name], interval_start=case['interval_start_0based'],
            resolution=resolution,
        )
        for player, resolution in STAGES
    }
    for location in CONTROL_LOCATIONS:
      stages = []
      for player, resolution in STAGES:
        intended = intended_by_stage[player]
        if location == 'intended':
          positions = intended
        else:
          positions = translated_control(
              intended,
              direction=-1 if location == 'upstream' else 1,
              resolution=resolution,
              forbidden=biological_by_stage[player],
          )
        enabled = player in CANDIDATE_PLAYERS
        stages.append({
            'player': player,
            'resolution_bp': resolution,
            'enabled': enabled,
            'positions': list(positions) if enabled else [],
        })
      conditions.append({
          'condition_id': f'{support_name}_{location}',
          'support': support_name,
          'location': location,
          'stages': stages,
      })
  return conditions


def build_plan() -> dict[str, Any]:
  cases = _load_development_cases()
  planned_cases = [
      {**case, 'conditions': _case_conditions(case)} for case in cases
  ]
  maximum_slots = max(
      len(stage['positions'])
      for case in planned_cases
      for condition in case['conditions']
      for stage in condition['stages']
  )
  if maximum_slots > 6:
    raise PlanError(f'Unexpected spatial slot count: {maximum_slots}.')
  condition_ids = [
      f'{support}_{location}'
      for support in SUPPORT_NAMES for location in CONTROL_LOCATIONS
  ]
  return {
      'schema_version': 'alphagenome-spatial-encoder-skip-plan-v1',
      'goal': (
          'Test whether the mask-110 encoder-skip route localizes around the '
          'variant, acceptor, donor, or acceptor/donor union.'
      ),
      'scope': {
          'development_only': True,
          'genes': list(DEVELOPMENT_GENES),
          'variant_count': 20,
          'effect_count': 12,
          'neutral_count': 8,
          'confirmation_access': False,
      },
      'inputs': {
          'selected_variants_path': SELECTED_PATH.relative_to(HERE).as_posix(),
          'selected_variants_sha256': _sha256(SELECTED_PATH),
          'frozen_exons_path': EXONS_PATH.relative_to(HERE).as_posix(),
          'frozen_exons_sha256': _sha256(EXONS_PATH),
          'exploratory_source_cube_sha256': (
              '95ddc79c634fecdd4b4e43e090ac760cdc26a268f09fed7cee76843f982e45de'
          ),
      },
      'model_behavior_design': {
          'context_bp': CONTEXT_BP,
          'candidate_mask': 110,
          'candidate_players': list(CANDIDATE_PLAYERS),
          'disabled_players': ['E64', 'E4'],
          'transformer_context': 'natural_recipient_T',
          'supports': list(SUPPORT_NAMES),
          'locations': list(CONTROL_LOCATIONS),
          'guard_tokens_per_contiguous_component_side': 1,
          'minimum_shift_distance_bp': MINIMUM_CONTROL_DISTANCE_BP,
          'maximum_position_slots': maximum_slots,
          'condition_ids': condition_ids,
          'condition_count_per_variant': len(condition_ids),
          'identity_applies_per_variant': 2,
          'applies_per_condition': 2,
          'planned_model_apply_count': 20 * 2 + 20 * len(condition_ids) * 2,
          'planned_compile_count': 1,
      },
      'analysis_rule': {
          'primary_statistic': 'bidirectional bottleneck recovery B',
          'spatial_contrast': 'B_intended - max(B_upstream, B_downstream)',
          'candidate_requires_each_gene': {
              'median_intended_B_at_least': 0.25,
              'median_spatial_contrast_strictly_greater_than': 0.0,
          },
          'secondary_specificity': (
              'median absolute effect movement versus neutral movement, '
              'reported per gene without treating experimental neutrals as '
              'AlphaGenome-null'
          ),
      },
      'controls': {
          'reciprocal_transfer': True,
          'same_allele_self_controls': True,
          'exact_repeat': True,
          'disabled_skip_noop_checks': True,
          'within_variant_equal_shape_shifted_controls': True,
          'os_kernel_is_a_gate': False,
      },
      'cases': planned_cases,
  }


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument('--stdout', action='store_true')
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  plan = build_plan()
  payload = json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + '\n'
  if args.stdout:
    print(payload, end='')
    return
  if args.output.exists() or args.output.is_symlink():
    raise PlanError(f'Output already exists: {args.output}.')
  args.output.write_text(payload, encoding='utf-8')
  print(args.output)


if __name__ == '__main__':
  main()
