#!/usr/bin/env python3
"""Prepare individual-channel necessity and 8-channel causal validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
REFINEMENT_PLAN_PATH = HERE / 'channel_refinement_plan_v1.json'
REFINEMENT_ANALYSIS_PATH = (
    HERE / 'results' / 'channel_refinement_v1_model_behavior_analysis'
    / 'ANALYSIS.json'
)
SPATIAL_PLAN_PATH = HERE / 'spatial_encoder_skip_plan_v1.json'
DEFAULT_OUTPUT = HERE / 'individual_channel_validation_plan_v1.json'
LOCATIONS = ('intended', 'upstream', 'downstream')
SELECTED_CHILD_IDS = (
    'E32_c0000_0007',
    'E16_c0000_0007',
    'E2_c0168_0175',
    'E1_c0168_0175',
)


class PlanError(RuntimeError):
  """Raised when an input or frozen individual-channel invariant changes."""


def _load(path: Path) -> dict[str, Any]:
  try:
    value = json.loads(path.read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError) as error:
    raise PlanError(f'Cannot read {path}.') from error
  if not isinstance(value, dict):
    raise PlanError(f'JSON root is not an object: {path}.')
  return value


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _feature(
    child: Mapping[str, Any], start: int, end: int
) -> dict[str, Any]:
  return {
      'selected_for_gene': (
          'BRAF' if child['group_id'].startswith(('E32_', 'E16_'))
          else 'SLC25A48'
      ),
      'parent_child_id': child['group_id'],
      'stage': child['stage'],
      'stage_index': child['stage_index'],
      'resolution_bp': child['resolution_bp'],
      'stage_width': child['stage_width'],
      'channel_start_inclusive': start,
      'channel_end_exclusive': end,
      'channel_count': end - start,
  }


def _v_positions(case: Mapping[str, Any]) -> dict[str, Any]:
  result = {}
  for location in LOCATIONS:
    matches = [
        condition for condition in case['conditions']
        if condition['condition_id'] == f'V_{location}'
    ]
    if len(matches) != 1:
      raise PlanError(f'Missing V_{location} condition.')
    result[location] = [
        {
            'stage': stage['player'],
            'resolution_bp': stage['resolution_bp'],
            'route_enabled': stage['enabled'],
            'positions': stage['positions'],
        }
        for stage in matches[0]['stages']
    ]
  return result


def build_plan() -> dict[str, Any]:
  """Build the frozen 32-channel necessity plus sufficiency design."""
  refinement = _load(REFINEMENT_PLAN_PATH)
  analysis = _load(REFINEMENT_ANALYSIS_PATH)
  spatial = _load(SPATIAL_PLAN_PATH)
  if (
      refinement.get('schema_version')
      != 'alphagenome-channel-refinement-plan-v1'
      or analysis.get('schema_version')
      != 'alphagenome-channel-refinement-analysis-v1'
      or spatial.get('scope', {}).get('confirmation_access') is not False
      or analysis.get('scope', {}).get('confirmation_access') is not False
  ):
    raise PlanError('Inputs are not completed development-only sources.')
  observed = analysis['recommended_individual_channel_refinement'][
      'deduplicated_child_ids'
  ]
  if observed != list(SELECTED_CHILD_IDS):
    raise PlanError('Selected 8-channel candidates changed.')
  children_by_id = {
      group['group_id']: group for group in refinement['groups']
  }
  selected = [children_by_id[group_id] for group_id in SELECTED_CHILD_IDS]

  conditions = []
  for child in selected:
    for channel in range(
        child['channel_start_inclusive'], child['channel_end_exclusive']
    ):
      conditions.append({
          'condition_index': len(conditions),
          'condition_id': f"necessity_{child['stage']}_c{channel:04d}",
          'kind': 'individual_channel_necessity',
          'channel_policy': 'all_candidate_channels_except_named_channel',
          'location': 'intended',
          'feature': _feature(child, channel, channel + 1),
      })
  necessity_count = len(conditions)
  for child in selected:
    for location in LOCATIONS:
      conditions.append({
          'condition_index': len(conditions),
          'condition_id': (
              f"sufficiency_{child['group_id']}_{location}"
          ),
          'kind': 'eight_channel_only_sufficiency',
          'channel_policy': 'only_named_eight_channel_child',
          'location': location,
          'feature': _feature(
              child,
              child['channel_start_inclusive'],
              child['channel_end_exclusive'],
          ),
      })
  if necessity_count != 32 or len(conditions) != 44:
    raise PlanError('Expected 32 necessity and 12 sufficiency conditions.')

  spatial_by_variant = {
      case['variant_id']: case for case in spatial['cases']
  }
  cases = []
  for case in refinement['cases']:
    spatial_case = spatial_by_variant.get(case['variant_id'])
    if (
        spatial_case is None
        or spatial_case['order'] != case['order']
        or spatial_case['gene'] != case['gene']
    ):
      raise PlanError('Spatial and refinement cases are not aligned.')
    positions = _v_positions(spatial_case)
    intended = positions['intended']
    if any(
        current['stage'] != prior['stage']
        or current['positions'] != prior['positions']
        for current, prior in zip(intended, case['stages'], strict=True)
    ):
      raise PlanError('Intended V positions differ from refinement plan.')
    cases.append({
        **case,
        'positions_by_location': positions,
    })
  return {
      'schema_version': 'alphagenome-individual-channel-validation-plan-v1',
      'question': (
          'Which individual channels are necessary inside the four leading '
          'exon-specific subspaces, and are those eight-channel subspaces '
          'sufficient only at the intended variant neighborhood?'
      ),
      'scope': {
          **refinement['scope'],
          'confirmation_access': False,
      },
      'inputs': {
          'refinement_plan_path': REFINEMENT_PLAN_PATH.relative_to(
              HERE
          ).as_posix(),
          'refinement_plan_sha256': _sha256(REFINEMENT_PLAN_PATH),
          'refinement_analysis_path': REFINEMENT_ANALYSIS_PATH.relative_to(
              HERE
          ).as_posix(),
          'refinement_analysis_sha256': _sha256(REFINEMENT_ANALYSIS_PATH),
          'refinement_raw_result_tree_sha256': analysis['source'][
              'raw_result_tree_sha256'
          ],
          'spatial_plan_path': SPATIAL_PLAN_PATH.relative_to(HERE).as_posix(),
          'spatial_plan_sha256': _sha256(SPATIAL_PLAN_PATH),
      },
      'design': {
          'support': 'V',
          'selected_child_count': len(selected),
          'selected_individual_channel_count': 32,
          'individual_necessity_condition_count': necessity_count,
          'eight_channel_sufficiency_condition_count': 12,
          'condition_count': len(conditions),
          'identity_applies_per_variant': 2,
          'full_V_route_applies_per_variant': 2,
          'condition_applies_per_variant': 1,
          'planned_model_apply_count': 20 * (4 + len(conditions)),
          'planned_compile_count': 1,
      },
      'analysis': {
          'individual_necessity': (
              'B_full_V - B_without_channel, ranked separately within each '
              'selected child and gene'
          ),
          'eight_channel_sufficiency': (
              'B_only_child at intended V positions and per-variant spatial '
              'contrast versus max(upstream, downstream)'
          ),
          'individual_advance_rule': (
              'positive effect median necessity, effect median greater than '
              'gene-matched neutral median, and positive loss in at least '
              'four of six effect variants'
          ),
          'interpretation_limit': (
              'A channel is a model coordinate, not a biological factor; '
              'sequence-level validation is required.'
          ),
      },
      'controls': {
          'reciprocal_transfer': True,
          'same_allele_self_controls': True,
          'identity_and_full_route_exact_repeat': True,
          'selected_channel_donor_exact': True,
          'withheld_channel_natural_exact': True,
          'non_V_routes_disabled': True,
          'equal_shape_upstream_and_downstream_sufficiency_controls': True,
          'neutral_variants_secondary_specificity': True,
          'os_kernel_is_a_gate': False,
      },
      'selected_children': selected,
      'conditions': conditions,
      'cases': cases,
  }


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument('--stdout', action='store_true')
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  payload = json.dumps(
      build_plan(), indent=2, sort_keys=True, allow_nan=False
  ) + '\n'
  if args.stdout:
    print(payload, end='')
    return
  if args.output.exists() or args.output.is_symlink():
    raise PlanError(f'Output already exists: {args.output}.')
  args.output.write_text(payload, encoding='utf-8')
  print(args.output)


if __name__ == '__main__':
  main()
