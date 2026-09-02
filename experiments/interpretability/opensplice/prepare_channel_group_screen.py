#!/usr/bin/env python3
"""Prepare a deterministic grouped-channel screen for the V-local skip route."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
SPATIAL_PLAN_PATH = HERE / 'spatial_encoder_skip_plan_v1.json'
SPATIAL_ANALYSIS_PATH = (
    HERE / 'results' / 'spatial_encoder_skip_v1_model_behavior_analysis'
    / 'ANALYSIS.json'
)
DEFAULT_OUTPUT = HERE / 'channel_group_screen_plan_v1.json'
STAGES = (
    ('E64', 64, 1536, False),
    ('E32', 32, 1408, True),
    ('E16', 16, 1280, True),
    ('E8', 8, 1152, True),
    ('E4', 4, 1024, False),
    ('E2', 2, 896, True),
    ('E1', 1, 768, True),
)
GROUP_SIZE = 32
MAX_CHANNELS = 1536


class PlanError(RuntimeError):
  """Raised when a channel-screen input or invariant changes."""


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
  try:
    value = json.loads(path.read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError) as error:
    raise PlanError(f'Cannot load {path}.') from error
  if not isinstance(value, dict):
    raise PlanError(f'JSON root is not an object: {path}.')
  return value


def channel_groups() -> list[dict[str, Any]]:
  groups = []
  for stage_index, (stage, resolution, width, enabled) in enumerate(STAGES):
    if not enabled:
      continue
    for start in range(0, width, GROUP_SIZE):
      end = min(start + GROUP_SIZE, width)
      groups.append({
          'group_index': len(groups),
          'group_id': f'{stage}_c{start:04d}_{end - 1:04d}',
          'stage': stage,
          'stage_index': stage_index,
          'resolution_bp': resolution,
          'stage_width': width,
          'channel_start_inclusive': start,
          'channel_end_exclusive': end,
          'channel_count': end - start,
      })
  return groups


def _v_condition(case: Mapping[str, Any]) -> Mapping[str, Any]:
  matches = [
      condition for condition in case['conditions']
      if condition['condition_id'] == 'V_intended'
  ]
  if len(matches) != 1:
    raise PlanError('Spatial case does not have one V_intended condition.')
  return matches[0]


def build_plan() -> dict[str, Any]:
  spatial = _load(SPATIAL_PLAN_PATH)
  analysis = _load(SPATIAL_ANALYSIS_PATH)
  if (
      spatial.get('scope', {}).get('development_only') is not True
      or spatial.get('scope', {}).get('confirmation_access') is not False
      or analysis.get('scope', {}).get('confirmation_access') is not False
      or analysis.get('supports_passing_frozen_rule_in_both_genes')
      != ['V', 'A', 'S']
  ):
    raise PlanError(
        'Spatial source result is not the completed development result.'
    )
  groups = channel_groups()
  if (
      len(groups) != 172
      or sum(group['channel_count'] for group in groups) != 5504
  ):
    raise PlanError('Unexpected grouped-channel partition.')
  cases = []
  for case in spatial['cases']:
    condition = _v_condition(case)
    stages = []
    for planned, expected in zip(condition['stages'], STAGES, strict=True):
      stage, resolution, width, enabled = expected
      if (
          planned['player'] != stage
          or planned['resolution_bp'] != resolution
          or planned['enabled'] is not enabled
      ):
        raise PlanError('V-local stage binding changed.')
      stages.append({
          'stage': stage,
          'stage_index': len(stages),
          'resolution_bp': resolution,
          'channel_width': width,
          'route_enabled': enabled,
          'positions': planned['positions'],
      })
    cases.append({
        'order': case['order'],
        'gene': case['gene'],
        'variant_id': case['variant_id'],
        'selection_class': case['selection_class'],
        'chromosome': case['chromosome'],
        'strand': case['strand'],
        'interval_start_0based': case['interval_start_0based'],
        'interval_end_0based_exclusive': case['interval_end_0based_exclusive'],
        'variant_position_1based': case['variant_position_1based'],
        'stages': stages,
    })
  applies_per_case = 2 + 2 + len(groups)
  return {
      'schema_version': 'alphagenome-channel-group-screen-plan-v1',
      'question': (
          'Which contiguous 32-channel blocks are necessary for V-local '
          'bidirectional recovery in the E32/E16/E8/E2/E1 skip route?'
      ),
      'scope': {
          'development_only': True,
          'confirmation_access': False,
          'genes': ['BRAF', 'SLC25A48'],
          'variant_count': 20,
          'effect_count': 12,
          'neutral_count': 8,
      },
      'inputs': {
          'spatial_plan_path': SPATIAL_PLAN_PATH.relative_to(HERE).as_posix(),
          'spatial_plan_sha256': _sha256(SPATIAL_PLAN_PATH),
          'spatial_analysis_path': (
              SPATIAL_ANALYSIS_PATH.relative_to(HERE).as_posix()
          ),
          'spatial_analysis_sha256': _sha256(SPATIAL_ANALYSIS_PATH),
          'spatial_raw_result_tree_sha256': analysis['source'][
              'raw_result_tree_sha256'
          ],
      },
      'design': {
          'support': 'V',
          'candidate_players': [
              stage for stage, _, _, enabled in STAGES if enabled
          ],
          'grouping': 'contiguous_nonoverlapping_channel_blocks',
          'nominal_group_size': GROUP_SIZE,
          'group_count': len(groups),
          'total_candidate_channels': sum(
              width for _, _, width, enabled in STAGES if enabled
          ),
          'maximum_channel_axis': MAX_CHANNELS,
          'condition': (
              'transfer every candidate channel except the named group'
          ),
          'identity_applies_per_variant': 2,
          'full_V_route_applies_per_variant': 2,
          'group_applies_per_variant': 1,
          'planned_model_apply_count': 20 * applies_per_case,
          'planned_compile_count': 1,
      },
      'analysis': {
          'primary_population': 'significant_effect',
          'per_case_necessity_loss': 'B_full_V - B_without_group',
          'cross_gene_ranking': (
              'descending minimum of the two gene-level median necessity losses'
          ),
          'gene_specific_ranking': (
              'descending median necessity loss within each gene'
          ),
          'interpretation_limit': (
              'A group screen identifies necessary subspaces; it does not '
              'assign effects additively or resolve individual channels.'
          ),
      },
      'controls': {
          'full_V_route_closure_against_spatial_result': (
              'aggregate recovery comparison; cross-executable target '
              'equality is diagnostic'
          ),
          'reciprocal_transfer': True,
          'same_allele_self_controls': True,
          'identity_and_full_route_exact_repeat': True,
          'per_group_selected_channel_donor_exact': True,
          'per_group_withheld_channel_noop_exact': True,
          'non_V_routes_disabled': True,
          'neutral_variants_secondary_specificity': True,
          'os_kernel_is_a_gate': False,
      },
      'follow_up': {
          'top_cross_gene_groups': (
              'recursively split to individual channels and test only-group '
              'sufficiency plus V/upstream/downstream localization'
          ),
          'top_gene_specific_groups': (
              'retain separately to test exon-specific feature programs'
          ),
      },
      'groups': groups,
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
