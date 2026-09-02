#!/usr/bin/env python3
"""Prepare spatial sufficiency tests for the three advancing channels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SOURCE_PLAN_PATH = HERE / 'individual_channel_validation_plan_v1.json'
SOURCE_ANALYSIS_PATH = (
    HERE / 'results' /
    'individual_channel_validation_v1_model_behavior_analysis' /
    'ANALYSIS.json'
)
DEFAULT_OUTPUT = HERE / 'single_channel_sufficiency_plan_v1.json'
LOCATIONS = ('intended', 'upstream', 'downstream')
EXPECTED = (
    ('BRAF', 'E16', 2, 3),
    ('SLC25A48', 'E1', 6, 175),
    ('SLC25A48', 'E2', 5, 175),
)


class PlanError(RuntimeError):
  """Raised when a source or frozen single-channel invariant changes."""


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


def build_plan() -> dict[str, Any]:
  source = _load(SOURCE_PLAN_PATH)
  analysis = _load(SOURCE_ANALYSIS_PATH)
  if (
      source.get('schema_version')
      != 'alphagenome-individual-channel-validation-plan-v1'
      or analysis.get('schema_version')
      != 'alphagenome-individual-channel-validation-analysis-v1'
      or analysis.get('scope', {}).get('confirmation_access') is not False
      or analysis.get('individual_necessity', {}).get(
          'advancing_channel_count'
      ) != 3
  ):
    raise PlanError('Source is not the completed individual-channel result.')
  advancing = analysis['individual_necessity']['advancing_channels']
  observed = tuple(
      (
          row['selected_for_gene'], row['stage'], row['stage_index'],
          row['channel_start_inclusive'],
      )
      for row in advancing
  )
  if observed != EXPECTED:
    raise PlanError('Advancing channel set changed from the frozen rule.')
  channels = []
  for row in advancing:
    channels.append({
        'selected_for_gene': row['selected_for_gene'],
        'source_condition_id': row['condition_id'],
        'stage': row['stage'],
        'stage_index': row['stage_index'],
        'resolution_bp': row['resolution_bp'],
        'stage_width': row['stage_width'],
        'channel_start_inclusive': row['channel_start_inclusive'],
        'channel_end_exclusive': row['channel_end_exclusive'],
        'channel_count': 1,
    })
  conditions = []
  for channel in channels:
    for location in LOCATIONS:
      conditions.append({
          'condition_index': len(conditions),
          'condition_id': (
              f"only_{channel['stage']}_c"
              f"{channel['channel_start_inclusive']:04d}_{location}"
          ),
          'kind': 'single_channel_only_sufficiency',
          'channel_policy': 'only_named_single_channel',
          'location': location,
          'feature': channel,
      })
  if len(conditions) != 9:
    raise PlanError('Expected nine single-channel spatial conditions.')
  return {
      'schema_version': 'alphagenome-single-channel-sufficiency-plan-v1',
      'question': (
          'Can each advancing coordinate transfer reciprocal splice effect '
          'by itself, specifically at the intended variant neighborhood?'
      ),
      'scope': {
          **source['scope'],
          'confirmation_access': False,
      },
      'inputs': {
          'individual_validation_plan_path': SOURCE_PLAN_PATH.relative_to(
              HERE
          ).as_posix(),
          'individual_validation_plan_sha256': _sha256(SOURCE_PLAN_PATH),
          'individual_validation_analysis_path': (
              SOURCE_ANALYSIS_PATH.relative_to(HERE).as_posix()
          ),
          'individual_validation_analysis_sha256': _sha256(
              SOURCE_ANALYSIS_PATH
          ),
          'individual_validation_raw_tree_sha256': analysis['source'][
              'raw_result_tree_sha256'
          ],
      },
      'design': {
          'support': 'V',
          'selected_channel_count': len(channels),
          'locations': list(LOCATIONS),
          'condition_count': len(conditions),
          'condition': (
              'transfer only the named channel at the named V location'
          ),
          'identity_applies_per_variant': 2,
          'full_V_route_applies_per_variant': 2,
          'condition_applies_per_variant': 1,
          'planned_model_apply_count': 20 * (4 + len(conditions)),
          'planned_compile_count': 1,
      },
      'analysis': {
          'primary_population': 'selected gene significant effects',
          'sufficiency': 'bidirectional bottleneck recovery B',
          'spatial_contrast': (
              'B_intended - max(B_upstream, B_downstream) per variant'
          ),
          'pass_rule': (
              'positive median intended B, positive median spatial contrast, '
              'positive contrast in at least four of six effects, and effect '
              'median intended B greater than the gene-matched neutral median'
          ),
      },
      'controls': {
          'reciprocal_transfer': True,
          'same_allele_self_controls': True,
          'identity_and_full_route_exact_repeat': True,
          'selected_channel_donor_exact': True,
          'all_other_channels_natural_exact': True,
          'non_V_routes_disabled': True,
          'equal_shape_upstream_and_downstream_controls': True,
          'os_kernel_is_a_gate': False,
      },
      'channels': channels,
      'conditions': conditions,
      'cases': source['cases'],
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
