#!/usr/bin/env python3
"""Prepare the deterministic 8-channel refinement of selected parent blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SCREEN_PLAN_PATH = HERE / 'channel_group_screen_plan_v1.json'
SCREEN_ANALYSIS_PATH = (
    HERE / 'results' / 'channel_group_screen_v1_model_behavior_analysis'
    / 'ANALYSIS.json'
)
DEFAULT_OUTPUT = HERE / 'channel_refinement_plan_v1.json'
CHILD_SIZE = 8


class PlanError(RuntimeError):
  """Raised when a refinement input or deterministic invariant changes."""


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
  """Build the fixed five-parent, twenty-child development plan."""
  screen = _load(SCREEN_PLAN_PATH)
  analysis = _load(SCREEN_ANALYSIS_PATH)
  if (
      screen.get('schema_version')
      != 'alphagenome-channel-group-screen-plan-v1'
      or screen.get('scope', {}).get('confirmation_access') is not False
      or analysis.get('schema_version')
      != 'alphagenome-channel-group-screen-analysis-v1'
      or analysis.get('scope', {}).get('confirmation_access') is not False
      or analysis.get('scope', {}).get('group_result_count') != 3440
  ):
    raise PlanError('Source is not the completed development channel screen.')
  selected_ids = analysis['recommended_refinement'][
      'deterministic_parent_group_ids'
  ]
  if selected_ids != [
      'E1_c0160_0191',
      'E32_c0000_0031',
      'E16_c0000_0031',
      'E16_c0512_0543',
      'E2_c0160_0191',
  ]:
    raise PlanError('Selected parent set changed from the frozen rule.')
  by_id = {group['group_id']: group for group in screen['groups']}
  parents = [by_id[group_id] for group_id in selected_ids]
  groups = []
  for parent_rank, parent in enumerate(parents):
    start = parent['channel_start_inclusive']
    end = parent['channel_end_exclusive']
    if end - start != 32:
      raise PlanError('A refinement parent is not 32 channels wide.')
    children = []
    for child_start in range(start, end, CHILD_SIZE):
      child_end = child_start + CHILD_SIZE
      child = {
          'group_index': len(groups),
          'group_id': (
              f"{parent['stage']}_c{child_start:04d}_{child_end - 1:04d}"
          ),
          'stage': parent['stage'],
          'stage_index': parent['stage_index'],
          'resolution_bp': parent['resolution_bp'],
          'stage_width': parent['stage_width'],
          'channel_start_inclusive': child_start,
          'channel_end_exclusive': child_end,
          'channel_count': CHILD_SIZE,
          'parent_rank': parent_rank,
          'parent_group_id': parent['group_id'],
      }
      children.append(child)
      groups.append(child)
    if (
        children[0]['channel_start_inclusive'] != start
        or children[-1]['channel_end_exclusive'] != end
    ):
      raise PlanError('Children do not exactly partition their parent.')
  if len(groups) != 20 or len({group['group_id'] for group in groups}) != 20:
    raise PlanError('Refinement must have 20 unique child groups.')
  return {
      'schema_version': 'alphagenome-channel-refinement-plan-v1',
      'question': (
          'Which 8-channel children account for the necessity of the three '
          'shared and two top gene-ranked 32-channel parent blocks?'
      ),
      'scope': {
          **screen['scope'],
          'confirmation_access': False,
      },
      'inputs': {
          'channel_screen_plan_path': SCREEN_PLAN_PATH.relative_to(
              HERE
          ).as_posix(),
          'channel_screen_plan_sha256': _sha256(SCREEN_PLAN_PATH),
          'channel_screen_analysis_path': SCREEN_ANALYSIS_PATH.relative_to(
              HERE
          ).as_posix(),
          'channel_screen_analysis_sha256': _sha256(SCREEN_ANALYSIS_PATH),
          'channel_screen_raw_result_tree_sha256': analysis['source'][
              'raw_result_tree_sha256'
          ],
      },
      'design': {
          'support': 'V',
          'parent_selection_rule': (
              'top three positive cross-gene maximin blocks with effect '
              'median exceeding neutral in both genes, plus the top block '
              'within each gene, deduplicated in that order'
          ),
          'parent_group_count': len(parents),
          'child_grouping': 'contiguous_nonoverlapping_channel_blocks',
          'child_group_size': CHILD_SIZE,
          'group_count': len(groups),
          'condition': (
              'transfer every V-local candidate channel except the named '
              '8-channel child'
          ),
          'identity_applies_per_variant': 2,
          'full_V_route_applies_per_variant': 2,
          'group_applies_per_variant': 1,
          'planned_model_apply_count': 20 * (4 + len(groups)),
          'planned_compile_count': 1,
      },
      'analysis': {
          'primary_population': 'significant_effect',
          'per_case_necessity_loss': 'B_full_V - B_without_child',
          'within_parent_ranking': (
              'descending minimum of BRAF and SLC25A48 median necessity loss '
              'for shared parents; descending gene median for gene-ranked '
              'parents'
          ),
          'interpretation_limit': (
              'An 8-channel necessity result is still a subspace result and '
              'does not establish a motif or biochemical mechanism.'
          ),
      },
      'controls': {
          'reciprocal_transfer': True,
          'same_allele_self_controls': True,
          'identity_and_full_route_exact_repeat': True,
          'per_child_selected_channel_donor_exact': True,
          'per_child_withheld_channel_noop_exact': True,
          'non_V_routes_disabled': True,
          'neutral_variants_secondary_specificity': True,
          'cross_executable_full_route_comparison_is_diagnostic': True,
          'os_kernel_is_a_gate': False,
      },
      'follow_up': {
          'leading_children': (
              'split to individual channels, then test only-group sufficiency '
              'and V/upstream/downstream localization'
          ),
      },
      'parents': parents,
      'groups': groups,
      'cases': screen['cases'],
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
