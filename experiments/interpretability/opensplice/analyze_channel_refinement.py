#!/usr/bin/env python3
"""Validate and analyze the completed 8-channel refinement experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import analyze_channel_group_screen as screen_analysis
# pylint: enable=wrong-import-position


DEFAULT_INPUT = HERE / 'results' / 'channel_refinement_v1'
DEFAULT_OUTPUT = (
    HERE / 'results' / 'channel_refinement_v1_model_behavior_analysis'
)
PLAN_PATH = HERE / 'channel_refinement_plan_v1.json'
GENES = screen_analysis.GENES
SHARED_PARENTS = (
    'E1_c0160_0191', 'E32_c0000_0031', 'E16_c0000_0031'
)


class AnalysisError(RuntimeError):
  """Raised when refinement raw data or its frozen bindings fail."""


def load_results(
    root: Path = DEFAULT_INPUT, plan_path: Path = PLAN_PATH
) -> dict[str, Any]:
  """Validate every identity, full-route and 8-channel raw record."""
  try:
    if not root.is_dir() or root.is_symlink():
      raise AnalysisError(f'Result root is absent or unsafe: {root}.')
    plan = screen_analysis._load(plan_path)  # pylint: disable=protected-access
    plan_sha256 = screen_analysis._sha256(  # pylint: disable=protected-access
        plan_path
    )
    if (
        plan.get('schema_version')
        != 'alphagenome-channel-refinement-plan-v1'
        or plan.get('scope', {}).get('confirmation_access') is not False
        or plan.get('design', {}).get('parent_group_count') != 5
        or plan.get('design', {}).get('group_count') != 20
    ):
      raise AnalysisError('Plan is not the frozen 8-channel refinement.')
    summary_path = root / 'SUMMARY.json'
    summary = screen_analysis._load(  # pylint: disable=protected-access
        summary_path
    )
    if (
        summary.get('status') != 'complete'
        or summary.get('variant_count') != 20
        or summary.get('parent_group_count') != 5
        or summary.get('group_count') != 20
        or summary.get('group_result_count') != 400
        or summary.get('model_apply_count_in_full_nonresume_run') != 480
        or summary.get('full_frozen_design_completed') is not True
        or summary.get('binding', {}).get('confirmation_access') is not False
        or summary.get('binding', {}).get('plan_sha256') != plan_sha256
    ):
      raise AnalysisError('Summary is not the complete frozen refinement.')
    closure = summary.get('prior_channel_screen_closure', {})
    if (
        closure.get('comparison_is_diagnostic_not_a_gate') is not True
        or closure.get('case_count') != 20
    ):
      raise AnalysisError('Cross-executable closure record is incomplete.')

    identity_paths = sorted((root / 'raw' / 'identity').glob('*.json'))
    full_paths = sorted((root / 'raw' / 'full').glob('*.json'))
    group_paths = sorted((root / 'raw' / 'groups').glob('*/*.json'))
    if (
        len(identity_paths) != 20 or len(full_paths) != 20
        or len(group_paths) != 400
    ):
      raise AnalysisError('Raw refinement lacks 20 + 20 + 400 files.')

    rows = []
    full_rows = []
    for planned in plan['cases']:
      stem = (
          f"{planned['order']:03d}_"
          f"{screen_analysis._slug(planned['variant_id'])}"  # pylint: disable=protected-access
      )
      identity_path = root / 'raw' / 'identity' / f'{stem}.json'
      full_path = root / 'raw' / 'full' / f'{stem}.json'
      identity = screen_analysis._load(  # pylint: disable=protected-access
          identity_path
      )
      full = screen_analysis._load(  # pylint: disable=protected-access
          full_path
      )
      screen_analysis._validate_identity(  # pylint: disable=protected-access
          identity, planned, plan_sha256
      )
      full_b = screen_analysis._validate_full(  # pylint: disable=protected-access
          full, planned, identity, plan_sha256
      )
      full_rows.append({
          'case': full['configuration']['case'],
          'bidirectional_bottleneck': full_b,
      })
      paths = sorted((root / 'raw' / 'groups' / stem).glob('*.json'))
      if len(paths) != 20:
        raise AnalysisError(f'{stem} does not have 20 child records.')
      for group, path in zip(plan['groups'], paths, strict=True):
        expected = f"{group['group_index']:03d}_{group['group_id']}.json"
        if path.name != expected:
          raise AnalysisError(f'Child file order changed in {stem}.')
        record = screen_analysis._load(  # pylint: disable=protected-access
            path
        )
        without_b = screen_analysis._validate_group(  # pylint: disable=protected-access
            record, planned, group, identity, full, plan_sha256
        )
        rows.append({
            'case': record['configuration']['case'],
            'group': group,
            'full_bidirectional_bottleneck': full_b,
            'without_group_bidirectional_bottleneck': without_b,
            'necessity_loss': full_b - without_b,
        })
    all_paths = [summary_path, *identity_paths, *full_paths, *group_paths]
    return {
        'plan': plan,
        'summary': summary,
        'rows': rows,
        'full_rows': full_rows,
        'source': {
            'result_root': str(root),
            'plan_path': str(plan_path),
            'plan_sha256': plan_sha256,
            'raw_identity_file_count': len(identity_paths),
            'raw_full_file_count': len(full_paths),
            'raw_group_file_count': len(group_paths),
            'raw_result_tree_sha256': (
                screen_analysis._tree_digest(  # pylint: disable=protected-access
                    root, all_paths
                )
            ),
        },
    }
  except screen_analysis.AnalysisError as error:
    raise AnalysisError(str(error)) from error


def _compact(row: Mapping[str, Any]) -> dict[str, Any]:
  return {
      'group_id': row['group_id'],
      'parent_group_id': row['parent_group_id'],
      'stage': row['stage'],
      'channel_start_inclusive': row['channel_start_inclusive'],
      'channel_end_exclusive': row['channel_end_exclusive'],
      'cross_gene_maximin_effect_median_loss': row[
          'cross_gene_maximin_effect_median_loss'
      ],
      'per_gene': row['per_gene'],
  }


def analyze(loaded: Mapping[str, Any]) -> dict[str, Any]:
  """Rank children and test whether shared parents share a dominant child."""
  summaries = screen_analysis.summarize_groups(
      loaded['rows'], loaded['plan']['groups']
  )
  shared_ranking = sorted(summaries, key=lambda row: (
      -row['cross_gene_maximin_effect_median_loss'], row['group_id']
  ))
  gene_rankings = {
      gene: sorted(summaries, key=lambda row, gene=gene: (
          -row['per_gene'][gene]['effect']['median_necessity_loss'],
          row['group_id'],
      )) for gene in GENES
  }
  parent_summaries = {}
  for parent in loaded['plan']['parents']:
    parent_id = parent['group_id']
    children = [
        row for row in summaries if row['parent_group_id'] == parent_id
    ]
    if len(children) != 4:
      raise AnalysisError(f'{parent_id} does not have four children.')
    ranked_by_gene = {
        gene: sorted(children, key=lambda row, gene=gene: (
            -row['per_gene'][gene]['effect']['median_necessity_loss'],
            row['group_id'],
        )) for gene in GENES
    }
    parent_summaries[parent_id] = {
        'stage': parent['stage'],
        'children': [_compact(row) for row in children],
        'top_child_by_gene': {
            gene: _compact(ranked_by_gene[gene][0]) for gene in GENES
        },
        'same_top_child_in_both_genes': (
            ranked_by_gene['BRAF'][0]['group_id']
            == ranked_by_gene['SLC25A48'][0]['group_id']
        ),
    }

  full_route = {}
  for gene in GENES:
    full_route[gene] = {}
    for selection_class in ('significant_effect', 'neutral_control'):
      values = [
          row['bidirectional_bottleneck'] for row in loaded['full_rows']
          if row['case']['gene'] == gene
          and row['case']['selection_class'] == selection_class
      ]
      full_route[gene][selection_class] = {
          'variant_count': len(values),
          'median_bidirectional_bottleneck': float(statistics.median(values)),
      }

  candidates = {}
  for gene in GENES:
    eligible = [
        row for row in gene_rankings[gene]
        if row['per_gene'][gene]['effect']['median_necessity_loss'] > 0
        and row['per_gene'][gene]['effect']['median_necessity_loss']
        > row['per_gene'][gene]['neutral']['median_necessity_loss']
    ]
    if len(eligible) < 2:
      raise AnalysisError(f'Fewer than two specific candidates for {gene}.')
    candidates[gene] = [_compact(row) for row in eligible[:2]]
  selected_ids = []
  for gene in GENES:
    for row in candidates[gene]:
      if row['group_id'] not in selected_ids:
        selected_ids.append(row['group_id'])

  shared_parent_divergence = {
      parent_id: parent_summaries[parent_id]
      for parent_id in SHARED_PARENTS
  }
  return {
      'schema_version': 'alphagenome-channel-refinement-analysis-v1',
      'source': loaded['source'],
      'scope': {
          'development_only': True,
          'confirmation_access': False,
          'variant_count': 20,
          'effect_count': 12,
          'neutral_count': 8,
          'parent_group_count': 5,
          'child_group_count': 20,
          'group_result_count': 400,
          'model_apply_count': 480,
      },
      'control_summary': {
          'all_runtime_controls_passed': True,
          'identity_and_full_route_repeats_bit_exact': True,
          'child_calls_are_single_shot': True,
          'prior_executable_comparison_is_diagnostic_not_a_gate': True,
          'prior_executable_maximum_absolute_target_mean_difference': (
              loaded['summary']['prior_channel_screen_closure'][
                  'maximum_absolute_target_mean_difference'
              ]
          ),
      },
      'full_V_route': full_route,
      'child_summaries': summaries,
      'rankings': {
          'cross_gene_all_child_ids': [
              row['group_id'] for row in shared_ranking
          ],
          'top_10_cross_gene': [
              _compact(row) for row in shared_ranking[:10]
          ],
          'all_child_ids_by_gene': {
              gene: [row['group_id'] for row in gene_rankings[gene]]
              for gene in GENES
          },
          'top_10_by_gene': {
              gene: [_compact(row) for row in gene_rankings[gene][:10]]
              for gene in GENES
          },
      },
      'parent_summaries': parent_summaries,
      'shared_parent_divergence': {
          'parent_count': len(shared_parent_divergence),
          'different_top_child_in_both_gene_count': sum(
              not value['same_top_child_in_both_genes']
              for value in shared_parent_divergence.values()
          ),
          'per_parent': shared_parent_divergence,
      },
      'recommended_individual_channel_refinement': {
          'selection_rule': (
              'top two children per gene with positive effect median '
              'necessity exceeding the gene-matched neutral median'
          ),
          'per_gene': candidates,
          'deduplicated_child_ids': selected_ids,
          'individual_channel_count': 8 * len(selected_ids),
      },
      'interpretation': {
          'main_result': (
              'All three parent blocks selected as cross-gene candidates '
              'have different dominant 8-channel children in BRAF and '
              'SLC25A48. Parent-level sharing therefore reflects adjacent '
              'gene-specific subspaces more than one dominant common child.'
          ),
          'braf_program': (
              'BRAF preferentially uses channels 0-7 at E32 and E16.'
          ),
          'slc25a48_program': (
              'SLC25A48 preferentially uses channels 168-175 at E2 and E1, '
              'strengthening the persistent multiscale-band hypothesis.'
          ),
          'shared_residual': (
              'E1 channels 160-167 and 176-183 retain small positive median '
              'necessity in both genes, but their maximin losses are about '
              '0.001 and should not be overstated.'
          ),
          'next_experiment': (
              'Resolve the four specific 8-channel children to individual '
              'channels while adding only-child sufficiency and shifted '
              'spatial controls before sequence-feature attribution.'
          ),
      },
  }


def _fmt(value: float) -> str:
  return f'{value:.5f}'


def result_markdown(result: Mapping[str, Any]) -> str:
  lines = [
      '# Eight-channel refinement model-behavior result',
      '',
      'The refinement separates the coarse cross-gene candidates into mostly',
      'adjacent, gene-specific subspaces. In all three 32-channel parents',
      'selected as shared candidates, BRAF and SLC25A48 have different',
      'dominant 8-channel children.',
      '',
      '## Dominant child within each shared parent',
      '',
      '| Parent | BRAF top child (median loss) | SLC25A48 top child '
      '(median loss) | Same child? |',
      '|---|---:|---:|:---:|',
  ]
  for parent_id in SHARED_PARENTS:
    parent = result['parent_summaries'][parent_id]
    braf = parent['top_child_by_gene']['BRAF']
    slc = parent['top_child_by_gene']['SLC25A48']
    braf_loss = braf['per_gene']['BRAF']['effect']['median_necessity_loss']
    slc_loss = slc['per_gene']['SLC25A48']['effect'][
        'median_necessity_loss'
    ]
    lines.append(
        f"| `{parent_id}` | `{braf['group_id']}` "
        f'({_fmt(braf_loss)}) | '
        f"`{slc['group_id']}` "
        f'({_fmt(slc_loss)}) | '
        f"{'yes' if parent['same_top_child_in_both_genes'] else 'no'} |"
    )
  lines.extend([
      '',
      'The repeated offset is informative. BRAF favors channels 0-7 at both',
      'E32 and E16. SLC25A48 favors 16-23 in those same parents, and its two',
      'strongest children overall are channels 168-175 at E2 and E1. Thus the',
      'persistent 160-191 SLC25A48 band from the coarse screen narrows to the',
      'same eight-channel slice at two resolutions.',
      '',
      '## Locked individual-channel candidates',
      '',
      'Using the top two positive effect-over-neutral children per gene:',
      '',
  ])
  for gene in GENES:
    values = result['recommended_individual_channel_refinement']['per_gene'][
        gene
    ]
    lines.append(
        f"- {gene}: " + ', '.join(f"`{row['group_id']}`" for row in values)
    )
  lines.extend([
      '',
      '## Boundaries',
      '',
      'All 480 planned applies completed and every causal runtime control',
      'passed. Identity and full-route repeats were bit-exact; child calls',
      'were intentionally single-shot. Confirmation data remained sealed.',
      '',
      'These are model subspaces, not biological factors. The maximin shared',
      'losses at 8-channel resolution are small, losses are nonadditive, and',
      'there are six effect variants per gene. The result argues for parallel',
      'exon-specific programs rather than a universal splice channel.',
      '',
      'Next, test all 32 individual channels within the four locked children,',
      'together with only-child sufficiency and shifted-position controls.',
      'Only channels surviving those tests should be mapped to activating',
      'sequences, motifs or candidate splicing factors.',
  ])
  return '\n'.join(lines) + '\n'


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
  parser.add_argument('--plan', type=Path, default=PLAN_PATH)
  parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if args.output_dir.exists() or args.output_dir.is_symlink():
    raise AnalysisError(f'Output already exists: {args.output_dir}.')
  result = analyze(load_results(args.input, args.plan))
  args.output_dir.mkdir(parents=True)
  (args.output_dir / 'ANALYSIS.json').write_text(
      json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n',
      encoding='utf-8',
  )
  (args.output_dir / 'RESULT.md').write_text(
      result_markdown(result), encoding='utf-8'
  )
  print(args.output_dir)


if __name__ == '__main__':
  main()
