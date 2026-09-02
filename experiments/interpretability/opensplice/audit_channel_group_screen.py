#!/usr/bin/env python3
"""Independently audit channel necessity rankings from raw target means."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_RAW = HERE / 'results' / 'channel_group_screen_v1' / 'raw'
DEFAULT_ANALYSIS = (
    HERE / 'results' / 'channel_group_screen_v1_model_behavior_analysis'
    / 'ANALYSIS.json'
)
DEFAULT_OUTPUT = (
    HERE / 'results' / 'channel_group_screen_v1_model_behavior_analysis'
    / 'INDEPENDENT_AUDIT.md'
)
GENES = ('BRAF', 'SLC25A48')


class AuditError(RuntimeError):
  """Raised when a raw record or independent comparison fails."""


def _load(path: Path) -> dict[str, Any]:
  try:
    value = json.loads(path.read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError) as error:
    raise AuditError(f'Cannot read {path}.') from error
  if not isinstance(value, dict):
    raise AuditError(f'JSON root is not an object: {path}.')
  return value


def _alternative_bottleneck(means: list[float]) -> float:
  """Use sign-reversed denominators to independently recover B."""
  if len(means) != 6 or not all(math.isfinite(value) for value in means):
    raise AuditError('Raw target means are malformed.')
  ref, alt, ref_alt, alt_alt, alt_ref, ref_ref = means
  alt_minus_ref = alt - ref
  if alt_minus_ref == 0:
    raise AuditError('Zero baseline allele effect.')
  ref_into_alt = (alt_alt - ref_alt) / alt_minus_ref
  alt_into_ref = (ref_ref - alt_ref) / (ref - alt)
  return min(ref_into_alt, alt_into_ref)


def _tree_digest(root: Path, paths: list[Path]) -> str:
  digest = hashlib.sha256()
  for path in sorted(paths):
    digest.update(path.relative_to(root).as_posix().encode('utf-8'))
    digest.update(b'\0')
    digest.update(path.read_bytes())
    digest.update(b'\0')
  return digest.hexdigest()


def audit(raw_root: Path, analysis_path: Path) -> dict[str, Any]:
  """Reread raw means and reproduce rankings without the main analyzer."""
  full_paths = sorted((raw_root / 'full').glob('*.json'))
  group_paths = sorted((raw_root / 'groups').glob('*/*.json'))
  if len(full_paths) != 20 or len(group_paths) != 3440:
    raise AuditError('Expected 20 full and 3440 group records.')
  full = {}
  maximum_raw_recovery_difference = 0.0
  for path in full_paths:
    record = _load(path)
    checks = record.get('checks', {})
    if (
        record.get('status') != 'complete'
        or checks.get('passed') is not True
        or checks.get('repeat_checked') is not True
        or checks.get('target_repeat_exact') is not True
        or checks.get('trace_repeat_exact') is not True
        or checks.get('target_readout') != checks.get('repeat_target_readout')
    ):
      raise AuditError(f'Full-route control failed: {path}.')
    means = list(map(float, checks['target_readout']['means']))
    bottleneck = _alternative_bottleneck(means)
    difference = abs(
        bottleneck - checks['recovery']['bidirectional_bottleneck']
    )
    maximum_raw_recovery_difference = max(
        maximum_raw_recovery_difference, difference
    )
    case = record['configuration']['case']
    full[case['variant_id']] = bottleneck
  if len(full) != 20:
    raise AuditError('Full-route variants are not unique.')

  rows = []
  for path in group_paths:
    record = _load(path)
    checks = record.get('checks', {})
    if (
        record.get('status') != 'complete'
        or checks.get('passed') is not True
        or checks.get('repeat_checked') is not False
        or checks.get('target_repeat_exact') is not None
        or checks.get('trace_repeat_exact') is not None
        or checks.get('repeat_target_readout') is not None
    ):
      raise AuditError(f'Group control failed: {path}.')
    means = list(map(float, checks['target_readout']['means']))
    bottleneck = _alternative_bottleneck(means)
    difference = abs(
        bottleneck - checks['recovery']['bidirectional_bottleneck']
    )
    maximum_raw_recovery_difference = max(
        maximum_raw_recovery_difference, difference
    )
    configuration = record['configuration']
    case = configuration['case']
    group = configuration['group']
    rows.append({
        'group_id': group['group_id'],
        'gene': case['gene'],
        'selection_class': case['selection_class'],
        'variant_id': case['variant_id'],
        'necessity_loss': full[case['variant_id']] - bottleneck,
    })
  if maximum_raw_recovery_difference != 0:
    raise AuditError('Alternative raw recovery differs from stored recovery.')

  group_ids = sorted({row['group_id'] for row in rows})
  if len(group_ids) != 172:
    raise AuditError('Expected 172 unique channel groups.')
  summaries = {}
  for group_id in group_ids:
    summaries[group_id] = {}
    for gene in GENES:
      for selection_class, label in (
          ('significant_effect', 'effect'), ('neutral_control', 'neutral')
      ):
        values = [
            row['necessity_loss'] for row in rows
            if row['group_id'] == group_id and row['gene'] == gene
            and row['selection_class'] == selection_class
        ]
        expected = 6 if label == 'effect' else 4
        if len(values) != expected:
          raise AuditError('Gene/class grouping differs from frozen design.')
        summaries[group_id][f'{gene}_{label}_median'] = statistics.median(
            values
        )
    summaries[group_id]['maximin'] = min(
        summaries[group_id][f'{gene}_effect_median'] for gene in GENES
    )
  ranking = sorted(group_ids, key=lambda group_id: (
      -summaries[group_id]['maximin'], group_id
  ))
  eligible = [
      group_id for group_id in ranking
      if summaries[group_id]['maximin'] > 0
      and all(
          summaries[group_id][f'{gene}_effect_median']
          > summaries[group_id][f'{gene}_neutral_median']
          for gene in GENES
      )
  ]
  top_gene = {
      gene: sorted(group_ids, key=lambda group_id, gene=gene: (
          -summaries[group_id][f'{gene}_effect_median'], group_id
      ))[0]
      for gene in GENES
  }

  analysis = _load(analysis_path)
  reported_by_group = {
      row['group_id']: row for row in analysis['group_summaries']
  }
  differences = []
  for group_id in group_ids:
    reported = reported_by_group[group_id]
    differences.append(abs(
        summaries[group_id]['maximin']
        - reported['cross_gene_maximin_effect_median_loss']
    ))
    for gene in GENES:
      for label in ('effect', 'neutral'):
        differences.append(abs(
            summaries[group_id][f'{gene}_{label}_median']
            - reported['per_gene'][gene][label]['median_necessity_loss']
        ))
  maximum_analysis_difference = max(differences)
  if maximum_analysis_difference != 0:
    raise AuditError('Independent group aggregates differ from analysis.')
  if ranking != analysis['rankings']['cross_gene_all_group_ids']:
    raise AuditError('Independent cross-gene ranking differs from analysis.')
  reported_refinement = analysis['recommended_refinement']
  if eligible[:3] != [
      row['ranked_group_id']
      for row in reported_refinement['shared_specificity_qualified_parents']
  ]:
    raise AuditError('Independent shared refinement set differs.')
  if top_gene != {
      gene: reported_refinement['top_within_gene_parent'][gene][
          'ranked_group_id'
      ] for gene in GENES
  }:
    raise AuditError('Independent top within-gene parents differ.')

  return {
      'raw_full_count': len(full_paths),
      'raw_group_count': len(group_paths),
      'raw_active_tree_sha256': _tree_digest(
          raw_root, [*full_paths, *group_paths]
      ),
      'maximum_raw_recovery_difference': maximum_raw_recovery_difference,
      'maximum_analysis_aggregation_difference': maximum_analysis_difference,
      'cross_gene_ranking_matches_exactly': True,
      'top_cross_gene_groups': ranking[:3],
      'top_within_gene_group': top_gene,
  }


def markdown(result: dict[str, Any]) -> str:
  lines = [
      '# Independent channel-screen audit',
      '',
      'This audit reread the raw full-route and group target means, recomputed',
      'reciprocal recovery with an algebraically equivalent sign-reversed',
      'formula, and independently rebuilt all gene medians and rankings. It',
      'did not call the main analyzer.',
      '',
      f"- Full-route records: {result['raw_full_count']}",
      f"- Group records: {result['raw_group_count']}",
      f"- Raw active-result tree SHA-256: `{result['raw_active_tree_sha256']}`",
      '- Maximum raw recovery difference: '
      f"`{result['maximum_raw_recovery_difference']:.1f}`",
      '- Maximum aggregate difference from `ANALYSIS.json`: '
      f"`{result['maximum_analysis_aggregation_difference']:.1f}`",
      '- Full 172-group ranking matches exactly: true',
      '',
      'Independently selected shared refinement parents:',
      '',
  ]
  lines.extend(
      f'- `{group_id}`' for group_id in result['top_cross_gene_groups']
  )
  lines.extend([
      '',
      'Top within-gene parents:',
      '',
      f"- BRAF: `{result['top_within_gene_group']['BRAF']}`",
      f"- SLC25A48: `{result['top_within_gene_group']['SLC25A48']}`",
      '',
      'The independent arithmetic, aggregation and deterministic selection',
      'match exactly.',
  ])
  return '\n'.join(lines) + '\n'


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--raw', type=Path, default=DEFAULT_RAW)
  parser.add_argument('--analysis', type=Path, default=DEFAULT_ANALYSIS)
  parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if args.output.exists() or args.output.is_symlink():
    raise AuditError(f'Output already exists: {args.output}.')
  result = audit(args.raw, args.analysis)
  args.output.write_text(markdown(result), encoding='utf-8')
  print(args.output)


if __name__ == '__main__':
  main()
