#!/usr/bin/env python3
"""Independently audit 8-channel refinement rankings from raw means."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_RAW = HERE / 'results' / 'channel_refinement_v1' / 'raw'
DEFAULT_ANALYSIS = (
    HERE / 'results' / 'channel_refinement_v1_model_behavior_analysis'
    / 'ANALYSIS.json'
)
DEFAULT_OUTPUT = (
    HERE / 'results' / 'channel_refinement_v1_model_behavior_analysis'
    / 'INDEPENDENT_AUDIT.md'
)
GENES = ('BRAF', 'SLC25A48')


class AuditError(RuntimeError):
  """Raised when raw arithmetic or a reported refinement result differs."""


def _load(path: Path) -> dict[str, Any]:
  try:
    value = json.loads(path.read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError) as error:
    raise AuditError(f'Cannot read {path}.') from error
  if not isinstance(value, dict):
    raise AuditError(f'JSON root is not an object: {path}.')
  return value


def _bottleneck(means: list[float]) -> float:
  if len(means) != 6 or not all(math.isfinite(value) for value in means):
    raise AuditError('Raw target means are malformed.')
  ref, alt, ref_alt, alt_alt, alt_ref, ref_ref = means
  if ref == alt:
    raise AuditError('Zero baseline allele effect.')
  return min(
      (alt_alt - ref_alt) / (alt - ref),
      (ref_ref - alt_ref) / (ref - alt),
  )


def _digest(root: Path, paths: list[Path]) -> str:
  digest = hashlib.sha256()
  for path in sorted(paths):
    digest.update(path.relative_to(root).as_posix().encode('utf-8'))
    digest.update(b'\0')
    digest.update(path.read_bytes())
    digest.update(b'\0')
  return digest.hexdigest()


def audit(raw_root: Path, analysis_path: Path) -> dict[str, Any]:
  """Independently reproduce every median, ranking and selected child."""
  full_paths = sorted((raw_root / 'full').glob('*.json'))
  child_paths = sorted((raw_root / 'groups').glob('*/*.json'))
  if len(full_paths) != 20 or len(child_paths) != 400:
    raise AuditError('Expected 20 full and 400 child records.')
  full = {}
  maximum_raw_difference = 0.0
  for path in full_paths:
    record = _load(path)
    checks = record.get('checks', {})
    if (
        record.get('status') != 'complete'
        or checks.get('passed') is not True
        or checks.get('target_repeat_exact') is not True
        or checks.get('trace_repeat_exact') is not True
        or checks.get('target_readout') != checks.get('repeat_target_readout')
    ):
      raise AuditError(f'Full-route control failed: {path}.')
    value = _bottleneck(list(map(float, checks['target_readout']['means'])))
    maximum_raw_difference = max(
        maximum_raw_difference,
        abs(value - checks['recovery']['bidirectional_bottleneck']),
    )
    case = record['configuration']['case']
    full[case['variant_id']] = value

  rows = []
  for path in child_paths:
    record = _load(path)
    checks = record.get('checks', {})
    if (
        record.get('status') != 'complete'
        or checks.get('passed') is not True
        or checks.get('repeat_checked') is not False
        or checks.get('target_repeat_exact') is not None
        or checks.get('trace_repeat_exact') is not None
    ):
      raise AuditError(f'Child control failed: {path}.')
    value = _bottleneck(list(map(float, checks['target_readout']['means'])))
    maximum_raw_difference = max(
        maximum_raw_difference,
        abs(value - checks['recovery']['bidirectional_bottleneck']),
    )
    configuration = record['configuration']
    case = configuration['case']
    group = configuration['group']
    rows.append({
        'group_id': group['group_id'],
        'gene': case['gene'],
        'selection_class': case['selection_class'],
        'variant_id': case['variant_id'],
        'necessity_loss': full[case['variant_id']] - value,
    })
  if maximum_raw_difference != 0:
    raise AuditError('Independent recovery differs from stored recovery.')

  group_ids = sorted({row['group_id'] for row in rows})
  if len(group_ids) != 20:
    raise AuditError('Expected 20 unique child groups.')
  medians = {}
  for group_id in group_ids:
    medians[group_id] = {}
    for gene in GENES:
      for selection_class, label in (
          ('significant_effect', 'effect'), ('neutral_control', 'neutral')
      ):
        values = [
            row['necessity_loss'] for row in rows
            if row['group_id'] == group_id and row['gene'] == gene
            and row['selection_class'] == selection_class
        ]
        if len(values) != (6 if label == 'effect' else 4):
          raise AuditError('Gene/class grouping differs from frozen design.')
        medians[group_id][f'{gene}_{label}'] = statistics.median(values)
    medians[group_id]['maximin'] = min(
        medians[group_id][f'{gene}_effect'] for gene in GENES
    )

  cross_gene = sorted(group_ids, key=lambda group_id: (
      -medians[group_id]['maximin'], group_id
  ))
  by_gene = {
      gene: sorted(group_ids, key=lambda group_id, gene=gene: (
          -medians[group_id][f'{gene}_effect'], group_id
      )) for gene in GENES
  }
  selected = {}
  for gene in GENES:
    eligible = [
        group_id for group_id in by_gene[gene]
        if medians[group_id][f'{gene}_effect'] > 0
        and medians[group_id][f'{gene}_effect']
        > medians[group_id][f'{gene}_neutral']
    ]
    selected[gene] = eligible[:2]

  analysis = _load(analysis_path)
  reported = {row['group_id']: row for row in analysis['child_summaries']}
  differences = []
  for group_id in group_ids:
    for gene in GENES:
      for label in ('effect', 'neutral'):
        differences.append(abs(
            medians[group_id][f'{gene}_{label}']
            - reported[group_id]['per_gene'][gene][label][
                'median_necessity_loss'
            ]
        ))
  maximum_analysis_difference = max(differences)
  if maximum_analysis_difference != 0:
    raise AuditError('Independent medians differ from ANALYSIS.json.')
  if cross_gene != analysis['rankings']['cross_gene_all_child_ids']:
    raise AuditError('Independent cross-gene child ranking differs.')
  if by_gene != analysis['rankings']['all_child_ids_by_gene']:
    raise AuditError('Independent within-gene child ranking differs.')
  reported_selected = {
      gene: [row['group_id'] for row in analysis[
          'recommended_individual_channel_refinement'
      ]['per_gene'][gene]]
      for gene in GENES
  }
  if selected != reported_selected:
    raise AuditError('Independent individual-channel selection differs.')
  return {
      'raw_full_count': len(full_paths),
      'raw_child_count': len(child_paths),
      'raw_active_tree_sha256': _digest(
          raw_root, [*full_paths, *child_paths]
      ),
      'maximum_raw_recovery_difference': maximum_raw_difference,
      'maximum_analysis_median_difference': maximum_analysis_difference,
      'all_rankings_match_exactly': True,
      'selected_children': selected,
  }


def markdown(result: dict[str, Any]) -> str:
  return '\n'.join([
      '# Independent eight-channel refinement audit',
      '',
      'This audit reread all raw active target means, used a sign-reversed',
      'recovery formula, and independently rebuilt every median, cross-gene',
      'ranking, within-gene ranking and individual-channel candidate set. It',
      'did not call the main analyzer.',
      '',
      f"- Full-route records: {result['raw_full_count']}",
      f"- Child records: {result['raw_child_count']}",
      f"- Raw active tree SHA-256: `{result['raw_active_tree_sha256']}`",
      '- Maximum raw recovery difference: '
      f"`{result['maximum_raw_recovery_difference']:.1f}`",
      '- Maximum median difference from `ANALYSIS.json`: '
      f"`{result['maximum_analysis_median_difference']:.1f}`",
      '- All cross-gene and within-gene rankings match exactly: true',
      '',
      'Independently selected children:',
      '',
      '- BRAF: ' + ', '.join(
          f'`{value}`' for value in result['selected_children']['BRAF']
      ),
      '- SLC25A48: ' + ', '.join(
          f'`{value}`' for value in result['selected_children']['SLC25A48']
      ),
      '',
      'The independent arithmetic, aggregation and selection match exactly.',
  ]) + '\n'


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
