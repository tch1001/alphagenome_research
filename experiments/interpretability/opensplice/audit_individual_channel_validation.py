#!/usr/bin/env python3
"""Independently audit individual necessity and subspace sufficiency."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_RAW = HERE / 'results' / 'individual_channel_validation_v1' / 'raw'
DEFAULT_ANALYSIS = (
    HERE / 'results' /
    'individual_channel_validation_v1_model_behavior_analysis' /
    'ANALYSIS.json'
)
DEFAULT_OUTPUT = DEFAULT_ANALYSIS.parent / 'INDEPENDENT_AUDIT.md'
GENES = ('BRAF', 'SLC25A48')
LOCATIONS = ('intended', 'upstream', 'downstream')


class AuditError(RuntimeError):
  """Raised when raw arithmetic or the reported result differs."""


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
  """Recompute recovery, aggregation, advancement and localization."""
  full_paths = sorted((raw_root / 'full').glob('*.json'))
  condition_paths = sorted((raw_root / 'conditions').glob('*/*.json'))
  if len(full_paths) != 20 or len(condition_paths) != 880:
    raise AuditError('Expected 20 full and 880 condition records.')
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
  for path in condition_paths:
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
      raise AuditError(f'Condition control failed: {path}.')
    value = _bottleneck(list(map(float, checks['target_readout']['means'])))
    maximum_raw_difference = max(
        maximum_raw_difference,
        abs(value - checks['recovery']['bidirectional_bottleneck']),
    )
    configuration = record['configuration']
    case = configuration['case']
    condition = configuration['condition']
    rows.append({
        'condition': condition,
        'gene': case['gene'],
        'selection_class': case['selection_class'],
        'variant_id': case['variant_id'],
        'bottleneck': value,
        'necessity_loss': full[case['variant_id']] - value,
    })
  if maximum_raw_difference != 0:
    raise AuditError('Independent recovery differs from stored recovery.')

  necessity = [
      row for row in rows
      if row['condition']['kind'] == 'individual_channel_necessity'
  ]
  condition_ids = sorted({
      row['condition']['condition_id'] for row in necessity
  })
  if len(condition_ids) != 32:
    raise AuditError('Expected 32 individual conditions.')
  summaries = {}
  for condition_id in condition_ids:
    condition = next(
        row['condition'] for row in necessity
        if row['condition']['condition_id'] == condition_id
    )
    summaries[condition_id] = {'condition': condition}
    for gene in GENES:
      for selection_class, label in (
          ('significant_effect', 'effect'), ('neutral_control', 'neutral')
      ):
        values = [
            row['necessity_loss'] for row in necessity
            if row['condition']['condition_id'] == condition_id
            and row['gene'] == gene
            and row['selection_class'] == selection_class
        ]
        if len(values) != (6 if label == 'effect' else 4):
          raise AuditError('Necessity population grouping changed.')
        summaries[condition_id][f'{gene}_{label}_median'] = (
            statistics.median(values)
        )
        summaries[condition_id][f'{gene}_{label}_positive_count'] = sum(
            value > 0 for value in values
        )
  advancing = []
  rankings = {}
  child_ids = sorted({
      value['condition']['feature']['parent_child_id']
      for value in summaries.values()
  })
  for child_id in child_ids:
    values = [
        (condition_id, value) for condition_id, value in summaries.items()
        if value['condition']['feature']['parent_child_id'] == child_id
    ]
    gene = values[0][1]['condition']['feature']['selected_for_gene']
    values.sort(key=lambda item, gene=gene: (
        -item[1][f'{gene}_effect_median'], item[0]
    ))
    rankings[child_id] = [item[0] for item in values]
    for condition_id, value in values:
      if (
          value[f'{gene}_effect_median'] > 0
          and value[f'{gene}_effect_median']
          > value[f'{gene}_neutral_median']
          and value[f'{gene}_effect_positive_count'] >= 4
      ):
        advancing.append(condition_id)
  advancing.sort()

  sufficiency = [
      row for row in rows
      if row['condition']['kind'] == 'eight_channel_only_sufficiency'
  ]
  shifted = [
      row['bottleneck'] for row in sufficiency
      if row['condition']['location'] != 'intended'
  ]
  if len(shifted) != 160 or not all(value == 0 for value in shifted):
    raise AuditError('Shifted sufficiency controls are not all zero.')
  sufficiency_medians = {}
  for child_id in child_ids:
    sufficiency_medians[child_id] = {}
    for gene in GENES:
      for selection_class, label in (
          ('significant_effect', 'effect'), ('neutral_control', 'neutral')
      ):
        per_variant = {}
        for row in sufficiency:
          feature = row['condition']['feature']
          if (
              feature['parent_child_id'] == child_id
              and row['gene'] == gene
              and row['selection_class'] == selection_class
          ):
            per_variant.setdefault(row['variant_id'], {})[
                row['condition']['location']
            ] = row['bottleneck']
        if len(per_variant) != (6 if label == 'effect' else 4):
          raise AuditError('Sufficiency population grouping changed.')
        sufficiency_medians[child_id][f'{gene}_{label}'] = {
            'locations': {
                location: statistics.median(
                    values[location] for values in per_variant.values()
                ) for location in LOCATIONS
            },
            'contrast': statistics.median(
                values['intended']
                - max(values['upstream'], values['downstream'])
                for values in per_variant.values()
            ),
        }

  analysis = _load(analysis_path)
  reported_necessity = {}
  reported_rankings = {}
  for child_id, value in analysis['individual_necessity'][
      'ranked_by_selected_child'
  ].items():
    reported_rankings[child_id] = value['ranked_condition_ids']
    for row in value['summaries']:
      reported_necessity[row['condition_id']] = row
  differences = []
  for condition_id, value in summaries.items():
    reported = reported_necessity[condition_id]
    for gene in GENES:
      for label in ('effect', 'neutral'):
        differences.append(abs(
            value[f'{gene}_{label}_median']
            - reported['per_gene'][gene][label]['median']
        ))
        if (
            value[f'{gene}_{label}_positive_count']
            != reported['per_gene'][gene][label]['positive_count']
        ):
          raise AuditError('Independent positive count differs.')
  reported_sufficiency = analysis['eight_channel_sufficiency']['per_child']
  for child_id, value in sufficiency_medians.items():
    for gene in GENES:
      for label in ('effect', 'neutral'):
        reported = reported_sufficiency[child_id]['per_gene'][gene][label]
        for location in LOCATIONS:
          differences.append(abs(
              value[f'{gene}_{label}']['locations'][location]
              - reported['median_bidirectional_bottleneck_by_location'][
                  location
              ]
          ))
        differences.append(abs(
            value[f'{gene}_{label}']['contrast']
            - reported['median_per_variant_spatial_contrast']
        ))
  maximum_analysis_difference = max(differences)
  if maximum_analysis_difference != 0:
    raise AuditError('Independent medians differ from analysis.')
  if rankings != reported_rankings:
    raise AuditError('Independent within-child rankings differ.')
  reported_advancing = sorted(
      row['condition_id']
      for row in analysis['individual_necessity']['advancing_channels']
  )
  if advancing != reported_advancing:
    raise AuditError('Independent advancing-channel set differs.')
  return {
      'raw_full_count': len(full_paths),
      'raw_condition_count': len(condition_paths),
      'raw_active_tree_sha256': _digest(
          raw_root, [*full_paths, *condition_paths]
      ),
      'maximum_raw_recovery_difference': maximum_raw_difference,
      'maximum_analysis_aggregation_difference': maximum_analysis_difference,
      'all_rankings_match_exactly': True,
      'all_160_shifted_sufficiency_controls_exactly_zero': True,
      'advancing_condition_ids': advancing,
  }


def markdown(result: dict[str, Any]) -> str:
  lines = [
      '# Independent individual-channel validation audit',
      '',
      'This audit reread all raw active target means, used an algebraically',
      'equivalent sign-reversed recovery formula, and independently rebuilt',
      'the necessity rankings, advance rule, sufficiency medians and spatial',
      'contrasts. It did not call the main analyzer.',
      '',
      f"- Full-route records: {result['raw_full_count']}",
      f"- Condition records: {result['raw_condition_count']}",
      f"- Raw active tree SHA-256: `{result['raw_active_tree_sha256']}`",
      '- Maximum raw recovery difference: '
      f"`{result['maximum_raw_recovery_difference']:.1f}`",
      '- Maximum aggregation difference from `ANALYSIS.json`: '
      f"`{result['maximum_analysis_aggregation_difference']:.1f}`",
      '- All rankings match exactly: true',
      '- All 160 shifted sufficiency controls exactly zero: true',
      '',
      'Independently advancing channels:',
      '',
  ]
  lines.extend(
      f'- `{condition_id}`'
      for condition_id in result['advancing_condition_ids']
  )
  lines.extend([
      '',
      'The independent arithmetic, aggregation and selection match exactly.',
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
