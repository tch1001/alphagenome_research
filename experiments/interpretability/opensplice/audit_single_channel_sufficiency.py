#!/usr/bin/env python3
"""Independently audit single-channel spatial sufficiency from raw means."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_RAW = HERE / 'results' / 'single_channel_sufficiency_v1' / 'raw'
DEFAULT_ANALYSIS = (
    HERE / 'results' /
    'single_channel_sufficiency_v1_model_behavior_analysis' /
    'ANALYSIS.json'
)
DEFAULT_OUTPUT = DEFAULT_ANALYSIS.parent / 'INDEPENDENT_AUDIT.md'
GENES = ('BRAF', 'SLC25A48')
LOCATIONS = ('intended', 'upstream', 'downstream')


class AuditError(RuntimeError):
  """Raised when a raw control or independent aggregate differs."""


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
  full_paths = sorted((raw_root / 'full').glob('*.json'))
  condition_paths = sorted((raw_root / 'conditions').glob('*/*.json'))
  if len(full_paths) != 20 or len(condition_paths) != 180:
    raise AuditError('Expected 20 full and 180 condition records.')
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
    feature = condition['feature']
    rows.append({
        'key': f"{feature['stage']}_c{feature['channel_start_inclusive']:04d}",
        'selected_for_gene': feature['selected_for_gene'],
        'gene': case['gene'],
        'selection_class': case['selection_class'],
        'variant_id': case['variant_id'],
        'location': condition['location'],
        'bottleneck': value,
    })
  if maximum_raw_difference != 0:
    raise AuditError('Independent recovery differs from stored recovery.')
  shifted = [
      row['bottleneck'] for row in rows if row['location'] != 'intended'
  ]
  if len(shifted) != 120 or not all(value == 0 for value in shifted):
    raise AuditError('Shifted controls are not all zero.')

  summaries = {}
  keys = sorted({row['key'] for row in rows})
  for key in keys:
    summaries[key] = {}
    gene_selected = next(
        row['selected_for_gene'] for row in rows if row['key'] == key
    )
    for gene in GENES:
      summaries[key][gene] = {}
      for selection_class, label in (
          ('significant_effect', 'effect'), ('neutral_control', 'neutral')
      ):
        population = {}
        for row in rows:
          if (
              row['key'] == key and row['gene'] == gene
              and row['selection_class'] == selection_class
          ):
            population.setdefault(row['variant_id'], {})[
                row['location']
            ] = row['bottleneck']
        if len(population) != (6 if label == 'effect' else 4):
          raise AuditError('Population grouping changed.')
        medians = {
            location: statistics.median(
                value[location] for value in population.values()
            ) for location in LOCATIONS
        }
        contrasts = [
            value['intended']
            - max(value['upstream'], value['downstream'])
            for value in population.values()
        ]
        summaries[key][gene][label] = {
            'medians': medians,
            'contrast': statistics.median(contrasts),
            'positive': sum(value > 0 for value in contrasts),
        }
    effect = summaries[key][gene_selected]['effect']
    neutral = summaries[key][gene_selected]['neutral']
    summaries[key]['passes'] = (
        effect['medians']['intended'] > 0
        and effect['contrast'] > 0
        and effect['positive'] >= 4
        and effect['medians']['intended'] > neutral['medians']['intended']
    )

  analysis = _load(analysis_path)
  differences = []
  for key, value in summaries.items():
    reported = analysis['per_channel'][key]
    if value['passes'] is not reported['passes_selected_gene_sufficiency_rule']:
      raise AuditError('Independent pass result differs.')
    for gene in GENES:
      for label in ('effect', 'neutral'):
        observed = value[gene][label]
        expected = reported['per_gene'][gene][label]
        for location in LOCATIONS:
          differences.append(abs(
              observed['medians'][location]
              - expected['median_bidirectional_bottleneck_by_location'][
                  location
              ]
          ))
        differences.append(abs(
            observed['contrast']
            - expected['median_per_variant_spatial_contrast']
        ))
        if observed['positive'] != expected['positive_spatial_contrast_count']:
          raise AuditError('Independent positive count differs.')
  maximum_analysis_difference = max(differences)
  if maximum_analysis_difference != 0:
    raise AuditError('Independent aggregates differ from analysis.')
  passing = [key for key in keys if summaries[key]['passes']]
  if passing != analysis['channels_passing_frozen_sufficiency_rule']:
    raise AuditError('Independent passing set differs.')
  return {
      'raw_full_count': len(full_paths),
      'raw_condition_count': len(condition_paths),
      'raw_active_tree_sha256': _digest(
          raw_root, [*full_paths, *condition_paths]
      ),
      'maximum_raw_recovery_difference': maximum_raw_difference,
      'maximum_analysis_aggregation_difference': maximum_analysis_difference,
      'all_120_shifted_values_exactly_zero': True,
      'passing_channels': passing,
  }


def markdown(result: dict[str, Any]) -> str:
  lines = [
      '# Independent single-channel sufficiency audit',
      '',
      'This audit reread every raw target mean, used an algebraically',
      'equivalent sign-reversed recovery formula, and independently rebuilt',
      'all spatial medians, contrasts and pass decisions. It did not call the',
      'main analyzer.',
      '',
      f"- Full-route records: {result['raw_full_count']}",
      f"- Condition records: {result['raw_condition_count']}",
      f"- Raw active tree SHA-256: `{result['raw_active_tree_sha256']}`",
      '- Maximum raw recovery difference: '
      f"`{result['maximum_raw_recovery_difference']:.1f}`",
      '- Maximum aggregation difference from `ANALYSIS.json`: '
      f"`{result['maximum_analysis_aggregation_difference']:.1f}`",
      '- All 120 shifted values exactly zero: true',
      '',
      'Independently passing coordinates:',
      '',
  ]
  lines.extend(f'- `{value}`' for value in result['passing_channels'])
  lines.extend([
      '',
      'The independent arithmetic, aggregation and decisions match exactly.',
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
