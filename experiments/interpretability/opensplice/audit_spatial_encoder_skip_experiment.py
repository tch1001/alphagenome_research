#!/usr/bin/env python3
"""Independently audit spatial recovery directly from raw target means."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_RAW = (
    HERE / 'results' / 'spatial_encoder_skip_v1' / 'raw' / 'conditions'
)
DEFAULT_ANALYSIS = (
    HERE / 'results' / 'spatial_encoder_skip_v1_model_behavior_analysis'
    / 'ANALYSIS.json'
)
DEFAULT_OUTPUT = (
    HERE / 'results' / 'spatial_encoder_skip_v1_model_behavior_analysis'
    / 'INDEPENDENT_AUDIT.md'
)
GENES = ('BRAF', 'SLC25A48')
SUPPORTS = ('V', 'A', 'D', 'S')
LOCATIONS = ('intended', 'upstream', 'downstream')


class AuditError(RuntimeError):
  """Raised when the raw result cube or independent comparison fails."""


def _load(path: Path) -> dict[str, Any]:
  try:
    value = json.loads(path.read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError) as error:
    raise AuditError(f'Cannot read {path}.') from error
  if not isinstance(value, dict):
    raise AuditError(f'JSON root is not an object: {path}.')
  return value


def _alternative_bottleneck(means: list[float]) -> float:
  """Uses sign-reversed denominators as an independent recovery formula."""
  if len(means) != 6 or not all(math.isfinite(value) for value in means):
    raise AuditError('Raw target means are malformed.')
  ref, alt, ref_alt, alt_alt, alt_ref, ref_ref = means
  alt_minus_ref = alt - ref
  if alt_minus_ref == 0:
    raise AuditError('Zero baseline allele effect.')
  ref_into_alt = (alt_alt - ref_alt) / alt_minus_ref
  alt_into_ref = (ref_ref - alt_ref) / (ref - alt)
  return min(ref_into_alt, alt_into_ref)


def audit(raw_root: Path, analysis_path: Path) -> dict[str, Any]:
  paths = sorted(raw_root.glob('*/*.json'))
  if len(paths) != 240:
    raise AuditError(f'Expected 240 raw conditions, observed {len(paths)}.')
  tree = hashlib.sha256()
  rows = []
  for path in paths:
    tree.update(path.relative_to(raw_root).as_posix().encode('utf-8'))
    tree.update(b'\0')
    payload = path.read_bytes()
    tree.update(payload)
    tree.update(b'\0')
    record = json.loads(payload)
    if record.get('status') != 'complete':
      raise AuditError(f'Incomplete raw condition: {path}.')
    configuration = record['configuration']
    checks = record['checks']
    if (
        checks.get('passed') is not True
        or checks.get('target_repeat_exact') is not True
        or checks.get('trace_repeat_exact') is not True
        or checks['target_readout'] != checks['repeat_target_readout']
    ):
      raise AuditError(f'Raw control failed: {path}.')
    means = list(map(float, checks['target_readout']['means']))
    bottleneck = _alternative_bottleneck(means)
    rows.append({
        'gene': configuration['case']['gene'],
        'selection_class': configuration['case']['selection_class'],
        'variant_id': configuration['case']['variant_id'],
        'support': configuration['condition']['support'],
        'location': configuration['condition']['location'],
        'bottleneck': bottleneck,
        'stored_bottleneck': checks['recovery']['bidirectional_bottleneck'],
    })
  maximum_raw_difference = max(
      abs(row['bottleneck'] - row['stored_bottleneck']) for row in rows
  )
  if maximum_raw_difference != 0:
    raise AuditError('Alternative raw recovery differs from stored recovery.')

  medians = {}
  contrasts = {}
  for support in SUPPORTS:
    medians[support] = {}
    contrasts[support] = {}
    for gene in GENES:
      selected = [
          row for row in rows
          if row['gene'] == gene
          and row['selection_class'] == 'significant_effect'
          and row['support'] == support
      ]
      grouped: dict[str, dict[str, float]] = {}
      for row in selected:
        grouped.setdefault(row['variant_id'], {})[row['location']] = row[
            'bottleneck'
        ]
      if len(grouped) != 6 or any(
          set(values) != set(LOCATIONS) for values in grouped.values()
      ):
        raise AuditError('Effect grouping differs from the frozen design.')
      medians[support][gene] = statistics.median(
          values['intended'] for values in grouped.values()
      )
      contrasts[support][gene] = statistics.median(
          values['intended']
          - max(values['upstream'], values['downstream'])
          for values in grouped.values()
      )

  analysis = _load(analysis_path)
  differences = []
  for support in SUPPORTS:
    for gene in GENES:
      reported = analysis['effect_variants_primary'][support]['per_gene'][gene]
      differences.extend((
          abs(
              medians[support][gene]
              - reported['median_bidirectional_bottleneck']['intended']
          ),
          abs(
              contrasts[support][gene]
              - reported['median_per_variant_spatial_contrast']
          ),
      ))
  maximum_analysis_difference = max(differences)
  if maximum_analysis_difference != 0:
    raise AuditError('Independent aggregation differs from ANALYSIS.json.')
  shifted = [
      row['bottleneck'] for row in rows if row['location'] != 'intended'
  ]
  return {
      'raw_condition_count': len(rows),
      'raw_condition_tree_sha256': tree.hexdigest(),
      'maximum_raw_recovery_difference': maximum_raw_difference,
      'maximum_analysis_aggregation_difference': maximum_analysis_difference,
      'all_shifted_controls_exactly_zero': all(value == 0 for value in shifted),
      'medians': medians,
      'contrasts': contrasts,
  }


def markdown(result: dict[str, Any]) -> str:
  lines = [
      '# Independent spatial-analysis audit',
      '',
      'This audit reread all 240 raw condition records and recomputed',
      'bidirectional recovery with an algebraically equivalent, sign-reversed',
      'formula. It did not call the main analyzer.',
      '',
      f"- Raw condition count: {result['raw_condition_count']}",
      f"- Raw condition tree SHA-256: `{result['raw_condition_tree_sha256']}`",
      '- Maximum raw recovery difference: '
      f"`{result['maximum_raw_recovery_difference']:.1f}`",
      '- Maximum aggregation difference from `ANALYSIS.json`: '
      f"`{result['maximum_analysis_aggregation_difference']:.1f}`",
      '- All shifted controls exactly zero: '
      f"`{str(result['all_shifted_controls_exactly_zero']).lower()}`",
      '',
      '| Support | BRAF median B | SLC25A48 median B |',
      '|---|---:|---:|',
  ]
  for support in SUPPORTS:
    lines.append(
        f"| {support} | {result['medians'][support]['BRAF']:.5f} | "
        f"{result['medians'][support]['SLC25A48']:.5f} |"
    )
  lines.extend([
      '',
      'The independent arithmetic and aggregation match exactly.',
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
  result = audit(args.raw, args.analysis)
  if args.output.exists() or args.output.is_symlink():
    raise AuditError(f'Output already exists: {args.output}.')
  args.output.write_text(markdown(result), encoding='utf-8')
  print(args.output)


if __name__ == '__main__':
  main()
