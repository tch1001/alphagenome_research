#!/usr/bin/env python3
"""Independently audit key encoder feature-decomposition claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_RAW = HERE / 'results' / 'encoder_feature_decomposition_v1'
DEFAULT_ANALYSIS = (
    HERE / 'results'
    / 'encoder_feature_decomposition_v1_model_behavior_analysis'
    / 'ANALYSIS.json'
)
DEFAULT_PLAN = HERE / 'encoder_feature_decomposition_plan_v1.json'
DEFAULT_OUTPUT = DEFAULT_ANALYSIS.parent / 'INDEPENDENT_AUDIT.md'


class AuditError(RuntimeError):
  """Raised when independent arithmetic disagrees with the analysis."""


def _load(path: Path) -> dict[str, Any]:
  try:
    return json.loads(path.read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError) as error:
    raise AuditError(f'Cannot load {path}.') from error


def _vector(record, planned, stage, component):
  case = record['configuration']['case']
  resolution = (1, 2)[stage]
  positions = record['configuration']['positions'][stage]
  roles = (
      case['position_1based'], case['exon_start_1based'],
      case['exon_end_1based'],
  )
  indices = [
      positions.index(
          (position - 1 - planned['interval_start_0based']) // resolution
      )
      for position in roles
  ]
  values = np.asarray(record['components'][component], dtype=float)[stage]
  delta = values[1, :, 1] - values[0, :, 1]
  return delta[indices]


def audit(
    raw_root: Path = DEFAULT_RAW,
    analysis_path: Path = DEFAULT_ANALYSIS,
    plan_path: Path = DEFAULT_PLAN,
) -> dict[str, Any]:
  analysis = _load(analysis_path)
  plan = _load(plan_path)
  weights = _load(raw_root / 'WEIGHTS.json')
  paths = sorted((raw_root / 'raw').glob('*.json'))
  records = [_load(path) for path in paths]
  if len(records) != 20 or any(
      row.get('checks', {}).get('passed') is not True for row in records
  ):
    raise AuditError('Raw result is incomplete or has failed controls.')
  planned = {row['variant_id']: row for row in plan['cases']}
  effects = [
      row for row in records
      if row['configuration']['case']['gene'] == 'SLC25A48'
      and row['configuration']['case']['selection_class']
      == 'significant_effect'
  ]
  neutrals = [
      row for row in records
      if row['configuration']['case']['gene'] == 'SLC25A48'
      and row['configuration']['case']['selection_class'] == 'neutral_control'
  ]

  def median_norm(rows, stage, component):
    return float(statistics.median(
        np.linalg.norm(_vector(
            row, planned[row['configuration']['case']['variant_id']],
            stage, component,
        ))
        for row in rows
    ))

  ratios = []
  for row in effects:
    case_plan = planned[row['configuration']['case']['variant_id']]
    carried = _vector(row, case_plan, 1, 'carried')
    output = _vector(row, case_plan, 1, 'output')
    ratios.append(float(np.linalg.norm(output) / np.linalg.norm(carried)))

  alphabet = weights['dna_alphabet']
  raw = np.asarray(weights['dna_direct_conv']['raw_kernel'])[:, :, 1]
  base_index = {base: index for index, base in enumerate(alphabet)}
  direct_errors = []
  for row in records:
    case = row['configuration']['case']
    if case['gene'] != 'SLC25A48':
      continue
    offset = case['position_1based'] - case['exon_start_1based']
    expected = 0.0
    if -7 <= offset <= 7:
      expected = float(
          raw[offset + 7, base_index[case['alternate_bases']]]
          - raw[offset + 7, base_index[case['reference_bases']]]
      )
    observed = float(_vector(
        row, planned[case['variant_id']], 0, 'carried'
    )[1])
    direct_errors.append(abs(expected - observed))

  result = {
      'raw_file_count': len(records),
      'slc_effect_count': len(effects),
      'slc_neutral_count': len(neutrals),
      'e1_effect_direct_median_l2': median_norm(effects, 0, 'carried'),
      'e1_effect_update_median_l2': median_norm(effects, 0, 'first_update'),
      'e1_effect_output_median_l2': median_norm(effects, 0, 'output'),
      'e1_neutral_output_median_l2': median_norm(neutrals, 0, 'output'),
      'e2_effect_output_median_l2': median_norm(effects, 1, 'output'),
      'e2_neutral_output_median_l2': median_norm(neutrals, 1, 'output'),
      'e2_effect_median_amplification': float(statistics.median(ratios)),
      'e2_effect_amplified_count': sum(value > 1 for value in ratios),
      'maximum_direct_weight_activation_error': max(direct_errors),
  }
  expected = {
      'e1_effect_direct_median_l2': analysis['features']['E1_c0175'][
          'by_selection_class']['significant_effect']['carried'][
              'median_l2_allele_difference'
          ],
      'e1_effect_update_median_l2': analysis['features']['E1_c0175'][
          'by_selection_class']['significant_effect']['first_update'][
              'median_l2_allele_difference'
          ],
      'e1_effect_output_median_l2': analysis['features']['E1_c0175'][
          'by_selection_class']['significant_effect']['output'][
              'median_l2_allele_difference'
          ],
      'e1_neutral_output_median_l2': analysis['features']['E1_c0175'][
          'by_selection_class']['neutral_control']['output'][
              'median_l2_allele_difference'
          ],
      'e2_effect_output_median_l2': analysis['features']['E2_c0175'][
          'by_selection_class']['significant_effect']['output'][
              'median_l2_allele_difference'
          ],
      'e2_neutral_output_median_l2': analysis['features']['E2_c0175'][
          'by_selection_class']['neutral_control']['output'][
              'median_l2_allele_difference'
          ],
      'e2_effect_median_amplification': analysis[
          'slc25a48_e2_amplification']['significant_effect'][
              'median_output_to_carried_l2_ratio'
          ],
      'maximum_direct_weight_activation_error': analysis['weight_analysis'][
          'maximum_direct_weight_to_activation_absolute_difference'
      ],
  }
  differences = {
      key: abs(result[key] - value) for key, value in expected.items()
  }
  result['maximum_analysis_difference'] = max(differences.values())
  if (
      result['maximum_analysis_difference'] != 0
      or result['e2_effect_amplified_count'] != 6
  ):
    raise AuditError('Independent decomposition metrics disagree.')
  return result


def _markdown(result: dict[str, Any]) -> str:
  direct = result['e1_effect_direct_median_l2']
  update = result['e1_effect_update_median_l2']
  e1_effect = result['e1_effect_output_median_l2']
  e1_neutral = result['e1_neutral_output_median_l2']
  e2_neutral = result['e2_neutral_output_median_l2']
  amplification = result['e2_effect_median_amplification']
  amplified_count = result['e2_effect_amplified_count']
  weight_error = result['maximum_direct_weight_activation_error']
  analysis_difference = result['maximum_analysis_difference']
  return (
      '# Independent encoder feature-decomposition audit\n\n'
      'The audit independently reread all raw component arrays and selected '
      'weights. It rebuilt role-aligned allele-difference vectors without '
      'importing the main analyzer.\n\n'
      f"- Raw records: {result['raw_file_count']}\n"
      f"- SLC25A48 effects/neutrals: {result['slc_effect_count']}/"
      f"{result['slc_neutral_count']}\n"
      f'- E1 direct/update/output effect median L2: {direct:.6f} / '
      f'{update:.6f} / {e1_effect:.6f}\n'
      f'- E1/E2 neutral output median L2: {e1_neutral:.6f} / '
      f'{e2_neutral:.6f}\n'
      f'- E2 effect median amplification: {amplification:.6f} '
      f'({amplified_count}/6 amplified)\n'
      f'- Maximum direct-weight/activation error: {weight_error:.9f}\n'
      f'- Maximum difference from the main analysis: '
      f'{analysis_difference}\n\n'
      'The independently recomputed claims match exactly.\n'
  )


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--raw-root', type=Path, default=DEFAULT_RAW)
  parser.add_argument('--analysis', type=Path, default=DEFAULT_ANALYSIS)
  parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
  parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  result = audit(args.raw_root, args.analysis, args.plan)
  if args.output.exists() or args.output.is_symlink():
    raise AuditError(f'Output already exists: {args.output}.')
  args.output.write_text(_markdown(result), encoding='utf-8')
  print(args.output)


if __name__ == '__main__':
  main()
