#!/usr/bin/env python3
"""Independently audit key SLC25A48 channel-175 ISM claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_RAW = HERE / 'results' / 'slc25a48_channel175_ism_v1'
DEFAULT_ANALYSIS = (
    HERE / 'results'
    / 'slc25a48_channel175_ism_v1_model_behavior_analysis'
    / 'ANALYSIS.json'
)
DEFAULT_PLAN = HERE / 'slc25a48_channel175_ism_plan_v1.json'
DEFAULT_OUTPUT = DEFAULT_ANALYSIS.parent / 'INDEPENDENT_AUDIT.md'


class AuditError(RuntimeError):
  """Raised when independently recomputed ISM metrics disagree."""


def _load(path: Path) -> dict[str, Any]:
  try:
    return json.loads(path.read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError) as error:
    raise AuditError(f'Cannot load {path}.') from error


def audit(
    raw_root: Path = DEFAULT_RAW,
    analysis_path: Path = DEFAULT_ANALYSIS,
    plan_path: Path = DEFAULT_PLAN,
) -> dict[str, Any]:
  analysis = _load(analysis_path)
  plan = _load(plan_path)
  paths = sorted((raw_root / 'raw').glob('batch_*.json'))
  records = [_load(path) for path in paths]
  if len(records) != 25 or any(
      record.get('checks', {}).get('passed') is not True
      for record in records
  ):
    raise AuditError('Raw batches are incomplete or failed controls.')
  acceptor = records[0]['configuration']['anchor_case'][
      'exon_start_1based'
  ]
  interval_start = plan['anchor']['interval_start_0based']
  edits = []
  references = []
  for record in records:
    positions = record['configuration']['feature_positions'][1]
    slot = positions.index((acceptor - 1 - interval_start) // 2)
    margins = np.asarray(record['target']['margins'], dtype=float)
    feature = np.asarray(
        record['feature_components']['output'], dtype=float
    )[1, :, slot, 0]
    references.append((margins[0].tolist(), feature[0]))
    for row_index, row in enumerate(record['rows']):
      if row['kind'] != 'edit':
        continue
      edits.append({
          'offset': row['edit']['offset_from_acceptor_bp'],
          'acceptor_delta': float(margins[row_index, 0] - margins[0, 0]),
          'e2_delta': float(feature[row_index] - feature[0]),
      })
  if (
      len(edits) != 123
      or not all(reference == references[0] for reference in references)
  ):
    raise AuditError('Candidate count or repeated reference differs.')
  x = np.asarray([row['e2_delta'] for row in edits])
  y = np.asarray([row['acceptor_delta'] for row in edits])
  pearson = float(np.corrcoef(x, y)[0, 1])
  core = [
      row['acceptor_delta'] for row in edits if -3 <= row['offset'] <= 0
  ]
  invariant = [
      row['acceptor_delta'] for row in edits if -2 <= row['offset'] <= -1
  ]
  outside = [
      row['acceptor_delta'] for row in edits
      if not -3 <= row['offset'] <= 0
  ]
  result = {
      'raw_batch_count': len(records),
      'candidate_edit_count': len(edits),
      'reference_across_batches_exact': True,
      'invariant_ag_negative_count': sum(value < 0 for value in invariant),
      'invariant_ag_median_acceptor_delta': float(statistics.median(invariant)),
      'core_median_absolute_acceptor_delta': float(
          statistics.median(map(abs, core))
      ),
      'outside_median_absolute_acceptor_delta': float(
          statistics.median(map(abs, outside))
      ),
      'e2_acceptor_pearson': pearson,
  }
  expected = {
      'invariant_ag_negative_count': analysis['groups'][
          'invariant_AG_minus2_through_minus1'
      ]['negative_acceptor_margin_count'],
      'invariant_ag_median_acceptor_delta': analysis['groups'][
          'invariant_AG_minus2_through_minus1'
      ]['median_acceptor_margin_delta'],
      'core_median_absolute_acceptor_delta': analysis['groups'][
          'TAGG_core_minus3_through_0'
      ]['median_absolute_acceptor_margin_delta'],
      'outside_median_absolute_acceptor_delta': analysis['groups'][
          'outside_core'
      ]['median_absolute_acceptor_margin_delta'],
      'e2_acceptor_pearson': analysis[
          'channel_to_acceptor_output_associations'
      ]['e2_output_delta_at_acceptor']['pearson'],
  }
  differences = [
      abs(result[key] - value) for key, value in expected.items()
  ]
  result['maximum_analysis_difference'] = max(differences)
  if result['maximum_analysis_difference'] != 0:
    raise AuditError('Independent ISM metrics disagree with main analysis.')
  return result


def _markdown(result: dict[str, Any]) -> str:
  return (
      '# Independent SLC25A48 channel-175 ISM audit\n\n'
      'The audit independently reread all 25 raw batches and reconstructed '
      'edit deltas without importing the main analyzer.\n\n'
      f"- Candidate edits: {result['candidate_edit_count']}\n"
      '- Reference exact across batches: '
      f"{str(result['reference_across_batches_exact']).lower()}\n"
      '- Invariant AG edits decreasing acceptor output: '
      f"{result['invariant_ag_negative_count']}/6\n"
      '- Invariant AG median acceptor delta: '
      f"{result['invariant_ag_median_acceptor_delta']:.6f}\n"
      '- Core/outside median absolute delta: '
      f"{result['core_median_absolute_acceptor_delta']:.6f} / "
      f"{result['outside_median_absolute_acceptor_delta']:.6f}\n"
      f"- E2:175/acceptor Pearson r: {result['e2_acceptor_pearson']:.9f}\n"
      '- Maximum difference from the main analysis: '
      f"{result['maximum_analysis_difference']}\n\n"
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
