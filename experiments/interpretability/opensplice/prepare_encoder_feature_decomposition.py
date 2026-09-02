#!/usr/bin/env python3
"""Prepare natural-allele and weight analysis of causal encoder channels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SOURCE_ANALYSIS = (
    HERE / 'results' / 'single_channel_sufficiency_v1_model_behavior_analysis'
    / 'ANALYSIS.json'
)
CASE_PLAN = HERE / 'spatial_encoder_skip_plan_v1.json'
DEFAULT_OUTPUT = HERE / 'encoder_feature_decomposition_plan_v1.json'
FEATURES = (
    ('BRAF', 'E16', 16, 3),
    ('SLC25A48', 'E1', 1, 175),
    ('SLC25A48', 'E2', 2, 175),
)
ENCODER_RESOLUTIONS = (1, 2, 4, 8, 16, 32, 64)


class PlanError(RuntimeError):
  """Raised when a source result or frozen design invariant changes."""


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
  analysis = _load(SOURCE_ANALYSIS)
  cases_source = _load(CASE_PLAN)
  observed = tuple(
      (row['selected_for_gene'], row['stage'], row['resolution_bp'],
       row['channel_start_inclusive'])
      for row in analysis['per_channel'].values()
  )
  if (
      analysis.get('schema_version')
      != 'alphagenome-single-channel-sufficiency-analysis-v1'
      or analysis.get('scope', {}).get('confirmation_access') is not False
      or observed != FEATURES
      or len(cases_source.get('cases', ())) != 20
  ):
    raise PlanError('Source result differs from the completed causal finding.')

  cases = []
  for source in cases_source['cases']:
    genomic_positions = tuple(dict.fromkeys((
        source['variant_position_1based'],
        source['acceptor_position_1based'],
        source['donor_position_1based'],
    )))
    stages = []
    for stage_index, resolution in enumerate(ENCODER_RESOLUTIONS):
      selected = list(dict.fromkeys(
          (position - 1 - source['interval_start_0based']) // resolution
          for position in genomic_positions
      ))
      stages.append({
          'stage_index': stage_index,
          'stage': f'E{resolution}',
          'resolution_bp': resolution,
          'positions': selected,
      })
    cases.append({
        key: source[key] for key in (
            'order', 'gene', 'variant_id', 'selection_class', 'chromosome',
            'strand', 'interval_start_0based',
            'interval_end_0based_exclusive', 'variant_position_1based',
        )
    } | {'stages': stages})

  features = [
      {
          'gene': gene,
          'stage': stage,
          'resolution_bp': resolution,
          'encoder_stage_index': ENCODER_RESOLUTIONS.index(resolution),
          'channel_index': channel,
      }
      for gene, stage, resolution, channel in FEATURES
  ]
  return {
      'schema_version': 'alphagenome-encoder-feature-decomposition-plan-v1',
      'question': (
          'What natural allele-dependent computation and learned weights '
          'produce the causal encoder channels?'
      ),
      'scope': {
          'development_only': True,
          'confirmation_access': False,
          'variant_count': 20,
          'effect_count': 12,
          'neutral_count': 8,
      },
      'inputs': {
          'single_channel_analysis_path': SOURCE_ANALYSIS.relative_to(
              HERE
          ).as_posix(),
          'single_channel_analysis_sha256': _sha256(SOURCE_ANALYSIS),
          'case_plan_path': CASE_PLAN.relative_to(HERE).as_posix(),
          'case_plan_sha256': _sha256(CASE_PLAN),
      },
      'design': {
          'features': features,
          'traced_channel_indices': [3, 175],
          'encoder_resolutions_bp': list(ENCODER_RESOLUTIONS),
          'positions': (
              'the same unique V/acceptor/donor tokens as the causal tests'
          ),
          'components': [
              'carried', 'first_update', 'second_update', 'output'
          ],
          'natural_rows': ['reference', 'alternate'],
          'exact_repeats_per_case': 2,
          'planned_model_apply_count': 40,
          'weight_targets': [
              'DnaEmbedder direct Conv1D channels 3 and 175',
              'DownResBlock learned branches targeting channel 175',
          ],
      },
      'analysis': {
          'primary_population': 'gene-matched significant effects',
          'natural_allele_difference': 'alternate minus reference',
          'robustification_test': (
              'compare cross-effect alignment of SLC25A48 channel-175 '
              'allele-difference vectors at E1, the E2 carried path, both '
              'E2 learned updates, and the E2 output'
          ),
          'interpretation_limit': (
              'Raw kernels are mechanistic weight evidence, not by themselves '
              'a transcription- or splicing-factor assignment.'
          ),
      },
      'controls': {
          'checkpoint_compatible_no_new_parameters': True,
          'normal_encoder_output_exact_in_unit_tests': True,
          'exact_runtime_repeat': True,
          'invalid_padded_positions_zero': True,
          'all_values_finite': True,
          'confirmation_access': False,
          'os_kernel_is_a_gate': False,
      },
      'cases': cases,
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
