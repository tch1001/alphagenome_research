#!/usr/bin/env python3
"""Prepare controlled single-base edits around the SLC25A48 acceptor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SOURCE_ANALYSIS = (
    HERE / 'results'
    / 'encoder_feature_decomposition_v1_model_behavior_analysis'
    / 'ANALYSIS.json'
)
CASE_PLAN = HERE / 'encoder_feature_decomposition_plan_v1.json'
DEFAULT_OUTPUT = HERE / 'slc25a48_channel175_ism_plan_v1.json'
ANCHOR_VARIANT = 'SLC25A48_e8_G70A'
SCAN_OFFSETS = tuple(range(-20, 21))
TRACE_OFFSETS = tuple(range(-24, 25))
EDIT_ROWS_PER_BATCH = 5


class PlanError(RuntimeError):
  """Raised when source evidence or frozen ISM design changes."""


def _load(path: Path) -> dict[str, Any]:
  try:
    value = json.loads(path.read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError) as error:
    raise PlanError(f'Cannot load {path}.') from error
  if not isinstance(value, dict):
    raise PlanError(f'JSON root is not an object: {path}.')
  return value


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def build_plan() -> dict[str, Any]:
  analysis = _load(SOURCE_ANALYSIS)
  case_plan = _load(CASE_PLAN)
  interpretation = analysis.get('interpretation', {})
  if (
      analysis.get('schema_version')
      != 'alphagenome-encoder-feature-decomposition-analysis-v1'
      or analysis.get('scope', {}).get('confirmation_access') is not False
      or analysis.get('weight_analysis', {}).get(
          'direct_kernel_preferred_core'
      ) != 'TAGG'
      or interpretation.get(
          'e1_channel_175_is_nonlinear_composite_acceptor_detector'
      ) is not True
      or interpretation.get('e2_amplifies_all_six_effect_vectors') is not True
  ):
    raise PlanError('Source is not the completed channel-175 interpretation.')
  matches = [
      case for case in case_plan['cases']
      if case['variant_id'] == ANCHOR_VARIANT
  ]
  if len(matches) != 1:
    raise PlanError('Anchor case is not unique.')
  anchor = matches[0]
  candidate_count = len(SCAN_OFFSETS) * 3
  batch_count = math.ceil(candidate_count / EDIT_ROWS_PER_BATCH)
  return {
      'schema_version': 'alphagenome-slc25a48-channel175-ism-plan-v1',
      'question': (
          'Which single-base edits around the SLC25A48 exon-8 acceptor '
          'change channel 175 and the canonical splice-logit output?'
      ),
      'scope': {
          'development_only': True,
          'confirmation_access': False,
          'gene': 'SLC25A48',
          'exon_id': 'SLC25A48_e8',
      },
      'inputs': {
          'decomposition_analysis_path': SOURCE_ANALYSIS.relative_to(
              HERE
          ).as_posix(),
          'decomposition_analysis_sha256': _sha256(SOURCE_ANALYSIS),
          'case_plan_path': CASE_PLAN.relative_to(HERE).as_posix(),
          'case_plan_sha256': _sha256(CASE_PLAN),
      },
      'anchor': anchor,
      'design': {
          'reference': 'GRCh38 reference allele sequence',
          'scan_offsets_from_acceptor_bp': list(SCAN_OFFSETS),
          'alternate_bases_per_position': 3,
          'candidate_edit_count': candidate_count,
          'trace_offsets_from_acceptor_bp': list(TRACE_OFFSETS),
          'traced_channel': 175,
          'traced_encoder_resolutions_bp': [1, 2],
          'target': (
              'canonical acceptor and donor splice-classification '
              'class-minus-padding logit margins'
          ),
          'batch_layout': (
              'reference row followed by five single-edit rows; final unused '
              'rows are exact reference padding controls'
          ),
          'batch_size': 6,
          'edit_rows_per_batch': EDIT_ROWS_PER_BATCH,
          'batch_count': batch_count,
          'encoder_model_apply_count': batch_count,
          'full_model_apply_count': batch_count,
          'planned_model_apply_count': batch_count * 2,
      },
      'analysis': {
          'primary': (
              'per-edit E1/E2 channel-175 change at the acceptor token and '
              'acceptor splice-logit-margin change'
          ),
          'association': (
              'Pearson and rank correlation across all 123 edits, plus '
              'position/base maps and the preregistered -3..0 TAGG core'
          ),
          'interpretation_limit': (
              'ISM supports a sequence-to-model mechanism at this exon; it '
              'does not identify a cellular factor or prove genome-wide use.'
          ),
      },
      'controls': {
          'reference_repeated_in_every_batch': True,
          'padding_rows_exact_reference': True,
          'every_candidate_is_one_snv': True,
          'all_values_finite': True,
          'confirmation_access': False,
          'os_kernel_is_a_gate': False,
      },
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
