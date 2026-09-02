#!/usr/bin/env python3
"""Run intended and shifted sufficiency tests for three model coordinates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import jax
import numpy as np

from alphagenome_research.model import dna_model


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import prepare_single_channel_sufficiency as planner
import run_channel_group_screen as screen
import run_individual_channel_validation as individual
import run_inference_trace as v2
import run_route_census_v3 as route_v3
# pylint: enable=wrong-import-position


SCRIPT_VERSION = 'alphagenome-single-channel-sufficiency-v1.0.0'
DEFAULT_PLAN = HERE / 'single_channel_sufficiency_plan_v1.json'
DEFAULT_OUTPUT = HERE / 'results' / 'single_channel_sufficiency_v1'
PRIOR_FULL = (
    HERE / 'results' / 'individual_channel_validation_v1' / 'raw' / 'full'
)


class SufficiencyError(RuntimeError):
  """Raised when a design or single-channel runtime invariant fails."""


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plan(path: Path = DEFAULT_PLAN) -> tuple[dict[str, Any], str]:
  if not path.is_file() or path.is_symlink():
    raise SufficiencyError(f'Plan must be a regular file: {path}.')
  try:
    plan = json.loads(path.read_text(encoding='utf-8'))
  except json.JSONDecodeError as error:
    raise SufficiencyError(f'Plan is not valid JSON: {path}.') from error
  if plan != planner.build_plan():
    raise SufficiencyError('Plan differs from deterministic source plan.')
  if (
      plan.get('schema_version')
      != 'alphagenome-single-channel-sufficiency-plan-v1'
      or plan.get('scope', {}).get('confirmation_access') is not False
      or plan.get('design', {}).get('selected_channel_count') != 3
      or plan.get('design', {}).get('condition_count') != 9
      or plan.get('design', {}).get('planned_model_apply_count') != 260
  ):
    raise SufficiencyError('Single-channel frozen design changed.')
  return plan, _sha256(path)


def _binding(
    checkpoint: Path, plan_path: Path, plan_sha256: str
) -> dict[str, Any]:
  if checkpoint.name != route_v3.CHECKPOINT_SNAPSHOT:
    raise SufficiencyError(
        f'Expected checkpoint {route_v3.CHECKPOINT_SNAPSHOT}, '
        f'observed {checkpoint.name}.'
    )
  repo = HERE.parents[2]
  return {
      'script_version': SCRIPT_VERSION,
      'script_sha256': _sha256(Path(__file__).resolve()),
      'individual_runner_sha256': _sha256(
          HERE / 'run_individual_channel_validation.py'
      ),
      'interpretability_sha256': _sha256(
          repo / 'src/alphagenome_research/model/interpretability.py'
      ),
      'model_sha256': _sha256(
          repo / 'src/alphagenome_research/model/model.py'
      ),
      'plan_path': str(plan_path.resolve()),
      'plan_sha256': plan_sha256,
      'checkpoint_snapshot': checkpoint.name,
      'context_bp': screen.spatial.planner.CONTEXT_BP,
      'attention_backend': route_v3.ATTENTION_BACKEND,
      'confirmation_access': False,
  }


def single_channel_mask(condition: Mapping[str, Any]) -> np.ndarray:
  """Select only the one channel named by a sufficiency condition."""
  feature = condition['feature']
  if (
      condition.get('kind') != 'single_channel_only_sufficiency'
      or condition.get('channel_policy') != 'only_named_single_channel'
      or feature.get('channel_count') != 1
  ):
    raise SufficiencyError('Single-channel condition changed.')
  result = np.zeros((7, screen.planner.MAX_CHANNELS), dtype=bool)
  start = feature['channel_start_inclusive']
  end = feature['channel_end_exclusive']
  stage = feature['stage_index']
  if (
      feature['stage'] != screen.STAGE_NAMES[stage]
      or not 0 <= start < end <= screen.STAGE_WIDTHS[stage]
  ):
    raise SufficiencyError('Single-channel bounds changed.')
  result[stage, start:end] = True
  if int(result.sum()) != 1:
    raise SufficiencyError('Sufficiency condition does not select one channel.')
  return result


def build_dry_run(
    bindings: Sequence[tuple[v2.Case, Mapping[str, Any]]],
    conditions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
  return {
      'script_version': SCRIPT_VERSION,
      'development_only': True,
      'confirmation_access': False,
      'variant_count': len(bindings),
      'selected_channel_count': len({
          value['feature']['source_condition_id'] for value in conditions
      }),
      'condition_count': len(conditions),
      'model_apply_count': len(bindings) * (4 + len(conditions)),
      'fixed_position_shape': [7, screen.spatial.SLOTS],
      'fixed_channel_shape': [7, screen.planner.MAX_CHANNELS],
  }


def _prior_full(case: v2.Case) -> dict[str, Any]:
  path = PRIOR_FULL / f'{individual._case_stem(case)}.json'  # pylint: disable=protected-access
  try:
    record = json.loads(path.read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError) as error:
    raise SufficiencyError(f'Cannot read prior full result: {path}.') from error
  if record.get('status') != 'complete':
    raise SufficiencyError(f'Prior full result is incomplete: {path}.')
  return record


def closure(
    bindings: Sequence[tuple[v2.Case, Mapping[str, Any]]], output: Path
) -> dict[str, Any]:
  target_differences = []
  recovery_differences = []
  exact = 0
  for case, _ in bindings:
    stem = individual._case_stem(case)  # pylint: disable=protected-access
    current = json.loads(
        (output / 'raw' / 'full' / f'{stem}.json').read_text(encoding='utf-8')
    )
    prior = _prior_full(case)
    left = current['checks']['target_readout']
    right = prior['checks']['target_readout']
    exact += left == right
    target_differences.extend(
        abs(float(a) - float(b))
        for a, b in zip(left['means'], right['means'], strict=True)
    )
    recovery_differences.append(
        current['checks']['recovery']['bidirectional_bottleneck']
        - prior['checks']['recovery']['bidirectional_bottleneck']
    )
  if not all(
      math.isfinite(value)
      for value in (*target_differences, *recovery_differences)
  ):
    raise SufficiencyError('Cross-executable closure is non-finite.')
  return {
      'comparison_is_diagnostic_not_a_gate': True,
      'case_count': len(bindings),
      'exact_target_readout_case_count': exact,
      'maximum_absolute_target_mean_difference': max(target_differences),
      'maximum_absolute_bidirectional_bottleneck_difference': max(
          map(abs, recovery_differences)
      ),
  }


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', type=Path)
  parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
  parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument('--max-variants', type=int, default=0)
  parser.add_argument('--max-conditions', type=int, default=0)
  parser.add_argument('--dry-run', action='store_true')
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if args.max_variants < 0 or args.max_conditions < 0:
    raise SufficiencyError('Limits must be nonnegative.')
  plan, plan_sha256 = load_plan(args.plan)
  bindings = screen.bind_cases(plan, route_v3.load_development_cases())
  conditions = plan['conditions']
  if args.max_variants:
    bindings = bindings[:args.max_variants]
  if args.max_conditions:
    conditions = conditions[:args.max_conditions]
  if args.dry_run:
    print(json.dumps(build_dry_run(bindings, conditions), indent=2))
    return
  devices = jax.devices()
  if not any(device.platform == 'gpu' for device in devices):
    raise SufficiencyError(f'Sufficiency run requires GPU; observed {devices}.')
  checkpoint = v2._checkpoint_path(args.checkpoint)  # pylint: disable=protected-access
  binding = _binding(checkpoint, args.plan, plan_sha256)
  model_instance = dna_model.create(
      checkpoint,
      model_settings=dna_model.ModelSettings(
          attention_backend=route_v3.ATTENTION_BACKEND
      ),
  )
  apply_fn = jax.jit(
      dna_model.create_splice_classification_logit_margin_route_census_apply(
          model_instance._metadata,  # pylint: disable=protected-access
          attention_backend=route_v3.ATTENTION_BACKEND,
      )
  )
  completed = 0
  for case, planned in bindings:
    count = individual._run_case(  # pylint: disable=protected-access
        model_instance, apply_fn, case, planned, conditions, binding,
        args.output_dir, condition_channel_fn=single_channel_mask,
    )
    completed += count
    print(
        f'completed {case.order:03d} {case.variant_id}: {count} conditions',
        flush=True,
    )
  summary = {
      'status': 'complete',
      'binding': binding,
      'variant_count': len(bindings),
      'selected_channel_count': len({
          value['feature']['source_condition_id'] for value in conditions
      }),
      'condition_count': len(conditions),
      'condition_result_count': completed,
      'model_apply_count_in_full_nonresume_run': (
          len(bindings) * (4 + len(conditions))
      ),
      'full_frozen_design_completed': (
          len(bindings) == 20 and len(conditions) == 9
      ),
      'prior_individual_validation_closure': closure(
          bindings, args.output_dir
      ),
      'created_at_unix_s': time.time(),
  }
  v2._write_atomic(  # pylint: disable=protected-access
      args.output_dir / 'SUMMARY.json', summary
  )
  print(json.dumps(summary, indent=2))


if __name__ == '__main__':
  main()
