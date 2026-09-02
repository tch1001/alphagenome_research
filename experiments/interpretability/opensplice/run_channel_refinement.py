#!/usr/bin/env python3
"""Run the development-only 8-channel V-local refinement."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import jax

from alphagenome_research.model import dna_model


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import prepare_channel_refinement as planner
import run_channel_group_screen as screen
import run_inference_trace as v2
import run_route_census_v3 as route_v3
# pylint: enable=wrong-import-position


SCRIPT_VERSION = 'alphagenome-channel-refinement-v1.0.0'
DEFAULT_PLAN = HERE / 'channel_refinement_plan_v1.json'
DEFAULT_OUTPUT = HERE / 'results' / 'channel_refinement_v1'
SCREEN_RAW = HERE / 'results' / 'channel_group_screen_v1' / 'raw' / 'full'


class RefinementError(RuntimeError):
  """Raised when the refinement design or an execution invariant fails."""


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plan(path: Path = DEFAULT_PLAN) -> tuple[dict[str, Any], str]:
  if not path.is_file() or path.is_symlink():
    raise RefinementError(f'Plan must be a regular file: {path}.')
  try:
    plan = json.loads(path.read_text(encoding='utf-8'))
  except json.JSONDecodeError as error:
    raise RefinementError(f'Plan is not valid JSON: {path}.') from error
  if plan != planner.build_plan():
    raise RefinementError('Plan differs from the deterministic source plan.')
  if (
      plan.get('schema_version')
      != 'alphagenome-channel-refinement-plan-v1'
      or plan.get('scope', {}).get('confirmation_access') is not False
      or plan.get('design', {}).get('parent_group_count') != 5
      or plan.get('design', {}).get('group_count') != 20
      or plan.get('design', {}).get('planned_model_apply_count') != 480
  ):
    raise RefinementError('Refinement scope or fixed design changed.')
  return plan, _sha256(path)


def _binding(
    checkpoint: Path, plan_path: Path, plan_sha256: str
) -> dict[str, Any]:
  if checkpoint.name != route_v3.CHECKPOINT_SNAPSHOT:
    raise RefinementError(
        f'Expected checkpoint {route_v3.CHECKPOINT_SNAPSHOT}, '
        f'observed {checkpoint.name}.'
    )
  repo = HERE.parents[2]
  return {
      'script_version': SCRIPT_VERSION,
      'script_sha256': _sha256(Path(__file__).resolve()),
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


def build_dry_run(
    bindings: tuple[tuple[v2.Case, Mapping[str, Any]], ...],
    groups: list[Mapping[str, Any]],
) -> dict[str, Any]:
  return {
      'script_version': SCRIPT_VERSION,
      'development_only': True,
      'confirmation_access': False,
      'variant_count': len(bindings),
      'parent_group_count': len({group['parent_group_id'] for group in groups}),
      'group_count': len(groups),
      'model_apply_count': len(bindings) * (4 + len(groups)),
      'fixed_position_shape': [7, screen.spatial.SLOTS],
      'fixed_channel_shape': [7, screen.planner.MAX_CHANNELS],
  }


def _screen_full(case: v2.Case) -> dict[str, Any]:
  # pylint: disable=protected-access
  path = SCREEN_RAW / (
      f'{case.order:03d}_{v2._slug(case.variant_id)}.json'
  )
  try:
    record = json.loads(path.read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError) as error:
    raise RefinementError(
        f'Cannot read prior full-route result: {path}.'
    ) from error
  if (
      record.get('status') != 'complete'
      or record.get('configuration', {}).get('case', {}).get('variant_id')
      != case.variant_id
      or record.get('checks', {}).get('passed') is not True
  ):
    raise RefinementError(f'Prior full-route result is invalid: {path}.')
  return record


def summarize_cross_executable_closure(
    bindings: tuple[tuple[v2.Case, Mapping[str, Any]], ...], output: Path
) -> dict[str, Any]:
  """Record, but do not gate on, full-route cross-executable differences."""
  target_differences = []
  recovery_differences = []
  exact = 0
  for case, _ in bindings:
    # pylint: disable=protected-access
    stem = f'{case.order:03d}_{v2._slug(case.variant_id)}'
    current = json.loads(
        (output / 'raw' / 'full' / f'{stem}.json').read_text(encoding='utf-8')
    )
    prior = _screen_full(case)
    current_readout = current['checks']['target_readout']
    prior_readout = prior['checks']['target_readout']
    exact += current_readout == prior_readout
    target_differences.extend(
        abs(float(left) - float(right)) for left, right in zip(
            current_readout['means'], prior_readout['means'], strict=True
        )
    )
    recovery_differences.append(
        current['checks']['recovery']['bidirectional_bottleneck']
        - prior['checks']['recovery']['bidirectional_bottleneck']
    )
  if not all(
      math.isfinite(value)
      for value in (*target_differences, *recovery_differences)
  ):
    raise RefinementError('Cross-executable closure is non-finite.')
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
  parser.add_argument('--max-groups', type=int, default=0)
  parser.add_argument('--dry-run', action='store_true')
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if args.max_variants < 0 or args.max_groups < 0:
    raise RefinementError('Limits must be nonnegative.')
  plan, plan_sha256 = load_plan(args.plan)
  bindings = screen.bind_cases(plan, route_v3.load_development_cases())
  groups = plan['groups']
  if args.max_variants:
    bindings = bindings[:args.max_variants]
  if args.max_groups:
    groups = groups[:args.max_groups]
  if args.dry_run:
    print(json.dumps(build_dry_run(bindings, groups), indent=2))
    return
  devices = jax.devices()
  if not any(device.platform == 'gpu' for device in devices):
    raise RefinementError(f'Refinement requires a GPU; observed {devices}.')
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
    count = screen._run_case(  # pylint: disable=protected-access
        model_instance, apply_fn, case, planned, groups, binding,
        args.output_dir,
    )
    completed += count
    print(
        f'completed {case.order:03d} {case.variant_id}: {count} children',
        flush=True,
    )
  closure = summarize_cross_executable_closure(bindings, args.output_dir)
  summary = {
      'status': 'complete',
      'binding': binding,
      'variant_count': len(bindings),
      'parent_group_count': len({
          group['parent_group_id'] for group in groups
      }),
      'group_count': len(groups),
      'group_result_count': completed,
      'model_apply_count_in_full_nonresume_run': (
          len(bindings) * (4 + len(groups))
      ),
      'full_frozen_design_completed': (
          len(bindings) == 20 and len(groups) == 20
      ),
      'prior_channel_screen_closure': closure,
      'created_at_unix_s': time.time(),
  }
  v2._write_atomic(  # pylint: disable=protected-access
      args.output_dir / 'SUMMARY.json', summary
  )
  print(json.dumps(summary, indent=2))


if __name__ == '__main__':
  main()
