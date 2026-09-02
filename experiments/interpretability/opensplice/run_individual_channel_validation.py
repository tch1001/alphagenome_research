#!/usr/bin/env python3
"""Run individual necessity and eight-channel sufficiency/localization tests."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from alphagenome.models import dna_model as public_dna_model
from alphagenome_research.model import dna_model
from alphagenome_research.model import interpretability


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import prepare_individual_channel_validation as planner
import run_channel_group_screen as screen
import run_inference_trace as v2
import run_route_census_v3 as route_v3
# pylint: enable=wrong-import-position


SCRIPT_VERSION = 'alphagenome-individual-channel-validation-v1.0.0'
DEFAULT_PLAN = HERE / 'individual_channel_validation_plan_v1.json'
DEFAULT_OUTPUT = HERE / 'results' / 'individual_channel_validation_v1'
REFINEMENT_FULL = HERE / 'results' / 'channel_refinement_v1' / 'raw' / 'full'


class ValidationError(RuntimeError):
  """Raised when an individual-channel design or runtime control fails."""


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plan(path: Path = DEFAULT_PLAN) -> tuple[dict[str, Any], str]:
  if not path.is_file() or path.is_symlink():
    raise ValidationError(f'Plan must be a regular file: {path}.')
  try:
    plan = json.loads(path.read_text(encoding='utf-8'))
  except json.JSONDecodeError as error:
    raise ValidationError(f'Plan is not valid JSON: {path}.') from error
  if plan != planner.build_plan():
    raise ValidationError('Plan differs from the deterministic source plan.')
  if (
      plan.get('schema_version')
      != 'alphagenome-individual-channel-validation-plan-v1'
      or plan.get('scope', {}).get('confirmation_access') is not False
      or plan.get('design', {}).get(
          'individual_necessity_condition_count'
      ) != 32
      or plan.get('design', {}).get(
          'eight_channel_sufficiency_condition_count'
      ) != 12
      or plan.get('design', {}).get('planned_model_apply_count') != 960
  ):
    raise ValidationError('Individual-channel frozen design changed.')
  return plan, _sha256(path)


def _binding(
    checkpoint: Path, plan_path: Path, plan_sha256: str
) -> dict[str, Any]:
  if checkpoint.name != route_v3.CHECKPOINT_SNAPSHOT:
    raise ValidationError(
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


def condition_selection(
    case: v2.Case, planned: Mapping[str, Any], location: str
) -> interpretability.CausalRouteTraceSelection:
  """Build the fixed position selector for a planned V location."""
  if location not in planner.LOCATIONS:
    raise ValidationError(f'Unknown spatial location: {location}.')
  interval = v2.centered_interval(case, screen.spatial.planner.CONTEXT_BP)
  selection = route_v3.route_selection(case, interval)
  positions = np.zeros((7, screen.spatial.SLOTS), np.int32)
  valid = np.zeros((7, screen.spatial.SLOTS), bool)
  stages = planned['positions_by_location'][location]
  for index, (stage, expected) in enumerate(zip(
      stages, screen.planner.STAGES, strict=True
  )):
    name, resolution, _, enabled = expected
    values = stage['positions']
    if (
        stage['stage'] != name
        or stage['resolution_bp'] != resolution
        or stage['route_enabled'] is not enabled
        or len(values) > screen.spatial.SLOTS
        or (not enabled and values)
    ):
      raise ValidationError('Condition position binding changed.')
    positions[index, :len(values)] = values
    valid[index, :len(values)] = True
  return dataclasses.replace(
      selection,
      decoder_skip_positions=jnp.asarray(positions),
      decoder_skip_valid_mask=jnp.asarray(valid),
  )


def condition_channels(condition: Mapping[str, Any]) -> np.ndarray:
  """Build all-except-one or only-eight channel masks."""
  feature = condition['feature']
  if condition['kind'] == 'individual_channel_necessity':
    if (
        condition['channel_policy']
        != 'all_candidate_channels_except_named_channel'
        or feature['channel_count'] != 1
    ):
      raise ValidationError('Individual necessity condition changed.')
    result = screen.channel_mask(feature)
    if int(screen.channel_mask().sum() - result.sum()) != 1:
      raise ValidationError('Necessity mask does not withhold one channel.')
    return result
  if condition['kind'] == 'eight_channel_only_sufficiency':
    if (
        condition['channel_policy'] != 'only_named_eight_channel_child'
        or feature['channel_count'] != 8
    ):
      raise ValidationError('Eight-channel sufficiency condition changed.')
    result = np.zeros((7, screen.planner.MAX_CHANNELS), dtype=bool)
    start = feature['channel_start_inclusive']
    end = feature['channel_end_exclusive']
    result[feature['stage_index'], start:end] = True
    if int(result.sum()) != 8:
      raise ValidationError('Sufficiency mask does not select eight channels.')
    return result
  raise ValidationError(f"Unknown condition kind: {condition.get('kind')}.")


def _case_stem(case: v2.Case) -> str:
  # pylint: disable=protected-access
  return f'{case.order:03d}_{v2._slug(case.variant_id)}'


def _run_case(
    model_instance, apply_fn, case: v2.Case, planned: Mapping[str, Any],
    conditions: Sequence[Mapping[str, Any]], binding: Mapping[str, Any],
    output: Path,
) -> int:
  intended = condition_selection(case, planned, 'intended')
  full_channels = screen.channel_mask()
  interval = v2.centered_interval(case, screen.spatial.planner.CONTEXT_BP)
  metadata = model_instance._metadata[  # pylint: disable=protected-access
      public_dna_model.Organism.HOMO_SAPIENS
  ].splice_sites
  target_selection, resolved = route_v3.target_selection(
      metadata, case, interval
  )
  dna_batch, sequence_sha256 = route_v3._build_six_row_batch(  # pylint: disable=protected-access
      model_instance, case, interval
  )
  common = (
      model_instance._params, model_instance._state,  # pylint: disable=protected-access
      dna_batch, jnp.zeros((6,), jnp.int32),
  )
  base_configuration = {
      **binding,
      'case': v2._case_record(case),  # pylint: disable=protected-access
      'sequence_sha256': sequence_sha256,
      'target': {
          'endpoints': [
              dataclasses.asdict(value) for value in resolved.endpoints
          ],
          'padding_track_index': resolved.padding_track_index,
      },
  }

  identity_intervention = screen.interventions(
      intended, full_channels, active_positions=False
  )
  screen.validate_runtime_contract(
      intended, identity_intervention, full_channels, active_positions=False
  )
  identity_configuration = {
      **base_configuration, 'kind': 'individual_validation_identity'
  }
  identity_path = output / 'raw' / 'identity' / f'{_case_stem(case)}.json'
  fingerprint = v2._fingerprint(  # pylint: disable=protected-access
      identity_configuration
  )
  identity = v2._load_completed(  # pylint: disable=protected-access
      identity_path, fingerprint
  )
  if identity is None:
    first, first_seconds = route_v3._timed_apply(  # pylint: disable=protected-access
        apply_fn, *common, intended, identity_intervention, target_selection
    )
    repeated, repeated_seconds = route_v3._timed_apply(  # pylint: disable=protected-access
        apply_fn, *common, intended, identity_intervention, target_selection
    )
    checks = route_v3.validate_identity_audit(
        first[0], first[1], repeated[0], repeated[1]
    )
    identity = {
        'status': 'complete', 'fingerprint': fingerprint,
        'configuration': identity_configuration, 'checks': checks,
        'seconds': {'first': first_seconds, 'repeat': repeated_seconds},
        'created_at_unix_s': time.time(),
    }
    v2._write_atomic(identity_path, identity)  # pylint: disable=protected-access
  identity_means = [
      identity['checks']['target_means'][role] for role in screen.ROLES
  ]

  full_intervention = screen.interventions(
      intended, full_channels, active_positions=True
  )
  screen.validate_runtime_contract(
      intended, full_intervention, full_channels, active_positions=True
  )
  full_configuration = {
      **base_configuration,
      'kind': 'individual_validation_full_V_route',
      'identity_fingerprint': identity['fingerprint'],
  }
  full_path = output / 'raw' / 'full' / f'{_case_stem(case)}.json'
  full_fingerprint = v2._fingerprint(  # pylint: disable=protected-access
      full_configuration
  )
  full = v2._load_completed(  # pylint: disable=protected-access
      full_path, full_fingerprint
  )
  if full is None:
    first, first_seconds = route_v3._timed_apply(  # pylint: disable=protected-access
        apply_fn, *common, intended, full_intervention, target_selection
    )
    repeated, repeated_seconds = route_v3._timed_apply(  # pylint: disable=protected-access
        apply_fn, *common, intended, full_intervention, target_selection
    )
    checks = screen.validate_active_call(
        first, identity_means, intended, full_channels, repeated
    )
    full = {
        'status': 'complete', 'fingerprint': full_fingerprint,
        'configuration': full_configuration, 'checks': checks,
        'seconds': {'first': first_seconds, 'repeat': repeated_seconds},
        'created_at_unix_s': time.time(),
    }
    v2._write_atomic(full_path, full)  # pylint: disable=protected-access

  completed = 0
  condition_dir = output / 'raw' / 'conditions' / _case_stem(case)
  for condition in conditions:
    selection = condition_selection(case, planned, condition['location'])
    channels = condition_channels(condition)
    intervention = screen.interventions(
        selection, channels, active_positions=True
    )
    screen.validate_runtime_contract(
        selection, intervention, channels, active_positions=True
    )
    configuration = {
        **base_configuration,
        'kind': 'individual_channel_validation_condition',
        'identity_fingerprint': identity['fingerprint'],
        'full_fingerprint': full['fingerprint'],
        'condition': condition,
    }
    filename = (
        f"{condition['condition_index']:03d}_"
        f"{condition['condition_id']}.json"
    )
    path = condition_dir / filename
    condition_fingerprint = v2._fingerprint(  # pylint: disable=protected-access
        configuration
    )
    record = v2._load_completed(  # pylint: disable=protected-access
        path, condition_fingerprint
    )
    if record is None:
      result, seconds = route_v3._timed_apply(  # pylint: disable=protected-access
          apply_fn, *common, selection, intervention, target_selection
      )
      checks = screen.validate_active_call(
          result, identity_means, selection, channels
      )
      record = {
          'status': 'complete', 'fingerprint': condition_fingerprint,
          'configuration': configuration, 'checks': checks,
          'seconds': seconds, 'created_at_unix_s': time.time(),
      }
      v2._write_atomic(path, record)  # pylint: disable=protected-access
    completed += 1
  return completed


def build_dry_run(
    bindings: Sequence[tuple[v2.Case, Mapping[str, Any]]],
    conditions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
  return {
      'script_version': SCRIPT_VERSION,
      'development_only': True,
      'confirmation_access': False,
      'variant_count': len(bindings),
      'condition_count': len(conditions),
      'individual_necessity_condition_count': sum(
          value['kind'] == 'individual_channel_necessity'
          for value in conditions
      ),
      'sufficiency_condition_count': sum(
          value['kind'] == 'eight_channel_only_sufficiency'
          for value in conditions
      ),
      'model_apply_count': len(bindings) * (4 + len(conditions)),
      'fixed_position_shape': [7, screen.spatial.SLOTS],
      'fixed_channel_shape': [7, screen.planner.MAX_CHANNELS],
  }


def _refinement_full(case: v2.Case) -> dict[str, Any]:
  path = REFINEMENT_FULL / f'{_case_stem(case)}.json'
  try:
    record = json.loads(path.read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError) as error:
    raise ValidationError(
        f'Cannot read refinement full result: {path}.'
    ) from error
  if (
      record.get('status') != 'complete'
      or record.get('configuration', {}).get('case', {}).get('variant_id')
      != case.variant_id
      or record.get('checks', {}).get('passed') is not True
  ):
    raise ValidationError(f'Refinement full result is invalid: {path}.')
  return record


def summarize_cross_executable_closure(
    bindings: Sequence[tuple[v2.Case, Mapping[str, Any]]], output: Path
) -> dict[str, Any]:
  differences = []
  recovery_differences = []
  exact = 0
  for case, _ in bindings:
    current = json.loads(
        (output / 'raw' / 'full' / f'{_case_stem(case)}.json').read_text(
            encoding='utf-8'
        )
    )
    prior = _refinement_full(case)
    left = current['checks']['target_readout']
    right = prior['checks']['target_readout']
    exact += left == right
    differences.extend(
        abs(float(a) - float(b))
        for a, b in zip(left['means'], right['means'], strict=True)
    )
    recovery_differences.append(
        current['checks']['recovery']['bidirectional_bottleneck']
        - prior['checks']['recovery']['bidirectional_bottleneck']
    )
  if not all(
      math.isfinite(value)
      for value in (*differences, *recovery_differences)
  ):
    raise ValidationError('Cross-executable closure is non-finite.')
  return {
      'comparison_is_diagnostic_not_a_gate': True,
      'case_count': len(bindings),
      'exact_target_readout_case_count': exact,
      'maximum_absolute_target_mean_difference': max(differences),
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
    raise ValidationError('Limits must be nonnegative.')
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
    raise ValidationError(f'Validation requires a GPU; observed {devices}.')
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
    count = _run_case(
        model_instance, apply_fn, case, planned, conditions, binding,
        args.output_dir,
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
      'condition_count': len(conditions),
      'condition_result_count': completed,
      'model_apply_count_in_full_nonresume_run': (
          len(bindings) * (4 + len(conditions))
      ),
      'full_frozen_design_completed': (
          len(bindings) == 20 and len(conditions) == 44
      ),
      'prior_refinement_closure': summarize_cross_executable_closure(
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
