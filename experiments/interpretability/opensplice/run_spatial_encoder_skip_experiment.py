#!/usr/bin/env python3
"""Run the development-only spatial encoder-skip localization experiment.

The runner uses AlphaGenome's existing positional decoder-skip transfer seam.
It never loads confirmation examples.  Every call has the same six-row batch
and fixed selector shapes so identity, biological supports, and translated
controls reuse one compiled executable.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
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
import prepare_spatial_encoder_skip_experiment as planner  # pylint: disable=g-import-not-at-top
import run_inference_trace as v2  # pylint: disable=g-import-not-at-top
import run_route_census_v3 as route_v3  # pylint: disable=g-import-not-at-top
# pylint: enable=wrong-import-position


SCRIPT_VERSION = 'alphagenome-spatial-encoder-skip-v1.0.0'
DEFAULT_PLAN = HERE / 'spatial_encoder_skip_plan_v1.json'
DEFAULT_OUTPUT = HERE / 'results' / 'spatial_encoder_skip_v1'
STAGE_NAMES = tuple(name for name, _ in planner.STAGES)
STAGE_RESOLUTIONS = tuple(resolution for _, resolution in planner.STAGES)
EXPECTED_ENABLED = (False, True, True, True, False, True, True)
SLOTS = 6
ROLES = route_v3.TRACE_BATCH_ROLES
DONOR_ROWS = np.asarray(route_v3.TRACE_BATCH_DONORS, np.int32)
IDENTITY_ROWS = np.asarray((0, 1, 1, 1, 0, 0), np.int32)
RECIPIENT_ROWS = np.asarray((False, False, True, True, True, True), bool)


class SpatialRunError(RuntimeError):
  """Raised when the frozen spatial design or a causal control fails."""


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plan(path: Path = DEFAULT_PLAN) -> tuple[dict[str, Any], str]:
  """Loads and validates the exact deterministic development-only plan."""
  if not path.is_file() or path.is_symlink():
    raise SpatialRunError(f'Plan must be a regular file: {path}.')
  try:
    plan = json.loads(path.read_text(encoding='utf-8'))
  except json.JSONDecodeError as error:
    raise SpatialRunError(f'Plan is not valid JSON: {path}.') from error
  if plan != planner.build_plan():
    raise SpatialRunError(
        'Spatial plan differs from the deterministic metadata-derived plan.'
    )
  if (
      plan.get('schema_version')
      != 'alphagenome-spatial-encoder-skip-plan-v1'
      or plan['scope'].get('development_only') is not True
      or plan['scope'].get('confirmation_access') is not False
      or plan['model_behavior_design'].get('maximum_position_slots') != SLOTS
      or tuple(plan['model_behavior_design'].get('candidate_players', ()))
      != planner.CANDIDATE_PLAYERS
  ):
    raise SpatialRunError('Spatial plan scope or fixed-shape contract changed.')
  return plan, _sha256(path)


def bind_cases(
    plan: Mapping[str, Any], cases: Sequence[v2.Case]
) -> tuple[tuple[v2.Case, Mapping[str, Any]], ...]:
  """Binds plan rows to the independently validated model-input cases."""
  plan_cases = plan['cases']
  if len(cases) != 20 or len(plan_cases) != 20:
    raise SpatialRunError('Spatial experiment requires exactly 20 cases.')
  result = []
  for case, planned in zip(cases, plan_cases, strict=True):
    observed = (
        case.order, case.gene, case.variant_id, case.selection_class,
        case.chromosome, case.strand, case.position_1based,
    )
    expected = (
        planned['order'], planned['gene'], planned['variant_id'],
        planned['selection_class'], planned['chromosome'], planned['strand'],
        planned['variant_position_1based'],
    )
    if observed != expected:
      raise SpatialRunError(
          f'Plan/model case binding differs at order {case.order}.'
      )
    interval = v2.centered_interval(case, planner.CONTEXT_BP)
    if (
        interval.start != planned['interval_start_0based']
        or interval.end != planned['interval_end_0based_exclusive']
    ):
      raise SpatialRunError(f'Interval binding differs for {case.variant_id}.')
    result.append((case, planned))
  return tuple(result)


def _padded_stage_positions(
    condition: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
  positions = np.zeros((len(STAGE_NAMES), SLOTS), np.int32)
  valid = np.zeros((len(STAGE_NAMES), SLOTS), bool)
  stages = condition.get('stages', ())
  if len(stages) != len(STAGE_NAMES):
    raise SpatialRunError('Spatial condition does not have seven skip stages.')
  for index, (stage, name, resolution, expected_enabled) in enumerate(zip(
      stages, STAGE_NAMES, STAGE_RESOLUTIONS, EXPECTED_ENABLED, strict=True
  )):
    if (
        stage.get('player') != name
        or stage.get('resolution_bp') != resolution
        or stage.get('enabled') is not expected_enabled
    ):
      raise SpatialRunError('Spatial condition stage contract changed.')
    values = tuple(stage.get('positions', ()))
    if not expected_enabled and values:
      raise SpatialRunError(f'Disabled stage {name} has selected positions.')
    if len(values) > SLOTS or len(set(values)) != len(values):
      raise SpatialRunError(f'Invalid fixed-slot positions for stage {name}.')
    sequence_length = planner.CONTEXT_BP // resolution
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        or value < 0 or value >= sequence_length
        for value in values
    ):
      raise SpatialRunError(f'Out-of-range position for stage {name}.')
    positions[index, :len(values)] = values
    valid[index, :len(values)] = True
  return positions, valid


def spatial_selection(
    case: v2.Case, condition: Mapping[str, Any]
) -> interpretability.CausalRouteTraceSelection:
  """Builds the fixed six-slot selector for one dynamic spatial condition."""
  interval = v2.centered_interval(case, planner.CONTEXT_BP)
  selection = route_v3.route_selection(case, interval)
  positions, valid = _padded_stage_positions(condition)
  return dataclasses.replace(
      selection,
      decoder_skip_positions=jnp.asarray(positions),
      decoder_skip_valid_mask=jnp.asarray(valid),
  )


def spatial_interventions(
    selection: interpretability.CausalRouteTraceSelection, *, active: bool
) -> interpretability.CausalRouteInterventions:
  identity = interpretability.no_causal_route_interventions(
      selection,
      batch_size=interpretability.PAIRED_CAUSAL_BATCH_SIZE,
      num_edges=route_v3.NUM_DUMMY_EDGES,
  )
  if not active:
    return identity
  return dataclasses.replace(
      identity,
      decoder_skip_states=interpretability.paired_six_row_batch_transfer(
          selection.decoder_skip_valid_mask
      ),
  )


def validate_runtime_contract(
    selection: interpretability.CausalRouteTraceSelection,
    interventions: interpretability.CausalRouteInterventions,
    *, active: bool,
) -> None:
  """Checks the actual arrays that will be passed to the JIT executable."""
  interpretability.validate_causal_route_interventions(
      selection, interventions,
      batch_size=interpretability.PAIRED_CAUSAL_BATCH_SIZE,
  )
  transfer = interventions.decoder_skip_states
  donors = np.asarray(transfer.donor_batch_indices, np.int32)
  mask = np.asarray(transfer.transfer_mask, bool)
  expected_donors = np.broadcast_to(
      DONOR_ROWS[None, :, None], (len(STAGE_NAMES), 6, SLOTS)
  )
  if active:
    expected_mask = (
        np.asarray(selection.decoder_skip_valid_mask, bool)[:, None, :]
        & RECIPIENT_ROWS[None, :, None]
    )
  else:
    expected_mask = np.zeros((len(STAGE_NAMES), 6, SLOTS), bool)
    expected_donors = np.broadcast_to(
        np.arange(6, dtype=np.int32)[None, :, None],
        (len(STAGE_NAMES), 6, SLOTS),
    )
  if not np.array_equal(donors, expected_donors):
    raise SpatialRunError('Runtime decoder-skip donor map changed.')
  if not np.array_equal(mask, expected_mask):
    raise SpatialRunError('Runtime decoder-skip transfer mask changed.')
  for name in ('encoder_outputs', 'decoder_outputs', 'final_embeddings'):
    other = getattr(interventions, name)
    if np.asarray(other.transfer_mask, bool).any():
      raise SpatialRunError(f'Non-spatial route is active: {name}.')
  transformer = interventions.transformer
  if not np.all(np.asarray(transformer.head_masks) == 1):
    raise SpatialRunError('Transformer head mask changed from identity.')
  if np.asarray(transformer.pair_bias_replace_mask, bool).any():
    raise SpatialRunError('Transformer pair bias route is active.')
  for name in (
      'pre_attention_residual_transfer',
      'post_attention_residual_transfer',
      'post_mlp_residual_transfer',
  ):
    if np.asarray(getattr(transformer, name).transfer_mask, bool).any():
      raise SpatialRunError(f'Transformer residual route is active: {name}.')


def target_record(target: interpretability.TargetSummary) -> dict[str, Any]:
  means = np.asarray(target.mean, np.float32)
  totals = np.asarray(target.total, np.float32)
  count = int(np.asarray(target.num_values))
  if means.shape != (6,) or totals.shape != (6,) or count != 2:
    raise SpatialRunError('Splice target shape or endpoint count changed.')
  if not np.isfinite(means).all() or not np.isfinite(totals).all():
    raise SpatialRunError('Splice target contains a non-finite value.')
  if not np.array_equal(means * np.float32(2), totals):
    raise SpatialRunError('Splice target mean/total algebra changed.')
  return {
      'means': means.tolist(),
      'totals': totals.tolist(),
      'num_values': count,
      'row_roles': list(ROLES),
  }


def _traces_exact(
    first: interpretability.CausalRouteTrace,
    second: interpretability.CausalRouteTrace,
) -> bool:
  first_arrays = route_v3._trace_arrays(first)  # pylint: disable=protected-access
  second_arrays = route_v3._trace_arrays(second)  # pylint: disable=protected-access
  return len(first_arrays) == len(second_arrays) and all(
      name == other_name and axis == other_axis
      and np.array_equal(value, other)
      for (name, value, axis), (other_name, other, other_axis)
      in zip(first_arrays, second_arrays, strict=True)
  )


def recovery_from_means(values: Sequence[float]) -> dict[str, Any]:
  if len(values) != 6:
    raise ValueError('Recovery requires six target means.')
  ref, alt, ref_alt, alt_alt, alt_ref, ref_ref = map(float, values)
  denominator = ref - alt
  if denominator == 0:
    raise SpatialRunError('Cannot normalize a zero AlphaGenome allele effect.')
  forward = (ref_alt - alt_alt) / denominator
  reverse = (alt_ref - ref_ref) / -denominator
  return {
      'reference_into_alternate': forward,
      'alternate_into_reference': reverse,
      'bidirectional_bottleneck': min(forward, reverse),
      'bidirectional_mean': (forward + reverse) / 2.0,
  }


def validate_condition(
    first: tuple[interpretability.TargetSummary,
                 interpretability.CausalRouteTrace],
    repeated: tuple[interpretability.TargetSummary,
                    interpretability.CausalRouteTrace],
    identity_means: Sequence[float],
    selection: interpretability.CausalRouteTraceSelection,
) -> dict[str, Any]:
  """Validates exact repeat, no-op, self-control, and donor-vector behavior."""
  first_target, first_trace = first
  repeated_target, repeated_trace = repeated
  readout = target_record(first_target)
  repeated_readout = target_record(repeated_target)
  if readout != repeated_readout or not _traces_exact(
      first_trace, repeated_trace
  ):
    raise SpatialRunError('Spatial intervention repeat is not bit-exact.')
  values = np.asarray(readout['means'], np.float32)
  identity = np.asarray(identity_means, np.float32)
  if not np.array_equal(values[:2], identity[:2]):
    raise SpatialRunError('Spatial call changed baseline output rows.')
  if values[3] != values[1] or values[5] != values[0]:
    raise SpatialRunError('Spatial same-allele self controls changed output.')

  valid_by_stage = np.asarray(selection.decoder_skip_valid_mask, bool)
  donor_checks = {}
  no_op_checks = {}
  same_allele_checks = {}
  for stage, name in enumerate(STAGE_NAMES):
    natural = np.asarray(first_trace.decoder_skip_states[stage])
    effective = np.asarray(first_trace.effective_decoder_skip_states[stage])
    valid = valid_by_stage[stage]
    if natural.shape[:2] != (6, SLOTS) or effective.shape != natural.shape:
      raise SpatialRunError(f'Unexpected traced tensor shape at {name}.')
    same_allele = all(
        np.array_equal(natural[row], natural[IDENTITY_ROWS[row]])
        for row in range(6)
    )
    same_allele_checks[name] = bool(same_allele)
    if not same_allele:
      raise SpatialRunError(f'Natural same-allele traces differ at {name}.')
    inactive = ~valid
    no_op = (
        np.array_equal(effective[:2], natural[:2])
        and np.array_equal(effective[:, inactive], natural[:, inactive])
        and (valid.any() or np.array_equal(effective, natural))
    )
    no_op_checks[name] = bool(no_op)
    if not no_op:
      raise SpatialRunError(f'Disabled decoder-skip values changed at {name}.')
    if valid.any():
      donor_exact = all(
          np.array_equal(
              effective[row, valid], natural[DONOR_ROWS[row], valid]
          )
          for row in range(2, 6)
      )
    else:
      donor_exact = True
    donor_checks[name] = bool(donor_exact) if valid.any() else None
    if not donor_exact:
      raise SpatialRunError(f'Enabled live donor vectors differ at {name}.')
  return {
      'passed': True,
      'target_repeat_exact': True,
      'trace_repeat_exact': True,
      'baseline_targets_exact_from_identity': True,
      'self_targets_exact': True,
      'natural_same_allele_exact_by_stage': same_allele_checks,
      'disabled_or_unselected_noop_exact_by_stage': no_op_checks,
      'enabled_donor_vectors_exact_by_stage': donor_checks,
      'target_readout': readout,
      'repeat_target_readout': repeated_readout,
      'raw_movement': {
          'reference_into_alternate': float(values[2] - values[3]),
          'alternate_into_reference': float(values[4] - values[5]),
      },
      'recovery': recovery_from_means(values),
  }


def _condition_record(condition: Mapping[str, Any]) -> dict[str, Any]:
  positions, valid = _padded_stage_positions(condition)
  return {
      'condition_id': condition['condition_id'],
      'support': condition['support'],
      'location': condition['location'],
      'stage_names': list(STAGE_NAMES),
      'stage_resolutions_bp': list(STAGE_RESOLUTIONS),
      'positions': positions.tolist(),
      'valid_mask': valid.tolist(),
  }


def build_dry_run(
    plan: Mapping[str, Any],
    bindings: Sequence[tuple[v2.Case, Mapping[str, Any]]],
    *, max_conditions: int,
) -> dict[str, Any]:
  conditions_per_case = (
      max_conditions if max_conditions else
      plan['model_behavior_design']['condition_count_per_variant']
  )
  return {
      'script_version': SCRIPT_VERSION,
      'development_only': True,
      'confirmation_access': False,
      'variant_count': len(bindings),
      'condition_count_per_variant': conditions_per_case,
      'model_apply_count': len(bindings) * (2 + 2 * conditions_per_case),
      'fixed_decoder_skip_shape': [7, 6],
      'candidate_players': list(planner.CANDIDATE_PLAYERS),
      'condition_ids': [
          condition['condition_id']
          for condition in bindings[0][1]['conditions'][:conditions_per_case]
      ] if bindings else [],
  }


def _binding(
    checkpoint: Path, plan_path: Path, plan_sha256: str
) -> dict[str, Any]:
  if checkpoint.name != route_v3.CHECKPOINT_SNAPSHOT:
    raise SpatialRunError(
        f'Expected checkpoint snapshot {route_v3.CHECKPOINT_SNAPSHOT}, '
        f'observed {checkpoint.name}.'
    )
  return {
      'script_version': SCRIPT_VERSION,
      'script_sha256': _sha256(Path(__file__).resolve()),
      'plan_path': str(plan_path.resolve()),
      'plan_sha256': plan_sha256,
      'checkpoint_path': str(checkpoint),
      'checkpoint_snapshot': checkpoint.name,
      'context_bp': planner.CONTEXT_BP,
      'attention_backend': route_v3.ATTENTION_BACKEND,
      'confirmation_access': False,
  }


def _identity_path(output: Path, case: v2.Case) -> Path:
  return output / 'raw' / 'identity' / (
      f'{case.order:03d}_{v2._slug(case.variant_id)}.json'  # pylint: disable=protected-access
  )


def _condition_path(
    output: Path, case: v2.Case, condition: Mapping[str, Any]
) -> Path:
  return output / 'raw' / 'conditions' / (
      f'{case.order:03d}_{v2._slug(case.variant_id)}'  # pylint: disable=protected-access
  ) / f"{condition['condition_id']}.json"


def _run_identity(
    model_instance, apply_fn, case: v2.Case, planned: Mapping[str, Any],
    binding: Mapping[str, Any], output: Path,
) -> tuple[dict[str, Any], tuple[Any, ...], Any]:
  condition = planned['conditions'][0]
  selection = spatial_selection(case, condition)
  interventions = spatial_interventions(selection, active=False)
  validate_runtime_contract(selection, interventions, active=False)
  interval = v2.centered_interval(case, planner.CONTEXT_BP)
  metadata = model_instance._metadata[  # pylint: disable=protected-access
      public_dna_model.Organism.HOMO_SAPIENS
  ].splice_sites
  target_selection, resolved = route_v3.target_selection(
      metadata, case, interval
  )
  dna_batch, sequence_sha256 = route_v3._build_six_row_batch(  # pylint: disable=protected-access
      model_instance, case, interval
  )
  configuration = {
      **binding,
      'kind': 'spatial_all_false_identity',
      'case': v2._case_record(case),  # pylint: disable=protected-access
      'condition_selector': _condition_record(condition),
      'sequence_sha256': sequence_sha256,
      'target': {
          'endpoints': [dataclasses.asdict(x) for x in resolved.endpoints],
          'padding_track_index': resolved.padding_track_index,
      },
  }
  fingerprint = v2._fingerprint(configuration)  # pylint: disable=protected-access
  path = _identity_path(output, case)
  completed = v2._load_completed(path, fingerprint)  # pylint: disable=protected-access
  organism = jnp.zeros((6,), jnp.int32)
  common = (
      model_instance._params, model_instance._state,  # pylint: disable=protected-access
      dna_batch, organism,
  )
  if completed is None:
    first, first_seconds = route_v3._timed_apply(  # pylint: disable=protected-access
        apply_fn, *common, selection, interventions, target_selection
    )
    repeated, repeated_seconds = route_v3._timed_apply(  # pylint: disable=protected-access
        apply_fn, *common, selection, interventions, target_selection
    )
    checks = route_v3.validate_identity_audit(
        first[0], first[1], repeated[0], repeated[1]
    )
    completed = {
        'status': 'complete',
        'fingerprint': fingerprint,
        'configuration': configuration,
        'checks': checks,
        'seconds': {'first': first_seconds, 'repeat': repeated_seconds},
        'created_at_unix_s': time.time(),
    }
    v2._write_atomic(path, completed)  # pylint: disable=protected-access
  return completed, common, target_selection


def _run_condition(
    apply_fn, case: v2.Case, condition: Mapping[str, Any],
    identity: Mapping[str, Any], common: tuple[Any, ...], target_selection,
    binding: Mapping[str, Any], output: Path,
) -> dict[str, Any]:
  selection = spatial_selection(case, condition)
  interventions = spatial_interventions(selection, active=True)
  validate_runtime_contract(selection, interventions, active=True)
  configuration = {
      **binding,
      'kind': 'spatial_decoder_skip_live_transfer',
      'case': v2._case_record(case),  # pylint: disable=protected-access
      'condition': _condition_record(condition),
      'identity_fingerprint': identity['fingerprint'],
      'sequence_sha256': identity['configuration']['sequence_sha256'],
  }
  fingerprint = v2._fingerprint(configuration)  # pylint: disable=protected-access
  path = _condition_path(output, case, condition)
  completed = v2._load_completed(path, fingerprint)  # pylint: disable=protected-access
  if completed is not None:
    return completed
  first, first_seconds = route_v3._timed_apply(  # pylint: disable=protected-access
      apply_fn, *common, selection, interventions, target_selection
  )
  repeated, repeated_seconds = route_v3._timed_apply(  # pylint: disable=protected-access
      apply_fn, *common, selection, interventions, target_selection
  )
  identity_means = [
      identity['checks']['target_means'][role] for role in ROLES
  ]
  checks = validate_condition(
      first, repeated, identity_means, selection
  )
  completed = {
      'status': 'complete',
      'fingerprint': fingerprint,
      'configuration': configuration,
      'checks': checks,
      'seconds': {'first': first_seconds, 'repeat': repeated_seconds},
      'created_at_unix_s': time.time(),
  }
  v2._write_atomic(path, completed)  # pylint: disable=protected-access
  return completed


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
    raise SpatialRunError('Limits must be nonnegative.')
  plan, plan_sha256 = load_plan(args.plan)
  cases = route_v3.load_development_cases()
  bindings = bind_cases(plan, cases)
  if args.max_variants:
    bindings = bindings[:args.max_variants]
  total_conditions = plan[
      'model_behavior_design'
  ]['condition_count_per_variant']
  if args.max_conditions > total_conditions:
    raise SpatialRunError('Condition limit exceeds the frozen design.')
  if args.dry_run:
    print(json.dumps(
        build_dry_run(plan, bindings, max_conditions=args.max_conditions),
        indent=2, allow_nan=False,
    ))
    return

  devices = jax.devices()
  if not any(device.platform == 'gpu' for device in devices):
    raise SpatialRunError(
        f'This AlphaGenome experiment requires a GPU; observed {devices}.'
    )
  checkpoint = v2._checkpoint_path(  # pylint: disable=protected-access
      args.checkpoint
  )
  binding = _binding(checkpoint, args.plan, plan_sha256)
  model_instance = dna_model.create(
      checkpoint,
      model_settings=dna_model.ModelSettings(
          attention_backend=route_v3.ATTENTION_BACKEND
      ),
  )
  apply_fn = (
      dna_model.create_splice_classification_logit_margin_route_census_apply(
          model_instance._metadata,  # pylint: disable=protected-access
          attention_backend=route_v3.ATTENTION_BACKEND,
      )
  )
  apply_fn = jax.jit(apply_fn)
  completed_count = 0
  condition_limit = args.max_conditions or total_conditions
  for case, planned in bindings:
    identity, common, target_selection = _run_identity(
        model_instance, apply_fn, case, planned, binding, args.output_dir
    )
    for condition in planned['conditions'][:condition_limit]:
      _run_condition(
          apply_fn, case, condition, identity, common, target_selection,
          binding, args.output_dir,
      )
      completed_count += 1
    print(
        f'completed {case.order:03d} {case.variant_id}: '
        f'{condition_limit} spatial conditions',
        flush=True,
    )

  summary = {
      'status': 'complete',
      'binding': binding,
      'variant_count': len(bindings),
      'condition_count': completed_count,
      'model_apply_count_in_full_nonresume_run': (
          len(bindings) * (2 + 2 * condition_limit)
      ),
      'full_frozen_design_completed': (
          len(bindings) == 20 and condition_limit == total_conditions
      ),
      'created_at_unix_s': time.time(),
  }
  v2._write_atomic(args.output_dir / 'SUMMARY.json', summary)  # pylint: disable=protected-access
  print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
  main()
