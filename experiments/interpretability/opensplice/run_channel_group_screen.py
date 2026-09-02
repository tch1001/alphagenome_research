#!/usr/bin/env python3
"""Run the V-local grouped-channel necessity screen on development variants."""

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
import prepare_channel_group_screen as planner
import run_inference_trace as v2
import run_route_census_v3 as route_v3
import run_spatial_encoder_skip_experiment as spatial
# pylint: enable=wrong-import-position


SCRIPT_VERSION = 'alphagenome-channel-group-screen-v1.0.0'
DEFAULT_PLAN = HERE / 'channel_group_screen_plan_v1.json'
DEFAULT_OUTPUT = HERE / 'results' / 'channel_group_screen_v1'
SPATIAL_RAW = HERE / 'results' / 'spatial_encoder_skip_v1'
STAGE_NAMES = tuple(stage for stage, _, _, _ in planner.STAGES)
STAGE_WIDTHS = tuple(width for _, _, width, _ in planner.STAGES)
ROLES = route_v3.TRACE_BATCH_ROLES
DONOR_ROWS = np.asarray(route_v3.TRACE_BATCH_DONORS, np.int32)
IDENTITY_ROWS = np.asarray((0, 1, 1, 1, 0, 0), np.int32)
RECIPIENT_ROWS = np.asarray((False, False, True, True, True, True), bool)


class ScreenError(RuntimeError):
  """Raised when the screen design or a causal runtime control fails."""


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plan(path: Path = DEFAULT_PLAN) -> tuple[dict[str, Any], str]:
  if not path.is_file() or path.is_symlink():
    raise ScreenError(f'Plan must be a regular file: {path}.')
  try:
    plan = json.loads(path.read_text(encoding='utf-8'))
  except json.JSONDecodeError as error:
    raise ScreenError(f'Plan is not valid JSON: {path}.') from error
  if plan != planner.build_plan():
    raise ScreenError(
        'Channel plan differs from the deterministic source plan.'
    )
  if (
      plan.get('schema_version')
      != 'alphagenome-channel-group-screen-plan-v1'
      or plan['scope'].get('confirmation_access') is not False
      or plan['design'].get('group_count') != 172
      or plan['design'].get('maximum_channel_axis') != planner.MAX_CHANNELS
  ):
    raise ScreenError('Channel plan scope or fixed-shape contract changed.')
  return plan, _sha256(path)


def bind_cases(
    plan: Mapping[str, Any], cases: Sequence[v2.Case]
) -> tuple[tuple[v2.Case, Mapping[str, Any]], ...]:
  if len(cases) != 20 or len(plan['cases']) != 20:
    raise ScreenError('Channel screen requires exactly 20 development cases.')
  result = []
  for case, planned in zip(cases, plan['cases'], strict=True):
    if (
        case.order != planned['order']
        or case.gene != planned['gene']
        or case.variant_id != planned['variant_id']
        or case.selection_class != planned['selection_class']
        or case.position_1based != planned['variant_position_1based']
    ):
      raise ScreenError(f'Case binding differs at order {case.order}.')
    interval = v2.centered_interval(case, spatial.planner.CONTEXT_BP)
    if (
        interval.start != planned['interval_start_0based']
        or interval.end != planned['interval_end_0based_exclusive']
    ):
      raise ScreenError(f'Interval binding differs for {case.variant_id}.')
    result.append((case, planned))
  return tuple(result)


def channel_selection(
    case: v2.Case, planned: Mapping[str, Any]
) -> interpretability.CausalRouteTraceSelection:
  interval = v2.centered_interval(case, spatial.planner.CONTEXT_BP)
  selection = route_v3.route_selection(case, interval)
  positions = np.zeros((7, spatial.SLOTS), np.int32)
  valid = np.zeros((7, spatial.SLOTS), bool)
  if len(planned['stages']) != 7:
    raise ScreenError('Channel case does not have seven skip stages.')
  for index, (stage, expected) in enumerate(zip(
      planned['stages'], planner.STAGES, strict=True
  )):
    name, resolution, width, enabled = expected
    if (
        stage['stage'] != name
        or stage['stage_index'] != index
        or stage['resolution_bp'] != resolution
        or stage['channel_width'] != width
        or stage['route_enabled'] is not enabled
    ):
      raise ScreenError('Channel case stage binding changed.')
    selected = stage['positions']
    if len(selected) > spatial.SLOTS or (not enabled and selected):
      raise ScreenError(f'Invalid position selection at {name}.')
    positions[index, :len(selected)] = selected
    valid[index, :len(selected)] = True
  return dataclasses.replace(
      selection,
      decoder_skip_positions=jnp.asarray(positions),
      decoder_skip_valid_mask=jnp.asarray(valid),
  )


def channel_mask(
    group: Mapping[str, Any] | None = None,
) -> np.ndarray:
  """Returns all candidate channels, optionally withholding one group."""
  mask = np.zeros((7, planner.MAX_CHANNELS), bool)
  for index, (_, _, width, enabled) in enumerate(planner.STAGES):
    if enabled:
      mask[index, :width] = True
  if group is not None:
    index = group['stage_index']
    if group['stage'] != STAGE_NAMES[index]:
      raise ScreenError('Group stage name/index binding changed.')
    start = group['channel_start_inclusive']
    end = group['channel_end_exclusive']
    if not 0 <= start < end <= STAGE_WIDTHS[index]:
      raise ScreenError('Group channel bounds are invalid.')
    mask[index, start:end] = False
  return mask


def interventions(
    selection: interpretability.CausalRouteTraceSelection,
    channels: np.ndarray,
    *, active_positions: bool,
) -> interpretability.CausalRouteInterventions:
  identity = interpretability.no_causal_route_interventions(
      selection,
      batch_size=interpretability.PAIRED_CAUSAL_BATCH_SIZE,
      num_edges=route_v3.NUM_DUMMY_EDGES,
  )
  positions = (
      selection.decoder_skip_valid_mask
      if active_positions
      else jnp.zeros_like(selection.decoder_skip_valid_mask)
  )
  transfer = interpretability.paired_six_row_batch_transfer(
      positions, jnp.asarray(channels)
  )
  return dataclasses.replace(identity, decoder_skip_states=transfer)


def validate_runtime_contract(
    selection: interpretability.CausalRouteTraceSelection,
    value: interpretability.CausalRouteInterventions,
    channels: np.ndarray,
    *, active_positions: bool,
) -> None:
  interpretability.validate_causal_route_interventions(
      selection, value, batch_size=6
  )
  transfer = value.decoder_skip_states
  if not np.array_equal(np.asarray(transfer.channel_mask), channels):
    raise ScreenError('Runtime channel mask differs from the planned mask.')
  expected_donors = np.broadcast_to(
      DONOR_ROWS[None, :, None], (7, 6, spatial.SLOTS)
  )
  if not np.array_equal(transfer.donor_batch_indices, expected_donors):
    raise ScreenError('Runtime donor rows differ from the six-row protocol.')
  expected_positions = (
      np.asarray(selection.decoder_skip_valid_mask)[:, None, :]
      & RECIPIENT_ROWS[None, :, None]
      if active_positions
      else np.zeros((7, 6, spatial.SLOTS), bool)
  )
  if not np.array_equal(transfer.transfer_mask, expected_positions):
    raise ScreenError('Runtime spatial transfer mask changed.')
  for name in ('encoder_outputs', 'decoder_outputs', 'final_embeddings'):
    other = getattr(value, name)
    if np.asarray(other.transfer_mask).any() or other.channel_mask is not None:
      raise ScreenError(f'Non-screen route is active: {name}.')
  transformer = value.transformer
  if not np.all(np.asarray(transformer.head_masks) == 1):
    raise ScreenError('Transformer heads changed from identity.')
  if np.asarray(transformer.pair_bias_replace_mask).any():
    raise ScreenError('Transformer pair-bias intervention is active.')
  for name in (
      'pre_attention_residual_transfer',
      'post_attention_residual_transfer',
      'post_mlp_residual_transfer',
  ):
    route = getattr(transformer, name)
    if np.asarray(route.transfer_mask).any() or route.channel_mask is not None:
      raise ScreenError(f'Transformer route is active: {name}.')


def _other_routes_are_noops(trace: interpretability.CausalRouteTrace) -> bool:
  for natural_name, effective_name in (
      ('encoder_outputs', 'effective_encoder_outputs'),
      ('decoder_outputs', 'effective_decoder_outputs'),
      ('final_embeddings', 'effective_final_embeddings'),
  ):
    natural = getattr(trace, natural_name)
    effective = getattr(trace, effective_name)
    if len(natural) != len(effective) or any(
        not np.array_equal(left, right)
        for left, right in zip(natural, effective, strict=True)
    ):
      return False
  transformer = trace.transformer
  for natural_name, effective_name in (
      ('compact_pair_bias_edges', 'effective_compact_pair_bias_edges'),
      ('head_value_outputs', 'effective_head_value_outputs'),
      ('pre_attention_residuals', 'effective_pre_attention_residuals'),
      ('post_attention_residuals', 'effective_post_attention_residuals'),
      ('post_mlp_residuals', 'effective_post_mlp_residuals'),
  ):
    if not np.array_equal(
        getattr(transformer, natural_name),
        getattr(transformer, effective_name),
    ):
      return False
  return True


def validate_active_call(
    first: tuple[interpretability.TargetSummary,
                 interpretability.CausalRouteTrace],
    identity_means: Sequence[float],
    selection: interpretability.CausalRouteTraceSelection,
    channels: np.ndarray,
    repeated: tuple[interpretability.TargetSummary,
                    interpretability.CausalRouteTrace] | None = None,
) -> dict[str, Any]:
  """Checks selected-channel donors and withheld-channel natural values."""
  target, trace = first
  readout = spatial.target_record(target)
  if repeated is not None:
    repeat_readout = spatial.target_record(repeated[0])
    if (
        readout != repeat_readout
        or not spatial._traces_exact(trace, repeated[1])  # pylint: disable=protected-access
    ):
      raise ScreenError('Repeated full-channel call is not bit-exact.')
  else:
    repeat_readout = None
  values = np.asarray(readout['means'], np.float32)
  identity = np.asarray(identity_means, np.float32)
  if not np.array_equal(values[:2], identity[:2]):
    raise ScreenError('Channel call changed baseline output rows.')
  if values[3] != values[1] or values[5] != values[0]:
    raise ScreenError('Channel self-control output rows changed.')
  if not _other_routes_are_noops(trace):
    raise ScreenError('A non-channel route changed effective values.')

  valid_by_stage = np.asarray(selection.decoder_skip_valid_mask, bool)
  donor_checks = {}
  withheld_checks = {}
  same_allele_checks = {}
  for stage, (name, width) in enumerate(zip(
      STAGE_NAMES, STAGE_WIDTHS, strict=True
  )):
    natural = np.asarray(trace.decoder_skip_states[stage])
    effective = np.asarray(trace.effective_decoder_skip_states[stage])
    if natural.shape != (6, spatial.SLOTS, width):
      raise ScreenError(f'Unexpected traced channel shape at {name}.')
    valid = valid_by_stage[stage]
    selected = channels[stage, :width]
    same_allele = all(
        np.array_equal(natural[row], natural[IDENTITY_ROWS[row]])
        for row in range(6)
    )
    same_allele_checks[name] = bool(same_allele)
    if not same_allele:
      raise ScreenError(f'Natural same-allele values differ at {name}.')
    if not np.array_equal(effective[:2], natural[:2]):
      raise ScreenError(f'Baseline decoder-skip values changed at {name}.')
    if not np.array_equal(effective[:, ~valid], natural[:, ~valid]):
      raise ScreenError(f'Invalid position slots changed at {name}.')
    withheld = ~selected
    withheld_exact = np.array_equal(
        effective[:, :, withheld], natural[:, :, withheld]
    )
    withheld_checks[name] = bool(withheld_exact)
    if not withheld_exact:
      raise ScreenError(f'Withheld channels changed at {name}.')
    if valid.any() and selected.any():
      donor_exact = all(
          np.array_equal(
              effective[row][np.ix_(valid, selected)],
              natural[DONOR_ROWS[row]][np.ix_(valid, selected)],
          )
          for row in range(2, 6)
      )
    else:
      donor_exact = True
    donor_checks[name] = (
        bool(donor_exact) if valid.any() and selected.any() else None
    )
    if not donor_exact:
      raise ScreenError(f'Selected donor channels differ at {name}.')
  recovery = spatial.recovery_from_means(values)
  return {
      'passed': True,
      'repeat_checked': repeated is not None,
      'target_repeat_exact': True if repeated is not None else None,
      'trace_repeat_exact': True if repeated is not None else None,
      'baseline_targets_exact_from_identity': True,
      'self_targets_exact': True,
      'non_channel_routes_noop_exact': True,
      'natural_same_allele_exact_by_stage': same_allele_checks,
      'selected_donor_channels_exact_by_stage': donor_checks,
      'withheld_channels_natural_exact_by_stage': withheld_checks,
      'target_readout': readout,
      'repeat_target_readout': repeat_readout,
      'raw_movement': {
          'reference_into_alternate': float(values[2] - values[3]),
          'alternate_into_reference': float(values[4] - values[5]),
      },
      'recovery': recovery,
  }


def _binding(
    checkpoint: Path, plan_path: Path, plan_sha256: str
) -> dict[str, Any]:
  if checkpoint.name != route_v3.CHECKPOINT_SNAPSHOT:
    raise ScreenError(
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
      'context_bp': spatial.planner.CONTEXT_BP,
      'attention_backend': route_v3.ATTENTION_BACKEND,
      'confirmation_access': False,
  }


def _case_stem(case: v2.Case) -> str:
  return f'{case.order:03d}_{v2._slug(case.variant_id)}'  # pylint: disable=protected-access


def _spatial_closure(case: v2.Case) -> dict[str, Any]:
  path = (
      SPATIAL_RAW / 'raw' / 'conditions' / _case_stem(case) / 'V_intended.json'
  )
  try:
    record = json.loads(path.read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError) as error:
    raise ScreenError(f'Cannot load spatial closure result: {path}.') from error
  if (
      record.get('status') != 'complete'
      or record.get('configuration', {}).get('case', {}).get('variant_id')
      != case.variant_id
      or record.get('configuration', {}).get('condition', {}).get(
          'condition_id'
      ) != 'V_intended'
      or record.get('checks', {}).get('passed') is not True
  ):
    raise ScreenError(f'Invalid spatial closure result: {path}.')
  return record


def _load_completed(path: Path, configuration: Mapping[str, Any]):
  fingerprint = v2._fingerprint(configuration)  # pylint: disable=protected-access
  return (
      v2._load_completed(path, fingerprint),  # pylint: disable=protected-access
      fingerprint,
  )


def _run_case(
    model_instance, apply_fn, case: v2.Case, planned: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]], binding: Mapping[str, Any],
    output: Path,
) -> int:
  selection = channel_selection(case, planned)
  full_channels = channel_mask()
  interval = v2.centered_interval(case, spatial.planner.CONTEXT_BP)
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
      dna_batch, jnp.zeros((6,), jnp.int32), selection,
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

  identity_intervention = interventions(
      selection, full_channels, active_positions=False
  )
  validate_runtime_contract(
      selection, identity_intervention, full_channels, active_positions=False
  )
  identity_configuration = {
      **base_configuration, 'kind': 'channel_screen_identity'
  }
  identity_path = output / 'raw' / 'identity' / f'{_case_stem(case)}.json'
  identity, fingerprint = _load_completed(
      identity_path, identity_configuration
  )
  if identity is None:
    first, first_seconds = route_v3._timed_apply(  # pylint: disable=protected-access
        apply_fn, *common, identity_intervention, target_selection
    )
    repeated, repeated_seconds = route_v3._timed_apply(  # pylint: disable=protected-access
        apply_fn, *common, identity_intervention, target_selection
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
  identity_means = [identity['checks']['target_means'][role] for role in ROLES]

  full_intervention = interventions(
      selection, full_channels, active_positions=True
  )
  validate_runtime_contract(
      selection, full_intervention, full_channels, active_positions=True
  )
  full_configuration = {
      **base_configuration,
      'kind': 'channel_screen_full_V_route',
      'identity_fingerprint': identity['fingerprint'],
  }
  full_path = output / 'raw' / 'full' / f'{_case_stem(case)}.json'
  full, full_fingerprint = _load_completed(full_path, full_configuration)
  if full is None:
    first, first_seconds = route_v3._timed_apply(  # pylint: disable=protected-access
        apply_fn, *common, full_intervention, target_selection
    )
    repeated, repeated_seconds = route_v3._timed_apply(  # pylint: disable=protected-access
        apply_fn, *common, full_intervention, target_selection
    )
    checks = validate_active_call(
        first, identity_means, selection, full_channels, repeated
    )
    closure = _spatial_closure(case)
    observed = np.asarray(checks['target_readout']['means'])
    expected = np.asarray(closure['checks']['target_readout']['means'])
    observed_b = checks['recovery']['bidirectional_bottleneck']
    expected_b = closure['checks']['recovery']['bidirectional_bottleneck']
    full = {
        'status': 'complete', 'fingerprint': full_fingerprint,
        'configuration': full_configuration, 'checks': checks,
        'spatial_V_closure': {
            'comparison_recorded': True,
            'source_fingerprint': closure['fingerprint'],
            'target_readout_exact': checks['target_readout']
            == closure['checks']['target_readout'],
            'maximum_absolute_target_mean_difference': float(
                np.max(np.abs(observed - expected))
            ),
            'bidirectional_bottleneck_difference': observed_b - expected_b,
        },
        'seconds': {'first': first_seconds, 'repeat': repeated_seconds},
        'created_at_unix_s': time.time(),
    }
    v2._write_atomic(full_path, full)  # pylint: disable=protected-access

  completed = 0
  group_dir = output / 'raw' / 'groups' / _case_stem(case)
  for group in groups:
    channels = channel_mask(group)
    group_intervention = interventions(
        selection, channels, active_positions=True
    )
    validate_runtime_contract(
        selection, group_intervention, channels, active_positions=True
    )
    configuration = {
        **base_configuration,
        'kind': 'channel_screen_without_group',
        'identity_fingerprint': identity['fingerprint'],
        'full_fingerprint': full['fingerprint'],
        'group': group,
    }
    path = group_dir / f"{group['group_index']:03d}_{group['group_id']}.json"
    record, group_fingerprint = _load_completed(path, configuration)
    if record is None:
      result, seconds = route_v3._timed_apply(  # pylint: disable=protected-access
          apply_fn, *common, group_intervention, target_selection
      )
      checks = validate_active_call(
          result, identity_means, selection, channels
      )
      record = {
          'status': 'complete', 'fingerprint': group_fingerprint,
          'configuration': configuration, 'checks': checks,
          'seconds': seconds, 'created_at_unix_s': time.time(),
      }
      v2._write_atomic(path, record)  # pylint: disable=protected-access
    completed += 1
  return completed


def build_dry_run(
    bindings: Sequence[tuple[v2.Case, Mapping[str, Any]]],
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
  return {
      'script_version': SCRIPT_VERSION,
      'development_only': True,
      'confirmation_access': False,
      'variant_count': len(bindings),
      'group_count': len(groups),
      'model_apply_count': len(bindings) * (4 + len(groups)),
      'fixed_position_shape': [7, spatial.SLOTS],
      'fixed_channel_shape': [7, planner.MAX_CHANNELS],
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
    raise ScreenError('Limits must be nonnegative.')
  plan, plan_sha256 = load_plan(args.plan)
  bindings = bind_cases(plan, route_v3.load_development_cases())
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
    raise ScreenError(f'Channel screen requires a GPU; observed {devices}.')
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
        model_instance, apply_fn, case, planned, groups, binding,
        args.output_dir,
    )
    completed += count
    print(
        f'completed {case.order:03d} {case.variant_id}: {count} groups',
        flush=True,
    )
  summary = {
      'status': 'complete',
      'binding': binding,
      'variant_count': len(bindings),
      'group_count': len(groups),
      'group_result_count': completed,
      'model_apply_count_in_full_nonresume_run': (
          len(bindings) * (4 + len(groups))
      ),
      'full_frozen_design_completed': (
          len(bindings) == 20 and len(groups) == 172
      ),
      'created_at_unix_s': time.time(),
  }
  v2._write_atomic(  # pylint: disable=protected-access
      args.output_dir / 'SUMMARY.json', summary
  )
  print(json.dumps(summary, indent=2))


if __name__ == '__main__':
  main()
