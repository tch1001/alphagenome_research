#!/usr/bin/env python3
"""Trace natural causal-channel terms and selected frozen encoder weights."""

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
import jax.numpy as jnp
import numpy as np

from alphagenome_research.model import attention
from alphagenome_research.model import dna_model


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import prepare_encoder_feature_decomposition as planner
import run_inference_trace as v2
import run_route_census_v3 as route_v3
# pylint: enable=wrong-import-position


SCRIPT_VERSION = 'alphagenome-encoder-feature-decomposition-v1.0.0'
DEFAULT_PLAN = HERE / 'encoder_feature_decomposition_plan_v1.json'
DEFAULT_OUTPUT = HERE / 'results' / 'encoder_feature_decomposition_v1'
CHANNELS = (3, 175)
COMPONENTS = ('carried', 'first_update', 'second_update', 'output')


class DecompositionError(RuntimeError):
  """Raised when a frozen design or runtime invariant fails."""


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plan(path: Path = DEFAULT_PLAN) -> tuple[dict[str, Any], str]:
  if not path.is_file() or path.is_symlink():
    raise DecompositionError(f'Plan must be a regular file: {path}.')
  try:
    plan = json.loads(path.read_text(encoding='utf-8'))
  except json.JSONDecodeError as error:
    raise DecompositionError(f'Plan is not valid JSON: {path}.') from error
  if plan != planner.build_plan():
    raise DecompositionError('Plan differs from deterministic source plan.')
  if (
      plan.get('schema_version')
      != 'alphagenome-encoder-feature-decomposition-plan-v1'
      or plan.get('scope', {}).get('confirmation_access') is not False
      or plan.get('design', {}).get('traced_channel_indices') != [3, 175]
      or plan.get('design', {}).get('planned_model_apply_count') != 40
  ):
    raise DecompositionError('Frozen decomposition design changed.')
  return plan, _sha256(path)


def bind_cases(
    plan: Mapping[str, Any], cases: Sequence[v2.Case]
) -> tuple[tuple[v2.Case, Mapping[str, Any]], ...]:
  if len(cases) != 20 or len(plan['cases']) != 20:
    raise DecompositionError('Expected 20 development cases.')
  result = []
  for case, planned in zip(cases, plan['cases'], strict=True):
    interval = v2.centered_interval(case, route_v3.CONTEXT_BP)
    if (
        case.order != planned['order']
        or case.gene != planned['gene']
        or case.variant_id != planned['variant_id']
        or case.selection_class != planned['selection_class']
        or interval.start != planned['interval_start_0based']
        or interval.end != planned['interval_end_0based_exclusive']
    ):
      raise DecompositionError(f'Case binding differs at {case.order}.')
    result.append((case, planned))
  return tuple(result)


def trace_selection(
    case: v2.Case, planned: Mapping[str, Any]
) -> tuple[jax.Array, jax.Array]:
  interval = v2.centered_interval(case, route_v3.CONTEXT_BP)
  route_selection = route_v3.route_selection(case, interval)
  positions = np.zeros((7, 3), np.int32)
  valid = np.zeros((7, 3), bool)
  if len(planned['stages']) != 7:
    raise DecompositionError('Expected seven encoder skip stages.')
  for stage, expected_resolution in zip(
      planned['stages'], planner.ENCODER_RESOLUTIONS, strict=True
  ):
    index = stage['stage_index']
    selected = stage['positions']
    if (
        index != planner.ENCODER_RESOLUTIONS.index(expected_resolution)
        or stage['resolution_bp'] != expected_resolution
        or stage['stage'] != f'E{expected_resolution}'
        or not 1 <= len(selected) <= 3
    ):
      raise DecompositionError('Encoder position design changed.')
    positions[index, :len(selected)] = selected
    valid[index, :len(selected)] = True
  if not np.array_equal(
      positions, np.asarray(route_selection.encoder_positions[:7])
  ) or not np.array_equal(
      valid, np.asarray(route_selection.encoder_valid_mask[:7])
  ):
    raise DecompositionError('Planned positions differ from route selection.')
  return jnp.asarray(positions), jnp.asarray(valid)


def build_dry_run(
    bindings: Sequence[tuple[v2.Case, Mapping[str, Any]]]
) -> dict[str, Any]:
  return {
      'script_version': SCRIPT_VERSION,
      'development_only': True,
      'confirmation_access': False,
      'variant_count': len(bindings),
      'natural_rows_per_variant': 2,
      'traced_channels': list(CHANNELS),
      'encoder_stage_count': 7,
      'components': list(COMPONENTS),
      'model_apply_count': len(bindings) * 2,
  }


def _binding(
    checkpoint: Path, plan_path: Path, plan_sha256: str
) -> dict[str, Any]:
  if checkpoint.name != route_v3.CHECKPOINT_SNAPSHOT:
    raise DecompositionError('Unexpected checkpoint snapshot.')
  repo = HERE.parents[2]
  source_paths = (
      Path(__file__).resolve(),
      repo / 'src/alphagenome_research/model/convolutions.py',
      repo / 'src/alphagenome_research/model/interpretability.py',
      repo / 'src/alphagenome_research/model/model.py',
      repo / 'src/alphagenome_research/model/dna_model.py',
  )
  return {
      'script_version': SCRIPT_VERSION,
      'source_sha256': {
          str(path.relative_to(repo)): _sha256(path) for path in source_paths
      },
      'plan_path': str(plan_path.resolve()),
      'plan_sha256': plan_sha256,
      'checkpoint_snapshot': checkpoint.name,
      'context_bp': route_v3.CONTEXT_BP,
      'confirmation_access': False,
  }


def _standardized_kernel(
    parameters: Mapping[str, jax.Array], target_channel: int
) -> np.ndarray:
  raw = np.asarray(parameters['w'][:, :, target_channel], dtype=np.float64)
  centered = raw - np.mean(raw)
  variance = np.var(raw)
  fan_in = raw.shape[0] * raw.shape[1]
  scale = float(np.asarray(parameters['scale'])[0, 0, target_channel])
  return centered * scale / math.sqrt(max(fan_in * variance, 1e-4))


def _kernel_summary(kernel: np.ndarray, target_channel: int) -> dict[str, Any]:
  strength = np.sqrt(np.sum(np.square(kernel), axis=0))
  top = np.argsort(-strength)[:32]
  return {
      'target_channel': target_channel,
      'shape': list(kernel.shape),
      'self_channel_spatial_weights': kernel[:, target_channel].tolist(),
      'top_input_channels_by_l2': [
          {
              'input_channel': int(index),
              'l2': float(strength[index]),
              'spatial_weights': kernel[:, index].tolist(),
          }
          for index in top
      ],
  }


def extract_weight_evidence(params: Mapping[str, Any]) -> dict[str, Any]:
  """Extracts reproducible selected-kernel evidence from the checkpoint."""
  direct_module = 'alphagenome/sequence_encoder/dna_embedder/conv1_d'
  direct = params[direct_module]
  raw = np.asarray(direct['w'][:, :, list(CHANNELS)], dtype=np.float64)
  bias = np.asarray(direct['b'], dtype=np.float64)[list(CHANNELS)]
  position_centered = raw - np.mean(raw, axis=1, keepdims=True)
  digest = hashlib.sha256()
  digest.update(raw.tobytes())
  digest.update(bias.tobytes())
  blocks = []
  for block_index, resolution in enumerate((2, 4, 8, 16, 32, 64)):
    branches = []
    for branch_name in ('conv_block', 'conv_block_1'):
      module = (
          f'alphagenome/sequence_encoder/downres_block_{block_index}/'
          f'{branch_name}/standardized_conv1_d'
      )
      summaries = []
      for target in CHANNELS:
        kernel = _standardized_kernel(params[module], target)
        digest.update(kernel.tobytes())
        summaries.append(_kernel_summary(kernel, target))
      branches.append({'branch': branch_name, 'module': module,
                       'targets': summaries})
    blocks.append({'output_resolution_bp': resolution, 'branches': branches})
  return {
      'schema_version': 'alphagenome-selected-encoder-weight-evidence-v1',
      'dna_alphabet': ['A', 'C', 'G', 'T'],
      'dna_kernel_offsets_bp': list(range(-7, 8)),
      'dna_direct_conv': {
          'module': direct_module,
          'channels': list(CHANNELS),
          'bias': bias.tolist(),
          'raw_kernel': raw.tolist(),
          'per_position_base_centered_kernel': position_centered.tolist(),
      },
      'downres_blocks': blocks,
      'selected_weight_evidence_sha256': digest.hexdigest(),
      'interpretation_limit': (
          'Direct and standardized kernels are model weights, but downstream '
          'normalization/nonlinearity means no kernel is a named motif alone.'
      ),
  }


def _trace_arrays(trace) -> dict[str, np.ndarray]:
  return {
      name: np.asarray(getattr(trace, name), dtype=np.float32)
      for name in COMPONENTS
  }


def _run_case(
    model_instance, apply_fn, case: v2.Case, planned: Mapping[str, Any],
    binding: Mapping[str, Any], output: Path,
) -> None:
  interval = v2.centered_interval(case, route_v3.CONTEXT_BP)
  dna, sequence_sha256 = route_v3._build_six_row_batch(  # pylint: disable=protected-access
      model_instance, case, interval
  )
  positions, valid = trace_selection(case, planned)
  configuration = {
      **binding,
      'case': v2._case_record(case),  # pylint: disable=protected-access
      'sequence_sha256': sequence_sha256,
      'positions': [stage['positions'] for stage in planned['stages']],
      'channels': list(CHANNELS),
  }
  fingerprint = v2._fingerprint(configuration)  # pylint: disable=protected-access
  path = output / 'raw' / f'{case.order:03d}_{v2._slug(case.variant_id)}.json'  # pylint: disable=protected-access
  completed = v2._load_completed(path, fingerprint)  # pylint: disable=protected-access
  if completed is not None:
    return
  common = (
      model_instance._params, model_instance._state,  # pylint: disable=protected-access
      dna[:2], positions, valid, jnp.asarray(CHANNELS, jnp.int32),
  )
  start = time.perf_counter()
  first = apply_fn(*common)
  jax.block_until_ready(first)
  first_seconds = time.perf_counter() - start
  start = time.perf_counter()
  repeated = apply_fn(*common)
  jax.block_until_ready(repeated)
  repeat_seconds = time.perf_counter() - start
  arrays = _trace_arrays(first)
  repeated_arrays = _trace_arrays(repeated)
  exact_repeat = all(
      np.array_equal(arrays[name], repeated_arrays[name])
      for name in COMPONENTS
  )
  finite = all(np.isfinite(value).all() for value in arrays.values())
  invalid_zero = all(
      np.count_nonzero(value * (~np.asarray(valid))[:, None, :, None]) == 0
      for value in arrays.values()
  )
  reconstruction = (
      arrays['carried'] + arrays['first_update'] + arrays['second_update']
  )
  max_residual = float(np.max(np.abs(arrays['output'] - reconstruction)))
  if not (exact_repeat and finite and invalid_zero):
    raise DecompositionError(f'Runtime control failed for {case.variant_id}.')
  record = {
      'status': 'complete',
      'fingerprint': fingerprint,
      'configuration': configuration,
      'checks': {
          'passed': True,
          'exact_repeat': exact_repeat,
          'all_values_finite': finite,
          'invalid_padded_positions_zero': invalid_zero,
          'maximum_float32_reconstruction_residual': max_residual,
          'reconstruction_note': (
              'Diagnostic only: separately returned bfloat16 terms can differ '
              'from their fused output by bfloat16 rounding.'
          ),
      },
      'axis_order': ['encoder_stage', 'allele_row', 'position_slot', 'channel'],
      'allele_rows': ['reference', 'alternate'],
      'components': {name: arrays[name].tolist() for name in COMPONENTS},
      'seconds': {'first': first_seconds, 'repeat': repeat_seconds},
      'created_at_unix_s': time.time(),
  }
  v2._write_atomic(path, record)  # pylint: disable=protected-access


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', type=Path)
  parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
  parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument('--max-variants', type=int, default=0)
  parser.add_argument('--dry-run', action='store_true')
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if args.max_variants < 0:
    raise DecompositionError('max-variants must be nonnegative.')
  plan, plan_sha256 = load_plan(args.plan)
  bindings = bind_cases(plan, route_v3.load_development_cases())
  if args.max_variants:
    bindings = bindings[:args.max_variants]
  if args.dry_run:
    print(json.dumps(build_dry_run(bindings), indent=2))
    return
  devices = jax.devices()
  if not any(device.platform == 'gpu' for device in devices):
    raise DecompositionError(f'GPU required; observed {devices}.')
  checkpoint = v2._checkpoint_path(args.checkpoint)  # pylint: disable=protected-access
  binding = _binding(checkpoint, args.plan, plan_sha256)
  model_instance = dna_model.create(
      checkpoint,
      model_settings=dna_model.ModelSettings(
          attention_backend=attention.ATTENTION_BACKEND_DENSE
      ),
  )
  apply_fn = jax.jit(dna_model.create_encoder_block_decomposition_apply(
      model_instance._metadata  # pylint: disable=protected-access
  ))
  for case, planned in bindings:
    _run_case(
        model_instance, apply_fn, case, planned, binding, args.output_dir
    )
    print(f'completed {case.order:03d} {case.variant_id}', flush=True)
  weights = extract_weight_evidence(
      model_instance._params  # pylint: disable=protected-access
  )
  v2._write_atomic(args.output_dir / 'WEIGHTS.json', weights)  # pylint: disable=protected-access
  raw_files = sorted((args.output_dir / 'raw').glob('*.json'))
  summary = {
      'status': 'complete',
      'binding': binding,
      'variant_count': len(raw_files),
      'full_frozen_design_completed': len(raw_files) == 20,
      'model_apply_count_in_full_nonresume_run': len(bindings) * 2,
      'all_runtime_controls_passed': all(
          json.loads(path.read_text(encoding='utf-8'))['checks']['passed']
          for path in raw_files
      ),
      'selected_weight_evidence_sha256': weights[
          'selected_weight_evidence_sha256'
      ],
      'confirmation_access': False,
      'created_at_unix_s': time.time(),
  }
  v2._write_atomic(args.output_dir / 'SUMMARY.json', summary)  # pylint: disable=protected-access
  print(json.dumps(summary, indent=2))


if __name__ == '__main__':
  main()
