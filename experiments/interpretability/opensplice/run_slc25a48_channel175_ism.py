#!/usr/bin/env python3
"""Run controlled SLC25A48 acceptor edits through channel 175 and output."""

from __future__ import annotations

import argparse
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
from alphagenome_research.model import attention
from alphagenome_research.model import dna_model


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

# pylint: disable=wrong-import-position
import prepare_slc25a48_channel175_ism as planner
import run_inference_trace as v2
import run_route_census_v3 as route_v3
# pylint: enable=wrong-import-position


SCRIPT_VERSION = 'alphagenome-slc25a48-channel175-ism-v1.0.0'
DEFAULT_PLAN = HERE / 'slc25a48_channel175_ism_plan_v1.json'
DEFAULT_OUTPUT = HERE / 'results' / 'slc25a48_channel175_ism_v1'
ALPHABET = 'ACGT'
COMPONENTS = ('carried', 'first_update', 'second_update', 'output')


class IsmError(RuntimeError):
  """Raised when the design or a controlled-edit invariant fails."""


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plan(path: Path = DEFAULT_PLAN) -> tuple[dict[str, Any], str]:
  if not path.is_file() or path.is_symlink():
    raise IsmError(f'Plan must be a regular file: {path}.')
  try:
    plan = json.loads(path.read_text(encoding='utf-8'))
  except json.JSONDecodeError as error:
    raise IsmError(f'Plan is not valid JSON: {path}.') from error
  if plan != planner.build_plan():
    raise IsmError('Plan differs from deterministic source plan.')
  if (
      plan.get('schema_version')
      != 'alphagenome-slc25a48-channel175-ism-plan-v1'
      or plan.get('scope', {}).get('confirmation_access') is not False
      or plan.get('design', {}).get('candidate_edit_count') != 123
      or plan.get('design', {}).get('planned_model_apply_count') != 50
  ):
    raise IsmError('Frozen ISM design changed.')
  return plan, _sha256(path)


def anchor_case(cases: Sequence[v2.Case]) -> v2.Case:
  matches = [
      case for case in cases if case.variant_id == planner.ANCHOR_VARIANT
  ]
  if len(matches) != 1:
    raise IsmError('Anchor development case is not unique.')
  return matches[0]


def feature_selection(
    acceptor_position_1based: int, interval_start_0based: int
) -> tuple[jax.Array, jax.Array]:
  positions = np.zeros((7, len(planner.TRACE_OFFSETS)), np.int32)
  valid = np.zeros_like(positions, dtype=bool)
  genomic = [
      acceptor_position_1based + offset
      for offset in planner.TRACE_OFFSETS
  ]
  for stage, resolution in enumerate((1, 2, 4, 8, 16, 32, 64)):
    tokens = list(dict.fromkeys(
        (position - 1 - interval_start_0based) // resolution
        for position in genomic
    ))
    positions[stage, :len(tokens)] = tokens
    valid[stage, :len(tokens)] = True
  return jnp.asarray(positions), jnp.asarray(valid)


def build_candidates(
    reference: str, acceptor_index: int
) -> list[dict[str, Any]]:
  candidates = []
  for offset in planner.SCAN_OFFSETS:
    index = acceptor_index + offset
    reference_base = reference[index].upper()
    if reference_base not in ALPHABET:
      raise IsmError(f'Non-ACGT reference base at scan offset {offset}.')
    for alternate_base in ALPHABET:
      if alternate_base == reference_base:
        continue
      candidates.append({
          'candidate_index': len(candidates),
          'offset_from_acceptor_bp': offset,
          'sequence_index': index,
          'reference_base': reference_base,
          'alternate_base': alternate_base,
          'edit_id': f'{offset:+d}_{reference_base}>{alternate_base}',
      })
  if len(candidates) != 123:
    raise IsmError('Expected exactly 123 single-base candidates.')
  return candidates


def batch_layout(candidates: Sequence[Mapping[str, Any]]):
  size = planner.EDIT_ROWS_PER_BATCH
  result = []
  for start in range(0, len(candidates), size):
    edits = list(candidates[start:start + size])
    rows = [{'kind': 'reference', 'edit': None}]
    rows.extend({'kind': 'edit', 'edit': edit} for edit in edits)
    rows.extend(
        {'kind': 'reference_padding', 'edit': None}
        for _ in range(6 - len(rows))
    )
    result.append(rows)
  if len(result) != 25 or any(len(rows) != 6 for rows in result):
    raise IsmError('Unexpected fixed batch layout.')
  return result


def build_dry_run(plan: Mapping[str, Any]) -> dict[str, Any]:
  return {
      'script_version': SCRIPT_VERSION,
      'development_only': True,
      'confirmation_access': False,
      'candidate_edit_count': plan['design']['candidate_edit_count'],
      'batch_count': plan['design']['batch_count'],
      'batch_size': plan['design']['batch_size'],
      'encoder_model_apply_count': plan['design'][
          'encoder_model_apply_count'
      ],
      'full_model_apply_count': plan['design']['full_model_apply_count'],
      'model_apply_count': plan['design']['planned_model_apply_count'],
  }


def _binding(checkpoint: Path, plan_path: Path, plan_sha256: str):
  if checkpoint.name != route_v3.CHECKPOINT_SNAPSHOT:
    raise IsmError('Unexpected checkpoint snapshot.')
  repo = HERE.parents[2]
  files = (
      Path(__file__).resolve(),
      repo / 'src/alphagenome_research/model/convolutions.py',
      repo / 'src/alphagenome_research/model/interpretability.py',
      repo / 'src/alphagenome_research/model/model.py',
      repo / 'src/alphagenome_research/model/dna_model.py',
  )
  return {
      'script_version': SCRIPT_VERSION,
      'source_sha256': {
          str(path.relative_to(repo)): _sha256(path) for path in files
      },
      'plan_path': str(plan_path.resolve()),
      'plan_sha256': plan_sha256,
      'checkpoint_snapshot': checkpoint.name,
      'context_bp': route_v3.CONTEXT_BP,
      'confirmation_access': False,
  }


def _dna_batch(model_instance, reference: str, rows):
  encoder = model_instance._one_hot_encoder  # pylint: disable=protected-access
  sequences = []
  for row in rows:
    sequence = reference
    if row['kind'] == 'edit':
      edit = row['edit']
      index = edit['sequence_index']
      sequence = (
          reference[:index] + edit['alternate_base'] + reference[index + 1:]
      )
      if sum(left != right for left, right in zip(reference, sequence)) != 1:
        raise IsmError('Candidate is not exactly one SNV from reference.')
    sequences.append(np.asarray(encoder.encode(sequence), dtype=np.float32))
  return jnp.asarray(np.stack(sequences))


def _array(value) -> np.ndarray:
  return np.asarray(value, dtype=np.float32)


def _run_batch(
    model_instance, encoder_apply, target_apply, reference: str, rows,
    positions, valid, target_selection, configuration, path: Path,
):
  fingerprint = v2._fingerprint(configuration)  # pylint: disable=protected-access
  completed = v2._load_completed(path, fingerprint)  # pylint: disable=protected-access
  if completed is not None:
    return
  dna = _dna_batch(model_instance, reference, rows)
  params = model_instance._params  # pylint: disable=protected-access
  state = model_instance._state  # pylint: disable=protected-access
  start = time.perf_counter()
  feature = encoder_apply(
      params, state, dna, positions, valid, jnp.array([175], jnp.int32)
  )
  jax.block_until_ready(feature)
  encoder_seconds = time.perf_counter() - start
  start = time.perf_counter()
  target = target_apply(
      params, state, dna, jnp.zeros((6,), jnp.int32), target_selection
  )
  jax.block_until_ready(target)
  target_seconds = time.perf_counter() - start
  components = {
      name: _array(getattr(feature, name)[:2]).tolist()
      for name in COMPONENTS
  }
  margins = _array(target.margins)
  means = _array(target.target.mean)
  padding_rows = [
      index for index, row in enumerate(rows)
      if row['kind'] == 'reference_padding'
  ]
  reference_exact = all(
      np.array_equal(margins[index], margins[0])
      and means[index] == means[0]
      and all(
          np.array_equal(
              np.asarray(components[name])[:, index],
              np.asarray(components[name])[:, 0],
          )
          for name in COMPONENTS
      )
      for index in padding_rows
  )
  finite = bool(all(
      np.isfinite(np.asarray(value)).all() for value in components.values()
  ) and np.isfinite(margins).all() and np.isfinite(means).all())
  if not (reference_exact and finite):
    raise IsmError('Batch runtime control failed.')
  record = {
      'status': 'complete',
      'fingerprint': fingerprint,
      'configuration': configuration,
      'rows': rows,
      'checks': {
          'passed': True,
          'padding_reference_exact': reference_exact,
          'all_values_finite': finite,
      },
      'feature_axis_order': [
          'encoder_stage_E1_E2', 'batch_row', 'position_slot', 'channel'
      ],
      'feature_components': components,
      'target': {
          'endpoint_order': ['acceptor', 'donor'],
          'margins': margins.tolist(),
          'mean': means.tolist(),
      },
      'seconds': {
          'encoder': encoder_seconds,
          'full_model_target': target_seconds,
      },
      'created_at_unix_s': time.time(),
  }
  v2._write_atomic(path, record)  # pylint: disable=protected-access


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', type=Path)
  parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
  parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument('--max-batches', type=int, default=0)
  parser.add_argument('--dry-run', action='store_true')
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if args.max_batches < 0:
    raise IsmError('max-batches must be nonnegative.')
  plan, plan_sha256 = load_plan(args.plan)
  if args.dry_run:
    print(json.dumps(build_dry_run(plan), indent=2))
    return
  if not any(device.platform == 'gpu' for device in jax.devices()):
    raise IsmError('ISM requires a GPU.')
  case = anchor_case(route_v3.load_development_cases())
  checkpoint = v2._checkpoint_path(args.checkpoint)  # pylint: disable=protected-access
  binding = _binding(checkpoint, args.plan, plan_sha256)
  model_instance = dna_model.create(
      checkpoint,
      model_settings=dna_model.ModelSettings(
          attention_backend=attention.ATTENTION_BACKEND_DENSE
      ),
  )
  interval = v2.centered_interval(case, route_v3.CONTEXT_BP)
  extractor = model_instance._get_fasta_extractor(  # pylint: disable=protected-access
      public_dna_model.Organism.HOMO_SAPIENS
  )
  reference = extractor.extract(interval).upper()
  acceptor = case.exon_start_1based
  acceptor_index = acceptor - 1 - interval.start
  candidates = build_candidates(reference, acceptor_index)
  batches = batch_layout(candidates)
  if args.max_batches:
    batches = batches[:args.max_batches]
  positions, valid = feature_selection(acceptor, interval.start)
  metadata = model_instance._metadata[  # pylint: disable=protected-access
      public_dna_model.Organism.HOMO_SAPIENS
  ].splice_sites
  target_selection, resolved = route_v3.target_selection(
      metadata, case, interval
  )
  encoder_apply = jax.jit(
      dna_model.create_encoder_block_decomposition_apply(
          model_instance._metadata  # pylint: disable=protected-access
      )
  )
  target_apply = jax.jit(
      dna_model.create_splice_classification_logit_margin_evidence_apply(
          model_instance._metadata,  # pylint: disable=protected-access
          attention_backend=attention.ATTENTION_BACKEND_DENSE,
      )
  )
  common_configuration = {
      **binding,
      'anchor_case': v2._case_record(case),  # pylint: disable=protected-access
      'reference_sequence_sha256': hashlib.sha256(
          reference.encode('ascii')
      ).hexdigest(),
      'reference_acceptor_window_minus32_plus32': reference[
          acceptor_index - 32:acceptor_index + 33
      ],
      'acceptor_sequence_index': acceptor_index,
      'feature_positions': np.asarray(positions).tolist(),
      'feature_valid_mask': np.asarray(valid).tolist(),
      'target_endpoints': [
          {
              'role': endpoint.role,
              'position_1based': endpoint.position_1based,
              'position_index': endpoint.position_index,
              'track_index': endpoint.track_index,
          }
          for endpoint in resolved.endpoints
      ],
  }
  for batch_index, rows in enumerate(batches):
    configuration = {
        **common_configuration,
        'batch_index': batch_index,
        'rows': rows,
    }
    path = args.output_dir / 'raw' / f'batch_{batch_index:03d}.json'
    _run_batch(
        model_instance, encoder_apply, target_apply, reference, rows,
        positions, valid, target_selection, configuration, path,
    )
    print(f'completed batch {batch_index + 1}/{len(batches)}', flush=True)
  raw_paths = sorted((args.output_dir / 'raw').glob('batch_*.json'))
  records = [json.loads(path.read_text(encoding='utf-8')) for path in raw_paths]
  references = [
      (
          record['target']['margins'][0],
          [
              record['feature_components'][name][stage][0]
              for name in COMPONENTS for stage in range(2)
          ],
      )
      for record in records
  ]
  reference_across_batches_exact = all(
      value == references[0] for value in references
  ) if references else False
  candidate_count = sum(
      row['kind'] == 'edit'
      for record in records for row in record['rows']
  )
  summary = {
      'status': 'complete',
      'binding': binding,
      'batch_count': len(records),
      'candidate_edit_count': candidate_count,
      'full_frozen_design_completed': (
          len(records) == 25 and candidate_count == 123
      ),
      'model_apply_count_in_full_nonresume_run': len(batches) * 2,
      'reference_across_batches_exact': reference_across_batches_exact,
      'all_runtime_controls_passed': (
          reference_across_batches_exact
          and all(record['checks']['passed'] for record in records)
      ),
      'reference_acceptor_window_minus32_plus32': common_configuration[
          'reference_acceptor_window_minus32_plus32'
      ],
      'confirmation_access': False,
      'created_at_unix_s': time.time(),
  }
  v2._write_atomic(args.output_dir / 'SUMMARY.json', summary)  # pylint: disable=protected-access
  print(json.dumps(summary, indent=2))


if __name__ == '__main__':
  main()
