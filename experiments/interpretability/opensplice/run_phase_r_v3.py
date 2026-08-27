#!/usr/bin/env python3
"""Development-only Phase-R rerun of the frozen v2 residual grid.

Only the scalar target changes: this runner uses the canonical acceptor/donor
splice-classification pre-softmax logit margin.  The residual experiment stays
fixed to transformer layers 0--5, the pre/post-attention and post-MLP seams,
and the v2 V/A/D/S position sets plus their exact shifted controls.
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


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))
import run_inference_trace as v2  # pylint: disable=g-import-not-at-top
import run_route_census_v3 as route_v3  # pylint: disable=g-import-not-at-top


SCRIPT_VERSION = 'opensplice-phase-r-v3.0.0'
STAGES = ('pre_attention', 'post_attention', 'post_mlp')
LAYERS = tuple(range(6))
CANDIDATE_POSITION_SETS = ('V', 'A', 'D', 'S')
POSITION_SET_NAMES = (
    'V',
    'A',
    'D',
    'S',
    'V_control_upstream',
    'V_control_downstream',
    'A_control_upstream',
    'A_control_downstream',
    'D_control_upstream',
    'D_control_downstream',
    'S_control_upstream',
    'S_control_downstream',
)
EXPECTED_CANDIDATES = 72
EXPECTED_GROUPS = 216
DEFAULT_OUTPUT_DIR = (
    _HERE / 'results' / 'v3_development_phase_r_logit_margin'
)


@dataclasses.dataclass(frozen=True)
class ResidualGridGroup:
  order: int
  stage: str
  layer: int
  position_set: v2.TracePositionSet

  @property
  def is_candidate(self) -> bool:
    return self.position_set.name in CANDIDATE_POSITION_SETS

  @property
  def key(self) -> str:
    return (
        f'{self.order:03d}_{self.stage}_layer{self.layer:02d}_'
        f'{self.position_set.name}'
    )


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', type=Path)
  parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument(
      '--max-variants',
      type=int,
      default=0,
      help='Development variants to audit; 0 means all 20.',
  )
  parser.add_argument(
      '--max-groups',
      type=int,
      default=0,
      help='Residual groups per eligible effect; 0 means all 216.',
  )
  parser.add_argument('--dry-run', action='store_true')
  return parser.parse_args()


def enumerate_groups(
    case: v2.Case, interval
) -> tuple[ResidualGridGroup, ...]:
  """Returns the exact v2 stage-major, layer-major 216-call grid."""
  position_sets = v2.trace_position_sets(case, interval)
  observed_names = tuple(item.name for item in position_sets)
  if observed_names != POSITION_SET_NAMES:
    raise ValueError(
        f'Frozen v2 position-set order changed: {observed_names}.'
    )
  groups = tuple(
      ResidualGridGroup(
          order=order,
          stage=stage,
          layer=layer,
          position_set=position_set,
      )
      for order, (stage, layer, position_set) in enumerate(
          (stage, layer, position_set)
          for stage in STAGES
          for layer in LAYERS
          for position_set in position_sets
      )
  )
  if len(groups) != EXPECTED_GROUPS:
    raise ValueError(f'Expected {EXPECTED_GROUPS} Phase-R groups.')
  if sum(group.is_candidate for group in groups) != EXPECTED_CANDIDATES:
    raise ValueError(f'Expected {EXPECTED_CANDIDATES} Phase-R candidates.')
  return groups


def phase_r_trace_selection(
    position_sets: Sequence[v2.TracePositionSet],
) -> interpretability.CausalRouteTraceSelection:
  """Wraps the unchanged v2 transformer selection with disabled route slots."""
  transformer = v2.transformer_trace_selection(position_sets)

  def disabled(num_stages: int):
    return (
        jnp.zeros((num_stages, 1), jnp.int32),
        jnp.zeros((num_stages, 1), jnp.bool),
    )

  encoder_positions, encoder_valid = disabled(
      len(interpretability.ENCODER_ROUTE_RESOLUTIONS)
  )
  decoder_positions, decoder_valid = disabled(
      len(interpretability.DECODER_ROUTE_RESOLUTIONS)
  )
  final_positions, final_valid = disabled(
      len(interpretability.FINAL_EMBEDDING_ROUTE_RESOLUTIONS)
  )
  return interpretability.CausalRouteTraceSelection(
      transformer=transformer,
      encoder_positions=encoder_positions,
      encoder_valid_mask=encoder_valid,
      decoder_skip_positions=decoder_positions,
      decoder_skip_valid_mask=decoder_valid,
      decoder_output_positions=decoder_positions,
      decoder_output_valid_mask=decoder_valid,
      final_embedding_positions=final_positions,
      final_embedding_valid_mask=final_valid,
  )


def group_interventions(
    selection: interpretability.CausalRouteTraceSelection,
    group: ResidualGridGroup | None,
) -> interpretability.CausalRouteInterventions:
  """Builds an all-false identity or one exact v2 live residual transfer."""
  identity = interpretability.no_causal_route_interventions(
      selection,
      batch_size=interpretability.PAIRED_CAUSAL_BATCH_SIZE,
      num_edges=v2.PAIR_PADDING_SIZE,
  )
  if group is None:
    return identity
  transformer = v2._live_batch_residual_transfer(  # pylint: disable=protected-access
      identity.transformer,
      stage=group.stage,
      layer=group.layer,
      slots=group.position_set.slots,
  )
  return dataclasses.replace(identity, transformer=transformer)


def _active_slot_mask(group: ResidualGridGroup) -> np.ndarray:
  mask = np.zeros((v2.NUM_TRACE_SLOTS,), bool)
  mask[np.asarray(group.position_set.slots, np.int32)] = True
  return mask


def phase_configuration(checkpoint: Path) -> dict[str, Any]:
  """Freezes the target-only change and exact inherited v2 grid."""
  configuration = route_v3.frozen_configuration(checkpoint)
  protocol_path = _HERE / 'v3_wider_mechanism' / 'prospective_protocol_v3.md'
  target_protocol_path = _HERE / 'target_readout_protocol_v3.md'
  return {
      **configuration,
      'script_version': SCRIPT_VERSION,
      'phase': 'R_readout_isolation',
      'phase_runner_sha256': v2._sha256(  # pylint: disable=protected-access
          Path(__file__)
      ),
      'v2_runner_sha256': v2._sha256(  # pylint: disable=protected-access
          _HERE / 'run_inference_trace.py'
      ),
      'prospective_protocol_sha256': v2._sha256(  # pylint: disable=protected-access
          protocol_path
      ),
      'target_protocol_sha256': v2._sha256(  # pylint: disable=protected-access
          target_protocol_path
      ),
      'residual_grid': {
          'stages': STAGES,
          'layers': LAYERS,
          'candidate_position_sets': CANDIDATE_POSITION_SETS,
          'resolved_position_sets_per_variant': POSITION_SET_NAMES,
          'candidate_count': EXPECTED_CANDIDATES,
          'executed_groups_per_eligible_effect': EXPECTED_GROUPS,
          'control_start_distance_tokens': v2.CONTROL_START_DISTANCE_TOKENS,
      },
      'non_transformer_route_transfers': 'all_false_not_enumerated',
  }


def _case_configuration(
    frozen: Mapping[str, Any],
    case: v2.Case,
    interval,
    resolved_target,
    sequence_sha256: Mapping[str, str],
) -> dict[str, Any]:
  return {
      **frozen,
      'case': v2._case_record(case),  # pylint: disable=protected-access
      'interval': {
          'chromosome': interval.chromosome,
          'start_0based': interval.start,
          'end_0based_exclusive': interval.end,
      },
      'exon': {
          'start_1based': case.exon_start_1based,
          'end_1based': case.exon_end_1based,
          'strand': case.strand,
      },
      'canonical_target': {
          'endpoints': [
              dataclasses.asdict(endpoint)
              for endpoint in resolved_target.endpoints
          ],
          'padding_track_index': resolved_target.padding_track_index,
      },
      'sequence_sha256': sequence_sha256,
  }


def _artifact_path(
    output_dir: Path,
    case: v2.Case,
    group: ResidualGridGroup | None = None,
) -> Path:
  slug = v2._slug(case.variant_id)  # pylint: disable=protected-access
  case_name = f'{case.order:03d}_{slug}'
  if group is None:
    return output_dir / 'identity' / f'{case_name}.json'
  return output_dir / 'groups' / case_name / f'{group.key}.json'


def _run_identity(
    model_instance,
    apply_fn,
    case: v2.Case,
    frozen: Mapping[str, Any],
    output_dir: Path,
):
  interval = v2.centered_interval(case, route_v3.CONTEXT_BP)
  position_sets = v2.trace_position_sets(case, interval)
  selection = phase_r_trace_selection(position_sets)
  metadata = model_instance._metadata[  # pylint: disable=protected-access
      public_dna_model.Organism.HOMO_SAPIENS
  ].splice_sites
  target, resolved_target = route_v3.target_selection(
      metadata, case, interval
  )
  # Reuses the exact aligned-SNV six-row DNA construction from route v3.
  dna_batch, sequence_sha256 = (  # pylint: disable=protected-access
      route_v3._build_six_row_batch(
          model_instance, case, interval
      )
  )
  configuration = {
      **_case_configuration(
          frozen,
          case,
          interval,
          resolved_target,
          sequence_sha256,
      ),
      'kind': 'phase_r_gate0_all_false_identity_duplicate_repeat',
      'resolved_position_sets': [
          dataclasses.asdict(position_set) for position_set in position_sets
      ],
  }
  fingerprint = v2._fingerprint(configuration)  # pylint: disable=protected-access
  path = _artifact_path(output_dir, case)
  completed = v2._load_completed(  # pylint: disable=protected-access
      path, fingerprint
  )
  organism = jnp.zeros((interpretability.PAIRED_CAUSAL_BATCH_SIZE,), jnp.int32)
  common = (
      model_instance._params,  # pylint: disable=protected-access
      model_instance._state,  # pylint: disable=protected-access
      dna_batch,
      organism,
      selection,
  )
  if completed is None:
    identity = group_interventions(selection, None)
    first_output = route_v3._timed_apply(  # pylint: disable=protected-access
        apply_fn, *common, identity, target
    )
    (first_target, first_trace), first_seconds = first_output
    second_output = route_v3._timed_apply(  # pylint: disable=protected-access
        apply_fn, *common, identity, target
    )
    (second_target, second_trace), second_seconds = second_output
    checks = route_v3.validate_identity_audit(
        first_target, first_trace, second_target, second_trace
    )
    gate = route_v3.direction_gate(case, np.asarray(first_target.mean))
    completed = {
        'status': 'complete',
        'fingerprint': fingerprint,
        'configuration': configuration,
        'checks': checks,
        'direction_gate': gate,
        'seconds': {
            'first_compile_and_run': first_seconds,
            'exact_repeat': second_seconds,
        },
        'created_at_unix_s': time.time(),
    }
    v2._write_atomic(path, completed)  # pylint: disable=protected-access
  return completed, common, target, selection, resolved_target


def _run_group(
    apply_fn,
    case: v2.Case,
    identity: Mapping[str, Any],
    common,
    target,
    selection,
    resolved_target,
    group: ResidualGridGroup,
    frozen: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
  interval = v2.centered_interval(case, route_v3.CONTEXT_BP)
  configuration = {
      **_case_configuration(
          frozen,
          case,
          interval,
          resolved_target,
          identity['configuration']['sequence_sha256'],
      ),
      'kind': 'phase_r_six_row_live_transformer_residual',
      'gate0_fingerprint': identity['fingerprint'],
      'stage': group.stage,
      'layer': group.layer,
      'position_set': dataclasses.asdict(group.position_set),
      'grid_order': group.order,
  }
  fingerprint = v2._fingerprint(configuration)  # pylint: disable=protected-access
  path = _artifact_path(output_dir, case, group)
  completed = v2._load_completed(  # pylint: disable=protected-access
      path, fingerprint
  )
  if completed:
    return completed
  interventions = group_interventions(selection, group)
  batch_output = route_v3._timed_apply(  # pylint: disable=protected-access
      apply_fn, *common, interventions, target
  )
  (batch_target, batch_trace), seconds = batch_output
  route_component = route_v3.RouteComponent(
      order=group.order,
      family=f'transformer_{group.stage}',
      stage=group.layer,
      resolution_bp=128,
      channel_width=1536,
      seam=group.stage,
  )
  checks = route_v3.validate_component_audit(
      batch_target,
      batch_trace,
      route_component,
      tuple(
          identity['checks']['target_means'][role]
          for role in route_v3.TRACE_BATCH_ROLES
      ),
      _active_slot_mask(group),
  )
  result = {
      'status': 'complete',
      'fingerprint': fingerprint,
      'configuration': configuration,
      'checks': checks,
      'seconds_compile_and_run': seconds,
      'created_at_unix_s': time.time(),
  }
  v2._write_atomic(path, result)  # pylint: disable=protected-access
  return result


def build_dry_run_plan(
    cases: Sequence[v2.Case], max_groups: int
) -> dict[str, Any]:
  per_variant = []
  for case in cases:
    interval = v2.centered_interval(case, route_v3.CONTEXT_BP)
    groups = enumerate_groups(case, interval)
    selected_groups = groups[:max_groups] if max_groups else groups
    per_variant.append({
        'order': case.order,
        'gene': case.gene,
        'variant_id': case.variant_id,
        'selection_class': case.selection_class,
        'resolved_position_sets': [
            dataclasses.asdict(position_set)
            for position_set in v2.trace_position_sets(case, interval)
        ],
        'bounded_group_count_if_eligible': len(selected_groups),
    })
  return {
      'script_version': SCRIPT_VERSION,
      'dry_run': True,
      'partition': 'development_only',
      'development_genes': route_v3.DEVELOPMENT_GENES,
      'selected_variants_sha256': route_v3.SELECTED_SHA256,
      'frozen_exons_sha256': route_v3.EXONS_SHA256,
      'context_bp': route_v3.CONTEXT_BP,
      'attention_backend': route_v3.ATTENTION_BACKEND,
      'target_head_key': 'splice_sites_classification/logits',
      'target_scalar': 'TargetSummary.mean',
      'identity_calls_per_variant': 2,
      'stages': STAGES,
      'layers': LAYERS,
      'candidate_position_sets': CANDIDATE_POSITION_SETS,
      'position_set_names': POSITION_SET_NAMES,
      'full_candidate_count': EXPECTED_CANDIDATES,
      'full_group_count_per_eligible_effect': EXPECTED_GROUPS,
      'bounded_group_count_per_eligible_effect': (
          min(max_groups, EXPECTED_GROUPS) if max_groups else EXPECTED_GROUPS
      ),
      'non_transformer_route_components': 0,
      'variant_count': len(cases),
      'effect_count': sum(case.is_effect for case in cases),
      'neutral_count': sum(not case.is_effect for case in cases),
      'variants': per_variant,
  }


def main() -> None:
  args = _parse_args()
  if args.max_variants < 0 or args.max_groups < 0:
    raise ValueError('Limits must be nonnegative.')
  all_cases = route_v3.load_development_cases()
  cases = all_cases[:args.max_variants] if args.max_variants else all_cases
  if args.dry_run:
    print(json.dumps(
        v2._to_json(  # pylint: disable=protected-access
            build_dry_run_plan(cases, args.max_groups)
        ),
        indent=2,
        allow_nan=False,
    ))
    return

  checkpoint = v2._checkpoint_path(args.checkpoint)  # pylint: disable=protected-access
  frozen = phase_configuration(checkpoint)
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

  identities = {}
  live_inputs = {}
  for case in cases:
    identity, common, target, selection, resolved = _run_identity(
        model_instance, apply_fn, case, frozen, args.output_dir
    )
    identities[case.variant_id] = identity
    live_inputs[case.variant_id] = (common, target, selection, resolved)

  if not args.max_variants:
    eligible_by_gene = {
        gene: sum(
            case.gene == gene
            and case.is_effect
            and identities[case.variant_id]['direction_gate'][
                'eligible_for_causal_census'
            ]
            for case in cases
        )
        for gene in route_v3.DEVELOPMENT_GENES
    }
    if any(count < 3 for count in eligible_by_gene.values()):
      raise ValueError(
          'Phase R requires at least three eligible effects per development '
          f'gene; observed {eligible_by_gene}.'
      )

  completed_groups = []
  for case in cases:
    identity = identities[case.variant_id]
    if not identity['direction_gate']['eligible_for_causal_census']:
      continue
    common, target, selection, resolved = live_inputs[case.variant_id]
    interval = v2.centered_interval(case, route_v3.CONTEXT_BP)
    groups = enumerate_groups(case, interval)
    if args.max_groups:
      groups = groups[:args.max_groups]
    for group in groups:
      completed_groups.append(_run_group(
          apply_fn,
          case,
          identity,
          common,
          target,
          selection,
          resolved,
          group,
          frozen,
          args.output_dir,
      ))

  summary = {
      'status': 'complete',
      'script_version': SCRIPT_VERSION,
      'partition': 'development_only',
      'variant_count': len(cases),
      'eligible_effect_count': sum(
          identities[case.variant_id]['direction_gate'][
              'eligible_for_causal_census'
          ]
          for case in cases
      ),
      'completed_group_count': len(completed_groups),
      'group_limit_per_eligible_effect': (
          min(args.max_groups, EXPECTED_GROUPS)
          if args.max_groups
          else EXPECTED_GROUPS
      ),
      'created_at_unix_s': time.time(),
  }
  v2._write_atomic(  # pylint: disable=protected-access
      args.output_dir / 'summary.json', summary
  )
  print((args.output_dir / 'summary.json').resolve())


if __name__ == '__main__':
  main()
