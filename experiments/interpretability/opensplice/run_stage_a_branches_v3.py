#!/usr/bin/env python3
"""Development-only Stage-A closure and whole T/E branch experiment.

This dependency-ordered first Stage-A slice runs two mandatory closure controls
before the branch partition: the final post-GELU A/D embedding and joint whole
transformer-output plus all-seven encoder-skip transfer.  Only after both close
bit-exactly does it measure isolated whole T and E branches and their Shapley
decomposition.  Confirmation variants are unavailable to this runner.
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
import run_phase_r_v3 as phase_r_v3  # pylint: disable=g-import-not-at-top
import run_route_census_v3 as route_v3  # pylint: disable=g-import-not-at-top


SCRIPT_VERSION = 'opensplice-stage-a-branches-v3.1.0'
COMPONENTS = (
    'final_embedding_A_D_closure',
    'joint_T_plus_E_closure',
    'whole_T',
    'whole_E',
)
DEFAULT_OUTPUT_DIR = (
    _HERE / 'results' / 'v3_development_stage_a_branches_dual_reference'
)
LOCKED_PHASE_R_DIR = (
    _HERE / 'results' / 'v3_development_phase_r_logit_margin'
)
LOCKED_PHASE_R_IDENTITY_TREE_SHA256 = (
    'ff7182be96e4b5be52e022e613ac16f476651924ff36d6a11b397b95613a3436'
)
LOCKED_PHASE_R_ANALYSIS_SHA256 = (
    '0131d591197fb187b9f291479e028c32c87313e40addd411235cb650df018a21'
)
CROSS_EXECUTABLE_TARGET_TOLERANCE = 2**-8
PHASE_R_RUNNER_PATH = _HERE / 'run_phase_r_v3.py'
V2_RUNNER_PATH = _HERE / 'run_inference_trace.py'
DUAL_REFERENCE_AMENDMENT_PATH = (
    _HERE
    / 'v3_wider_mechanism'
    / 'gate0_dual_reference_amendment_v3_1.md'
)


@dataclasses.dataclass(frozen=True)
class StageAComponent:
  order: int
  name: str
  transformer_output: bool
  encoder_skips: bool
  final_embedding: bool
  closure_required: bool

  @property
  def key(self) -> str:
    return f'{self.order:02d}_{self.name}'


def enumerate_components() -> tuple[StageAComponent, ...]:
  return (
      StageAComponent(0, COMPONENTS[0], False, False, True, True),
      StageAComponent(1, COMPONENTS[1], True, True, False, True),
      StageAComponent(2, COMPONENTS[2], True, False, False, False),
      StageAComponent(3, COMPONENTS[3], False, True, False, False),
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
      '--max-components',
      type=int,
      default=0,
      help='Dependency-ordered components per eligible effect; 0 means all 4.',
  )
  parser.add_argument('--dry-run', action='store_true')
  return parser.parse_args()


def branch_selection(resolved_target) -> interpretability.StageABranchSelection:
  endpoints = resolved_target.endpoints
  if tuple(endpoint.role for endpoint in endpoints) != ('acceptor', 'donor'):
    raise ValueError('Stage-A closure endpoints must be [acceptor, donor].')
  return interpretability.StageABranchSelection(
      final_embedding_positions=jnp.asarray(
          [endpoint.position_index for endpoint in endpoints], jnp.int32
      ),
      final_embedding_valid_mask=jnp.ones((2,), jnp.bool),
  )


def component_interventions(
    selection: interpretability.StageABranchSelection,
    component: StageAComponent | None,
) -> interpretability.StageABranchInterventions:
  identity = interpretability.no_stage_a_branch_interventions(
      selection, batch_size=interpretability.PAIRED_CAUSAL_BATCH_SIZE
  )
  if component is None:
    return identity
  transformer = interpretability.paired_six_row_whole_sequence_transfer(
      jnp.asarray([component.transformer_output], jnp.bool)
  )
  skips = interpretability.paired_six_row_whole_sequence_transfer(
      jnp.full(
          (len(interpretability.DECODER_ROUTE_RESOLUTIONS),),
          component.encoder_skips,
          jnp.bool,
      )
  )
  final_mask = jnp.full(
      (1, selection.final_embedding_positions.shape[0]),
      component.final_embedding,
      jnp.bool,
  )
  final_embedding = interpretability.paired_six_row_batch_transfer(final_mask)
  return interpretability.StageABranchInterventions(
      transformer_output=transformer,
      encoder_skips=skips,
      final_embedding=final_embedding,
  )


def _target_values(target: interpretability.TargetSummary) -> np.ndarray:
  return route_v3._target_values(target)  # pylint: disable=protected-access


def _trace_arrays(trace: interpretability.StageABranchTrace):
  return (
      ('transformer_output_natural_matches_identity',
       np.asarray(trace.transformer_output_natural_matches_identity)),
      ('transformer_output_effective_matches_natural',
       np.asarray(trace.transformer_output_effective_matches_natural)),
      ('transformer_output_effective_matches_intervention_donor',
       np.asarray(
           trace.transformer_output_effective_matches_intervention_donor
       )),
      ('transformer_output_natural_fingerprint',
       np.asarray(trace.transformer_output_natural_fingerprint)),
      ('encoder_skips_natural_match_identity',
       np.asarray(trace.encoder_skips_natural_match_identity)),
      ('encoder_skips_effective_match_natural',
       np.asarray(trace.encoder_skips_effective_match_natural)),
      ('encoder_skips_effective_match_intervention_donor',
       np.asarray(
           trace.encoder_skips_effective_match_intervention_donor
       )),
      ('encoder_skips_natural_fingerprints',
       np.asarray(trace.encoder_skips_natural_fingerprints)),
      ('natural_final_embeddings',
       np.asarray(trace.natural_final_embeddings)),
      ('effective_final_embeddings',
       np.asarray(trace.effective_final_embeddings)),
  )


def validate_identity_audit(
    first_target: interpretability.TargetSummary,
    first_trace: interpretability.StageABranchTrace,
    second_target: interpretability.TargetSummary,
    second_trace: interpretability.StageABranchTrace,
) -> dict[str, Any]:
  first = _target_values(first_target)
  second = _target_values(second_target)
  if not np.array_equal(first, second):
    raise ValueError('Stage-A Gate 0 target repeat is not exact.')
  if not (first[0] == first[4] == first[5]):
    raise ValueError('Stage-A Gate 0 REF duplicate targets differ.')
  if not (first[1] == first[2] == first[3]):
    raise ValueError('Stage-A Gate 0 ALT duplicate targets differ.')
  for (name, values), (_, repeated) in zip(
      _trace_arrays(first_trace), _trace_arrays(second_trace), strict=True
  ):
    if not np.array_equal(values, repeated):
      raise ValueError(f'Stage-A Gate 0 repeat failed at {name}.')
  natural_final = np.asarray(first_trace.natural_final_embeddings)
  for donor, recipients in ((0, (4, 5)), (1, (2, 3))):
    for recipient in recipients:
      if not np.array_equal(natural_final[donor], natural_final[recipient]):
        raise ValueError('Stage-A natural final duplicate trace failed.')
  if not np.array_equal(
      first_trace.natural_final_embeddings,
      first_trace.effective_final_embeddings,
  ):
    raise ValueError('All-false final embedding transfer is not a no-op.')
  if not np.asarray(
      first_trace.transformer_output_natural_matches_identity
  ).all():
    raise ValueError('All-false T natural duplicate audit must be true.')
  if not np.asarray(
      first_trace.transformer_output_effective_matches_natural
  ).all():
    raise ValueError('All-false T natural/effective no-op audit failed.')
  if not np.asarray(
      first_trace.encoder_skips_natural_match_identity
  ).all():
    raise ValueError('All-false E natural duplicate audit must be true.')
  if not np.asarray(
      first_trace.encoder_skips_effective_match_natural
  ).all():
    raise ValueError('All-false E natural/effective no-op audit failed.')
  return {
      'passed': True,
      'target_repeat_exact': True,
      'target_duplicate_rows_exact': True,
      'trace_repeat_exact': True,
      'natural_final_duplicate_rows_exact': True,
      'all_false_final_embedding_exact': True,
      'natural_T_duplicate_tensors_exact': True,
      'all_false_T_natural_effective_exact': True,
      'natural_E_duplicate_tensors_exact': True,
      'all_false_E_natural_effective_exact': True,
      'natural_T_fingerprint_repeat_exact': True,
      'natural_E_fingerprint_repeat_exact': True,
      'fingerprint_repeat_semantics': (
          'four uint32 reductions; catches ordinary drift but collisions are '
          'possible and it is not a cryptographic proof of tensor equality'
      ),
      'num_values': 2,
      'target_means': dict(
          zip(route_v3.TRACE_BATCH_ROLES, first.tolist(), strict=True)
      ),
  }


def validate_component_audit(
    target: interpretability.TargetSummary,
    trace: interpretability.StageABranchTrace,
    component: StageAComponent,
    identity_target_means: Sequence[float],
) -> dict[str, Any]:
  values = _target_values(target)
  identity = np.asarray(identity_target_means)
  if not np.array_equal(values[:2], identity[:2]):
    raise ValueError('Stage-A component baseline rows drifted from Gate 0.')
  if values[3] != values[1] or values[5] != values[0]:
    raise ValueError('Stage-A component self targets are not bit-exact.')

  recipient_rows = (2, 3, 4, 5)
  self_rows = (3, 5)
  transformer_natural = np.asarray(
      trace.transformer_output_natural_matches_identity
  )
  transformer_effective = np.asarray(
      trace.transformer_output_effective_matches_intervention_donor
  )
  transformer_noop = np.asarray(
      trace.transformer_output_effective_matches_natural
  )
  skip_natural = np.asarray(trace.encoder_skips_natural_match_identity)
  skip_effective = np.asarray(
      trace.encoder_skips_effective_match_intervention_donor
  )
  skip_noop = np.asarray(trace.encoder_skips_effective_match_natural)
  if component.transformer_output:
    if not transformer_natural[list(self_rows)].all():
      raise ValueError('Whole T natural self tensor audit failed.')
    if not transformer_effective[list(recipient_rows)].all():
      raise ValueError('Whole T effective donor tensor audit failed.')
  elif not transformer_noop.all():
    raise ValueError('Disabled whole T transfer is not an exact no-op.')
  if component.encoder_skips:
    if not skip_natural[:, list(self_rows)].all():
      raise ValueError('Whole E natural self tensor audit failed.')
    if not skip_effective[:, list(recipient_rows)].all():
      raise ValueError('Whole E effective donor tensor audit failed.')
  elif not skip_noop.all():
    raise ValueError('Disabled whole E transfer is not an exact no-op.')
  if component.final_embedding:
    natural = np.asarray(trace.natural_final_embeddings)
    effective = np.asarray(trace.effective_final_embeddings)
    for recipient, donor in ((2, 0), (3, 1), (4, 1), (5, 0)):
      if not np.array_equal(natural[donor], effective[recipient]):
        raise ValueError('Final A/D embedding donor-vector audit failed.')
  elif not np.array_equal(
      trace.natural_final_embeddings, trace.effective_final_embeddings
  ):
    raise ValueError('Disabled final embedding seam changed selected values.')

  ref, alt, ref_alt, alt_alt, alt_ref, ref_ref = values.tolist()
  closure = {
      'reference_into_alternate_target_equals_donor': ref_alt == ref,
      'alternate_into_reference_target_equals_donor': alt_ref == alt,
  }
  closure['passed'] = all(closure.values())
  if component.closure_required and not closure['passed']:
    raise ValueError(f'Mandatory Stage-A target closure failed: {closure}.')

  def recovery(patched, self_control, donor, recipient):
    denominator = donor - recipient
    return None if denominator == 0 else (patched - self_control) / denominator

  forward = recovery(ref_alt, alt_alt, ref, alt)
  reciprocal = recovery(alt_ref, ref_ref, alt, ref)
  return {
      'passed': True,
      'self_targets_exact': True,
      'transformer_natural_self_tensors_exact': bool(
          transformer_natural[list(self_rows)].all()
      ) if component.transformer_output else None,
      'transformer_effective_donor_tensors_exact': bool(
          transformer_effective[list(recipient_rows)].all()
      ) if component.transformer_output else None,
      'transformer_disabled_natural_effective_exact': (
          None if component.transformer_output else bool(transformer_noop.all())
      ),
      'all_seven_skip_natural_self_tensors_exact': bool(
          skip_natural[:, list(self_rows)].all()
      ) if component.encoder_skips else None,
      'all_seven_skip_effective_donor_tensors_exact': bool(
          skip_effective[:, list(recipient_rows)].all()
      ) if component.encoder_skips else None,
      'all_seven_skips_disabled_natural_effective_exact': (
          None if component.encoder_skips else bool(skip_noop.all())
      ),
      'final_embedding_donor_vectors_exact': (
          True if component.final_embedding else None
      ),
      'closure': closure if component.closure_required else None,
      'target_means': dict(
          zip(route_v3.TRACE_BATCH_ROLES, values.tolist(), strict=True)
      ),
      'raw_movement': {
          'reference_into_alternate': ref_alt - alt_alt,
          'alternate_into_reference': alt_ref - ref_ref,
      },
      'recovery': {
          'reference_into_alternate': forward,
          'alternate_into_reference': reciprocal,
          'bidirectional_bottleneck': (
              None if forward is None or reciprocal is None
              else min(forward, reciprocal)
          ),
      },
  }


def shapley_partition(
    identity: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
  """Computes raw and normalized two-route Shapley values in both directions."""
  empty = identity['checks']['target_means']
  t = results['whole_T']['checks']['target_means']
  e = results['whole_E']['checks']['target_means']
  joint = results['joint_T_plus_E_closure']['checks']['target_means']
  specifications = {
      'reference_into_alternate': {
          'empty': empty['alternate_baseline'],
          'donor': empty['reference_baseline'],
          'T': t['reference_into_alternate'],
          'E': e['reference_into_alternate'],
          'T_plus_E': joint['reference_into_alternate'],
      },
      'alternate_into_reference': {
          'empty': empty['reference_baseline'],
          'donor': empty['alternate_baseline'],
          'T': t['alternate_into_reference'],
          'E': e['alternate_into_reference'],
          'T_plus_E': joint['alternate_into_reference'],
      },
  }
  output = {}
  for direction, values in specifications.items():
    phi_t = 0.5 * (
        (values['T'] - values['empty'])
        + (values['T_plus_E'] - values['E'])
    )
    phi_e = 0.5 * (
        (values['E'] - values['empty'])
        + (values['T_plus_E'] - values['T'])
    )
    interaction = (
        values['T_plus_E']
        - values['T']
        - values['E']
        + values['empty']
    )
    denominator = values['donor'] - values['empty']
    output[direction] = {
        **values,
        'raw_phi_T': phi_t,
        'raw_phi_E': phi_e,
        'raw_interaction': interaction,
        'raw_joint_movement': values['T_plus_E'] - values['empty'],
        'normalized_phi_T': None if denominator == 0 else phi_t / denominator,
        'normalized_phi_E': None if denominator == 0 else phi_e / denominator,
        'normalized_interaction': (
            None if denominator == 0 else interaction / denominator
        ),
    }
  return output


def _tree_digest(paths: Sequence[Path], root: Path) -> str:
  digest = hashlib.sha256()
  for path in sorted(paths):
    digest.update(str(path.relative_to(root)).encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(
        v2._sha256(path)  # pylint: disable=protected-access
    ))
  return digest.hexdigest()


def load_locked_phase_r_identities(
    cases: Sequence[v2.Case],
) -> dict[str, Mapping[str, Any]]:
  """Loads only the frozen development identities and verifies their lock."""
  if any('confirm' in part.lower() for part in LOCKED_PHASE_R_DIR.parts):
    raise ValueError('Locked Phase-R path must be development-only.')
  analysis = LOCKED_PHASE_R_DIR / 'PHASE_R_ANALYSIS.json'
  observed_analysis_sha = v2._sha256(  # pylint: disable=protected-access
      analysis
  )
  if observed_analysis_sha != LOCKED_PHASE_R_ANALYSIS_SHA256:
    raise ValueError('Locked Phase-R analysis hash mismatch.')
  paths = sorted((LOCKED_PHASE_R_DIR / 'identity').glob('*.json'))
  if len(paths) != 20:
    raise ValueError(
        f'Expected 20 locked Phase-R identities, got {len(paths)}.'
    )
  observed_tree = _tree_digest(paths, LOCKED_PHASE_R_DIR)
  if observed_tree != LOCKED_PHASE_R_IDENTITY_TREE_SHA256:
    raise ValueError('Locked Phase-R identity tree hash mismatch.')

  allowed = {case.variant_id: case for case in cases}
  current_phase_r_sha = v2._sha256(  # pylint: disable=protected-access
      PHASE_R_RUNNER_PATH
  )
  current_v2_sha = v2._sha256(  # pylint: disable=protected-access
      V2_RUNNER_PATH
  )
  records = {}
  for path in paths:
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('status') != 'complete':
      raise ValueError(f'Incomplete locked Phase-R identity: {path.name}.')
    case_record = record.get('configuration', {}).get('case', {})
    variant_id = case_record.get('variant_id')
    if variant_id not in allowed:
      raise ValueError(f'Unexpected locked Phase-R variant: {variant_id!r}.')
    if case_record.get('gene') not in route_v3.DEVELOPMENT_GENES:
      raise ValueError('Locked Phase-R identity is outside development genes.')
    locked_configuration = record['configuration']
    if locked_configuration.get('phase_runner_sha256') != current_phase_r_sha:
      raise ValueError(
          f'Current Phase-R runner differs from lock for {variant_id}.'
      )
    if locked_configuration.get('v2_runner_sha256') != current_v2_sha:
      raise ValueError(
          f'Current v2 helper runner differs from lock for {variant_id}.'
      )
    if variant_id in records:
      raise ValueError(f'Duplicate locked Phase-R variant: {variant_id}.')
    means = record.get('checks', {}).get('target_means', {})
    if tuple(means) != route_v3.TRACE_BATCH_ROLES:
      raise ValueError(f'Invalid locked target roles for {variant_id}.')
    if not np.isfinite([float(means[role]) for role in means]).all():
      raise ValueError(f'Non-finite locked target for {variant_id}.')
    records[variant_id] = record
  if set(records) != set(allowed):
    raise ValueError(
        'Locked Phase-R identities do not match development cases.'
    )
  return records


def _json_normalized(value: Any) -> Any:
  return json.loads(json.dumps(
      v2._to_json(value),  # pylint: disable=protected-access
      sort_keys=True,
      separators=(',', ':'),
      allow_nan=False,
  ))


def validate_locked_phase_r_linkage(
    case: v2.Case,
    interval,
    resolved_target,
    sequence_sha256: Mapping[str, str],
    position_sets: Sequence[Any],
    locked_record: Mapping[str, Any],
) -> dict[str, Any]:
  """Binds a prospective reference call to its exact locked Phase-R inputs."""
  locked = locked_record['configuration']
  current = {
      'case': v2._case_record(case),  # pylint: disable=protected-access
      'interval': {
          'chromosome': interval.chromosome,
          'start_0based': interval.start,
          'end_0based_exclusive': interval.end,
      },
      'sequence_sha256': dict(sequence_sha256),
      'canonical_target': {
          'endpoints': [
              dataclasses.asdict(endpoint)
              for endpoint in resolved_target.endpoints
          ],
          'padding_track_index': resolved_target.padding_track_index,
      },
      'resolved_position_sets': [
          dataclasses.asdict(position_set) for position_set in position_sets
      ],
  }
  for name, value in current.items():
    if _json_normalized(value) != _json_normalized(locked.get(name)):
      raise ValueError(
          f'Current Phase-R {name} differs from lock for {case.variant_id}.'
      )
  return {
      'passed': True,
      'fields_exact': tuple(current),
      'phase_runner_sha256': locked['phase_runner_sha256'],
      'v2_runner_sha256': locked['v2_runner_sha256'],
  }


def validate_locked_phase_r_identity(
    case: v2.Case,
    reference_target_means: Sequence[float],
    locked_record: Mapping[str, Any],
) -> dict[str, Any]:
  """Fails closed if a current Phase-R graph drifts from its frozen identity."""
  reference = np.asarray(reference_target_means, np.float64)
  if reference.shape != (interpretability.PAIRED_CAUSAL_BATCH_SIZE,):
    raise ValueError(
        f'Expected six Phase-R reference target means, got {reference.shape}.'
    )
  locked_map = locked_record['checks']['target_means']
  locked = np.asarray(
      [locked_map[role] for role in route_v3.TRACE_BATCH_ROLES], np.float64
  )
  differences = np.abs(reference - locked)
  if not np.isfinite(reference).all() or not np.isfinite(locked).all():
    raise ValueError(
        'Cross-executable target identity contains non-finite values.'
    )
  if not (differences <= CROSS_EXECUTABLE_TARGET_TOLERANCE).all():
    raise ValueError(
        'Current Phase-R reference graph differs from its locked identity by '
        'more '
        f'than {CROSS_EXECUTABLE_TARGET_TOLERANCE}: {differences.tolist()}.'
    )

  reference_delta = float(reference[1] - reference[0])
  locked_delta = float(locked[1] - locked[0])
  denominator_gate = None
  if case.is_effect:
    experimental_sign = int(np.sign(case.delta_logit))
    denominator_gate = {
        'minimum_absolute_logit_margin_delta': route_v3.EFFECT_THRESHOLD,
        'experimental_delta_logit_sign': experimental_sign,
        'reference_graph_absolute_delta_passes': (
            abs(reference_delta) >= route_v3.EFFECT_THRESHOLD
        ),
        'locked_phase_r_absolute_delta_passes': (
            abs(locked_delta) >= route_v3.EFFECT_THRESHOLD
        ),
        'reference_graph_direction_matches': (
            int(np.sign(reference_delta)) == experimental_sign
        ),
        'locked_phase_r_direction_matches': (
            int(np.sign(locked_delta)) == experimental_sign
        ),
    }
    if not all(
        value for key, value in denominator_gate.items()
        if key not in (
            'minimum_absolute_logit_margin_delta',
            'experimental_delta_logit_sign',
        )
    ):
      raise ValueError(
          'Current/locked Phase-R denominator or direction gate failed: '
          f'{denominator_gate}.'
      )
  return {
      'passed': True,
      'tolerance': CROSS_EXECUTABLE_TARGET_TOLERANCE,
      'locked_phase_r_identity_fingerprint': locked_record['fingerprint'],
      'locked_target_means': dict(zip(
          route_v3.TRACE_BATCH_ROLES, locked.tolist(), strict=True
      )),
      'current_phase_r_reference_target_means': dict(zip(
          route_v3.TRACE_BATCH_ROLES, reference.tolist(), strict=True
      )),
      'absolute_differences': dict(zip(
          route_v3.TRACE_BATCH_ROLES, differences.tolist(), strict=True
      )),
      'locked_alt_minus_ref_logit_margin': locked_delta,
      'current_phase_r_alt_minus_ref_logit_margin': reference_delta,
      'effect_denominator_and_direction_gate': denominator_gate,
  }


def compare_stage_a_to_phase_r_reference(
    stage_target_means: Sequence[float],
    reference_target_means: Sequence[float],
) -> dict[str, Any]:
  """Records, but never uses, cross-graph BF16 drift for causal ranking."""
  stage = np.asarray(stage_target_means, np.float64)
  reference = np.asarray(reference_target_means, np.float64)
  expected = (interpretability.PAIRED_CAUSAL_BATCH_SIZE,)
  if stage.shape != expected or reference.shape != expected:
    raise ValueError('Dual-reference diagnostics require two six-row targets.')
  if not np.isfinite(stage).all() or not np.isfinite(reference).all():
    raise ValueError('Dual-reference target contains non-finite values.')
  differences = np.abs(stage - reference)
  return {
      'diagnostic_only_not_a_gate': True,
      'reason': (
          'the distinct Stage-A and Phase-R transformed graphs can compile '
          'to different BF16 arithmetic; Stage-A causal contrasts use only '
          'its exact within-graph baselines and self controls'
      ),
      'stage_a_target_means': dict(zip(
          route_v3.TRACE_BATCH_ROLES, stage.tolist(), strict=True
      )),
      'current_phase_r_reference_target_means': dict(zip(
          route_v3.TRACE_BATCH_ROLES, reference.tolist(), strict=True
      )),
      'absolute_differences': dict(zip(
          route_v3.TRACE_BATCH_ROLES, differences.tolist(), strict=True
      )),
      'maximum_absolute_difference': float(differences.max()),
      'frozen_cross_executable_tolerance_for_reference_graph': (
          CROSS_EXECUTABLE_TARGET_TOLERANCE
      ),
  }


def frozen_configuration(checkpoint: Path) -> dict[str, Any]:
  base = route_v3.frozen_configuration(checkpoint)
  protocol = _HERE / 'v3_wider_mechanism' / 'prospective_protocol_v3.md'
  return {
      **base,
      'script_version': SCRIPT_VERSION,
      'stage': 'A_dependency_slice_closure_and_whole_T_E',
      'runner_sha256': v2._sha256(  # pylint: disable=protected-access
          Path(__file__)
      ),
      'protocol_sha256': v2._sha256(  # pylint: disable=protected-access
          protocol
      ),
      'phase_r_runner_sha256': v2._sha256(  # pylint: disable=protected-access
          PHASE_R_RUNNER_PATH
      ),
      'v2_runner_sha256': v2._sha256(  # pylint: disable=protected-access
          V2_RUNNER_PATH
      ),
      'dual_reference_amendment': {
          'path': str(DUAL_REFERENCE_AMENDMENT_PATH.resolve()),
          'sha256': v2._sha256(  # pylint: disable=protected-access
              DUAL_REFERENCE_AMENDMENT_PATH
          ),
      },
      'components': COMPONENTS,
      'whole_tensor_host_policy': 'return_exact_boolean_audits_only',
      'dual_reference_design': {
          'semantic_reference': (
              'current create_splice_classification_logit_margin_route_census_'
              'apply with exact frozen Phase-R selection and all-false pytree'
          ),
          'causal_graph': (
              'Stage-A whole-branch apply with exact internal baselines and '
              'self controls'
          ),
          'cross_graph_stage_a_difference': 'diagnostic_only_not_a_gate',
      },
      'locked_phase_r': {
          'path': str(LOCKED_PHASE_R_DIR.resolve()),
          'identity_tree_sha256': LOCKED_PHASE_R_IDENTITY_TREE_SHA256,
          'analysis_sha256': LOCKED_PHASE_R_ANALYSIS_SHA256,
          'target_tolerance': CROSS_EXECUTABLE_TARGET_TOLERANCE,
          'denominator_rule': (
              'for effects, abs(ALT-REF logit margin)>=0.01 and sign must '
              'match experimental delta_logit in both executables'
          ),
      },
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


def _artifact_path(output_dir: Path, case: v2.Case, component=None) -> Path:
  slug = v2._slug(case.variant_id)  # pylint: disable=protected-access
  case_name = f'{case.order:03d}_{slug}'
  if component is None:
    return output_dir / 'identity' / f'{case_name}.json'
  return output_dir / 'components' / case_name / f'{component.key}.json'


def _phase_r_reference_path(output_dir: Path, case: v2.Case) -> Path:
  slug = v2._slug(case.variant_id)  # pylint: disable=protected-access
  return output_dir / 'phase_r_reference' / f'{case.order:03d}_{slug}.json'


def _run_phase_r_reference(
    model_instance,
    phase_r_reference_apply_fn,
    case: v2.Case,
    frozen: Mapping[str, Any],
    output_dir: Path,
    locked_phase_r_identity: Mapping[str, Any],
) -> dict[str, Any]:
  interval = v2.centered_interval(case, route_v3.CONTEXT_BP)
  metadata = model_instance._metadata[  # pylint: disable=protected-access
      public_dna_model.Organism.HOMO_SAPIENS
  ].splice_sites
  target, resolved = route_v3.target_selection(metadata, case, interval)
  phase_r_position_sets = v2.trace_position_sets(case, interval)
  phase_r_selection = phase_r_v3.phase_r_trace_selection(
      phase_r_position_sets
  )
  build_batch = (  # pylint: disable=protected-access
      route_v3._build_six_row_batch
  )
  dna_batch, sequence_sha = build_batch(
      model_instance, case, interval
  )
  linkage = validate_locked_phase_r_linkage(
      case,
      interval,
      resolved,
      sequence_sha,
      phase_r_position_sets,
      locked_phase_r_identity,
  )
  configuration = {
      **_case_configuration(
          frozen, case, interval, resolved, sequence_sha
      ),
      'kind': 'exact_current_phase_r_gate0_dual_reference',
      'resolved_position_sets': [
          dataclasses.asdict(position_set)
          for position_set in phase_r_position_sets
      ],
      'locked_input_linkage': linkage,
      'locked_phase_r_identity_fingerprint': locked_phase_r_identity[
          'fingerprint'
      ],
  }
  fingerprint = (  # pylint: disable=protected-access
      v2._fingerprint(configuration)
  )
  path = _phase_r_reference_path(output_dir, case)
  completed = v2._load_completed(  # pylint: disable=protected-access
      path, fingerprint
  )
  organism = jnp.zeros((6,), jnp.int32)
  common = (
      model_instance._params,  # pylint: disable=protected-access
      model_instance._state,  # pylint: disable=protected-access
      dna_batch,
      organism,
      phase_r_selection,
  )
  if completed is None:
    timed_apply = route_v3._timed_apply  # pylint: disable=protected-access
    phase_r_interventions = phase_r_v3.group_interventions(
        phase_r_selection, None
    )
    phase_r_first, phase_r_first_seconds = timed_apply(
        phase_r_reference_apply_fn,
        *common,
        phase_r_interventions,
        target,
    )
    phase_r_second, phase_r_second_seconds = timed_apply(
        phase_r_reference_apply_fn,
        *common,
        phase_r_interventions,
        target,
    )
    phase_r_current_checks = route_v3.validate_identity_audit(
        phase_r_first[0],
        phase_r_first[1],
        phase_r_second[0],
        phase_r_second[1],
    )
    phase_r_comparison = validate_locked_phase_r_identity(
        case, np.asarray(phase_r_first[0].mean), locked_phase_r_identity
    )
    completed = {
        'status': 'complete',
        'fingerprint': fingerprint,
        'configuration': configuration,
        'checks': phase_r_current_checks,
        'phase_r_cross_executable_identity': phase_r_comparison,
        'direction_gate': route_v3.direction_gate(
            case, np.asarray(phase_r_first[0].mean)
        ),
        'seconds': {
            'first_compile_and_run': phase_r_first_seconds,
            'exact_repeat': phase_r_second_seconds,
        },
        'created_at_unix_s': time.time(),
    }
    v2._write_atomic(path, completed)  # pylint: disable=protected-access
  return completed


def _run_identity(
    model_instance,
    stage_apply_fn,
    case: v2.Case,
    frozen: Mapping[str, Any],
    output_dir: Path,
    phase_r_reference: Mapping[str, Any],
):
  interval = v2.centered_interval(case, route_v3.CONTEXT_BP)
  metadata = model_instance._metadata[  # pylint: disable=protected-access
      public_dna_model.Organism.HOMO_SAPIENS
  ].splice_sites
  target, resolved = route_v3.target_selection(metadata, case, interval)
  selection = branch_selection(resolved)
  dna_batch, sequence_sha = (  # pylint: disable=protected-access
      route_v3._build_six_row_batch(model_instance, case, interval)
  )
  if _json_normalized(sequence_sha) != _json_normalized(
      phase_r_reference['configuration']['sequence_sha256']
  ):
    raise ValueError('Stage-A sequence differs from its Phase-R reference.')
  configuration = {
      **_case_configuration(
          frozen, case, interval, resolved, sequence_sha
      ),
      'kind': 'stage_a_all_false_identity_duplicate_repeat',
      'phase_r_reference_fingerprint': phase_r_reference['fingerprint'],
  }
  fingerprint = v2._fingerprint(  # pylint: disable=protected-access
      configuration
  )
  path = _artifact_path(output_dir, case)
  completed = v2._load_completed(  # pylint: disable=protected-access
      path, fingerprint
  )
  organism = jnp.zeros((6,), jnp.int32)
  common = (
      model_instance._params,  # pylint: disable=protected-access
      model_instance._state,  # pylint: disable=protected-access
      dna_batch,
      organism,
      selection,
  )
  if completed is None:
    interventions = component_interventions(selection, None)
    timed_apply = route_v3._timed_apply  # pylint: disable=protected-access
    first, first_seconds = timed_apply(
        stage_apply_fn, *common, interventions, target
    )
    second, second_seconds = timed_apply(
        stage_apply_fn, *common, interventions, target
    )
    checks = validate_identity_audit(first[0], first[1], second[0], second[1])
    reference_means = tuple(
        phase_r_reference['checks']['target_means'][role]
        for role in route_v3.TRACE_BATCH_ROLES
    )
    diagnostic = compare_stage_a_to_phase_r_reference(
        np.asarray(first[0].mean), reference_means
    )
    completed = {
        'status': 'complete',
        'fingerprint': fingerprint,
        'configuration': configuration,
        'checks': checks,
        'stage_a_vs_phase_r_cross_graph_diagnostic': diagnostic,
        'direction_gate': route_v3.direction_gate(
            case, np.asarray(first[0].mean)
        ),
        'seconds': {
            'first_compile_and_run': first_seconds,
            'exact_repeat': second_seconds,
        },
        'created_at_unix_s': time.time(),
    }
    v2._write_atomic(path, completed)  # pylint: disable=protected-access
  return completed, common, target, selection, resolved


def _run_component(
    apply_fn,
    case: v2.Case,
    identity: Mapping[str, Any],
    common,
    target,
    selection,
    resolved,
    component: StageAComponent,
    frozen: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
  interval = v2.centered_interval(case, route_v3.CONTEXT_BP)
  configuration = {
      **_case_configuration(
          frozen,
          case,
          interval,
          resolved,
          identity['configuration']['sequence_sha256'],
      ),
      'kind': 'stage_a_live_whole_branch_or_closure',
      'gate0_fingerprint': identity['fingerprint'],
      'component': dataclasses.asdict(component),
  }
  fingerprint = (  # pylint: disable=protected-access
      v2._fingerprint(configuration)
  )
  path = _artifact_path(output_dir, case, component)
  completed = v2._load_completed(  # pylint: disable=protected-access
      path, fingerprint
  )
  if completed:
    return completed
  interventions = component_interventions(selection, component)
  timed_apply = route_v3._timed_apply  # pylint: disable=protected-access
  (batch_target, batch_trace), seconds = timed_apply(
      apply_fn, *common, interventions, target
  )
  checks = validate_component_audit(
      batch_target,
      batch_trace,
      component,
      tuple(
          identity['checks']['target_means'][role]
          for role in route_v3.TRACE_BATCH_ROLES
      ),
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
    cases: Sequence[v2.Case], components: Sequence[StageAComponent]
) -> dict[str, Any]:
  closure_count = sum(component.closure_required for component in components)
  branch_count = len(components) - closure_count
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
      'stage_a_identity_calls_per_variant': 2,
      'phase_r_semantic_reference_calls_per_variant': 2,
      'dual_reference_policy': (
          'locked threshold applies to an exact current Phase-R graph; '
          'Stage-A cross-graph difference is diagnostic only'
      ),
      'execution_order': (
          'all_20_current_phase_r_references',
          'all_20_stage_a_identities',
          'all_12_final_embedding_closures',
          'all_12_joint_T_plus_E_closures',
          'eligible_isolated_whole_T_and_E',
      ),
      'bounded_gpu_execution': 'forbidden_after_v3.1_amendment',
      'locked_phase_r_identity_tree_sha256': (
          LOCKED_PHASE_R_IDENTITY_TREE_SHA256
      ),
      'locked_phase_r_analysis_sha256': LOCKED_PHASE_R_ANALYSIS_SHA256,
      'phase_r_runner_sha256': v2._sha256(  # pylint: disable=protected-access
          PHASE_R_RUNNER_PATH
      ),
      'v2_runner_sha256': v2._sha256(  # pylint: disable=protected-access
          V2_RUNNER_PATH
      ),
      'dual_reference_amendment_sha256': (
          v2._sha256(  # pylint: disable=protected-access
              DUAL_REFERENCE_AMENDMENT_PATH
          )
      ),
      'cross_executable_target_tolerance': (
          CROSS_EXECUTABLE_TARGET_TOLERANCE
      ),
      'components': [dataclasses.asdict(component) for component in components],
      'dependency_order': COMPONENTS,
      'variant_count': len(cases),
      'effect_count': sum(case.is_effect for case in cases),
      'neutral_count': sum(not case.is_effect for case in cases),
      'mandatory_closure_calls_per_effect': closure_count,
      'isolated_branch_calls_per_eligible_effect': branch_count,
      'component_calls_per_eligible_effect': len(components),
      'confirmation_access': 'disabled',
      'variants': [
          {
              'order': case.order,
              'gene': case.gene,
              'variant_id': case.variant_id,
              'selection_class': case.selection_class,
          }
          for case in cases
      ],
      'remaining_stage_a_work': (
          'receptive_field skip supports, decoder ancestral supports, '
          'pre-normalization output addends, shifted/random controls'
      ),
  }


def mandatory_closure_cases(
    cases: Sequence[v2.Case],
) -> tuple[v2.Case, ...]:
  """Returns every frozen effect, independent of direction eligibility."""
  return tuple(case for case in cases if case.is_effect)


def isolated_branch_cases(
    cases: Sequence[v2.Case], identities: Mapping[str, Mapping[str, Any]]
) -> tuple[v2.Case, ...]:
  """Applies the Phase-R eligibility gate only to isolated/ranking branches."""
  return tuple(
      case
      for case in cases
      if case.is_effect
      and identities[case.variant_id]['direction_gate'][
          'eligible_for_causal_census'
      ]
  )


def main() -> None:
  args = _parse_args()
  if args.max_variants < 0 or args.max_components < 0:
    raise ValueError('Limits must be nonnegative.')
  all_cases = route_v3.load_development_cases()
  locked_phase_r_identities = load_locked_phase_r_identities(all_cases)
  cases = all_cases[:args.max_variants] if args.max_variants else all_cases
  components = enumerate_components()
  if args.max_components:
    components = components[:args.max_components]
  if args.dry_run:
    print(json.dumps(
        v2._to_json(  # pylint: disable=protected-access
            build_dry_run_plan(cases, components)
        ),
        indent=2,
        allow_nan=False,
    ))
    return
  if args.max_variants or args.max_components:
    raise ValueError(
        'v3.1 GPU execution is all-or-nothing: all 20 references and '
        'identities must pass before active components.'
    )

  checkpoint = (  # pylint: disable=protected-access
      v2._checkpoint_path(args.checkpoint)
  )
  frozen = frozen_configuration(checkpoint)
  model_instance = dna_model.create(
      checkpoint,
      model_settings=dna_model.ModelSettings(
          attention_backend=route_v3.ATTENTION_BACKEND
      ),
  )
  stage_apply_fn = (
      dna_model.create_splice_classification_logit_margin_stage_a_branch_apply(
          model_instance._metadata,  # pylint: disable=protected-access
          attention_backend=route_v3.ATTENTION_BACKEND,
      )
  )
  stage_apply_fn = jax.jit(stage_apply_fn)
  phase_r_reference_apply_fn = jax.jit(
      dna_model.create_splice_classification_logit_margin_route_census_apply(
          model_instance._metadata,  # pylint: disable=protected-access
          attention_backend=route_v3.ATTENTION_BACKEND,
      )
  )

  # Pass 1 is intentionally complete before the first Stage-A identity call.
  phase_r_references = {}
  for case in cases:
    phase_r_references[case.variant_id] = _run_phase_r_reference(
        model_instance,
        phase_r_reference_apply_fn,
        case,
        frozen,
        args.output_dir,
        locked_phase_r_identities[case.variant_id],
    )

  # Pass 2 completes all Stage-A identities before any active component.
  identities = {}
  live_inputs = {}
  for case in cases:
    identity, common, target, selection, resolved = _run_identity(
        model_instance,
        stage_apply_fn,
        case,
        frozen,
        args.output_dir,
        phase_r_references[case.variant_id],
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
          'Stage A requires at least three eligible effects per development '
          f'gene; observed {eligible_by_gene}.'
      )

  eligible_cases = isolated_branch_cases(cases, identities)
  effect_cases = mandatory_closure_cases(cases)
  component_results = {case.variant_id: {} for case in effect_cases}
  closure_components = tuple(
      component for component in components if component.closure_required
  )
  branch_components = tuple(
      component for component in components if not component.closure_required
  )

  # Every available mandatory closure is completed across the entire
  # development cohort before any isolated branch result is produced.
  for component in closure_components:
    for case in effect_cases:
      identity = identities[case.variant_id]
      common, target, selection, resolved = live_inputs[case.variant_id]
      component_results[case.variant_id][component.name] = _run_component(
          stage_apply_fn,
          case,
          identity,
          common,
          target,
          selection,
          resolved,
          component,
          frozen,
          args.output_dir,
      )

  for case in eligible_cases:
    identity = identities[case.variant_id]
    common, target, selection, resolved = live_inputs[case.variant_id]
    results = component_results[case.variant_id]
    for component in branch_components:
      results[component.name] = _run_component(
          stage_apply_fn,
          case,
          identity,
          common,
          target,
          selection,
          resolved,
          component,
          frozen,
          args.output_dir,
      )
    required = {'joint_T_plus_E_closure', 'whole_T', 'whole_E'}
    if required.issubset(results):
      partition = {
          'status': 'complete',
          'case': v2._case_record(case),  # pylint: disable=protected-access
          'identity_fingerprint': identity['fingerprint'],
          'component_fingerprints': {
              name: results[name]['fingerprint'] for name in sorted(required)
          },
          'shapley': shapley_partition(identity, results),
          'created_at_unix_s': time.time(),
      }
      slug = v2._slug(case.variant_id)  # pylint: disable=protected-access
      v2._write_atomic(  # pylint: disable=protected-access
          args.output_dir / 'partitions' / f'{case.order:03d}_{slug}.json',
          partition,
      )

  summary = {
      'status': 'complete',
      'script_version': SCRIPT_VERSION,
      'partition': 'development_only',
      'runner_sha256': frozen['runner_sha256'],
      'phase_r_runner_sha256': frozen['phase_r_runner_sha256'],
      'v2_runner_sha256': frozen['v2_runner_sha256'],
      'dual_reference_amendment': frozen['dual_reference_amendment'],
      'locked_phase_r': frozen['locked_phase_r'],
      'variant_count': len(cases),
      'completed_phase_r_reference_count': len(phase_r_references),
      'completed_stage_a_identity_count': len(identities),
      'eligible_effect_count': sum(
          identities[case.variant_id]['direction_gate'][
              'eligible_for_causal_census'
          ]
          for case in cases
      ),
      'completed_component_count': sum(
          len(results) for results in component_results.values()
      ),
      'effect_count': len(effect_cases),
      'mandatory_closures_apply_to': 'every frozen development effect',
      'component_limit_per_eligible_effect': len(components),
      'mandatory_closure_components': COMPONENTS[:2],
      'remaining_stage_a_work': (
          'receptive_field skip supports, decoder ancestral supports, '
          'pre-normalization output addends, shifted/random controls'
      ),
      'created_at_unix_s': time.time(),
  }
  v2._write_atomic(  # pylint: disable=protected-access
      args.output_dir / 'summary.json', summary
  )
  print((args.output_dir / 'summary.json').resolve())


if __name__ == '__main__':
  main()
