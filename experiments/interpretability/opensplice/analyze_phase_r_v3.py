#!/usr/bin/env python3
"""Fail-closed CPU analyzer for development-only OpenSplice Phase R.

The analyzer reads only ``summary.json``, ``identity/*.json``, and
``groups/*/*.json`` from a Phase-R v3 run. It independently recomputes the
frozen v2 B/q/Q ranking and never imports the model or opens confirmation data.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence


ANALYSIS_VERSION = 'opensplice-phase-r-analysis-v3.0.0'
SOURCE_SCRIPT_VERSION = 'opensplice-phase-r-v3.0.0'
DEVELOPMENT_GENES = ('BRAF', 'SLC25A48')
STAGE_ORDER = ('pre_attention', 'post_attention', 'post_mlp')
LAYERS = tuple(range(6))
CANDIDATE_ORDER = ('V', 'A', 'D', 'S')
POSITION_SET_ORDER = (
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
TRACE_BATCH_ROLES = (
    'reference_baseline',
    'alternate_baseline',
    'reference_into_alternate',
    'alternate_into_alternate_self_control',
    'alternate_into_reference',
    'reference_into_reference_self_control',
)
DONOR_VECTOR_ROLES = TRACE_BATCH_ROLES[2:]
EXPECTED_IDENTITIES = 20
EXPECTED_EFFECTS_PER_GENE = 6
EXPECTED_NEUTRALS_PER_GENE = 4
EXPECTED_GROUPS_PER_ELIGIBLE_EFFECT = 216
EXPECTED_CANDIDATES = 72
EFFECT_THRESHOLD = 0.01


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run-dir', type=Path, required=True)
  parser.add_argument('--output-json', type=Path, required=True)
  parser.add_argument('--output-markdown', type=Path)
  return parser.parse_args()


def _guard_development_path(path: Path) -> None:
  if any('confirm' in part.lower() for part in path.resolve().parts):
    raise ValueError('Refusing to inspect a confirmation-named path.')


def _read_complete(path: Path) -> dict[str, Any]:
  try:
    value = json.loads(path.read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError) as error:
    raise ValueError(f'Cannot read JSON artifact {path}.') from error
  if not isinstance(value, dict) or value.get('status') != 'complete':
    raise ValueError(f'Artifact is not complete: {path}.')
  return value


def _canonical_bytes(value: Any) -> bytes:
  return json.dumps(
      value, sort_keys=True, separators=(',', ':'), allow_nan=False
  ).encode('utf-8')


def _fingerprint(configuration: Mapping[str, Any]) -> str:
  return hashlib.sha256(_canonical_bytes(configuration)).hexdigest()


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_digest(paths: Iterable[Path], root: Path) -> str:
  digest = hashlib.sha256()
  for path in sorted(paths):
    digest.update(str(path.relative_to(root)).encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(_sha256(path)))
  return digest.hexdigest()


def _finite_float(value: Any, label: str) -> float:
  if isinstance(value, bool):
    raise ValueError(f'{label} must be numeric, not boolean.')
  try:
    result = float(value)
  except (TypeError, ValueError) as error:
    raise ValueError(f'{label} is not numeric: {value!r}.') from error
  if not math.isfinite(result):
    raise ValueError(f'{label} is non-finite.')
  return result


def _same_number(observed: Any, expected: float, label: str) -> None:
  observed = _finite_float(observed, label)
  if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
    raise ValueError(f'{label} mismatch: {observed} != {expected}.')


def _median(values: Iterable[float]) -> float:
  values = list(values)
  if not values or any(not math.isfinite(value) for value in values):
    raise ValueError('A required median is empty or non-finite.')
  return float(statistics.median(values))


def _is_effect(case: Mapping[str, Any]) -> bool:
  selection_class = str(case.get('selection_class', '')).lower()
  if not selection_class:
    raise ValueError('Case has no selection_class.')
  return 'neutral' not in selection_class


def _require_true(
    mapping: Mapping[str, Any], fields: Sequence[str], label: str
):
  for field in fields:
    if mapping.get(field) is not True:
      raise ValueError(f'{label}.{field} is not true.')


def _target_means(record: Mapping[str, Any], label: str) -> dict[str, float]:
  try:
    values = record['checks']['target_means']
  except (KeyError, TypeError) as error:
    raise ValueError(f'{label} has no target means.') from error
  if not isinstance(values, Mapping) or set(values) != set(TRACE_BATCH_ROLES):
    raise ValueError(f'{label} has an invalid six-row target schema.')
  return {
      role: _finite_float(values[role], f'{label}.{role}')
      for role in TRACE_BATCH_ROLES
  }


def _common_configuration(configuration: Mapping[str, Any], kind: str) -> Any:
  excluded = {
      'case',
      'interval',
      'exon',
      'canonical_target',
      'sequence_sha256',
      'kind',
  }
  if kind == 'identity':
    excluded.add('resolved_position_sets')
  else:
    excluded.update({
        'gate0_fingerprint', 'stage', 'layer', 'position_set', 'grid_order'
    })
  return {
      key: value for key, value in configuration.items() if key not in excluded
  }


def _case_configuration(configuration: Mapping[str, Any], kind: str) -> Any:
  excluded = {'kind'}
  if kind == 'identity':
    excluded.add('resolved_position_sets')
  else:
    excluded.update({
        'gate0_fingerprint', 'stage', 'layer', 'position_set', 'grid_order'
    })
  return {
      key: value for key, value in configuration.items() if key not in excluded
  }


def _validate_shared_configuration(configuration: Mapping[str, Any]) -> None:
  if configuration.get('script_version') != SOURCE_SCRIPT_VERSION:
    raise ValueError('Unexpected Phase-R source script version.')
  if configuration.get('phase') != 'R_readout_isolation':
    raise ValueError('Artifact is not from Phase R.')
  if tuple(configuration.get('development_genes', ())) != DEVELOPMENT_GENES:
    raise ValueError('Artifact development-gene allowlist changed.')
  if configuration.get('target_head_key') != (
      'splice_sites_classification/logits'
  ):
    raise ValueError('Phase R did not use the locked internal logit head.')
  if configuration.get('context_bp') != 16_384:
    raise ValueError('Phase-R context is not the locked 16,384 bp.')
  if configuration.get('attention_backend') != 'dense':
    raise ValueError('Phase-R attention backend is not dense.')
  grid = configuration.get('residual_grid')
  if not isinstance(grid, Mapping):
    raise ValueError('Phase-R residual grid is missing.')
  expected = {
      'stages': STAGE_ORDER,
      'layers': LAYERS,
      'candidate_position_sets': CANDIDATE_ORDER,
      'resolved_position_sets_per_variant': POSITION_SET_ORDER,
  }
  for key, value in expected.items():
    if tuple(grid.get(key, ())) != value:
      raise ValueError(f'Frozen residual-grid field {key} changed.')
  if grid.get('candidate_count') != EXPECTED_CANDIDATES:
    raise ValueError('Frozen candidate count changed.')
  if grid.get('executed_groups_per_eligible_effect') != (
      EXPECTED_GROUPS_PER_ELIGIBLE_EFFECT
  ):
    raise ValueError('Frozen group count changed.')
  if configuration.get('non_transformer_route_transfers') != (
      'all_false_not_enumerated'
  ):
    raise ValueError('Non-transformer transfers were not disabled.')


def _validate_fingerprint(record: Mapping[str, Any], label: str) -> None:
  configuration = record.get('configuration')
  if not isinstance(configuration, Mapping):
    raise ValueError(f'{label} has no configuration.')
  observed = record.get('fingerprint')
  expected = _fingerprint(configuration)
  if observed != expected:
    raise ValueError(f'{label} configuration fingerprint mismatch.')


def _validate_identity(
    record: Mapping[str, Any], label: str
) -> dict[str, Any]:
  _validate_fingerprint(record, label)
  configuration = record['configuration']
  _validate_shared_configuration(configuration)
  if configuration.get('kind') != (
      'phase_r_gate0_all_false_identity_duplicate_repeat'
  ):
    raise ValueError(f'{label} has the wrong identity kind.')
  case = configuration.get('case')
  if not isinstance(case, Mapping):
    raise ValueError(f'{label} has no case.')
  gene = case.get('gene')
  if gene not in DEVELOPMENT_GENES:
    raise ValueError(f'{label} contains non-development gene {gene!r}.')
  checks = record.get('checks')
  if not isinstance(checks, Mapping):
    raise ValueError(f'{label} has no Gate-0 checks.')
  _require_true(
      checks,
      (
          'passed',
          'target_repeat_exact',
          'target_duplicate_rows_exact',
          'trace_repeat_exact',
          'trace_duplicate_rows_exact',
          'target_total_equals_two_times_mean',
      ),
      f'{label}.checks',
  )
  if checks.get('num_values') != 2:
    raise ValueError(f'{label} target is not the two-endpoint mean.')
  means = _target_means(record, label)
  ref = means['reference_baseline']
  alt = means['alternate_baseline']
  if not (
      ref
      == means['alternate_into_reference']
      == means['reference_into_reference_self_control']
  ):
    raise ValueError(f'{label} REF duplicate rows are not exact.')
  if not (
      alt
      == means['reference_into_alternate']
      == means['alternate_into_alternate_self_control']
  ):
    raise ValueError(f'{label} ALT duplicate rows are not exact.')
  delta = alt - ref
  experimental = _finite_float(case.get('delta_logit'), f'{label}.delta_logit')
  effect = _is_effect(case)
  same_sign = math.copysign(1.0, delta) == math.copysign(1.0, experimental)
  same_sign = same_sign and delta != 0 and experimental != 0
  expected_eligible = effect and abs(delta) >= EFFECT_THRESHOLD and same_sign
  gate = record.get('direction_gate')
  if not isinstance(gate, Mapping):
    raise ValueError(f'{label} has no direction gate.')
  _same_number(
      gate.get('predicted_alt_minus_ref_logit_margin'), delta,
      f'{label}.predicted_delta',
  )
  _same_number(
      gate.get('experimental_delta_logit'), experimental,
      f'{label}.experimental_delta',
  )
  _same_number(
      gate.get('minimum_absolute_predicted_effect'), EFFECT_THRESHOLD,
      f'{label}.effect_threshold',
  )
  expected_direction = same_sign if effect else None
  if gate.get('direction_matches_delta_logit') is not expected_direction:
    raise ValueError(f'{label} direction-match flag is inconsistent.')
  if gate.get('eligible_for_causal_census') is not expected_eligible:
    raise ValueError(f'{label} eligibility flag is inconsistent.')
  return {
      'case': dict(case),
      'fingerprint': record['fingerprint'],
      'means': means,
      'predicted_delta': delta,
      'effect': effect,
      'eligible': expected_eligible,
      'configuration': configuration,
  }


def _position_set_contract(
    value: Mapping[str, Any], expected_name: str
) -> None:
  if value.get('name') != expected_name:
    raise ValueError('Position-set name does not match its grid key.')
  candidate = expected_name.split('_control_', maxsplit=1)[0]
  is_control = '_control_' in expected_name
  expected_role = 'width_matched_control' if is_control else 'candidate'
  expected_match = candidate if is_control else None
  if value.get('role') != expected_role:
    raise ValueError(f'{expected_name} has an invalid position-set role.')
  if value.get('matched_candidate') != expected_match:
    raise ValueError(f'{expected_name} has an invalid matched candidate.')
  tokens = value.get('tokens')
  slots = value.get('slots')
  if (
      not isinstance(tokens, list)
      or not tokens
      or not isinstance(slots, list)
      or len(tokens) != len(slots)
  ):
    raise ValueError(f'{expected_name} has invalid tokens/slots.')


def _validate_group(
    record: Mapping[str, Any],
    label: str,
    identity: Mapping[str, Any],
) -> tuple[tuple[str, int, str], dict[str, float]]:
  _validate_fingerprint(record, label)
  configuration = record['configuration']
  _validate_shared_configuration(configuration)
  if configuration.get('kind') != (
      'phase_r_six_row_live_transformer_residual'
  ):
    raise ValueError(f'{label} has the wrong group kind.')
  if configuration.get('gate0_fingerprint') != identity['fingerprint']:
    raise ValueError(f'{label} does not link to its Gate-0 identity.')
  if _case_configuration(configuration, 'group') != _case_configuration(
      identity['configuration'], 'identity'
  ):
    raise ValueError(f'{label} configuration differs from Gate 0.')
  stage = configuration.get('stage')
  layer = configuration.get('layer')
  position_set = configuration.get('position_set')
  if stage not in STAGE_ORDER or layer not in LAYERS:
    raise ValueError(f'{label} has an out-of-grid stage/layer.')
  if not isinstance(position_set, Mapping):
    raise ValueError(f'{label} has no position-set definition.')
  name = position_set.get('name')
  if name not in POSITION_SET_ORDER:
    raise ValueError(f'{label} has unexpected position set {name!r}.')
  _position_set_contract(position_set, name)
  expected_order = (
      STAGE_ORDER.index(stage) * len(LAYERS) * len(POSITION_SET_ORDER)
      + int(layer) * len(POSITION_SET_ORDER)
      + POSITION_SET_ORDER.index(name)
  )
  if configuration.get('grid_order') != expected_order:
    raise ValueError(f'{label} has an invalid grid order.')
  checks = record.get('checks')
  if not isinstance(checks, Mapping):
    raise ValueError(f'{label} has no checks.')
  _require_true(
      checks,
      ('passed', 'baseline_targets_exact_from_gate0', 'self_targets_exact'),
      f'{label}.checks',
  )
  donors = checks.get('donor_vectors_exact')
  if not isinstance(donors, Mapping) or set(donors) != set(DONOR_VECTOR_ROLES):
    raise ValueError(f'{label} has an invalid donor-vector audit.')
  _require_true(donors, DONOR_VECTOR_ROLES, f'{label}.donor_vectors_exact')
  means = _target_means(record, label)
  identity_means = identity['means']
  ref = identity_means['reference_baseline']
  alt = identity_means['alternate_baseline']
  if means['reference_baseline'] != ref or means['alternate_baseline'] != alt:
    raise ValueError(f'{label} baseline targets differ from Gate 0.')
  if means['alternate_into_alternate_self_control'] != alt:
    raise ValueError(f'{label} ALT self target drifted.')
  if means['reference_into_reference_self_control'] != ref:
    raise ValueError(f'{label} REF self target drifted.')
  if alt == ref:
    raise ValueError(f'{label} has an undefined zero baseline effect.')
  ref_alt_movement = means['reference_into_alternate'] - alt
  alt_ref_movement = means['alternate_into_reference'] - ref
  raw = checks.get('raw_movement')
  if not isinstance(raw, Mapping):
    raise ValueError(f'{label} has no raw movement.')
  _same_number(
      raw.get('reference_into_alternate'), ref_alt_movement,
      f'{label}.ref_into_alt_movement',
  )
  _same_number(
      raw.get('alternate_into_reference'), alt_ref_movement,
      f'{label}.alt_into_ref_movement',
  )
  forward = ref_alt_movement / (ref - alt)
  reciprocal = alt_ref_movement / (alt - ref)
  bottleneck = min(forward, reciprocal)
  recovery = checks.get('self_control_corrected_recovery')
  if not isinstance(recovery, Mapping):
    raise ValueError(f'{label} has no recovery metrics.')
  _same_number(
      recovery.get('reference_into_alternate'), forward,
      f'{label}.forward_recovery',
  )
  _same_number(
      recovery.get('alternate_into_reference'), reciprocal,
      f'{label}.reciprocal_recovery',
  )
  _same_number(
      recovery.get('bidirectional_bottleneck'), bottleneck,
      f'{label}.bidirectional_bottleneck',
  )
  return (stage, int(layer), name), {
      'B': bottleneck,
      'forward_recovery': forward,
      'reciprocal_recovery': reciprocal,
  }


def _expected_group_keys() -> set[tuple[str, int, str]]:
  return {
      (stage, layer, position_set)
      for stage in STAGE_ORDER
      for layer in LAYERS
      for position_set in POSITION_SET_ORDER
  }


def _rank(
    identities: Mapping[str, Mapping[str, Any]],
    groups: Mapping[str, Mapping[tuple[str, int, str], Mapping[str, float]]],
) -> list[dict[str, Any]]:
  rankings = []
  for stage_index, stage in enumerate(STAGE_ORDER):
    for layer in LAYERS:
      for candidate_index, candidate in enumerate(CANDIDATE_ORDER):
        per_gene = {gene: [] for gene in DEVELOPMENT_GENES}
        for variant_id, identity in identities.items():
          if not identity['eligible']:
            continue
          metrics = groups[variant_id]
          candidate_b = metrics[(stage, layer, candidate)]['B']
          upstream_b = metrics[
              (stage, layer, f'{candidate}_control_upstream')
          ]['B']
          downstream_b = metrics[
              (stage, layer, f'{candidate}_control_downstream')
          ]['B']
          per_gene[identity['case']['gene']].append({
              'B': candidate_b,
              'q': candidate_b - max(upstream_b, downstream_b),
          })
        medians = {
            gene: {
                'median_B': _median(row['B'] for row in rows),
                'median_q': _median(row['q'] for row in rows),
                'eligible_effects': len(rows),
            }
            for gene, rows in per_gene.items()
        }
        q_statistic = min(row['median_q'] for row in medians.values())
        passes = all(
            row['eligible_effects'] >= 3
            and row['median_B'] >= 0.25
            and row['median_q'] > 0
            for row in medians.values()
        )
        rankings.append({
            'stage': stage,
            'layer': layer,
            'position_set': candidate,
            'Q': q_statistic,
            'per_exon': medians,
            'passes_development_selection_gate': passes,
            '_tie_order': (stage_index, layer, candidate_index),
        })
  rankings.sort(key=lambda row: (-row['Q'], row['_tie_order']))
  for rank, row in enumerate(rankings, start=1):
    row['rank'] = rank
    del row['_tie_order']
  return rankings


def analyze(
    run_dir: Path, *, ignored_paths: Sequence[Path] = ()
) -> dict[str, Any]:
  """Validates a complete development Phase-R tree and recomputes ranking."""
  run_dir = run_dir.resolve()
  _guard_development_path(run_dir)
  temporary_artifacts = sorted(run_dir.rglob('*.tmp'))
  if temporary_artifacts:
    raise ValueError(
        f'Incomplete temporary artifacts remain: {temporary_artifacts}.'
    )
  summary_path = run_dir / 'summary.json'
  identity_paths = sorted((run_dir / 'identity').glob('*.json'))
  group_paths = sorted((run_dir / 'groups').glob('*/*.json'))
  allowed_paths = {summary_path, *identity_paths, *group_paths}
  ignored = {path.resolve() for path in ignored_paths}
  unexpected = {
      path.resolve()
      for path in run_dir.rglob('*.json')
      if path.resolve() not in allowed_paths and path.resolve() not in ignored
  }
  if unexpected:
    raise ValueError(
        f'Unexpected JSON artifacts in Phase-R tree: {unexpected}.'
    )
  summary = _read_complete(summary_path)
  if summary.get('script_version') != SOURCE_SCRIPT_VERSION:
    raise ValueError('Summary source script version mismatch.')
  if summary.get('partition') != 'development_only':
    raise ValueError('Summary is not development-only.')
  if summary.get('variant_count') != EXPECTED_IDENTITIES:
    raise ValueError('Summary does not contain all 20 development variants.')
  if summary.get('group_limit_per_eligible_effect') != (
      EXPECTED_GROUPS_PER_ELIGIBLE_EFFECT
  ):
    raise ValueError('Summary is a bounded run, not complete Phase R.')
  if len(identity_paths) != EXPECTED_IDENTITIES:
    raise ValueError(
        f'Expected {EXPECTED_IDENTITIES} identities, found '
        f'{len(identity_paths)}.'
    )

  identities = {}
  common_hashes = set()
  case_orders = set()
  for path in identity_paths:
    identity = _validate_identity(_read_complete(path), str(path))
    case = identity['case']
    variant_id = str(case.get('variant_id', ''))
    if not variant_id or variant_id in identities:
      raise ValueError(f'Duplicate or empty identity variant {variant_id!r}.')
    identities[variant_id] = identity
    case_orders.add(case.get('order'))
    common_hashes.add(hashlib.sha256(_canonical_bytes(
        _common_configuration(identity['configuration'], 'identity')
    )).hexdigest())
  if case_orders != set(range(EXPECTED_IDENTITIES)):
    raise ValueError('Development case orders are not exactly 0 through 19.')
  if len(common_hashes) != 1:
    raise ValueError('Identity artifacts mix frozen Phase-R configurations.')

  counts = Counter(
      (identity['case']['gene'], identity['effect'])
      for identity in identities.values()
  )
  for gene in DEVELOPMENT_GENES:
    if counts[(gene, True)] != EXPECTED_EFFECTS_PER_GENE:
      raise ValueError(f'{gene} does not have six frozen effects.')
    if counts[(gene, False)] != EXPECTED_NEUTRALS_PER_GENE:
      raise ValueError(f'{gene} does not have four frozen neutrals.')
  eligible_by_gene = {
      gene: sum(
          identity['eligible'] and identity['case']['gene'] == gene
          for identity in identities.values()
      )
      for gene in DEVELOPMENT_GENES
  }
  if any(count < 3 for count in eligible_by_gene.values()):
    raise ValueError(
        f'Phase-R eligibility gate failed before ranking: {eligible_by_gene}.'
    )
  eligible_ids = {
      variant_id for variant_id, row in identities.items() if row['eligible']
  }
  expected_group_count = (
      len(eligible_ids) * EXPECTED_GROUPS_PER_ELIGIBLE_EFFECT
  )
  if len(group_paths) != expected_group_count:
    raise ValueError(
        f'Expected {expected_group_count} groups, found {len(group_paths)}.'
    )
  if summary.get('eligible_effect_count') != len(eligible_ids):
    raise ValueError('Summary eligible-effect count is inconsistent.')
  if summary.get('completed_group_count') != len(group_paths):
    raise ValueError('Summary completed-group count is inconsistent.')

  groups: dict[str, dict[tuple[str, int, str], dict[str, float]]] = {
      variant_id: {} for variant_id in eligible_ids
  }
  for path in group_paths:
    record = _read_complete(path)
    case = record.get('configuration', {}).get('case', {})
    variant_id = case.get('variant_id')
    if variant_id not in eligible_ids:
      raise ValueError(
          f'Unexpected or ineligible grouped variant {variant_id}.'
      )
    key, metrics = _validate_group(
        record, str(path), identities[variant_id]
    )
    if key in groups[variant_id]:
      raise ValueError(f'Duplicate group {key} for {variant_id}.')
    groups[variant_id][key] = metrics
  expected_keys = _expected_group_keys()
  for variant_id, observed in groups.items():
    if set(observed) != expected_keys:
      missing = expected_keys - set(observed)
      extra = set(observed) - expected_keys
      raise ValueError(
          f'Incomplete grid for {variant_id}: missing={missing}, extra={extra}.'
      )

  rankings = _rank(identities, groups)
  raw_paths = [summary_path, *identity_paths, *group_paths]
  eligibility_rows = []
  for gene in DEVELOPMENT_GENES:
    gene_rows = [
        row for row in identities.values() if row['case']['gene'] == gene
    ]
    eligibility_rows.append({
        'gene': gene,
        'eligible_effects': sum(row['eligible'] for row in gene_rows),
        'effect_total': sum(row['effect'] for row in gene_rows),
        'neutral_total': sum(not row['effect'] for row in gene_rows),
        'ineligible_effect_variants': sorted(
            row['case']['variant_id']
            for row in gene_rows
            if row['effect'] and not row['eligible']
        ),
    })
  top = rankings[0]
  return {
      'analysis_version': ANALYSIS_VERSION,
      'scope': 'development_only_confirmation_unopened',
      'source_script_version': SOURCE_SCRIPT_VERSION,
      'run_dir': str(run_dir),
      'hash_tree': {
          'raw_artifact_count': len(raw_paths),
          'raw_json_tree_sha256': _tree_digest(raw_paths, run_dir),
          'summary_sha256': _sha256(summary_path),
          'identity_tree_sha256': _tree_digest(identity_paths, run_dir),
          'groups_tree_sha256': _tree_digest(group_paths, run_dir),
          'shared_configuration_sha256': next(iter(common_hashes)),
      },
      'audit': {
          'gate0_all_identities_pass': True,
          'eligibility_gate_passes': True,
          'completeness_passes': True,
          'identity_count': len(identity_paths),
          'eligible_effect_count': len(eligible_ids),
          'group_count': len(group_paths),
          'groups_per_eligible_effect': EXPECTED_GROUPS_PER_ELIGIBLE_EFFECT,
          'eligibility_by_exon': eligibility_rows,
      },
      'development_search': {
          'candidate_count': len(rankings),
          'top_candidate': top,
          'any_candidate_passes': any(
              row['passes_development_selection_gate'] for row in rankings
          ),
          'rankings': rankings,
      },
      'decision': (
          'lock_top_phase_r_candidate_stop_wider_search'
          if top['passes_development_selection_gate']
          else 'phase_r_negative_continue_wider_ladder_confirmation_closed'
      ),
  }


def render_markdown(result: Mapping[str, Any]) -> str:
  """Renders a concise human-readable result without changing the decision."""
  search = result['development_search']
  top = search['top_candidate']
  lines = [
      '# OpenSplice Phase-R development result',
      '',
      f"**Decision:** `{result['decision']}`",
      '',
      'All Gate-0, eligibility, completeness, configuration-fingerprint, and '
      'hash-tree checks passed. Confirmation remained unopened.',
      '',
      '## Eligibility',
      '',
      '| Exon | Eligible effects | Frozen effects | Neutrals |',
      '|---|---:|---:|---:|',
  ]
  for row in result['audit']['eligibility_by_exon']:
    lines.append(
        f"| {row['gene']} | {row['eligible_effects']} | "
        f"{row['effect_total']} | {row['neutral_total']} |"
    )
  lines.extend([
      '',
      '## Frozen ranking',
      '',
      f"Top candidate: `{top['stage']}/layer{top['layer']}/"
      f"{top['position_set']}`; Q = {top['Q']:.6g}; "
      f"gate pass = {top['passes_development_selection_gate']}.",
      '',
      '| Rank | Candidate | Q | BRAF median B/q | SLC25A48 median B/q | Pass |',
      '|---:|---|---:|---:|---:|:---:|',
  ])
  for row in search['rankings'][:10]:
    braf = row['per_exon']['BRAF']
    slc = row['per_exon']['SLC25A48']
    label = f"{row['stage']}/L{row['layer']}/{row['position_set']}"
    lines.append(
        f"| {row['rank']} | {label} | {row['Q']:.6g} | "
        f"{braf['median_B']:.6g}/{braf['median_q']:.6g} | "
        f"{slc['median_B']:.6g}/{slc['median_q']:.6g} | "
        f"{row['passes_development_selection_gate']} |"
    )
  lines.extend([
      '',
      'The ranking is development-only. A passing top candidate is a locked '
      'logit-margin residual hypothesis, not biological validation; a failing '
      'top candidate leaves confirmation closed.',
      '',
      f"Raw JSON tree SHA-256: "
      f"`{result['hash_tree']['raw_json_tree_sha256']}`",
      '',
  ])
  return '\n'.join(lines)


def _write_atomic(path: Path, text: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(text, encoding='utf-8')
  temporary.replace(path)


def main() -> None:
  args = _parse_args()
  run_dir = args.run_dir.resolve()
  output_json = args.output_json.resolve()
  raw_roots = (
      run_dir / 'summary.json',
      run_dir / 'identity',
      run_dir / 'groups',
  )
  if output_json == raw_roots[0] or any(
      root in output_json.parents for root in raw_roots[1:]
  ):
    raise ValueError('Analysis output must not overwrite a raw artifact.')
  ignored = [args.output_json]
  result = analyze(run_dir, ignored_paths=ignored)
  _write_atomic(
      args.output_json,
      json.dumps(result, indent=2, allow_nan=False) + '\n',
  )
  if args.output_markdown is not None:
    _write_atomic(args.output_markdown, render_markdown(result))
  print(args.output_json.resolve())


if __name__ == '__main__':
  main()
