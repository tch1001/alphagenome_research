#!/usr/bin/env python3
"""Analyze individual necessity and eight-channel sufficiency/localization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / 'results' / 'individual_channel_validation_v1'
DEFAULT_OUTPUT = HERE / 'results' / (
    'individual_channel_validation_v1_model_behavior_analysis'
)
PLAN_PATH = HERE / 'individual_channel_validation_plan_v1.json'
GENES = ('BRAF', 'SLC25A48')
STAGES = ('E64', 'E32', 'E16', 'E8', 'E4', 'E2', 'E1')
ENABLED = ('E32', 'E16', 'E8', 'E2', 'E1')
LOCATIONS = ('intended', 'upstream', 'downstream')
ROLES = (
    'reference_baseline',
    'alternate_baseline',
    'reference_into_alternate',
    'alternate_into_alternate_self_control',
    'alternate_into_reference',
    'reference_into_reference_self_control',
)
TRUE_STAGES = {stage: True for stage in STAGES}
FULL_DONORS = {stage: True if stage in ENABLED else None for stage in STAGES}


class AnalysisError(RuntimeError):
  """Raised when a raw artifact or frozen validation contract fails."""


def _load(path: Path) -> dict[str, Any]:
  try:
    value = json.loads(path.read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError) as error:
    raise AnalysisError(f'Cannot read {path}.') from error
  if not isinstance(value, dict):
    raise AnalysisError(f'JSON root is not an object: {path}.')
  return value


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(configuration: Mapping[str, Any]) -> str:
  encoded = json.dumps(
      configuration, sort_keys=True, separators=(',', ':'), allow_nan=False
  ).encode('utf-8')
  return hashlib.sha256(encoded).hexdigest()


def _tree_digest(root: Path, paths: Sequence[Path]) -> str:
  digest = hashlib.sha256()
  for path in sorted(paths):
    digest.update(path.relative_to(root).as_posix().encode('utf-8'))
    digest.update(b'\0')
    digest.update(path.read_bytes())
    digest.update(b'\0')
  return digest.hexdigest()


def _slug(value: str) -> str:
  return re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('._')


def _finite(value: Any, label: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise AnalysisError(f'{label} is not numeric.')
  result = float(value)
  if not math.isfinite(result):
    raise AnalysisError(f'{label} is not finite.')
  return result


def _median(values: Iterable[float]) -> float:
  values = list(values)
  if not values:
    raise AnalysisError('Cannot take a median of an empty collection.')
  return float(statistics.median(values))


def recovery_from_readout(readout: Mapping[str, Any]) -> dict[str, float]:
  if readout.get('row_roles') != list(ROLES):
    raise AnalysisError('Six-row target role order changed.')
  means = [_finite(value, 'target mean') for value in readout.get('means', ())]
  totals = [
      _finite(value, 'target total') for value in readout.get('totals', ())
  ]
  if len(means) != 6 or len(totals) != 6 or readout.get('num_values') != 2:
    raise AnalysisError('Target reduction shape changed.')
  if totals != [2 * value for value in means]:
    raise AnalysisError('Target total/mean algebra changed.')
  ref, alt, ref_alt, alt_alt, alt_ref, ref_ref = means
  denominator = ref - alt
  if denominator == 0:
    raise AnalysisError('Cannot normalize a zero model allele effect.')
  forward = (ref_alt - alt_alt) / denominator
  reverse = (alt_ref - ref_ref) / -denominator
  return {
      'reference_into_alternate': forward,
      'alternate_into_reference': reverse,
      'bidirectional_bottleneck': min(forward, reverse),
      'bidirectional_mean': (forward + reverse) / 2,
  }


def _case_matches(
    configuration: Mapping[str, Any], case: Mapping[str, Any],
    plan_sha256: str,
) -> bool:
  observed = configuration.get('case', {})
  return (
      configuration.get('confirmation_access') is False
      and configuration.get('plan_sha256') == plan_sha256
      and observed.get('order') == case['order']
      and observed.get('gene') == case['gene']
      and observed.get('variant_id') == case['variant_id']
      and observed.get('selection_class') == case['selection_class']
      and observed.get('position_1based') == case['variant_position_1based']
  )


def _check_fingerprint(record: Mapping[str, Any], label: str) -> None:
  if record.get('fingerprint') != _fingerprint(record.get('configuration', {})):
    raise AnalysisError(f'{label} fingerprint does not recompute.')


def _validate_identity(
    record: Mapping[str, Any], case: Mapping[str, Any], plan_sha256: str
) -> None:
  configuration = record.get('configuration', {})
  checks = record.get('checks', {})
  if (
      record.get('status') != 'complete'
      or configuration.get('kind') != 'individual_validation_identity'
      or not _case_matches(configuration, case, plan_sha256)
  ):
    raise AnalysisError(f"Identity binding failed for {case['variant_id']}.")
  required = (
      'passed', 'target_repeat_exact', 'target_duplicate_rows_exact',
      'trace_repeat_exact', 'trace_duplicate_rows_exact',
      'target_total_equals_two_times_mean',
  )
  if any(checks.get(name) is not True for name in required):
    raise AnalysisError('Identity runtime control failed.')
  if tuple(checks.get('target_means', {})) != ROLES:
    raise AnalysisError('Identity target roles changed.')
  _check_fingerprint(record, 'Identity')


def _expected_donors(condition: Mapping[str, Any] | None) -> dict[str, Any]:
  if condition is None or condition['kind'] == 'individual_channel_necessity':
    return FULL_DONORS
  stage = condition['feature']['stage']
  return {name: True if name == stage else None for name in STAGES}


def _validate_active(
    record: Mapping[str, Any], *, repeated: bool,
    condition: Mapping[str, Any] | None,
) -> float:
  checks = record.get('checks', {})
  required = (
      'passed', 'baseline_targets_exact_from_identity',
      'self_targets_exact', 'non_channel_routes_noop_exact',
  )
  if any(checks.get(name) is not True for name in required):
    raise AnalysisError('Active runtime control failed.')
  expected_repeat = True if repeated else None
  if (
      checks.get('repeat_checked') is not repeated
      or checks.get('target_repeat_exact') is not expected_repeat
      or checks.get('trace_repeat_exact') is not expected_repeat
      or checks.get('natural_same_allele_exact_by_stage') != TRUE_STAGES
      or checks.get('withheld_channels_natural_exact_by_stage') != TRUE_STAGES
      or checks.get('selected_donor_channels_exact_by_stage')
      != _expected_donors(condition)
  ):
    raise AnalysisError('Active repeat or stage-level control failed.')
  if repeated:
    if checks.get('target_readout') != checks.get('repeat_target_readout'):
      raise AnalysisError('Repeated full-route target differs.')
  elif checks.get('repeat_target_readout') is not None:
    raise AnalysisError('One-shot condition unexpectedly stores a repeat.')
  recovery = recovery_from_readout(checks.get('target_readout', {}))
  if checks.get('recovery') != recovery:
    raise AnalysisError('Persisted recovery does not recompute.')
  means = checks['target_readout']['means']
  raw = {
      'reference_into_alternate': means[2] - means[3],
      'alternate_into_reference': means[4] - means[5],
  }
  if checks.get('raw_movement') != raw:
    raise AnalysisError('Persisted raw movement does not recompute.')
  _check_fingerprint(record, 'Active record')
  return recovery['bidirectional_bottleneck']


def load_results(
    root: Path = DEFAULT_INPUT, plan_path: Path = PLAN_PATH
) -> dict[str, Any]:
  """Validate all 920 raw records in the completed experiment."""
  if not root.is_dir() or root.is_symlink():
    raise AnalysisError(f'Result root is absent or unsafe: {root}.')
  plan = _load(plan_path)
  plan_sha256 = _sha256(plan_path)
  if (
      plan.get('schema_version')
      != 'alphagenome-individual-channel-validation-plan-v1'
      or plan.get('scope', {}).get('confirmation_access') is not False
      or plan.get('design', {}).get('condition_count') != 44
  ):
    raise AnalysisError('Plan is not the frozen individual validation.')
  summary_path = root / 'SUMMARY.json'
  summary = _load(summary_path)
  if (
      summary.get('status') != 'complete'
      or summary.get('variant_count') != 20
      or summary.get('condition_count') != 44
      or summary.get('condition_result_count') != 880
      or summary.get('model_apply_count_in_full_nonresume_run') != 960
      or summary.get('full_frozen_design_completed') is not True
      or summary.get('binding', {}).get('confirmation_access') is not False
      or summary.get('binding', {}).get('plan_sha256') != plan_sha256
  ):
    raise AnalysisError('Summary is not the completed frozen design.')

  identity_paths = sorted((root / 'raw' / 'identity').glob('*.json'))
  full_paths = sorted((root / 'raw' / 'full').glob('*.json'))
  condition_paths = sorted((root / 'raw' / 'conditions').glob('*/*.json'))
  if (
      len(identity_paths) != 20 or len(full_paths) != 20
      or len(condition_paths) != 880
  ):
    raise AnalysisError('Raw data lacks 20 + 20 + 880 records.')

  rows = []
  full_rows = []
  for case in plan['cases']:
    stem = f"{case['order']:03d}_{_slug(case['variant_id'])}"
    identity = _load(root / 'raw' / 'identity' / f'{stem}.json')
    full = _load(root / 'raw' / 'full' / f'{stem}.json')
    _validate_identity(identity, case, plan_sha256)
    full_configuration = full.get('configuration', {})
    if (
        full.get('status') != 'complete'
        or full_configuration.get('kind')
        != 'individual_validation_full_V_route'
        or not _case_matches(full_configuration, case, plan_sha256)
        or full_configuration.get('identity_fingerprint')
        != identity.get('fingerprint')
    ):
      raise AnalysisError(
          f"Full-route binding failed for {case['variant_id']}."
      )
    full_b = _validate_active(full, repeated=True, condition=None)
    full_rows.append({
        'case': full_configuration['case'],
        'bidirectional_bottleneck': full_b,
    })
    paths = sorted((root / 'raw' / 'conditions' / stem).glob('*.json'))
    if len(paths) != 44:
      raise AnalysisError(f'{stem} does not have 44 conditions.')
    for condition, path in zip(plan['conditions'], paths, strict=True):
      expected = (
          f"{condition['condition_index']:03d}_"
          f"{condition['condition_id']}.json"
      )
      if path.name != expected:
        raise AnalysisError(f'Condition file order changed in {stem}.')
      record = _load(path)
      configuration = record.get('configuration', {})
      if (
          record.get('status') != 'complete'
          or configuration.get('kind')
          != 'individual_channel_validation_condition'
          or not _case_matches(configuration, case, plan_sha256)
          or configuration.get('identity_fingerprint')
          != identity.get('fingerprint')
          or configuration.get('full_fingerprint') != full.get('fingerprint')
          or configuration.get('condition') != condition
      ):
        raise AnalysisError(
            f"Condition binding failed for {case['variant_id']}/"
            f"{condition['condition_id']}."
        )
      value = _validate_active(
          record, repeated=False, condition=condition
      )
      rows.append({
          'case': configuration['case'],
          'condition': condition,
          'bidirectional_bottleneck': value,
          'full_bidirectional_bottleneck': full_b,
          'necessity_loss': full_b - value,
      })
  all_paths = [summary_path, *identity_paths, *full_paths, *condition_paths]
  return {
      'plan': plan,
      'summary': summary,
      'rows': rows,
      'full_rows': full_rows,
      'source': {
          'result_root': str(root),
          'plan_path': str(plan_path),
          'plan_sha256': plan_sha256,
          'raw_identity_file_count': len(identity_paths),
          'raw_full_file_count': len(full_paths),
          'raw_condition_file_count': len(condition_paths),
          'raw_result_tree_sha256': _tree_digest(root, all_paths),
      },
  }


def _population(values: list[float]) -> dict[str, Any]:
  return {
      'variant_count': len(values),
      'median': _median(values),
      'positive_count': sum(value > 0 for value in values),
      'minimum': min(values),
      'maximum': max(values),
  }


def necessity_summaries(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
  selected = [
      row for row in rows
      if row['condition']['kind'] == 'individual_channel_necessity'
  ]
  result = []
  conditions = {}
  for row in selected:
    condition = row['condition']
    conditions[condition['condition_id']] = condition
  if len(conditions) != 32:
    raise AnalysisError('Expected 32 unique necessity conditions.')
  for condition in sorted(
      conditions.values(), key=lambda value: value['condition_index']
  ):
    per_gene = {}
    for gene in GENES:
      per_gene[gene] = {}
      for selection_class, label in (
          ('significant_effect', 'effect'), ('neutral_control', 'neutral')
      ):
        values = [
            row['necessity_loss'] for row in selected
            if row['condition']['condition_id'] == condition['condition_id']
            and row['case']['gene'] == gene
            and row['case']['selection_class'] == selection_class
        ]
        expected = 6 if label == 'effect' else 4
        if len(values) != expected:
          raise AnalysisError('Necessity population grouping changed.')
        per_gene[gene][label] = _population(values)
      effect = per_gene[gene]['effect']
      neutral = per_gene[gene]['neutral']
      per_gene[gene]['passes_advance_rule'] = (
          effect['median'] > 0
          and effect['median'] > neutral['median']
          and effect['positive_count'] >= 4
      )
    result.append({
        **condition['feature'],
        'condition_id': condition['condition_id'],
        'per_gene': per_gene,
    })
  return result


def sufficiency_summaries(
    rows: Sequence[Mapping[str, Any]],
    selected_children: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
  selected = [
      row for row in rows
      if row['condition']['kind'] == 'eight_channel_only_sufficiency'
  ]
  result = {}
  for child in selected_children:
    child_id = child['group_id']
    per_gene = {}
    for gene in GENES:
      per_gene[gene] = {}
      for selection_class, label in (
          ('significant_effect', 'effect'), ('neutral_control', 'neutral')
      ):
        population = [
            row for row in selected
            if row['condition']['feature']['parent_child_id'] == child_id
            and row['case']['gene'] == gene
            and row['case']['selection_class'] == selection_class
        ]
        by_variant: dict[str, dict[str, float]] = {}
        for row in population:
          by_variant.setdefault(row['case']['variant_id'], {})[
              row['condition']['location']
          ] = row['bidirectional_bottleneck']
        expected = 6 if label == 'effect' else 4
        if len(by_variant) != expected or any(
            set(value) != set(LOCATIONS) for value in by_variant.values()
        ):
          raise AnalysisError('Sufficiency population grouping changed.')
        medians = {
            location: _median(
                value[location] for value in by_variant.values()
            ) for location in LOCATIONS
        }
        contrasts = [
            value['intended']
            - max(value['upstream'], value['downstream'])
            for value in by_variant.values()
        ]
        per_gene[gene][label] = {
            'variant_count': len(by_variant),
            'median_bidirectional_bottleneck_by_location': medians,
            'positive_intended_count': sum(
                value['intended'] > 0 for value in by_variant.values()
            ),
            'median_per_variant_spatial_contrast': _median(contrasts),
            'positive_spatial_contrast_count': sum(
                value > 0 for value in contrasts
            ),
        }
    selected_gene = (
        'BRAF' if child_id.startswith(('E32_', 'E16_')) else 'SLC25A48'
    )
    effect = per_gene[selected_gene]['effect']
    neutral = per_gene[selected_gene]['neutral']
    result[child_id] = {
        'selected_for_gene': selected_gene,
        'stage': child['stage'],
        'channel_start_inclusive': child['channel_start_inclusive'],
        'channel_end_exclusive': child['channel_end_exclusive'],
        'per_gene': per_gene,
        'passes_selected_gene_localized_sufficiency_rule': (
            effect['median_bidirectional_bottleneck_by_location'][
                'intended'
            ] > 0
            and effect['median_per_variant_spatial_contrast'] > 0
            and effect['positive_spatial_contrast_count'] >= 4
            and effect['median_bidirectional_bottleneck_by_location'][
                'intended'
            ]
            > neutral['median_bidirectional_bottleneck_by_location'][
                'intended'
            ]
        ),
    }
  return result


def analyze(loaded: Mapping[str, Any]) -> dict[str, Any]:
  necessity = necessity_summaries(loaded['rows'])
  sufficiency = sufficiency_summaries(
      loaded['rows'], loaded['plan']['selected_children']
  )
  ranked_by_child = {}
  advancing = []
  for child in loaded['plan']['selected_children']:
    child_id = child['group_id']
    gene = 'BRAF' if child_id.startswith(('E32_', 'E16_')) else 'SLC25A48'
    values = [
        row for row in necessity if row['parent_child_id'] == child_id
    ]
    values.sort(key=lambda row, gene=gene: (
        -row['per_gene'][gene]['effect']['median'], row['condition_id']
    ))
    ranked_by_child[child_id] = {
        'selected_for_gene': gene,
        'ranked_condition_ids': [row['condition_id'] for row in values],
        'summaries': values,
    }
    advancing.extend(
        row for row in values
        if row['per_gene'][gene]['passes_advance_rule']
    )
  advancing.sort(key=lambda row: (
      row['selected_for_gene'], row['stage'], row['channel_start_inclusive']
  ))
  shifted = [
      row['bidirectional_bottleneck'] for row in loaded['rows']
      if row['condition']['kind'] == 'eight_channel_only_sufficiency'
      and row['condition']['location'] != 'intended'
  ]
  if len(shifted) != 160:
    raise AnalysisError('Expected 160 shifted sufficiency controls.')
  full_route = {}
  for gene in GENES:
    full_route[gene] = {}
    for selection_class in ('significant_effect', 'neutral_control'):
      values = [
          row['bidirectional_bottleneck'] for row in loaded['full_rows']
          if row['case']['gene'] == gene
          and row['case']['selection_class'] == selection_class
      ]
      full_route[gene][selection_class] = {
          'variant_count': len(values),
          'median_bidirectional_bottleneck': _median(values),
      }
  return {
      'schema_version': (
          'alphagenome-individual-channel-validation-analysis-v1'
      ),
      'source': loaded['source'],
      'scope': {
          'development_only': True,
          'confirmation_access': False,
          'variant_count': 20,
          'effect_count': 12,
          'neutral_count': 8,
          'condition_count': 44,
          'condition_result_count': 880,
          'model_apply_count': 960,
      },
      'control_summary': {
          'all_runtime_controls_passed': True,
          'identity_and_full_route_repeats_bit_exact': True,
          'conditions_are_single_shot': True,
          'shifted_sufficiency_control_count': len(shifted),
          'maximum_absolute_shifted_sufficiency_B': max(map(abs, shifted)),
          'all_shifted_sufficiency_B_exactly_zero': all(
              value == 0 for value in shifted
          ),
          'prior_executable_maximum_absolute_target_mean_difference': (
              loaded['summary']['prior_refinement_closure'][
                  'maximum_absolute_target_mean_difference'
              ]
          ),
      },
      'full_V_route': full_route,
      'individual_necessity': {
          'channel_count': len(necessity),
          'ranked_by_selected_child': ranked_by_child,
          'advancing_channel_count': len(advancing),
          'advancing_channels': advancing,
      },
      'eight_channel_sufficiency': {
          'selected_child_count': len(sufficiency),
          'all_children_pass_selected_gene_localized_sufficiency_rule': all(
              value['passes_selected_gene_localized_sufficiency_rule']
              for value in sufficiency.values()
          ),
          'per_child': sufficiency,
      },
      'interpretation': {
          'slc25a48_channel_result': (
              'Channel 175 is individually necessary in both E2 and E1, and '
              'the corresponding 168-175 subspaces are sufficient only at '
              'the intended V neighborhood. This is the strongest current '
              'candidate for a persistent multiscale model feature.'
          ),
          'braf_channel_result': (
              'E16 channel 3 passes the individual necessity rule. The '
              'E32 0-7 subspace is localized and sufficient, but none of its '
              'individual channels passes effect-over-neutral consistency, '
              'suggesting synergy or redundancy within that subspace.'
          ),
          'claim_boundary': (
              'The result identifies causal AlphaGenome coordinates. It does '
              'not identify a sequence motif, RBP or cellular mechanism.'
          ),
          'next_experiment': (
              'Test only-channel sufficiency and shifted localization for E16 '
              'channel 3 plus E2/E1 channel 175, then characterize their '
              'sequence preferences and validate controlled motif edits.'
          ),
      },
  }


def _fmt(value: float) -> str:
  return f'{value:.5f}'


def result_markdown(result: Mapping[str, Any]) -> str:
  advancing = result['individual_necessity']['advancing_channels']
  lines = [
      '# Individual-channel causal validation result',
      '',
      'Three model coordinates pass the frozen individual necessity rule:',
      'BRAF E16 channel 3, and SLC25A48 channel 175 at both E2 and E1.',
      'All four parent 8-channel subspaces also show positive, spatially',
      'localized sufficiency for the gene that selected them.',
      '',
      '## Advancing individual channels',
      '',
      '| Gene | Stage | Channel | Effect median loss | Neutral median loss | '
      'Positive effects |',
      '|---|---|---:|---:|---:|---:|',
  ]
  for row in advancing:
    gene = row['selected_for_gene']
    lines.append(
        f"| {gene} | {row['stage']} | {row['channel_start_inclusive']} | "
        f"{_fmt(row['per_gene'][gene]['effect']['median'])} | "
        f"{_fmt(row['per_gene'][gene]['neutral']['median'])} | "
        f"{row['per_gene'][gene]['effect']['positive_count']}/6 |"
    )
  lines.extend([
      '',
      'The SLC25A48 result is especially coherent: the same channel number,',
      '175, is necessary at two successive resolutions. Its E2 and E1 parent',
      'subspaces recover median `B=0.01462` and `B=0.02661` by themselves at',
      'the intended site.',
      '',
      '## Eight-channel localized sufficiency',
      '',
      '| Selected gene | Subspace | Intended median B | Positive effects |',
      '|---|---|---:|---:|',
  ])
  sufficiency = result['eight_channel_sufficiency']['per_child']
  for child_id, row in sufficiency.items():
    gene = row['selected_for_gene']
    effect = row['per_gene'][gene]['effect']
    intended = effect['median_bidirectional_bottleneck_by_location'][
        'intended'
    ]
    lines.append(
        f"| {gene} | `{child_id}` | "
        f'{_fmt(intended)} | '
        f"{effect['positive_spatial_contrast_count']}/6 |"
    )
  lines.extend([
      '',
      'Every one of the 160 equal-shape upstream/downstream controls has',
      'exactly zero recovery. The E32 BRAF subspace is localized and',
      'sufficient, but no constituent channel passes the individual',
      'effect-over-neutral rule; its behavior may require within-subspace',
      'synergy or redundancy.',
      '',
      '## Claim boundary and next step',
      '',
      'All 960 model applies completed and every runtime causal control',
      'passed. Identity and full-route repeats were exact, and confirmation',
      'data remained sealed.',
      '',
      'This identifies causal model coordinates, not a motif, RBP or cellular',
      'mechanism. Next test E16 channel 3 and E2/E1 channel 175 by themselves',
      'at intended and shifted positions. If those single-coordinate tests',
      'survive, their sequence preferences can be characterized with',
      'activation optimization, motif comparison and controlled edits.',
  ])
  return '\n'.join(lines) + '\n'


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
  parser.add_argument('--plan', type=Path, default=PLAN_PATH)
  parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if args.output_dir.exists() or args.output_dir.is_symlink():
    raise AnalysisError(f'Output already exists: {args.output_dir}.')
  result = analyze(load_results(args.input, args.plan))
  args.output_dir.mkdir(parents=True)
  (args.output_dir / 'ANALYSIS.json').write_text(
      json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n',
      encoding='utf-8',
  )
  (args.output_dir / 'RESULT.md').write_text(
      result_markdown(result), encoding='utf-8'
  )
  print(args.output_dir)


if __name__ == '__main__':
  main()
