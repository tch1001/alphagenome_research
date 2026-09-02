#!/usr/bin/env python3
"""Validate and analyze the completed V-local channel-group screen."""

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
DEFAULT_INPUT = HERE / 'results' / 'channel_group_screen_v1'
DEFAULT_OUTPUT = (
    HERE / 'results' / 'channel_group_screen_v1_model_behavior_analysis'
)
PLAN_PATH = HERE / 'channel_group_screen_plan_v1.json'
CONVOLUTIONS_PATH = HERE.parents[2] / (
    'src/alphagenome_research/model/convolutions.py'
)
GENES = ('BRAF', 'SLC25A48')
STAGES = ('E64', 'E32', 'E16', 'E8', 'E4', 'E2', 'E1')
ENABLED = ('E32', 'E16', 'E8', 'E2', 'E1')
ROLES = (
    'reference_baseline',
    'alternate_baseline',
    'reference_into_alternate',
    'alternate_into_alternate_self_control',
    'alternate_into_reference',
    'reference_into_reference_self_control',
)
EXPECTED_DONORS = {
    stage: True if stage in ENABLED else None for stage in STAGES
}
EXPECTED_TRUE_STAGES = {stage: True for stage in STAGES}


class AnalysisError(RuntimeError):
  """Raised when the raw screen or its frozen bindings fail validation."""


def _load(path: Path) -> dict[str, Any]:
  try:
    value = json.loads(path.read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError) as error:
    raise AnalysisError(f'Cannot read JSON artifact {path}.') from error
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


def _slug(value: str) -> str:
  return re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('._')


def recovery_from_readout(readout: Mapping[str, Any]) -> dict[str, float]:
  """Recompute reciprocal recovery from the persisted six-row target."""
  if readout.get('row_roles') != list(ROLES):
    raise AnalysisError('Six-row target role order changed.')
  means = [_finite(value, 'target mean') for value in readout.get('means', ())]
  totals = [
      _finite(value, 'target total') for value in readout.get('totals', ())
  ]
  if len(means) != 6 or len(totals) != 6 or readout.get('num_values') != 2:
    raise AnalysisError('Target reduction does not have the frozen shape.')
  if totals != [value * 2 for value in means]:
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
      'bidirectional_mean': (forward + reverse) / 2.0,
  }


def _case_matches(
    configuration: Mapping[str, Any], planned: Mapping[str, Any],
    plan_sha256: str,
) -> bool:
  case = configuration.get('case', {})
  return (
      configuration.get('confirmation_access') is False
      and configuration.get('plan_sha256') == plan_sha256
      and case.get('order') == planned['order']
      and case.get('gene') == planned['gene']
      and case.get('variant_id') == planned['variant_id']
      and case.get('selection_class') == planned['selection_class']
      and case.get('position_1based') == planned['variant_position_1based']
  )


def _validate_fingerprint(record: Mapping[str, Any], label: str) -> None:
  if record.get('fingerprint') != _fingerprint(record.get('configuration', {})):
    raise AnalysisError(f'{label} fingerprint does not recompute.')


def _validate_identity(
    record: Mapping[str, Any], planned: Mapping[str, Any], plan_sha256: str
) -> None:
  configuration = record.get('configuration', {})
  checks = record.get('checks', {})
  if (
      record.get('status') != 'complete'
      or configuration.get('kind') != 'channel_screen_identity'
      or not _case_matches(configuration, planned, plan_sha256)
  ):
    raise AnalysisError(f"Identity binding failed for {planned['variant_id']}.")
  required = (
      'passed', 'target_repeat_exact', 'target_duplicate_rows_exact',
      'trace_repeat_exact', 'trace_duplicate_rows_exact',
      'target_total_equals_two_times_mean',
  )
  if any(checks.get(name) is not True for name in required):
    raise AnalysisError(f"Identity control failed for {planned['variant_id']}.")
  means = checks.get('target_means', {})
  if tuple(means) != ROLES or checks.get('num_values') != 2:
    raise AnalysisError(f"Identity target changed for {planned['variant_id']}.")
  for role, value in means.items():
    _finite(value, f'identity {role}')
  _validate_fingerprint(record, 'Identity')


def _validate_active_checks(
    checks: Mapping[str, Any], *, repeated: bool
) -> dict[str, float]:
  required = (
      'passed', 'baseline_targets_exact_from_identity',
      'self_targets_exact', 'non_channel_routes_noop_exact',
  )
  if any(checks.get(name) is not True for name in required):
    raise AnalysisError('An active-call runtime control failed.')
  if checks.get('repeat_checked') is not repeated:
    raise AnalysisError('Repeat contract changed.')
  expected_repeat = True if repeated else None
  if (
      checks.get('target_repeat_exact') is not expected_repeat
      or checks.get('trace_repeat_exact') is not expected_repeat
  ):
    raise AnalysisError('Active-call repeat result changed.')
  if repeated:
    if checks.get('target_readout') != checks.get('repeat_target_readout'):
      raise AnalysisError('Repeated target readout differs.')
  elif checks.get('repeat_target_readout') is not None:
    raise AnalysisError('One-shot group unexpectedly stores a repeat.')
  if checks.get('natural_same_allele_exact_by_stage') != EXPECTED_TRUE_STAGES:
    raise AnalysisError('Natural same-allele stage control failed.')
  if checks.get('selected_donor_channels_exact_by_stage') != EXPECTED_DONORS:
    raise AnalysisError('Selected donor-channel stage control failed.')
  if (
      checks.get('withheld_channels_natural_exact_by_stage')
      != EXPECTED_TRUE_STAGES
  ):
    raise AnalysisError('Withheld-channel stage control failed.')
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
  return recovery


def _validate_full(
    record: Mapping[str, Any], planned: Mapping[str, Any],
    identity: Mapping[str, Any], plan_sha256: str,
) -> float:
  configuration = record.get('configuration', {})
  if (
      record.get('status') != 'complete'
      or configuration.get('kind') != 'channel_screen_full_V_route'
      or not _case_matches(configuration, planned, plan_sha256)
      or configuration.get('identity_fingerprint')
      != identity.get('fingerprint')
  ):
    raise AnalysisError(
        f"Full-route binding failed for {planned['variant_id']}."
    )
  closure = record.get('spatial_V_closure', {})
  if closure.get('comparison_recorded') is not True:
    raise AnalysisError('Spatial closure comparison was not recorded.')
  _finite(
      closure.get('maximum_absolute_target_mean_difference'),
      'spatial closure target difference',
  )
  _finite(
      closure.get('bidirectional_bottleneck_difference'),
      'spatial closure recovery difference',
  )
  _validate_fingerprint(record, 'Full route')
  return _validate_active_checks(record.get('checks', {}), repeated=True)[
      'bidirectional_bottleneck'
  ]


def _validate_group(
    record: Mapping[str, Any], planned: Mapping[str, Any],
    planned_group: Mapping[str, Any], identity: Mapping[str, Any],
    full: Mapping[str, Any], plan_sha256: str,
) -> float:
  configuration = record.get('configuration', {})
  if (
      record.get('status') != 'complete'
      or configuration.get('kind') != 'channel_screen_without_group'
      or not _case_matches(configuration, planned, plan_sha256)
      or configuration.get('identity_fingerprint')
      != identity.get('fingerprint')
      or configuration.get('full_fingerprint') != full.get('fingerprint')
      or configuration.get('group') != planned_group
  ):
    raise AnalysisError(
        f"Group binding failed for {planned['variant_id']}/"
        f"{planned_group['group_id']}."
    )
  _validate_fingerprint(record, 'Group')
  return _validate_active_checks(record.get('checks', {}), repeated=False)[
      'bidirectional_bottleneck'
  ]


def load_results(
    root: Path = DEFAULT_INPUT, plan_path: Path = PLAN_PATH
) -> dict[str, Any]:
  """Load and validate every raw identity, full-route and group record."""
  if not root.is_dir() or root.is_symlink():
    raise AnalysisError(f'Result root is absent or unsafe: {root}.')
  plan = _load(plan_path)
  plan_sha256 = _sha256(plan_path)
  if (
      plan.get('schema_version')
      != 'alphagenome-channel-group-screen-plan-v1'
      or plan.get('scope', {}).get('confirmation_access') is not False
      or plan.get('scope', {}).get('variant_count') != 20
      or plan.get('design', {}).get('group_count') != 172
  ):
    raise AnalysisError('Plan is not the frozen development-only design.')
  summary_path = root / 'SUMMARY.json'
  summary = _load(summary_path)
  if (
      summary.get('status') != 'complete'
      or summary.get('variant_count') != 20
      or summary.get('group_count') != 172
      or summary.get('group_result_count') != 3440
      or summary.get('model_apply_count_in_full_nonresume_run') != 3520
      or summary.get('full_frozen_design_completed') is not True
      or summary.get('binding', {}).get('confirmation_access') is not False
      or summary.get('binding', {}).get('plan_sha256') != plan_sha256
  ):
    raise AnalysisError('Summary is not the completed frozen screen.')

  identity_paths = sorted((root / 'raw' / 'identity').glob('*.json'))
  full_paths = sorted((root / 'raw' / 'full').glob('*.json'))
  group_paths = sorted((root / 'raw' / 'groups').glob('*/*.json'))
  if (
      len(identity_paths) != 20 or len(full_paths) != 20
      or len(group_paths) != 3440
  ):
    raise AnalysisError('Raw screen does not contain 20 + 20 + 3440 files.')

  rows = []
  full_rows = []
  groups = plan['groups']
  for planned in plan['cases']:
    stem = f"{planned['order']:03d}_{_slug(planned['variant_id'])}"
    identity_path = root / 'raw' / 'identity' / f'{stem}.json'
    full_path = root / 'raw' / 'full' / f'{stem}.json'
    if identity_path not in identity_paths or full_path not in full_paths:
      raise AnalysisError(f'Missing identity/full result for {stem}.')
    identity = _load(identity_path)
    full = _load(full_path)
    _validate_identity(identity, planned, plan_sha256)
    full_b = _validate_full(full, planned, identity, plan_sha256)
    full_rows.append({
        'case': full['configuration']['case'],
        'bidirectional_bottleneck': full_b,
        'spatial_V_closure': full['spatial_V_closure'],
    })
    case_dir = root / 'raw' / 'groups' / stem
    paths = sorted(case_dir.glob('*.json'))
    if len(paths) != 172:
      raise AnalysisError(f'{stem} does not have 172 group records.')
    for group, path in zip(groups, paths, strict=True):
      expected_name = f"{group['group_index']:03d}_{group['group_id']}.json"
      if path.name != expected_name:
        raise AnalysisError(f'Group file order changed in {stem}.')
      record = _load(path)
      without_b = _validate_group(
          record, planned, group, identity, full, plan_sha256
      )
      rows.append({
          'case': record['configuration']['case'],
          'group': group,
          'full_bidirectional_bottleneck': full_b,
          'without_group_bidirectional_bottleneck': without_b,
          'necessity_loss': full_b - without_b,
      })

  all_paths = [summary_path, *identity_paths, *full_paths, *group_paths]
  return {
      'plan': plan,
      'rows': rows,
      'full_rows': full_rows,
      'source': {
          'result_root': str(root),
          'plan_path': str(plan_path),
          'plan_sha256': plan_sha256,
          'raw_identity_file_count': len(identity_paths),
          'raw_full_file_count': len(full_paths),
          'raw_group_file_count': len(group_paths),
          'raw_result_tree_sha256': _tree_digest(root, all_paths),
      },
  }


def _population_summary(values: Sequence[float]) -> dict[str, Any]:
  return {
      'variant_count': len(values),
      'median_necessity_loss': _median(values),
      'positive_necessity_count': sum(value > 0 for value in values),
      'minimum_necessity_loss': min(values),
      'maximum_necessity_loss': max(values),
  }


def summarize_groups(
    rows: Sequence[Mapping[str, Any]], groups: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
  """Summarize every group by gene and selection class."""
  summaries = []
  for group in groups:
    selected_group = [
        row for row in rows if row['group']['group_id'] == group['group_id']
    ]
    if len(selected_group) != 20:
      raise AnalysisError(f"Incomplete group {group['group_id']}.")
    per_gene = {}
    for gene in GENES:
      effect = [
          row['necessity_loss'] for row in selected_group
          if row['case']['gene'] == gene
          and row['case']['selection_class'] == 'significant_effect'
      ]
      neutral = [
          row['necessity_loss'] for row in selected_group
          if row['case']['gene'] == gene
          and row['case']['selection_class'] == 'neutral_control'
      ]
      if len(effect) != 6 or len(neutral) != 4:
        raise AnalysisError('Gene/selection-class grouping changed.')
      per_gene[gene] = {
          'effect': _population_summary(effect),
          'neutral': _population_summary(neutral),
          'effect_median_exceeds_neutral': _median(effect) > _median(neutral),
      }
    maximin = min(
        per_gene[gene]['effect']['median_necessity_loss'] for gene in GENES
    )
    summaries.append({
        **group,
        'per_gene': per_gene,
        'cross_gene_maximin_effect_median_loss': maximin,
        'positive_effect_median_in_both_genes': maximin > 0,
        'effect_median_exceeds_neutral_in_both_genes': all(
            per_gene[gene]['effect_median_exceeds_neutral'] for gene in GENES
        ),
    })
  return summaries


def _rank_shared(
    summaries: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
  return sorted(
      summaries,
      key=lambda row: (
          -row['cross_gene_maximin_effect_median_loss'], row['group_id']
      ),
  )


def _rank_gene(
    summaries: Sequence[Mapping[str, Any]], gene: str
) -> list[Mapping[str, Any]]:
  return sorted(
      summaries,
      key=lambda row: (
          -row['per_gene'][gene]['effect']['median_necessity_loss'],
          row['group_id'],
      ),
  )


def _compact_group(row: Mapping[str, Any]) -> dict[str, Any]:
  return {
      'ranked_group_id': row['group_id'],
      'stage': row['stage'],
      'channel_start_inclusive': row['channel_start_inclusive'],
      'channel_end_exclusive': row['channel_end_exclusive'],
      'cross_gene_maximin_effect_median_loss': row[
          'cross_gene_maximin_effect_median_loss'
      ],
      'per_gene': row['per_gene'],
  }


def analyze(loaded: Mapping[str, Any]) -> dict[str, Any]:
  """Aggregate necessity losses and select deterministic refinements."""
  summaries = summarize_groups(loaded['rows'], loaded['plan']['groups'])
  shared = _rank_shared(summaries)
  eligible = [
      row for row in shared
      if row['positive_effect_median_in_both_genes']
      and row['effect_median_exceeds_neutral_in_both_genes']
  ]
  if len(eligible) < 3:
    raise AnalysisError('Fewer than three specificity-qualified shared groups.')
  gene_rankings = {gene: _rank_gene(summaries, gene) for gene in GENES}

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

  stage_summary = {}
  for stage in ENABLED:
    selected = [row for row in summaries if row['stage'] == stage]
    stage_summary[stage] = {
        'group_count': len(selected),
        'median_group_necessity_loss_by_gene': {
            gene: _median(
                row['per_gene'][gene]['effect']['median_necessity_loss']
                for row in selected
            ) for gene in GENES
        },
        'maximum_cross_gene_maximin_effect_median_loss': max(
            row['cross_gene_maximin_effect_median_loss'] for row in selected
        ),
    }

  closure_differences = [
      row['spatial_V_closure']['maximum_absolute_target_mean_difference']
      for row in loaded['full_rows']
  ]
  closure_b = {
      gene: _median(
          row['spatial_V_closure']['bidirectional_bottleneck_difference']
          for row in loaded['full_rows']
          if row['case']['gene'] == gene
          and row['case']['selection_class'] == 'significant_effect'
      ) for gene in GENES
  }
  persistent_band = [
      _compact_group(row) for row in summaries
      if row['channel_start_inclusive'] == 160
      and row['channel_end_exclusive'] == 192
  ]
  persistent_band.sort(key=lambda row: ENABLED.index(row['stage']))
  top_gene = {
      gene: [_compact_group(row) for row in gene_rankings[gene][:10]]
      for gene in GENES
  }
  recommended = {
      'shared_specificity_qualified_parents': [
          _compact_group(row) for row in eligible[:3]
      ],
      'top_within_gene_parent': {
          gene: _compact_group(gene_rankings[gene][0]) for gene in GENES
      },
      'deterministic_parent_group_ids': [
          *(row['group_id'] for row in eligible[:3]),
          *(
              gene_rankings[gene][0]['group_id'] for gene in GENES
              if gene_rankings[gene][0]['group_id']
              not in {row['group_id'] for row in eligible[:3]}
          ),
      ],
      'next_split': 'four contiguous nonoverlapping 8-channel children',
  }

  return {
      'schema_version': 'alphagenome-channel-group-screen-analysis-v1',
      'source': {
          **loaded['source'],
          'convolutions_path': str(CONVOLUTIONS_PATH),
          'convolutions_sha256': _sha256(CONVOLUTIONS_PATH),
      },
      'scope': {
          'development_only': True,
          'confirmation_access': False,
          'variant_count': 20,
          'effect_count': 12,
          'neutral_count': 8,
          'group_count': 172,
          'group_result_count': 3440,
          'model_apply_count': 3520,
      },
      'control_summary': {
          'all_runtime_controls_passed': True,
          'identity_and_full_route_repeats_bit_exact': True,
          'group_calls_are_single_shot': True,
          'within_executable_comparisons_used_for_ranking': True,
          'maximum_absolute_target_mean_difference_from_spatial_executable': (
              max(closure_differences)
          ),
          'median_full_B_difference_from_spatial_by_effect_gene': closure_b,
      },
      'full_V_route': full_route,
      'group_summaries': summaries,
      'rankings': {
          'cross_gene_all_group_ids': [row['group_id'] for row in shared],
          'top_10_cross_gene': [
              _compact_group(row) for row in shared[:10]
          ],
          'top_10_by_gene': top_gene,
          'positive_effect_median_in_both_gene_count': sum(
              row['positive_effect_median_in_both_genes'] for row in summaries
          ),
          'positive_effect_median_group_count_by_gene': {
              gene: sum(
                  row['per_gene'][gene]['effect']['median_necessity_loss'] > 0
                  for row in summaries
              ) for gene in GENES
          },
      },
      'stage_summary': stage_summary,
      'persistent_channel_band_160_191': persistent_band,
      'recommended_refinement': recommended,
      'interpretation': {
          'shared_subspace_result': (
              'E1 channels 160-191 are the strongest maximin cross-gene '
              '32-channel necessity candidate and pass the secondary '
              'effect-versus-neutral median check in both genes.'
          ),
          'exon_specific_result': (
              'BRAF is comparatively sparse and ranks E16 channels 512-543 '
              'first, whereas SLC25A48 ranks E2 channels 160-191 first.'
          ),
          'architecture_link_inference': (
              'The repeated SLC25A48 ranking of channels 160-191 at all five '
              'resolutions is consistent with a persistent feature family: '
              'DownResBlock explicitly carries existing channels through a '
              'zero-padded residual path while appending 128 channels. '
              'Convolutional mixing means this is an architectural clue, not '
              'proof that channel identity is invariant across resolutions.'
          ),
          'distributed_nonlinear_warning': (
              'No single 32-channel block explains most full-route recovery; '
              'necessity losses are nonlinear, nonadditive and based on six '
              'effect variants per gene.'
          ),
          'next_experiment': (
              'Split the three specificity-qualified shared parents plus the '
              'top within-gene parents into 8-channel children, then test '
              'individual channels, only-group sufficiency and shifted '
              'spatial controls before sequence or motif interpretation.'
          ),
      },
  }


def _fmt(value: float) -> str:
  return f'{value:.5f}'


def result_markdown(analysis: Mapping[str, Any]) -> str:
  top = analysis['rankings']['top_10_cross_gene']
  parents = analysis['recommended_refinement']
  lines = [
      '# V-local channel-group model-behavior result',
      '',
      'The completed development screen identifies reproducible channel',
      'subspaces inside the spatially localized five-skip route. The strongest',
      'shared candidate is `E1` channels 160-191. SLC25A48 additionally shows',
      'a striking 160-191 signal at every tested resolution, while BRAF ranks',
      '`E16` channels 512-543 first within that gene.',
      '',
      '## Screen and controls',
      '',
      '- 20 development variants, 172 nonoverlapping 32-channel blocks and',
      '  3,520 model applies completed.',
      '- All selected-channel donor, withheld-channel natural-value,',
      '  same-allele, baseline and non-route no-op controls passed.',
      '- Identity and full-route repeats were bit-exact. Group conditions were',
      '  intentionally single-shot after the prior spatial repeat cube was',
      '  exact.',
      '- Confirmation examples were not accessed.',
      '',
      '## Leading cross-gene necessity blocks',
      '',
      '| Rank | Block | BRAF median loss | SLC25A48 median loss | Maximin |',
      '|---:|---|---:|---:|---:|',
  ]
  for rank, row in enumerate(top, 1):
    braf = row['per_gene']['BRAF']['effect']['median_necessity_loss']
    slc25a48 = row['per_gene']['SLC25A48']['effect'][
        'median_necessity_loss'
    ]
    lines.append(
        f"| {rank} | `{row['ranked_group_id']}` | "
        f'{_fmt(braf)} | '
        f'{_fmt(slc25a48)} | '
        f"{_fmt(row['cross_gene_maximin_effect_median_loss'])} |"
    )
  lines.extend([
      '',
      'Loss is `B_full V - B_without block`; positive values mean that',
      'withholding the block reduced reciprocal recovery. The first three',
      'blocks also have a larger median loss for effects than experimental',
      'neutrals in both genes, so they are the locked shared refinement set:',
      '',
  ])
  for row in parents['shared_specificity_qualified_parents']:
    lines.append(f"- `{row['ranked_group_id']}`")
  lines.extend([
      '',
      '## Architecture-linked clue',
      '',
      'For SLC25A48, the same channel-number band 160-191 is highly ranked at',
      'E32, E16, E8, E2 and E1; it is the top SLC25A48 block at E2 and the',
      'strongest shared block at E1. `DownResBlock` preserves the existing',
      'channel prefix through a zero-padded residual connection while adding',
      '128 new channels at each downsampling step. The causal recurrence is',
      'therefore consistent with a persistent multiscale feature family.',
      '',
      'This is an inference, not yet a biological label: learned convolutions',
      'mix channels, so equal indices do not prove an invariant feature.',
      '',
      '## Boundaries and next experiment',
      '',
      'The screen is a causal search result, not a motif or molecular-',
      'mechanism claim. Effects are nonlinear and nonadditive, only six',
      'effect variants per gene were used, and no one 32-channel block',
      'accounts for most of the full-route recovery.',
      '',
      'Next, split the three shared parents and the two top gene-ranked',
      'parents into contiguous 8-channel children. Surviving children then',
      'advance to individual-channel necessity, only-group sufficiency, and',
      'V-versus-shifted localization. Sequence optimization or motif',
      'attribution should start only after those causal checks.',
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
