#!/usr/bin/env python3
"""Aggregate the frozen OpenSplice development run without opening confirmation.

This analyzer consumes only the first two frozen exons. It recomputes the
development statistic in causal_interpretability_protocol_v2.md, audits
self-patch drift, and records a content hash over every raw JSON artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping


EFFECT_THRESHOLD = 0.01
SELF_DRIFT_ABSOLUTE_LIMIT = 1e-4
SELF_DRIFT_RELATIVE_LIMIT = 0.01
DEVELOPMENT_EXONS = ('BRAF', 'SLC25A48')
STAGE_ORDER = ('pre_attention', 'post_attention', 'post_mlp')
CANDIDATE_ORDER = ('V', 'A', 'D', 'S')
EXPECTED_BASELINES = 20
EXPECTED_EFFECTS = 12
EXPECTED_GROUPS_PER_EFFECT = 216
EXPECTED_IDENTITY_AUDITS = EXPECTED_EFFECTS
PAIRED_BATCH_SCRIPT_VERSION = 'opensplice-inference-trace-v1.2.0'


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run-dir', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  return parser.parse_args()


def _read_json(path: Path) -> Mapping[str, Any]:
  try:
    value = json.loads(path.read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError) as error:
    raise ValueError(f'Cannot read complete JSON artifact {path}.') from error
  if value.get('status') != 'complete':
    raise ValueError(f'Artifact is not complete: {path}.')
  return value


def _tree_digest(paths: list[Path], root: Path) -> str:
  digest = hashlib.sha256()
  for path in sorted(paths):
    digest.update(str(path.relative_to(root)).encode('utf-8'))
    digest.update(b'\0')
    digest.update(hashlib.sha256(path.read_bytes()).digest())
  return digest.hexdigest()


def _median(values) -> float:
  values = list(values)
  if not values or any(not math.isfinite(value) for value in values):
    raise ValueError('A required metric is empty or non-finite.')
  return float(statistics.median(values))


def _effect_case(record: Mapping[str, Any]) -> bool:
  return 'neutral' not in record['selection_class'].lower()


def analyze(run_dir: Path) -> dict[str, Any]:
  baseline_paths = sorted(run_dir.glob('baseline/16384bp/*.json'))
  all_trace_paths = sorted(run_dir.glob('trace/16384bp/*/*.json'))
  identity_paths = [
      path for path in all_trace_paths if path.name == 'gate0_identity_repeat.json'
  ]
  trace_paths = [
      path for path in all_trace_paths if path.name != 'gate0_identity_repeat.json'
  ]
  if len(baseline_paths) != EXPECTED_BASELINES:
    raise ValueError(
        f'Expected {EXPECTED_BASELINES} development baselines, found '
        f'{len(baseline_paths)}.'
    )
  baselines = [_read_json(path) for path in baseline_paths]
  script_versions = {
      record['configuration']['script_version'] for record in baselines
  }
  if len(script_versions) != 1:
    raise ValueError(f'Mixed baseline script versions: {script_versions}.')
  script_version = script_versions.pop()
  paired_batch_run = script_version == PAIRED_BATCH_SCRIPT_VERSION
  expected_identity_count = EXPECTED_IDENTITY_AUDITS if paired_batch_run else 0
  if len(identity_paths) != expected_identity_count:
    raise ValueError(
        f'Expected {expected_identity_count} Gate-0 identity audits for '
        f'{script_version}, found {len(identity_paths)}.'
    )
  cases = [record['configuration']['case'] for record in baselines]
  if tuple(dict.fromkeys(case['gene'] for case in cases)) != DEVELOPMENT_EXONS:
    raise ValueError('Run contains data outside the two frozen development exons.')
  effects = {
      case['variant_id']: (case, record)
      for case, record in zip(cases, baselines, strict=True)
      if _effect_case(case)
  }
  if len(effects) != EXPECTED_EFFECTS:
    raise ValueError(
        f'Expected {EXPECTED_EFFECTS} development effects, found {len(effects)}.'
    )
  identity_audits: dict[str, Mapping[str, Any]] = {}
  for path in identity_paths:
    record = _read_json(path)
    case = record['configuration']['case']
    variant_id = case['variant_id']
    if variant_id not in effects:
      raise ValueError(f'Unexpected Gate-0 variant {variant_id}.')
    if variant_id in identity_audits:
      raise ValueError(f'Duplicate Gate-0 identity audit for {variant_id}.')
    if not record['checks']['passed']:
      raise ValueError(f'Gate-0 identity audit did not pass for {variant_id}.')
    identity_audits[variant_id] = record
  if paired_batch_run and set(identity_audits) != set(effects):
    raise ValueError('Gate-0 identity audits do not cover every effect variant.')
  expected_trace_count = EXPECTED_EFFECTS * EXPECTED_GROUPS_PER_EFFECT
  if len(trace_paths) != expected_trace_count:
    raise ValueError(
        f'Expected {expected_trace_count} trace groups, found {len(trace_paths)}.'
    )

  traces: dict[str, dict[tuple[str, int, str], Mapping[str, Any]]] = {
      variant_id: {} for variant_id in effects
  }
  self_audit: dict[str, dict[str, Any]] = {}
  for path in trace_paths:
    record = _read_json(path)
    configuration = record['configuration']
    case = configuration['case']
    variant_id = case['variant_id']
    if variant_id not in effects:
      raise ValueError(f'Unexpected traced variant {variant_id}.')
    position_set = configuration['position_set']['name']
    key = (configuration['stage'], int(configuration['layer']), position_set)
    if key in traces[variant_id]:
      raise ValueError(f'Duplicate trace group for {variant_id}: {key}.')
    patches = record['patches']
    forward = patches['reference_into_alternate'][
        'self_control_corrected_recovery'
    ]
    reciprocal = patches['alternate_into_reference'][
        'self_control_corrected_recovery'
    ]
    if forward is None or reciprocal is None:
      raise ValueError(f'Undefined recovery for {variant_id}: {key}.')
    self_drift = max(
        abs(
            patches['alternate_into_alternate_self_control'][
                'delta_from_baseline'
            ]
        ),
        abs(
            patches['reference_into_reference_self_control'][
                'delta_from_baseline'
            ]
        ),
    )
    traces[variant_id][key] = {
        'bidirectional_recovery': min(float(forward), float(reciprocal)),
        'forward_recovery': float(forward),
        'reciprocal_recovery': float(reciprocal),
        'self_drift': float(self_drift),
    }
    audit = self_audit.setdefault(
        variant_id,
        {'max_self_drift': 0.0, 'failed_groups': 0, 'total_groups': 0},
    )
    denominator = abs(float(record['direct_baseline']['mean_delta_splice']))
    relative_limit = SELF_DRIFT_RELATIVE_LIMIT * denominator
    failed = self_drift > SELF_DRIFT_ABSOLUTE_LIMIT or self_drift > relative_limit
    audit['max_self_drift'] = max(audit['max_self_drift'], self_drift)
    audit['failed_groups'] += int(failed)
    audit['total_groups'] += 1

  rankings = []
  for stage_index, stage in enumerate(STAGE_ORDER):
    for layer in range(6):
      for candidate_index, candidate in enumerate(CANDIDATE_ORDER):
        per_exon: dict[str, list[dict[str, float]]] = {
            gene: [] for gene in DEVELOPMENT_EXONS
        }
        for variant_id, (case, _) in effects.items():
          values = traces[variant_id]
          candidate_b = values[(stage, layer, candidate)][
              'bidirectional_recovery'
          ]
          upstream_b = values[
              (stage, layer, f'{candidate}_control_upstream')
          ]['bidirectional_recovery']
          downstream_b = values[
              (stage, layer, f'{candidate}_control_downstream')
          ]['bidirectional_recovery']
          per_exon[case['gene']].append({
              'B': candidate_b,
              'q': candidate_b - max(upstream_b, downstream_b),
          })
        medians = {
            gene: {
                'median_B': _median(row['B'] for row in rows),
                'median_q': _median(row['q'] for row in rows),
                'eligible_effects': len(rows),
            }
            for gene, rows in per_exon.items()
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
            '_tie_order': [stage_index, layer, candidate_index],
        })
  rankings.sort(key=lambda row: (-row['Q'], row['_tie_order']))
  for rank, row in enumerate(rankings, start=1):
    row['rank'] = rank
    del row['_tie_order']

  baseline_rows = []
  for case, record in zip(cases, baselines, strict=True):
    prediction = record['prediction']
    baseline_rows.append({
        'gene': case['gene'],
        'variant_id': case['variant_id'],
        'selection_class': case['selection_class'],
        'experimental_delta_logit': case['delta_logit'],
        'predicted_mean_delta': prediction['mean_delta_splice'],
        'predicted_acceptor_delta': prediction['delta_acceptor'],
        'predicted_donor_delta': prediction['delta_donor'],
        'direction_gate': record['direction_gate'],
    })
  effect_gate_by_exon = {}
  for gene in DEVELOPMENT_EXONS:
    rows = [row for row in baseline_rows if row['gene'] == gene]
    effect_rows = [
        row for row in rows if 'neutral' not in row['selection_class'].lower()
    ]
    neutral_rows = [
        row for row in rows if 'neutral' in row['selection_class'].lower()
    ]
    effect_gate_by_exon[gene] = {
        'correct_and_above_threshold': sum(
            row['direction_gate']['gated_for_tracing'] for row in effect_rows
        ),
        'effect_total': len(effect_rows),
        'neutral_total': len(neutral_rows),
        'neutral_max_absolute_predicted_delta': max(
            abs(row['predicted_mean_delta']) for row in neutral_rows
        ),
        'neutral_median_absolute_predicted_delta': _median(
            abs(row['predicted_mean_delta']) for row in neutral_rows
        ),
    }

  failed_variants = sum(
      audit['failed_groups'] > 0 for audit in self_audit.values()
  )
  identity_audit_summary = {
      'required': paired_batch_run,
      'passed': len(identity_audits),
      'expected': expected_identity_count,
      'all_target_repeats_exact': all(
          record['checks']['target_repeat_exact']
          for record in identity_audits.values()
      ),
      'all_trace_repeats_exact': all(
          record['checks']['trace_repeat_exact']
          for record in identity_audits.values()
      ),
      'all_duplicate_rows_exact': all(
          record['checks']['reference_duplicate_rows_exact']
          and record['checks']['alternate_duplicate_rows_exact']
          and record['checks']['trace_duplicate_rows_exact']
          for record in identity_audits.values()
      ),
  }
  all_identity_audits_pass = (
      not paired_batch_run
      or identity_audit_summary['passed'] == identity_audit_summary['expected']
      and identity_audit_summary['all_target_repeats_exact']
      and identity_audit_summary['all_trace_repeats_exact']
      and identity_audit_summary['all_duplicate_rows_exact']
  )
  return {
      'analysis_version': 'opensplice-development-analysis-v1.2',
      'scope': 'development_exons_only_confirmation_unopened',
      'source_script_version': script_version,
      'run_dir': str(run_dir),
      'raw_artifact_count': (
          len(baseline_paths) + len(trace_paths) + len(identity_paths) + 1
      ),
      'raw_json_tree_sha256': _tree_digest(
          baseline_paths
          + trace_paths
          + identity_paths
          + [run_dir / 'summary.json'],
          run_dir,
      ),
      'baseline': {
          'effect_gate_by_exon': effect_gate_by_exon,
          'rows': baseline_rows,
      },
      'tooling_gate': {
          'absolute_self_drift_limit': SELF_DRIFT_ABSOLUTE_LIMIT,
          'relative_self_drift_limit_fraction': SELF_DRIFT_RELATIVE_LIMIT,
          'variants_with_any_failed_group': failed_variants,
          'variants_total': len(self_audit),
          'identity_repeat_audits': identity_audit_summary,
          'passes_gate_0': (
              failed_variants / len(self_audit) <= 0.05
              and all_identity_audits_pass
          ),
          'per_variant': self_audit,
      },
      'development_search': {
          'candidate_count': len(rankings),
          'top_candidate': rankings[0],
          'any_candidate_passes': any(
              row['passes_development_selection_gate'] for row in rankings
          ),
          'rankings': rankings,
      },
      'decision': (
          'tooling_invalid_fix_self_patch_before_interpretation'
          if failed_variants or not all_identity_audits_pass
          else 'lock_top_candidate'
          if rankings[0]['passes_development_selection_gate']
          else 'negative_do_not_open_confirmation_internals'
      ),
  }


def main() -> None:
  args = _parse_args()
  result = analyze(args.run_dir.resolve())
  args.output.parent.mkdir(parents=True, exist_ok=True)
  temporary = args.output.with_suffix(args.output.suffix + '.tmp')
  temporary.write_text(
      json.dumps(result, indent=2, sort_keys=False, allow_nan=False) + '\n',
      encoding='utf-8',
  )
  temporary.replace(args.output)
  print(args.output.resolve())


if __name__ == '__main__':
  main()
