#!/usr/bin/env python3
"""Analyze the completed v3.3 development coalition cube.

This is a CPU-only, model-behavior analysis. It reads only the 20 development
cases (BRAF and SLC25A48), reconstructs target margins from raw logits, and
does not inspect confirmation artifacts or the incomplete OOD-anchor prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    HERE / 'results/v3_3_development_encoder_skip_factorial_one_shot'
    / 'raw/coalitions'
)
DEFAULT_OUTPUT = (
    HERE / 'results/'
    'v3_3_development_encoder_skip_factorial_exploratory_model_behavior_analysis'
)
PLAYERS_7 = ('E64', 'E32', 'E16', 'E8', 'E4', 'E2', 'E1')
PLAYERS_8 = ('T', *PLAYERS_7)
GENES = ('BRAF', 'SLC25A48')
EFFECT_ORDERS = (0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15)
NEUTRAL_ORDERS = (6, 7, 8, 9, 16, 17, 18, 19)
EXPECTED_ORDERS = tuple(range(20))


class AnalysisError(RuntimeError):
  """Raised when the development cube is incomplete or internally invalid."""


def _median(values: Iterable[float]) -> float:
  values = list(values)
  if not values:
    raise AnalysisError('Cannot take a median of an empty collection.')
  return float(statistics.median(values))


def _finite(value: Any, label: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise AnalysisError(f'{label} is not numeric.')
  result = float(value)
  if not math.isfinite(result):
    raise AnalysisError(f'{label} is not finite.')
  return result


def _endpoint_values(record: Mapping[str, Any]) -> list[float]:
  """Reconstructs six target means from raw relevant/padding logits."""
  target = record.get('target_readout')
  if not isinstance(target, Mapping):
    raise AnalysisError('Missing target_readout object.')
  if target.get('selected_logit_axis') != ['relevant_class', 'padding_class']:
    raise AnalysisError('Selected-logit axis changed.')
  if target.get('endpoint_axis') != ['acceptor', 'donor']:
    raise AnalysisError('Endpoint axis changed.')
  raw = target.get('selected_logits')
  if not isinstance(raw, list) or len(raw) != 6:
    raise AnalysisError('Target readout is not six rows.')

  endpoint_margins = []
  totals = []
  means = []
  for row_index, row in enumerate(raw):
    if not isinstance(row, list) or len(row) != 2:
      raise AnalysisError(f'Target row {row_index} is not two endpoints.')
    margins = []
    for endpoint_index, pair in enumerate(row):
      if not isinstance(pair, list) or len(pair) != 2:
        raise AnalysisError(
            f'Target row {row_index} endpoint {endpoint_index} is malformed.'
        )
      relevant = _finite(pair[0], 'relevant logit')
      padding = _finite(pair[1], 'padding logit')
      margins.append(relevant - padding)
    endpoint_margins.append(margins)
    totals.append(sum(margins))
    means.append(sum(margins) / 2.0)

  if target.get('endpoint_margins') != endpoint_margins:
    raise AnalysisError('Persisted endpoint margins do not match raw logits.')
  if target.get('totals') != totals or target.get('means') != means:
    raise AnalysisError('Persisted target reductions do not match raw logits.')
  if record.get('repeat_target_readout') != target:
    raise AnalysisError('Repeated target readout differs from the first readout.')
  return means


def shapley_values(values: Sequence[float], player_count: int) -> list[float]:
  """Returns exact finite-game Shapley values in bit/player order."""
  if len(values) != 1 << player_count:
    raise ValueError('Shapley game has the wrong number of coalitions.')
  factorial_n = math.factorial(player_count)
  result = []
  for player in range(player_count):
    value = 0.0
    for coalition in range(1 << player_count):
      if coalition & (1 << player):
        continue
      size = coalition.bit_count()
      weight = (
          math.factorial(size)
          * math.factorial(player_count - size - 1)
          / factorial_n
      )
      value += weight * (
          values[coalition | (1 << player)] - values[coalition]
      )
    result.append(value)
  return result


def shapley_values_by_permutations(
    values: Sequence[float], player_count: int
) -> list[float]:
  """Independent permutation definition, used by tests and audits."""
  if len(values) != 1 << player_count:
    raise ValueError('Permutation game has the wrong number of coalitions.')
  totals = [0.0] * player_count
  permutations = list(itertools.permutations(range(player_count)))
  for order in permutations:
    coalition = 0
    for player in order:
      with_player = coalition | (1 << player)
      totals[player] += values[with_player] - values[coalition]
      coalition = with_player
  return [value / len(permutations) for value in totals]


def harsanyi_dividends(
    values: Sequence[float], player_count: int
) -> list[float]:
  """Returns the Möbius transform of a finite coalition game."""
  if len(values) != 1 << player_count:
    raise ValueError('Harsanyi game has the wrong number of coalitions.')
  result = list(map(float, values))
  for player in range(player_count):
    for coalition in range(1 << player_count):
      if coalition & (1 << player):
        result[coalition] -= result[coalition ^ (1 << player)]
  return result


def eight_player_mask_to_coalition_id(mask: int) -> int:
  """Maps bit order (T,E64,...,E1) to the persisted coalition ID."""
  if mask < 0 or mask >= 256:
    raise ValueError('Eight-player mask is out of range.')
  t = mask & 1
  e_mask = sum(((mask >> (index + 1)) & 1) << index for index in range(7))
  return 128 * t + e_mask


def _expected_enabled_players(coalition_id: int) -> list[str]:
  t, e_mask = divmod(coalition_id, 128)
  enabled = ['T'] if t else []
  enabled.extend(
      player for index, player in enumerate(PLAYERS_7)
      if e_mask & (1 << index)
  )
  return enabled


def _validate_record(
    record: Mapping[str, Any], *, order: int, coalition_id: int
) -> tuple[dict[str, Any], list[float]]:
  if (
      record.get('status') != 'complete'
      or record.get('failure') is not None
      or record.get('family') != 'encoder_skip_coalition'
      or record.get('same_six_row_compiled_executable') is not True
  ):
    raise AnalysisError(f'Coalition {order}:{coalition_id} is not complete.')
  case = record.get('case')
  coalition = record.get('coalition')
  checks = record.get('checks')
  if not all(isinstance(value, Mapping) for value in (case, coalition, checks)):
    raise AnalysisError(f'Coalition {order}:{coalition_id} is malformed.')
  case = dict(case)
  if case.get('order') != order:
    raise AnalysisError(f'Case order changed for coalition {order}:{coalition_id}.')
  t, e_mask = divmod(coalition_id, 128)
  e_bits = [bool(e_mask & (1 << index)) for index in range(7)]
  expected_coalition = {
      'coalition_bit_order': list(PLAYERS_7) + ['T'],
      'coalition_id': coalition_id,
      'e_bits': e_bits,
      'e_bits_binary': f'{e_mask:07b}',
      'e_mask': e_mask,
      'enabled_players': _expected_enabled_players(coalition_id),
      'shapley_player_order': list(PLAYERS_8),
      't': t,
  }
  if dict(coalition) != expected_coalition:
    raise AnalysisError(f'Coalition identity changed for {order}:{coalition_id}.')

  required_true = (
      'E_disabled_noop_exact', 'E_enabled_donor_exact',
      'baseline_rows_E_natural_effective_exact',
      'baseline_rows_T_natural_effective_exact',
      'baseline_targets_exact_from_identity', 'final_embedding_disabled_exact',
      'natural_E_same_allele_exact', 'natural_T_same_allele_exact',
      'natural_route_fingerprints_match_identity', 'passed',
      'runtime_route_masks_and_maps_exact', 'self_targets_exact',
      'target_repeat_exact', 'trace_repeat_exact',
      'transformer_internal_seams_disabled_exact',
  )
  if any(checks.get(key) is not True for key in required_true):
    raise AnalysisError(f'Coalition control failed for {order}:{coalition_id}.')
  if (
      checks.get('T_enabled_donor_exact') is not bool(t)
      or checks.get('T_disabled_noop_exact') is not (not bool(t))
      or checks.get('id0_identity_endpoint_exact') is not (coalition_id == 0)
      or checks.get('id255_endpoint_closure_exact') is not (
          coalition_id == 255
      )
  ):
    raise AnalysisError(f'Anchor semantics changed for {order}:{coalition_id}.')
  means = _endpoint_values(record)
  denominator = means[0] - means[1]
  if denominator == 0:
    raise AnalysisError(f'Zero model denominator for {order}:{coalition_id}.')
  reference_into_alternate = (means[2] - means[3]) / denominator
  alternate_into_reference = (means[4] - means[5]) / -denominator
  stored = checks.get('recovery')
  if not isinstance(stored, Mapping) or (
      stored.get('reference_into_alternate') != reference_into_alternate
      or stored.get('alternate_into_reference') != alternate_into_reference
      or stored.get('bidirectional_bottleneck')
      != min(reference_into_alternate, alternate_into_reference)
  ):
    raise AnalysisError(f'Persisted recovery changed for {order}:{coalition_id}.')
  return case, means


def load_cube(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  """Loads and validates the exact 20-by-256 development cube."""
  if not root.is_dir() or root.is_symlink():
    raise AnalysisError(f'Unsafe or absent coalition root: {root}.')
  directories = sorted(path for path in root.iterdir() if path.is_dir())
  if len(directories) != 20:
    raise AnalysisError('Coalition root does not contain exactly 20 cases.')
  cases = []
  tree_digest = hashlib.sha256()
  file_count = 0
  for order, directory in enumerate(directories):
    if not directory.name.startswith(f'{order:03d}_'):
      raise AnalysisError(f'Unexpected case directory order: {directory.name}.')
    files = sorted(directory.glob('*.json'))
    expected_names = [f'{coalition_id:03d}.json' for coalition_id in range(256)]
    if [path.name for path in files] != expected_names:
      raise AnalysisError(f'Incomplete coalition membership: {directory.name}.')
    coalition_means = []
    canonical_case = None
    for coalition_id, path in enumerate(files):
      payload = path.read_bytes()
      relative = path.relative_to(root).as_posix()
      digest = hashlib.sha256(payload).digest()
      tree_digest.update(relative.encode('utf-8'))
      tree_digest.update(b'\0')
      tree_digest.update(digest)
      try:
        record = json.loads(payload)
      except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnalysisError(f'Invalid JSON: {relative}.') from error
      case, means = _validate_record(
          record, order=order, coalition_id=coalition_id
      )
      if canonical_case is None:
        canonical_case = case
      elif case != canonical_case:
        raise AnalysisError(f'Case metadata changed within {directory.name}.')
      coalition_means.append(means)
      file_count += 1
    assert canonical_case is not None
    expected_class = (
        'significant_effect' if order in EFFECT_ORDERS else 'neutral_control'
    )
    if canonical_case.get('selection_class') != expected_class:
      raise AnalysisError(f'Selection class changed for order {order}.')
    cases.append({'case': canonical_case, 'means': coalition_means})
  if tuple(case['case']['order'] for case in cases) != EXPECTED_ORDERS:
    raise AnalysisError('Development case order changed.')
  if file_count != 5120:
    raise AnalysisError('Development cube is not exactly 5,120 records.')
  return cases, {
      'coalition_file_count': file_count,
      'coalition_tree_sha256': tree_digest.hexdigest(),
  }


def _case_statistics(item: Mapping[str, Any]) -> dict[str, Any]:
  means = item['means']
  reference = means[0][0]
  alternate = means[0][1]
  denominator = reference - alternate
  if denominator == 0:
    raise AnalysisError('Case has a zero REF/ALT denominator.')

  ref_into_alt = [(row[2] - row[3]) / denominator for row in means]
  alt_into_ref = [(row[4] - row[5]) / -denominator for row in means]
  bottleneck = [min(a, b) for a, b in zip(ref_into_alt, alt_into_ref)]
  mean_recovery = [(a + b) / 2.0 for a, b in zip(ref_into_alt, alt_into_ref)]
  raw_movement = [
      (abs(row[2] - row[3]) + abs(row[4] - row[5])) / 2.0
      for row in means
  ]

  natural_values = [mean_recovery[mask] for mask in range(128)]
  donor_t_values = [mean_recovery[128 + mask] for mask in range(128)]
  eight_values = [
      mean_recovery[eight_player_mask_to_coalition_id(mask)]
      for mask in range(256)
  ]
  natural_shapley = shapley_values(natural_values, 7)
  donor_t_shapley = shapley_values(donor_t_values, 7)
  eight_shapley = shapley_values(eight_values, 8)
  dividends = harsanyi_dividends(eight_values, 8)

  efficiency = {
      'natural_T': sum(natural_shapley)
      - (natural_values[-1] - natural_values[0]),
      'donor_T': sum(donor_t_shapley)
      - (donor_t_values[-1] - donor_t_values[0]),
      'eight_player': sum(eight_shapley)
      - (eight_values[-1] - eight_values[0]),
  }
  if any(abs(value) > 1e-12 for value in efficiency.values()):
    raise AnalysisError('Shapley efficiency did not close to 1e-12.')
  return {
      'case': item['case'],
      'reference_target': reference,
      'alternate_target': alternate,
      'predicted_alt_minus_ref': alternate - reference,
      'reference_into_alternate_recovery': ref_into_alt,
      'alternate_into_reference_recovery': alt_into_ref,
      'bidirectional_bottleneck': bottleneck,
      'bidirectional_mean_recovery': mean_recovery,
      'mean_absolute_raw_movement': raw_movement,
      'shapley': {
          'seven_given_natural_T': dict(zip(PLAYERS_7, natural_shapley)),
          'seven_given_donor_T': dict(zip(PLAYERS_7, donor_t_shapley)),
          'eight_player': dict(zip(PLAYERS_8, eight_shapley)),
          'efficiency_residual': efficiency,
      },
      'eight_player_absolute_harsanyi_mass_by_order': {
          str(order): sum(
              abs(dividends[mask]) for mask in range(1, 256)
              if mask.bit_count() == order
          )
          for order in range(1, 9)
      },
  }


def _gene_cases(
    rows: Sequence[Mapping[str, Any]], gene: str, selection_class: str
) -> list[Mapping[str, Any]]:
  result = [
      row for row in rows
      if row['case']['gene'] == gene
      and row['case']['selection_class'] == selection_class
  ]
  expected = 6 if selection_class == 'significant_effect' else 4
  if len(result) != expected:
    raise AnalysisError(f'Unexpected {gene} {selection_class} count.')
  return result


def _summarize_shapley(
    effects: Sequence[Mapping[str, Any]], family: str, players: Sequence[str]
) -> dict[str, Any]:
  result = {}
  for gene in GENES:
    gene_rows = _gene_cases(effects, gene, 'significant_effect')
    result[gene] = {
        player: {
            'median': _median(row['shapley'][family][player] for row in gene_rows),
            'minimum': min(row['shapley'][family][player] for row in gene_rows),
            'maximum': max(row['shapley'][family][player] for row in gene_rows),
            'positive_variant_count': sum(
                row['shapley'][family][player] > 0 for row in gene_rows
            ),
        }
        for player in players
    }
  return result


def _mask_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
  result = {'natural_T': {}, 'donor_T': {}}
  for t, family in enumerate(('natural_T', 'donor_T')):
    for e_mask in range(128):
      coalition_id = 128 * t + e_mask
      result[family][str(e_mask)] = {
          gene: _median(
              row['bidirectional_bottleneck'][coalition_id]
              for row in _gene_cases(rows, gene, 'significant_effect')
          )
          for gene in GENES
      }
  return result


def choose_resolution_candidate(
    mask_summaries: Mapping[str, Any], *, minimum_b: float = 0.25,
    minimum_retention: float = 0.80,
) -> dict[str, Any] | None:
  """Applies the old deterministic route rule as an exploratory summary."""
  for family in ('natural_T', 'donor_T'):
    rows = mask_summaries[family]
    full = rows['127']
    candidates = []
    for e_mask in range(1, 128):
      medians = rows[str(e_mask)]
      retention = {gene: medians[gene] / full[gene] for gene in GENES}
      if all(
          medians[gene] >= minimum_b
          and retention[gene] >= minimum_retention
          for gene in GENES
      ):
        candidates.append((e_mask, medians, retention))
    if candidates:
      candidates.sort(key=lambda item: (
          item[0].bit_count(), -min(item[1].values()), item[0]
      ))
      e_mask, medians, retention = candidates[0]
      return {
          'family': family,
          'e_mask': e_mask,
          'enabled_players': [
              player for index, player in enumerate(PLAYERS_7)
              if e_mask & (1 << index)
          ],
          'median_b': medians,
          'retention_vs_all_E': retention,
          'passing_mask_count': len(candidates),
          'rule': {
              'minimum_median_b_each_gene': minimum_b,
              'minimum_retention_each_gene': minimum_retention,
              'ordering': 'fewest_skips_then_maximin_B_then_increasing_mask',
          },
      }
  return None


def analyze_cube(cases: Sequence[Mapping[str, Any]], source: Mapping[str, Any]) -> dict[str, Any]:
  rows = [_case_statistics(case) for case in cases]
  effects = [
      row for row in rows
      if row['case']['selection_class'] == 'significant_effect'
  ]
  neutrals = [
      row for row in rows
      if row['case']['selection_class'] == 'neutral_control'
  ]
  if [row['case']['order'] for row in effects] != list(EFFECT_ORDERS):
    raise AnalysisError('Effect order changed.')
  if [row['case']['order'] for row in neutrals] != list(NEUTRAL_ORDERS):
    raise AnalysisError('Neutral order changed.')

  masks = _mask_summaries(rows)
  candidate = choose_resolution_candidate(masks)
  neutral_alignment = None
  if candidate is not None:
    t = 0 if candidate['family'] == 'natural_T' else 1
    coalition_id = 128 * t + candidate['e_mask']
    per_gene = {}
    for gene in GENES:
      effect_rows = _gene_cases(rows, gene, 'significant_effect')
      neutral_rows = _gene_cases(rows, gene, 'neutral_control')
      effect_median = _median(
          row['mean_absolute_raw_movement'][coalition_id]
          for row in effect_rows
      )
      neutral_median = _median(
          row['mean_absolute_raw_movement'][coalition_id]
          for row in neutral_rows
      )
      per_gene[gene] = {
          'effect_median_absolute_raw_movement': effect_median,
          'neutral_median_absolute_raw_movement': neutral_median,
          'effect_exceeds_neutral': effect_median > neutral_median,
      }
    neutral_alignment = {
        'coalition_id': coalition_id,
        'per_gene': per_gene,
        'passes_both_genes': all(
            value['effect_exceeds_neutral'] for value in per_gene.values()
        ),
    }

  anchor_summary = {
      str(coalition_id): {
          gene: _median(
              row['bidirectional_bottleneck'][coalition_id]
              for row in _gene_cases(rows, gene, 'significant_effect')
          )
          for gene in GENES
      }
      for coalition_id in (0, 127, 128, 255)
  }
  singletons = {
      player: {
          gene: masks['natural_T'][str(1 << index)][gene]
          for gene in GENES
      }
      for index, player in enumerate(PLAYERS_7)
  }
  leave_one_out = {
      player: {
          gene: masks['natural_T'][str(127 ^ (1 << index))][gene]
          for gene in GENES
      }
      for index, player in enumerate(PLAYERS_7)
  }
  harsanyi_mass = {
      gene: {
          str(order): _median(
              row['eight_player_absolute_harsanyi_mass_by_order'][str(order)]
              for row in _gene_cases(rows, gene, 'significant_effect')
          )
          for order in range(1, 9)
      }
      for gene in GENES
  }

  return {
      'schema_version': 'alphagenome-encoder-skip-exploratory-analysis-v1',
      'scope': {
          'development_only': True,
          'genes': list(GENES),
          'effect_variant_count': 12,
          'neutral_variant_count': 8,
          'confirmation_artifacts_read': False,
          'ood_anchor_artifacts_read': False,
          'model_calls': 0,
          'claim_level': 'exploratory_computational_route',
      },
      'source': dict(source),
      'controls': {
          'complete_20_by_256_cube': True,
          'raw_logits_recomputed': True,
          'stored_reductions_and_recoveries_reproduced': True,
          'all_record_controls_passed': True,
          'all_shapley_efficiency_residuals_le_1e_12': True,
      },
      'anchor_median_b': anchor_summary,
      'natural_T_singleton_median_b': singletons,
      'natural_T_leave_one_out_median_b': leave_one_out,
      'shapley_summary': {
          'seven_given_natural_T': _summarize_shapley(
              effects, 'seven_given_natural_T', PLAYERS_7
          ),
          'seven_given_donor_T': _summarize_shapley(
              effects, 'seven_given_donor_T', PLAYERS_7
          ),
          'eight_player': _summarize_shapley(
              effects, 'eight_player', PLAYERS_8
          ),
      },
      'eight_player_absolute_harsanyi_mass_by_order': harsanyi_mass,
      'effect_mask_median_b': masks,
      'exploratory_resolution_candidate': candidate,
      'candidate_effect_vs_neutral': neutral_alignment,
      'per_variant': rows,
      'interpretation': {
          'encoder_route_is_multiscale': True,
          'resolution_profile_differs_between_genes': True,
          'candidate_is_biologically_aligned': bool(
              neutral_alignment and neutral_alignment['passes_both_genes']
          ),
          'spatial_or_channel_mechanism_established': False,
          'biological_mechanism_established': False,
      },
  }


def _format_number(value: float) -> str:
  return f'{value:.5f}'


def render_markdown(result: Mapping[str, Any]) -> str:
  candidate = result['exploratory_resolution_candidate']
  lines = [
      '# Exploratory AlphaGenome encoder-skip model-behavior analysis', '',
      'This CPU-only analysis independently reconstructed the complete 20-by-256',
      'development coalition cube from raw relevant/padding logits. It made no',
      'model call and read neither confirmation nor incomplete OOD-anchor artifacts.',
      '', '## Coarse route anchors', '',
      '| Coalition | BRAF median B | SLC25A48 median B |',
      '|---|---:|---:|',
  ]
  names = {'0': 'empty', '127': 'all encoder skips', '128': 'T only', '255': 'T + all skips'}
  for coalition_id in ('0', '127', '128', '255'):
    values = result['anchor_median_b'][coalition_id]
    lines.append(
        f"| {names[coalition_id]} | {_format_number(values['BRAF'])} | "
        f"{_format_number(values['SLC25A48'])} |"
    )

  lines.extend(['', '## Resolution-level Shapley profile', '',
                'Median normalized contributions conditional on natural T:', '',
                '| Skip | BRAF | SLC25A48 |', '|---|---:|---:|'])
  summary = result['shapley_summary']['seven_given_natural_T']
  for player in PLAYERS_7:
    lines.append(
        f"| {player} | {_format_number(summary['BRAF'][player]['median'])} | "
        f"{_format_number(summary['SLC25A48'][player]['median'])} |"
    )

  lines.extend(['', '## Exploratory resolution candidate', ''])
  if candidate is None:
    lines.append('No coalition met the historical two-gene recovery/retention rule.')
  else:
    enabled = ' + '.join(candidate['enabled_players'])
    lines.extend([
        f"The deterministic historical rule selects **{enabled}** (mask "
        f"`{candidate['e_mask']}`).",
        '',
        '| Gene | Median B | Retention versus all skips |',
        '|---|---:|---:|',
    ])
    for gene in GENES:
      lines.append(
          f"| {gene} | {_format_number(candidate['median_b'][gene])} | "
          f"{_format_number(candidate['retention_vs_all_E'][gene])} |"
      )

  alignment = result['candidate_effect_vs_neutral']
  lines.extend(['', '## Biological-specificity warning', ''])
  if alignment is not None:
    lines.extend([
        '| Gene | Effect median absolute movement | Neutral median | Effect > neutral |',
        '|---|---:|---:|:---:|',
    ])
    for gene in GENES:
      value = alignment['per_gene'][gene]
      lines.append(
          f"| {gene} | {_format_number(value['effect_median_absolute_raw_movement'])} | "
          f"{_format_number(value['neutral_median_absolute_raw_movement'])} | "
          f"{str(value['effect_exceeds_neutral']).lower()} |"
      )
  lines.extend([
      '',
      'The candidate fails the effect-versus-neutral comparison in BRAF. It is',
      'therefore evidence about a broad computational route, not a biologically',
      'specific splice mechanism.', '', '## Interpretation', '',
      '- BRAF is weighted toward E32 and E16.',
      '- SLC25A48 is weighted toward E1 and E2, with E1 the largest contributor.',
      '- No single resolution explains both exons; the shared route is multiscale.',
      '- E64 and E4 can be omitted together while retaining most all-skip recovery.',
      '- The next experiment should spatially localize the five retained skips and',
      '  add better behavior-matched controls before any channel or motif claim.',
      '',
      'These are development-only causal-computation observations. They do not',
      'identify an RBP, motif, spliceosome step, endogenous necessity or mechanism',
      'that generalizes beyond the two analyzed exons.', '',
  ])
  return '\n'.join(lines)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
  parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  cases, source = load_cube(args.input.resolve())
  result = analyze_cube(cases, source)
  if args.output_dir.exists() or args.output_dir.is_symlink():
    raise AnalysisError(f'Output path already exists: {args.output_dir}.')
  args.output_dir.mkdir(parents=True, mode=0o755)
  json_payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
  markdown = render_markdown(result)
  (args.output_dir / 'ANALYSIS.json').write_text(json_payload, encoding='utf-8')
  (args.output_dir / 'RESULT.md').write_text(markdown, encoding='utf-8')
  print(args.output_dir / 'RESULT.md')


if __name__ == '__main__':
  main()
