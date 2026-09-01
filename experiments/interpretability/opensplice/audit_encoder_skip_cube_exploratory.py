#!/usr/bin/env python3
"""Independent arithmetic audit of the exploratory encoder-skip analysis."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
from typing import Sequence


HERE = Path(__file__).resolve().parent
RAW_ROOT = (
    HERE / 'results/v3_3_development_encoder_skip_factorial_one_shot'
    / 'raw/coalitions'
)
OUTPUT_ROOT = (
    HERE / 'results/'
    'v3_3_development_encoder_skip_factorial_exploratory_model_behavior_analysis'
)
ANALYSIS_PATH = OUTPUT_ROOT / 'ANALYSIS.json'
AUDIT_PATH = OUTPUT_ROOT / 'INDEPENDENT_AUDIT.md'
PLAYERS_7 = ('E64', 'E32', 'E16', 'E8', 'E4', 'E2', 'E1')
PLAYERS_8 = ('T', *PLAYERS_7)
GENES = ('BRAF', 'SLC25A48')


def _mean_targets(path: Path) -> list[float]:
  record = json.loads(path.read_text(encoding='utf-8'))
  logits = record['target_readout']['selected_logits']
  return [
      statistics.fmean(float(pair[0]) - float(pair[1]) for pair in row)
      for row in logits
  ]


def _permutation_shapley(
    values: Sequence[float], player_count: int
) -> list[float]:
  totals = [0.0] * player_count
  count = 0
  for order in itertools.permutations(range(player_count)):
    coalition = 0
    for player in order:
      extended = coalition | (1 << player)
      totals[player] += values[extended] - values[coalition]
      coalition = extended
    count += 1
  return [value / count for value in totals]


def _eight_mask_to_id(mask: int) -> int:
  return 128 * (mask & 1) + sum(
      ((mask >> (index + 1)) & 1) << index for index in range(7)
  )


def _median(values: Sequence[float]) -> float:
  return float(statistics.median(values))


def main() -> None:
  primary_bytes = ANALYSIS_PATH.read_bytes()
  primary = json.loads(primary_bytes)
  primary_rows = {
      row['case']['order']: row for row in primary['per_variant']
  }
  raw = {}
  cases = {}
  tree_digest = hashlib.sha256()
  for directory in sorted(RAW_ROOT.iterdir()):
    order = int(directory.name[:3])
    values = []
    for coalition_id in range(256):
      path = directory / f'{coalition_id:03d}.json'
      payload = path.read_bytes()
      relative = path.relative_to(RAW_ROOT).as_posix()
      tree_digest.update(relative.encode('utf-8'))
      tree_digest.update(b'\0')
      tree_digest.update(hashlib.sha256(payload).digest())
      record = json.loads(payload)
      cases[order] = record['case']
      values.append(_mean_targets(path))
    raw[order] = values

  if tree_digest.hexdigest() != primary['source']['coalition_tree_sha256']:
    raise RuntimeError('Raw coalition tree differs from the primary analysis.')

  effect_orders = [
      order for order in sorted(cases)
      if cases[order]['selection_class'] == 'significant_effect'
  ]
  neutral_orders = [
      order for order in sorted(cases)
      if cases[order]['selection_class'] == 'neutral_control'
  ]
  maximum_shapley_difference = 0.0
  bottleneck = {}
  movement = {}
  for order in sorted(raw):
    rows = raw[order]
    denominator = rows[0][0] - rows[0][1]
    ref_into_alt = [(row[2] - row[3]) / denominator for row in rows]
    alt_into_ref = [(row[4] - row[5]) / -denominator for row in rows]
    mean_recovery = [
        statistics.fmean(pair)
        for pair in zip(ref_into_alt, alt_into_ref)
    ]
    bottleneck[order] = [
        min(pair) for pair in zip(ref_into_alt, alt_into_ref)
    ]
    movement[order] = [
        statistics.fmean((abs(row[2] - row[3]), abs(row[4] - row[5])))
        for row in rows
    ]
    if order not in effect_orders:
      continue
    games = {
        'seven_given_natural_T': (mean_recovery[:128], PLAYERS_7),
        'seven_given_donor_T': (mean_recovery[128:], PLAYERS_7),
        'eight_player': (
            [mean_recovery[_eight_mask_to_id(mask)] for mask in range(256)],
            PLAYERS_8,
        ),
    }
    for family, (game, players) in games.items():
      independent = _permutation_shapley(game, len(players))
      expected = primary_rows[order]['shapley'][family]
      for player, observed in zip(players, independent):
        difference = abs(observed - expected[player])
        maximum_shapley_difference = max(
            maximum_shapley_difference, difference
        )
        if difference > 1e-12:
          raise RuntimeError(
              f'Shapley mismatch at order {order}, {family}, {player}.'
          )

  independent_anchor = {}
  for coalition_id in (0, 127, 128, 255):
    independent_anchor[str(coalition_id)] = {
        gene: _median([
            bottleneck[order][coalition_id]
            for order in effect_orders if cases[order]['gene'] == gene
        ])
        for gene in GENES
    }
  if independent_anchor != primary['anchor_median_b']:
    raise RuntimeError('Independent anchor medians differ.')

  mask_medians = {}
  for mask in range(128):
    mask_medians[mask] = {
        gene: _median([
            bottleneck[order][mask]
            for order in effect_orders if cases[order]['gene'] == gene
        ])
        for gene in GENES
    }
  full = mask_medians[127]
  passing = [
      mask for mask in range(1, 128)
      if all(
          mask_medians[mask][gene] >= 0.25
          and mask_medians[mask][gene] / full[gene] >= 0.80
          for gene in GENES
      )
  ]
  passing.sort(key=lambda mask: (
      mask.bit_count(), -min(mask_medians[mask].values()), mask
  ))
  if not passing or passing[0] != primary[
      'exploratory_resolution_candidate'
  ]['e_mask']:
    raise RuntimeError('Independent resolution candidate differs.')
  candidate = passing[0]

  independent_neutral = {}
  for gene in GENES:
    effect_values = [
        movement[order][candidate]
        for order in effect_orders if cases[order]['gene'] == gene
    ]
    neutral_values = [
        movement[order][candidate]
        for order in neutral_orders if cases[order]['gene'] == gene
    ]
    independent_neutral[gene] = {
        'effect_median_absolute_raw_movement': _median(effect_values),
        'neutral_median_absolute_raw_movement': _median(neutral_values),
        'effect_exceeds_neutral': _median(effect_values) > _median(neutral_values),
    }
  if independent_neutral != primary['candidate_effect_vs_neutral']['per_gene']:
    raise RuntimeError('Independent neutral comparison differs.')

  if AUDIT_PATH.exists() or AUDIT_PATH.is_symlink():
    raise RuntimeError(f'Audit output already exists: {AUDIT_PATH}.')
  analysis_sha = hashlib.sha256(primary_bytes).hexdigest()
  lines = [
      '# Independent audit of the exploratory encoder-skip analysis', '',
      'A separate standard-library implementation reread all 5,120 development',
      'coalition records and recomputed target means directly from raw',
      'relevant/padding logits. It used the permutation definition of Shapley',
      'values rather than the subset-weight formula used by the primary analyzer.',
      '',
      f'- Primary `ANALYSIS.json` SHA-256: `{analysis_sha}`',
      f"- Raw coalition tree SHA-256: `{tree_digest.hexdigest()}`",
      f'- Maximum absolute Shapley difference: `{maximum_shapley_difference:.3e}`',
      '- Anchor medians, mask selection and effect-versus-neutral medians: exact',
      '- Confirmation artifacts read: no',
      '- Incomplete OOD-anchor artifacts read: no',
      '- Model calls: zero', '',
      'The independent audit confirms the exploratory mask-110 computational',
      'route and the BRAF neutral-control failure. It does not convert that route',
      'into a biologically specific mechanism.', '',
  ]
  AUDIT_PATH.write_text('\n'.join(lines), encoding='utf-8')
  print(AUDIT_PATH)


if __name__ == '__main__':
  main()
