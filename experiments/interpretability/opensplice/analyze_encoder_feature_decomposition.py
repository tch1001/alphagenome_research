#!/usr/bin/env python3
"""Analyze natural activations and weights of causal encoder channels."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / 'results' / 'encoder_feature_decomposition_v1'
DEFAULT_PLAN = HERE / 'encoder_feature_decomposition_plan_v1.json'
DEFAULT_OUTPUT = (
    HERE / 'results'
    / 'encoder_feature_decomposition_v1_model_behavior_analysis'
)
RESOLUTIONS = (1, 2, 4, 8, 16, 32, 64)
COMPONENTS = ('carried', 'first_update', 'second_update', 'output')
FEATURES = (
    ('BRAF', 'E16_c0003', 4, 0),
    ('SLC25A48', 'E1_c0175', 0, 1),
    ('SLC25A48', 'E2_c0175', 1, 1),
)


class AnalysisError(RuntimeError):
  """Raised when raw results or an analysis invariant are invalid."""


def _load(path: Path) -> dict[str, Any]:
  try:
    value = json.loads(path.read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError) as error:
    raise AnalysisError(f'Cannot load {path}.') from error
  if not isinstance(value, dict):
    raise AnalysisError(f'JSON root is not an object: {path}.')
  return value


def _tree_sha256(paths: Iterable[Path], root: Path) -> str:
  digest = hashlib.sha256()
  for path in sorted(paths):
    digest.update(path.relative_to(root).as_posix().encode())
    digest.update(b'\0')
    digest.update(path.read_bytes())
    digest.update(b'\0')
  return digest.hexdigest()


def _median(values: Iterable[float]) -> float:
  return float(statistics.median(values))


def _role_vector(
    record: Mapping[str, Any], planned: Mapping[str, Any], stage: int,
    channel_slot: int, component: str,
) -> np.ndarray:
  case = record['configuration']['case']
  resolution = RESOLUTIONS[stage]
  positions = record['configuration']['positions'][stage]
  genomic = (
      case['position_1based'],
      case['exon_start_1based'],
      case['exon_end_1based'],
  )
  role_slots = [
      positions.index(
          (position - 1 - planned['interval_start_0based']) // resolution
      )
      for position in genomic
  ]
  values = np.asarray(record['components'][component], dtype=float)[stage]
  delta = values[1, :, channel_slot] - values[0, :, channel_slot]
  return delta[role_slots]


def _pairwise_cosines(vectors: list[np.ndarray]) -> list[float]:
  result = []
  for left, right in itertools.combinations(vectors, 2):
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator:
      result.append(float(np.dot(left, right) / denominator))
  return result


def _aggregate(
    selected: list[tuple[Mapping[str, Any], Mapping[str, Any]]],
    stage: int, channel_slot: int, component: str,
) -> dict[str, Any]:
  vectors = [
      _role_vector(record, planned, stage, channel_slot, component)
      for record, planned in selected
  ]
  norms = [float(np.linalg.norm(vector)) for vector in vectors]
  cosines = _pairwise_cosines(vectors)
  return {
      'variant_count': len(vectors),
      'median_l2_allele_difference': _median(norms),
      'canonical_acceptor_delta_median': _median(
          vector[1] for vector in vectors
      ),
      'canonical_acceptor_negative_count': int(sum(
          vector[1] < 0 for vector in vectors
      )),
      'median_pairwise_cosine': _median(cosines) if cosines else None,
      'minimum_pairwise_cosine': min(cosines) if cosines else None,
      'per_variant': [
          {
              'variant_id': record['configuration']['case']['variant_id'],
              'delta_by_role_V_A_D': vector.tolist(),
              'l2': norm,
          }
          for (record, _), vector, norm in zip(
              selected, vectors, norms, strict=True
          )
      ],
  }


def _feature_analysis(
    records: list[Mapping[str, Any]],
    plan_by_id: Mapping[str, Mapping[str, Any]],
    gene: str, feature_id: str, stage: int, channel_slot: int,
) -> dict[str, Any]:
  by_class = {}
  for selection_class in ('significant_effect', 'neutral_control'):
    selected = [
        (record, plan_by_id[record['configuration']['case']['variant_id']])
        for record in records
        if record['configuration']['case']['gene'] == gene
        and record['configuration']['case']['selection_class']
        == selection_class
    ]
    by_class[selection_class] = {
        component: _aggregate(selected, stage, channel_slot, component)
        for component in COMPONENTS
    }
  return {
      'feature_id': feature_id,
      'selected_gene': gene,
      'encoder_stage_index': stage,
      'resolution_bp': RESOLUTIONS[stage],
      'channel_index': (3, 175)[channel_slot],
      'by_selection_class': by_class,
  }


def _slc_e2_amplification(
    records: list[Mapping[str, Any]],
    plan_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
  result = {}
  for selection_class in ('significant_effect', 'neutral_control'):
    ratios = []
    projections = []
    rows = []
    for record in records:
      case = record['configuration']['case']
      if (
          case['gene'] != 'SLC25A48'
          or case['selection_class'] != selection_class
      ):
        continue
      planned = plan_by_id[case['variant_id']]
      carried = _role_vector(record, planned, 1, 1, 'carried')
      output = _role_vector(record, planned, 1, 1, 'output')
      learned = output - carried
      carried_norm = float(np.linalg.norm(carried))
      ratio = float(np.linalg.norm(output) / carried_norm)
      projection = float(np.dot(learned, carried) / np.dot(carried, carried))
      ratios.append(ratio)
      projections.append(projection)
      rows.append({
          'variant_id': case['variant_id'],
          'output_to_carried_l2_ratio': ratio,
          'learned_update_projection_onto_carried': projection,
      })
    result[selection_class] = {
        'variant_count': len(rows),
        'median_output_to_carried_l2_ratio': _median(ratios),
        'count_output_norm_greater_than_carried': sum(x > 1 for x in ratios),
        'median_learned_update_projection_onto_carried': _median(projections),
        'per_variant': rows,
    }
  return result


def _weight_analysis(
    weights: Mapping[str, Any], records: list[Mapping[str, Any]],
    plan_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
  direct = weights['dna_direct_conv']
  raw = np.asarray(direct['raw_kernel'], dtype=float)[:, :, 1]
  centered = np.asarray(
      direct['per_position_base_centered_kernel'], dtype=float
  )[:, :, 1]
  alphabet = weights['dna_alphabet']
  offsets = weights['dna_kernel_offsets_bp']
  preferences = []
  for offset, row in zip(offsets, centered, strict=True):
    order = np.argsort(-row)
    preferences.append({
        'offset_bp': offset,
        'preferred_base': alphabet[int(order[0])],
        'preferred_centered_weight': float(row[order[0]]),
        'preference_margin_over_second': float(row[order[0]] - row[order[1]]),
    })
  core = ''.join(
      row['preferred_base'] for row in preferences
      if -3 <= row['offset_bp'] <= 0
  )
  base_index = {base: index for index, base in enumerate(alphabet)}
  checks = []
  known_reference = {}
  for record in records:
    case = record['configuration']['case']
    if case['gene'] != 'SLC25A48':
      continue
    offset = case['position_1based'] - case['exon_start_1based']
    known = known_reference.setdefault(str(offset), case['reference_bases'])
    if known != case['reference_bases']:
      raise AnalysisError('Inconsistent SLC25A48 reference allele.')
    expected = 0.0
    if -7 <= offset <= 7:
      expected = float(
          raw[offset + 7, base_index[case['alternate_bases']]]
          - raw[offset + 7, base_index[case['reference_bases']]]
      )
    planned = plan_by_id[case['variant_id']]
    observed = float(_role_vector(record, planned, 0, 1, 'carried')[1])
    checks.append({
        'variant_id': case['variant_id'],
        'offset_from_acceptor_bp': offset,
        'expected_direct_acceptor_delta_from_weights': expected,
        'observed_direct_acceptor_delta': observed,
        'absolute_difference': abs(expected - observed),
    })
  return {
      'channel': 175,
      'direct_kernel_core_offsets': [-3, -2, -1, 0],
      'direct_kernel_preferred_core': core,
      'known_SLC25A48_reference_alleles_by_acceptor_offset': known_reference,
      'per_offset_preferences': preferences,
      'direct_weight_to_activation_checks': checks,
      'maximum_direct_weight_to_activation_absolute_difference': max(
          row['absolute_difference'] for row in checks
      ),
  }


def analyze(root: Path = DEFAULT_INPUT, plan_path: Path = DEFAULT_PLAN):
  plan = _load(plan_path)
  summary = _load(root / 'SUMMARY.json')
  weights = _load(root / 'WEIGHTS.json')
  paths = sorted((root / 'raw').glob('*.json'))
  if (
      len(paths) != 20
      or summary.get('full_frozen_design_completed') is not True
      or summary.get('all_runtime_controls_passed') is not True
      or plan.get('scope', {}).get('confirmation_access') is not False
  ):
    raise AnalysisError('Raw decomposition result is incomplete or invalid.')
  records = [_load(path) for path in paths]
  if any(
      record.get('checks', {}).get('passed') is not True for record in records
  ):
    raise AnalysisError('A raw runtime control failed.')
  plan_by_id = {case['variant_id']: case for case in plan['cases']}
  features = [
      _feature_analysis(records, plan_by_id, *feature) for feature in FEATURES
  ]
  result = {
      'schema_version': 'alphagenome-encoder-feature-decomposition-analysis-v1',
      'scope': plan['scope'],
      'source': {
          'raw_file_count': len(paths),
          'raw_result_tree_sha256': _tree_sha256(paths, root / 'raw'),
          'selected_weight_evidence_sha256': weights[
              'selected_weight_evidence_sha256'
          ],
      },
      'features': {row['feature_id']: row for row in features},
      'slc25a48_e2_amplification': _slc_e2_amplification(
          records, plan_by_id
      ),
      'weight_analysis': _weight_analysis(weights, records, plan_by_id),
  }
  effect_e1 = result['features']['E1_c0175']['by_selection_class'][
      'significant_effect'
  ]
  effect_e2 = result['features']['E2_c0175']['by_selection_class'][
      'significant_effect'
  ]
  neutral_e1 = result['features']['E1_c0175']['by_selection_class'][
      'neutral_control'
  ]
  result['interpretation'] = {
      'e1_channel_175_is_nonlinear_composite_acceptor_detector': (
          effect_e1['first_update']['median_l2_allele_difference']
          > 20 * effect_e1['carried']['median_l2_allele_difference']
      ),
      'effect_acceptor_delta_negative_count_E1': effect_e1['output'][
          'canonical_acceptor_negative_count'
      ],
      'effect_acceptor_delta_negative_count_E2': effect_e2['output'][
          'canonical_acceptor_negative_count'
      ],
      'neutral_acceptor_delta_negative_count_E1': neutral_e1['output'][
          'canonical_acceptor_negative_count'
      ],
      'e2_amplifies_all_six_effect_vectors': (
          result['slc25a48_e2_amplification']['significant_effect'][
              'count_output_norm_greater_than_carried'
          ] == 6
      ),
      'e2_increases_cross_effect_alignment': (
          effect_e2['output']['median_pairwise_cosine']
          > effect_e1['output']['median_pairwise_cosine']
      ),
  }
  return result


def _result_markdown(analysis: Mapping[str, Any]) -> str:
  features = analysis['features']
  e1 = features['E1_c0175']['by_selection_class']['significant_effect']
  e2 = features['E2_c0175']['by_selection_class']['significant_effect']
  neutral_e1 = features['E1_c0175']['by_selection_class']['neutral_control']
  neutral_e2 = features['E2_c0175']['by_selection_class']['neutral_control']
  amp = analysis['slc25a48_e2_amplification']
  weights = analysis['weight_analysis']
  effect_amplification = amp['significant_effect'][
      'median_output_to_carried_l2_ratio'
  ]
  neutral_amplification = amp['neutral_control'][
      'median_output_to_carried_l2_ratio'
  ]
  neutral_e2_l2 = neutral_e2['output']['median_l2_allele_difference']
  return f"""# Encoder feature-decomposition result

SLC25A48 channel 175 is a learned splice-acceptor detector. The direct
15-bp DNA kernel prefers `{weights['direct_kernel_preferred_core']}` at
offsets -3..0 relative to its output base, matching the core of the tested
acceptor neighborhood. Across all ten SLC25A48 variants, direct-kernel weight
differences predict the measured direct-convolution allele differences to a
maximum absolute error of
`{weights['maximum_direct_weight_to_activation_absolute_difference']:.6g}`.

The causal E1 feature is not merely that short linear filter. In the six
effects, the direct branch has median allele-difference L2
`{e1['carried']['median_l2_allele_difference']:.3f}`, while the learned E1
residual branch has median L2
`{e1['first_update']['median_l2_allele_difference']:.3f}` and produces a
negative acceptor activation change in all 6/6 effects. The final E1 effect
vectors are already strongly aligned (median pairwise cosine
`{e1['output']['median_pairwise_cosine']:.3f}`), versus an effect median L2 of
`{e1['output']['median_l2_allele_difference']:.3f}` and neutral median L2 of
`{neutral_e1['output']['median_l2_allele_difference']:.3f}`.

E2 mostly inherits that composite E1 detector through the explicit
zero-padded residual path, then selectively strengthens it. The E2 output
norm exceeds its carried-input norm in 6/6 effects, with median amplification
`{effect_amplification:.3f}x`; the neutral median is
`{neutral_amplification:.3f}x`.
Effect median L2 rises to `{e2['output']['median_l2_allele_difference']:.3f}`
while neutral median L2 is `{neutral_e2_l2:.3f}`.

The decomposition revises the earlier robustness hypothesis: E2 does not make
the six natural effect vectors more mutually aligned (median cosine changes
from `{e1['output']['median_pairwise_cosine']:.3f}` to
`{e2['output']['median_pairwise_cosine']:.3f}`). Instead, its learned updates
amplify an already coherent E1 acceptor-disruption signal. The greater causal
portability of E2:175 therefore likely also reflects how the decoder consumes
the coarser skip, not just a cleaner encoder feature.

BRAF E16:3 remains a weaker, distributed inherited feature and is not assigned
a biological motif here. All 40 natural-allele applies passed exact-repeat,
finite-value and padded-position controls; confirmation data remained sealed.
"""


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
  parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
  parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if args.output_dir.exists() or args.output_dir.is_symlink():
    raise AnalysisError(f'Output already exists: {args.output_dir}.')
  analysis = analyze(args.input, args.plan)
  args.output_dir.mkdir(parents=True)
  (args.output_dir / 'ANALYSIS.json').write_text(
      json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + '\n',
      encoding='utf-8',
  )
  (args.output_dir / 'RESULT.md').write_text(
      _result_markdown(analysis), encoding='utf-8'
  )
  print(args.output_dir)


if __name__ == '__main__':
  main()
