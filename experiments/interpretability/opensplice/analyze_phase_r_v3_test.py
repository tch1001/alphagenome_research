"""Synthetic CPU tests for the standalone Phase-R v3 analyzer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


_MODULE_PATH = Path(__file__).with_name('analyze_phase_r_v3.py')
_SPEC = importlib.util.spec_from_file_location(
    'analyze_phase_r_v3', _MODULE_PATH
)
analyzer = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = analyzer
_SPEC.loader.exec_module(analyzer)


def _write(path: Path, value) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def _position_set(name: str) -> dict:
  candidate = name.split('_control_', maxsplit=1)[0]
  is_control = '_control_' in name
  return {
      'name': name,
      'tokens': [10, 11] if candidate == 'S' else [10],
      'slots': [0, 1] if candidate == 'S' else [0],
      'role': 'width_matched_control' if is_control else 'candidate',
      'matched_candidate': candidate if is_control else None,
      'genomic_intervals': [[0, 128]],
  }


def _shared_configuration() -> dict:
  return {
      'script_version': analyzer.SOURCE_SCRIPT_VERSION,
      'phase': 'R_readout_isolation',
      'development_genes': list(analyzer.DEVELOPMENT_GENES),
      'target_head_key': 'splice_sites_classification/logits',
      'target': (
          'mean_of_acceptor_and_donor_classification_logit_minus_padding_logit'
      ),
      'context_bp': 16_384,
      'attention_backend': 'dense',
      'checkpoint_snapshot': 'synthetic-checkpoint',
      'code': {'git_head': 'synthetic'},
      'residual_grid': {
          'stages': list(analyzer.STAGE_ORDER),
          'layers': list(analyzer.LAYERS),
          'candidate_position_sets': list(analyzer.CANDIDATE_ORDER),
          'resolved_position_sets_per_variant': list(
              analyzer.POSITION_SET_ORDER
          ),
          'candidate_count': analyzer.EXPECTED_CANDIDATES,
          'executed_groups_per_eligible_effect': (
              analyzer.EXPECTED_GROUPS_PER_ELIGIBLE_EFFECT
          ),
          'control_start_distance_tokens': 4,
      },
      'non_transformer_route_transfers': 'all_false_not_enumerated',
  }


def _case(order: int, gene: str, effect: bool) -> dict:
  label = 'E' if effect else 'N'
  return {
      'order': order,
      'selection_class': 'significant_effect' if effect else 'neutral_control',
      'gene': gene,
      'variant_id': f'{gene}_{label}{order}',
      'delta_logit': 1.0 if effect else 0.0,
  }


def _case_configuration(shared: dict, case: dict) -> dict:
  return {
      **shared,
      'case': case,
      'interval': {
          'chromosome': 'chr1',
          'start_0based': 0,
          'end_0based_exclusive': 16_384,
      },
      'exon': {'start_1based': 100, 'end_1based': 200, 'strand': '+'},
      'canonical_target': {
          'endpoints': [
              {
                  'role': 'acceptor',
                  'position_1based': 100,
                  'position_index': 99,
                  'track_index': 1,
              },
              {
                  'role': 'donor',
                  'position_1based': 200,
                  'position_index': 199,
                  'track_index': 0,
              },
          ],
          'padding_track_index': 4,
      },
      'sequence_sha256': {'reference': 'ref', 'alternate': 'alt'},
  }


def _target_means(ref: float, alt: float, recovery: float | None = None):
  if recovery is None:
    ref_into_alt = alt
    alt_into_ref = ref
  else:
    ref_into_alt = alt + recovery * (ref - alt)
    alt_into_ref = ref + recovery * (alt - ref)
  return {
      'reference_baseline': ref,
      'alternate_baseline': alt,
      'reference_into_alternate': ref_into_alt,
      'alternate_into_alternate_self_control': alt,
      'alternate_into_reference': alt_into_ref,
      'reference_into_reference_self_control': ref,
  }


def _identity(shared: dict, case: dict) -> dict:
  effect = 'neutral' not in case['selection_class']
  ref, alt = (0.0, 1.0) if effect else (0.0, 0.0)
  configuration = {
      **_case_configuration(shared, case),
      'kind': 'phase_r_gate0_all_false_identity_duplicate_repeat',
      'resolved_position_sets': [
          _position_set(name) for name in analyzer.POSITION_SET_ORDER
      ],
  }
  return {
      'status': 'complete',
      'fingerprint': analyzer._fingerprint(configuration),
      'configuration': configuration,
      'checks': {
          'passed': True,
          'target_repeat_exact': True,
          'target_duplicate_rows_exact': True,
          'trace_repeat_exact': True,
          'trace_duplicate_rows_exact': True,
          'target_total_equals_two_times_mean': True,
          'num_values': 2,
          'target_means': _target_means(ref, alt),
      },
      'direction_gate': {
          'predicted_alt_minus_ref_logit_margin': alt - ref,
          'experimental_delta_logit': case['delta_logit'],
          'minimum_absolute_predicted_effect': analyzer.EFFECT_THRESHOLD,
          'direction_matches_delta_logit': True if effect else None,
          'eligible_for_causal_census': effect,
      },
  }


def _recovery(stage: str, layer: int, name: str) -> float:
  if '_control_' in name:
    return 0.1
  if stage == 'pre_attention' and layer == 0 and name == 'V':
    return 0.6
  return 0.3


def _group(identity: dict, stage: str, layer: int, name: str) -> dict:
  configuration = {
      **{
          key: value
          for key, value in identity['configuration'].items()
          if key not in {'kind', 'resolved_position_sets'}
      },
      'kind': 'phase_r_six_row_live_transformer_residual',
      'gate0_fingerprint': identity['fingerprint'],
      'stage': stage,
      'layer': layer,
      'position_set': _position_set(name),
      'grid_order': (
          analyzer.STAGE_ORDER.index(stage)
          * len(analyzer.LAYERS)
          * len(analyzer.POSITION_SET_ORDER)
          + layer * len(analyzer.POSITION_SET_ORDER)
          + analyzer.POSITION_SET_ORDER.index(name)
      ),
  }
  recovery = _recovery(stage, layer, name)
  means = _target_means(0.0, 1.0, recovery)
  return {
      'status': 'complete',
      'fingerprint': analyzer._fingerprint(configuration),
      'configuration': configuration,
      'checks': {
          'passed': True,
          'baseline_targets_exact_from_gate0': True,
          'self_targets_exact': True,
          'donor_vectors_exact': {
              role: True for role in analyzer.DONOR_VECTOR_ROLES
          },
          'target_means': means,
          'raw_movement': {
              'reference_into_alternate': -recovery,
              'alternate_into_reference': recovery,
          },
          'self_control_corrected_recovery': {
              'reference_into_alternate': recovery,
              'alternate_into_reference': recovery,
              'bidirectional_bottleneck': recovery,
          },
      },
  }


def _make_tree(root: Path) -> None:
  shared = _shared_configuration()
  identities = []
  order = 0
  for gene in analyzer.DEVELOPMENT_GENES:
    for effect in (True,) * 6 + (False,) * 4:
      case = _case(order, gene, effect)
      identity = _identity(shared, case)
      identities.append(identity)
      _write(
          root / 'identity' / f'{order:03d}_{case["variant_id"]}.json',
          identity,
      )
      order += 1
  group_count = 0
  for identity in identities:
    case = identity['configuration']['case']
    if not identity['direction_gate']['eligible_for_causal_census']:
      continue
    case_dir = root / 'groups' / f'{case["order"]:03d}_{case["variant_id"]}'
    for stage in analyzer.STAGE_ORDER:
      for layer in analyzer.LAYERS:
        for name in analyzer.POSITION_SET_ORDER:
          group = _group(identity, stage, layer, name)
          grid_order = group['configuration']['grid_order']
          _write(
              case_dir / f'{grid_order:03d}_{stage}_{layer}_{name}.json',
              group,
          )
          group_count += 1
  _write(root / 'summary.json', {
      'status': 'complete',
      'script_version': analyzer.SOURCE_SCRIPT_VERSION,
      'partition': 'development_only',
      'variant_count': analyzer.EXPECTED_IDENTITIES,
      'eligible_effect_count': 12,
      'completed_group_count': group_count,
      'group_limit_per_eligible_effect': (
          analyzer.EXPECTED_GROUPS_PER_ELIGIBLE_EFFECT
      ),
  })


class PhaseRAnalyzerTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._temporary = tempfile.TemporaryDirectory()
    cls.run_dir = Path(cls._temporary.name) / 'phase_r_development'
    _make_tree(cls.run_dir)

  @classmethod
  def tearDownClass(cls):
    cls._temporary.cleanup()
    super().tearDownClass()

  def test_full_tree_reproduces_ranking_gates_hashes_and_markdown(self):
    result = analyzer.analyze(self.run_dir)

    self.assertTrue(result['audit']['gate0_all_identities_pass'])
    self.assertTrue(result['audit']['eligibility_gate_passes'])
    self.assertTrue(result['audit']['completeness_passes'])
    self.assertEqual(result['audit']['identity_count'], 20)
    self.assertEqual(result['audit']['eligible_effect_count'], 12)
    self.assertEqual(result['audit']['group_count'], 12 * 216)
    self.assertEqual(result['development_search']['candidate_count'], 72)
    top = result['development_search']['top_candidate']
    self.assertEqual(
        (top['stage'], top['layer'], top['position_set']),
        ('pre_attention', 0, 'V'),
    )
    self.assertAlmostEqual(top['Q'], 0.5)
    self.assertAlmostEqual(top['per_exon']['BRAF']['median_B'], 0.6)
    self.assertTrue(top['passes_development_selection_gate'])
    self.assertEqual(
        result['development_search']['first_passing_candidate'], top
    )
    self.assertEqual(
        result['decision'],
        'lock_first_passing_phase_r_candidate_stop_wider_search',
    )
    tied = result['development_search']['rankings'][1:5]
    self.assertEqual(
        [(row['stage'], row['layer'], row['position_set']) for row in tied],
        [
            ('pre_attention', 0, 'A'),
            ('pre_attention', 0, 'D'),
            ('pre_attention', 0, 'S'),
            ('pre_attention', 1, 'V'),
        ],
    )
    self.assertEqual(len(result['hash_tree']['raw_json_tree_sha256']), 64)
    markdown = analyzer.render_markdown(result)
    self.assertIn('Confirmation remained unopened', markdown)
    self.assertIn('pre_attention/layer0/V', markdown)

  def test_missing_group_fails_completeness(self):
    path = next((self.run_dir / 'groups').glob('*/*.json'))
    saved = path.read_bytes()
    path.unlink()
    try:
      with self.assertRaisesRegex(ValueError, 'Expected 2592 groups'):
        analyzer.analyze(self.run_dir)
    finally:
      path.write_bytes(saved)

  def test_group_fingerprint_tampering_fails_closed(self):
    path = next((self.run_dir / 'groups').glob('*/*.json'))
    saved = path.read_bytes()
    value = json.loads(saved)
    value['fingerprint'] = 'bad'
    _write(path, value)
    try:
      with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
        analyzer.analyze(self.run_dir)
    finally:
      path.write_bytes(saved)

  def test_group_position_set_must_match_linked_gate0(self):
    path = next((self.run_dir / 'groups').glob('*/*.json'))
    saved = path.read_bytes()
    value = json.loads(saved)
    value['configuration']['position_set']['tokens'][0] += 1
    value['fingerprint'] = analyzer._fingerprint(value['configuration'])
    _write(path, value)
    try:
      with self.assertRaisesRegex(ValueError, 'differs from linked Gate 0'):
        analyzer.analyze(self.run_dir)
    finally:
      path.write_bytes(saved)

  def test_nondevelopment_gene_is_rejected(self):
    path = sorted((self.run_dir / 'identity').glob('*.json'))[0]
    saved = path.read_bytes()
    value = json.loads(saved)
    value['configuration']['case']['gene'] = 'ELN'
    value['fingerprint'] = analyzer._fingerprint(value['configuration'])
    _write(path, value)
    try:
      with self.assertRaisesRegex(ValueError, 'non-development gene'):
        analyzer.analyze(self.run_dir)
    finally:
      path.write_bytes(saved)

  def test_confirmation_named_path_is_rejected_before_read(self):
    with tempfile.TemporaryDirectory(prefix='confirmation_') as directory:
      with self.assertRaisesRegex(ValueError, 'Refusing to inspect'):
        analyzer.analyze(Path(directory))


if __name__ == '__main__':
  unittest.main()
