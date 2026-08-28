#!/usr/bin/env python3
"""CPU-only tests for the prospective v3.3.1 analyzer repair."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


_HERE = Path(__file__).resolve().parent
_ANALYZER_PATH = _HERE / 'analyze_encoder_skip_localization_v3_3_1.py'
_SPEC = importlib.util.spec_from_file_location(
    'analyze_encoder_skip_localization_v3_3_1', _ANALYZER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
analyzer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analyzer)


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_start() -> dict:
  return json.loads(
      (analyzer._RUN_DIR / 'ATTEMPT_STARTED.json').read_text(encoding='utf-8')
  )


def _controlled_result() -> dict:
  return {
      'analysis_version': analyzer._v33.ANALYSIS_VERSION,
      'analysis_dir': str(analyzer._ANALYSIS_DIR),
      'decision': 'controlled_stop_ood_tooling_failure',
      'nomination': None,
      'resolution_analysis': None,
      'hash_tree': {
          'raw_artifact_count': analyzer.RAW_ARTIFACT_COUNT,
          'raw_artifact_tree_sha256': analyzer.RAW_ARTIFACT_TREE_SHA256,
          'raw_manifest_sha256': analyzer.RAW_MANIFEST_SHA256,
          'run_complete_sha256': analyzer.RUN_COMPLETE_SHA256,
      },
      'controlled_stop': {
          'reason': 'ood_tooling_failure',
          'identity_count': 20,
          'identity_invalid_count': 0,
          'eligible_effect_count': 12,
          'coalition_record_count': 5_120,
          'coalition_invalid_count': 0,
          'ood_anchor_record_count': 2,
          'ood_invalid_count': 1,
          'shapley_computed': False,
          'nomination_performed': False,
      },
  }


class GeneratedBindingTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.start = _read_start()

  def test_real_seven_role_evidence_normalizes_to_two_outputs(self):
    normalized, audit = analyzer._validate_generated_bindings(self.start)
    actual = self.start['same_process_pre_import_bootstrap'][
        'generated_bindings'
    ]['artifacts']
    narrowed = normalized['same_process_pre_import_bootstrap'][
        'generated_bindings'
    ]['artifacts']
    self.assertEqual(set(actual), analyzer._BOOTSTRAP_ROLES)
    self.assertEqual(set(narrowed), {'generated_pb2', 'generated_pyi'})
    self.assertEqual(audit['bootstrap_artifact_count'], 7)
    self.assertTrue(audit['all_current_bytes_verified'])
    self.assertEqual(
        {row['path'] for row in narrowed.values()},
        set(analyzer._GENERATED_OUTPUTS),
    )
    # Reproduce the frozen defect without invoking any scientific reader.
    self.assertNotEqual(
        {Path(row['path']).resolve() for row in actual.values()},
        {Path(path).resolve() for path in analyzer._GENERATED_OUTPUTS},
    )
    self.assertTrue(
        {Path(path).resolve() for path in analyzer._GENERATED_OUTPUTS}.issubset(
            {Path(row['path']).resolve() for row in actual.values()}
        )
    )

  def test_real_original_start_failure_and_normalized_start_success(self):
    cases = analyzer._v33._load_cases()
    with self.assertRaisesRegex(
        analyzer._v33.AnalysisError, 'generated-output path set'
    ):
      analyzer._v33._validate_start(
          analyzer._RUN_DIR.resolve(), bundle_root=analyzer._REPO_ROOT,
          cases=cases,
      )
    normalized, _ = analyzer._validate_generated_bindings(self.start)
    with analyzer._normalized_start_reader(analyzer._RUN_DIR, normalized):
      freeze, freeze_sha, start_audit = analyzer._v33._validate_start(
          analyzer._RUN_DIR.resolve(), bundle_root=analyzer._REPO_ROOT,
          cases=cases,
      )
    self.assertEqual(freeze_sha, analyzer.ORIGINAL_FREEZE_SHA256)
    self.assertEqual(freeze['protocol_sha256'], analyzer.ORIGINAL_PROTOCOL_SHA256)
    self.assertTrue(start_audit['tracked_head_clean_at_pre_import'])
    self.assertTrue(start_audit['runtime_environment_verified'])

  def _artifacts(self, start):
    return start['same_process_pre_import_bootstrap'][
        'generated_bindings'
    ]['artifacts']

  def test_missing_and_extra_role_fail(self):
    missing = copy.deepcopy(self.start)
    del self._artifacts(missing)['tensor_proto']
    with self.assertRaisesRegex(analyzer.AmendmentError, 'seven-role'):
      analyzer._validate_generated_bindings(missing)
    extra = copy.deepcopy(self.start)
    self._artifacts(extra)['unexpected'] = copy.deepcopy(
        self._artifacts(extra)['tensor_proto']
    )
    with self.assertRaisesRegex(analyzer.AmendmentError, 'seven-role'):
      analyzer._validate_generated_bindings(extra)

  def test_swapped_generated_roles_fail(self):
    value = copy.deepcopy(self.start)
    artifacts = self._artifacts(value)
    artifacts['generated_pb2'], artifacts['generated_pyi'] = (
        artifacts['generated_pyi'], artifacts['generated_pb2']
    )
    with self.assertRaisesRegex(analyzer.AmendmentError, 'exact frozen output'):
      analyzer._validate_generated_bindings(value)

  def test_duplicate_path_fails(self):
    value = copy.deepcopy(self.start)
    artifacts = self._artifacts(value)
    artifacts['dependency_proto'] = copy.deepcopy(artifacts['tensor_proto'])
    with self.assertRaisesRegex(analyzer.AmendmentError, 'duplicated path'):
      analyzer._validate_generated_bindings(value)

  def test_generated_output_escape_fails(self):
    value = copy.deepcopy(self.start)
    outside = Path('/etc/hosts').resolve()
    artifacts = self._artifacts(value)
    artifacts['generated_pb2'] = {
        'path': str(outside), 'sha256': _sha(outside),
        'size_bytes': outside.stat().st_size,
    }
    with self.assertRaisesRegex(analyzer.AmendmentError, 'escaped'):
      analyzer._validate_generated_bindings(value)

  def test_source_dependency_misclassification_fails(self):
    value = copy.deepcopy(self.start)
    artifacts = self._artifacts(value)
    artifacts['source_proto'], artifacts['dependency_proto'] = (
        artifacts['dependency_proto'], artifacts['source_proto']
    )
    with self.assertRaisesRegex(analyzer.AmendmentError, 'misclassified'):
      analyzer._validate_generated_bindings(value)

  def test_hash_and_size_tampering_fail_for_every_role(self):
    for role in sorted(analyzer._BOOTSTRAP_ROLES):
      with self.subTest(role=role, field='sha256'):
        value = copy.deepcopy(self.start)
        self._artifacts(value)[role]['sha256'] = '0' * 64
        with self.assertRaisesRegex(analyzer.AmendmentError, 'current bytes'):
          analyzer._validate_generated_bindings(value)
      with self.subTest(role=role, field='size_bytes'):
        value = copy.deepcopy(self.start)
        self._artifacts(value)[role]['size_bytes'] += 1
        with self.assertRaisesRegex(analyzer.AmendmentError, 'current bytes'):
          analyzer._validate_generated_bindings(value)

  def test_path_and_schema_tampering_fail(self):
    for role in sorted(analyzer._BOOTSTRAP_ROLES):
      with self.subTest(role=role):
        value = copy.deepcopy(self.start)
        original = Path(self._artifacts(value)[role]['path'])
        self._artifacts(value)[role]['path'] = str(
            original.parent / '..' / original.parent.name / original.name
        )
        with self.assertRaisesRegex(analyzer.AmendmentError, 'canonical'):
          analyzer._validate_generated_bindings(value)
    value = copy.deepcopy(self.start)
    self._artifacts(value)['tensor_pb2']['extra'] = True
    with self.assertRaisesRegex(analyzer.AmendmentError, 'schema'):
      analyzer._validate_generated_bindings(value)

  def test_all_disclosure_claims_are_exact(self):
    generated = self.start['same_process_pre_import_bootstrap'][
        'generated_bindings'
    ]
    mutations = {
        'pre_import_gate': False,
        'historical_generator_argv': ['protoc'],
        'exact_regeneration_claim': True,
        'generated_artifact_exception': [],
        'embedded_header': ['changed'],
        'protobuf_runtime_version': '0',
    }
    for field, changed in mutations.items():
      with self.subTest(field=field):
        value = copy.deepcopy(self.start)
        value['same_process_pre_import_bootstrap'][
            'generated_bindings'
        ][field] = changed
        with self.assertRaisesRegex(analyzer.AmendmentError, 'claims'):
          analyzer._validate_generated_bindings(value)
    self.assertTrue(generated['pre_import_gate'])

  def test_frozen_output_and_import_cross_bindings_are_exact(self):
    value = copy.deepcopy(self.start)
    outputs = value['freeze']['protobuf_binding']['generated_outputs']
    next(iter(outputs.values()))['size_bytes'] += 1
    with self.assertRaisesRegex(analyzer.AmendmentError, 'mapping'):
      analyzer._validate_generated_bindings(value)
    value = copy.deepcopy(self.start)
    value['freeze']['protobuf_binding']['imported_pb2']['size_bytes'] += 1
    with self.assertRaisesRegex(analyzer.AmendmentError, 'Imported protobuf'):
      analyzer._validate_generated_bindings(value)


class DelegationTest(unittest.TestCase):

  def test_start_reader_is_exactly_scoped_and_restored(self):
    with tempfile.TemporaryDirectory() as temporary:
      run_dir = Path(temporary)
      start = run_dir / 'ATTEMPT_STARTED.json'
      other = run_dir / 'OTHER.json'
      start.write_text('{"source": "disk"}\n', encoding='utf-8')
      other.write_text('{"source": "other"}\n', encoding='utf-8')
      original = analyzer._v33._read_json
      with analyzer._normalized_start_reader(run_dir, {'source': 'normalized'}):
        self.assertEqual(analyzer._v33._read_json(start)['source'], 'normalized')
        self.assertEqual(analyzer._v33._read_json(other)['source'], 'other')
      self.assertIs(analyzer._v33._read_json, original)
      with self.assertRaisesRegex(RuntimeError, 'boom'):
        with analyzer._normalized_start_reader(run_dir, {'source': 'normalized'}):
          raise RuntimeError('boom')
      self.assertIs(analyzer._v33._read_json, original)

  def test_delegation_preserves_controlled_stop_and_refuses_claims(self):
    expected = _controlled_result()
    binding = {'verified': True}
    with mock.patch.object(
        analyzer, '_validate_amendment_preconditions', return_value=binding
    ), mock.patch.object(analyzer._v33, 'analyze', return_value=expected):
      result = analyzer.analyze(
          analyzer._RUN_DIR, bundle_root=analyzer._REPO_ROOT,
          amendment_binding=binding, attempt_started_sha256='a' * 64,
      )
    self.assertEqual(result['decision'], 'controlled_stop_ood_tooling_failure')
    self.assertIsNone(result['nomination'])
    self.assertIsNone(result['resolution_analysis'])
    self.assertFalse(result['controlled_stop']['shapley_computed'])
    self.assertFalse(result['controlled_stop']['nomination_performed'])
    self.assertTrue(result['analyzer_amendment']['controlled_stop_preserved'])
    for field, changed in (
        ('nomination', {'coalition_id': 1}),
        ('resolution_analysis', {}),
        ('decision', 'complete'),
    ):
      bad = _controlled_result()
      bad[field] = changed
      with self.subTest(field=field), mock.patch.object(
          analyzer, '_validate_amendment_preconditions', return_value=binding
      ), mock.patch.object(analyzer._v33, 'analyze', return_value=bad), (
          self.assertRaisesRegex(analyzer.AmendmentError, 'controlled-stop')
      ):
        analyzer.analyze(
            analyzer._RUN_DIR, bundle_root=analyzer._REPO_ROOT,
            amendment_binding=binding, attempt_started_sha256='a' * 64,
        )

  def test_no_model_import_gate(self):
    with mock.patch.dict(analyzer.sys.modules, {'jax': mock.Mock()}):
      with self.assertRaisesRegex(analyzer.AmendmentError, 'forbidden'):
        analyzer._assert_no_model_imports('test')

  def test_analysis_has_no_nonstandard_or_attemptless_bypass(self):
    binding = {'verified': True}
    with mock.patch.object(
        analyzer, '_validate_amendment_preconditions', return_value=binding
    ), self.assertRaisesRegex(analyzer.AmendmentError, 'attempt binding'):
      analyzer.analyze(
          analyzer._RUN_DIR, bundle_root=analyzer._REPO_ROOT,
          amendment_binding=binding, attempt_started_sha256=None,
      )


class ImmutableAndAppendOnlyTest(unittest.TestCase):

  def test_real_immutable_tree_structural_audit(self):
    audit = analyzer._validate_immutable_run(analyzer._RUN_DIR)
    self.assertEqual(audit['whole_run_file_count'], 5_158)
    self.assertEqual(audit['raw_artifact_count'], 5_142)
    self.assertEqual(audit['compiler_file_count'], 8)
    self.assertEqual(audit['stop_reason'], 'ood_tooling_failure')

  def test_tree_digest_uses_relative_nul_raw_digest_framing(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      first = root / 'a'
      second = root / 'd' / 'b'
      second.parent.mkdir()
      first.write_bytes(b'one')
      second.write_bytes(b'two')
      manual = hashlib.sha256()
      for path in (first, second):
        manual.update(str(path.relative_to(root)).encode('utf-8'))
        manual.update(b'\0')
        manual.update(hashlib.sha256(path.read_bytes()).digest())
      self.assertEqual(
          analyzer._tree_digest([second, first], root), manual.hexdigest()
      )

  def test_attempt_start_is_exclusive_and_records_no_prior_scientific_read(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      attempt = root / 'attempt'
      run = root / 'run'
      output_json = root / 'analysis' / 'ANALYSIS.json'
      output_markdown = root / 'analysis' / 'RESULT.md'
      binding = {'amendment_sha256': analyzer.AMENDMENT_SHA256}
      digest = analyzer._start_attempt(
          run, output_json, output_markdown, binding, attempt_dir=attempt
      )
      record_path = attempt / 'ANALYSIS_ATTEMPT_STARTED.json'
      self.assertEqual(_sha(record_path), digest)
      record = json.loads(record_path.read_text(encoding='utf-8'))
      self.assertFalse(record['scientific_values_read_before_attempt'])
      self.assertFalse(record['model_rerun_permitted'])
      self.assertEqual(record['confirmation_model_calls_permitted'], 0)
      with self.assertRaises(FileExistsError):
        analyzer._start_attempt(
            run, output_json, output_markdown, binding, attempt_dir=attempt
        )

  def test_binding_is_canonical_before_append_only_serialization(self):
    value = {'sequence_bindings': {0: {'sha256': 'a' * 64}}}
    canonical = analyzer._json_canonical_object(value)
    self.assertEqual(canonical, {'sequence_bindings': {'0': {'sha256': 'a' * 64}}})
    self.assertEqual(
        canonical, json.loads(json.dumps(canonical, sort_keys=True))
    )

  def test_started_attempt_revalidates_exact_singleton_and_binding(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      run = root / 'run'
      analysis = root / 'analysis'
      attempt = root / 'attempt'
      binding = analyzer._json_canonical_object({
          'sequence_bindings': {0: {'sha256': 'a' * 64}}
      })
      with mock.patch.object(analyzer, '_RUN_DIR', run), mock.patch.object(
          analyzer, '_ANALYSIS_DIR', analysis
      ), mock.patch.object(analyzer, '_ATTEMPT_DIR', attempt):
        digest = analyzer._start_attempt(
            run, analysis / 'ANALYSIS.json', analysis / 'RESULT.md', binding,
            attempt_dir=attempt,
        )
        analyzer._validate_started_attempt(binding, digest)
        (attempt / 'EXTRA').write_text('unexpected\n', encoding='utf-8')
        with self.assertRaisesRegex(analyzer.AmendmentError, 'extra'):
          analyzer._validate_started_attempt(binding, digest)

  def test_amendment_and_original_analyzer_hashes_are_stable(self):
    self.assertEqual(_sha(analyzer._AMENDMENT_PATH), analyzer.AMENDMENT_SHA256)
    self.assertEqual(
        _sha(analyzer._ORIGINAL_ANALYZER_PATH),
        analyzer.ORIGINAL_ANALYZER_SHA256,
    )
    self.assertEqual(
        _sha(analyzer._ORIGINAL_TEST_PATH),
        analyzer.ORIGINAL_ANALYZER_TEST_SHA256,
    )


if __name__ == '__main__':
  unittest.main()
