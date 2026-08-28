"""CPU-only tests for the v3.3.2.1 controlled-stop analyzer repair."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock


_MODULE_PATH = Path(__file__).with_name(
    'analyze_encoder_skip_ood_sidecar_v3_3_2_1.py'
)
_SPEC = importlib.util.spec_from_file_location(
    'analyze_encoder_skip_ood_sidecar_v3_3_2_1', _MODULE_PATH
)
analyzer = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = analyzer
_SPEC.loader.exec_module(analyzer)


def _write_json(path: Path, value) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n',
      encoding='utf-8',
  )


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _controlled_result() -> dict:
  return {
      'analysis_version': analyzer._v332.ANALYSIS_VERSION,
      'status': 'complete_controlled_stop_audited',
      'decision': 'controlled_stop_compiler_graph_mismatch',
      'controlled_stop': {
          'reason': 'compiler_graph_mismatch',
          'message': 'New eight-row graph/HLO differs from frozen v3.3.',
          'audited_record_count': 0,
      },
      'scientific_summary_computed': False,
      'shapley_or_nomination_computed': False,
      'nomination_performed': False,
      'nomination': None,
      'resolution_analysis': None,
      'combined_analysis_permitted': False,
      'sidecar_audit': {
          'audited_record_count': 0,
          'valid_record_count': 0,
          'invalid_record_count': 0,
          'audited_model_apply_count': 0,
          'raw_artifact_count': 0,
          'id0_all20': False,
          'id255_all20': False,
      },
      'provenance_audit': {
          'compiler': {'graph_and_hlo_exact_to_original_v3_3': False}
      },
  }


class LiteralPreflightSchemaTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.start_path = analyzer._RUN_DIR / 'ATTEMPT_STARTED.json'
    cls.preflight_path = analyzer._PREFLIGHT_DIR / 'preflight_0000.json'
    cls.start = json.loads(cls.start_path.read_text(encoding='utf-8'))
    cls.preflight = json.loads(cls.preflight_path.read_text(encoding='utf-8'))

  def test_captured_literal_key_sets_and_pid(self):
    same = self.start['same_process_preflight']
    external = self.start['external_preflight']['observation']
    self.assertEqual(set(same), analyzer._SAME_PROCESS_KEYS)
    self.assertEqual(set(external), analyzer._EXTERNAL_OBSERVATION_KEYS)
    self.assertEqual(same['pid'], analyzer.BOOTSTRAP_PID)
    self.assertEqual(self.start['bootstrap']['pid'], analyzer.BOOTSTRAP_PID)
    analyzer._validate_same_process(
        same, bootstrap=self.start['bootstrap'], start=self.start
    )

  def test_external_artifact_and_embedded_copy_validate(self):
    digest = analyzer._validate_external_tail(
        self.start,
        freeze=json.loads(analyzer._ORIGINAL_FREEZE_PATH.read_text()),
        freeze_sha=analyzer.V3_3_2_FREEZE_SHA256,
    )
    self.assertEqual(digest, _sha(self.preflight_path))

  def test_external_only_field_is_rejected_on_same_process(self):
    same = copy.deepcopy(self.start['same_process_preflight'])
    same['jax_enable_compilation_cache'] = False
    with self.assertRaisesRegex(analyzer.AmendmentError, 'literal key set'):
      analyzer._validate_same_process(
          same, bootstrap=self.start['bootstrap'], start=self.start
      )

  def test_missing_same_process_field_is_rejected(self):
    same = copy.deepcopy(self.start['same_process_preflight'])
    same.pop('runtime_environment')
    with self.assertRaisesRegex(analyzer.AmendmentError, 'literal key set'):
      analyzer._validate_same_process(
          same, bootstrap=self.start['bootstrap'], start=self.start
      )

  def test_pid_and_runtime_tamper_fail(self):
    same = copy.deepcopy(self.start['same_process_preflight'])
    same['pid'] += 1
    with self.assertRaisesRegex(analyzer.AmendmentError, 'PID'):
      analyzer._validate_same_process(
          same, bootstrap=self.start['bootstrap'], start=self.start
      )
    same = copy.deepcopy(self.start['same_process_preflight'])
    same['runtime_environment']['JAX_ENABLE_COMPILATION_CACHE'] = 'true'
    with self.assertRaisesRegex(analyzer.AmendmentError, 'runtime-environment'):
      analyzer._validate_same_process(
          same, bootstrap=self.start['bootstrap'], start=self.start
      )

  def test_gpu_uuid_and_package_tamper_fail(self):
    same = copy.deepcopy(self.start['same_process_preflight'])
    same['nvidia_smi']['parsed_single_gpu']['uuid'] = 'GPU-tamper'
    with self.assertRaisesRegex(analyzer._v332._v33.AnalysisError, 'GPU/UUID'):
      analyzer._validate_same_process(
          same, bootstrap=self.start['bootstrap'], start=self.start
      )
    start = copy.deepcopy(self.start)
    package = next(iter(start['same_process_preflight']['packages']))
    start['same_process_preflight']['packages'][package] = 'tamper'
    with self.assertRaisesRegex(
        analyzer._v332._v33.AnalysisError, 'package versions'
    ):
      analyzer._validate_external_tail(
          start,
          freeze=json.loads(analyzer._ORIGINAL_FREEZE_PATH.read_text()),
          freeze_sha=analyzer.V3_3_2_FREEZE_SHA256,
      )

  def test_external_derived_field_missing_or_tampered_fails(self):
    start = copy.deepcopy(self.start)
    start['external_preflight']['observation'].pop(
        'v3_3_2_runtime_environment'
    )
    # Embedded bytes remain immutable, so even an in-memory mutation fails at
    # the exact artifact/copy comparison before the key-set gate.
    with self.assertRaisesRegex(analyzer.AmendmentError, 'embedded copy'):
      analyzer._validate_external_tail(
          start,
          freeze=json.loads(analyzer._ORIGINAL_FREEZE_PATH.read_text()),
          freeze_sha=analyzer.V3_3_2_FREEZE_SHA256,
      )

  def test_external_path_and_hash_tamper_fail(self):
    start = copy.deepcopy(self.start)
    start['external_preflight']['path'] = str(
        analyzer._PREFLIGHT_DIR / 'preflight_0000.stdout.log'
    )
    with self.assertRaisesRegex(analyzer.AmendmentError, 'lexical path'):
      analyzer._validate_external_tail(
          start,
          freeze=json.loads(analyzer._ORIGINAL_FREEZE_PATH.read_text()),
          freeze_sha=analyzer.V3_3_2_FREEZE_SHA256,
      )
    start = copy.deepcopy(self.start)
    start['external_preflight']['sha256'] = '0' * 64
    with self.assertRaisesRegex(analyzer.AmendmentError, 'path/hash'):
      analyzer._validate_external_tail(
          start,
          freeze=json.loads(analyzer._ORIGINAL_FREEZE_PATH.read_text()),
          freeze_sha=analyzer.V3_3_2_FREEZE_SHA256,
      )


class ExactTreeTest(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)

  def tearDown(self):
    self.temporary.cleanup()

  def fixture(self):
    (self.root / 'nested').mkdir()
    (self.root / 'one').write_bytes(b'one')
    (self.root / 'nested/two').write_bytes(b'two')
    files = {
        'one': (3, _sha(self.root / 'one')),
        'nested/two': (3, _sha(self.root / 'nested/two')),
    }
    return files

  def test_exact_tree_hashes_every_file(self):
    files = self.fixture()
    audit = analyzer._walk_exact_tree(
        self.root, files, {'.', 'nested'}, 'fixture'
    )
    self.assertEqual(audit['file_count'], 2)
    self.assertTrue(analyzer._is_sha256(audit['tree_sha256']))

  def test_extra_empty_directory_fails(self):
    files = self.fixture()
    (self.root / 'empty').mkdir()
    with self.assertRaisesRegex(analyzer.AmendmentError, 'directory membership'):
      analyzer._walk_exact_tree(
          self.root, files, {'.', 'nested'}, 'fixture'
      )

  def test_symlink_and_special_entry_fail(self):
    files = self.fixture()
    (self.root / 'link').symlink_to(self.root / 'one')
    with self.assertRaisesRegex(analyzer.AmendmentError, 'symlink/special'):
      analyzer._walk_exact_tree(
          self.root, files, {'.', 'nested'}, 'fixture'
      )
    (self.root / 'link').unlink()
    fifo = self.root / 'fifo'
    os.mkfifo(fifo)
    self.assertTrue(stat.S_ISFIFO(fifo.lstat().st_mode))
    with self.assertRaisesRegex(analyzer.AmendmentError, 'symlink/special'):
      analyzer._walk_exact_tree(
          self.root, files, {'.', 'nested'}, 'fixture'
      )

  def test_hash_and_size_tamper_fail(self):
    files = self.fixture()
    (self.root / 'one').write_bytes(b'changed')
    with self.assertRaisesRegex(analyzer.AmendmentError, 'bytes changed'):
      analyzer._walk_exact_tree(
          self.root, files, {'.', 'nested'}, 'fixture'
      )

  def test_bound_real_run_and_preflight_trees(self):
    run = analyzer._walk_exact_tree(
        analyzer._RUN_DIR, analyzer._RUN_FILES,
        {'.', 'compiler', 'compiler/eight_row'}, 'captured run'
    )
    self.assertEqual(run['tree_sha256'], analyzer.RUN_TREE_SHA256)
    compiler = [
        analyzer._RUN_DIR / relative for relative in analyzer._RUN_FILES
        if relative.startswith('compiler/')
    ]
    self.assertEqual(
        analyzer._tree_digest(compiler, analyzer._RUN_DIR),
        analyzer.COMPILER_TREE_SHA256,
    )
    preflight = analyzer._walk_exact_tree(
        analyzer._PREFLIGHT_DIR, analyzer._PREFLIGHT_FILES, {'.'},
        'captured preflight',
    )
    self.assertEqual(
        preflight['tree_sha256'], analyzer.PREFLIGHT_TREE_SHA256
    )


class RepairAndClaimTest(unittest.TestCase):

  def test_analysis_freeze_is_exact_and_hash_bound(self):
    with (
        mock.patch.object(analyzer.subprocess, 'run'),
        mock.patch.object(
            analyzer.subprocess, 'check_output',
            return_value=analyzer._FREEZE_PATH.read_bytes(),
        ),
    ):
      freeze = analyzer._validate_freeze(analyzer._REPO_ROOT)
    self.assertEqual(freeze['run_tree_sha256'], analyzer.RUN_TREE_SHA256)
    self.assertEqual(
        freeze['compiler_tree_sha256'], analyzer.COMPILER_TREE_SHA256
    )
    self.assertEqual(
        freeze['original_protocol_sha256'], analyzer.ORIGINAL_PROTOCOL_SHA256
    )

  def test_untracked_analysis_freeze_fails(self):
    def tracked(command, **unused):
      if str(analyzer._FREEZE_PATH.relative_to(analyzer._REPO_ROOT)) in command:
        raise analyzer.subprocess.CalledProcessError(1, command)
      return mock.DEFAULT

    with mock.patch.object(analyzer.subprocess, 'run', side_effect=tracked):
      with self.assertRaisesRegex(analyzer.AmendmentError, 'not committed'):
        analyzer._validate_freeze(analyzer._REPO_ROOT)

  def test_analysis_freeze_head_blob_tamper_fails(self):
    with (
        mock.patch.object(analyzer.subprocess, 'run'),
        mock.patch.object(
            analyzer.subprocess, 'check_output', return_value=b'{}\n'
        ),
    ):
      with self.assertRaisesRegex(analyzer.AmendmentError, 'HEAD blob'):
        analyzer._validate_freeze(analyzer._REPO_ROOT)

  def test_all_75_frozen_sources_match_model_run_commit(self):
    freeze = json.loads(analyzer._FREEZE_PATH.read_text())
    audit = analyzer._validate_model_run_bundle(freeze, analyzer._REPO_ROOT)
    self.assertEqual(audit['model_run_commit'], analyzer.MODEL_RUN_COMMIT)
    self.assertEqual(audit['frozen_file_count'], 75)
    self.assertTrue(audit['all_current_and_historical_bytes_exact'])

  def test_only_exact_frozen_error_is_repairable(self):
    with mock.patch.object(
        analyzer._v332, '_validate_freeze_and_start',
        side_effect=analyzer._v332.AnalysisError('different failure'),
    ):
      with self.assertRaisesRegex(
          analyzer._v332.AnalysisError, 'different failure'
      ):
        analyzer._validate_freeze_and_start_repaired(
            analyzer._RUN_DIR, bundle_root=analyzer._REPO_ROOT
        )

  def test_exact_zero_apply_result_is_accepted(self):
    analyzer._enforce_exact_controlled_stop(_controlled_result())

  def test_terminal_count_and_compiler_tamper_fail(self):
    result = _controlled_result()
    result['sidecar_audit']['audited_model_apply_count'] = 1
    with self.assertRaisesRegex(analyzer.AmendmentError, 'zero-apply'):
      analyzer._enforce_exact_controlled_stop(result)
    result = _controlled_result()
    result['provenance_audit']['compiler'][
        'graph_and_hlo_exact_to_original_v3_3'
    ] = True
    with self.assertRaisesRegex(analyzer.AmendmentError, 'zero-apply'):
      analyzer._enforce_exact_controlled_stop(result)

  def test_scientific_payload_is_rejected(self):
    result = _controlled_result()
    result['nomination'] = {'player': 'E64'}
    with self.assertRaisesRegex(analyzer.AmendmentError, 'zero-apply'):
      analyzer._enforce_exact_controlled_stop(result)
    with self.assertRaisesRegex(analyzer.AmendmentError, 'scientific payload'):
      analyzer._assert_no_scientific_payload({'shapley': {'E64': 1.0}})

  def test_analyze_delegates_under_scoped_context_only(self):
    binding = {'bound': True}
    result = _controlled_result()
    with (
        mock.patch.object(
            analyzer, '_validate_amendment_preconditions',
            return_value=binding,
        ),
        mock.patch.object(analyzer, '_repaired_validator') as repaired,
        mock.patch.object(
            analyzer._v332, 'analyze', return_value=copy.deepcopy(result)
        ) as delegated,
    ):
      observed = analyzer.analyze(
          analyzer._RUN_DIR, amendment_binding=binding,
          attempt_started_sha256='a' * 64,
      )
    repaired.assert_called_once_with()
    delegated.assert_called_once()
    self.assertEqual(observed['analysis_version'], analyzer.ANALYSIS_VERSION)
    self.assertEqual(
        observed['decision'], 'controlled_stop_compiler_graph_mismatch'
    )
    self.assertFalse(
        observed['analyzer_amendment']['scientific_summary_computed']
    )

  def test_provenance_failure_precedes_run_tree_read(self):
    expected = {
        analyzer._AMENDMENT_PATH: analyzer.AMENDMENT_SHA256,
        analyzer._ORIGINAL_ANALYZER_PATH: analyzer.V3_3_2_ANALYZER_SHA256,
        analyzer._ORIGINAL_TEST_PATH: analyzer.V3_3_2_TEST_SHA256,
        analyzer._ORIGINAL_FREEZE_PATH: analyzer.V3_3_2_FREEZE_SHA256,
    }
    with (
        mock.patch.object(
            analyzer, '_validate_freeze',
            side_effect=analyzer.AmendmentError('bundle failed'),
        ),
        mock.patch.object(analyzer, '_walk_exact_tree') as tree,
        mock.patch.object(
            analyzer, '_sha256', side_effect=lambda path: expected[Path(path)]
        ),
    ):
      with self.assertRaisesRegex(analyzer.AmendmentError, 'bundle failed'):
        analyzer._validate_amendment_preconditions(
            analyzer._RUN_DIR, analyzer._REPO_ROOT
        )
    tree.assert_not_called()

  def test_frozen_v332_bytes_remain_exact(self):
    self.assertEqual(
        _sha(analyzer._ORIGINAL_ANALYZER_PATH),
        analyzer.V3_3_2_ANALYZER_SHA256,
    )
    self.assertEqual(
        _sha(analyzer._ORIGINAL_TEST_PATH), analyzer.V3_3_2_TEST_SHA256
    )
    self.assertEqual(
        _sha(analyzer._ORIGINAL_FREEZE_PATH), analyzer.V3_3_2_FREEZE_SHA256
    )


class AppendOnlyTest(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)

  def tearDown(self):
    self.temporary.cleanup()

  def test_start_is_create_exclusive_and_bound(self):
    attempt = self.root / 'attempt'
    analysis = self.root / 'analysis'
    with (
        mock.patch.object(analyzer, '_ATTEMPT_DIR', attempt),
        mock.patch.object(analyzer, '_ANALYSIS_DIR', analysis),
        mock.patch.object(analyzer, '_RUN_DIR', self.root / 'run'),
    ):
      digest = analyzer._start_attempt({'binding': True})
      self.assertTrue(analyzer._is_sha256(digest))
      value = json.loads(
          (attempt / 'ANALYSIS_ATTEMPT_STARTED.json').read_text()
      )
      self.assertFalse(value['model_rerun_permitted'])
      self.assertFalse(value['scientific_values_read_before_attempt'])
      with self.assertRaises(FileExistsError):
        analyzer._start_attempt({'binding': True})

  def test_failure_terminal_is_create_exclusive(self):
    attempt = self.root / 'attempt'
    attempt.mkdir()
    with mock.patch.object(analyzer, '_ATTEMPT_DIR', attempt):
      analyzer._persist_terminal(
          'ANALYSIS_FAILURE.json', {'status': 'failed_consumed_no_retry'},
          'a' * 64,
      )
      with self.assertRaises(FileExistsError):
        analyzer._persist_terminal(
            'ANALYSIS_FAILURE.json', {'status': 'retry'}, 'a' * 64
        )

  def test_rendered_markdown_has_no_biological_claim(self):
    text = analyzer.render_markdown(_controlled_result())
    self.assertIn('no biological evidence', text)
    self.assertIn('zero model applies', text)
    self.assertIn('Combined analysis permitted:** no', text)


if __name__ == '__main__':
  unittest.main()
