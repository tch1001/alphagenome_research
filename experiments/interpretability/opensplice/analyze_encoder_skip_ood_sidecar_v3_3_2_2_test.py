"""CPU-only tests for the v3.3.2.2 saved-validator repair."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock


_MODULE_PATH = Path(__file__).with_name(
    'analyze_encoder_skip_ood_sidecar_v3_3_2_2.py'
)
_SPEC = importlib.util.spec_from_file_location(
    'analyze_encoder_skip_ood_sidecar_v3_3_2_2', _MODULE_PATH
)
analyzer = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = analyzer
_SPEC.loader.exec_module(analyzer)


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
          'audited_record_count': 0, 'valid_record_count': 0,
          'invalid_record_count': 0, 'audited_model_apply_count': 0,
          'raw_artifact_count': 0, 'id0_all20': False, 'id255_all20': False,
      },
      'provenance_audit': {
          'compiler': {'graph_and_hlo_exact_to_original_v3_3': False}
      },
  }


class SavedReferenceTest(unittest.TestCase):

  def tearDown(self):
    analyzer._v332._validate_freeze_and_start = (  # pylint: disable=protected-access
        analyzer._SAVED_FROZEN_VALIDATOR
    )

  def test_saved_reference_is_original_and_not_repaired(self):
    self.assertIs(
        analyzer._v332._validate_freeze_and_start,  # pylint: disable=protected-access
        analyzer._SAVED_FROZEN_VALIDATOR,
    )
    self.assertIsNot(
        analyzer._SAVED_FROZEN_VALIDATOR,
        analyzer._validate_freeze_and_start_repaired,
    )

  def test_context_installs_and_restores_exact_identity(self):
    saved = analyzer._SAVED_FROZEN_VALIDATOR
    with analyzer._scoped_repair():
      self.assertIs(
          analyzer._v332._validate_freeze_and_start,  # pylint: disable=protected-access
          analyzer._validate_freeze_and_start_repaired,
      )
    self.assertIs(analyzer._v332._validate_freeze_and_start, saved)  # pylint: disable=protected-access

  def test_context_restores_after_body_exception(self):
    saved = analyzer._SAVED_FROZEN_VALIDATOR
    with self.assertRaisesRegex(RuntimeError, 'body failed'):
      with analyzer._scoped_repair():
        raise RuntimeError('body failed')
    self.assertIs(analyzer._v332._validate_freeze_and_start, saved)  # pylint: disable=protected-access

  def test_direct_or_concurrent_prepatch_fails(self):
    analyzer._v332._validate_freeze_and_start = lambda *a, **k: None  # pylint: disable=protected-access
    with self.assertRaisesRegex(analyzer.AmendmentError, 'prepatched'):
      with analyzer._scoped_repair():
        pass

  def test_concurrent_change_inside_context_fails_and_restores(self):
    with self.assertRaisesRegex(analyzer.AmendmentError, 'concurrently'):
      with analyzer._scoped_repair():
        analyzer._v332._validate_freeze_and_start = lambda *a, **k: None  # pylint: disable=protected-access
    self.assertIs(
        analyzer._v332._validate_freeze_and_start,  # pylint: disable=protected-access
        analyzer._SAVED_FROZEN_VALIDATOR,
    )

  def test_saved_callable_called_once_without_recursion(self):
    calls = []

    def saved(*args, **kwargs):
      calls.append((args, kwargs))
      raise analyzer._v332.AnalysisError(analyzer.KNOWN_SCHEMA_ERROR)

    old_limit = sys.getrecursionlimit()
    try:
      sys.setrecursionlimit(80)
      with mock.patch.object(analyzer, '_SAVED_FROZEN_VALIDATOR', saved):
        analyzer._call_saved_frozen_validator(
            analyzer._v3321._RUN_DIR, bundle_root=analyzer._REPO_ROOT  # pylint: disable=protected-access
        )
    finally:
      sys.setrecursionlimit(old_limit)
    self.assertEqual(len(calls), 1)

  def test_every_other_saved_error_propagates(self):
    def saved(*unused_args, **unused_kwargs):
      raise analyzer._v332.AnalysisError('different frozen failure')

    with mock.patch.object(analyzer, '_SAVED_FROZEN_VALIDATOR', saved):
      with self.assertRaisesRegex(
          analyzer._v332.AnalysisError, 'different frozen failure'
      ):
        analyzer._call_saved_frozen_validator(
            analyzer._v3321._RUN_DIR, bundle_root=analyzer._REPO_ROOT  # pylint: disable=protected-access
        )

  def test_context_restores_after_repaired_validator_exception(self):
    def saved(*unused_args, **unused_kwargs):
      raise analyzer._v332.AnalysisError('different frozen failure')

    with (
        mock.patch.object(analyzer, '_SAVED_FROZEN_VALIDATOR', saved),
        mock.patch.object(analyzer._v332, '_validate_freeze_and_start', saved),
    ):
      with self.assertRaisesRegex(
          analyzer._v332.AnalysisError, 'different frozen failure'
      ):
        with analyzer._scoped_repair():
          analyzer._v332._validate_freeze_and_start(  # pylint: disable=protected-access
              analyzer._v3321._RUN_DIR, bundle_root=analyzer._REPO_ROOT  # pylint: disable=protected-access
          )
      self.assertIs(analyzer._v332._validate_freeze_and_start, saved)  # pylint: disable=protected-access

  def test_real_context_delegated_analyze_calls_saved_once(self):
    calls = []
    reconstructed = ({}, 'f' * 64, {}, {}, {})
    result = _controlled_result()

    def saved(*args, **kwargs):
      calls.append((args, kwargs))
      raise analyzer._v332.AnalysisError(analyzer.KNOWN_SCHEMA_ERROR)

    def delegated(run_dir, *, bundle_root):
      self.assertIs(
          analyzer._v332._validate_freeze_and_start,  # pylint: disable=protected-access
          analyzer._validate_freeze_and_start_repaired,
      )
      observed = analyzer._v332._validate_freeze_and_start(  # pylint: disable=protected-access
          run_dir, bundle_root=bundle_root
      )
      self.assertEqual(observed, reconstructed)
      return copy.deepcopy(result)

    binding = {'bound': True}
    with (
        mock.patch.object(analyzer, '_SAVED_FROZEN_VALIDATOR', saved),
        mock.patch.object(
            analyzer._v332, '_validate_freeze_and_start', saved
        ),
        mock.patch.object(
            analyzer, '_validate_amendment_preconditions', return_value=binding
        ),
        mock.patch.object(
            analyzer, '_reconstruct_frozen_return', return_value=reconstructed
        ),
        mock.patch.object(analyzer._v332, 'analyze', side_effect=delegated),
    ):
      observed = analyzer.analyze(
          analyzer._v3321._RUN_DIR,  # pylint: disable=protected-access
          amendment_binding=binding, attempt_started_sha256='a' * 64,
      )
      self.assertIs(analyzer._v332._validate_freeze_and_start, saved)  # pylint: disable=protected-access
    self.assertEqual(len(calls), 1)
    self.assertEqual(observed['analysis_version'], analyzer.ANALYSIS_VERSION)
    self.assertFalse(observed['combined_analysis_permitted'])


class ConsumedFailureTest(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.copy_index = 0

  def tearDown(self):
    self.temporary.cleanup()

  def copy_attempt(self) -> Path:
    target = self.root / f'attempt_{self.copy_index}'
    self.copy_index += 1
    shutil.copytree(analyzer._V3_3_2_1_ATTEMPT_DIR, target)
    return target

  def validate(self, target: Path):
    with (
        mock.patch.object(analyzer, '_V3_3_2_1_ATTEMPT_DIR', target),
        mock.patch.object(analyzer, '_V3_3_2_1_ANALYSIS_DIR', self.root / 'old-output'),
        mock.patch.object(analyzer, '_V3_3_2_ANALYSIS_DIR', self.root / 'older-output'),
    ):
      return analyzer._validate_consumed_v3_3_2_1()

  def test_exact_real_failure_attempt(self):
    audit = analyzer._validate_consumed_v3_3_2_1()
    self.assertEqual(audit['file_count'], 2)
    self.assertEqual(audit['tree_sha256'], analyzer.FAILED_ATTEMPT_TREE_SHA256)
    self.assertFalse(audit['scientific_values_read'])

  def test_missing_extra_and_mutated_file_fail(self):
    target = self.copy_attempt()
    (target / 'ANALYSIS_FAILURE.json').unlink()
    with self.assertRaisesRegex(analyzer.AmendmentError, 'membership'):
      self.validate(target)
    target = self.copy_attempt()
    (target / 'extra').write_bytes(b'')
    with self.assertRaisesRegex(analyzer.AmendmentError, 'membership'):
      self.validate(target)
    target = self.copy_attempt()
    failure_path = target / 'ANALYSIS_FAILURE.json'
    failure_path.chmod(0o644)
    failure_path.write_bytes(b'{}\n')
    with self.assertRaisesRegex(analyzer.AmendmentError, 'bytes changed'):
      self.validate(target)

  def test_symlink_directory_and_special_fail(self):
    target = self.copy_attempt()
    (target / 'ANALYSIS_FAILURE.json').unlink()
    (target / 'ANALYSIS_FAILURE.json').symlink_to(
        analyzer._V3_3_2_1_ATTEMPT_DIR / 'ANALYSIS_FAILURE.json'
    )
    with self.assertRaisesRegex(analyzer.AmendmentError, 'directory/symlink'):
      self.validate(target)
    target = self.copy_attempt()
    (target / 'empty').mkdir()
    with self.assertRaisesRegex(analyzer.AmendmentError, 'directory/symlink'):
      self.validate(target)
    target = self.copy_attempt()
    (target / 'fifo').touch()
    # A regular extra already exercises membership; special-node behavior is
    # covered by replacing it with a FIFO where supported.
    (target / 'fifo').unlink()
    import os
    os.mkfifo(target / 'fifo')
    with self.assertRaisesRegex(analyzer.AmendmentError, 'directory/symlink'):
      self.validate(target)

  def test_later_output_or_terminal_fails(self):
    target = self.copy_attempt()
    (target / 'ANALYSIS_COMPLETE.json').write_bytes(b'{}\n')
    with self.assertRaisesRegex(analyzer.AmendmentError, 'membership'):
      self.validate(target)
    target = self.copy_attempt()
    output = self.root / 'old-output'
    output.mkdir()
    with (
        mock.patch.object(analyzer, '_V3_3_2_1_ATTEMPT_DIR', target),
        mock.patch.object(analyzer, '_V3_3_2_1_ANALYSIS_DIR', output),
        mock.patch.object(analyzer, '_V3_3_2_ANALYSIS_DIR', self.root / 'older-output'),
    ):
      with self.assertRaisesRegex(analyzer.AmendmentError, 'forbidden'):
        analyzer._validate_consumed_v3_3_2_1()


class BoundaryTest(unittest.TestCase):

  def test_all_five_prior_files_match_historical_commit(self):
    audit = analyzer._validate_v3_3_2_1_historical_bundle(
        analyzer._REPO_ROOT
    )
    self.assertEqual(audit['file_count'], 5)
    self.assertTrue(audit['current_and_historical_bytes_exact'])

  def test_missing_or_tampered_historical_blob_fails(self):
    with mock.patch.object(
        analyzer.subprocess, 'check_output',
        side_effect=analyzer.subprocess.CalledProcessError(1, ('git', 'show')),
    ):
      with self.assertRaisesRegex(analyzer.AmendmentError, 'blob is absent'):
        analyzer._validate_v3_3_2_1_historical_bundle(
            analyzer._REPO_ROOT
        )
    with mock.patch.object(
        analyzer.subprocess, 'check_output', return_value=b'tampered blob'
    ):
      with self.assertRaisesRegex(analyzer.AmendmentError, 'blob bytes changed'):
        analyzer._validate_v3_3_2_1_historical_bundle(
            analyzer._REPO_ROOT
        )

  def test_new_freeze_schema_and_head_binding(self):
    with (
        mock.patch.object(analyzer.subprocess, 'run'),
        mock.patch.object(
            analyzer.subprocess, 'check_output',
            return_value=analyzer._FREEZE_PATH.read_bytes(),
        ),
    ):
      freeze = analyzer._validate_freeze(analyzer._REPO_ROOT)
    self.assertEqual(
        freeze['v3_3_2_1_attempt_tree_sha256'],
        analyzer.FAILED_ATTEMPT_TREE_SHA256,
    )

  def test_untracked_or_changed_new_freeze_fails(self):
    freeze_relative = str(analyzer._FREEZE_PATH.relative_to(analyzer._REPO_ROOT))

    def untracked(command, **unused):
      if freeze_relative in command:
        raise analyzer.subprocess.CalledProcessError(1, command)
      return mock.DEFAULT

    with mock.patch.object(analyzer.subprocess, 'run', side_effect=untracked):
      with self.assertRaisesRegex(analyzer.AmendmentError, 'not committed'):
        analyzer._validate_freeze(analyzer._REPO_ROOT)
    with (
        mock.patch.object(analyzer.subprocess, 'run'),
        mock.patch.object(analyzer.subprocess, 'check_output', return_value=b'{}\n'),
    ):
      with self.assertRaisesRegex(analyzer.AmendmentError, 'HEAD blob'):
        analyzer._validate_freeze(analyzer._REPO_ROOT)

  def test_prior_bundle_hashes_remain_exact(self):
    for path, digest in (
        (analyzer._V3_3_2_1_ANALYZER_PATH, analyzer.V3_3_2_1_ANALYZER_SHA256),
        (analyzer._V3_3_2_1_TEST_PATH, analyzer.V3_3_2_1_TEST_SHA256),
        (analyzer._V3_3_2_1_FREEZE_PATH, analyzer.V3_3_2_1_FREEZE_SHA256),
        (analyzer._V3_3_2_1_SHELL_PATH, analyzer.V3_3_2_1_SHELL_SHA256),
        (analyzer._V3_3_2_1_AMENDMENT_PATH, analyzer.V3_3_2_1_AMENDMENT_SHA256),
    ):
      self.assertEqual(_sha(path), digest)

  def test_only_zero_apply_controlled_result_is_accepted(self):
    analyzer._v3321._enforce_exact_controlled_stop(_controlled_result())  # pylint: disable=protected-access
    result = _controlled_result()
    result['sidecar_audit']['audited_model_apply_count'] = 1
    with self.assertRaises(analyzer._v3321.AmendmentError):
      analyzer._v3321._enforce_exact_controlled_stop(result)  # pylint: disable=protected-access

  def test_scientific_payload_fails(self):
    with self.assertRaises(analyzer._v3321.AmendmentError):
      analyzer._v3321._assert_no_scientific_payload(  # pylint: disable=protected-access
          {'shapley': {'E64': 1.0}}
      )

  def test_fresh_destinations_absent(self):
    self.assertFalse(analyzer._ATTEMPT_DIR.exists())
    self.assertFalse(analyzer._ANALYSIS_DIR.exists())

  def test_markdown_plainly_denies_biology(self):
    text = analyzer.render_markdown(_controlled_result())
    self.assertIn('no biological evidence', text)
    self.assertIn('Combined analysis permitted:** no', text)

  def test_fresh_attempt_is_create_exclusive(self):
    with tempfile.TemporaryDirectory() as temporary:
      attempt = Path(temporary) / 'attempt'
      output = Path(temporary) / 'output'
      with (
          mock.patch.object(analyzer, '_ATTEMPT_DIR', attempt),
          mock.patch.object(analyzer, '_ANALYSIS_DIR', output),
      ):
        digest = analyzer._start_attempt({'bound': True})
        self.assertTrue(analyzer._is_sha256(digest))
        with self.assertRaises(FileExistsError):
          analyzer._start_attempt({'bound': True})


class StartedAttemptSchemaTest(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.attempt = self.root / 'attempt'
    self.output = self.root / 'output'
    self.binding = {'bound': True}

  def tearDown(self):
    self.temporary.cleanup()

  def make(self):
    with (
        mock.patch.object(analyzer, '_ATTEMPT_DIR', self.attempt),
        mock.patch.object(analyzer, '_ANALYSIS_DIR', self.output),
    ):
      digest = analyzer._start_attempt(self.binding)
    path = self.attempt / 'ANALYSIS_ATTEMPT_STARTED.json'
    return path, json.loads(path.read_text()), digest

  def validate(self, path: Path, value: dict):
    path.chmod(0o644)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    )
    digest = _sha(path)
    with (
        mock.patch.object(analyzer, '_ATTEMPT_DIR', self.attempt),
        mock.patch.object(analyzer, '_ANALYSIS_DIR', self.output),
    ):
      analyzer._validate_started_attempt(self.binding, digest)

  def test_exact_started_attempt_passes(self):
    path, value, _ = self.make()
    self.validate(path, value)

  def test_extra_and_missing_key_fail(self):
    for mutation in ('extra', 'missing'):
      with self.subTest(mutation=mutation):
        self.attempt = self.root / f'attempt_{mutation}'
        path, value, _ = self.make()
        if mutation == 'extra':
          value['extra'] = None
        else:
          value.pop('status')
        with self.assertRaisesRegex(analyzer.AmendmentError, 'content changed'):
          self.validate(path, value)

  def test_each_path_field_tamper_fails(self):
    for field in ('run_dir', 'output_json', 'output_markdown'):
      with self.subTest(field=field):
        self.attempt = self.root / f'attempt_{field}'
        path, value, _ = self.make()
        value[field] = str(self.root / f'tampered_{field}')
        with self.assertRaisesRegex(analyzer.AmendmentError, 'content changed'):
          self.validate(path, value)

  def test_nonfinite_or_malformed_time_fails(self):
    for index, timestamp in enumerate((True, None, '1', float('inf'))):
      with self.subTest(timestamp=timestamp):
        self.attempt = self.root / f'attempt_time_{index}'
        path, value, _ = self.make()
        value['started_at_unix_s'] = timestamp
        path.chmod(0o644)
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + '\n'
        )
        digest = _sha(path)
        with (
            mock.patch.object(analyzer, '_ATTEMPT_DIR', self.attempt),
            mock.patch.object(analyzer, '_ANALYSIS_DIR', self.output),
        ):
          with self.assertRaisesRegex(analyzer.AmendmentError, 'content changed'):
            analyzer._validate_started_attempt(self.binding, digest)


if __name__ == '__main__':
  unittest.main()
