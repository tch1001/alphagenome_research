#!/usr/bin/env python3
"""CPU-only tests for the standalone v3.3.4.5.1 structural analyzer."""

from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
ANALYZER_PATH = HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_4_5_1.py'
GENERATOR_PATH = (
    HERE / 'generate_encoder_skip_ood_sidecar_analysis_v3_3_4_5_1_freeze.py'
)
SHELL_PATH = HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_4_5_1.sh'
FREEZE_PATH = HERE / 'encoder_skip_ood_sidecar_analysis_v3_3_4_5_1_freeze.json'


def load(path: Path, name: str):
  spec = importlib.util.spec_from_file_location(name, path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


analyzer = load(ANALYZER_PATH, '_v33451_analyzer_test_target')
generator = load(GENERATOR_PATH, '_v33451_generator_test_target')


@contextmanager
def publication_sandbox():
  with tempfile.TemporaryDirectory() as directory:
    parent = Path(directory)
    roots = {
        'analysis_attempt': parent / 'attempt',
        'analysis_output': parent / 'output',
    }
    with (
        mock.patch.object(analyzer, '_PUBLICATION_ROOTS', roots),
        mock.patch.object(analyzer, '_ANALYSIS_ATTEMPT_DIR', roots['analysis_attempt']),
        mock.patch.object(analyzer, '_ANALYSIS_DIR', roots['analysis_output']),
    ):
      analyzer._PUBLICATION_DIRECTORIES.clear()
      analyzer._PUBLICATION_SUCCESSES.clear()
      analyzer._PUBLICATION_TEMP_ORPHANS.clear()
      analyzer._PUBLICATION_UNCERTAIN_FINALS.clear()
      analyzer._PUBLICATION_PREEXISTING.clear()
      analyzer._PUBLICATION_FAILURE.clear()
      analyzer._PUBLICATION_FAILURE_TERMINAL_USED.clear()
      analyzer._PUBLICATION_UNBINDABLE_ROOTS.clear()
      analyzer._PUBLICATION_ORDINAL = 0
      analyzer._PUBLICATION_TEST_FAIL_STAGE = None
      try:
        yield parent, roots
      finally:
        analyzer._PUBLICATION_TEST_FAIL_STAGE = None
        for descriptor, _dev, _ino in analyzer._PUBLICATION_DIRECTORIES.values():
          try:
            os.close(descriptor)
          except OSError:
            pass
        analyzer._PUBLICATION_DIRECTORIES.clear()
        analyzer._PUBLICATION_UNBINDABLE_ROOTS.clear()


def cache_fixture(root: Path) -> tuple[tuple[object, ...], ...]:
  root.mkdir(mode=0o700)
  (root / 'triton').mkdir(mode=0o700)
  (root / 'xdg').mkdir(mode=0o700)
  (root / 'xdg' / 'matplotlib').mkdir(mode=0o700)
  file_path = root / 'xdg' / 'matplotlib' / 'fontlist-v3.11.0.json'
  file_path.write_bytes(b'{"cache":"fixture"}\n')
  file_path.chmod(0o600)
  rows = []
  for relative, kind in (
      ('.', 'directory'), ('triton', 'directory'), ('xdg', 'directory'),
      ('xdg/matplotlib', 'directory'),
      ('xdg/matplotlib/fontlist-v3.11.0.json', 'regular'),
  ):
    path = root if relative == '.' else root / relative
    observed = path.lstat()
    rows.append((
        relative, kind, f'{stat.S_IMODE(observed.st_mode):04o}',
        observed.st_size, observed.st_dev, observed.st_ino,
        observed.st_nlink,
        hashlib.sha256(path.read_bytes()).hexdigest() if kind == 'regular'
        else None,
    ))
  return tuple(rows)


def cache_digest(rows: tuple[tuple[object, ...], ...]) -> tuple[str, str]:
  file_relative = str(rows[-1][0])
  file_sha = str(rows[-1][-1])
  file_digest = hashlib.sha256(
      file_relative.encode() + b'\0' + bytes.fromhex(file_sha)
  ).hexdigest()
  combined = hashlib.sha256()
  for row in rows[:-1]:
    combined.update(b'D\0' + str(row[0]).encode() + b'\0')
  combined.update(b'F\0' + file_relative.encode() + b'\0')
  combined.update(bytes.fromhex(file_sha))
  return file_digest, combined.hexdigest()


class StaticContractTest(unittest.TestCase):

  def test_exact_record_key_counts(self):
    self.assertEqual(len(analyzer._V33451_FREEZE_KEYS), 20)
    self.assertEqual(len(analyzer._V33451_START_KEYS), 22)
    self.assertEqual(len(analyzer._V33451_ANALYSIS_KEYS), 26)
    self.assertEqual(len(analyzer._V33451_COMPLETE_KEYS), 11)
    self.assertEqual(len(analyzer._V33451_FAILURE_KEYS), 13)
    self.assertEqual(len(analyzer.PUBLICATION_SUCCESS_KEYS), 19)
    self.assertEqual(len(analyzer.PUBLICATION_FAILURE_KEYS), 19)
    self.assertEqual(len(analyzer.PUBLICATION_AUDIT_KEYS), 15)
    self.assertEqual(len(analyzer._V33451_OUTPUT_STATE_KEYS), 13)

  def test_namespace_split_is_literal(self):
    self.assertEqual(
        analyzer.PUBLICATION_SCHEMA_VERSION,
        'v3.3.4.5-named-temp-renameat2-noreplace-v1',
    )
    self.assertEqual(
        analyzer._V33451_PUBLICATION_SCHEMA_VERSION,
        'v3.3.4.5.1-named-temp-renameat2-noreplace-v1',
    )
    self.assertNotEqual(
        analyzer.PUBLICATION_SCHEMA_VERSION,
        analyzer._V33451_PUBLICATION_SCHEMA_VERSION,
    )

  def test_no_old_analyzer_or_model_import(self):
    forbidden = (
        'analyze_encoder_skip_ood_sidecar_v3_3_4_5',
        'validate_encoder_skip_ood_sidecar_bootstrap',
        'run_encoder_skip_ood_sidecar', 'jax', 'jaxlib',
        'alphagenome', 'alphagenome_research.model',
    )
    for path in (ANALYZER_PATH, GENERATOR_PATH):
      tree = ast.parse(path.read_text())
      imports = []
      for node in ast.walk(tree):
        if isinstance(node, ast.Import):
          imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
          imports.append(node.module or '')
      self.assertFalse([
          name for name in imports
          if any(name == item or name.startswith(item + '.') for item in forbidden)
      ])
    analyzer_source = ANALYZER_PATH.read_text()
    for unsafe in ('.read_text(', '.read_bytes(', 'Path.open('):
      self.assertNotIn(unsafe, analyzer_source)

  def test_gate_order_is_source_then_immutable_then_start_allocation(self):
    freeze_gate = inspect.getsource(analyzer._v33451_validate_analysis_freeze)
    self.assertLess(
        freeze_gate.index('_v33451_validate_source_inventory'),
        freeze_gate.index('_v33451_validate_immutable_contract'),
    )
    main_source = inspect.getsource(analyzer.main)
    self.assertLess(
        main_source.index("phase='pre_start'"),
        main_source.index('ensure_publication_directory'),
    )

  def test_shell_is_exact_cpu_entrypoint(self):
    text = SHELL_PATH.read_text()
    self.assertIn('umask 077', text)
    self.assertIn('/usr/bin/python3', text)
    self.assertIn('analyze_encoder_skip_ood_sidecar_v3_3_4_5_1.py', text)
    self.assertIn('--acknowledge-structural-only-v3-3-4-5-1', text)
    self.assertNotIn('analyze_encoder_skip_ood_sidecar_v3_3_4_5.py"', text)
    subprocess.check_call(('bash', '-n', str(SHELL_PATH)))

  def test_modes_and_freeze_absence(self):
    self.assertEqual(stat.S_IMODE(ANALYZER_PATH.stat().st_mode), 0o644)
    self.assertEqual(stat.S_IMODE(Path(__file__).stat().st_mode), 0o644)
    self.assertEqual(stat.S_IMODE(GENERATOR_PATH.stat().st_mode), 0o644)
    self.assertEqual(stat.S_IMODE(SHELL_PATH.stat().st_mode), 0o755)
    if FREEZE_PATH.exists():
      value = json.loads(FREEZE_PATH.read_text())
      self.assertEqual(len(value), 20)
      self.assertEqual(value['schema_version'], analyzer.ANALYSIS_SCHEMA_VERSION)
      self.assertEqual(stat.S_IMODE(FREEZE_PATH.stat().st_mode), 0o644)
    else:
      self.assertFalse(FREEZE_PATH.is_symlink())

  def test_consumed_failure_exact(self):
    value = analyzer._v33451_expected_consumed_failure()
    stderr = value['stderr_text'].encode()
    self.assertEqual(len(stderr), 1587)
    self.assertEqual(
        hashlib.sha256(stderr).hexdigest(),
        '0158926b7b41b6636bfacd2acdcf268bae7f7082b9f935edb483aa184bdd6967',
    )
    self.assertTrue(value['failed_before_start'])
    self.assertFalse(value['retry_permitted'])

  def test_fresh_paths_use_exact_semantic_keys_and_probe_all_targets(self):
    names = {
        'old_v3345_attempt': '_OLD_V3345_ANALYSIS_ATTEMPT_DIR',
        'old_v3345_output': '_OLD_V3345_ANALYSIS_DIR',
        'old_v333_attempt': '_PRIOR_ANALYZER_ATTEMPT_DIR',
        'old_v333_output': '_PRIOR_ANALYZER_OUTPUT_DIR',
        'new_attempt': '_ANALYSIS_ATTEMPT_DIR',
        'new_output': '_ANALYSIS_DIR',
    }
    with tempfile.TemporaryDirectory() as directory:
      targets = {
          attribute: Path(directory) / name
          for name, attribute in names.items()
      }
      with mock.patch.multiple(analyzer, **targets):
        self.assertEqual(
            analyzer._v33451_fresh_paths(),
            {name: 'absent' for name in names},
        )
        for name, attribute in names.items():
          with self.subTest(name=name):
            targets[attribute].write_bytes(b'collision')
            with self.assertRaises(analyzer.AnalysisError):
              analyzer._v33451_fresh_paths()
            targets[attribute].unlink()

  def test_consumed_failure_every_leaf_and_binding_fail_closed(self):
    value = analyzer._v33451_expected_consumed_failure()
    binding = analyzer._v33451_canonical_binding(value)
    analyzer._v33451_validate_consumed_failure(value, binding)

    def leaves(node, prefix=()):
      if isinstance(node, dict):
        for key, child in node.items():
          yield from leaves(child, (*prefix, key))
      elif isinstance(node, list):
        for index, child in enumerate(node):
          yield from leaves(child, (*prefix, index))
      else:
        yield prefix

    for path in leaves(value):
      with self.subTest(path=path):
        tampered = json.loads(json.dumps(value))
        parent = tampered
        for component in path[:-1]:
          parent = parent[component]
        current = parent[path[-1]]
        if isinstance(current, bool):
          parent[path[-1]] = not current
        elif current is None:
          parent[path[-1]] = 0
        elif isinstance(current, int):
          parent[path[-1]] = current + 1
        elif isinstance(current, float):
          parent[path[-1]] = current + 1.0
        else:
          parent[path[-1]] = current + 'x'
        with self.assertRaises(analyzer.AnalysisError):
          analyzer._v33451_validate_consumed_failure(tampered, binding)
    for key in ('sha256', 'size_bytes'):
      with self.subTest(binding=key):
        tampered_binding = dict(binding)
        tampered_binding[key] = (
            '0' * 64 if key == 'sha256' else binding[key] + 1
        )
        with self.assertRaises(analyzer.AnalysisError):
          analyzer._v33451_validate_consumed_failure(value, tampered_binding)

  def test_freeze_authorization_requires_live_and_git_mode_100644(self):
    head = '1' * 40
    with tempfile.TemporaryDirectory() as directory:
      freeze = Path(directory) / 'freeze.json'
      payload = b'{"freeze":true}\n'
      freeze.write_bytes(payload)
      freeze.chmod(0o644)
      relative = 'experiments/freeze.json'
      node = {
          'git_head': head, 'freeze_path': str(freeze.resolve()),
          'freeze_sha256': hashlib.sha256(payload).hexdigest(),
          'freeze_size_bytes': len(payload), 'live_equals_git_show': True,
          'tracked_clean': True,
          'authorization_source': 'external_post_commit_audit',
      }
      git_mode = ['100644']
      def check_output(command, **kwargs):
        if command[-2:] == ('rev-parse', 'HEAD'):
          return head + '\n'
        if command[-2] == 'show':
          return payload
        if command[-4] == 'ls-tree' and command[-3] == head:
          return f'{git_mode[0]} blob deadbeef\t{relative}\n'
        raise AssertionError(command)
      with (
          mock.patch.object(analyzer, '_ANALYSIS_FREEZE_PATH', freeze),
          mock.patch.object(analyzer, '_REPO_ROOT', Path(directory)),
          mock.patch.object(
              Path, 'relative_to', return_value=Path(relative)
          ),
          mock.patch.object(
              analyzer.subprocess, 'check_output', side_effect=check_output
          ),
          mock.patch.object(analyzer.subprocess, 'check_call'),
      ):
        analyzer._v33451_validate_authorization(node, {})
        freeze.chmod(0o600)
        with self.assertRaises(analyzer.AnalysisError):
          analyzer._v33451_validate_authorization(node, {})
        freeze.chmod(0o644)
        git_mode[0] = '100755'
        with self.assertRaises(analyzer.AnalysisError):
          analyzer._v33451_validate_authorization(node, {})

  def test_claim_boundary_is_structural_only(self):
    boundary = analyzer._V33451_CLAIM_BOUNDARY
    self.assertTrue(all(boundary[key] for key in boundary if key != 'combined_analysis_permitted'))
    self.assertFalse(boundary['combined_analysis_permitted'])

  def test_cpu_guard_rejects_jaxlib_and_model(self):
    for name in ('jaxlib', 'jaxlib.xla_extension', 'alphagenome_research.model.foo'):
      with self.subTest(name=name), mock.patch.dict(analyzer.sys.modules, {name: object()}):
        with self.assertRaises(analyzer.AnalysisError):
          analyzer._assert_cpu_only('sentinel')

  def test_direct_analysis_token_is_rejected(self):
    with self.assertRaisesRegex(analyzer.AnalysisError, 'Direct'):
      analyzer._v33451_structural_analyze(
          token=object(), started_sha256='0' * 64, authorization={}
      )
    with self.assertRaisesRegex(analyzer.AnalysisError, 'legacy'):
      analyzer.analyze(analyzer._RUN_DIR)
    with self.assertRaisesRegex(analyzer.AnalysisError, 'precheck'):
      analyzer._analysis_attempt_precheck(analyzer._RUN_DIR)

  def test_inherited_132_rows_are_exact_old_freeze_rows(self):
    rows = analyzer._v33451_expected_inherited_source_rows()
    self.assertEqual(len(rows), 132)
    analyzer._v33451_require_inherited_source_rows(rows)
    for index in range(len(rows)):
      for key in ('path', 'sha256', 'size_bytes', 'git_mode', 'authority_commit'):
        with self.subTest(index=index, key=key):
          tampered = json.loads(json.dumps(rows))
          if key == 'size_bytes':
            tampered[index][key] += 1
          elif key == 'authority_commit':
            tampered[index][key] = '0' * 40
          else:
            tampered[index][key] = str(tampered[index][key]) + 'x'
          with self.assertRaises(analyzer.AnalysisError):
            analyzer._v33451_require_inherited_source_rows(tampered)
    reordered = list(reversed(rows))
    with self.assertRaises(analyzer.AnalysisError):
      analyzer._v33451_require_inherited_source_rows(reordered)


class PriorCacheTest(unittest.TestCase):

  def test_live_prior_cache_passes_directory_aware_gate(self):
    value = analyzer._validate_prior_cache_directory_aware()
    self.assertEqual(value['directory_paths'], ['.', 'triton', 'xdg', 'xdg/matplotlib'])
    self.assertEqual(value['file_count'], 1)
    self.assertEqual(value['file_tree_sha256'], analyzer.PRIOR_CACHE_FILE_TREE_SHA256)
    self.assertEqual(value['directory_file_tree_sha256'], analyzer.PRIOR_CACHE_TREE_SHA256)

  def _patch_fixture(self, root: Path, rows):
    file_digest, combined = cache_digest(rows)
    return mock.patch.multiple(
        analyzer, _PRIOR_CACHE_DIR=root, _PRIOR_CACHE_LSTAT_ROWS=rows,
        PRIOR_CACHE_FILE_TREE_SHA256=file_digest,
        PRIOR_CACHE_TREE_SHA256=combined,
    )

  def test_empty_triton_is_accepted(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / 'cache'
      rows = cache_fixture(root)
      with self._patch_fixture(root, rows):
        value = analyzer._validate_prior_cache_directory_aware()
      self.assertIn('triton', value['directory_paths'])

  def test_extra_empty_directory_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / 'cache'
      rows = cache_fixture(root)
      (root / 'extra').mkdir()
      with self._patch_fixture(root, rows), self.assertRaises(analyzer.AnalysisError):
        analyzer._validate_prior_cache_directory_aware()

  def test_fifo_replacement_fails_without_blocking(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / 'cache'
      rows = cache_fixture(root)
      target = root / 'xdg/matplotlib/fontlist-v3.11.0.json'
      target.unlink()
      os.mkfifo(target, 0o600)
      with self._patch_fixture(root, rows), self.assertRaises(analyzer.AnalysisError):
        analyzer._validate_prior_cache_directory_aware()

  def test_root_swap_during_read_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      parent = Path(directory)
      root = parent / 'cache'
      rows = cache_fixture(root)
      original_read = analyzer.os.read
      swapped = False
      def swap_then_read(fd, size):
        nonlocal swapped
        if not swapped:
          swapped = True
          root.rename(parent / 'old-cache')
          root.mkdir(mode=0o700)
        return original_read(fd, size)
      with (
          self._patch_fixture(root, rows),
          mock.patch.object(analyzer.os, 'read', side_effect=swap_then_read),
          self.assertRaises(analyzer.AnalysisError),
      ):
        analyzer._validate_prior_cache_directory_aware()

  def test_each_frozen_cache_row_field_is_fail_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / 'cache'
      rows = cache_fixture(root)
      file_digest, combined = cache_digest(rows)
      for index in range(len(rows)):
        for field in range(2, 8):
          with self.subTest(row=index, field=field):
            tampered = [list(row) for row in rows]
            value = tampered[index][field]
            if isinstance(value, int):
              tampered[index][field] = value + 1
            elif value is None:
              tampered[index][field] = '0' * 64
            else:
              tampered[index][field] = str(value) + 'x'
            with (
                mock.patch.multiple(
                    analyzer, _PRIOR_CACHE_DIR=root,
                    _PRIOR_CACHE_LSTAT_ROWS=tuple(tuple(row) for row in tampered),
                    PRIOR_CACHE_FILE_TREE_SHA256=file_digest,
                    PRIOR_CACHE_TREE_SHA256=combined,
                ),
                self.assertRaises(analyzer.AnalysisError),
            ):
              analyzer._validate_prior_cache_directory_aware()

  def test_cache_byte_mode_hardlink_and_membership_tampers(self):
    mutations = (
        'bytes', 'mode', 'hardlink', 'missing_file', 'missing_matplotlib',
        'missing_xdg', 'missing_triton', 'symlink_file', 'symlink_triton',
        'fifo_triton', 'extra_file',
    )
    for mutation in mutations:
      with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / 'cache'
        rows = cache_fixture(root)
        target = root / 'xdg/matplotlib/fontlist-v3.11.0.json'
        if mutation == 'bytes':
          target.write_bytes(b'drift')
        elif mutation == 'mode':
          target.chmod(0o644)
        elif mutation == 'hardlink':
          os.link(target, root / 'xdg/matplotlib/other')
        elif mutation == 'missing_file':
          target.unlink()
        elif mutation == 'missing_matplotlib':
          target.unlink()
          (root / 'xdg/matplotlib').rmdir()
        elif mutation == 'missing_xdg':
          target.unlink()
          (root / 'xdg/matplotlib').rmdir()
          (root / 'xdg').rmdir()
        elif mutation == 'missing_triton':
          (root / 'triton').rmdir()
        elif mutation == 'symlink_file':
          target.unlink()
          target.symlink_to(root / 'xdg')
        elif mutation == 'symlink_triton':
          (root / 'triton').rmdir()
          (root / 'triton').symlink_to('xdg')
        elif mutation == 'fifo_triton':
          (root / 'triton').rmdir()
          os.mkfifo(root / 'triton', 0o700)
        else:
          (root / 'extra').write_bytes(b'extra')
        with self._patch_fixture(root, rows), self.assertRaises(analyzer.AnalysisError):
          analyzer._validate_prior_cache_directory_aware()


class PublicationTest(unittest.TestCase):

  def test_absent_output_state_is_exact13(self):
    with publication_sandbox():
      state = analyzer._v33451_output_state('analysis_output')
      self.assertEqual(set(state), analyzer._V33451_OUTPUT_STATE_KEYS)
      self.assertEqual(state['state'], 'absent')
      self.assertIsNone(state['root_lstat'])
      self.assertEqual(state['file_tree_sha256'], analyzer.EMPTY_SHA256)

  def test_success_roundtrip_and_exact_audit(self):
    with publication_sandbox():
      analyzer.ensure_publication_directory(
          'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json', 'start'
      )
      success = analyzer.publish_bytes(
          'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json', b'{"ok":true}\n',
          'start',
      )
      self.assertEqual(set(success), set(analyzer.PUBLICATION_SUCCESS_KEYS))
      self.assertEqual(success['mode'], '0400')
      value = analyzer._v33451_read_publication_json(
          'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json', 'START'
      )
      self.assertEqual(value, {'ok': True})
      audit = analyzer.publication_audit('analysis_attempt')
      self.assertEqual(set(audit), set(analyzer.PUBLICATION_AUDIT_KEYS))
      self.assertEqual(audit['successful_final_count_before_terminal'], 1)
      state = analyzer._v33451_output_state('analysis_attempt')
      self.assertEqual(set(state), set(analyzer._V33451_OUTPUT_STATE_KEYS))
      self.assertEqual(state['state'], 'published_prefix')

  def test_active_start_reader_rejects_same_byte_path_swap(self):
    with publication_sandbox() as (_parent, roots):
      analyzer.ensure_publication_directory(
          'analysis_attempt', 'START.json', 'start'
      )
      payload = b'{"ok":true}\n'
      analyzer.publish_bytes('analysis_attempt', 'START.json', payload, 'start')
      original_read = analyzer.os.read
      calls = 0
      def swap_on_second_reader(descriptor, size):
        nonlocal calls
        calls += 1
        if calls == 3:
          target = roots['analysis_attempt'] / 'START.json'
          target.rename(roots['analysis_attempt'] / 'old-START.json')
          target.write_bytes(payload)
          target.chmod(0o400)
        return original_read(descriptor, size)
      with (
          mock.patch.object(
              analyzer.os, 'read', side_effect=swap_on_second_reader
          ),
          self.assertRaises(analyzer.AnalysisError),
      ):
        analyzer._v33451_read_publication_json(
            'analysis_attempt', 'START.json', 'START'
        )

  def test_every_root_failure_stage_has_exact_flags(self):
    for stage in (
        'root_parent_open', 'root_parent_validation',
        'root_final_preexistence', 'root_mkdir', 'root_parent_fsync',
        'root_revalidation',
    ):
      with self.subTest(stage=stage), publication_sandbox():
        analyzer._PUBLICATION_TEST_FAIL_STAGE = stage
        with self.assertRaises(analyzer.PublicationError) as caught:
          analyzer.ensure_publication_directory(
              'analysis_attempt', 'START.json', 'start'
          )
        failure = analyzer._v33451_validate_publication_failure(
            caught.exception.publication_failure, 'failure'
        )
        self.assertIsNone(failure['temp_relative_path'])
        self.assertEqual(
            (failure['parent_fsync_attempted'], failure['parent_fsync_succeeded']),
            (stage in {'root_parent_fsync', 'root_revalidation'},
             stage == 'root_revalidation'),
        )

  def test_publication_failure_nullability_and_stage_flags_fail_closed(self):
    with publication_sandbox():
      analyzer._PUBLICATION_TEST_FAIL_STAGE = 'root_parent_open'
      with self.assertRaises(analyzer.PublicationError) as caught:
        analyzer.ensure_publication_directory(
            'analysis_attempt', 'START.json', 'start'
        )
      root_failure = caught.exception.publication_failure
      for key, value in (
          ('temp_relative_path', '.v33451.tmp.1.000000.' + '0' * 32),
          ('publication_ordinal', 0),
          ('rename_noreplace_attempted', True),
          ('parent_fsync_attempted', True),
      ):
        with self.subTest(root_key=key):
          tampered = json.loads(json.dumps(root_failure))
          tampered[key] = value
          with self.assertRaises(analyzer.AnalysisError):
            analyzer._v33451_validate_publication_failure(tampered, 'tamper')
    with publication_sandbox():
      analyzer.ensure_publication_directory(
          'analysis_output', 'RESULT.md', 'result'
      )
      analyzer._PUBLICATION_TEST_FAIL_STAGE = 'temp_write'
      with self.assertRaises(analyzer.PublicationError) as caught:
        analyzer.publish_bytes(
            'analysis_output', 'RESULT.md', b'payload', 'result'
        )
      file_failure = caught.exception.publication_failure
      for key, value in (
          ('temp_relative_path', None), ('publication_ordinal', None),
          ('rename_noreplace_succeeded', True),
          ('parent_fsync_succeeded', True),
      ):
        with self.subTest(file_key=key):
          tampered = json.loads(json.dumps(file_failure))
          tampered[key] = value
          with self.assertRaises(analyzer.AnalysisError):
            analyzer._v33451_validate_publication_failure(tampered, 'tamper')

  def test_nonregular_entry_nullability_is_exact(self):
    state = {
        'state': 'present', 'entry_type': 'fifo', 'mode': '0600',
        'size_bytes': None, 'sha256': None, 'st_dev': 1, 'st_ino': 2,
        'st_nlink': 1,
    }
    analyzer._v33451_validate_entry_state(state, 'fifo')
    for key, value in (('size_bytes', 0), ('sha256', analyzer.EMPTY_SHA256)):
      tampered = dict(state)
      tampered[key] = value
      with self.assertRaises(analyzer.AnalysisError):
        analyzer._v33451_validate_entry_state(tampered, 'fifo')
    unreadable = analyzer._absent_publication_entry()
    unreadable['state'] = 'unreadable'
    with self.assertRaises(analyzer.AnalysisError):
      analyzer._v33451_validate_entry_state(unreadable, 'unreadable')

  def test_every_file_failure_stage_is_archivable(self):
    stages = (
        'parent_open', 'parent_validation', 'final_preexistence', 'temp_open',
        'temp_write', 'file_fsync_before_rename', 'fchmod',
        'file_fsync_after_fchmod', 'readback', 'rename_noreplace',
        'parent_fsync', 'post_publish_revalidation',
    )
    for stage in stages:
      with self.subTest(stage=stage), publication_sandbox():
        analyzer.ensure_publication_directory(
            'analysis_output', 'RESULT.md', 'result'
        )
        analyzer._PUBLICATION_TEST_FAIL_STAGE = stage
        with self.assertRaises(analyzer.PublicationError) as caught:
          analyzer.publish_bytes(
              'analysis_output', 'RESULT.md', b'payload', 'result'
          )
        failure = analyzer._v33451_validate_publication_failure(
            caught.exception.publication_failure, 'failure'
        )
        self.assertRegex(failure['temp_relative_path'], r'^\.v33451\.tmp\.')
        audit = analyzer.publication_audit('analysis_output', failure)
        self.assertEqual(set(audit), set(analyzer.PUBLICATION_AUDIT_KEYS))
        state = analyzer._v33451_output_state('analysis_output', failure)
        self.assertEqual(set(state), set(analyzer._V33451_OUTPUT_STATE_KEYS))
        present = (
            set(state['regular_final_bindings'])
            | set(state['temporary_orphan_bindings'])
            | set(state['durability_uncertain_final_bindings'])
            | {
                path for path, value in state['preexisting_entry_states'].items()
                if value['state'] == 'present'
            }
        )
        self.assertEqual(
            present,
            set(state['regular_final_bindings'])
            | set(state['temporary_orphan_bindings'])
            | set(state['durability_uncertain_final_bindings'])
            | {
                path for path, value in state['preexisting_entry_states'].items()
                if value['state'] == 'present'
            },
        )

  def test_start_publication_failure_is_consumed_without_terminal(self):
    with publication_sandbox():
      analyzer.ensure_publication_directory(
          'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json', 'start'
      )
      analyzer._PUBLICATION_TEST_FAIL_STAGE = 'temp_write'
      with self.assertRaises(analyzer.PublicationError) as caught:
        analyzer.publish_bytes(
            'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json', b'{}\n', 'start'
        )
      failure = caught.exception.publication_failure
      audit = analyzer.publication_audit('analysis_attempt', failure)
      self.assertEqual(audit['successful_final_count_before_terminal'], 0)
      self.assertEqual(audit['temporary_orphan_count'], 1)
      self.assertNotIn(
          'ANALYSIS_FAILURE.json',
          analyzer._PUBLICATION_SUCCESSES.get('analysis_attempt', {}),
      )

  def test_failure_terminal_publication_failure_is_terminal_less(self):
    with publication_sandbox():
      analyzer.ensure_publication_directory(
          'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json', 'start'
      )
      analyzer.publish_bytes(
          'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json', b'{}\n', 'start'
      )
      started_sha = analyzer._v33451_publication_file_binding(
          'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json'
      )['sha256']
      failure_record = analyzer._v33451_failure_record(
          RuntimeError('analysis failed'), started_sha,
          phase='structural_terminal_audit',
      )
      analyzer._PUBLICATION_TEST_FAIL_STAGE = 'temp_write'
      with self.assertRaises(analyzer.PublicationError) as caught:
        analyzer._v33451_write_json(
            'analysis_attempt', 'ANALYSIS_FAILURE.json', failure_record,
            'analysis_failure',
        )
      analyzer._PUBLICATION_TEST_FAIL_STAGE = None
      audit = analyzer.publication_audit(
          'analysis_attempt', caught.exception.publication_failure
      )
      self.assertNotIn(
          'ANALYSIS_FAILURE.json',
          audit['successful_final_bindings_before_terminal'],
      )
      self.assertEqual(audit['temporary_orphan_count'], 1)
      with self.assertRaises(analyzer.AnalysisError):
        analyzer.publish_bytes(
            'analysis_attempt', 'ANALYSIS_FAILURE.json', b'{}\n',
            'analysis_failure',
        )

  def test_preexisting_regular_is_bound_in_physical_tree(self):
    with publication_sandbox() as (_parent, roots):
      analyzer.ensure_publication_directory(
          'analysis_output', 'RESULT.md', 'result'
      )
      (roots['analysis_output'] / 'RESULT.md').write_bytes(b'collision')
      with self.assertRaises(analyzer.PublicationError) as caught:
        analyzer.publish_bytes(
            'analysis_output', 'RESULT.md', b'payload', 'result'
        )
      failure = caught.exception.publication_failure
      state = analyzer._v33451_output_state('analysis_output', failure)
      self.assertIn('RESULT.md', state['preexisting_entry_states'])
      self.assertNotEqual(state['file_tree_sha256'], analyzer.EMPTY_SHA256)

  def test_special_entry_states_have_null_size_and_hash(self):
    with publication_sandbox() as (_parent, roots):
      analyzer.ensure_publication_directory(
          'analysis_output', 'RESULT.md', 'result'
      )
      root = roots['analysis_output']
      os.mkfifo(root / 'fifo')
      (root / 'directory').mkdir()
      (root / 'symlink').symlink_to('directory')
      sock = socket.socket(socket.AF_UNIX)
      try:
        sock.bind(str(root / 'socket'))
        fd = analyzer._PUBLICATION_DIRECTORIES['analysis_output'][0]
        for name, kind in (
            ('fifo', 'fifo'), ('directory', 'directory'),
            ('symlink', 'symlink'), ('socket', 'socket'),
        ):
          state = analyzer._publication_entry_at(fd, name)
          self.assertEqual(state['entry_type'], kind)
          self.assertIsNone(state['size_bytes'])
          self.assertIsNone(state['sha256'])
          analyzer._v33451_validate_entry_state(state, name)
      finally:
        sock.close()

  def test_registered_root_path_swap_is_rejected(self):
    with publication_sandbox() as (parent, roots):
      analyzer.ensure_publication_directory(
          'analysis_attempt', 'START.json', 'start'
      )
      roots['analysis_attempt'].rename(parent / 'old-attempt')
      roots['analysis_attempt'].mkdir(mode=0o700)
      with self.assertRaises(analyzer.AnalysisError):
        analyzer._v33451_tree_binding(roots['analysis_attempt'])

  def test_publish_reauthenticates_fixed_root_path(self):
    with publication_sandbox() as (parent, roots):
      analyzer.ensure_publication_directory(
          'analysis_attempt', 'START.json', 'start'
      )
      roots['analysis_attempt'].rename(parent / 'detached-attempt')
      roots['analysis_attempt'].mkdir(mode=0o700)
      with self.assertRaises(analyzer.PublicationError) as caught:
        analyzer.publish_bytes(
            'analysis_attempt', 'START.json', b'{}\n', 'start'
        )
      self.assertEqual(
          caught.exception.publication_failure['failure_stage'],
          'parent_validation',
      )

  def test_post_publish_revalidation_binds_one_final_inode(self):
    with publication_sandbox() as (_parent, roots):
      analyzer.ensure_publication_directory(
          'analysis_output', 'RESULT.md', 'result'
      )
      original = analyzer._publication_entry_at
      swapped = False
      def replace_after_entry(directory_fd, relative):
        nonlocal swapped
        state = original(directory_fd, relative)
        if relative == 'RESULT.md' and state['state'] == 'present' and not swapped:
          swapped = True
          target = roots['analysis_output'] / relative
          target.unlink()
          target.write_bytes(b'payload')
          target.chmod(0o400)
        return state
      with (
          mock.patch.object(
              analyzer, '_publication_entry_at', side_effect=replace_after_entry
          ),
          self.assertRaises(analyzer.PublicationError) as caught,
      ):
        analyzer.publish_bytes(
            'analysis_output', 'RESULT.md', b'payload', 'result'
        )
      self.assertEqual(
          caught.exception.publication_failure['failure_stage'],
          'post_publish_revalidation',
      )

  def test_publication_entry_rejects_path_swap_during_fd_read(self):
    with publication_sandbox() as (_parent, roots):
      analyzer.ensure_publication_directory(
          'analysis_output', 'RESULT.md', 'result'
      )
      target = roots['analysis_output'] / 'bound'
      target.write_bytes(b'same')
      target.chmod(0o400)
      original_read = analyzer.os.read
      swapped = False
      def swap_then_read(descriptor, size):
        nonlocal swapped
        if not swapped:
          swapped = True
          target.rename(roots['analysis_output'] / 'old-bound')
          target.write_bytes(b'same')
          target.chmod(0o400)
        return original_read(descriptor, size)
      root_fd = analyzer._PUBLICATION_DIRECTORIES['analysis_output'][0]
      with (
          mock.patch.object(analyzer.os, 'read', side_effect=swap_then_read),
          self.assertRaises(analyzer.AnalysisError),
      ):
        analyzer._publication_entry_at(root_fd, 'bound')

  def test_publication_entry_rejects_nonregular_path_swap(self):
    with publication_sandbox() as (_parent, roots):
      analyzer.ensure_publication_directory(
          'analysis_output', 'RESULT.md', 'result'
      )
      root = roots['analysis_output']
      os.mkfifo(root / 'special', 0o600)
      original_stat = analyzer.os.stat
      observations = 0
      def swap_on_final(relative, *args, **kwargs):
        nonlocal observations
        if relative == 'special':
          observations += 1
          if observations == 2:
            (root / 'special').rename(root / 'old-special')
            os.mkfifo(root / 'special', 0o600)
        return original_stat(relative, *args, **kwargs)
      root_fd = analyzer._PUBLICATION_DIRECTORIES['analysis_output'][0]
      with (
          mock.patch.object(analyzer.os, 'stat', side_effect=swap_on_final),
          self.assertRaises(analyzer.AnalysisError),
      ):
        analyzer._publication_entry_at(root_fd, 'special')

  def test_lost_invocation_created_entries_are_terminal_unbindable(self):
    for lost in ('temp', 'final', 'root'):
      with self.subTest(lost=lost), publication_sandbox() as (_parent, roots):
        if lost == 'root':
          def inject(stage):
            if stage == 'root_parent_fsync':
              roots['analysis_output'].rmdir()
              raise OSError('lost created root')
          with (
              mock.patch.object(
                  analyzer, '_injected_publication_failure', side_effect=inject
              ),
              self.assertRaises(analyzer.PublicationError),
          ):
            analyzer.ensure_publication_directory(
                'analysis_output', 'RESULT.md', 'result'
            )
        else:
          analyzer.ensure_publication_directory(
              'analysis_output', 'RESULT.md', 'result'
          )
          def inject(stage):
            if lost == 'temp' and stage == 'temp_write':
              next(roots['analysis_output'].glob('.v33451.tmp.*')).unlink()
              raise OSError('lost created temp')
            if lost == 'final' and stage == 'parent_fsync':
              (roots['analysis_output'] / 'RESULT.md').unlink()
              raise OSError('lost renamed final')
          with (
              mock.patch.object(
                  analyzer, '_injected_publication_failure', side_effect=inject
              ),
              self.assertRaises(analyzer.PublicationError),
          ):
            analyzer.publish_bytes(
                'analysis_output', 'RESULT.md', b'payload', 'result'
            )
        self.assertIn('analysis_output', analyzer._PUBLICATION_UNBINDABLE_ROOTS)
        with self.assertRaises(analyzer.AnalysisError):
          analyzer.publication_audit('analysis_output')
        with self.assertRaises(analyzer.AnalysisError):
          analyzer._v33451_output_state('analysis_output')

  def test_created_root_parent_path_swap_is_terminal_unbindable(self):
    for failure_stage in ('root_parent_fsync', 'root_revalidation'):
      with self.subTest(stage=failure_stage), publication_sandbox() as (base, roots):
        fixed_parent = base / 'fixed-parent'
        fixed_parent.mkdir(mode=0o700)
        fixed_root = fixed_parent / 'output'
        replacement_roots = {
            'analysis_attempt': roots['analysis_attempt'],
            'analysis_output': fixed_root,
        }
        moved_parent = base / 'moved-parent'
        def swap_parent(stage):
          if stage == failure_stage:
            fixed_parent.rename(moved_parent)
            fixed_parent.mkdir(mode=0o700)
            raise OSError('injected fixed-parent pathname swap')
        with (
            mock.patch.object(
                analyzer, '_PUBLICATION_ROOTS', replacement_roots
            ),
            mock.patch.object(analyzer, '_ANALYSIS_DIR', fixed_root),
            mock.patch.object(
                analyzer, '_injected_publication_failure',
                side_effect=swap_parent,
            ),
            self.assertRaises(analyzer.PublicationError),
        ):
          analyzer.ensure_publication_directory(
              'analysis_output', 'RESULT.md', 'result'
          )
        self.assertTrue((moved_parent / 'output').is_dir())
        self.assertFalse(fixed_root.exists())
        self.assertIn('analysis_output', analyzer._PUBLICATION_UNBINDABLE_ROOTS)
        with mock.patch.object(analyzer, '_PUBLICATION_ROOTS', replacement_roots):
          with self.assertRaises(analyzer.AnalysisError):
            analyzer._v33451_output_state('analysis_output')

  def test_created_root_parent_reauth_error_is_terminal_unbindable(self):
    with publication_sandbox() as (base, roots):
      fixed_parent = base / 'fixed-parent'
      fixed_parent.mkdir(mode=0o700)
      fixed_root = fixed_parent / 'output'
      replacement_roots = {
          'analysis_attempt': roots['analysis_attempt'],
          'analysis_output': fixed_root,
      }
      def deny_parent(stage):
        if stage == 'root_parent_fsync':
          fixed_parent.chmod(0o000)
          raise OSError('injected fixed-parent permission loss')
      try:
        with (
            mock.patch.object(
                analyzer, '_PUBLICATION_ROOTS', replacement_roots
            ),
            mock.patch.object(analyzer, '_ANALYSIS_DIR', fixed_root),
            mock.patch.object(
                analyzer, '_injected_publication_failure',
                side_effect=deny_parent,
            ),
            self.assertRaises(analyzer.PublicationError),
        ):
          analyzer.ensure_publication_directory(
              'analysis_output', 'RESULT.md', 'result'
          )
      finally:
        fixed_parent.chmod(0o700)
      self.assertTrue(fixed_root.is_dir())
      self.assertIn('analysis_output', analyzer._PUBLICATION_UNBINDABLE_ROOTS)
      with mock.patch.object(analyzer, '_PUBLICATION_ROOTS', replacement_roots):
        with self.assertRaises(analyzer.AnalysisError):
          analyzer._v33451_output_state('analysis_output')

  def test_root_preexistence_types_are_archivable_without_child_probe(self):
    for kind in ('directory', 'regular', 'symlink', 'fifo'):
      with self.subTest(kind=kind), publication_sandbox() as (parent, roots):
        root = roots['analysis_output']
        if kind == 'directory':
          root.mkdir(mode=0o700)
        elif kind == 'regular':
          root.write_bytes(b'preexisting')
        elif kind == 'symlink':
          target = parent / 'target'
          target.mkdir()
          root.symlink_to(target)
        else:
          os.mkfifo(root, 0o600)
        with self.assertRaises(analyzer.PublicationError) as caught:
          analyzer.ensure_publication_directory(
              'analysis_output', 'RESULT.md', 'result'
          )
        failure = analyzer._v33451_validate_publication_failure(
            caught.exception.publication_failure, 'root collision'
        )
        self.assertEqual(failure['failure_stage'], 'root_final_preexistence')
        self.assertEqual(failure['temp_state'], analyzer._absent_publication_entry())
        self.assertEqual(failure['final_state'], analyzer._absent_publication_entry())
        state = analyzer._v33451_output_state('analysis_output', failure)
        self.assertEqual(state['root_lstat']['entry_type'], kind)
        self.assertEqual(state['state'], 'publication_failure_prefix')

  def test_nonempty_preexisting_root_children_are_bound(self):
    with publication_sandbox() as (_parent, roots):
      root = roots['analysis_output']
      root.mkdir(mode=0o700)
      (root / 'regular').write_bytes(b'preserved')
      os.mkfifo(root / 'fifo', 0o600)
      (root / 'empty').mkdir(mode=0o700)
      with self.assertRaises(analyzer.PublicationError) as caught:
        analyzer.ensure_publication_directory(
            'analysis_output', 'RESULT.md', 'result'
        )
      state = analyzer._v33451_output_state(
          'analysis_output', caught.exception.publication_failure
      )
      self.assertEqual(
          set(state['preexisting_entry_states']),
          {'regular', 'fifo', 'empty'},
      )
      self.assertNotEqual(state['file_tree_sha256'], analyzer.EMPTY_SHA256)
      self.assertIn('empty', state['directory_paths'])
      audit = analyzer.publication_audit(
          'analysis_output', caught.exception.publication_failure
      )
      self.assertEqual(audit['preexisting_entry_count'], 3)

  def test_successful_start_result_analysis_complete_lifecycle(self):
    with publication_sandbox():
      analyzer.ensure_publication_directory(
          'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json', 'start'
      )
      analyzer.publish_bytes(
          'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json', b'{}\n', 'start'
      )
      started_sha = analyzer._v33451_publication_file_binding(
          'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json'
      )['sha256']
      analyzer.ensure_publication_directory(
          'analysis_output', 'RESULT.md', 'result'
      )
      analyzer.publish_bytes('analysis_output', 'RESULT.md', b'result\n', 'result')
      analyzer.publish_bytes(
          'analysis_output', 'ANALYSIS.json', b'{}\n', 'analysis'
      )
      complete = analyzer._v33451_complete_record(started_sha)
      self.assertEqual(set(complete), analyzer._V33451_COMPLETE_KEYS)
      self.assertEqual(
          complete['publication_audit']['successful_final_count_before_terminal'], 3
      )
      analyzer._v33451_write_json(
          'analysis_attempt', 'ANALYSIS_COMPLETE.json', complete, 'complete'
      )
      attempt = analyzer._v33451_tree_binding(
          analyzer._PUBLICATION_ROOTS['analysis_attempt']
      )
      self.assertEqual(
          set(attempt['file_bindings']),
          {'ANALYSIS_ATTEMPT_STARTED.json', 'ANALYSIS_COMPLETE.json'},
      )

  def test_output_publication_failure_archives_once(self):
    with publication_sandbox():
      analyzer.ensure_publication_directory(
          'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json', 'start'
      )
      analyzer.publish_bytes(
          'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json', b'{}\n', 'start'
      )
      started_sha = analyzer._v33451_publication_file_binding(
          'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json'
      )['sha256']
      analyzer.ensure_publication_directory(
          'analysis_output', 'RESULT.md', 'result'
      )
      analyzer._PUBLICATION_TEST_FAIL_STAGE = 'temp_write'
      with self.assertRaises(analyzer.PublicationError) as caught:
        analyzer.publish_bytes('analysis_output', 'RESULT.md', b'result', 'result')
      analyzer._PUBLICATION_TEST_FAIL_STAGE = None
      failure = analyzer._v33451_failure_record(
          caught.exception, started_sha, phase='result_publication'
      )
      self.assertEqual(set(failure), analyzer._V33451_FAILURE_KEYS)
      analyzer._v33451_write_json(
          'analysis_attempt', 'ANALYSIS_FAILURE.json', failure, 'analysis_failure'
      )
      with self.assertRaises(analyzer.AnalysisError):
        analyzer.publish_bytes(
            'analysis_output', 'RESULT.md', b'retry', 'result'
        )

  def test_complete_publication_failure_allows_one_failure_terminal(self):
    with publication_sandbox():
      analyzer.ensure_publication_directory(
          'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json', 'start'
      )
      analyzer.publish_bytes(
          'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json', b'{}\n', 'start'
      )
      started_sha = analyzer._v33451_publication_file_binding(
          'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json'
      )['sha256']
      analyzer._PUBLICATION_TEST_FAIL_STAGE = 'parent_fsync'
      with self.assertRaises(analyzer.PublicationError) as caught:
        analyzer.publish_bytes(
            'analysis_attempt', 'ANALYSIS_COMPLETE.json', b'{}\n', 'complete'
        )
      analyzer._PUBLICATION_TEST_FAIL_STAGE = None
      failure = analyzer._v33451_failure_record(
          caught.exception, started_sha, phase='complete_publication'
      )
      analyzer._v33451_write_json(
          'analysis_attempt', 'ANALYSIS_FAILURE.json', failure, 'analysis_failure'
      )
      with self.assertRaises(analyzer.AnalysisError):
        analyzer.publish_bytes(
            'analysis_attempt', 'ANALYSIS_FAILURE_2.json', b'{}\n',
            'analysis_failure',
        )

  def test_all_failure_phases_serialize_exactly(self):
    for phase in analyzer._V33451_FAILURE_PHASES:
      with self.subTest(phase=phase), publication_sandbox():
        analyzer.ensure_publication_directory(
            'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json', 'start'
        )
        analyzer.publish_bytes(
            'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json', b'{}\n', 'start'
        )
        started_sha = analyzer._v33451_publication_file_binding(
            'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json'
        )['sha256']
        value = analyzer._v33451_failure_record(
            RuntimeError('fixture'), started_sha, phase=phase
        )
        self.assertEqual(value['failure_phase'], phase)
        self.assertFalse(value['raw_access_reached'])

  def test_active_start_and_output_prefixes_are_phase_exact(self):
    with publication_sandbox():
      authorization = {'authorization': True}
      start = {key: None for key in analyzer._V33451_START_KEYS}
      start.update({
          'attempt_id': analyzer.ANALYSIS_ATTEMPT_ID,
          'acknowledgement': analyzer.ANALYSIS_ACKNOWLEDGEMENT,
          'external_freeze_authorization': authorization,
      })
      analyzer.ensure_publication_directory(
          'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json', 'start'
      )
      analyzer._v33451_write_json(
          'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json', start, 'start'
      )
      started_sha = analyzer._v33451_publication_file_binding(
          'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json'
      )['sha256']
      analyzer._v33451_validate_active_start(
          started_sha, authorization, output_prefix=None
      )
      analyzer.ensure_publication_directory(
          'analysis_output', 'RESULT.md', 'result'
      )
      analyzer.publish_bytes('analysis_output', 'RESULT.md', b'result', 'result')
      analyzer._v33451_validate_active_start(
          started_sha, authorization, output_prefix={'RESULT.md'}
      )
      analyzer.publish_bytes(
          'analysis_output', 'ANALYSIS.json', b'{}\n', 'analysis'
      )
      analyzer._v33451_validate_active_start(
          started_sha, authorization,
          output_prefix={'RESULT.md', 'ANALYSIS.json'},
      )
      with self.assertRaises(analyzer.AnalysisError):
        analyzer._v33451_validate_active_start(
            started_sha, authorization, output_prefix={'RESULT.md'}
        )

  def test_every_old_destination_is_rechecked(self):
    with tempfile.TemporaryDirectory() as directory:
      paths = tuple(Path(directory) / f'old-{index}' for index in range(4))
      with mock.patch.object(analyzer, '_v33451_old_destinations', return_value=paths):
        analyzer._v33451_require_old_absent()
        for path in paths:
          with self.subTest(path=path):
            path.write_bytes(b'drift')
            with self.assertRaises(analyzer.AnalysisError):
              analyzer._v33451_require_old_absent()
            path.unlink()


class StructuralBoundaryTest(unittest.TestCase):

  def test_safe_json_reader_rejects_swap_without_opening_confirmation(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      allowed = root / 'allowed.json'
      confirmation = root / 'confirmation-secret.json'
      allowed.write_text('{"allowed":true}\n')
      confirmation.write_text('{"score":99}\n')
      original_read = analyzer.os.read
      original_open = analyzer.os.open
      reads = 0
      opened = []
      def observed_open(path, *args, **kwargs):
        opened.append(os.fspath(path))
        return original_open(path, *args, **kwargs)
      def swap_then_read(descriptor, size):
        nonlocal reads
        if reads == 0:
          reads += 1
          allowed.rename(root / 'old-allowed.json')
          allowed.symlink_to(confirmation)
        return original_read(descriptor, size)
      with (
          mock.patch.object(analyzer.os, 'open', side_effect=observed_open),
          mock.patch.object(analyzer.os, 'read', side_effect=swap_then_read),
          self.assertRaises(analyzer.AnalysisError),
      ):
        analyzer._read_json(allowed, 'allowed structural JSON')
      self.assertEqual(opened, [str(allowed)])
      self.assertNotIn(str(confirmation), opened)

  def test_raw_membership_tamper_fails_before_hash(self):
    for namespace in ('raw', 'dispatch_started', 'dispatch_completed', 'confirmation'):
      with self.subTest(namespace=namespace), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'allowed.json').write_text('{}')
        (root / namespace).mkdir()
        (root / namespace / 'score.json').write_text('{"score":1}')
        expected = {
            'directory_paths': ['.'],
            'file_bindings': {'allowed.json': {}},
        }
        with mock.patch.object(
            analyzer, '_sha256_no_follow',
            side_effect=AssertionError('scientific byte read'),
        ) as read_sentinel:
          with self.assertRaises(analyzer.AnalysisError):
            analyzer._v33451_assert_tree_membership(root, expected, 'run')
        read_sentinel.assert_not_called()

  def test_malicious_frozen_raw_allowlist_fails_before_tree_hash(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      (root / 'raw').mkdir()
      (root / 'raw/score.json').write_text('{}')
      malicious = {
          name: {} for name in (
              'run_root_binding', 'compiler_tree_binding',
              'preflight_tree_binding', 'external_cache_tree_binding',
              'model_cache_tree_binding', 'run_terminal_binding',
              'raw_manifest_binding', 'old_analyzer_bundle',
          )
      }
      malicious['run_root_binding'] = {
          'file_count': 1, 'directory_paths': ['.', 'raw'],
          'file_bindings': {'raw/score.json': {'sha256': '0' * 64}},
          'file_tree_sha256': '0' * 64,
          'directory_file_tree_sha256': '0' * 64,
      }
      with (
          mock.patch.object(analyzer, '_RUN_DIR', root),
          mock.patch.object(
              analyzer, '_v33451_tree_binding',
              side_effect=AssertionError('tree byte read'),
          ) as sentinel,
          self.assertRaises(analyzer.AnalysisError),
      ):
        analyzer._v33451_validate_immutable_contract(malicious)
      sentinel.assert_not_called()

  def test_raw_inserted_between_membership_and_hash_is_never_opened(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      (root / 'allowed.json').write_text('{}')
      original_hash = analyzer._sha256_no_follow
      inserted = False
      opened = []
      def insert_then_hash(path, expected=None):
        nonlocal inserted
        relative = path.relative_to(root).as_posix()
        opened.append(relative)
        if not inserted:
          inserted = True
          (root / 'raw').mkdir()
          (root / 'raw/score.json').write_text('{"score":1}')
        if relative.startswith('raw/'):
          raise AssertionError('raw byte was opened')
        return original_hash(path, expected)
      with (
          mock.patch.object(
              analyzer, '_sha256_no_follow', side_effect=insert_then_hash
          ),
          self.assertRaises(analyzer.AnalysisError),
      ):
        analyzer._v33451_tree_binding(
            root, expected_files={'allowed.json'},
            expected_directories={'.'}, label='run',
        )
      self.assertEqual(opened, ['allowed.json'])

  def test_model_start_and_terminal_helpers_accept_live_archive(self):
    prior333 = analyzer._v33451_prior333_binding()
    prior331 = analyzer._v33451_prior331_binding()
    freeze = analyzer._read_json(analyzer._FREEZE_PATH, 'model freeze')
    freeze_sha = analyzer._v33451_file_binding(analyzer._FREEZE_PATH)['sha256']
    start = analyzer._validate_start_v3345(
        analyzer._RUN_DIR, freeze, freeze_sha,
        prior333=prior333, prior331=prior331,
    )
    completion = analyzer._read_json(
        analyzer._RUN_DIR / 'RUN_COMPLETE.json', 'RUN_COMPLETE'
    )
    completion, _ = analyzer._validate_terminal_common(
        completion, freeze_sha=freeze_sha, start=start
    )
    compiler = analyzer._validate_compiler_v3345(
        analyzer._RUN_DIR, completion, freeze, start
    )
    self.assertEqual(completion['model_apply_success_count'], 0)
    self.assertEqual(completion['valid_record_count'], 0)
    self.assertEqual(compiler['state'], 'diagnostic_provenance_failed')
    self.assertEqual(
        compiler['audit']['trigger_operation']['trigger_type'],
        'DiagnosticPersistenceFailure',
    )

  def test_internal_structural_archive_reaches_exact_no_science_result(self):
    with publication_sandbox():
      analyzer.ensure_publication_directory(
          'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json', 'start'
      )
      analyzer.publish_bytes(
          'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json', b'{}\n', 'start'
      )
      started_sha = analyzer._v33451_publication_file_binding(
          'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json'
      )['sha256']
      immutable = {
          'run_root_binding': analyzer._v33451_tree_binding(analyzer._RUN_DIR),
          'compiler_tree_binding': analyzer._v33451_tree_binding(
              analyzer._RUN_DIR / 'compiler'
          ),
          'preflight_tree_binding': analyzer._v33451_tree_binding(
              analyzer._PREFLIGHT_DIR
          ),
          'external_cache_tree_binding': analyzer._v33451_tree_binding(
              analyzer._PREFLIGHT_CACHE_DIR
          ),
          'model_cache_tree_binding': analyzer._v33451_tree_binding(
              analyzer._MODEL_CACHE_DIR
          ),
      }
      precheck = {
          'immutable': immutable,
          'consumed': analyzer._v33451_expected_consumed_failure(),
          'prior_cache': analyzer._validate_prior_cache_directory_aware(),
      }
      original_read = analyzer._read_json
      def raw_sentinel(path, label):
        relative = Path(path).relative_to(analyzer._RUN_DIR).as_posix() if (
            Path(path).is_relative_to(analyzer._RUN_DIR)
        ) else ''
        if relative.startswith(('raw/', 'dispatch_started/', 'dispatch_completed/')):
          raise AssertionError('scientific raw read')
        return original_read(path, label)
      with (
          mock.patch.object(analyzer, '_v33451_validate_active_start'),
          mock.patch.object(
              analyzer, '_v33451_validate_analysis_freeze',
              return_value=precheck,
          ),
          mock.patch.object(analyzer, '_read_json', side_effect=raw_sentinel),
      ):
        result = analyzer._v33451_structural_analyze(
            token=analyzer._V33451_ACTIVE_TOKEN,
            started_sha256=started_sha, authorization={},
        )
      self.assertEqual(set(result), analyzer._V33451_ANALYSIS_KEYS)
      self.assertEqual(
          result['decision'],
          'controlled_stop_diagnostic_provenance_failure',
      )
      for name in (
          'scientific_summary_computed', 'donor_normalization_computed',
          'shapley_or_nomination_computed',
          'interaction_or_resolution_computed', 'nomination_performed',
          'combined_analysis_permitted',
      ):
        self.assertFalse(result[name])

  def test_structural_result_has_no_science_fields_true(self):
    # The literal result contract itself is frozen independent of publication.
    for name in (
        'scientific_summary_computed', 'donor_normalization_computed',
        'shapley_or_nomination_computed',
        'interaction_or_resolution_computed', 'nomination_performed',
        'combined_analysis_permitted',
    ):
      self.assertIn(name, analyzer._V33451_ANALYSIS_KEYS)
    self.assertNotIn('scores', analyzer._V33451_ANALYSIS_KEYS)


class GeneratorTest(unittest.TestCase):

  def test_contract_counts_and_target(self):
    self.assertEqual(len(generator.START_KEYS), 22)
    self.assertEqual(len(generator.ANALYSIS_KEYS), 26)
    self.assertEqual(len(generator.COMPLETE_KEYS), 11)
    self.assertEqual(len(generator.FAILURE_KEYS), 13)
    self.assertEqual(len(generator.PUBLICATION_SUCCESS_KEYS), 19)
    self.assertEqual(len(generator.PUBLICATION_FAILURE_KEYS), 19)
    self.assertEqual(len(generator.PUBLICATION_AUDIT_KEYS), 15)
    self.assertEqual(generator.FREEZE, FREEZE_PATH)

  def test_source_contract_is_exact_132_plus_docs_plus_four(self):
    source_head = '2' * 40
    def git_blob(_commit, relative):
      return (generator.REPO / relative).read_bytes()
    def git_output(*args, **_kwargs):
      if args[0] == 'ls-tree':
        path = generator.REPO / args[-1]
        mode = '100755' if stat.S_IMODE(path.stat().st_mode) & 0o111 else '100644'
        return f'{mode} blob deadbeef\t{args[-1]}\n'
      if args == ('rev-parse', 'HEAD'):
        return source_head + '\n'
      raise AssertionError(args)
    with (
        mock.patch.object(generator, 'git_blob', side_effect=git_blob),
        mock.patch.object(generator, 'git_output', side_effect=git_output),
        mock.patch.object(generator.subprocess, 'check_call'),
    ):
      value = generator.source_contract(source_head)
    self.assertEqual(value['row_count'], 137)
    self.assertEqual(
        value['authority_partitions']['inherited_132']['row_count'], 132
    )
    self.assertEqual(value['authority_partitions']['amendment']['row_count'], 1)
    self.assertEqual(
        value['authority_partitions']['new_implementation_4']['row_count'], 4
    )
    self.assertEqual(
        [row['path'] for row in value['rows']],
        sorted(row['path'] for row in value['rows']),
    )

  def test_source_contract_rejects_each_authority_family_drift(self):
    model_rows = json.loads(generator.MODEL_FREEZE.read_text())[
        'source_inventory_contract'
    ]['rows']
    paths = (
        model_rows[0]['path'],
        generator.AMENDMENT.relative_to(generator.REPO).as_posix(),
        *(path.relative_to(generator.REPO).as_posix()
          for path in generator.NEW_FILES),
    )
    for drift_path in paths:
      with self.subTest(drift_path=drift_path):
        def git_blob(_commit, relative):
          value = (generator.REPO / relative).read_bytes()
          return value + b'drift' if relative == drift_path else value
        def git_output(*args, **_kwargs):
          if args[0] == 'ls-tree':
            path = generator.REPO / args[-1]
            mode = (
                '100755' if stat.S_IMODE(path.stat().st_mode) & 0o111
                else '100644'
            )
            return f'{mode} blob deadbeef\t{args[-1]}\n'
          raise AssertionError(args)
        with (
            mock.patch.object(generator, 'git_blob', side_effect=git_blob),
            mock.patch.object(generator, 'git_output', side_effect=git_output),
            self.assertRaises(RuntimeError),
        ):
          generator.source_contract('2' * 40)

  def test_immutable_literal_validator_accepts_live_facts(self):
    old_bundle = {
        'git_head': generator.MODEL_HEAD,
        'analyzer': generator.file_binding(
            HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_4_5.py', absolute=True
        ),
        'test': generator.file_binding(
            HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_4_5_test.py', absolute=True
        ),
        'shell': generator.file_binding(
            HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_4_5.sh', absolute=True
        ),
        'freeze': generator.file_binding(generator.MODEL_FREEZE, absolute=True),
    }
    value = {
        'run_root_binding': generator.tree_binding(generator.RUN),
        'compiler_tree_binding': generator.tree_binding(generator.RUN / 'compiler'),
        'preflight_tree_binding': generator.tree_binding(generator.PREFLIGHT),
        'external_cache_tree_binding': generator.tree_binding(generator.EXTERNAL_CACHE),
        'model_cache_tree_binding': generator.tree_binding(generator.MODEL_CACHE),
        'run_terminal_binding': generator.file_binding(
            generator.RUN / 'RUN_COMPLETE.json', absolute=True
        ),
        'raw_manifest_binding': generator.file_binding(
            generator.RUN / 'RAW_MANIFEST.json', absolute=True
        ),
        'old_analyzer_bundle': old_bundle,
    }
    generator.validate_immutable_facts(value)
    valid = json.loads(json.dumps(value))
    value['run_root_binding']['file_count'] = 13
    with self.assertRaises(RuntimeError):
      generator.validate_immutable_facts(value)
    for name in ('analyzer', 'test', 'shell', 'freeze'):
      for key in ('path', 'sha256', 'size_bytes'):
        with self.subTest(old_bundle=name, key=key):
          tampered = json.loads(json.dumps(valid))
          current = tampered['old_analyzer_bundle'][name][key]
          tampered['old_analyzer_bundle'][name][key] = (
              current + 1 if isinstance(current, int) else current + 'x'
          )
          with self.assertRaises(RuntimeError):
            generator.validate_immutable_facts(tampered)
    tampered = json.loads(json.dumps(valid))
    tampered['old_analyzer_bundle']['git_head'] = '0' * 40
    with self.assertRaises(RuntimeError):
      generator.validate_immutable_facts(tampered)

  def test_generator_membership_gate_is_byte_blind(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      (root / 'allowed').write_bytes(b'ok')
      (root / 'raw').mkdir()
      (root / 'raw' / 'secret').write_bytes(b'score')
      with mock.patch.object(
          generator, 'sha256_no_follow', side_effect=AssertionError('read')
      ) as sentinel:
        with self.assertRaises(RuntimeError):
          generator.assert_tree_membership(
              root, {'allowed'}, {'.'}, 'run'
          )
      sentinel.assert_not_called()

  def test_generator_expected_tree_rejects_inserted_raw_without_open(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      (root / 'allowed').write_bytes(b'ok')
      original_hash = generator.sha256_no_follow
      opened = []
      def insert_then_hash(path, expected):
        relative = path.relative_to(root).as_posix()
        opened.append(relative)
        if len(opened) == 1:
          (root / 'raw').mkdir()
          (root / 'raw/secret').write_bytes(b'score')
        if relative.startswith('raw/'):
          raise AssertionError('raw byte was opened')
        return original_hash(path, expected)
      with (
          mock.patch.object(
              generator, 'sha256_no_follow', side_effect=insert_then_hash
          ),
          self.assertRaises(RuntimeError),
      ):
        generator.tree_binding(
            root, expected_files={'allowed'}, expected_directories={'.'},
            label='run',
        )
      self.assertEqual(opened, ['allowed'])

  def test_generator_main_uses_fchmod_and_full_readback(self):
    source = GENERATOR_PATH.read_text()
    self.assertIn('os.fchmod(descriptor, 0o644)', source)
    self.assertIn('bytes(observed) != payload', source)
    self.assertIn('os.fsync(parent_fd)', source)
    self.assertIn('os.O_EXCL', source)

  def test_build_freeze_is_exact20_and_deterministic_under_fixed_inputs(self):
    source = {
        'row_count': 137, 'rows': [],
        'authority_partitions': {}, 'source_authority_head': '2' * 40,
        'source_authority_tree_exact': True,
        'all_rows_authority_exact': True,
        'all_rows_live_at_generation_exact': True, 'tree_sha256': '3' * 64,
    }
    binding = {'path': '/fixed', 'sha256': '4' * 64, 'size_bytes': 1}
    tree = {
        'root': '/fixed', 'file_count': 0, 'directory_count': 1,
        'file_bindings': {}, 'file_tree_sha256': generator.EMPTY_SHA256,
        'directory_paths': ['.'], 'directory_tree_sha256': '5' * 64,
        'directory_file_tree_sha256': '6' * 64,
    }
    def git_output(*args, **_kwargs):
      if args[:2] == ('rev-parse', 'HEAD'):
        return '2' * 40 + '\n'
      if args[:3] == ('rev-list', '--parents', '-n'):
        return '2' * 40 + ' ' + generator.DOCS_HEAD + '\n'
      if args[:2] == ('diff', '--name-status'):
        return '\n'.join(
            f'A\t{path.relative_to(generator.REPO).as_posix()}'
            for path in sorted(generator.NEW_FILES)
        ) + '\n'
      raise AssertionError(args)
    with (
        mock.patch.object(generator.subprocess, 'check_call'),
        mock.patch.object(generator, 'git_output', side_effect=git_output),
        mock.patch.object(generator, 'source_contract', return_value=source),
        mock.patch.object(generator, 'prior_cache_contract', return_value={'cache': True}),
        mock.patch.object(generator, 'consumed_failure', return_value={'failure': True}),
        mock.patch.object(generator, 'file_binding', return_value=binding),
        mock.patch.object(generator, 'tree_binding', return_value=tree),
        mock.patch.object(generator, 'assert_tree_membership'),
        mock.patch.object(generator, 'validate_immutable_facts'),
        mock.patch.object(Path, 'exists', return_value=False),
        mock.patch.object(Path, 'is_symlink', return_value=False),
    ):
      first = generator.build_freeze()
      second = generator.build_freeze()
    self.assertEqual(first, second)
    self.assertEqual(len(first), 20)
    self.assertEqual(first['source_inventory_contract']['row_count'], 137)
    self.assertEqual(
        first['publication_contract'], analyzer._v33451_publication_contract()
    )
    self.assertEqual(
        first['record_contracts'], analyzer._v33451_record_contracts()
    )
    self.assertEqual(first['claim_boundary'], analyzer._V33451_CLAIM_BOUNDARY)

  def test_build_freeze_rejects_non_child_or_nonexact_delta(self):
    source_head = '2' * 40
    cases = ('wrong_parent', 'wrong_delta')
    for case in cases:
      with self.subTest(case=case):
        def git_output(*args, **_kwargs):
          if args[:2] == ('rev-parse', 'HEAD'):
            return source_head + '\n'
          if args[:3] == ('rev-list', '--parents', '-n'):
            parent = '3' * 40 if case == 'wrong_parent' else generator.DOCS_HEAD
            return source_head + ' ' + parent + '\n'
          if args[:2] == ('diff', '--name-status'):
            return '' if case == 'wrong_delta' else '\n'.join(
                f'A\t{path.relative_to(generator.REPO).as_posix()}'
                for path in sorted(generator.NEW_FILES)
            ) + '\n'
          raise AssertionError(args)
        with (
            mock.patch.object(generator.subprocess, 'check_call'),
            mock.patch.object(generator, 'git_output', side_effect=git_output),
            mock.patch.object(Path, 'exists', return_value=False),
            mock.patch.object(Path, 'is_symlink', return_value=False),
            self.assertRaises(RuntimeError),
        ):
          generator.build_freeze()

  def test_precheck_failure_creates_no_destination(self):
    with tempfile.TemporaryDirectory() as directory:
      absent_freeze = Path(directory) / 'missing-freeze.json'
      attempt = Path(directory) / 'attempt'
      output = Path(directory) / 'output'
      with (
          mock.patch.object(analyzer, '_ANALYSIS_FREEZE_PATH', absent_freeze),
          mock.patch.object(analyzer, '_ANALYSIS_ATTEMPT_DIR', attempt),
          mock.patch.object(analyzer, '_ANALYSIS_DIR', output),
          self.assertRaises((FileNotFoundError, analyzer.AnalysisError)),
      ):
        analyzer._v33451_validate_analysis_freeze(
            {
                'git_head': '0' * 40, 'freeze_path': str(absent_freeze),
                'freeze_sha256': '0' * 64, 'freeze_size_bytes': 0,
                'live_equals_git_show': True, 'tracked_clean': True,
                'authorization_source': 'external_post_commit_audit',
            },
            phase='pre_start',
        )
      self.assertFalse(attempt.exists())
      self.assertFalse(output.exists())


if __name__ == '__main__':
  unittest.main()
