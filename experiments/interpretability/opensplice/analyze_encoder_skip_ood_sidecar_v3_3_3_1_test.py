#!/usr/bin/env python3
"""CPU-only tests for the v3.3.3.1 structural archive."""

from __future__ import annotations

import copy
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_3_1.py'
SPEC = importlib.util.spec_from_file_location('v3331_analyzer_tested', MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


def _records(root: Path, names: list[str]) -> dict[str, tuple[int, str]]:
  return {
      name: ((root / name).stat().st_size, analyzer._sha256(root / name))
      for name in names
  }


class RepresentationTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    cls.start = analyzer._read_json(analyzer._RUN_DIR / 'ATTEMPT_STARTED.json', 'start')
    cls.compiler = analyzer._read_json(
        analyzer._RUN_DIR / 'compiler/eight_row/COMPILER_PROVENANCE.json',
        'compiler',
    )
    cls.old = analyzer._load_old_validator()

  def test_exact_representation_diagnosis(self) -> None:
    audit = analyzer._representation_audit(self.compiler, self.start, self.old)
    self.assertEqual(audit['signature_object_count'], 3)
    self.assertEqual(audit['leaves_tuple_count'], 3)
    self.assertEqual(audit['shape_tuple_count'], 29)
    self.assertEqual(audit['tuple_container_count'], 32)
    self.assertEqual(audit['canonical_size_bytes'], 2877)
    self.assertEqual(audit['canonical_sha256'], analyzer.PROGRAM_SIGNATURES_SHA256)
    self.assertFalse(audit['runtime_tuple_direct_equals_stored_list'])
    self.assertTrue(audit['runtime_tuple_canonical_equals_stored_list'])

  def test_runtime_tuple_locations_are_narrow(self) -> None:
    stored = self.compiler['program_signatures']
    runtime = analyzer._tupleize_runtime_signature(stored)
    self.assertTrue(all(isinstance(row['leaves'], tuple) for row in runtime.values()))
    self.assertTrue(all(
        isinstance(leaf['shape'], tuple)
        for row in runtime.values() for leaf in row['leaves']
    ))
    self.assertEqual(analyzer._canonical_bytes(runtime), analyzer._canonical_bytes(stored))

  def test_rejects_shape_dtype_treedef_and_leaf_order_changes(self) -> None:
    for mutation in ('shape', 'dtype', 'treedef', 'order'):
      compiler = copy.deepcopy(self.compiler)
      current = compiler['program_signatures']
      if mutation == 'shape':
        current['target']['leaves'][0]['shape'][0] += 1
      elif mutation == 'dtype':
        current['target']['leaves'][0]['dtype'] = 'float64'
      elif mutation == 'treedef':
        current['target']['treedef'] += 'changed'
      else:
        current['target']['leaves'].reverse()
      compiler['program_signatures_sha256'] = analyzer._canonical_sha(current)
      with self.subTest(mutation=mutation), self.assertRaises(analyzer.AnalysisError):
        analyzer._representation_audit(compiler, self.start, self.old)

  def test_rejects_current_prior_or_hash_change(self) -> None:
    compiler = copy.deepcopy(self.compiler)
    compiler['program_signatures_sha256'] = '0' * 64
    with self.assertRaisesRegex(analyzer.AnalysisError, 'canonical'):
      analyzer._representation_audit(compiler, self.start, self.old)
    start = copy.deepcopy(self.start)
    start['v3_3_2_run_binding']['eight_row_compiler']['program_signatures'][
        'target'
    ]['treedef'] += 'x'
    with self.assertRaisesRegex(analyzer.AnalysisError, 'stored signature'):
      analyzer._representation_audit(self.compiler, start, self.old)

  def test_rejects_each_source_gate_flag_change(self) -> None:
    flags = (
        'stablehlo_exact', 'pre_backend_hlo_exact', 'entry_abi_exact',
        'source_runtime_device_toolchain_checkpoint_reference_exact',
        'same_lowered_compiled_object', 'program_signatures_exact',
        'source_program_exact',
    )
    for flag in flags:
      compiler = copy.deepcopy(self.compiler)
      compiler['source_program_gate'][flag] = not compiler['source_program_gate'][flag]
      with self.subTest(flag=flag), self.assertRaisesRegex(
          analyzer.AnalysisError, 'gate failure pattern'
      ):
        analyzer._representation_audit(compiler, self.start, self.old)

  def test_rejects_source_input_audit_false(self) -> None:
    compiler = copy.deepcopy(self.compiler)
    compiler['source_program_gate']['source_input_audit'][
        'checkpoint_exact'
    ] = False
    with self.assertRaisesRegex(analyzer.AnalysisError, 'source/input'):
      analyzer._representation_audit(compiler, self.start, self.old)


class TreeAndProvenanceTest(unittest.TestCase):

  def _freeze_fixture(self, mutate=None, *, head_freeze_exact=True, diff=b''):
    temporary = tempfile.TemporaryDirectory(dir=analyzer._HERE)
    path = Path(temporary.name) / 'freeze.json'
    value = json.loads(analyzer._FREEZE_PATH.read_text())
    if mutate is not None:
      mutate(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

    def git_blob(commit, relative, *, bundle_root):
      del commit
      candidate = bundle_root / relative
      if candidate.resolve() == path.resolve() and not head_freeze_exact:
        return b'changed freeze'
      return candidate.read_bytes()

    return temporary, path, git_blob, diff

  def test_new_freeze_schema_inventory_and_head_gate(self) -> None:
    temporary, path, git_blob, diff = self._freeze_fixture()
    expected_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with temporary, mock.patch.object(analyzer, '_FREEZE_PATH', path), mock.patch.object(
        analyzer, '_git_blob', side_effect=git_blob
    ), mock.patch.object(analyzer.subprocess, 'run'), mock.patch.object(
        analyzer.subprocess, 'check_output', return_value=diff
    ):
      freeze, digest = analyzer._validate_new_freeze(analyzer._REPO_ROOT)
    self.assertEqual(freeze['analysis_version'], analyzer.ANALYSIS_VERSION)
    self.assertEqual(digest, expected_digest)

  def test_new_freeze_rejects_schema_inventory_head_and_dirty_tamper(self) -> None:
    cases = (
        ('schema', lambda row: row.update({'extra': True}), True, b''),
        ('inventory', lambda row: row['file_sha256'].__setitem__(
            str(analyzer._OLD_ANALYZER.relative_to(analyzer._REPO_ROOT)), '0' * 64
        ), True, b''),
        ('head', None, False, b''),
        ('dirty', None, True, b'changed'),
    )
    for label, mutate, head_exact, diff in cases:
      temporary, path, git_blob, diff = self._freeze_fixture(
          mutate, head_freeze_exact=head_exact, diff=diff
      )
      with self.subTest(label=label), temporary, mock.patch.object(
          analyzer, '_FREEZE_PATH', path
      ), mock.patch.object(
          analyzer, '_git_blob', side_effect=git_blob
      ), mock.patch.object(analyzer.subprocess, 'run'), mock.patch.object(
          analyzer.subprocess, 'check_output', return_value=diff
      ), self.assertRaises(analyzer.AnalysisError):
        analyzer._validate_new_freeze(analyzer._REPO_ROOT)

  def test_every_new_source_rejects_live_and_head_drift(self) -> None:
    inventory = analyzer._read_json(analyzer._FREEZE_PATH, 'new freeze')[
        'file_sha256'
    ]
    self.assertEqual(len(inventory), 7)
    for relative, digest in inventory.items():
      with self.subTest(relative=relative, phase='live'), mock.patch.object(
          analyzer, '_strict_regular'
      ), mock.patch.object(analyzer, '_sha256', return_value='0' * 64), self.assertRaisesRegex(
          analyzer.AnalysisError, 'source bytes'
      ):
        analyzer._validate_new_source_row(
            relative, digest, bundle_root=analyzer._REPO_ROOT
        )
      with self.subTest(relative=relative, phase='HEAD'), mock.patch.object(
          analyzer, '_strict_regular'
      ), mock.patch.object(analyzer, '_sha256', return_value=digest), mock.patch.object(
          analyzer.subprocess, 'run'
      ), mock.patch.object(
          analyzer, '_git_blob', return_value=b'HEAD drift'
      ), self.assertRaisesRegex(analyzer.AnalysisError, 'HEAD source'):
        analyzer._validate_new_source_row(
            relative, digest, bundle_root=analyzer._REPO_ROOT
        )

  def test_exact_live_run_preflight_and_cache_trees(self) -> None:
    audit = analyzer._validate_fixed_trees()
    self.assertEqual(audit['run']['file_count'], 11)
    self.assertEqual(audit['run']['tree_sha256'], analyzer._RUN_TREE_SHA256)
    self.assertEqual(audit['preflight']['file_count'], 5)
    self.assertEqual(audit['preflight']['tree_sha256'], analyzer._PREFLIGHT_TREE_SHA256)
    self.assertEqual(audit['model_cache']['file_count'], 1)
    self.assertEqual(audit['model_cache']['tree_sha256'], analyzer._MODEL_CACHE_TREE_SHA256)

  def test_cache_and_generic_tree_framing_are_explicitly_distinct(self) -> None:
    path = analyzer._MODEL_CACHE_DIR / 'xdg/matplotlib/fontlist-v3.11.0.json'
    generic = analyzer._tree_digest([path.resolve()], analyzer._MODEL_CACHE_DIR.resolve())
    cache = analyzer._cache_tree_digest(
        [path.resolve()],
        [
            analyzer._MODEL_CACHE_DIR.resolve(),
            (analyzer._MODEL_CACHE_DIR / 'triton').resolve(),
            (analyzer._MODEL_CACHE_DIR / 'xdg').resolve(),
            (analyzer._MODEL_CACHE_DIR / 'xdg/matplotlib').resolve(),
        ],
        analyzer._MODEL_CACHE_DIR.resolve(),
    )
    self.assertEqual(generic, 'd1d11bc6dc48b302cf675fb48727bd6ededec09142429eaa9e368f7631463717')
    self.assertEqual(cache, analyzer._MODEL_CACHE_TREE_SHA256)
    self.assertNotEqual(generic, cache)

  def test_strict_tree_rejects_missing_extra_symlink_and_directory(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      (root / 'a').write_bytes(b'a')
      records = _records(root, ['a'])
      tree = analyzer._tree_digest([root / 'a'], root)
      analyzer._strict_tree(root, records, {'.'}, tree, 'fixture')
      (root / 'extra').write_bytes(b'x')
      with self.assertRaises(analyzer.AnalysisError):
        analyzer._strict_tree(root, records, {'.'}, tree, 'fixture')
      (root / 'extra').unlink()
      (root / 'empty').mkdir()
      with self.assertRaises(analyzer.AnalysisError):
        analyzer._strict_tree(root, records, {'.'}, tree, 'fixture')
      (root / 'empty').rmdir()
      (root / 'link').symlink_to(root / 'a')
      with self.assertRaises(analyzer.AnalysisError):
        analyzer._strict_tree(root, records, {'.'}, tree, 'fixture')
      (root / 'link').unlink()
      (root / 'a').unlink()
      with self.assertRaises(analyzer.AnalysisError):
        analyzer._strict_tree(root, records, {'.'}, tree, 'fixture')

  def test_fixed_tree_rejects_bound_hash_or_size_tamper(self) -> None:
    files = copy.deepcopy(analyzer._RUN_FILES)
    size, digest = files['RAW_MANIFEST.json']
    files['RAW_MANIFEST.json'] = (size + 1, digest)
    with self.assertRaisesRegex(analyzer.AnalysisError, 'binding changed'):
      analyzer._strict_tree(
          analyzer._RUN_DIR, files, {'.', 'compiler', 'compiler/eight_row'},
          analyzer._RUN_TREE_SHA256, 'run fixture',
      )

  def test_every_run_compiler_and_preflight_binding_rejects_tamper(self) -> None:
    fixtures = (
        (
            analyzer._RUN_DIR, analyzer._RUN_FILES,
            {'.', 'compiler', 'compiler/eight_row'}, analyzer._RUN_TREE_SHA256,
            'run', False,
        ),
        (
            analyzer._PREFLIGHT_DIR, analyzer._PREFLIGHT_FILES,
            {'.'}, analyzer._PREFLIGHT_TREE_SHA256, 'preflight', False,
        ),
        (
            analyzer._MODEL_CACHE_DIR, analyzer._MODEL_CACHE_FILES,
            {'.', 'triton', 'xdg', 'xdg/matplotlib'},
            analyzer._MODEL_CACHE_TREE_SHA256, 'cache', True,
        ),
    )
    for root, records, directories, tree, label, cache_framing in fixtures:
      for relative, (size, digest) in records.items():
        for phase, changed_binding in (
            ('size', (size + 1, digest)), ('hash', (size, '0' * 64))
        ):
          changed = dict(records)
          changed[relative] = changed_binding
          with self.subTest(
              family=label, relative=relative, phase=phase
          ), mock.patch.object(
              analyzer, '_sha256', side_effect=lambda path, r=root, rows=records: (
                  rows[path.resolve().relative_to(r.resolve()).as_posix()][1]
              )
          ), self.assertRaisesRegex(analyzer.AnalysisError, 'binding changed'):
            analyzer._strict_tree(
                root, changed, directories, tree, label,
                cache_framing=cache_framing,
            )
    compiler_rows = {
        name for name in analyzer._RUN_FILES if name.startswith('compiler/')
    }
    self.assertEqual(len(compiler_rows), 4)

  def test_every_cache_directory_binding_is_required(self) -> None:
    directories = {'.', 'triton', 'xdg', 'xdg/matplotlib'}
    for missing in directories:
      with self.subTest(missing=missing), self.assertRaisesRegex(
          analyzer.AnalysisError, 'membership changed'
      ):
        analyzer._strict_tree(
            analyzer._MODEL_CACHE_DIR, analyzer._MODEL_CACHE_FILES,
            directories - {missing}, analyzer._MODEL_CACHE_TREE_SHA256,
            'cache', cache_framing=True,
        )

  def test_full_original_96_file_and_historical_bundle_gate(self) -> None:
    prevalidated = analyzer._validate_original_inventory_stdlib(
        analyzer._REPO_ROOT
    )
    old = analyzer._load_old_validator()
    freeze, digest, validated = analyzer._validate_original_bundle(
        analyzer._REPO_ROOT, old, prevalidated=prevalidated
    )
    self.assertEqual(digest, analyzer.ORIGINAL_FREEZE_SHA256)
    self.assertEqual(len(freeze['file_sha256']), 96)
    self.assertEqual(len(validated), 8)

  def test_every_original_source_rejects_live_and_historical_drift(self) -> None:
    freeze = analyzer._read_json(analyzer._OLD_FREEZE, 'old freeze')
    inventory = freeze['file_sha256']
    self.assertEqual(len(inventory), 96)
    self.assertTrue({Path(path).suffix for path in inventory}.issuperset({
        '.py', '.json', '.md', '.tsv', '.toml', '.proto', '.sh'
    }))
    for relative, digest in inventory.items():
      with self.subTest(relative=relative, phase='live'), mock.patch.object(
          analyzer, '_strict_regular'
      ), mock.patch.object(analyzer, '_sha256', return_value='0' * 64), self.assertRaisesRegex(
          analyzer.AnalysisError, 'live bytes'
      ):
        analyzer._validate_original_source_row(
            relative, digest, bundle_root=analyzer._REPO_ROOT
        )
      with self.subTest(relative=relative, phase='historical'), mock.patch.object(
          analyzer, '_strict_regular'
      ), mock.patch.object(analyzer, '_sha256', return_value=digest), mock.patch.object(
          analyzer, '_git_blob', return_value=b'historical drift'
      ), self.assertRaisesRegex(analyzer.AnalysisError, 'historical bytes'):
        analyzer._validate_original_source_row(
            relative, digest, bundle_root=analyzer._REPO_ROOT
        )

  def test_source_gate_precedes_run_read_and_helper_import_pre_start(self) -> None:
    events = []
    with mock.patch.object(analyzer, '_assert_cpu_only'), mock.patch.object(
        analyzer, '_assert_destinations_fresh'
    ), mock.patch.object(
        analyzer, '_validate_new_freeze', side_effect=lambda root: (
            events.append('new_sources') or ({}, 'f' * 64)
        )
    ), mock.patch.object(
        analyzer, '_validate_original_inventory_stdlib',
        side_effect=lambda root: events.append('original96') or ({}, 'e' * 64)
    ), mock.patch.object(
        analyzer, '_validate_fixed_trees',
        side_effect=lambda: events.append('run_read') or {}
    ), mock.patch.object(
        analyzer, '_load_old_validator',
        side_effect=lambda: events.append('old_import') or object()
    ), mock.patch.object(
        analyzer, '_validate_original_bundle', return_value=({}, 'e' * 64, ())
    ), mock.patch.object(
        analyzer, '_validate_terminal', return_value={}
    ):
      analyzer._provenance_precheck(analyzer._REPO_ROOT)
    self.assertLess(events.index('original96'), events.index('run_read'))
    self.assertLess(events.index('original96'), events.index('old_import'))

  def test_source_drift_aborts_before_run_read_or_helper_import(self) -> None:
    fixed, loaded = mock.Mock(), mock.Mock()
    with mock.patch.object(analyzer, '_assert_cpu_only'), mock.patch.object(
        analyzer, '_assert_destinations_fresh'
    ), mock.patch.object(
        analyzer, '_validate_new_freeze', return_value=({}, 'f' * 64)
    ), mock.patch.object(
        analyzer, '_validate_original_inventory_stdlib',
        side_effect=analyzer.AnalysisError('source drift')
    ), mock.patch.object(analyzer, '_validate_fixed_trees', fixed), mock.patch.object(
        analyzer, '_load_old_validator', loaded
    ), self.assertRaisesRegex(analyzer.AnalysisError, 'source drift'):
      analyzer._provenance_precheck(analyzer._REPO_ROOT)
    fixed.assert_not_called()
    loaded.assert_not_called()

  def test_exact_preflight_and_terminal_integration(self) -> None:
    old = analyzer._load_old_validator()
    freeze, digest, validated = analyzer._validate_original_bundle(
        analyzer._REPO_ROOT, old
    )
    terminal = analyzer._validate_terminal(old, freeze, digest, validated)
    self.assertEqual(terminal['representation']['tuple_container_count'], 32)
    self.assertEqual(terminal['run_complete_sha256'], analyzer._RUN_FILES['RUN_COMPLETE.json'][1])
    self.assertTrue(terminal['imports']['stable_shared_module_bytes'])
    self.assertTrue(terminal['protobuf']['seven_role_two_generated_output_repair_exact'])

  def test_pre_start_sections_defer_representation_branch(self) -> None:
    old = analyzer._load_old_validator()
    freeze, digest, validated = analyzer._validate_original_bundle(
        analyzer._REPO_ROOT, old
    )
    with mock.patch.object(
        analyzer, '_representation_audit',
        side_effect=AssertionError('must be post-START'),
    ):
      terminal = analyzer._validate_terminal(
          old, freeze, digest, validated, representation=False
      )
    self.assertIsNone(terminal['representation'])

  def test_preflight_transition_tamper_rejected(self) -> None:
    old = analyzer._load_old_validator()
    start = analyzer._read_json(analyzer._RUN_DIR / 'ATTEMPT_STARTED.json', 'start')
    freeze = analyzer._read_json(analyzer._OLD_FREEZE, 'freeze')
    changed = copy.deepcopy(start)
    changed['cache_role_transition']['roles_and_roots_distinct'] = False
    with self.assertRaisesRegex(analyzer.AnalysisError, 'transition'):
      analyzer._validate_preflight_standalone(
          old, changed, freeze, analyzer.ORIGINAL_FREEZE_SHA256
      )

  def test_run_has_exact_empty_raw_state(self) -> None:
    manifest = analyzer._read_json(analyzer._RUN_DIR / 'RAW_MANIFEST.json', 'manifest')
    self.assertEqual(manifest, {
        'artifact_count': 0, 'artifact_sha256': {},
        'artifact_tree_sha256': analyzer.EMPTY_SHA256,
    })
    self.assertFalse((analyzer._RUN_DIR / 'raw').exists())

  def test_original_analyzer_destinations_are_absent(self) -> None:
    self.assertFalse(analyzer._OLD_ATTEMPT_DIR.exists())
    self.assertFalse(analyzer._OLD_ANALYSIS_DIR.exists())
    self.assertFalse(analyzer._ATTEMPT_DIR.exists())
    self.assertFalse(analyzer._ANALYSIS_DIR.exists())

  def test_both_original_destinations_reject_regular_and_symlink_tamper(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      target = root / 'target'; target.write_text('x')
      for attribute in ('_OLD_ATTEMPT_DIR', '_OLD_ANALYSIS_DIR'):
        path = root / attribute
        for kind in ('regular', 'symlink'):
          if kind == 'regular':
            path.write_text('appeared')
          else:
            path.symlink_to(target)
          other = root / f'other_{attribute}'
          with self.subTest(attribute=attribute, kind=kind), mock.patch.object(
              analyzer, attribute, path
          ), mock.patch.object(
              analyzer,
              '_OLD_ANALYSIS_DIR' if attribute == '_OLD_ATTEMPT_DIR'
              else '_OLD_ATTEMPT_DIR',
              other,
          ), self.assertRaisesRegex(analyzer.AnalysisError, 'not absent'):
            analyzer._assert_old_destinations_absent()
          path.unlink()


class NoScienceAndAttemptTest(unittest.TestCase):

  def test_local_model_import_sentinel_is_rejected(self) -> None:
    for name in (
        'alphagenome_research.model.sentinel_v3331', 'jaxlib.sentinel_v3331'
    ):
      sys.modules[name] = types.ModuleType(name)
      try:
        with self.subTest(name=name), self.assertRaisesRegex(
            analyzer.AnalysisError, 'forbidden'
        ):
          analyzer._assert_cpu_only('sentinel')
      finally:
        del sys.modules[name]

  def test_source_never_calls_old_entrypoints_or_rebinds(self) -> None:
    source = MODULE_PATH.read_text(encoding='utf-8')
    self.assertNotIn('old.main(', source)
    self.assertNotIn('old.analyze(', source)
    self.assertNotIn('setattr(old', source)
    imported = []
    for node in ast.walk(ast.parse(source)):
      if isinstance(node, ast.Import):
        imported.extend(alias.name for alias in node.names)
      elif isinstance(node, ast.ImportFrom) and node.module:
        imported.append(node.module)
    self.assertFalse(any(
        name == 'jax' or name.startswith('jax.')
        or name == 'jaxlib' or name.startswith('jaxlib.')
        or name == 'alphagenome' or name.startswith('alphagenome.')
        or name == 'alphagenome_research.model'
        or name.startswith('alphagenome_research.model.')
        for name in imported
    ))

  def test_direct_analysis_bypass_rejected(self) -> None:
    with self.assertRaisesRegex(analyzer.AnalysisError, 'post-START'):
      analyzer.analyze()

  def test_post_start_source_gate_precedes_run_read_and_helper_import(self) -> None:
    events = []
    terminal = {
        'representation': {}, 'run_complete_sha256': 'a' * 64,
        'compiler_provenance_sha256': 'b' * 64, 'start': {}, 'imports': {},
        'protobuf': {}, 'kernel_cache': {},
    }
    with mock.patch.object(analyzer, '_assert_cpu_only'), mock.patch.object(
        analyzer, '_validate_active_attempt'
    ), mock.patch.object(
        analyzer, '_validate_new_freeze', side_effect=lambda root: (
            events.append('new_sources') or ({'analysis_version': analyzer.ANALYSIS_VERSION}, 'f' * 64)
        )
    ), mock.patch.object(
        analyzer, '_validate_original_inventory_stdlib',
        side_effect=lambda root: events.append('original96') or ({}, 'e' * 64)
    ), mock.patch.object(
        analyzer, '_validate_fixed_trees',
        side_effect=lambda: events.append('run_read') or {
            'run': {}, 'preflight': {}, 'model_cache': {}
        }
    ), mock.patch.object(
        analyzer, '_load_old_validator',
        side_effect=lambda: events.append('old_import') or object()
    ), mock.patch.object(
        analyzer, '_validate_original_bundle', return_value=({}, 'e' * 64, ())
    ), mock.patch.object(
        analyzer, '_validate_terminal', return_value=terminal
    ), mock.patch.object(analyzer, '_assert_old_destinations_absent'):
      analyzer.analyze(token=analyzer._ATTEMPT_TOKEN, started_sha256='f' * 64)
    self.assertLess(events.index('original96'), events.index('run_read'))
    self.assertLess(events.index('original96'), events.index('old_import'))

  def test_active_attempt_exact_start_and_token(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      attempt = Path(temporary) / 'attempt'
      attempt.mkdir()
      freeze = Path(temporary) / 'freeze.json'
      freeze.write_text('{}')
      with mock.patch.object(analyzer, '_ATTEMPT_DIR', attempt), mock.patch.object(
          analyzer, '_FREEZE_PATH', freeze
      ):
        start = analyzer._start_record({'freeze_sha256': analyzer._sha256(freeze)})
        path = attempt / 'ANALYSIS_ATTEMPT_STARTED.json'
        path.write_text(json.dumps(start, indent=2, sort_keys=True) + '\n')
        digest = analyzer._sha256(path)
        analyzer._validate_active_attempt(analyzer._ATTEMPT_TOKEN, digest)
        with self.assertRaises(analyzer.AnalysisError):
          analyzer._validate_active_attempt(object(), digest)
        start['model_apply_count'] = 1
        path.write_text(json.dumps(start, indent=2, sort_keys=True) + '\n')
        with self.assertRaises(analyzer.AnalysisError):
          analyzer._validate_active_attempt(analyzer._ATTEMPT_TOKEN, analyzer._sha256(path))

  def test_start_extra_missing_and_path_tamper_rejected(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      attempt = Path(temporary) / 'attempt'; attempt.mkdir()
      freeze = Path(temporary) / 'freeze.json'; freeze.write_text('{}')
      with mock.patch.object(analyzer, '_ATTEMPT_DIR', attempt), mock.patch.object(
          analyzer, '_FREEZE_PATH', freeze
      ):
        baseline = analyzer._start_record({'freeze_sha256': analyzer._sha256(freeze)})
        for label, mutate in (
            ('extra', lambda row: row.update({'extra': 1})),
            ('missing', lambda row: row.pop('output_json')),
            ('path', lambda row: row.update({'analysis_dir': '/tmp/wrong'})),
            ('time', lambda row: row.update({'started_at_unix_s': float('nan')})),
        ):
          row = copy.deepcopy(baseline); mutate(row)
          path = attempt / 'ANALYSIS_ATTEMPT_STARTED.json'
          path.write_text(json.dumps(row, indent=2, sort_keys=True) + '\n')
          with self.subTest(label=label), self.assertRaises(analyzer.AnalysisError):
            analyzer._validate_active_attempt(analyzer._ATTEMPT_TOKEN, analyzer._sha256(path))
          path.unlink()

  def test_output_state_records_absent_partial_and_complete(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      output = Path(temporary) / 'output'
      with mock.patch.object(analyzer, '_ANALYSIS_DIR', output):
        self.assertEqual(analyzer._output_state()['state'], 'absent')
        output.mkdir(); (output / 'ANALYSIS.json').write_text('{}')
        self.assertEqual(analyzer._output_state()['state'], 'partial')
        (output / 'RESULT.md').write_text('result')
        self.assertEqual(analyzer._output_state()['state'], 'complete')

  def test_main_append_only_success_and_rerun_refusal(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      attempt, output = root / 'attempt', root / 'analysis'
      argv = [
          str(MODULE_PATH), '--run-dir', str(analyzer._RUN_DIR),
          '--bundle-root', str(analyzer._REPO_ROOT),
          '--output-json', str(output / 'ANALYSIS.json'),
          '--output-markdown', str(output / 'RESULT.md'),
      ]
      fake_result = {
          'representation_audit': {
              'canonical_sha256': analyzer.PROGRAM_SIGNATURES_SHA256,
          }
      }

      def delegated(*, token=None, started_sha256=None):
        analyzer._validate_active_attempt(token, started_sha256)
        return fake_result

      with mock.patch.object(analyzer, '_ATTEMPT_DIR', attempt), mock.patch.object(
          analyzer, '_ANALYSIS_DIR', output
      ), mock.patch.object(
          analyzer, '_provenance_precheck', return_value={
              'freeze_sha256': analyzer._sha256(analyzer._FREEZE_PATH)
          }
      ), mock.patch.object(analyzer, 'analyze', side_effect=delegated), mock.patch.object(
          sys, 'argv', argv
      ):
        analyzer.main()
        self.assertEqual(
            {path.name for path in attempt.iterdir()},
            {'ANALYSIS_ATTEMPT_STARTED.json', 'ANALYSIS_COMPLETE.json'},
        )
        self.assertEqual(
            {path.name for path in output.iterdir()}, {'ANALYSIS.json', 'RESULT.md'}
        )
        complete = json.loads((attempt / 'ANALYSIS_COMPLETE.json').read_text())
        self.assertEqual(complete['analysis_output_state']['state'], 'complete')
        self.assertFalse(complete['scientific_summary_computed'])
        with self.assertRaises(FileExistsError):
          analyzer.main()

  def test_main_failure_persists_partial_output_and_consumes_attempt(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      attempt, output = root / 'attempt', root / 'analysis'
      argv = [
          str(MODULE_PATH), '--run-dir', str(analyzer._RUN_DIR),
          '--bundle-root', str(analyzer._REPO_ROOT),
          '--output-json', str(output / 'ANALYSIS.json'),
          '--output-markdown', str(output / 'RESULT.md'),
      ]
      fake_result = {
          'representation_audit': {
              'canonical_sha256': analyzer.PROGRAM_SIGNATURES_SHA256,
          }
      }
      with mock.patch.object(analyzer, '_ATTEMPT_DIR', attempt), mock.patch.object(
          analyzer, '_ANALYSIS_DIR', output
      ), mock.patch.object(
          analyzer, '_provenance_precheck', return_value={
              'freeze_sha256': analyzer._sha256(analyzer._FREEZE_PATH)
          }
      ), mock.patch.object(analyzer, 'analyze', return_value=fake_result), mock.patch.object(
          analyzer, '_markdown', side_effect=RuntimeError('second output failed')
      ), mock.patch.object(sys, 'argv', argv):
        with self.assertRaisesRegex(RuntimeError, 'second output'):
          analyzer.main()
        self.assertEqual(
            {path.name for path in attempt.iterdir()},
            {'ANALYSIS_ATTEMPT_STARTED.json', 'ANALYSIS_FAILURE.json'},
        )
        failure = json.loads((attempt / 'ANALYSIS_FAILURE.json').read_text())
        self.assertEqual(failure['analysis_output_state']['state'], 'partial')
        self.assertEqual(set(failure['analysis_output_state']['files']), {'ANALYSIS.json'})
        self.assertFalse(failure['scientific_raw_evidence_reached'])

  def test_main_failure_before_output_persists_absent_state(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      attempt, output = root / 'attempt', root / 'analysis'
      argv = [
          str(MODULE_PATH), '--run-dir', str(analyzer._RUN_DIR),
          '--bundle-root', str(analyzer._REPO_ROOT),
          '--output-json', str(output / 'ANALYSIS.json'),
          '--output-markdown', str(output / 'RESULT.md'),
      ]
      with mock.patch.object(analyzer, '_ATTEMPT_DIR', attempt), mock.patch.object(
          analyzer, '_ANALYSIS_DIR', output
      ), mock.patch.object(
          analyzer, '_provenance_precheck', return_value={
              'freeze_sha256': analyzer._sha256(analyzer._FREEZE_PATH)
          }
      ), mock.patch.object(
          analyzer, 'analyze', side_effect=analyzer.AnalysisError('post-start gate')
      ), mock.patch.object(sys, 'argv', argv):
        with self.assertRaisesRegex(analyzer.AnalysisError, 'post-start'):
          analyzer.main()
        failure = json.loads((attempt / 'ANALYSIS_FAILURE.json').read_text())
        self.assertEqual(failure['analysis_output_state'], {
            'state': 'absent', 'file_count': 0, 'files': {},
            'tree_sha256': analyzer.EMPTY_SHA256,
        })
        self.assertFalse(output.exists())

  def test_mid_analysis_original_destination_appearance_is_consumed_failure(self) -> None:
    for destination in ('attempt', 'output'):
      for kind in ('regular', 'symlink'):
        with self.subTest(destination=destination, kind=kind), tempfile.TemporaryDirectory() as temporary:
          root = Path(temporary)
          attempt, output = root / 'attempt', root / 'analysis'
          old_attempt, old_output = root / 'old_attempt', root / 'old_output'
          target = root / 'target'; target.write_text('target')
          appeared_path = old_attempt if destination == 'attempt' else old_output
          argv = [
              str(MODULE_PATH), '--run-dir', str(analyzer._RUN_DIR),
              '--bundle-root', str(analyzer._REPO_ROOT),
              '--output-json', str(output / 'ANALYSIS.json'),
              '--output-markdown', str(output / 'RESULT.md'),
          ]

          def appeared(**unused):
            if kind == 'regular':
              appeared_path.write_text('appeared')
            else:
              appeared_path.symlink_to(target)
            return {'representation_audit': {
                'canonical_sha256': analyzer.PROGRAM_SIGNATURES_SHA256
            }}

          with mock.patch.object(analyzer, '_ATTEMPT_DIR', attempt), mock.patch.object(
              analyzer, '_ANALYSIS_DIR', output
          ), mock.patch.object(analyzer, '_OLD_ATTEMPT_DIR', old_attempt), mock.patch.object(
              analyzer, '_OLD_ANALYSIS_DIR', old_output
          ), mock.patch.object(
              analyzer, '_provenance_precheck', return_value={
                  'freeze_sha256': analyzer._sha256(analyzer._FREEZE_PATH)
              }
          ), mock.patch.object(analyzer, 'analyze', side_effect=appeared), mock.patch.object(
              sys, 'argv', argv
          ):
            with self.assertRaisesRegex(analyzer.AnalysisError, 'destination appeared'):
              analyzer.main()
            failure = json.loads((attempt / 'ANALYSIS_FAILURE.json').read_text())
            self.assertEqual(failure['status'], 'analysis_failed_consumed_no_retry')
            self.assertEqual(failure['analysis_output_state']['state'], 'absent')
            self.assertIn('destination appeared', failure['error']['message'])

  def test_result_schema_is_structural_only(self) -> None:
    source = MODULE_PATH.read_text(encoding='utf-8')
    for literal in (
        "'model_apply_count': 0", "'raw_record_count': 0",
        "'scientific_summary_computed': False",
        "'donor_normalization_computed': False",
        "'shapley_or_nomination_computed': False",
        "'nomination_performed': False", "'combined_analysis_permitted': False",
    ):
      self.assertIn(literal, source)

  def test_path_guard_rejects_confirmation_paths(self) -> None:
    with self.assertRaises(analyzer.AnalysisError):
      analyzer._guard_path(Path('/tmp/confirmation/results'))


if __name__ == '__main__':
  unittest.main()
