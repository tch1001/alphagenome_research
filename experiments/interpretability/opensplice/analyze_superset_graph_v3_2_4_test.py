"""CPU-only tests for the exact v3.2.4 import-alias amendment."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


_HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
  specification = importlib.util.spec_from_file_location(name, path)
  assert specification is not None and specification.loader is not None
  module = importlib.util.module_from_spec(specification)
  specification.loader.exec_module(module)
  return module


analyzer = _load(
    'analyze_superset_graph_v3_2_4_test_target',
    _HERE / 'analyze_superset_graph_v3_2_4.py',
)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n',
      encoding='utf-8',
  )


class ImportFixture:

  def __init__(self, root: Path):
    self.bundle = root / 'repo'
    self.upstream = root / 'alphagenome'
    self.bundle.mkdir()
    self.upstream.mkdir()
    self.run_dir = root / 'run'
    self.run_dir.mkdir()
    self.runner = (
        self.bundle / 'experiments/interpretability/opensplice/'
        'run_superset_graph_v3_2.py'
    )
    self.runner.parent.mkdir(parents=True)
    self.runner.write_text('synthetic runner\n', encoding='utf-8')
    self.rows = [
        self._row('__main__', self.runner),
        self._row('__mp_main__', self.runner),
    ]
    required = [
        'alphagenome_research.model.dna_model',
        'alphagenome_research.model.model',
        'target_reducers_v3',
    ]
    for index, name in enumerate(required):
      path = self.bundle / f'modules/required_{index}.py'
      path.parent.mkdir(exist_ok=True)
      path.write_text(f'{name}\n', encoding='utf-8')
      self.rows.append(self._row(name, path))
    for index in range(69):
      name = f'synthetic.module_{index:03d}'
      path = self.bundle / f'modules/synthetic_{index:03d}.py'
      path.write_text(f'{name}\n', encoding='utf-8')
      self.rows.append(self._row(name, path))
    self.rows.sort(key=lambda row: row['name'])
    assert len(self.rows) == 74
    self.value = {'module_count': 74, 'modules': self.rows}
    self.paths = {
        phase: self.run_dir / filename
        for phase, filename in analyzer._IMPORT_FILENAMES.items()
    }
    self.write_all()

  @staticmethod
  def _row(name: str, path: Path) -> dict:
    return {
        'name': name,
        'path': str(path.absolute()),
        'root': 'alphagenome_research_checkout',
        'sha256': _sha256(path),
        'size_bytes': path.stat().st_size,
    }

  def write_all(self) -> None:
    for path in self.paths.values():
      _write_json(path, self.value)

  def write_one(self, phase: str = 'pre_model') -> Path:
    path = self.paths[phase]
    _write_json(path, self.value)
    return path

  def expectations(self, artifact: Path | None = None) -> dict:
    artifact = artifact or self.paths['pre_model']
    return {
        'artifact_sha256': _sha256(artifact),
        'artifact_size': artifact.stat().st_size,
        'module_count': 74,
        'path_group_count': 73,
        'runner_path': self.runner,
        'runner_sha256': self.rows[0]['sha256'],
        'runner_size': self.rows[0]['size_bytes'],
        'required_modules': set(analyzer._REQUIRED_MODULES),
        'allowed_artifact_paths': {
            path.absolute() for path in self.paths.values()
        },
    }

  def validate_one(self, phase: str = 'pre_model'):
    path = self.write_one(phase)
    expectations = self.expectations(path)
    return analyzer._validate_import_provenance_impl(
        path, expectations['artifact_sha256'], bundle_root=self.bundle,
        **expectations,
    )

  def complete(self) -> dict:
    digest = _sha256(self.paths['pre_model'])
    return {
        'import_provenance_phases': {
            phase: digest for phase in analyzer._IMPORT_FILENAMES
        },
        'import_provenance_sha256': digest,
    }


class ImportAliasValidationTest(unittest.TestCase):

  def test_exact_alias_pair_preserves_both_rows(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_4_exact_') as directory:
      fixture = ImportFixture(Path(directory))
      result = fixture.validate_one()
      self.assertEqual(result['module_count'], 74)
      self.assertEqual(len(result['modules']), 74)
      self.assertTrue(result['alias_audit']['accepted_exact_alias'])
      self.assertTrue(result['alias_audit']['rows_preserved'])
      self.assertEqual(
          [row['name'] for row in result['modules'][:2]],
          ['__main__', '__mp_main__'],
      )

  def test_duplicate_name_or_unrelated_duplicate_path_is_rejected(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_4_dup_name_') as directory:
      fixture = ImportFixture(Path(directory))
      fixture.rows[3]['name'] = fixture.rows[2]['name']
      with self.assertRaisesRegex(ValueError, 'unsorted or duplicated'):
        fixture.validate_one()
    with tempfile.TemporaryDirectory(prefix='v3_2_4_dup_path_') as directory:
      fixture = ImportFixture(Path(directory))
      fixture.rows[3]['path'] = fixture.rows[2]['path']
      fixture.rows[3]['sha256'] = fixture.rows[2]['sha256']
      fixture.rows[3]['size_bytes'] = fixture.rows[2]['size_bytes']
      with self.assertRaisesRegex(ValueError, 'path group count changed'):
        fixture.validate_one()

  def test_third_alias_or_wrong_alias_name_is_rejected(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_4_third_') as directory:
      fixture = ImportFixture(Path(directory))
      fixture.rows[2]['path'] = fixture.rows[0]['path']
      fixture.rows[2]['sha256'] = fixture.rows[0]['sha256']
      fixture.rows[2]['size_bytes'] = fixture.rows[0]['size_bytes']
      with self.assertRaisesRegex(ValueError, 'path group count changed'):
        fixture.validate_one()
    with tempfile.TemporaryDirectory(prefix='v3_2_4_wrong_name_') as directory:
      fixture = ImportFixture(Path(directory))
      fixture.rows[1]['name'] = '__mp_main_bad__'
      with self.assertRaisesRegex(ValueError, 'alias is not exact'):
        fixture.validate_one()

  def test_wrong_alias_path_root_hash_or_size_is_rejected(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_4_wrong_path_') as directory:
      fixture = ImportFixture(Path(directory))
      alternate = fixture.bundle / 'alternate.py'
      alternate.write_text('synthetic runner\n', encoding='utf-8')
      fixture.rows[1] = fixture._row('__mp_main__', alternate)
      with self.assertRaisesRegex(ValueError, 'path group count changed'):
        fixture.validate_one()
    with tempfile.TemporaryDirectory(prefix='v3_2_4_wrong_root_') as directory:
      fixture = ImportFixture(Path(directory))
      fixture.rows[1]['root'] = 'upstream_alphagenome_checkout'
      with self.assertRaisesRegex(ValueError, 'escaped its declared root'):
        fixture.validate_one()
    with tempfile.TemporaryDirectory(prefix='v3_2_4_wrong_hash_') as directory:
      fixture = ImportFixture(Path(directory))
      fixture.rows[1]['sha256'] = '0' * 64
      with self.assertRaisesRegex(ValueError, 'module hash changed'):
        fixture.validate_one()
    with tempfile.TemporaryDirectory(prefix='v3_2_4_wrong_size_') as directory:
      fixture = ImportFixture(Path(directory))
      fixture.rows[1]['size_bytes'] += 1
      with self.assertRaisesRegex(ValueError, 'module size changed'):
        fixture.validate_one()

  def test_live_bytes_symlink_or_nonfile_is_rejected(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_4_live_bytes_') as directory:
      fixture = ImportFixture(Path(directory))
      fixture.runner.write_text('changed runner\n', encoding='utf-8')
      with self.assertRaisesRegex(ValueError, 'module hash changed'):
        fixture.validate_one()
    with tempfile.TemporaryDirectory(prefix='v3_2_4_symlink_') as directory:
      fixture = ImportFixture(Path(directory))
      target = fixture.runner.with_name('runner_target.py')
      fixture.runner.rename(target)
      fixture.runner.symlink_to(target)
      with self.assertRaisesRegex(ValueError, 'symlinked or not a regular file'):
        fixture.validate_one()
    with tempfile.TemporaryDirectory(prefix='v3_2_4_nonfile_') as directory:
      fixture = ImportFixture(Path(directory))
      fixture.runner.unlink()
      fixture.runner.mkdir()
      with self.assertRaisesRegex(ValueError, 'symlinked or not a regular file'):
        fixture.validate_one()

  def test_missing_extra_unsorted_or_malformed_row_is_rejected(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_4_missing_') as directory:
      fixture = ImportFixture(Path(directory))
      fixture.rows.pop()
      with self.assertRaisesRegex(ValueError, 'list/count is invalid'):
        fixture.validate_one()
    with tempfile.TemporaryDirectory(prefix='v3_2_4_extra_') as directory:
      fixture = ImportFixture(Path(directory))
      fixture.rows.append(dict(fixture.rows[-1], name='zz_extra'))
      with self.assertRaisesRegex(ValueError, 'list/count is invalid'):
        fixture.validate_one()
    with tempfile.TemporaryDirectory(prefix='v3_2_4_unsorted_') as directory:
      fixture = ImportFixture(Path(directory))
      fixture.rows[2], fixture.rows[3] = fixture.rows[3], fixture.rows[2]
      with self.assertRaisesRegex(ValueError, 'unsorted or duplicated'):
        fixture.validate_one()
    with tempfile.TemporaryDirectory(prefix='v3_2_4_malformed_') as directory:
      fixture = ImportFixture(Path(directory))
      fixture.rows[2]['extra'] = True
      with self.assertRaisesRegex(ValueError, 'schema changed'):
        fixture.validate_one()

  def test_changed_phase_artifact_or_binding_is_rejected(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_4_phase_') as directory:
      fixture = ImportFixture(Path(directory))
      expectations = fixture.expectations()
      result = analyzer._validate_import_phases_impl(
          fixture.run_dir, fixture.complete(), bundle_root=fixture.bundle,
          expectations=expectations, expected_run_dir=fixture.run_dir,
      )
      self.assertTrue(result['phase_artifacts_byte_identical'])
      changed_binding = fixture.complete()
      changed_binding['import_provenance_phases']['pre_model'] = '0' * 64
      with self.assertRaisesRegex(ValueError, 'bindings changed'):
        analyzer._validate_import_phases_impl(
            fixture.run_dir, changed_binding, bundle_root=fixture.bundle,
            expectations=expectations, expected_run_dir=fixture.run_dir,
        )
      fixture.paths['postcompile'].write_bytes(b'changed')
      with self.assertRaisesRegex(ValueError, 'size/hash binding mismatch'):
        analyzer._validate_import_phases_impl(
            fixture.run_dir, fixture.complete(), bundle_root=fixture.bundle,
            expectations=expectations, expected_run_dir=fixture.run_dir,
        )

  def test_real_three_phase_provenance_validates_without_scientific_reads(self):
    run_dir = (
        _HERE / 'results/v3_2_development_superset_graph_one_shot'
    )
    complete = json.loads((run_dir / 'RUN_COMPLETE.json').read_text())
    result = analyzer._validate_import_phases(
        run_dir, complete, bundle_root=analyzer._base._REPO_ROOT
    )
    self.assertEqual(set(result['module_counts'].values()), {74})
    self.assertTrue(result['phase_artifacts_byte_identical'])
    self.assertTrue(
        all(x['rows_preserved'] for x in result['exact_alias_audits'].values())
    )
    analyzer._v321._assert_no_model_imports('v3.2.4 real import test')

  def test_consumed_v3_2_3_attempt_is_exactly_bound(self):
    audit = analyzer._validate_consumed_v3_2_3_attempt()
    self.assertEqual(
        audit['attempt_started_sha256'], analyzer.V3_2_3_ATTEMPT_SHA256
    )
    self.assertFalse(audit['scientific_output_written'])

  def test_v3_2_4_attempt_is_single_use_and_hash_bound(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_4_attempt_') as directory:
      saved = analyzer._ATTEMPT_DIR
      analyzer._ATTEMPT_DIR = Path(directory) / 'attempt'
      binding = {'git_head': '1' * 40}
      try:
        digest = analyzer._start_attempt(
            Path(directory) / 'run', Path(directory) / 'analysis.json', None,
            binding,
        )
        analyzer._validate_started_attempt(binding, digest)
        with self.assertRaisesRegex(ValueError, 'content/binding changed'):
          analyzer._validate_started_attempt({'git_head': '2' * 40}, digest)
        with self.assertRaisesRegex(FileExistsError, 'already consumed'):
          analyzer._start_attempt(
              Path(directory) / 'run', Path(directory) / 'analysis.json', None,
              binding,
          )
      finally:
        analyzer._ATTEMPT_DIR = saved

  def test_prior_analyzer_bundle_bytes_remain_unchanged(self):
    bundle = analyzer._validate_prior_analyzer_bundles(
        analyzer._base._REPO_ROOT
    )
    self.assertGreaterEqual(len(bundle), 10)
    self.assertEqual(
        _sha256(analyzer._V3_2_3_PATH), analyzer.V3_2_3_ANALYZER_SHA256
    )
    self.assertEqual(
        _sha256(analyzer._V3_2_3_TEST_PATH), analyzer.V3_2_3_TEST_SHA256
    )
    self.assertEqual(
        _sha256(analyzer._V3_2_3_AMENDMENT_PATH),
        analyzer.V3_2_3_AMENDMENT_SHA256,
    )


if __name__ == '__main__':
  unittest.main()
