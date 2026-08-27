"""CPU-only tests for the v3.2.3 external-preflight schema amendment."""

from __future__ import annotations

import copy
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
    'analyze_superset_graph_v3_2_3_test_target',
    _HERE / 'analyze_superset_graph_v3_2_3.py',
)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n',
      encoding='utf-8',
  )


class PreflightFixture:

  def __init__(self, root: Path):
    self.root = root / 'preflight'
    self.root.mkdir()
    self.stdout = self.root / 'preflight_0000.stdout.log'
    self.stderr = self.root / 'preflight_0000.stderr.log'
    self.stdout.write_bytes(b'')
    self.stderr.write_bytes(b'')
    self.raw = {
        'script_version': 'synthetic-preflight',
        'status': 'pass',
        'logs': {
            'stdout': {
                'path': str(self.stdout.resolve()),
                'sha256': _sha256(self.stdout),
            },
            'stderr': {
                'path': str(self.stderr.resolve()),
                'sha256': _sha256(self.stderr),
            },
        },
        'shared_value': {'exact': True},
    }
    self.path = self.root / 'preflight_0000.json'
    self.freeze = {'preflight_dir': str(self.root.resolve())}
    self.start = {}
    self.refresh()

  def refresh(self) -> None:
    _write_json(self.path, self.raw)
    self.start = {
        'external_preflight': {
            'path': str(self.path.resolve()),
            'sha256': _sha256(self.path),
            **self.raw,
            'validated_logs': copy.deepcopy(self.raw['logs']),
        }
    }

  def normalize(self):
    return analyzer._normalize_external_preflight(
        self.start, self.freeze,
        expected_preflight_sha256=_sha256(self.path),
        expected_log_sha256=hashlib.sha256(b'').hexdigest(),
    )


class ExternalPreflightNormalizationTest(unittest.TestCase):

  def test_exact_historical_augmented_schema_removes_only_derived_field(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_3_exact_') as directory:
      fixture = PreflightFixture(Path(directory))
      normalized, audit = fixture.normalize()
      self.assertNotIn(
          'validated_logs', normalized['external_preflight']
      )
      self.assertEqual(
          {
              key: value for key, value in normalized['external_preflight'].items()
              if key not in {'path', 'sha256'}
          },
          fixture.raw,
      )
      self.assertTrue(audit['derived_validated_logs_verified'])

  def test_missing_or_extra_embedded_field_is_rejected(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_3_missing_') as directory:
      fixture = PreflightFixture(Path(directory))
      del fixture.start['external_preflight']['shared_value']
      with self.assertRaisesRegex(ValueError, 'unexpected fields'):
        fixture.normalize()
    with tempfile.TemporaryDirectory(prefix='v3_2_3_extra_') as directory:
      fixture = PreflightFixture(Path(directory))
      fixture.start['external_preflight']['unexpected'] = True
      with self.assertRaisesRegex(ValueError, 'unexpected fields'):
        fixture.normalize()

  def test_changed_shared_artifact_value_is_rejected(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_3_shared_') as directory:
      fixture = PreflightFixture(Path(directory))
      fixture.start['external_preflight']['shared_value'] = {'exact': False}
      with self.assertRaisesRegex(ValueError, 'differs at shared_value'):
        fixture.normalize()

  def test_missing_extra_or_unequal_validated_logs_is_rejected(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_3_logs_missing_') as directory:
      fixture = PreflightFixture(Path(directory))
      del fixture.start['external_preflight']['validated_logs']
      with self.assertRaisesRegex(ValueError, 'unexpected fields'):
        fixture.normalize()
    with tempfile.TemporaryDirectory(prefix='v3_2_3_logs_extra_') as directory:
      fixture = PreflightFixture(Path(directory))
      fixture.start['external_preflight']['validated_logs']['extra'] = {}
      with self.assertRaisesRegex(ValueError, 'validated_logs changed'):
        fixture.normalize()
    with tempfile.TemporaryDirectory(prefix='v3_2_3_logs_diff_') as directory:
      fixture = PreflightFixture(Path(directory))
      fixture.start['external_preflight']['validated_logs']['stdout'][
          'sha256'
      ] = '0' * 64
      with self.assertRaisesRegex(ValueError, 'validated_logs changed'):
        fixture.normalize()
    with tempfile.TemporaryDirectory(prefix='v3_2_3_logs_malformed_') as directory:
      fixture = PreflightFixture(Path(directory))
      fixture.start['external_preflight']['validated_logs'] = []
      with self.assertRaisesRegex(ValueError, 'validated_logs changed'):
        fixture.normalize()

  def test_saved_log_schema_extra_is_rejected(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_3_raw_logs_extra_') as directory:
      fixture = PreflightFixture(Path(directory))
      fixture.raw['logs']['extra'] = {}
      fixture.refresh()
      with self.assertRaisesRegex(ValueError, 'log schema changed'):
        fixture.normalize()

  def test_log_escape_symlink_and_missing_file_are_rejected(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_3_escape_') as directory:
      fixture = PreflightFixture(Path(directory))
      outside = Path(directory) / 'outside.log'
      outside.write_bytes(b'')
      fixture.raw['logs']['stdout']['path'] = str(outside.resolve())
      fixture.refresh()
      with self.assertRaisesRegex(ValueError, 'log path/hash changed'):
        fixture.normalize()
    with tempfile.TemporaryDirectory(prefix='v3_2_3_symlink_') as directory:
      fixture = PreflightFixture(Path(directory))
      target = fixture.root / 'target.log'
      target.write_bytes(b'')
      fixture.stdout.unlink()
      fixture.stdout.symlink_to(target)
      with self.assertRaisesRegex(ValueError, 'log path/hash changed'):
        fixture.normalize()
    with tempfile.TemporaryDirectory(prefix='v3_2_3_missing_file_') as directory:
      fixture = PreflightFixture(Path(directory))
      fixture.stdout.unlink()
      with self.assertRaisesRegex(ValueError, 'log path/hash changed'):
        fixture.normalize()

  def test_current_log_byte_or_hash_tampering_is_rejected(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_3_tamper_') as directory:
      fixture = PreflightFixture(Path(directory))
      fixture.stdout.write_bytes(b'tamper')
      with self.assertRaisesRegex(ValueError, 'log path/hash changed'):
        fixture.normalize()

  def test_real_frozen_preflight_successfully_delegates_without_raw_read(self):
    start_path = (
        _HERE / 'results/v3_2_development_superset_graph_one_shot/'
        'ATTEMPT_STARTED.json'
    )
    start = json.loads(start_path.read_text(encoding='utf-8'))
    audit = analyzer._validate_preflight(
        start, start['freeze'], start['freeze']['sha256']
    )
    self.assertTrue(audit['derived_validated_logs_verified'])
    self.assertTrue(audit['external_preflight_logs_verified'])

  def test_consumed_v3_2_2_attempt_is_exactly_bound(self):
    audit = analyzer._validate_consumed_v3_2_2_attempt()
    self.assertEqual(
        audit['attempt_started_sha256'], analyzer.V3_2_2_ATTEMPT_SHA256
    )
    self.assertFalse(audit['scientific_output_written'])

  def test_v3_2_3_attempt_start_revalidation_and_single_use(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_3_attempt_') as directory:
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

  def test_prior_analyzer_bytes_remain_unchanged(self):
    self.assertEqual(
        _sha256(analyzer._V3_2_2_PATH), analyzer.V3_2_2_ANALYZER_SHA256
    )
    self.assertEqual(
        _sha256(analyzer._V3_2_2_TEST_PATH), analyzer.V3_2_2_TEST_SHA256
    )


if __name__ == '__main__':
  unittest.main()
