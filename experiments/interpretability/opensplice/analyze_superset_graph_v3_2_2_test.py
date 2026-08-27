"""CPU-only tests for the v3.2.2 protobuf provenance amendment."""

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
    'analyze_superset_graph_v3_2_2_test_target',
    _HERE / 'analyze_superset_graph_v3_2_2.py',
)


def _binding(path: Path) -> dict:
  data = path.read_bytes()
  return {'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}


class ProtobufFixture:

  def __init__(self, bundle: Path):
    self.bundle = bundle
    self.proto_dir = bundle / 'src/alphagenome_research/protos'
    self.proto_dir.mkdir(parents=True)
    self.pb2 = self.proto_dir / 'calibration_scores_pb2.py'
    self.pyi = self.proto_dir / 'calibration_scores_pb2.pyi'
    self.pb2.write_bytes(b'# exact generated pb2\n')
    self.pyi.write_bytes(b'# exact generated pyi\n')
    self.freeze = {
        'protobuf_binding': {
            'generated_outputs': {
                str(self.pb2.resolve()): _binding(self.pb2),
                str(self.pyi.resolve()): _binding(self.pyi),
            },
            # The historical schema redundantly binds only the .py explicitly.
            'imported_pb2': {
                'path': str(self.pb2.resolve()), **_binding(self.pb2)
            },
        },
    }

  def normalize(self):
    return analyzer._normalize_frozen_protobuf_binding(
        self.freeze, bundle_root=self.bundle
    )


class ProtobufPathKeyNormalizationTest(unittest.TestCase):

  def test_exact_path_key_schema_includes_pyi_only_keyed_binding(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_2_exact_') as directory:
      fixture = ProtobufFixture(Path(directory))
      normalized, audit = fixture.normalize()
      pyi = normalized['protobuf_binding']['generated_outputs'][
          str(fixture.pyi.resolve())
      ]
      self.assertEqual(pyi['path'], str(fixture.pyi.resolve()))
      self.assertTrue(audit['path_key_schema_verified'])
      self.assertEqual(audit['generated_output_count'], 2)

  def test_missing_or_extra_generated_output_is_rejected(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_2_missing_') as directory:
      fixture = ProtobufFixture(Path(directory))
      del fixture.freeze['protobuf_binding']['generated_outputs'][
          str(fixture.pyi.resolve())
      ]
      with self.assertRaisesRegex(ValueError, 'exactly pb2/pyi'):
        fixture.normalize()
    with tempfile.TemporaryDirectory(prefix='v3_2_2_extra_') as directory:
      fixture = ProtobufFixture(Path(directory))
      extra = fixture.proto_dir / 'extra.py'
      extra.write_bytes(b'extra')
      fixture.freeze['protobuf_binding']['generated_outputs'][
          str(extra.resolve())
      ] = _binding(extra)
      with self.assertRaisesRegex(ValueError, 'exactly pb2/pyi'):
        fixture.normalize()

  def test_malformed_value_or_path_escape_is_rejected(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_2_malformed_') as directory:
      fixture = ProtobufFixture(Path(directory))
      fixture.freeze['protobuf_binding']['generated_outputs'][
          str(fixture.pyi.resolve())
      ]['path'] = str(fixture.pyi.resolve())
      with self.assertRaisesRegex(ValueError, 'value schema'):
        fixture.normalize()
    with tempfile.TemporaryDirectory(prefix='v3_2_2_escape_') as directory:
      fixture = ProtobufFixture(Path(directory))
      original = fixture.freeze['protobuf_binding']['generated_outputs'].pop(
          str(fixture.pyi.resolve())
      )
      fixture.freeze['protobuf_binding']['generated_outputs'][
          str(Path(directory).parent / fixture.pyi.name)
      ] = original
      with self.assertRaisesRegex(ValueError, 'exactly pb2/pyi'):
        fixture.normalize()

  def test_live_generated_byte_and_size_tampering_is_rejected(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_2_tamper_') as directory:
      fixture = ProtobufFixture(Path(directory))
      fixture.pyi.write_bytes(b'# altered generated pyi\n')
      with self.assertRaisesRegex(ValueError, 'bytes changed'):
        fixture.normalize()

  def test_explicit_path_collision_agreement_and_disagreement(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_2_collision_') as directory:
      fixture = ProtobufFixture(Path(directory))
      fixture.normalize()
      fixture.freeze['protobuf_binding']['imported_pb2']['sha256'] = '0' * 64
      with self.assertRaisesRegex(ValueError, 'bindings disagree'):
        fixture.normalize()

  def test_consumed_v3_2_1_attempt_is_exactly_bound(self):
    audit = analyzer._validate_consumed_v3_2_1_attempt()
    self.assertEqual(
        audit['attempt_started_sha256'], analyzer.V3_2_1_ATTEMPT_SHA256
    )
    self.assertFalse(audit['scientific_output_written'])

  def test_real_frozen_bootstrap_path_key_schema_validates_without_raw_read(self):
    start_path = (
        _HERE / 'results/v3_2_development_superset_graph_one_shot/'
        'ATTEMPT_STARTED.json'
    )
    start = json.loads(start_path.read_text(encoding='utf-8'))
    audit = analyzer._validate_bootstrap_attestation(
        start, start['freeze'], start['freeze']['sha256'],
        bundle_root=analyzer._base._REPO_ROOT,
    )
    self.assertTrue(audit['frozen_generated_outputs_path_key_schema_verified'])
    self.assertEqual(audit['frozen_generated_output_count'], 2)

  def test_v3_2_2_attempt_is_append_only_and_single_use(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_2_attempt_') as directory:
      saved = analyzer._ATTEMPT_DIR
      analyzer._ATTEMPT_DIR = Path(directory) / 'attempt'
      try:
        digest = analyzer._start_attempt(
            Path(directory) / 'run', Path(directory) / 'analysis.json', None,
            {'git_head': '1' * 40},
        )
        self.assertEqual(len(digest), 64)
        analyzer._validate_started_attempt({'git_head': '1' * 40}, digest)
        with self.assertRaisesRegex(ValueError, 'content/binding changed'):
          analyzer._validate_started_attempt({'git_head': '2' * 40}, digest)
        with self.assertRaisesRegex(FileExistsError, 'already consumed'):
          analyzer._start_attempt(
              Path(directory) / 'run', Path(directory) / 'analysis.json', None,
              {'git_head': '1' * 40},
          )
      finally:
        analyzer._ATTEMPT_DIR = saved

  def test_v3_2_and_v3_2_1_analyzers_are_unchanged(self):
    self.assertEqual(
        analyzer._sha256(analyzer._V3_2_1_PATH),
        analyzer.V3_2_1_ANALYZER_SHA256,
    )
    self.assertEqual(
        analyzer._sha256(analyzer._V3_2_1_TEST_PATH),
        analyzer.V3_2_1_TEST_SHA256,
    )
    self.assertEqual(
        analyzer._sha256(analyzer._v321._ORIGINAL_PATH),
        analyzer._v321.ORIGINAL_ANALYZER_SHA256,
    )


if __name__ == '__main__':
  unittest.main()
