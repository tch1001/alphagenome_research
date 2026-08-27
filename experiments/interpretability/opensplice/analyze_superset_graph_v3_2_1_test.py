"""CPU-only tests for the v3.2.1 offline-analyzer symlink amendment."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest


_HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
  specification = importlib.util.spec_from_file_location(name, path)
  assert specification is not None and specification.loader is not None
  module = importlib.util.module_from_spec(specification)
  specification.loader.exec_module(module)
  return module


analyzer = _load(
    'analyze_superset_graph_v3_2_1_test_target',
    _HERE / 'analyze_superset_graph_v3_2_1.py',
)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class CheckpointFixture:

  def __init__(self, root: Path, *, symlink_relative: str | None = None):
    self.model_root = root / 'models--google--alphagenome-all-folds'
    self.snapshot = (
        self.model_root / 'snapshots' / analyzer._v3.CHECKPOINT_SNAPSHOT
    )
    self.blobs = self.model_root / 'blobs'
    self.blobs.mkdir(parents=True)
    names = [
        '.gitattributes', 'README.md', '_CHECKPOINT_METADATA', '_METADATA',
        'd/part0', 'manifest.ocdbt', 'notebook.ipynb',
        'ocdbt.process_0/d/part1', 'ocdbt.process_0/d/part2',
        'ocdbt.process_0/d/part3', 'ocdbt.process_0/d/part4',
        'ocdbt.process_0/manifest.ocdbt',
    ]
    self.records = []
    self.symlink_path = None
    self.symlink_blob = None
    for index, relative in enumerate(sorted(names)):
      data = f'frozen checkpoint content {index}\n'.encode()
      lexical = self.snapshot / relative
      lexical.parent.mkdir(parents=True, exist_ok=True)
      if relative == symlink_relative:
        blob_id = hashlib.sha256(f'blob {index}'.encode()).hexdigest()
        blob = self.blobs / blob_id
        blob.write_bytes(data)
        depth = len(Path(relative).parts) + 1
        link = Path(*(('..',) * depth), 'blobs', blob_id)
        lexical.symlink_to(link)
        self.symlink_path = lexical
        self.symlink_blob = blob
      else:
        lexical.write_bytes(data)
      self.records.append({
          'relative_path': relative,
          'size_bytes': len(data),
          'sha256': hashlib.sha256(data).hexdigest(),
      })
    self.manifest = root / 'checkpoint_manifest.tsv'
    self.manifest.write_text(
        ''.join(
            f"{row['relative_path']}\t{row['size_bytes']}\t{row['sha256']}\n"
            for row in self.records
        ),
        encoding='utf-8',
    )
    self.binding = {
        'snapshot_path': str(self.snapshot.resolve()),
        'snapshot_name': analyzer._v3.CHECKPOINT_SNAPSHOT,
        'manifest_path': str(self.manifest.resolve()),
        'manifest_sha256': _sha256(self.manifest),
        'file_count': 12,
        'files': self.records,
    }

  def validate(self):
    return analyzer._validate_checkpoint_tree(
        self.snapshot, self.manifest, _sha256(self.manifest), self.binding, {}
    )


class CheckpointSymlinkPolicyTest(unittest.TestCase):

  def test_valid_real_files_remain_accepted(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_1_real_') as directory:
      fixture = CheckpointFixture(Path(directory))
      audit = fixture.validate()
      self.assertEqual(audit['checkpoint_file_count'], 12)
      self.assertEqual(audit['checkpoint_symlink_count'], 0)

  def test_valid_depth_derived_one_hop_hf_symlinks_are_accepted(self):
    for relative in ('.gitattributes', 'd/part0', 'ocdbt.process_0/d/part1'):
      with self.subTest(relative=relative), tempfile.TemporaryDirectory(
          prefix='v3_2_1_link_'
      ) as directory:
        fixture = CheckpointFixture(
            Path(directory), symlink_relative=relative
        )
        audit = fixture.validate()
        self.assertEqual(audit['checkpoint_symlink_count'], 1)
        self.assertEqual(
            audit['checkpoint_symlinks'][0]['relative_path'], relative
        )

  def test_absolute_or_escaping_symlink_is_rejected(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_1_escape_') as directory:
      root = Path(directory)
      fixture = CheckpointFixture(root, symlink_relative='.gitattributes')
      outside = root / 'outside'
      outside.write_bytes(fixture.symlink_blob.read_bytes())
      fixture.symlink_path.unlink()
      fixture.symlink_path.symlink_to(outside)
      with self.assertRaisesRegex(ValueError, 'exact depth-derived'):
        fixture.validate()

  def test_chained_blob_symlink_is_rejected(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_1_chain_') as directory:
      root = Path(directory)
      fixture = CheckpointFixture(root, symlink_relative='.gitattributes')
      target = root / 'real_blob'
      target.write_bytes(fixture.symlink_blob.read_bytes())
      fixture.symlink_blob.unlink()
      fixture.symlink_blob.symlink_to(target)
      with self.assertRaisesRegex(ValueError, 'chained'):
        fixture.validate()

  def test_dangling_blob_symlink_is_rejected(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_1_dangling_') as directory:
      fixture = CheckpointFixture(
          Path(directory), symlink_relative='.gitattributes'
      )
      fixture.symlink_blob.unlink()
      with self.assertRaisesRegex(ValueError, 'cannot be resolved'):
        fixture.validate()

  def test_blob_byte_tampering_is_rejected(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_1_tamper_') as directory:
      fixture = CheckpointFixture(
          Path(directory), symlink_relative='.gitattributes'
      )
      original = fixture.symlink_blob.read_bytes()
      fixture.symlink_blob.write_bytes(b'X' + original[1:])
      with self.assertRaisesRegex(ValueError, 'content changed'):
        fixture.validate()

  def test_extra_file_and_extra_directory_are_rejected(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_1_extra_file_') as directory:
      fixture = CheckpointFixture(Path(directory))
      (fixture.snapshot / 'extra').write_text('extra', encoding='utf-8')
      with self.assertRaisesRegex(ValueError, 'exact 12-file manifest'):
        fixture.validate()
    with tempfile.TemporaryDirectory(prefix='v3_2_1_extra_dir_') as directory:
      fixture = CheckpointFixture(Path(directory))
      (fixture.snapshot / 'unused').mkdir()
      with self.assertRaisesRegex(ValueError, 'extra directory'):
        fixture.validate()

  def test_append_only_attempt_directory_cannot_be_reused(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_1_attempt_') as directory:
      saved = analyzer._ATTEMPT_DIR
      analyzer._ATTEMPT_DIR = Path(directory) / 'attempt'
      try:
        _, digest = analyzer._start_analysis_attempt(
            Path(directory) / 'run', Path(directory) / 'analysis.json', None,
            {'git_head': '1' * 40},
        )
        self.assertEqual(len(digest), 64)
        with self.assertRaisesRegex(FileExistsError, 'already consumed'):
          analyzer._start_analysis_attempt(
              Path(directory) / 'run', Path(directory) / 'analysis.json', None,
              {'git_head': '1' * 40},
          )
      finally:
        analyzer._ATTEMPT_DIR = saved

  def test_original_frozen_analyzer_is_unchanged(self):
    self.assertEqual(
        _sha256(analyzer._ORIGINAL_PATH), analyzer.ORIGINAL_ANALYZER_SHA256
    )

  def test_same_process_jax_or_model_import_is_rejected(self):
    sys.modules['jax.synthetic_v3_2_1_test'] = types.ModuleType(
        'jax.synthetic_v3_2_1_test'
    )
    try:
      with self.assertRaisesRegex(RuntimeError, 'forbidden model/JAX imports'):
        analyzer._assert_no_model_imports('synthetic')
    finally:
      del sys.modules['jax.synthetic_v3_2_1_test']

  def test_unrelated_tracked_tree_tampering_is_rejected(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_1_git_') as directory:
      repository = Path(directory)
      tracked = repository / 'unrelated_tracked.txt'
      tracked.write_text('frozen\n', encoding='utf-8')
      subprocess.run(('git', 'init', '-q'), cwd=repository, check=True)
      subprocess.run(('git', 'add', 'unrelated_tracked.txt'), cwd=repository, check=True)
      subprocess.run(
          (
              'git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
              '-c', 'commit.gpgsign=false', 'commit', '-qm', 'freeze',
          ),
          cwd=repository, check=True,
      )
      analyzer._assert_global_tracked_head_clean(repository)
      tracked.write_text('tampered\n', encoding='utf-8')
      with self.assertRaisesRegex(ValueError, 'tracked files differ'):
        analyzer._assert_global_tracked_head_clean(repository)


if __name__ == '__main__':
  unittest.main()
