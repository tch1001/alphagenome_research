#!/usr/bin/env python3
"""CPU-only analyzer repair for the exact v3.3 bootstrap role-set defect.

This wrapper preserves the frozen v3.3 analyzer and changes only how the
seven-role pre-import protobuf attestation is presented to its two-output
generated-file check.  All scientific and controlled-stop validation remains
delegated to the byte-bound v3.3 analyzer.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import traceback
from typing import Any, Iterator, Mapping, Sequence


ANALYSIS_VERSION = 'opensplice-encoder-skip-localization-analysis-v3.3.1'
AMENDMENT_REASON = 'seven_role_two_generated_output_provenance_classification'
AMENDMENT_SHA256 = (
    '37e23b251f53ab87bae99b63024a381c367ce33bbc950a2227b3267fbc9668d1'
)
ORIGINAL_COMMIT = '9e8324434a3c0b2d5c53d209ad04c2d88b99f4bc'
ORIGINAL_PROTOCOL_SHA256 = (
    '85151029297cf4b4c1a07dd2a0f47b10b0663acd08ddfe25ba72b9898b1beea0'
)
ORIGINAL_FREEZE_SHA256 = (
    '98860ed4e60c427a76ac05879d800f36b65c10a310f4b2b981819fa48af767b3'
)
ORIGINAL_ANALYZER_SHA256 = (
    '0a65a27a5c424bb9dddacb5475e02d28a0999fc7cd593d4dd63ba4be06c39a46'
)
ORIGINAL_ANALYZER_TEST_SHA256 = (
    'd027f73fb07682e8cb54d46653a5bd9aa900aaeac433b1748dbdf4886c6d5034'
)
ATTEMPT_STARTED_SHA256 = (
    'b74081fd0cbd1c8d6ec5445b3b71661f40ac4d47dd77fd2d9bd3675b4cf9c3c3'
)
RUN_COMPLETE_SHA256 = (
    'ddc8350361ae9091ac47878a2c2d043897c46ef1d7722a401869d8d69e4be463'
)
RAW_MANIFEST_SHA256 = (
    '6c50c86153fbce5136ed99205ca4726f87a00ef56216f1205dba5c25d3d27cd7'
)
RAW_ARTIFACT_COUNT = 5_142
RAW_ARTIFACT_TREE_SHA256 = (
    'e7376062ce31090b349e88b91bd41700caf4e690511c15993e50f2bd0d47f770'
)
WHOLE_RUN_FILE_COUNT = 5_158
WHOLE_RUN_TREE_SHA256 = (
    '2d8125fe6d13773ba9621e527870361b6a195c516c5b4f044c7dad64c9310aaa'
)
COMPILER_FILE_COUNT = 8
COMPILER_TREE_SHA256 = (
    '9a03dcbc9d439cb9bf197941af3bbdb3e6bda067cf661b90de6d7eab1f4d87eb'
)
IMPORT_PROVENANCE_SHA256 = (
    '64a5538499e5b06e29cb506a2b08585bb002b3766bd1be210d1a568b9ec5110e'
)
PROTOBUF_PROVENANCE_SHA256 = (
    '2498a940f6ee15e54e72e8f51587d4c42ffc1b49851873c31ad09085315d0ba8'
)
TARGET_ELIGIBILITY_SHA256 = (
    'b216692d8028faab09b5f6590e3e68d9c8805d3c715ddedd99e8956019cedcf0'
)
DEVICE_PREFLIGHT_SHA256 = (
    'b983c7f4910ef4fc5f68bc72486552063f4497f90bba64497eb29a09d3d1809d'
)
EMPTY_SHA256 = (
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
)

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
_UPSTREAM_ROOT = _REPO_ROOT.parent / 'alphagenome'
_ORIGINAL_ANALYZER_PATH = _HERE / 'analyze_encoder_skip_localization_v3_3.py'
_ORIGINAL_TEST_PATH = _HERE / 'analyze_encoder_skip_localization_v3_3_test.py'
_FREEZE_PATH = _HERE / 'encoder_skip_factorial_v3_3_freeze.json'
_AMENDMENT_PATH = (
    _HERE / 'v3_wider_mechanism/encoder_skip_analysis_amendment_v3_3_1.md'
)
_TEST_PATH = _HERE / 'analyze_encoder_skip_localization_v3_3_1_test.py'
_RUN_DIR = (
    _HERE / 'results/v3_3_development_encoder_skip_factorial_one_shot'
)
_ANALYSIS_DIR = (
    _HERE / 'results/v3_3_development_encoder_skip_factorial_analysis'
)
_ATTEMPT_DIR = (
    _HERE
    / 'results/v3_3_development_encoder_skip_factorial_analysis_v3_3_1_attempt'
)

_BOOTSTRAP_ROLES = {
    'dependency_pb2', 'dependency_proto', 'generated_pb2', 'generated_pyi',
    'source_proto', 'tensor_pb2', 'tensor_proto',
}
_NAMED_NODE_BY_ROLE = {
    'source_proto': 'source_proto',
    'dependency_pb2': 'dependency_pb2',
    'dependency_proto': 'dependency_proto',
    'tensor_pb2': 'tensor_pb2',
    'tensor_proto': 'tensor_proto',
}
_GENERATED_OUTPUTS = {
    str(
        (_REPO_ROOT / 'src/alphagenome_research/protos/calibration_scores_pb2.py')
        .resolve()
    ): {
        'sha256': '4673289dd481fd8c4976f602ab36b07646304107e352e3e6d27b2abe4f9e9ebc',
        'size_bytes': 2_794,
    },
    str(
        (_REPO_ROOT / 'src/alphagenome_research/protos/calibration_scores_pb2.pyi')
        .resolve()
    ): {
        'sha256': '329dc390abeb187084fff28fbe6cb6d9868aa8867326bf53f9a52d4c83f527f9',
        'size_bytes': 1_815,
    },
}
_GENERATED_ROLE_PATH = {
    'generated_pb2': next(
        path for path in _GENERATED_OUTPUTS if path.endswith('_pb2.py')
    ),
    'generated_pyi': next(
        path for path in _GENERATED_OUTPUTS if path.endswith('_pb2.pyi')
    ),
}
_GENERATED_EXCEPTION = [
    '?? src/alphagenome_research/protos/calibration_scores_pb2.py',
    '?? src/alphagenome_research/protos/calibration_scores_pb2.pyi',
]
_TOP_LEVEL_HASHES = {
    'ATTEMPT_STARTED.json': ATTEMPT_STARTED_SHA256,
    'IMPORT_PROVENANCE.json': IMPORT_PROVENANCE_SHA256,
    'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json': IMPORT_PROVENANCE_SHA256,
    'IMPORT_PROVENANCE_PRE_MODEL.json': IMPORT_PROVENANCE_SHA256,
    'PROTOBUF_PROVENANCE.json': PROTOBUF_PROVENANCE_SHA256,
    'RAW_MANIFEST.json': RAW_MANIFEST_SHA256,
    'RUN_COMPLETE.json': RUN_COMPLETE_SHA256,
    'TARGET_ELIGIBILITY.json': TARGET_ELIGIBILITY_SHA256,
}


class AmendmentError(RuntimeError):
  """Raised when the prospective analyzer-only contract is violated."""


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
  return (
      isinstance(value, str) and len(value) == 64
      and all(character in '0123456789abcdef' for character in value)
  )


def _assert_no_model_imports(label: str) -> None:
  forbidden = sorted(
      name for name in sys.modules
      if name in {'jax', 'jaxlib', 'alphagenome'}
      or name.startswith(('jax.', 'jaxlib.', 'alphagenome.'))
      or name.startswith('alphagenome_research.model')
  )
  if forbidden:
    raise AmendmentError(f'{label} imported forbidden model/JAX modules: {forbidden}.')


def _strict_regular(path: Path, label: str) -> None:
  try:
    mode = path.lstat().st_mode
  except OSError as error:
    raise AmendmentError(f'{label} cannot be statted: {path}.') from error
  if path.is_symlink() or not stat.S_ISREG(mode):
    raise AmendmentError(f'{label} is symlinked or not a regular file: {path}.')


def _strict_directory(path: Path, label: str) -> None:
  try:
    mode = path.lstat().st_mode
  except OSError as error:
    raise AmendmentError(f'{label} cannot be statted: {path}.') from error
  if path.is_symlink() or not stat.S_ISDIR(mode):
    raise AmendmentError(f'{label} is symlinked or not a directory: {path}.')


def _canonical_path(value: Any, label: str) -> Path:
  if not isinstance(value, str) or not value or not Path(value).is_absolute():
    raise AmendmentError(f'{label} path is not an absolute canonical string.')
  path = Path(value)
  if str(path.resolve()) != value:
    raise AmendmentError(f'{label} path is not canonical.')
  lowered = tuple(part.lower() for part in path.parts)
  if any('confirm' in part for part in lowered):
    raise AmendmentError(f'{label} uses a confirmation-named path.')
  return path


def _require_contained(path: Path, root: Path, label: str) -> None:
  try:
    path.relative_to(root.resolve())
  except ValueError as error:
    raise AmendmentError(f'{label} escaped its frozen repository root.') from error


def _read_object(path: Path, label: str) -> dict[str, Any]:
  _strict_regular(path, label)
  try:
    value = json.loads(path.read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError) as error:
    raise AmendmentError(f'{label} is not readable JSON.') from error
  if not isinstance(value, dict):
    raise AmendmentError(f'{label} must be a JSON object.')
  return value


def _json_canonical_object(value: Mapping[str, Any]) -> dict[str, Any]:
  """Returns the exact JSON-domain form that append-only records will store."""
  encoded = json.dumps(value, sort_keys=True, allow_nan=False)
  decoded = json.loads(encoded)
  if not isinstance(decoded, dict):
    raise AmendmentError('Expected a JSON-domain object.')
  return decoded


def _load_original():
  _assert_no_model_imports('v3.3.1 pre-import process')
  _strict_regular(_ORIGINAL_ANALYZER_PATH, 'Frozen v3.3 analyzer')
  if _sha256(_ORIGINAL_ANALYZER_PATH) != ORIGINAL_ANALYZER_SHA256:
    raise AmendmentError('Frozen v3.3 analyzer bytes changed before import.')
  specification = importlib.util.spec_from_file_location(
      '_opensplice_frozen_encoder_skip_analyzer_v3_3', _ORIGINAL_ANALYZER_PATH
  )
  if specification is None or specification.loader is None:
    raise AmendmentError('Cannot load the frozen v3.3 analyzer.')
  module = importlib.util.module_from_spec(specification)
  specification.loader.exec_module(module)
  _assert_no_model_imports('v3.3.1 post-import process')
  return module


_v33 = _load_original()


def _artifact_row(value: Any, label: str) -> dict[str, Any]:
  if not isinstance(value, Mapping) or set(value) != {
      'path', 'sha256', 'size_bytes'
  }:
    raise AmendmentError(f'{label} schema is not exactly path/hash/size.')
  path = _canonical_path(value['path'], label)
  digest, size = value['sha256'], value['size_bytes']
  if not _is_sha256(digest) or isinstance(size, bool) or not isinstance(size, int) or size < 0:
    raise AmendmentError(f'{label} hash/size is malformed.')
  _strict_regular(path, label)
  allowed_root = _REPO_ROOT if path.is_relative_to(_REPO_ROOT) else _UPSTREAM_ROOT
  _require_contained(path, allowed_root, label)
  if path.stat().st_size != size or _sha256(path) != digest:
    raise AmendmentError(f'{label} current bytes differ from the attestation.')
  return {'path': str(path), 'sha256': digest, 'size_bytes': size}


def _validate_generated_bindings(
    start: Mapping[str, Any], *, freeze: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Validates exact seven-role evidence and returns a two-output START copy."""
  freeze_binding = start.get('freeze') if freeze is None else freeze
  if not isinstance(freeze_binding, Mapping):
    raise AmendmentError('START freeze binding is absent.')
  protobuf = freeze_binding.get('protobuf_binding')
  if not isinstance(protobuf, Mapping):
    raise AmendmentError('Frozen protobuf binding is absent.')
  bootstrap = start.get('same_process_pre_import_bootstrap')
  generated = bootstrap.get('generated_bindings') if isinstance(bootstrap, Mapping) else None
  if not isinstance(generated, Mapping) or set(generated) != {
      'pre_import_gate', 'historical_generator_argv',
      'exact_regeneration_claim', 'generated_artifact_exception', 'artifacts',
      'embedded_header', 'protobuf_runtime_version',
  }:
    raise AmendmentError('Pre-import generated-binding schema changed.')
  if (
      generated.get('pre_import_gate') is not True
      or generated.get('historical_generator_argv') != 'unknown'
      or generated.get('exact_regeneration_claim') is not False
      or generated.get('generated_artifact_exception') != _GENERATED_EXCEPTION
      or generated.get('embedded_header') != protobuf.get('embedded_generated_header')
      or generated.get('protobuf_runtime_version')
      != protobuf.get('protobuf_runtime_version')
  ):
    raise AmendmentError('Pre-import generated-binding claims changed.')
  outputs = protobuf.get('generated_outputs')
  if outputs != _GENERATED_OUTPUTS:
    raise AmendmentError('Frozen generated-output mapping changed.')
  artifacts = generated.get('artifacts')
  if not isinstance(artifacts, Mapping) or set(artifacts) != _BOOTSTRAP_ROLES:
    raise AmendmentError('Pre-import artifact roles are not the exact seven-role set.')
  rows = {
      role: _artifact_row(value, f'generated_bindings.artifacts.{role}')
      for role, value in artifacts.items()
  }
  paths = [row['path'] for row in rows.values()]
  if len(paths) != len(set(paths)):
    raise AmendmentError('Pre-import artifact roles contain a duplicated path.')
  for role, expected_path in _GENERATED_ROLE_PATH.items():
    row = rows[role]
    if row['path'] != expected_path or {
        'sha256': row['sha256'], 'size_bytes': row['size_bytes']
    } != _GENERATED_OUTPUTS[expected_path]:
      raise AmendmentError(f'{role} does not match its exact frozen output.')
  for role, node_name in _NAMED_NODE_BY_ROLE.items():
    row = rows[role]
    node = protobuf.get(node_name)
    if not isinstance(node, Mapping) or set(node) not in (
        {'path', 'sha256'}, {'path', 'sha256', 'size_bytes'}
    ):
      raise AmendmentError(f'Frozen protobuf node {node_name} is malformed.')
    expected = {'path': row['path'], 'sha256': row['sha256']}
    if {key: node.get(key) for key in expected} != expected:
      raise AmendmentError(f'{role} is misclassified against {node_name}.')
    if 'size_bytes' in node and node['size_bytes'] != row['size_bytes']:
      raise AmendmentError(f'{role} size differs from {node_name}.')
  imported = protobuf.get('imported_dependency_pb2')
  if not isinstance(imported, Mapping) or (
      imported.get('alphagenome.protos.dna_model_pb2') != rows['dependency_pb2']
      or imported.get('alphagenome.protos.tensor_pb2') != rows['tensor_pb2']
      or protobuf.get('imported_pb2') != rows['generated_pb2']
  ):
    raise AmendmentError('Imported protobuf bindings differ from seven-role evidence.')

  normalized = copy.deepcopy(dict(start))
  normalized_artifacts = normalized[
      'same_process_pre_import_bootstrap'
  ]['generated_bindings']['artifacts']
  normalized_artifacts.clear()
  normalized_artifacts.update({
      role: copy.deepcopy(rows[role])
      for role in ('generated_pb2', 'generated_pyi')
  })
  audit = {
      'repair': AMENDMENT_REASON,
      'bootstrap_role_order': sorted(rows),
      'bootstrap_artifact_count': 7,
      'generated_output_roles': ['generated_pb2', 'generated_pyi'],
      'generated_output_count': 2,
      'source_dependency_roles': sorted(_NAMED_NODE_BY_ROLE),
      'all_current_bytes_verified': True,
      'scientific_gate_or_estimand_changed': False,
  }
  return normalized, audit


@contextmanager
def _normalized_start_reader(
    run_dir: Path, normalized_start: Mapping[str, Any]
) -> Iterator[None]:
  original_reader = _v33._read_json  # pylint: disable=protected-access
  start_path = (run_dir / 'ATTEMPT_STARTED.json').resolve()

  def reader(path: Path) -> dict[str, Any]:
    if path.resolve() == start_path:
      return copy.deepcopy(dict(normalized_start))
    return original_reader(path)

  _v33._read_json = reader  # pylint: disable=protected-access
  try:
    yield
  finally:
    _v33._read_json = original_reader  # pylint: disable=protected-access


def _tree_digest(paths: Sequence[Path], root: Path) -> str:
  digest = hashlib.sha256()
  root = root.resolve()
  for path in sorted(path.resolve() for path in paths):
    relative = path.relative_to(root)
    digest.update(str(relative).encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(_sha256(path)))
  return digest.hexdigest()


def _validate_immutable_run(run_dir: Path) -> dict[str, Any]:
  run_dir = run_dir.resolve()
  if run_dir != _RUN_DIR.resolve():
    raise AmendmentError('Run directory differs from the immutable v3.3 run.')
  _strict_directory(run_dir, 'Immutable v3.3 run directory')
  entries = sorted(run_dir.rglob('*'))
  files: list[Path] = []
  directories: set[Path] = set()
  for path in entries:
    mode = path.lstat().st_mode
    if path.is_symlink():
      raise AmendmentError(f'Immutable run contains a symlink: {path}.')
    if stat.S_ISREG(mode):
      files.append(path.resolve())
    elif stat.S_ISDIR(mode):
      directories.add(path.resolve())
    else:
      raise AmendmentError(f'Immutable run contains a special entry: {path}.')
  expected_directories = {
      parent
      for path in files
      for parent in path.parents
      if parent != run_dir and run_dir in parent.parents
  }
  if directories != expected_directories:
    raise AmendmentError('Immutable run contains an empty or unexpected directory.')
  if len(files) != WHOLE_RUN_FILE_COUNT or _tree_digest(files, run_dir) != WHOLE_RUN_TREE_SHA256:
    raise AmendmentError('Whole immutable run file count/tree changed.')
  for filename, expected_sha in _TOP_LEVEL_HASHES.items():
    path = run_dir / filename
    _strict_regular(path, filename)
    if _sha256(path) != expected_sha:
      raise AmendmentError(f'Immutable top-level artifact changed: {filename}.')

  manifest = _read_object(run_dir / 'RAW_MANIFEST.json', 'RAW_MANIFEST')
  if set(manifest) != {
      'artifact_count', 'artifact_sha256', 'artifact_tree_sha256'
  } or (
      manifest.get('artifact_count') != RAW_ARTIFACT_COUNT
      or manifest.get('artifact_tree_sha256') != RAW_ARTIFACT_TREE_SHA256
      or not isinstance(manifest.get('artifact_sha256'), Mapping)
      or len(manifest['artifact_sha256']) != RAW_ARTIFACT_COUNT
  ):
    raise AmendmentError('RAW_MANIFEST count/tree/schema changed.')
  raw_paths: list[Path] = []
  for relative, expected_sha in manifest['artifact_sha256'].items():
    if (
        not isinstance(relative, str) or Path(relative).is_absolute()
        or '..' in Path(relative).parts or not relative.startswith('raw/')
        or not _is_sha256(expected_sha)
    ):
      raise AmendmentError('RAW_MANIFEST contains a malformed path/hash row.')
    path = (run_dir / relative).resolve()
    _require_contained(path, run_dir / 'raw', 'RAW_MANIFEST artifact')
    _strict_regular(path, 'RAW_MANIFEST artifact')
    if _sha256(path) != expected_sha:
      raise AmendmentError(f'Raw artifact bytes changed: {relative}.')
    raw_paths.append(path)
  if _tree_digest(raw_paths, run_dir) != RAW_ARTIFACT_TREE_SHA256:
    raise AmendmentError('Raw artifact tree changed.')
  compiler_paths = sorted(
      path.resolve() for path in (run_dir / 'compiler').rglob('*')
      if path.is_file() and not path.is_symlink()
  )
  if (
      len(compiler_paths) != COMPILER_FILE_COUNT
      or _tree_digest(compiler_paths, run_dir) != COMPILER_TREE_SHA256
  ):
    raise AmendmentError('Compiler file count/tree changed.')
  expected_files = {
      (run_dir / name).resolve() for name in _TOP_LEVEL_HASHES
  } | set(raw_paths) | set(compiler_paths)
  if set(files) != expected_files:
    raise AmendmentError('Immutable run membership changed.')

  complete = _read_object(run_dir / 'RUN_COMPLETE.json', 'RUN_COMPLETE')
  expected_complete = {
      'status': 'controlled_stop', 'stop_reason': 'ood_tooling_failure',
      'identity_count': 20, 'identity_invalid_count': 0,
      'eligible_effect_count': 12, 'coalition_record_count': 5_120,
      'coalition_invalid_count': 0, 'ood_anchor_record_count': 2,
      'ood_invalid_count': 1, 'scientific_record_count': 5_142,
      'model_apply_count': 10_288, 'compile_count': 2,
      'confirmation_model_calls': 0, 'all_effects_target_eligible': True,
      'all_neutrals_retained': True, 'id0_noop_all20': True,
      'id255_closure_all20': True,
  }
  for key, expected in expected_complete.items():
    if complete.get(key) != expected:
      raise AmendmentError(f'Controlled-stop field changed: {key}.')
  if complete.get('raw_manifest') != manifest:
    raise AmendmentError('RUN_COMPLETE no longer embeds the exact RAW_MANIFEST.')
  return {
      'whole_run_file_count': len(files),
      'whole_run_tree_sha256': WHOLE_RUN_TREE_SHA256,
      'raw_artifact_count': len(raw_paths),
      'raw_artifact_tree_sha256': RAW_ARTIFACT_TREE_SHA256,
      'compiler_file_count': len(compiler_paths),
      'compiler_tree_sha256': COMPILER_TREE_SHA256,
      'run_complete_status': 'controlled_stop',
      'stop_reason': 'ood_tooling_failure',
  }


def _validate_original_bundle(bundle_root: Path) -> dict[str, Any]:
  bundle_root = bundle_root.resolve()
  _strict_regular(_FREEZE_PATH, 'Frozen v3.3 freeze')
  if _sha256(_FREEZE_PATH) != ORIGINAL_FREEZE_SHA256:
    raise AmendmentError('Original v3.3 freeze bytes changed.')
  freeze = _read_object(_FREEZE_PATH, 'Frozen v3.3 freeze')
  if freeze.get('protocol_sha256') != ORIGINAL_PROTOCOL_SHA256:
    raise AmendmentError('Original v3.3 protocol binding changed.')
  inventory = freeze.get('file_sha256')
  if not isinstance(inventory, Mapping) or len(inventory) != 61:
    raise AmendmentError('Original v3.3 bundle is not the exact 61-file inventory.')
  if (
      inventory.get(str(_ORIGINAL_ANALYZER_PATH.relative_to(bundle_root)))
      != ORIGINAL_ANALYZER_SHA256
      or inventory.get(str(_ORIGINAL_TEST_PATH.relative_to(bundle_root)))
      != ORIGINAL_ANALYZER_TEST_SHA256
  ):
    raise AmendmentError('Original analyzer/test are not freeze-bound.')
  for relative, expected_sha in inventory.items():
    if (
        not isinstance(relative, str) or Path(relative).is_absolute()
        or '..' in Path(relative).parts or not _is_sha256(expected_sha)
    ):
      raise AmendmentError('Original bundle inventory is malformed.')
    path = (bundle_root / relative).resolve()
    _require_contained(path, bundle_root, 'Original bundle file')
    _strict_regular(path, 'Original bundle file')
    if _sha256(path) != expected_sha:
      raise AmendmentError(f'Original bundle bytes changed: {relative}.')
    historical = subprocess.check_output(
        ('git', '-C', str(bundle_root), 'show', f'{ORIGINAL_COMMIT}:{relative}')
    )
    if hashlib.sha256(historical).hexdigest() != expected_sha:
      raise AmendmentError(f'Original commit bytes differ: {relative}.')
  return {
      'original_commit': ORIGINAL_COMMIT,
      'freeze_sha256': ORIGINAL_FREEZE_SHA256,
      'bundle_file_count': len(inventory),
      'analyzer_sha256': ORIGINAL_ANALYZER_SHA256,
      'analyzer_test_sha256': ORIGINAL_ANALYZER_TEST_SHA256,
  }


def _assert_global_tracked_clean(bundle_root: Path) -> str:
  subprocess.run(
      ('git', '-C', str(bundle_root), 'diff', '--binary', '--exit-code', 'HEAD', '--'),
      check=True, capture_output=True,
  )
  head = subprocess.check_output(
      ('git', '-C', str(bundle_root), 'rev-parse', 'HEAD'), text=True
  ).strip()
  if len(head) != 40 or any(character not in '0123456789abcdef' for character in head):
    raise AmendmentError('Current Git HEAD is malformed.')
  return head


def _validate_started_attempt(binding: Mapping[str, Any], digest: str) -> None:
  if not _is_sha256(digest):
    raise AmendmentError('v3.3.1 attempt-start digest is malformed.')
  _strict_directory(_ATTEMPT_DIR, 'v3.3.1 attempt directory')
  entries = list(_ATTEMPT_DIR.iterdir())
  if len(entries) != 1 or entries[0].name != 'ANALYSIS_ATTEMPT_STARTED.json':
    raise AmendmentError('v3.3.1 attempt has extra or terminal artifacts.')
  path = entries[0]
  _strict_regular(path, 'v3.3.1 attempt-start artifact')
  if _sha256(path) != digest:
    raise AmendmentError('v3.3.1 attempt-start bytes changed.')
  record = _read_object(path, 'v3.3.1 attempt-start artifact')
  if (
      set(record) != {
          'analysis_version', 'status', 'reason', 'started_at_unix_s',
          'run_dir', 'output_json', 'output_markdown', 'amendment_binding',
          'model_rerun_permitted', 'scientific_values_read_before_attempt',
          'confirmation_model_calls_permitted',
      }
      or record.get('analysis_version') != ANALYSIS_VERSION
      or record.get('status') != 'started_append_only_one_shot'
      or record.get('reason') != AMENDMENT_REASON
      or not isinstance(record.get('started_at_unix_s'), (int, float))
      or isinstance(record.get('started_at_unix_s'), bool)
      or record.get('run_dir') != str(_RUN_DIR.resolve())
      or record.get('output_json') != str((_ANALYSIS_DIR / 'ANALYSIS.json').resolve())
      or record.get('output_markdown') != str((_ANALYSIS_DIR / 'RESULT.md').resolve())
      or record.get('amendment_binding') != binding
      or record.get('model_rerun_permitted') is not False
      or record.get('scientific_values_read_before_attempt') is not False
      or record.get('confirmation_model_calls_permitted') != 0
  ):
    raise AmendmentError('v3.3.1 attempt-start content/binding changed.')


def _validate_amendment_preconditions(
    run_dir: Path, bundle_root: Path, *,
    expected_attempt_started_sha256: str | None = None,
) -> dict[str, Any]:
  _assert_no_model_imports('v3.3.1 precondition process')
  bundle_root = bundle_root.resolve()
  if bundle_root != _REPO_ROOT.resolve():
    raise AmendmentError('Repository root differs from the prospective contract.')
  if run_dir.resolve() != _RUN_DIR.resolve():
    raise AmendmentError('Run directory differs from the prospective contract.')
  if _ANALYSIS_DIR.exists():
    raise FileExistsError('Frozen analysis destination already exists; never overwrite.')
  if expected_attempt_started_sha256 is None and _ATTEMPT_DIR.exists():
    raise FileExistsError('The append-only v3.3.1 attempt was already consumed.')
  _strict_regular(_AMENDMENT_PATH, 'v3.3.1 amendment')
  if _sha256(_AMENDMENT_PATH) != AMENDMENT_SHA256:
    raise AmendmentError('v3.3.1 amendment bytes changed.')
  tracked_paths = (_AMENDMENT_PATH, Path(__file__).resolve(), _TEST_PATH.resolve())
  for path in tracked_paths:
    relative = str(path.relative_to(bundle_root))
    subprocess.run(
        ('git', '-C', str(bundle_root), 'ls-files', '--error-unmatch', relative),
        check=True, capture_output=True,
    )
  head = _assert_global_tracked_clean(bundle_root)
  bundle_audit = _validate_original_bundle(bundle_root)
  run_audit = _validate_immutable_run(run_dir)
  start = _read_object(run_dir / 'ATTEMPT_STARTED.json', 'ATTEMPT_STARTED')
  normalized, generated_audit = _validate_generated_bindings(start)
  cases = _v33._load_cases()  # pylint: disable=protected-access
  with _normalized_start_reader(run_dir, normalized):
    _, observed_freeze_sha, start_audit = _v33._validate_start(  # pylint: disable=protected-access
        run_dir.resolve(), bundle_root=bundle_root, cases=cases
    )
  if observed_freeze_sha != ORIGINAL_FREEZE_SHA256:
    raise AmendmentError('Frozen v3.3 START validation returned a changed freeze.')
  external = start.get('external_preflight')
  if not isinstance(external, Mapping):
    raise AmendmentError('External preflight binding is absent.')
  if (
      external.get('sha256') != DEVICE_PREFLIGHT_SHA256
      or _sha256(Path(str(external.get('path')))) != DEVICE_PREFLIGHT_SHA256
  ):
    raise AmendmentError('Device preflight artifact changed.')
  logs = external.get('logs')
  if not isinstance(logs, Mapping):
    raise AmendmentError('Device preflight log bindings are absent.')
  for stream in ('stdout', 'stderr'):
    row = logs.get(stream)
    if (
        not isinstance(row, Mapping) or row.get('sha256') != EMPTY_SHA256
        or _sha256(Path(str(row.get('path')))) != EMPTY_SHA256
    ):
      raise AmendmentError(f'Device preflight {stream} changed.')
  final_head = _assert_global_tracked_clean(bundle_root)
  if final_head != head:
    raise AmendmentError('Git HEAD changed during v3.3.1 precondition audit.')
  binding = {
      'analysis_version': ANALYSIS_VERSION,
      'git_head': head,
      'tracked_head_clean': True,
      'file_sha256': {
          str(path.relative_to(bundle_root)): _sha256(path)
          for path in tracked_paths
      },
      'amendment_sha256': AMENDMENT_SHA256,
      'original_bundle': bundle_audit,
      'immutable_run': run_audit,
      'generated_binding_audit': generated_audit,
      'original_start_provenance_audit': start_audit,
      'original_analysis_output_absent': True,
      'model_rerun_permitted': False,
      'scientific_gate_or_estimand_changed': False,
  }
  binding = _json_canonical_object(binding)
  if expected_attempt_started_sha256 is not None:
    _validate_started_attempt(binding, expected_attempt_started_sha256)
  _assert_no_model_imports('v3.3.1 completed precondition process')
  return binding


def _enforce_controlled_stop(result: Mapping[str, Any]) -> None:
  controlled = result.get('controlled_stop')
  hash_tree = result.get('hash_tree')
  if (
      result.get('analysis_version') != _v33.ANALYSIS_VERSION
      or Path(str(result.get('analysis_dir'))).resolve() != _ANALYSIS_DIR.resolve()
      or result.get('decision') != 'controlled_stop_ood_tooling_failure'
      or result.get('nomination') is not None
      or result.get('resolution_analysis') is not None
      or not isinstance(hash_tree, Mapping)
      or hash_tree.get('raw_artifact_count') != RAW_ARTIFACT_COUNT
      or hash_tree.get('raw_artifact_tree_sha256') != RAW_ARTIFACT_TREE_SHA256
      or hash_tree.get('raw_manifest_sha256') != RAW_MANIFEST_SHA256
      or hash_tree.get('run_complete_sha256') != RUN_COMPLETE_SHA256
      or not isinstance(controlled, Mapping)
      or controlled.get('reason') != 'ood_tooling_failure'
      or controlled.get('identity_count') != 20
      or controlled.get('identity_invalid_count') != 0
      or controlled.get('eligible_effect_count') != 12
      or controlled.get('coalition_record_count') != 5_120
      or controlled.get('coalition_invalid_count') != 0
      or controlled.get('ood_anchor_record_count') != 2
      or controlled.get('ood_invalid_count') != 1
      or controlled.get('shapley_computed') is not False
      or controlled.get('nomination_performed') is not False
  ):
    raise AmendmentError('Delegated result exceeds the frozen controlled-stop claim.')


def analyze(
    run_dir: Path, *, bundle_root: Path = _REPO_ROOT,
    amendment_binding: Mapping[str, Any] | None = None,
    attempt_started_sha256: str | None = None,
) -> dict[str, Any]:
  _assert_no_model_imports('v3.3.1 pre-analysis process')
  verified = _validate_amendment_preconditions(
      run_dir, bundle_root,
      expected_attempt_started_sha256=attempt_started_sha256,
  )
  if amendment_binding != verified:
    raise AmendmentError('v3.3.1 amendment binding is absent or changed.')
  if not _is_sha256(attempt_started_sha256):
    raise AmendmentError('v3.3.1 append-only attempt binding is absent.')
  start = _read_object(run_dir / 'ATTEMPT_STARTED.json', 'ATTEMPT_STARTED')
  normalized, generated_audit = _validate_generated_bindings(start)
  with _normalized_start_reader(run_dir, normalized):
    result = _v33.analyze(run_dir, bundle_root=bundle_root)
  _assert_no_model_imports('v3.3.1 post-analysis process')
  _enforce_controlled_stop(result)
  result['analyzer_amendment'] = {
      'model_run_analysis_version': _v33.ANALYSIS_VERSION,
      'offline_analysis_version': ANALYSIS_VERSION,
      'amendment_sha256': AMENDMENT_SHA256,
      'amendment_binding': verified,
      'analysis_attempt_started_sha256': attempt_started_sha256,
      'generated_binding_audit': generated_audit,
      'preserved_original_failure': {
          'analysis_version': _v33.ANALYSIS_VERSION,
          'reason': 'seven_bootstrap_roles_compared_to_two_generated_outputs',
          'scientific_record_values_read': False,
          'scientific_output_written': False,
      },
      'model_rerun_permitted': False,
      'scientific_gate_or_estimand_changed': False,
      'controlled_stop_preserved': True,
      'shapley_computed': False,
      'nomination_performed': False,
  }
  return result


def _write_new_json(path: Path, value: Mapping[str, Any]) -> str:
  data = (
      json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
  ).encode('utf-8')
  descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
  try:
    with os.fdopen(descriptor, 'wb') as handle:
      handle.write(data)
      handle.flush()
      os.fsync(handle.fileno())
  except BaseException:
    path.unlink(missing_ok=True)
    raise
  return hashlib.sha256(data).hexdigest()


def _start_attempt(
    run_dir: Path, output_json: Path, output_markdown: Path,
    amendment_binding: Mapping[str, Any], *, attempt_dir: Path = _ATTEMPT_DIR,
) -> str:
  try:
    attempt_dir.mkdir(mode=0o755, parents=False, exist_ok=False)
  except FileExistsError as error:
    raise FileExistsError(
        'The append-only v3.3.1 corrected-analysis attempt was already consumed.'
    ) from error
  return _write_new_json(attempt_dir / 'ANALYSIS_ATTEMPT_STARTED.json', {
      'analysis_version': ANALYSIS_VERSION,
      'status': 'started_append_only_one_shot',
      'reason': AMENDMENT_REASON,
      'started_at_unix_s': time.time(),
      'run_dir': str(run_dir.resolve()),
      'output_json': str(output_json.resolve()),
      'output_markdown': str(output_markdown.resolve()),
      'amendment_binding': dict(amendment_binding),
      'model_rerun_permitted': False,
      'scientific_values_read_before_attempt': False,
      'confirmation_model_calls_permitted': 0,
  })


def _persist_terminal(
    filename: str, value: Mapping[str, Any], *, started_sha256: str
) -> None:
  _write_new_json(_ATTEMPT_DIR / filename, {
      'analysis_version': ANALYSIS_VERSION,
      'attempt_started_sha256': started_sha256,
      'recorded_at_unix_s': time.time(),
      **value,
  })


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run-dir', type=Path, required=True)
  parser.add_argument('--output-json', type=Path, required=True)
  parser.add_argument('--output-markdown', type=Path, required=True)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if args.run_dir.resolve() != _RUN_DIR.resolve():
    raise AmendmentError('CLI run directory differs from the immutable v3.3 run.')
  if args.output_json.resolve() != _ANALYSIS_DIR.resolve() / 'ANALYSIS.json':
    raise AmendmentError('JSON output differs from the frozen analysis destination.')
  if args.output_markdown.resolve() != _ANALYSIS_DIR.resolve() / 'RESULT.md':
    raise AmendmentError('Markdown output differs from the frozen analysis destination.')
  binding = _validate_amendment_preconditions(args.run_dir, _REPO_ROOT)
  started_sha = _start_attempt(
      args.run_dir, args.output_json, args.output_markdown, binding
  )
  try:
    result = analyze(
        args.run_dir, amendment_binding=binding,
        attempt_started_sha256=started_sha,
    )
    _v33._write_atomic(  # pylint: disable=protected-access
        args.output_json,
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n',
    )
    markdown = _v33.render_markdown(result) + (
        '\n\n## Analyzer-only amendment\n\n'
        'The GPU run used frozen v3.3. Offline validation used the prospective '
        'v3.3.1 seven-role/two-generated-output provenance repair. The result '
        'remains an OOD tooling controlled stop; no Shapley decomposition, '
        'resolution nomination, spatial trigger, or mechanistic claim was made.\n'
    )
    _v33._write_atomic(args.output_markdown, markdown)  # pylint: disable=protected-access
    _persist_terminal('ANALYSIS_COMPLETE.json', {
        'status': 'complete_controlled_stop_audited',
        'analysis_json_sha256': _sha256(args.output_json),
        'analysis_markdown_sha256': _sha256(args.output_markdown),
        'decision': result['decision'],
        'shapley_computed': False,
        'nomination_performed': False,
    }, started_sha256=started_sha)
  except BaseException as error:
    try:
      _persist_terminal('ANALYSIS_FAILURE.json', {
          'status': 'failed_consumed_no_retry',
          'failure': {
              'type': type(error).__name__, 'message': str(error),
              'traceback': traceback.format_exc(),
          },
          'analysis_json_exists': args.output_json.exists(),
          'analysis_markdown_exists': args.output_markdown.exists(),
      }, started_sha256=started_sha)
    except BaseException:
      pass
    raise
  print(args.output_json.resolve())


if __name__ == '__main__':
  main()
