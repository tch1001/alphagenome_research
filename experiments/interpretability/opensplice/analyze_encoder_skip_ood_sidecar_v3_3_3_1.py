#!/usr/bin/env python3
"""CPU-only structural archive for the consumed v3.3.3 compiler-gate stop.

This analyzer implements only the prospective v3.3.3.1 representation audit.
It never imports JAX or AlphaGenome, never reads a scientific raw record, and
never invokes or modifies the frozen v3.3.3 analyzer entry points.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import stat
import subprocess
import sys
import time
import traceback
from types import ModuleType
from typing import Any, Iterable, Mapping


ANALYSIS_VERSION = 'opensplice-encoder-skip-ood-sidecar-analysis-v3.3.3.1'
AMENDMENT_COMMIT = 'd2a013944a399ddac59a023d7d84ea5a7c23e9f4'
AMENDMENT_SHA256 = (
    '4d2957d144e56e58c5b2058076bbcdb7f1495f3172e1b8829a0affa10a0ea4a9'
)
MODEL_RUN_COMMIT = '228083b931dbc62d4a283e68df01011f5ef4bff9'
ORIGINAL_PROTOCOL_COMMIT = '783a7d0dfbd5f26e22152d1201dacf82f2b01d15'
ORIGINAL_PROTOCOL_SHA256 = (
    'c9b00398296e683ac6e1c321fd8c4302f96b2e62bb23828e8b5ef2fe9de3f70b'
)
ORIGINAL_FREEZE_SHA256 = (
    '0e4c16a306f734e016c64509a3b7f0d76f26baf399ee0b1d41c6fb073203741b'
)
ORIGINAL_ANALYZER_SHA256 = (
    '5cf50aa7a9403df5d8f5555b1d2fc50ab2feb69e0508c37e64bddd5a8a1e3783'
)
ORIGINAL_ANALYZER_TEST_SHA256 = (
    'fe174d0546e97e1ac1de0ad03204df4e77cf8baa0e7edb27791d6b46ed6ed58f'
)
PROGRAM_SIGNATURES_SHA256 = (
    'd8f95fb9d3637fd263cc3da0f6a33409d5fb2a5cf37e348723ecc89b3224c300'
)
PROGRAM_SIGNATURES_CANONICAL_SIZE = 2_877
EMPTY_SHA256 = (
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
)
CONFIRMATION_DISCLOSURE = (
    'Later-exon metadata/labels were exposed after protocol freeze; '
    'no later-exon model outputs, activations, or interventions are used.'
)

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
_AMENDMENT_PATH = (
    _HERE / 'v3_wider_mechanism'
    / 'encoder_skip_ood_sidecar_analysis_amendment_v3_3_3_1.md'
)
_FREEZE_PATH = _HERE / 'encoder_skip_ood_sidecar_analysis_v3_3_3_1_freeze.json'
_RUN_DIR = _HERE / 'results/v3_3_3_development_ood_sidecar_one_shot'
_PREFLIGHT_DIR = _HERE / 'results/v3_3_3_device_preflight'
_MODEL_CACHE_DIR = _HERE / 'results/v3_3_3_model_kernel_cache'
_OLD_ANALYZER = _HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_3.py'
_OLD_ANALYZER_TEST = _HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_3_test.py'
_OLD_FREEZE = _HERE / 'encoder_skip_ood_sidecar_v3_3_3_freeze.json'
_OLD_ATTEMPT_DIR = (
    _HERE / 'results/v3_3_3_development_ood_sidecar_analysis_attempt'
)
_OLD_ANALYSIS_DIR = _HERE / 'results/v3_3_3_development_ood_sidecar_analysis'
_ATTEMPT_DIR = (
    _HERE
    / 'results/v3_3_3_development_ood_sidecar_analysis_v3_3_3_1_attempt'
)
_ANALYSIS_DIR = (
    _HERE / 'results/v3_3_3_development_ood_sidecar_analysis_v3_3_3_1'
)
_ATTEMPT_TOKEN = object()

_RUN_FILES = {
    'ATTEMPT_STARTED.json': (871_020, 'e5f7c33f2e8c82af51ed98a3884d7df83e1828e92e322df8aa8a054ec7464c65'),
    'IMPORT_PROVENANCE.json': (41_572, 'aa5072c505ebe54d0a7812a7fa3e6bda249a74c17d2b90df3f235c2f4cd6bb4b'),
    'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json': (41_572, 'aa5072c505ebe54d0a7812a7fa3e6bda249a74c17d2b90df3f235c2f4cd6bb4b'),
    'IMPORT_PROVENANCE_PRE_MODEL.json': (41_572, 'aa5072c505ebe54d0a7812a7fa3e6bda249a74c17d2b90df3f235c2f4cd6bb4b'),
    'PROTOBUF_PROVENANCE.json': (3_339, '2498a940f6ee15e54e72e8f51587d4c42ffc1b49851873c31ad09085315d0ba8'),
    'RAW_MANIFEST.json': (145, 'fadbff5ab512dea3d25edb39dc9a91ffe25473b73f1e33fd53f97b60fc8436fd'),
    'RUN_COMPLETE.json': (227_159, '43e0ff055e9f7fa4032a75120c551a2b5762e4fbd85119e80e3694f8b9f54bba'),
    'compiler/eight_row/COMPILER_PROVENANCE.json': (102_245, 'ae07b0f10784ea3c6dd26d2b87eb718c5e28d3834112ae4f0566d1c4fb7e3125'),
    'compiler/eight_row/graph.compiled.hlo.txt': (16_603_075, 'f0fe2fa0b7e8326390c8f2ed38ce52ef6c64c355bc462450d85d3b2f040645f4'),
    'compiler/eight_row/graph.pre_backend.hlo.txt': (1_829_833, '675b9cf26fe8f59a1d138e856ee84f48edb20d8c896ac9e539b6b064dcef2750'),
    'compiler/eight_row/graph.stablehlo.mlir': (3_196_162, '69dbf2a054cf89e56a9000dc0c04e5cd4ba425f016885ddc127d22c01661bddd'),
}
_RUN_TREE_SHA256 = 'bb13aa4de212c3896781401374057bc0cdfc0c7527772cc36b08b57c70451805'
_COMPILER_TREE_SHA256 = '7ee5ad1bb94ecbd97606fcccae3abcad6b0ebec74dd9f983d81b4fc179142ef0'
_PREFLIGHT_FILES = {
    '.allocation.lock': (0, EMPTY_SHA256),
    '.preflight_0000.reserved': (0, EMPTY_SHA256),
    'preflight_0000.json': (704_213, '79e2c9937025830b309854cff4f5c93c607b7574fb44a9d51f45564b14246224'),
    'preflight_0000.stderr.log': (0, EMPTY_SHA256),
    'preflight_0000.stdout.log': (0, EMPTY_SHA256),
}
_PREFLIGHT_TREE_SHA256 = 'f2bae99e3b0a59a50419e0507146e26f4eea1c67f2595ddccec4e8d5aef7a0e1'
_MODEL_CACHE_FILES = {
    'xdg/matplotlib/fontlist-v3.11.0.json': (
        163_240,
        'a777469f8f54be8cc9107788bce2b3cd23709aa317114392cdc365b3fb127125',
    )
}
_MODEL_CACHE_TREE_SHA256 = 'a1dafb75097282b3d28174e4bb6c79a81d2f1c8c25cd8c64a1ea84642ba7f43a'


class AnalysisError(RuntimeError):
  """A fail-closed structural audit error."""


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
  return json.dumps(
      value, sort_keys=True, separators=(',', ':'), ensure_ascii=False
  ).encode('utf-8')


def _canonical_sha(value: Any) -> str:
  return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
  return (
      isinstance(value, str) and len(value) == 64
      and all(character in '0123456789abcdef' for character in value)
  )


def _assert_cpu_only(label: str) -> None:
  forbidden = sorted(
      name for name in sys.modules
      if name == 'jax' or name.startswith('jax.')
      or name == 'jaxlib' or name.startswith('jaxlib.')
      or name == 'alphagenome' or name.startswith('alphagenome.')
      or name == 'alphagenome_research.model'
      or name.startswith('alphagenome_research.model.')
  )
  if forbidden:
    raise AnalysisError(f'{label}: forbidden model/JAX imports: {forbidden}.')


def _guard_path(path: Path) -> None:
  for part in path.resolve().parts:
    lowered = part.lower()
    if 'confirm' in lowered or lowered in {'eln', 'eif4a2', 'dmd'}:
      raise AnalysisError(f'Refusing confirmation path: {path}.')


def _strict_regular(path: Path, label: str) -> None:
  _guard_path(path)
  try:
    mode = path.lstat().st_mode
  except OSError as error:
    raise AnalysisError(f'{label} cannot be statted.') from error
  if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
    raise AnalysisError(f'{label} is not a regular non-symlink file.')


def _read_json(path: Path, label: str) -> dict[str, Any]:
  _strict_regular(path, label)
  try:
    value = json.loads(path.read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError) as error:
    raise AnalysisError(f'{label} is not readable JSON.') from error
  if not isinstance(value, dict):
    raise AnalysisError(f'{label} is not a JSON object.')
  return value


def _finite(value: Any, label: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise AnalysisError(f'{label} is not numeric.')
  result = float(value)
  if not math.isfinite(result):
    raise AnalysisError(f'{label} is not finite.')
  return result


def _tree_digest(paths: Iterable[Path], root: Path) -> str:
  digest = hashlib.sha256()
  for path in sorted(paths):
    digest.update(path.relative_to(root).as_posix().encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(_sha256(path)))
  return digest.hexdigest()


def _cache_tree_digest(files: Iterable[Path], directories: Iterable[Path], root: Path) -> str:
  digest = hashlib.sha256()
  for directory in sorted(directories):
    relative = '.' if directory == root else directory.relative_to(root).as_posix()
    digest.update(b'D\0')
    digest.update(relative.encode('utf-8'))
    digest.update(b'\0')
  for path in sorted(files):
    digest.update(b'F\0')
    digest.update(path.relative_to(root).as_posix().encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(_sha256(path)))
  return digest.hexdigest()


def _strict_tree(
    root: Path, files: Mapping[str, tuple[int, str]], expected_dirs: set[str],
    expected_tree: str, label: str, *, cache_framing: bool = False,
) -> dict[str, Any]:
  _guard_path(root)
  if root.is_symlink() or not root.is_dir():
    raise AnalysisError(f'{label} root is absent or unsafe.')
  expected_paths = {(root / relative).resolve() for relative in files}
  expected_directory_paths = {(root / relative).resolve() for relative in expected_dirs}
  observed_files: set[Path] = set()
  observed_dirs = {root.resolve()}
  for entry in root.rglob('*'):
    mode = entry.lstat().st_mode
    if stat.S_ISLNK(mode):
      raise AnalysisError(f'{label} contains a symlink.')
    if stat.S_ISREG(mode):
      observed_files.add(entry.resolve())
    elif stat.S_ISDIR(mode):
      observed_dirs.add(entry.resolve())
    else:
      raise AnalysisError(f'{label} contains a special entry.')
  if observed_files != expected_paths or observed_dirs != expected_directory_paths:
    raise AnalysisError(f'{label} exact membership changed.')
  for relative, (size, digest) in files.items():
    path = root / relative
    if path.stat().st_size != size or _sha256(path) != digest:
      raise AnalysisError(f'{label} binding changed: {relative}.')
  tree = (
      _cache_tree_digest(observed_files, observed_dirs, root.resolve())
      if cache_framing else _tree_digest(observed_files, root.resolve())
  )
  if tree != expected_tree:
    raise AnalysisError(f'{label} tree digest changed.')
  return {'file_count': len(files), 'tree_sha256': tree}


def _git_blob(commit: str, relative: str, *, bundle_root: Path) -> bytes:
  try:
    return subprocess.check_output(
        ('git', '-C', str(bundle_root), 'show', f'{commit}:{relative}')
    )
  except subprocess.CalledProcessError as error:
    raise AnalysisError(f'Historical source missing: {commit}:{relative}.') from error


def _validate_new_source_row(
    relative: str, digest: str, *, bundle_root: Path,
) -> None:
  if (
      not isinstance(relative, str) or not relative
      or Path(relative).is_absolute() or '..' in Path(relative).parts
      or not _is_sha256(digest)
  ):
    raise AnalysisError('v3.3.3.1 source inventory is malformed.')
  path = (bundle_root / relative).resolve()
  try:
    path.relative_to(bundle_root)
  except ValueError as error:
    raise AnalysisError('v3.3.3.1 source escaped repository.') from error
  _strict_regular(path, f'v3.3.3.1 source {relative}')
  if _sha256(path) != digest:
    raise AnalysisError(f'v3.3.3.1 source bytes changed: {relative}.')
  subprocess.run(
      ('git', '-C', str(bundle_root), 'ls-files', '--error-unmatch', relative),
      check=True, capture_output=True,
  )
  head_bytes = _git_blob('HEAD', relative, bundle_root=bundle_root)
  if hashlib.sha256(head_bytes).hexdigest() != digest:
    raise AnalysisError(f'v3.3.3.1 HEAD source bytes changed: {relative}.')


def _validate_new_freeze(bundle_root: Path) -> tuple[dict[str, Any], str]:
  freeze = _read_json(_FREEZE_PATH, 'v3.3.3.1 analysis freeze')
  keys = {
      'analysis_version', 'amendment_path', 'amendment_sha256',
      'amendment_commit', 'model_run_commit', 'original_protocol_commit',
      'original_protocol_sha256', 'original_freeze_path',
      'original_freeze_sha256', 'original_freeze_key_count',
      'original_freeze_file_count', 'original_analyzer_path',
      'original_analyzer_sha256', 'original_analyzer_test_path',
      'original_analyzer_test_sha256', 'original_attempt_dir',
      'original_analysis_dir', 'run_dir', 'preflight_dir',
      'model_cache_dir', 'attempt_dir', 'analysis_dir',
      'run_file_count', 'run_tree_sha256', 'compiler_file_count',
      'compiler_tree_sha256', 'preflight_file_count',
      'preflight_tree_sha256', 'model_cache_file_count',
      'model_cache_tree_sha256', 'program_signatures_sha256',
      'program_signatures_canonical_size_bytes', 'leaves_tuple_count',
      'shape_tuple_count', 'tuple_container_count', 'file_sha256',
  }
  if set(freeze) != keys:
    raise AnalysisError('v3.3.3.1 freeze key set changed.')
  expected = {
      'analysis_version': ANALYSIS_VERSION,
      'amendment_path': str(_AMENDMENT_PATH.resolve()),
      'amendment_sha256': AMENDMENT_SHA256,
      'amendment_commit': AMENDMENT_COMMIT,
      'model_run_commit': MODEL_RUN_COMMIT,
      'original_protocol_commit': ORIGINAL_PROTOCOL_COMMIT,
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'original_freeze_path': str(_OLD_FREEZE.resolve()),
      'original_freeze_sha256': ORIGINAL_FREEZE_SHA256,
      'original_freeze_key_count': 69,
      'original_freeze_file_count': 96,
      'original_analyzer_path': str(_OLD_ANALYZER.resolve()),
      'original_analyzer_sha256': ORIGINAL_ANALYZER_SHA256,
      'original_analyzer_test_path': str(_OLD_ANALYZER_TEST.resolve()),
      'original_analyzer_test_sha256': ORIGINAL_ANALYZER_TEST_SHA256,
      'original_attempt_dir': str(_OLD_ATTEMPT_DIR.resolve()),
      'original_analysis_dir': str(_OLD_ANALYSIS_DIR.resolve()),
      'run_dir': str(_RUN_DIR.resolve()),
      'preflight_dir': str(_PREFLIGHT_DIR.resolve()),
      'model_cache_dir': str(_MODEL_CACHE_DIR.resolve()),
      'attempt_dir': str(_ATTEMPT_DIR.resolve()),
      'analysis_dir': str(_ANALYSIS_DIR.resolve()),
      'run_file_count': 11, 'run_tree_sha256': _RUN_TREE_SHA256,
      'compiler_file_count': 4, 'compiler_tree_sha256': _COMPILER_TREE_SHA256,
      'preflight_file_count': 5,
      'preflight_tree_sha256': _PREFLIGHT_TREE_SHA256,
      'model_cache_file_count': 1,
      'model_cache_tree_sha256': _MODEL_CACHE_TREE_SHA256,
      'program_signatures_sha256': PROGRAM_SIGNATURES_SHA256,
      'program_signatures_canonical_size_bytes': PROGRAM_SIGNATURES_CANONICAL_SIZE,
      'leaves_tuple_count': 3, 'shape_tuple_count': 29,
      'tuple_container_count': 32,
  }
  for key, value in expected.items():
    if freeze.get(key) != value:
      raise AnalysisError(f'v3.3.3.1 freeze changed at {key}.')
  inventory = freeze.get('file_sha256')
  if not isinstance(inventory, Mapping) or not inventory:
    raise AnalysisError('v3.3.3.1 source inventory is absent.')
  required = {
      str(path.relative_to(bundle_root)) for path in (
          _AMENDMENT_PATH, Path(__file__).resolve(),
          _HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_3_1_test.py',
          _HERE / 'run_encoder_skip_ood_sidecar_analysis_v3_3_3_1.sh',
          _OLD_ANALYZER, _OLD_ANALYZER_TEST, _OLD_FREEZE,
      )
  }
  if set(inventory) != required:
    raise AnalysisError('v3.3.3.1 source inventory membership changed.')
  for relative, digest in inventory.items():
    _validate_new_source_row(relative, digest, bundle_root=bundle_root)
  freeze_relative = str(_FREEZE_PATH.relative_to(bundle_root))
  subprocess.run(
      ('git', '-C', str(bundle_root), 'ls-files', '--error-unmatch', freeze_relative),
      check=True, capture_output=True,
  )
  if _git_blob('HEAD', freeze_relative, bundle_root=bundle_root) != _FREEZE_PATH.read_bytes():
    raise AnalysisError('v3.3.3.1 freeze differs from its HEAD blob.')
  if subprocess.check_output(
      ('git', '-C', str(bundle_root), 'diff', '--binary', 'HEAD', '--')
  ):
    raise AnalysisError('v3.3.3.1 requires a globally tracked-clean HEAD.')
  if _sha256(_AMENDMENT_PATH) != AMENDMENT_SHA256 or hashlib.sha256(
      _git_blob(AMENDMENT_COMMIT, str(_AMENDMENT_PATH.relative_to(bundle_root)),
                bundle_root=bundle_root)
  ).hexdigest() != AMENDMENT_SHA256:
    raise AnalysisError('v3.3.3.1 amendment binding changed.')
  return dict(freeze), _sha256(_FREEZE_PATH)


def _load_old_validator() -> ModuleType:
  """Loads only the frozen helper module after validating its live bytes."""
  _assert_cpu_only('before frozen helper import')
  if _sha256(_OLD_ANALYZER) != ORIGINAL_ANALYZER_SHA256:
    raise AnalysisError('Frozen v3.3.3 analyzer bytes changed.')
  spec = importlib.util.spec_from_file_location(
      '_opensplice_frozen_v333_helpers_for_v3331', _OLD_ANALYZER
  )
  if spec is None or spec.loader is None:
    raise AnalysisError('Frozen v3.3.3 helpers cannot be loaded.')
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  _assert_cpu_only('after frozen helper import')
  return module


def _validate_original_bundle(
    bundle_root: Path, old: ModuleType, *,
    prevalidated: tuple[dict[str, Any], str] | None = None,
) -> tuple[dict[str, Any], str, tuple[Any, ...]]:
  if prevalidated is None:
    prevalidated = _validate_original_inventory_stdlib(bundle_root)
  freeze, freeze_sha = prevalidated
  try:
    validated = old._validate_freeze(_RUN_DIR, bundle_root=bundle_root)
  except old.AnalysisError as error:
    raise AnalysisError(str(error)) from error
  return freeze, freeze_sha, validated


def _validate_original_source_row(
    relative: str, digest: str, *, bundle_root: Path,
) -> None:
  if (
      not isinstance(relative, str) or not relative
      or Path(relative).is_absolute() or '..' in Path(relative).parts
      or not _is_sha256(digest)
  ):
    raise AnalysisError('Original source inventory row is malformed.')
  path = (bundle_root / relative).resolve()
  try:
    path.relative_to(bundle_root)
  except ValueError as error:
    raise AnalysisError('Original source path escaped repository.') from error
  _strict_regular(path, f'original bundle {relative}')
  if _sha256(path) != digest:
    raise AnalysisError(f'Original bundle live bytes changed: {relative}.')
  historical = _git_blob(MODEL_RUN_COMMIT, relative, bundle_root=bundle_root)
  if hashlib.sha256(historical).hexdigest() != digest:
    raise AnalysisError(f'Original bundle historical bytes changed: {relative}.')
  subprocess.run(
      ('git', '-C', str(bundle_root), 'ls-files', '--error-unmatch', relative),
      check=True, capture_output=True,
  )


def _validate_original_inventory_stdlib(
    bundle_root: Path,
) -> tuple[dict[str, Any], str]:
  """Validates all 96 source rows before run reads or helper imports."""
  _assert_cpu_only('original inventory entry')
  _assert_old_destinations_absent()
  if bundle_root.resolve() != _REPO_ROOT:
    raise AnalysisError('Original inventory repository root changed.')
  if _sha256(_OLD_ANALYZER) != ORIGINAL_ANALYZER_SHA256:
    raise AnalysisError('Frozen v3.3.3 analyzer bytes changed.')
  if _sha256(_OLD_ANALYZER_TEST) != ORIGINAL_ANALYZER_TEST_SHA256:
    raise AnalysisError('Frozen v3.3.3 analyzer test bytes changed.')
  if _sha256(_OLD_FREEZE) != ORIGINAL_FREEZE_SHA256:
    raise AnalysisError('Frozen v3.3.3 freeze bytes changed.')
  freeze = _read_json(_OLD_FREEZE, 'v3.3.3 freeze')
  inventory = freeze.get('file_sha256')
  if len(freeze) != 69 or not isinstance(inventory, Mapping) or len(inventory) != 96:
    raise AnalysisError('Frozen v3.3.3 inventory shape/count changed.')
  for relative, digest in inventory.items():
    _validate_original_source_row(relative, digest, bundle_root=bundle_root)
  if inventory.get(str(_OLD_ANALYZER.relative_to(bundle_root))) != ORIGINAL_ANALYZER_SHA256:
    raise AnalysisError('Original analyzer is not freeze-bound exactly.')
  if subprocess.check_output(
      ('git', '-C', str(bundle_root), 'diff', '--binary', 'HEAD', '--')
  ):
    raise AnalysisError('Original inventory requires a globally tracked-clean HEAD.')
  _assert_cpu_only('original inventory exit')
  return dict(freeze), ORIGINAL_FREEZE_SHA256


def _validate_fixed_trees() -> dict[str, Any]:
  run = _strict_tree(
      _RUN_DIR, _RUN_FILES, {'.', 'compiler', 'compiler/eight_row'},
      _RUN_TREE_SHA256, 'v3.3.3 run',
  )
  compiler_files = {
      relative: binding for relative, binding in _RUN_FILES.items()
      if relative.startswith('compiler/')
  }
  compiler_paths = [(_RUN_DIR / relative).resolve() for relative in compiler_files]
  if len(compiler_paths) != 4 or _tree_digest(compiler_paths, _RUN_DIR) != _COMPILER_TREE_SHA256:
    raise AnalysisError('v3.3.3 compiler tree changed.')
  preflight = _strict_tree(
      _PREFLIGHT_DIR, _PREFLIGHT_FILES, {'.'}, _PREFLIGHT_TREE_SHA256,
      'v3.3.3 preflight',
  )
  cache = _strict_tree(
      _MODEL_CACHE_DIR, _MODEL_CACHE_FILES,
      {'.', 'triton', 'xdg', 'xdg/matplotlib'}, _MODEL_CACHE_TREE_SHA256,
      'v3.3.3 model cache', cache_framing=True,
  )
  return {'run': run, 'preflight': preflight, 'model_cache': cache}


def _tupleize_runtime_signature(stored: Mapping[str, Any]) -> dict[str, Any]:
  if set(stored) != {'eight_interventions', 'selection', 'target'}:
    raise AnalysisError('Program-signature object names changed.')
  result: dict[str, Any] = {}
  leaf_count = 0
  for name, signature in stored.items():
    if not isinstance(signature, Mapping) or set(signature) != {'treedef', 'leaves'}:
      raise AnalysisError(f'Program signature {name} schema changed.')
    if not isinstance(signature['treedef'], str) or not signature['treedef']:
      raise AnalysisError(f'Program signature {name} treedef changed.')
    leaves = signature['leaves']
    if not isinstance(leaves, list):
      raise AnalysisError(f'Persisted program signature {name} leaves are not lists.')
    runtime_leaves = []
    for leaf in leaves:
      if not isinstance(leaf, Mapping) or set(leaf) != {'dtype', 'shape'}:
        raise AnalysisError(f'Program signature {name} leaf schema changed.')
      if not isinstance(leaf['dtype'], str) or not leaf['dtype']:
        raise AnalysisError(f'Program signature {name} dtype changed.')
      shape = leaf['shape']
      if (
          not isinstance(shape, list)
          or any(isinstance(item, bool) or not isinstance(item, int) or item < 0
                 for item in shape)
      ):
        raise AnalysisError(f'Program signature {name} shape changed.')
      runtime_leaves.append({'dtype': leaf['dtype'], 'shape': tuple(shape)})
      leaf_count += 1
    result[name] = {'treedef': signature['treedef'], 'leaves': tuple(runtime_leaves)}
  if len(result) != 3 or leaf_count != 29:
    raise AnalysisError('Program signature tuple cardinality changed.')
  return result


def _representation_audit(
    compiler: Mapping[str, Any], start: Mapping[str, Any], old: ModuleType,
) -> dict[str, Any]:
  current = compiler.get('program_signatures')
  prior = start.get('v3_3_2_run_binding', {}).get(
      'eight_row_compiler', {}
  ).get('program_signatures')
  if not isinstance(current, Mapping) or not isinstance(prior, Mapping):
    raise AnalysisError('Current/prior program signatures are absent.')
  if current != prior:
    raise AnalysisError('Current/prior stored signature literals changed.')
  current_bytes, prior_bytes = _canonical_bytes(current), _canonical_bytes(prior)
  if (
      len(current_bytes) != PROGRAM_SIGNATURES_CANONICAL_SIZE
      or current_bytes != prior_bytes
      or hashlib.sha256(current_bytes).hexdigest() != PROGRAM_SIGNATURES_SHA256
      or compiler.get('program_signatures_sha256') != PROGRAM_SIGNATURES_SHA256
  ):
    raise AnalysisError('Program signature canonical bytes/hash changed.')
  runtime = _tupleize_runtime_signature(current)
  if runtime == prior:
    raise AnalysisError('Expected tuple/list direct inequality disappeared.')
  if _canonical_bytes(runtime) != current_bytes or _canonical_sha(runtime) != PROGRAM_SIGNATURES_SHA256:
    raise AnalysisError('Tuple/list canonical equality disappeared.')
  tuple_locations = 3 + sum(
      len(signature['leaves']) for signature in runtime.values()
  )
  if tuple_locations != 32:
    raise AnalysisError('Tuple-container count changed.')

  gate = compiler.get('source_program_gate')
  gate_keys = {
      'contract', 'observed', 'stablehlo_exact', 'pre_backend_hlo_exact',
      'program_signatures_exact', 'entry_abi_exact',
      'source_runtime_device_toolchain_checkpoint_reference_exact',
      'source_input_audit', 'same_lowered_compiled_object',
      'source_program_exact',
  }
  if not isinstance(gate, Mapping) or set(gate) != gate_keys:
    raise AnalysisError('Stored source-program gate is absent.')
  required_true = {
      'stablehlo_exact', 'pre_backend_hlo_exact', 'entry_abi_exact',
      'source_runtime_device_toolchain_checkpoint_reference_exact',
      'same_lowered_compiled_object',
  }
  if (
      gate.get('program_signatures_exact') is not False
      or gate.get('source_program_exact') is not False
      or any(gate.get(key) is not True for key in required_true)
  ):
    raise AnalysisError('Stored source-program gate failure pattern changed.')
  source_audit = gate.get('source_input_audit')
  source_audit_keys = {
      'bootstrap_sources_and_prior_trees_exact',
      'tracked_head_and_frozen_inventory_exact',
      'external_device_runtime_environment_exact',
      'same_process_device_runtime_environment_exact', 'checkpoint_exact',
      'reference_object_and_sequences_exact', 'protobuf_binding_exact',
      'three_import_inventories_stable_exact', 'freeze_sha256',
      'external_preflight_sha256', 'checkpoint_binding',
      'reference_object_binding',
  }
  audit_true = {
      'bootstrap_sources_and_prior_trees_exact',
      'tracked_head_and_frozen_inventory_exact',
      'external_device_runtime_environment_exact',
      'same_process_device_runtime_environment_exact', 'checkpoint_exact',
      'reference_object_and_sequences_exact', 'protobuf_binding_exact',
      'three_import_inventories_stable_exact',
  }
  if (
      not isinstance(source_audit, Mapping) or set(source_audit) != source_audit_keys
      or any(
      source_audit.get(key) is not True for key in audit_true
      )
  ):
    raise AnalysisError('Stored source/input audit changed.')
  try:
    old._validate_source_program_gate(
        compiler, compiler['artifacts'], expected_signatures=prior,
        expected_source_input_audit=source_audit,
    )
  except old.AnalysisError as error:
    expected = 'Compiler source-program gate changed at program_signatures_exact.'
    if str(error) != expected:
      raise AnalysisError('Frozen analyzer rejection point changed.') from error
    old_rejection = expected
  else:
    raise AnalysisError('Frozen analyzer no longer rejects the stored gate.')
  return {
      'diagnosis': 'tuple_leaves_and_shapes_vs_json_lists_only',
      'stored_current_equals_stored_prior': True,
      'runtime_tuple_direct_equals_stored_list': False,
      'runtime_tuple_canonical_equals_stored_list': True,
      'canonical_size_bytes': len(current_bytes),
      'canonical_sha256': PROGRAM_SIGNATURES_SHA256,
      'signature_object_count': 3,
      'leaves_tuple_count': 3,
      'shape_tuple_count': 29,
      'tuple_container_count': tuple_locations,
      'stored_program_signatures_exact': False,
      'stored_source_program_exact': False,
      'all_other_source_program_terms_exact': True,
      'frozen_analyzer_helper_rejection': old_rejection,
  }


def _validate_preflight_standalone(
    old: ModuleType, start: Mapping[str, Any], freeze: Mapping[str, Any],
    freeze_sha: str,
) -> dict[str, Any]:
  external = start.get('external_preflight')
  same = start.get('same_process_preflight')
  bootstrap = start.get('bootstrap')
  if not isinstance(external, Mapping) or not isinstance(same, Mapping):
    raise AnalysisError('Device preflight records are absent.')
  path = Path(str(external.get('path', ''))).resolve()
  if (
      path != (_PREFLIGHT_DIR / 'preflight_0000.json').resolve()
      or external.get('sha256') != _PREFLIGHT_FILES['preflight_0000.json'][1]
      or _sha256(path) != external.get('sha256')
  ):
    raise AnalysisError('External preflight path/hash changed.')
  raw = _read_json(path, 'external preflight')
  embedded = {
      key: value for key, value in external.items()
      if key not in {'path', 'sha256', 'validated_logs', 'directory_binding'}
  }
  if raw != embedded:
    raise AnalysisError('Embedded external preflight differs from current bytes.')
  if (
      raw.get('status') != 'pass' or raw.get('failure') is not None
      or raw.get('preflight_attempt_number') != 0
      or raw.get('freeze_sha256') != freeze_sha
      or raw.get('no_model_or_biological_access') is not True
      or raw.get('no_jit_or_array_kernel') is not True
  ):
    raise AnalysisError('External preflight pass/no-model contract changed.')
  try:
    old._v332._v33._validate_device_observation(
        raw.get('observation'), 'external preflight'
    )
    old._v332._v33._validate_device_observation(same, 'same-process preflight')
  except old._v332._v33.AnalysisError as error:
    raise AnalysisError(str(error)) from error
  if same.get('pid') != bootstrap.get('pid'):
    raise AnalysisError('Same-process preflight PID changed.')
  external_cache = old._expected_cache_environment(freeze, 'external_preflight')
  model_cache = old._expected_cache_environment(freeze, 'model')
  observation_runtime = raw.get('observation', {}).get(
      'v3_3_3_runtime_environment'
  )
  if (
      not isinstance(observation_runtime, Mapping)
      or observation_runtime.get('cache_environment') != external_cache
      or same.get('v3_3_3_cache_environment') != model_cache
      or bootstrap.get('sanitized_environment', {}).get('cache_environment')
      != model_cache
  ):
    raise AnalysisError('External/model pre-import cache routing changed.')
  live = observation_runtime.get('live_cache_environment')
  expected_live = {
      'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
      'JAX_ENABLE_COMPILATION_CACHE': 'false', 'CUDA_CACHE_DISABLE': '1',
      'cache_role': 'external_preflight',
      'cache_root': external_cache['cache_root'],
      'triton_cache_dir': external_cache['triton_cache_dir'],
      'xdg_cache_home': external_cache['xdg_cache_home'],
      'present_forbidden_names': [], 'exact_to_pre_import_routing': True,
  }
  if live != expected_live:
    raise AnalysisError('External live cache routing changed.')
  transition = start.get('cache_role_transition')
  expected_transition = {
      'contract': freeze['cache_isolation_contract'],
      'external_preflight': external_cache, 'model': model_cache,
      'roles_and_roots_distinct': True, 'shared_policy_exact': True,
      'default_user_cache_paths_eligible': False,
      'cache_output_equality_is_a_gate': False,
  }
  if transition != expected_transition:
    raise AnalysisError('External/model cache-role transition changed.')
  directory = external.get('directory_binding')
  expected_files = {
      name: {'sha256': digest, 'size_bytes': size}
      for name, (size, digest) in _PREFLIGHT_FILES.items()
  }
  if directory != {
      'path': str(_PREFLIGHT_DIR.resolve()), 'file_count': 5,
      'file_sha256': expected_files,
      'tree_sha256': _PREFLIGHT_TREE_SHA256,
      'sole_preflight_attempt_exact': True,
  }:
    raise AnalysisError('External preflight directory binding changed.')
  return {
      'external_preflight_sha256': external['sha256'],
      'external_preflight_file_count': 5,
      'external_preflight_tree_sha256': _PREFLIGHT_TREE_SHA256,
      'external_exact_rtx3090_uuid_gate': True,
      'same_process_exact_rtx3090_uuid_gate': True,
      'same_process_pid_bound_to_bootstrap': True,
      'cache_inputs_absent': True,
  }


def _validate_terminal(
    old: ModuleType, freeze: Mapping[str, Any], freeze_sha: str,
    validated: tuple[Any, ...], *, representation: bool = True,
) -> dict[str, Any]:
  (
      _, _, original_audit, _, _, v332_run, v3321_failure, v3322_archive,
  ) = validated
  start = _read_json(_RUN_DIR / 'ATTEMPT_STARTED.json', 'ATTEMPT_STARTED')
  completion = _read_json(_RUN_DIR / 'RUN_COMPLETE.json', 'RUN_COMPLETE')
  start = _read_json(_RUN_DIR / 'ATTEMPT_STARTED.json', 'ATTEMPT_STARTED')
  start_keys = {
      'active_recipient_rows_without_cross_call_predicate', 'amendment',
      'attempt_id', 'bootstrap', 'cache_role_transition',
      'checkpoint_binding', 'checkpoint_path', 'combined_analysis_permitted',
      'compile_count_contract', 'compiled_backend_equality_is_a_gate',
      'confirmation_model_calls', 'confirmation_scope_disclosure',
      'execution_order_contract', 'external_preflight', 'freeze',
      'freeze_sha256', 'invariant_rows_between_calls', 'max_output_bytes',
      'max_wall_time_seconds', 'model_apply_count_contract',
      'original_protocol_sha256', 'original_run_binding',
      'original_run_revalidated_in_full', 'program_signatures',
      'record_count_contract', 'reference_object_binding',
      'reference_sequence_bindings', 'rerun_count_contract',
      'runtime_environment', 'runtime_version_binding',
      'same_process_preflight', 'scientific_summary_computed',
      'script_version', 'shapley_or_nomination_computed',
      'source_program_contract', 'started_at_unix_s', 'status',
      'v3_3_1_status', 'v3_3_2_1_failure_status',
      'v3_3_2_2_archive_status', 'v3_3_2_run_binding',
  }
  if set(start) != start_keys:
    raise AnalysisError('ATTEMPT_STARTED exact key set changed.')
  required_start = {
      'attempt_id': old.ATTEMPT_ID, 'script_version': old.SCRIPT_VERSION,
      'status': 'started_append_only_one_shot',
      'freeze': dict(freeze), 'freeze_sha256': freeze_sha,
      'original_protocol_sha256': old.ORIGINAL_PROTOCOL_SHA256,
      'original_run_revalidated_in_full': True,
      'v3_3_2_run_binding': v332_run,
      'v3_3_2_1_failure_status': v3321_failure,
      'v3_3_2_2_archive_status': v3322_archive,
      'record_count_contract': 80, 'model_apply_count_contract': 320,
      'compile_count_contract': {'eight_row': 1, 'six_row': 0},
      'rerun_count_contract': {'identity': 0, 'main_cube': 0},
      'compiled_backend_equality_is_a_gate': False,
      'confirmation_model_calls': 0, 'scientific_summary_computed': False,
      'shapley_or_nomination_computed': False,
      'combined_analysis_permitted': False,
      'confirmation_scope_disclosure': CONFIRMATION_DISCLOSURE,
  }
  for key, expected in required_start.items():
    if start.get(key) != expected:
      raise AnalysisError(f'ATTEMPT_STARTED changed at {key}.')
  _finite(start.get('started_at_unix_s'), 'ATTEMPT_STARTED.started_at_unix_s')
  bootstrap = start.get('bootstrap')
  bootstrap_keys = {
      'bootstrap_path', 'bootstrap_sha256', 'created_at_unix_s',
      'freeze_path', 'freeze_sha256', 'generated_bindings', 'git_head',
      'launcher_path', 'launcher_sha256', 'original_bundle',
      'original_eight_row_compiler', 'original_run',
      'original_run_all_5158_files_rehashed', 'pid', 'preflight_state',
      'sanitized_environment', 'tracked_head_clean', 'v3_3_1_status',
      'v3_3_2_1_failure_status', 'v3_3_2_2_archive_status', 'v3_3_2_run',
  }
  if not isinstance(bootstrap, Mapping) or set(bootstrap) != bootstrap_keys:
    raise AnalysisError('ATTEMPT_STARTED.bootstrap exact key set changed.')
  if (
      not isinstance(bootstrap.get('pid'), int) or bootstrap['pid'] <= 0
      or bootstrap.get('freeze_path') != str(_OLD_FREEZE.resolve())
      or bootstrap.get('freeze_sha256') != freeze_sha
      or bootstrap.get('tracked_head_clean') is not True
      or bootstrap.get('original_run_all_5158_files_rehashed') is not True
      or bootstrap.get('v3_3_2_run') != v332_run
      or bootstrap.get('v3_3_2_1_failure_status') != v3321_failure
      or bootstrap.get('v3_3_2_2_archive_status') != v3322_archive
  ):
    raise AnalysisError('ATTEMPT_STARTED.bootstrap binding changed.')
  _finite(bootstrap.get('created_at_unix_s'), 'bootstrap.created_at_unix_s')
  try:
    preflight_audit = _validate_preflight_standalone(
        old, start, freeze, freeze_sha
    )
    cases = old._v332._v33._load_cases()
    checkpoint_audit = old._v332._v33._validate_checkpoint_reference(
        start, freeze, cases
    )
    runtime_audit = old._v332._v33._validate_runtime_manifest(start, freeze)
    normalized = dict(start)
    normalized['same_process_pre_import_bootstrap'] = {
        'freeze': bootstrap['original_bundle']
    }
    _, upstream_audit = old._v332._v33._validate_upstream_checkout(
        normalized, freeze, bundle_root=_REPO_ROOT
    )
  except (old.AnalysisError, old._v332._v33.AnalysisError) as error:
    raise AnalysisError(str(error)) from error
  start_audit = {
      'git_head': bootstrap['git_head'], **preflight_audit,
      **checkpoint_audit, **runtime_audit, **upstream_audit,
  }
  compiler = _read_json(
      _RUN_DIR / 'compiler/eight_row/COMPILER_PROVENANCE.json',
      'COMPILER_PROVENANCE',
  )
  predicates = {
      'status': 'controlled_stop', 'stop_reason': 'source_program_mismatch',
      'model_apply_count': 0, 'ood_anchor_record_count': 0,
      'ood_invalid_count': 0, 'unique_recipient_anchor_count': 0,
      'all_80_recipient_anchors_complete': False,
      'eight_row_compile_count': 1, 'eight_row_compile_attempt_count': 1,
      'eight_row_successful_compile_count': 1, 'six_row_compile_count': 0,
      'identity_rerun_count': 0, 'main_cube_rerun_count': 0,
      'old_ood_records_reused': 0, 'confirmation_model_calls': 0,
      'id0_all20': False, 'id255_all20': False,
      'scientific_summary_computed': False,
      'shapley_or_nomination_computed': False,
      'interaction_or_resolution_computed': False,
      'combined_analysis_permitted': False,
      'source_program_exact': False,
      'compiled_backend_diagnostic_only': True,
      'original_run_revalidated_in_full': True,
      'original_ood_records_provenance_only': True,
  }
  if {key: completion.get(key) for key in predicates} != predicates:
    raise AnalysisError('RUN_COMPLETE structural/no-science predicates changed.')
  manifest = _read_json(_RUN_DIR / 'RAW_MANIFEST.json', 'RAW_MANIFEST')
  empty = {
      'artifact_count': 0, 'artifact_sha256': {},
      'artifact_tree_sha256': EMPTY_SHA256,
  }
  if manifest != empty or completion.get('raw_manifest') != empty:
    raise AnalysisError('Empty raw-manifest state changed.')
  if (_RUN_DIR / 'raw').exists() or (_RUN_DIR / 'raw').is_symlink():
    raise AnalysisError('Unexpected raw directory exists.')
  if (
      completion.get('eight_row_compiler') != compiler
      or completion.get('source_program_gate') != compiler.get('source_program_gate')
      or completion.get('eight_row_executable_fingerprint')
      != compiler.get('executable_fingerprint')
      or completion.get('backend_diagnostics') != compiler.get('backend_diagnostics')
      or completion.get('diagnostic_comparisons')
      != compiler.get('diagnostic_comparisons')
  ):
    raise AnalysisError('RUN_COMPLETE/compiler linkage changed.')
  artifacts = compiler.get('artifacts')
  if not isinstance(artifacts, Mapping) or set(artifacts) != {
      'stablehlo', 'hlo', 'compiled_hlo'
  }:
    raise AnalysisError('Compiler artifacts changed.')
  names = {
      'stablehlo': 'graph.stablehlo.mlir',
      'hlo': 'graph.pre_backend.hlo.txt',
      'compiled_hlo': 'graph.compiled.hlo.txt',
  }
  for role, name in names.items():
    row = artifacts[role]
    path = (_RUN_DIR / 'compiler/eight_row' / name).resolve()
    if (
        not isinstance(row, Mapping) or row.get('path') != str(path)
        or row.get('sha256') != _sha256(path)
        or row.get('size_bytes') != path.stat().st_size
    ):
      raise AnalysisError(f'Compiler artifact binding changed: {role}.')
  expected_fingerprint = hashlib.sha256(
      bytes.fromhex(artifacts['compiled_hlo']['sha256'])
  ).hexdigest()
  if compiler.get('executable_fingerprint') != expected_fingerprint:
    raise AnalysisError('Executable fingerprint changed.')
  source_audit = compiler['source_program_gate']['source_input_audit']
  try:
    imports = old._validate_imports(
        _RUN_DIR, completion, bundle_root=_REPO_ROOT, freeze=freeze
    )
    protobuf = old._validate_protobuf(_RUN_DIR, completion, freeze)
    cache = old._validate_kernel_cache_provenance(
        compiler['kernel_cache_provenance'],
        expected_preimport=old._expected_cache_environment(freeze, 'model'),
        phase='post_compile',
    )
    terminal_cache = old._validate_final_model_cache(
        completion['model_kernel_cache_final'], freeze,
        compiler_cache_audit=cache,
    )
  except old.AnalysisError as error:
    raise AnalysisError(str(error)) from error
  if (
      imports.get('stable_shared_module_bytes') is not True
      or protobuf.get('seven_role_two_generated_output_repair_exact') is not True
      or source_audit.get('three_import_inventories_stable_exact') is not True
      or source_audit.get('protobuf_binding_exact') is not True
  ):
    raise AnalysisError('Import/protobuf source-input linkage changed.')
  representation_audit = (
      _representation_audit(compiler, start, old) if representation else None
  )
  return {
      'start': start_audit,
      'representation': representation_audit,
      'imports': imports, 'protobuf': protobuf,
      'kernel_cache': terminal_cache,
      'run_complete_sha256': _sha256(_RUN_DIR / 'RUN_COMPLETE.json'),
      'compiler_provenance_sha256': _sha256(
          _RUN_DIR / 'compiler/eight_row/COMPILER_PROVENANCE.json'
      ),
  }


def _assert_old_destinations_absent() -> None:
  for path, label in (
      (_OLD_ATTEMPT_DIR, 'original v3.3.3 analysis attempt'),
      (_OLD_ANALYSIS_DIR, 'original v3.3.3 analysis output'),
  ):
    if path.exists() or path.is_symlink():
      raise AnalysisError(f'{label} is not absent/fresh.')


def _assert_destinations_fresh() -> None:
  _assert_old_destinations_absent()
  for path, label in (
      (_ATTEMPT_DIR, 'v3.3.3.1 analysis attempt'),
      (_ANALYSIS_DIR, 'v3.3.3.1 analysis output'),
  ):
    if path.exists() or path.is_symlink():
      raise AnalysisError(f'{label} is not absent/fresh.')


def _provenance_precheck(bundle_root: Path) -> dict[str, Any]:
  _assert_cpu_only('v3.3.3.1 precheck entry')
  if bundle_root.resolve() != _REPO_ROOT:
    raise AnalysisError('Repository root changed.')
  _assert_destinations_fresh()
  freeze, freeze_sha = _validate_new_freeze(bundle_root)
  original_inventory = _validate_original_inventory_stdlib(bundle_root)
  trees = _validate_fixed_trees()
  old = _load_old_validator()
  original_freeze, original_freeze_sha, validated = _validate_original_bundle(
      bundle_root, old, prevalidated=original_inventory
  )
  # Sections 1--3 only. The representation branch is deliberately deferred
  # until after the durable append-only START has consumed this one-shot audit.
  terminal = _validate_terminal(
      old, original_freeze, original_freeze_sha, validated,
      representation=False,
  )
  _assert_cpu_only('v3.3.3.1 precheck exit')
  return {
      'freeze': freeze, 'freeze_sha256': freeze_sha,
      'trees': trees, 'terminal': terminal,
  }


def _run_bindings() -> dict[str, Any]:
  return {
      name: {'path': str((_RUN_DIR / name).resolve()), 'size_bytes': size,
             'sha256': digest}
      for name, (size, digest) in _RUN_FILES.items()
  }


_START_KEYS = {
    'analysis_version', 'status', 'amendment', 'analyzer', 'freeze',
    'run_dir', 'preflight_dir', 'model_cache_dir', 'attempt_dir',
    'analysis_dir', 'output_json', 'output_markdown', 'run_artifacts',
    'original_analyzer_attempt_absent', 'original_analyzer_output_absent',
    'model_apply_count', 'raw_record_count',
    'scientific_raw_evidence_reached', 'scientific_summary_computed',
    'donor_normalization_computed', 'shapley_or_nomination_computed',
    'interaction_or_resolution_computed', 'nomination_performed',
    'combined_analysis_permitted',
    'confirmation_model_outputs_activations_interventions_unopened',
    'started_at_unix_s',
}


def _start_record(precheck: Mapping[str, Any]) -> dict[str, Any]:
  return {
      'analysis_version': ANALYSIS_VERSION,
      'status': 'analysis_started_append_only_one_shot',
      'amendment': {'path': str(_AMENDMENT_PATH.resolve()),
                    'sha256': AMENDMENT_SHA256, 'commit': AMENDMENT_COMMIT},
      'analyzer': {'path': str(Path(__file__).resolve()),
                   'sha256': _sha256(Path(__file__).resolve())},
      'freeze': {'path': str(_FREEZE_PATH.resolve()),
                 'sha256': precheck['freeze_sha256']},
      'run_dir': str(_RUN_DIR.resolve()),
      'preflight_dir': str(_PREFLIGHT_DIR.resolve()),
      'model_cache_dir': str(_MODEL_CACHE_DIR.resolve()),
      'attempt_dir': str(_ATTEMPT_DIR.resolve()),
      'analysis_dir': str(_ANALYSIS_DIR.resolve()),
      'output_json': str((_ANALYSIS_DIR / 'ANALYSIS.json').resolve()),
      'output_markdown': str((_ANALYSIS_DIR / 'RESULT.md').resolve()),
      'run_artifacts': _run_bindings(),
      'original_analyzer_attempt_absent': True,
      'original_analyzer_output_absent': True,
      'model_apply_count': 0, 'raw_record_count': 0,
      'scientific_raw_evidence_reached': False,
      'scientific_summary_computed': False,
      'donor_normalization_computed': False,
      'shapley_or_nomination_computed': False,
      'interaction_or_resolution_computed': False,
      'nomination_performed': False, 'combined_analysis_permitted': False,
      'confirmation_model_outputs_activations_interventions_unopened': True,
      'started_at_unix_s': time.time(),
  }


def _validate_active_attempt(token: object | None, started_sha: str | None) -> dict[str, Any]:
  if token is not _ATTEMPT_TOKEN or not _is_sha256(started_sha):
    raise AnalysisError('Standalone archive requires the internal post-START gate.')
  paths = list(_ATTEMPT_DIR.iterdir()) if _ATTEMPT_DIR.is_dir() else []
  if (
      _ATTEMPT_DIR.is_symlink() or len(paths) != 1
      or paths[0].name != 'ANALYSIS_ATTEMPT_STARTED.json'
  ):
    raise AnalysisError('Active v3.3.3.1 attempt membership changed.')
  start_path = paths[0]
  _strict_regular(start_path, 'ANALYSIS_ATTEMPT_STARTED')
  if _sha256(start_path) != started_sha:
    raise AnalysisError('Active v3.3.3.1 START hash changed.')
  start = _read_json(start_path, 'ANALYSIS_ATTEMPT_STARTED')
  if set(start) != _START_KEYS:
    raise AnalysisError('Active v3.3.3.1 START key set changed.')
  expected = _start_record({'freeze_sha256': _sha256(_FREEZE_PATH)})
  expected.pop('started_at_unix_s')
  for key, value in expected.items():
    if start.get(key) != value:
      raise AnalysisError(f'Active v3.3.3.1 START changed at {key}.')
  _finite(start.get('started_at_unix_s'), 'START.started_at_unix_s')
  for name, binding in start['run_artifacts'].items():
    size, digest = _RUN_FILES[name]
    path = _RUN_DIR / name
    if binding != {'path': str(path.resolve()), 'size_bytes': size, 'sha256': digest}:
      raise AnalysisError(f'Active START run binding changed: {name}.')
  _assert_old_destinations_absent()
  if _ANALYSIS_DIR.exists() or _ANALYSIS_DIR.is_symlink():
    raise AnalysisError('v3.3.3.1 output appeared before delegated analysis.')
  return start


def analyze(
    *, token: object | None = None, started_sha256: str | None = None,
) -> dict[str, Any]:
  _assert_cpu_only('v3.3.3.1 analyze entry')
  _validate_active_attempt(token, started_sha256)
  # Revalidate source bytes first after START. No run/compiler byte is touched
  # and no frozen helper is imported until both source inventories pass.
  freeze, _ = _validate_new_freeze(_REPO_ROOT)
  original_inventory = _validate_original_inventory_stdlib(_REPO_ROOT)
  trees = _validate_fixed_trees()
  old = _load_old_validator()
  original_freeze, original_freeze_sha, validated = _validate_original_bundle(
      _REPO_ROOT, old, prevalidated=original_inventory
  )
  terminal = _validate_terminal(old, original_freeze, original_freeze_sha, validated)
  result = {
      'analysis_version': ANALYSIS_VERSION,
      'status': 'complete_controlled_stop_structural_archive',
      'decision': 'controlled_stop_source_program_mismatch_representation_only',
      'model_apply_count': 0, 'raw_record_count': 0,
      'id0_all20': False, 'id255_all20': False,
      'scientific_raw_evidence_reached': False,
      'scientific_summary_computed': False,
      'donor_normalization_computed': False,
      'shapley_or_nomination_computed': False,
      'interaction_or_resolution_computed': False,
      'nomination_performed': False, 'nomination': None,
      'resolution_analysis': None, 'combined_analysis_permitted': False,
      'source_program_gate_passed_by_v3_3_3': False,
      'ood_experiment_executed_by_v3_3_3': False,
      'representation_audit': terminal['representation'],
      'provenance_audit': {
          'amendment_sha256': AMENDMENT_SHA256,
          'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
          'original_freeze_sha256': ORIGINAL_FREEZE_SHA256,
          'model_run_commit': MODEL_RUN_COMMIT,
          'run_tree': trees['run'], 'preflight_tree': trees['preflight'],
          'model_cache_tree': trees['model_cache'],
          'run_complete_sha256': terminal['run_complete_sha256'],
          'compiler_provenance_sha256': terminal['compiler_provenance_sha256'],
          'start': terminal['start'], 'imports': terminal['imports'],
          'protobuf': terminal['protobuf'],
          'kernel_cache': terminal['kernel_cache'],
          'old_analyzer_entry_point_invoked': False,
          'old_analyzer_binding_replaced': False,
      },
      'claim_boundary': (
          'Structural archive only: v3.3.3 stopped before apply zero; no '
          'biological or mechanistic evidence was produced.'
      ),
      'confirmation_model_outputs_activations_interventions_unopened': True,
      'confirmation_scope_disclosure': CONFIRMATION_DISCLOSURE,
  }
  if freeze.get('analysis_version') != ANALYSIS_VERSION:
    raise AnalysisError('Active v3.3.3.1 freeze changed.')
  _assert_old_destinations_absent()
  _assert_cpu_only('v3.3.3.1 analyze exit')
  return result


def _markdown(result: Mapping[str, Any]) -> str:
  audit = result['representation_audit']
  return '\n'.join((
      '# OpenSplice v3.3.3.1 structural archive', '',
      '**Decision:** controlled stop archived; no experiment execution.', '',
      'The v3.3.3 runner performed one successful eight-row compilation but '
      'stopped before model apply zero and wrote no raw record. The sole '
      'failed gate was host representation: three `leaves` tuples and 29 '
      '`shape` tuples compared against JSON lists. The literals, treedefs, '
      'dtypes, numeric shapes, order, and canonical signature SHA remained '
      f"exact (`{audit['canonical_sha256']}`).", '',
      'No scientific summary, donor normalization, Shapley value, interaction, '
      'resolution analysis, or nomination was computed. Combined analysis is '
      'not permitted by this archive.', '',
  ))


def _write_new(path: Path, data: bytes) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('xb') as handle:
    handle.write(data)


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
  _write_new(path, (json.dumps(value, indent=2, sort_keys=True) + '\n').encode())


def _output_state() -> dict[str, Any]:
  if not _ANALYSIS_DIR.exists() and not _ANALYSIS_DIR.is_symlink():
    return {'state': 'absent', 'file_count': 0, 'files': {},
            'tree_sha256': EMPTY_SHA256}
  if _ANALYSIS_DIR.is_symlink() or not _ANALYSIS_DIR.is_dir():
    raise AnalysisError('Analysis output root is unsafe.')
  files = []
  for entry in _ANALYSIS_DIR.iterdir():
    mode = entry.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
      raise AnalysisError('Analysis output contains an unsafe entry.')
    if entry.name not in {'ANALYSIS.json', 'RESULT.md'}:
      raise AnalysisError('Analysis output contains an extra entry.')
    files.append(entry)
  bindings = {
      path.name: {'sha256': _sha256(path), 'size_bytes': path.stat().st_size}
      for path in sorted(files)
  }
  return {
      'state': 'complete' if set(bindings) == {'ANALYSIS.json', 'RESULT.md'} else 'partial',
      'file_count': len(bindings), 'files': bindings,
      'tree_sha256': _tree_digest(files, _ANALYSIS_DIR),
  }


def _terminal_record(
    *, status: str, started_sha: str, error: BaseException | None = None,
) -> dict[str, Any]:
  record = {
      'analysis_version': ANALYSIS_VERSION, 'status': status,
      'attempt_started_sha256': started_sha,
      'analysis_output_state': _output_state(),
      'model_apply_count': 0, 'raw_record_count': 0,
      'scientific_raw_evidence_reached': False,
      'scientific_summary_computed': False,
      'donor_normalization_computed': False,
      'shapley_or_nomination_computed': False,
      'interaction_or_resolution_computed': False,
      'nomination_performed': False, 'combined_analysis_permitted': False,
      'confirmation_model_outputs_activations_interventions_unopened': True,
      'terminal_at_unix_s': time.time(),
  }
  if error is not None:
    record['error'] = {
        'type': type(error).__name__, 'message': str(error),
        'traceback': ''.join(traceback.format_exception(error)),
    }
  return record


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run-dir', type=Path, required=True)
  parser.add_argument('--bundle-root', type=Path, required=True)
  parser.add_argument('--output-json', type=Path, required=True)
  parser.add_argument('--output-markdown', type=Path, required=True)
  args = parser.parse_args()
  expected_json = (_ANALYSIS_DIR / 'ANALYSIS.json').resolve()
  expected_markdown = (_ANALYSIS_DIR / 'RESULT.md').resolve()
  if (
      args.run_dir.resolve() != _RUN_DIR.resolve()
      or args.bundle_root.resolve() != _REPO_ROOT
      or args.output_json.resolve() != expected_json
      or args.output_markdown.resolve() != expected_markdown
  ):
    raise AnalysisError('CLI paths differ from the frozen v3.3.3.1 paths.')
  precheck = _provenance_precheck(_REPO_ROOT)
  _ATTEMPT_DIR.mkdir(parents=True, exist_ok=False)
  start_path = _ATTEMPT_DIR / 'ANALYSIS_ATTEMPT_STARTED.json'
  _write_json_new(start_path, _start_record(precheck))
  started_sha = _sha256(start_path)
  try:
    result = analyze(token=_ATTEMPT_TOKEN, started_sha256=started_sha)
    _assert_old_destinations_absent()
    _ANALYSIS_DIR.mkdir(parents=True, exist_ok=False)
    _write_json_new(_ANALYSIS_DIR / 'ANALYSIS.json', result)
    _write_new(_ANALYSIS_DIR / 'RESULT.md', _markdown(result).encode())
    _assert_old_destinations_absent()
    _write_json_new(
        _ATTEMPT_DIR / 'ANALYSIS_COMPLETE.json',
        _terminal_record(
            status='analysis_complete_controlled_stop_structural_archive',
            started_sha=started_sha,
        ),
    )
  except BaseException as error:
    terminal_error = error
    try:
      _assert_old_destinations_absent()
    except BaseException as destination_error:
      terminal_error = AnalysisError(
          'Original v3.3.3 analyzer destination appeared during the '
          f'consumed audit: {destination_error}; prior error: {error}'
      )
    _write_json_new(
        _ATTEMPT_DIR / 'ANALYSIS_FAILURE.json',
        _terminal_record(
            status='analysis_failed_consumed_no_retry',
            started_sha=started_sha, error=terminal_error,
        ),
    )
    raise terminal_error


if __name__ == '__main__':
  main()
