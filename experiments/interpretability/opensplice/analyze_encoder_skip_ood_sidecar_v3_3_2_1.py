#!/usr/bin/env python3
"""CPU-only v3.3.2.1 archive of the zero-apply compiler controlled stop.

This wrapper leaves the frozen v3.3.2 analyzer and run untouched.  It repairs
only that analyzer's conflation of the external and same-process preflight
schemas, then delegates every remaining structural check to the frozen bytes.
It never imports JAX/AlphaGenome and never computes a scientific statistic.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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
from typing import Any, Iterator, Mapping


ANALYSIS_VERSION = 'opensplice-encoder-skip-ood-sidecar-analysis-v3.3.2.1'
AMENDMENT_REASON = 'same_process_external_preflight_schema_repair'
AMENDMENT_SHA256 = (
    '81a4f4c126b83225b02c7de5cf0dc6fd0baf6085b84b9ed5dd7a3677744090ba'
)
AMENDMENT_COMMIT = '214729e8cc84c821fe45b4373fcf93469a88a487'
MODEL_RUN_COMMIT = '24e2214168eeca41d4f3b60b62094b6befcadcc1'
ORIGINAL_PROTOCOL_SHA256 = (
    '85151029297cf4b4c1a07dd2a0f47b10b0663acd08ddfe25ba72b9898b1beea0'
)
V3_3_2_AMENDMENT_SHA256 = (
    '42cd43bbc25517d35b3e21dd5df7bf68a37ee46c51cb71bcdf363ac4de3b19e3'
)
V3_3_2_FREEZE_SHA256 = (
    'baa257dfa8d0bd1afca40b11038d71c9157499eeb654523ad59df119eb2c0295'
)
V3_3_2_ANALYZER_SHA256 = (
    '90f00a6d51f33ac456a0fd799f2b9caf456b58944928886dc1577731707f205e'
)
V3_3_2_TEST_SHA256 = (
    '2e7424a76840c776ff550db3961c1fb1b63dee273bd8d030800efd27bf81d11e'
)
RUN_FILE_COUNT = 11
RUN_TREE_SHA256 = (
    '4ac66e45a4d7d65af2785904d11b23bf7e809e07f3f617e190772242b2e7a4ab'
)
COMPILER_FILE_COUNT = 4
COMPILER_TREE_SHA256 = (
    '4378048568ff58a2bbee55ba9da750498b89fdef72c97911815cf895c8a8b7d1'
)
PREFLIGHT_FILE_COUNT = 5
PREFLIGHT_TREE_SHA256 = (
    '797211382478ba249fe94e7ccbcc11c7192d30423d5e289ce03cf3cea37f65f5'
)
EMPTY_SHA256 = (
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
)
BOOTSTRAP_PID = 2_327_369
EXPECTED_PYTHON_EXECUTABLE = (
    '/home/degen2/alphafold-stuff/agvenv/bin/python'
)
EXPECTED_PYTHON_RESOLVED = '/home/degen2/anaconda3/bin/python3.13'

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
_AMENDMENT_PATH = (
    _HERE / 'v3_wider_mechanism/'
    'encoder_skip_ood_sidecar_analysis_amendment_v3_3_2_1.md'
)
_ORIGINAL_ANALYZER_PATH = (
    _HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_2.py'
)
_ORIGINAL_TEST_PATH = (
    _HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_2_test.py'
)
_ORIGINAL_FREEZE_PATH = _HERE / 'encoder_skip_ood_sidecar_v3_3_2_freeze.json'
_FREEZE_PATH = _HERE / 'encoder_skip_ood_sidecar_analysis_v3_3_2_1_freeze.json'
_TEST_PATH = _HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_2_1_test.py'
_WRAPPER_PATH = _HERE / 'run_encoder_skip_ood_sidecar_analysis_v3_3_2_1.sh'
_RUN_DIR = _HERE / 'results/v3_3_2_development_ood_sidecar_one_shot'
_PREFLIGHT_DIR = _HERE / 'results/v3_3_2_device_preflight'
_ATTEMPT_DIR = (
    _HERE / 'results/v3_3_2_development_ood_sidecar_analysis_v3_3_2_1_attempt'
)
_ANALYSIS_DIR = (
    _HERE / 'results/v3_3_2_development_ood_sidecar_analysis_v3_3_2_1'
)

_RUN_FILES = {
    'ATTEMPT_STARTED.json': (774_186, 'd1bb85b3c35d892f10f67a5510b85784eb1b6b989c9ae32290b8c0fdc3c8d235'),
    'IMPORT_PROVENANCE.json': (41_558, 'a74f3c9658e9d2286724680b52f4ea788d492f4fa9d7c52b20a53c90d57edc99'),
    'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json': (41_558, 'a74f3c9658e9d2286724680b52f4ea788d492f4fa9d7c52b20a53c90d57edc99'),
    'IMPORT_PROVENANCE_PRE_MODEL.json': (41_558, 'a74f3c9658e9d2286724680b52f4ea788d492f4fa9d7c52b20a53c90d57edc99'),
    'PROTOBUF_PROVENANCE.json': (3_339, '2498a940f6ee15e54e72e8f51587d4c42ffc1b49851873c31ad09085315d0ba8'),
    'RAW_MANIFEST.json': (145, 'fadbff5ab512dea3d25edb39dc9a91ffe25473b73f1e33fd53f97b60fc8436fd'),
    'RUN_COMPLETE.json': (15_145, 'd88c8becc96a24f1f56a1cee82da35fdd331ee3b40ba4108ae98f44ae24b4ad7'),
    'compiler/eight_row/COMPILER_PROVENANCE.json': (8_196, 'bd20e21a56a9ca5498d7119771bb1da9ac2e156ed3190d3ce3aa09ff2d2e312c'),
    'compiler/eight_row/graph.compiled.hlo.txt': (16_601_836, 'b436435ebb14b87cf9929ee9b16fc2c74d1764460c701f8160f1dc092687b718'),
    'compiler/eight_row/graph.pre_backend.hlo.txt': (1_829_833, '675b9cf26fe8f59a1d138e856ee84f48edb20d8c896ac9e539b6b064dcef2750'),
    'compiler/eight_row/graph.stablehlo.mlir': (3_196_162, '69dbf2a054cf89e56a9000dc0c04e5cd4ba425f016885ddc127d22c01661bddd'),
}
_PREFLIGHT_FILES = {
    '.allocation.lock': (0, EMPTY_SHA256),
    '.preflight_0000.reserved': (0, EMPTY_SHA256),
    'preflight_0000.json': (682_921, '309bdec9544cd4ff2cdbc83207663e81d7f5d27ece83e4ad569e1859ba412a8c'),
    'preflight_0000.stderr.log': (0, EMPTY_SHA256),
    'preflight_0000.stdout.log': (0, EMPTY_SHA256),
}
_SAME_PROCESS_KEYS = {
    'environment', 'hostname', 'jax_default_backend', 'jax_gpu_devices',
    'jax_module_version', 'jaxlib_module_version', 'kernel',
    'no_jit_no_array_no_model', 'nvidia_smi', 'packages', 'pid', 'platform',
    'python_executable', 'python_version', 'runtime_environment',
}
_EXTERNAL_OBSERVATION_KEYS = _SAME_PROCESS_KEYS | {
    'jax_enable_compilation_cache', 'v3_3_runtime_environment',
    'v3_3_2_runtime_environment',
}
_FORBIDDEN_RESULT_KEYS = {
    'shapley', 'interaction', 'resolution_analysis', 'nomination',
    'scientific_summary',
}


class AmendmentError(RuntimeError):
  """Raised when the one-shot analyzer amendment fails closed."""


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


def _assert_cpu_only(label: str) -> None:
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


def _load_cpu_module(path: Path, digest: str, name: str):
  _assert_cpu_only(f'{name} pre-import')
  _strict_regular(path, name)
  if _sha256(path) != digest:
    raise AmendmentError(f'{name} bytes changed before import.')
  specification = importlib.util.spec_from_file_location(name, path)
  if specification is None or specification.loader is None:
    raise AmendmentError(f'Cannot load {name}.')
  module = importlib.util.module_from_spec(specification)
  specification.loader.exec_module(module)
  _assert_cpu_only(f'{name} post-import')
  return module


_v332 = _load_cpu_module(
    _ORIGINAL_ANALYZER_PATH, V3_3_2_ANALYZER_SHA256,
    '_opensplice_frozen_ood_sidecar_analyzer_v3_3_2',
)


def _read_object(path: Path, label: str) -> dict[str, Any]:
  _strict_regular(path, label)
  try:
    value = json.loads(path.read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError) as error:
    raise AmendmentError(f'{label} is not readable JSON.') from error
  if not isinstance(value, dict):
    raise AmendmentError(f'{label} must be a JSON object.')
  return value


def _tree_digest(paths: list[Path], root: Path) -> str:
  digest = hashlib.sha256()
  root = root.resolve()
  for path in sorted(path.resolve() for path in paths):
    digest.update(path.relative_to(root).as_posix().encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(_sha256(path)))
  return digest.hexdigest()


def _walk_exact_tree(
    root: Path, files: Mapping[str, tuple[int, str]],
    directories: set[str], label: str,
) -> dict[str, Any]:
  """Lstat-walks an exact tree without following links or ignoring dirs."""
  _strict_directory(root, label)
  seen_files: dict[str, Path] = {}
  seen_dirs = {'.'}

  def walk(directory: Path) -> None:
    with os.scandir(directory) as entries:
      for entry in entries:
        path = Path(entry.path)
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode) and not path.is_symlink():
          seen_dirs.add(relative)
          walk(path)
        elif stat.S_ISREG(mode) and not path.is_symlink():
          seen_files[relative] = path
        else:
          raise AmendmentError(
              f'{label} contains a symlink/special entry: {relative}.'
          )

  walk(root)
  if seen_dirs != directories:
    raise AmendmentError(f'{label} directory membership changed.')
  if set(seen_files) != set(files):
    raise AmendmentError(f'{label} file membership changed.')
  for relative, (size, digest) in files.items():
    path = seen_files[relative]
    if path.stat().st_size != size or _sha256(path) != digest:
      raise AmendmentError(f'{label} bytes changed: {relative}.')
  return {
      'file_count': len(seen_files),
      'tree_sha256': _tree_digest(list(seen_files.values()), root),
      'files': {
          relative: {'size_bytes': size, 'sha256': digest}
          for relative, (size, digest) in files.items()
      },
  }


def _assert_global_tracked_clean(bundle_root: Path) -> str:
  if subprocess.check_output(
      ('git', '-C', str(bundle_root), 'diff', '--binary', 'HEAD', '--')
  ):
    raise AmendmentError('v3.3.2.1 requires a globally tracked-clean HEAD.')
  return subprocess.check_output(
      ('git', '-C', str(bundle_root), 'rev-parse', 'HEAD'), text=True
  ).strip()


def _validate_freeze(bundle_root: Path) -> dict[str, Any]:
  freeze = _read_object(_FREEZE_PATH, 'v3.3.2.1 analysis freeze')
  exact_keys = {
      'analysis_version', 'amendment_path', 'amendment_sha256',
      'amendment_commit', 'model_run_commit', 'run_dir', 'preflight_dir',
      'attempt_dir', 'analysis_dir', 'original_analyzer_path',
      'original_analyzer_sha256', 'original_analyzer_test_path',
      'original_analyzer_test_sha256', 'original_freeze_path',
      'original_freeze_sha256', 'original_protocol_sha256',
      'v3_3_2_amendment_sha256', 'run_file_count', 'run_tree_sha256',
      'compiler_file_count', 'compiler_tree_sha256', 'preflight_file_count',
      'preflight_tree_sha256', 'file_sha256',
  }
  if set(freeze) != exact_keys:
    raise AmendmentError('v3.3.2.1 analysis freeze schema changed.')
  expected = {
      'analysis_version': ANALYSIS_VERSION,
      'amendment_path': str(_AMENDMENT_PATH.resolve()),
      'amendment_sha256': AMENDMENT_SHA256,
      'amendment_commit': AMENDMENT_COMMIT,
      'model_run_commit': MODEL_RUN_COMMIT,
      'run_dir': str(_RUN_DIR.resolve()),
      'preflight_dir': str(_PREFLIGHT_DIR.resolve()),
      'attempt_dir': str(_ATTEMPT_DIR.resolve()),
      'analysis_dir': str(_ANALYSIS_DIR.resolve()),
      'original_analyzer_path': str(_ORIGINAL_ANALYZER_PATH.resolve()),
      'original_analyzer_sha256': V3_3_2_ANALYZER_SHA256,
      'original_analyzer_test_path': str(_ORIGINAL_TEST_PATH.resolve()),
      'original_analyzer_test_sha256': V3_3_2_TEST_SHA256,
      'original_freeze_path': str(_ORIGINAL_FREEZE_PATH.resolve()),
      'original_freeze_sha256': V3_3_2_FREEZE_SHA256,
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'v3_3_2_amendment_sha256': V3_3_2_AMENDMENT_SHA256,
      'run_file_count': RUN_FILE_COUNT,
      'run_tree_sha256': RUN_TREE_SHA256,
      'compiler_file_count': COMPILER_FILE_COUNT,
      'compiler_tree_sha256': COMPILER_TREE_SHA256,
      'preflight_file_count': PREFLIGHT_FILE_COUNT,
      'preflight_tree_sha256': PREFLIGHT_TREE_SHA256,
  }
  for key, value in expected.items():
    if freeze.get(key) != value:
      raise AmendmentError(f'v3.3.2.1 analysis freeze changed at {key}.')
  inventory = freeze.get('file_sha256')
  required = {
      str(_AMENDMENT_PATH.relative_to(bundle_root)),
      str(Path(__file__).resolve().relative_to(bundle_root)),
      str(_TEST_PATH.relative_to(bundle_root)),
      str(_WRAPPER_PATH.relative_to(bundle_root)),
      str(_ORIGINAL_ANALYZER_PATH.relative_to(bundle_root)),
      str(_ORIGINAL_TEST_PATH.relative_to(bundle_root)),
      str(_ORIGINAL_FREEZE_PATH.relative_to(bundle_root)),
  }
  if not isinstance(inventory, Mapping) or set(inventory) != required:
    raise AmendmentError('v3.3.2.1 source inventory changed.')
  for relative, digest in inventory.items():
    path = (bundle_root / relative).resolve()
    try:
      path.relative_to(bundle_root)
    except ValueError as error:
      raise AmendmentError('v3.3.2.1 source inventory escaped repository.') from error
    _strict_regular(path, f'v3.3.2.1 source {relative}')
    if not _is_sha256(digest) or _sha256(path) != digest:
      raise AmendmentError(f'v3.3.2.1 source bytes changed: {relative}.')
    subprocess.run(
        ('git', '-C', str(bundle_root), 'ls-files', '--error-unmatch', relative),
        check=True, capture_output=True,
    )
  freeze_relative = str(_FREEZE_PATH.relative_to(bundle_root))
  try:
    subprocess.run(
        (
            'git', '-C', str(bundle_root), 'ls-files', '--error-unmatch',
            freeze_relative,
        ),
        check=True, capture_output=True,
    )
    committed_freeze = subprocess.check_output(
        ('git', '-C', str(bundle_root), 'show', f'HEAD:{freeze_relative}')
    )
  except subprocess.CalledProcessError as error:
    raise AmendmentError(
        'v3.3.2.1 analysis freeze is not committed at HEAD.'
    ) from error
  if hashlib.sha256(committed_freeze).hexdigest() != _sha256(_FREEZE_PATH):
    raise AmendmentError('v3.3.2.1 analysis freeze differs from its HEAD blob.')
  return freeze


def _validate_model_run_bundle(
    freeze: Mapping[str, Any], bundle_root: Path,
) -> dict[str, Any]:
  """Binds the live frozen files to their exact model-run commit bytes."""
  inventory = freeze['file_sha256']
  frozen_inventory = _read_object(
      _ORIGINAL_FREEZE_PATH, 'frozen v3.3.2 freeze'
  ).get('file_sha256')
  if not isinstance(frozen_inventory, Mapping) or len(frozen_inventory) != 75:
    raise AmendmentError('Frozen v3.3.2 75-file inventory changed.')
  for relative, digest in frozen_inventory.items():
    if not isinstance(relative, str) or not _is_sha256(digest):
      raise AmendmentError('Frozen v3.3.2 source inventory is malformed.')
    path = (bundle_root / relative).resolve()
    try:
      path.relative_to(bundle_root)
    except ValueError as error:
      raise AmendmentError('Frozen v3.3.2 source escaped repository.') from error
    _strict_regular(path, f'frozen v3.3.2 source {relative}')
    if _sha256(path) != digest:
      raise AmendmentError(f'Frozen v3.3.2 source bytes changed: {relative}.')
    historical = subprocess.check_output(
        ('git', '-C', str(bundle_root), 'show', f'{MODEL_RUN_COMMIT}:{relative}')
    )
    if hashlib.sha256(historical).hexdigest() != digest:
      raise AmendmentError(
          f'Model-run commit source bytes changed: {relative}.'
      )
  # Ensure this analyzer's own freeze binds, but does not redefine, the three
  # central frozen-v3.3.2 files.
  for relative, expected in (
      (str(_ORIGINAL_ANALYZER_PATH.relative_to(bundle_root)), V3_3_2_ANALYZER_SHA256),
      (str(_ORIGINAL_TEST_PATH.relative_to(bundle_root)), V3_3_2_TEST_SHA256),
      (str(_ORIGINAL_FREEZE_PATH.relative_to(bundle_root)), V3_3_2_FREEZE_SHA256),
  ):
    if inventory.get(relative) != expected:
      raise AmendmentError('v3.3.2.1 freeze lost a central v3.3.2 binding.')
  return {
      'model_run_commit': MODEL_RUN_COMMIT,
      'frozen_file_count': 75,
      'all_current_and_historical_bytes_exact': True,
  }


def _validate_same_process(
    same: Any, *, bootstrap: Mapping[str, Any], start: Mapping[str, Any],
) -> None:
  if not isinstance(same, Mapping) or set(same) != _SAME_PROCESS_KEYS:
    raise AmendmentError('Same-process preflight literal key set changed.')
  if same.get('pid') != bootstrap.get('pid') or same.get('pid') != BOOTSTRAP_PID:
    raise AmendmentError('Same-process preflight/bootstrap PID changed.')
  _v332._v33._validate_device_observation(  # pylint: disable=protected-access
      same, 'same-process preflight'
  )
  external = start.get('external_preflight')
  external_observation = (
      external.get('observation') if isinstance(external, Mapping) else None
  )
  if (
      not isinstance(external_observation, Mapping)
      or same.get('runtime_environment')
      != external_observation.get('runtime_environment')
  ):
    raise AmendmentError('Same-process runtime-environment binding changed.')
  executable = same.get('python_executable')
  if executable != EXPECTED_PYTHON_EXECUTABLE:
    raise AmendmentError('Same-process Python executable path is malformed.')
  # The captured venv launcher is a normal relative symlink (python->python3).
  # Bind its literal path, require its resolved target to remain inside the
  # exact venv bin directory, and require that target to be a regular file.
  path = Path(executable)
  target = path.resolve()
  if str(target) != EXPECTED_PYTHON_RESOLVED:
    raise AmendmentError('Same-process Python executable target changed.')
  _strict_regular(target, 'same-process resolved Python executable')


def _validate_external_tail(
    start: Mapping[str, Any], *, freeze: Mapping[str, Any], freeze_sha: str,
) -> str:
  external = start.get('external_preflight')
  if not isinstance(external, Mapping):
    raise AmendmentError('External preflight is absent.')
  expected_preflight_path = (_PREFLIGHT_DIR / 'preflight_0000.json').resolve()
  if external.get('path') != str(expected_preflight_path):
    raise AmendmentError('External preflight lexical path changed.')
  preflight_path = Path(str(external.get('path'))).resolve()
  _strict_regular(preflight_path, 'external preflight')
  if (
      preflight_path.parent != _PREFLIGHT_DIR.resolve()
      or _sha256(preflight_path) != external.get('sha256')
  ):
    raise AmendmentError('External preflight path/hash changed.')
  raw = _read_object(preflight_path, 'external preflight')
  embedded = {
      key: value for key, value in external.items()
      if key not in {'path', 'sha256', 'validated_logs'}
  }
  if raw != embedded:
    raise AmendmentError('External preflight embedded copy changed.')
  if (
      raw.get('script_version') != 'opensplice-device-preflight-v3.3.2'
      or raw.get('status') != 'pass'
      or raw.get('amendment_sha256') != V3_3_2_AMENDMENT_SHA256
      or raw.get('original_protocol_sha256') != ORIGINAL_PROTOCOL_SHA256
      or raw.get('freeze_sha256') != freeze_sha
      or raw.get('failure') is not None
      or raw.get('no_model_or_biological_access') is not True
      or raw.get('no_jit_or_array_kernel') is not True
  ):
    raise AmendmentError('External preflight contract changed.')
  observation = raw.get('observation')
  if not isinstance(observation, Mapping) or set(observation) != _EXTERNAL_OBSERVATION_KEYS:
    raise AmendmentError('External preflight observation key set changed.')
  _v332._v33._validate_device_observation(  # pylint: disable=protected-access
      observation, 'external preflight'
  )
  if (
      observation.get('python_executable') != EXPECTED_PYTHON_EXECUTABLE
      or
      observation.get('jax_enable_compilation_cache') is not False
      or observation.get('v3_3_runtime_environment') != start.get('runtime_environment')
      or observation.get('v3_3_2_runtime_environment') != start.get('runtime_environment')
  ):
    raise AmendmentError('External preflight derived runtime fields changed.')
  logs = raw.get('logs')
  validated = external.get('validated_logs')
  expected_logs = {}
  for stream in ('stdout', 'stderr'):
    row = logs.get(stream) if isinstance(logs, Mapping) else None
    if not isinstance(row, Mapping) or set(row) != {'path', 'sha256'}:
      raise AmendmentError(f'External preflight {stream} schema changed.')
    expected_path = (_PREFLIGHT_DIR / f'preflight_0000.{stream}.log').resolve()
    if row.get('path') != str(expected_path):
      raise AmendmentError(f'External preflight {stream} lexical path changed.')
    path = Path(str(row.get('path'))).resolve()
    _strict_regular(path, f'external preflight {stream}')
    if path.parent != _PREFLIGHT_DIR.resolve() or _sha256(path) != row.get('sha256'):
      raise AmendmentError(f'External preflight {stream} changed.')
    expected_logs[stream] = {'path': str(path), 'sha256': row['sha256']}
  if validated != expected_logs:
    raise AmendmentError('External preflight validated-log binding changed.')
  # The frozen runtime validator cross-checks every package/platform/GPU field.
  _v332._v33._validate_runtime_manifest(  # pylint: disable=protected-access
      start, _read_object(_v332._ORIGINAL_FREEZE_PATH, 'original v3.3 freeze')
  )
  return str(external['sha256'])


def _validate_freeze_and_start_repaired(
    run_dir: Path, *, bundle_root: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, str], dict[int, Any]]:
  """Runs the frozen prefix, repairs only the known same-process mismatch."""
  # The frozen model-run bundle records the then-current model-run commit.
  # This separately committed wrapper first proves current tracked cleanliness
  # and byte identity to that commit, then presents that historical HEAD only
  # to the frozen validator's immutable-run attestation comparison.
  real_check_output = subprocess.check_output

  def historical_check_output(command, *args, **kwargs):
    command_tuple = tuple(command)
    if (
        command_tuple[-2:] == ('rev-parse', 'HEAD')
        and '-C' in command_tuple
        and Path(command_tuple[command_tuple.index('-C') + 1]).resolve()
        == bundle_root.resolve()
    ):
      return MODEL_RUN_COMMIT if kwargs.get('text') else MODEL_RUN_COMMIT.encode()
    return real_check_output(command, *args, **kwargs)

  try:
    subprocess.check_output = historical_check_output
    try:
      _v332._validate_freeze_and_start(  # pylint: disable=protected-access
          run_dir, bundle_root=bundle_root
      )
    finally:
      subprocess.check_output = real_check_output
  except _v332.AnalysisError as error:
    if str(error) != 'Same-process preflight PID differs from bootstrap.':
      raise
  else:
    raise AmendmentError('Frozen analyzer no longer exhibits the bound schema defect.')

  freeze = _v332._read_json(_v332._FREEZE_PATH, 'v3.3.2 freeze')  # pylint: disable=protected-access
  freeze_sha = _sha256(_v332._FREEZE_PATH)  # pylint: disable=protected-access
  start = _read_object(run_dir / 'ATTEMPT_STARTED.json', 'ATTEMPT_STARTED')
  bootstrap = start.get('bootstrap')
  if not isinstance(bootstrap, Mapping):
    raise AmendmentError('START bootstrap is absent.')
  _validate_same_process(
      start.get('same_process_preflight'), bootstrap=bootstrap, start=start
  )
  external_sha = _validate_external_tail(
      start, freeze=freeze, freeze_sha=freeze_sha
  )

  # Reconstruct only the frozen return object.  All prerequisite gates above
  # were already executed byte-for-byte by the frozen validator before its
  # known same-process exception.
  v331_status = _v332._validate_v3_3_1_status(  # pylint: disable=protected-access
      freeze.get('v3_3_1_status')
  )
  original_audit, original_manifest, sequence_bindings = (
      _v332._validate_original_v3_3()  # pylint: disable=protected-access
  )
  compatibility = {
      'freeze': freeze,
      'same_process_pre_import_bootstrap': {
          'generated_bindings': bootstrap['generated_bindings']
      },
  }
  _, generated_audit = _v332._v331._validate_generated_bindings(  # pylint: disable=protected-access
      compatibility, freeze=freeze
  )
  cases = _v332._v33._load_cases()  # pylint: disable=protected-access
  original_freeze = _read_object(
      _v332._ORIGINAL_FREEZE_PATH, 'original v3.3 freeze'  # pylint: disable=protected-access
  )
  checkpoint_audit = _v332._v33._validate_checkpoint_reference(  # pylint: disable=protected-access
      start, original_freeze, cases
  )
  runtime_audit = _v332._v33._validate_runtime_manifest(  # pylint: disable=protected-access
      start, original_freeze
  )
  return (
      freeze,
      freeze_sha,
      {
          'git_head': MODEL_RUN_COMMIT,
          'model_run_commit_attestation_replayed_after_exact_byte_audit': True,
          'generated_binding_audit': generated_audit,
          'original_run_audit': original_audit,
          'v3_3_1_status': v331_status,
          'checkpoint_reference_audit': checkpoint_audit,
          'runtime_audit': runtime_audit,
          'external_preflight_sha256': external_sha,
          'same_process_exact_rtx3090_uuid_gate': True,
          'external_exact_rtx3090_uuid_gate': True,
      },
      original_manifest,
      sequence_bindings,
  )


@contextmanager
def _repaired_validator() -> Iterator[None]:
  original = _v332._validate_freeze_and_start  # pylint: disable=protected-access
  _v332._validate_freeze_and_start = _validate_freeze_and_start_repaired  # pylint: disable=protected-access
  try:
    yield
  finally:
    _v332._validate_freeze_and_start = original  # pylint: disable=protected-access


def _validate_amendment_preconditions(
    run_dir: Path, bundle_root: Path, *,
    expected_attempt_started_sha256: str | None = None,
) -> dict[str, Any]:
  _assert_cpu_only('v3.3.2.1 precondition process')
  bundle_root = bundle_root.resolve()
  if bundle_root != _REPO_ROOT.resolve() or run_dir.resolve() != _RUN_DIR.resolve():
    raise AmendmentError('Repository/run path differs from v3.3.2.1 contract.')
  if _ANALYSIS_DIR.exists():
    raise FileExistsError('v3.3.2.1 analysis destination already exists.')
  if expected_attempt_started_sha256 is None and _ATTEMPT_DIR.exists():
    raise FileExistsError('The append-only v3.3.2.1 attempt was already consumed.')
  if _sha256(_AMENDMENT_PATH) != AMENDMENT_SHA256:
    raise AmendmentError('v3.3.2.1 amendment bytes changed.')
  if _sha256(_ORIGINAL_ANALYZER_PATH) != V3_3_2_ANALYZER_SHA256:
    raise AmendmentError('Frozen v3.3.2 analyzer changed.')
  if _sha256(_ORIGINAL_TEST_PATH) != V3_3_2_TEST_SHA256:
    raise AmendmentError('Frozen v3.3.2 analyzer test changed.')
  if _sha256(_ORIGINAL_FREEZE_PATH) != V3_3_2_FREEZE_SHA256:
    raise AmendmentError('Frozen v3.3.2 freeze changed.')
  freeze = _validate_freeze(bundle_root)
  head = _assert_global_tracked_clean(bundle_root)
  model_run_bundle = _validate_model_run_bundle(freeze, bundle_root)
  run_audit = _walk_exact_tree(
      run_dir, _RUN_FILES, {'.', 'compiler', 'compiler/eight_row'},
      'immutable v3.3.2 run',
  )
  if run_audit['file_count'] != RUN_FILE_COUNT or run_audit['tree_sha256'] != RUN_TREE_SHA256:
    raise AmendmentError('Immutable v3.3.2 run tree digest changed.')
  compiler_paths = [
      run_dir / relative for relative in _RUN_FILES
      if relative.startswith('compiler/')
  ]
  compiler_tree = _tree_digest(compiler_paths, run_dir)
  if len(compiler_paths) != COMPILER_FILE_COUNT or compiler_tree != COMPILER_TREE_SHA256:
    raise AmendmentError('Immutable v3.3.2 compiler tree digest changed.')
  preflight_audit = _walk_exact_tree(
      _PREFLIGHT_DIR, _PREFLIGHT_FILES, {'.'}, 'immutable v3.3.2 preflight'
  )
  if (
      preflight_audit['file_count'] != PREFLIGHT_FILE_COUNT
      or preflight_audit['tree_sha256'] != PREFLIGHT_TREE_SHA256
  ):
    raise AmendmentError('Immutable v3.3.2 preflight tree digest changed.')
  if (run_dir / 'raw').exists():
    raise AmendmentError('Zero-apply compiler stop unexpectedly contains raw data.')
  binding = {
      'analysis_version': ANALYSIS_VERSION,
      'git_head': head,
      'tracked_head_clean': True,
      'amendment_sha256': AMENDMENT_SHA256,
      'analysis_freeze_sha256': _sha256(_FREEZE_PATH),
      'analysis_freeze': freeze,
      'model_run_bundle': model_run_bundle,
      'immutable_run': run_audit,
      'compiler_file_count': COMPILER_FILE_COUNT,
      'compiler_tree_sha256': compiler_tree,
      'immutable_preflight': preflight_audit,
      'model_rerun_permitted': False,
      'model_apply_count': 0,
      'scientific_values_read': False,
      'confirmation_model_calls_permitted': 0,
  }
  if expected_attempt_started_sha256 is not None:
    _validate_started_attempt(binding, expected_attempt_started_sha256)
  _assert_cpu_only('v3.3.2.1 completed precondition process')
  return json.loads(json.dumps(binding, sort_keys=True, allow_nan=False))


def _validate_started_attempt(binding: Mapping[str, Any], digest: str) -> None:
  _strict_directory(_ATTEMPT_DIR, 'v3.3.2.1 attempt directory')
  entries = list(_ATTEMPT_DIR.iterdir())
  if len(entries) != 1 or entries[0].name != 'ANALYSIS_ATTEMPT_STARTED.json':
    raise AmendmentError('v3.3.2.1 attempt has extra/terminal artifacts.')
  if not _is_sha256(digest) or _sha256(entries[0]) != digest:
    raise AmendmentError('v3.3.2.1 attempt-start bytes changed.')
  value = _read_object(entries[0], 'v3.3.2.1 attempt start')
  if (
      value.get('analysis_version') != ANALYSIS_VERSION
      or value.get('status') != 'started_append_only_one_shot'
      or value.get('reason') != AMENDMENT_REASON
      or value.get('amendment_binding') != binding
      or value.get('run_dir') != str(_RUN_DIR.resolve())
      or value.get('output_json') != str((_ANALYSIS_DIR / 'ANALYSIS.json').resolve())
      or value.get('output_markdown') != str((_ANALYSIS_DIR / 'RESULT.md').resolve())
      or value.get('model_rerun_permitted') is not False
      or value.get('scientific_values_read_before_attempt') is not False
      or value.get('confirmation_model_calls_permitted') != 0
  ):
    raise AmendmentError('v3.3.2.1 attempt-start content changed.')


def _assert_no_scientific_payload(value: Any, path: str = 'result') -> None:
  if isinstance(value, Mapping):
    for key, child in value.items():
      lowered = str(key).lower()
      if any(token in lowered for token in _FORBIDDEN_RESULT_KEYS):
        # Frozen structural booleans/nulls are permitted; payloads are not.
        if child not in (False, None):
          raise AmendmentError(f'Forbidden scientific payload at {path}.{key}.')
      _assert_no_scientific_payload(child, f'{path}.{key}')
  elif isinstance(value, list):
    for index, child in enumerate(value):
      _assert_no_scientific_payload(child, f'{path}[{index}]')


def _enforce_exact_controlled_stop(result: Mapping[str, Any]) -> None:
  stop = result.get('controlled_stop')
  sidecar = result.get('sidecar_audit')
  compiler = result.get('provenance_audit', {}).get('compiler')
  if (
      result.get('analysis_version') != _v332.ANALYSIS_VERSION
      or result.get('status') != 'complete_controlled_stop_audited'
      or result.get('decision') != 'controlled_stop_compiler_graph_mismatch'
      or not isinstance(stop, Mapping)
      or stop.get('reason') != 'compiler_graph_mismatch'
      or stop.get('audited_record_count') != 0
      or result.get('scientific_summary_computed') is not False
      or result.get('shapley_or_nomination_computed') is not False
      or result.get('nomination_performed') is not False
      or result.get('nomination') is not None
      or result.get('resolution_analysis') is not None
      or result.get('combined_analysis_permitted') is not False
      or not isinstance(sidecar, Mapping)
      or sidecar.get('audited_record_count') != 0
      or sidecar.get('valid_record_count') != 0
      or sidecar.get('invalid_record_count') != 0
      or sidecar.get('audited_model_apply_count') != 0
      or sidecar.get('raw_artifact_count') != 0
      or sidecar.get('id0_all20') is not False
      or sidecar.get('id255_all20') is not False
      or not isinstance(compiler, Mapping)
      or compiler.get('graph_and_hlo_exact_to_original_v3_3') is not False
  ):
    raise AmendmentError('Delegated result exceeds the exact zero-apply stop.')
  _assert_no_scientific_payload(result)


def analyze(
    run_dir: Path, *, bundle_root: Path = _REPO_ROOT,
    amendment_binding: Mapping[str, Any] | None = None,
    attempt_started_sha256: str | None = None,
) -> dict[str, Any]:
  _assert_cpu_only('v3.3.2.1 pre-analysis process')
  verified = _validate_amendment_preconditions(
      run_dir, bundle_root,
      expected_attempt_started_sha256=attempt_started_sha256,
  )
  if amendment_binding != verified or not _is_sha256(attempt_started_sha256):
    raise AmendmentError('v3.3.2.1 append-only binding is absent or changed.')
  with _repaired_validator():
    result = _v332.analyze(run_dir, bundle_root=bundle_root)
  _assert_cpu_only('v3.3.2.1 post-analysis process')
  _enforce_exact_controlled_stop(result)
  result = dict(result)
  result['analysis_version'] = ANALYSIS_VERSION
  result['analyzer_amendment'] = {
      'reason': AMENDMENT_REASON,
      'amendment_sha256': AMENDMENT_SHA256,
      'amendment_binding': verified,
      'attempt_started_sha256': attempt_started_sha256,
      'frozen_analyzer_sha256': V3_3_2_ANALYZER_SHA256,
      'frozen_analyzer_failure_preserved': (
          'same-process preflight schema was validated as external schema'
      ),
      'repair_scope': 'preflight_schema_only',
      'model_rerun_permitted': False,
      'model_apply_count': 0,
      'scientific_summary_computed': False,
      'shapley_or_nomination_computed': False,
      'combined_analysis_permitted': False,
  }
  _assert_no_scientific_payload(result)
  return result


def render_markdown(result: Mapping[str, Any]) -> str:
  return '\n'.join((
      '# OpenSplice v3.3.2.1 controlled-stop structural archive', '',
      f"**Decision:** `{result['decision']}`", '',
      'The frozen v3.3.2 attempt stopped at the compiler-identity gate before ',
      'any model apply. StableHLO and pre-backend HLO matched the frozen ',
      'v3.3 executable, but the fresh backend-compiled HLO was not byte-',
      'identical, so the prospective gate correctly stopped the run.', '',
      'This sidecar produced **no biological evidence**: zero endpoint records, ',
      'zero movements, zero model applies, and no ID-0 or ID-255 completion.', '',
      '**Scientific summary computed:** no  ',
      '**Shapley, interaction, resolution, or nomination computed:** no  ',
      '**Combined analysis permitted:** no', '',
      'The v3.3.2.1 repair distinguishes the literal same-process preflight ',
      'schema from the augmented external-preflight schema. It changes no ',
      'scientific gate, estimator, or model artifact.', '',
      'Confirmation model outputs, activations, and interventions remained ',
      'unopened. Previously disclosed later-exon metadata/labels remain within ',
      'the recorded scope disclosure.', '',
  ))


def _write_new_json(path: Path, value: Mapping[str, Any]) -> str:
  payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
  descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
  try:
    with os.fdopen(descriptor, 'wb') as handle:
      handle.write(payload)
      handle.flush()
      os.fsync(handle.fileno())
  except BaseException:
    path.unlink(missing_ok=True)
    raise
  return hashlib.sha256(payload).hexdigest()


def _start_attempt(binding: Mapping[str, Any]) -> str:
  _ATTEMPT_DIR.mkdir(mode=0o755, parents=False, exist_ok=False)
  return _write_new_json(_ATTEMPT_DIR / 'ANALYSIS_ATTEMPT_STARTED.json', {
      'analysis_version': ANALYSIS_VERSION,
      'status': 'started_append_only_one_shot',
      'reason': AMENDMENT_REASON,
      'started_at_unix_s': time.time(),
      'run_dir': str(_RUN_DIR.resolve()),
      'output_json': str((_ANALYSIS_DIR / 'ANALYSIS.json').resolve()),
      'output_markdown': str((_ANALYSIS_DIR / 'RESULT.md').resolve()),
      'amendment_binding': dict(binding),
      'model_rerun_permitted': False,
      'scientific_values_read_before_attempt': False,
      'confirmation_model_calls_permitted': 0,
  })


def _persist_terminal(filename: str, value: Mapping[str, Any], started_sha: str) -> None:
  _write_new_json(_ATTEMPT_DIR / filename, {
      'analysis_version': ANALYSIS_VERSION,
      'attempt_started_sha256': started_sha,
      'recorded_at_unix_s': time.time(),
      **value,
  })


def _write_outputs(result: Mapping[str, Any]) -> dict[str, Any]:
  if _ANALYSIS_DIR.exists():
    raise FileExistsError('v3.3.2.1 output directory already exists.')
  _ANALYSIS_DIR.mkdir(mode=0o755, parents=False, exist_ok=False)
  json_path = _ANALYSIS_DIR / 'ANALYSIS.json'
  markdown_path = _ANALYSIS_DIR / 'RESULT.md'
  json_sha = _write_new_json(json_path, result)
  markdown = render_markdown(result).encode('utf-8')
  descriptor = os.open(markdown_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
  with os.fdopen(descriptor, 'wb') as handle:
    handle.write(markdown)
    handle.flush()
    os.fsync(handle.fileno())
  return {
      'analysis_json_sha256': json_sha,
      'analysis_markdown_sha256': hashlib.sha256(markdown).hexdigest(),
      'analysis_file_count': 2,
      'analysis_tree_sha256': _tree_digest([json_path, markdown_path], _ANALYSIS_DIR),
  }


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run-dir', type=Path, required=True)
  parser.add_argument('--bundle-root', type=Path, required=True)
  parser.add_argument('--output-json', type=Path, required=True)
  parser.add_argument('--output-markdown', type=Path, required=True)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if args.run_dir.resolve() != _RUN_DIR.resolve():
    raise AmendmentError('CLI run path differs from immutable v3.3.2 run.')
  if args.bundle_root.resolve() != _REPO_ROOT.resolve():
    raise AmendmentError('CLI repository path differs from v3.3.2.1 freeze.')
  if args.output_json.resolve() != (_ANALYSIS_DIR / 'ANALYSIS.json').resolve():
    raise AmendmentError('CLI JSON output differs from append-only destination.')
  if args.output_markdown.resolve() != (_ANALYSIS_DIR / 'RESULT.md').resolve():
    raise AmendmentError('CLI Markdown output differs from append-only destination.')
  binding = _validate_amendment_preconditions(args.run_dir, args.bundle_root)
  started_sha = _start_attempt(binding)
  try:
    result = analyze(
        args.run_dir, bundle_root=args.bundle_root,
        amendment_binding=binding, attempt_started_sha256=started_sha,
    )
    output = _write_outputs(result)
    _persist_terminal('ANALYSIS_COMPLETE.json', {
        'status': 'complete_controlled_stop_audited',
        'decision': result['decision'],
        **output,
        'model_apply_count': 0,
        'scientific_summary_computed': False,
        'shapley_or_nomination_computed': False,
        'combined_analysis_permitted': False,
    }, started_sha)
  except BaseException as error:
    try:
      _persist_terminal('ANALYSIS_FAILURE.json', {
          'status': 'failed_consumed_no_retry',
          'failure': {
              'type': type(error).__name__, 'message': str(error),
              'traceback': traceback.format_exc(),
          },
          'analysis_dir_exists': _ANALYSIS_DIR.exists(),
          'model_apply_count': 0,
          'scientific_summary_computed': False,
          'shapley_or_nomination_computed': False,
          'combined_analysis_permitted': False,
      }, started_sha)
    except BaseException:
      pass
    raise
  print((_ANALYSIS_DIR / 'ANALYSIS.json').resolve())


if __name__ == '__main__':
  main()
