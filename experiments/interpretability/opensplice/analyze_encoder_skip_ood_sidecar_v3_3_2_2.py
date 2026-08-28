#!/usr/bin/env python3
"""CPU-only v3.3.2.2 archive using a saved frozen validator callable.

The v3.3.2 and v3.3.2.1 bundles and both consumed attempts remain immutable.
This version changes only the Python function-reference control flow: the
frozen original validator is captured before patching, invoked directly, and
restored by exact identity in a finally block.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import traceback
from typing import Any, Iterator, Mapping


ANALYSIS_VERSION = 'opensplice-encoder-skip-ood-sidecar-analysis-v3.3.2.2'
AMENDMENT_REASON = 'saved_frozen_validator_function_reference_repair'
AMENDMENT_SHA256 = (
    '3188b44f85d8315eb4a099b42930b2d08f76074ffb190ef11b67a9f39e788a3d'
)
AMENDMENT_COMMIT = '2a2cc59136f5b83f3a7c265b5197e30cdecd7c11'
V3_3_2_1_COMMIT = 'b43051aa4a893e24a38e932900d349278c9ead88'
V3_3_2_1_ANALYZER_SHA256 = (
    '35db9ca198cb5d7f03621ccf322ea116f98cea3bfdc711006dfd20bc809e8048'
)
V3_3_2_1_TEST_SHA256 = (
    'a8733f3ffb35920dda2f6a856076cbe82a9e90aa2fe483169150dbad4421a1b8'
)
V3_3_2_1_FREEZE_SHA256 = (
    '3871ab41b16105a94673e89381d32d7253b014c64eda5c6789eaecf16477c061'
)
V3_3_2_1_SHELL_SHA256 = (
    'ea5cce6ae631ba3fa2bf0082d691d0896ef0fed7b20f0d908034e17775060caa'
)
V3_3_2_1_AMENDMENT_SHA256 = (
    '81a4f4c126b83225b02c7de5cf0dc6fd0baf6085b84b9ed5dd7a3677744090ba'
)
FAILED_START_SHA256 = (
    'a87c4e15ed67a363d07c434ca232540687950d145e67492b9ed9c17d9adebf1d'
)
FAILED_TERMINAL_SHA256 = (
    '1cd933623ecdfb328d5db458b16df909e632a560361dbb547f83c22cf13ab7c7'
)
FAILED_ATTEMPT_TREE_SHA256 = (
    '5e97b191e781c5141d2f308deefacfa8f6a196449fd7e11f36c22828a13f036a'
)
KNOWN_SCHEMA_ERROR = 'Same-process preflight PID differs from bootstrap.'

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
_AMENDMENT_PATH = (
    _HERE / 'v3_wider_mechanism/'
    'encoder_skip_ood_sidecar_analysis_amendment_v3_3_2_2.md'
)
_V3_3_2_1_ANALYZER_PATH = (
    _HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_2_1.py'
)
_V3_3_2_1_TEST_PATH = (
    _HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_2_1_test.py'
)
_V3_3_2_1_FREEZE_PATH = (
    _HERE / 'encoder_skip_ood_sidecar_analysis_v3_3_2_1_freeze.json'
)
_V3_3_2_1_SHELL_PATH = (
    _HERE / 'run_encoder_skip_ood_sidecar_analysis_v3_3_2_1.sh'
)
_V3_3_2_1_AMENDMENT_PATH = (
    _HERE / 'v3_wider_mechanism/'
    'encoder_skip_ood_sidecar_analysis_amendment_v3_3_2_1.md'
)
_V3_3_2_1_ATTEMPT_DIR = (
    _HERE / 'results/'
    'v3_3_2_development_ood_sidecar_analysis_v3_3_2_1_attempt'
)
_V3_3_2_1_ANALYSIS_DIR = (
    _HERE / 'results/v3_3_2_development_ood_sidecar_analysis_v3_3_2_1'
)
_V3_3_2_ANALYSIS_DIR = (
    _HERE / 'results/v3_3_2_development_ood_sidecar_analysis'
)
_FREEZE_PATH = (
    _HERE / 'encoder_skip_ood_sidecar_analysis_v3_3_2_2_freeze.json'
)
_TEST_PATH = _HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_2_2_test.py'
_WRAPPER_PATH = _HERE / 'run_encoder_skip_ood_sidecar_analysis_v3_3_2_2.sh'
_ATTEMPT_DIR = (
    _HERE / 'results/'
    'v3_3_2_development_ood_sidecar_analysis_v3_3_2_2_attempt'
)
_ANALYSIS_DIR = (
    _HERE / 'results/v3_3_2_development_ood_sidecar_analysis_v3_3_2_2'
)
_FAILED_ATTEMPT_FILES = {
    'ANALYSIS_ATTEMPT_STARTED.json': (8_616, FAILED_START_SHA256),
    'ANALYSIS_FAILURE.json': (2_163, FAILED_TERMINAL_SHA256),
}


class AmendmentError(RuntimeError):
  """Raised when the saved-validator amendment fails closed."""


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
    raise AmendmentError(f'{label} is symlinked or not regular: {path}.')


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


_v3321 = _load_cpu_module(
    _V3_3_2_1_ANALYZER_PATH, V3_3_2_1_ANALYZER_SHA256,
    '_opensplice_frozen_ood_sidecar_analyzer_v3_3_2_1',
)
_v332 = _v3321._v332  # pylint: disable=protected-access
# Capture once, immediately after exact-byte import and before any patch.
_SAVED_FROZEN_VALIDATOR = _v332._validate_freeze_and_start  # pylint: disable=protected-access
if _SAVED_FROZEN_VALIDATOR is _v3321._validate_freeze_and_start_repaired:  # pylint: disable=protected-access
  raise AmendmentError('Saved validator unexpectedly refers to repaired v3.3.2.1.')


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
    digest.update(path.relative_to(root).as_posix().encode())
    digest.update(b'\0')
    digest.update(bytes.fromhex(_sha256(path)))
  return digest.hexdigest()


def _walk_exact_flat_attempt(
    directory: Path, files: Mapping[str, tuple[int, str]], label: str,
) -> dict[str, Any]:
  _strict_directory(directory, label)
  entries = list(directory.iterdir())
  if any(
      path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode)
      for path in entries
  ):
    raise AmendmentError(f'{label} contains a directory/symlink/special entry.')
  if {path.name for path in entries} != set(files):
    raise AmendmentError(f'{label} file membership changed.')
  for relative, (size, digest) in files.items():
    path = directory / relative
    if path.stat().st_size != size or _sha256(path) != digest:
      raise AmendmentError(f'{label} bytes changed: {relative}.')
  return {
      'file_count': len(entries),
      'tree_sha256': _tree_digest(entries, directory),
      'files': {
          relative: {'size_bytes': size, 'sha256': digest}
          for relative, (size, digest) in files.items()
      },
  }


def _validate_consumed_v3_3_2_1() -> dict[str, Any]:
  audit = _walk_exact_flat_attempt(
      _V3_3_2_1_ATTEMPT_DIR, _FAILED_ATTEMPT_FILES,
      'consumed v3.3.2.1 attempt',
  )
  if audit['tree_sha256'] != FAILED_ATTEMPT_TREE_SHA256:
    raise AmendmentError('Consumed v3.3.2.1 attempt tree changed.')
  if _V3_3_2_1_ANALYSIS_DIR.exists() or _V3_3_2_ANALYSIS_DIR.exists():
    raise AmendmentError('A forbidden prior analysis output now exists.')
  start = _read_object(
      _V3_3_2_1_ATTEMPT_DIR / 'ANALYSIS_ATTEMPT_STARTED.json',
      'consumed v3.3.2.1 START',
  )
  failure = _read_object(
      _V3_3_2_1_ATTEMPT_DIR / 'ANALYSIS_FAILURE.json',
      'consumed v3.3.2.1 FAILURE',
  )
  if set(start) != {
      'amendment_binding', 'analysis_version',
      'confirmation_model_calls_permitted', 'model_rerun_permitted',
      'output_json', 'output_markdown', 'reason', 'run_dir',
      'scientific_values_read_before_attempt', 'started_at_unix_s', 'status',
  }:
    raise AmendmentError('Consumed v3.3.2.1 START schema changed.')
  binding = start.get('amendment_binding')
  if (
      start.get('analysis_version') != _v3321.ANALYSIS_VERSION
      or start.get('status') != 'started_append_only_one_shot'
      or start.get('reason') != _v3321.AMENDMENT_REASON
      or start.get('run_dir') != str(_v3321._RUN_DIR.resolve())  # pylint: disable=protected-access
      or start.get('output_json')
      != str((_V3_3_2_1_ANALYSIS_DIR / 'ANALYSIS.json').resolve())
      or start.get('output_markdown')
      != str((_V3_3_2_1_ANALYSIS_DIR / 'RESULT.md').resolve())
      or start.get('model_rerun_permitted') is not False
      or start.get('scientific_values_read_before_attempt') is not False
      or start.get('confirmation_model_calls_permitted') != 0
      or not isinstance(start.get('started_at_unix_s'), (int, float))
      or not isinstance(binding, Mapping)
      or binding.get('git_head') != V3_3_2_1_COMMIT
      or binding.get('tracked_head_clean') is not True
      or binding.get('model_apply_count') != 0
      or binding.get('scientific_values_read') is not False
      or binding.get('analysis_freeze_sha256') != V3_3_2_1_FREEZE_SHA256
      or binding.get('immutable_run', {}).get('tree_sha256')
      != _v3321.RUN_TREE_SHA256
      or binding.get('compiler_tree_sha256') != _v3321.COMPILER_TREE_SHA256
      or binding.get('immutable_preflight', {}).get('tree_sha256')
      != _v3321.PREFLIGHT_TREE_SHA256
      or binding.get('model_run_bundle', {}).get('frozen_file_count') != 75
      or binding.get('model_run_bundle', {}).get(
          'all_current_and_historical_bytes_exact'
      ) is not True
  ):
    raise AmendmentError('Consumed v3.3.2.1 START content changed.')
  if set(failure) != {
      'analysis_dir_exists', 'analysis_version', 'attempt_started_sha256',
      'combined_analysis_permitted', 'failure', 'model_apply_count',
      'recorded_at_unix_s', 'scientific_summary_computed',
      'shapley_or_nomination_computed', 'status',
  }:
    raise AmendmentError('Consumed v3.3.2.1 FAILURE schema changed.')
  failure_row = failure.get('failure')
  if (
      failure.get('analysis_version') != _v3321.ANALYSIS_VERSION
      or failure.get('status') != 'failed_consumed_no_retry'
      or failure.get('attempt_started_sha256') != FAILED_START_SHA256
      or failure.get('analysis_dir_exists') is not False
      or failure.get('model_apply_count') != 0
      or failure.get('scientific_summary_computed') is not False
      or failure.get('shapley_or_nomination_computed') is not False
      or failure.get('combined_analysis_permitted') is not False
      or not isinstance(failure.get('recorded_at_unix_s'), (int, float))
      or not isinstance(failure_row, Mapping)
      or set(failure_row) != {'type', 'message', 'traceback'}
      or failure_row.get('type') != 'RecursionError'
      or failure_row.get('message') != 'maximum recursion depth exceeded'
      or not isinstance(failure_row.get('traceback'), str)
      or '[Previous line repeated 993 more times]' not in failure_row['traceback']
      or not failure_row['traceback'].endswith(
          'RecursionError: maximum recursion depth exceeded\n'
      )
  ):
    raise AmendmentError('Consumed v3.3.2.1 failure content changed.')
  return {
      **audit,
      'start_sha256': FAILED_START_SHA256,
      'failure_sha256': FAILED_TERMINAL_SHA256,
      'exact_recursion_traceback_bound_by_file_hash': True,
      'analysis_output_absent': True,
      'model_apply_count': 0,
      'scientific_values_read': False,
  }


def _validate_freeze(bundle_root: Path) -> dict[str, Any]:
  freeze = _read_object(_FREEZE_PATH, 'v3.3.2.2 freeze')
  exact_keys = {
      'analysis_version', 'amendment_path', 'amendment_sha256',
      'amendment_commit', 'run_dir', 'preflight_dir', 'attempt_dir',
      'analysis_dir', 'v3_3_2_1_commit', 'v3_3_2_1_attempt_dir',
      'v3_3_2_1_attempt_file_count', 'v3_3_2_1_attempt_tree_sha256',
      'v3_3_2_1_start_sha256', 'v3_3_2_1_failure_sha256',
      'file_sha256',
  }
  if set(freeze) != exact_keys:
    raise AmendmentError('v3.3.2.2 freeze schema changed.')
  expected = {
      'analysis_version': ANALYSIS_VERSION,
      'amendment_path': str(_AMENDMENT_PATH.resolve()),
      'amendment_sha256': AMENDMENT_SHA256,
      'amendment_commit': AMENDMENT_COMMIT,
      'run_dir': str(_v3321._RUN_DIR.resolve()),  # pylint: disable=protected-access
      'preflight_dir': str(_v3321._PREFLIGHT_DIR.resolve()),  # pylint: disable=protected-access
      'attempt_dir': str(_ATTEMPT_DIR.resolve()),
      'analysis_dir': str(_ANALYSIS_DIR.resolve()),
      'v3_3_2_1_commit': V3_3_2_1_COMMIT,
      'v3_3_2_1_attempt_dir': str(_V3_3_2_1_ATTEMPT_DIR.resolve()),
      'v3_3_2_1_attempt_file_count': 2,
      'v3_3_2_1_attempt_tree_sha256': FAILED_ATTEMPT_TREE_SHA256,
      'v3_3_2_1_start_sha256': FAILED_START_SHA256,
      'v3_3_2_1_failure_sha256': FAILED_TERMINAL_SHA256,
  }
  for key, value in expected.items():
    if freeze.get(key) != value:
      raise AmendmentError(f'v3.3.2.2 freeze changed at {key}.')
  inventory = freeze.get('file_sha256')
  required = {
      str(path.relative_to(bundle_root)) for path in (
          _AMENDMENT_PATH, Path(__file__).resolve(), _TEST_PATH, _WRAPPER_PATH,
          _V3_3_2_1_ANALYZER_PATH, _V3_3_2_1_TEST_PATH,
          _V3_3_2_1_FREEZE_PATH, _V3_3_2_1_SHELL_PATH,
          _V3_3_2_1_AMENDMENT_PATH,
          _v3321._ORIGINAL_ANALYZER_PATH,  # pylint: disable=protected-access
          _v3321._ORIGINAL_TEST_PATH,  # pylint: disable=protected-access
          _v3321._ORIGINAL_FREEZE_PATH,  # pylint: disable=protected-access
      )
  }
  if not isinstance(inventory, Mapping) or set(inventory) != required:
    raise AmendmentError('v3.3.2.2 source inventory changed.')
  for relative, digest in inventory.items():
    path = (bundle_root / relative).resolve()
    try:
      path.relative_to(bundle_root)
    except ValueError as error:
      raise AmendmentError('v3.3.2.2 source escaped repository.') from error
    _strict_regular(path, f'v3.3.2.2 source {relative}')
    if not _is_sha256(digest) or _sha256(path) != digest:
      raise AmendmentError(f'v3.3.2.2 source bytes changed: {relative}.')
    subprocess.run(
        ('git', '-C', str(bundle_root), 'ls-files', '--error-unmatch', relative),
        check=True, capture_output=True,
    )
  freeze_relative = str(_FREEZE_PATH.relative_to(bundle_root))
  try:
    subprocess.run(
        ('git', '-C', str(bundle_root), 'ls-files', '--error-unmatch', freeze_relative),
        check=True, capture_output=True,
    )
    committed = subprocess.check_output(
        ('git', '-C', str(bundle_root), 'show', f'HEAD:{freeze_relative}')
    )
  except subprocess.CalledProcessError as error:
    raise AmendmentError('v3.3.2.2 freeze is not committed at HEAD.') from error
  if hashlib.sha256(committed).hexdigest() != _sha256(_FREEZE_PATH):
    raise AmendmentError('v3.3.2.2 freeze differs from its HEAD blob.')
  return freeze


def _assert_global_tracked_clean(bundle_root: Path) -> str:
  subprocess.run(
      ('git', '-C', str(bundle_root), 'diff', '--binary', '--exit-code', 'HEAD', '--'),
      check=True, capture_output=True,
  )
  return subprocess.check_output(
      ('git', '-C', str(bundle_root), 'rev-parse', 'HEAD'), text=True
  ).strip()


def _validate_v3_3_2_1_historical_bundle(bundle_root: Path) -> dict[str, Any]:
  rows = (
      (_V3_3_2_1_ANALYZER_PATH, V3_3_2_1_ANALYZER_SHA256),
      (_V3_3_2_1_TEST_PATH, V3_3_2_1_TEST_SHA256),
      (_V3_3_2_1_FREEZE_PATH, V3_3_2_1_FREEZE_SHA256),
      (_V3_3_2_1_SHELL_PATH, V3_3_2_1_SHELL_SHA256),
      (_V3_3_2_1_AMENDMENT_PATH, V3_3_2_1_AMENDMENT_SHA256),
  )
  verified = {}
  for path, expected in rows:
    relative = str(path.relative_to(bundle_root))
    try:
      historical = subprocess.check_output(
          (
              'git', '-C', str(bundle_root), 'show',
              f'{V3_3_2_1_COMMIT}:{relative}',
          )
      )
    except subprocess.CalledProcessError as error:
      raise AmendmentError(
          f'v3.3.2.1 historical blob is absent: {relative}.'
      ) from error
    if hashlib.sha256(historical).hexdigest() != expected:
      raise AmendmentError(
          f'v3.3.2.1 historical blob bytes changed: {relative}.'
      )
    verified[relative] = expected
  return {
      'commit': V3_3_2_1_COMMIT,
      'file_count': 5,
      'current_and_historical_bytes_exact': True,
      'file_sha256': verified,
  }


def _call_saved_frozen_validator(run_dir: Path, *, bundle_root: Path) -> None:
  """Calls the captured object once under the frozen historical-HEAD shim."""
  real_check_output = subprocess.check_output

  def historical_check_output(command, *args, **kwargs):
    command_tuple = tuple(command)
    if (
        command_tuple[-2:] == ('rev-parse', 'HEAD')
        and '-C' in command_tuple
        and Path(command_tuple[command_tuple.index('-C') + 1]).resolve()
        == bundle_root.resolve()
    ):
      value = _v3321.MODEL_RUN_COMMIT
      return value if kwargs.get('text') else value.encode()
    return real_check_output(command, *args, **kwargs)

  try:
    subprocess.check_output = historical_check_output
    try:
      _SAVED_FROZEN_VALIDATOR(run_dir, bundle_root=bundle_root)
    finally:
      subprocess.check_output = real_check_output
  except _v332.AnalysisError as error:
    if str(error) != KNOWN_SCHEMA_ERROR:
      raise
  else:
    raise AmendmentError('Frozen validator no longer exhibits the bound schema defect.')


def _reconstruct_frozen_return(
    run_dir: Path, *, bundle_root: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, str], dict[int, Any]]:
  freeze = _v332._read_json(_v332._FREEZE_PATH, 'v3.3.2 freeze')  # pylint: disable=protected-access
  freeze_sha = _sha256(_v332._FREEZE_PATH)  # pylint: disable=protected-access
  start = _read_object(run_dir / 'ATTEMPT_STARTED.json', 'ATTEMPT_STARTED')
  bootstrap = start.get('bootstrap')
  if not isinstance(bootstrap, Mapping):
    raise AmendmentError('START bootstrap is absent.')
  _v3321._validate_same_process(  # pylint: disable=protected-access
      start.get('same_process_preflight'), bootstrap=bootstrap, start=start
  )
  external_sha = _v3321._validate_external_tail(  # pylint: disable=protected-access
      start, freeze=freeze, freeze_sha=freeze_sha
  )
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
          'git_head': _v3321.MODEL_RUN_COMMIT,
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


def _validate_freeze_and_start_repaired(
    run_dir: Path, *, bundle_root: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, str], dict[int, Any]]:
  _call_saved_frozen_validator(run_dir, bundle_root=bundle_root)
  return _reconstruct_frozen_return(run_dir, bundle_root=bundle_root)


@contextmanager
def _scoped_repair() -> Iterator[None]:
  if _v332._validate_freeze_and_start is not _SAVED_FROZEN_VALIDATOR:  # pylint: disable=protected-access
    raise AmendmentError('Frozen validator was prepatched or concurrently changed.')
  _v332._validate_freeze_and_start = _validate_freeze_and_start_repaired  # pylint: disable=protected-access
  observed_error: AmendmentError | None = None
  try:
    yield
  finally:
    if _v332._validate_freeze_and_start is not _validate_freeze_and_start_repaired:  # pylint: disable=protected-access
      observed_error = AmendmentError('Scoped validator was concurrently changed.')
    _v332._validate_freeze_and_start = _SAVED_FROZEN_VALIDATOR  # pylint: disable=protected-access
    if _v332._validate_freeze_and_start is not _SAVED_FROZEN_VALIDATOR:  # pylint: disable=protected-access
      observed_error = AmendmentError('Frozen validator identity was not restored.')
    if observed_error is not None:
      raise observed_error


def _validate_amendment_preconditions(
    run_dir: Path, bundle_root: Path, *,
    expected_attempt_started_sha256: str | None = None,
) -> dict[str, Any]:
  _assert_cpu_only('v3.3.2.2 precondition process')
  bundle_root = bundle_root.resolve()
  if bundle_root != _REPO_ROOT.resolve() or run_dir.resolve() != _v3321._RUN_DIR.resolve():  # pylint: disable=protected-access
    raise AmendmentError('Repository/run path differs from v3.3.2.2 contract.')
  if _ANALYSIS_DIR.exists():
    raise FileExistsError('v3.3.2.2 analysis output already exists.')
  if expected_attempt_started_sha256 is None and _ATTEMPT_DIR.exists():
    raise FileExistsError('The append-only v3.3.2.2 attempt was consumed.')
  if _sha256(_AMENDMENT_PATH) != AMENDMENT_SHA256:
    raise AmendmentError('v3.3.2.2 amendment bytes changed.')
  for path, digest, label in (
      (_V3_3_2_1_ANALYZER_PATH, V3_3_2_1_ANALYZER_SHA256, 'v3.3.2.1 analyzer'),
      (_V3_3_2_1_TEST_PATH, V3_3_2_1_TEST_SHA256, 'v3.3.2.1 test'),
      (_V3_3_2_1_FREEZE_PATH, V3_3_2_1_FREEZE_SHA256, 'v3.3.2.1 freeze'),
      (_V3_3_2_1_SHELL_PATH, V3_3_2_1_SHELL_SHA256, 'v3.3.2.1 shell'),
      (_V3_3_2_1_AMENDMENT_PATH, V3_3_2_1_AMENDMENT_SHA256, 'v3.3.2.1 amendment'),
  ):
    _strict_regular(path, label)
    if _sha256(path) != digest:
      raise AmendmentError(f'{label} bytes changed.')
  freeze = _validate_freeze(bundle_root)
  head = _assert_global_tracked_clean(bundle_root)
  historical_v3321 = _validate_v3_3_2_1_historical_bundle(bundle_root)
  prior_failure = _validate_consumed_v3_3_2_1()
  # Reuse the exact v3.3.2.1 byte/tree gates, but not its consumed-attempt gate.
  v3321_freeze = _v3321._validate_freeze(bundle_root)  # pylint: disable=protected-access
  model_bundle = _v3321._validate_model_run_bundle(  # pylint: disable=protected-access
      v3321_freeze, bundle_root
  )
  run_audit = _v3321._walk_exact_tree(  # pylint: disable=protected-access
      run_dir, _v3321._RUN_FILES, {'.', 'compiler', 'compiler/eight_row'},  # pylint: disable=protected-access
      'immutable v3.3.2 run',
  )
  preflight_audit = _v3321._walk_exact_tree(  # pylint: disable=protected-access
      _v3321._PREFLIGHT_DIR, _v3321._PREFLIGHT_FILES, {'.'},  # pylint: disable=protected-access
      'immutable v3.3.2 preflight',
  )
  if (
      run_audit['tree_sha256'] != _v3321.RUN_TREE_SHA256
      or preflight_audit['tree_sha256'] != _v3321.PREFLIGHT_TREE_SHA256
  ):
    raise AmendmentError('Immutable v3.3.2 tree binding changed.')
  binding = {
      'analysis_version': ANALYSIS_VERSION,
      'git_head': head,
      'tracked_head_clean': True,
      'amendment_sha256': AMENDMENT_SHA256,
      'analysis_freeze_sha256': _sha256(_FREEZE_PATH),
      'analysis_freeze': freeze,
      'consumed_v3_3_2_1_attempt': prior_failure,
      'v3_3_2_1_bundle': {
          'commit': V3_3_2_1_COMMIT,
          'analyzer_sha256': V3_3_2_1_ANALYZER_SHA256,
          'test_sha256': V3_3_2_1_TEST_SHA256,
          'freeze_sha256': V3_3_2_1_FREEZE_SHA256,
          'shell_sha256': V3_3_2_1_SHELL_SHA256,
          'amendment_sha256': V3_3_2_1_AMENDMENT_SHA256,
          'historical_commit_audit': historical_v3321,
      },
      'model_run_bundle': model_bundle,
      'immutable_run': run_audit,
      'immutable_preflight': preflight_audit,
      'saved_validator_is_frozen_original': (
          _SAVED_FROZEN_VALIDATOR
          is _v332._validate_freeze_and_start  # pylint: disable=protected-access
      ),
      'model_rerun_permitted': False,
      'model_apply_count': 0,
      'scientific_values_read': False,
      'confirmation_model_calls_permitted': 0,
  }
  if binding['saved_validator_is_frozen_original'] is not True:
    raise AmendmentError('Frozen validator was changed before analysis attempt.')
  binding = json.loads(json.dumps(binding, sort_keys=True, allow_nan=False))
  if expected_attempt_started_sha256 is not None:
    _validate_started_attempt(binding, expected_attempt_started_sha256)
  _assert_cpu_only('v3.3.2.2 completed precondition process')
  return binding


def _validate_started_attempt(binding: Mapping[str, Any], digest: str) -> None:
  _strict_directory(_ATTEMPT_DIR, 'v3.3.2.2 attempt')
  entries = list(_ATTEMPT_DIR.iterdir())
  if len(entries) != 1 or entries[0].name != 'ANALYSIS_ATTEMPT_STARTED.json':
    raise AmendmentError('v3.3.2.2 attempt has extra/terminal artifacts.')
  if not _is_sha256(digest) or _sha256(entries[0]) != digest:
    raise AmendmentError('v3.3.2.2 attempt START changed.')
  value = _read_object(entries[0], 'v3.3.2.2 attempt START')
  expected_keys = {
      'analysis_version', 'status', 'reason', 'started_at_unix_s',
      'run_dir', 'output_json', 'output_markdown', 'amendment_binding',
      'model_rerun_permitted', 'scientific_values_read_before_attempt',
      'confirmation_model_calls_permitted',
  }
  timestamp = value.get('started_at_unix_s')
  if (
      set(value) != expected_keys
      or
      value.get('analysis_version') != ANALYSIS_VERSION
      or value.get('status') != 'started_append_only_one_shot'
      or value.get('reason') != AMENDMENT_REASON
      or value.get('run_dir') != str(_v3321._RUN_DIR.resolve())  # pylint: disable=protected-access
      or value.get('output_json')
      != str((_ANALYSIS_DIR / 'ANALYSIS.json').resolve())
      or value.get('output_markdown')
      != str((_ANALYSIS_DIR / 'RESULT.md').resolve())
      or value.get('amendment_binding') != binding
      or value.get('model_rerun_permitted') is not False
      or value.get('scientific_values_read_before_attempt') is not False
      or value.get('confirmation_model_calls_permitted') != 0
      or isinstance(timestamp, bool)
      or not isinstance(timestamp, (int, float))
      or not math.isfinite(float(timestamp))
  ):
    raise AmendmentError('v3.3.2.2 attempt START content changed.')


def analyze(
    run_dir: Path, *, bundle_root: Path = _REPO_ROOT,
    amendment_binding: Mapping[str, Any] | None = None,
    attempt_started_sha256: str | None = None,
) -> dict[str, Any]:
  _assert_cpu_only('v3.3.2.2 pre-analysis process')
  verified = _validate_amendment_preconditions(
      run_dir, bundle_root,
      expected_attempt_started_sha256=attempt_started_sha256,
  )
  if amendment_binding != verified or not _is_sha256(attempt_started_sha256):
    raise AmendmentError('v3.3.2.2 append-only binding changed.')
  with _scoped_repair():
    result = _v332.analyze(run_dir, bundle_root=bundle_root)
  if _v332._validate_freeze_and_start is not _SAVED_FROZEN_VALIDATOR:  # pylint: disable=protected-access
    raise AmendmentError('Frozen validator was not restored after delegation.')
  _v3321._enforce_exact_controlled_stop(result)  # pylint: disable=protected-access
  result = dict(result)
  result['analysis_version'] = ANALYSIS_VERSION
  result['analyzer_amendment'] = {
      'reason': AMENDMENT_REASON,
      'amendment_sha256': AMENDMENT_SHA256,
      'amendment_binding': verified,
      'attempt_started_sha256': attempt_started_sha256,
      'saved_frozen_validator_called_directly': True,
      'saved_frozen_validator_restored_exactly': True,
      'v3_3_2_1_failure_preserved': {
          'start_sha256': FAILED_START_SHA256,
          'failure_sha256': FAILED_TERMINAL_SHA256,
          'attempt_tree_sha256': FAILED_ATTEMPT_TREE_SHA256,
      },
      'model_rerun_permitted': False,
      'model_apply_count': 0,
      'scientific_summary_computed': False,
      'shapley_or_nomination_computed': False,
      'combined_analysis_permitted': False,
  }
  _v3321._assert_no_scientific_payload(result)  # pylint: disable=protected-access
  _assert_cpu_only('v3.3.2.2 post-analysis process')
  return result


def render_markdown(result: Mapping[str, Any]) -> str:
  return '\n'.join((
      '# OpenSplice v3.3.2.2 controlled-stop structural archive', '',
      f"**Decision:** `{result['decision']}`", '',
      'The frozen v3.3.2 attempt stopped before any model apply because its ',
      'fresh backend-compiled HLO was not byte-identical to the frozen v3.3 ',
      'backend artifact. StableHLO and pre-backend HLO were identical.', '',
      'This archive contains **no biological evidence**: zero endpoint records, ',
      'zero model applies, and no completed ID-0 or ID-255 controls.', '',
      '**Scientific summary computed:** no  ',
      '**Shapley, interaction, resolution, or nomination computed:** no  ',
      '**Combined analysis permitted:** no', '',
      'v3.3.2.2 changes only Python function-reference control flow. The saved ',
      'frozen validator was called directly and restored exactly.', '',
      'Confirmation model outputs, activations, and interventions remained ',
      'unopened.', '',
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
      'run_dir': str(_v3321._RUN_DIR.resolve()),  # pylint: disable=protected-access
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
    raise FileExistsError('v3.3.2.2 output directory already exists.')
  _ANALYSIS_DIR.mkdir(mode=0o755, parents=False, exist_ok=False)
  json_path = _ANALYSIS_DIR / 'ANALYSIS.json'
  markdown_path = _ANALYSIS_DIR / 'RESULT.md'
  json_sha = _write_new_json(json_path, result)
  markdown = render_markdown(result).encode()
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


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run-dir', type=Path, required=True)
  parser.add_argument('--bundle-root', type=Path, required=True)
  parser.add_argument('--output-json', type=Path, required=True)
  parser.add_argument('--output-markdown', type=Path, required=True)
  args = parser.parse_args()
  if args.run_dir.resolve() != _v3321._RUN_DIR.resolve():  # pylint: disable=protected-access
    raise AmendmentError('CLI run path changed.')
  if args.bundle_root.resolve() != _REPO_ROOT.resolve():
    raise AmendmentError('CLI repository path changed.')
  if args.output_json.resolve() != (_ANALYSIS_DIR / 'ANALYSIS.json').resolve():
    raise AmendmentError('CLI JSON output path changed.')
  if args.output_markdown.resolve() != (_ANALYSIS_DIR / 'RESULT.md').resolve():
    raise AmendmentError('CLI Markdown output path changed.')
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
