#!/usr/bin/env python3
"""Analyzer-only v3.2.4 fix for the exact multiprocessing import alias."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import time
import traceback
from typing import Any, Mapping, Sequence


ANALYSIS_VERSION = 'opensplice-superset-analysis-v3.2.4'
AMENDMENT_REASON = 'exact_main_mp_main_import_path_alias'
V3_2_3_ANALYZER_SHA256 = (
    '9f84b094aa064b3dc789e72f53e6d028e346eecfe6dcfc8a0c7febc7fcc1397c'
)
V3_2_3_TEST_SHA256 = (
    '9c8a937791f9ae966dfd45be5384989917912a1714fb1b147f8e7a5268cf1bd8'
)
V3_2_3_AMENDMENT_SHA256 = (
    'e0c18b53dfdce93be443c84c178766bbb744ce7b017463b83ca449833f91e95e'
)
V3_2_3_ATTEMPT_SHA256 = (
    '00dbe3749a1a810e1bf7de02d1eeecf02a9a363ee196709d8c7c76283b778325'
)
V3_2_3_FAILURE_SHA256 = (
    '70b4521dfbcfbc0702bd3e8c3039c45f1a6d8f66081d2c51e5181d4fa2320871'
)
FROZEN_IMPORT_PROVENANCE_SHA256 = (
    'b542e76b7db0cbe74a5322af9c6e647a0dfd5bea931657af746c41560949dd7d'
)
FROZEN_IMPORT_PROVENANCE_SIZE = 24_700
FROZEN_IMPORT_MODULE_COUNT = 74
FROZEN_IMPORT_PATH_GROUP_COUNT = 73
FROZEN_RUNNER_SHA256 = (
    'd3d3335ee47fcd477f25fd8dbf13f515a6ec909b7054e35c7209acac45f2f1eb'
)
FROZEN_RUNNER_SIZE = 68_642
AMENDMENT_SHA256 = (
    '9a274b7b945c82113d71ede892bd7f987220f28bef7515860665d35fc92ca554'
)

_HERE = Path(__file__).resolve().parent
_V3_2_3_PATH = _HERE / 'analyze_superset_graph_v3_2_3.py'
_V3_2_3_TEST_PATH = _HERE / 'analyze_superset_graph_v3_2_3_test.py'
_V3_2_3_AMENDMENT_PATH = (
    _HERE / 'v3_wider_mechanism/superset_graph_analysis_amendment_v3_2_3.md'
)
_AMENDMENT_PATH = (
    _HERE / 'v3_wider_mechanism/superset_graph_analysis_amendment_v3_2_4.md'
)
_TEST_PATH = _HERE / 'analyze_superset_graph_v3_2_4_test.py'
_RUNNER_PATH = _HERE / 'run_superset_graph_v3_2.py'
_CONSUMED_V3_2_3_DIR = (
    _HERE / 'results/v3_2_development_superset_graph_analysis_v3_2_3_attempt'
)
_ATTEMPT_DIR = (
    _HERE / 'results/v3_2_development_superset_graph_analysis_v3_2_4_attempt'
)
_IMPORT_FILENAMES = {
    'pre_model': 'IMPORT_PROVENANCE_PRE_MODEL.json',
    'post_model_precompile': 'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
    'postcompile': 'IMPORT_PROVENANCE.json',
}
_MODULE_ROW_KEYS = {'name', 'path', 'root', 'sha256', 'size_bytes'}
_REQUIRED_MODULES = {
    'alphagenome_research.model.model',
    'alphagenome_research.model.dna_model',
    'target_reducers_v3',
}


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def _load_v3_2_3():
  if _sha256(_V3_2_3_PATH) != V3_2_3_ANALYZER_SHA256:
    raise ValueError('Frozen v3.2.3 analyzer bytes changed before import.')
  specification = importlib.util.spec_from_file_location(
      '_opensplice_frozen_analyzer_v3_2_3', _V3_2_3_PATH
  )
  if specification is None or specification.loader is None:
    raise RuntimeError('Cannot load the frozen v3.2.3 analyzer.')
  module = importlib.util.module_from_spec(specification)
  specification.loader.exec_module(module)
  return module


_v323 = _load_v3_2_3()
_v322 = _v323._v322  # pylint: disable=protected-access
_v321 = _v323._v321  # pylint: disable=protected-access
_base = _v323._base  # pylint: disable=protected-access
_LAST_IMPORT_ALIAS_AUDIT: dict[str, Any] | None = None


def _strict_regular_nonsymlink(path: Path, label: str) -> None:
  try:
    mode = path.lstat().st_mode
  except OSError as error:
    raise ValueError(f'{label} cannot be statted.') from error
  if path.is_symlink() or not stat.S_ISREG(mode):
    raise ValueError(f'{label} is symlinked or not a regular file.')


def _validate_import_provenance_impl(
    path: Path, expected_sha: Any, *, bundle_root: Path,
    artifact_sha256: str, artifact_size: int, module_count: int,
    path_group_count: int, runner_path: Path, runner_sha256: str,
    runner_size: int, required_modules: set[str],
    allowed_artifact_paths: set[Path] | None,
) -> dict[str, Any]:
  """Validates one inventory with only the exact main/mp-main path alias."""
  path = path.absolute()
  _base._guard_path(path)  # pylint: disable=protected-access
  if allowed_artifact_paths is not None and path not in allowed_artifact_paths:
    raise ValueError('IMPORT_PROVENANCE path is not an exact frozen phase file.')
  _strict_regular_nonsymlink(path, 'IMPORT_PROVENANCE artifact')
  if (
      not _base._is_sha256(expected_sha)  # pylint: disable=protected-access
      or expected_sha != artifact_sha256
      or path.stat().st_size != artifact_size
      or _sha256(path) != artifact_sha256
  ):
    raise ValueError('IMPORT_PROVENANCE size/hash binding mismatch.')
  value = _base._read_json(path)  # pylint: disable=protected-access
  if set(value) != {'module_count', 'modules'}:
    raise ValueError('IMPORT_PROVENANCE top-level schema changed.')
  modules = value.get('modules')
  if (
      not isinstance(modules, list)
      or value.get('module_count') != module_count
      or len(modules) != module_count
  ):
    raise ValueError('IMPORT_PROVENANCE module list/count is invalid.')
  upstream_root = (bundle_root.parent / 'alphagenome').resolve()
  resolved_bundle_root = bundle_root.resolve()
  names: list[str] = []
  groups: dict[str, list[dict[str, Any]]] = {}
  validated_rows: list[dict[str, Any]] = []
  for index, row in enumerate(modules):
    if not isinstance(row, Mapping) or set(row) != _MODULE_ROW_KEYS:
      raise ValueError(f'IMPORT_PROVENANCE row {index} schema changed.')
    name, path_value, root = row['name'], row['path'], row['root']
    digest, size = row['sha256'], row['size_bytes']
    if (
        not isinstance(name, str) or not name
        or not isinstance(path_value, str) or not path_value
        or not _base._is_sha256(digest)  # pylint: disable=protected-access
        or isinstance(size, bool) or not isinstance(size, int) or size < 0
    ):
      raise ValueError(f'IMPORT_PROVENANCE row {index} is malformed.')
    lexical_path = Path(path_value)
    _base._guard_path(lexical_path)  # pylint: disable=protected-access
    _strict_regular_nonsymlink(lexical_path, f'Imported module {name}')
    module_path = lexical_path.resolve()
    expected_root = (
        resolved_bundle_root if root == 'alphagenome_research_checkout'
        else upstream_root if root == 'upstream_alphagenome_checkout'
        else None
    )
    if expected_root is None:
      raise ValueError('IMPORT_PROVENANCE contains an undeclared root.')
    try:
      module_path.relative_to(expected_root)
    except ValueError as error:
      raise ValueError('IMPORT_PROVENANCE module escaped its declared root.') from error
    if _sha256(module_path) != digest:
      raise ValueError(f'Imported module hash changed: {name}.')
    if module_path.stat().st_size != size:
      raise ValueError(f'Imported module size changed: {name}.')
    names.append(name)
    validated = dict(row)
    validated_rows.append(validated)
    groups.setdefault(str(module_path), []).append(validated)
  if names != sorted(names) or len(names) != len(set(names)):
    raise ValueError('IMPORT_PROVENANCE names are unsorted or duplicated.')
  if not required_modules.issubset(names):
    missing = sorted(required_modules - set(names))
    raise ValueError(f'IMPORT_PROVENANCE misses required modules: {missing}.')
  if len(groups) != path_group_count:
    raise ValueError('IMPORT_PROVENANCE resolved-path group count changed.')
  collisions = [rows for rows in groups.values() if len(rows) != 1]
  if len(collisions) != 1 or len(collisions[0]) != 2:
    raise ValueError('IMPORT_PROVENANCE has an unexpected path collision.')
  aliases = collisions[0]
  alias_names = {row['name'] for row in aliases}
  resolved_runner = runner_path.resolve()
  alias_path = Path(aliases[0]['path']).resolve()
  without_names = [
      {key: value for key, value in row.items() if key != 'name'}
      for row in aliases
  ]
  if (
      alias_names != {'__main__', '__mp_main__'}
      or alias_path != resolved_runner
      or any(Path(row['path']).resolve() != resolved_runner for row in aliases)
      or without_names[0] != without_names[1]
      or aliases[0]['root'] != 'alphagenome_research_checkout'
      or aliases[0]['sha256'] != runner_sha256
      or aliases[0]['size_bytes'] != runner_size
      or _sha256(resolved_runner) != runner_sha256
      or resolved_runner.stat().st_size != runner_size
  ):
    raise ValueError('IMPORT_PROVENANCE main/mp-main alias is not exact.')
  return {
      'sha256': artifact_sha256,
      'module_count': module_count,
      'modules': validated_rows,
      'roots': [
          'alphagenome_research_checkout',
          'upstream_alphagenome_checkout',
      ],
      'alias_audit': {
          'accepted_exact_alias': True,
          'names': ['__main__', '__mp_main__'],
          'path': str(resolved_runner),
          'root': 'alphagenome_research_checkout',
          'sha256': runner_sha256,
          'size_bytes': runner_size,
          'rows_preserved': True,
          'path_identity_count': 1,
      },
  }


def _production_import_expectations(bundle_root: Path) -> dict[str, Any]:
  run_dir = _base._OUTPUT_DIR.resolve()  # pylint: disable=protected-access
  return {
      'artifact_sha256': FROZEN_IMPORT_PROVENANCE_SHA256,
      'artifact_size': FROZEN_IMPORT_PROVENANCE_SIZE,
      'module_count': FROZEN_IMPORT_MODULE_COUNT,
      'path_group_count': FROZEN_IMPORT_PATH_GROUP_COUNT,
      'runner_path': _RUNNER_PATH,
      'runner_sha256': FROZEN_RUNNER_SHA256,
      'runner_size': FROZEN_RUNNER_SIZE,
      'required_modules': set(_REQUIRED_MODULES),
      'allowed_artifact_paths': {
          (run_dir / filename).absolute()
          for filename in _IMPORT_FILENAMES.values()
      },
  }


def _validate_import_provenance(
    path: Path, expected_sha: Any, *, bundle_root: Path
) -> dict[str, Any]:
  return _validate_import_provenance_impl(
      path, expected_sha, bundle_root=bundle_root,
      **_production_import_expectations(bundle_root),
  )


def _validate_import_phases_impl(
    run_dir: Path, complete: Mapping[str, Any], *, bundle_root: Path,
    expectations: Mapping[str, Any], expected_run_dir: Path,
) -> dict[str, Any]:
  run_dir = run_dir.resolve()
  if run_dir != expected_run_dir.resolve():
    raise ValueError('Import-provenance run directory changed.')
  bindings = complete.get('import_provenance_phases')
  expected_digest = expectations['artifact_sha256']
  if (
      not isinstance(bindings, Mapping)
      or set(bindings) != set(_IMPORT_FILENAMES)
      or any(value != expected_digest for value in bindings.values())
      or complete.get('import_provenance_sha256') != expected_digest
  ):
    raise ValueError('RUN_COMPLETE import-provenance bindings changed.')
  phase_bytes = []
  phases = {}
  for phase, filename in _IMPORT_FILENAMES.items():
    artifact = run_dir / filename
    phases[phase] = _validate_import_provenance_impl(
        artifact, bindings[phase], bundle_root=bundle_root, **expectations
    )
    phase_bytes.append(artifact.read_bytes())
  if any(data != phase_bytes[0] for data in phase_bytes[1:]):
    raise ValueError('Import-provenance phase artifacts are not byte-identical.')
  lazy_additions = {}
  for earlier_name, later_name in (
      ('pre_model', 'post_model_precompile'),
      ('post_model_precompile', 'postcompile'),
  ):
    earlier = {row['name']: row for row in phases[earlier_name]['modules']}
    later = {row['name']: row for row in phases[later_name]['modules']}
    if earlier != later:
      raise ValueError('Import provenance changed across phases.')
    lazy_additions[f'{earlier_name}_to_{later_name}'] = []
  return {
      'phase_sha256': dict(bindings),
      'module_counts': {
          name: value['module_count'] for name, value in phases.items()
      },
      'lazy_additions': lazy_additions,
      'stable_shared_module_bytes': True,
      'exact_alias_audits': {
          name: value['alias_audit'] for name, value in phases.items()
      },
      'phase_artifacts_byte_identical': True,
  }


def _validate_import_phases(
    run_dir: Path, complete: Mapping[str, Any], *, bundle_root: Path
) -> dict[str, Any]:
  global _LAST_IMPORT_ALIAS_AUDIT
  result = _validate_import_phases_impl(
      run_dir, complete, bundle_root=bundle_root,
      expectations=_production_import_expectations(bundle_root),
      expected_run_dir=_base._OUTPUT_DIR,  # pylint: disable=protected-access
  )
  _LAST_IMPORT_ALIAS_AUDIT = {
      'phase_sha256': result['phase_sha256'],
      'module_counts': result['module_counts'],
      'path_group_count': FROZEN_IMPORT_PATH_GROUP_COUNT,
      'phase_artifacts_byte_identical': True,
      'phase_aliases': result['exact_alias_audits'],
      'both_alias_rows_preserved': True,
      'cross_phase_missing_changed_or_lazy_additions': False,
  }
  return result


def _validate_consumed_v3_2_3_attempt() -> dict[str, Any]:
  expected = {
      'ANALYSIS_ATTEMPT_STARTED.json': V3_2_3_ATTEMPT_SHA256,
      'ANALYSIS_FAILURE.json': V3_2_3_FAILURE_SHA256,
  }
  observed = {
      path.name for path in _CONSUMED_V3_2_3_DIR.iterdir()
  } if _CONSUMED_V3_2_3_DIR.is_dir() else set()
  if observed != set(expected):
    raise ValueError('Consumed v3.2.3 attempt tree is incomplete or has extras.')
  for filename, digest in expected.items():
    path = _CONSUMED_V3_2_3_DIR / filename
    if path.is_symlink() or not path.is_file() or _sha256(path) != digest:
      raise ValueError(f'Consumed v3.2.3 artifact changed: {filename}.')
  started = json.loads(
      (_CONSUMED_V3_2_3_DIR / 'ANALYSIS_ATTEMPT_STARTED.json').read_text(
          encoding='utf-8'
      )
  )
  failure = json.loads(
      (_CONSUMED_V3_2_3_DIR / 'ANALYSIS_FAILURE.json').read_text(
          encoding='utf-8'
      )
  )
  if (
      started.get('analysis_version') != 'opensplice-superset-analysis-v3.2.3'
      or started.get('status') != 'started_append_only_one_shot'
      or started.get('reason')
      != 'external_preflight_validated_logs_normalization'
      or started.get('model_rerun_permitted') is not False
      or started.get('scientific_gate_or_estimand_changed') is not False
      or started.get('confirmation_model_calls_permitted') != 0
      or failure.get('status') != 'failed_consumed_no_retry'
      or failure.get('attempt_started_sha256') != V3_2_3_ATTEMPT_SHA256
      or failure.get('analysis_json_exists') is not False
      or failure.get('analysis_markdown_exists') is not False
      or failure.get('failure', {}).get('type') != 'ValueError'
      or failure.get('failure', {}).get('message')
      != 'IMPORT_PROVENANCE contains duplicate module/path rows.'
  ):
    raise ValueError('Consumed v3.2.3 failure boundary changed.')
  return {
      'attempt_started_sha256': V3_2_3_ATTEMPT_SHA256,
      'failure_sha256': V3_2_3_FAILURE_SHA256,
      'failure_type': 'ValueError',
      'failure_message': failure['failure']['message'],
      'scientific_output_written': False,
  }


def _validate_prior_analyzer_bundles(bundle_root: Path) -> dict[str, str]:
  expected = {
      _v321._ORIGINAL_PATH: _v321.ORIGINAL_ANALYZER_SHA256,  # pylint: disable=protected-access
      _v322._V3_2_1_PATH: _v322.V3_2_1_ANALYZER_SHA256,  # pylint: disable=protected-access
      _v322._V3_2_1_TEST_PATH: _v322.V3_2_1_TEST_SHA256,  # pylint: disable=protected-access
      _v321._AMENDMENT_PATH: _v321.AMENDMENT_SHA256,  # pylint: disable=protected-access
      _v323._V3_2_2_PATH: _v323.V3_2_2_ANALYZER_SHA256,  # pylint: disable=protected-access
      _v323._V3_2_2_TEST_PATH: _v323.V3_2_2_TEST_SHA256,  # pylint: disable=protected-access
      _v323._V3_2_2_AMENDMENT_PATH: _v323.V3_2_2_AMENDMENT_SHA256,  # pylint: disable=protected-access
      _V3_2_3_PATH: V3_2_3_ANALYZER_SHA256,
      _V3_2_3_TEST_PATH: V3_2_3_TEST_SHA256,
      _V3_2_3_AMENDMENT_PATH: V3_2_3_AMENDMENT_SHA256,
  }
  root = bundle_root.resolve()
  observed = {}
  for path, digest in expected.items():
    resolved = path.resolve()
    _base._guard_path(resolved)  # pylint: disable=protected-access
    try:
      relative = str(resolved.relative_to(root))
    except ValueError as error:
      raise ValueError('Prior analyzer bundle escapes repository.') from error
    if resolved.is_symlink() or not resolved.is_file() or _sha256(resolved) != digest:
      raise ValueError(f'Prior analyzer bundle changed: {relative}.')
    observed[relative] = digest
  return observed


def _validate_amendment_preconditions(
    run_dir: Path, bundle_root: Path, *,
    expected_attempt_started_sha256: str | None = None,
) -> dict[str, Any]:
  _v321._assert_no_model_imports('v3.2.4 precondition process')  # pylint: disable=protected-access
  if _sha256(_V3_2_3_PATH) != V3_2_3_ANALYZER_SHA256:
    raise ValueError('Frozen v3.2.3 analyzer bytes changed.')
  if _sha256(_V3_2_3_TEST_PATH) != V3_2_3_TEST_SHA256:
    raise ValueError('Frozen v3.2.3 analyzer test bytes changed.')
  if _sha256(_V3_2_3_AMENDMENT_PATH) != V3_2_3_AMENDMENT_SHA256:
    raise ValueError('Frozen v3.2.3 amendment bytes changed.')
  prior_binding = _v321._validate_amendment_preconditions(  # pylint: disable=protected-access
      run_dir, bundle_root
  )
  consumed_v321 = _v322._validate_consumed_v3_2_1_attempt()  # pylint: disable=protected-access
  consumed_v322 = _v323._validate_consumed_v3_2_2_attempt()  # pylint: disable=protected-access
  consumed_v323 = _validate_consumed_v3_2_3_attempt()
  prior_bundles = _validate_prior_analyzer_bundles(bundle_root)
  if expected_attempt_started_sha256 is None and _ATTEMPT_DIR.exists():
    raise FileExistsError('The append-only v3.2.4 attempt was already consumed.')
  if _sha256(_AMENDMENT_PATH) != AMENDMENT_SHA256:
    raise ValueError('Prospective v3.2.4 amendment bytes changed/unbound.')
  tracked_paths = (_AMENDMENT_PATH, Path(__file__).resolve(), _TEST_PATH.resolve())
  for path in tracked_paths:
    _base._guard_path(path)  # pylint: disable=protected-access
    try:
      relative = str(path.relative_to(bundle_root.resolve()))
    except ValueError as error:
      raise ValueError('v3.2.4 amendment file escapes repository.') from error
    subprocess.run(
        ('git', '-C', str(bundle_root), 'ls-files', '--error-unmatch', relative),
        check=True, capture_output=True,
    )
  _v321._assert_global_tracked_head_clean(bundle_root)  # pylint: disable=protected-access
  git_head = subprocess.check_output(
      ('git', '-C', str(bundle_root), 'rev-parse', 'HEAD'), text=True
  ).strip()
  result = {
      'git_head': git_head,
      'tracked_head_clean': True,
      'file_sha256': {
          str(path.relative_to(bundle_root.resolve())): _sha256(path)
          for path in tracked_paths
      },
      'amendment_sha256': AMENDMENT_SHA256,
      'v3_2_1_binding': prior_binding,
      'prior_analyzer_bundle_sha256': prior_bundles,
      'v3_2_3_bundle_sha256': {
          str(_V3_2_3_PATH.relative_to(bundle_root.resolve())): (
              V3_2_3_ANALYZER_SHA256
          ),
          str(_V3_2_3_TEST_PATH.relative_to(bundle_root.resolve())): (
              V3_2_3_TEST_SHA256
          ),
          str(_V3_2_3_AMENDMENT_PATH.relative_to(bundle_root.resolve())): (
              V3_2_3_AMENDMENT_SHA256
          ),
      },
      'consumed_v3_2_1_attempt': consumed_v321,
      'consumed_v3_2_2_attempt': consumed_v322,
      'consumed_v3_2_3_attempt': consumed_v323,
  }
  if expected_attempt_started_sha256 is not None:
    _validate_started_attempt(result, expected_attempt_started_sha256)
  return result


def analyze(
    run_dir: Path, *, bundle_root: Path = _base._REPO_ROOT,  # pylint: disable=protected-access
    ignored_paths: Sequence[Path] = (), enforce_standard_locations: bool = True,
    amendment_binding: Mapping[str, Any] | None = None,
    attempt_started_sha256: str | None = None,
) -> dict[str, Any]:
  global _LAST_IMPORT_ALIAS_AUDIT
  _LAST_IMPORT_ALIAS_AUDIT = None
  _v323._LAST_PREFLIGHT_NORMALIZATION_AUDIT = None  # pylint: disable=protected-access
  _v322._LAST_PROTOBUF_AUDIT = None  # pylint: disable=protected-access
  _v321._LAST_CHECKPOINT_AUDIT = None  # pylint: disable=protected-access
  verified = None
  if enforce_standard_locations:
    verified = _validate_amendment_preconditions(
        run_dir, bundle_root,
        expected_attempt_started_sha256=attempt_started_sha256,
    )
    if amendment_binding != verified:
      raise ValueError('v3.2.4 amendment binding is absent or changed.')
    if not _base._is_sha256(attempt_started_sha256):  # pylint: disable=protected-access
      raise ValueError('v3.2.4 append-only attempt binding is absent.')
  _base._validate_checkpoint_and_reference_inputs = (  # pylint: disable=protected-access
      _v321._validate_checkpoint_and_reference_inputs  # pylint: disable=protected-access
  )
  _base._validate_bootstrap_attestation = _v322._validate_bootstrap_attestation  # pylint: disable=protected-access
  _base._validate_preflight = _v323._validate_preflight  # pylint: disable=protected-access
  _base._validate_import_phases = _validate_import_phases  # pylint: disable=protected-access
  _base.ANALYSIS_VERSION = ANALYSIS_VERSION
  try:
    result = _base.analyze(
        run_dir, bundle_root=bundle_root, ignored_paths=ignored_paths,
        enforce_standard_locations=enforce_standard_locations,
    )
  finally:
    _v321._assert_no_model_imports('v3.2.4 post-analysis process')  # pylint: disable=protected-access
  if _v323._LAST_PREFLIGHT_NORMALIZATION_AUDIT is None:  # pylint: disable=protected-access
    raise RuntimeError('v3.2.3 preflight normalization audit was not recorded.')
  if _v322._LAST_PROTOBUF_AUDIT is None:  # pylint: disable=protected-access
    raise RuntimeError('v3.2.2 protobuf normalization audit was not recorded.')
  if _v321._LAST_CHECKPOINT_AUDIT is None:  # pylint: disable=protected-access
    raise RuntimeError('v3.2.1 checkpoint normalization audit was not recorded.')
  if _LAST_IMPORT_ALIAS_AUDIT is None:
    raise RuntimeError('v3.2.4 import-alias audit was not recorded.')
  result['analyzer_amendments'] = {
      'model_run_analysis_version': 'opensplice-superset-analysis-v3.2.0',
      'offline_analysis_version': ANALYSIS_VERSION,
      'v3_2_4_amendment_sha256': AMENDMENT_SHA256,
      'v3_2_4_amendment_binding': verified,
      'analysis_attempt_started_sha256': attempt_started_sha256,
      'checkpoint_symlink_audit': _v321._LAST_CHECKPOINT_AUDIT,  # pylint: disable=protected-access
      'protobuf_generated_outputs_audit': _v322._LAST_PROTOBUF_AUDIT,  # pylint: disable=protected-access
      'external_preflight_normalization_audit': (
          _v323._LAST_PREFLIGHT_NORMALIZATION_AUDIT  # pylint: disable=protected-access
      ),
      'import_main_alias_audit': _LAST_IMPORT_ALIAS_AUDIT,
      'preserved_analyzer_failures': [
          {
              'analysis_version': 'opensplice-superset-analysis-v3.2.0',
              'reason': 'manifest_bound_hf_snapshot_symlink_rejected',
              'scientific_output_written': False,
          },
          {
              'analysis_version': 'opensplice-superset-analysis-v3.2.1',
              'reason': 'path_keyed_generated_pyi_omitted',
              'attempt_started_sha256': _v322.V3_2_1_ATTEMPT_SHA256,
              'failure_sha256': _v322.V3_2_1_FAILURE_SHA256,
              'scientific_output_written': False,
          },
          {
              'analysis_version': 'opensplice-superset-analysis-v3.2.2',
              'reason': 'derived_validated_logs_compared_as_artifact_field',
              'attempt_started_sha256': _v323.V3_2_2_ATTEMPT_SHA256,
              'failure_sha256': _v323.V3_2_2_FAILURE_SHA256,
              'scientific_output_written': False,
          },
          {
              'analysis_version': 'opensplice-superset-analysis-v3.2.3',
              'reason': 'exact_main_mp_main_path_alias_rejected',
              'attempt_started_sha256': V3_2_3_ATTEMPT_SHA256,
              'failure_sha256': V3_2_3_FAILURE_SHA256,
              'scientific_output_written': False,
          },
      ],
      'model_rerun_permitted': False,
      'scientific_gate_or_estimand_changed': False,
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
    run_dir: Path, output_json: Path, output_markdown: Path | None,
    amendment_binding: Mapping[str, Any],
) -> str:
  try:
    _ATTEMPT_DIR.mkdir(mode=0o755, parents=False, exist_ok=False)
  except FileExistsError as error:
    raise FileExistsError(
        'The append-only v3.2.4 corrected-analysis attempt was already consumed.'
    ) from error
  return _write_new_json(_ATTEMPT_DIR / 'ANALYSIS_ATTEMPT_STARTED.json', {
      'analysis_version': ANALYSIS_VERSION,
      'status': 'started_append_only_one_shot',
      'reason': AMENDMENT_REASON,
      'started_at_unix_s': time.time(),
      'run_dir': str(run_dir.resolve()),
      'output_json': str(output_json.resolve()),
      'output_markdown': (
          str(output_markdown.resolve()) if output_markdown is not None else None
      ),
      'amendment_binding': dict(amendment_binding),
      'model_rerun_permitted': False,
      'scientific_gate_or_estimand_changed': False,
      'confirmation_model_calls_permitted': 0,
  })


def _validate_started_attempt(
    amendment_binding: Mapping[str, Any], started_sha256: str,
) -> None:
  if not _base._is_sha256(started_sha256):  # pylint: disable=protected-access
    raise ValueError('v3.2.4 attempt-start digest is malformed.')
  entries = list(_ATTEMPT_DIR.iterdir()) if _ATTEMPT_DIR.is_dir() else []
  if len(entries) != 1 or entries[0].name != 'ANALYSIS_ATTEMPT_STARTED.json':
    raise ValueError('v3.2.4 started attempt has extra or terminal artifacts.')
  path = entries[0]
  if path.is_symlink() or not path.is_file() or _sha256(path) != started_sha256:
    raise ValueError('v3.2.4 attempt-start artifact hash/type changed.')
  record = json.loads(path.read_text(encoding='utf-8'))
  if (
      record.get('analysis_version') != ANALYSIS_VERSION
      or record.get('status') != 'started_append_only_one_shot'
      or record.get('reason') != AMENDMENT_REASON
      or record.get('amendment_binding') != amendment_binding
      or record.get('model_rerun_permitted') is not False
      or record.get('scientific_gate_or_estimand_changed') is not False
      or record.get('confirmation_model_calls_permitted') != 0
  ):
    raise ValueError('v3.2.4 attempt-start content/binding changed.')


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
  parser.add_argument('--output-markdown', type=Path)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  for path in (args.run_dir, args.output_json, args.output_markdown):
    if path is not None:
      _base._guard_path(path)  # pylint: disable=protected-access
  analysis_dir = _base._ANALYSIS_DIR.resolve()  # pylint: disable=protected-access
  if args.output_json.resolve() != analysis_dir / 'ANALYSIS.json':
    raise ValueError('JSON output path differs from frozen analysis destination.')
  if args.output_markdown is not None and args.output_markdown.resolve() != (
      analysis_dir / 'RESULT.md'
  ):
    raise ValueError('Markdown output path differs from frozen destination.')
  if analysis_dir.exists():
    raise FileExistsError('Frozen analysis directory already exists; never overwrite.')
  binding = _validate_amendment_preconditions(
      args.run_dir, _base._REPO_ROOT  # pylint: disable=protected-access
  )
  started_sha = _start_attempt(
      args.run_dir, args.output_json, args.output_markdown, binding
  )
  ignored = [args.output_json]
  if args.output_markdown is not None:
    ignored.append(args.output_markdown)
  try:
    result = analyze(
        args.run_dir, ignored_paths=ignored, amendment_binding=binding,
        attempt_started_sha256=started_sha,
    )
    _base._write_atomic(  # pylint: disable=protected-access
        args.output_json,
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n',
    )
    if args.output_markdown is not None:
      markdown = _base.render_markdown(result) + (
          '\n## Analyzer-only amendments\n\n'
          'The model run used v3.2.0. Offline analysis used the prospective '
          'v3.2.1 checkpoint-symlink, v3.2.2 protobuf path-key, v3.2.3 '
          'preflight-log, and v3.2.4 exact import-alias repairs. Both import '
          'alias rows remain present. No repair changed a scientific gate or '
          'permitted a model rerun.\n'
      )
      _base._write_atomic(args.output_markdown, markdown)  # pylint: disable=protected-access
    _persist_terminal(
        'ANALYSIS_COMPLETE.json',
        {
            'status': 'complete',
            'analysis_json_sha256': _sha256(args.output_json),
            'analysis_markdown_sha256': (
                _sha256(args.output_markdown)
                if args.output_markdown is not None else None
            ),
            'decision': result['decision'],
        },
        started_sha256=started_sha,
    )
  except BaseException as error:
    try:
      _persist_terminal(
          'ANALYSIS_FAILURE.json',
          {
              'status': 'failed_consumed_no_retry',
              'failure': {
                  'type': type(error).__name__, 'message': str(error),
                  'traceback': traceback.format_exc(),
              },
              'analysis_json_exists': args.output_json.exists(),
              'analysis_markdown_exists': (
                  args.output_markdown.exists()
                  if args.output_markdown is not None else False
              ),
          },
          started_sha256=started_sha,
      )
    except BaseException:
      pass
    raise
  print(args.output_json.resolve())


if __name__ == '__main__':
  main()
