#!/usr/bin/env python3
"""Analyzer-only v3.2.3 fix for derived external-preflight log bindings."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any, Mapping, Sequence


ANALYSIS_VERSION = 'opensplice-superset-analysis-v3.2.3'
AMENDMENT_REASON = 'external_preflight_validated_logs_normalization'
V3_2_2_ANALYZER_SHA256 = (
    '874d66b8e9f9e39e8a9e56329f2d04cb219eb7d62be38a0bc038cc8de010e2b4'
)
V3_2_2_TEST_SHA256 = (
    '15477a6e171a42d24d75fd549960ba831804427209238125745e0158174099b9'
)
V3_2_2_AMENDMENT_SHA256 = (
    'a25d08c8a609703532a749ac0e5d0246614446627b84b4196f28be91ffdecb4f'
)
V3_2_2_ATTEMPT_SHA256 = (
    '11bf87f577d6aafbaa37139b5b98dc2c96c4dec317552564b4ca2bf6db88f117'
)
V3_2_2_FAILURE_SHA256 = (
    'f1904bf9593d6f43acad6bd3394e6010404b8e9e2e4b8c43f8b90344554d92a9'
)
FROZEN_EXTERNAL_PREFLIGHT_SHA256 = (
    '06e0d79f751dc2beb63355a87964f8c3f88d6ae8f843bc1132af5ea6d7ea2b35'
)
FROZEN_EMPTY_LOG_SHA256 = (
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
)
AMENDMENT_SHA256 = (
    'e0c18b53dfdce93be443c84c178766bbb744ce7b017463b83ca449833f91e95e'
)

_HERE = Path(__file__).resolve().parent
_V3_2_2_PATH = _HERE / 'analyze_superset_graph_v3_2_2.py'
_V3_2_2_TEST_PATH = _HERE / 'analyze_superset_graph_v3_2_2_test.py'
_V3_2_2_AMENDMENT_PATH = (
    _HERE / 'v3_wider_mechanism/superset_graph_analysis_amendment_v3_2_2.md'
)
_AMENDMENT_PATH = (
    _HERE / 'v3_wider_mechanism/superset_graph_analysis_amendment_v3_2_3.md'
)
_TEST_PATH = _HERE / 'analyze_superset_graph_v3_2_3_test.py'
_CONSUMED_V3_2_2_DIR = (
    _HERE / 'results/v3_2_development_superset_graph_analysis_v3_2_2_attempt'
)
_ATTEMPT_DIR = (
    _HERE / 'results/v3_2_development_superset_graph_analysis_v3_2_3_attempt'
)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def _load_v3_2_2():
  if _sha256(_V3_2_2_PATH) != V3_2_2_ANALYZER_SHA256:
    raise ValueError('Frozen v3.2.2 analyzer bytes changed before import.')
  specification = importlib.util.spec_from_file_location(
      '_opensplice_frozen_analyzer_v3_2_2', _V3_2_2_PATH
  )
  if specification is None or specification.loader is None:
    raise RuntimeError('Cannot load the frozen v3.2.2 analyzer.')
  module = importlib.util.module_from_spec(specification)
  specification.loader.exec_module(module)
  return module


_v322 = _load_v3_2_2()
_v321 = _v322._v321  # pylint: disable=protected-access
_base = _v322._base  # pylint: disable=protected-access
_FROZEN_PREFLIGHT_VALIDATOR = _base._validate_preflight  # pylint: disable=protected-access
_LAST_PREFLIGHT_NORMALIZATION_AUDIT: dict[str, Any] | None = None


def _normalize_external_preflight(
    start: Mapping[str, Any], freeze: Mapping[str, Any],
    *, expected_preflight_sha256: str = FROZEN_EXTERNAL_PREFLIGHT_SHA256,
    expected_log_sha256: str = FROZEN_EMPTY_LOG_SHA256,
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Checks and removes only the runner-derived ``validated_logs`` field."""
  external = start.get('external_preflight')
  if not isinstance(external, Mapping):
    raise ValueError('Embedded external preflight binding is missing.')
  path_text, digest = external.get('path'), external.get('sha256')
  if not isinstance(path_text, str) or not _base._is_sha256(digest):  # pylint: disable=protected-access
    raise ValueError('Embedded external preflight path/hash is malformed.')
  path = Path(path_text).resolve()
  _base._guard_path(path)  # pylint: disable=protected-access
  preflight_root = Path(str(freeze.get('preflight_dir'))).resolve()
  _base._guard_path(preflight_root)  # pylint: disable=protected-access
  if (
      digest != expected_preflight_sha256
      or path.parent != preflight_root or _sha256(path) != digest
  ):
    raise ValueError('External preflight artifact path/hash differs from freeze.')
  raw = _base._read_json(path)  # pylint: disable=protected-access
  expected_external_keys = set(raw) | {'path', 'sha256', 'validated_logs'}
  if set(external) != expected_external_keys:
    raise ValueError('Embedded external preflight schema has unexpected fields.')
  for key, value in raw.items():
    if external.get(key) != value:
      raise ValueError(f'Embedded external preflight differs at {key}.')
  logs = raw.get('logs')
  if not isinstance(logs, Mapping) or set(logs) != {'stdout', 'stderr'}:
    raise ValueError('External preflight artifact log schema changed.')
  expected_validated_logs = {}
  for stream in ('stdout', 'stderr'):
    binding = logs[stream]
    if not isinstance(binding, Mapping) or set(binding) != {'path', 'sha256'}:
      raise ValueError(f'External preflight {stream} binding schema changed.')
    log_path_text, log_digest = binding.get('path'), binding.get('sha256')
    if not isinstance(log_path_text, str) or not _base._is_sha256(log_digest):  # pylint: disable=protected-access
      raise ValueError(f'External preflight {stream} path/hash is malformed.')
    lexical_log_path = Path(log_path_text)
    _base._guard_path(lexical_log_path)  # pylint: disable=protected-access
    log_path = lexical_log_path.resolve()
    if (
        log_digest != expected_log_sha256
        or
        log_path.parent != preflight_root
        or lexical_log_path.is_symlink() or not lexical_log_path.is_file()
        or _sha256(log_path) != log_digest
    ):
      raise ValueError(f'External preflight {stream} log path/hash changed.')
    expected_validated_logs[stream] = {
        'path': str(log_path), 'sha256': log_digest,
    }
  if external.get('validated_logs') != expected_validated_logs:
    raise ValueError('Derived external preflight validated_logs changed.')
  if dict(logs) != expected_validated_logs:
    raise ValueError('Saved preflight logs are not canonical resolved bindings.')
  normalized = copy.deepcopy(dict(start))
  del normalized['external_preflight']['validated_logs']
  return normalized, {
      'derived_validated_logs_verified': True,
      'external_preflight_sha256': digest,
      'validated_logs': expected_validated_logs,
      'normalization': 'removed_exact_runner_derived_validated_logs_only',
  }


def _validate_preflight(
    start: Mapping[str, Any], freeze: Mapping[str, Any], freeze_sha: str
) -> dict[str, Any]:
  global _LAST_PREFLIGHT_NORMALIZATION_AUDIT
  normalized, audit = _normalize_external_preflight(start, freeze)
  result = _FROZEN_PREFLIGHT_VALIDATOR(normalized, freeze, freeze_sha)
  _LAST_PREFLIGHT_NORMALIZATION_AUDIT = audit
  return {**result, 'derived_validated_logs_verified': True}


def _validate_consumed_v3_2_2_attempt() -> dict[str, Any]:
  expected = {
      'ANALYSIS_ATTEMPT_STARTED.json': V3_2_2_ATTEMPT_SHA256,
      'ANALYSIS_FAILURE.json': V3_2_2_FAILURE_SHA256,
  }
  observed = {
      path.name for path in _CONSUMED_V3_2_2_DIR.iterdir()
  } if _CONSUMED_V3_2_2_DIR.is_dir() else set()
  if observed != set(expected):
    raise ValueError('Consumed v3.2.2 attempt tree is incomplete or has extras.')
  for filename, digest in expected.items():
    path = _CONSUMED_V3_2_2_DIR / filename
    if path.is_symlink() or not path.is_file() or _sha256(path) != digest:
      raise ValueError(f'Consumed v3.2.2 artifact changed: {filename}.')
  started = json.loads(
      (_CONSUMED_V3_2_2_DIR / 'ANALYSIS_ATTEMPT_STARTED.json').read_text(
          encoding='utf-8'
      )
  )
  failure = json.loads(
      (_CONSUMED_V3_2_2_DIR / 'ANALYSIS_FAILURE.json').read_text(
          encoding='utf-8'
      )
  )
  if (
      started.get('analysis_version') != 'opensplice-superset-analysis-v3.2.2'
      or started.get('status') != 'started_append_only_one_shot'
      or started.get('model_rerun_permitted') is not False
      or started.get('confirmation_model_calls_permitted') != 0
      or failure.get('status') != 'failed_consumed_no_retry'
      or failure.get('attempt_started_sha256') != V3_2_2_ATTEMPT_SHA256
      or failure.get('analysis_json_exists') is not False
      or failure.get('analysis_markdown_exists') is not False
      or failure.get('failure', {}).get('type') != 'ValueError'
      or failure.get('failure', {}).get('message')
      != 'Embedded external preflight differs from its artifact.'
  ):
    raise ValueError('Consumed v3.2.2 failure boundary changed.')
  return {
      'attempt_started_sha256': V3_2_2_ATTEMPT_SHA256,
      'failure_sha256': V3_2_2_FAILURE_SHA256,
      'failure_type': 'ValueError',
      'failure_message': failure['failure']['message'],
      'scientific_output_written': False,
  }


def _validate_amendment_preconditions(
    run_dir: Path, bundle_root: Path, *,
    expected_attempt_started_sha256: str | None = None,
) -> dict[str, Any]:
  _v321._assert_no_model_imports('v3.2.3 precondition process')  # pylint: disable=protected-access
  if _sha256(_V3_2_2_PATH) != V3_2_2_ANALYZER_SHA256:
    raise ValueError('Frozen v3.2.2 analyzer bytes changed.')
  if _sha256(_V3_2_2_TEST_PATH) != V3_2_2_TEST_SHA256:
    raise ValueError('Frozen v3.2.2 analyzer test bytes changed.')
  if _sha256(_V3_2_2_AMENDMENT_PATH) != V3_2_2_AMENDMENT_SHA256:
    raise ValueError('Frozen v3.2.2 amendment bytes changed.')
  prior_binding = _v321._validate_amendment_preconditions(  # pylint: disable=protected-access
      run_dir, bundle_root
  )
  consumed_v321 = _v322._validate_consumed_v3_2_1_attempt()  # pylint: disable=protected-access
  consumed_v322 = _validate_consumed_v3_2_2_attempt()
  if expected_attempt_started_sha256 is None and _ATTEMPT_DIR.exists():
    raise FileExistsError('The append-only v3.2.3 attempt was already consumed.')
  if not AMENDMENT_SHA256 or _sha256(_AMENDMENT_PATH) != AMENDMENT_SHA256:
    raise ValueError('Prospective v3.2.3 amendment bytes changed/unbound.')
  tracked_paths = (_AMENDMENT_PATH, Path(__file__).resolve(), _TEST_PATH.resolve())
  for path in tracked_paths:
    _base._guard_path(path)  # pylint: disable=protected-access
    try:
      relative = str(path.relative_to(bundle_root.resolve()))
    except ValueError as error:
      raise ValueError('v3.2.3 amendment file escapes repository.') from error
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
      'v3_2_2_bundle_sha256': {
          str(_V3_2_2_PATH.relative_to(bundle_root.resolve())): (
              V3_2_2_ANALYZER_SHA256
          ),
          str(_V3_2_2_TEST_PATH.relative_to(bundle_root.resolve())): (
              V3_2_2_TEST_SHA256
          ),
          str(_V3_2_2_AMENDMENT_PATH.relative_to(bundle_root.resolve())): (
              V3_2_2_AMENDMENT_SHA256
          ),
      },
      'consumed_v3_2_1_attempt': consumed_v321,
      'consumed_v3_2_2_attempt': consumed_v322,
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
  global _LAST_PREFLIGHT_NORMALIZATION_AUDIT
  _LAST_PREFLIGHT_NORMALIZATION_AUDIT = None
  _v322._LAST_PROTOBUF_AUDIT = None  # pylint: disable=protected-access
  _v321._LAST_CHECKPOINT_AUDIT = None  # pylint: disable=protected-access
  verified = None
  if enforce_standard_locations:
    verified = _validate_amendment_preconditions(
        run_dir, bundle_root,
        expected_attempt_started_sha256=attempt_started_sha256,
    )
    if amendment_binding != verified:
      raise ValueError('v3.2.3 amendment binding is absent or changed.')
    if not _base._is_sha256(attempt_started_sha256):  # pylint: disable=protected-access
      raise ValueError('v3.2.3 append-only attempt binding is absent.')
  _base._validate_checkpoint_and_reference_inputs = (  # pylint: disable=protected-access
      _v321._validate_checkpoint_and_reference_inputs  # pylint: disable=protected-access
  )
  _base._validate_bootstrap_attestation = _v322._validate_bootstrap_attestation  # pylint: disable=protected-access
  _base._validate_preflight = _validate_preflight  # pylint: disable=protected-access
  _base.ANALYSIS_VERSION = ANALYSIS_VERSION
  try:
    result = _base.analyze(
        run_dir, bundle_root=bundle_root, ignored_paths=ignored_paths,
        enforce_standard_locations=enforce_standard_locations,
    )
  finally:
    _v321._assert_no_model_imports('v3.2.3 post-analysis process')  # pylint: disable=protected-access
  if _LAST_PREFLIGHT_NORMALIZATION_AUDIT is None:
    raise RuntimeError('v3.2.3 preflight normalization audit was not recorded.')
  if _v322._LAST_PROTOBUF_AUDIT is None:  # pylint: disable=protected-access
    raise RuntimeError('v3.2.2 protobuf normalization audit was not recorded.')
  if _v321._LAST_CHECKPOINT_AUDIT is None:  # pylint: disable=protected-access
    raise RuntimeError('v3.2.1 checkpoint normalization audit was not recorded.')
  result['analyzer_amendments'] = {
      'model_run_analysis_version': 'opensplice-superset-analysis-v3.2.0',
      'offline_analysis_version': ANALYSIS_VERSION,
      'v3_2_3_amendment_sha256': AMENDMENT_SHA256,
      'v3_2_3_amendment_binding': verified,
      'analysis_attempt_started_sha256': attempt_started_sha256,
      'checkpoint_symlink_audit': _v321._LAST_CHECKPOINT_AUDIT,  # pylint: disable=protected-access
      'protobuf_generated_outputs_audit': _v322._LAST_PROTOBUF_AUDIT,  # pylint: disable=protected-access
      'external_preflight_normalization_audit': (
          _LAST_PREFLIGHT_NORMALIZATION_AUDIT
      ),
      'preserved_analyzer_failures': [
          {
              'analysis_version': 'opensplice-superset-analysis-v3.2.0',
              'reason': 'manifest_bound_hf_snapshot_symlink_rejected',
              'scientific_output_written': False,
          },
          {
              'analysis_version': 'opensplice-superset-analysis-v3.2.1',
              'reason': 'path_keyed_generated_pyi_omitted',
              'scientific_output_written': False,
          },
          {
              'analysis_version': 'opensplice-superset-analysis-v3.2.2',
              'reason': 'derived_validated_logs_compared_as_artifact_field',
              'attempt_started_sha256': V3_2_2_ATTEMPT_SHA256,
              'failure_sha256': V3_2_2_FAILURE_SHA256,
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
        'The append-only v3.2.3 corrected-analysis attempt was already consumed.'
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
    raise ValueError('v3.2.3 attempt-start digest is malformed.')
  entries = list(_ATTEMPT_DIR.iterdir()) if _ATTEMPT_DIR.is_dir() else []
  if len(entries) != 1 or entries[0].name != 'ANALYSIS_ATTEMPT_STARTED.json':
    raise ValueError('v3.2.3 started attempt has extra or terminal artifacts.')
  path = entries[0]
  if path.is_symlink() or not path.is_file() or _sha256(path) != started_sha256:
    raise ValueError('v3.2.3 attempt-start artifact hash/type changed.')
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
    raise ValueError('v3.2.3 attempt-start content/binding changed.')


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
          'The model run used v3.2.0. Offline analysis used prospective v3.2.1 '
          'checkpoint-symlink, v3.2.2 protobuf path-key, and v3.2.3 derived '
          'preflight-log validation repairs. No repair changed a scientific '
          'gate or permitted a model rerun.\n'
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
