#!/usr/bin/env python3
"""Analyzer-only v3.2.2 fix for path-keyed protobuf output bindings.

The v3.2.1 checkpoint repair remains unchanged.  This wrapper normalizes only
the already-frozen ``protobuf_binding.generated_outputs`` path-key schema for
the original bootstrap validator.  Scientific evidence, reducers, thresholds,
rankings, and decisions continue to run through the frozen v3.2 analyzer.
"""

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


ANALYSIS_VERSION = 'opensplice-superset-analysis-v3.2.2'
AMENDMENT_REASON = 'protobuf_generated_outputs_path_key_normalization'
V3_2_1_ANALYZER_SHA256 = (
    '250236f4e1fd4c7712f7862da06e4bf933d731343dd306ae967a03a19e274313'
)
V3_2_1_TEST_SHA256 = (
    'deb96ebdf4d2d8c961bea87249565d721e046a85b23defefe9932160275f128c'
)
V3_2_1_ATTEMPT_SHA256 = (
    '2a77efe48eff892fd3d61b227b486c6820107236461de80a888414bed88ba38b'
)
V3_2_1_FAILURE_SHA256 = (
    '174e4bdc56126cab1c9fddd358e06ecaa796d770bbbf526676ac2262dc26311e'
)
AMENDMENT_SHA256 = (
    'a25d08c8a609703532a749ac0e5d0246614446627b84b4196f28be91ffdecb4f'
)

_HERE = Path(__file__).resolve().parent
_V3_2_1_PATH = _HERE / 'analyze_superset_graph_v3_2_1.py'
_V3_2_1_TEST_PATH = _HERE / 'analyze_superset_graph_v3_2_1_test.py'
_AMENDMENT_PATH = (
    _HERE / 'v3_wider_mechanism/superset_graph_analysis_amendment_v3_2_2.md'
)
_TEST_PATH = _HERE / 'analyze_superset_graph_v3_2_2_test.py'
_CONSUMED_V3_2_1_DIR = (
    _HERE / 'results/v3_2_development_superset_graph_analysis_v3_2_1_attempt'
)
_ATTEMPT_DIR = (
    _HERE / 'results/v3_2_development_superset_graph_analysis_v3_2_2_attempt'
)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def _load_v3_2_1():
  if _sha256(_V3_2_1_PATH) != V3_2_1_ANALYZER_SHA256:
    raise ValueError('Frozen v3.2.1 analyzer bytes changed before import.')
  specification = importlib.util.spec_from_file_location(
      '_opensplice_frozen_analyzer_v3_2_1', _V3_2_1_PATH
  )
  if specification is None or specification.loader is None:
    raise RuntimeError('Cannot load the frozen v3.2.1 analyzer.')
  module = importlib.util.module_from_spec(specification)
  specification.loader.exec_module(module)
  return module


_v321 = _load_v3_2_1()
_base = _v321._v3  # pylint: disable=protected-access
_FROZEN_BOOTSTRAP_VALIDATOR = _base._validate_bootstrap_attestation  # pylint: disable=protected-access
_LAST_PROTOBUF_AUDIT: dict[str, Any] | None = None


def _normalize_frozen_protobuf_binding(
    freeze: Mapping[str, Any], *, bundle_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Validates and expands the exact frozen path-key output schema."""
  protobuf = freeze.get('protobuf_binding')
  if not isinstance(protobuf, Mapping):
    raise ValueError('Frozen protobuf binding is missing.')
  outputs = protobuf.get('generated_outputs')
  expected_paths = {
      str((
          bundle_root
          / 'src/alphagenome_research/protos/calibration_scores_pb2.py'
      ).resolve()),
      str((
          bundle_root
          / 'src/alphagenome_research/protos/calibration_scores_pb2.pyi'
      ).resolve()),
  }
  if not isinstance(outputs, Mapping) or set(outputs) != expected_paths:
    raise ValueError(
        'Frozen protobuf generated_outputs must contain exactly pb2/pyi path keys.'
    )
  explicit_by_path: dict[str, list[Mapping[str, Any]]] = {}

  def collect_explicit(node: Any, *, inside_outputs: bool = False) -> None:
    if isinstance(node, Mapping):
      if not inside_outputs and isinstance(node.get('path'), str) and (
          'sha256' in node
      ):
        normalized_path = str(Path(node['path']).resolve())
        explicit_by_path.setdefault(normalized_path, []).append(node)
      for key, child in node.items():
        collect_explicit(
            child,
            inside_outputs=inside_outputs or key == 'generated_outputs',
        )
    elif isinstance(node, list):
      for child in node:
        collect_explicit(child, inside_outputs=inside_outputs)

  collect_explicit(protobuf)
  normalized = copy.deepcopy(dict(freeze))
  normalized_outputs = {}
  audit_outputs = {}
  for path_text in sorted(expected_paths):
    binding = outputs[path_text]
    if not isinstance(binding, Mapping) or set(binding) != {
        'sha256', 'size_bytes'
    }:
      raise ValueError('Frozen generated-output value schema changed.')
    digest, size = binding.get('sha256'), binding.get('size_bytes')
    if (
        not _base._is_sha256(digest)  # pylint: disable=protected-access
        or not isinstance(size, int) or isinstance(size, bool) or size < 0
    ):
      raise ValueError('Frozen generated-output hash/size is malformed.')
    path = Path(path_text).resolve()
    _base._guard_path(path)  # pylint: disable=protected-access
    try:
      path.relative_to(bundle_root.resolve())
    except ValueError as error:
      raise ValueError('Frozen generated output escapes the repository.') from error
    if path.is_symlink() or not path.is_file():
      raise ValueError('Frozen generated output is missing, symlinked, or non-file.')
    if path.stat().st_size != size or _sha256(path) != digest:
      raise ValueError('Frozen generated-output bytes changed.')
    for collision in explicit_by_path.get(path_text, []):
      if (
          collision.get('sha256') != digest
          or collision.get('size_bytes') != size
      ):
        raise ValueError(
            'Path-keyed and explicit-path protobuf bindings disagree.'
        )
    explicit = {'path': path_text, 'sha256': digest, 'size_bytes': size}
    normalized_outputs[path_text] = explicit
    audit_outputs[path_text] = {'sha256': digest, 'size_bytes': size}
  normalized['protobuf_binding']['generated_outputs'] = normalized_outputs
  return normalized, {
      'path_key_schema_verified': True,
      'generated_output_count': 2,
      'generated_outputs': audit_outputs,
  }


def _validate_bootstrap_attestation(
    start: Mapping[str, Any], freeze: Mapping[str, Any], freeze_sha: str, *,
    bundle_root: Path,
) -> dict[str, Any]:
  """Delegates to v3.2 after exact path-key normalization."""
  global _LAST_PROTOBUF_AUDIT
  protobuf = freeze.get('protobuf_binding')
  generated = start.get('same_process_pre_import_bootstrap', {}).get(
      'generated_bindings'
  )
  if not isinstance(protobuf, Mapping) or (
      protobuf.get('historical_generation_provenance')
      != 'unknown_not_reconstructed'
      or protobuf.get('regeneration_claim') is not False
      or protobuf.get('current_protoc_was_used_to_generate_frozen_outputs')
      is not False
      or protobuf.get('protobuf_runtime_version') != '7.35.1'
      or protobuf.get('byte_level_reproducibility')
      != (
          'generated outputs remain intentionally untracked and are frozen '
          'by exact path, size, and SHA256'
      )
  ):
    raise ValueError('Frozen protobuf provenance disclosures changed.')
  if not isinstance(generated, Mapping) or (
      generated.get('historical_generator_argv') != 'unknown'
      or generated.get('exact_regeneration_claim') is not False
      or generated.get('protobuf_runtime_version') != '7.35.1'
  ):
    raise ValueError('Bootstrap protobuf generation disclosures changed.')
  normalized, audit = _normalize_frozen_protobuf_binding(
      freeze, bundle_root=bundle_root
  )
  result = _FROZEN_BOOTSTRAP_VALIDATOR(
      start, normalized, freeze_sha, bundle_root=bundle_root
  )
  _LAST_PROTOBUF_AUDIT = audit
  return {
      **result,
      'frozen_generated_outputs_path_key_schema_verified': True,
      'frozen_generated_output_count': 2,
      'protobuf_generation_history_unknown_disclosed': True,
      'protobuf_regeneration_claim_made': False,
  }


def _validate_consumed_v3_2_1_attempt() -> dict[str, Any]:
  expected = {
      'ANALYSIS_ATTEMPT_STARTED.json': V3_2_1_ATTEMPT_SHA256,
      'ANALYSIS_FAILURE.json': V3_2_1_FAILURE_SHA256,
  }
  observed = {
      path.name for path in _CONSUMED_V3_2_1_DIR.iterdir()
  } if _CONSUMED_V3_2_1_DIR.is_dir() else set()
  if observed != set(expected):
    raise ValueError('Consumed v3.2.1 attempt tree is incomplete or has extras.')
  for filename, digest in expected.items():
    path = _CONSUMED_V3_2_1_DIR / filename
    if path.is_symlink() or not path.is_file() or _sha256(path) != digest:
      raise ValueError(f'Consumed v3.2.1 artifact changed: {filename}.')
  started = json.loads(
      (_CONSUMED_V3_2_1_DIR / 'ANALYSIS_ATTEMPT_STARTED.json').read_text(
          encoding='utf-8'
      )
  )
  failure = json.loads(
      (_CONSUMED_V3_2_1_DIR / 'ANALYSIS_FAILURE.json').read_text(
          encoding='utf-8'
      )
  )
  if (
      started.get('analysis_version') != 'opensplice-superset-analysis-v3.2.1'
      or started.get('status') != 'started_append_only_one_shot'
      or started.get('model_rerun_permitted') is not False
      or started.get('confirmation_model_calls_permitted') != 0
      or failure.get('status') != 'failed_consumed_no_retry'
      or failure.get('attempt_started_sha256') != V3_2_1_ATTEMPT_SHA256
      or failure.get('analysis_json_exists') is not False
      or failure.get('analysis_markdown_exists') is not False
      or failure.get('failure', {}).get('type') != 'ValueError'
      or failure.get('failure', {}).get('message')
      != 'Bootstrap artifact generated_pyi differs from frozen protobuf binding.'
  ):
    raise ValueError('Consumed v3.2.1 failure boundary changed.')
  return {
      'attempt_started_sha256': V3_2_1_ATTEMPT_SHA256,
      'failure_sha256': V3_2_1_FAILURE_SHA256,
      'failure_type': 'ValueError',
      'failure_message': failure['failure']['message'],
      'scientific_output_written': False,
  }


def _validate_amendment_preconditions(
    run_dir: Path, bundle_root: Path, *,
    expected_attempt_started_sha256: str | None = None,
) -> dict[str, Any]:
  _v321._assert_no_model_imports('v3.2.2 precondition process')  # pylint: disable=protected-access
  if _sha256(_V3_2_1_PATH) != V3_2_1_ANALYZER_SHA256:
    raise ValueError('Frozen v3.2.1 analyzer bytes changed.')
  if _sha256(_V3_2_1_TEST_PATH) != V3_2_1_TEST_SHA256:
    raise ValueError('Frozen v3.2.1 analyzer test bytes changed.')
  prior_binding = _v321._validate_amendment_preconditions(  # pylint: disable=protected-access
      run_dir, bundle_root
  )
  if expected_attempt_started_sha256 is None and _ATTEMPT_DIR.exists():
    raise FileExistsError('The append-only v3.2.2 attempt was already consumed.')
  consumed = _validate_consumed_v3_2_1_attempt()
  if not AMENDMENT_SHA256 or _sha256(_AMENDMENT_PATH) != AMENDMENT_SHA256:
    raise ValueError('Prospective v3.2.2 amendment bytes changed/unbound.')
  tracked_paths = (_AMENDMENT_PATH, Path(__file__).resolve(), _TEST_PATH.resolve())
  relative_paths = []
  for path in tracked_paths:
    _base._guard_path(path)  # pylint: disable=protected-access
    try:
      relative = str(path.relative_to(bundle_root.resolve()))
    except ValueError as error:
      raise ValueError('v3.2.2 amendment file escapes repository.') from error
    subprocess.run(
        ('git', '-C', str(bundle_root), 'ls-files', '--error-unmatch', relative),
        check=True, capture_output=True,
    )
    relative_paths.append(relative)
  _v321._assert_global_tracked_head_clean(bundle_root)  # pylint: disable=protected-access
  git_head = subprocess.check_output(
      ('git', '-C', str(bundle_root), 'rev-parse', 'HEAD'), text=True
  ).strip()
  hashes = {
      str(path.relative_to(bundle_root.resolve())): _sha256(path)
      for path in tracked_paths
  }
  result = {
      'git_head': git_head,
      'tracked_head_clean': True,
      'file_sha256': hashes,
      'amendment_sha256': AMENDMENT_SHA256,
      'v3_2_1_binding': prior_binding,
      'consumed_v3_2_1_attempt': consumed,
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
  """Runs unchanged v3.2 science with v3.2.1+v3.2.2 input repairs."""
  global _LAST_PROTOBUF_AUDIT
  _LAST_PROTOBUF_AUDIT = None
  verified = None
  if enforce_standard_locations:
    verified = _validate_amendment_preconditions(
        run_dir, bundle_root,
        expected_attempt_started_sha256=attempt_started_sha256,
    )
    if amendment_binding != verified:
      raise ValueError('v3.2.2 amendment binding is absent or changed.')
    if not _base._is_sha256(attempt_started_sha256):  # pylint: disable=protected-access
      raise ValueError('v3.2.2 append-only attempt binding is absent.')
  _base._validate_checkpoint_and_reference_inputs = (  # pylint: disable=protected-access
      _v321._validate_checkpoint_and_reference_inputs  # pylint: disable=protected-access
  )
  _base._validate_bootstrap_attestation = _validate_bootstrap_attestation  # pylint: disable=protected-access
  _base.ANALYSIS_VERSION = ANALYSIS_VERSION
  try:
    result = _base.analyze(
        run_dir, bundle_root=bundle_root, ignored_paths=ignored_paths,
        enforce_standard_locations=enforce_standard_locations,
    )
  finally:
    _v321._assert_no_model_imports('v3.2.2 post-analysis process')  # pylint: disable=protected-access
  if _LAST_PROTOBUF_AUDIT is None:
    raise RuntimeError('v3.2.2 protobuf normalization audit was not recorded.')
  checkpoint_audit = _v321._LAST_CHECKPOINT_AUDIT  # pylint: disable=protected-access
  if checkpoint_audit is None:
    raise RuntimeError('v3.2.1 checkpoint symlink audit was not recorded.')
  result['analyzer_amendments'] = {
      'model_run_analysis_version': 'opensplice-superset-analysis-v3.2.0',
      'offline_analysis_version': ANALYSIS_VERSION,
      'v3_2_1_analyzer_sha256': V3_2_1_ANALYZER_SHA256,
      'v3_2_1_consumed_attempt': (
          verified['consumed_v3_2_1_attempt'] if verified else None
      ),
      'v3_2_1_checkpoint_symlink_audit': {
          'checkpoint_symlink_count': checkpoint_audit[
              'checkpoint_symlink_count'
          ],
          'checkpoint_symlink_policy': checkpoint_audit[
              'checkpoint_symlink_policy'
          ],
          'checkpoint_symlinks': checkpoint_audit['checkpoint_symlinks'],
      },
      'v3_2_2_amendment_sha256': AMENDMENT_SHA256,
      'v3_2_2_amendment_binding': verified,
      'analysis_attempt_started_sha256': attempt_started_sha256,
      'protobuf_generated_outputs_audit': _LAST_PROTOBUF_AUDIT,
      'preserved_analyzer_failures': [
          {
              'analysis_version': 'opensplice-superset-analysis-v3.2.0',
              'reason': 'manifest_bound_hf_snapshot_symlink_rejected',
              'scientific_output_written': False,
          },
          {
              'analysis_version': 'opensplice-superset-analysis-v3.2.1',
              'reason': (
                  'path_keyed_generated_pyi_omitted_from_frozen_path_collection'
              ),
              'attempt_started_sha256': V3_2_1_ATTEMPT_SHA256,
              'failure_sha256': V3_2_1_FAILURE_SHA256,
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
        'The append-only v3.2.2 corrected-analysis attempt was already consumed.'
    ) from error
  record = {
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
  }
  return _write_new_json(_ATTEMPT_DIR / 'ANALYSIS_ATTEMPT_STARTED.json', record)


def _validate_started_attempt(
    amendment_binding: Mapping[str, Any], started_sha256: str,
) -> None:
  if not _base._is_sha256(started_sha256):  # pylint: disable=protected-access
    raise ValueError('v3.2.2 attempt-start digest is malformed.')
  if not _ATTEMPT_DIR.is_dir():
    raise ValueError('v3.2.2 attempt directory is absent after start.')
  entries = list(_ATTEMPT_DIR.iterdir())
  if len(entries) != 1 or entries[0].name != 'ANALYSIS_ATTEMPT_STARTED.json':
    raise ValueError('v3.2.2 started attempt has extra or terminal artifacts.')
  path = entries[0]
  if path.is_symlink() or not path.is_file() or _sha256(path) != started_sha256:
    raise ValueError('v3.2.2 attempt-start artifact hash/type changed.')
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
    raise ValueError('v3.2.2 attempt-start content/binding changed.')


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
  run_dir = args.run_dir.resolve()
  analysis_dir = _base._ANALYSIS_DIR.resolve()  # pylint: disable=protected-access
  if args.output_json.resolve() != analysis_dir / 'ANALYSIS.json':
    raise ValueError('JSON output path differs from frozen analysis destination.')
  if args.output_markdown is not None and args.output_markdown.resolve() != (
      analysis_dir / 'RESULT.md'
  ):
    raise ValueError('Markdown output path differs from frozen destination.')
  if analysis_dir.exists():
    raise FileExistsError('Frozen analysis directory already exists; never overwrite.')
  binding = _validate_amendment_preconditions(run_dir, _base._REPO_ROOT)  # pylint: disable=protected-access
  started_sha = _start_attempt(
      run_dir, args.output_json, args.output_markdown, binding
  )
  ignored = [args.output_json]
  if args.output_markdown is not None:
    ignored.append(args.output_markdown)
  try:
    result = analyze(
        run_dir, ignored_paths=ignored, amendment_binding=binding,
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
          'checkpoint-symlink and v3.2.2 protobuf path-key validation repairs. '
          'Neither repair changed a scientific gate or permitted a model rerun.\n'
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
