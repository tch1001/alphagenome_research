#!/usr/bin/env python3
"""Analyzer-only v3.2.1 amendment for manifest-bound HF snapshot symlinks.

The frozen v3.2 analyzer rejected every checkpoint symlink, even though a
Hugging Face snapshot is intentionally a directory of symlinks into its sibling
``blobs`` directory.  This wrapper leaves the frozen analyzer untouched and
replaces only its checkpoint/reference-input validator.  Scientific reducers,
gates, rankings, decisions, and confirmation isolation remain v3.2 code.
"""

from __future__ import annotations

import argparse
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
from typing import Any, Mapping, Sequence


ANALYSIS_VERSION = 'opensplice-superset-analysis-v3.2.1'
AMENDMENT_REASON = 'hf_snapshot_manifest_symlink_compatibility'
ORIGINAL_ANALYZER_SHA256 = (
    'a098cba37b172ab741e159ce9e4701219a0280b7376afe408d6c0d2d8f58c7cc'
)
AMENDMENT_SHA256 = (
    '69207fbee072bfacca3af7d21361b6257839b6a0bab3791643f1c4d36bf75ed3'
)
ATTEMPT_STARTED_SHA256 = (
    '2fb661d35c431ce03f2f62199a32a2ef5a4827294c91a980029c233616e060c8'
)
RUN_COMPLETE_SHA256 = (
    '4d59b74584528d6ed8149a36f80e506c37d6ee939cd75b4a22da5bb230fd2425'
)
RAW_MANIFEST_SHA256 = (
    '2d63d7dfeaa69e2c1ad8cde731e656e134e37e639023f0745daadb564f17a665'
)
RAW_ARTIFACT_TREE_SHA256 = (
    '4171d8aebae7fff3b9981d7ab0dc914c659c6fe2916cb9a48bbee87e205beed8'
)
ORIGINAL_FREEZE_SHA256 = (
    '526b40899736e1a0442f51bf5b5dae3a2cea89ab4aa680b7c0c6189ce0d8dc4f'
)
_HERE = Path(__file__).resolve().parent
_ORIGINAL_PATH = _HERE / 'analyze_superset_graph_v3_2.py'
_AMENDMENT_PATH = (
    _HERE / 'v3_wider_mechanism/superset_graph_analysis_amendment_v3_2_1.md'
)
_TEST_PATH = _HERE / 'analyze_superset_graph_v3_2_1_test.py'
_ATTEMPT_DIR = (
    _HERE / 'results/v3_2_development_superset_graph_analysis_v3_2_1_attempt'
)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def _assert_no_model_imports(label: str) -> None:
  forbidden = sorted(
      name for name in sys.modules
      if name == 'jax' or name.startswith('jax.')
      or name == 'alphagenome' or name.startswith('alphagenome.')
      or name == 'alphagenome_research'
      or name.startswith('alphagenome_research.')
  )
  if forbidden:
    raise RuntimeError(f'{label} contains forbidden model/JAX imports: {forbidden}.')


def _load_frozen_analyzer():
  _assert_no_model_imports('pre-analyzer-import process')
  if _sha256(_ORIGINAL_PATH) != ORIGINAL_ANALYZER_SHA256:
    raise ValueError('Frozen v3.2 analyzer bytes changed before import.')
  specification = importlib.util.spec_from_file_location(
      '_opensplice_frozen_analyzer_v3_2', _ORIGINAL_PATH
  )
  if specification is None or specification.loader is None:
    raise RuntimeError('Cannot load the frozen v3.2 analyzer.')
  module = importlib.util.module_from_spec(specification)
  specification.loader.exec_module(module)
  _assert_no_model_imports('post-analyzer-import process')
  return module


_v3 = _load_frozen_analyzer()
_LAST_CHECKPOINT_AUDIT: dict[str, Any] | None = None


def _assert_global_tracked_head_clean(bundle_root: Path) -> None:
  try:
    difference = subprocess.check_output(
        ('git', '-C', str(bundle_root), 'diff', '--binary', 'HEAD', '--')
    )
  except subprocess.CalledProcessError as error:
    raise ValueError('Cannot audit the repository tracked HEAD state.') from error
  if difference:
    raise ValueError('Repository tracked files differ from committed HEAD.')


def _validate_amendment_preconditions(
    run_dir: Path, bundle_root: Path,
) -> dict[str, Any]:
  """Binds the prospective amendment before any individual raw record read."""
  _assert_no_model_imports('v3.2.1 precondition process')
  run_dir = run_dir.resolve()
  bundle_root = bundle_root.resolve()
  _v3._guard_path(run_dir)  # pylint: disable=protected-access
  if run_dir != _v3._OUTPUT_DIR.resolve():  # pylint: disable=protected-access
    raise ValueError('v3.2.1 accepts only the frozen completed run directory.')
  expected_top_hashes = {
      'ATTEMPT_STARTED.json': ATTEMPT_STARTED_SHA256,
      'RUN_COMPLETE.json': RUN_COMPLETE_SHA256,
      'RAW_MANIFEST.json': RAW_MANIFEST_SHA256,
  }
  for filename, expected in expected_top_hashes.items():
    if _sha256(run_dir / filename) != expected:
      raise ValueError(f'Completed raw-run binding changed: {filename}.')
  start = json.loads((run_dir / 'ATTEMPT_STARTED.json').read_text(encoding='utf-8'))
  complete = json.loads((run_dir / 'RUN_COMPLETE.json').read_text(encoding='utf-8'))
  manifest = json.loads((run_dir / 'RAW_MANIFEST.json').read_text(encoding='utf-8'))
  if (
      start.get('freeze', {}).get('sha256') != ORIGINAL_FREEZE_SHA256
      or start.get('confirmation_model_calls') != 0
  ):
    raise ValueError('Completed attempt freeze/confirmation binding changed.')
  exact_complete = {
      'status': 'complete',
      'identity_count': 20,
      'eligible_effect_count': 12,
      'phase_r_group_count': 2592,
      'phase_r_invalid_count': 0,
      'stage_a_group_count': 48,
      'stage_a_invalid_count': 0,
      'closures_passed': True,
      'confirmation_model_calls': 0,
  }
  for field, expected in exact_complete.items():
    if complete.get(field) != expected:
      raise ValueError(f'Completed-run field changed at {field}.')
  if (
      manifest.get('artifact_count') != 2660
      or manifest.get('artifact_tree_sha256') != RAW_ARTIFACT_TREE_SHA256
      or complete.get('raw_manifest') != manifest
  ):
    raise ValueError('Completed raw manifest count/tree binding changed.')
  if _sha256(_AMENDMENT_PATH) != AMENDMENT_SHA256:
    raise ValueError('Prospective v3.2.1 amendment bytes changed.')
  if _sha256(_ORIGINAL_PATH) != ORIGINAL_ANALYZER_SHA256:
    raise ValueError('Frozen v3.2 analyzer bytes changed.')
  _assert_global_tracked_head_clean(bundle_root)
  tracked_paths = (_AMENDMENT_PATH, Path(__file__).resolve(), _TEST_PATH.resolve())
  relative_paths = []
  for path in tracked_paths:
    _v3._guard_path(path)  # pylint: disable=protected-access
    try:
      relative = str(path.relative_to(bundle_root))
    except ValueError as error:
      raise ValueError('v3.2.1 amendment file escapes repository.') from error
    subprocess.run(
        ('git', '-C', str(bundle_root), 'ls-files', '--error-unmatch', relative),
        check=True, capture_output=True,
    )
    relative_paths.append(relative)
  if subprocess.check_output(
      ('git', '-C', str(bundle_root), 'diff', '--binary', 'HEAD', '--',
       *relative_paths)
  ):
    raise ValueError('v3.2.1 analyzer/amendment/tests differ from committed HEAD.')
  git_head = subprocess.check_output(
      ('git', '-C', str(bundle_root), 'rev-parse', 'HEAD'), text=True
  ).strip()
  if (
      len(git_head) != 40
      or any(character not in '0123456789abcdef' for character in git_head)
  ):
    raise ValueError('v3.2.1 Git HEAD is malformed.')
  if _v3._ANALYSIS_DIR.exists():  # pylint: disable=protected-access
    raise FileExistsError('Frozen analysis destination already exists.')
  hashes = {str(path.relative_to(bundle_root)): _sha256(path) for path in tracked_paths}
  return {
      'git_head': git_head,
      'tracked_head_clean': True,
      'file_sha256': hashes,
      'amendment_sha256': AMENDMENT_SHA256,
      'completed_run_sha256': expected_top_hashes,
      'raw_artifact_count': 2660,
      'raw_artifact_tree_sha256': RAW_ARTIFACT_TREE_SHA256,
  }


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


def _start_analysis_attempt(
    run_dir: Path, output_json: Path, output_markdown: Path | None,
    amendment_binding: Mapping[str, Any],
) -> tuple[Path, str]:
  _v3._guard_path(_ATTEMPT_DIR)  # pylint: disable=protected-access
  try:
    _ATTEMPT_DIR.mkdir(mode=0o755, parents=False, exist_ok=False)
  except FileExistsError as error:
    raise FileExistsError(
        'The append-only v3.2.1 corrected-analysis attempt was already consumed.'
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
  path = _ATTEMPT_DIR / 'ANALYSIS_ATTEMPT_STARTED.json'
  return path, _write_new_json(path, record)


def _persist_attempt_terminal(
    filename: str, value: Mapping[str, Any], *, started_sha256: str
) -> None:
  record = {
      'analysis_version': ANALYSIS_VERSION,
      'attempt_started_sha256': started_sha256,
      'recorded_at_unix_s': time.time(),
      **value,
  }
  _write_new_json(_ATTEMPT_DIR / filename, record)


def _strict_regular_file(path: Path, label: str) -> None:
  try:
    mode = path.stat().st_mode
  except OSError as error:
    raise ValueError(f'{label} cannot be statted.') from error
  if not stat.S_ISREG(mode):
    raise ValueError(f'{label} does not resolve to a regular file.')


def _pinned_hf_blobs_root(checkpoint_path: Path) -> Path | None:
  model_cache = checkpoint_path.parent.parent
  expected_hf_model = 'models--google--alphagenome-all-folds'
  if (
      checkpoint_path.parent.name == 'snapshots'
      and checkpoint_path.name == _v3.CHECKPOINT_SNAPSHOT
      and model_cache.name == expected_hf_model
  ):
    root = (model_cache / 'blobs').resolve()
    _v3._guard_path(root)  # pylint: disable=protected-access
    return root
  return None


def _validate_checkpoint_tree(
    checkpoint_path: Path, manifest_path: Path,
    manifest_sha256: str, checkpoint_binding: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
  """Validates exact lexical entries and contained symlink targets."""
  checkpoint_path = checkpoint_path.resolve()
  _v3._guard_path(checkpoint_path)  # pylint: disable=protected-access
  if (
      checkpoint_path.name != _v3.CHECKPOINT_SNAPSHOT
      or not checkpoint_path.is_dir()
  ):
    raise ValueError('Checkpoint snapshot path/name is not the frozen snapshot.')
  if _v3._sha256(manifest_path) != manifest_sha256:  # pylint: disable=protected-access
    raise ValueError('Checkpoint-manifest content differs from the freeze.')
  blobs_root = _pinned_hf_blobs_root(checkpoint_path)
  records = []
  files = []
  symlink_records = []
  seen_relative = set()
  try:
    lines = manifest_path.read_text(encoding='utf-8').splitlines()
  except OSError as error:
    raise ValueError('Cannot read frozen checkpoint manifest.') from error
  for line_number, line in enumerate(lines, start=1):
    fields = line.split('\t')
    if len(fields) != 3:
      raise ValueError(
          f'Checkpoint manifest row {line_number} does not have three columns.'
      )
    relative, size_text, digest = fields
    relative_path = Path(relative)
    if (
        not relative or relative_path.is_absolute() or '..' in relative_path.parts
        or relative in seen_relative or not _v3._is_sha256(digest)  # pylint: disable=protected-access
    ):
      raise ValueError(f'Checkpoint manifest row {line_number} is unsafe/invalid.')
    try:
      size = int(size_text)
    except ValueError as error:
      raise ValueError('Checkpoint manifest size is not an integer.') from error
    if size < 0 or str(size) != size_text:
      raise ValueError('Checkpoint manifest size is non-canonical.')
    lexical_path = checkpoint_path / relative_path
    _v3._guard_path(lexical_path)  # pylint: disable=protected-access
    try:
      lexical_path.relative_to(checkpoint_path)
    except ValueError as error:
      raise ValueError('Checkpoint lexical path escapes its snapshot.') from error
    is_symlink = lexical_path.is_symlink()
    try:
      resolved_path = lexical_path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
      raise ValueError(f'Checkpoint path cannot be resolved: {relative}.') from error
    _v3._guard_path(resolved_path)  # pylint: disable=protected-access
    _strict_regular_file(resolved_path, f'Checkpoint target {relative}')
    if is_symlink:
      try:
        link_text = os.readlink(lexical_path)
      except OSError as error:
        raise ValueError(f'Cannot read checkpoint symlink: {relative}.') from error
      link_parts = Path(link_text).parts
      blob_id = link_parts[-1] if link_parts else ''
      expected_link_parts = (
          ('..',) * (len(relative_path.parts) + 1)
          + ('blobs', blob_id)
      )
      if (
          Path(link_text).is_absolute()
          or link_parts != expected_link_parts
          or len(blob_id) not in (40, 64)
          or any(character not in '0123456789abcdef' for character in blob_id)
      ):
        raise ValueError(
            f'Checkpoint symlink is not the exact depth-derived relative '
            f'link to blobs/<hex>: {relative}.'
        )
      direct_target = Path(os.path.abspath(lexical_path.parent / link_text))
      if (
          blobs_root is None
          or direct_target.parent.resolve() != blobs_root
          or direct_target.name != blob_id
          or direct_target.is_symlink()
          or resolved_path.parent != blobs_root
          or resolved_path != direct_target
      ):
        raise ValueError(
            f'Checkpoint symlink is chained, cross-repository, or escaping: '
            f'{relative}.'
        )
      symlink_records.append({
          'relative_path': relative,
          'resolved_path': str(resolved_path),
          'resolved_root': str(blobs_root),
          'link_text': link_text,
      })
    elif not resolved_path.is_relative_to(checkpoint_path):
      raise ValueError(f'Checkpoint file escapes its snapshot: {relative}.')
    if (
        resolved_path.stat().st_size != size
        or _v3._sha256(resolved_path) != digest  # pylint: disable=protected-access
    ):
      raise ValueError(f'Checkpoint file content changed: {relative}.')
    seen_relative.add(relative)
    files.append(lexical_path)
    records.append({
        'relative_path': relative, 'size_bytes': size, 'sha256': digest,
    })
  if (
      len(records) != 12
      or [row['relative_path'] for row in records]
      != sorted(row['relative_path'] for row in records)
  ):
    raise ValueError('Checkpoint manifest must contain exactly 12 sorted files.')
  expected_files = [row['relative_path'] for row in records]
  allowed_directories = {
      str(parent)
      for row in records
      for parent in Path(row['relative_path']).parents
      if str(parent) != '.'
  }
  observed_files = []
  for path in checkpoint_path.rglob('*'):
    relative = str(path.relative_to(checkpoint_path))
    if path.is_symlink() or path.is_file():
      observed_files.append(relative)
    elif path.is_dir():
      if relative not in allowed_directories:
        raise ValueError(f'Checkpoint tree has an extra directory: {relative}.')
    else:
      raise ValueError(f'Checkpoint tree has an unrecognized entry: {relative}.')
  if sorted(observed_files) != expected_files:
    raise ValueError('Checkpoint tree differs from its exact 12-file manifest.')
  expected_binding = {
      'snapshot_path': str(checkpoint_path),
      'snapshot_name': _v3.CHECKPOINT_SNAPSHOT,
      'manifest_path': str(manifest_path),
      'manifest_sha256': manifest_sha256,
      'file_count': 12,
      'files': records,
  }
  if checkpoint_binding != expected_binding:
    raise ValueError('START checkpoint binding differs from the verified tree.')
  tree_digest = hashlib.sha256()
  for row in records:
    tree_digest.update(row['relative_path'].encode('utf-8'))
    tree_digest.update(b'\0')
    tree_digest.update(bytes.fromhex(row['sha256']))
  return {
      'checkpoint_binding': expected_binding,
      'checkpoint_manifest_sha256': manifest_sha256,
      'checkpoint_file_count': 12,
      'checkpoint_tree_sha256': tree_digest.hexdigest(),
      'checkpoint_symlink_count': len(symlink_records),
      'checkpoint_symlinks': symlink_records,
      'checkpoint_symlink_policy': (
          'exact_depth_derived_one_hop_relative_links_to_direct_regular_children_of_'
          'pinned_hf_blobs_with_manifest_size_and_sha256'
      ),
  }


def _validate_checkpoint_and_reference_inputs(
    start: Mapping[str, Any], freeze: Mapping[str, Any],
    expected_cases: Sequence[Mapping[str, Any]], *,
    enforce_standard_locations: bool,
) -> dict[str, Any]:
  """v3.2 input audit with the sole symlink-policy amendment."""
  global _LAST_CHECKPOINT_AUDIT
  manifest_path = Path(str(freeze['checkpoint_manifest_path'])).resolve()
  reference_path = Path(str(freeze['reference_bindings_path'])).resolve()
  for path in (manifest_path, reference_path):
    _v3._guard_path(path)  # pylint: disable=protected-access
  if enforce_standard_locations:
    standard_paths = {
        'development_variants_path': _v3._SELECTED_PATH.resolve(),  # pylint: disable=protected-access
        'development_exons_path': _v3._EXONS_PATH.resolve(),  # pylint: disable=protected-access
        'checkpoint_manifest_path': _v3._CHECKPOINT_MANIFEST_PATH.resolve(),  # pylint: disable=protected-access
        'reference_bindings_path': _v3._REFERENCE_BINDINGS_PATH.resolve(),  # pylint: disable=protected-access
    }
    for name, expected in standard_paths.items():
      if Path(str(freeze[name])).resolve() != expected:
        raise ValueError(f'Frozen standard input path changed at {name}.')
    if freeze['checkpoint_manifest_sha256'] != _v3.CHECKPOINT_MANIFEST_SHA256:
      raise ValueError('Frozen production checkpoint-manifest hash changed.')
    if freeze['reference_bindings_sha256'] != _v3.REFERENCE_BINDINGS_SHA256:
      raise ValueError('Frozen production reference-binding hash changed.')
    if set(freeze['file_sha256']) != _v3._expected_production_bundle_paths():  # pylint: disable=protected-access
      raise ValueError('Frozen production file-hash inventory is not exact.')
  if _v3._sha256(reference_path) != freeze['reference_bindings_sha256']:  # pylint: disable=protected-access
    raise ValueError('Reference-sequence binding content differs from the freeze.')

  checkpoint_value = start.get('checkpoint_path')
  checkpoint_binding = start.get('checkpoint_binding')
  if not isinstance(checkpoint_value, str) or not isinstance(
      checkpoint_binding, Mapping
  ):
    raise ValueError('Attempt checkpoint path/binding is missing.')
  checkpoint_audit = _validate_checkpoint_tree(
      Path(checkpoint_value), manifest_path,
      str(freeze['checkpoint_manifest_sha256']), checkpoint_binding, freeze,
  )

  reference_binding = _v3._read_json(reference_path)  # pylint: disable=protected-access
  if set(reference_binding) != {'reference_url', 'context_bp', 'cases'}:
    raise ValueError('Reference-sequence binding has unexpected/missing keys.')
  if (
      reference_binding['reference_url'] != _v3._REFERENCE_URL  # pylint: disable=protected-access
      or reference_binding['context_bp'] != 16_384
      or not isinstance(reference_binding['cases'], Mapping)
  ):
    raise ValueError('Reference-sequence binding header changed.')
  expected_ids = [case['variant_id'] for case in expected_cases]
  if set(reference_binding['cases']) != set(expected_ids):
    raise ValueError('Reference-sequence binding does not contain exact 20 cases.')
  sequence_bindings = {}
  for case in expected_cases:
    variant = case['variant_id']
    row = reference_binding['cases'][variant]
    center = (case['exon_start_1based'] + case['exon_end_1based']) // 2
    expected_start = center - 1 - 8192
    if (
        not isinstance(row, list) or len(row) != 6
        or row[:4] != [
            case['order'], case['chromosome'], expected_start,
            expected_start + 16_384,
        ]
        or not _v3._is_sha256(row[4]) or not _v3._is_sha256(row[5])  # pylint: disable=protected-access
        or row[4] == row[5]
    ):
      raise ValueError(f'Reference/sequence binding changed for {variant}.')
    sequence_bindings[variant] = {
        'chromosome': row[1], 'start_0based': row[2],
        'end_0based_exclusive': row[3],
        'reference': row[4], 'alternate': row[5],
    }
  expected_reference_sequence_binding = {
      'path': str(reference_path),
      'sha256': freeze['reference_bindings_sha256'],
  }
  if start.get('reference_sequence_bindings') != expected_reference_sequence_binding:
    raise ValueError('START reference-sequence file binding changed.')
  expected_reference_object_binding = {
      **_v3._REFERENCE_OBJECT,  # pylint: disable=protected-access
      'observed_generation': _v3._REFERENCE_OBJECT['generation'],  # pylint: disable=protected-access
      'observed_size_bytes': _v3._REFERENCE_OBJECT['size_bytes'],  # pylint: disable=protected-access
      'observed_etag': _v3._REFERENCE_OBJECT['etag'],  # pylint: disable=protected-access
      'observed_md5_base64': _v3._REFERENCE_OBJECT['md5_base64'],  # pylint: disable=protected-access
      'observed_crc32c_base64': _v3._REFERENCE_OBJECT['crc32c_base64'],  # pylint: disable=protected-access
  }
  if start.get('reference_object_binding') != expected_reference_object_binding:
    raise ValueError('START GCS reference-object metadata binding changed.')
  _LAST_CHECKPOINT_AUDIT = checkpoint_audit
  return {
      **checkpoint_audit,
      'reference_object': dict(_v3._REFERENCE_OBJECT),  # pylint: disable=protected-access
      'reference_object_metadata_verified': True,
      'reference_bindings_sha256': freeze['reference_bindings_sha256'],
      'sequence_bindings': sequence_bindings,
  }


def analyze(
    run_dir: Path, *, bundle_root: Path = _v3._REPO_ROOT,  # pylint: disable=protected-access
    ignored_paths: Sequence[Path] = (), enforce_standard_locations: bool = True,
    amendment_binding: Mapping[str, Any] | None = None,
    attempt_started_sha256: str | None = None,
) -> dict[str, Any]:
  """Runs the frozen v3.2 scientific analyzer with the v3.2.1 path amendment."""
  global _LAST_CHECKPOINT_AUDIT
  _LAST_CHECKPOINT_AUDIT = None
  if _sha256(_ORIGINAL_PATH) != ORIGINAL_ANALYZER_SHA256:
    raise ValueError('Frozen v3.2 analyzer bytes changed.')
  verified_amendment = None
  if enforce_standard_locations:
    verified_amendment = _validate_amendment_preconditions(
        run_dir, bundle_root
    )
    if amendment_binding != verified_amendment:
      raise ValueError('v3.2.1 amendment binding is absent or changed.')
    if not _v3._is_sha256(attempt_started_sha256):  # pylint: disable=protected-access
      raise ValueError('v3.2.1 append-only attempt binding is absent.')
  _v3._validate_checkpoint_and_reference_inputs = (  # pylint: disable=protected-access
      _validate_checkpoint_and_reference_inputs
  )
  _v3.ANALYSIS_VERSION = ANALYSIS_VERSION
  try:
    result = _v3.analyze(
        run_dir, bundle_root=bundle_root, ignored_paths=ignored_paths,
        enforce_standard_locations=enforce_standard_locations,
    )
  finally:
    _assert_no_model_imports('post-analysis process')
  if _LAST_CHECKPOINT_AUDIT is None:
    raise RuntimeError('Checkpoint amendment audit was not recorded.')
  result['analyzer_amendment'] = {
      'analysis_version': ANALYSIS_VERSION,
      'reason': AMENDMENT_REASON,
      'frozen_v3_2_analyzer_sha256': ORIGINAL_ANALYZER_SHA256,
      'prospective_amendment_sha256': AMENDMENT_SHA256,
      'amendment_binding': verified_amendment,
      'analysis_attempt_started_sha256': attempt_started_sha256,
      'initial_v3_2_analyzer_failure_preserved': (
          'manifest-bound Hugging Face snapshot symlink rejected as '
          'missing_or_symlinked before any scientific result was emitted'
      ),
      'model_rerun_permitted': False,
      'scientific_gate_or_estimand_changed': False,
      'checkpoint_symlink_count': _LAST_CHECKPOINT_AUDIT[
          'checkpoint_symlink_count'
      ],
      'checkpoint_symlink_policy': _LAST_CHECKPOINT_AUDIT[
          'checkpoint_symlink_policy'
      ],
      'checkpoint_symlinks': _LAST_CHECKPOINT_AUDIT['checkpoint_symlinks'],
  }
  return result


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run-dir', type=Path, required=True)
  parser.add_argument('--output-json', type=Path, required=True)
  parser.add_argument('--output-markdown', type=Path)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  _v3._guard_path(args.output_json)  # pylint: disable=protected-access
  if args.output_markdown is not None:
    _v3._guard_path(args.output_markdown)  # pylint: disable=protected-access
  run_dir = args.run_dir.resolve()
  if args.output_json.resolve() == run_dir or run_dir in args.output_json.resolve().parents:
    raise ValueError('Analysis output cannot be inside the append-only raw run.')
  ignored = [args.output_json]
  if args.output_markdown is not None:
    ignored.append(args.output_markdown)
  amendment_binding = _validate_amendment_preconditions(
      args.run_dir, _v3._REPO_ROOT  # pylint: disable=protected-access
  )
  analysis_dir = _v3._ANALYSIS_DIR.resolve()  # pylint: disable=protected-access
  if args.output_json.resolve() != analysis_dir / 'ANALYSIS.json':
    raise ValueError('JSON output path differs from frozen analysis destination.')
  if args.output_markdown is not None and args.output_markdown.resolve() != (
      analysis_dir / 'RESULT.md'
  ):
    raise ValueError('Markdown output path differs from frozen destination.')
  if analysis_dir.exists():
    raise FileExistsError('Frozen analysis directory already exists; never overwrite.')
  _, started_sha = _start_analysis_attempt(
      args.run_dir, args.output_json, args.output_markdown, amendment_binding
  )
  try:
    result = analyze(
        args.run_dir, ignored_paths=ignored,
        amendment_binding=amendment_binding,
        attempt_started_sha256=started_sha,
    )
    _v3._write_atomic(  # pylint: disable=protected-access
        args.output_json,
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n',
    )
    if args.output_markdown is not None:
      markdown = _v3.render_markdown(result)
      amendment = (
          '\n## Analyzer-only v3.2.1 amendment\n\n'
          'The frozen v3.2 analyzer first failed because it rejected manifest-'
          'bound Hugging Face snapshot symlinks. No model rerun or scientific '
          'gate change was permitted. v3.2.1 accepts only exact-hash regular-'
          'file targets reached by one exact relative link into the pinned '
          'Hugging Face blobs directory.\n'
      )
      _v3._write_atomic(args.output_markdown, markdown + amendment)  # pylint: disable=protected-access
    _persist_attempt_terminal(
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
      _persist_attempt_terminal(
          'ANALYSIS_FAILURE.json',
          {
              'status': 'failed_consumed_no_retry',
              'failure': {
                  'type': type(error).__name__,
                  'message': str(error),
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
