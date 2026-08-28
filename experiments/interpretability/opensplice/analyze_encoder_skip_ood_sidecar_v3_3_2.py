#!/usr/bin/env python3
"""CPU-only structural audit for the prospective v3.3.2 OOD sidecar.

This analyzer never imports AlphaGenome or JAX and never computes a Shapley
value, ranking, resolution gate, or nomination.  It independently validates
the immutable v3.3 cube, the fresh sidecar tree, raw endpoint algebra, the
seven-role/two-generated-output provenance repair, and exact controlled-stop
prefixes.  A complete sidecar can only make a future, separately prospective
combined analysis structurally eligible.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


ANALYSIS_VERSION = 'opensplice-encoder-skip-ood-sidecar-analysis-v3.3.2'
SCRIPT_VERSION = 'opensplice-encoder-skip-ood-sidecar-v3.3.2'
ATTEMPT_ID = 'opensplice-v3.3.2-development-ood-sidecar-one-shot'
AMENDMENT_SHA256 = (
    '42cd43bbc25517d35b3e21dd5df7bf68a37ee46c51cb71bcdf363ac4de3b19e3'
)
AMENDMENT_COMMIT = '95d028f'
ORIGINAL_PROTOCOL_SHA256 = (
    '85151029297cf4b4c1a07dd2a0f47b10b0663acd08ddfe25ba72b9898b1beea0'
)
ORIGINAL_FREEZE_SHA256 = (
    '98860ed4e60c427a76ac05879d800f36b65c10a310f4b2b981819fa48af767b3'
)
V3_3_1_ANALYZER_SHA256 = (
    '0b10ba857bf27e1e48a90122dc1015933d0327ed84fc213cf0dafa95790bc6e7'
)
V3_3_1_TEST_SHA256 = (
    '27d68183bbfe77cb64d37eca2ff568e07a4fa893283c500118e46c4477054d8e'
)
V3_3_1_AMENDMENT_SHA256 = (
    '37e23b251f53ab87bae99b63024a381c367ce33bbc950a2227b3267fbc9668d1'
)
EXPECTED_RECORD_COUNT = 80
EXPECTED_APPLY_COUNT = 320
ANCHOR_IDS = (0, 127, 128, 255)
RECIPIENT_ORDERS = tuple(range(20))
INVARIANT_ROWS = (0, 1, 3, 5, 6, 7)
ACTIVE_ROWS = (2, 4)
EIGHT_ROLES = (
    'reference_baseline', 'alternate_baseline',
    'reference_into_alternate', 'alternate_into_alternate_self_control',
    'alternate_into_reference', 'reference_into_reference_self_control',
    'unrelated_reference_donor', 'unrelated_alternate_donor',
)
IDENTITY_ROWS = (0, 1, 1, 1, 0, 0, 6, 7)
INTENDED_DONOR_ROWS = (0, 1, 0, 1, 1, 0, 6, 7)
UNRELATED_DONOR_ROWS = (0, 1, 6, 1, 7, 0, 6, 7)
CONFIRMATION_DISCLOSURE = (
    'Later-exon metadata/labels were exposed after protocol freeze; '
    'no later-exon model outputs, activations, or interventions are used.'
)
MAX_WALL_TIME_SECONDS = 7_200
MAX_OUTPUT_BYTES = 1_073_741_824

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
_UPSTREAM_ROOT = _REPO_ROOT.parent / 'alphagenome'
_AMENDMENT_PATH = (
    _HERE / 'v3_wider_mechanism/encoder_skip_ood_sidecar_amendment_v3_3_2.md'
)
_FREEZE_PATH = _HERE / 'encoder_skip_ood_sidecar_v3_3_2_freeze.json'
_ORIGINAL_FREEZE_PATH = _HERE / 'encoder_skip_factorial_v3_3_freeze.json'
_ORIGINAL_RUN_DIR = (
    _HERE / 'results/v3_3_development_encoder_skip_factorial_one_shot'
)
_RUN_DIR = _HERE / 'results/v3_3_2_development_ood_sidecar_one_shot'
_ANALYSIS_DIR = _HERE / 'results/v3_3_2_development_ood_sidecar_analysis'
_PREFLIGHT_DIR = _HERE / 'results/v3_3_2_device_preflight'
_V3_3_1_ANALYZER_PATH = _HERE / 'analyze_encoder_skip_localization_v3_3_1.py'
_V3_3_1_TEST_PATH = _HERE / 'analyze_encoder_skip_localization_v3_3_1_test.py'
_V3_3_1_AMENDMENT_PATH = (
    _HERE / 'v3_wider_mechanism/encoder_skip_analysis_amendment_v3_3_1.md'
)
_V3_3_1_ATTEMPT_DIR = (
    _HERE / 'results/v3_3_development_encoder_skip_factorial_analysis_v3_3_1_attempt'
)
_V3_3_1_ANALYSIS_DIR = (
    _HERE / 'results/v3_3_development_encoder_skip_factorial_analysis'
)
_TEST_PATH = _HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_2_test.py'
_ORIGINAL_PROTOCOL_PATH = (
    _HERE / 'v3_wider_mechanism/encoder_skip_localization_protocol_v3_3.md'
)

_ORIGINAL_BINDING = {
    'git_commit': '9e8324434a3c0b2d5c53d209ad04c2d88b99f4bc',
    'protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
    'freeze_sha256': ORIGINAL_FREEZE_SHA256,
    'runner_sha256': '56eef2cc5b87f3ff9ad5837d19b891b98bbb4a7e126e20713ea9bc8b21c409c5',
    'analyzer_sha256': '0a65a27a5c424bb9dddacb5475e02d28a0999fc7cd593d4dd63ba4be06c39a46',
    'analyzer_test_sha256': 'd027f73fb07682e8cb54d46653a5bd9aa900aaeac433b1748dbdf4886c6d5034',
    'model_sha256': '7aee357d776f1f10f9ef04b1602103496ad543d89f49d5e59af459afca217ea1',
    'interpretability_sha256': 'd00a4dd8a4e62c2d8a7d583a74cbf5632121f98892e901c7f8927539ee156500',
    'attempt_started_sha256': 'b74081fd0cbd1c8d6ec5445b3b71661f40ac4d47dd77fd2d9bd3675b4cf9c3c3',
    'run_complete_sha256': 'ddc8350361ae9091ac47878a2c2d043897c46ef1d7722a401869d8d69e4be463',
    'raw_manifest_sha256': '6c50c86153fbce5136ed99205ca4726f87a00ef56216f1205dba5c25d3d27cd7',
    'raw_artifact_count': 5_142,
    'raw_artifact_tree_sha256': 'e7376062ce31090b349e88b91bd41700caf4e690511c15993e50f2bd0d47f770',
    'whole_run_file_count': 5_158,
    'whole_run_tree_sha256': '2d8125fe6d13773ba9621e527870361b6a195c516c5b4f044c7dad64c9310aaa',
    'compiler_file_count': 8,
    'compiler_tree_sha256': '9a03dcbc9d439cb9bf197941af3bbdb3e6bda067cf661b90de6d7eab1f4d87eb',
    'import_provenance_sha256': '64a5538499e5b06e29cb506a2b08585bb002b3766bd1be210d1a568b9ec5110e',
    'protobuf_provenance_sha256': '2498a940f6ee15e54e72e8f51587d4c42ffc1b49851873c31ad09085315d0ba8',
    'target_eligibility_sha256': 'b216692d8028faab09b5f6590e3e68d9c8805d3c715ddedd99e8956019cedcf0',
    'device_preflight_sha256': 'b983c7f4910ef4fc5f68bc72486552063f4497f90bba64497eb29a09d3d1809d',
    'preflight_stdout_sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    'preflight_stderr_sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
}


class AnalysisError(RuntimeError):
  """Raised when any structural or raw-evidence gate fails closed."""


_FREEZE_KEYS = {
    'analysis_dir', 'amendment_commit', 'amendment_path',
    'amendment_sha256', 'attempt_id', 'attention_backend',
    'checkpoint_manifest_path', 'checkpoint_manifest_sha256',
    'checkpoint_snapshot', 'context_bp', 'development_exons_path',
    'development_exons_sha256', 'development_variants_path',
    'development_variants_sha256', 'eight_row_compile_count',
    'eight_row_intended_donor_rows', 'eight_row_natural_identity_rows',
    'eight_row_roles', 'eight_row_unrelated_donor_rows',
    'environment_contract', 'expected_compute_capability',
    'expected_device_kind', 'expected_gpu_uuid', 'file_sha256',
    'identity_rerun_count', 'invariant_rows_between_calls',
    'main_cube_rerun_count', 'max_output_bytes', 'max_wall_time_seconds',
    'mixed_precision_policy', 'model_apply_count', 'old_ood_records_reused',
    'ood_anchor_ids', 'ood_record_count', 'original_freeze_path',
    'original_freeze_sha256', 'original_protocol_path',
    'original_protocol_sha256', 'original_run', 'output_dir',
    'preflight_dir', 'preflight_script_version', 'protobuf_binding',
    'recipient_orders', 'reference_bindings_path',
    'reference_bindings_sha256', 'reference_object', 'reference_url',
    'runtime_version_manifest', 'script_version', 'six_row_compile_count',
    'upstream_alphagenome_git_head',
    'upstream_generated_binding_exception', 'upstream_imported_modules',
    'v3_3_1_status',
}


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
    raise AnalysisError(f'{label} imported forbidden model/JAX modules: {forbidden}.')


def _load_cpu_module(path: Path, digest: str, name: str):
  _assert_cpu_only(f'{name} pre-import')
  if path.is_symlink() or not path.is_file() or _sha256(path) != digest:
    raise AnalysisError(f'{name} bytes changed before import.')
  specification = importlib.util.spec_from_file_location(name, path)
  if specification is None or specification.loader is None:
    raise AnalysisError(f'Cannot load {name}.')
  module = importlib.util.module_from_spec(specification)
  specification.loader.exec_module(module)
  _assert_cpu_only(f'{name} post-import')
  return module


_v331 = _load_cpu_module(
    _V3_3_1_ANALYZER_PATH, V3_3_1_ANALYZER_SHA256,
    '_opensplice_frozen_analyzer_v3_3_1',
)
_v33 = _v331._v33  # pylint: disable=protected-access


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
  if path.is_symlink() or not stat.S_ISREG(mode):
    raise AnalysisError(f'{label} is symlinked or not a regular file.')


def _read_json(path: Path, label: str) -> dict[str, Any]:
  _strict_regular(path, label)
  try:
    value = json.loads(path.read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError) as error:
    raise AnalysisError(f'{label} is not readable JSON.') from error
  if not isinstance(value, dict):
    raise AnalysisError(f'{label} must be a JSON object.')
  return value


def _finite(value: Any, label: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise AnalysisError(f'{label} is not numeric.')
  result = float(value)
  if not math.isfinite(result):
    raise AnalysisError(f'{label} is non-finite.')
  return result


def _require_exact_keys(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
  if not isinstance(value, Mapping) or set(value) != keys:
    observed = set(value) if isinstance(value, Mapping) else set()
    raise AnalysisError(
        f'{label} schema changed (missing={sorted(keys-observed)}, '
        f'extra={sorted(observed-keys)}).'
    )
  return value


def _tree_digest(paths: Iterable[Path], root: Path) -> str:
  digest = hashlib.sha256()
  root = root.resolve()
  for path in sorted(path.resolve() for path in paths):
    relative = path.relative_to(root)
    digest.update(str(relative).encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(_sha256(path)))
  return digest.hexdigest()


def _validate_exact_flat_tree(
    directory: Path, mapping: Mapping[str, Mapping[str, Any]],
    expected_tree: str, label: str,
) -> None:
  if directory.is_symlink() or not directory.is_dir():
    raise AnalysisError(f'{label} directory is absent or unsafe.')
  entries = sorted(directory.iterdir())
  if any(
      path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode)
      for path in entries
  ):
    raise AnalysisError(f'{label} contains a directory/symlink/special entry.')
  if {path.name for path in entries} != set(mapping):
    raise AnalysisError(f'{label} file membership changed.')
  for path in entries:
    row = _require_exact_keys(
        mapping[path.name], {'sha256', 'size_bytes'},
        f'{label}.{path.name}.binding',
    )
    if (
        not _is_sha256(row.get('sha256'))
        or _sha256(path) != row['sha256']
        or path.stat().st_size != row.get('size_bytes')
    ):
      raise AnalysisError(f'{label}.{path.name} changed.')
  if not _is_sha256(expected_tree) or _tree_digest(entries, directory) != expected_tree:
    raise AnalysisError(f'{label} tree changed.')


def _slug(value: str) -> str:
  return ''.join(character if character.isalnum() else '_' for character in value).strip('_')


def _execution_order() -> tuple[tuple[int, int], ...]:
  result = tuple(
      (order, anchor) for order in RECIPIENT_ORDERS for anchor in ANCHOR_IDS
  )
  if len(result) != 80 or len(set(result)) != 80:
    raise AssertionError('Internal sidecar execution order changed.')
  return result


def _artifact_relative(case: Mapping[str, Any], anchor: int) -> str:
  return (
      f"raw/ood_anchors/{case['order']:03d}_{_slug(str(case['variant_id']))}/"
      f'{anchor:03d}.json'
  )


def _donor_order(order: int) -> int:
  if 0 <= order < 10:
    return order + 10
  if 10 <= order < 20:
    return order - 10
  raise AnalysisError(f'Invalid development recipient order: {order}.')


def _original_relative(case: Mapping[str, Any], family: str, anchor: int | None) -> str:
  key = f"{case['order']:03d}_{_slug(str(case['variant_id']))}"
  if family == 'identity' and anchor is None:
    return f'raw/identity/{key}.json'
  if family == 'coalition' and anchor is not None:
    return f'raw/coalitions/{key}/{anchor:03d}.json'
  raise AnalysisError('Invalid original-artifact family/anchor request.')


def _validate_binding(
    value: Any, *, expected_relative: str, expected_manifest: Mapping[str, str],
    root: Path, label: str,
) -> None:
  row = _require_exact_keys(value, {'path', 'sha256'}, label)
  expected_sha = expected_manifest.get(expected_relative)
  if row.get('path') != expected_relative or row.get('sha256') != expected_sha:
    raise AnalysisError(f'{label} differs from the immutable original manifest.')
  path = (root / expected_relative).resolve()
  try:
    path.relative_to(root.resolve())
  except ValueError as error:
    raise AnalysisError(f'{label} escaped its immutable root.') from error
  _strict_regular(path, label)
  if _sha256(path) != expected_sha:
    raise AnalysisError(f'{label} current bytes changed.')


def _expected_coalition(anchor: int) -> dict[str, Any]:
  t, e_mask = divmod(anchor, 128)
  e_players = ('E64', 'E32', 'E16', 'E8', 'E4', 'E2', 'E1')
  bits = [bool(e_mask & (1 << index)) for index in range(7)]
  return {
      'coalition_id': anchor,
      't': t,
      'e_mask': e_mask,
      'e_bits': bits,
      'e_bits_binary': format(e_mask, '07b'),
      'enabled_players': (['T'] if t else []) + [
          player for player, enabled in zip(e_players, bits, strict=True) if enabled
      ],
      'coalition_bit_order': list(e_players) + ['T'],
      'shapley_player_order': ['T', *e_players],
  }


def _readouts(record: Mapping[str, Any], label: str) -> dict[str, dict[str, Any]]:
  fields = {
      'intended': 'intended_target_readout',
      'intended_repeat': 'intended_repeat_target_readout',
      'unrelated': 'unrelated_target_readout',
      'unrelated_repeat': 'unrelated_repeat_target_readout',
  }
  return {
      call: _v33._readout(record, field, label, rows=8)  # pylint: disable=protected-access
      for call, field in fields.items()
  }


def _same_readout_row(
    left: Mapping[str, Any], left_row: int,
    right: Mapping[str, Any], right_row: int,
) -> bool:
  return _v33._row_bytes(left, left_row) == _v33._row_bytes(  # pylint: disable=protected-access
      right, right_row
  )


def _validate_readout_relations(
    readouts: Mapping[str, Mapping[str, Any]], anchor: int, label: str,
) -> None:
  intended = readouts['intended']
  unrelated = readouts['unrelated']
  for call in ('intended', 'unrelated'):
    first, repeated = readouts[call], readouts[f'{call}_repeat']
    if any(not _same_readout_row(first, row, repeated, row) for row in range(8)):
      raise AnalysisError(f'{label}.{call} endpoint repeat differs.')
    if not _same_readout_row(first, 3, first, 1):
      raise AnalysisError(f'{label}.{call} alternate self row differs.')
    if not _same_readout_row(first, 5, first, 0):
      raise AnalysisError(f'{label}.{call} reference self row differs.')
  if any(
      not _same_readout_row(intended, row, unrelated, row)
      for row in INVARIANT_ROWS
  ):
    raise AnalysisError(f'{label} endpoint invariant-row equality failed.')
  if anchor == 0:
    for call in ('intended', 'unrelated'):
      value = readouts[call]
      if (
          not _same_readout_row(value, 2, value, 1)
          or not _same_readout_row(value, 4, value, 0)
      ):
        raise AnalysisError(f'{label}.{call} ID0 recipient no-op failed.')
    if any(
        not _same_readout_row(intended, row, unrelated, row)
        for row in range(8)
    ):
      raise AnalysisError(f'{label} ID0 all-row endpoint equality failed.')
  if anchor == 255 and (
      not _same_readout_row(intended, 2, intended, 0)
      or not _same_readout_row(intended, 4, intended, 1)
      or not _same_readout_row(unrelated, 2, unrelated, 6)
      or not _same_readout_row(unrelated, 4, unrelated, 7)
  ):
    raise AnalysisError(f'{label} ID255 endpoint closure failed.')


def _validate_raw_movement(
    emitted: Any, readouts: Mapping[str, Mapping[str, Any]], label: str,
) -> None:
  value = _require_exact_keys(emitted, {'intended', 'unrelated'}, f'{label}.raw_movement')
  for call in ('intended', 'unrelated'):
    row = _require_exact_keys(
        value[call],
        {'reference_into_alternate', 'alternate_into_reference'},
        f'{label}.raw_movement.{call}',
    )
    expected = _v33._raw_movements(readouts[call])['movements']  # pylint: disable=protected-access
    for direction, expected_number in expected.items():
      if _finite(row.get(direction), f'{label}.{call}.{direction}') != expected_number:
        raise AnalysisError(f'{label}.{call}.{direction} raw movement changed.')


def _validate_trace_fingerprints(
    record: Mapping[str, Any], label: str, *, require_repeat: bool = True,
) -> None:
  pairs = (
      ('intended_trace_fingerprint', 'intended_repeat_trace_fingerprint'),
      ('unrelated_trace_fingerprint', 'unrelated_repeat_trace_fingerprint'),
  )
  for first, repeated in pairs:
    _v33._validate_trace_fingerprint(record.get(first), f'{label}.{first}')  # pylint: disable=protected-access
    _v33._validate_trace_fingerprint(record.get(repeated), f'{label}.{repeated}')  # pylint: disable=protected-access
    if require_repeat and record.get(first) != record.get(repeated):
      raise AnalysisError(f'{label} compact trace repeat changed.')


def _fingerprint_rows(value: Any, label: str) -> list[dict[str, Any]]:
  node = _require_exact_keys(
      value, {'full_shape', 'dtype', 'row_count', 'rows', 'collision_semantics'},
      label,
  )
  if (
      node.get('row_count') != 8
      or not isinstance(node.get('full_shape'), list)
      or not node['full_shape'] or node['full_shape'][0] != 8
      or not isinstance(node.get('dtype'), str) or not node['dtype']
      or node.get('collision_semantics')
      != 'SHA-256 per exact row byte string; direct live equality is the gate.'
      or not isinstance(node.get('rows'), list) or len(node['rows']) != 8
  ):
    raise AnalysisError(f'{label} compact rowwise header changed.')
  rows = []
  expected_row_shape: list[int] | None = None
  expected_dtype: str | None = None
  dtype_bytes = {'float16': 2, 'bfloat16': 2, 'float32': 4, 'float64': 8}
  for index, raw in enumerate(node['rows']):
    row = _require_exact_keys(
        raw, {'row', 'shape', 'dtype', 'size_bytes', 'sha256'},
        f'{label}.rows[{index}]',
    )
    if (
        row.get('row') != index
        or not isinstance(row.get('shape'), list)
        or any(isinstance(size, bool) or not isinstance(size, int) or size < 0
               for size in row['shape'])
        or row.get('dtype') != node['dtype']
        or isinstance(row.get('size_bytes'), bool)
        or not isinstance(row.get('size_bytes'), int)
        or row['size_bytes'] < 0
        or not _is_sha256(row.get('sha256'))
    ):
      raise AnalysisError(f'{label}.rows[{index}] is malformed.')
    if row['dtype'] not in dtype_bytes:
      raise AnalysisError(f'{label}.rows[{index}] dtype is not frozen numeric data.')
    if row['size_bytes'] != math.prod(row['shape']) * dtype_bytes[row['dtype']]:
      raise AnalysisError(f'{label}.rows[{index}] shape/size arithmetic changed.')
    if expected_row_shape is None:
      expected_row_shape, expected_dtype = list(row['shape']), row['dtype']
    elif row['shape'] != expected_row_shape or row['dtype'] != expected_dtype:
      raise AnalysisError(f'{label} rows have inconsistent shapes/dtypes.')
    rows.append(dict(row))
  if node['full_shape'] != [8, *(expected_row_shape or [])]:
    raise AnalysisError(f'{label} full shape differs from its eight row shapes.')
  return rows


def _nested_shape(value: Any) -> tuple[int, ...]:
  if not isinstance(value, list):
    return ()
  if not value:
    return (0,)
  child = {_nested_shape(item) for item in value}
  if len(child) != 1:
    raise AnalysisError('Compact upstream fingerprint values are ragged.')
  return (len(value),) + child.pop()


def _validate_upstream_compact(
    value: Any, label: str, expected_shape: tuple[int, ...],
) -> dict[str, Any]:
  node = _require_exact_keys(value, {'shape', 'dtype', 'values'}, label)
  if (
      not isinstance(node.get('shape'), list)
      or tuple(node['shape']) != _nested_shape(node.get('values'))
      or tuple(node['shape']) != expected_shape
      or node.get('dtype') not in {'float16', 'bfloat16', 'float32', 'float64'}
  ):
    raise AnalysisError(f'{label} compact upstream schema changed.')
  for leaf in _v33._array_leaves(node['values']):  # pylint: disable=protected-access
    _finite(leaf, f'{label}.value')
  return dict(node)


def _validate_rowwise_fingerprints(
    value: Any, anchor: int, label: str, *, require_live_gates: bool,
) -> None:
  calls = _require_exact_keys(
      value, {'intended', 'intended_repeat', 'unrelated', 'unrelated_repeat'},
      f'{label}.rowwise_trace_fingerprints',
  )
  parsed: dict[str, dict[str, Any]] = {}
  call_keys = {
      'natural_final_embeddings', 'effective_final_embeddings',
      'transformer_output_natural_fingerprint',
      'encoder_skips_natural_fingerprints',
  }
  for call, raw in calls.items():
    node = _require_exact_keys(raw, call_keys, f'{label}.rowwise.{call}')
    parsed[call] = {
        'natural': _fingerprint_rows(
            node['natural_final_embeddings'], f'{label}.{call}.natural_final'
        ),
        'effective': _fingerprint_rows(
            node['effective_final_embeddings'], f'{label}.{call}.effective_final'
        ),
        'T': _validate_upstream_compact(
            node['transformer_output_natural_fingerprint'],
            f'{label}.{call}.T', (8, 4),
        ),
        'E': _validate_upstream_compact(
            node['encoder_skips_natural_fingerprints'],
            f'{label}.{call}.E', (7, 8, 4),
        ),
    }
  if not require_live_gates:
    return
  for call in ('intended', 'unrelated'):
    if parsed[call] != parsed[f'{call}_repeat']:
      raise AnalysisError(f'{label}.{call} rowwise repeat differs.')
    if parsed[call]['natural'] != parsed[call]['effective']:
      raise AnalysisError(f'{label}.{call} disabled final seam differs.')
  for field in ('T', 'E'):
    if parsed['intended'][field] != parsed['unrelated'][field]:
      raise AnalysisError(f'{label} upstream natural {field} differs between calls.')
  rows = range(8) if anchor == 0 else INVARIANT_ROWS
  if any(
      parsed['intended']['natural'][row]
      != parsed['unrelated']['natural'][row]
      for row in rows
  ):
    raise AnalysisError(f'{label} natural-final invariant rows differ.')


_CHECK_KEYS = {
    'passed', 'corrected_host_assertion_version',
    'upstream_transformer_natural_tensors_all8_exact_between_calls',
    'upstream_T_E_natural_fingerprints_all8_exact_between_calls',
    'natural_final_invariant_rows_exact_between_calls',
    'natural_final_invariant_rows', 'active_rows_cross_call_equality_not_required',
    'active_rows_forced_difference_not_required',
    'full_within_call_natural_effective_final_exact',
    'endpoint_invariant_rows_exact_between_calls',
    'self_rows_exact_within_each_call',
    'id0_all8_natural_final_exact_between_calls',
    'id0_within_call_natural_final_recipient_noop_exact',
    'id0_all8_endpoint_exact_between_calls', 'id0_recipient_noop_exact',
    'id255_intended_endpoint_closure_exact',
    'id255_unrelated_endpoint_closure_exact',
    'intended_route_tensor_donor_exact', 'unrelated_route_tensor_donor_exact',
    'enabled_disabled_T_E_exact', 'runtime_route_masks_and_maps_exact',
    'intended_target_repeat_exact', 'intended_trace_repeat_exact',
    'unrelated_target_repeat_exact', 'unrelated_trace_repeat_exact',
    'transformer_internal_seams_disabled_exact',
    'final_embedding_disabled_exact', 'normalization_computed',
}


def _validate_checks(value: Any, anchor: int, label: str) -> None:
  checks = _require_exact_keys(value, _CHECK_KEYS, f'{label}.checks')
  expected_nonbool = {
      'corrected_host_assertion_version': 'v3.3.2',
      'natural_final_invariant_rows': list(INVARIANT_ROWS),
      'active_rows_cross_call_equality_not_required': list(ACTIVE_ROWS),
  }
  for key, expected in expected_nonbool.items():
    if checks.get(key) != expected:
      raise AnalysisError(f'{label}.checks.{key} changed.')
  false_fields = {'normalization_computed'}
  conditional = {
      'id0_all8_natural_final_exact_between_calls': anchor == 0,
      'id0_within_call_natural_final_recipient_noop_exact': anchor == 0,
      'id0_all8_endpoint_exact_between_calls': anchor == 0,
      'id0_recipient_noop_exact': anchor == 0,
      'id255_intended_endpoint_closure_exact': anchor == 255,
      'id255_unrelated_endpoint_closure_exact': anchor == 255,
  }
  for key in _CHECK_KEYS - set(expected_nonbool) - set(conditional) - false_fields:
    if checks.get(key) is not True:
      raise AnalysisError(f'{label}.checks.{key} is not true.')
  for key, expected in conditional.items():
    if checks.get(key) is not expected:
      raise AnalysisError(f'{label}.checks.{key} applicability changed.')
  if checks.get('normalization_computed') is not False:
    raise AnalysisError(f'{label} computed forbidden donor normalization.')


_RECORD_KEYS = {
    'status', 'family', 'script_version', 'amendment_sha256',
    'amendment_commit', 'original_protocol_sha256', 'freeze_sha256',
    'execution_index', 'sidecar_execution_index', 'execution_order',
    'eight_row_executable_fingerprint', 'same_eight_row_compiled_executable',
    'six_row_executable_used', 'recipient_case', 'donor_case', 'coalition',
    'batch_roles', 'natural_identity_rows', 'intended_donor_rows',
    'unrelated_donor_rows', 'invariant_rows_between_calls',
    'active_recipient_rows', 'active_recipient_cross_call_equality_gate',
    'active_recipient_cross_call_inequality_gate',
    'original_artifact_bindings', 'original_ood_records_used_as_data',
    'recipient_sequence_sha256', 'donor_sequence_sha256',
    'runtime_interventions', 'intended_target_readout',
    'intended_repeat_target_readout', 'unrelated_target_readout',
    'unrelated_repeat_target_readout', 'intended_trace_fingerprint',
    'intended_repeat_trace_fingerprint', 'unrelated_trace_fingerprint',
    'unrelated_repeat_trace_fingerprint', 'rowwise_trace_fingerprints',
    'raw_movement', 'checks', 'failure', 'seconds', 'created_at_unix_s',
    'model_apply_count_through_record',
}


def _validate_original_artifact_bindings(
    value: Any, *, case: Mapping[str, Any], donor_case: Mapping[str, Any],
    anchor: int, original_manifest: Mapping[str, str], label: str,
) -> None:
  bindings = _require_exact_keys(
      value,
      {'recipient_identity', 'donor_identity', 'recipient_six_row_coalition'},
      f'{label}.original_artifact_bindings',
  )
  for name, linked_case, family, linked_anchor in (
      ('recipient_identity', case, 'identity', None),
      ('donor_identity', donor_case, 'identity', None),
      ('recipient_six_row_coalition', case, 'coalition', anchor),
  ):
    _validate_binding(
        bindings[name],
        expected_relative=_original_relative(
            linked_case, family, linked_anchor
        ),
        expected_manifest=original_manifest, root=_ORIGINAL_RUN_DIR,
        label=f'{label}.{name}',
    )


def _validate_invalid_payload(
    record: Mapping[str, Any], *, case: Mapping[str, Any],
    donor_case: Mapping[str, Any], anchor: int,
    original_manifest: Mapping[str, str], label: str,
) -> None:
  """Audits the exact persisted prefix of the runner's post-call try block."""
  readout_fields = (
      'intended_target_readout', 'intended_repeat_target_readout',
      'unrelated_target_readout', 'unrelated_repeat_target_readout',
  )
  present = [record.get(field) is not None for field in readout_fields]
  if present != sorted(present, reverse=True):
    raise AnalysisError(f'{label} invalid readouts are not an exact prefix.')
  parsed: dict[str, dict[str, Any]] = {}
  for field, is_present in zip(readout_fields, present, strict=True):
    if is_present:
      parsed[field] = _v33._readout(  # pylint: disable=protected-access
          record, field, label, rows=8
      )

  rowwise_present = record.get('rowwise_trace_fingerprints') is not None
  trace_fields = (
      'intended_trace_fingerprint', 'intended_repeat_trace_fingerprint',
      'unrelated_trace_fingerprint', 'unrelated_repeat_trace_fingerprint',
  )
  trace_present = [record.get(field) is not None for field in trace_fields]
  if any(trace_present) and not all(trace_present):
    raise AnalysisError(f'{label} invalid trace-fingerprint stage is partial.')
  bindings_present = record.get('original_artifact_bindings') is not None
  movement_present = record.get('raw_movement') is not None
  stages = [all(present), rowwise_present, all(trace_present), bindings_present,
            movement_present]
  if stages != sorted(stages, reverse=True):
    raise AnalysisError(f'{label} invalid evidence stages are not dependency-ordered.')
  if rowwise_present:
    _validate_rowwise_fingerprints(
        record['rowwise_trace_fingerprints'], anchor, label,
        require_live_gates=False,
    )
  if all(trace_present):
    _validate_trace_fingerprints(record, label, require_repeat=False)
  if bindings_present:
    _validate_original_artifact_bindings(
        record['original_artifact_bindings'], case=case,
        donor_case=donor_case, anchor=anchor,
        original_manifest=original_manifest, label=label,
    )
  if movement_present:
    readouts = {
        'intended': parsed['intended_target_readout'],
        'intended_repeat': parsed['intended_repeat_target_readout'],
        'unrelated': parsed['unrelated_target_readout'],
        'unrelated_repeat': parsed['unrelated_repeat_target_readout'],
    }
    _validate_raw_movement(record['raw_movement'], readouts, label)


def _validate_record(
    record: Mapping[str, Any], *, case: Mapping[str, Any],
    donor_case: Mapping[str, Any], anchor: int, execution_index: int,
    freeze_sha256: str, executable_fingerprint: str,
    original_manifest: Mapping[str, str], sequence_bindings: Mapping[int, Any],
    allow_invalid: bool,
) -> dict[str, Any]:
  label = f'order={case["order"]},anchor={anchor}'
  _require_exact_keys(record, _RECORD_KEYS, label)
  expected_common = {
      'family': 'v3_3_2_unrelated_donor_sidecar_anchor',
      'script_version': SCRIPT_VERSION,
      'amendment_sha256': AMENDMENT_SHA256,
      'amendment_commit': AMENDMENT_COMMIT,
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'freeze_sha256': freeze_sha256,
      'execution_index': execution_index,
      'sidecar_execution_index': execution_index,
      'execution_order': 'recipient-major, anchor-minor',
      'eight_row_executable_fingerprint': executable_fingerprint,
      'same_eight_row_compiled_executable': True,
      'six_row_executable_used': False,
      'recipient_case': dict(case),
      'donor_case': dict(donor_case),
      'coalition': _expected_coalition(anchor),
      'batch_roles': list(EIGHT_ROLES),
      'natural_identity_rows': list(IDENTITY_ROWS),
      'intended_donor_rows': list(INTENDED_DONOR_ROWS),
      'unrelated_donor_rows': list(UNRELATED_DONOR_ROWS),
      'invariant_rows_between_calls': list(INVARIANT_ROWS),
      'active_recipient_rows': list(ACTIVE_ROWS),
      'active_recipient_cross_call_equality_gate': False,
      'active_recipient_cross_call_inequality_gate': False,
      'original_ood_records_used_as_data': False,
      'recipient_sequence_sha256': sequence_bindings[case['order']],
      'donor_sequence_sha256': sequence_bindings[donor_case['order']],
      'model_apply_count_through_record': 4 * (execution_index + 1),
  }
  for key, expected in expected_common.items():
    if record.get(key) != expected:
      raise AnalysisError(f'{label}.{key} changed.')
  if not _is_sha256(executable_fingerprint):
    raise AnalysisError(f'{label} executable fingerprint is malformed.')
  _finite(record.get('created_at_unix_s'), f'{label}.created_at_unix_s')
  seconds = _require_exact_keys(
      record.get('seconds'),
      {'intended', 'intended_repeat', 'unrelated', 'unrelated_repeat'},
      f'{label}.seconds',
  )
  for call, value in seconds.items():
    if _finite(value, f'{label}.seconds.{call}') < 0:
      raise AnalysisError(f'{label}.seconds.{call} is negative.')

  runtime = _require_exact_keys(
      record.get('runtime_interventions'), {'intended', 'unrelated'},
      f'{label}.runtime_interventions',
  )
  _v33._runtime_route(  # pylint: disable=protected-access
      runtime['intended'], rows=8, coalition_id=anchor,
      donor_rows=INTENDED_DONOR_ROWS, label=f'{label}.runtime.intended',
  )
  _v33._runtime_route(  # pylint: disable=protected-access
      runtime['unrelated'], rows=8, coalition_id=anchor,
      donor_rows=UNRELATED_DONOR_ROWS, label=f'{label}.runtime.unrelated',
  )
  complete = record.get('status') == 'complete'
  invalid = record.get('status') == 'invalid'
  if not complete and not (allow_invalid and invalid):
    raise AnalysisError(f'{label} has an invalid status for this prefix.')
  if complete:
    if record.get('failure') is not None:
      raise AnalysisError(f'{label} complete record contains a failure.')
    _validate_original_artifact_bindings(
        record.get('original_artifact_bindings'), case=case,
        donor_case=donor_case, anchor=anchor,
        original_manifest=original_manifest, label=label,
    )
    readouts = _readouts(record, label)
    _validate_trace_fingerprints(record, label)
    _validate_rowwise_fingerprints(
        record.get('rowwise_trace_fingerprints'), anchor, label,
        require_live_gates=True,
    )
    _validate_raw_movement(record.get('raw_movement'), readouts, label)
    _validate_readout_relations(readouts, anchor, label)
    _validate_checks(record.get('checks'), anchor, label)
  else:
    if record.get('checks') is not None:
      raise AnalysisError(f'{label} invalid record contains checks.')
    failure = _require_exact_keys(
        record.get('failure'), {'type', 'message'}, f'{label}.failure'
    )
    if (
        not isinstance(failure.get('type'), str)
        or not failure['type'] or not failure['type'].isidentifier()
        or not isinstance(failure.get('message'), str)
        or not failure['message']
    ):
      raise AnalysisError(f'{label} invalid failure is malformed.')
    _validate_invalid_payload(
        record, case=case, donor_case=donor_case, anchor=anchor,
        original_manifest=original_manifest, label=label,
    )
  return {'status': record['status'], 'anchor': anchor, 'order': case['order']}


def _validate_v3_3_1_status(expected: Any) -> dict[str, Any]:
  value = _require_exact_keys(
      expected,
      {
          'amendment_path', 'amendment_sha256', 'amendment_commit',
          'attempt_dir', 'analysis_dir', 'state', 'attempt_file_count',
          'attempt_tree_sha256', 'attempt_files', 'analysis_file_count',
          'analysis_tree_sha256', 'analysis_files', 'structural_predicates',
      },
      'v3.3.1 completed status',
  )
  fixed = {
      'amendment_path': str(_V3_3_1_AMENDMENT_PATH.resolve()),
      'amendment_sha256': V3_3_1_AMENDMENT_SHA256,
      'amendment_commit': '186c25f',
      'attempt_dir': str(_V3_3_1_ATTEMPT_DIR.resolve()),
      'analysis_dir': str(_V3_3_1_ANALYSIS_DIR.resolve()),
      'state': 'completed',
      'attempt_file_count': 2,
      'attempt_tree_sha256': 'b0e788f0df3db1678ca410da7b0c409a18ceeaa6ddcbb97c61a155188a6e719f',
      'attempt_files': {
          'ANALYSIS_ATTEMPT_STARTED.json': {
              'sha256': '1c4738026210ddd7f4d62b21f04eb1305cc86041daadb61cb3cfe0e549af8922',
              'size_bytes': 16_116,
          },
          'ANALYSIS_COMPLETE.json': {
              'sha256': 'ee7d9fa0d0d06abbc52beda8801f411b8725d59e7d5683c256bd51010d732e99',
              'size_bytes': 574,
          },
      },
      'analysis_file_count': 2,
      'analysis_tree_sha256': 'f3e6eee31c3fc978356a5766c190061ae3f8fd709da6c5c0836f7ce3d47de8f0',
      'analysis_files': {
          'ANALYSIS.json': {
              'sha256': 'ed18cce580c578d3cd750756d882a7b120a87736849077abffaee5781c09dd6b',
              'size_bytes': 653_362,
          },
          'RESULT.md': {
              'sha256': '6a4884040677e194c9b43c115af8c045cd67d2ffa857088710ab852399d2a440',
              'size_bytes': 862,
          },
      },
      'structural_predicates': {
          'status': 'complete_controlled_stop_audited',
          'decision': 'controlled_stop_ood_tooling_failure',
          'shapley_computed': False,
          'nomination_performed': False,
          'analysis_version': (
              'opensplice-encoder-skip-localization-analysis-v3.3.1'
          ),
      },
  }
  if dict(value) != fixed:
    raise AnalysisError('v3.3.1 completed status differs from the frozen table.')
  for path, digest, label in (
      (_V3_3_1_AMENDMENT_PATH, V3_3_1_AMENDMENT_SHA256, 'v3.3.1 amendment'),
      (_V3_3_1_ANALYZER_PATH, V3_3_1_ANALYZER_SHA256, 'v3.3.1 analyzer'),
      (_V3_3_1_TEST_PATH, V3_3_1_TEST_SHA256, 'v3.3.1 analyzer test'),
  ):
    _strict_regular(path, label)
    if _sha256(path) != digest:
      raise AnalysisError(f'{label} changed.')
  for directory, mapping, count, tree, label in (
      (
          _V3_3_1_ATTEMPT_DIR, fixed['attempt_files'], 2,
          fixed['attempt_tree_sha256'], 'v3.3.1 attempt',
      ),
      (
          _V3_3_1_ANALYSIS_DIR, fixed['analysis_files'], 2,
          fixed['analysis_tree_sha256'], 'v3.3.1 analysis',
      ),
  ):
    if count != len(mapping):
      raise AnalysisError(f'{label} frozen count/mapping changed.')
    _validate_exact_flat_tree(directory, mapping, tree, label)
  attempt_complete = _read_json(
      _V3_3_1_ATTEMPT_DIR / 'ANALYSIS_COMPLETE.json', 'v3.3.1 completion'
  )
  analysis = _read_json(
      _V3_3_1_ANALYSIS_DIR / 'ANALYSIS.json', 'v3.3.1 analysis'
  )
  if (
      attempt_complete.get('status') != 'complete_controlled_stop_audited'
      or attempt_complete.get('decision') != 'controlled_stop_ood_tooling_failure'
      or attempt_complete.get('shapley_computed') is not False
      or attempt_complete.get('nomination_performed') is not False
      or analysis.get('decision') != 'controlled_stop_ood_tooling_failure'
      or analysis.get('nomination') is not None
      or analysis.get('resolution_analysis') is not None
  ):
    raise AnalysisError('v3.3.1 did not complete its structural controlled-stop audit.')
  return fixed


def _validate_original_v3_3() -> tuple[dict[str, Any], dict[str, str], dict[int, Any]]:
  """Revalidates v3.3 without opening an original scientific raw record."""
  bundle = _v331._validate_original_bundle(_REPO_ROOT)  # pylint: disable=protected-access
  run = _v331._validate_immutable_run(_ORIGINAL_RUN_DIR)  # pylint: disable=protected-access
  if (
      bundle.get('original_commit') != _ORIGINAL_BINDING['git_commit']
      or bundle.get('freeze_sha256') != ORIGINAL_FREEZE_SHA256
      or run.get('whole_run_file_count') != 5_158
      or run.get('whole_run_tree_sha256') != _ORIGINAL_BINDING['whole_run_tree_sha256']
      or run.get('stop_reason') != 'ood_tooling_failure'
  ):
    raise AnalysisError('Immutable original-v3.3 audit changed.')
  original_start = _read_json(
      _ORIGINAL_RUN_DIR / 'ATTEMPT_STARTED.json', 'original ATTEMPT_STARTED'
  )
  normalized, generated_audit = _v331._validate_generated_bindings(  # pylint: disable=protected-access
      original_start
  )
  cases = _v33._load_cases()  # pylint: disable=protected-access
  with _v331._normalized_start_reader(  # pylint: disable=protected-access
      _ORIGINAL_RUN_DIR, normalized
  ):
    _, freeze_sha, start_audit = _v33._validate_start(  # pylint: disable=protected-access
        _ORIGINAL_RUN_DIR.resolve(), bundle_root=_REPO_ROOT, cases=cases
    )
  if freeze_sha != ORIGINAL_FREEZE_SHA256:
    raise AnalysisError('Original START normalized to a changed freeze.')
  manifest = _read_json(
      _ORIGINAL_RUN_DIR / 'RAW_MANIFEST.json', 'original RAW_MANIFEST'
  )
  mapping = manifest.get('artifact_sha256')
  if (
      not isinstance(mapping, Mapping) or len(mapping) != 5_142
      or manifest.get('artifact_tree_sha256')
      != _ORIGINAL_BINDING['raw_artifact_tree_sha256']
  ):
    raise AnalysisError('Original raw manifest changed after full rehash.')
  return ({
      **dict(_ORIGINAL_BINDING),
      'seven_role_two_output_repair_verified': generated_audit,
      'start_provenance_audit': start_audit,
      'original_bundle_binding': copy.deepcopy(
          original_start['same_process_pre_import_bootstrap']['freeze']
      ),
  }, dict(mapping), start_audit['sequence_bindings'])


def _validate_freeze_and_start(
    run_dir: Path, *, bundle_root: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, str], dict[int, Any]]:
  if run_dir.resolve() != _RUN_DIR.resolve() or bundle_root.resolve() != _REPO_ROOT:
    raise AnalysisError('v3.3.2 run/repository path changed.')
  if _sha256(_AMENDMENT_PATH) != AMENDMENT_SHA256:
    raise AnalysisError('v3.3.2 amendment bytes changed.')
  freeze = _read_json(_FREEZE_PATH, 'v3.3.2 freeze')
  _require_exact_keys(freeze, _FREEZE_KEYS, 'v3.3.2 freeze')
  freeze_sha = _sha256(_FREEZE_PATH)
  original_freeze = _read_json(_ORIGINAL_FREEZE_PATH, 'original v3.3 freeze')
  required_freeze = {
      'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION,
      'amendment_path': str(_AMENDMENT_PATH.resolve()),
      'amendment_sha256': AMENDMENT_SHA256,
      'amendment_commit': AMENDMENT_COMMIT,
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'original_protocol_path': str(_ORIGINAL_PROTOCOL_PATH.resolve()),
      'original_freeze_sha256': ORIGINAL_FREEZE_SHA256,
      'original_freeze_path': str(_ORIGINAL_FREEZE_PATH.resolve()),
      'output_dir': str(_RUN_DIR.resolve()),
      'analysis_dir': str(_ANALYSIS_DIR.resolve()),
      'preflight_dir': str(_PREFLIGHT_DIR.resolve()),
      'preflight_script_version': 'opensplice-device-preflight-v3.3.2',
      'ood_anchor_ids': list(ANCHOR_IDS),
      'recipient_orders': list(RECIPIENT_ORDERS),
      'ood_record_count': 80,
      'model_apply_count': 320,
      'eight_row_compile_count': 1,
      'six_row_compile_count': 0,
      'identity_rerun_count': 0,
      'main_cube_rerun_count': 0,
      'old_ood_records_reused': 0,
      'invariant_rows_between_calls': list(INVARIANT_ROWS),
      'eight_row_roles': list(EIGHT_ROLES),
      'eight_row_natural_identity_rows': list(IDENTITY_ROWS),
      'eight_row_intended_donor_rows': list(INTENDED_DONOR_ROWS),
      'eight_row_unrelated_donor_rows': list(UNRELATED_DONOR_ROWS),
      'max_wall_time_seconds': MAX_WALL_TIME_SECONDS,
      'max_output_bytes': MAX_OUTPUT_BYTES,
      'original_run': dict(_ORIGINAL_BINDING),
  }
  for key, expected in required_freeze.items():
    if freeze.get(key) != expected:
      raise AnalysisError(f'v3.3.2 freeze changed at {key}.')
  inherited = (
      'attention_backend', 'checkpoint_manifest_path',
      'checkpoint_manifest_sha256', 'checkpoint_snapshot', 'context_bp',
      'development_exons_path', 'development_exons_sha256',
      'development_variants_path', 'development_variants_sha256',
      'environment_contract', 'expected_compute_capability',
      'expected_device_kind', 'expected_gpu_uuid', 'mixed_precision_policy',
      'protobuf_binding', 'reference_bindings_path',
      'reference_bindings_sha256', 'reference_object', 'reference_url',
      'runtime_version_manifest', 'upstream_alphagenome_git_head',
      'upstream_generated_binding_exception', 'upstream_imported_modules',
  )
  for key in inherited:
    if freeze.get(key) != original_freeze.get(key):
      raise AnalysisError(f'v3.3.2 inherited freeze binding changed at {key}.')
  v331_status = _validate_v3_3_1_status(freeze.get('v3_3_1_status'))
  inventory = freeze.get('file_sha256')
  if not isinstance(inventory, Mapping) or not inventory:
    raise AnalysisError('v3.3.2 freeze file inventory is absent.')
  required_files = {
      str(_AMENDMENT_PATH.relative_to(bundle_root)),
      str(Path(__file__).resolve().relative_to(bundle_root)),
      str(_TEST_PATH.relative_to(bundle_root)),
      'experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_2.py',
      'experiments/interpretability/opensplice/validate_encoder_skip_ood_sidecar_bootstrap_v3_3_2.py',
      'experiments/interpretability/opensplice/launch_encoder_skip_ood_sidecar_v3_3_2.py',
      'experiments/interpretability/opensplice/run_device_preflight_v3_3_2.py',
      'experiments/interpretability/opensplice/run_device_preflight_v3_3_2_test.py',
      'experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_2.sh',
      'experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_2_test.py',
      str(_ORIGINAL_PROTOCOL_PATH.relative_to(bundle_root)),
      str(_ORIGINAL_FREEZE_PATH.relative_to(bundle_root)),
      'experiments/interpretability/opensplice/run_encoder_skip_factorial_v3_3.py',
      'experiments/interpretability/opensplice/analyze_encoder_skip_localization_v3_3.py',
      'experiments/interpretability/opensplice/analyze_encoder_skip_localization_v3_3_test.py',
      str(_V3_3_1_AMENDMENT_PATH.relative_to(bundle_root)),
      str(_V3_3_1_ANALYZER_PATH.relative_to(bundle_root)),
      str(_V3_3_1_TEST_PATH.relative_to(bundle_root)),
      'src/alphagenome_research/model/model.py',
      'src/alphagenome_research/model/interpretability.py',
      'experiments/interpretability/opensplice/run_superset_graph_v3_2.py',
  }
  if not required_files.issubset(inventory):
    raise AnalysisError('v3.3.2 freeze misses a required source file.')
  for relative, digest in inventory.items():
    if (
        not isinstance(relative, str) or Path(relative).is_absolute()
        or '..' in Path(relative).parts or not _is_sha256(digest)
    ):
      raise AnalysisError('v3.3.2 file inventory is malformed.')
    path = (bundle_root / relative).resolve()
    try:
      path.relative_to(bundle_root)
    except ValueError as error:
      raise AnalysisError('v3.3.2 source path escaped the repository.') from error
    _strict_regular(path, f'v3.3.2 source {relative}')
    if _sha256(path) != digest:
      raise AnalysisError(f'v3.3.2 source bytes changed: {relative}.')
    subprocess.run(
        ('git', '-C', str(bundle_root), 'ls-files', '--error-unmatch', relative),
        check=True, capture_output=True,
    )
  subprocess.run(
      (
          'git', '-C', str(bundle_root), 'ls-files', '--error-unmatch',
          str(_FREEZE_PATH.relative_to(bundle_root)),
      ),
      check=True, capture_output=True,
  )
  if subprocess.check_output(
      ('git', '-C', str(bundle_root), 'diff', '--binary', 'HEAD', '--')
  ):
    raise AnalysisError('v3.3.2 requires a globally tracked-clean HEAD.')
  git_head = subprocess.check_output(
      ('git', '-C', str(bundle_root), 'rev-parse', 'HEAD'), text=True
  ).strip()
  original_audit, original_manifest, sequence_bindings = _validate_original_v3_3()
  expected_original_bundle = copy.deepcopy(
      original_audit['original_bundle_binding']
  )
  expected_original_bundle['git_head'] = git_head
  expected_original_bundle['tracked_paths'] = sorted({
      str(_ORIGINAL_FREEZE_PATH.relative_to(bundle_root)),
      str(Path(original_freeze['protocol_path']).resolve().relative_to(bundle_root)),
      str(Path(original_freeze['supersession_path']).resolve().relative_to(bundle_root)),
      *original_freeze['file_sha256'].keys(),
  })

  start = _read_json(run_dir / 'ATTEMPT_STARTED.json', 'ATTEMPT_STARTED')
  _require_exact_keys(start, {
      'attempt_id', 'script_version', 'status', 'amendment',
      'original_protocol_sha256', 'freeze', 'freeze_sha256', 'bootstrap',
      'external_preflight', 'same_process_preflight', 'runtime_environment',
      'runtime_version_binding', 'checkpoint_path', 'checkpoint_binding',
      'reference_object_binding', 'reference_sequence_bindings',
      'original_run_binding', 'original_run_revalidated_in_full',
      'v3_3_1_status', 'record_count_contract', 'model_apply_count_contract',
      'compile_count_contract', 'rerun_count_contract',
      'execution_order_contract', 'invariant_rows_between_calls',
      'active_recipient_rows_without_cross_call_predicate',
      'max_wall_time_seconds', 'max_output_bytes', 'confirmation_model_calls',
      'confirmation_scope_disclosure', 'started_at_unix_s',
  }, 'ATTEMPT_STARTED')
  expected_start = {
      'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION,
      'status': 'started_append_only_one_shot',
      'amendment': {
          'path': str(_AMENDMENT_PATH.resolve()), 'sha256': AMENDMENT_SHA256,
          'commit': AMENDMENT_COMMIT,
      },
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'freeze': freeze,
      'freeze_sha256': freeze_sha,
      'original_run_binding': dict(_ORIGINAL_BINDING),
      'original_run_revalidated_in_full': True,
      'v3_3_1_status': v331_status,
      'record_count_contract': 80,
      'model_apply_count_contract': 320,
      'compile_count_contract': {'eight_row': 1, 'six_row': 0},
      'rerun_count_contract': {'identity': 0, 'main_cube': 0},
      'execution_order_contract': {
          'recipient_orders': list(RECIPIENT_ORDERS),
          'anchor_ids': list(ANCHOR_IDS), 'major': 'recipient',
          'minor': 'anchor', 'indices': [0, 79],
      },
      'invariant_rows_between_calls': list(INVARIANT_ROWS),
      'active_recipient_rows_without_cross_call_predicate': list(ACTIVE_ROWS),
      'max_wall_time_seconds': MAX_WALL_TIME_SECONDS,
      'max_output_bytes': MAX_OUTPUT_BYTES,
      'confirmation_model_calls': 0,
      'confirmation_scope_disclosure': CONFIRMATION_DISCLOSURE,
  }
  for key, expected in expected_start.items():
    if start.get(key) != expected:
      raise AnalysisError(f'ATTEMPT_STARTED.{key} changed.')
  _finite(start.get('started_at_unix_s'), 'ATTEMPT_STARTED.started_at_unix_s')

  bootstrap = _require_exact_keys(start.get('bootstrap'), {
      'pid', 'created_at_unix_s', 'sanitized_environment',
      'generated_bindings', 'launcher_path', 'launcher_sha256',
      'bootstrap_path', 'bootstrap_sha256', 'freeze_path', 'freeze_sha256',
      'git_head', 'tracked_head_clean', 'original_run',
      'original_eight_row_compiler', 'original_run_all_5158_files_rehashed',
      'v3_3_1_status', 'original_bundle',
  }, 'ATTEMPT_STARTED.bootstrap')
  if (
      not isinstance(bootstrap.get('pid'), int) or bootstrap['pid'] <= 0
      or bootstrap.get('sanitized_environment') != {
          'LD_LIBRARY_PATH': 'absent',
          'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
          'JAX_ENABLE_COMPILATION_CACHE': 'false',
      }
      or bootstrap.get('freeze_path') != str(_FREEZE_PATH.resolve())
      or bootstrap.get('freeze_sha256') != freeze_sha
      or bootstrap.get('git_head') != git_head
      or bootstrap.get('tracked_head_clean') is not True
      or bootstrap.get('original_run') != _ORIGINAL_BINDING
      or bootstrap.get('original_run_all_5158_files_rehashed') is not True
      or bootstrap.get('v3_3_1_status') != v331_status
      or bootstrap.get('original_bundle')
      != expected_original_bundle
  ):
    raise AnalysisError('START bootstrap binding changed.')
  _finite(bootstrap.get('created_at_unix_s'), 'bootstrap.created_at_unix_s')
  for field, path in (
      ('launcher', _HERE / 'launch_encoder_skip_ood_sidecar_v3_3_2.py'),
      ('bootstrap', _HERE / 'validate_encoder_skip_ood_sidecar_bootstrap_v3_3_2.py'),
  ):
    if (
        bootstrap.get(f'{field}_path') != str(path.resolve())
        or bootstrap.get(f'{field}_sha256') != _sha256(path)
    ):
      raise AnalysisError(f'Bootstrap {field} binding changed.')
  compatibility = {
      'freeze': freeze,
      'same_process_pre_import_bootstrap': {
          'generated_bindings': bootstrap['generated_bindings']
      },
  }
  _, generated_audit = _v331._validate_generated_bindings(  # pylint: disable=protected-access
      compatibility, freeze=freeze
  )
  original_complete = _read_json(
      _ORIGINAL_RUN_DIR / 'RUN_COMPLETE.json', 'original RUN_COMPLETE'
  )
  if bootstrap.get('original_eight_row_compiler') != original_complete.get(
      'eight_row_compiler'
  ):
    raise AnalysisError('Bootstrap original eight-row compiler changed.')
  cases = _v33._load_cases()  # pylint: disable=protected-access
  checkpoint_audit = _v33._validate_checkpoint_reference(  # pylint: disable=protected-access
      start, original_freeze, cases
  )
  runtime_audit = _v33._validate_runtime_manifest(  # pylint: disable=protected-access
      start, original_freeze
  )
  if start.get('runtime_environment') != _read_json(
      _ORIGINAL_RUN_DIR / 'ATTEMPT_STARTED.json', 'original ATTEMPT_STARTED'
  ).get('runtime_environment'):
    raise AnalysisError('Sidecar runtime environment differs from frozen v3.3.')
  same = start.get('same_process_preflight')
  _v33._validate_device_observation(same, 'same-process preflight')  # pylint: disable=protected-access
  if (
      same.get('pid') != bootstrap['pid']
      or same.get('jax_enable_compilation_cache') is not False
      or same.get('v3_3_runtime_environment')
      != start.get('runtime_environment')
  ):
    raise AnalysisError('Same-process preflight PID differs from bootstrap.')
  external = start.get('external_preflight')
  if not isinstance(external, Mapping):
    raise AnalysisError('External preflight is absent.')
  preflight_path = Path(str(external.get('path'))).resolve()
  if preflight_path.parent != _PREFLIGHT_DIR.resolve() or _sha256(
      preflight_path
  ) != external.get('sha256'):
    raise AnalysisError('External preflight path/hash changed.')
  raw_preflight = _read_json(preflight_path, 'external preflight')
  embedded = {
      key: value for key, value in external.items()
      if key not in {'path', 'sha256', 'validated_logs'}
  }
  if raw_preflight != embedded or (
      raw_preflight.get('script_version') != 'opensplice-device-preflight-v3.3.2'
      or raw_preflight.get('status') != 'pass'
      or raw_preflight.get('amendment_sha256') != AMENDMENT_SHA256
      or raw_preflight.get('original_protocol_sha256') != ORIGINAL_PROTOCOL_SHA256
      or raw_preflight.get('freeze_sha256') != freeze_sha
      or raw_preflight.get('failure') is not None
      or raw_preflight.get('no_model_or_biological_access') is not True
      or raw_preflight.get('no_jit_or_array_kernel') is not True
  ):
    raise AnalysisError('External preflight contract changed.')
  logs = raw_preflight.get('logs')
  validated_logs = external.get('validated_logs')
  expected_logs = {}
  for stream in ('stdout', 'stderr'):
    row = logs.get(stream) if isinstance(logs, Mapping) else None
    if not isinstance(row, Mapping) or set(row) != {'path', 'sha256'}:
      raise AnalysisError(f'External preflight {stream} schema changed.')
    path = Path(str(row['path'])).resolve()
    if path.parent != _PREFLIGHT_DIR.resolve() or _sha256(path) != row['sha256']:
      raise AnalysisError(f'External preflight {stream} changed.')
    expected_logs[stream] = {'path': str(path), 'sha256': row['sha256']}
  if validated_logs != expected_logs:
    raise AnalysisError('External preflight validated-log binding changed.')
  _v33._validate_device_observation(  # pylint: disable=protected-access
      raw_preflight.get('observation'), 'external preflight'
  )
  if (
      raw_preflight.get('observation', {}).get(
          'jax_enable_compilation_cache'
      ) is not False
      or raw_preflight.get('observation', {}).get(
          'v3_3_runtime_environment'
      ) != start.get('runtime_environment')
  ):
    raise AnalysisError('External preflight runtime/cache binding changed.')
  return (
      freeze, freeze_sha,
      {
          'git_head': git_head,
          'generated_binding_audit': generated_audit,
          'original_run_audit': original_audit,
          'v3_3_1_status': v331_status,
          'checkpoint_reference_audit': checkpoint_audit,
          'runtime_audit': runtime_audit,
          'external_preflight_sha256': external['sha256'],
          'same_process_exact_rtx3090_uuid_gate': True,
          'external_exact_rtx3090_uuid_gate': True,
      },
      original_manifest, sequence_bindings,
  )


def _validate_compiler(
    run_dir: Path, emitted: Any, original_compiler: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
  """Validates the sole eight-row executable and its frozen-v3.3 identity."""
  compiler = _require_exact_keys(emitted, {
      'executable_name', 'compile_count', 'compile_seconds',
      'executable_fingerprint', 'artifacts', 'program_signatures',
      'original_v3_3_compiler_binding', 'original_graph_comparison',
      'original_executable_fingerprint_exact',
      'graph_and_hlo_exact_to_original_v3_3',
  }, 'eight-row compiler')
  if compiler.get('executable_name') != 'eight_row' or compiler.get(
      'compile_count'
  ) != 1 or _finite(
      compiler.get('compile_seconds'), 'compiler.compile_seconds'
  ) < 0:
    raise AnalysisError('Eight-row compiler identity/count/timing changed.')
  if compiler.get('original_v3_3_compiler_binding') != dict(original_compiler):
    raise AnalysisError('Eight-row compiler original binding changed.')
  directory = run_dir / 'compiler/eight_row'
  provenance = directory / 'COMPILER_PROVENANCE.json'
  if _read_json(provenance, 'compiler provenance') != dict(compiler):
    raise AnalysisError('Compiler provenance file differs from RUN_COMPLETE.')
  expected_names = {
      'stablehlo': 'graph.stablehlo.mlir',
      'hlo': 'graph.pre_backend.hlo.txt',
      'compiled_hlo': 'graph.compiled.hlo.txt',
  }
  artifacts = _require_exact_keys(
      compiler.get('artifacts'), set(expected_names), 'compiler.artifacts'
  )
  comparisons = _require_exact_keys(
      compiler.get('original_graph_comparison'), set(expected_names),
      'compiler.original_graph_comparison',
  )
  expected_files = {provenance.resolve()}
  computed_comparisons = {}
  for name, filename in expected_names.items():
    binding = _require_exact_keys(
        artifacts[name], {'path', 'sha256', 'size_bytes'},
        f'compiler.artifacts.{name}',
    )
    path = Path(str(binding.get('path'))).resolve()
    expected_path = (directory / filename).resolve()
    if path != expected_path:
      raise AnalysisError(f'Compiler {name} path changed.')
    _strict_regular(path, f'compiler.{name}')
    if (
        not _is_sha256(binding.get('sha256'))
        or _sha256(path) != binding['sha256']
        or path.stat().st_size != binding.get('size_bytes')
    ):
      raise AnalysisError(f'Compiler {name} bytes changed.')
    original = original_compiler.get('artifacts', {}).get(name, {})
    computed_comparisons[name] = {
        'sha256_exact': binding['sha256'] == original.get('sha256'),
        'size_exact': binding['size_bytes'] == original.get('size_bytes'),
    }
    if comparisons[name] != computed_comparisons[name]:
      raise AnalysisError(f'Compiler {name} original comparison changed.')
    expected_files.add(path)
  observed = set()
  if directory.is_symlink() or not directory.is_dir():
    raise AnalysisError('Eight-row compiler directory is absent or unsafe.')
  for path in directory.iterdir():
    mode = path.lstat().st_mode
    if path.is_symlink() or not stat.S_ISREG(mode):
      raise AnalysisError('Compiler tree contains a symlink/directory/special entry.')
    observed.add(path.resolve())
  if observed != expected_files:
    raise AnalysisError('Compiler tree does not contain exactly four files.')
  fingerprint = hashlib.sha256(
      bytes.fromhex(artifacts['compiled_hlo']['sha256'])
  ).hexdigest()
  if compiler.get('executable_fingerprint') != fingerprint:
    raise AnalysisError('Eight-row executable fingerprint changed.')
  fingerprint_exact = fingerprint == original_compiler.get(
      'executable_fingerprint'
  )
  graph_exact = fingerprint_exact and all(
      row['sha256_exact'] and row['size_exact']
      for row in computed_comparisons.values()
  )
  if (
      compiler.get('original_executable_fingerprint_exact') is not fingerprint_exact
      or compiler.get('graph_and_hlo_exact_to_original_v3_3') is not graph_exact
  ):
    raise AnalysisError('Compiler exactness flags do not match current bytes.')
  signatures = compiler.get('program_signatures')
  if not isinstance(signatures, Mapping) or set(signatures) != {
      'selection', 'target', 'eight_interventions'
  }:
    raise AnalysisError('Compiler program-signature schema changed.')
  return fingerprint, {
      'executable_fingerprint': fingerprint,
      'graph_and_hlo_exact_to_original_v3_3': graph_exact,
      'artifact_sha256': {
          name: artifacts[name]['sha256'] for name in expected_names
      },
  }


def _sidecar_source_paths() -> tuple[Path, ...]:
  return (
      _HERE / 'run_encoder_skip_ood_sidecar_v3_3_2.py',
      _HERE / 'launch_encoder_skip_ood_sidecar_v3_3_2.py',
      _HERE / 'validate_encoder_skip_ood_sidecar_bootstrap_v3_3_2.py',
      _HERE / 'run_device_preflight_v3_3_2.py',
      _AMENDMENT_PATH,
      _FREEZE_PATH,
      _HERE / 'run_encoder_skip_factorial_v3_3.py',
  )


def _validate_import_file(
    path: Path, expected_sha: Any, *, bundle_root: Path,
    freeze: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
  if not _is_sha256(expected_sha) or _sha256(path) != expected_sha:
    raise AnalysisError(f'Import-provenance hash mismatch: {path.name}.')
  value = _read_json(path, path.name)
  _require_exact_keys(value, {
      'module_count', 'modules', 'upstream_source_attestation',
      'v3_3_2_sidecar_sources',
  }, path.name)
  expected_sources = {
      str(source.resolve()): {
          'sha256': _sha256(source.resolve()),
          'size_bytes': source.resolve().stat().st_size,
      }
      for source in _sidecar_source_paths()
  }
  if len(expected_sources) != 7 or value.get(
      'v3_3_2_sidecar_sources'
  ) != expected_sources:
    raise AnalysisError(f'{path.name} seven-source sidecar binding changed.')
  inventory = freeze.get('upstream_imported_modules')
  exception = freeze.get('upstream_generated_binding_exception')
  upstream_head = freeze.get('upstream_alphagenome_git_head')
  if not isinstance(inventory, Mapping) or len(inventory) != 26:
    raise AnalysisError('Frozen upstream source inventory is incomplete.')
  upstream_root = (bundle_root.parent / 'alphagenome').resolve()
  generated_names = set(_v33.UPSTREAM_GENERATED_MODULE_NAMES)
  expected_attestation = {
      'git_head': upstream_head,
      'tracked_head_clean': True,
      'imported_module_count': 26,
      'imported_modules': {
          name: {
              **binding,
              'path': str((upstream_root / binding['relative_path']).resolve()),
              'source_kind': (
                  'generated_exact_byte_exception'
                  if name in generated_names else 'tracked'
              ),
          }
          for name, binding in inventory.items()
      },
      'tracked_imported_module_count': 22,
      'generated_imported_module_count': 4,
      'generated_binding_exception': exception,
  }
  if value.get('upstream_source_attestation') != expected_attestation:
    raise AnalysisError(f'{path.name} upstream source attestation changed.')
  modules = value.get('modules')
  if (
      not isinstance(modules, list) or not modules
      or value.get('module_count') != len(modules)
  ):
    raise AnalysisError(f'{path.name} module list/count is invalid.')
  by_name: dict[str, Mapping[str, Any]] = {}
  by_path: dict[str, list[str]] = defaultdict(list)
  observed_upstream = {}
  for raw in modules:
    row = _require_exact_keys(
        raw, {'name', 'path', 'root', 'sha256', 'size_bytes'},
        f'{path.name}.module',
    )
    name = row.get('name')
    if not isinstance(name, str) or not name or name in by_name:
      raise AnalysisError(f'{path.name} has a duplicate/malformed module name.')
    root = {
        'alphagenome_research_checkout': bundle_root.resolve(),
        'upstream_alphagenome_checkout': upstream_root,
    }.get(row.get('root'))
    if root is None:
      raise AnalysisError(f'{path.name} module root changed.')
    module_path = Path(str(row.get('path'))).resolve()
    try:
      module_path.relative_to(root)
    except ValueError as error:
      raise AnalysisError(f'{path.name} module escaped its declared root.') from error
    _strict_regular(module_path, f'{path.name}.{name}')
    if (
        not _is_sha256(row.get('sha256'))
        or _sha256(module_path) != row['sha256']
        or module_path.stat().st_size != row.get('size_bytes')
    ):
      raise AnalysisError(f'{path.name} module bytes changed: {name}.')
    by_name[name] = dict(row)
    by_path[str(module_path)].append(name)
    if row['root'] == 'upstream_alphagenome_checkout':
      observed_upstream[name] = {
          'relative_path': str(module_path.relative_to(upstream_root)),
          'sha256': row['sha256'], 'size_bytes': row['size_bytes'],
      }
  if observed_upstream != dict(inventory):
    raise AnalysisError(f'{path.name} upstream import inventory changed.')
  for duplicate_path, names in {
      key: value for key, value in by_path.items() if len(value) > 1
  }.items():
    if (
        set(names) != {'__main__', '__mp_main__'}
        or Path(duplicate_path).name
        != 'run_encoder_skip_ood_sidecar_v3_3_2.py'
    ):
      raise AnalysisError(f'{path.name} has an unapproved duplicate path alias.')
    rows = [by_name[name] for name in names]
    if any(
        (row['sha256'], row['size_bytes'], row['root'])
        != (rows[0]['sha256'], rows[0]['size_bytes'], rows[0]['root'])
        for row in rows[1:]
    ):
      raise AnalysisError(f'{path.name} approved alias bytes differ.')
  return by_name


def _validate_imports(
    run_dir: Path, completion: Mapping[str, Any], *, bundle_root: Path,
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
  filenames = {
      'pre_model': 'IMPORT_PROVENANCE_PRE_MODEL.json',
      'post_model_precompile': 'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
      'postcompile': 'IMPORT_PROVENANCE.json',
  }
  bindings = _require_exact_keys(
      completion.get('import_provenance_phases'), set(filenames),
      'RUN_COMPLETE.import_provenance_phases',
  )
  phases = {
      phase: _validate_import_file(
          run_dir / filename, bindings[phase], bundle_root=bundle_root,
          freeze=freeze,
      )
      for phase, filename in filenames.items()
  }
  if completion.get('import_provenance_sha256') != bindings['postcompile']:
    raise AnalysisError('Final import-provenance binding changed.')
  lazy = {}
  for earlier, later in (
      ('pre_model', 'post_model_precompile'),
      ('post_model_precompile', 'postcompile'),
  ):
    missing = set(phases[earlier]) - set(phases[later])
    changed = {
        name for name in phases[earlier]
        if name in phases[later] and phases[earlier][name] != phases[later][name]
    }
    if missing or changed:
      raise AnalysisError('Import-provenance shared module bytes changed.')
    lazy[f'{earlier}_to_{later}'] = sorted(
        set(phases[later]) - set(phases[earlier])
    )
  required = {
      'alphagenome_research.model.model',
      'alphagenome_research.model.dna_model',
      'alphagenome_research.model.interpretability',
  }
  if not required.issubset(phases['postcompile']):
    raise AnalysisError('Final import provenance lacks required model modules.')
  return {
      'phase_sha256': dict(bindings),
      'module_counts': {key: len(value) for key, value in phases.items()},
      'lazy_additions': lazy,
      'stable_shared_module_bytes': True,
      'seven_sidecar_source_bindings_exact': True,
  }


def _validate_protobuf(
    run_dir: Path, completion: Mapping[str, Any], freeze: Mapping[str, Any],
) -> dict[str, Any]:
  path = run_dir / 'PROTOBUF_PROVENANCE.json'
  expected = completion.get('protobuf_provenance_sha256')
  if not _is_sha256(expected) or _sha256(path) != expected:
    raise AnalysisError('Protobuf provenance hash binding changed.')
  value = _read_json(path, 'PROTOBUF_PROVENANCE')
  if value != freeze.get('protobuf_binding'):
    raise AnalysisError('Protobuf provenance differs from the freeze.')

  def visit(node: Any) -> None:
    if isinstance(node, Mapping):
      if isinstance(node.get('path'), str):
        bound = Path(node['path']).resolve()
        _strict_regular(bound, 'protobuf bound path')
        if 'sha256' in node and _sha256(bound) != node['sha256']:
          raise AnalysisError('Protobuf bound file hash changed.')
        if 'size_bytes' in node and bound.stat().st_size != node['size_bytes']:
          raise AnalysisError('Protobuf bound file size changed.')
      for child in node.values():
        visit(child)
    elif isinstance(node, list):
      for child in node:
        visit(child)
  visit(value)
  generated = value.get('generated_outputs')
  if not isinstance(generated, Mapping) or len(generated) != 2 or {
      Path(item).name for item in generated
  } != {'calibration_scores_pb2.py', 'calibration_scores_pb2.pyi'}:
    raise AnalysisError('Protobuf generated-output set is not exactly two files.')
  for path_value, binding in generated.items():
    if not isinstance(path_value, str):
      raise AnalysisError('Protobuf generated-output path is malformed.')
    row = _require_exact_keys(
        binding, {'sha256', 'size_bytes'}, 'protobuf.generated_output'
    )
    output_path = Path(path_value).resolve()
    _strict_regular(output_path, 'protobuf generated output')
    if (
        not _is_sha256(row.get('sha256'))
        or _sha256(output_path) != row['sha256']
        or output_path.stat().st_size != row.get('size_bytes')
    ):
      raise AnalysisError('Protobuf generated-output bytes changed.')
  return {
      'sha256': expected,
      'generated_output_count': 2,
      'seven_role_two_generated_output_repair_exact': True,
  }


_COMPLETION_KEYS = {
    'status', 'stop_reason', 'message', 'attempt_id', 'script_version',
    'amendment_sha256', 'amendment_commit', 'original_protocol_sha256',
    'freeze_sha256', 'ood_anchor_record_count', 'ood_invalid_count',
    'unique_recipient_anchor_count', 'all_80_recipient_anchors_complete',
    'model_apply_count', 'expected_model_apply_count',
    'eight_row_compile_count', 'six_row_compile_count',
    'identity_rerun_count', 'main_cube_rerun_count',
    'old_ood_records_reused', 'one_fixed_eight_row_executable',
    'eight_row_compiler', 'eight_row_executable_fingerprint',
    'graph_and_hlo_exact_to_original_v3_3', 'id0_all20', 'id255_all20',
    'invariant_rows_between_calls',
    'active_rows_have_no_forced_cross_call_predicate',
    'original_run_binding', 'original_run_revalidated_in_full',
    'original_ood_records_provenance_only', 'v3_3_1_status',
    'import_provenance_phases', 'import_provenance_sha256',
    'protobuf_provenance_sha256', 'raw_manifest',
    'confirmation_model_calls', 'confirmation_scope_disclosure',
    'scientific_summary_computed', 'shapley_or_nomination_computed',
    'completed_at_unix_s',
}


def _completion_prefix(
    completion: Mapping[str, Any], *, freeze_sha: str,
    v331_status: Mapping[str, Any],
) -> tuple[tuple[tuple[int, int], ...], bool]:
  _require_exact_keys(completion, _COMPLETION_KEYS, 'RUN_COMPLETE')
  common = {
      'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION,
      'amendment_sha256': AMENDMENT_SHA256,
      'amendment_commit': AMENDMENT_COMMIT,
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'freeze_sha256': freeze_sha,
      'expected_model_apply_count': EXPECTED_APPLY_COUNT,
      'eight_row_compile_count': 1,
      'six_row_compile_count': 0,
      'identity_rerun_count': 0,
      'main_cube_rerun_count': 0,
      'old_ood_records_reused': 0,
      'one_fixed_eight_row_executable': True,
      'invariant_rows_between_calls': list(INVARIANT_ROWS),
      'active_rows_have_no_forced_cross_call_predicate': True,
      'original_run_binding': dict(_ORIGINAL_BINDING),
      'original_run_revalidated_in_full': True,
      'original_ood_records_provenance_only': True,
      'v3_3_1_status': dict(v331_status),
      'confirmation_model_calls': 0,
      'confirmation_scope_disclosure': CONFIRMATION_DISCLOSURE,
      'scientific_summary_computed': False,
      'shapley_or_nomination_computed': False,
  }
  for key, expected in common.items():
    if completion.get(key) != expected:
      raise AnalysisError(f'RUN_COMPLETE.{key} changed.')
  _finite(completion.get('completed_at_unix_s'), 'RUN_COMPLETE.completed_at_unix_s')
  for key in (
      'ood_anchor_record_count', 'ood_invalid_count',
      'unique_recipient_anchor_count', 'model_apply_count',
  ):
    value = completion.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
      raise AnalysisError(f'RUN_COMPLETE.{key} is not a nonnegative integer.')
  count = completion['ood_anchor_record_count']
  if (
      count > EXPECTED_RECORD_COUNT
      or completion['unique_recipient_anchor_count'] != count
      or completion['model_apply_count'] != 4 * count
  ):
    raise AnalysisError('RUN_COMPLETE sidecar count/apply arithmetic changed.')
  status, reason = completion.get('status'), completion.get('stop_reason')
  full = status == 'complete' and reason is None
  if full:
    valid = (
        count == 80 and completion['ood_invalid_count'] == 0
        and completion.get('all_80_recipient_anchors_complete') is True
        and completion.get('id0_all20') is True
        and completion.get('id255_all20') is True
        and completion.get('graph_and_hlo_exact_to_original_v3_3') is True
        and completion.get('message')
        == 'All 80 frozen v3.3.2 OOD sidecar records completed.'
    )
  elif status == 'controlled_stop' and reason == 'compiler_graph_mismatch':
    valid = (
        count == completion['ood_invalid_count'] == 0
        and completion.get('all_80_recipient_anchors_complete') is False
        and completion.get('id0_all20') is False
        and completion.get('id255_all20') is False
        and completion.get('graph_and_hlo_exact_to_original_v3_3') is False
        and completion.get('message')
        == 'New eight-row graph/HLO differs from frozen v3.3.'
    )
  elif status == 'controlled_stop' and reason == 'ood_tooling_failure':
    if not (1 <= count <= 80 and completion['ood_invalid_count'] == 1):
      valid = False
    else:
      order, anchor = _execution_order()[count - 1]
      valid = completion.get('message') == (
          f'OOD sidecar audit failed at order={order}, anchor_id={anchor}.'
      ) and completion.get('all_80_recipient_anchors_complete') is False
  else:
    valid = False
  if not valid:
    raise AnalysisError('RUN_COMPLETE status/count/message does not match an allowed prefix.')
  if not isinstance(completion.get('id0_all20'), bool) or not isinstance(
      completion.get('id255_all20'), bool
  ):
    raise AnalysisError('RUN_COMPLETE anchor-closure flags are not booleans.')
  return _execution_order()[:count], full


def _validate_manifest(
    run_dir: Path, cases: Mapping[int, Mapping[str, Any]],
    prefix: Sequence[tuple[int, int]],
) -> tuple[dict[str, Any], dict[str, str]]:
  manifest = _read_json(run_dir / 'RAW_MANIFEST.json', 'RAW_MANIFEST')
  _require_exact_keys(
      manifest, {'artifact_count', 'artifact_sha256', 'artifact_tree_sha256'},
      'RAW_MANIFEST',
  )
  expected_paths = {
      _artifact_relative(cases[order], anchor) for order, anchor in prefix
  }
  raw_root = run_dir / 'raw'
  observed_files: set[str] = set()
  if expected_paths:
    if raw_root.is_symlink() or not raw_root.is_dir():
      raise AnalysisError('Sidecar raw directory is absent or unsafe.')
    expected_directories = {raw_root.resolve()}
    for relative in expected_paths:
      path = (run_dir / relative).resolve()
      expected_directories.update(
          parent for parent in path.parents
          if parent == raw_root.resolve() or raw_root.resolve() in parent.parents
      )
    for lexical in raw_root.rglob('*'):
      mode = lexical.lstat().st_mode
      if lexical.is_symlink():
        raise AnalysisError('Sidecar raw tree contains a symlink.')
      if stat.S_ISREG(mode):
        observed_files.add(str(lexical.relative_to(run_dir)))
      elif stat.S_ISDIR(mode):
        if lexical.resolve() not in expected_directories:
          raise AnalysisError('Sidecar raw tree contains an extra/empty directory.')
      else:
        raise AnalysisError('Sidecar raw tree contains a special entry.')
  elif raw_root.exists():
    raise AnalysisError('Zero-record controlled stop must not contain a raw tree.')
  if observed_files != expected_paths:
    raise AnalysisError('Raw tree differs from the exact frozen sidecar prefix.')
  mapping = manifest.get('artifact_sha256')
  if not isinstance(mapping, Mapping) or set(mapping) != expected_paths:
    raise AnalysisError('RAW_MANIFEST does not enumerate the exact prefix.')
  clean = {}
  for relative in sorted(expected_paths):
    digest = mapping[relative]
    path = run_dir / relative
    if not _is_sha256(digest) or _sha256(path) != digest:
      raise AnalysisError(f'RAW_MANIFEST hash mismatch: {relative}.')
    clean[relative] = digest
  if (
      manifest.get('artifact_count') != len(expected_paths)
      or not _is_sha256(manifest.get('artifact_tree_sha256'))
      or manifest.get('artifact_tree_sha256')
      != _tree_digest((run_dir / relative for relative in expected_paths), run_dir)
  ):
    raise AnalysisError('RAW_MANIFEST count/tree digest changed.')
  return dict(manifest), clean


def _validate_top_level_tree(run_dir: Path, *, has_raw: bool) -> None:
  expected_files = {
      'ATTEMPT_STARTED.json', 'IMPORT_PROVENANCE_PRE_MODEL.json',
      'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
      'IMPORT_PROVENANCE.json', 'PROTOBUF_PROVENANCE.json',
      'RAW_MANIFEST.json', 'RUN_COMPLETE.json',
  }
  expected_directories = {'compiler'} | ({'raw'} if has_raw else set())
  observed_files, observed_directories = set(), set()
  for path in run_dir.iterdir():
    mode = path.lstat().st_mode
    if path.is_symlink():
      raise AnalysisError('Run root contains a symlink.')
    if stat.S_ISREG(mode):
      observed_files.add(path.name)
    elif stat.S_ISDIR(mode):
      observed_directories.add(path.name)
    else:
      raise AnalysisError('Run root contains a special entry.')
  if observed_files != expected_files or observed_directories != expected_directories:
    raise AnalysisError('Run root contains missing or extra artifacts.')
  compiler_root = run_dir / 'compiler'
  children = list(compiler_root.iterdir())
  if (
      len(children) != 1 or children[0].name != 'eight_row'
      or children[0].is_symlink() or not children[0].is_dir()
  ):
    raise AnalysisError('Compiler root is not exactly compiler/eight_row.')


def analyze(
    run_dir: Path, *, bundle_root: Path | None = None,
) -> dict[str, Any]:
  """Audits only structure/evidence; it never computes a scientific summary."""
  _assert_cpu_only('v3.3.2 analyzer entry')
  run_dir = run_dir.resolve()
  bundle_root = _REPO_ROOT if bundle_root is None else bundle_root.resolve()
  _guard_path(run_dir)
  _guard_path(bundle_root)
  if not run_dir.is_dir() or run_dir.is_symlink():
    raise AnalysisError('v3.3.2 run directory is absent or unsafe.')
  if (run_dir / 'TERMINAL_FAILURE.json').exists():
    raise AnalysisError('Unexpected TERMINAL_FAILURE is not an auditable outcome.')
  freeze, freeze_sha, start_audit, original_manifest, sequence_bindings = (
      _validate_freeze_and_start(run_dir, bundle_root=bundle_root)
  )
  cases = _v33._load_cases()  # pylint: disable=protected-access
  completion = _read_json(run_dir / 'RUN_COMPLETE.json', 'RUN_COMPLETE')
  prefix, fully_complete = _completion_prefix(
      completion, freeze_sha=freeze_sha,
      v331_status=start_audit['v3_3_1_status'],
  )
  _validate_top_level_tree(run_dir, has_raw=bool(prefix))
  manifest, raw_hashes = _validate_manifest(run_dir, cases, prefix)
  if completion.get('raw_manifest') != manifest:
    raise AnalysisError('RUN_COMPLETE embedded raw manifest changed.')

  original_complete = _read_json(
      _ORIGINAL_RUN_DIR / 'RUN_COMPLETE.json', 'original RUN_COMPLETE'
  )
  original_compiler = original_complete.get('eight_row_compiler')
  if not isinstance(original_compiler, Mapping):
    raise AnalysisError('Original v3.3 eight-row compiler binding is absent.')
  fingerprint, compiler_audit = _validate_compiler(
      run_dir, completion.get('eight_row_compiler'), original_compiler
  )
  if (
      completion.get('eight_row_executable_fingerprint') != fingerprint
      or completion.get('graph_and_hlo_exact_to_original_v3_3')
      is not compiler_audit['graph_and_hlo_exact_to_original_v3_3']
  ):
    raise AnalysisError('RUN_COMPLETE compiler linkage changed.')
  imports_audit = _validate_imports(
      run_dir, completion, bundle_root=bundle_root, freeze=freeze
  )
  protobuf_audit = _validate_protobuf(run_dir, completion, freeze)

  record_audits = []
  for index, (order, anchor) in enumerate(prefix):
    case = cases[order]
    donor_case = cases[_donor_order(order)]
    relative = _artifact_relative(case, anchor)
    record = _read_json(run_dir / relative, relative)
    allow_invalid = (
        completion.get('status') == 'controlled_stop'
        and completion.get('stop_reason') == 'ood_tooling_failure'
        and index == len(prefix) - 1
    )
    record_audits.append(_validate_record(
        record, case=case, donor_case=donor_case, anchor=anchor,
        execution_index=index, freeze_sha256=freeze_sha,
        executable_fingerprint=fingerprint,
        original_manifest=original_manifest,
        sequence_bindings=sequence_bindings, allow_invalid=allow_invalid,
    ))
  if record_audits:
    invalid_indices = [
        index for index, row in enumerate(record_audits)
        if row['status'] != 'complete'
    ]
    expected_invalid = (
        [len(record_audits) - 1]
        if completion.get('stop_reason') == 'ood_tooling_failure' else []
    )
    if invalid_indices != expected_invalid:
      raise AnalysisError('Sidecar invalid artifact is not the exact final prefix row.')
  observed_id0 = sum(
      row['anchor'] == 0 and row['status'] == 'complete'
      for row in record_audits
  ) == 20
  observed_id255 = sum(
      row['anchor'] == 255 and row['status'] == 'complete'
      for row in record_audits
  ) == 20
  if (
      completion.get('id0_all20') is not observed_id0
      or completion.get('id255_all20') is not observed_id255
  ):
    raise AnalysisError('RUN_COMPLETE ID0/ID255 flags differ from audited prefix.')
  if fully_complete and (
      len(record_audits) != 80
      or any(row['status'] != 'complete' for row in record_audits)
      or not compiler_audit['graph_and_hlo_exact_to_original_v3_3']
  ):
    raise AnalysisError('Complete sidecar lacks exact 80-record/compiler closure.')

  if fully_complete:
    decision = 'sidecar_complete_structural_audit'
  else:
    decision = f"controlled_stop_{completion['stop_reason']}"
  combined_permitted = bool(
      fully_complete
      and start_audit['v3_3_1_status'].get('state') == 'completed'
      and compiler_audit['graph_and_hlo_exact_to_original_v3_3']
  )
  result = {
      'analysis_version': ANALYSIS_VERSION,
      'status': (
          'complete_structural_audit' if fully_complete
          else 'complete_controlled_stop_audited'
      ),
      'decision': decision,
      'controlled_stop': (
          None if fully_complete else {
              'reason': completion['stop_reason'],
              'message': completion['message'],
              'audited_record_count': len(prefix),
          }
      ),
      'scientific_summary_computed': False,
      'shapley_or_nomination_computed': False,
      'nomination_performed': False,
      'nomination': None,
      'resolution_analysis': None,
      'combined_analysis_permitted': combined_permitted,
      'combined_analysis_requirement': (
          'A separately prospective analyzer is required before any Shapley, '
          'resolution, or nomination calculation.'
      ),
      'sidecar_audit': {
          'expected_record_count': 80,
          'audited_record_count': len(prefix),
          'valid_record_count': sum(
              row['status'] == 'complete' for row in record_audits
          ),
          'invalid_record_count': sum(
              row['status'] != 'complete' for row in record_audits
          ),
          'expected_model_apply_count': 320,
          'audited_model_apply_count': completion['model_apply_count'],
          'execution_order_exact': True,
          'raw_manifest_sha256': _sha256(run_dir / 'RAW_MANIFEST.json'),
          'raw_artifact_tree_sha256': manifest['artifact_tree_sha256'],
          'raw_artifact_count': manifest['artifact_count'],
          'raw_hash_count': len(raw_hashes),
          'raw_endpoint_evidence_recomputed_where_emitted': True,
          'all_four_readouts_present_for_every_record': fully_complete,
          'donor_normalization_computed': False,
          'old_ood_records_used_as_data': False,
          'invariant_rows_between_calls': list(INVARIANT_ROWS),
          'active_rows_without_forced_cross_call_predicate': list(ACTIVE_ROWS),
          'id0_all20': observed_id0,
          'id255_all20': observed_id255,
      },
      'provenance_audit': {
          **start_audit,
          'freeze_sha256': freeze_sha,
          'amendment_sha256': AMENDMENT_SHA256,
          'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
          'original_cube_bound_separately': True,
          'sidecar_bound_separately': True,
          'compiler': compiler_audit,
          'imports': imports_audit,
          'protobuf': protobuf_audit,
          'seven_role_two_generated_output_repair_exact': True,
      },
      'confirmation_model_outputs_activations_interventions_unopened': True,
      'confirmation_scope_disclosure': CONFIRMATION_DISCLOSURE,
  }
  _assert_cpu_only('v3.3.2 analyzer exit')
  return result


def render_markdown(result: Mapping[str, Any]) -> str:
  lines = [
      '# OpenSplice v3.3.2 OOD sidecar structural audit', '',
      f"**Decision:** `{result['decision']}`", '',
  ]
  if result.get('controlled_stop') is None:
    lines.extend([
        'All 80 fresh OOD sidecar records passed the frozen structural, raw-',
        'endpoint, repeat, route-map, provenance, and compiler-identity gates.',
        '',
        'This makes a future combined scientific analysis structurally '
        'eligible. It does not itself compute Shapley values, a resolution '
        'gate, or a mechanism nomination.', '',
    ])
  else:
    lines.extend([
        'The exact append-only controlled-stop prefix was audited. The '
        'sidecar is not structurally complete, so combined analysis is not '
        'permitted.', '',
    ])
  lines.extend([
      '**Scientific summary computed:** no',
      '**Shapley or nomination computed:** no',
      f"**Combined analysis permitted:** {'yes' if result['combined_analysis_permitted'] else 'no'}",
      '',
      'Any combined scientific interpretation requires a separately '
      'prospective analyzer committed after this sidecar completes.', '',
      'Confirmation model outputs, activations, and interventions remained '
      'unopened. Previously disclosed later-exon metadata/labels were exposed '
      'after the original protocol freeze.', '',
  ])
  return '\n'.join(lines)


def _write_analysis_outputs(
    output_json: Path, output_markdown: Path, result: Mapping[str, Any],
) -> None:
  expected_json = (_ANALYSIS_DIR / 'ANALYSIS.json').resolve()
  expected_markdown = (_ANALYSIS_DIR / 'RESULT.md').resolve()
  if output_json.resolve() != expected_json or output_markdown.resolve() != expected_markdown:
    raise AnalysisError('Analysis output destinations differ from the freeze.')
  if _ANALYSIS_DIR.exists():
    raise FileExistsError('Append-only v3.3.2 analysis directory already exists.')
  _ANALYSIS_DIR.mkdir(parents=False, exist_ok=False)
  payloads = (
      (output_json, json.dumps(result, indent=2, sort_keys=True) + '\n'),
      (output_markdown, render_markdown(result)),
  )
  for path, text_value in payloads:
    with path.open('x', encoding='utf-8') as handle:
      handle.write(text_value)
      handle.flush()
      os.fsync(handle.fileno())


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run-dir', type=Path, required=True)
  parser.add_argument('--bundle-root', type=Path, required=True)
  parser.add_argument('--output-json', type=Path, required=True)
  parser.add_argument('--output-markdown', type=Path, required=True)
  args = parser.parse_args()
  if args.run_dir.resolve() != _RUN_DIR.resolve():
    parser.error('--run-dir differs from the frozen sidecar output directory.')
  if args.bundle_root.resolve() != _REPO_ROOT.resolve():
    parser.error('--bundle-root differs from the frozen repository.')
  if args.output_json.resolve() != (_ANALYSIS_DIR / 'ANALYSIS.json').resolve():
    parser.error('--output-json differs from the frozen analysis path.')
  if args.output_markdown.resolve() != (_ANALYSIS_DIR / 'RESULT.md').resolve():
    parser.error('--output-markdown differs from the frozen analysis path.')
  result = analyze(args.run_dir, bundle_root=args.bundle_root)
  _write_analysis_outputs(args.output_json, args.output_markdown, result)
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
  main()
