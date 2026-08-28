#!/usr/bin/env python3
"""CPU-only, fail-closed analysis for the OpenSplice v3.3 skip census.

The module deliberately imports neither AlphaGenome nor JAX.  It reconstructs
the frozen endpoint-logit reducer from raw evidence before computing recovery,
exact Shapley values, Shapley interactions, Harsanyi dividends, and the
predeclared resolution nomination.  File/provenance validation is kept in the
same module so a malformed or partial cube cannot reach the estimators.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import stat
import struct
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


ANALYSIS_VERSION = 'opensplice-encoder-skip-localization-analysis-v3.3.0'
SOURCE_SCRIPT_VERSION = 'opensplice-encoder-skip-factorial-v3.3.0'
ATTEMPT_ID = 'opensplice-v3.3-development-encoder-skip-factorial-one-shot'
PROTOCOL_SHA256 = (
    '85151029297cf4b4c1a07dd2a0f47b10b0663acd08ddfe25ba72b9898b1beea0'
)
ORIGINAL_PROTOCOL_SHA256 = (
    '89a3c5ebf7a6af85de58f37952047694fd14c61ef11e72668ce4392f6077a342'
)
ORDERING_CLARIFICATION_COMMIT = '6c77e3f'
CAPACITY_CLARIFICATION_COMMIT = '62ec610'
UPSTREAM_PROVENANCE_AMENDMENT_COMMIT = '93227d4'
EXPECTED_MODEL_APPLIES = 10_600
MAX_WALL_TIME_SECONDS = 14_400
MAX_OUTPUT_BYTES = 8_589_934_592
CONFIRMATION_SCOPE_DISCLOSURE = (
    'Later-exon metadata/labels were exposed after protocol freeze; '
    'no later-exon model outputs, activations, or interventions are used.'
)
EXPECTED_DEVICE_KIND = 'NVIDIA GeForce RTX 3090'
EXPECTED_GPU_UUID = 'GPU-64111645-1e42-a96d-f192-4abbec4b8090'
EXPECTED_COMPUTE_CAPABILITY = '8.6'
SUPERSESSION_COMMIT = 'c64def455412153b12a008f6b48be58cf6bf8d59'
SUPERSESSION_SHA256 = (
    'ca29860eae1d41b9c5c69908b2209c0b5fe06b5d1b9c2f70225ee2d0656fa0dd'
)
PLAYERS_E = ('E64', 'E32', 'E16', 'E8', 'E4', 'E2', 'E1')
PLAYERS_8 = ('T',) + PLAYERS_E
EFFECT_ORDERS = tuple(range(0, 6)) + tuple(range(10, 16))
NEUTRAL_ORDERS = tuple(range(6, 10)) + tuple(range(16, 20))
ALL_ORDERS = tuple(range(20))
OOD_ANCHORS = (0, 127, 128, 255)
EXPECTED_IDENTITIES = 20
EXPECTED_COALITIONS = 5_120
EXPECTED_OOD = 80
EXPECTED_RAW_RECORDS = 5_220
EFFECT_THRESHOLD = 0.01
RECOVERY_THRESHOLD = 0.25
RETENTION_THRESHOLD = 0.80
SHAPLEY_ABS_TOLERANCE = 1e-9
SHAPLEY_REL_TOLERANCE = 1e-9
DEVELOPMENT_GENES = ('BRAF', 'SLC25A48')
FORBIDDEN_GENES = ('ELN', 'EIF4A2', 'DMD')
TRACE_ROLES = (
    'reference_baseline',
    'alternate_baseline',
    'reference_into_alternate',
    'alternate_into_alternate_self_control',
    'alternate_into_reference',
    'reference_into_reference_self_control',
)
RUNTIME_PACKAGES = (
    'jax', 'jaxlib', 'jax-cuda12-pjrt', 'jax-cuda12-plugin',
    'nvidia-cublas-cu12', 'nvidia-cuda-runtime-cu12',
    'nvidia-cudnn-cu12', 'nvidia-cusparse-cu12', 'dm-haiku', 'jmp',
    'protobuf', 'numpy', 'orbax-checkpoint', 'grpcio',
)
UPSTREAM_GENERATED_MODULE_NAMES = (
    'alphagenome.protos.dna_model_pb2',
    'alphagenome.protos.dna_model_service_pb2',
    'alphagenome.protos.dna_model_service_pb2_grpc',
    'alphagenome.protos.tensor_pb2',
)
UPSTREAM_GENERATED_OUTPUTS = {
    'alphagenome.protos.dna_model_pb2': {
        'relative_path': 'src/alphagenome/protos/dna_model_pb2.py',
        'sha256': 'd97564536e77ec09bdf144ba1204d4e08f79095fb9ed6c0cba7b065dc6f252ee',
        'size_bytes': 16279,
    },
    'alphagenome.protos.dna_model_service_pb2': {
        'relative_path': 'src/alphagenome/protos/dna_model_service_pb2.py',
        'sha256': '062ed92fd13a0757c3c029275edd698c2327ea56be4a5e074a28d91fa0dabd6a',
        'size_bytes': 10114,
    },
    'alphagenome.protos.dna_model_service_pb2_grpc': {
        'relative_path': 'src/alphagenome/protos/dna_model_service_pb2_grpc.py',
        'sha256': 'db3816f2f0ccfe08d0e6bec0771079bbca6cfc399bd58e79e928cb1264bdaeb3',
        'size_bytes': 16361,
    },
    'alphagenome.protos.tensor_pb2': {
        'relative_path': 'src/alphagenome/protos/tensor_pb2.py',
        'sha256': 'dea7a5207e82601b6763e95ee4b69356345e95bc2de91632928d6161f873cdb8',
        'size_bytes': 3155,
    },
}
UPSTREAM_GENERATED_SOURCE_INPUTS = {
    'hatch_build.py': {
        'sha256': '7e517361e85e8f85ca0c5c8497b9281234f54215475e0e86a7695cdab7945d5b',
        'size_bytes': 1739,
    },
    'pyproject.toml': {
        'sha256': 'e4b06b106b2a4836d1b94d0fb97128ec23a2aeb0b7ebf043aed302a321075777',
        'size_bytes': 3073,
    },
    'src/alphagenome/.gitignore': {
        'sha256': '7405614ddb7a97b8acdf99e14ba9f4adf07f2318480c733733da03f51cd704ec',
        'size_bytes': 358,
    },
    'src/alphagenome/protos/dna_model.proto': {
        'sha256': 'd19a7208ec34953ca021efbff32516f1aa277f0477276f7699d9567fd616329a',
        'size_bytes': 15103,
    },
    'src/alphagenome/protos/dna_model_service.proto': {
        'sha256': 'bdcfea14fb629fb4b6e9304909023a872267a69e5f0e3a23bdb9a769f26dc317',
        'size_bytes': 8472,
    },
    'src/alphagenome/protos/tensor.proto': {
        'sha256': '07779023b2868377cbfc3c2ce96cd266ae425a0a1116aea755691c263d6238f7',
        'size_bytes': 2856,
    },
}
UPSTREAM_GENERATED_HEADERS = {
    'alphagenome.protos.dna_model_pb2': [
        '# -*- coding: utf-8 -*-',
        '# Generated by the protocol buffer compiler.  DO NOT EDIT!',
        '# NO CHECKED-IN PROTOBUF GENCODE',
        '# source: alphagenome/protos/dna_model.proto',
        '# Protobuf Python Version: 5.27.2',
    ],
    'alphagenome.protos.dna_model_service_pb2': [
        '# -*- coding: utf-8 -*-',
        '# Generated by the protocol buffer compiler.  DO NOT EDIT!',
        '# NO CHECKED-IN PROTOBUF GENCODE',
        '# source: alphagenome/protos/dna_model_service.proto',
        '# Protobuf Python Version: 5.27.2',
    ],
    'alphagenome.protos.dna_model_service_pb2_grpc': [
        '# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!',
        "GRPC_GENERATED_VERSION = '1.67.1'",
    ],
    'alphagenome.protos.tensor_pb2': [
        '# -*- coding: utf-8 -*-',
        '# Generated by the protocol buffer compiler.  DO NOT EDIT!',
        '# NO CHECKED-IN PROTOBUF GENCODE',
        '# source: alphagenome/protos/tensor.proto',
        '# Protobuf Python Version: 5.27.2',
    ],
}
_HERE = Path(__file__).resolve().parent
_PROTOCOL_PATH = (
    _HERE / 'v3_wider_mechanism' / 'encoder_skip_localization_protocol_v3_3.md'
)
_SUPERSESSION_PATH = (
    _HERE / 'v3_wider_mechanism' / 'seven_skip_factorial_analysis_plan.md'
)
_CASES_PATH = _HERE / 'superset_graph_v3_2_development_variants.tsv'
_EXONS_PATH = _HERE / 'superset_graph_v3_2_development_exons.tsv'
_PRIOR_V3_2_RUN_DIR = _HERE / 'results' / 'v3_2_development_superset_graph_one_shot'
_PRIOR_V3_2_ANALYSIS_DIR = (
    _HERE / 'results' / 'v3_2_development_superset_graph_analysis'
)
EXPECTED_PRIOR_V3_2_EVIDENCE = {
    'protocol': {
        'path': str(
            (_HERE / 'v3_wider_mechanism' / 'superset_graph_protocol_v3_2.md').resolve()
        ),
        'sha256': '1e87839250e838d1d9aa95162e76a3acb6b1eaab59b481c60dab943b62f6caea',
    },
    'raw_manifest': {
        'path': str((_PRIOR_V3_2_RUN_DIR / 'RAW_MANIFEST.json').resolve()),
        'sha256': '2d63d7dfeaa69e2c1ad8cde731e656e134e37e639023f0745daadb564f17a665',
        'artifact_count': 2660,
        'artifact_tree_sha256': '4171d8aebae7fff3b9981d7ab0dc914c659c6fe2916cb9a48bbee87e205beed8',
    },
    'analysis': {
        'path': str((_PRIOR_V3_2_ANALYSIS_DIR / 'ANALYSIS.json').resolve()),
        'sha256': 'a46c827e16fb4e054e7cf702f7147da70e0a3b35f677b430edf64e9a3055013c',
    },
    'result': {
        'path': str((_PRIOR_V3_2_ANALYSIS_DIR / 'RESULT.md').resolve()),
        'sha256': 'e4c8d45c1b35d8934734c4d8c18bd2ca10b78dd0099a9b37c296e60b15b7e52c',
    },
}


def _tree_digest(paths: Iterable[Path], root: Path) -> str:
  digest = hashlib.sha256()
  root = root.resolve()
  for path in sorted(paths):
    try:
      relative = path.resolve().relative_to(root)
    except ValueError as error:
      raise AnalysisError(f'Artifact escaped tree root: {path}.') from error
    digest.update(str(relative).encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(_sha256(path)))
  return digest.hexdigest()


class AnalysisError(ValueError):
  """A fail-closed v3.3 validation failure."""


def _assert_cpu_only_module_boundary() -> None:
  forbidden = sorted(
      name for name in sys.modules
      if name in {'jax', 'jaxlib', 'alphagenome'}
      or name.startswith(('jax.', 'jaxlib.', 'alphagenome.'))
      or name.startswith('alphagenome_research.model')
  )
  if forbidden:
    raise AnalysisError(
        f'Offline analyzer process imported forbidden model/JAX modules: {forbidden}.'
    )


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run-dir', type=Path, required=True)
  parser.add_argument('--output-json', type=Path, required=True)
  parser.add_argument('--output-markdown', type=Path, required=True)
  return parser.parse_args()


def _guard_path(path: Path) -> None:
  resolved = path.resolve()
  for part in resolved.parts:
    lowered = part.lower()
    if 'confirm' in lowered or any(gene.lower() in lowered for gene in FORBIDDEN_GENES):
      raise AnalysisError(f'Refusing forbidden confirmation path: {path}.')


def _sha256(path: Path) -> str:
  _guard_path(path)
  digest = hashlib.sha256()
  try:
    with path.open('rb') as handle:
      for block in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(block)
  except OSError as error:
    raise AnalysisError(f'Cannot hash required file {path}.') from error
  return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
  _guard_path(path)
  try:
    value = json.loads(path.read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError) as error:
    raise AnalysisError(f'Cannot read JSON artifact {path}.') from error
  if not isinstance(value, dict):
    raise AnalysisError(f'JSON artifact must be an object: {path}.')
  _reject_forbidden_content(value, str(path))
  return value


def _reject_forbidden_content(value: Any, label: str) -> None:
  """Rejects confirmation names/genes even when hidden in metadata values."""
  if isinstance(value, Mapping):
    for key, item in value.items():
      key_text = str(key).lower()
      if isinstance(item, str) and any(
          token in key_text for token in ('path', 'dir', 'file', 'output')
      ):
        _guard_path(Path(item))
      _reject_forbidden_content(item, label)
  elif isinstance(value, list):
    for item in value:
      _reject_forbidden_content(item, label)
  elif isinstance(value, str):
    words = {
        ''.join(character for character in word.upper() if character.isalnum())
        for word in value.replace('/', ' ').replace('_', ' ').split()
    }
    if any(gene in words for gene in FORBIDDEN_GENES):
      raise AnalysisError(f'{label} exposes a forbidden confirmation name.')


def _is_sha256(value: Any) -> bool:
  return (
      isinstance(value, str)
      and len(value) == 64
      and all(character in '0123456789abcdef' for character in value)
  )


def _finite(value: Any, label: str) -> float:
  if isinstance(value, bool):
    raise AnalysisError(f'{label} must be numeric, not boolean.')
  try:
    result = float(value)
  except (TypeError, ValueError) as error:
    raise AnalysisError(f'{label} must be numeric.') from error
  if not math.isfinite(result):
    raise AnalysisError(f'{label} is non-finite.')
  return result


def _f32(value: Any, label: str = 'value') -> float:
  result = _finite(value, label)
  try:
    return struct.unpack('<f', struct.pack('<f', result))[0]
  except OverflowError as error:
    raise AnalysisError(f'{label} is outside float32 range.') from error


def _same_f32(observed: Any, expected: Any, label: str) -> None:
  observed_value = _finite(observed, label)
  expected_value = _f32(expected, f'{label}.expected')
  if observed_value != _f32(observed_value, label):
    raise AnalysisError(f'{label} is not an exact float32 value.')
  if struct.pack('<f', observed_value) != struct.pack('<f', expected_value):
    raise AnalysisError(f'{label} differs from recomputed float32 evidence.')


def _readout(
    record: Mapping[str, Any], field: str, label: str, *, rows: int
) -> dict[str, Any]:
  """Independently reconstructs margins/totals/means from selected logits."""
  value = record.get(field)
  if not isinstance(value, Mapping):
    raise AnalysisError(f'{label}.{field} raw endpoint evidence is absent.')
  if value.get('endpoint_axis') != ['acceptor', 'donor']:
    raise AnalysisError(f'{label}.{field} endpoint axis changed.')
  if value.get('selected_logit_axis') != ['relevant_class', 'padding_class']:
    raise AnalysisError(f'{label}.{field} selected-logit axis changed.')
  if value.get('num_values') != 2:
    raise AnalysisError(f'{label}.{field} must reduce exactly two endpoints.')
  logits = value.get('selected_logits')
  margins = value.get('endpoint_margins')
  totals = value.get('totals')
  means = value.get('means')
  if not isinstance(logits, list) or len(logits) != rows:
    raise AnalysisError(f'{label}.{field}.selected_logits row count changed.')
  if not isinstance(margins, list) or len(margins) != rows:
    raise AnalysisError(f'{label}.{field}.endpoint_margins row count changed.')
  if not isinstance(totals, list) or len(totals) != rows:
    raise AnalysisError(f'{label}.{field}.totals row count changed.')
  if not isinstance(means, list) or len(means) != rows:
    raise AnalysisError(f'{label}.{field}.means row count changed.')
  clean_logits, clean_margins, clean_totals, clean_means = [], [], [], []
  for row in range(rows):
    row_logits, row_margins = logits[row], margins[row]
    if (
        not isinstance(row_logits, list) or len(row_logits) != 2
        or any(not isinstance(endpoint, list) or len(endpoint) != 2
               for endpoint in row_logits)
        or not isinstance(row_margins, list) or len(row_margins) != 2
    ):
      raise AnalysisError(f'{label}.{field} has a malformed endpoint tensor.')
    clean_logit_row, clean_margin_row = [], []
    for endpoint in range(2):
      relevant = _f32(row_logits[endpoint][0], f'{label}.relevant')
      padding = _f32(row_logits[endpoint][1], f'{label}.padding')
      margin = _f32(relevant - padding)
      _same_f32(row_margins[endpoint], margin, f'{label}.margin[{row},{endpoint}]')
      clean_logit_row.append([relevant, padding])
      clean_margin_row.append(margin)
    total = _f32(clean_margin_row[0] + clean_margin_row[1])
    mean = _f32(total / 2.0)
    _same_f32(totals[row], total, f'{label}.total[{row}]')
    _same_f32(means[row], mean, f'{label}.mean[{row}]')
    clean_logits.append(clean_logit_row)
    clean_margins.append(clean_margin_row)
    clean_totals.append(total)
    clean_means.append(mean)
  return {
      'selected_logits': clean_logits,
      'endpoint_margins': clean_margins,
      'totals': clean_totals,
      'means': clean_means,
  }


def _row_bytes(readout: Mapping[str, Any], row: int) -> bytes:
  values: list[float] = []
  for endpoint in readout['selected_logits'][row]:
    values.extend(endpoint)
  values.extend(readout['endpoint_margins'][row])
  values.extend((readout['totals'][row], readout['means'][row]))
  return b''.join(struct.pack('<f', value) for value in values)


def _require_repeat(
    record: Mapping[str, Any], field: str, repeat_field: str, label: str, *, rows: int
) -> dict[str, Any]:
  first = _readout(record, field, label, rows=rows)
  repeat = _readout(record, repeat_field, label, rows=rows)
  if any(_row_bytes(first, row) != _row_bytes(repeat, row) for row in range(rows)):
    raise AnalysisError(f'{label} endpoint repeat is not bit-exact.')
  return first


def _validate_trace_fingerprint(value: Any, label: str) -> None:
  if not isinstance(value, Mapping) or set(value) != {'sha256', 'leaves'}:
    raise AnalysisError(f'{label} trace-fingerprint schema changed.')
  if not _is_sha256(value.get('sha256')) or not isinstance(value.get('leaves'), list):
    raise AnalysisError(f'{label} trace-fingerprint content is malformed.')
  for index, leaf in enumerate(value['leaves']):
    if not isinstance(leaf, Mapping) or set(leaf) != {'shape', 'dtype'}:
      raise AnalysisError(f'{label} trace leaf {index} schema changed.')
    shape = leaf.get('shape')
    if (
        not isinstance(shape, list)
        or any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in shape)
        or not isinstance(leaf.get('dtype'), str) or not leaf['dtype']
    ):
      raise AnalysisError(f'{label} trace leaf {index} is malformed.')


def _metrics(readout: Mapping[str, Any]) -> dict[str, Any]:
  means = readout['means']
  if len(means) < 6:
    raise AnalysisError('Scientific recovery requires six recipient rows.')
  ref, alt, ref_alt, alt_self, alt_ref, ref_self = means[:6]
  denominator_forward = ref - alt
  denominator_reverse = alt - ref
  movement_forward = ref_alt - alt_self
  movement_reverse = alt_ref - ref_self
  result: dict[str, Any] = {
      'baseline_delta': alt - ref,
      'movements': {
          'reference_into_alternate': movement_forward,
          'alternate_into_reference': movement_reverse,
      },
      'mean_absolute_movement': statistics.fmean(
          (abs(movement_forward), abs(movement_reverse))
      ),
  }
  if denominator_forward == 0.0:
    result['recoveries'] = None
    result['B'] = None
  else:
    forward = movement_forward / denominator_forward
    reverse = movement_reverse / denominator_reverse
    result['recoveries'] = {
        'reference_into_alternate': forward,
        'alternate_into_reference': reverse,
    }
    result['B'] = min(forward, reverse)
  return result


def _raw_movements(readout: Mapping[str, Any]) -> dict[str, Any]:
  """Computes only OOD-permitted raw movement, never recovery or B."""
  means = readout['means']
  if len(means) < 6:
    raise AnalysisError('Raw directional movement requires six recipient rows.')
  movement_forward = _f32(means[2] - means[3], 'raw_movement.forward')
  movement_reverse = _f32(means[4] - means[5], 'raw_movement.reverse')
  return {
      'movements': {
          'reference_into_alternate': movement_forward,
          'alternate_into_reference': movement_reverse,
      },
      'mean_absolute_movement': statistics.fmean(
          (abs(movement_forward), abs(movement_reverse))
      ),
  }


def _subset_members(mask: int, players: Sequence[str]) -> list[str]:
  return [player for index, player in enumerate(players) if mask & (1 << index)]


def _require_cube(values: Mapping[int, float], n: int, label: str) -> None:
  expected = set(range(1 << n))
  if set(values) != expected:
    missing = sorted(expected - set(values))[:5]
    extra = sorted(set(values) - expected)[:5]
    raise AnalysisError(f'{label} cube is incomplete (missing={missing}, extra={extra}).')
  for mask, value in values.items():
    _finite(value, f'{label}[{mask}]')


def exact_shapley(values: Mapping[int, float], players: Sequence[str]) -> dict[str, float]:
  """Returns exact-enumeration Shapley values for a complete Boolean cube."""
  n = len(players)
  _require_cube(values, n, 'Shapley')
  denominator = math.factorial(n)
  result = {}
  for index, player in enumerate(players):
    bit = 1 << index
    total = 0.0
    for mask in range(1 << n):
      if mask & bit:
        continue
      size = mask.bit_count()
      weight = math.factorial(size) * math.factorial(n - size - 1) / denominator
      total += weight * (values[mask | bit] - values[mask])
    result[player] = total
  expected = values[(1 << n) - 1] - values[0]
  if not math.isclose(
      math.fsum(result.values()), expected,
      rel_tol=SHAPLEY_REL_TOLERANCE, abs_tol=SHAPLEY_ABS_TOLERANCE,
  ):
    raise AnalysisError('Shapley efficiency failed the frozen tolerance.')
  return result


def pairwise_shapley_interactions(
    values: Mapping[int, float], players: Sequence[str]
) -> dict[str, float]:
  n = len(players)
  _require_cube(values, n, 'interaction')
  if n < 2:
    return {}
  denominator = math.factorial(n - 1)
  result = {}
  for i in range(n):
    for j in range(i + 1, n):
      bits = (1 << i) | (1 << j)
      total = 0.0
      for mask in range(1 << n):
        if mask & bits:
          continue
        size = mask.bit_count()
        weight = math.factorial(size) * math.factorial(n - size - 2) / denominator
        second = (
            values[mask | bits] - values[mask | (1 << i)]
            - values[mask | (1 << j)] + values[mask]
        )
        total += weight * second
      result[f'{players[i]}:{players[j]}'] = total
  return result


def harsanyi_dividends(
    values: Mapping[int, float], players: Sequence[str]
) -> tuple[dict[str, float], dict[str, float]]:
  n = len(players)
  _require_cube(values, n, 'Harsanyi')
  dividends: dict[str, float] = {}
  mass = {str(order): 0.0 for order in range(1, n + 1)}
  for mask in range(1 << n):
    total = 0.0
    submask = mask
    while True:
      sign = -1.0 if (mask.bit_count() - submask.bit_count()) % 2 else 1.0
      total += sign * values[submask]
      if submask == 0:
        break
      submask = (submask - 1) & mask
    key = format(mask, f'0{n}b')
    dividends[key] = total
    if mask:
      mass[str(mask.bit_count())] += abs(total)
  return dividends, mass


def _direction_values(
    cube: Mapping[int, Mapping[str, Any]], direction: str
) -> dict[int, float]:
  if direction not in ('reference_into_alternate', 'alternate_into_reference'):
    raise AnalysisError(f'Unknown direction {direction}.')
  return {identifier: row['movements'][direction] for identifier, row in cube.items()}


def _normalized(values: Mapping[str, float], denominator: float) -> dict[str, float] | None:
  if denominator == 0.0:
    return None
  return {key: value / denominator for key, value in values.items()}


def decompose_variant(cube: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
  """Computes the three frozen decompositions for one validated 256 cube."""
  if set(cube) != set(range(256)):
    raise AnalysisError('Variant coalition cube must contain IDs 0--255 exactly.')
  result: dict[str, Any] = {}
  for direction in ('reference_into_alternate', 'alternate_into_reference'):
    all_values = _direction_values(cube, direction)
    natural = {mask: all_values[mask] for mask in range(128)}
    donor = {mask: all_values[128 + mask] for mask in range(128)}
    # The eight-player bit order is (T,E64,...,E1), whereas coalition IDs store
    # E in bits 0..6 and T in bit 7.  Remap rather than silently reusing IDs.
    eight = {}
    for mask8 in range(256):
      t = mask8 & 1
      e_mask = mask8 >> 1
      eight[mask8] = all_values[128 * t + e_mask]
    denominator = (
        cube[0]['baseline_delta'] * -1.0
        if direction == 'reference_into_alternate'
        else cube[0]['baseline_delta']
    )
    views = {}
    for name, values, players in (
        ('T_natural', natural, PLAYERS_E),
        ('T_donor', donor, PLAYERS_E),
        ('joint_8_player', eight, PLAYERS_8),
    ):
      phi = exact_shapley(values, players)
      interactions = pairwise_shapley_interactions(values, players)
      dividends, mass = harsanyi_dividends(values, players)
      views[name] = {
          'player_order': list(players),
          'raw_shapley': phi,
          'normalized_shapley': _normalized(phi, denominator),
          'pairwise_interactions': interactions,
          'normalized_pairwise_interactions': _normalized(interactions, denominator),
          'harsanyi_dividends': dividends,
          'absolute_harsanyi_mass_by_order': mass,
          'efficiency': {
              'sum_phi': math.fsum(phi.values()),
              'grand_difference': values[(1 << len(players)) - 1] - values[0],
              'absolute_tolerance': SHAPLEY_ABS_TOLERANCE,
              'relative_tolerance': SHAPLEY_REL_TOLERANCE,
          },
      }
    result[direction] = views
  return result


def _median(values: Iterable[float], label: str) -> float:
  clean = [_finite(value, label) for value in values]
  if not clean:
    raise AnalysisError(f'{label} has an empty denominator.')
  return statistics.median(clean)


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
  if len(x) != len(y) or not x:
    raise AnalysisError('Spearman vectors must have equal nonzero length.')

  def ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
      end = cursor + 1
      while end < len(order) and values[order[end]] == values[order[cursor]]:
        end += 1
      rank = (cursor + 1 + end) / 2.0
      for position in order[cursor:end]:
        result[position] = rank
      cursor = end
    return result

  rx, ry = ranks(x), ranks(y)
  mean_x, mean_y = statistics.fmean(rx), statistics.fmean(ry)
  centered_x = [value - mean_x for value in rx]
  centered_y = [value - mean_y for value in ry]
  denominator = math.sqrt(
      math.fsum(value * value for value in centered_x)
      * math.fsum(value * value for value in centered_y)
  )
  if denominator == 0.0:
    return None
  return math.fsum(a * b for a, b in zip(centered_x, centered_y, strict=True)) / denominator


def _candidate_table(
    cubes: Mapping[int, Mapping[int, Mapping[str, Any]]],
    cases: Mapping[int, Mapping[str, Any]], *, t: int,
) -> tuple[list[dict[str, Any]], str | None]:
  baseline_id = 127 if t == 0 else 255
  all_e = {}
  for gene, orders in (
      ('BRAF', range(0, 6)), ('SLC25A48', range(10, 16))
  ):
    all_e[gene] = _median(
        (cubes[order][baseline_id]['B'] for order in orders),
        f'{gene}.all_E_B',
    )
    if not math.isfinite(all_e[gene]) or all_e[gene] <= 0.0:
      return [], f'{gene} all-E denominator is nonpositive/non-finite'
  rows = []
  for e_mask in range(1, 128):
    coalition_id = 128 * t + e_mask
    medians = {}
    retention = {}
    for gene, orders in (
        ('BRAF', range(0, 6)), ('SLC25A48', range(10, 16))
    ):
      medians[gene] = _median(
          (cubes[order][coalition_id]['B'] for order in orders),
          f'{gene}.coalition[{coalition_id}].B',
      )
      retention[gene] = medians[gene] / all_e[gene]
    available = all(
        medians[gene] >= RECOVERY_THRESHOLD
        and retention[gene] >= RETENTION_THRESHOLD
        for gene in DEVELOPMENT_GENES
    )
    rows.append({
        'coalition_id': coalition_id,
        't': t,
        'e_mask': e_mask,
        'e_bits': format(e_mask, '07b'),
        'enabled_skips': _subset_members(e_mask, PLAYERS_E),
        'cardinality': e_mask.bit_count(),
        'median_B': medians,
        'all_E_median_B': dict(all_e),
        'retention': retention,
        'minimum_exon_median_B': min(medians.values()),
        'available': available,
    })
  return rows, None


def _select_candidate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
  available = [dict(row) for row in rows if row['available']]
  if not available:
    return None
  available.sort(key=lambda row: (
      row['cardinality'], -row['minimum_exon_median_B'], row['e_mask']
  ))
  return available[0]


def _neutral_alignment(
    candidate_id: int,
    cubes: Mapping[int, Mapping[int, Mapping[str, Any]]],
    cases: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
  result = {}
  passes = True
  for gene, effects, neutrals in (
      ('BRAF', range(0, 6), range(6, 10)),
      ('SLC25A48', range(10, 16), range(16, 20)),
  ):
    effect_values = [cubes[order][candidate_id]['mean_absolute_movement'] for order in effects]
    neutral_values = [cubes[order][candidate_id]['mean_absolute_movement'] for order in neutrals]
    effect_median = _median(effect_values, f'{gene}.effect_movement')
    neutral_median = _median(neutral_values, f'{gene}.neutral_movement')
    gene_pass = effect_median > neutral_median
    passes = passes and gene_pass
    all_orders = list(effects) + list(neutrals)
    signed_movements = [
        _oriented_alt_movement(cubes[order][candidate_id]) for order in all_orders
    ]
    experimental = [_finite(cases[order]['delta_logit'], 'delta_logit') for order in all_orders]
    result[gene] = {
        'effect_median_absolute_movement': effect_median,
        'neutral_median_absolute_movement': neutral_median,
        'effect_minus_neutral': effect_median - neutral_median,
        'passes': gene_pass,
        'rank_correlation_with_experimental_delta_logit': _spearman(
            signed_movements, experimental
        ),
        'n_effects': len(effect_values),
        'n_neutrals': len(neutral_values),
    }
  return {'passes_both_exons': passes, 'by_exon': result}


def _oriented_alt_movement(metrics: Mapping[str, Any]) -> float:
  """Orients reciprocal patches to the ALT-minus-REF biological direction.

  REF->ALT moves an ALT recipient toward REF and therefore has the opposite
  sign from ALT-REF.  ALT->REF already has the ALT-REF sign.  Averaging the
  two un-oriented movements would incorrectly cancel a perfect rescue.
  """
  movements = metrics.get('movements')
  if not isinstance(movements, Mapping):
    raise AnalysisError('Directional movement evidence is missing.')
  forward = _finite(
      movements.get('reference_into_alternate'), 'reference_into_alternate'
  )
  reverse = _finite(
      movements.get('alternate_into_reference'), 'alternate_into_reference'
  )
  return statistics.fmean((-forward, reverse))


def _median_mapping(rows: Sequence[Mapping[str, float]], keys: Sequence[str]) -> dict[str, float]:
  return {
      key: _median((row[key] for row in rows), f'median.{key}') for key in keys
  }


def _per_exon_summaries(
    cubes: Mapping[int, Mapping[int, Mapping[str, Any]]],
    decompositions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
  """Returns protocol-required medians across all six effects per exon."""
  result = {}
  for gene, orders in (
      ('BRAF', tuple(range(0, 6))),
      ('SLC25A48', tuple(range(10, 16))),
  ):
    exon: dict[str, Any] = {'effect_orders': list(orders), 'directions': {}}
    for direction in ('reference_into_alternate', 'alternate_into_reference'):
      direction_result = {}
      for view, players in (
          ('T_natural', PLAYERS_E),
          ('T_donor', PLAYERS_E),
          ('joint_8_player', PLAYERS_8),
      ):
        raw_rows = [
            decompositions[str(order)][direction][view]['raw_shapley']
            for order in orders
        ]
        normalized_rows = [
            decompositions[str(order)][direction][view]['normalized_shapley']
            for order in orders
        ]
        interaction_keys = list(
            decompositions[str(orders[0])][direction][view][
                'pairwise_interactions'
            ]
        )
        interaction_rows = [
            decompositions[str(order)][direction][view]['pairwise_interactions']
            for order in orders
        ]
        normalized_interaction_rows = [
            decompositions[str(order)][direction][view][
                'normalized_pairwise_interactions'
            ]
            for order in orders
        ]
        direction_result[view] = {
            'median_raw_shapley': _median_mapping(raw_rows, players),
            'median_normalized_shapley': (
                None if any(row is None for row in normalized_rows)
                else _median_mapping(normalized_rows, players)  # type: ignore[arg-type]
            ),
            'median_pairwise_interactions': _median_mapping(
                interaction_rows, interaction_keys
            ),
            'median_normalized_pairwise_interactions': (
                None if any(row is None for row in normalized_interaction_rows)
                else _median_mapping(  # type: ignore[arg-type]
                    normalized_interaction_rows, interaction_keys
                )
            ),
            'median_absolute_harsanyi_mass_by_order': {
                str(order_index): _median(
                    (
                        decompositions[str(order)][direction][view][
                            'absolute_harsanyi_mass_by_order'
                        ][str(order_index)]
                        for order in orders
                    ),
                    f'{gene}.{direction}.{view}.harsanyi_order_{order_index}',
                )
                for order_index in range(1, len(players) + 1)
            },
        }
      exon['directions'][direction] = direction_result

    context_views = {}
    for t, context in ((0, 'T_natural'), (1, 'T_donor')):
      offset = 128 * t
      singleton = {}
      leave_one_out = {}
      for index, player in enumerate(PLAYERS_E):
        singleton_id = offset + (1 << index)
        leave_out_id = offset + (127 ^ (1 << index))
        full_id = offset + 127
        singleton[player] = _median(
            (cubes[order][singleton_id]['B'] for order in orders),
            f'{gene}.{context}.{player}.singleton_B',
        )
        leave_one_out[player] = _median(
            (
                cubes[order][full_id]['B'] - cubes[order][leave_out_id]['B']
                for order in orders
            ),
            f'{gene}.{context}.{player}.leave_one_out',
        )
      cumulative = {}
      for name, indices in (
          ('coarse_to_fine', tuple(range(7))),
          ('fine_to_coarse', tuple(reversed(range(7)))),
      ):
        mask = 0
        path = []
        for index in indices:
          mask |= 1 << index
          identifier = offset + mask
          path.append({
              'added': PLAYERS_E[index],
              'e_mask': mask,
              'coalition_id': identifier,
              'median_B': _median(
                  (cubes[order][identifier]['B'] for order in orders),
                  f'{gene}.{context}.{name}.{mask}',
              ),
          })
        cumulative[name] = path
      context_views[context] = {
          'median_singleton_B': singleton,
          'median_leave_one_out_decrement': leave_one_out,
          'cumulative_paths': cumulative,
          'median_full_coalition_B': _median(
              (cubes[order][offset + 127]['B'] for order in orders),
              f'{gene}.{context}.full_B',
          ),
      }
    exon['derived_coalition_views'] = context_views
    result[gene] = exon
  return result


def _all_behavior_controls(
    cubes: Mapping[int, Mapping[int, Mapping[str, Any]]],
    cases: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
  """Reports every coalition's effect/neutral descriptive comparison."""
  rows = []
  for coalition_id in range(256):
    by_exon = {}
    for gene, effects, neutrals in (
        ('BRAF', tuple(range(0, 6)), tuple(range(6, 10))),
        ('SLC25A48', tuple(range(10, 16)), tuple(range(16, 20))),
    ):
      effect_values = [
          cubes[order][coalition_id]['mean_absolute_movement'] for order in effects
      ]
      neutral_values = [
          cubes[order][coalition_id]['mean_absolute_movement'] for order in neutrals
      ]
      orders = effects + neutrals
      signed_movement = [
          _oriented_alt_movement(cubes[order][coalition_id]) for order in orders
      ]
      delta_logit = [_finite(cases[order]['delta_logit'], 'delta_logit') for order in orders]
      signed_agreement = [
          (movement > 0) == (experimental > 0)
          for movement, experimental in zip(signed_movement, delta_logit, strict=True)
          if movement != 0.0 and experimental != 0.0
      ]
      effect_median = _median(effect_values, f'{gene}.effect')
      neutral_median = _median(neutral_values, f'{gene}.neutral')
      by_exon[gene] = {
          'effect_median_absolute_movement': effect_median,
          'neutral_median_absolute_movement': neutral_median,
          'effect_minus_neutral': effect_median - neutral_median,
          'signed_agreement_count': sum(signed_agreement),
          'signed_agreement_denominator': len(signed_agreement),
          'rank_correlation_with_experimental_delta_logit': _spearman(
              signed_movement, delta_logit
          ),
      }
    rows.append({'coalition_id': coalition_id, 'by_exon': by_exon})
  return rows


def summarize_cubes(
    cubes: Mapping[int, Mapping[int, Mapping[str, Any]]],
    cases: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
  """Analyzes 20 already-evidence-validated cubes and applies exact gates."""
  if set(cubes) != set(ALL_ORDERS) or set(cases) != set(ALL_ORDERS):
    raise AnalysisError('Analysis requires the exact frozen 20-case cohort.')
  for order in ALL_ORDERS:
    if set(cubes[order]) != set(range(256)):
      raise AnalysisError(f'Order {order} does not have a complete 256 cube.')
    for coalition_id, metrics in cubes[order].items():
      if metrics.get('B') is None:
        if order in EFFECT_ORDERS:
          raise AnalysisError(f'Effect order {order} has undefined recovery.')
      elif not math.isfinite(_finite(metrics['B'], 'B')):
        raise AnalysisError(f'Order {order} coalition {coalition_id} is invalid.')
  skip_rows, skip_failure = _candidate_table(cubes, cases, t=0)
  skip = None if skip_failure else _select_candidate(skip_rows)
  donor_rows: list[dict[str, Any]] = []
  donor_failure = None
  donor = None
  if skip is None:
    donor_rows, donor_failure = _candidate_table(cubes, cases, t=1)
    donor = None if donor_failure else _select_candidate(donor_rows)
  selected = skip if skip is not None else donor
  if selected is None:
    decision = 'no_resolution_coalition_passed'
    alignment = None
  else:
    decision = 'skip_only_route' if selected['t'] == 0 else 'T_dependent_joint_route'
    alignment = _neutral_alignment(selected['coalition_id'], cubes, cases)
    if not alignment['passes_both_exons']:
      decision += '_not_biologically_aligned'
  per_variant = {
      str(order): decompose_variant(cubes[order]) for order in ALL_ORDERS
  }
  return {
      'decision': decision,
      'nomination': selected,
      'biological_alignment': alignment,
      'skip_only_family': {
          'failure': skip_failure, 'candidates': skip_rows,
      },
      'T_dependent_family': {
          'evaluated': skip is None,
          'failure': donor_failure,
          'candidates': donor_rows,
      },
      'per_variant_decomposition': per_variant,
      'per_exon_summaries': _per_exon_summaries(cubes, per_variant),
      'coalition_behavior_controls': _all_behavior_controls(cubes, cases),
  }


def _load_cases() -> dict[int, dict[str, Any]]:
  """Loads only the two committed development projection files."""
  _guard_path(_CASES_PATH)
  _guard_path(_EXONS_PATH)
  exons = {}
  with _EXONS_PATH.open('r', encoding='utf-8', newline='') as handle:
    for row in csv.DictReader(handle, delimiter='\t'):
      if row['gene'] not in DEVELOPMENT_GENES:
        raise AnalysisError('Development exon projection contains another gene.')
      exons[row['ensembl_exon_id']] = row
  rows = []
  with _CASES_PATH.open('r', encoding='utf-8', newline='') as handle:
    rows = list(csv.DictReader(handle, delimiter='\t'))
  if len(rows) != 20 or len(exons) != 2:
    raise AnalysisError('Development projections are not exactly 20 rows/two exons.')
  cases = {}
  for order, row in enumerate(rows):
    exon = exons.get(row['ensembl_exon_id'])
    if exon is None or exon['gene'] not in DEVELOPMENT_GENES:
      raise AnalysisError('Development case references an unfrozen exon.')
    cases[order] = {
        'order': order,
        'selection_version': row['selection_version'],
        'selection_class': row['selection_class'],
        'observed_effect_sign': row['observed_effect_sign'].strip().lower(),
        'variant_id': row['variant_id'],
        'gene': exon['gene'],
        'exon_id': exon['exon_id'],
        'ensembl_exon_id': exon['ensembl_exon_id'],
        'chromosome': (
            exon['chromosome'] if exon['chromosome'].startswith('chr')
            else f"chr{exon['chromosome']}"
        ),
        'strand': exon['strand'],
        'exon_start_1based': int(exon['exon_start_1based']),
        'exon_end_1based': int(exon['exon_end_1based']),
        'position_1based': int(row['position_1based']),
        'reference_bases': row['reference_bases'].upper(),
        'alternate_bases': row['alternate_bases'].upper(),
        'region': row['region'],
        'mut_type': row['mut_type'],
        'delta_psi': float(row['delta_psi']),
        'delta_logit': float(row['delta_logit']),
    }
  if tuple(cases) != ALL_ORDERS:
    raise AnalysisError('Development order differs from 0--19.')
  return cases


def _slug(value: str) -> str:
  return ''.join(character if character.isalnum() else '_' for character in value).strip('_')


def _expected_execution_order() -> list[tuple[str, int, int | None]]:
  """Returns the exact dependency-first record order frozen at 6c77e3f."""
  order: list[tuple[str, int, int | None]] = [
      ('identity', case_order, None) for case_order in ALL_ORDERS
  ]
  order.extend(
      ('coalition', case_order, coalition_id)
      for case_order in ALL_ORDERS for coalition_id in (0, 255)
  )
  order.extend(
      ('coalition', case_order, coalition_id)
      for case_order in EFFECT_ORDERS for coalition_id in range(1, 255)
  )
  order.extend(
      ('coalition', case_order, coalition_id)
      for case_order in NEUTRAL_ORDERS for coalition_id in range(1, 255)
  )
  order.extend(
      ('ood', case_order, coalition_id)
      for case_order in ALL_ORDERS for coalition_id in OOD_ANCHORS
  )
  if len(order) != EXPECTED_RAW_RECORDS or len(set(order)) != len(order):
    raise AssertionError('Internal v3.3 execution-order contract is invalid.')
  return order


def _artifact_relative(
    family: str, case: Mapping[str, Any], coalition_id: int | None
) -> str:
  case_key = f"{case['order']:03d}_{_slug(str(case['variant_id']))}"
  if family == 'identity':
    return f'raw/identity/{case_key}.json'
  if coalition_id is None:
    raise AssertionError('Coalition/OOD path requires an identifier.')
  directory = 'coalitions' if family == 'coalition' else 'ood_anchors'
  return f'raw/{directory}/{case_key}/{coalition_id:03d}.json'


def _validate_case_record(
    observed: Any, expected: Mapping[str, Any], label: str
) -> None:
  if not isinstance(observed, Mapping):
    raise AnalysisError(f'{label} case metadata is missing.')
  if dict(observed) != dict(expected):
    raise AnalysisError(f'{label} case differs from the frozen projection.')
  if observed.get('gene') not in DEVELOPMENT_GENES:
    raise AnalysisError(f'{label} includes a non-development gene.')


def _validate_canonical_target(
    record: Mapping[str, Any], case: Mapping[str, Any], label: str
) -> None:
  interval = record.get('interval')
  target = record.get('canonical_target')
  if not isinstance(interval, Mapping) or not isinstance(target, Mapping):
    raise AnalysisError(f'{label} interval/canonical target metadata is absent.')
  start, end = interval.get('start_0based'), interval.get('end_0based_exclusive')
  center = (case['exon_start_1based'] + case['exon_end_1based']) // 2
  expected_start = center - 1 - 8192
  if (
      interval.get('chromosome') != case['chromosome']
      or not isinstance(start, int) or not isinstance(end, int)
      or (start, end) != (expected_start, expected_start + 16_384)
  ):
    raise AnalysisError(f'{label} context is not the frozen unshifted 16,384 bp.')
  if case['strand'] == '+':
    expected = (
        ('acceptor', case['exon_start_1based'], 1),
        ('donor', case['exon_end_1based'], 0),
    )
  elif case['strand'] == '-':
    expected = (
        ('acceptor', case['exon_end_1based'], 3),
        ('donor', case['exon_start_1based'], 2),
    )
  else:
    raise AnalysisError(f'{label} frozen strand is invalid.')
  endpoints = target.get('endpoints')
  if not isinstance(endpoints, list) or len(endpoints) != 2:
    raise AnalysisError(f'{label} canonical target must have exactly two endpoints.')
  for observed, (role, position, track) in zip(endpoints, expected, strict=True):
    if not isinstance(observed, Mapping) or (
        observed.get('role'), observed.get('position_1based'),
        observed.get('position_index'), observed.get('track_index'),
    ) != (role, position, position - 1 - start, track):
      raise AnalysisError(f'{label} strand-aware endpoint mapping changed.')
  if target.get('padding_track_index') != 4:
    raise AnalysisError(f'{label} padding-class index changed.')
  _validate_resolved_position_sets(record.get('resolved_position_sets'), case, start, label)


def _validate_resolved_position_sets(
    observed: Any, case: Mapping[str, Any], interval_start: int, label: str
) -> None:
  """Reconstructs the exact V/A/D/S and width-matched trace metadata."""
  def token(position_1based: int) -> int:
    return (position_1based - 1 - interval_start) // 128

  if case['strand'] == '+':
    acceptor_position, donor_position = (
        case['exon_start_1based'], case['exon_end_1based']
    )
  else:
    acceptor_position, donor_position = (
        case['exon_end_1based'], case['exon_start_1based']
    )
  variant = (token(case['position_1based']),)
  acceptor = (token(acceptor_position),)
  donor = (token(donor_position),)
  splice = tuple(dict.fromkeys(variant + acceptor + donor))
  candidates = (('V', variant), ('A', acceptor), ('D', donor), ('S', splice))
  occupied = set(splice)

  def shifted(tokens: tuple[int, ...], direction: int) -> tuple[int, ...]:
    for distance in range(4, 128):
      result = tuple(value + direction * distance for value in tokens)
      if min(result) >= 0 and max(result) < 128 and occupied.isdisjoint(result):
        return result
    raise AnalysisError(f'{label} has no valid frozen trace-position control.')

  specifications: list[tuple[str, tuple[int, ...], str, str | None]] = [
      (name, tokens, 'candidate', None) for name, tokens in candidates
  ]
  for name, tokens in candidates:
    specifications.extend((
        (f'{name}_control_upstream', shifted(tokens, -1),
         'width_matched_control', name),
        (f'{name}_control_downstream', shifted(tokens, 1),
         'width_matched_control', name),
    ))
  unique_tokens = tuple(dict.fromkeys(
      value for _, tokens, _, _ in specifications for value in tokens
  ))
  slot = {value: index for index, value in enumerate(unique_tokens)}
  expected = [{
      'name': name,
      'tokens': list(tokens),
      'slots': [slot[value] for value in tokens],
      'role': role,
      'matched_candidate': matched,
      'genomic_intervals': [
          [interval_start + 128 * value, interval_start + 128 * (value + 1)]
          for value in tokens
      ],
  } for name, tokens, role, matched in specifications]
  if observed != expected:
    raise AnalysisError(f'{label} resolved trace-position sets changed.')


def _require_common_linkage(
    record: Mapping[str, Any], *, family: str, freeze_sha: str,
    execution_index: int, label: str,
) -> None:
  expected = {
      'status': 'complete',
      'family': family,
      'script_version': SOURCE_SCRIPT_VERSION,
      'protocol_sha256': PROTOCOL_SHA256,
      'supersession_sha256': SUPERSESSION_SHA256,
      'supersession_commit': 'c64def4',
      'freeze_sha256': freeze_sha,
      'execution_index': execution_index,
  }
  for key, value in expected.items():
    if record.get(key) != value:
      raise AnalysisError(f'{label}.{key} differs from the frozen contract.')
  if record.get('failure') is not None:
    raise AnalysisError(f'{label} records a failure.')
  _finite(record.get('created_at_unix_s'), f'{label}.created_at_unix_s')


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
  if set(value) != expected:
    raise AnalysisError(
        f'{label} schema changed (missing={sorted(expected - set(value))}, '
        f'extra={sorted(set(value) - expected)}).'
    )


def _require_seconds(value: Any, names: set[str], label: str) -> None:
  if not isinstance(value, Mapping) or set(value) != names:
    raise AnalysisError(f'{label} timing schema changed.')
  for name in names:
    if _finite(value[name], f'{label}.{name}') < 0.0:
      raise AnalysisError(f'{label}.{name} is negative.')


def _require_checks(checks: Any, fields: Sequence[str], label: str) -> Mapping[str, Any]:
  if not isinstance(checks, Mapping):
    raise AnalysisError(f'{label}.checks are absent.')
  for field in fields:
    if checks.get(field) is not True:
      raise AnalysisError(f'{label}.checks.{field} is not true.')
  return checks


def _array_shape(value: Any) -> tuple[int, ...]:
  if not isinstance(value, list):
    return ()
  if not value:
    return (0,)
  child_shapes = {_array_shape(child) for child in value}
  if len(child_shapes) != 1:
    raise AnalysisError('Runtime intervention array is ragged.')
  return (len(value),) + child_shapes.pop()


def _array_leaves(value: Any) -> Iterable[Any]:
  if isinstance(value, list):
    for child in value:
      yield from _array_leaves(child)
  else:
    yield value


def _require_integer_array(value: Any, shape: tuple[int, ...], label: str) -> None:
  if _array_shape(value) != shape:
    raise AnalysisError(f'{label} shape changed: {_array_shape(value)} != {shape}.')
  if any(not isinstance(item, int) or isinstance(item, bool) for item in _array_leaves(value)):
    raise AnalysisError(f'{label} must contain only integer indices.')


def _require_bool_array(value: Any, shape: tuple[int, ...], label: str) -> None:
  if _array_shape(value) != shape:
    raise AnalysisError(f'{label} shape changed: {_array_shape(value)} != {shape}.')
  if any(not isinstance(item, bool) for item in _array_leaves(value)):
    raise AnalysisError(f'{label} must contain only booleans.')


def _runtime_route(
    value: Any, *, rows: int, coalition_id: int, donor_rows: Sequence[int],
    label: str,
) -> None:
  """Independently parses every persisted runtime donor/natural/mask array."""
  if not isinstance(value, Mapping) or set(value) != {
      'transformer_output', 'encoder_skips', 'final_embedding',
      'phase_r_residuals',
  }:
    raise AnalysisError(f'{label} runtime intervention schema changed.')
  t, e_mask = divmod(coalition_id, 128)
  natural_rows = (
      [0, 1, 1, 1, 0, 0] if rows == 6
      else [0, 1, 1, 1, 0, 0, 6, 7]
  )
  active = [False, False, True, True, True, True] + ([False, False] if rows == 8 else [])

  def whole_route(node: Any, components: int, enabled: Sequence[bool], name: str) -> None:
    if not isinstance(node, Mapping) or set(node) != {
        'donor_batch_indices', 'natural_identity_batch_indices', 'transfer_mask'
    }:
      raise AnalysisError(f'{label}.{name} schema changed.')
    _require_integer_array(node['donor_batch_indices'], (components, rows), f'{label}.{name}.donor')
    _require_integer_array(
        node['natural_identity_batch_indices'], (components, rows),
        f'{label}.{name}.natural',
    )
    _require_bool_array(node['transfer_mask'], (components, rows), f'{label}.{name}.mask')
    expected_donor = [list(donor_rows) for _ in range(components)]
    expected_natural = [list(natural_rows) for _ in range(components)]
    expected_mask = [
        [bool(component_enabled and row_active) for row_active in active]
        for component_enabled in enabled
    ]
    if node['donor_batch_indices'] != expected_donor:
      raise AnalysisError(f'{label}.{name} donor map changed.')
    if node['natural_identity_batch_indices'] != expected_natural:
      raise AnalysisError(f'{label}.{name} natural map changed.')
    if node['transfer_mask'] != expected_mask:
      raise AnalysisError(f'{label}.{name} transfer mask changed.')

  whole_route(value['transformer_output'], 1, [bool(t)], 'transformer_output')
  whole_route(
      value['encoder_skips'], 7,
      [bool(e_mask & (1 << index)) for index in range(7)], 'encoder_skips',
  )
  final = value['final_embedding']
  if not isinstance(final, Mapping) or set(final) != {'donor_batch_indices', 'transfer_mask'}:
    raise AnalysisError(f'{label}.final_embedding schema changed.')
  _require_integer_array(
      final['donor_batch_indices'], (1, rows, 2), f'{label}.final_embedding.donor'
  )
  _require_bool_array(
      final['transfer_mask'], (1, rows, 2), f'{label}.final_embedding.mask'
  )
  expected_final_donors = [
      [[row, row] for row in range(rows)]
  ]
  if (
      final['donor_batch_indices'] != expected_final_donors
      or any(_array_leaves(final['transfer_mask']))
  ):
    raise AnalysisError(f'{label}.final_embedding disabled-route audit failed.')
  residuals = value['phase_r_residuals']
  expected_residuals = {
      'pre_attention_residual_transfer',
      'post_attention_residual_transfer',
      'post_mlp_residual_transfer',
  }
  if not isinstance(residuals, Mapping) or set(residuals) != expected_residuals:
    raise AnalysisError(f'{label}.phase_r_residuals schema changed.')
  for name in expected_residuals:
    node = residuals[name]
    if not isinstance(node, Mapping) or set(node) != {'donor_batch_indices', 'transfer_mask'}:
      raise AnalysisError(f'{label}.{name} schema changed.')
    _require_integer_array(
        node['donor_batch_indices'], (9, rows, 24), f'{label}.{name}.donor'
    )
    _require_bool_array(
        node['transfer_mask'], (9, rows, 24), f'{label}.{name}.mask'
    )
    expected_residual_donors = [
        [[row] * 24 for row in range(rows)] for _ in range(9)
    ]
    if (
        node['donor_batch_indices'] != expected_residual_donors
        or any(_array_leaves(node['transfer_mask']))
    ):
      raise AnalysisError(f'{label}.{name} disabled-route audit failed.')


def _same_number(observed: Any, expected: float, label: str) -> None:
  if _finite(observed, label) != expected:
    raise AnalysisError(f'{label} differs from independently recomputed value.')


def _validate_emitted_metrics(
    checks: Mapping[str, Any], metrics: Mapping[str, Any], label: str
) -> None:
  means = checks.get('target_means')
  if not isinstance(means, Mapping) or set(means) != set(TRACE_ROLES):
    raise AnalysisError(f'{label}.checks.target_means schema changed.')
  raw = checks.get('raw_movement')
  recovery = checks.get('recovery')
  if not isinstance(raw, Mapping) or set(raw) != set(metrics['movements']):
    raise AnalysisError(f'{label}.checks.raw_movement schema changed.')
  for direction, expected in metrics['movements'].items():
    _same_number(raw.get(direction), expected, f'{label}.raw_movement.{direction}')
  if not isinstance(recovery, Mapping) or set(recovery) != {
      'reference_into_alternate', 'alternate_into_reference',
      'bidirectional_bottleneck',
  }:
    raise AnalysisError(f'{label}.checks.recovery schema changed.')
  if metrics['recoveries'] is None:
    if any(value is not None for value in recovery.values()):
      raise AnalysisError(f'{label} emitted recovery for a zero denominator.')
  else:
    for direction, expected in metrics['recoveries'].items():
      _same_number(recovery.get(direction), expected, f'{label}.recovery.{direction}')
    _same_number(
        recovery.get('bidirectional_bottleneck'), metrics['B'], f'{label}.recovery.B'
    )


def _validate_identity_record(
    record: Mapping[str, Any], case: Mapping[str, Any], *, freeze_sha: str,
    execution_index: int, label: str,
) -> dict[str, Any]:
  _require_exact_keys(record, {
      'status', 'family', 'script_version', 'protocol_sha256',
      'supersession_sha256', 'supersession_commit', 'freeze_sha256',
      'execution_index', 'six_row_executable_fingerprint',
      'same_six_row_compiled_executable', 'case', 'interval',
      'sequence_sha256', 'resolved_position_sets', 'canonical_target',
      'runtime_interventions', 'target_readout', 'repeat_target_readout',
      'trace_fingerprint', 'repeat_trace_fingerprint',
      'natural_route_fingerprints', 'checks', 'failure', 'direction_gate',
      'program_signatures', 'seconds', 'created_at_unix_s',
  }, label)
  _require_common_linkage(
      record, family='identity', freeze_sha=freeze_sha,
      execution_index=execution_index, label=label,
  )
  _validate_case_record(record.get('case'), case, label)
  _validate_canonical_target(record, case, label)
  if record.get('same_six_row_compiled_executable') is not True:
    raise AnalysisError(f'{label} did not use the one six-row executable.')
  _runtime_route(
      record.get('runtime_interventions'), rows=6, coalition_id=0,
      donor_rows=(0, 1, 0, 1, 1, 0), label=f'{label}.runtime_interventions',
  )
  fingerprint = record.get('six_row_executable_fingerprint')
  if not _is_sha256(fingerprint):
    raise AnalysisError(f'{label} six-row executable fingerprint is invalid.')
  sequence = record.get('sequence_sha256')
  if (
      not isinstance(sequence, Mapping) or set(sequence) != {'reference', 'alternate'}
      or not all(_is_sha256(value) for value in sequence.values())
      or sequence['reference'] == sequence['alternate']
  ):
    raise AnalysisError(f'{label} sequence binding is malformed.')
  readout = _require_repeat(
      record, 'target_readout', 'repeat_target_readout', label, rows=6
  )
  natural_rows = (0, 1, 1, 1, 0, 0)
  if any(_row_bytes(readout, row) != _row_bytes(readout, donor)
         for row, donor in enumerate(natural_rows)):
    raise AnalysisError(f'{label} natural six-row target identity failed.')
  checks = _require_checks(record.get('checks'), (
      'passed', 'target_repeat_exact', 'target_duplicates_exact',
      'trace_repeat_exact', 'natural_duplicates_exact',
      'all_false_natural_effective_exact',
      'target_total_equals_two_times_mean',
  ), label)
  _require_exact_keys(checks, {
      'passed', 'target_means', 'target_repeat_exact',
      'target_duplicates_exact', 'trace_repeat_exact',
      'natural_duplicates_exact', 'num_values',
      'all_false_natural_effective_exact',
      'target_total_equals_two_times_mean', 'trace_fingerprint_first',
      'trace_fingerprint_repeat',
  }, f'{label}.checks')
  if checks.get('num_values') != 2:
    raise AnalysisError(f'{label}.checks.num_values changed.')
  _validate_trace_fingerprint(record.get('trace_fingerprint'), f'{label}.trace')
  _validate_trace_fingerprint(
      record.get('repeat_trace_fingerprint'), f'{label}.repeat_trace'
  )
  if record.get('trace_fingerprint') != record.get('repeat_trace_fingerprint'):
    raise AnalysisError(f'{label} compact trace repeat differs.')
  if (
      checks.get('trace_fingerprint_first') != record.get('trace_fingerprint')
      or checks.get('trace_fingerprint_repeat')
      != record.get('repeat_trace_fingerprint')
  ):
    raise AnalysisError(f'{label} embedded trace fingerprints changed.')
  target_means = checks.get('target_means')
  if not isinstance(target_means, Mapping) or set(target_means) != set(TRACE_ROLES):
    raise AnalysisError(f'{label}.checks.target_means schema changed.')
  for role, expected in zip(TRACE_ROLES, readout['means'], strict=True):
    _same_number(target_means.get(role), expected, f'{label}.target_means.{role}')
  _require_seconds(record.get('seconds'), {'first', 'repeat'}, f'{label}.seconds')
  route_fingerprints = record.get('natural_route_fingerprints')
  if not isinstance(route_fingerprints, Mapping) or set(route_fingerprints) != {'T', 'E'}:
    raise AnalysisError(f'{label} natural-route fingerprints are absent.')
  t_fp, e_fp = route_fingerprints['T'], route_fingerprints['E']
  if (
      not isinstance(t_fp, list) or len(t_fp) != 6
      or any(not isinstance(row, list) or len(row) != 4 for row in t_fp)
      or not isinstance(e_fp, list) or len(e_fp) != 7
      or any(
          not isinstance(stage, list) or len(stage) != 6
          or any(not isinstance(row, list) or len(row) != 4 for row in stage)
          for stage in e_fp
      )
  ):
    raise AnalysisError(f'{label} natural-route fingerprint shapes changed.')
  for value in (
      item for stage in (t_fp, *e_fp) for row in stage for item in row
  ):
    _finite(value, f'{label}.natural_route_fingerprint')
  predicted_delta = _f32(
      readout['means'][1] - readout['means'][0], f'{label}.predicted_delta'
  )
  expected_eligible = (
      execution_index in EFFECT_ORDERS
      and abs(predicted_delta) >= EFFECT_THRESHOLD
      and (predicted_delta > 0) == (case['delta_logit'] > 0)
  )
  gate = record.get('direction_gate')
  if not isinstance(gate, Mapping):
    raise AnalysisError(f'{label} direction gate is absent.')
  _same_number(
      gate.get('predicted_alt_minus_ref_logit_margin'), predicted_delta,
      f'{label}.direction_gate.predicted_delta',
  )
  _same_number(
      gate.get('experimental_delta_logit'), case['delta_logit'],
      f'{label}.direction_gate.experimental_delta',
  )
  if gate.get('minimum_absolute_predicted_effect') != EFFECT_THRESHOLD:
    raise AnalysisError(f'{label} direction threshold changed.')
  if execution_index in EFFECT_ORDERS:
    if gate.get('direction_matches_delta_logit') is not (
        (predicted_delta > 0) == (case['delta_logit'] > 0)
    ):
      raise AnalysisError(f'{label} direction-match flag changed.')
    if gate.get('eligible_for_causal_census') is not expected_eligible:
      raise AnalysisError(f'{label} target eligibility was not recomputed exactly.')
  else:
    if (
        gate.get('direction_matches_delta_logit') is not None
        or gate.get('eligible_for_causal_census') is not False
    ):
      raise AnalysisError(f'{label} neutral was relabelled as eligible.')
  signatures = record.get('program_signatures')
  if not isinstance(signatures, Mapping) or set(signatures) != {
      'selection', 'target', 'six_interventions', 'eight_interventions'
  }:
    raise AnalysisError(f'{label} program-signature set changed.')
  return {
      'readout': readout,
      'sequence_sha256': dict(sequence),
      'natural_route_fingerprints': route_fingerprints,
      'six_row_executable_fingerprint': fingerprint,
      'predicted_delta': predicted_delta,
      'eligible': expected_eligible,
      'program_signatures': signatures,
  }


def _validate_frozen_sequence_binding(
    identity: Mapping[str, Any], expected: Any, label: str
) -> None:
  if identity.get('sequence_sha256') != expected:
    raise AnalysisError(
        f'{label} sequence binding differs from the frozen GRCh38 input.'
    )


def _validate_coalition_metadata(value: Any, coalition_id: int, label: str) -> None:
  if not isinstance(value, Mapping):
    raise AnalysisError(f'{label} coalition metadata is absent.')
  t, e_mask = divmod(coalition_id, 128)
  e_bits = [bool(e_mask & (1 << index)) for index in range(7)]
  enabled_e = [
      player for player, selected in zip(PLAYERS_E, e_bits, strict=True) if selected
  ]
  enabled = (['T'] if t else []) + enabled_e
  expected = {
      'coalition_id': coalition_id,
      't': t,
      'e_mask': e_mask,
      'e_bits': e_bits,
      'e_bits_binary': format(e_mask, '07b'),
      'enabled_players': enabled,
      'coalition_bit_order': list(PLAYERS_E) + ['T'],
      'shapley_player_order': list(PLAYERS_8),
  }
  # Backwards-in-development spelling is rejected once the stable serializer
  # emits both explicit orders; a single ambiguous player_order is unsafe.
  for key, expected_value in expected.items():
    if value.get(key) != expected_value:
      raise AnalysisError(f'{label}.coalition.{key} changed.')


def _validate_binding(
    binding: Any, *, expected_relative: str, run_dir: Path, label: str
) -> None:
  if not isinstance(binding, Mapping) or set(binding) != {'path', 'sha256'}:
    raise AnalysisError(f'{label} binding schema changed.')
  if binding.get('path') != expected_relative:
    raise AnalysisError(f'{label} binding path changed.')
  path = (run_dir / expected_relative).resolve()
  try:
    path.relative_to(run_dir.resolve())
  except ValueError as error:
    raise AnalysisError(f'{label} binding escaped the run directory.') from error
  if _sha256(path) != binding.get('sha256'):
    raise AnalysisError(f'{label} binding hash changed.')


def _validate_coalition_record(
    record: Mapping[str, Any], case: Mapping[str, Any], coalition_id: int, *,
    identity: Mapping[str, Any], identity_relative: str, run_dir: Path,
    freeze_sha: str, execution_index: int, label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
  _require_exact_keys(record, {
      'status', 'family', 'script_version', 'protocol_sha256',
      'supersession_sha256', 'supersession_commit', 'freeze_sha256',
      'execution_index', 'six_row_executable_fingerprint',
      'same_six_row_compiled_executable', 'case', 'coalition',
      'runtime_interventions', 'identity_binding', 'target_readout',
      'repeat_target_readout', 'trace_fingerprint',
      'repeat_trace_fingerprint', 'checks', 'failure', 'seconds',
      'created_at_unix_s',
  }, label)
  _require_common_linkage(
      record, family='encoder_skip_coalition', freeze_sha=freeze_sha,
      execution_index=execution_index, label=label,
  )
  _validate_case_record(record.get('case'), case, label)
  _validate_coalition_metadata(record.get('coalition'), coalition_id, label)
  _runtime_route(
      record.get('runtime_interventions'), rows=6, coalition_id=coalition_id,
      donor_rows=(0, 1, 0, 1, 1, 0), label=f'{label}.runtime_interventions',
  )
  _validate_binding(
      record.get('identity_binding'), expected_relative=identity_relative,
      run_dir=run_dir, label=f'{label}.identity',
  )
  if (
      record.get('same_six_row_compiled_executable') is not True
      or record.get('six_row_executable_fingerprint')
      != identity['six_row_executable_fingerprint']
  ):
    raise AnalysisError(f'{label} six-row executable linkage changed.')
  readout = _require_repeat(
      record, 'target_readout', 'repeat_target_readout', label, rows=6
  )
  for row, identity_row in ((0, 0), (1, 1), (3, 1), (5, 0)):
    if _row_bytes(readout, row) != _row_bytes(identity['readout'], identity_row):
      raise AnalysisError(f'{label} baseline/self endpoint evidence changed.')
  if coalition_id == 0 and any(
      _row_bytes(readout, row) != _row_bytes(identity['readout'], row)
      for row in range(6)
  ):
    raise AnalysisError(f'{label} ID0 is not an endpoint-exact no-op.')
  if coalition_id == 255 and (
      _row_bytes(readout, 2) != _row_bytes(identity['readout'], 0)
      or _row_bytes(readout, 4) != _row_bytes(identity['readout'], 1)
  ):
    raise AnalysisError(f'{label} ID255 does not close both raw endpoints.')
  checks = _require_checks(record.get('checks'), (
      'passed', 'baseline_targets_exact_from_identity', 'self_targets_exact',
      'target_repeat_exact', 'trace_repeat_exact',
      'transformer_internal_seams_disabled_exact',
      'natural_T_same_allele_exact', 'natural_E_same_allele_exact',
      'natural_route_fingerprints_match_identity',
      'baseline_rows_T_natural_effective_exact',
      'baseline_rows_E_natural_effective_exact',
      'E_enabled_donor_exact', 'E_disabled_noop_exact',
      'final_embedding_disabled_exact',
      'runtime_route_masks_and_maps_exact',
  ), label)
  _require_exact_keys(checks, {
      'passed', 'target_means', 'baseline_targets_exact_from_identity',
      'self_targets_exact', 'target_repeat_exact', 'trace_repeat_exact',
      'transformer_internal_seams_disabled_exact',
      'natural_T_same_allele_exact', 'natural_E_same_allele_exact',
      'natural_route_fingerprints_match_identity',
      'baseline_rows_T_natural_effective_exact',
      'baseline_rows_E_natural_effective_exact', 'T_enabled_donor_exact',
      'T_disabled_noop_exact', 'E_enabled_donor_exact',
      'E_disabled_noop_exact', 'final_embedding_disabled_exact',
      'runtime_route_masks_and_maps_exact', 'id0_identity_endpoint_exact',
      'id255_endpoint_closure_exact', 'raw_movement', 'recovery',
  }, f'{label}.checks')
  t = coalition_id // 128
  if checks.get('T_enabled_donor_exact' if t else 'T_disabled_noop_exact') is not True:
    raise AnalysisError(f'{label} T enabled/disabled tensor audit failed.')
  if checks.get('id0_identity_endpoint_exact') is not (coalition_id == 0):
    raise AnalysisError(f'{label} ID0 audit flag changed.')
  if checks.get('id255_endpoint_closure_exact') is not (coalition_id == 255):
    raise AnalysisError(f'{label} ID255 audit flag changed.')
  _validate_trace_fingerprint(record.get('trace_fingerprint'), f'{label}.trace')
  _validate_trace_fingerprint(
      record.get('repeat_trace_fingerprint'), f'{label}.repeat_trace'
  )
  if record.get('trace_fingerprint') != record.get('repeat_trace_fingerprint'):
    raise AnalysisError(f'{label} compact trace repeat differs.')
  _require_seconds(record.get('seconds'), {'first', 'repeat'}, f'{label}.seconds')
  metrics = _metrics(readout)
  _validate_emitted_metrics(checks, metrics, label)
  target_means = checks['target_means']
  for role, expected in zip(TRACE_ROLES, readout['means'], strict=True):
    _same_number(target_means.get(role), expected, f'{label}.target_means.{role}')
  return metrics, readout


def _validate_ood_record(
    record: Mapping[str, Any], recipient: Mapping[str, Any], donor: Mapping[str, Any],
    coalition_id: int, *, identity: Mapping[str, Any], identity_relative: str,
    donor_identity: Mapping[str, Any], donor_identity_relative: str,
    linked_relative: str, run_dir: Path, freeze_sha: str,
    execution_index: int, eight_fingerprint: str, label: str,
) -> dict[str, Any]:
  _require_exact_keys(record, {
      'status', 'family', 'script_version', 'protocol_sha256',
      'supersession_sha256', 'supersession_commit', 'freeze_sha256',
      'execution_index', 'eight_row_executable_fingerprint',
      'same_eight_row_compiled_executable', 'recipient_case', 'donor_case',
      'coalition', 'batch_roles', 'natural_identity_rows',
      'intended_donor_rows', 'unrelated_donor_rows', 'identity_binding',
      'donor_identity_binding', 'linked_six_row_coalition',
      'recipient_sequence_sha256', 'donor_sequence_sha256',
      'runtime_interventions', 'intended_target_readout',
      'intended_repeat_target_readout', 'unrelated_target_readout',
      'unrelated_repeat_target_readout', 'intended_trace_fingerprint',
      'intended_repeat_trace_fingerprint', 'unrelated_trace_fingerprint',
      'unrelated_repeat_trace_fingerprint', 'raw_movement', 'checks',
      'failure', 'seconds', 'created_at_unix_s',
  }, label)
  _require_common_linkage(
      record, family='unrelated_donor_anchor', freeze_sha=freeze_sha,
      execution_index=execution_index, label=label,
  )
  _validate_case_record(record.get('recipient_case'), recipient, label)
  _validate_case_record(record.get('donor_case'), donor, f'{label}.donor')
  _validate_coalition_metadata(record.get('coalition'), coalition_id, label)
  _validate_binding(
      record.get('identity_binding'), expected_relative=identity_relative,
      run_dir=run_dir, label=f'{label}.identity',
  )
  _validate_binding(
      record.get('donor_identity_binding'),
      expected_relative=donor_identity_relative,
      run_dir=run_dir, label=f'{label}.donor_identity',
  )
  _validate_binding(
      record.get('linked_six_row_coalition'), expected_relative=linked_relative,
      run_dir=run_dir, label=f'{label}.linked_six_row_coalition',
  )
  if (
      record.get('same_eight_row_compiled_executable') is not True
      or record.get('eight_row_executable_fingerprint') != eight_fingerprint
  ):
    raise AnalysisError(f'{label} eight-row executable linkage changed.')
  expected_maps = {
      'batch_roles': list(TRACE_ROLES) + [
          'unrelated_reference_donor', 'unrelated_alternate_donor'
      ],
      'natural_identity_rows': [0, 1, 1, 1, 0, 0, 6, 7],
      'intended_donor_rows': [0, 1, 0, 1, 1, 0, 6, 7],
      'unrelated_donor_rows': [0, 1, 6, 1, 7, 0, 6, 7],
  }
  for key, expected in expected_maps.items():
    if record.get(key) != expected:
      raise AnalysisError(f'{label}.{key} changed.')
  runtime = record.get('runtime_interventions')
  if not isinstance(runtime, Mapping) or set(runtime) != {'intended', 'unrelated'}:
    raise AnalysisError(f'{label}.runtime_interventions schema changed.')
  _runtime_route(
      runtime['intended'], rows=8, coalition_id=coalition_id,
      donor_rows=expected_maps['intended_donor_rows'],
      label=f'{label}.runtime_interventions.intended',
  )
  _runtime_route(
      runtime['unrelated'], rows=8, coalition_id=coalition_id,
      donor_rows=expected_maps['unrelated_donor_rows'],
      label=f'{label}.runtime_interventions.unrelated',
  )
  if record.get('recipient_sequence_sha256') != identity['sequence_sha256']:
    raise AnalysisError(f'{label} recipient sequence binding changed.')
  donor_sequence = record.get('donor_sequence_sha256')
  if donor_sequence != donor_identity['sequence_sha256']:
    raise AnalysisError(f'{label} donor sequence differs from donor identity.')
  intended = _require_repeat(
      record, 'intended_target_readout', 'intended_repeat_target_readout',
      f'{label}.intended', rows=8,
  )
  unrelated = _require_repeat(
      record, 'unrelated_target_readout', 'unrelated_repeat_target_readout',
      f'{label}.unrelated', rows=8,
  )
  for readout_name, readout in (('intended', intended), ('unrelated', unrelated)):
    if (
        _row_bytes(readout, 3) != _row_bytes(readout, 1)
        or _row_bytes(readout, 5) != _row_bytes(readout, 0)
    ):
      raise AnalysisError(f'{label}.{readout_name} within-8-row self target changed.')
  for row in (0, 1, 3, 5, 6, 7):
    if _row_bytes(intended, row) != _row_bytes(unrelated, row):
      raise AnalysisError(f'{label} intended/unrelated fixed row {row} differs.')
  trace_fields = (
      ('intended_trace_fingerprint', 'intended_repeat_trace_fingerprint'),
      ('unrelated_trace_fingerprint', 'unrelated_repeat_trace_fingerprint'),
  )
  for first, repeat in trace_fields:
    _validate_trace_fingerprint(record.get(first), f'{label}.{first}')
    _validate_trace_fingerprint(record.get(repeat), f'{label}.{repeat}')
    if record.get(first) != record.get(repeat):
      raise AnalysisError(f'{label} {first} repeat differs.')
  _require_seconds(
      record.get('seconds'),
      {'intended', 'intended_repeat', 'unrelated', 'unrelated_repeat'},
      f'{label}.seconds',
  )
  checks = _require_checks(record.get('checks'), (
      'passed', 'baseline_rows_exact_between_calls',
      'self_rows_exact_between_calls', 'donor_source_rows_exact_between_calls',
      'self_targets_exact',
      'intended_route_tensor_donor_exact',
      'unrelated_route_tensor_donor_exact',
      'enabled_disabled_T_E_exact',
      'runtime_route_masks_and_maps_exact',
      'natural_route_tensors_exact_between_calls',
      'transformer_internal_seams_disabled_exact',
      'final_embedding_disabled_exact',
      'intended_target_repeat_exact', 'intended_trace_repeat_exact',
      'unrelated_target_repeat_exact', 'unrelated_trace_repeat_exact',
  ), label)
  _require_exact_keys(checks, {
      'passed', 'baseline_rows_exact_between_calls',
      'self_rows_exact_between_calls', 'donor_source_rows_exact_between_calls',
      'natural_route_tensors_exact_between_calls', 'self_targets_exact',
      'id0_all_recipient_rows_noop_exact',
      'id0_intended_unrelated_all_rows_exact',
      'id255_intended_endpoint_closure_exact',
      'id255_unrelated_endpoint_closure_exact',
      'intended_route_tensor_donor_exact',
      'unrelated_route_tensor_donor_exact', 'enabled_disabled_T_E_exact',
      'runtime_route_masks_and_maps_exact', 'intended_target_repeat_exact',
      'intended_trace_repeat_exact', 'unrelated_target_repeat_exact',
      'unrelated_trace_repeat_exact',
      'transformer_internal_seams_disabled_exact',
      'final_embedding_disabled_exact', 'normalization_computed',
  }, f'{label}.checks')
  if checks.get('normalization_computed') is not False:
    raise AnalysisError(f'{label} improperly normalized an OOD stress control.')
  anchor_flags = {
      'id0_all_recipient_rows_noop_exact': coalition_id == 0,
      'id0_intended_unrelated_all_rows_exact': coalition_id == 0,
      'id255_intended_endpoint_closure_exact': coalition_id == 255,
      'id255_unrelated_endpoint_closure_exact': coalition_id == 255,
  }
  for field, expected in anchor_flags.items():
    if checks.get(field) is not expected:
      raise AnalysisError(f'{label}.checks.{field} changed.')
  if coalition_id == 0:
    for readout in (intended, unrelated):
      for row, natural_row in enumerate((0, 1, 1, 1, 0, 0)):
        if _row_bytes(readout, row) != _row_bytes(readout, natural_row):
          raise AnalysisError(f'{label} ID0 recipient endpoint no-op failed.')
    if any(_row_bytes(intended, row) != _row_bytes(unrelated, row) for row in range(8)):
      raise AnalysisError(f'{label} ID0 intended/unrelated endpoint outputs differ.')
  if coalition_id == 255:
    if (
        _row_bytes(intended, 2) != _row_bytes(intended, 0)
        or _row_bytes(intended, 4) != _row_bytes(intended, 1)
        or _row_bytes(unrelated, 2) != _row_bytes(unrelated, 6)
        or _row_bytes(unrelated, 4) != _row_bytes(unrelated, 7)
    ):
      raise AnalysisError(f'{label} ID255 endpoint closure failed.')
  intended_metrics = _raw_movements(intended)
  unrelated_metrics = _raw_movements(unrelated)
  emitted = record.get('raw_movement')
  if not isinstance(emitted, Mapping) or set(emitted) != {'intended', 'unrelated'}:
    raise AnalysisError(f'{label} OOD raw-movement evidence is missing.')
  for name, metrics in (
      ('intended', intended_metrics), ('unrelated', unrelated_metrics)
  ):
    values = emitted.get(name)
    if not isinstance(values, Mapping) or set(values) != set(metrics['movements']):
      raise AnalysisError(f'{label}.{name} movement schema changed.')
    for direction, expected in metrics['movements'].items():
      _same_number(values.get(direction), expected, f'{label}.{name}.{direction}')
  return {
      'recipient_order': recipient['order'],
      'donor_order': donor['order'],
      'coalition_id': coalition_id,
      'intended_raw_movement': intended_metrics['movements'],
      'unrelated_raw_movement': unrelated_metrics['movements'],
      'intended_mean_absolute_movement': intended_metrics['mean_absolute_movement'],
      'unrelated_mean_absolute_movement': unrelated_metrics['mean_absolute_movement'],
      'donor_normalized_recovery_computed': False,
  }


def _require_invalid_linkage(
    record: Mapping[str, Any], *, family: str, freeze_sha: str,
    execution_index: int, label: str,
) -> None:
  expected = {
      'status': 'invalid', 'family': family,
      'script_version': SOURCE_SCRIPT_VERSION,
      'protocol_sha256': PROTOCOL_SHA256,
      'supersession_sha256': SUPERSESSION_SHA256,
      'supersession_commit': 'c64def4', 'freeze_sha256': freeze_sha,
      'execution_index': execution_index,
  }
  for key, value in expected.items():
    if record.get(key) != value:
      raise AnalysisError(f'{label}.{key} differs for controlled invalid record.')
  if record.get('checks') is not None:
    raise AnalysisError(f'{label} invalid record unexpectedly has passing checks.')
  failure = record.get('failure')
  if (
      not isinstance(failure, Mapping) or set(failure) != {'type', 'message'}
      or not all(isinstance(failure[key], str) and failure[key] for key in failure)
  ):
    raise AnalysisError(f'{label} invalid failure evidence is malformed.')
  _finite(record.get('created_at_unix_s'), f'{label}.created_at_unix_s')


def _validate_invalid_record(
    record: Mapping[str, Any], *, family: str, case: Mapping[str, Any],
    expected_sequence: Mapping[str, str],
    coalition_id: int | None, identity: Mapping[str, Any] | None,
    identity_relative: str | None, donor_case: Mapping[str, Any] | None,
    donor_identity: Mapping[str, Any] | None,
    donor_identity_relative: str | None, linked_relative: str | None,
    run_dir: Path, freeze_sha: str, execution_index: int,
    six_fingerprint: str, eight_fingerprint: str, label: str,
) -> None:
  _require_invalid_linkage(
      record, family=family, freeze_sha=freeze_sha,
      execution_index=execution_index, label=label,
  )
  if family == 'identity':
    _require_exact_keys(record, {
        'status', 'family', 'script_version', 'protocol_sha256',
        'supersession_sha256', 'supersession_commit', 'freeze_sha256',
        'execution_index', 'six_row_executable_fingerprint',
        'same_six_row_compiled_executable', 'case', 'interval',
        'sequence_sha256', 'resolved_position_sets', 'canonical_target',
        'runtime_interventions', 'target_readout', 'repeat_target_readout',
        'trace_fingerprint', 'repeat_trace_fingerprint',
        'natural_route_fingerprints', 'checks', 'failure', 'direction_gate',
        'program_signatures', 'seconds', 'created_at_unix_s',
    }, label)
    _validate_case_record(record.get('case'), case, label)
    _validate_canonical_target(record, case, label)
    _validate_frozen_sequence_binding(
        {'sequence_sha256': record.get('sequence_sha256')}, expected_sequence, label
    )
    _runtime_route(
        record.get('runtime_interventions'), rows=6, coalition_id=0,
        donor_rows=(0, 1, 0, 1, 1, 0), label=f'{label}.runtime',
    )
    _readout(record, 'target_readout', label, rows=6)
    _readout(record, 'repeat_target_readout', label, rows=6)
    _validate_trace_fingerprint(record.get('trace_fingerprint'), f'{label}.trace')
    _validate_trace_fingerprint(
        record.get('repeat_trace_fingerprint'), f'{label}.repeat_trace'
    )
    if (
        record.get('same_six_row_compiled_executable') is not True
        or record.get('six_row_executable_fingerprint') != six_fingerprint
    ):
      raise AnalysisError(f'{label} invalid identity executable changed.')
    _require_seconds(record.get('seconds'), {'first', 'repeat'}, f'{label}.seconds')
    return
  assert coalition_id is not None and identity is not None and identity_relative is not None
  _validate_coalition_metadata(record.get('coalition'), coalition_id, label)
  if family == 'encoder_skip_coalition':
    _require_exact_keys(record, {
        'status', 'family', 'script_version', 'protocol_sha256',
        'supersession_sha256', 'supersession_commit', 'freeze_sha256',
        'execution_index', 'six_row_executable_fingerprint',
        'same_six_row_compiled_executable', 'case', 'coalition',
        'runtime_interventions', 'identity_binding', 'target_readout',
        'repeat_target_readout', 'trace_fingerprint',
        'repeat_trace_fingerprint', 'checks', 'failure', 'seconds',
        'created_at_unix_s',
    }, label)
    _validate_case_record(record.get('case'), case, label)
    _runtime_route(
        record.get('runtime_interventions'), rows=6, coalition_id=coalition_id,
        donor_rows=(0, 1, 0, 1, 1, 0), label=f'{label}.runtime',
    )
    _validate_binding(
        record.get('identity_binding'), expected_relative=identity_relative,
        run_dir=run_dir, label=f'{label}.identity',
    )
    _readout(record, 'target_readout', label, rows=6)
    _readout(record, 'repeat_target_readout', label, rows=6)
    _validate_trace_fingerprint(record.get('trace_fingerprint'), f'{label}.trace')
    _validate_trace_fingerprint(
        record.get('repeat_trace_fingerprint'), f'{label}.repeat_trace'
    )
    if (
        record.get('same_six_row_compiled_executable') is not True
        or record.get('six_row_executable_fingerprint') != six_fingerprint
    ):
      raise AnalysisError(f'{label} invalid coalition executable changed.')
    _require_seconds(record.get('seconds'), {'first', 'repeat'}, f'{label}.seconds')
    return
  assert donor_case is not None and donor_identity is not None
  assert donor_identity_relative is not None and linked_relative is not None
  _require_exact_keys(record, {
      'status', 'family', 'script_version', 'protocol_sha256',
      'supersession_sha256', 'supersession_commit', 'freeze_sha256',
      'execution_index', 'eight_row_executable_fingerprint',
      'same_eight_row_compiled_executable', 'recipient_case', 'donor_case',
      'coalition', 'batch_roles', 'natural_identity_rows',
      'intended_donor_rows', 'unrelated_donor_rows', 'identity_binding',
      'donor_identity_binding', 'linked_six_row_coalition',
      'recipient_sequence_sha256', 'donor_sequence_sha256',
      'runtime_interventions', 'intended_target_readout',
      'intended_repeat_target_readout', 'unrelated_target_readout',
      'unrelated_repeat_target_readout', 'intended_trace_fingerprint',
      'intended_repeat_trace_fingerprint', 'unrelated_trace_fingerprint',
      'unrelated_repeat_trace_fingerprint', 'raw_movement', 'checks',
      'failure', 'seconds', 'created_at_unix_s',
  }, label)
  _validate_case_record(record.get('recipient_case'), case, label)
  _validate_case_record(record.get('donor_case'), donor_case, f'{label}.donor')
  if record.get('recipient_sequence_sha256') != identity['sequence_sha256']:
    raise AnalysisError(f'{label} invalid OOD recipient sequence changed.')
  if record.get('donor_sequence_sha256') != donor_identity['sequence_sha256']:
    raise AnalysisError(f'{label} invalid OOD donor sequence changed.')
  runtime = record.get('runtime_interventions')
  if not isinstance(runtime, Mapping) or set(runtime) != {'intended', 'unrelated'}:
    raise AnalysisError(f'{label} invalid OOD runtime schema changed.')
  _runtime_route(
      runtime['intended'], rows=8, coalition_id=coalition_id,
      donor_rows=(0, 1, 0, 1, 1, 0, 6, 7), label=f'{label}.intended.runtime',
  )
  _runtime_route(
      runtime['unrelated'], rows=8, coalition_id=coalition_id,
      donor_rows=(0, 1, 6, 1, 7, 0, 6, 7), label=f'{label}.unrelated.runtime',
  )
  _validate_binding(
      record.get('identity_binding'), expected_relative=identity_relative,
      run_dir=run_dir, label=f'{label}.identity',
  )
  _validate_binding(
      record.get('donor_identity_binding'), expected_relative=donor_identity_relative,
      run_dir=run_dir, label=f'{label}.donor_identity',
  )
  _validate_binding(
      record.get('linked_six_row_coalition'), expected_relative=linked_relative,
      run_dir=run_dir, label=f'{label}.linked',
  )
  parsed_readouts = {}
  for field in (
      'intended_target_readout', 'intended_repeat_target_readout',
      'unrelated_target_readout', 'unrelated_repeat_target_readout',
  ):
    parsed_readouts[field] = _readout(record, field, label, rows=8)
  for field in (
      'intended_trace_fingerprint', 'intended_repeat_trace_fingerprint',
      'unrelated_trace_fingerprint', 'unrelated_repeat_trace_fingerprint',
  ):
    _validate_trace_fingerprint(record.get(field), f'{label}.{field}')
  emitted = record.get('raw_movement')
  if not isinstance(emitted, Mapping) or set(emitted) != {'intended', 'unrelated'}:
    raise AnalysisError(f'{label} invalid OOD raw movement is missing.')
  for call, field in (
      ('intended', 'intended_target_readout'),
      ('unrelated', 'unrelated_target_readout'),
  ):
    movements = _raw_movements(parsed_readouts[field])['movements']
    if not isinstance(emitted.get(call), Mapping) or set(emitted[call]) != set(movements):
      raise AnalysisError(f'{label} invalid OOD {call} movement schema changed.')
    for direction, expected in movements.items():
      _same_number(
          emitted[call].get(direction), expected,
          f'{label}.{call}.{direction}',
      )
  if (
      record.get('same_eight_row_compiled_executable') is not True
      or record.get('eight_row_executable_fingerprint') != eight_fingerprint
  ):
    raise AnalysisError(f'{label} invalid OOD executable changed.')
  _require_seconds(
      record.get('seconds'),
      {'intended', 'intended_repeat', 'unrelated', 'unrelated_repeat'},
      f'{label}.seconds',
  )


def _validate_protocol_bindings() -> dict[str, str]:
  if _sha256(_PROTOCOL_PATH) != PROTOCOL_SHA256:
    raise AnalysisError('Authoritative v3.3 protocol bytes changed.')
  if _sha256(_SUPERSESSION_PATH) != SUPERSESSION_SHA256:
    raise AnalysisError('v3.3 supersession record bytes changed.')
  return {
      'protocol_path': str(_PROTOCOL_PATH),
      'protocol_sha256': PROTOCOL_SHA256,
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'ordering_clarification_commit': ORDERING_CLARIFICATION_COMMIT,
      'capacity_clarification_commit': CAPACITY_CLARIFICATION_COMMIT,
      'upstream_provenance_amendment_commit': (
          UPSTREAM_PROVENANCE_AMENDMENT_COMMIT
      ),
      'expected_raw_record_count': EXPECTED_RAW_RECORDS,
      'expected_model_apply_count': EXPECTED_MODEL_APPLIES,
      'supersession_path': str(_SUPERSESSION_PATH),
      'supersession_sha256': SUPERSESSION_SHA256,
      'supersession_commit': SUPERSESSION_COMMIT,
      'superseded_plan_is_not_a_gate': True,
  }


def _validate_manifest(
    run_dir: Path, cases: Mapping[int, Mapping[str, Any]], *,
    expected_order: Sequence[tuple[str, int, int | None]] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
  manifest_path = run_dir / 'RAW_MANIFEST.json'
  manifest = _read_json(manifest_path)
  if expected_order is None:
    expected_order = _expected_execution_order()
  expected_paths = {
      _artifact_relative(family, cases[order], coalition_id)
      for family, order, coalition_id in expected_order
  }
  raw_root = run_dir / 'raw'
  if raw_root.is_symlink() or not raw_root.is_dir():
    raise AnalysisError('Raw artifact root is missing or symlinked.')
  observed_files = set()
  expected_directories = {raw_root.resolve()}
  for relative in expected_paths:
    path = (run_dir / relative).resolve()
    expected_directories.update(path.parents)
  for lexical in raw_root.rglob('*'):
    _guard_path(lexical)
    if lexical.is_symlink():
      raise AnalysisError(f'Raw artifact tree contains a symlink: {lexical}.')
    if lexical.is_file():
      observed_files.add(str(lexical.relative_to(run_dir)))
    elif lexical.is_dir():
      if lexical.resolve() not in expected_directories:
        raise AnalysisError(f'Raw artifact tree contains an extra directory: {lexical}.')
    else:
      raise AnalysisError(f'Raw artifact tree contains a special entry: {lexical}.')
  if observed_files != expected_paths:
    raise AnalysisError('Raw tree does not contain exactly the frozen execution prefix.')
  mapping = manifest.get('artifact_sha256')
  if not isinstance(mapping, Mapping) or set(mapping) != expected_paths:
    raise AnalysisError('RAW_MANIFEST does not enumerate the exact raw tree.')
  clean_mapping = {}
  for relative in sorted(expected_paths):
    digest = mapping[relative]
    if not _is_sha256(digest) or _sha256(run_dir / relative) != digest:
      raise AnalysisError(f'RAW_MANIFEST hash mismatch: {relative}.')
    clean_mapping[relative] = digest
  paths = [run_dir / relative for relative in expected_paths]
  if manifest.get('artifact_count') != len(expected_paths):
    raise AnalysisError('RAW_MANIFEST count differs from the frozen execution prefix.')
  if manifest.get('artifact_tree_sha256') != _tree_digest(paths, run_dir):
    raise AnalysisError('RAW_MANIFEST tree digest mismatch.')
  return manifest, clean_mapping


def _validate_top_level_tree(run_dir: Path, *, target_eligibility_exists: bool) -> None:
  expected_files = {
      'ATTEMPT_STARTED.json',
      'IMPORT_PROVENANCE_PRE_MODEL.json',
      'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
      'IMPORT_PROVENANCE.json',
      'PROTOBUF_PROVENANCE.json',
      'RAW_MANIFEST.json',
      'RUN_COMPLETE.json',
  }
  if target_eligibility_exists:
    expected_files.add('TARGET_ELIGIBILITY.json')
  expected_directories = {'raw', 'compiler'}
  observed_files, observed_directories = set(), set()
  for path in run_dir.iterdir():
    _guard_path(path)
    if path.is_symlink():
      raise AnalysisError(f'Run root contains a symlink: {path}.')
    if path.is_file():
      observed_files.add(path.name)
    elif path.is_dir():
      observed_directories.add(path.name)
    else:
      raise AnalysisError(f'Run root contains a special entry: {path}.')
  if observed_files != expected_files or observed_directories != expected_directories:
    raise AnalysisError('Run root contains missing or extra artifacts.')
  compiler_dirs = run_dir / 'compiler'
  if {
      path.name for path in compiler_dirs.iterdir() if path.is_dir()
  } != {'six_row', 'eight_row'} or any(
      not path.is_dir() or path.is_symlink() for path in compiler_dirs.iterdir()
  ):
    raise AnalysisError('Compiler root is not exactly six_row/eight_row.')


def _validate_compiler(
    compiler: Any, *, run_dir: Path, executable_name: str
) -> str:
  expected_keys = {
      'executable_name', 'compile_count', 'compile_seconds',
      'executable_fingerprint', 'artifacts',
  }
  if not isinstance(compiler, Mapping) or set(compiler) != expected_keys:
    raise AnalysisError(f'{executable_name} compiler schema changed.')
  if compiler.get('executable_name') != executable_name or compiler.get('compile_count') != 1:
    raise AnalysisError(f'{executable_name} compile identity/count changed.')
  _finite(compiler.get('compile_seconds'), f'{executable_name}.compile_seconds')
  directory = run_dir / 'compiler' / executable_name
  provenance_path = directory / 'COMPILER_PROVENANCE.json'
  if _read_json(provenance_path) != dict(compiler):
    raise AnalysisError(f'{executable_name} compiler provenance file differs.')
  expected_names = {
      'stablehlo': 'graph.stablehlo.mlir',
      'hlo': 'graph.pre_backend.hlo.txt',
      'compiled_hlo': 'graph.compiled.hlo.txt',
  }
  artifacts = compiler.get('artifacts')
  if not isinstance(artifacts, Mapping) or set(artifacts) != set(expected_names):
    raise AnalysisError(f'{executable_name} must bind exactly three IR artifacts.')
  expected_files = {provenance_path.resolve()}
  for name, filename in expected_names.items():
    binding = artifacts[name]
    if not isinstance(binding, Mapping) or set(binding) != {'path', 'sha256', 'size_bytes'}:
      raise AnalysisError(f'{executable_name}.{name} compiler binding changed.')
    path = Path(str(binding.get('path'))).resolve()
    expected = (directory / filename).resolve()
    _guard_path(path)
    if path != expected:
      raise AnalysisError(f'{executable_name}.{name} compiler path changed.')
    if _sha256(path) != binding.get('sha256') or path.stat().st_size != binding.get('size_bytes'):
      raise AnalysisError(f'{executable_name}.{name} compiler bytes changed.')
    expected_files.add(path)
  observed = {path.resolve() for path in directory.iterdir() if path.is_file()}
  if observed != expected_files or any(path.is_dir() for path in directory.iterdir()):
    raise AnalysisError(f'{executable_name} compiler tree contains extra entries.')
  fingerprint = hashlib.sha256(
      bytes.fromhex(artifacts['compiled_hlo']['sha256'])
  ).hexdigest()
  if compiler.get('executable_fingerprint') != fingerprint:
    raise AnalysisError(f'{executable_name} executable fingerprint mismatch.')
  return fingerprint


def _validate_import_file(
    path: Path, expected_sha: Any, *, bundle_root: Path,
    upstream_inventory: Mapping[str, Any], upstream_head: str,
    upstream_exception: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
  if not _is_sha256(expected_sha) or _sha256(path) != expected_sha:
    raise AnalysisError(f'Import-provenance hash mismatch: {path.name}.')
  value = _read_json(path)
  _require_exact_keys(
      value, {'module_count', 'modules', 'upstream_source_attestation'}, path.name
  )
  upstream_root = (bundle_root.parent / 'alphagenome').resolve()
  expected_upstream_attestation = {
      'git_head': upstream_head,
      'tracked_head_clean': True,
      'imported_module_count': len(upstream_inventory),
      'imported_modules': {
          name: {
              **binding,
              'path': str((upstream_root / binding['relative_path']).resolve()),
              'source_kind': (
                  'generated_exact_byte_exception'
                  if name in UPSTREAM_GENERATED_MODULE_NAMES else 'tracked'
              ),
          }
          for name, binding in upstream_inventory.items()
      },
      'tracked_imported_module_count': 22,
      'generated_imported_module_count': 4,
      'generated_binding_exception': dict(upstream_exception),
  }
  if value.get('upstream_source_attestation') != expected_upstream_attestation:
    raise AnalysisError(f'{path.name} upstream source attestation changed.')
  modules = value.get('modules')
  if not isinstance(modules, list) or not modules or value.get('module_count') != len(modules):
    raise AnalysisError(f'{path.name} module count/list is invalid.')
  upstream = upstream_root
  by_name: dict[str, Mapping[str, Any]] = {}
  by_path: dict[str, list[str]] = defaultdict(list)
  for row in modules:
    if not isinstance(row, Mapping) or set(row) != {
        'name', 'path', 'root', 'sha256', 'size_bytes'
    }:
      raise AnalysisError(f'{path.name} import row schema changed.')
    name, path_value = row['name'], row['path']
    if not isinstance(name, str) or not isinstance(path_value, str) or name in by_name:
      raise AnalysisError(f'{path.name} has a duplicate/malformed module name.')
    module_path = Path(path_value).resolve()
    declared_root = {
        'alphagenome_research_checkout': bundle_root,
        'upstream_alphagenome_checkout': upstream,
    }.get(row['root'])
    if declared_root is None:
      raise AnalysisError(f'{path.name} import row has an unknown root.')
    try:
      module_path.relative_to(declared_root.resolve())
    except ValueError as error:
      raise AnalysisError(f'{path.name} import escaped its declared root.') from error
    _guard_path(module_path)
    if _sha256(module_path) != row['sha256'] or module_path.stat().st_size != row['size_bytes']:
      raise AnalysisError(f'{path.name} imported module bytes changed: {name}.')
    by_name[name] = row
    by_path[str(module_path)].append(name)
  observed_upstream = {}
  for name, row in by_name.items():
    if row['root'] == 'upstream_alphagenome_checkout':
      module_path = Path(row['path']).resolve()
      observed_upstream[name] = {
          'relative_path': str(module_path.relative_to(upstream)),
          'sha256': row['sha256'], 'size_bytes': row['size_bytes'],
      }
  if observed_upstream != dict(upstream_inventory):
    raise AnalysisError(f'{path.name} loaded upstream source inventory changed.')
  duplicates = {key: names for key, names in by_path.items() if len(names) > 1}
  for duplicate_path, names in duplicates.items():
    if set(names) != {'__main__', '__mp_main__'} or Path(duplicate_path).name != (
        'run_encoder_skip_factorial_v3_3.py'
    ):
      raise AnalysisError(f'{path.name} contains an unapproved duplicate path alias.')
    rows = [by_name[name] for name in names]
    if any(
        (row['sha256'], row['size_bytes'], row['root'])
        != (rows[0]['sha256'], rows[0]['size_bytes'], rows[0]['root'])
        for row in rows[1:]
    ):
      raise AnalysisError(f'{path.name} approved alias bytes differ.')
  return by_name


def _validate_imports(
    run_dir: Path, complete: Mapping[str, Any], *, bundle_root: Path,
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
  files = {
      'pre_model': 'IMPORT_PROVENANCE_PRE_MODEL.json',
      'post_model_precompile': 'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
      'postcompile': 'IMPORT_PROVENANCE.json',
  }
  bindings = complete.get('import_provenance_phases')
  if not isinstance(bindings, Mapping) or set(bindings) != set(files):
    raise AnalysisError('Import-provenance phase bindings are incomplete.')
  phases = {
      name: _validate_import_file(
          run_dir / filename, bindings[name], bundle_root=bundle_root,
          upstream_inventory=freeze['upstream_imported_modules'],
          upstream_head=freeze['upstream_alphagenome_git_head'],
          upstream_exception=freeze['upstream_generated_binding_exception'],
      ) for name, filename in files.items()
  }
  if complete.get('import_provenance_sha256') != bindings['postcompile']:
    raise AnalysisError('Final import-provenance binding differs.')
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
      raise AnalysisError('Import-provenance shared bytes changed across phases.')
    lazy[f'{earlier}_to_{later}'] = sorted(set(phases[later]) - set(phases[earlier]))
  required = {
      'alphagenome_research.model.model',
      'alphagenome_research.model.dna_model',
      'alphagenome_research.model.interpretability',
  }
  if not required.issubset(phases['postcompile']):
    raise AnalysisError('Final import provenance lacks required model modules.')
  return {
      'phase_sha256': dict(bindings),
      'module_counts': {name: len(rows) for name, rows in phases.items()},
      'lazy_additions': lazy,
      'stable_shared_module_bytes': True,
  }


def _validate_protobuf(
    run_dir: Path, complete: Mapping[str, Any], freeze: Mapping[str, Any]
) -> dict[str, Any]:
  expected = complete.get('protobuf_provenance_sha256')
  path = run_dir / 'PROTOBUF_PROVENANCE.json'
  if not _is_sha256(expected) or _sha256(path) != expected:
    raise AnalysisError('Protobuf-provenance hash binding mismatch.')
  value = _read_json(path)
  if value != freeze.get('protobuf_binding'):
    raise AnalysisError('Protobuf provenance differs from the freeze.')

  def visit(node: Any) -> None:
    if isinstance(node, Mapping):
      if isinstance(node.get('path'), str):
        bound = Path(node['path']).resolve()
        _guard_path(bound)
        if 'sha256' in node and _sha256(bound) != node['sha256']:
          raise AnalysisError('Protobuf generated/source hash changed.')
        if 'size_bytes' in node and bound.stat().st_size != node['size_bytes']:
          raise AnalysisError('Protobuf generated/source size changed.')
      for child in node.values():
        visit(child)
    elif isinstance(node, list):
      for child in node:
        visit(child)
  visit(value)
  generated = value.get('generated_outputs')
  if not isinstance(generated, Mapping) or len(generated) != 2:
    raise AnalysisError('Protobuf generated-output binding set changed.')
  expected_generated_names = {
      'calibration_scores_pb2.py', 'calibration_scores_pb2.pyi'
  }
  if {Path(path).name for path in generated} != expected_generated_names:
    raise AnalysisError('Protobuf generated-output names changed.')
  for path_value, binding in generated.items():
    if not isinstance(path_value, str) or not isinstance(binding, Mapping):
      raise AnalysisError('Protobuf generated-output row is malformed.')
    generated_path = Path(path_value).resolve()
    _guard_path(generated_path)
    if set(binding) != {'sha256', 'size_bytes'}:
      raise AnalysisError('Protobuf generated-output binding schema changed.')
    if (
        _sha256(generated_path) != binding['sha256']
        or generated_path.stat().st_size != binding['size_bytes']
    ):
      raise AnalysisError('Protobuf generated-output bytes changed.')
  if (
      value.get('regeneration_claim') is not False
      or value.get('current_protoc_was_used_to_generate_frozen_outputs') is not False
  ):
    raise AnalysisError('Protobuf provenance makes a false regeneration claim.')
  return {'sha256': expected, 'binding_verified': True}


def _validate_device_observation(value: Any, label: str) -> None:
  if not isinstance(value, Mapping):
    raise AnalysisError(f'{label} device observation is absent.')
  if value.get('jax_default_backend') != 'gpu':
    raise AnalysisError(f'{label} JAX backend is not GPU.')
  devices = value.get('jax_gpu_devices')
  if not isinstance(devices, list) or len(devices) != 1:
    raise AnalysisError(f'{label} must expose exactly one JAX GPU.')
  device = devices[0]
  if not isinstance(device, Mapping) or (
      device.get('device_kind') != EXPECTED_DEVICE_KIND
      or device.get('platform') != 'gpu'
      or device.get('client_platform') != 'gpu'
  ):
    raise AnalysisError(f'{label} JAX device is not the frozen RTX 3090.')
  nvidia = value.get('nvidia_smi')
  parsed = nvidia.get('parsed_single_gpu') if isinstance(nvidia, Mapping) else None
  if not isinstance(parsed, Mapping) or (
      parsed.get('name') != EXPECTED_DEVICE_KIND
      or parsed.get('uuid') != EXPECTED_GPU_UUID
      or str(parsed.get('compute_capability')) != EXPECTED_COMPUTE_CAPABILITY
  ):
    raise AnalysisError(f'{label} nvidia-smi GPU/UUID/CC changed.')
  environment = value.get('environment')
  if not isinstance(environment, Mapping):
    raise AnalysisError(f'{label} environment observation is absent.')
  ld = environment.get('LD_LIBRARY_PATH')
  if not isinstance(ld, Mapping) or ld.get('present') is not False or ld.get('value') is not None:
    raise AnalysisError(f'{label} observed LD_LIBRARY_PATH as present.')
  if environment.get('XLA_PYTHON_CLIENT_PREALLOCATE') != 'false':
    raise AnalysisError(f'{label} preallocation setting changed.')
  if value.get('no_jit_no_array_no_model') is not True:
    raise AnalysisError(f'{label} did not attest the no-JIT preflight boundary.')


def _validate_bundle_and_preflight(
    start: Mapping[str, Any], freeze: Mapping[str, Any], freeze_sha: str,
    *, bundle_root: Path,
) -> dict[str, Any]:
  bundle = start.get('bundle')
  expected_generated = [
      '?? src/alphagenome_research/protos/calibration_scores_pb2.py',
      '?? src/alphagenome_research/protos/calibration_scores_pb2.pyi',
  ]
  if not isinstance(bundle, Mapping) or set(bundle) != {
      'git_head', 'tracked_clean', 'generated_artifact_exception'
  }:
    raise AnalysisError('START committed-bundle attestation schema changed.')
  if (
      bundle.get('tracked_clean') is not True
      or bundle.get('generated_artifact_exception') != expected_generated
      or not isinstance(bundle.get('git_head'), str)
      or len(bundle['git_head']) != 40
  ):
    raise AnalysisError('START committed-bundle gate failed.')
  bootstrap = start['same_process_pre_import_bootstrap']
  if bootstrap['freeze']['git_head'] != bundle['git_head']:
    raise AnalysisError('Bootstrap and bundle Git HEAD differ.')
  same = start.get('same_process_preflight')
  _validate_device_observation(same, 'same-process preflight')
  if same.get('pid') != start['same_process_pre_import_bootstrap']['pid']:
    raise AnalysisError('Same-process preflight PID differs from bootstrap PID.')
  external = start.get('external_preflight')
  if not isinstance(external, Mapping):
    raise AnalysisError('External preflight binding is absent.')
  path_value, digest = external.get('path'), external.get('sha256')
  if not isinstance(path_value, str) or not _is_sha256(digest):
    raise AnalysisError('External preflight path/hash is malformed.')
  path = Path(path_value).resolve()
  _guard_path(path)
  if _sha256(path) != digest:
    raise AnalysisError('External preflight artifact hash changed.')
  raw = _read_json(path)
  embedded = {
      key: value for key, value in external.items()
      if key not in {'path', 'sha256', 'validated_logs'}
  }
  if raw != embedded:
    raise AnalysisError('Embedded external preflight differs from its artifact.')
  if (
      raw.get('script_version') != 'opensplice-device-preflight-v3.3.0'
      or raw.get('status') != 'pass'
      or raw.get('protocol_sha256') != PROTOCOL_SHA256
      or raw.get('freeze_sha256') != freeze_sha
      or raw.get('failure') is not None
      or raw.get('no_model_or_biological_access') is not True
      or raw.get('no_jit_or_array_kernel') is not True
  ):
    raise AnalysisError('External preflight contract did not pass exactly.')
  preflight_dir = Path(str(freeze.get('preflight_dir'))).resolve()
  if path.parent != preflight_dir:
    raise AnalysisError('External preflight escaped the frozen directory.')
  logs, validated = raw.get('logs'), external.get('validated_logs')
  if not isinstance(logs, Mapping) or set(logs) != {'stdout', 'stderr'}:
    raise AnalysisError('External preflight logs are incomplete.')
  expected_validated = {}
  for stream in ('stdout', 'stderr'):
    binding = logs[stream]
    if not isinstance(binding, Mapping) or set(binding) != {'path', 'sha256'}:
      raise AnalysisError(f'External preflight {stream} log schema changed.')
    log_path = Path(str(binding['path'])).resolve()
    if log_path.parent != preflight_dir or _sha256(log_path) != binding['sha256']:
      raise AnalysisError(f'External preflight {stream} log changed.')
    expected_validated[stream] = {'path': str(log_path), 'sha256': binding['sha256']}
  if validated != expected_validated:
    raise AnalysisError('External preflight validated-log binding changed.')
  observation = raw.get('observation')
  _validate_device_observation(observation, 'external preflight')
  if (
      observation.get('jax_enable_compilation_cache') is not False
      or observation.get('v3_3_runtime_environment') != start.get('runtime_environment')
  ):
    raise AnalysisError('External/same-process runtime environment changed.')
  return {
      'git_head': bundle['git_head'],
      'external_preflight_sha256': digest,
      'external_exact_rtx3090_uuid_gate': True,
      'same_process_exact_rtx3090_uuid_gate': True,
  }


def _validate_checkpoint_reference(
    start: Mapping[str, Any], freeze: Mapping[str, Any],
    cases: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
  manifest_path = Path(str(freeze.get('checkpoint_manifest_path'))).resolve()
  reference_path = Path(str(freeze.get('reference_bindings_path'))).resolve()
  for path in (manifest_path, reference_path):
    _guard_path(path)
  if (
      _sha256(manifest_path) != freeze.get('checkpoint_manifest_sha256')
      or _sha256(reference_path) != freeze.get('reference_bindings_sha256')
  ):
    raise AnalysisError('Checkpoint/reference manifest bytes changed.')
  checkpoint_path = Path(str(start.get('checkpoint_path'))).resolve()
  if (
      checkpoint_path.name != freeze.get('checkpoint_snapshot')
      or not checkpoint_path.is_dir()
  ):
    raise AnalysisError('Checkpoint snapshot path/name changed.')
  records = []
  for number, line in enumerate(
      manifest_path.read_text(encoding='utf-8').splitlines(), start=1
  ):
    fields = line.split('\t')
    if len(fields) != 3:
      raise AnalysisError(f'Checkpoint manifest row {number} is malformed.')
    relative, size_text, digest = fields
    relative_path = Path(relative)
    if (
        not relative or relative_path.is_absolute() or '..' in relative_path.parts
        or not _is_sha256(digest)
    ):
      raise AnalysisError(f'Checkpoint manifest row {number} is unsafe.')
    try:
      size = int(size_text)
    except ValueError as error:
      raise AnalysisError('Checkpoint manifest size is not an integer.') from error
    if size < 0 or str(size) != size_text:
      raise AnalysisError('Checkpoint manifest size is noncanonical.')
    records.append({'relative_path': relative, 'size_bytes': size, 'sha256': digest})
  if len(records) != 12 or [row['relative_path'] for row in records] != sorted(
      row['relative_path'] for row in records
  ):
    raise AnalysisError('Checkpoint manifest is not exactly 12 sorted rows.')
  if start.get('checkpoint_binding') != {
      'snapshot_path': str(checkpoint_path),
      'snapshot_name': checkpoint_path.name,
      'manifest_path': str(manifest_path),
      'manifest_sha256': freeze['checkpoint_manifest_sha256'],
      'file_count': 12,
      'files': records,
  }:
    raise AnalysisError('START checkpoint binding differs from the manifest.')
  model_cache = checkpoint_path.parent.parent
  blobs_root = (model_cache / 'blobs').resolve()
  expected_relatives = {row['relative_path'] for row in records}
  allowed_directories = {
      str(parent) for relative in expected_relatives for parent in Path(relative).parents
      if str(parent) != '.'
  }
  for row in records:
    relative_path = Path(row['relative_path'])
    lexical = checkpoint_path / relative_path
    try:
      resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as error:
      raise AnalysisError(f'Checkpoint entry cannot resolve: {relative_path}.') from error
    if not stat.S_ISREG(resolved.stat().st_mode):
      raise AnalysisError(f'Checkpoint target is not regular: {relative_path}.')
    if lexical.is_symlink():
      link_text = os.readlink(lexical)
      parts = Path(link_text).parts
      blob = parts[-1] if parts else ''
      expected_parts = ('..',) * (len(relative_path.parts) + 1) + ('blobs', blob)
      direct = Path(os.path.abspath(lexical.parent / link_text))
      if (
          Path(link_text).is_absolute() or parts != expected_parts
          or len(blob) not in (40, 64)
          or any(character not in '0123456789abcdef' for character in blob)
          or direct.parent.resolve() != blobs_root or direct.is_symlink()
          or resolved != direct or resolved.parent != blobs_root
      ):
        raise AnalysisError(f'Checkpoint symlink escaped pinned blobs: {relative_path}.')
    else:
      try:
        resolved.relative_to(checkpoint_path)
      except ValueError as error:
        raise AnalysisError('Checkpoint regular file escaped snapshot.') from error
    if resolved.stat().st_size != row['size_bytes'] or _sha256(resolved) != row['sha256']:
      raise AnalysisError(f'Checkpoint bytes changed: {relative_path}.')
  observed_files = set()
  for path in checkpoint_path.rglob('*'):
    relative = str(path.relative_to(checkpoint_path))
    if path.is_symlink() or path.is_file():
      observed_files.add(relative)
    elif path.is_dir() and relative not in allowed_directories:
      raise AnalysisError(f'Checkpoint contains extra directory: {relative}.')
    elif not path.is_dir():
      raise AnalysisError(f'Checkpoint contains special entry: {relative}.')
  if observed_files != expected_relatives:
    raise AnalysisError('Checkpoint tree differs from exact manifest.')

  reference = _read_json(reference_path)
  if set(reference) != {'reference_url', 'context_bp', 'cases'} or (
      reference.get('reference_url') != freeze.get('reference_url')
      or reference.get('context_bp') != 16_384
      or not isinstance(reference.get('cases'), Mapping)
  ):
    raise AnalysisError('Reference-sequence binding schema changed.')
  if set(reference['cases']) != {case['variant_id'] for case in cases.values()}:
    raise AnalysisError('Reference binding does not contain the exact 20 cases.')
  sequence_bindings = {}
  for order, case in cases.items():
    row = reference['cases'][case['variant_id']]
    center = (case['exon_start_1based'] + case['exon_end_1based']) // 2
    expected_start = center - 1 - 8192
    if (
        not isinstance(row, list) or len(row) != 6
        or row[:4] != [order, case['chromosome'], expected_start, expected_start + 16384]
        or not _is_sha256(row[4]) or not _is_sha256(row[5]) or row[4] == row[5]
    ):
      raise AnalysisError(f'Reference sequence binding changed for order {order}.')
    sequence_bindings[order] = {'reference': row[4], 'alternate': row[5]}
  if start.get('reference_sequence_bindings') != {
      'path': str(reference_path), 'sha256': freeze['reference_bindings_sha256']
  }:
    raise AnalysisError('START reference-sequence file binding changed.')
  reference_object = freeze.get('reference_object')
  if not isinstance(reference_object, Mapping):
    raise AnalysisError('Frozen reference object is absent.')
  expected_object_binding = {
      **reference_object,
      'observed_generation': reference_object.get('generation'),
      'observed_size_bytes': reference_object.get('size_bytes'),
      'observed_etag': reference_object.get('etag'),
      'observed_md5_base64': reference_object.get('md5_base64'),
      'observed_crc32c_base64': reference_object.get('crc32c_base64'),
  }
  if start.get('reference_object_binding') != expected_object_binding:
    raise AnalysisError('START reference-object metadata binding changed.')
  return {
      'checkpoint_file_count': 12,
      'checkpoint_manifest_sha256': freeze['checkpoint_manifest_sha256'],
      'reference_bindings_sha256': freeze['reference_bindings_sha256'],
      'sequence_bindings': sequence_bindings,
  }


def _validate_prior_v3_2_evidence(value: Any) -> dict[str, Any]:
  if value != EXPECTED_PRIOR_V3_2_EVIDENCE:
    raise AnalysisError('Prior v3.2 evidence binding differs from protocol section 1.')
  for name in ('protocol', 'analysis', 'result'):
    path = Path(value[name]['path']).resolve()
    _guard_path(path)
    if _sha256(path) != value[name]['sha256']:
      raise AnalysisError(f'Prior v3.2 {name} bytes changed.')
  raw = value['raw_manifest']
  raw_path = Path(raw['path']).resolve()
  _guard_path(raw_path)
  if _sha256(raw_path) != raw['sha256']:
    raise AnalysisError('Prior v3.2 raw-manifest bytes changed.')
  manifest = _read_json(raw_path)
  mapping = manifest.get('artifact_sha256')
  if (
      manifest.get('artifact_count') != 2660
      or manifest.get('artifact_tree_sha256') != raw['artifact_tree_sha256']
      or not isinstance(mapping, Mapping) or len(mapping) != 2660
      or any(not _is_sha256(digest) for digest in mapping.values())
  ):
    raise AnalysisError('Prior v3.2 raw-manifest identity changed.')
  run_dir = raw_path.parent.resolve()
  expected_paths = set(mapping)
  artifact_paths = []
  for relative, digest in mapping.items():
    if (
        not isinstance(relative, str) or Path(relative).is_absolute()
        or '..' in Path(relative).parts
    ):
      raise AnalysisError('Prior v3.2 artifact path is unsafe.')
    lexical = run_dir / relative
    resolved = lexical.resolve()
    try:
      resolved.relative_to(run_dir)
    except ValueError as error:
      raise AnalysisError('Prior v3.2 artifact escaped its run directory.') from error
    if lexical.is_symlink() or not lexical.is_file() or _sha256(lexical) != digest:
      raise AnalysisError(f'Prior v3.2 artifact bytes changed: {relative}.')
    artifact_paths.append(lexical)
  raw_root = run_dir / 'raw'
  observed = set()
  for lexical in raw_root.rglob('*'):
    if lexical.is_symlink() or (not lexical.is_file() and not lexical.is_dir()):
      raise AnalysisError('Prior v3.2 raw tree contains a symlink/special entry.')
    if lexical.is_file():
      observed.add(str(lexical.relative_to(run_dir)))
  if observed != expected_paths:
    raise AnalysisError('Prior v3.2 raw tree contains missing or extra artifacts.')
  if _tree_digest(artifact_paths, run_dir) != raw['artifact_tree_sha256']:
    raise AnalysisError('Prior v3.2 raw artifact tree digest changed.')
  return {
      'prior_v3_2_protocol_sha256': value['protocol']['sha256'],
      'prior_v3_2_raw_manifest_sha256': raw['sha256'],
      'prior_v3_2_raw_artifact_count': 2660,
      'prior_v3_2_raw_tree_sha256': raw['artifact_tree_sha256'],
      'prior_v3_2_analysis_sha256': value['analysis']['sha256'],
      'prior_v3_2_result_sha256': value['result']['sha256'],
  }


def _validate_runtime_manifest(
    start: Mapping[str, Any], freeze: Mapping[str, Any]
) -> dict[str, Any]:
  manifest = freeze.get('runtime_version_manifest')
  if not isinstance(manifest, Mapping) or set(manifest) != {
      'python_version', 'platform', 'kernel', 'packages', 'nvidia_smi'
  }:
    raise AnalysisError('Frozen runtime-version manifest schema changed.')
  packages = manifest.get('packages')
  if (
      not isinstance(packages, Mapping) or set(packages) != set(RUNTIME_PACKAGES)
      or any(not isinstance(value, str) or not value for value in packages.values())
      or not all(isinstance(manifest.get(key), str) and manifest[key]
                 for key in ('python_version', 'platform', 'kernel'))
      or not isinstance(manifest.get('nvidia_smi'), Mapping)
  ):
    raise AnalysisError('Frozen runtime-version manifest is malformed.')
  binding = {
      key: manifest[key] for key in ('python_version', 'platform', 'kernel', 'packages')
  }
  if start.get('runtime_version_binding') != binding:
    raise AnalysisError('START runtime-version binding differs from freeze.')
  for label, observation in (
      ('same-process', start.get('same_process_preflight')),
      ('external', start.get('external_preflight', {}).get('observation')),
  ):
    if not isinstance(observation, Mapping):
      raise AnalysisError(f'{label} runtime observation is absent.')
    observed_packages = observation.get('packages')
    if not isinstance(observed_packages, Mapping) or any(
        observed_packages.get(name) != packages[name] for name in RUNTIME_PACKAGES
    ):
      raise AnalysisError(f'{label} GPU/JAX package versions changed.')
    if (
        str(observation.get('python_version', '')).split()[0]
        != manifest['python_version']
        or observation.get('platform') != manifest['platform']
        or observation.get('kernel') != manifest['kernel']
        or observation.get('nvidia_smi', {}).get('parsed_single_gpu')
        != manifest['nvidia_smi']
    ):
      raise AnalysisError(f'{label} runtime/platform/GPU versions changed.')
  return {
      'runtime_version_manifest_verified': True,
      'runtime_package_versions': dict(packages),
  }


def _validate_upstream_checkout(
    start: Mapping[str, Any], freeze: Mapping[str, Any], *, bundle_root: Path
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
  inventory = freeze.get('upstream_imported_modules')
  if not isinstance(inventory, Mapping) or len(inventory) != 26:
    raise AnalysisError('Frozen upstream imported-module inventory changed.')
  upstream_root = (bundle_root.parent / 'alphagenome').resolve()
  head = freeze.get('upstream_alphagenome_git_head')
  if (
      not isinstance(head, str) or len(head) != 40
      or any(character not in '0123456789abcdef' for character in head)
  ):
    raise AnalysisError('Frozen upstream Git HEAD is malformed.')
  try:
    current_head = subprocess.check_output(
        ('git', '-C', str(upstream_root), 'rev-parse', 'HEAD'), text=True
    ).strip()
    dirty = subprocess.check_output(
        ('git', '-C', str(upstream_root), 'diff', '--binary', 'HEAD', '--')
    )
  except (OSError, subprocess.CalledProcessError) as error:
    raise AnalysisError('Cannot audit upstream AlphaGenome checkout.') from error
  if current_head != head or dirty:
    raise AnalysisError('Upstream AlphaGenome checkout HEAD/clean gate changed.')
  exception = freeze.get('upstream_generated_binding_exception')
  if not isinstance(exception, Mapping) or set(exception) != {
      'module_names', 'generated_outputs', 'source_inputs',
      'embedded_headers', 'protobuf_runtime_version',
      'grpcio_runtime_version', 'grpcio_tools', 'historical_generator',
      'historical_generator_argv', 'exact_regeneration_claim',
  }:
    raise AnalysisError('Frozen upstream generated-code exception schema changed.')
  packages = freeze.get('runtime_version_manifest', {}).get('packages', {})
  if (
      exception.get('module_names') != list(UPSTREAM_GENERATED_MODULE_NAMES)
      or exception.get('generated_outputs') != UPSTREAM_GENERATED_OUTPUTS
      or exception.get('source_inputs') != UPSTREAM_GENERATED_SOURCE_INPUTS
      or exception.get('embedded_headers') != UPSTREAM_GENERATED_HEADERS
      or exception.get('protobuf_runtime_version') != packages.get('protobuf')
      or exception.get('grpcio_runtime_version') != packages.get('grpcio')
      or exception.get('grpcio_tools') != 'unavailable_not_used'
      or exception.get('historical_generator') != 'unknown'
      or exception.get('historical_generator_argv') != 'unknown'
      or exception.get('exact_regeneration_claim') is not False
  ):
    raise AnalysisError('Frozen upstream generated-code exception claims changed.')
  if any(inventory.get(name) != binding for name, binding in UPSTREAM_GENERATED_OUTPUTS.items()):
    raise AnalysisError('Frozen upstream generated outputs differ from protocol table.')
  for relative, binding in UPSTREAM_GENERATED_SOURCE_INPUTS.items():
    path = (upstream_root / relative).resolve()
    try:
      subprocess.run(
          ('git', '-C', str(upstream_root), 'ls-files', '--error-unmatch', relative),
          check=True, capture_output=True,
      )
    except (OSError, subprocess.CalledProcessError) as error:
      raise AnalysisError(f'Upstream generated source input is untracked: {relative}.') from error
    if (
        path.is_symlink() or not path.is_file()
        or _sha256(path) != binding['sha256']
        or path.stat().st_size != binding['size_bytes']
    ):
      raise AnalysisError(f'Upstream generated source input changed: {relative}.')
  expected_absolute = {}
  tracked_count = generated_count = 0
  for module_name, binding in sorted(inventory.items()):
    if (
        not isinstance(module_name, str)
        or not (module_name == 'alphagenome' or module_name.startswith('alphagenome.'))
        or not isinstance(binding, Mapping) or set(binding) != {
            'relative_path', 'sha256', 'size_bytes'
        }
    ):
      raise AnalysisError('Frozen upstream module binding schema changed.')
    relative = binding['relative_path']
    if (
        not isinstance(relative, str) or Path(relative).is_absolute()
        or '..' in Path(relative).parts or not _is_sha256(binding['sha256'])
        or not isinstance(binding['size_bytes'], int)
        or isinstance(binding['size_bytes'], bool) or binding['size_bytes'] < 0
    ):
      raise AnalysisError(f'Frozen upstream binding is malformed: {module_name}.')
    path = (upstream_root / relative).resolve()
    try:
      normalized = str(path.relative_to(upstream_root))
    except ValueError as error:
      raise AnalysisError('Upstream module escaped checkout.') from error
    if normalized != relative or path.suffix != '.py':
      raise AnalysisError(f'Upstream module path changed: {module_name}.')
    tracked = subprocess.run(
        ('git', '-C', str(upstream_root), 'ls-files', '--error-unmatch', relative),
        capture_output=True,
    ).returncode == 0
    is_generated = module_name in UPSTREAM_GENERATED_MODULE_NAMES
    if is_generated:
      generated_count += 1
      if tracked or exception['generated_outputs'].get(module_name) != binding:
        raise AnalysisError(f'Upstream generated-output contract changed: {module_name}.')
      try:
        ignored = subprocess.check_output(
            ('git', '-C', str(upstream_root), 'check-ignore', '-v', relative),
            text=True,
        ).strip()
      except (OSError, subprocess.CalledProcessError) as error:
        raise AnalysisError(f'Upstream generated output is not ignored: {module_name}.') from error
      if not ignored.endswith(f'**/*pb2*.py\t{relative}'):
        raise AnalysisError(f'Upstream generated ignore rule changed: {module_name}.')
    else:
      tracked_count += 1
      if not tracked:
        raise AnalysisError(f'Upstream source module is untracked: {module_name}.')
    if (
        path.is_symlink() or not path.is_file()
        or _sha256(path) != binding['sha256']
        or path.stat().st_size != binding['size_bytes']
    ):
      raise AnalysisError(f'Upstream imported source changed: {module_name}.')
    expected_absolute[module_name] = {
        **binding, 'path': str(path),
        'source_kind': (
            'generated_exact_byte_exception' if is_generated else 'tracked'
        ),
    }
    if is_generated:
      text = path.read_text(encoding='utf-8').splitlines()
      if any(line not in text for line in UPSTREAM_GENERATED_HEADERS[module_name]):
        raise AnalysisError(f'Upstream generated header changed: {module_name}.')
  if (tracked_count, generated_count) != (22, 4):
    raise AnalysisError('Upstream imported-source 22+4 split changed.')
  bootstrap = start['same_process_pre_import_bootstrap']['freeze']
  expected_attestation = {
      'git_head': head, 'tracked_head_clean': True,
      'imported_module_count': 26, 'imported_modules': expected_absolute,
      'tracked_imported_module_count': 22,
      'generated_imported_module_count': 4,
      'generated_binding_exception': dict(exception),
  }
  if bootstrap.get('upstream_checkout') != expected_attestation:
    raise AnalysisError('Bootstrap upstream-checkout attestation changed.')
  return dict(inventory), {
      'upstream_git_head': head,
      'upstream_tracked_head_clean': True,
      'upstream_imported_module_count': 26,
      'upstream_tracked_imported_module_count': 22,
      'upstream_generated_imported_module_count': 4,
      'upstream_generated_exception_verified': True,
  }


def _validate_start(
    run_dir: Path, *, bundle_root: Path,
    cases: Mapping[int, Mapping[str, Any]] | None = None,
    validate_external_inputs: bool = True,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
  start = _read_json(run_dir / 'ATTEMPT_STARTED.json')
  _require_exact_keys(start, {
      'attempt_id', 'script_version', 'status', 'protocol_sha256',
      'supersession_sha256', 'supersession_commit', 'supersession', 'freeze',
      'bundle', 'external_preflight', 'same_process_preflight',
      'runtime_environment', 'runtime_version_binding',
      'prior_v3_2_evidence', 'same_process_pre_import_bootstrap',
      'checkpoint_path', 'checkpoint_binding', 'reference_object_binding',
      'reference_sequence_bindings', 'compile_count_contract',
      'compile_count_by_executable', 'scientific_record_count_contract',
      'model_apply_count_contract', 'execution_order_contract',
      'max_wall_time_seconds', 'max_output_bytes', 'confirmation_model_calls',
      'confirmation_scope_disclosure', 'started_at_unix_s',
  }, 'ATTEMPT_STARTED')
  required = {
      'attempt_id': ATTEMPT_ID,
      'script_version': SOURCE_SCRIPT_VERSION,
      'status': 'started_append_only_one_shot',
      'protocol_sha256': PROTOCOL_SHA256,
      'supersession_sha256': SUPERSESSION_SHA256,
      'supersession_commit': 'c64def4',
      'compile_count_contract': 2,
      'scientific_record_count_contract': EXPECTED_RAW_RECORDS,
      'model_apply_count_contract': EXPECTED_MODEL_APPLIES,
      'confirmation_model_calls': 0,
      'max_wall_time_seconds': MAX_WALL_TIME_SECONDS,
      'max_output_bytes': MAX_OUTPUT_BYTES,
  }
  for key, expected in required.items():
    if start.get(key) != expected:
      raise AnalysisError(f'ATTEMPT_STARTED.{key} differs from the frozen contract.')
  if start.get('supersession') != {
      'path': str(_SUPERSESSION_PATH.resolve()),
      'sha256': SUPERSESSION_SHA256,
      'commit': 'c64def4',
  }:
    raise AnalysisError('ATTEMPT_STARTED supersession binding changed.')
  if start.get('compile_count_by_executable') != {'six_row': 1, 'eight_row': 1}:
    raise AnalysisError('ATTEMPT_STARTED per-executable compile contract changed.')
  expected_execution = {
      'identities': 'manifest orders 0..19',
      'coalition_anchors': 'ID0 then ID255 for orders 0..19',
      'remaining_effects': 'orders 0..5,10..15; each IDs1..254 increasing',
      'remaining_neutrals': 'orders 6..9,16..19; each IDs1..254 increasing',
      'ood': 'orders0..19; IDs0,127,128,255',
  }
  if start.get('execution_order_contract') != expected_execution:
    raise AnalysisError('ATTEMPT_STARTED execution order contract changed.')
  freeze_binding = start.get('freeze')
  if not isinstance(freeze_binding, Mapping):
    raise AnalysisError('ATTEMPT_STARTED freeze binding is absent.')
  freeze_path_value, freeze_sha = (
      freeze_binding.get('path'), freeze_binding.get('sha256')
  )
  if not isinstance(freeze_path_value, str) or not _is_sha256(freeze_sha):
    raise AnalysisError('ATTEMPT_STARTED freeze path/hash is malformed.')
  freeze_path = Path(freeze_path_value).resolve()
  _guard_path(freeze_path)
  if _sha256(freeze_path) != freeze_sha:
    raise AnalysisError('Frozen v3.3 bundle file hash changed.')
  freeze_file = _read_json(freeze_path)
  embedded = {
      key: value for key, value in freeze_binding.items() if key not in {'path', 'sha256'}
  }
  if freeze_file != embedded:
    raise AnalysisError('Embedded v3.3 freeze differs from its bound file.')
  freeze_required = {
      'attempt_id': ATTEMPT_ID,
      'script_version': SOURCE_SCRIPT_VERSION,
      'protocol_sha256': PROTOCOL_SHA256,
      'supersession_sha256': SUPERSESSION_SHA256,
      'supersession_commit': 'c64def4',
      'scientific_record_count': EXPECTED_RAW_RECORDS,
      'model_apply_count': EXPECTED_MODEL_APPLIES,
      'identity_record_count': EXPECTED_IDENTITIES,
      'coalition_record_count': EXPECTED_COALITIONS,
      'ood_anchor_record_count': EXPECTED_OOD,
      'six_row_compile_count': 1,
      'eight_row_compile_count': 1,
      'effect_order': list(EFFECT_ORDERS),
      'neutral_order': list(NEUTRAL_ORDERS),
      'gate0_anchor_ids': [0, 255],
      'ood_anchor_ids': list(OOD_ANCHORS),
      'coalition_bit_order': list(PLAYERS_E) + ['T'],
      'shapley_player_order': list(PLAYERS_8),
      'shapley_absolute_tolerance': SHAPLEY_ABS_TOLERANCE,
      'shapley_relative_tolerance': SHAPLEY_REL_TOLERANCE,
      'max_wall_time_seconds': MAX_WALL_TIME_SECONDS,
      'max_output_bytes': MAX_OUTPUT_BYTES,
      'protocol_path': str(_PROTOCOL_PATH.resolve()),
      'supersession_path': str(_SUPERSESSION_PATH.resolve()),
  }
  for key, expected in freeze_required.items():
    if freeze_file.get(key) != expected:
      raise AnalysisError(f'Frozen v3.3 contract changed at {key}.')
  output_dir = freeze_file.get('output_dir')
  analysis_dir = freeze_file.get('analysis_dir')
  if not isinstance(output_dir, str) or Path(output_dir).resolve() != run_dir:
    raise AnalysisError('Run directory differs from frozen output_dir.')
  if not isinstance(analysis_dir, str):
    raise AnalysisError('Frozen analysis_dir is absent.')
  analysis_path = Path(analysis_dir).resolve()
  _guard_path(analysis_path)
  if analysis_path == run_dir or run_dir in analysis_path.parents:
    raise AnalysisError('Analysis destination is inside the append-only raw tree.')
  file_hashes = freeze_file.get('file_sha256')
  if not isinstance(file_hashes, Mapping) or not file_hashes:
    raise AnalysisError('Freeze contains no implementation hash inventory.')
  for relative, digest in file_hashes.items():
    if (
        not isinstance(relative, str) or not _is_sha256(digest)
        or Path(relative).is_absolute() or '..' in Path(relative).parts
    ):
      raise AnalysisError('Freeze implementation hash inventory is malformed.')
    path = (bundle_root / relative).resolve()
    try:
      path.relative_to(bundle_root.resolve())
    except ValueError as error:
      raise AnalysisError('Freeze implementation path escaped repository.') from error
    if _sha256(path) != digest:
      raise AnalysisError(f'Frozen implementation bytes changed: {relative}.')
  bootstrap = start.get('same_process_pre_import_bootstrap')
  if not isinstance(bootstrap, Mapping):
    raise AnalysisError('Pre-import bootstrap attestation is missing.')
  if set(bootstrap) != {
      'pid', 'created_at_unix_s', 'generated_bindings',
      'sanitized_environment', 'freeze', 'launcher_path', 'launcher_sha256',
      'bootstrap_path', 'bootstrap_sha256',
  }:
    raise AnalysisError('Pre-import bootstrap attestation schema changed.')
  if not isinstance(bootstrap.get('pid'), int) or bootstrap['pid'] <= 0:
    raise AnalysisError('Pre-import bootstrap PID is invalid.')
  _finite(bootstrap.get('created_at_unix_s'), 'bootstrap.created_at_unix_s')
  expected_sanitized = {
      'LD_LIBRARY_PATH': 'absent',
      'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
      'JAX_ENABLE_COMPILATION_CACHE': 'false',
  }
  if bootstrap.get('sanitized_environment') != expected_sanitized:
    raise AnalysisError('Pre-import sanitized environment changed.')
  bootstrap_freeze = bootstrap.get('freeze')
  if not isinstance(bootstrap_freeze, Mapping) or set(bootstrap_freeze) != {
      'path', 'sha256', 'git_head', 'tracked_head_clean', 'tracked_paths',
      'upstream_checkout', 'prior_v3_2_evidence',
  } or (
      bootstrap_freeze.get('path') != str(freeze_path)
      or bootstrap_freeze.get('sha256') != freeze_sha
      or bootstrap_freeze.get('tracked_head_clean') is not True
  ):
    raise AnalysisError('Pre-import bootstrap clean/freeze gate is invalid.')
  git_head = bootstrap_freeze.get('git_head')
  if (
      not isinstance(git_head, str) or len(git_head) != 40
      or any(character not in '0123456789abcdef' for character in git_head)
  ):
    raise AnalysisError('Pre-import bootstrap Git HEAD is invalid.')
  expected_tracked = sorted({
      str(freeze_path.relative_to(bundle_root)),
      str(_PROTOCOL_PATH.resolve().relative_to(bundle_root)),
      str(_SUPERSESSION_PATH.resolve().relative_to(bundle_root)),
      *file_hashes.keys(),
  })
  if bootstrap_freeze.get('tracked_paths') != expected_tracked:
    raise AnalysisError('Pre-import tracked path inventory changed.')
  for name, expected_path in (
      ('launcher', _HERE / 'launch_encoder_skip_factorial_v3_3.py'),
      ('bootstrap', _HERE / 'validate_encoder_skip_bootstrap_v3_3.py'),
  ):
    path = Path(str(bootstrap.get(f'{name}_path'))).resolve()
    if path != expected_path.resolve() or _sha256(path) != bootstrap.get(f'{name}_sha256'):
      raise AnalysisError(f'Pre-import {name} path/hash changed.')
  generated = bootstrap.get('generated_bindings')
  if not isinstance(generated, Mapping) or set(generated) != {
      'pre_import_gate', 'historical_generator_argv',
      'exact_regeneration_claim', 'generated_artifact_exception', 'artifacts',
      'embedded_header', 'protobuf_runtime_version',
  }:
    raise AnalysisError('Pre-import generated-binding schema changed.')
  if (
      generated.get('pre_import_gate') is not True
      or generated.get('exact_regeneration_claim') is not False
      or generated.get('historical_generator_argv') != 'unknown'
      or generated.get('protobuf_runtime_version')
      != freeze_file.get('protobuf_binding', {}).get('protobuf_runtime_version')
      or generated.get('embedded_header')
      != freeze_file.get('protobuf_binding', {}).get('embedded_generated_header')
  ):
    raise AnalysisError('Pre-import generated-binding claims changed.')
  generated_outputs = freeze_file.get('protobuf_binding', {}).get('generated_outputs')
  artifacts = generated.get('artifacts')
  if not isinstance(generated_outputs, Mapping) or not isinstance(artifacts, Mapping):
    raise AnalysisError('Pre-import generated-output linkage is absent.')
  if {Path(row['path']).resolve() for row in artifacts.values()} != {
      Path(path).resolve() for path in generated_outputs
  }:
    raise AnalysisError('Pre-import generated-output path set changed.')
  for row in artifacts.values():
    if not isinstance(row, Mapping) or set(row) != {'path', 'sha256', 'size_bytes'}:
      raise AnalysisError('Pre-import generated-output artifact schema changed.')
    path = Path(row['path']).resolve()
    frozen_row = generated_outputs.get(str(path))
    if frozen_row != {'sha256': row['sha256'], 'size_bytes': row['size_bytes']}:
      raise AnalysisError('Pre-import generated-output freeze linkage changed.')
    if _sha256(path) != row['sha256'] or path.stat().st_size != row['size_bytes']:
      raise AnalysisError('Pre-import generated-output bytes changed.')
  runtime = start.get('runtime_environment')
  if not isinstance(runtime, Mapping) or (
      runtime.get('JAX_ENABLE_COMPILATION_CACHE') != 'false'
      or runtime.get('XLA_PYTHON_CLIENT_PREALLOCATE') != 'false'
  ):
    raise AnalysisError('START runtime cache/preallocation contract changed.')
  forbidden_env = {
      key for key in runtime
      if key in {'XLA_FLAGS', 'JAX_COMPILATION_CACHE_DIR'}
      or str(key).startswith('JAX_PERSISTENT_CACHE_')
      or (
          'AUTOTUNE' in str(key).upper()
          and any(
              term in str(key).upper() for term in ('LOAD', 'DUMP', 'CACHE')
          )
      )
  }
  if forbidden_env:
    raise AnalysisError(f'START contains forbidden cache inputs: {sorted(forbidden_env)}.')
  _finite(start.get('started_at_unix_s'), 'ATTEMPT_STARTED.started_at_unix_s')
  if start.get('confirmation_scope_disclosure') != CONFIRMATION_SCOPE_DISCLOSURE:
    raise AnalysisError('ATTEMPT_STARTED confirmation scope disclosure changed.')
  extra_audit = {}
  if validate_external_inputs:
    if cases is None:
      raise AnalysisError('START input validation requires the frozen cases.')
    if start.get('prior_v3_2_evidence') != freeze_file.get('prior_v3_2_evidence'):
      raise AnalysisError('START prior-v3.2 evidence differs from freeze.')
    if bootstrap_freeze.get('prior_v3_2_evidence') != start['prior_v3_2_evidence']:
      raise AnalysisError('Bootstrap prior-v3.2 evidence differs from START.')
    extra_audit.update(_validate_prior_v3_2_evidence(start['prior_v3_2_evidence']))
    upstream_inventory, upstream_audit = _validate_upstream_checkout(
        start, freeze_file, bundle_root=bundle_root
    )
    extra_audit.update(upstream_audit)
    extra_audit['upstream_imported_modules'] = upstream_inventory
    extra_audit.update(_validate_runtime_manifest(start, freeze_file))
    extra_audit.update(_validate_bundle_and_preflight(
        start, freeze_file, freeze_sha, bundle_root=bundle_root
    ))
    extra_audit.update(_validate_checkpoint_reference(start, freeze_file, cases))
  return dict(freeze_file), freeze_sha, {
      'freeze_sha256': freeze_sha,
      'tracked_head_clean_at_pre_import': True,
      'git_head': git_head,
      'runtime_environment_verified': True,
      **extra_audit,
  }


def _validate_completion(
    complete: Mapping[str, Any], manifest: Mapping[str, Any], *,
    freeze_sha: str, run_dir: Path,
) -> tuple[str, str, dict[str, Any]]:
  required = {
      'status': 'complete',
      'stop_reason': None,
      'attempt_id': ATTEMPT_ID,
      'script_version': SOURCE_SCRIPT_VERSION,
      'protocol_sha256': PROTOCOL_SHA256,
      'supersession_sha256': SUPERSESSION_SHA256,
      'supersession_commit': 'c64def4',
      'freeze_sha256': freeze_sha,
      'identity_count': EXPECTED_IDENTITIES,
      'eligible_effect_count': len(EFFECT_ORDERS),
      'identity_invalid_count': 0,
      'coalition_record_count': EXPECTED_COALITIONS,
      'coalition_invalid_count': 0,
      'ood_anchor_record_count': EXPECTED_OOD,
      'ood_invalid_count': 0,
      'scientific_record_count': EXPECTED_RAW_RECORDS,
      'model_apply_count': EXPECTED_MODEL_APPLIES,
      'id0_noop_all20': True,
      'id255_closure_all20': True,
      'all_effects_target_eligible': True,
      'all_neutrals_retained': True,
      'compile_count': 2,
      'confirmation_model_calls': 0,
  }
  for key, expected in required.items():
    if complete.get(key) != expected:
      raise AnalysisError(f'RUN_COMPLETE.{key} differs from the frozen contract.')
  return _validate_completion_bindings(
      complete, manifest, freeze_sha=freeze_sha, run_dir=run_dir
  )


def _completion_prefix(
    complete: Mapping[str, Any]
) -> tuple[list[tuple[str, int, int | None]], bool]:
  """Validates completion/controlled-stop counts and returns exact raw prefix."""
  _require_exact_keys(complete, {
      'status', 'stop_reason', 'message', 'attempt_id', 'script_version',
      'protocol_sha256', 'freeze_sha256', 'supersession_sha256',
      'supersession_commit', 'identity_count', 'eligible_effect_count',
      'all_effects_target_eligible', 'all_neutrals_retained',
      'identity_invalid_count', 'coalition_record_count',
      'coalition_invalid_count', 'ood_anchor_record_count',
      'ood_invalid_count', 'scientific_record_count', 'model_apply_count',
      'id0_noop_all20', 'id255_closure_all20', 'six_row_compiler',
      'eight_row_compiler', 'six_row_executable_fingerprint',
      'eight_row_executable_fingerprint', 'compile_count',
      'import_provenance_sha256', 'import_provenance_phases',
      'protobuf_provenance_sha256', 'raw_manifest',
      'confirmation_model_calls', 'confirmation_scope_disclosure',
      'completed_at_unix_s',
  }, 'RUN_COMPLETE')
  common = {
      'attempt_id': ATTEMPT_ID,
      'script_version': SOURCE_SCRIPT_VERSION,
      'protocol_sha256': PROTOCOL_SHA256,
      'supersession_sha256': SUPERSESSION_SHA256,
      'supersession_commit': 'c64def4',
      'confirmation_model_calls': 0,
      'compile_count': 2,
  }
  for key, expected in common.items():
    if complete.get(key) != expected:
      raise AnalysisError(f'RUN_COMPLETE.{key} differs from the frozen contract.')
  counts = {}
  for key in (
      'identity_count', 'eligible_effect_count', 'identity_invalid_count',
      'coalition_record_count', 'coalition_invalid_count',
      'ood_anchor_record_count', 'ood_invalid_count',
      'scientific_record_count', 'model_apply_count',
  ):
    value = complete.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
      raise AnalysisError(f'RUN_COMPLETE.{key} is not a nonnegative integer.')
    counts[key] = value
  identity_count = counts['identity_count']
  coalition_count = counts['coalition_record_count']
  ood_count = counts['ood_anchor_record_count']
  if counts['scientific_record_count'] != identity_count + coalition_count + ood_count:
    raise AnalysisError('RUN_COMPLETE scientific record arithmetic changed.')
  if counts['model_apply_count'] != 2 * identity_count + 2 * coalition_count + 4 * ood_count:
    raise AnalysisError('RUN_COMPLETE model-apply arithmetic changed.')
  if complete.get('all_effects_target_eligible') is not (
      counts['eligible_effect_count'] == 12
  ):
    raise AnalysisError('RUN_COMPLETE all-effects eligibility flag changed.')
  if not isinstance(complete.get('all_neutrals_retained'), bool):
    raise AnalysisError('RUN_COMPLETE neutral-retention flag is not boolean.')
  all_order = _expected_execution_order()
  status, reason = complete.get('status'), complete.get('stop_reason')
  if status == 'complete':
    if reason is not None:
      raise AnalysisError('Complete run has a stop reason.')
    if (identity_count, coalition_count, ood_count) != (20, 5120, 80):
      raise AnalysisError('Complete run does not have the exact full family counts.')
    if complete['all_neutrals_retained'] is not True:
      raise AnalysisError('Complete run did not retain all neutral controls.')
    return all_order, True
  if status != 'controlled_stop' or reason not in {
      'identity_tooling_failure', 'target_comparability_failure',
      'gate0_closure_failure', 'coalition_tooling_failure',
      'ood_tooling_failure',
  }:
    raise AnalysisError('RUN_COMPLETE status/stop_reason is not an allowed outcome.')
  if reason != 'identity_tooling_failure' and complete['all_neutrals_retained'] is not True:
    raise AnalysisError('Controlled stop lost a neutral after valid identities.')
  if reason == 'identity_tooling_failure':
    valid = (
        identity_count == 20 and counts['identity_invalid_count'] >= 1
        and coalition_count == 0 and ood_count == 0
        and counts['coalition_invalid_count'] == counts['ood_invalid_count'] == 0
    )
  elif reason == 'target_comparability_failure':
    valid = (
        identity_count == 20 and counts['identity_invalid_count'] == 0
        and counts['eligible_effect_count'] < 12
        and coalition_count == 0 and ood_count == 0
        and counts['coalition_invalid_count'] == counts['ood_invalid_count'] == 0
    )
  elif reason == 'gate0_closure_failure':
    valid = (
        identity_count == 20 and counts['identity_invalid_count'] == 0
        and counts['eligible_effect_count'] == 12
        and 1 <= coalition_count <= 40
        and counts['coalition_invalid_count'] == 1
        and ood_count == counts['ood_invalid_count'] == 0
    )
  elif reason == 'coalition_tooling_failure':
    valid = (
        identity_count == 20 and counts['identity_invalid_count'] == 0
        and counts['eligible_effect_count'] == 12
        and 41 <= coalition_count <= 5120
        and counts['coalition_invalid_count'] == 1
        and ood_count == counts['ood_invalid_count'] == 0
    )
  else:
    valid = (
        identity_count == 20 and counts['identity_invalid_count'] == 0
        and counts['eligible_effect_count'] == 12
        and coalition_count == 5120 and counts['coalition_invalid_count'] == 0
        and 1 <= ood_count <= 80 and counts['ood_invalid_count'] == 1
    )
  if not valid:
    raise AnalysisError(f'RUN_COMPLETE counts do not match controlled stop {reason}.')
  prefix = (
      all_order[:identity_count]
      + all_order[20:20 + coalition_count]
      + all_order[5140:5140 + ood_count]
  )
  return prefix, False


def _validate_completion_bindings(
    complete: Mapping[str, Any], manifest: Mapping[str, Any], *,
    freeze_sha: str, run_dir: Path,
) -> tuple[str, str, dict[str, Any]]:
  if complete.get('freeze_sha256') != freeze_sha:
    raise AnalysisError('RUN_COMPLETE freeze binding changed.')
  if complete.get('raw_manifest') != manifest:
    raise AnalysisError('RUN_COMPLETE embedded raw manifest differs.')
  if complete.get('confirmation_scope_disclosure') != CONFIRMATION_SCOPE_DISCLOSURE:
    raise AnalysisError('RUN_COMPLETE confirmation scope disclosure changed.')
  _finite(complete.get('completed_at_unix_s'), 'RUN_COMPLETE.completed_at_unix_s')
  six = complete.get('six_row_compiler')
  eight = complete.get('eight_row_compiler')
  six_fingerprint = _validate_compiler(six, run_dir=run_dir, executable_name='six_row')
  eight_fingerprint = _validate_compiler(eight, run_dir=run_dir, executable_name='eight_row')
  if complete.get('six_row_executable_fingerprint') != six_fingerprint:
    raise AnalysisError('RUN_COMPLETE six-row fingerprint differs.')
  if complete.get('eight_row_executable_fingerprint') != eight_fingerprint:
    raise AnalysisError('RUN_COMPLETE eight-row fingerprint differs.')
  return six_fingerprint, eight_fingerprint, {
      'six_row_compiler_verified': True,
      'eight_row_compiler_verified': True,
      'compile_count': 2,
      'model_apply_count': complete['model_apply_count'],
  }


def render_markdown(result: Mapping[str, Any]) -> str:
  nomination = result.get('nomination')
  lines = [
      '# OpenSplice v3.3 development encoder-skip localization', '',
      f"**Decision:** `{result['decision']}`", '',
      'The CPU-only analyzer independently reconstructed both canonical '
      'endpoint margins from raw relevant/padding logits before evaluating '
      'the complete frozen coalition cube.', '',
  ]
  if result.get('controlled_stop') is not None:
    lines.extend([
        'The append-only controlled-stop prefix was audited successfully. '
        'No Shapley values or mechanism nomination were computed.',
        '',
    ])
  elif nomination is None:
    lines.append('No whole-skip resolution coalition met the frozen two-exon gate.')
  else:
    lines.extend([
        f"Nominated coalition: `{nomination['coalition_id']}` "
        f"(`{nomination['e_bits']}`, {', '.join(nomination['enabled_skips'])}).",
        '',
        'This is a development computational-route result under whole-tensor '
        'transfer. It is not a molecular pathway, RBP, spliceosome step, '
        'endogenous necessity, or independent experimental replication.',
    ])
  lines.extend([
      '',
      'Confirmation model outputs, activations, and interventions remained '
      'unopened. Previously disclosed later-exon metadata/labels were exposed '
      'post-freeze.', '',
  ])
  return '\n'.join(lines)


def _write_atomic(path: Path, text: str) -> None:
  _guard_path(path)
  if path.exists():
    raise FileExistsError(f'Append-only analysis output exists: {path}.')
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
  try:
    with temporary.open('x', encoding='utf-8') as handle:
      handle.write(text)
      handle.flush()
      os.fsync(handle.fileno())
    os.link(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)


def analyze(
    run_dir: Path, *, bundle_root: Path | None = None
) -> dict[str, Any]:
  """Validates one exact complete raw tree and computes all frozen estimands."""
  _assert_cpu_only_module_boundary()
  run_dir = run_dir.resolve()
  bundle_root = (
      _HERE.parents[2].resolve() if bundle_root is None else bundle_root.resolve()
  )
  _guard_path(run_dir)
  _guard_path(bundle_root)
  protocol = _validate_protocol_bindings()
  cases = _load_cases()
  freeze, freeze_sha, start_audit = _validate_start(
      run_dir, bundle_root=bundle_root, cases=cases
  )
  complete = _read_json(run_dir / 'RUN_COMPLETE.json')
  expected_order, fully_complete = _completion_prefix(complete)
  stop_reason = complete.get('stop_reason')
  _validate_top_level_tree(
      run_dir, target_eligibility_exists=stop_reason != 'identity_tooling_failure'
  )
  manifest, raw_hashes = _validate_manifest(
      run_dir, cases, expected_order=expected_order
  )
  if fully_complete:
    six_fingerprint, eight_fingerprint, compiler_audit = _validate_completion(
        complete, manifest, freeze_sha=freeze_sha, run_dir=run_dir
    )
  else:
    six_fingerprint, eight_fingerprint, compiler_audit = (
        _validate_completion_bindings(
            complete, manifest, freeze_sha=freeze_sha, run_dir=run_dir
        )
    )
  imports = _validate_imports(
      run_dir, complete, bundle_root=bundle_root, freeze=freeze
  )
  protobuf = _validate_protobuf(run_dir, complete, freeze)

  identities: dict[int, dict[str, Any]] = {}
  identity_bindings: dict[int, str] = {}
  cubes: dict[int, dict[int, dict[str, Any]]] = {
      order: {} for order in ALL_ORDERS
  }
  coalition_readouts: dict[tuple[int, int], dict[str, Any]] = {}
  ood_controls = []
  invalid_records: list[tuple[int, str]] = []
  for execution_index, (family, order, coalition_id) in enumerate(expected_order):
    case = cases[order]
    relative = _artifact_relative(family, case, coalition_id)
    path = run_dir / relative
    record = _read_json(path)
    if record.get('status') == 'invalid':
      donor_order = order + 10 if order < 10 else order - 10
      donor_case = cases[donor_order] if family == 'ood' else None
      runtime_family = {
          'identity': 'identity',
          'coalition': 'encoder_skip_coalition',
          'ood': 'unrelated_donor_anchor',
      }[family]
      _validate_invalid_record(
          record, family=runtime_family, case=case,
          expected_sequence=start_audit['sequence_bindings'][order],
          coalition_id=coalition_id, identity=identities.get(order),
          identity_relative=identity_bindings.get(order),
          donor_case=donor_case, donor_identity=identities.get(donor_order),
          donor_identity_relative=identity_bindings.get(donor_order),
          linked_relative=(
              _artifact_relative('coalition', case, coalition_id)
              if family == 'ood' else None
          ),
          run_dir=run_dir, freeze_sha=freeze_sha,
          execution_index=execution_index, six_fingerprint=six_fingerprint,
          eight_fingerprint=eight_fingerprint, label=relative,
      )
      invalid_records.append((execution_index, family))
      continue
    if family == 'identity':
      identity = _validate_identity_record(
          record, case, freeze_sha=freeze_sha,
          execution_index=execution_index, label=relative,
      )
      _validate_frozen_sequence_binding(
          identity, start_audit.get('sequence_bindings', {}).get(order), relative
      )
      if identity['six_row_executable_fingerprint'] != six_fingerprint:
        raise AnalysisError(f'{relative} executable differs from compiler provenance.')
      identities[order] = identity
      identity_bindings[order] = relative
      continue
    if order not in identities:
      raise AnalysisError(f'{relative} appears before its identity dependency.')
    assert coalition_id is not None
    if family == 'coalition':
      metrics, readout = _validate_coalition_record(
          record, case, coalition_id, identity=identities[order],
          identity_relative=identity_bindings[order], run_dir=run_dir,
          freeze_sha=freeze_sha, execution_index=execution_index,
          label=relative,
      )
      cubes[order][coalition_id] = metrics
      coalition_readouts[(order, coalition_id)] = readout
      continue
    donor_order = order + 10 if order < 10 else order - 10
    donor = cases[donor_order]
    linked_relative = _artifact_relative('coalition', case, coalition_id)
    control = _validate_ood_record(
        record, case, donor, coalition_id,
        identity=identities[order], identity_relative=identity_bindings[order],
        donor_identity=identities[donor_order],
        donor_identity_relative=identity_bindings[donor_order],
        linked_relative=linked_relative, run_dir=run_dir,
        freeze_sha=freeze_sha, execution_index=execution_index,
        eight_fingerprint=eight_fingerprint, label=relative,
    )
    ood_controls.append(control)

  if not fully_complete:
    if identities:
      first_signatures = next(iter(identities.values()))['program_signatures']
      if any(
          identity['program_signatures'] != first_signatures
          for identity in identities.values()
      ):
        raise AnalysisError('Controlled-stop identity program signatures differ.')
    actual_eligible_count = sum(
        identities.get(order, {}).get('eligible', False) for order in EFFECT_ORDERS
    )
    actual_all_neutrals_retained = all(order in identities for order in NEUTRAL_ORDERS)
    actual_id0_all20 = all(0 in cubes[order] for order in ALL_ORDERS)
    actual_id255_all20 = all(255 in cubes[order] for order in ALL_ORDERS)
    if complete.get('id0_noop_all20') is not actual_id0_all20:
      raise AnalysisError('Controlled-stop ID0 all-20 closure flag changed.')
    if complete.get('id255_closure_all20') is not actual_id255_all20:
      raise AnalysisError('Controlled-stop ID255 all-20 closure flag changed.')
    if complete['all_neutrals_retained'] is not actual_all_neutrals_retained:
      raise AnalysisError('Controlled-stop neutral-retention flag changed.')
    if (
        stop_reason != 'identity_tooling_failure'
        and actual_eligible_count != complete['eligible_effect_count']
    ):
      raise AnalysisError('Controlled-stop eligible-effect count changed.')
    if stop_reason == 'identity_tooling_failure':
      if (
          len(invalid_records) != complete['identity_invalid_count']
          or any(family != 'identity' for _, family in invalid_records)
      ):
        raise AnalysisError('Identity controlled-stop invalid set changed.')
    elif stop_reason == 'target_comparability_failure':
      if invalid_records:
        raise AnalysisError('Target-comparability stop contains invalid tooling records.')
    else:
      expected_family = 'ood' if stop_reason == 'ood_tooling_failure' else 'coalition'
      if invalid_records != [(len(expected_order) - 1, expected_family)]:
        raise AnalysisError('Controlled stop did not end at its first invalid record.')
    if stop_reason != 'identity_tooling_failure':
      eligibility = _read_json(run_dir / 'TARGET_ELIGIBILITY.json')
      expected_eligible = [
          cases[order]['variant_id'] for order in EFFECT_ORDERS
          if identities[order]['eligible']
      ]
      expected_ineligible = [
          cases[order]['variant_id'] for order in EFFECT_ORDERS
          if not identities[order]['eligible']
      ]
      if eligibility != {
          'eligible_effects': expected_eligible,
          'ineligible_effects': expected_ineligible,
          'neutral_controls': [cases[order]['variant_id'] for order in NEUTRAL_ORDERS],
      }:
        raise AnalysisError('Controlled-stop TARGET_ELIGIBILITY changed.')
    result = {
        'analysis_version': ANALYSIS_VERSION,
        'analysis_dir': freeze['analysis_dir'],
        'decision': f'controlled_stop_{stop_reason}',
        'nomination': None,
        'resolution_analysis': None,
        'protocol': protocol,
        'freeze_sha256': freeze_sha,
        'raw_manifest': manifest,
        'hash_tree': {
            'raw_artifact_count': len(raw_hashes),
            'raw_artifact_tree_sha256': manifest['artifact_tree_sha256'],
            'raw_manifest_sha256': _sha256(run_dir / 'RAW_MANIFEST.json'),
            'run_complete_sha256': _sha256(run_dir / 'RUN_COMPLETE.json'),
        },
        'provenance_audit': {
            **start_audit, **compiler_audit, 'imports': imports,
            'protobuf': protobuf, 'execution_prefix_exact': True,
        },
        'controlled_stop': {
            'reason': stop_reason,
            'identity_count': complete['identity_count'],
            'identity_invalid_count': complete['identity_invalid_count'],
            'eligible_effect_count': complete['eligible_effect_count'],
            'coalition_record_count': complete['coalition_record_count'],
            'coalition_invalid_count': complete['coalition_invalid_count'],
            'ood_anchor_record_count': complete['ood_anchor_record_count'],
            'ood_invalid_count': complete['ood_invalid_count'],
            'shapley_computed': False,
            'nomination_performed': False,
        },
        'confirmation_scope_disclosure': (
            'Confirmation model outputs, activations, and interventions remained '
            'unopened; later-exon metadata/labels had been exposed post-freeze.'
        ),
        'claim_boundary': (
            'Audited development tooling/target stop; no v3.3 localization '
            'or mechanistic result exists.'
        ),
    }
    _assert_cpu_only_module_boundary()
    return result

  if len(identities) != EXPECTED_IDENTITIES:
    raise AnalysisError('Identity family is incomplete.')
  signatures = identities[0]['program_signatures']
  if any(identity['program_signatures'] != signatures for identity in identities.values()):
    raise AnalysisError('Identity program signatures changed across variants.')
  ineligible = [order for order in EFFECT_ORDERS if not identities[order]['eligible']]
  if ineligible:
    raise AnalysisError(f'Not all 12 development effects are target eligible: {ineligible}.')
  if sum(len(cube) for cube in cubes.values()) != EXPECTED_COALITIONS:
    raise AnalysisError('Coalition family does not contain exactly 5,120 records.')
  if len(ood_controls) != EXPECTED_OOD:
    raise AnalysisError('OOD anchor family does not contain exactly 80 records.')
  eligibility = _read_json(run_dir / 'TARGET_ELIGIBILITY.json')
  expected_eligibility = {
      'eligible_effects': [cases[order]['variant_id'] for order in EFFECT_ORDERS],
      'ineligible_effects': [],
      'neutral_controls': [cases[order]['variant_id'] for order in NEUTRAL_ORDERS],
  }
  if eligibility != expected_eligibility:
    raise AnalysisError('TARGET_ELIGIBILITY differs from independent identity gates.')
  scientific = summarize_cubes(cubes, cases)
  identity_behavior = []
  for order in ALL_ORDERS:
    identity = identities[order]
    readout = identity['readout']
    identity_behavior.append({
        'order': order,
        'variant_id': cases[order]['variant_id'],
        'gene': cases[order]['gene'],
        'selection_class': cases[order]['selection_class'],
        'reference_endpoint_margins': readout['endpoint_margins'][0],
        'alternate_endpoint_margins': readout['endpoint_margins'][1],
        'alternate_minus_reference_endpoint_margins': [
            _f32(alt - ref)
            for alt, ref in zip(
                readout['endpoint_margins'][1], readout['endpoint_margins'][0],
                strict=True,
            )
        ],
        'alternate_minus_reference_mean_margin': identity['predicted_delta'],
        'experimental_delta_logit': cases[order]['delta_logit'],
        'target_eligible_effect': identity['eligible'],
        'neutral_is_behavior_control_not_null': order in NEUTRAL_ORDERS,
    })
  ood_warnings = []
  for anchor in OOD_ANCHORS:
    for gene, orders in (
        ('BRAF', range(0, 10)), ('SLC25A48', range(10, 20))
    ):
      rows = [
          row for row in ood_controls
          if row['coalition_id'] == anchor and row['recipient_order'] in orders
      ]
      intended = _median(
          (row['intended_mean_absolute_movement'] for row in rows),
          f'OOD.{gene}.{anchor}.intended',
      )
      unrelated = _median(
          (row['unrelated_mean_absolute_movement'] for row in rows),
          f'OOD.{gene}.{anchor}.unrelated',
      )
      ood_warnings.append({
          'coalition_id': anchor,
          'gene': gene,
          'median_intended_absolute_movement': intended,
          'median_unrelated_absolute_movement': unrelated,
          'unrelated_at_least_intended_warning': unrelated >= intended,
          'formal_gate': False,
      })
  result = {
      'analysis_version': ANALYSIS_VERSION,
      'analysis_dir': freeze['analysis_dir'],
      'decision': scientific['decision'],
      'nomination': scientific['nomination'],
      'protocol': protocol,
      'freeze_sha256': freeze_sha,
      'raw_manifest': manifest,
      'hash_tree': {
          'raw_artifact_count': len(raw_hashes),
          'raw_artifact_tree_sha256': manifest['artifact_tree_sha256'],
          'raw_manifest_sha256': _sha256(run_dir / 'RAW_MANIFEST.json'),
          'run_complete_sha256': _sha256(run_dir / 'RUN_COMPLETE.json'),
      },
      'provenance_audit': {
          **start_audit,
          **compiler_audit,
          'imports': imports,
          'protobuf': protobuf,
          'single_six_row_executable': True,
          'single_eight_row_executable': True,
          'execution_order_exact': True,
      },
      'gate_0': {
          'identity_count': len(identities),
          'eligible_effect_count': len(EFFECT_ORDERS),
          'neutral_count': len(NEUTRAL_ORDERS),
          'coalition_count': sum(len(cube) for cube in cubes.values()),
          'ood_anchor_count': len(ood_controls),
          'id0_endpoint_noop_all20': True,
          'id255_endpoint_closure_all20': True,
          'complete_cube': True,
      },
      'identity_behavior_controls': identity_behavior,
      'resolution_analysis': scientific,
      'ood_anchor_controls': {
          'records': ood_controls,
          'warnings': ood_warnings,
          'donor_normalized_recovery_computed': False,
          'formal_nomination_gate': False,
      },
      'neutral_caveat': (
          'All eight experimentally nonsignificant variants are retained as '
          'AlphaGenome behavior controls, not assumed AlphaGenome-null.'
      ),
      'confirmation_scope_disclosure': (
          'Confirmation model outputs, activations, and interventions remained '
          'unopened; later-exon metadata/labels had been exposed post-freeze.'
      ),
      'claim_boundary': (
          'Development-only whole-tensor computational routing in BRAF e14 and '
          'SLC25A48 e8; not a molecular pathway, RBP, spliceosome step, '
          'endogenous necessity, experimental replication, or held-out result.'
      ),
  }
  _assert_cpu_only_module_boundary()
  return result


def main() -> None:
  args = _parse_args()
  _guard_path(args.output_json)
  _guard_path(args.output_markdown)
  if args.output_json.resolve() == args.run_dir.resolve() or (
      args.run_dir.resolve() in args.output_json.resolve().parents
  ):
    raise AnalysisError('Analysis output cannot be inside the append-only raw run.')
  result = analyze(args.run_dir)
  analysis_dir = Path(result['analysis_dir']).resolve()
  if args.output_json.resolve() != analysis_dir / 'ANALYSIS.json':
    raise AnalysisError('JSON output path differs from frozen analysis destination.')
  if args.output_markdown.resolve() != analysis_dir / 'RESULT.md':
    raise AnalysisError('Markdown output path differs from frozen analysis destination.')
  if analysis_dir.exists():
    raise FileExistsError('Frozen analysis directory already exists; never overwrite.')
  _write_atomic(
      args.output_json,
      json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n',
  )
  _write_atomic(args.output_markdown, render_markdown(result))


if __name__ == '__main__':
  main()
