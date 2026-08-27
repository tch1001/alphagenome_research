#!/usr/bin/env python3
"""Fail-closed CPU-only analyzer for the OpenSplice v3.2 superset run.

This module deliberately has no JAX, AlphaGenome, NumPy, or model imports.  It
accepts only the frozen BRAF/SLC25A48 development cohort, verifies the
append-only artifact/hash/provenance graph, reconstructs the two-endpoint
classification-logit target from raw logits, and then computes the preregistered
Phase-R and Stage-A estimands.  Confirmation-named paths and non-development
genes are rejected before scientific analysis.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
from typing import Any, Iterable, Mapping, Sequence


ANALYSIS_VERSION = 'opensplice-superset-analysis-v3.2.0'
SOURCE_SCRIPT_VERSION = 'opensplice-superset-graph-v3.2.0'
ATTEMPT_ID = 'opensplice-v3.2-development-superset-graph-one-shot'
PROTOCOL_SHA256 = (
    '1e87839250e838d1d9aa95162e76a3acb6b1eaab59b481c60dab943b62f6caea'
)
DEVELOPMENT_VARIANTS_SHA256 = (
    '24a0afec1c020803152c7f55a0a78ac345763173dd79a4175e889d9192db05f9'
)
DEVELOPMENT_EXONS_SHA256 = (
    '37c49637bba5484d1e29a7f21faf42668dcf62d604e9af91939a8776e61f0231'
)
SELECTED_VARIANTS_SHA256 = (
    '09cf0003317d742dfa742481ff6a96896b679342717867b31c85283262a6fdf6'
)
FROZEN_EXONS_SHA256 = (
    'b95f8fc540f19222546322bebfb817a6c0f2147dd41325091086833584a09a75'
)
CHECKPOINT_MANIFEST_SHA256 = (
    '1ed87db4c5bd7c5418c7734ec128faa4a9ecd186df2a024437484a8bc2b6e934'
)
REFERENCE_BINDINGS_SHA256 = (
    'da712cdca50f82113ac1d00cb2fa7171f7368f31aedf06c48ce92dbdb5897dca'
)
CHECKPOINT_SNAPSHOT = 'a8f293a76ee73d5b57f3bf2ae146510589fcf187'
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
STAGES = ('pre_attention', 'post_attention', 'post_mlp')
LAYERS = tuple(range(6))
CANDIDATES = ('V', 'A', 'D', 'S')
POSITION_SETS = (
    'V',
    'A',
    'D',
    'S',
    'V_control_upstream',
    'V_control_downstream',
    'A_control_upstream',
    'A_control_downstream',
    'D_control_upstream',
    'D_control_downstream',
    'S_control_upstream',
    'S_control_downstream',
)
STAGE_COMPONENTS = (
    '00_final_embedding_A_D_closure',
    '01_joint_T_plus_E_closure',
    '02_whole_T',
    '03_whole_E',
)
EXPECTED_IDENTITIES = 20
EXPECTED_EFFECTS = 12
EXPECTED_NEUTRALS = 8
EXPECTED_EFFECTS_PER_GENE = 6
EXPECTED_NEUTRALS_PER_GENE = 4
EXPECTED_GROUPS_PER_ELIGIBLE = 216
EXPECTED_CANDIDATES = 72
EFFECT_THRESHOLD = 0.01
INVALID_FAMILY_FRACTION = 0.05
EXPECTED_DEVICE_KIND = 'NVIDIA GeForce RTX 3090'
EXPECTED_GPU_UUID = 'GPU-64111645-1e42-a96d-f192-4abbec4b8090'
EXPECTED_COMPUTE_CAPABILITY = '8.6'
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
_PROTOCOL_PATH = (
    _HERE / 'v3_wider_mechanism' / 'superset_graph_protocol_v3_2.md'
)
_SELECTED_PATH = _HERE / 'superset_graph_v3_2_development_variants.tsv'
_EXONS_PATH = _HERE / 'superset_graph_v3_2_development_exons.tsv'
_CHECKPOINT_MANIFEST_PATH = _HERE / 'checkpoint_manifest_v3_2.tsv'
_REFERENCE_BINDINGS_PATH = _HERE / 'superset_graph_v3_2_reference_bindings.json'
_OUTPUT_DIR = _HERE / 'results' / 'v3_2_development_superset_graph_one_shot'
_ANALYSIS_DIR = _HERE / 'results' / 'v3_2_development_superset_graph_analysis'
_PREFLIGHT_DIR = _HERE / 'results' / 'v3_2_device_preflight'
_REFERENCE_URL = (
    'https://storage.googleapis.com/alphagenome/reference/gencode/hg38/'
    'GRCh38.p13.genome.fa'
)
_REFERENCE_OBJECT = {
    'url': _REFERENCE_URL,
    'bucket': 'alphagenome',
    'object': 'reference/gencode/hg38/GRCh38.p13.genome.fa',
    'generation': '1766084693379925',
    'size_bytes': 3_321_586_957,
    'etag': 'edee5408303f6c1c6bae1c76ffd23671',
    'md5_base64': '7e5UCDA/bBxrrhx2/9I2cQ==',
    'crc32c_base64': 'AyHUtA==',
}
_DISCLOSURE = (
    'Later-exon metadata/labels were exposed after protocol freeze; '
    'no later-exon model outputs, activations, or interventions are used.'
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run-dir', type=Path, required=True)
  parser.add_argument('--output-json', type=Path, required=True)
  parser.add_argument('--output-markdown', type=Path)
  return parser.parse_args()


def _guard_path(path: Path) -> None:
  if any('confirm' in part.lower() for part in path.resolve().parts):
    raise ValueError(f'Refusing to inspect a confirmation-named path: {path}.')


def _read_json(path: Path) -> dict[str, Any]:
  _guard_path(path)
  try:
    value = json.loads(path.read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError) as error:
    raise ValueError(f'Cannot read JSON artifact {path}.') from error
  if not isinstance(value, dict):
    raise ValueError(f'JSON artifact is not an object: {path}.')
  return value


def _sha256(path: Path) -> str:
  _guard_path(path)
  digest = hashlib.sha256()
  try:
    with path.open('rb') as handle:
      for block in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(block)
  except OSError as error:
    raise ValueError(f'Cannot hash required artifact {path}.') from error
  return digest.hexdigest()


def _tree_digest(paths: Iterable[Path], root: Path) -> str:
  digest = hashlib.sha256()
  for path in sorted(paths):
    try:
      relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
      raise ValueError(f'Artifact lies outside hash-tree root: {path}.') from error
    digest.update(str(relative).encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(_sha256(path)))
  return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
  return (
      isinstance(value, str)
      and len(value) == 64
      and all(character in '0123456789abcdef' for character in value)
  )


def _expected_production_bundle_paths() -> set[str]:
  """Returns the exact committed v3.2 provenance surface."""
  model_root = _REPO_ROOT / 'src' / 'alphagenome_research' / 'model'
  paths = {
      str(path.relative_to(_REPO_ROOT))
      for path in model_root.rglob('*.py')
      if not path.name.endswith('_test.py')
  }
  paths.add('src/alphagenome_research/model/interpretability_test.py')
  paths.update({
      'src/alphagenome_research/protos/__init__.py',
      'src/alphagenome_research/protos/calibration_scores.proto',
      'pyproject.toml',
      'hatch_build.py',
      'experiments/interpretability/opensplice/run_device_preflight_v3_1_2.py',
      'experiments/interpretability/opensplice/run_inference_trace.py',
      'experiments/interpretability/opensplice/run_phase_r_v3.py',
      'experiments/interpretability/opensplice/run_route_census_v3.py',
      'experiments/interpretability/opensplice/run_stage_a_branches_v3.py',
      'experiments/interpretability/opensplice/target_reducers_v3.py',
      'experiments/interpretability/opensplice/run_superset_graph_v3_2.py',
      'experiments/interpretability/opensplice/run_superset_graph_v3_2.sh',
      'experiments/interpretability/opensplice/run_superset_graph_v3_2_test.py',
      'experiments/interpretability/opensplice/run_device_preflight_v3_2.py',
      'experiments/interpretability/opensplice/run_device_preflight_v3_2_test.py',
      'experiments/interpretability/opensplice/'
      'validate_superset_graph_bootstrap_v3_2.py',
      'experiments/interpretability/opensplice/launch_superset_graph_v3_2.py',
      'experiments/interpretability/opensplice/analyze_superset_graph_v3_2.py',
      'experiments/interpretability/opensplice/analyze_superset_graph_v3_2_test.py',
      'experiments/interpretability/opensplice/checkpoint_manifest_v3_2.tsv',
      'experiments/interpretability/opensplice/'
      'superset_graph_v3_2_reference_bindings.json',
      'experiments/interpretability/opensplice/'
      'superset_graph_v3_2_development_variants.tsv',
      'experiments/interpretability/opensplice/'
      'superset_graph_v3_2_development_exons.tsv',
      'experiments/interpretability/opensplice/v3_wider_mechanism/'
      'superset_graph_protocol_v3_2.md',
  })
  return paths


def _finite(value: Any, label: str) -> float:
  if isinstance(value, bool):
    raise ValueError(f'{label} must be numeric, not boolean.')
  try:
    result = float(value)
  except (TypeError, ValueError) as error:
    raise ValueError(f'{label} is not numeric.') from error
  if not math.isfinite(result):
    raise ValueError(f'{label} is non-finite.')
  return result


def _f32(value: Any, label: str = 'value') -> float:
  result = _finite(value, label)
  try:
    return struct.unpack('<f', struct.pack('<f', result))[0]
  except OverflowError as error:
    raise ValueError(f'{label} is outside float32 range.') from error


def _f32_bits(value: Any, label: str = 'value') -> bytes:
  return struct.pack('<f', _f32(value, label))


def _same_f32(observed: Any, expected: Any, label: str) -> None:
  observed_value = _finite(observed, label)
  expected_value = _finite(expected, f'{label}.expected')
  if observed_value != _f32(observed_value, label):
    raise ValueError(f'{label} is not an exact finite float32 value.')
  if _f32_bits(observed_value, label) != _f32_bits(expected_value, label):
    raise ValueError(f'{label} differs bit-for-bit from its recomputed value.')


def _same_float(observed: Any, expected: float, label: str) -> None:
  value = _finite(observed, label)
  if value != expected:
    raise ValueError(f'{label} mismatch: {value!r} != {expected!r}.')


def _slug(value: str) -> str:
  return ''.join(char if char.isalnum() else '_' for char in value).strip('_')


def _require_true(mapping: Mapping[str, Any], fields: Sequence[str], label: str):
  for field in fields:
    if mapping.get(field) is not True:
      raise ValueError(f'{label}.{field} is not true.')


def _require_linkage(
    record: Mapping[str, Any], *, freeze_sha: str, executable: str, label: str
) -> None:
  if record.get('protocol_sha256') != PROTOCOL_SHA256:
    raise ValueError(f'{label} protocol hash mismatch.')
  if record.get('freeze_sha256') != freeze_sha:
    raise ValueError(f'{label} freeze hash mismatch.')
  if record.get('executable_fingerprint') != executable:
    raise ValueError(f'{label} executable fingerprint mismatch.')


def _readout(record: Mapping[str, Any], field: str, label: str) -> dict[str, Any]:
  """Reconstructs the float32 endpoint reducer from raw selected logits."""
  value = record.get(field)
  if not isinstance(value, Mapping):
    raise ValueError(f'{label}.{field} raw endpoint evidence is missing.')
  if value.get('selected_logit_axis') != [
      'relevant_class', 'padding_class'
  ]:
    raise ValueError(f'{label}.{field} selected-logit axis changed.')
  if value.get('endpoint_axis') != ['acceptor', 'donor']:
    raise ValueError(f'{label}.{field} endpoint axis changed.')
  if value.get('num_values') != 2:
    raise ValueError(f'{label}.{field} does not reduce exactly two endpoints.')
  logits = value.get('selected_logits')
  margins = value.get('endpoint_margins')
  totals = value.get('totals')
  means = value.get('means')
  if not isinstance(logits, list) or len(logits) != 6:
    raise ValueError(f'{label}.{field}.selected_logits must have shape [6,2,2].')
  if not isinstance(margins, list) or len(margins) != 6:
    raise ValueError(f'{label}.{field}.endpoint_margins must have shape [6,2].')
  if not isinstance(totals, list) or len(totals) != 6:
    raise ValueError(f'{label}.{field}.totals must have shape [6].')
  if not isinstance(means, list) or len(means) != 6:
    raise ValueError(f'{label}.{field}.means must have shape [6].')
  reconstructed_logits: list[list[list[float]]] = []
  reconstructed_margins: list[list[float]] = []
  reconstructed_totals: list[float] = []
  reconstructed_means: list[float] = []
  for row in range(6):
    row_logits = logits[row]
    row_margins = margins[row]
    if not isinstance(row_logits, list) or len(row_logits) != 2 or any(
        not isinstance(endpoint, list) or len(endpoint) != 2
        for endpoint in row_logits
    ):
      raise ValueError(
          f'{label}.{field}.selected_logits must have shape [6,2,2].'
      )
    if not isinstance(row_margins, list) or len(row_margins) != 2:
      raise ValueError(
          f'{label}.{field}.endpoint_margins must have shape [6,2].'
      )
    clean_logits = []
    clean_margins = []
    for endpoint in range(2):
      relevant = _f32(
          row_logits[endpoint][0],
          f'{label}.{field}.selected_logits[{row}][{endpoint}][0]',
      )
      padding = _f32(
          row_logits[endpoint][1],
          f'{label}.{field}.selected_logits[{row}][{endpoint}][1]',
      )
      expected_margin = _f32(relevant - padding)
      _same_f32(
          row_margins[endpoint], expected_margin,
          f'{label}.{field}.endpoint_margins[{row}][{endpoint}]',
      )
      clean_logits.append([relevant, padding])
      clean_margins.append(expected_margin)
    expected_total = _f32(clean_margins[0] + clean_margins[1])
    expected_mean = _f32(expected_total / 2.0)
    _same_f32(totals[row], expected_total, f'{label}.{field}.totals[{row}]')
    _same_f32(means[row], expected_mean, f'{label}.{field}.means[{row}]')
    reconstructed_logits.append(clean_logits)
    reconstructed_margins.append(clean_margins)
    reconstructed_totals.append(expected_total)
    reconstructed_means.append(expected_mean)
  return {
      'selected_logits': reconstructed_logits,
      'endpoint_margins': reconstructed_margins,
      'totals': reconstructed_totals,
      'means': reconstructed_means,
  }


def _row_bits(readout: Mapping[str, Any], row: int) -> tuple[Any, ...]:
  return (
      tuple(
          tuple(_f32_bits(value) for value in endpoint)
          for endpoint in readout['selected_logits'][row]
      ),
      tuple(_f32_bits(value) for value in readout['endpoint_margins'][row]),
      _f32_bits(readout['totals'][row]),
      _f32_bits(readout['means'][row]),
  )


def _readouts_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
  return all(_row_bits(left, row) == _row_bits(right, row) for row in range(6))


def _fingerprints_equal(left: Any, right: Any) -> bool:
  if not isinstance(left, Mapping) or not isinstance(right, Mapping):
    return False
  sha = left.get('sha256')
  return (
      isinstance(sha, str)
      and len(sha) == 64
      and sha == right.get('sha256')
      and left.get('leaves') == right.get('leaves')
  )


def _validate_readout_repeat(
    record: Mapping[str, Any], label: str
) -> tuple[dict[str, Any], bool]:
  first = _readout(record, 'target_readout', label)
  repeat = _readout(record, 'repeat_target_readout', label)
  return first, _readouts_equal(first, repeat)


def _case_is_effect(case: Mapping[str, Any]) -> bool:
  selection_class = str(case.get('selection_class', '')).lower()
  if not selection_class:
    raise ValueError('Case selection_class is missing.')
  return 'neutral' not in selection_class


def _load_frozen_cases() -> tuple[dict[str, Any], ...]:
  """Loads only committed 20-row/two-exon development projections."""
  if _sha256(_SELECTED_PATH) != DEVELOPMENT_VARIANTS_SHA256:
    raise ValueError('Frozen development-variant projection hash mismatch.')
  if _sha256(_EXONS_PATH) != DEVELOPMENT_EXONS_SHA256:
    raise ValueError('Frozen development-exon projection hash mismatch.')
  exons: dict[str, dict[str, Any]] = {}
  with _EXONS_PATH.open('r', encoding='utf-8', newline='') as handle:
    reader = csv.DictReader(handle, delimiter='\t')
    for row in reader:
      order = int(row['selection_order'])
      if order not in (1, 2) or row['gene'] not in DEVELOPMENT_GENES:
        raise ValueError('Development exon allowlist changed.')
      exons[row['ensembl_exon_id']] = row
  if len(exons) != 2:
    raise ValueError('Frozen development exon table is incomplete.')
  cases = []
  with _SELECTED_PATH.open('r', encoding='utf-8', newline='') as handle:
    reader = csv.DictReader(handle, delimiter='\t')
    rows = list(reader)
    if len(rows) != EXPECTED_IDENTITIES:
      raise ValueError('Development variant projection must have exactly 20 rows.')
    for source_order, row in enumerate(rows):
      if row['gene'] not in DEVELOPMENT_GENES:
        raise ValueError('A non-development gene entered orders 0--19.')
      try:
        exon = exons[row['ensembl_exon_id']]
      except KeyError as error:
        raise ValueError('Development case references an unfrozen exon.') from error
      strand = exon['strand']
      if strand not in ('+', '-'):
        raise ValueError('Development exon has invalid strand.')
      cases.append({
          'order': source_order,
          'selection_version': row['selection_version'],
          'selection_class': row['selection_class'],
          'observed_effect_sign': row['observed_effect_sign'].strip().lower(),
          'gene': exon['gene'],
          'exon_id': exon['exon_id'],
          'ensembl_exon_id': exon['ensembl_exon_id'],
          'chromosome': (
              exon['chromosome'] if exon['chromosome'].startswith('chr')
              else f"chr{exon['chromosome']}"
          ),
          'strand': strand,
          'exon_start_1based': int(exon['exon_start_1based']),
          'exon_end_1based': int(exon['exon_end_1based']),
          'variant_id': row['variant_id'],
          'position_1based': int(row['position_1based']),
          'reference_bases': row['reference_bases'].upper(),
          'alternate_bases': row['alternate_bases'].upper(),
          'region': row['region'],
          'mut_type': row['mut_type'],
          'delta_psi': float(row['delta_psi']),
          'delta_logit': float(row['delta_logit']),
      })
  return tuple(cases)


def _validate_case(observed: Any, expected: Mapping[str, Any], label: str) -> None:
  if not isinstance(observed, Mapping):
    raise ValueError(f'{label} case is missing.')
  if any(str(observed.get('gene', '')).upper() == gene for gene in FORBIDDEN_GENES):
    raise ValueError(f'{label} contains a forbidden confirmation gene.')
  if observed.get('gene') not in DEVELOPMENT_GENES:
    raise ValueError(f'{label} contains a non-development gene.')
  if dict(observed) != dict(expected):
    raise ValueError(f'{label} case differs from frozen manifest order.')


def _validate_canonical_target(
    record: Mapping[str, Any], case: Mapping[str, Any], label: str
) -> None:
  interval = record.get('interval')
  target = record.get('canonical_target')
  if not isinstance(interval, Mapping) or not isinstance(target, Mapping):
    raise ValueError(f'{label} is missing interval/canonical target metadata.')
  start = interval.get('start_0based')
  end = interval.get('end_0based_exclusive')
  if (
      interval.get('chromosome') != case['chromosome']
      or not isinstance(start, int)
      or not isinstance(end, int)
      or end - start != 16_384
  ):
    raise ValueError(f'{label} interval violates the frozen 16,384-bp target.')
  if case['strand'] == '+':
    expected = (
        ('acceptor', case['exon_start_1based'], 1),
        ('donor', case['exon_end_1based'], 0),
    )
  else:
    expected = (
        ('acceptor', case['exon_end_1based'], 3),
        ('donor', case['exon_start_1based'], 2),
    )
  endpoints = target.get('endpoints')
  if not isinstance(endpoints, list) or len(endpoints) != 2:
    raise ValueError(f'{label} must have exactly [acceptor, donor] endpoints.')
  for endpoint, (role, position, track) in zip(endpoints, expected, strict=True):
    expected_index = position - 1 - start
    if not isinstance(endpoint, Mapping) or (
        endpoint.get('role'),
        endpoint.get('position_1based'),
        endpoint.get('position_index'),
        endpoint.get('track_index'),
    ) != (role, position, expected_index, track):
      raise ValueError(f'{label} strand/class endpoint mapping is invalid.')
  if target.get('padding_track_index') != 4:
    raise ValueError(f'{label} padding-class track must be index 4.')


def _validate_freeze(
    start: Mapping[str, Any], *, bundle_root: Path
) -> tuple[dict[str, Any], str]:
  _guard_path(bundle_root)
  if start.get('attempt_id') != ATTEMPT_ID:
    raise ValueError('Attempt identifier mismatch.')
  if start.get('script_version') != SOURCE_SCRIPT_VERSION:
    raise ValueError('Attempt source-script version mismatch.')
  if start.get('status') != 'started_append_only_one_shot':
    raise ValueError('Attempt was not recorded as append-only one-shot.')
  if start.get('protocol_sha256') != PROTOCOL_SHA256:
    raise ValueError('Attempt protocol hash mismatch.')
  if start.get('compile_count_contract') != 1:
    raise ValueError('Attempt did not freeze one compilation.')
  if start.get('confirmation_model_calls') != 0:
    raise ValueError('Attempt reports a confirmation model call.')
  freeze = start.get('freeze')
  if not isinstance(freeze, Mapping):
    raise ValueError('Attempt has no bound freeze object.')
  freeze_path_value = freeze.get('path')
  freeze_sha = freeze.get('sha256')
  if not isinstance(freeze_path_value, str) or not isinstance(freeze_sha, str):
    raise ValueError('Attempt freeze path/hash is missing.')
  freeze_path = Path(freeze_path_value).resolve()
  _guard_path(freeze_path)
  if _sha256(freeze_path) != freeze_sha:
    raise ValueError('Bound freeze-file hash mismatch.')
  frozen_file = _read_json(freeze_path)
  embedded = {key: value for key, value in freeze.items() if key not in {'path', 'sha256'}}
  if frozen_file != embedded:
    raise ValueError('Embedded freeze differs from its bound file.')
  expected = {
      'script_version': SOURCE_SCRIPT_VERSION,
      'attempt_id': ATTEMPT_ID,
      'protocol_sha256': PROTOCOL_SHA256,
      'development_variants_sha256': DEVELOPMENT_VARIANTS_SHA256,
      'development_exons_sha256': DEVELOPMENT_EXONS_SHA256,
      'selected_variants_sha256': SELECTED_VARIANTS_SHA256,
      'frozen_exons_sha256': FROZEN_EXONS_SHA256,
      'checkpoint_snapshot': CHECKPOINT_SNAPSHOT,
      'context_bp': 16_384,
      'attention_backend': 'dense',
      'preflight_script_version': 'opensplice-device-preflight-v3.2.0',
      'reference_url': _REFERENCE_URL,
      'reference_object': _REFERENCE_OBJECT,
      'expected_device_kind': EXPECTED_DEVICE_KIND,
      'expected_gpu_uuid': EXPECTED_GPU_UUID,
      'expected_compute_capability': EXPECTED_COMPUTE_CAPABILITY,
      'mixed_precision_policy': (
          'params=float32,compute=bfloat16,output=bfloat16'
      ),
      'environment_contract': {
          'LD_LIBRARY_PATH': 'absent',
          'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
          'JAX_ENABLE_COMPILATION_CACHE': 'false',
          'compiler_and_autotune_cache_inputs': 'absent',
      },
      'paired_batch_roles': list(TRACE_ROLES),
      'paired_batch_donor_rows': [0, 1, 0, 1, 1, 0],
      'phase_r_groups_per_eligible_effect': EXPECTED_GROUPS_PER_ELIGIBLE,
      'phase_r_candidate_count': EXPECTED_CANDIDATES,
      'stage_a_component_keys': list(STAGE_COMPONENTS),
  }
  for key, value in expected.items():
    if freeze.get(key) != value:
      raise ValueError(f'Frozen contract mismatch at {key}.')
  for key in (
      'development_variants_path', 'development_exons_path',
      'checkpoint_manifest_path', 'reference_bindings_path',
      'output_dir', 'analysis_dir', 'preflight_dir',
  ):
    value = freeze.get(key)
    if not isinstance(value, str):
      raise ValueError(f'Frozen {key} path is missing.')
    _guard_path(Path(value))
  for key in ('checkpoint_manifest_sha256', 'reference_bindings_sha256'):
    if not _is_sha256(freeze.get(key)):
      raise ValueError(f'Frozen {key} is not a lowercase SHA-256 digest.')
  upstream_head = freeze.get('upstream_alphagenome_git_head')
  if (
      not isinstance(upstream_head, str) or len(upstream_head) != 40
      or any(character not in '0123456789abcdef' for character in upstream_head)
  ):
    raise ValueError('Frozen upstream AlphaGenome Git HEAD is invalid.')
  checkpoint_path = start.get('checkpoint_path')
  if not isinstance(checkpoint_path, str):
    raise ValueError('Attempt checkpoint path is missing.')
  _guard_path(Path(checkpoint_path))
  external = start.get('external_preflight')
  if not isinstance(external, Mapping) or not isinstance(external.get('path'), str):
    raise ValueError('Attempt external-preflight path is missing.')
  _guard_path(Path(external['path']))
  hashes = freeze.get('file_sha256')
  if not isinstance(hashes, Mapping) or not hashes:
    raise ValueError('Freeze has no bound implementation files.')
  for relative, digest in hashes.items():
    if not isinstance(relative, str) or not isinstance(digest, str):
      raise ValueError('Freeze file-hash mapping is malformed.')
    path = (bundle_root / relative).resolve()
    try:
      path.relative_to(bundle_root.resolve())
    except ValueError as error:
      raise ValueError('Freeze file escapes the repository root.') from error
    _guard_path(path)
    if _sha256(path) != digest:
      raise ValueError(f'Frozen implementation file changed: {relative}.')
  return dict(freeze), freeze_sha


def _validate_runtime_environment(environment: Any, label: str) -> None:
  if not isinstance(environment, Mapping):
    raise ValueError(f'{label} runtime environment is missing.')
  if environment.get('JAX_ENABLE_COMPILATION_CACHE') != 'false':
    raise ValueError(f'{label} compilation cache was not disabled.')
  if environment.get('XLA_PYTHON_CLIENT_PREALLOCATE') != 'false':
    raise ValueError(f'{label} preallocation was not disabled.')
  forbidden = []
  for name in environment:
    upper = str(name).upper()
    if name in {'XLA_FLAGS', 'JAX_COMPILATION_CACHE_DIR'}:
      forbidden.append(name)
    if name.startswith('JAX_PERSISTENT_CACHE_'):
      forbidden.append(name)
    if 'AUTOTUNE' in upper and any(word in upper for word in ('LOAD', 'DUMP', 'CACHE')):
      forbidden.append(name)
  if forbidden:
    raise ValueError(f'{label} contains forbidden cache/autotune flags: {forbidden}.')
  allowed_prefixes = ('XLA', 'JAX', 'CUDA', 'CUDNN', 'CUBLAS', 'TRITON')
  if any(not str(name).startswith(allowed_prefixes) for name in environment):
    raise ValueError(f'{label} contains an unscoped environment variable.')


def _validate_device_observation(observation: Any, label: str) -> None:
  if not isinstance(observation, Mapping):
    raise ValueError(f'{label} device observation is missing.')
  if observation.get('jax_default_backend') != 'gpu':
    raise ValueError(f'{label} JAX backend is not GPU.')
  devices = observation.get('jax_gpu_devices')
  if not isinstance(devices, list) or len(devices) != 1:
    raise ValueError(f'{label} must expose exactly one JAX GPU.')
  device = devices[0]
  if not isinstance(device, Mapping) or (
      device.get('device_kind') != EXPECTED_DEVICE_KIND
      or device.get('platform') != 'gpu'
      or device.get('client_platform') != 'gpu'
  ):
    raise ValueError(f'{label} JAX device is not the frozen RTX 3090.')
  nvidia = observation.get('nvidia_smi')
  parsed = nvidia.get('parsed_single_gpu') if isinstance(nvidia, Mapping) else None
  if not isinstance(parsed, Mapping) or (
      parsed.get('name') != EXPECTED_DEVICE_KIND
      or parsed.get('uuid') != EXPECTED_GPU_UUID
      or str(parsed.get('compute_capability')) != EXPECTED_COMPUTE_CAPABILITY
  ):
    raise ValueError(f'{label} nvidia-smi identity/UUID/CC mismatch.')
  environment = observation.get('environment')
  if not isinstance(environment, Mapping):
    raise ValueError(f'{label} environment observation is missing.')
  ld = environment.get('LD_LIBRARY_PATH')
  if not isinstance(ld, Mapping) or ld.get('present') is not False or ld.get('value') is not None:
    raise ValueError(f'{label} LD_LIBRARY_PATH was not absent.')
  if environment.get('XLA_PYTHON_CLIENT_PREALLOCATE') != 'false':
    raise ValueError(f'{label} preallocation observation mismatch.')


def _validate_preflight(
    start: Mapping[str, Any], freeze: Mapping[str, Any], freeze_sha: str
) -> dict[str, Any]:
  _validate_runtime_environment(start.get('runtime_environment'), 'START')
  same_process = start.get('same_process_preflight')
  _validate_device_observation(same_process, 'same-process preflight')
  external = start.get('external_preflight')
  if not isinstance(external, Mapping):
    raise ValueError('External preflight binding is missing.')
  path_value, digest = external.get('path'), external.get('sha256')
  if not isinstance(path_value, str) or not isinstance(digest, str):
    raise ValueError('External preflight path/hash is missing.')
  path = Path(path_value).resolve()
  _guard_path(path)
  if _sha256(path) != digest:
    raise ValueError('External preflight artifact hash mismatch.')
  raw = _read_json(path)
  embedded = {key: value for key, value in external.items() if key not in {'path', 'sha256'}}
  if raw != embedded:
    raise ValueError('Embedded external preflight differs from its artifact.')
  if (
      raw.get('script_version') != 'opensplice-device-preflight-v3.2.0'
      or raw.get('status') != 'pass'
      or raw.get('protocol_sha256') != PROTOCOL_SHA256
      or raw.get('freeze_sha256') != freeze_sha
      or raw.get('no_model_or_biological_access') is not True
      or raw.get('no_jit_or_array_kernel') is not True
      or raw.get('failure') is not None
  ):
    raise ValueError('External preflight contract did not pass exactly.')
  preflight_freeze = raw.get('freeze')
  if not isinstance(preflight_freeze, Mapping) or (
      preflight_freeze.get('path') != freeze.get('path')
      or preflight_freeze.get('sha256') != freeze_sha
      or preflight_freeze.get('tracked_clean') is not True
  ):
    raise ValueError('External preflight freeze/bundle binding mismatch.')
  if Path(freeze['preflight_dir']).resolve() != path.parent:
    raise ValueError('External preflight escaped the frozen preflight directory.')
  logs = raw.get('logs')
  if not isinstance(logs, Mapping) or set(logs) != {'stdout', 'stderr'}:
    raise ValueError('External preflight log bindings are incomplete.')
  for stream in ('stdout', 'stderr'):
    binding = logs[stream]
    if not isinstance(binding, Mapping) or not isinstance(binding.get('path'), str):
      raise ValueError(f'External preflight {stream} binding is malformed.')
    log_path = Path(binding['path']).resolve()
    _guard_path(log_path)
    if log_path.parent != path.parent or _sha256(log_path) != binding.get('sha256'):
      raise ValueError(f'External preflight {stream} log hash/path mismatch.')
  _validate_device_observation(raw.get('observation'), 'external preflight')
  external_observation = raw['observation']
  _validate_runtime_environment(
      external_observation.get('v3_2_runtime_environment'),
      'external preflight',
  )
  if external_observation.get('v3_2_runtime_environment') != start.get(
      'runtime_environment'
  ):
    raise ValueError(
        'External and same-process START runtime environments differ.'
    )
  if external_observation.get('jax_enable_compilation_cache') is not False:
    raise ValueError('External preflight JAX compilation cache was enabled.')
  return {
      'external_preflight_sha256': digest,
      'external_preflight_logs_verified': True,
      'external_exact_rtx3090_uuid_gate': True,
      'same_process_exact_rtx3090_uuid_gate': True,
      'runtime_environment_gate': True,
  }


def _validate_checkpoint_and_reference_inputs(
    start: Mapping[str, Any], freeze: Mapping[str, Any],
    expected_cases: Sequence[Mapping[str, Any]], *,
    enforce_standard_locations: bool,
) -> dict[str, Any]:
  """Independently binds all immutable model/reference inputs, without I/O."""
  manifest_path = Path(str(freeze['checkpoint_manifest_path'])).resolve()
  reference_path = Path(str(freeze['reference_bindings_path'])).resolve()
  for path in (manifest_path, reference_path):
    _guard_path(path)
  if enforce_standard_locations:
    standard_paths = {
        'development_variants_path': _SELECTED_PATH.resolve(),
        'development_exons_path': _EXONS_PATH.resolve(),
        'checkpoint_manifest_path': _CHECKPOINT_MANIFEST_PATH.resolve(),
        'reference_bindings_path': _REFERENCE_BINDINGS_PATH.resolve(),
    }
    for name, expected in standard_paths.items():
      if Path(str(freeze[name])).resolve() != expected:
        raise ValueError(f'Frozen standard input path changed at {name}.')
    if freeze['checkpoint_manifest_sha256'] != CHECKPOINT_MANIFEST_SHA256:
      raise ValueError('Frozen production checkpoint-manifest hash changed.')
    if freeze['reference_bindings_sha256'] != REFERENCE_BINDINGS_SHA256:
      raise ValueError('Frozen production reference-binding hash changed.')
    if set(freeze['file_sha256']) != _expected_production_bundle_paths():
      raise ValueError('Frozen production file-hash inventory is not exact.')
  if _sha256(manifest_path) != freeze['checkpoint_manifest_sha256']:
    raise ValueError('Checkpoint-manifest content differs from the freeze.')
  if _sha256(reference_path) != freeze['reference_bindings_sha256']:
    raise ValueError('Reference-sequence binding content differs from the freeze.')

  checkpoint_path_value = start.get('checkpoint_path')
  if not isinstance(checkpoint_path_value, str):
    raise ValueError('Attempt checkpoint path is missing.')
  checkpoint_path = Path(checkpoint_path_value).resolve()
  _guard_path(checkpoint_path)
  if checkpoint_path.name != CHECKPOINT_SNAPSHOT or not checkpoint_path.is_dir():
    raise ValueError('Checkpoint snapshot path/name is not the frozen snapshot.')
  manifest_records = []
  manifest_files = []
  seen_relative = set()
  try:
    manifest_lines = manifest_path.read_text(encoding='utf-8').splitlines()
  except OSError as error:
    raise ValueError('Cannot read frozen checkpoint manifest.') from error
  for line_number, line in enumerate(manifest_lines, start=1):
    fields = line.split('\t')
    if len(fields) != 3:
      raise ValueError(
          f'Checkpoint manifest row {line_number} does not have three columns.'
      )
    relative, size_text, digest = fields
    relative_path = Path(relative)
    if (
        not relative or relative_path.is_absolute() or '..' in relative_path.parts
        or relative in seen_relative or not _is_sha256(digest)
    ):
      raise ValueError(f'Checkpoint manifest row {line_number} is unsafe/invalid.')
    try:
      size = int(size_text)
    except ValueError as error:
      raise ValueError('Checkpoint manifest size is not an integer.') from error
    if size < 0 or str(size) != size_text:
      raise ValueError('Checkpoint manifest size is non-canonical.')
    path = checkpoint_path / relative_path
    _guard_path(path)
    if path.is_symlink() or not path.is_file():
      raise ValueError(f'Checkpoint file is missing or symlinked: {relative}.')
    try:
      path.resolve().relative_to(checkpoint_path)
    except ValueError as error:
      raise ValueError('Checkpoint manifest entry escapes its snapshot.') from error
    if path.stat().st_size != size or _sha256(path) != digest:
      raise ValueError(f'Checkpoint file content changed: {relative}.')
    seen_relative.add(relative)
    manifest_files.append(path)
    manifest_records.append({
        'relative_path': relative, 'size_bytes': size, 'sha256': digest,
    })
  if (
      len(manifest_records) != 12
      or [row['relative_path'] for row in manifest_records]
      != sorted(row['relative_path'] for row in manifest_records)
  ):
    raise ValueError('Checkpoint manifest must contain exactly 12 sorted files.')
  observed_checkpoint_files = sorted(
      str(path.relative_to(checkpoint_path))
      for path in checkpoint_path.rglob('*') if path.is_file()
  )
  if observed_checkpoint_files != [
      row['relative_path'] for row in manifest_records
  ]:
    raise ValueError('Checkpoint tree differs from its exact 12-file manifest.')
  expected_checkpoint_binding = {
      'snapshot_path': str(checkpoint_path),
      'snapshot_name': CHECKPOINT_SNAPSHOT,
      'manifest_path': str(manifest_path),
      'manifest_sha256': freeze['checkpoint_manifest_sha256'],
      'file_count': 12,
      'files': manifest_records,
  }
  if start.get('checkpoint_binding') != expected_checkpoint_binding:
    raise ValueError('START checkpoint binding differs from the verified tree.')

  reference_binding = _read_json(reference_path)
  if set(reference_binding) != {'reference_url', 'context_bp', 'cases'}:
    raise ValueError('Reference-sequence binding has unexpected/missing keys.')
  if (
      reference_binding['reference_url'] != _REFERENCE_URL
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
        or not _is_sha256(row[4]) or not _is_sha256(row[5])
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
      **_REFERENCE_OBJECT,
      'observed_generation': _REFERENCE_OBJECT['generation'],
      'observed_size_bytes': _REFERENCE_OBJECT['size_bytes'],
      'observed_etag': _REFERENCE_OBJECT['etag'],
      'observed_md5_base64': _REFERENCE_OBJECT['md5_base64'],
      'observed_crc32c_base64': _REFERENCE_OBJECT['crc32c_base64'],
  }
  if start.get('reference_object_binding') != expected_reference_object_binding:
    raise ValueError('START GCS reference-object metadata binding changed.')
  return {
      'checkpoint_manifest_sha256': freeze['checkpoint_manifest_sha256'],
      'checkpoint_file_count': 12,
      'checkpoint_tree_sha256': _tree_digest(manifest_files, checkpoint_path),
      'reference_object': dict(_REFERENCE_OBJECT),
      'reference_object_metadata_verified': True,
      'reference_bindings_sha256': freeze['reference_bindings_sha256'],
      'sequence_bindings': sequence_bindings,
  }


def _validate_import_provenance(
    path: Path, expected_sha: Any, *, bundle_root: Path
) -> dict[str, Any]:
  if not isinstance(expected_sha, str) or _sha256(path) != expected_sha:
    raise ValueError('IMPORT_PROVENANCE hash binding mismatch.')
  value = _read_json(path)
  modules = value.get('modules')
  if not isinstance(modules, list) or value.get('module_count') != len(modules) or not modules:
    raise ValueError('IMPORT_PROVENANCE module list/count is invalid.')
  upstream_root = (bundle_root.parent / 'alphagenome').resolve()
  seen_names = set()
  seen_paths = set()
  for index, module in enumerate(modules):
    if not isinstance(module, Mapping):
      raise ValueError('IMPORT_PROVENANCE module row is malformed.')
    name, path_value, root = (
        module.get('name'), module.get('path'), module.get('root')
    )
    if not isinstance(name, str) or not isinstance(path_value, str):
      raise ValueError('IMPORT_PROVENANCE name/path is malformed.')
    module_path = Path(path_value).resolve()
    _guard_path(module_path)
    expected_root = (
        bundle_root if root == 'alphagenome_research_checkout'
        else upstream_root if root == 'upstream_alphagenome_checkout'
        else None
    )
    if expected_root is None:
      raise ValueError('IMPORT_PROVENANCE contains an undeclared root.')
    try:
      module_path.relative_to(expected_root)
    except ValueError as error:
      raise ValueError('IMPORT_PROVENANCE module escaped its declared root.') from error
    if name in seen_names or str(module_path) in seen_paths:
      raise ValueError('IMPORT_PROVENANCE contains duplicate module/path rows.')
    seen_names.add(name)
    seen_paths.add(str(module_path))
    if _sha256(module_path) != module.get('sha256'):
      raise ValueError(f'Imported module hash changed: {name}.')
    if module_path.stat().st_size != module.get('size_bytes'):
      raise ValueError(f'Imported module size changed: {name}.')
  required_names = {
      'alphagenome_research.model.model',
      'alphagenome_research.model.dna_model',
      'target_reducers_v3',
  }
  if not required_names.issubset(seen_names):
    raise ValueError(
        f'IMPORT_PROVENANCE misses required transitive modules: '
        f'{sorted(required_names - seen_names)}.'
    )
  if not any(Path(path).name == 'run_superset_graph_v3_2.py' for path in seen_paths):
    raise ValueError('IMPORT_PROVENANCE does not bind the executed runner.')
  return {
      'sha256': expected_sha,
      'module_count': len(modules),
      'modules': modules,
      'roots': [
          'alphagenome_research_checkout',
          'upstream_alphagenome_checkout',
      ],
  }


def _validate_import_phases(
    run_dir: Path, complete: Mapping[str, Any], *, bundle_root: Path
) -> dict[str, Any]:
  filenames = {
      'pre_model': 'IMPORT_PROVENANCE_PRE_MODEL.json',
      'post_model_precompile': 'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
      'postcompile': 'IMPORT_PROVENANCE.json',
  }
  bindings = complete.get('import_provenance_phases')
  if not isinstance(bindings, Mapping) or set(bindings) != set(filenames):
    raise ValueError('RUN_COMPLETE import-provenance phases are incomplete.')
  phases = {}
  for name, filename in filenames.items():
    phases[name] = _validate_import_provenance(
        run_dir / filename, bindings[name], bundle_root=bundle_root
    )
  if complete.get('import_provenance_sha256') != bindings['postcompile']:
    raise ValueError('Legacy/final import-provenance hash binding differs.')
  lazy_additions = {}
  for earlier_name, later_name in (
      ('pre_model', 'post_model_precompile'),
      ('post_model_precompile', 'postcompile'),
  ):
    earlier = {row['name']: row for row in phases[earlier_name]['modules']}
    later = {row['name']: row for row in phases[later_name]['modules']}
    missing = sorted(set(earlier) - set(later))
    changed = sorted(
        name for name in earlier if name in later and earlier[name] != later[name]
    )
    if missing or changed:
      raise ValueError(
          f'Import provenance changed across phases: missing={missing}, '
          f'changed={changed}.'
      )
    lazy_additions[f'{earlier_name}_to_{later_name}'] = sorted(
        set(later) - set(earlier)
    )
  return {
      'phase_sha256': dict(bindings),
      'module_counts': {
          name: value['module_count'] for name, value in phases.items()
      },
      'lazy_additions': lazy_additions,
      'stable_shared_module_bytes': True,
  }


def _validate_protobuf_provenance(
    path: Path, expected_sha: Any, freeze: Mapping[str, Any]
) -> dict[str, Any]:
  if not isinstance(expected_sha, str) or _sha256(path) != expected_sha:
    raise ValueError('PROTOBUF_PROVENANCE hash binding mismatch.')
  value = _read_json(path)
  if value != freeze.get('protobuf_binding'):
    raise ValueError('PROTOBUF_PROVENANCE differs from the frozen binding.')

  def validate_paths(node: Any) -> None:
    if isinstance(node, Mapping):
      if 'path' in node:
        if not isinstance(node['path'], str):
          raise ValueError('Protobuf provenance path is malformed.')
        bound = Path(node['path']).resolve()
        _guard_path(bound)
        if 'sha256' in node and _sha256(bound) != node['sha256']:
          raise ValueError(f'Protobuf provenance hash mismatch: {bound}.')
        if 'size_bytes' in node and bound.stat().st_size != node['size_bytes']:
          raise ValueError(f'Protobuf provenance size mismatch: {bound}.')
      for child in node.values():
        validate_paths(child)
    elif isinstance(node, list):
      for child in node:
        validate_paths(child)

  validate_paths(value)
  if value.get('regeneration_claim') is not False:
    raise ValueError('Protobuf provenance makes an unsupported regeneration claim.')
  if value.get('current_protoc_was_used_to_generate_frozen_outputs') is not False:
    raise ValueError('Protobuf provenance misstates current protoc usage.')
  return {'sha256': expected_sha, 'binding_verified': True}


def _validate_compiler_provenance(
    compiler: Any, path: Path, run_dir: Path
) -> str:
  if not isinstance(compiler, Mapping) or set(compiler) != {
      'compile_count', 'compile_seconds', 'executable_fingerprint', 'artifacts'
  }:
    raise ValueError('Compiler provenance top-level schema changed.')
  if compiler.get('compile_count') != 1:
    raise ValueError('Run did not use exactly one compiled executable.')
  _finite(compiler.get('compile_seconds'), 'compiler.compile_seconds')
  artifacts = compiler.get('artifacts')
  expected_names = {
      'stablehlo': 'superset.stablehlo.mlir',
      'hlo': 'superset.pre_backend.hlo.txt',
      'compiled_hlo': 'superset.compiled.hlo.txt',
  }
  if not isinstance(artifacts, Mapping) or set(artifacts) != set(expected_names):
    raise ValueError('Compiler provenance must bind exactly three IR artifacts.')
  expected_paths = set()
  for name, filename in expected_names.items():
    binding = artifacts[name]
    if not isinstance(binding, Mapping) or set(binding) != {
        'path', 'sha256', 'size_bytes'
    }:
      raise ValueError(f'Compiler {name} binding schema changed.')
    if not isinstance(binding['path'], str):
      raise ValueError(f'Compiler {name} path is malformed.')
    artifact_path = Path(binding['path']).resolve()
    _guard_path(artifact_path)
    expected_path = (run_dir / 'compiler' / filename).resolve()
    if artifact_path != expected_path:
      raise ValueError(f'Compiler {name} artifact path changed/escaped.')
    if (
        _sha256(artifact_path) != binding['sha256']
        or artifact_path.stat().st_size != binding['size_bytes']
    ):
      raise ValueError(f'Compiler {name} artifact hash/size mismatch.')
    expected_paths.add(artifact_path)
  observed_files = {
      item.resolve() for item in (run_dir / 'compiler').iterdir()
      if item.is_file()
  }
  if observed_files != expected_paths | {path.resolve()}:
    raise ValueError('Compiler directory contains missing or extra files.')
  compiled_digest = artifacts['compiled_hlo']['sha256']
  recomputed_executable = hashlib.sha256(bytes.fromhex(compiled_digest)).hexdigest()
  if compiler.get('executable_fingerprint') != recomputed_executable:
    raise ValueError('Executable fingerprint does not match compiled HLO.')
  return recomputed_executable


def _validate_bootstrap_attestation(
    start: Mapping[str, Any], freeze: Mapping[str, Any], freeze_sha: str, *,
    bundle_root: Path,
) -> dict[str, Any]:
  record = start.get('same_process_pre_import_bootstrap')
  if not isinstance(record, Mapping):
    raise ValueError('START pre-import bootstrap attestation is missing.')
  if not isinstance(record.get('pid'), int) or record['pid'] <= 0:
    raise ValueError('Bootstrap attestation PID is invalid.')
  _finite(record.get('created_at_unix_s'), 'bootstrap.created_at_unix_s')
  attested_freeze = record.get('freeze')
  if not isinstance(attested_freeze, Mapping) or (
      attested_freeze.get('path') != freeze.get('path')
      or attested_freeze.get('sha256') != freeze_sha
  ):
    raise ValueError('Bootstrap attestation freeze binding mismatch.')
  git_head = attested_freeze.get('git_head')
  if (
      not isinstance(git_head, str) or len(git_head) != 40
      or any(character not in '0123456789abcdef' for character in git_head)
      or attested_freeze.get('tracked_head_clean') is not True
  ):
    raise ValueError('Bootstrap committed-HEAD gate is missing or invalid.')
  bundle = start.get('bundle')
  external_freeze = start.get('external_preflight', {}).get('freeze', {})
  if (
      not isinstance(bundle, Mapping)
      or bundle.get('git_head') != git_head
      or bundle.get('tracked_clean') is not True
      or not isinstance(external_freeze, Mapping)
      or external_freeze.get('git_head') != git_head
      or external_freeze.get('tracked_clean') is not True
  ):
    raise ValueError('Bootstrap/bundle/external-preflight commit binding differs.')
  try:
    freeze_relative = str(Path(str(freeze['path'])).resolve().relative_to(bundle_root))
  except ValueError as error:
    raise ValueError('Frozen contract file escapes the repository root.') from error
  protocol_relative = str(_PROTOCOL_PATH.relative_to(_REPO_ROOT))
  expected_tracked_paths = sorted({
      freeze_relative, protocol_relative, *freeze['file_sha256'].keys(),
  })
  if attested_freeze.get('tracked_paths') != expected_tracked_paths:
    raise ValueError('Bootstrap tracked-path inventory differs from the freeze.')
  expected_scripts = {
      'launcher': (
          bundle_root
          / 'experiments/interpretability/opensplice/launch_superset_graph_v3_2.py'
      ),
      'bootstrap': (
          bundle_root
          / 'experiments/interpretability/opensplice/'
          'validate_superset_graph_bootstrap_v3_2.py'
      ),
  }
  frozen_hashes = freeze.get('file_sha256')
  for name, expected_path in expected_scripts.items():
    expected_path = expected_path.resolve()
    _guard_path(expected_path)
    relative = str(expected_path.relative_to(bundle_root))
    observed_hash = _sha256(expected_path)
    if (
        record.get(f'{name}_path') != str(expected_path)
        or record.get(f'{name}_sha256') != observed_hash
        or not isinstance(frozen_hashes, Mapping)
        or frozen_hashes.get(relative) != observed_hash
    ):
      raise ValueError(f'Bootstrap {name} path/hash is not frozen exactly.')
  generated = record.get('generated_bindings')
  if not isinstance(generated, Mapping) or (
      generated.get('pre_import_gate') is not True
      or generated.get('exact_regeneration_claim') is not False
      or generated.get('protobuf_runtime_version') != '7.35.1'
  ):
    raise ValueError('Generated-binding pre-import gate did not pass exactly.')
  expected_exception = [
      '?? src/alphagenome_research/protos/calibration_scores_pb2.py',
      '?? src/alphagenome_research/protos/calibration_scores_pb2.pyi',
  ]
  if generated.get('generated_artifact_exception') != expected_exception:
    raise ValueError('Generated-binding untracked exception changed.')
  header = generated.get('embedded_header')
  if not isinstance(header, list) or not any(
      'Protobuf Python Version: 7.35.1' in str(line) for line in header
  ):
    raise ValueError('Generated-binding protobuf header/version is missing.')
  artifacts = generated.get('artifacts')
  expected_names = {
      'source_proto', 'generated_pb2', 'generated_pyi', 'dependency_proto',
      'dependency_pb2', 'tensor_proto', 'tensor_pb2',
  }
  if not isinstance(artifacts, Mapping) or set(artifacts) != expected_names:
    raise ValueError('Pre-import generated-binding artifact set changed.')

  frozen_paths: dict[str, tuple[Any, Any]] = {}
  def collect(node: Any) -> None:
    if isinstance(node, Mapping):
      if isinstance(node.get('path'), str) and 'sha256' in node:
        frozen_paths[str(Path(node['path']).resolve())] = (
            node.get('sha256'), node.get('size_bytes')
        )
      for child in node.values():
        collect(child)
    elif isinstance(node, list):
      for child in node:
        collect(child)
  collect(freeze.get('protobuf_binding'))

  roots = (bundle_root.resolve(), (bundle_root.parent / 'alphagenome').resolve())
  for name, binding in artifacts.items():
    if not isinstance(binding, Mapping) or set(binding) != {
        'path', 'sha256', 'size_bytes'
    }:
      raise ValueError(f'Bootstrap generated artifact {name} schema changed.')
    path = Path(binding['path']).resolve()
    _guard_path(path)
    if not any(path.is_relative_to(root) for root in roots):
      raise ValueError(f'Bootstrap generated artifact {name} escaped roots.')
    if (
        _sha256(path) != binding['sha256']
        or path.stat().st_size != binding['size_bytes']
    ):
      raise ValueError(f'Bootstrap generated artifact {name} hash/size mismatch.')
    if name not in {'tensor_proto', 'tensor_pb2'}:
      frozen = frozen_paths.get(str(path))
      if frozen is None or frozen[0] != binding['sha256'] or (
          frozen[1] is not None and frozen[1] != binding['size_bytes']
      ):
        raise ValueError(
            f'Bootstrap artifact {name} differs from frozen protobuf binding.'
        )
  return {
      'same_process_pre_import_bootstrap_verified': True,
      'bootstrap_pid': record['pid'],
      'generated_binding_artifact_count': len(artifacts),
  }


def _validate_position_set(value: Any, expected_name: str, label: str) -> None:
  if not isinstance(value, Mapping) or value.get('name') != expected_name:
    raise ValueError(f'{label} position-set name mismatch.')
  candidate = expected_name.split('_control_', maxsplit=1)[0]
  control = '_control_' in expected_name
  if value.get('role') != ('width_matched_control' if control else 'candidate'):
    raise ValueError(f'{label} position-set role mismatch.')
  if value.get('matched_candidate') != (candidate if control else None):
    raise ValueError(f'{label} matched-control linkage mismatch.')
  tokens, slots = value.get('tokens'), value.get('slots')
  if (
      not isinstance(tokens, list)
      or not tokens
      or not isinstance(slots, list)
      or len(tokens) != len(slots)
      or any(not isinstance(slot, int) or slot < 0 or slot >= 24 for slot in slots)
      or len(set(slots)) != len(slots)
  ):
    raise ValueError(f'{label} position-set slots/tokens are invalid.')


def _identity(
    record: Mapping[str, Any], expected_case: Mapping[str, Any], *,
    freeze_sha: str, executable: str, expected_sequence: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
  status = record.get('status')
  if status not in {'complete', 'invalid'}:
    raise ValueError(f'{label} identity status is not complete/invalid.')
  if record.get('script_version') != SOURCE_SCRIPT_VERSION:
    raise ValueError(f'{label} identity script version mismatch.')
  _validate_case(record.get('case'), expected_case, label)
  _require_linkage(record, freeze_sha=freeze_sha, executable=executable, label=label)
  _validate_canonical_target(record, expected_case, label)
  sequence_sha = record.get('sequence_sha256')
  if not isinstance(sequence_sha, Mapping) or set(sequence_sha) != {
      'reference', 'alternate'
  }:
    raise ValueError(f'{label} REF/ALT sequence hash binding is malformed.')
  for allele in ('reference', 'alternate'):
    digest = sequence_sha[allele]
    if not isinstance(digest, str) or len(digest) != 64 or any(
        char not in '0123456789abcdef' for char in digest
    ):
      raise ValueError(f'{label} {allele} sequence SHA-256 is invalid.')
  if sequence_sha['reference'] == sequence_sha['alternate']:
    raise ValueError(f'{label} REF/ALT sequence hashes unexpectedly match.')
  if sequence_sha != {
      'reference': expected_sequence['reference'],
      'alternate': expected_sequence['alternate'],
  }:
    raise ValueError(f'{label} sequence hashes differ from frozen reference binding.')
  if record.get('interval') != {
      'chromosome': expected_sequence['chromosome'],
      'start_0based': expected_sequence['start_0based'],
      'end_0based_exclusive': expected_sequence['end_0based_exclusive'],
  }:
    raise ValueError(f'{label} interval differs from frozen reference binding.')
  readout, repeat_exact = _validate_readout_repeat(record, label)
  reasons = []
  if not repeat_exact:
    reasons.append('target_readout_repeat_not_exact')
  for row in (4, 5):
    if _row_bits(readout, row) != _row_bits(readout, 0):
      reasons.append(f'reference_natural_duplicate_row_{row}_not_exact')
  for row in (2, 3):
    if _row_bits(readout, row) != _row_bits(readout, 1):
      reasons.append(f'alternate_natural_duplicate_row_{row}_not_exact')
  checks = record.get('checks')
  if isinstance(checks, Mapping):
    for field in (
        'passed', 'target_repeat_exact', 'target_duplicates_exact',
        'trace_repeat_exact', 'natural_duplicates_exact',
        'all_false_natural_effective_exact',
        'target_total_equals_two_times_mean',
    ):
      if checks.get(field) is not True:
        reasons.append(f'check_{field}_not_true')
  else:
    reasons.append('checks_missing')
  if not _fingerprints_equal(
      record.get('trace_fingerprint'),
      record.get('repeat_trace_fingerprint'),
  ):
    reasons.append('trace_fingerprint_repeat_mismatch')
  means_by_role = dict(zip(TRACE_ROLES, readout['means'], strict=True))
  emitted_means = checks.get('target_means') if isinstance(checks, Mapping) else None
  if not isinstance(emitted_means, Mapping) or set(emitted_means) != set(TRACE_ROLES):
    reasons.append('emitted_target_means_schema_invalid')
  else:
    for role, expected in means_by_role.items():
      try:
        _same_f32(
            emitted_means[role], expected,
            f'{label}.checks.target_means.{role}',
        )
      except ValueError:
        reasons.append(f'emitted_target_mean_mismatch_{role}')
  case = dict(expected_case)
  effect = _case_is_effect(case)
  delta = _f32(readout['means'][1] - readout['means'][0])
  experimental = _finite(case['delta_logit'], f'{label}.delta_logit')
  same_sign = delta != 0 and experimental != 0 and (
      (delta > 0) == (experimental > 0)
  )
  eligible = effect and abs(delta) >= EFFECT_THRESHOLD and same_sign
  direction = record.get('direction_gate')
  if isinstance(direction, Mapping):
    try:
      _same_f32(
          direction.get('predicted_alt_minus_ref_logit_margin'), delta,
          f'{label}.direction_gate.predicted_delta',
      )
      _same_float(
          direction.get('experimental_delta_logit'), experimental,
          f'{label}.direction_gate.experimental_delta',
      )
      _same_float(
          direction.get('minimum_absolute_predicted_effect'), EFFECT_THRESHOLD,
          f'{label}.direction_gate.threshold',
      )
    except ValueError:
      reasons.append('direction_gate_numeric_mismatch')
    if direction.get('direction_matches_delta_logit') is not (
        same_sign if effect else None
    ):
      reasons.append('direction_match_flag_inconsistent')
    if direction.get('eligible_for_causal_census') is not eligible:
      reasons.append('eligibility_flag_inconsistent')
  else:
    reasons.append('direction_gate_missing')
  resolved_sets = record.get('resolved_position_sets')
  if not isinstance(resolved_sets, list) or len(resolved_sets) != len(POSITION_SETS):
    raise ValueError(f'{label} frozen position sets are incomplete.')
  by_name = {}
  for expected_name, value in zip(POSITION_SETS, resolved_sets, strict=True):
    _validate_position_set(value, expected_name, label)
    by_name[expected_name] = value
  if status == 'complete' and reasons:
    raise ValueError(f'{label} is marked complete but identity checks fail: {reasons}.')
  if status == 'invalid' and not isinstance(record.get('failure'), Mapping):
    raise ValueError(f'{label} invalid identity has no failure description.')
  return {
      'case': case,
      'readout': readout,
      'means': means_by_role,
      'effect': effect,
      'eligible': eligible,
      'predicted_delta': delta,
      'position_sets': by_name,
      'valid': status == 'complete' and not reasons,
      'invalid_reasons': reasons,
  }


def _identity_binding(
    record: Mapping[str, Any], identity_path: Path, run_dir: Path, label: str
) -> None:
  binding = record.get('identity_binding')
  if not isinstance(binding, Mapping):
    raise ValueError(f'{label} identity binding is missing.')
  expected_path = str(identity_path.relative_to(run_dir))
  if binding.get('path') != expected_path or binding.get('sha256') != _sha256(identity_path):
    raise ValueError(f'{label} identity path/hash binding mismatch.')


def _active_common(
    record: Mapping[str, Any], identity: Mapping[str, Any], *,
    identity_path: Path, run_dir: Path, freeze_sha: str, executable: str,
    expected_case: Mapping[str, Any], label: str,
) -> tuple[dict[str, Any], list[str]]:
  _validate_case(record.get('case'), expected_case, label)
  _require_linkage(record, freeze_sha=freeze_sha, executable=executable, label=label)
  _identity_binding(record, identity_path, run_dir, label)
  if record.get('same_compiled_executable') is not True:
    raise ValueError(f'{label} does not attest the one compiled executable.')
  readout, repeat_exact = _validate_readout_repeat(record, label)
  reasons = []
  if not repeat_exact:
    reasons.append('target_repeat_not_exact')
  for row in (0, 1):
    if _row_bits(readout, row) != _row_bits(identity['readout'], row):
      reasons.append(f'baseline_row_{row}_differs_from_identity')
  if _row_bits(readout, 3) != _row_bits(identity['readout'], 1):
    reasons.append('alternate_self_target_drift')
  if _row_bits(readout, 5) != _row_bits(identity['readout'], 0):
    reasons.append('reference_self_target_drift')
  checks = record.get('checks')
  if isinstance(checks, Mapping):
    emitted = checks.get('target_means')
    if not isinstance(emitted, Mapping) or set(emitted) != set(TRACE_ROLES):
      reasons.append('emitted_target_means_schema_invalid')
    else:
      for role, expected in zip(TRACE_ROLES, readout['means'], strict=True):
        try:
          _same_f32(emitted[role], expected, f'{label}.checks.target_means.{role}')
        except ValueError:
          reasons.append(f'emitted_target_mean_mismatch_{role}')
    if not _fingerprints_equal(
        record.get('trace_fingerprint'),
        record.get('repeat_trace_fingerprint'),
    ):
      reasons.append('trace_fingerprint_repeat_mismatch')
  else:
    reasons.append('checks_missing')
  status = record.get('status')
  if status not in {'complete', 'invalid'}:
    raise ValueError(f'{label} status must be complete or invalid.')
  if status == 'complete' and reasons:
    raise ValueError(f'{label} is marked complete but invalid: {reasons}.')
  if status == 'invalid' and not isinstance(record.get('failure'), Mapping):
    raise ValueError(f'{label} invalid record has no failure description.')
  return readout, reasons


def _recovery(
    readout: Mapping[str, Any], label: str, *, allow_zero: bool = False
) -> dict[str, float | None]:
  ref, alt, ref_alt, alt_alt, alt_ref, ref_ref = readout['means']
  if ref == alt:
    if not allow_zero:
      raise ValueError(f'{label} donor-minus-recipient denominator is zero.')
    return {
        'reference_into_alternate': None,
        'alternate_into_reference': None,
        'B': None,
        'raw_reference_into_alternate': ref_alt - alt_alt,
        'raw_alternate_into_reference': alt_ref - ref_ref,
    }
  forward = (ref_alt - alt_alt) / (ref - alt)
  reciprocal = (alt_ref - ref_ref) / (alt - ref)
  if not math.isfinite(forward) or not math.isfinite(reciprocal):
    raise ValueError(f'{label} recovery is non-finite.')
  return {
      'reference_into_alternate': forward,
      'alternate_into_reference': reciprocal,
      'B': min(forward, reciprocal),
      'raw_reference_into_alternate': ref_alt - alt_alt,
      'raw_alternate_into_reference': alt_ref - ref_ref,
  }


def _validate_phase_group(
    record: Mapping[str, Any], identity: Mapping[str, Any], *,
    identity_path: Path, run_dir: Path, freeze_sha: str, executable: str,
    expected_case: Mapping[str, Any], label: str,
) -> tuple[tuple[str, int, str], dict[str, Any]]:
  if record.get('family') != 'phase_r':
    raise ValueError(f'{label} has the wrong intervention family.')
  group = record.get('group')
  if not isinstance(group, Mapping):
    raise ValueError(f'{label} Phase-R group metadata is missing.')
  stage, layer = group.get('stage'), group.get('layer')
  position = group.get('position_set')
  if stage not in STAGES or layer not in LAYERS or not isinstance(position, Mapping):
    raise ValueError(f'{label} Phase-R group is outside the frozen grid.')
  name = position.get('name')
  if name not in POSITION_SETS:
    raise ValueError(f'{label} Phase-R position set is outside the frozen grid.')
  expected_order = (
      STAGES.index(stage) * len(LAYERS) * len(POSITION_SETS)
      + int(layer) * len(POSITION_SETS)
      + POSITION_SETS.index(name)
  )
  if group.get('order') != expected_order:
    raise ValueError(f'{label} Phase-R group order changed.')
  if group.get('is_candidate') is not (name in CANDIDATES):
    raise ValueError(f'{label} candidate/control flag is inconsistent.')
  _validate_position_set(position, name, label)
  if position != identity['position_sets'][name]:
    raise ValueError(f'{label} position set differs from linked identity.')
  readout, reasons = _active_common(
      record, identity, identity_path=identity_path, run_dir=run_dir,
      freeze_sha=freeze_sha, executable=executable,
      expected_case=expected_case, label=label,
  )
  checks = record.get('checks')
  if isinstance(checks, Mapping):
    for field in (
        'passed', 'baseline_targets_exact_from_identity',
        'self_targets_exact', 'selected_donor_vectors_exact',
        'active_seam_natural_same_allele_exact',
        'baseline_rows_active_seam_natural_effective_exact',
        'disabled_seams_exact', 'target_repeat_exact', 'trace_repeat_exact',
    ):
      if checks.get(field) is not True:
        reasons.append(f'check_{field}_not_true')
  valid = record.get('status') == 'complete' and not reasons
  if record.get('status') == 'complete' and not valid:
    raise ValueError(f'{label} is marked complete but Phase-R checks fail.')
  metrics = _recovery(readout, label)
  if isinstance(checks, Mapping):
    emitted_raw = checks.get('raw_movement')
    emitted_recovery = checks.get('recovery')
    if not isinstance(emitted_raw, Mapping) or not isinstance(
        emitted_recovery, Mapping
    ):
      raise ValueError(f'{label} emitted Phase-R estimands are missing.')
    _same_float(
        emitted_raw.get('reference_into_alternate'),
        metrics['raw_reference_into_alternate'],
        f'{label}.raw_movement.reference_into_alternate',
    )
    _same_float(
        emitted_raw.get('alternate_into_reference'),
        metrics['raw_alternate_into_reference'],
        f'{label}.raw_movement.alternate_into_reference',
    )
    _same_float(
        emitted_recovery.get('reference_into_alternate'),
        metrics['reference_into_alternate'],
        f'{label}.recovery.reference_into_alternate',
    )
    _same_float(
        emitted_recovery.get('alternate_into_reference'),
        metrics['alternate_into_reference'],
        f'{label}.recovery.alternate_into_reference',
    )
    _same_float(
        emitted_recovery.get('bidirectional_bottleneck'),
        metrics['B'],
        f'{label}.recovery.bidirectional_bottleneck',
    )
  return (stage, int(layer), name), {
      **metrics,
      'valid': valid,
      'invalid_reasons': reasons,
  }


def _expected_component(key: str) -> dict[str, Any]:
  flags = {
      STAGE_COMPONENTS[0]: (0, 'final_embedding_A_D_closure', False, False, True, True),
      STAGE_COMPONENTS[1]: (1, 'joint_T_plus_E_closure', True, True, False, True),
      STAGE_COMPONENTS[2]: (2, 'whole_T', True, False, False, False),
      STAGE_COMPONENTS[3]: (3, 'whole_E', False, True, False, False),
  }[key]
  return dict(zip(
      ('order', 'name', 'transformer_output', 'encoder_skips',
       'final_embedding', 'closure_required'),
      flags,
      strict=True,
  ))


def _validate_stage_group(
    record: Mapping[str, Any], component_key: str, identity: Mapping[str, Any], *,
    identity_path: Path, run_dir: Path, freeze_sha: str, executable: str,
    expected_case: Mapping[str, Any], label: str,
) -> dict[str, Any]:
  if record.get('family') != 'stage_a':
    raise ValueError(f'{label} has the wrong intervention family.')
  if record.get('component') != _expected_component(component_key):
    raise ValueError(f'{label} Stage-A component contract changed.')
  readout, reasons = _active_common(
      record, identity, identity_path=identity_path, run_dir=run_dir,
      freeze_sha=freeze_sha, executable=executable,
      expected_case=expected_case, label=label,
  )
  checks = record.get('checks')
  if isinstance(checks, Mapping):
    for field in (
        'passed', 'baseline_targets_exact_from_identity', 'self_targets_exact',
        'target_repeat_exact', 'trace_repeat_exact',
        'transformer_residual_seams_disabled_exact',
        'baseline_rows_T_natural_effective_exact',
        'baseline_rows_E_natural_effective_exact',
        'baseline_rows_final_A_D_natural_effective_exact',
    ):
      if checks.get(field) is not True:
        reasons.append(f'check_{field}_not_true')
    expected = _expected_component(component_key)
    if expected['transformer_output']:
      for field in (
          'transformer_natural_self_tensors_exact',
          'transformer_effective_donor_tensors_exact',
      ):
        if checks.get(field) is not True:
          reasons.append(f'check_{field}_not_true')
    elif checks.get('transformer_disabled_natural_effective_exact') is not True:
      reasons.append('check_transformer_disabled_natural_effective_exact_not_true')
    if expected['encoder_skips']:
      for field in (
          'all_seven_skip_natural_self_tensors_exact',
          'all_seven_skip_effective_donor_tensors_exact',
      ):
        if checks.get(field) is not True:
          reasons.append(f'check_{field}_not_true')
    elif checks.get('all_seven_skips_disabled_natural_effective_exact') is not True:
      reasons.append('check_all_seven_skips_disabled_natural_effective_exact_not_true')
    if expected['final_embedding'] and checks.get(
        'final_embedding_donor_vectors_exact'
    ) is not True:
      reasons.append('check_final_embedding_donor_vectors_exact_not_true')
  closure_exact = None
  if component_key in STAGE_COMPONENTS[:2]:
    closure_exact = (
        _row_bits(readout, 2) == _row_bits(readout, 0)
        and _row_bits(readout, 4) == _row_bits(readout, 1)
    )
    if not closure_exact:
      reasons.append('endpoint_level_closure_failed')
    if isinstance(checks, Mapping):
      closure = checks.get('closure')
      if not isinstance(closure, Mapping) or closure.get('passed') is not True:
        reasons.append('emitted_closure_check_failed')
  valid = record.get('status') == 'complete' and not reasons
  if record.get('status') == 'complete' and not valid:
    raise ValueError(f'{label} is marked complete but Stage-A checks fail: {reasons}.')
  metrics = _recovery(readout, label, allow_zero=True)
  if isinstance(checks, Mapping):
    emitted_raw = checks.get('raw_movement')
    emitted_recovery = checks.get('recovery')
    if not isinstance(emitted_raw, Mapping) or not isinstance(
        emitted_recovery, Mapping
    ):
      raise ValueError(f'{label} emitted Stage-A estimands are missing.')
    _same_float(
        emitted_raw.get('reference_into_alternate'),
        metrics['raw_reference_into_alternate'],
        f'{label}.raw_movement.reference_into_alternate',
    )
    _same_float(
        emitted_raw.get('alternate_into_reference'),
        metrics['raw_alternate_into_reference'],
        f'{label}.raw_movement.alternate_into_reference',
    )
    for field, key in (
        ('reference_into_alternate', 'reference_into_alternate'),
        ('alternate_into_reference', 'alternate_into_reference'),
        ('bidirectional_bottleneck', 'B'),
    ):
      if metrics[key] is None:
        if emitted_recovery.get(field) is not None:
          raise ValueError(f'{label}.recovery.{field} must be null.')
      else:
        _same_float(
            emitted_recovery.get(field), metrics[key],
            f'{label}.recovery.{field}',
        )
  return {
      **metrics,
      'readout': readout,
      'valid': valid,
      'closure_exact': closure_exact,
      'invalid_reasons': reasons,
  }


def _median(values: Iterable[float]) -> float:
  values = list(values)
  if not values or any(not math.isfinite(value) for value in values):
    raise ValueError('A required median is empty or non-finite.')
  return float(statistics.median(values))


def _phase_rank(
    identities: Mapping[str, Mapping[str, Any]],
    groups: Mapping[str, Mapping[tuple[str, int, str], Mapping[str, Any]]],
    *, family_tooling_failure: bool,
) -> list[dict[str, Any]]:
  rows = []
  for stage_index, stage in enumerate(STAGES):
    for layer in LAYERS:
      for candidate_index, candidate in enumerate(CANDIDATES):
        per_gene_values = {gene: [] for gene in DEVELOPMENT_GENES}
        invalid_variants = []
        for variant, identity in identities.items():
          if not identity['eligible']:
            continue
          three = (
              groups[variant][(stage, layer, candidate)],
              groups[variant][(stage, layer, f'{candidate}_control_upstream')],
              groups[variant][(stage, layer, f'{candidate}_control_downstream')],
          )
          if not all(item['valid'] for item in three):
            invalid_variants.append(variant)
            continue
          candidate_b, upstream_b, downstream_b = (
              item['B'] for item in three
          )
          per_gene_values[identity['case']['gene']].append({
              'variant_id': variant,
              'B': candidate_b,
              'q': candidate_b - max(upstream_b, downstream_b),
          })
        selectable = not invalid_variants and not family_tooling_failure
        per_exon = None
        q_statistic = None
        passes = False
        if selectable:
          per_exon = {
              gene: {
                  'median_B': _median(item['B'] for item in values),
                  'median_q': _median(item['q'] for item in values),
                  'eligible_effects': len(values),
              }
              for gene, values in per_gene_values.items()
          }
          q_statistic = min(item['median_q'] for item in per_exon.values())
          passes = all(
              item['eligible_effects'] >= 3
              and item['median_B'] >= 0.25
              and item['median_q'] > 0
              for item in per_exon.values()
          )
        rows.append({
            'stage': stage,
            'layer': layer,
            'position_set': candidate,
            'Q': q_statistic,
            'per_exon': per_exon,
            'selectable': selectable,
            'invalid_or_missing_variants': invalid_variants,
            'passes_development_selection_gate': passes,
            '_tie': (stage_index, layer, candidate_index),
        })
  valid = [row for row in rows if row['selectable']]
  invalid = [row for row in rows if not row['selectable']]
  valid.sort(key=lambda row: (-row['Q'], row['_tie']))
  invalid.sort(key=lambda row: row['_tie'])
  rows = valid + invalid
  for rank, row in enumerate(rows, start=1):
    row['rank'] = rank if row['selectable'] else None
    del row['_tie']
  return rows


def _shapley(
    identity: Mapping[str, Any], t: Mapping[str, Any], e: Mapping[str, Any],
    joint: Mapping[str, Any]
) -> dict[str, Any]:
  specifications = {
      'reference_into_alternate': {
          'empty': identity['means']['alternate_baseline'],
          'donor': identity['means']['reference_baseline'],
          'T': t['readout']['means'][2],
          'E': e['readout']['means'][2],
          'T_plus_E': joint['readout']['means'][2],
      },
      'alternate_into_reference': {
          'empty': identity['means']['reference_baseline'],
          'donor': identity['means']['alternate_baseline'],
          'T': t['readout']['means'][4],
          'E': e['readout']['means'][4],
          'T_plus_E': joint['readout']['means'][4],
      },
  }
  output = {}
  for direction, values in specifications.items():
    phi_t = 0.5 * (
        values['T'] - values['empty'] + values['T_plus_E'] - values['E']
    )
    phi_e = 0.5 * (
        values['E'] - values['empty'] + values['T_plus_E'] - values['T']
    )
    interaction = (
        values['T_plus_E'] - values['T'] - values['E'] + values['empty']
    )
    denominator = values['donor'] - values['empty']
    if denominator == 0:
      raise ValueError('Stage-A Shapley denominator is zero.')
    output[direction] = {
        **values,
        'raw_phi_T': phi_t,
        'raw_phi_E': phi_e,
        'raw_interaction': interaction,
        'raw_joint_movement': values['T_plus_E'] - values['empty'],
        'normalized_phi_T': phi_t / denominator,
        'normalized_phi_E': phi_e / denominator,
        'normalized_interaction': interaction / denominator,
        'efficiency_residual': (
            phi_t + phi_e - (values['T_plus_E'] - values['empty'])
        ),
    }
  return output


def _validate_manifest(run_dir: Path) -> tuple[dict[str, Any], list[Path]]:
  manifest_path = run_dir / 'RAW_MANIFEST.json'
  manifest = _read_json(manifest_path)
  raw_paths = sorted((run_dir / 'raw').rglob('*.json'))
  for path in (run_dir / 'raw').rglob('*'):
    _guard_path(path)
  mapping = manifest.get('artifact_sha256')
  if not isinstance(mapping, Mapping):
    raise ValueError('RAW_MANIFEST has no artifact hash mapping.')
  observed_relative = {str(path.relative_to(run_dir)) for path in raw_paths}
  if set(mapping) != observed_relative:
    raise ValueError('RAW_MANIFEST does not exactly enumerate the raw tree.')
  for relative, digest in mapping.items():
    if _sha256(run_dir / relative) != digest:
      raise ValueError(f'Raw artifact hash mismatch: {relative}.')
  if manifest.get('artifact_count') != len(raw_paths):
    raise ValueError('RAW_MANIFEST artifact count mismatch.')
  if manifest.get('artifact_tree_sha256') != _tree_digest(raw_paths, run_dir):
    raise ValueError('RAW_MANIFEST artifact tree hash mismatch.')
  return manifest, raw_paths


def _identity_behavior_rows(
    expected_cases: Sequence[Mapping[str, Any]],
    identities: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
  rows = []
  for case in expected_cases:
    identity = identities[case['variant_id']]
    endpoint_ref = identity['readout']['endpoint_margins'][0]
    endpoint_alt = identity['readout']['endpoint_margins'][1]
    rows.append({
        'order': case['order'],
        'variant_id': case['variant_id'],
        'gene': case['gene'],
        'selection_class': case['selection_class'],
        'identity_valid': identity['valid'],
        'identity_invalid_reasons': identity['invalid_reasons'],
        'is_effect': identity['effect'],
        'eligible_for_causal_census': identity['eligible'],
        'reference_endpoint_margins': endpoint_ref,
        'alternate_endpoint_margins': endpoint_alt,
        'alternate_minus_reference_endpoint_margins': [
            _f32(alt - ref)
            for alt, ref in zip(endpoint_alt, endpoint_ref, strict=True)
        ],
        'alternate_minus_reference_mean_margin': identity['predicted_delta'],
        'experimental_delta_logit': case['delta_logit'],
    })
  return rows


def analyze(
    run_dir: Path, *, bundle_root: Path = _REPO_ROOT,
    ignored_paths: Sequence[Path] = (), enforce_standard_locations: bool = True,
) -> dict[str, Any]:
  """Validates one completed development run and computes frozen estimands."""
  run_dir = run_dir.resolve()
  bundle_root = bundle_root.resolve()
  _guard_path(run_dir)
  for path in run_dir.rglob('*'):
    _guard_path(path)
  temporary = sorted(run_dir.rglob('*.tmp'))
  if temporary:
    raise ValueError(f'Incomplete temporary artifacts remain: {temporary}.')
  if _sha256(_PROTOCOL_PATH) != PROTOCOL_SHA256:
    raise ValueError('Committed v3.2 protocol hash mismatch.')
  start = _read_json(run_dir / 'ATTEMPT_STARTED.json')
  freeze, freeze_sha = _validate_freeze(start, bundle_root=bundle_root)
  expected_cases = _load_frozen_cases()
  immutable_inputs = _validate_checkpoint_and_reference_inputs(
      start, freeze, expected_cases,
      enforce_standard_locations=enforce_standard_locations,
  )
  if Path(freeze['output_dir']).resolve() != run_dir:
    raise ValueError('Frozen output directory differs from analyzed run.')
  analysis_dir = Path(freeze['analysis_dir']).resolve()
  if analysis_dir == run_dir or run_dir in analysis_dir.parents:
    raise ValueError('Frozen analysis output must be separate from the raw run.')
  if enforce_standard_locations and (
      run_dir != _OUTPUT_DIR.resolve()
      or analysis_dir != _ANALYSIS_DIR.resolve()
      or Path(freeze['preflight_dir']).resolve() != _PREFLIGHT_DIR.resolve()
  ):
    raise ValueError('Frozen v3.2 output/preflight locations changed.')
  bootstrap_audit = _validate_bootstrap_attestation(
      start, freeze, freeze_sha, bundle_root=bundle_root
  )
  preflight_audit = _validate_preflight(start, freeze, freeze_sha)
  manifest, raw_paths = _validate_manifest(run_dir)
  complete = _read_json(run_dir / 'RUN_COMPLETE.json')
  if complete.get('status') not in {'complete', 'controlled_stop'}:
    raise ValueError('RUN_COMPLETE status is not complete/controlled_stop.')
  stop_reason = complete.get('stop_reason')
  if complete.get('status') == 'complete' and stop_reason is not None:
    raise ValueError('Completed RUN_COMPLETE unexpectedly has a stop reason.')
  if complete.get('status') == 'controlled_stop' and stop_reason not in {
      'identity_tooling_failure', 'target_predictive_failure',
      'closure_tooling_failure',
  }:
    raise ValueError('RUN_COMPLETE controlled-stop reason is invalid.')
  if (
      complete.get('attempt_id') != ATTEMPT_ID
      or complete.get('script_version') != SOURCE_SCRIPT_VERSION
      or complete.get('protocol_sha256') != PROTOCOL_SHA256
  ):
    raise ValueError('RUN_COMPLETE attempt/script/protocol binding mismatch.')
  if start.get('confirmation_scope_disclosure') != _DISCLOSURE or complete.get(
      'confirmation_scope_disclosure'
  ) != _DISCLOSURE:
    raise ValueError('Confirmation-scope disclosure is missing or changed.')
  if complete.get('confirmation_model_calls') != 0:
    raise ValueError('RUN_COMPLETE reports a confirmation model call.')
  if complete.get('raw_manifest') != manifest:
    raise ValueError('RUN_COMPLETE raw-manifest binding mismatch.')
  compiler = complete.get('single_executable')
  compiler_path = run_dir / 'compiler' / 'COMPILER_PROVENANCE.json'
  compiler_file = _read_json(compiler_path)
  if compiler_file != compiler:
    raise ValueError('Compiler provenance differs from RUN_COMPLETE.')
  executable = _validate_compiler_provenance(
      compiler, compiler_path, run_dir
  )
  if start.get('bundle', {}).get('tracked_clean') is not True:
    raise ValueError('Attempt bundle was not committed and clean.')
  import_path = run_dir / 'IMPORT_PROVENANCE.json'
  import_audit = _validate_import_phases(
      run_dir, complete, bundle_root=bundle_root
  )
  protobuf_path = run_dir / 'PROTOBUF_PROVENANCE.json'
  protobuf_audit = _validate_protobuf_provenance(
      protobuf_path, complete.get('protobuf_provenance_sha256'), freeze
  )

  expected_by_id = {case['variant_id']: case for case in expected_cases}
  identity_paths = sorted((run_dir / 'raw' / 'identity').glob('*.json'))
  if len(identity_paths) != EXPECTED_IDENTITIES:
    raise ValueError(f'Expected 20 identities, observed {len(identity_paths)}.')
  identities = {}
  identity_path_by_id = {}
  for expected_case in expected_cases:
    filename = f"{expected_case['order']:03d}_{_slug(expected_case['variant_id'])}.json"
    path = run_dir / 'raw' / 'identity' / filename
    if path not in identity_paths:
      raise ValueError(f'Missing exact identity artifact {filename}.')
    row = _identity(
        _read_json(path), expected_case, freeze_sha=freeze_sha,
        executable=executable,
        expected_sequence=immutable_inputs['sequence_bindings'][
            expected_case['variant_id']
        ],
        label=str(path),
    )
    identities[expected_case['variant_id']] = row
    identity_path_by_id[expected_case['variant_id']] = path
  counts = Counter((row['case']['gene'], row['effect']) for row in identities.values())
  for gene in DEVELOPMENT_GENES:
    if counts[(gene, True)] != EXPECTED_EFFECTS_PER_GENE:
      raise ValueError(f'{gene} does not have six frozen effects.')
    if counts[(gene, False)] != EXPECTED_NEUTRALS_PER_GENE:
      raise ValueError(f'{gene} does not have four frozen neutrals.')
  if sum(row['effect'] for row in identities.values()) != EXPECTED_EFFECTS:
    raise ValueError('Frozen effect count changed.')
  if sum(not row['effect'] for row in identities.values()) != EXPECTED_NEUTRALS:
    raise ValueError('Frozen neutral count changed.')
  invalid_identities = [
      variant for variant, row in identities.items() if not row['valid']
  ]
  identity_behavior = _identity_behavior_rows(expected_cases, identities)

  def controlled_result(reason: str, decision: str) -> dict[str, Any]:
    if complete.get('identity_count') != EXPECTED_IDENTITIES:
      raise ValueError('Controlled stop identity count mismatch.')
    if (
        complete.get('phase_r_group_count') != 0
        or complete.get('phase_r_invalid_count') != 0
        or complete.get('stage_a_group_count') != 0
        or complete.get('stage_a_invalid_count') != 0
        or complete.get('closures_passed') is not None
    ):
      raise ValueError('Early controlled stop contains active intervention work.')
    if len(raw_paths) != EXPECTED_IDENTITIES:
      raise ValueError('Early controlled stop raw tree is not identity-only.')
    if any((run_dir / 'raw' / name).exists() for name in ('phase_r', 'stage_a')):
      raise ValueError('Early controlled stop has forbidden active-family paths.')
    allowed = {
        (run_dir / 'ATTEMPT_STARTED.json').resolve(),
        (run_dir / 'RAW_MANIFEST.json').resolve(),
        (run_dir / 'RUN_COMPLETE.json').resolve(),
        compiler_path.resolve(), import_path.resolve(), protobuf_path.resolve(),
        (run_dir / 'IMPORT_PROVENANCE_PRE_MODEL.json').resolve(),
        (run_dir / 'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json').resolve(),
        *(path.resolve() for path in raw_paths),
        *(path.resolve() for path in ignored_paths),
    }
    if reason == 'target_predictive_failure':
      allowed.add((run_dir / 'TARGET_ELIGIBILITY.json').resolve())
    unexpected = {
        path.resolve() for path in run_dir.rglob('*.json')
        if path.resolve() not in allowed
    }
    if unexpected:
      raise ValueError(f'Unexpected JSON in controlled-stop tree: {unexpected}.')
    return {
        'analysis_version': ANALYSIS_VERSION,
        'scope': (
            'development_only_confirmation_model_outputs_activations_'
            'interventions_unopened_metadata_labels_exposed_post_freeze'
        ),
        'protocol_sha256': PROTOCOL_SHA256,
        'run_dir': str(run_dir),
        'analysis_dir': str(analysis_dir),
        'decision': decision,
        'controlled_stop_reason': reason,
        'hash_tree': {
            'freeze_sha256': freeze_sha,
            'raw_artifact_count': len(raw_paths),
            'raw_artifact_tree_sha256': manifest['artifact_tree_sha256'],
            'raw_manifest_sha256': _sha256(run_dir / 'RAW_MANIFEST.json'),
            'run_complete_sha256': _sha256(run_dir / 'RUN_COMPLETE.json'),
            'executable_fingerprint': executable,
            'import_provenance_sha256': import_audit['phase_sha256']['postcompile'],
            'import_provenance_phases': import_audit['phase_sha256'],
            'protobuf_provenance_sha256': protobuf_audit['sha256'],
            'checkpoint_manifest_sha256': (
                immutable_inputs['checkpoint_manifest_sha256']
            ),
            'checkpoint_tree_sha256': immutable_inputs['checkpoint_tree_sha256'],
            'reference_bindings_sha256': (
                immutable_inputs['reference_bindings_sha256']
            ),
        },
        'audit': {
            'one_compiled_executable': True,
            'raw_endpoint_reducer_recomputed': True,
            'identity_count': EXPECTED_IDENTITIES,
            'invalid_identity_variants': invalid_identities,
            'active_intervention_count': 0,
            **preflight_audit,
            **bootstrap_audit,
            'import_provenance_module_counts': import_audit['module_counts'],
            'import_provenance_lazy_additions': import_audit['lazy_additions'],
            'import_provenance_stable_shared_bytes': (
                import_audit['stable_shared_module_bytes']
            ),
            'protobuf_binding_verified': protobuf_audit['binding_verified'],
            'checkpoint_file_count': immutable_inputs['checkpoint_file_count'],
            'checkpoint_tree_verified': True,
            'reference_object_metadata_verified': (
                immutable_inputs['reference_object_metadata_verified']
            ),
            'identity_sequence_bindings_verified': True,
            'confirmation_model_outputs_opened': False,
            'confirmation_activations_opened': False,
            'confirmation_interventions_run': False,
            'confirmation_metadata_labels_exposed_post_freeze': True,
        },
        'target_behavior': {
            'predictive_denominator_effect_count': EXPECTED_EFFECTS,
            'all_twenty_identities': identity_behavior,
            'neutral_behavior_controls': [
                row for row in identity_behavior if not row['is_effect']
            ],
            'neutral_null_assumption_made': False,
        },
        'phase_r': None,
        'stage_a': None,
        'claim_boundary': (
            'Controlled development-only gate stop; no mechanism result exists. '
            'Later metadata/labels were exposed post-freeze, while confirmation '
            'model outputs, activations, and interventions remained unopened.'
        ),
    }

  if stop_reason == 'identity_tooling_failure':
    if not invalid_identities or (run_dir / 'TARGET_ELIGIBILITY.json').exists():
      raise ValueError('Identity controlled stop is inconsistent with raw identities.')
    if complete.get('eligible_effect_count') != 0:
      raise ValueError('Identity controlled stop has a nonzero eligible count.')
    return controlled_result(
        stop_reason, 'identity_tooling_failure_no_mechanism_result'
    )
  if invalid_identities:
    raise ValueError('Invalid identity exists without the exact identity stop.')
  eligible = {key for key, value in identities.items() if value['eligible']}
  eligible_by_gene = {
      gene: sum(
          row['eligible'] and row['case']['gene'] == gene
          for row in identities.values()
      )
      for gene in DEVELOPMENT_GENES
  }
  eligibility = _read_json(run_dir / 'TARGET_ELIGIBILITY.json')
  expected_eligible_order = [
      case['variant_id'] for case in expected_cases
      if identities[case['variant_id']]['eligible']
  ]
  expected_ineligible_order = [
      case['variant_id'] for case in expected_cases
      if identities[case['variant_id']]['effect']
      and not identities[case['variant_id']]['eligible']
  ]
  expected_neutrals = [
      case['variant_id'] for case in expected_cases
      if not identities[case['variant_id']]['effect']
  ]
  if (
      eligibility.get('eligible_effects') != expected_eligible_order
      or eligibility.get('ineligible_effects') != expected_ineligible_order
      or eligibility.get('neutral_controls') != expected_neutrals
      or eligibility.get('eligible_effects_per_gene') != eligible_by_gene
  ):
    raise ValueError('TARGET_ELIGIBILITY differs from raw identity evidence.')
  target_gate_failed = any(count < 3 for count in eligible_by_gene.values())
  if stop_reason == 'target_predictive_failure':
    if not target_gate_failed or complete.get('eligible_effect_count') != len(eligible):
      raise ValueError('Target controlled stop is inconsistent with identities.')
    return controlled_result(
        stop_reason, 'target_predictive_failure_no_mechanism_result'
    )
  if target_gate_failed:
    raise ValueError(f'Target eligibility failed without controlled stop: {eligible_by_gene}.')
  if stop_reason not in {None, 'closure_tooling_failure'}:
    raise ValueError('Unexpected controlled stop after target eligibility.')

  expected_group_keys = {
      (stage, layer, name)
      for stage in STAGES for layer in LAYERS for name in POSITION_SETS
  }
  phase_groups = {variant: {} for variant in eligible}
  phase_invalid = 0
  for expected_case in expected_cases:
    variant = expected_case['variant_id']
    case_dir = run_dir / 'raw' / 'phase_r' / (
        f"{expected_case['order']:03d}_{_slug(variant)}"
    )
    paths = sorted(case_dir.glob('*.json')) if variant in eligible else []
    if variant not in eligible:
      if case_dir.exists() and any(case_dir.glob('*.json')):
        raise ValueError(f'Ineligible/neutral variant received Phase-R grid: {variant}.')
      continue
    if len(paths) != EXPECTED_GROUPS_PER_ELIGIBLE:
      raise ValueError(f'{variant} Phase-R grid is incomplete: {len(paths)}.')
    for path in paths:
      record = _read_json(path)
      key, metrics = _validate_phase_group(
          record, identities[variant], identity_path=identity_path_by_id[variant],
          run_dir=run_dir, freeze_sha=freeze_sha, executable=executable,
          expected_case=expected_case, label=str(path),
      )
      expected_filename = (
          f'{record["group"]["order"]:03d}_{key[0]}_layer{key[1]:02d}_{key[2]}.json'
      )
      if path.name != expected_filename:
        raise ValueError(f'{path} does not follow frozen group filename/order.')
      if key in phase_groups[variant]:
        raise ValueError(f'Duplicate Phase-R group {key} for {variant}.')
      phase_groups[variant][key] = metrics
      phase_invalid += not metrics['valid']
    if set(phase_groups[variant]) != expected_group_keys:
      raise ValueError(f'{variant} Phase-R grid keys are incomplete.')
  phase_total = len(eligible) * EXPECTED_GROUPS_PER_ELIGIBLE
  phase_invalid_fraction = phase_invalid / phase_total
  phase_tooling_failure = phase_invalid_fraction > INVALID_FAMILY_FRACTION
  rankings = _phase_rank(
      identities, phase_groups, family_tooling_failure=phase_tooling_failure
  )
  first_passing = next(
      (row for row in rankings if row['passes_development_selection_gate']), None
  )

  effects = [case for case in expected_cases if _case_is_effect(case)]
  stage_results: dict[str, dict[str, dict[str, Any]]] = {
      key: {} for key in STAGE_COMPONENTS
  }
  closure_invalid = 0
  closure_failure_stage = None
  executed_closure_components = []
  for component_key in STAGE_COMPONENTS[:2]:
    directory = run_dir / 'raw' / 'stage_a' / component_key
    paths = sorted(directory.glob('*.json')) if directory.exists() else []
    if closure_failure_stage is not None:
      if paths:
        raise ValueError(
            f'{component_key} ran after {closure_failure_stage} closure failed.'
        )
      continue
    if len(paths) != EXPECTED_EFFECTS:
      raise ValueError(f'{component_key} closure cohort is incomplete.')
    executed_closure_components.append(component_key)
    component_invalid = 0
    for case in effects:
      variant = case['variant_id']
      path = directory / f"{case['order']:03d}_{_slug(variant)}.json"
      if path not in paths:
        raise ValueError(f'Missing closure artifact {path}.')
      metrics = _validate_stage_group(
          _read_json(path), component_key, identities[variant],
          identity_path=identity_path_by_id[variant], run_dir=run_dir,
          freeze_sha=freeze_sha, executable=executable,
          expected_case=case, label=str(path),
      )
      stage_results[component_key][variant] = metrics
      closure_invalid += not metrics['valid']
      component_invalid += not metrics['valid']
    if component_invalid:
      closure_failure_stage = (
          'final_embedding_A_D' if component_key == STAGE_COMPONENTS[0]
          else 'joint_T_plus_E'
      )
  closures_pass = closure_failure_stage is None
  if complete.get('closures_passed') is not closures_pass:
    raise ValueError('RUN_COMPLETE closure flag differs from endpoint-level audit.')
  if (stop_reason == 'closure_tooling_failure') is not (not closures_pass):
    raise ValueError('Closure controlled-stop reason differs from raw closures.')

  isolated_invalid = 0
  isolated_total = len(eligible) * 2
  for component_key in STAGE_COMPONENTS[2:]:
    directory = run_dir / 'raw' / 'stage_a' / component_key
    paths = sorted(directory.glob('*.json')) if directory.exists() else []
    if not closures_pass:
      if paths:
        raise ValueError('Isolated T/E ran after a failed mandatory closure.')
      continue
    if len(paths) != len(eligible):
      raise ValueError(f'{component_key} eligible cohort is incomplete.')
    for case in expected_cases:
      variant = case['variant_id']
      if variant not in eligible:
        continue
      path = directory / f"{case['order']:03d}_{_slug(variant)}.json"
      if path not in paths:
        raise ValueError(f'Missing isolated route artifact {path}.')
      metrics = _validate_stage_group(
          _read_json(path), component_key, identities[variant],
          identity_path=identity_path_by_id[variant], run_dir=run_dir,
          freeze_sha=freeze_sha, executable=executable,
          expected_case=case, label=str(path),
      )
      stage_results[component_key][variant] = metrics
      isolated_invalid += not metrics['valid']
  route_complete = closures_pass and isolated_invalid == 0

  shapley = {}
  route_summary = None
  if route_complete:
    for variant in expected_eligible_order:
      shapley[variant] = _shapley(
          identities[variant],
          stage_results[STAGE_COMPONENTS[2]][variant],
          stage_results[STAGE_COMPONENTS[3]][variant],
          stage_results[STAGE_COMPONENTS[1]][variant],
      )
    route_summary = {}
    for component_key, label in zip(STAGE_COMPONENTS[2:], ('T', 'E'), strict=True):
      per_exon = {}
      for gene in DEVELOPMENT_GENES:
        values = [
            stage_results[component_key][variant]
            for variant in expected_eligible_order
            if identities[variant]['case']['gene'] == gene
        ]
        per_exon[gene] = {
            'eligible_effects': len(values),
            'median_B': _median(item['B'] for item in values),
            'median_reference_into_alternate': _median(
                item['reference_into_alternate'] for item in values
            ),
            'median_alternate_into_reference': _median(
                item['alternate_into_reference'] for item in values
            ),
        }
      route_summary[label] = per_exon
    route_summary['shapley'] = {}
    for gene in DEVELOPMENT_GENES:
      gene_variants = [
          variant for variant in expected_eligible_order
          if identities[variant]['case']['gene'] == gene
      ]
      route_summary['shapley'][gene] = {}
      for direction in (
          'reference_into_alternate', 'alternate_into_reference'
      ):
        rows = [shapley[variant][direction] for variant in gene_variants]
        max_efficiency_error = max(abs(row['efficiency_residual']) for row in rows)
        if max_efficiency_error > 1e-12:
          raise ValueError('Stage-A Shapley efficiency accounting failed.')
        route_summary['shapley'][gene][direction] = {
            'eligible_effects': len(rows),
            'median_raw_phi_T': _median(row['raw_phi_T'] for row in rows),
            'median_raw_phi_E': _median(row['raw_phi_E'] for row in rows),
            'median_raw_interaction': _median(
                row['raw_interaction'] for row in rows
            ),
            'median_raw_joint_movement': _median(
                row['raw_joint_movement'] for row in rows
            ),
            'maximum_absolute_efficiency_residual': max_efficiency_error,
        }

  expected_phase_count = phase_total
  expected_stage_count = EXPECTED_EFFECTS * len(executed_closure_components) + (
      isolated_total if closures_pass else 0
  )
  if complete.get('identity_count') != EXPECTED_IDENTITIES:
    raise ValueError('RUN_COMPLETE identity count mismatch.')
  if complete.get('eligible_effect_count') != len(eligible):
    raise ValueError('RUN_COMPLETE eligibility count mismatch.')
  if complete.get('phase_r_group_count') != expected_phase_count:
    raise ValueError('RUN_COMPLETE Phase-R count mismatch.')
  if complete.get('phase_r_invalid_count') != phase_invalid:
    raise ValueError('RUN_COMPLETE Phase-R invalid count mismatch.')
  if complete.get('stage_a_group_count') != expected_stage_count:
    raise ValueError('RUN_COMPLETE Stage-A count mismatch.')
  if complete.get('stage_a_invalid_count') != closure_invalid + isolated_invalid:
    raise ValueError('RUN_COMPLETE Stage-A invalid count mismatch.')

  expected_raw_count = EXPECTED_IDENTITIES + expected_phase_count + expected_stage_count
  if len(raw_paths) != expected_raw_count:
    raise ValueError(
        f'Raw tree has {len(raw_paths)} files; expected {expected_raw_count}.'
    )
  ignored = {path.resolve() for path in ignored_paths}
  allowed_json = {
      (run_dir / 'ATTEMPT_STARTED.json').resolve(),
      (run_dir / 'TARGET_ELIGIBILITY.json').resolve(),
      (run_dir / 'RAW_MANIFEST.json').resolve(),
      (run_dir / 'RUN_COMPLETE.json').resolve(),
      compiler_path.resolve(),
      import_path.resolve(),
      (run_dir / 'IMPORT_PROVENANCE_PRE_MODEL.json').resolve(),
      (run_dir / 'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json').resolve(),
      protobuf_path.resolve(),
      *(path.resolve() for path in raw_paths),
      *ignored,
  }
  unexpected = {
      path.resolve() for path in run_dir.rglob('*.json')
      if path.resolve() not in allowed_json
  }
  if unexpected:
    raise ValueError(f'Unexpected JSON artifacts in v3.2 run tree: {unexpected}.')

  phase_decision = (
      'phase_r_family_tooling_failure'
      if phase_tooling_failure else
      'phase_r_development_hypothesis_requires_localized_controls'
      if first_passing is not None else
      'phase_r_negative_in_superset_graph'
  )
  stage_decision = (
      f'{closure_failure_stage}_closure_tooling_failure_no_mechanism_result'
      if not closures_pass else
      'isolated_stage_a_tooling_failure_no_shapley'
      if not route_complete else
      'stage_a_routes_descriptive_upper_bounds_only'
  )
  if not closures_pass:
    decision = f'{closure_failure_stage}_closure_tooling_failure_no_mechanism_result'
  elif phase_tooling_failure:
    decision = 'phase_r_family_tooling_failure_confirmation_closed'
  elif first_passing is not None:
    decision = 'phase_r_development_hypothesis_requires_localized_controls'
  elif not route_complete:
    decision = 'phase_r_negative_isolated_stage_a_tooling_failure_no_shapley'
  else:
    decision = 'phase_r_negative_stage_a_routes_descriptive_only'
  return {
      'analysis_version': ANALYSIS_VERSION,
      'scope': (
          'development_only_confirmation_model_outputs_activations_'
          'interventions_unopened_metadata_labels_exposed_post_freeze'
      ),
      'protocol_sha256': PROTOCOL_SHA256,
      'run_dir': str(run_dir),
      'analysis_dir': str(analysis_dir),
      'decision': decision,
      'hash_tree': {
          'freeze_sha256': freeze_sha,
          'raw_artifact_count': len(raw_paths),
          'raw_artifact_tree_sha256': manifest['artifact_tree_sha256'],
          'raw_manifest_sha256': _sha256(run_dir / 'RAW_MANIFEST.json'),
          'run_complete_sha256': _sha256(run_dir / 'RUN_COMPLETE.json'),
          'executable_fingerprint': executable,
          'import_provenance_sha256': import_audit['phase_sha256']['postcompile'],
          'import_provenance_phases': import_audit['phase_sha256'],
          'protobuf_provenance_sha256': protobuf_audit['sha256'],
          'checkpoint_manifest_sha256': (
              immutable_inputs['checkpoint_manifest_sha256']
          ),
          'checkpoint_tree_sha256': immutable_inputs['checkpoint_tree_sha256'],
          'reference_bindings_sha256': (
              immutable_inputs['reference_bindings_sha256']
          ),
      },
      'audit': {
          'one_compiled_executable': True,
          'raw_endpoint_reducer_recomputed': True,
          'identity_count': EXPECTED_IDENTITIES,
          'effect_count': EXPECTED_EFFECTS,
          'neutral_count': EXPECTED_NEUTRALS,
          'eligible_effect_count': len(eligible),
          'eligible_effects_per_gene': eligible_by_gene,
          'confirmation_model_calls': 0,
          **preflight_audit,
          **bootstrap_audit,
          'import_provenance_module_counts': import_audit['module_counts'],
          'import_provenance_lazy_additions': import_audit['lazy_additions'],
          'import_provenance_stable_shared_bytes': (
              import_audit['stable_shared_module_bytes']
          ),
          'protobuf_binding_verified': protobuf_audit['binding_verified'],
          'checkpoint_file_count': immutable_inputs['checkpoint_file_count'],
          'checkpoint_tree_verified': True,
          'reference_object_metadata_verified': (
              immutable_inputs['reference_object_metadata_verified']
          ),
          'identity_sequence_bindings_verified': True,
          'confirmation_model_outputs_opened': False,
          'confirmation_activations_opened': False,
          'confirmation_interventions_run': False,
          'confirmation_metadata_labels_exposed_post_freeze': True,
      },
      'target_behavior': {
          'predictive_denominator_effect_count': EXPECTED_EFFECTS,
          'all_twenty_identities': identity_behavior,
          'ineligible_effect_variants': expected_ineligible_order,
          'neutral_behavior_controls': [
              row for row in identity_behavior if not row['is_effect']
          ],
          'neutral_null_assumption_made': False,
      },
      'phase_r': {
          'family_decision': phase_decision,
          'group_count': phase_total,
          'invalid_group_count': phase_invalid,
          'invalid_fraction': phase_invalid_fraction,
          'family_tooling_failure': phase_tooling_failure,
          'candidate_count': len(rankings),
          'first_passing_candidate': first_passing,
          'rankings': rankings,
      },
      'stage_a': {
          'family_decision': stage_decision,
          'closures_pass': closures_pass,
          'closure_failure_stage': closure_failure_stage,
          'executed_closure_components': executed_closure_components,
          'closure_invalid_count': closure_invalid,
          'isolated_route_complete': route_complete,
          'isolated_invalid_count': isolated_invalid,
          'route_summary': route_summary,
          'shapley_by_variant': shapley,
      },
      'claim_boundary': (
          'Development-only computational intervention result. It does not '
          'establish an RBP, biochemical pathway, endogenous mechanism, or '
          'experimental replication, and it cannot open confirmation.'
      ),
  }


def render_markdown(result: Mapping[str, Any]) -> str:
  phase = result['phase_r']
  stage = result['stage_a']
  if phase is None or stage is None:
    return '\n'.join([
        '# OpenSplice v3.2 development superset result',
        '',
        f"**Decision:** `{result['decision']}`",
        '',
        'The CPU-only analyzer verified this controlled gate stop and its '
        'identity-only raw hash tree. No active causal intervention was run, '
        'so no mechanism result exists.',
        '',
        'Confirmation model outputs, activations and interventions remained '
        'unopened. Later confirmation metadata/labels had been exposed '
        'post-freeze; this is not a claim of complete metadata blindness.',
        '',
        f"Raw artifact tree SHA-256: "
        f"`{result['hash_tree']['raw_artifact_tree_sha256']}`",
        '',
    ])
  first = phase['first_passing_candidate']
  lines = [
      '# OpenSplice v3.2 development superset result',
      '',
      f"**Decision:** `{result['decision']}`",
      '',
      'The CPU-only analyzer independently reconstructed all endpoint margins '
      'and means from the raw relevant/padding logits. Confirmation model '
      'outputs, activations and interventions remained unopened. Later '
      'confirmation metadata/labels had been exposed post-freeze, so this is '
      'not a claim of complete metadata blindness.',
      '',
      '## Gates',
      '',
      f"- Eligible effects: {result['audit']['eligible_effect_count']} "
      f"({result['audit']['eligible_effects_per_gene']})",
      f"- Phase-R invalid groups: {phase['invalid_group_count']}/"
      f"{phase['group_count']}",
      f"- Mandatory endpoint-level closures: {stage['closures_pass']}",
      f"- Complete isolated T/E account: {stage['isolated_route_complete']}",
      '',
  ]
  if first is None:
    lines.append('No frozen Phase-R candidate passed the development gate.')
  else:
    lines.append(
        'First passing development hypothesis: '
        f"`{first['stage']}/L{first['layer']}/{first['position_set']}` "
        f"(Q={first['Q']:.6g})."
    )
  lines.extend([
      '',
      result['claim_boundary'],
      '',
      f"Raw artifact tree SHA-256: "
      f"`{result['hash_tree']['raw_artifact_tree_sha256']}`",
      '',
  ])
  return '\n'.join(lines)


def _write_atomic(path: Path, text: str) -> None:
  if path.exists():
    raise FileExistsError(f'Append-only analysis artifact exists: {path}.')
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


def main() -> None:
  args = _parse_args()
  _guard_path(args.output_json)
  if args.output_markdown is not None:
    _guard_path(args.output_markdown)
  if args.output_json.resolve() == args.run_dir.resolve() or (
      args.run_dir.resolve() in args.output_json.resolve().parents
  ):
    raise ValueError('Analysis output cannot be inside the append-only raw run.')
  ignored = [args.output_json]
  if args.output_markdown is not None:
    ignored.append(args.output_markdown)
  result = analyze(args.run_dir, ignored_paths=ignored)
  analysis_dir = Path(result['analysis_dir']).resolve()
  if args.output_json.resolve() != analysis_dir / 'ANALYSIS.json':
    raise ValueError('JSON output path differs from frozen analysis destination.')
  if args.output_markdown is not None and args.output_markdown.resolve() != (
      analysis_dir / 'RESULT.md'
  ):
    raise ValueError('Markdown output path differs from frozen destination.')
  if analysis_dir.exists():
    raise FileExistsError('Frozen analysis directory already exists; never overwrite.')
  _write_atomic(
      args.output_json,
      json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n',
  )
  if args.output_markdown is not None:
    _write_atomic(args.output_markdown, render_markdown(result))
  print(args.output_json.resolve())


if __name__ == '__main__':
  main()
