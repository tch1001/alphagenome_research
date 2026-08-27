#!/usr/bin/env python3
"""One-shot development-only OpenSplice v3.2 superset-graph runner."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np

from alphagenome.models import dna_model as public_dna_model
from alphagenome_research.model import dna_model
from alphagenome_research.model import interpretability


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))
import run_device_preflight_v3_1_2 as device_gate  # pylint: disable=g-import-not-at-top
import run_inference_trace as v2  # pylint: disable=g-import-not-at-top
import run_phase_r_v3 as phase_r  # pylint: disable=g-import-not-at-top
import run_route_census_v3 as route_v3  # pylint: disable=g-import-not-at-top
import run_stage_a_branches_v3 as stage_a  # pylint: disable=g-import-not-at-top


SCRIPT_VERSION = 'opensplice-superset-graph-v3.2.0'
ATTEMPT_ID = 'opensplice-v3.2-development-superset-graph-one-shot'
PROTOCOL_PATH = (
    _HERE / 'v3_wider_mechanism' / 'superset_graph_protocol_v3_2.md'
)
PROTOCOL_SHA256 = (
    '1e87839250e838d1d9aa95162e76a3acb6b1eaab59b481c60dab943b62f6caea'
)
FREEZE_PATH = _HERE / 'superset_graph_v3_2_freeze.json'
CHECKPOINT_MANIFEST_PATH = _HERE / 'checkpoint_manifest_v3_2.tsv'
CHECKPOINT_MANIFEST_SHA256 = (
    '1ed87db4c5bd7c5418c7734ec128faa4a9ecd186df2a024437484a8bc2b6e934'
)
REFERENCE_BINDINGS_PATH = _HERE / 'superset_graph_v3_2_reference_bindings.json'
REFERENCE_BINDINGS_SHA256 = (
    'da712cdca50f82113ac1d00cb2fa7171f7368f31aedf06c48ce92dbdb5897dca'
)
DEVELOPMENT_VARIANTS_PATH = (
    _HERE / 'superset_graph_v3_2_development_variants.tsv'
)
DEVELOPMENT_EXONS_PATH = _HERE / 'superset_graph_v3_2_development_exons.tsv'
DEVELOPMENT_VARIANTS_SHA256 = (
    '24a0afec1c020803152c7f55a0a78ac345763173dd79a4175e889d9192db05f9'
)
DEVELOPMENT_EXONS_SHA256 = (
    '37c49637bba5484d1e29a7f21faf42668dcf62d604e9af91939a8776e61f0231'
)
CALIBRATION_PROTO_PATH = (
    _HERE.parents[2]
    / 'src/alphagenome_research/protos/calibration_scores.proto'
)
CALIBRATION_PB2_PATH = CALIBRATION_PROTO_PATH.with_name(
    'calibration_scores_pb2.py'
)
CALIBRATION_PYI_PATH = CALIBRATION_PROTO_PATH.with_name(
    'calibration_scores_pb2.pyi'
)
DEPENDENCY_PROTO_PATH = (
    _HERE.parents[3] / 'alphagenome/src/alphagenome/protos/dna_model.proto'
)
DEPENDENCY_PB2_PATH = DEPENDENCY_PROTO_PATH.with_name('dna_model_pb2.py')
TENSOR_PROTO_PATH = DEPENDENCY_PROTO_PATH.with_name('tensor.proto')
TENSOR_PB2_PATH = DEPENDENCY_PROTO_PATH.with_name('tensor_pb2.py')
OUTPUT_DIR = _HERE / 'results' / 'v3_2_development_superset_graph_one_shot'
ANALYSIS_DIR = _HERE / 'results' / 'v3_2_development_superset_graph_analysis'
START_PATH = OUTPUT_DIR / 'ATTEMPT_STARTED.json'
EXPECTED_DEVICE_KIND = device_gate.EXPECTED_DEVICE_KIND
EXPECTED_GPU_UUID = device_gate.EXPECTED_GPU_UUID
EXPECTED_COMPUTE_CAPABILITY = device_gate.EXPECTED_COMPUTE_CAPABILITY
REFERENCE_URL = (
    'https://storage.googleapis.com/alphagenome/reference/gencode/hg38/'
    'GRCh38.p13.genome.fa'
)
REFERENCE_OBJECT = {
    'url': REFERENCE_URL,
    'bucket': 'alphagenome',
    'object': 'reference/gencode/hg38/GRCh38.p13.genome.fa',
    'generation': '1766084693379925',
    'size_bytes': 3321586957,
    'etag': 'edee5408303f6c1c6bae1c76ffd23671',
    'md5_base64': '7e5UCDA/bBxrrhx2/9I2cQ==',
    'crc32c_base64': 'AyHUtA==',
}
TRACE_ROLES = route_v3.TRACE_BATCH_ROLES
RECIPIENT_DONOR_ROWS = ((2, 0), (3, 1), (4, 1), (5, 0))
NATURAL_IDENTITY_ROWS = (0, 1, 1, 1, 0, 0)
STAGE_COMPONENT_NAMES = stage_a.COMPONENTS
ENVIRONMENT_PREFIXES = ('XLA', 'JAX', 'CUDA', 'CUDNN', 'CUBLAS', 'TRITON')
BOOTSTRAP_ATTESTATION_MODULE = '_opensplice_v3_2_bootstrap_attestation'


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', type=Path)
  parser.add_argument('--successful-preflight', type=Path)
  parser.add_argument('--dry-run', action='store_true')
  parser.add_argument('--max-variants', type=int, default=0)
  parser.add_argument('--max-groups', type=int, default=0)
  args = parser.parse_args()
  if not args.dry_run and (args.max_variants or args.max_groups):
    parser.error('Bounded flags are dry-run-only; v3.2 execution is all-or-none.')
  return args


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
  return (
      json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str)
      + '\n'
  ).encode('utf-8')


def _write_new(path: Path, value: Any) -> str:
  """Publishes one append-only JSON artifact."""
  data = _json_bytes(value)
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
  descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
  try:
    with os.fdopen(descriptor, 'wb') as handle:
      handle.write(data)
      handle.flush()
      os.fsync(handle.fileno())
    try:
      os.link(temporary, path)
    except FileExistsError as error:
      raise FileExistsError(f'Append-only artifact exists: {path}.') from error
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
      os.fsync(directory_fd)
    finally:
      os.close(directory_fd)
  finally:
    temporary.unlink(missing_ok=True)
  return hashlib.sha256(data).hexdigest()


def _write_new_text(path: Path, value: str) -> str:
  data = value.encode('utf-8')
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
  descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
  try:
    with os.fdopen(descriptor, 'wb') as handle:
      handle.write(data)
      handle.flush()
      os.fsync(handle.fileno())
    os.link(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)
  return hashlib.sha256(data).hexdigest()


def _slug(value: str) -> str:
  return ''.join(ch if ch.isalnum() else '_' for ch in value).strip('_')


def _reject_confirmation_path(path: Path) -> None:
  if any('confirm' in part.lower() for part in path.resolve().parts):
    raise ValueError(f'Confirmation-named path is forbidden: {path}.')


def consume_bootstrap_attestation() -> dict[str, Any]:
  module = sys.modules.pop(BOOTSTRAP_ATTESTATION_MODULE, None)
  record = getattr(module, 'record', None)
  if not isinstance(record, dict):
    raise RuntimeError(
        'Direct runner invocation is forbidden; use the v3.2 launcher.'
    )
  if record.get('pid') != os.getpid():
    raise RuntimeError('Bootstrap attestation came from another process.')
  if record.get('freeze', {}).get('sha256') != _sha256(FREEZE_PATH):
    raise RuntimeError('Bootstrap attestation used another freeze.')
  for field, path in (
      ('launcher', _HERE / 'launch_superset_graph_v3_2.py'),
      ('bootstrap', _HERE / 'validate_superset_graph_bootstrap_v3_2.py'),
  ):
    if record.get(f'{field}_path') != str(path.resolve()):
      raise RuntimeError(f'Bootstrap {field} path changed.')
    if record.get(f'{field}_sha256') != _sha256(path):
      raise RuntimeError(f'Bootstrap {field} hash changed.')
  return record


def assert_v3_2_environment() -> dict[str, str]:
  """Enforces the prospective no-cache/no-autotune runtime contract."""
  device_gate.assert_sanitized_environment()
  if os.environ.get('JAX_ENABLE_COMPILATION_CACHE') != 'false':
    raise ValueError('JAX_ENABLE_COMPILATION_CACHE must be literal false.')
  forbidden_exact = ('XLA_FLAGS', 'JAX_COMPILATION_CACHE_DIR')
  present = [name for name in forbidden_exact if name in os.environ]
  present.extend(
      name for name in os.environ if name.startswith('JAX_PERSISTENT_CACHE_')
  )
  present.extend(
      name for name in os.environ
      if 'AUTOTUNE' in name.upper()
      and any(term in name.upper() for term in ('LOAD', 'DUMP', 'CACHE'))
  )
  if present:
    raise ValueError(
        'Forbidden compiler/autotune environment variables are present: '
        f'{sorted(set(present))}.'
    )
  if bool(jax.config.jax_enable_compilation_cache):
    raise ValueError('JAX runtime compilation cache is enabled.')
  return {
      name: value for name, value in sorted(os.environ.items())
      if name.startswith(ENVIRONMENT_PREFIXES)
  }


def import_provenance() -> dict[str, Any]:
  """Hashes every loaded project/upstream model module with a real file."""
  repo = _HERE.parents[2].resolve()
  upstream = (repo.parent / 'alphagenome').resolve()
  modules = []
  for name, module in sorted(sys.modules.items()):
    raw_path = getattr(module, '__file__', None)
    is_local_helper = bool(
        raw_path
        and Path(raw_path).resolve().is_relative_to(_HERE.resolve())
    )
    if not (
        name == 'alphagenome_research'
        or name.startswith('alphagenome_research.')
        or name == 'alphagenome'
        or name.startswith('alphagenome.')
        or is_local_helper
    ):
      continue
    if raw_path is None:
      continue
    path = Path(raw_path).resolve()
    if path.suffix == '.pyc' and path.with_suffix('').exists():
      path = path.with_suffix('')
    if not path.is_file():
      raise ValueError(f'Loaded module has no hashable file: {name}: {path}.')
    if path.is_relative_to(repo):
      root = 'alphagenome_research_checkout'
    elif path.is_relative_to(upstream):
      root = 'upstream_alphagenome_checkout'
    else:
      raise ValueError(f'Model/helper module escaped declared roots: {name}.')
    modules.append({
        'name': name,
        'path': str(path),
        'root': root,
        'sha256': _sha256(path),
        'size_bytes': path.stat().st_size,
    })
  if not any(x['name'] == 'alphagenome_research.model.model' for x in modules):
    raise ValueError('Superset model module missing from import provenance.')
  return {'module_count': len(modules), 'modules': modules}


def assert_import_provenance_stable(
    earlier: Mapping[str, Any], later: Mapping[str, Any]
) -> None:
  old = {item['name']: item for item in earlier['modules']}
  new = {item['name']: item for item in later['modules']}
  missing = sorted(set(old) - set(new))
  changed = sorted(name for name in old if name in new and old[name] != new[name])
  if missing or changed:
    raise ValueError(
        f'Imported module provenance changed: missing={missing}, '
        f'changed={changed}.'
    )


def protobuf_provenance() -> dict[str, Any]:
  """Binds the imported generated bytes without claiming regeneration."""
  paths = (
      CALIBRATION_PROTO_PATH, CALIBRATION_PB2_PATH, CALIBRATION_PYI_PATH,
      DEPENDENCY_PROTO_PATH, DEPENDENCY_PB2_PATH,
      TENSOR_PROTO_PATH, TENSOR_PB2_PATH,
  )
  for path in paths:
    _reject_confirmation_path(path)
    if not path.is_file():
      raise ValueError(f'Required protobuf artifact is missing: {path}.')
  imported = sys.modules.get('alphagenome_research.protos.calibration_scores_pb2')
  imported_path = Path(getattr(imported, '__file__', '')).resolve()
  if imported_path != CALIBRATION_PB2_PATH.resolve():
    raise ValueError('Imported calibration protobuf does not use frozen bytes.')
  imported_dependencies = {}
  for name, expected_path in (
      ('alphagenome.protos.dna_model_pb2', DEPENDENCY_PB2_PATH),
      ('alphagenome.protos.tensor_pb2', TENSOR_PB2_PATH),
  ):
    module = sys.modules.get(name)
    path = Path(getattr(module, '__file__', '')).resolve()
    if path != expected_path.resolve():
      raise ValueError(f'Imported protobuf path changed: {name}.')
    imported_dependencies[name] = {
        'path': str(path), 'sha256': _sha256(path),
        'size_bytes': path.stat().st_size,
    }
  compiler = shutil.which('protoc')
  compiler_version = None
  if compiler is not None:
    result = subprocess.run(
        (compiler, '--version'), check=False, capture_output=True, text=True
    )
    compiler_version = {
        'path': str(Path(compiler).resolve()),
        'returncode': result.returncode,
        'stdout': result.stdout.strip(),
        'stderr': result.stderr.strip(),
    }
  header = CALIBRATION_PB2_PATH.read_text(
      encoding='utf-8'
  ).splitlines()[:8]
  runtime_version = importlib.metadata.version('protobuf')
  if runtime_version != '7.35.1':
    raise ValueError(f'Frozen protobuf runtime changed: {runtime_version}.')
  if not any('Protobuf Python Version: 7.35.1' in line for line in header):
    raise ValueError('Generated binding embedded protobuf version changed.')
  if not any(
      'source: alphagenome_research/protos/calibration_scores.proto' in line
      for line in header
  ):
    raise ValueError('Generated binding embedded source path changed.')
  return {
      'historical_generation_provenance': 'unknown_not_reconstructed',
      'regeneration_claim': False,
      'byte_level_reproducibility': (
          'generated outputs remain intentionally untracked and are frozen by '
          'exact path, size, and SHA256'
      ),
      'embedded_generated_header': header,
      'protobuf_runtime_version': runtime_version,
      'current_standalone_protoc': compiler_version,
      'current_protoc_was_used_to_generate_frozen_outputs': False,
      'source_proto': {
          'path': str(CALIBRATION_PROTO_PATH.resolve()),
          'sha256': _sha256(CALIBRATION_PROTO_PATH),
      },
      'dependency_proto': {
          'path': str(DEPENDENCY_PROTO_PATH.resolve()),
          'sha256': _sha256(DEPENDENCY_PROTO_PATH),
      },
      'dependency_pb2': {
          'path': str(DEPENDENCY_PB2_PATH.resolve()),
          'sha256': _sha256(DEPENDENCY_PB2_PATH),
          'size_bytes': DEPENDENCY_PB2_PATH.stat().st_size,
      },
      'tensor_proto': {
          'path': str(TENSOR_PROTO_PATH.resolve()),
          'sha256': _sha256(TENSOR_PROTO_PATH),
      },
      'tensor_pb2': {
          'path': str(TENSOR_PB2_PATH.resolve()),
          'sha256': _sha256(TENSOR_PB2_PATH),
          'size_bytes': TENSOR_PB2_PATH.stat().st_size,
      },
      'generated_outputs': {
          str(path.resolve()): {
              'sha256': _sha256(path), 'size_bytes': path.stat().st_size
          }
          for path in (CALIBRATION_PB2_PATH, CALIBRATION_PYI_PATH)
      },
      'imported_pb2': {
          'path': str(imported_path),
          'sha256': _sha256(imported_path),
          'size_bytes': imported_path.stat().st_size,
      },
      'imported_dependency_pb2': imported_dependencies,
  }


def load_development_cases() -> tuple[v2.Case, ...]:
  for path in (DEVELOPMENT_VARIANTS_PATH, DEVELOPMENT_EXONS_PATH):
    _reject_confirmation_path(path)
  if _sha256(DEVELOPMENT_VARIANTS_PATH) != DEVELOPMENT_VARIANTS_SHA256:
    raise ValueError('v3.2 development variant projection hash changed.')
  if _sha256(DEVELOPMENT_EXONS_PATH) != DEVELOPMENT_EXONS_SHA256:
    raise ValueError('v3.2 development exon projection hash changed.')
  cases = v2.load_cases(DEVELOPMENT_VARIANTS_PATH, DEVELOPMENT_EXONS_PATH)
  if tuple(case.order for case in cases) != tuple(range(20)):
    raise ValueError('v3.2 requires exact manifest orders 0--19.')
  if {case.gene for case in cases} != {'BRAF', 'SLC25A48'}:
    raise ValueError('v3.2 development gene allowlist changed.')
  if any(case.gene in {'ELN', 'EIF4A2', 'DMD'} for case in cases):
    raise ValueError('Confirmation gene entered v3.2 development cases.')
  return cases


@functools.cache
def load_reference_bindings() -> Mapping[str, Any]:
  _reject_confirmation_path(REFERENCE_BINDINGS_PATH)
  if _sha256(REFERENCE_BINDINGS_PATH) != REFERENCE_BINDINGS_SHA256:
    raise ValueError('v3.2 reference-sequence binding hash changed.')
  binding = json.loads(REFERENCE_BINDINGS_PATH.read_text(encoding='utf-8'))
  if binding.get('reference_url') != REFERENCE_URL:
    raise ValueError('Reference-sequence binding URL changed.')
  if binding.get('context_bp') != route_v3.CONTEXT_BP:
    raise ValueError('Reference-sequence binding context changed.')
  cases = binding.get('cases', {})
  if len(cases) != 20:
    raise ValueError('Reference-sequence binding must contain 20 dev cases.')
  return binding


def validate_reference_case(
    case: v2.Case, interval: Any, sequence_sha: Mapping[str, str]
) -> None:
  try:
    expected = load_reference_bindings()['cases'][case.variant_id]
  except KeyError as error:
    raise ValueError(f'No frozen reference binding for {case.variant_id}.') from error
  observed = [
      case.order, interval.chromosome, interval.start, interval.end,
      sequence_sha['reference'], sequence_sha['alternate'],
  ]
  if observed != expected:
    raise ValueError(f'Reference/sequence binding changed for {case.variant_id}.')
  for digest in sequence_sha.values():
    if len(digest) != 64 or any(ch not in '0123456789abcdef' for ch in digest):
      raise ValueError('Sequence binding is not a lowercase SHA-256 digest.')


def validate_checkpoint(checkpoint: Path) -> dict[str, Any]:
  _reject_confirmation_path(CHECKPOINT_MANIFEST_PATH)
  if _sha256(CHECKPOINT_MANIFEST_PATH) != CHECKPOINT_MANIFEST_SHA256:
    raise ValueError('Checkpoint manifest hash changed.')
  records = []
  for line in CHECKPOINT_MANIFEST_PATH.read_text(encoding='utf-8').splitlines():
    fields = line.split('\t')
    if len(fields) != 3:
      raise ValueError('Checkpoint manifest row must have three columns.')
    relative, size_text, expected_sha = fields
    path = checkpoint / relative
    _reject_confirmation_path(path)
    if not path.is_file():
      raise ValueError(f'Checkpoint file is missing: {relative}.')
    size = path.stat().st_size
    observed_sha = _sha256(path)
    if size != int(size_text) or observed_sha != expected_sha:
      raise ValueError(f'Checkpoint file changed: {relative}.')
    records.append({
        'relative_path': relative, 'size_bytes': size,
        'sha256': observed_sha,
    })
  if len(records) != 12 or [x['relative_path'] for x in records] != sorted(
      x['relative_path'] for x in records
  ):
    raise ValueError('Checkpoint manifest must contain 12 sorted files.')
  observed_files = sorted(
      str(path.relative_to(checkpoint))
      for path in checkpoint.rglob('*') if path.is_file()
  )
  if observed_files != [x['relative_path'] for x in records]:
    raise ValueError('Checkpoint snapshot contains files outside the manifest.')
  return {
      'snapshot_path': str(checkpoint.resolve()),
      'snapshot_name': checkpoint.name,
      'manifest_path': str(CHECKPOINT_MANIFEST_PATH.resolve()),
      'manifest_sha256': CHECKPOINT_MANIFEST_SHA256,
      'file_count': len(records),
      'files': records,
  }


def validate_reference_object() -> dict[str, Any]:
  request = urllib.request.Request(REFERENCE_URL, method='HEAD')
  with urllib.request.urlopen(request, timeout=30) as response:
    headers = {name.lower(): value for name, value in response.headers.items()}
    hash_headers = response.headers.get_all('x-goog-hash', [])
  hashes = {}
  for item in ','.join(hash_headers).split(','):
    if '=' in item:
      name, value = item.strip().split('=', 1)
      hashes[name] = value
  observed = {
      **REFERENCE_OBJECT,
      'observed_generation': headers.get('x-goog-generation'),
      'observed_size_bytes': int(headers.get('content-length', '-1')),
      'observed_etag': headers.get('etag', '').strip('"'),
      'observed_md5_base64': hashes.get('md5'),
      'observed_crc32c_base64': hashes.get('crc32c'),
  }
  for name in ('generation', 'etag'):
    if observed[f'observed_{name}'] != REFERENCE_OBJECT[name]:
      raise ValueError(f'GCS reference {name} changed.')
  if observed['observed_size_bytes'] != REFERENCE_OBJECT['size_bytes']:
    raise ValueError('GCS reference object size changed.')
  for name in ('md5_base64', 'crc32c_base64'):
    if observed[f'observed_{name}'] != REFERENCE_OBJECT[name]:
      raise ValueError(f'GCS reference {name} changed.')
  return observed


def superset_selection(
    position_sets: Sequence[v2.TracePositionSet], resolved_target: Any
) -> interpretability.SupersetGraphSelection:
  selection = interpretability.SupersetGraphSelection(
      transformer=v2.transformer_trace_selection(position_sets),
      stage_a=stage_a.branch_selection(resolved_target),
  )
  residual = selection.transformer.residual_positions
  if residual is None or residual.positions.shape != (v2.NUM_TRACE_SLOTS,):
    raise ValueError('Frozen 24-slot transformer selector shape changed.')
  if selection.stage_a.final_embedding_positions.shape != (2,):
    raise ValueError('Frozen A/D final-embedding selector shape changed.')
  return selection


def identity_interventions(
    selection: interpretability.SupersetGraphSelection,
) -> interpretability.SupersetGraphInterventions:
  return interpretability.no_superset_graph_interventions(
      selection,
      batch_size=interpretability.PAIRED_CAUSAL_BATCH_SIZE,
      num_edges=v2.PAIR_PADDING_SIZE,
  )


def phase_r_interventions(
    selection: interpretability.SupersetGraphSelection,
    group: phase_r.ResidualGridGroup,
) -> interpretability.SupersetGraphInterventions:
  identity = identity_interventions(selection)
  transformer = v2._live_batch_residual_transfer(  # pylint: disable=protected-access
      identity.transformer,
      stage=group.stage,
      layer=group.layer,
      slots=group.position_set.slots,
  )
  return dataclasses.replace(identity, transformer=transformer)


def stage_a_interventions(
    selection: interpretability.SupersetGraphSelection,
    component: stage_a.StageAComponent,
) -> interpretability.SupersetGraphInterventions:
  identity = identity_interventions(selection)
  branches = stage_a.component_interventions(selection.stage_a, component)
  return dataclasses.replace(identity, stage_a=branches)


def pytree_signature(value: Any) -> dict[str, Any]:
  leaves, treedef = jax.tree.flatten(value)
  return {
      'treedef': str(treedef),
      'leaves': tuple({
          'shape': tuple(np.shape(leaf)),
          'dtype': str(getattr(leaf, 'dtype', type(leaf).__name__)),
      } for leaf in leaves),
  }


def assert_same_program_signature(reference: Mapping[str, Any], value: Any) -> None:
  if pytree_signature(value) != dict(reference):
    raise ValueError('Runtime selector/intervention pytree signature changed.')


def _array_bytes(value: Any) -> bytes:
  array = np.asarray(value)
  return (
      str(array.dtype).encode('ascii') + b'\0'
      + repr(tuple(array.shape)).encode('ascii') + b'\0'
      + array.tobytes(order='C')
  )


def trace_fingerprint(trace: interpretability.SupersetGraphTrace) -> dict[str, Any]:
  digest = hashlib.sha256()
  fields = []
  for index, leaf in enumerate(jax.tree.leaves(trace)):
    array = np.asarray(leaf)
    digest.update(index.to_bytes(4, 'little'))
    digest.update(_array_bytes(array))
    fields.append({'shape': tuple(array.shape), 'dtype': str(array.dtype)})
  return {'sha256': digest.hexdigest(), 'leaves': tuple(fields)}


def _all_trace_arrays(trace: interpretability.SupersetGraphTrace):
  return tuple(np.asarray(leaf) for leaf in jax.tree.leaves(trace))


def _assert_trace_repeat(first: Any, second: Any) -> None:
  left = _all_trace_arrays(first)
  right = _all_trace_arrays(second)
  if len(left) != len(right):
    raise ValueError('Superset trace structure changed across repeat.')
  for index, (a, b) in enumerate(zip(left, right, strict=True)):
    if not np.array_equal(a, b):
      raise ValueError(f'Superset trace repeat failed at leaf {index}.')


_TRANSFORMER_PAIRS = (
    ('compact_pair_bias_edges', 'effective_compact_pair_bias_edges'),
    ('head_value_outputs', 'effective_head_value_outputs'),
    ('pre_attention_residuals', 'effective_pre_attention_residuals'),
    ('post_attention_residuals', 'effective_post_attention_residuals'),
    ('post_mlp_residuals', 'effective_post_mlp_residuals'),
)


def _assert_transformer_noop(trace: interpretability.TransformerTrace) -> None:
  for natural_name, effective_name in _TRANSFORMER_PAIRS:
    if not np.array_equal(
        getattr(trace, natural_name), getattr(trace, effective_name)
    ):
      raise ValueError(f'Disabled transformer seam changed: {natural_name}.')


def _assert_stage_a_noop(trace: interpretability.StageABranchTrace) -> None:
  if not np.asarray(
      trace.transformer_output_natural_matches_identity
  ).all():
    raise ValueError('Natural T same-allele duplicates differ.')
  if not np.asarray(
      trace.transformer_output_effective_matches_natural
  ).all():
    raise ValueError('Disabled whole T changed its natural tensor.')
  if not np.asarray(trace.encoder_skips_natural_match_identity).all():
    raise ValueError('Natural E same-allele duplicates differ.')
  if not np.asarray(trace.encoder_skips_effective_match_natural).all():
    raise ValueError('Disabled whole E changed its natural tensor.')
  if not np.array_equal(
      trace.natural_final_embeddings, trace.effective_final_embeddings
  ):
    raise ValueError('Disabled final A/D embedding seam changed values.')


def _target_values(
    evidence: interpretability.SpliceClassificationLogitMarginEvidence,
) -> np.ndarray:
  return route_v3._target_values(  # pylint: disable=protected-access
      evidence.target
  )


def target_readout(
    evidence: interpretability.SpliceClassificationLogitMarginEvidence,
) -> dict[str, Any]:
  """Serializes and verifies every stage of the frozen endpoint reducer."""
  selected = np.asarray(evidence.selected_logits, np.float32)
  margins = np.asarray(evidence.margins, np.float32)
  means = np.asarray(evidence.target.mean, np.float32)
  totals = np.asarray(evidence.target.total, np.float32)
  num_values = int(np.asarray(evidence.target.num_values))
  if selected.shape != (6, 2, 2):
    raise ValueError(f'Endpoint logits have wrong shape {selected.shape}.')
  if margins.shape != (6, 2) or means.shape != (6,) or totals.shape != (6,):
    raise ValueError('Endpoint reducer output shapes changed.')
  recomputed_margins = selected[..., 0] - selected[..., 1]
  if not np.array_equal(recomputed_margins, margins):
    raise ValueError('Persisted endpoint margins do not equal class-padding.')
  if num_values != 2:
    raise ValueError('Endpoint reducer no longer contains exactly two values.')
  if not np.array_equal(margins.sum(axis=1, dtype=np.float32), totals):
    raise ValueError('Persisted endpoint totals do not equal summed margins.')
  if not np.array_equal(totals / np.float32(2), means):
    raise ValueError('Persisted target means do not equal endpoint mean.')
  if not np.isfinite(selected).all():
    raise ValueError('Endpoint evidence contains non-finite logits.')
  return {
      'endpoint_axis': ['acceptor', 'donor'],
      'selected_logit_axis': ['relevant_class', 'padding_class'],
      'selected_logits': selected.tolist(),
      'endpoint_margins': margins.tolist(),
      'means': means.tolist(),
      'totals': totals.tolist(),
      'num_values': num_values,
  }


def _readout_array(readout: Mapping[str, Any], field: str) -> np.ndarray:
  return np.asarray(readout[field], np.float32)


def _assert_evidence_repeat(
    first: interpretability.SpliceClassificationLogitMarginEvidence,
    second: interpretability.SpliceClassificationLogitMarginEvidence,
) -> None:
  for name in ('selected_logits', 'margins'):
    if not np.array_equal(np.asarray(getattr(first, name)),
                          np.asarray(getattr(second, name))):
      raise ValueError(f'Endpoint evidence repeat failed at {name}.')
  for name in ('total', 'mean', 'num_values'):
    if not np.array_equal(np.asarray(getattr(first.target, name)),
                          np.asarray(getattr(second.target, name))):
      raise ValueError(f'Target repeat failed at {name}.')


def _target_map(values: Sequence[float]) -> dict[str, float]:
  return dict(zip(TRACE_ROLES, [float(x) for x in values], strict=True))


def recovery_statistics(values: Sequence[float]) -> dict[str, Any]:
  ref, alt, ref_alt, alt_alt, alt_ref, ref_ref = map(float, values)

  def recovery(patched, self_control, donor, recipient):
    denominator = donor - recipient
    return None if denominator == 0 else (patched - self_control) / denominator

  forward = recovery(ref_alt, alt_alt, ref, alt)
  reciprocal = recovery(alt_ref, ref_ref, alt, ref)
  return {
      'raw_movement': {
          'reference_into_alternate': ref_alt - alt_alt,
          'alternate_into_reference': alt_ref - ref_ref,
      },
      'recovery': {
          'reference_into_alternate': forward,
          'alternate_into_reference': reciprocal,
          'bidirectional_bottleneck': (
              None if forward is None or reciprocal is None
              else min(forward, reciprocal)
          ),
      },
  }


def validate_identity(
    first_target: interpretability.SpliceClassificationLogitMarginEvidence,
    first_trace: interpretability.SupersetGraphTrace,
    second_target: interpretability.SpliceClassificationLogitMarginEvidence,
    second_trace: interpretability.SupersetGraphTrace,
) -> dict[str, Any]:
  first = _target_values(first_target)
  second = _target_values(second_target)
  _assert_evidence_repeat(first_target, second_target)
  for name, values in (
      ('selected logits', np.asarray(first_target.selected_logits)),
      ('endpoint margins', np.asarray(first_target.margins)),
      ('target means', first),
  ):
    if not (
        np.array_equal(values[0], values[4])
        and np.array_equal(values[0], values[5])
    ):
      raise ValueError(f'Superset identity REF duplicate {name} differ.')
    if not (
        np.array_equal(values[1], values[2])
        and np.array_equal(values[1], values[3])
    ):
      raise ValueError(f'Superset identity ALT duplicate {name} differ.')
  _assert_trace_repeat(first_trace, second_trace)
  _assert_transformer_noop(first_trace.transformer)
  _assert_stage_a_noop(first_trace.stage_a)
  for natural_name, _ in _TRANSFORMER_PAIRS:
    values = np.asarray(getattr(first_trace.transformer, natural_name))
    for row, donor in enumerate(NATURAL_IDENTITY_ROWS):
      if not np.array_equal(values[:, row], values[:, donor]):
        raise ValueError(f'Transformer natural duplicate failed: {natural_name}.')
  natural_final = np.asarray(first_trace.stage_a.natural_final_embeddings)
  for row, donor in enumerate(NATURAL_IDENTITY_ROWS):
    if not np.array_equal(natural_final[row], natural_final[donor]):
      raise ValueError('Final-embedding natural duplicate failed.')
  return {
      'passed': True,
      'target_means': _target_map(first),
      'target_repeat_exact': True,
      'target_duplicates_exact': True,
      'trace_repeat_exact': True,
      'natural_duplicates_exact': True,
      'num_values': 2,
      'all_false_natural_effective_exact': True,
      'target_total_equals_two_times_mean': True,
      'trace_fingerprint_first': trace_fingerprint(first_trace),
      'trace_fingerprint_repeat': trace_fingerprint(second_trace),
  }


def _validate_common_active(
    target: interpretability.SpliceClassificationLogitMarginEvidence,
    trace: interpretability.SupersetGraphTrace,
    repeated_target: interpretability.SpliceClassificationLogitMarginEvidence,
    repeated_trace: interpretability.SupersetGraphTrace,
    identity_readout: Mapping[str, Any],
) -> np.ndarray:
  values = _target_values(target)
  repeated = _target_values(repeated_target)
  _assert_evidence_repeat(target, repeated_target)
  _assert_trace_repeat(trace, repeated_trace)
  for field, current in (
      ('selected_logits', np.asarray(target.selected_logits)),
      ('endpoint_margins', np.asarray(target.margins)),
      ('means', values),
  ):
    identity = _readout_array(identity_readout, field)
    if not np.array_equal(current[:2], identity[:2]):
      raise ValueError(f'Active baseline rows differ at {field}.')
    if not np.array_equal(current[3], identity[1]):
      raise ValueError(f'Active ALT self row differs at {field}.')
    if not np.array_equal(current[5], identity[0]):
      raise ValueError(f'Active REF self row differs at {field}.')
  return values


def validate_phase_r_group(
    target: interpretability.SpliceClassificationLogitMarginEvidence,
    trace: interpretability.SupersetGraphTrace,
    repeated_target: interpretability.SpliceClassificationLogitMarginEvidence,
    repeated_trace: interpretability.SupersetGraphTrace,
    identity_readout: Mapping[str, Any],
    group: phase_r.ResidualGridGroup,
) -> dict[str, Any]:
  values = _validate_common_active(
      target, trace, repeated_target, repeated_trace, identity_readout
  )
  stage_trace = trace.stage_a
  if not np.asarray(
      stage_trace.transformer_output_effective_matches_natural
  ).all():
    raise ValueError('Phase-R changed disabled whole T seam.')
  if not np.asarray(stage_trace.encoder_skips_natural_match_identity).all():
    raise ValueError('Phase-R encoder-skip natural duplicates differ.')
  if not np.asarray(
      stage_trace.encoder_skips_effective_match_natural
  ).all():
    raise ValueError('Phase-R changed disabled encoder-skip seam.')
  if not np.array_equal(
      stage_trace.natural_final_embeddings,
      stage_trace.effective_final_embeddings,
  ):
    raise ValueError('Phase-R changed disabled final-embedding seam.')
  active_natural = f'{group.stage}_residuals'
  active_effective = f'effective_{group.stage}_residuals'
  slots = np.asarray(group.position_set.slots, np.int32)
  natural = np.asarray(getattr(trace.transformer, active_natural))[group.layer]
  effective = np.asarray(getattr(trace.transformer, active_effective))[group.layer]
  if not len(slots):
    raise ValueError('Phase-R group has no active slots.')
  for row, donor in enumerate(NATURAL_IDENTITY_ROWS):
    if not np.array_equal(natural[row, slots], natural[donor, slots]):
      raise ValueError('Phase-R active-seam natural same-allele audit failed.')
  if not np.array_equal(effective[:2, slots], natural[:2, slots]):
    raise ValueError('Phase-R active seam changed baseline rows 0/1.')
  for recipient, donor in RECIPIENT_DONOR_ROWS:
    if not np.array_equal(effective[recipient, slots], natural[donor, slots]):
      raise ValueError('Phase-R selected live donor vector mismatch.')
  disabled_slots = np.ones((v2.NUM_TRACE_SLOTS,), bool)
  disabled_slots[slots] = False
  if not np.array_equal(effective[:, disabled_slots], natural[:, disabled_slots]):
    raise ValueError('Phase-R transfer changed a disabled residual slot.')
  for natural_name, effective_name in _TRANSFORMER_PAIRS:
    natural_all = np.asarray(getattr(trace.transformer, natural_name))
    effective_all = np.asarray(getattr(trace.transformer, effective_name))
    if natural_name == active_natural:
      for layer in range(interpretability.NUM_TRANSFORMER_LAYERS):
        if layer != group.layer and not np.array_equal(
            natural_all[layer], effective_all[layer]
        ):
          raise ValueError('Phase-R changed a disabled transformer layer.')
    elif not np.array_equal(natural_all, effective_all):
      raise ValueError(f'Phase-R changed disabled seam {natural_name}.')
  result = recovery_statistics(values)
  return {
      'passed': True,
      'target_means': _target_map(values),
      'baseline_targets_exact_from_identity': True,
      'self_targets_exact': True,
      'selected_donor_vectors_exact': True,
      'active_seam_natural_same_allele_exact': True,
      'baseline_rows_active_seam_natural_effective_exact': True,
      'disabled_seams_exact': True,
      'target_repeat_exact': True,
      'trace_repeat_exact': True,
      'trace_fingerprint': trace_fingerprint(trace),
      'repeat_trace_fingerprint': trace_fingerprint(repeated_trace),
      **result,
  }


def validate_stage_a_group(
    target: interpretability.SpliceClassificationLogitMarginEvidence,
    trace: interpretability.SupersetGraphTrace,
    repeated_target: interpretability.SpliceClassificationLogitMarginEvidence,
    repeated_trace: interpretability.SupersetGraphTrace,
    identity_readout: Mapping[str, Any],
    component: stage_a.StageAComponent,
) -> dict[str, Any]:
  _validate_common_active(
      target, trace, repeated_target, repeated_trace, identity_readout
  )
  _assert_transformer_noop(trace.transformer)
  branch = trace.stage_a
  if not np.asarray(
      branch.transformer_output_natural_matches_identity
  ).all():
    raise ValueError('Stage-A natural whole-T tensors differ by same allele.')
  if not np.asarray(branch.encoder_skips_natural_match_identity).all():
    raise ValueError('Stage-A natural encoder skips differ by same allele.')
  if component.transformer_output and not np.asarray(
      branch.transformer_output_effective_matches_natural
  )[:2].all():
    raise ValueError('Active whole T changed baseline rows 0/1.')
  if component.encoder_skips and not np.asarray(
      branch.encoder_skips_effective_match_natural
  )[:, :2].all():
    raise ValueError('Active whole E changed baseline rows 0/1.')
  if component.final_embedding and not np.array_equal(
      np.asarray(branch.natural_final_embeddings)[:2],
      np.asarray(branch.effective_final_embeddings)[:2],
  ):
    raise ValueError('Active final A/D seam changed baseline rows 0/1.')
  identity_means = _readout_array(identity_readout, 'means')
  result = stage_a.validate_component_audit(
      target.target, branch, component, identity_means
  )
  if component.final_embedding:
    natural = np.asarray(branch.natural_final_embeddings)
    for row, donor in enumerate(NATURAL_IDENTITY_ROWS):
      if not np.array_equal(natural[row], natural[donor]):
        raise ValueError('Natural final A/D embeddings differ by same allele.')
  if component.closure_required:
    selected = np.asarray(target.selected_logits)
    margins = np.asarray(target.margins)
    for recipient, donor in ((2, 0), (4, 1)):
      if not np.array_equal(selected[recipient], selected[donor]):
        raise ValueError('Closure failed at raw endpoint logits.')
      if not np.array_equal(margins[recipient], margins[donor]):
        raise ValueError('Closure failed at endpoint margins.')
  return {
      **result,
      'baseline_targets_exact_from_identity': True,
      'target_repeat_exact': True,
      'trace_repeat_exact': True,
      'transformer_residual_seams_disabled_exact': True,
      'trace_fingerprint': trace_fingerprint(trace),
      'repeat_trace_fingerprint': trace_fingerprint(repeated_trace),
      'endpoint_level_closure_exact': component.closure_required,
      'active_route_baseline_rows_noop_exact': True,
      'baseline_rows_T_natural_effective_exact': True,
      'baseline_rows_E_natural_effective_exact': True,
      'baseline_rows_final_A_D_natural_effective_exact': True,
  }


def _case_inputs(model_instance: Any, case: v2.Case):
  interval = v2.centered_interval(case, route_v3.CONTEXT_BP)
  position_sets = v2.trace_position_sets(case, interval)
  metadata = model_instance._metadata[  # pylint: disable=protected-access
      public_dna_model.Organism.HOMO_SAPIENS
  ].splice_sites
  target, resolved = route_v3.target_selection(metadata, case, interval)
  selection = superset_selection(position_sets, resolved)
  dna_batch, sequence_sha = route_v3._build_six_row_batch(  # pylint: disable=protected-access
      model_instance, case, interval
  )
  validate_reference_case(case, interval, sequence_sha)
  return interval, position_sets, selection, target, resolved, dna_batch, sequence_sha


def _timed_apply(compiled: Any, args: Sequence[Any]):
  start = time.perf_counter()
  output = compiled(*args)
  jax.block_until_ready(output)
  return output, time.perf_counter() - start


def _compiler_artifacts(lowered: Any, compiled: Any, seconds: float) -> dict[str, Any]:
  directory = OUTPUT_DIR / 'compiler'
  stable = str(lowered.compiler_ir(dialect='stablehlo'))
  hlo_object = lowered.compiler_ir(dialect='hlo')
  hlo = hlo_object.as_hlo_text() if hasattr(hlo_object, 'as_hlo_text') else str(hlo_object)
  compiled_hlo = compiled.as_text()
  records = {}
  for name, filename, text in (
      ('stablehlo', 'superset.stablehlo.mlir', stable),
      ('hlo', 'superset.pre_backend.hlo.txt', hlo),
      ('compiled_hlo', 'superset.compiled.hlo.txt', compiled_hlo),
  ):
    path = directory / filename
    digest = _write_new_text(path, text)
    records[name] = {
        'path': str(path), 'sha256': digest,
        'size_bytes': len(text.encode('utf-8')),
    }
  executable_fingerprint = hashlib.sha256(
      bytes.fromhex(records['compiled_hlo']['sha256'])
  ).hexdigest()
  provenance = {
      'compile_count': 1,
      'compile_seconds': seconds,
      'executable_fingerprint': executable_fingerprint,
      'artifacts': records,
  }
  _write_new(directory / 'COMPILER_PROVENANCE.json', provenance)
  return provenance


def _artifact_path(kind: str, case: v2.Case, key: str = '') -> Path:
  case_name = f'{case.order:03d}_{_slug(case.variant_id)}'
  if kind == 'identity':
    return OUTPUT_DIR / 'raw' / 'identity' / f'{case_name}.json'
  if kind == 'phase_r':
    return OUTPUT_DIR / 'raw' / 'phase_r' / case_name / f'{key}.json'
  if kind == 'stage_a':
    return OUTPUT_DIR / 'raw' / 'stage_a' / key / f'{case_name}.json'
  raise ValueError(f'Unknown artifact kind {kind!r}.')


def _case_record(case: v2.Case) -> dict[str, Any]:
  return v2._case_record(case)  # pylint: disable=protected-access


def _run_identity(
    compiled: Any,
    model_instance: Any,
    case: v2.Case,
    params: Any,
    state: Any,
    selection_signature: Mapping[str, Any],
    intervention_signature: Mapping[str, Any],
    target_signature: Mapping[str, Any],
    freeze_sha256: str,
    executable_fingerprint: str,
) -> tuple[dict[str, Any], tuple[Any, ...]]:
  interval, position_sets, selection, target, resolved, dna, sequence_sha = (
      _case_inputs(model_instance, case)
  )
  identity = identity_interventions(selection)
  assert_same_program_signature(selection_signature, selection)
  assert_same_program_signature(intervention_signature, identity)
  assert_same_program_signature(target_signature, target)
  args = (
      params, state, dna, jnp.zeros((6,), jnp.int32),
      selection, identity, target,
  )
  first, first_seconds = _timed_apply(compiled, args)
  second, second_seconds = _timed_apply(compiled, args)
  first_readout = target_readout(first[0])
  repeat_readout = target_readout(second[0])
  try:
    checks = validate_identity(first[0], first[1], second[0], second[1])
    direction = route_v3.direction_gate(
        case, np.asarray(first[0].target.mean)
    )
    status = 'complete'
    failure = None
  except ValueError as error:
    checks = None
    direction = None
    status = 'invalid'
    failure = {'type': type(error).__name__, 'message': str(error)}
  artifact = {
      'status': status,
      'script_version': SCRIPT_VERSION,
      'protocol_sha256': PROTOCOL_SHA256,
      'freeze_sha256': freeze_sha256,
      'executable_fingerprint': executable_fingerprint,
      'case': _case_record(case),
      'interval': {
          'chromosome': interval.chromosome,
          'start_0based': interval.start,
          'end_0based_exclusive': interval.end,
      },
      'sequence_sha256': sequence_sha,
      'resolved_position_sets': [dataclasses.asdict(x) for x in position_sets],
      'canonical_target': {
          'endpoints': [dataclasses.asdict(x) for x in resolved.endpoints],
          'padding_track_index': resolved.padding_track_index,
      },
      'target_readout': first_readout,
      'repeat_target_readout': repeat_readout,
      'trace_fingerprint': trace_fingerprint(first[1]),
      'repeat_trace_fingerprint': trace_fingerprint(second[1]),
      'checks': checks,
      'failure': failure,
      'direction_gate': direction,
      'program_signatures': {
          'selection': selection_signature,
          'interventions': intervention_signature,
          'target': target_signature,
      },
      'seconds': {'first': first_seconds, 'repeat': second_seconds},
      'created_at_unix_s': time.time(),
  }
  path = _artifact_path('identity', case)
  digest = _write_new(path, artifact)
  artifact['identity_binding'] = {
      'path': str(path.relative_to(OUTPUT_DIR)), 'sha256': digest
  }
  return artifact, (dna, selection, target)


def _run_phase_group(
    compiled: Any,
    case: v2.Case,
    common: tuple[Any, ...],
    params: Any,
    state: Any,
    identity: Mapping[str, Any],
    group: phase_r.ResidualGridGroup,
    intervention_signature: Mapping[str, Any],
    target_signature: Mapping[str, Any],
    freeze_sha256: str,
    executable_fingerprint: str,
) -> dict[str, Any]:
  dna, selection, target = common
  interventions = phase_r_interventions(selection, group)
  assert_same_program_signature(intervention_signature, interventions)
  assert_same_program_signature(target_signature, target)
  args = (params, state, dna, jnp.zeros((6,), jnp.int32), selection,
          interventions, target)
  first, first_seconds = _timed_apply(compiled, args)
  second, repeat_seconds = _timed_apply(compiled, args)
  first_readout = target_readout(first[0])
  repeat_readout = target_readout(second[0])
  try:
    checks = validate_phase_r_group(
        first[0], first[1], second[0], second[1],
        identity['target_readout'], group
    )
    status = 'complete'
    failure = None
  except ValueError as error:
    checks = None
    status = 'invalid'
    failure = {'type': type(error).__name__, 'message': str(error)}
  artifact = {
      'status': status,
      'script_version': SCRIPT_VERSION,
      'protocol_sha256': PROTOCOL_SHA256,
      'freeze_sha256': freeze_sha256,
      'executable_fingerprint': executable_fingerprint,
      'identity_binding': identity['identity_binding'],
      'case': _case_record(case),
      'family': 'phase_r',
      'group': {
          'order': group.order, 'stage': group.stage, 'layer': group.layer,
          'position_set': dataclasses.asdict(group.position_set),
          'is_candidate': group.is_candidate,
      },
      'target_readout': first_readout,
      'repeat_target_readout': repeat_readout,
      'trace_fingerprint': trace_fingerprint(first[1]),
      'repeat_trace_fingerprint': trace_fingerprint(second[1]),
      'checks': checks,
      'failure': failure,
      'same_compiled_executable': True,
      'seconds': {'first': first_seconds, 'repeat': repeat_seconds},
      'created_at_unix_s': time.time(),
  }
  _write_new(_artifact_path('phase_r', case, group.key), artifact)
  return artifact


def _run_stage_group(
    compiled: Any,
    case: v2.Case,
    common: tuple[Any, ...],
    params: Any,
    state: Any,
    identity: Mapping[str, Any],
    component: stage_a.StageAComponent,
    intervention_signature: Mapping[str, Any],
    target_signature: Mapping[str, Any],
    freeze_sha256: str,
    executable_fingerprint: str,
) -> dict[str, Any]:
  dna, selection, target = common
  interventions = stage_a_interventions(selection, component)
  assert_same_program_signature(intervention_signature, interventions)
  assert_same_program_signature(target_signature, target)
  args = (params, state, dna, jnp.zeros((6,), jnp.int32), selection,
          interventions, target)
  first, first_seconds = _timed_apply(compiled, args)
  second, repeat_seconds = _timed_apply(compiled, args)
  first_readout = target_readout(first[0])
  repeat_readout = target_readout(second[0])
  try:
    checks = validate_stage_a_group(
        first[0], first[1], second[0], second[1],
        identity['target_readout'], component
    )
    status = 'complete'
    failure = None
  except ValueError as error:
    checks = None
    status = 'invalid'
    failure = {'type': type(error).__name__, 'message': str(error)}
  artifact = {
      'status': status,
      'script_version': SCRIPT_VERSION,
      'protocol_sha256': PROTOCOL_SHA256,
      'freeze_sha256': freeze_sha256,
      'executable_fingerprint': executable_fingerprint,
      'identity_binding': identity['identity_binding'],
      'case': _case_record(case),
      'family': 'stage_a',
      'component': dataclasses.asdict(component),
      'target_readout': first_readout,
      'repeat_target_readout': repeat_readout,
      'trace_fingerprint': trace_fingerprint(first[1]),
      'repeat_trace_fingerprint': trace_fingerprint(second[1]),
      'checks': checks,
      'failure': failure,
      'same_compiled_executable': True,
      'seconds': {'first': first_seconds, 'repeat': repeat_seconds},
      'created_at_unix_s': time.time(),
  }
  _write_new(_artifact_path('stage_a', case, component.key), artifact)
  return artifact


def _tree_digest(paths: Sequence[Path], root: Path) -> str:
  digest = hashlib.sha256()
  for path in sorted(paths):
    digest.update(str(path.relative_to(root)).encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(_sha256(path)))
  return digest.hexdigest()


def _raw_manifest() -> dict[str, Any]:
  paths = sorted(path for path in (OUTPUT_DIR / 'raw').rglob('*.json'))
  return {
      'artifact_count': len(paths),
      'artifact_sha256': {
          str(path.relative_to(OUTPUT_DIR)): _sha256(path) for path in paths
      },
      'artifact_tree_sha256': _tree_digest(paths, OUTPUT_DIR),
  }


def _write_completion(
    *,
    stop_reason: str | None,
    message: str,
    identity_count: int,
    eligible_effect_count: int,
    phase_results: Sequence[Mapping[str, Any]],
    stage_results: Sequence[Mapping[str, Any]],
    closures_passed: bool | None,
    compiler: Mapping[str, Any],
) -> dict[str, Any]:
  """Writes the controlled scientific stop/completion record."""
  raw = _raw_manifest()
  _write_new(OUTPUT_DIR / 'RAW_MANIFEST.json', raw)
  record = {
      'status': 'complete' if stop_reason is None else 'controlled_stop',
      'stop_reason': stop_reason,
      'message': message,
      'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION,
      'protocol_sha256': PROTOCOL_SHA256,
      'identity_count': identity_count,
      'eligible_effect_count': eligible_effect_count,
      'phase_r_group_count': len(phase_results),
      'phase_r_invalid_count': sum(
          x['status'] != 'complete' for x in phase_results
      ),
      'stage_a_group_count': len(stage_results),
      'stage_a_invalid_count': sum(
          x['status'] != 'complete' for x in stage_results
      ),
      'closures_passed': closures_passed,
      'single_executable': dict(compiler),
      'import_provenance_sha256': _sha256(
          OUTPUT_DIR / 'IMPORT_PROVENANCE.json'
      ),
      'import_provenance_phases': {
          name: _sha256(OUTPUT_DIR / filename)
          for name, filename in (
              ('pre_model', 'IMPORT_PROVENANCE_PRE_MODEL.json'),
              ('post_model_precompile',
               'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json'),
              ('postcompile', 'IMPORT_PROVENANCE.json'),
          )
      },
      'protobuf_provenance_sha256': _sha256(
          OUTPUT_DIR / 'PROTOBUF_PROVENANCE.json'
      ),
      'raw_manifest': raw,
      'confirmation_model_calls': 0,
      'confirmation_scope_disclosure': (
          'Later-exon metadata/labels were exposed after protocol freeze; '
          'no later-exon model outputs, activations, or interventions are used.'
      ),
      'completed_at_unix_s': time.time(),
  }
  _write_new(OUTPUT_DIR / 'RUN_COMPLETE.json', record)
  return record


def build_dry_run_plan(
    cases: Sequence[v2.Case], *, max_variants: int, max_groups: int
) -> dict[str, Any]:
  shown = cases[:max_variants] if max_variants else cases
  groups = phase_r.enumerate_groups(
      cases[0], v2.centered_interval(cases[0], route_v3.CONTEXT_BP)
  )
  shown_groups = groups[:max_groups] if max_groups else groups
  return {
      'script_version': SCRIPT_VERSION,
      'protocol_sha256': PROTOCOL_SHA256,
      'dry_run': True,
      'development_case_count': len(cases),
      'displayed_case_count': len(shown),
      'identity_calls': len(cases) * 2,
      'phase_r_groups_per_eligible_effect': len(groups),
      'displayed_phase_r_groups': len(shown_groups),
      'phase_r_candidates': sum(group.is_candidate for group in groups),
      'stage_a_components': STAGE_COMPONENT_NAMES,
      'compile_count': 1,
      'confirmation_model_calls': 0,
      'output_dir': str(OUTPUT_DIR),
      'analysis_dir': str(ANALYSIS_DIR),
  }


def validate_freeze() -> dict[str, Any]:
  frozen = json.loads(FREEZE_PATH.read_text(encoding='utf-8'))
  observed_protobuf = protobuf_provenance()
  upstream_head = subprocess.check_output(
      ('git', '-C', str(_HERE.parents[3] / 'alphagenome'), 'rev-parse', 'HEAD'),
      text=True,
  ).strip()
  expected = {
      'script_version': SCRIPT_VERSION,
      'attempt_id': ATTEMPT_ID,
      'protocol_sha256': PROTOCOL_SHA256,
      'selected_variants_sha256': route_v3.SELECTED_SHA256,
      'frozen_exons_sha256': route_v3.EXONS_SHA256,
      'development_variants_sha256': DEVELOPMENT_VARIANTS_SHA256,
      'development_exons_sha256': DEVELOPMENT_EXONS_SHA256,
      'development_variants_path': str(DEVELOPMENT_VARIANTS_PATH.resolve()),
      'development_exons_path': str(DEVELOPMENT_EXONS_PATH.resolve()),
      'checkpoint_snapshot': route_v3.CHECKPOINT_SNAPSHOT,
      'context_bp': route_v3.CONTEXT_BP,
      'attention_backend': route_v3.ATTENTION_BACKEND,
      'reference_url': REFERENCE_URL,
      'reference_object': REFERENCE_OBJECT,
      'reference_bindings_path': str(REFERENCE_BINDINGS_PATH.resolve()),
      'reference_bindings_sha256': REFERENCE_BINDINGS_SHA256,
      'checkpoint_manifest_path': str(CHECKPOINT_MANIFEST_PATH.resolve()),
      'checkpoint_manifest_sha256': CHECKPOINT_MANIFEST_SHA256,
      'output_dir': str(OUTPUT_DIR.resolve()),
      'analysis_dir': str(ANALYSIS_DIR.resolve()),
      'preflight_dir': str(
          (_HERE / 'results' / 'v3_2_device_preflight').resolve()
      ),
      'preflight_script_version': 'opensplice-device-preflight-v3.2.0',
      'expected_device_kind': EXPECTED_DEVICE_KIND,
      'expected_gpu_uuid': EXPECTED_GPU_UUID,
      'expected_compute_capability': EXPECTED_COMPUTE_CAPABILITY,
      'upstream_alphagenome_git_head': upstream_head,
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
      'phase_r_groups_per_eligible_effect': 216,
      'phase_r_candidate_count': 72,
      'stage_a_component_keys': [
          component.key for component in stage_a.enumerate_components()
      ],
      'protobuf_binding': observed_protobuf,
  }
  for name, value in expected.items():
    if frozen.get(name) != value:
      raise ValueError(f'v3.2 freeze mismatch: {name}.')
  for relative, digest in frozen['file_sha256'].items():
    path = _HERE.parents[2] / relative
    _reject_confirmation_path(path)
    if _sha256(path) != digest:
      raise ValueError(f'v3.2 frozen file changed: {relative}.')
  return {**frozen, 'path': str(FREEZE_PATH), 'sha256': _sha256(FREEZE_PATH)}


def validate_bundle_committed(frozen: Mapping[str, Any]) -> dict[str, Any]:
  repo = _HERE.parents[2]
  relative = tuple(frozen['file_sha256']) + (
      str(FREEZE_PATH.relative_to(repo)),
  )
  for path in relative:
    subprocess.run(
        ('git', '-C', str(repo), 'ls-files', '--error-unmatch', path),
        check=True, capture_output=True,
    )
  if subprocess.check_output(
      ('git', '-C', str(repo), 'diff', '--binary', 'HEAD', '--', *relative)
  ):
    raise ValueError('Committed v3.2 bundle differs from the working tree.')
  proto_status = subprocess.check_output(
      ('git', '-C', str(repo), 'status', '--porcelain=v1',
       '--untracked-files=all', '--', 'src/alphagenome_research/protos'),
      text=True,
  ).splitlines()
  expected_generated = {
      '?? src/alphagenome_research/protos/calibration_scores_pb2.py',
      '?? src/alphagenome_research/protos/calibration_scores_pb2.pyi',
  }
  if set(proto_status) != expected_generated:
    raise ValueError(
        'Generated protobuf status differs from exact untracked allowlist: '
        f'{proto_status}.'
    )
  return {
      'git_head': subprocess.check_output(
          ('git', '-C', str(repo), 'rev-parse', 'HEAD'), text=True
      ).strip(),
      'tracked_clean': True,
      'generated_artifact_exception': sorted(expected_generated),
  }


def validate_external_preflight(path: Path, frozen: Mapping[str, Any]) -> dict[str, Any]:
  _reject_confirmation_path(path)
  record = json.loads(path.read_text(encoding='utf-8'))
  if record.get('script_version') != 'opensplice-device-preflight-v3.2.0':
    raise ValueError('Wrong v3.2 external preflight version.')
  if record.get('status') != 'pass':
    raise ValueError('v3.2 external preflight did not pass.')
  if record.get('protocol_sha256') != PROTOCOL_SHA256:
    raise ValueError('v3.2 external preflight used another protocol.')
  if record.get('freeze_sha256') != frozen['sha256']:
    raise ValueError('v3.2 external preflight used another freeze.')
  if not record.get('no_model_or_biological_access'):
    raise ValueError('v3.2 preflight imported biological/model code.')
  if not record.get('no_jit_or_array_kernel'):
    raise ValueError('v3.2 preflight performed a JIT/array operation.')
  observation = record['observation']
  if observation.get('jax_enable_compilation_cache') is not False:
    raise ValueError('v3.2 preflight observed an enabled JAX cache.')
  current_environment = {
      name: value for name, value in sorted(os.environ.items())
      if name.startswith(ENVIRONMENT_PREFIXES)
  }
  if observation.get('v3_2_runtime_environment') != current_environment:
    raise ValueError('External and same-launch v3.2 environments differ.')
  preflight_root = Path(frozen['preflight_dir']).resolve()
  log_bindings = {}
  for stream in ('stdout', 'stderr'):
    binding = record['logs'][stream]
    log_path = Path(binding['path']).resolve()
    _reject_confirmation_path(log_path)
    if not log_path.is_relative_to(preflight_root):
      raise ValueError(f'External preflight {stream} escaped its directory.')
    if _sha256(log_path) != binding['sha256']:
      raise ValueError(f'External preflight {stream} log changed.')
    log_bindings[stream] = {
        'path': str(log_path), 'sha256': binding['sha256']
    }
  device_gate.validate_device_observation(observation)
  return {
      'path': str(path.resolve()), 'sha256': _sha256(path),
      **record, 'validated_logs': log_bindings,
  }


def main() -> None:
  args = _parse_args()
  bootstrap_attestation = consume_bootstrap_attestation()
  for path in (
      OUTPUT_DIR, ANALYSIS_DIR, PROTOCOL_PATH, FREEZE_PATH,
      DEVELOPMENT_VARIANTS_PATH, DEVELOPMENT_EXONS_PATH,
      CHECKPOINT_MANIFEST_PATH, REFERENCE_BINDINGS_PATH,
  ):
    _reject_confirmation_path(path)
  if args.successful_preflight is not None:
    _reject_confirmation_path(args.successful_preflight)
  if args.checkpoint is not None:
    _reject_confirmation_path(args.checkpoint)
  if _sha256(PROTOCOL_PATH) != PROTOCOL_SHA256:
    raise ValueError('v3.2 protocol hash mismatch.')
  cases = load_development_cases()
  if args.dry_run:
    print(json.dumps(build_dry_run_plan(
        cases, max_variants=args.max_variants, max_groups=args.max_groups
    ), indent=2))
    return

  environment = assert_v3_2_environment()
  frozen = validate_freeze()
  bundle = validate_bundle_committed(frozen)
  if args.successful_preflight is None:
    raise ValueError('v3.2 requires its successful external preflight.')
  external = validate_external_preflight(args.successful_preflight, frozen)
  same_process = device_gate.collect_device_observation()
  device_gate.validate_device_observation(same_process)
  if OUTPUT_DIR.exists() or ANALYSIS_DIR.exists():
    raise FileExistsError('v3.2 output/analysis exists; never resume or retry.')
  checkpoint = v2._checkpoint_path(args.checkpoint)  # pylint: disable=protected-access
  _reject_confirmation_path(checkpoint)
  if checkpoint.name != route_v3.CHECKPOINT_SNAPSHOT:
    raise ValueError('v3.2 checkpoint snapshot changed.')
  checkpoint_binding = validate_checkpoint(checkpoint)
  reference_object_binding = validate_reference_object()
  start = {
      'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION,
      'status': 'started_append_only_one_shot',
      'protocol_sha256': PROTOCOL_SHA256,
      'freeze': frozen,
      'bundle': bundle,
      'external_preflight': external,
      'same_process_preflight': same_process,
      'runtime_environment': environment,
      'same_process_pre_import_bootstrap': bootstrap_attestation,
      'checkpoint_path': str(checkpoint),
      'checkpoint_binding': checkpoint_binding,
      'reference_object_binding': reference_object_binding,
      'reference_sequence_bindings': {
          'path': str(REFERENCE_BINDINGS_PATH.resolve()),
          'sha256': REFERENCE_BINDINGS_SHA256,
      },
      'compile_count_contract': 1,
      'confirmation_model_calls': 0,
      'confirmation_scope_disclosure': (
          'Later-exon metadata/labels were exposed after protocol freeze; '
          'no later-exon model outputs, activations, or interventions are used.'
      ),
      'started_at_unix_s': time.time(),
  }
  _write_new(START_PATH, start)
  _write_new(
      OUTPUT_DIR / 'PROTOBUF_PROVENANCE.json', frozen['protobuf_binding']
  )
  imports_pre_model = import_provenance()
  _write_new(
      OUTPUT_DIR / 'IMPORT_PROVENANCE_PRE_MODEL.json', imports_pre_model
  )

  try:
    model_instance = dna_model.create(
        checkpoint,
        model_settings=dna_model.ModelSettings(
            attention_backend=route_v3.ATTENTION_BACKEND
        ),
    )
    params = model_instance._params  # pylint: disable=protected-access
    state = model_instance._state  # pylint: disable=protected-access
    prototype = cases[0]
    _, _, prototype_selection, prototype_target, _, prototype_dna, _ = (
        _case_inputs(model_instance, prototype)
    )
    prototype_interventions = identity_interventions(prototype_selection)
    selection_signature = pytree_signature(prototype_selection)
    intervention_signature = pytree_signature(prototype_interventions)
    target_signature = pytree_signature(prototype_target)
    raw_apply = (
        dna_model
        .create_splice_classification_logit_margin_superset_graph_apply(
            model_instance._metadata,  # pylint: disable=protected-access
            attention_backend=route_v3.ATTENTION_BACKEND,
        )
    )
    imports_post_model = import_provenance()
    assert_import_provenance_stable(imports_pre_model, imports_post_model)
    _write_new(
        OUTPUT_DIR / 'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
        imports_post_model,
    )
    prototype_args = (
        params, state, prototype_dna, jnp.zeros((6,), jnp.int32),
        prototype_selection, prototype_interventions, prototype_target,
    )
    compile_start = time.perf_counter()
    lowered = jax.jit(raw_apply).lower(*prototype_args)
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - compile_start
    compiler = _compiler_artifacts(lowered, compiled, compile_seconds)
    imports = import_provenance()
    assert_import_provenance_stable(imports_post_model, imports)
    _write_new(OUTPUT_DIR / 'IMPORT_PROVENANCE.json', imports)

    identities = {}
    live_inputs = {}
    identity_failures = []
    for case in cases:
      artifact, common = _run_identity(
          compiled, model_instance, case, params, state,
          selection_signature, intervention_signature,
          target_signature,
          frozen['sha256'], compiler['executable_fingerprint'],
      )
      identities[case.variant_id] = artifact
      live_inputs[case.variant_id] = common
      if artifact['status'] != 'complete':
        identity_failures.append(case.variant_id)
    if identity_failures:
      _write_completion(
          stop_reason='identity_tooling_failure',
          message=f'Identity Gate 0 failed for {identity_failures}.',
          identity_count=len(identities),
          eligible_effect_count=0,
          phase_results=(),
          stage_results=(),
          closures_passed=None,
          compiler=compiler,
      )
      return

    eligible = tuple(
        case for case in cases
        if identities[case.variant_id]['direction_gate'][
            'eligible_for_causal_census'
        ]
    )
    per_gene = {
        gene: sum(case.gene == gene and case.is_effect for case in eligible)
        for gene in route_v3.DEVELOPMENT_GENES
    }
    _write_new(OUTPUT_DIR / 'TARGET_ELIGIBILITY.json', {
        'eligible_effects': [case.variant_id for case in eligible],
        'ineligible_effects': [
            case.variant_id for case in cases
            if case.is_effect and case not in eligible
        ],
        'neutral_controls': [case.variant_id for case in cases if not case.is_effect],
        'eligible_effects_per_gene': per_gene,
    })
    if any(count < 3 for count in per_gene.values()):
      _write_completion(
          stop_reason='target_predictive_failure',
          message=f'Target eligibility failed: {per_gene}.',
          identity_count=len(identities),
          eligible_effect_count=len(eligible),
          phase_results=(),
          stage_results=(),
          closures_passed=None,
          compiler=compiler,
      )
      return

    phase_results = []
    for case in eligible:
      interval = v2.centered_interval(case, route_v3.CONTEXT_BP)
      for group in phase_r.enumerate_groups(case, interval):
        phase_results.append(_run_phase_group(
            compiled, case, live_inputs[case.variant_id], params, state,
            identities[case.variant_id], group, intervention_signature,
            target_signature,
            frozen['sha256'], compiler['executable_fingerprint'],
        ))

    effects = tuple(case for case in cases if case.is_effect)
    components = stage_a.enumerate_components()
    stage_results = []
    for component in components[:2]:
      family_results = []
      for case in effects:
        result = _run_stage_group(
            compiled, case, live_inputs[case.variant_id], params, state,
            identities[case.variant_id], component, intervention_signature,
            target_signature,
            frozen['sha256'], compiler['executable_fingerprint'],
        )
        family_results.append(result)
        stage_results.append(result)
      if not all(x['status'] == 'complete' for x in family_results):
        _write_completion(
            stop_reason='closure_tooling_failure',
            message=f'Mandatory closure family failed: {component.key}.',
            identity_count=len(identities),
            eligible_effect_count=len(eligible),
            phase_results=phase_results,
            stage_results=stage_results,
            closures_passed=False,
            compiler=compiler,
        )
        return
    closure_passed = True
    for component in components[2:]:
      for case in eligible:
        stage_results.append(_run_stage_group(
            compiled, case, live_inputs[case.variant_id], params, state,
            identities[case.variant_id], component, intervention_signature,
            target_signature,
            frozen['sha256'], compiler['executable_fingerprint'],
        ))

    _write_completion(
        stop_reason=None,
        message='All frozen development raw families completed.',
        identity_count=len(identities),
        eligible_effect_count=len(eligible),
        phase_results=phase_results,
        stage_results=stage_results,
        closures_passed=closure_passed,
        compiler=compiler,
    )
  except Exception as error:
    _write_new(OUTPUT_DIR / 'TERMINAL_FAILURE.json', {
        'status': 'terminal_failure',
        'type': type(error).__name__,
        'message': str(error),
        'traceback': ''.join(traceback.format_exception(error)),
        'created_at_unix_s': time.time(),
    })
    raise


if __name__ == '__main__':
  main()
