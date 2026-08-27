#!/usr/bin/env python3
"""One-shot exact-source Gate-0 identity diagnostic for OpenSplice v3.1.1.

This launcher intentionally lives outside the historical source tree.  It
loads every model and OpenSplice helper module from a clean detached worktree
at the original Phase-R commit, runs only the 20 frozen development identities,
and publishes append-only raw artifacts.  It never constructs an active
intervention or accesses the confirmation partition.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence


SCRIPT_VERSION = 'opensplice-exact-source-gate0-v3.1.1'
ATTEMPT_ID = 'opensplice-v3.1.1-exact-source-gate0-one-shot'
LOCK_COMMIT = 'fd4dc6913335a6966420d60ef04bc4643b751a27'
LOCKED_PHASE_R_TREE_SHA256 = (
    'ff7182be96e4b5be52e022e613ac16f476651924ff36d6a11b397b95613a3436'
)
LOCKED_PHASE_R_ANALYSIS_SHA256 = (
    '0131d591197fb187b9f291479e028c32c87313e40addd411235cb650df018a21'
)
TARGET_TOLERANCE = 2**-8
EFFECT_THRESHOLD = 0.01
EXPECTED_CASES = 20

_HERE = Path(__file__).resolve().parent
_CURRENT_REPO = _HERE.parents[2]
_PROTOCOL_PATH = (
    _HERE
    / 'v3_wider_mechanism'
    / 'exact_source_gate0_diagnostic_v3_1_1.md'
)
_FREEZE_CONFIG_PATH = _HERE / 'exact_source_gate0_v3_1_1_freeze.json'
PROTOCOL_SHA256 = (
    'fe0ca92a2e56bca168361c68d4103da637a1fc0a2ba8831fb8059c72574477c3'
)
_LOCKED_RESULT_DIR = (
    _HERE / 'results' / 'v3_development_phase_r_logit_margin'
)
OUTPUT_DIR = (
    _HERE / 'results' / 'v3_1_1_exact_source_gate0_identity_one_shot'
)
START_PATH = OUTPUT_DIR / 'ATTEMPT_STARTED.json'

TRACE_BATCH_ROLES = (
    'reference_baseline',
    'alternate_baseline',
    'reference_into_alternate',
    'alternate_into_alternate_self_control',
    'alternate_into_reference',
    'reference_into_reference_self_control',
)
EXPECTED_MODULES = (
    'run_inference_trace',
    'target_reducers_v3',
    'run_route_census_v3',
    'run_phase_r_v3',
    'alphagenome_research.model.interpretability',
    'alphagenome_research.model.model',
    'alphagenome_research.model.dna_model',
    'alphagenome_research.protos.calibration_scores_pb2',
)
GENERATED_PROTO_PATH = 'src/alphagenome_research/protos/calibration_scores.proto'
GENERATED_PB2_PATH = 'src/alphagenome_research/protos/calibration_scores_pb2.py'
GENERATED_PYI_PATH = 'src/alphagenome_research/protos/calibration_scores_pb2.pyi'
GENERATED_PROTO_SHA256 = (
    '356f08689a4bafa0761f88f08dac08468a2de2c8aef38dcef093457eceee2f34'
)
MIXED_PRECISION_POLICY = (
    'params=float32,compute=bfloat16,output=bfloat16'
)


@dataclasses.dataclass(frozen=True)
class LockedModules:
  """References imported exclusively from the historical worktree."""

  jax: Any
  jnp: Any
  np: Any
  public_dna_model: Any
  dna_model: Any
  interpretability: Any
  v2: Any
  route_v3: Any
  phase_r_v3: Any
  imported_modules: Mapping[str, Any]


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--locked-checkout',
      required=True,
      type=Path,
      help='Clean detached worktree at the exact historical lock commit.',
  )
  parser.add_argument('--checkpoint', type=Path)
  parser.add_argument('--dry-run', action='store_true')
  return parser.parse_args()


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
  return (
      json.dumps(
          value,
          indent=2,
          sort_keys=True,
          allow_nan=False,
          default=str,
      )
      + '\n'
  ).encode('utf-8')


def _write_new(path: Path, value: Any) -> str:
  """Atomically publishes one immutable JSON artifact and returns its hash."""
  path.parent.mkdir(parents=True, exist_ok=True)
  data = _json_bytes(value)
  temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
  try:
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    with os.fdopen(descriptor, 'wb') as handle:
      handle.write(data)
      handle.flush()
      os.fsync(handle.fileno())
    try:
      os.link(temporary, path)
    except FileExistsError as error:
      raise FileExistsError(f'Append-only artifact already exists: {path}') from error
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
      os.fsync(directory_fd)
    finally:
      os.close(directory_fd)
  finally:
    temporary.unlink(missing_ok=True)
  return hashlib.sha256(data).hexdigest()


def _write_new_text(path: Path, text: str) -> str:
  path.parent.mkdir(parents=True, exist_ok=True)
  data = text.encode('utf-8')
  temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
  try:
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    with os.fdopen(descriptor, 'wb') as handle:
      handle.write(data)
      handle.flush()
      os.fsync(handle.fileno())
    try:
      os.link(temporary, path)
    except FileExistsError as error:
      raise FileExistsError(f'Append-only artifact already exists: {path}') from error
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
      os.fsync(directory_fd)
    finally:
      os.close(directory_fd)
  finally:
    temporary.unlink(missing_ok=True)
  return hashlib.sha256(data).hexdigest()


def _run_text(command: Sequence[str], *, cwd: Path | None = None) -> str:
  return subprocess.check_output(
      tuple(command), cwd=cwd, text=True, stderr=subprocess.STDOUT
  ).strip()


def validate_locked_worktree(
    root: Path, expected_commit: str = LOCK_COMMIT
) -> dict[str, Any]:
  root = root.resolve()
  if not (root / '.git').exists():
    raise ValueError(f'Locked checkout is not a git worktree: {root}.')
  head = _run_text(('git', '-C', str(root), 'rev-parse', 'HEAD'))
  if head != expected_commit:
    raise ValueError(f'Locked checkout HEAD is {head}, expected {expected_commit}.')
  branch = _run_text(
      ('git', '-C', str(root), 'symbolic-ref', '--short', '-q', 'HEAD')
  ) if subprocess.run(
      ('git', '-C', str(root), 'symbolic-ref', '-q', 'HEAD'),
      check=False,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
  ).returncode == 0 else ''
  if branch:
    raise ValueError(f'Locked checkout must be detached, observed {branch!r}.')
  tracked_status = _run_text((
      'git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'
  ))
  if tracked_status:
    raise ValueError(f'Locked checkout has tracked changes: {tracked_status}.')
  dirty_diff = subprocess.check_output((
      'git', '-C', str(root), 'diff', '--binary', 'HEAD', '--'
  ))
  return {
      'root': str(root),
      'head': head,
      'detached': True,
      'tracked_clean': True,
      'tracked_dirty_diff_sha256': hashlib.sha256(dirty_diff).hexdigest(),
  }


def validate_generated_proto_binding(locked_root: Path) -> dict[str, Any]:
  """Binds the repo build hook's required, untracked protobuf outputs."""
  proto = locked_root / GENERATED_PROTO_PATH
  generated = (
      locked_root / GENERATED_PB2_PATH,
      locked_root / GENERATED_PYI_PATH,
  )
  if _sha256(proto) != GENERATED_PROTO_SHA256:
    raise ValueError('Locked calibration proto hash mismatch.')
  if any(not path.is_file() for path in generated):
    raise ValueError(
        'Locked checkout lacks generated calibration protobuf bindings; run '
        'the exact-source wrapper so the recorded build step can create them.'
    )
  allowed = {GENERATED_PB2_PATH, GENERATED_PYI_PATH}
  untracked = _run_text((
      'git', '-C', str(locked_root), 'ls-files', '--others',
      '--exclude-standard'
  )).splitlines()
  unexpected = sorted(set(untracked) - allowed)
  if unexpected:
    raise ValueError(f'Locked checkout has unexpected untracked files: {unexpected}.')
  return {
      'source_proto': {
          'path': GENERATED_PROTO_PATH,
          'sha256': GENERATED_PROTO_SHA256,
      },
      'generated_bindings': {
          str(path.relative_to(locked_root)): {
              'sha256': _sha256(path),
              'size_bytes': path.stat().st_size,
          }
          for path in generated
      },
      'tracked_source_unchanged': True,
      'unexpected_untracked_files': (),
  }


def validate_freeze_configuration(
    proto_build: Mapping[str, Any],
    import_provenance: Mapping[str, Any],
    mixed_precision: Mapping[str, Any],
) -> dict[str, Any]:
  """Binds the final launchers and dry-run build artifact before GPU use."""
  frozen = json.loads(_FREEZE_CONFIG_PATH.read_text(encoding='utf-8'))
  expected = {
      'script_version': SCRIPT_VERSION,
      'attempt_id': ATTEMPT_ID,
      'lock_commit': LOCK_COMMIT,
      'protocol_sha256': PROTOCOL_SHA256,
      'launcher_sha256': _sha256(Path(__file__).resolve()),
      'wrapper_sha256': _sha256(
          Path(__file__).with_suffix('.sh').resolve()
      ),
      'proto_build_manifest_sha256': proto_build['manifest_sha256'],
      'generated_output_sha256': {
          name: value['sha256']
          for name, value in proto_build['generated_outputs'].items()
      },
      'proto_generation': {
          'vector': 'standalone_protoc_not_grpc_tools',
          'grpcio_tools_used': False,
          'grpcio_tools_version': proto_build['tool']['grpcio_tools_version'],
          'protoc_path': proto_build['tool']['protoc_path'],
          'protoc_version': proto_build['tool']['protoc_version'],
          'command_argv': proto_build['command_argv'],
          'input_protos': proto_build['input_protos'],
          'build_recipe': proto_build['build_recipe'],
      },
      'initial_transitive_import_tree_sha256': import_provenance[
          'module_tree_sha256'
      ],
      'alphagenome_dependency_binding': import_provenance[
          'alphagenome_dependency_binding'
      ],
      'mixed_precision': mixed_precision,
  }
  for name, value in expected.items():
    if _json_normalized(frozen.get(name)) != _json_normalized(value):
      raise ValueError(f'Frozen v3.1.1 launch configuration mismatch: {name}.')
  return {
      **frozen,
      'path': str(_FREEZE_CONFIG_PATH.resolve()),
      'sha256': _sha256(_FREEZE_CONFIG_PATH),
  }


def prepare_generated_proto_binding(
    locked_root: Path,
    *,
    protoc_bin: Path,
    alphagenome_proto_root: Path,
) -> dict[str, Any]:
  """Performs or verifies the one declared out-of-tree-recorded build step."""
  proto = locked_root / GENERATED_PROTO_PATH
  pb2 = locked_root / GENERATED_PB2_PATH
  pyi = locked_root / GENERATED_PYI_PATH
  manifest_path = locked_root.parent / f'.{locked_root.name}.v3_1_1_proto_build.json'
  outputs_exist = (pb2.exists(), pyi.exists())
  if manifest_path.exists():
    if outputs_exist != (True, True):
      raise ValueError('Recorded protobuf build exists but its outputs are missing.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for path in (pb2, pyi):
      relative = str(path.relative_to(locked_root))
      if manifest['generated_outputs'][relative]['sha256'] != _sha256(path):
        raise ValueError(f'Generated protobuf changed after its build: {relative}.')
    return {**manifest, 'manifest_path': str(manifest_path),
            'manifest_sha256': _sha256(manifest_path), 'reused': True}
  if outputs_exist != (False, False):
    raise ValueError('Protobuf outputs are partial or pre-exist without provenance.')
  if not protoc_bin.is_file() or not os.access(protoc_bin, os.X_OK):
    raise ValueError(f'protoc is not executable: {protoc_bin}.')
  dependency_proto = alphagenome_proto_root / 'alphagenome/protos/dna_model.proto'
  if not dependency_proto.is_file():
    raise ValueError(f'AlphaGenome dependency proto is missing: {dependency_proto}.')
  command = (
      str(protoc_bin),
      f'--proto_path={locked_root / "src"}',
      f'--proto_path={alphagenome_proto_root}',
      f'--python_out={locked_root / "src"}',
      f'--pyi_out={locked_root / "src"}',
      str(proto),
  )
  result = subprocess.run(command, check=False, capture_output=True, text=True)
  if result.returncode or not pb2.is_file() or not pyi.is_file():
    raise RuntimeError(
        'Exact-source protobuf generation failed before model/GPU access: '
        f'rc={result.returncode}, stdout={result.stdout!r}, '
        f'stderr={result.stderr!r}.'
    )
  manifest = {
      'status': 'complete',
      'purpose': 'declared_repository_build_hook_output_not_model_source_patch',
      'initial_outputs_absent': True,
      'locked_commit': LOCK_COMMIT,
      'tracked_source_clean_before_generation': True,
      'command_argv': command,
      'include_roots': (str(locked_root / 'src'), str(alphagenome_proto_root)),
      'tool': {
          'protoc_path': str(protoc_bin.resolve()),
          'protoc_version': _safe_command((str(protoc_bin), '--version')),
          'python': sys.version,
          'grpcio_tools_version': _package_version('grpcio-tools'),
          'protobuf_version': _package_version('protobuf'),
      },
      'build_recipe': {
          'pyproject_toml_sha256': _sha256(locked_root / 'pyproject.toml'),
          'hatch_build_py_sha256': _sha256(locked_root / 'hatch_build.py'),
      },
      'input_protos': {
          str(proto.relative_to(locked_root)): _sha256(proto),
          str(dependency_proto): _sha256(dependency_proto),
      },
      'generated_outputs': {
          str(path.relative_to(locked_root)): {
              'sha256': _sha256(path), 'size_bytes': path.stat().st_size
          }
          for path in (pb2, pyi)
      },
      'stdout': result.stdout,
      'stderr': result.stderr,
      'created_at_unix_s': time.time(),
  }
  _write_new(manifest_path, manifest)
  return {**manifest, 'manifest_path': str(manifest_path),
          'manifest_sha256': _sha256(manifest_path), 'reused': False}


def disclose_proto_generation(proto_build: Mapping[str, Any]) -> dict[str, Any]:
  """States the actual generation vector without implying hatch-hook identity."""
  grpcio_tools_version = proto_build['tool']['grpcio_tools_version']
  if grpcio_tools_version is not None:
    raise ValueError(
        'Frozen v3.1.1 generation used standalone protoc, not grpcio-tools.'
    )
  return {
      **proto_build,
      'generation_disclosure': {
          'vector': 'standalone_protoc_29_3_command_not_hatch_build_hook',
          'standalone_protoc_used': True,
          'hatch_build_hook_invoked': False,
          'grpcio_tools_used': False,
          'grpcio_tools_version': None,
          'historical_generated_bytes_or_toolchain_reproduced': False,
          'interpretation': (
              'semantically generated from the exact locked proto and declared '
              'dependency; not evidence of byte-identical historical build output'
          ),
      },
  }


def _tree_digest(paths: Sequence[Path], root: Path) -> str:
  digest = hashlib.sha256()
  for path in sorted(paths):
    digest.update(str(path.relative_to(root)).encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(_sha256(path)))
  return digest.hexdigest()


def load_locked_identity_records() -> tuple[Mapping[str, Any], ...]:
  analysis = _LOCKED_RESULT_DIR / 'PHASE_R_ANALYSIS.json'
  if _sha256(analysis) != LOCKED_PHASE_R_ANALYSIS_SHA256:
    raise ValueError('Locked Phase-R analysis hash mismatch.')
  paths = sorted((_LOCKED_RESULT_DIR / 'identity').glob('*.json'))
  if len(paths) != EXPECTED_CASES:
    raise ValueError(f'Expected {EXPECTED_CASES} locked identities.')
  if _tree_digest(paths, _LOCKED_RESULT_DIR) != LOCKED_PHASE_R_TREE_SHA256:
    raise ValueError('Locked Phase-R identity tree hash mismatch.')
  records = tuple(json.loads(path.read_text(encoding='utf-8')) for path in paths)
  variant_ids = []
  for order, record in enumerate(records):
    if record.get('status') != 'complete':
      raise ValueError(f'Locked identity {order} is incomplete.')
    configuration = record.get('configuration', {})
    case = configuration.get('case', {})
    if case.get('order') != order:
      raise ValueError(f'Locked case order changed at {order}.')
    if case.get('gene') not in ('BRAF', 'SLC25A48'):
      raise ValueError('Locked identity is not in the development partition.')
    variant_ids.append(case.get('variant_id'))
    if tuple(record.get('checks', {}).get('target_means', {})) != TRACE_BATCH_ROLES:
      raise ValueError(f'Locked target role order changed at {order}.')
  if len(set(variant_ids)) != EXPECTED_CASES:
    raise ValueError('Locked identity variant IDs are not unique.')
  return records


def validate_locked_sources(
    locked_root: Path, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
  """Validates the serialized five-file compilation map and helper scripts."""
  expected = records[0]['configuration']['code']
  if expected['git_head'] != LOCK_COMMIT:
    raise ValueError('Locked identity does not name the expected source commit.')
  expected_files = expected['file_sha256']
  if len(expected_files) != 5:
    raise ValueError('Locked identity does not contain exactly five source hashes.')
  for record in records[1:]:
    if record['configuration']['code'] != expected:
      raise ValueError('Locked identities disagree on source provenance.')
  observed_files = {}
  for relative, expected_hash in expected_files.items():
    path = locked_root / relative
    observed = _sha256(path)
    if observed != expected_hash:
      raise ValueError(f'Locked source hash mismatch for {relative}.')
    observed_files[relative] = observed

  configuration = records[0]['configuration']
  helpers = {
      'experiments/interpretability/opensplice/run_phase_r_v3.py':
          configuration['phase_runner_sha256'],
      'experiments/interpretability/opensplice/run_inference_trace.py':
          configuration['v2_runner_sha256'],
  }
  for relative, expected_hash in helpers.items():
    observed = _sha256(locked_root / relative)
    if observed != expected_hash:
      raise ValueError(f'Locked helper hash mismatch for {relative}.')
  return {
      'locked_five_file_sha256': observed_files,
      'locked_helper_sha256': helpers,
  }


def _is_within(path: Path, root: Path) -> bool:
  try:
    path.resolve().relative_to(root.resolve())
    return True
  except ValueError:
    return False


def _prepare_module_search(locked_root: Path) -> tuple[str, ...]:
  """Removes current-checkout paths and prepends the historical worktree."""
  locked_src = locked_root / 'src'
  locked_experiments = (
      locked_root / 'experiments' / 'interpretability' / 'opensplice'
  )
  retained = []
  for item in sys.path:
    candidate = Path(item or os.getcwd())
    if _is_within(candidate, _CURRENT_REPO):
      continue
    retained.append(item)
  sys.path[:] = [str(locked_experiments), str(locked_src), *retained]
  return tuple(sys.path)


def load_locked_modules(locked_root: Path) -> LockedModules:
  """Imports compilation-path modules only after exact-source preflight."""
  already_loaded = sorted(name for name in EXPECTED_MODULES if name in sys.modules)
  if already_loaded:
    raise RuntimeError(
        'Exact-source diagnostic requires a fresh Python process; modules are '
        f'already loaded: {already_loaded}.'
    )
  _prepare_module_search(locked_root)
  jax = importlib.import_module('jax')
  jnp = importlib.import_module('jax.numpy')
  np = importlib.import_module('numpy')
  public_dna_model = importlib.import_module('alphagenome.models.dna_model')
  dna_model = importlib.import_module('alphagenome_research.model.dna_model')
  interpretability = importlib.import_module(
      'alphagenome_research.model.interpretability'
  )
  v2 = importlib.import_module('run_inference_trace')
  route_v3 = importlib.import_module('run_route_census_v3')
  phase_r_v3 = importlib.import_module('run_phase_r_v3')
  imported = {name: sys.modules[name] for name in EXPECTED_MODULES}
  for name, module in imported.items():
    path = Path(module.__file__).resolve()
    if not _is_within(path, locked_root):
      raise RuntimeError(f'{name} resolved outside locked worktree: {path}.')
  return LockedModules(
      jax=jax,
      jnp=jnp,
      np=np,
      public_dna_model=public_dna_model,
      dna_model=dna_model,
      interpretability=interpretability,
      v2=v2,
      route_v3=route_v3,
      phase_r_v3=phase_r_v3,
      imported_modules=imported,
  )


def _module_file_record(
    name: str,
    module: Any,
    *,
    allowed_root: Path,
    provenance_class: str,
) -> dict[str, Any]:
  file_name = getattr(module, '__file__', None)
  if file_name is not None:
    path = Path(file_name).resolve()
    if not _is_within(path, allowed_root):
      raise RuntimeError(
          f'{name} resolved outside declared {provenance_class} root: {path}.'
      )
    return {
        'kind': 'filesystem_module',
        'provenance_class': provenance_class,
        'path': str(path),
        'relative_path': str(path.relative_to(allowed_root)),
        'sha256': _sha256(path),
        'size_bytes': path.stat().st_size,
    }
  namespace_paths = tuple(
      str(Path(path).resolve())
      for path in getattr(module, '__path__', ())
  )
  if not namespace_paths or any(
      not _is_within(Path(path), allowed_root) for path in namespace_paths
  ):
    raise RuntimeError(
        f'{name} has no auditable file or allowed namespace path.'
    )
  return {
      'kind': 'namespace_package',
      'provenance_class': provenance_class,
      'namespace_paths': namespace_paths,
  }


def _module_tree_digest(records: Mapping[str, Mapping[str, Any]]) -> str:
  digest = hashlib.sha256()
  for name, record in sorted(records.items()):
    digest.update(name.encode('utf-8'))
    digest.update(b'\0')
    digest.update(_json_bytes(record))
  return digest.hexdigest()


def imported_module_provenance(
    modules: LockedModules, locked_root: Path
) -> dict[str, Any]:
  """Enumerates every loaded research, AlphaGenome, and helper module."""
  del modules  # The complete sys.modules inventory is the source of truth.
  locked_experiments = (
      locked_root / 'experiments' / 'interpretability' / 'opensplice'
  )
  alphagenome_root = Path(os.environ['ALPHAGENOME_PROTO_ROOT']).resolve()
  records = {}
  for name, module in sorted(sys.modules.items()):
    if module is None:
      continue
    file_name = getattr(module, '__file__', None)
    file_path = Path(file_name).resolve() if file_name else None
    if name == 'alphagenome_research' or name.startswith('alphagenome_research.'):
      records[name] = _module_file_record(
          name,
          module,
          allowed_root=locked_root,
          provenance_class='exact_fd4dc69_worktree',
      )
    elif name == 'alphagenome' or name.startswith('alphagenome.'):
      records[name] = _module_file_record(
          name,
          module,
          allowed_root=alphagenome_root,
          provenance_class='declared_third_party_alphagenome_source',
      )
    elif file_path is not None and _is_within(file_path, locked_experiments):
      records[name] = _module_file_record(
          name,
          module,
          allowed_root=locked_root,
          provenance_class='exact_fd4dc69_opensplice_helper',
      )
  required = set(EXPECTED_MODULES) | {
      'alphagenome.protos.dna_model_pb2',
      'alphagenome_research.protos.calibration_scores_pb2',
  }
  missing = sorted(required - set(records))
  if missing:
    raise RuntimeError(f'Required transitive modules were not loaded: {missing}.')

  dependency_proto = alphagenome_root / 'alphagenome/protos/dna_model.proto'
  dependency_pb2 = Path(
      sys.modules['alphagenome.protos.dna_model_pb2'].__file__
  ).resolve()
  alphagenome_repo = alphagenome_root.parent
  package_git = {
      'repository_root': str(alphagenome_repo),
      'head': _run_text(('git', '-C', str(alphagenome_repo), 'rev-parse', 'HEAD')),
      'tracked_status': _run_text((
          'git', '-C', str(alphagenome_repo), 'status', '--porcelain',
          '--untracked-files=no'
      )),
  }
  package_git['tracked_clean'] = not package_git['tracked_status']
  return {
      'module_count': len(records),
      'modules': records,
      'module_tree_sha256': _module_tree_digest(records),
      'declared_roots': {
          'exact_worktree': str(locked_root),
          'exact_opensplice_helpers': str(locked_experiments),
          'third_party_alphagenome_source': str(alphagenome_root),
      },
      'alphagenome_package_source': package_git,
      'alphagenome_dependency_binding': {
          'dna_model_proto_path': str(dependency_proto),
          'dna_model_proto_sha256': _sha256(dependency_proto),
          'dna_model_pb2_path': str(dependency_pb2),
          'dna_model_pb2_sha256': _sha256(dependency_pb2),
          'dna_model_pb2_module_record': records[
              'alphagenome.protos.dna_model_pb2'
          ],
      },
  }


def mixed_precision_provenance(locked_root: Path) -> dict[str, Any]:
  source = locked_root / 'src/alphagenome_research/model/dna_model.py'
  text = source.read_text(encoding='utf-8')
  expression = f"jmp.get_policy('{MIXED_PRECISION_POLICY}')"
  count = text.count(expression)
  if count < 1:
    raise ValueError('Exact fd4dc69 mixed-precision policy source changed.')
  return {
      'policy': MIXED_PRECISION_POLICY,
      'constructor_expression': expression,
      'source_path': str(source),
      'source_sha256': _sha256(source),
      'source_occurrences': count,
      'scope': 'hk.mixed_precision policy pushed for model.AlphaGenome',
  }


def _json_normalized(value: Any) -> Any:
  return json.loads(json.dumps(value, sort_keys=True, separators=(',', ':')))


def validate_static_cases(
    modules: LockedModules,
    records: Sequence[Mapping[str, Any]],
) -> tuple[Any, ...]:
  """Checks case, interval, and position sets before checkpoint/model load."""
  cases = modules.route_v3.load_development_cases()
  if len(cases) != EXPECTED_CASES:
    raise ValueError('Exact-source helper did not return 20 development cases.')
  for case, record in zip(cases, records, strict=True):
    locked = record['configuration']
    current_case = modules.v2._case_record(case)  # pylint: disable=protected-access
    if _json_normalized(current_case) != _json_normalized(locked['case']):
      raise ValueError(f'Case linkage mismatch for {case.variant_id}.')
    interval = modules.v2.centered_interval(case, modules.route_v3.CONTEXT_BP)
    current_interval = {
        'chromosome': interval.chromosome,
        'start_0based': interval.start,
        'end_0based_exclusive': interval.end,
    }
    if _json_normalized(current_interval) != _json_normalized(locked['interval']):
      raise ValueError(f'Interval linkage mismatch for {case.variant_id}.')
    positions = [
        dataclasses.asdict(item)
        for item in modules.v2.trace_position_sets(case, interval)
    ]
    if _json_normalized(positions) != _json_normalized(
        locked['resolved_position_sets']
    ):
      raise ValueError(f'Position-set linkage mismatch for {case.variant_id}.')
  return tuple(cases)


def validate_live_case_linkage(
    modules: LockedModules,
    case: Any,
    interval: Any,
    resolved_target: Any,
    sequence_sha256: Mapping[str, str],
    record: Mapping[str, Any],
) -> dict[str, Any]:
  locked = record['configuration']
  current_target = {
      'endpoints': [
          dataclasses.asdict(endpoint) for endpoint in resolved_target.endpoints
      ],
      'padding_track_index': resolved_target.padding_track_index,
  }
  checks = {
      'case': _json_normalized(
          modules.v2._case_record(case)  # pylint: disable=protected-access
      ) ==
              _json_normalized(locked['case']),
      'interval': _json_normalized({
          'chromosome': interval.chromosome,
          'start_0based': interval.start,
          'end_0based_exclusive': interval.end,
      }) == _json_normalized(locked['interval']),
      'sequence_sha256': _json_normalized(dict(sequence_sha256)) ==
                         _json_normalized(locked['sequence_sha256']),
      'canonical_target': _json_normalized(current_target) ==
                          _json_normalized(locked['canonical_target']),
      'resolved_position_sets': _json_normalized([
          dataclasses.asdict(item)
          for item in modules.v2.trace_position_sets(case, interval)
      ]) == _json_normalized(locked['resolved_position_sets']),
  }
  if not all(checks.values()):
    failed = [name for name, passed in checks.items() if not passed]
    raise ValueError(
        f'Exact input linkage failed for {case.variant_id}: {failed}.'
    )
  return {'passed': True, 'fields_exact': checks}


def compare_locked_target(
    current: Sequence[float],
    locked: Sequence[float],
    *,
    is_effect: bool,
    experimental_delta_logit: float,
) -> dict[str, Any]:
  if len(current) != 6 or len(locked) != 6:
    raise ValueError('Target comparison requires six frozen batch roles.')
  current = tuple(float(value) for value in current)
  locked = tuple(float(value) for value in locked)
  if any(not __import__('math').isfinite(value) for value in current + locked):
    raise ValueError('Target comparison contains non-finite values.')
  signed = tuple(value - expected for value, expected in zip(
      current, locked, strict=True
  ))
  absolute = tuple(abs(value) for value in signed)
  tolerance_pass = all(value <= TARGET_TOLERANCE for value in absolute)
  current_delta = current[1] - current[0]
  locked_delta = locked[1] - locked[0]
  direction = None
  if is_effect:
    experimental_sign = 1 if experimental_delta_logit > 0 else -1
    direction = {
        'experimental_delta_logit': experimental_delta_logit,
        'experimental_sign': experimental_sign,
        'minimum_absolute_margin': EFFECT_THRESHOLD,
        'current_magnitude_passes': abs(current_delta) >= EFFECT_THRESHOLD,
        'locked_magnitude_passes': abs(locked_delta) >= EFFECT_THRESHOLD,
        'current_direction_passes': (
            (1 if current_delta > 0 else -1 if current_delta < 0 else 0)
            == experimental_sign
        ),
        'locked_direction_passes': (
            (1 if locked_delta > 0 else -1 if locked_delta < 0 else 0)
            == experimental_sign
        ),
    }
  direction_pass = direction is None or all(
      value for key, value in direction.items()
      if key not in ('experimental_delta_logit', 'experimental_sign',
                     'minimum_absolute_margin')
  )
  return {
      'passed': tolerance_pass and direction_pass,
      'inclusive_tolerance': TARGET_TOLERANCE,
      'locked_target_means': dict(zip(TRACE_BATCH_ROLES, locked, strict=True)),
      'current_target_means': dict(zip(TRACE_BATCH_ROLES, current, strict=True)),
      'signed_current_minus_locked': dict(zip(
          TRACE_BATCH_ROLES, signed, strict=True
      )),
      'absolute_current_minus_locked': dict(zip(
          TRACE_BATCH_ROLES, absolute, strict=True
      )),
      'all_six_within_tolerance': tolerance_pass,
      'maximum_absolute_difference': max(absolute),
      'current_alt_minus_ref_logit_margin': current_delta,
      'locked_alt_minus_ref_logit_margin': locked_delta,
      'effect_denominator_and_direction_gate': direction,
  }


def _safe_command(command: Sequence[str]) -> dict[str, Any]:
  try:
    result = subprocess.run(
        tuple(command), check=False, capture_output=True, text=True, timeout=10
    )
  except (OSError, subprocess.SubprocessError) as error:
    return {'status': 'unavailable', 'reason': str(error)}
  return {
      'status': 'available' if result.returncode == 0 else 'error',
      'returncode': result.returncode,
      'stdout': result.stdout.strip(),
      'stderr': result.stderr.strip(),
  }


def _package_version(name: str) -> str | None:
  try:
    return importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:
    return None


def _compiler_environment() -> dict[str, Any]:
  prefixes = ('JAX_', 'CUDA_', 'NVIDIA_', 'ALPHAGENOME_PROTOC_')
  explicit = (
      'XLA_FLAGS', 'TF_XLA_FLAGS', 'LD_LIBRARY_PATH', 'PATH',
      'XDG_CACHE_HOME', 'HOME',
  )
  names = sorted(
      set(explicit)
      | {name for name in os.environ if name.startswith(prefixes)}
  )
  output = {}
  for name in names:
    value = os.environ.get(name)
    if any(token in name.upper() for token in ('SECRET', 'TOKEN', 'PASSWORD')):
      value = '<redacted>'
    output[name] = {'state': 'unset' if value is None else 'set', 'value': value}
  return output


def runtime_provenance(modules: LockedModules) -> dict[str, Any]:
  jax = modules.jax
  device_records = []
  try:
    for order, device in enumerate(jax.devices()):
      device_records.append({
          'visible_order': order,
          'id': getattr(device, 'id', None),
          'platform': getattr(device, 'platform', None),
          'device_kind': getattr(device, 'device_kind', None),
          'client_platform': getattr(getattr(device, 'client', None),
                                     'platform', None),
      })
  except Exception as error:  # provenance must not hide the attempted runtime
    device_records = [{'status': 'unavailable', 'reason': str(error)}]
  precision = None
  try:
    precision = jax.config.read('jax_default_matmul_precision')
  except Exception as error:
    precision = f'unavailable: {error}'
  return {
      'captured_at_unix_s': time.time(),
      'hostname': socket.gethostname(),
      'pid': os.getpid(),
      'platform': platform.platform(),
      'kernel': platform.release(),
      'python': sys.version,
      'packages': {
          name: _package_version(name) for name in (
              'numpy', 'jax', 'jaxlib', 'jax-cuda12-plugin',
              'jax-cuda12-pjrt', 'nvidia-cuda-runtime-cu12',
              'nvidia-cudnn-cu12', 'nvidia-cublas-cu12', 'jmp', 'dm-haiku',
          )
      },
      'jax_default_matmul_precision': precision,
      'mixed_precision_policy_runtime': runtime_mixed_precision_policy(),
      'devices': device_records,
      'nvidia_smi': _safe_command((
          'nvidia-smi',
          '--query-gpu=index,name,uuid,compute_cap,vbios_version,driver_version',
          '--format=csv,noheader',
      )),
      'nvcc': _safe_command(('nvcc', '--version')),
      'protoc': _safe_command((
          os.environ.get('ALPHAGENOME_PROTOC_BIN', 'protoc'), '--version'
      )),
      'compiler_environment': _compiler_environment(),
  }


def runtime_mixed_precision_policy() -> dict[str, Any]:
  jmp = importlib.import_module('jmp')
  jnp = importlib.import_module('jax.numpy')
  policy = jmp.get_policy(MIXED_PRECISION_POLICY)
  _assert_mixed_precision_dtypes(policy, jnp)
  return {
      'input_literal': MIXED_PRECISION_POLICY,
      'repr': repr(policy),
      'str': str(policy),
      'param_dtype': str(policy.param_dtype),
      'compute_dtype': str(policy.compute_dtype),
      'output_dtype': str(policy.output_dtype),
      'jmp_version': _package_version('jmp'),
      'dm_haiku_version': _package_version('dm-haiku'),
      'expected_dtypes_asserted_before_lowering': True,
  }


def _assert_mixed_precision_dtypes(policy: Any, jnp: Any) -> None:
  observed = (
      policy.param_dtype,
      policy.compute_dtype,
      policy.output_dtype,
  )
  expected = (jnp.float32, jnp.bfloat16, jnp.bfloat16)
  if observed != expected:
    raise ValueError(
        'Runtime mixed-precision policy differs from frozen '
        'float32/bfloat16/bfloat16 dtypes: '
        f'observed={observed}, expected={expected}.'
    )


def _xla_flag_value(name: str) -> str | None:
  try:
    tokens = shlex.split(os.environ.get('XLA_FLAGS', ''))
  except ValueError:
    return None
  prefix = f'--{name}='
  for token in tokens:
    if token.startswith(prefix):
      return token[len(prefix):]
  return None


def autotune_provenance() -> dict[str, Any]:
  """Hashes preconfigured XLA autotune files without selecting an algorithm."""
  entries = {}
  for flag in (
      'xla_gpu_dump_autotune_results_to',
      'xla_gpu_load_autotune_results_from',
  ):
    configured = _xla_flag_value(flag)
    if configured is None:
      entries[flag] = {
          'status': 'unavailable',
          'reason': (
              'flag was unset before the one-shot attempt; JAX 0.11 exposes '
              'no out-of-band Python API for the selected GPU autotune result'
          ),
      }
      continue
    path = Path(configured).expanduser().resolve()
    entries[flag] = {
        'configured_path': str(path),
        'status': 'available' if path.is_file() else 'unavailable',
        'sha256': _sha256(path) if path.is_file() else None,
        'size_bytes': path.stat().st_size if path.is_file() else None,
        'reason': None if path.is_file() else 'configured artifact was not created',
    }
  return entries


def _compiler_ir_text(lowered: Any, dialect: str) -> str:
  ir = lowered.compiler_ir(dialect=dialect)
  if dialect == 'hlo' and hasattr(ir, 'as_hlo_text'):
    return ir.as_hlo_text()
  return str(ir)


def compile_with_provenance(
    jitted_apply: Any, args: Sequence[Any]
) -> tuple[Any, dict[str, Any], float]:
  """Lowers the unchanged callable and captures IR out of band."""
  started = time.perf_counter()
  lowered = jitted_apply.lower(*args)
  provenance = {}
  for dialect, filename in (
      ('stablehlo', 'pre_optimization.stablehlo.mlir'),
      ('hlo', 'pre_backend.hlo.txt'),
  ):
    try:
      text = _compiler_ir_text(lowered, dialect)
      path = OUTPUT_DIR / 'compiler' / filename
      provenance[dialect] = {
          'status': 'available',
          'path': str(path),
          'sha256': _write_new_text(path, text),
          'size_bytes': len(text.encode('utf-8')),
      }
    except Exception as error:  # best-effort provenance only
      provenance[dialect] = {
          'status': 'unavailable',
          'reason': f'{type(error).__name__}: {error}',
      }
  compiled = lowered.compile()
  try:
    optimized = compiled.as_text()
    path = OUTPUT_DIR / 'compiler' / 'compiled_executable.hlo.txt'
    provenance['compiled_hlo'] = {
        'status': 'available',
        'path': str(path),
        'sha256': _write_new_text(path, optimized),
        'size_bytes': len(optimized.encode('utf-8')),
    }
  except Exception as error:
    provenance['compiled_hlo'] = {
        'status': 'unavailable',
        'reason': f'{type(error).__name__}: {error}',
    }
  provenance['autotune'] = autotune_provenance()
  seconds = time.perf_counter() - started
  _write_new(OUTPUT_DIR / 'compiler' / 'COMPILER_PROVENANCE.json', provenance)
  return compiled, provenance, seconds


def _timed_apply(modules: LockedModules, apply_fn: Any, args: Sequence[Any]):
  started = time.perf_counter()
  output = apply_fn(*args)
  modules.jax.block_until_ready(output)
  return output, time.perf_counter() - started


def _case_path(case: Any) -> Path:
  slug = ''.join(character if character.isalnum() else '_' for character in
                 case.variant_id).strip('_')
  return OUTPUT_DIR / 'raw_identity' / f'{case.order:03d}_{slug}.json'


def _locked_target(record: Mapping[str, Any]) -> tuple[float, ...]:
  values = record['checks']['target_means']
  return tuple(float(values[role]) for role in TRACE_BATCH_ROLES)


def _error_record(error: BaseException, stage: str) -> dict[str, Any]:
  return {
      'stage': stage,
      'exception_type': type(error).__name__,
      'message': str(error),
      'traceback': ''.join(traceback.format_exception(error)),
  }


def run_identity_unit(
    modules: LockedModules,
    model_instance: Any,
    compiled_apply: Any,
    case: Any,
    record: Mapping[str, Any],
    *,
    compile_seconds: float,
) -> tuple[dict[str, Any], bool]:
  """Runs one two-call identity unit and returns artifact plus terminal flag."""
  base = {
      'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION,
      'case': modules.v2._case_record(case),  # pylint: disable=protected-access
      'locked_identity_fingerprint': record['fingerprint'],
      'locked_target_means': dict(zip(
          TRACE_BATCH_ROLES, _locked_target(record), strict=True
      )),
      'input_linkage': None,
      'current_target_means': None,
      'comparison': None,
      'within_executable_checks': None,
      'seconds': {'compile': compile_seconds, 'first_run': None, 'warm_repeat': None},
      'created_at_unix_s': time.time(),
  }
  try:
    interval = modules.v2.centered_interval(case, modules.route_v3.CONTEXT_BP)
    positions = modules.v2.trace_position_sets(case, interval)
    selection = modules.phase_r_v3.phase_r_trace_selection(positions)
    metadata = model_instance._metadata[  # pylint: disable=protected-access
        modules.public_dna_model.Organism.HOMO_SAPIENS
    ].splice_sites
    target, resolved = modules.route_v3.target_selection(metadata, case, interval)
    dna_batch, sequence_sha = modules.route_v3._build_six_row_batch(  # pylint: disable=protected-access
        model_instance, case, interval
    )
    base['input_linkage'] = validate_live_case_linkage(
        modules, case, interval, resolved, sequence_sha, record
    )
    interventions = modules.phase_r_v3.group_interventions(selection, None)
    organism = modules.jnp.zeros((6,), modules.jnp.int32)
    args = (
        model_instance._params,  # pylint: disable=protected-access
        model_instance._state,  # pylint: disable=protected-access
        dna_batch,
        organism,
        selection,
        interventions,
        target,
    )
  except Exception as error:
    base.update({
        'status': 'validation_failure',
        'failure': _error_record(error, 'exact_input_linkage'),
    })
    return base, False

  first = None
  try:
    first, first_seconds = _timed_apply(modules, compiled_apply, args)
    base['seconds']['first_run'] = first_seconds
    second, second_seconds = _timed_apply(modules, compiled_apply, args)
    base['seconds']['warm_repeat'] = second_seconds
  except Exception as error:
    if first is not None:
      try:
        values = modules.route_v3._target_values(first[0])  # pylint: disable=protected-access
        base['current_target_means'] = dict(zip(
            TRACE_BATCH_ROLES, [float(x) for x in values], strict=True
        ))
      except Exception:
        pass
    base.update({
        'status': 'runtime_failure',
        'failure': _error_record(error, 'all_false_apply'),
    })
    return base, True

  try:
    first_values = modules.route_v3._target_values(first[0])  # pylint: disable=protected-access
    current = tuple(float(value) for value in first_values)
  except Exception as error:
    base.update({
        'status': 'validation_failure',
        'failure': _error_record(error, 'target_materialization'),
    })
    return base, False
  base['current_target_means'] = dict(zip(TRACE_BATCH_ROLES, current, strict=True))
  comparison = compare_locked_target(
      current,
      _locked_target(record),
      is_effect=case.is_effect,
      experimental_delta_logit=case.delta_logit,
  )
  base['comparison'] = comparison
  try:
    base['within_executable_checks'] = modules.route_v3.validate_identity_audit(
        first[0], first[1], second[0], second[1]
    )
  except Exception as error:
    base.update({
        'status': 'validation_failure',
        'failure': _error_record(error, 'within_executable_identity_audit'),
    })
    return base, False
  if not comparison['passed']:
    base.update({
        'status': 'numerical_failure',
        'failure': {
            'stage': 'locked_identity_comparison',
            'exception_type': 'FrozenDecisionRuleFailure',
            'message': (
                'At least one unchanged tolerance or effect direction rule failed.'
            ),
        },
    })
  else:
    base['status'] = 'pass'
    base['failure'] = None
  return base, False


def _artifact_tree_digest(paths: Sequence[Path]) -> str:
  digest = hashlib.sha256()
  for path in sorted(paths):
    digest.update(str(path.relative_to(OUTPUT_DIR)).encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(_sha256(path)))
  return digest.hexdigest()


def _summary(
    artifacts: Sequence[Path], statuses: Sequence[Mapping[str, Any]],
    *, terminal: bool
) -> dict[str, Any]:
  counts = {name: 0 for name in (
      'pass', 'numerical_failure', 'validation_failure', 'runtime_failure'
  )}
  for status in statuses:
    counts[status['status']] += 1
  hashes = {str(path.relative_to(OUTPUT_DIR)): _sha256(path) for path in artifacts}
  return {
      'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION,
      'status': 'pass' if len(statuses) == EXPECTED_CASES and not terminal and
                counts['pass'] == EXPECTED_CASES else 'failed',
      'cohort_passed': len(statuses) == EXPECTED_CASES and not terminal and
                       counts['pass'] == EXPECTED_CASES,
      'terminal_failure': terminal,
      'attempted_count': len(statuses),
      'expected_count': EXPECTED_CASES,
      'counts': counts,
      'ordered_case_statuses': list(statuses),
      'artifact_sha256': hashes,
      'artifact_tree_sha256': _artifact_tree_digest(artifacts),
      'active_intervention_calls': 0,
      'confirmation_model_calls': 0,
      'unconditional_stop_after_identity_cohort': True,
      'created_at_unix_s': time.time(),
  }


def build_dry_run_plan(
    locked_root: Path,
    worktree: Mapping[str, Any],
    source: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
  return {
      'script_version': SCRIPT_VERSION,
      'attempt_id': ATTEMPT_ID,
      'dry_run': True,
      'one_shot_output_dir': str(OUTPUT_DIR),
      'refuses_any_preexisting_attempt_directory': True,
      'resume_after_attempt_start': False,
      'locked_worktree': dict(worktree),
      'locked_source': dict(source),
      'locked_checkout_argument': str(locked_root),
      'protocol_path': str(_PROTOCOL_PATH),
      'protocol_sha256': PROTOCOL_SHA256,
      'identity_units': len(records),
      'all_false_apply_calls': len(records) * 2,
      'active_intervention_calls': 0,
      'development_genes': ('BRAF', 'SLC25A48'),
      'variant_ids': tuple(
          record['configuration']['case']['variant_id'] for record in records
      ),
      'ordinary_case_failures_are_persisted_and_collection_continues': True,
      'runtime_failure_is_terminal': True,
      'confirmation_model_calls': 0,
      'stablehlo_hlo_capture': 'best_effort_out_of_band_jax_lowering',
      'autotune_capture': (
          'hash preconfigured XLA dump/load artifact; otherwise record '
          'unavailable because JAX exposes no out-of-band selected-result API'
      ),
  }


def _ensure_fresh_attempt(start_record: Mapping[str, Any]) -> None:
  """Creates the sole attempt; any existing directory permanently refuses."""
  try:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
  except FileExistsError as error:
    raise FileExistsError(
        f'v3.1.1 is one-shot and cannot resume or retry: {OUTPUT_DIR} exists.'
    ) from error
  _write_new(START_PATH, start_record)


def main() -> None:
  args = _parse_args()
  if _sha256(_PROTOCOL_PATH) != PROTOCOL_SHA256:
    raise ValueError('Frozen v3.1.1 protocol hash mismatch.')
  locked_root = args.locked_checkout.resolve()
  worktree = validate_locked_worktree(locked_root)
  protoc_text = os.environ.get('ALPHAGENOME_PROTOC_BIN')
  proto_root_text = os.environ.get('ALPHAGENOME_PROTO_ROOT')
  if not protoc_text or not proto_root_text:
    raise ValueError(
        'Wrapper must declare ALPHAGENOME_PROTOC_BIN and '
        'ALPHAGENOME_PROTO_ROOT.'
    )
  proto_build = prepare_generated_proto_binding(
      locked_root,
      protoc_bin=Path(protoc_text).resolve(),
      alphagenome_proto_root=Path(proto_root_text).resolve(),
  )
  proto_build = disclose_proto_generation(proto_build)
  generated_proto = validate_generated_proto_binding(locked_root)
  records = load_locked_identity_records()
  sources = validate_locked_sources(locked_root, records)
  modules = load_locked_modules(locked_root)
  module_sources = imported_module_provenance(modules, locked_root)
  mixed_precision = mixed_precision_provenance(locked_root)
  launch_freeze = validate_freeze_configuration(
      proto_build, module_sources, mixed_precision
  )
  cases = validate_static_cases(modules, records)
  checkpoint = modules.v2._checkpoint_path(args.checkpoint)  # pylint: disable=protected-access
  if checkpoint.name != modules.route_v3.CHECKPOINT_SNAPSHOT:
    raise ValueError('Checkpoint snapshot differs from the frozen Phase-R lock.')
  if _json_normalized(str(checkpoint)) != _json_normalized(
      records[0]['configuration']['checkpoint_path']
  ):
    raise ValueError('Checkpoint path differs from the frozen Phase-R lock.')

  if args.dry_run:
    print(json.dumps(
        {
            **build_dry_run_plan(locked_root, worktree, sources, records),
            'generated_proto_build_artifact': proto_build,
            'generated_proto_validation': generated_proto,
            'initial_transitive_import_provenance': module_sources,
            'mixed_precision': mixed_precision,
            'launch_freeze': launch_freeze,
        },
        indent=2,
        default=str,
    ))
    return

  pre_run = runtime_provenance(modules)
  start_record = {
      'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION,
      'status': 'started_append_only_one_shot',
      'started_at_unix_s': time.time(),
      'launcher_path': str(Path(__file__).resolve()),
      'launcher_sha256': _sha256(Path(__file__).resolve()),
      'protocol_path': str(_PROTOCOL_PATH),
      'protocol_sha256': PROTOCOL_SHA256,
      'worktree': worktree,
      'generated_proto_build_artifact': generated_proto,
      'generated_proto_build_provenance': proto_build,
      'launch_freeze': launch_freeze,
      'locked_sources': sources,
      'import_resolution': module_sources,
      'mixed_precision': mixed_precision,
      'checkpoint_path': str(checkpoint),
      'checkpoint_snapshot': checkpoint.name,
      'locked_phase_r_identity_tree_sha256': LOCKED_PHASE_R_TREE_SHA256,
      'runtime_before_model_creation': pre_run,
      'one_shot_policy': {
          'identity_units': EXPECTED_CASES,
          'calls_per_identity': 2,
          'active_interventions': 0,
          'resume': False,
          'retry': False,
          'overwrite': False,
      },
  }
  _ensure_fresh_attempt(start_record)

  statuses = []
  artifact_paths = []
  terminal = False
  compiled_apply = None
  compile_seconds = 0.0
  try:
    model_instance = modules.dna_model.create(
        checkpoint,
        model_settings=modules.dna_model.ModelSettings(
            attention_backend=modules.route_v3.ATTENTION_BACKEND
        ),
    )
    post_model_imports = imported_module_provenance(modules, locked_root)
    _write_new(
        OUTPUT_DIR / 'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
        {
            'attempt_id': ATTEMPT_ID,
            'stage': 'post_model_precompile',
            'provenance': post_model_imports,
            'created_at_unix_s': time.time(),
        },
    )
    raw_apply = (
        modules.dna_model
        .create_splice_classification_logit_margin_route_census_apply(
            model_instance._metadata,  # pylint: disable=protected-access
            attention_backend=modules.route_v3.ATTENTION_BACKEND,
        )
    )
    jitted_apply = modules.jax.jit(raw_apply)

    for index, (case, record) in enumerate(zip(cases, records, strict=True)):
      if compiled_apply is None:
        interval = modules.v2.centered_interval(case, modules.route_v3.CONTEXT_BP)
        positions = modules.v2.trace_position_sets(case, interval)
        selection = modules.phase_r_v3.phase_r_trace_selection(positions)
        metadata = model_instance._metadata[  # pylint: disable=protected-access
            modules.public_dna_model.Organism.HOMO_SAPIENS
        ].splice_sites
        target, resolved = modules.route_v3.target_selection(metadata, case, interval)
        dna_batch, sequence_sha = modules.route_v3._build_six_row_batch(  # pylint: disable=protected-access
            model_instance, case, interval
        )
        validate_live_case_linkage(
            modules, case, interval, resolved, sequence_sha, record
        )
        interventions = modules.phase_r_v3.group_interventions(selection, None)
        compile_args = (
            model_instance._params,  # pylint: disable=protected-access
            model_instance._state,  # pylint: disable=protected-access
            dna_batch,
            modules.jnp.zeros((6,), modules.jnp.int32),
            selection,
            interventions,
            target,
        )
        compiled_apply, _, compile_seconds = compile_with_provenance(
            jitted_apply, compile_args
        )
      artifact, case_terminal = run_identity_unit(
          modules,
          model_instance,
          compiled_apply,
          case,
          record,
          compile_seconds=compile_seconds if index == 0 else 0.0,
      )
      path = _case_path(case)
      _write_new(path, artifact)
      artifact_paths.append(path)
      statuses.append({
          'order': case.order,
          'variant_id': case.variant_id,
          'gene': case.gene,
          'status': artifact['status'],
          'artifact': str(path.relative_to(OUTPUT_DIR)),
      })
      if case_terminal:
        terminal = True
        break
  except Exception as error:
    terminal = True
    failure_path = OUTPUT_DIR / 'TERMINAL_FAILURE.json'
    _write_new(failure_path, {
        'attempt_id': ATTEMPT_ID,
        'status': 'runtime_failure',
        'failure': _error_record(error, 'model_load_or_compile'),
        'created_at_unix_s': time.time(),
    })
    artifact_paths.append(failure_path)

  completion_path = OUTPUT_DIR / 'COMPLETION_PROVENANCE.json'
  _write_new(completion_path, {
      'attempt_id': ATTEMPT_ID,
      'runtime_after_cohort': runtime_provenance(modules),
      'transitive_import_provenance_at_completion': (
          imported_module_provenance(modules, locked_root)
      ),
      'autotune': autotune_provenance(),
      'created_at_unix_s': time.time(),
  })
  artifact_paths.append(completion_path)
  # The summary binds every durable artifact published before itself, including
  # the start record and best-effort compiler material, not only case JSON.
  all_published = sorted(
      path for path in OUTPUT_DIR.rglob('*')
      if path.is_file() and path.name != 'SUMMARY.json'
  )
  summary = _summary(all_published, statuses, terminal=terminal)
  summary_path = OUTPUT_DIR / 'SUMMARY.json'
  _write_new(summary_path, summary)
  print(summary_path.resolve())
  if not summary['cohort_passed']:
    raise SystemExit(2)


if __name__ == '__main__':
  main()
