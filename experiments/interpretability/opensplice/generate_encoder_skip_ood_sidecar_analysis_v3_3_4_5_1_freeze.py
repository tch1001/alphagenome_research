#!/usr/bin/env python3
"""Generate only the acyclic v3.3.4.5.1 CPU-analysis freeze.

Run this once from the clean source-authority commit.  The generated freeze is
deliberately excluded from its own 137-row inventory and must be committed by
a later freeze-only launch commit.  This module imports no experiment, model,
JAX, runner, analyzer, or publication-helper code.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Iterable, Mapping


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MODEL_HEAD = '0da8f47ea6e576a72a1cda204ce868ef79cc2ce5'
DOCS_HEAD = '564a01dc2981d57c8f8298f3efca5b22fcb381e0'
AMENDMENT_SHA256 = (
    '16af8ccb65f3e08739c3792c5c9ab3affcb19a3ca9993904260729a898afd5c4'
)
EMPTY_SHA256 = hashlib.sha256(b'').hexdigest()
SCHEMA_VERSION = 'v3.3.4.5.1-analysis-freeze-v1'
ANALYSIS_VERSION = 'v3.3.4.5.1-structural-analyzer-v1'
ATTEMPT_ID = 'v3.3.4.5.1-development-ood-sidecar-structural-analysis'
ACK = '--acknowledge-structural-only-v3-3-4-5-1'
AMENDMENT = (
    HERE / 'v3_wider_mechanism'
    / 'encoder_skip_ood_sidecar_analysis_amendment_v3_3_4_5_1.md'
)
MODEL_FREEZE = HERE / 'encoder_skip_ood_sidecar_v3_3_4_5_freeze.json'
FREEZE = HERE / 'encoder_skip_ood_sidecar_analysis_v3_3_4_5_1_freeze.json'
RUN = HERE / 'results/v3_3_4_5_development_ood_sidecar_one_shot'
PREFLIGHT = HERE / 'results/v3_3_4_5_device_preflight'
EXTERNAL_CACHE = HERE / 'results/v3_3_4_5_preflight_kernel_cache'
MODEL_CACHE = HERE / 'results/v3_3_4_5_model_kernel_cache'
PRIOR_CACHE = HERE / 'results/v3_3_3_model_kernel_cache'
ATTEMPT = (
    HERE
    / 'results/v3_3_4_5_development_ood_sidecar_analysis_v3_3_4_5_1_attempt'
)
ANALYSIS = (
    HERE
    / 'results/v3_3_4_5_development_ood_sidecar_analysis_v3_3_4_5_1'
)
NEW_FILES = (
    HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_4_5_1.py',
    HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_4_5_1_test.py',
    HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_4_5_1.sh',
    HERE / 'generate_encoder_skip_ood_sidecar_analysis_v3_3_4_5_1_freeze.py',
)
OLD_DESTINATIONS = (
    HERE / 'results/v3_3_4_5_development_ood_sidecar_analysis_attempt',
    HERE / 'results/v3_3_4_5_development_ood_sidecar_analysis',
    HERE / 'results/v3_3_3_development_ood_sidecar_analysis_attempt',
    HERE / 'results/v3_3_3_development_ood_sidecar_analysis',
)
PUBLICATION_SUCCESS_KEYS = (
    'schema_version', 'method', 'root_role', 'final_relative_path',
    'temp_basename', 'publication_ordinal', 'runner_pid', 'nonce_hex',
    'sha256', 'size_bytes', 'mode', 'st_dev', 'st_ino', 'st_nlink',
    'file_fsync_before_rename', 'file_fsync_after_fchmod',
    'rename_noreplace_succeeded', 'parent_fsync_succeeded',
    'post_publish_revalidation_exact',
)
PUBLICATION_FAILURE_KEYS = (
    'schema_version', 'method', 'root_role', 'artifact_role',
    'final_relative_path', 'temp_relative_path', 'publication_ordinal',
    'runner_pid', 'failure_stage', 'errno', 'error_type', 'message',
    'rename_noreplace_attempted', 'rename_noreplace_succeeded',
    'parent_fsync_attempted', 'parent_fsync_succeeded', 'temp_state',
    'final_state', 'created_at_unix_s',
)
PUBLICATION_AUDIT_KEYS = (
    'schema_version', 'method', 'successful_final_count_before_terminal',
    'successful_final_bindings_before_terminal', 'temporary_orphan_count',
    'temporary_orphan_bindings', 'durability_uncertain_final_count',
    'durability_uncertain_final_bindings', 'preexisting_entry_count',
    'preexisting_entry_states', 'publication_failure',
    'no_new_entry_failure', 'no_publication_retry',
    'no_published_final_deleted', 'no_temp_or_final_reused',
)
ENTRY_STATE_KEYS = (
    'state', 'entry_type', 'mode', 'size_bytes', 'sha256', 'st_dev',
    'st_ino', 'st_nlink',
)
FAILURE_STAGES = (
    'root_parent_open', 'root_parent_validation', 'root_final_preexistence',
    'root_mkdir', 'root_parent_fsync', 'root_revalidation', 'parent_open',
    'parent_validation', 'final_preexistence', 'temp_open', 'temp_write',
    'file_fsync_before_rename', 'fchmod', 'file_fsync_after_fchmod',
    'readback', 'rename_noreplace', 'parent_fsync',
    'post_publish_revalidation',
)
START_KEYS = {
    'status', 'schema_version', 'analysis_version', 'attempt_id',
    'acknowledgement', 'git_head', 'external_freeze_authorization',
    'freeze_binding', 'analyzer_binding', 'test_binding', 'shell_binding',
    'generator_binding', 'amendment_binding', 'run_terminal_binding',
    'source_inventory_attestation', 'immutable_input_audit',
    'consumed_analyzer_failure',
    'consumed_analyzer_failure_content_binding', 'prior_cache_audit',
    'prior_cache_audit_content_binding', 'fresh_paths', 'started_at_unix_s',
}
ANALYSIS_KEYS = {
    'status', 'decision', 'analysis_version',
    'analysis_attempt_start_binding', 'run_binding', 'preflight_binding',
    'external_cache_binding', 'model_cache_binding',
    'source_and_prior_audit', 'consumed_analyzer_failure_audit',
    'prior_cache_audit', 'compiler_and_signature_audit',
    'dispatch_journal_audit', 'raw_prefix_audit', 'control_audit',
    'terminal_audit', 'publication_audit', 'confirmation_boundary',
    'claim_boundary', 'scientific_summary_computed',
    'donor_normalization_computed', 'shapley_or_nomination_computed',
    'interaction_or_resolution_computed', 'nomination_performed',
    'combined_analysis_permitted', 'completed_at_unix_s',
}
COMPLETE_KEYS = {
    'status', 'schema_version', 'analysis_version', 'attempt_id',
    'start_binding', 'analysis_binding', 'result_binding',
    'attempt_tree_before_complete', 'output_tree_complete',
    'publication_audit', 'completed_at_unix_s',
}
FAILURE_KEYS = {
    'status', 'schema_version', 'analysis_version', 'attempt_id',
    'start_binding', 'failure', 'failure_phase', 'raw_access_reached',
    'analysis_output_state', 'attempt_output_state', 'publication_audit',
    'old_destinations_absent', 'failed_at_unix_s',
}
OUTPUT_STATE_KEYS = {
    'state', 'root_role', 'root_lstat', 'regular_final_bindings',
    'temporary_orphan_bindings', 'durability_uncertain_final_bindings',
    'preexisting_entry_states', 'directory_paths', 'directory_tree_sha256',
    'directory_file_tree_sha256', 'file_tree_sha256',
    'entry_state_tree_sha256', 'publication_failure',
}
FAILURE_PHASES = {
    'post_start_source_gate', 'post_start_prior_cache_gate',
    'model_input_rehash', 'structural_terminal_audit', 'result_publication',
    'analysis_publication', 'final_toctou', 'complete_publication',
}


def sha256(path: Path) -> str:
  observed = path.lstat()
  return sha256_no_follow(path, observed)


def sha256_no_follow(path: Path, expected: os.stat_result) -> str:
  descriptor = os.open(
      path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
  )
  try:
    before = os.fstat(descriptor)
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_nlink, value.st_mode,
        value.st_size,
    )
    if not stat.S_ISREG(before.st_mode) or identity(before) != identity(expected):
      raise RuntimeError(f'No-follow file identity changed: {path}')
    digest = hashlib.sha256()
    for block in iter(lambda: os.read(descriptor, 1024 * 1024), b''):
      digest.update(block)
    if identity(os.fstat(descriptor)) != identity(before):
      raise RuntimeError(f'File changed during no-follow read: {path}')
    try:
      final_path = path.lstat()
    except FileNotFoundError as error:
      raise RuntimeError(f'File pathname disappeared during read: {path}') from error
    if identity(final_path) != identity(before):
      raise RuntimeError(f'File pathname changed during read: {path}')
    return digest.hexdigest()
  finally:
    os.close(descriptor)


def canonical_binding(value: Any) -> dict[str, Any]:
  payload = json.dumps(
      value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
      allow_nan=False,
  ).encode()
  return {'sha256': hashlib.sha256(payload).hexdigest(), 'size_bytes': len(payload)}


def git_output(*args: str, binary: bool = False) -> Any:
  return subprocess.check_output(
      ('git', '-C', str(REPO), *args), text=not binary
  )


def git_blob(commit: str, relative: str) -> bytes:
  return git_output('show', f'{commit}:{relative}', binary=True)


def file_binding(path: Path, *, absolute: bool = False) -> dict[str, Any]:
  status = path.lstat()
  if not stat.S_ISREG(status.st_mode) or stat.S_ISLNK(status.st_mode):
    raise RuntimeError(f'Unsafe file: {path}')
  result = {'sha256': sha256_no_follow(path, status), 'size_bytes': status.st_size}
  return {'path': str(path.resolve()), **result} if absolute else result


def binding_map_digest(bindings: Mapping[str, Mapping[str, Any]]) -> str:
  digest = hashlib.sha256()
  for relative in sorted(bindings):
    digest.update(relative.encode())
    digest.update(b'\0')
    digest.update(bytes.fromhex(str(bindings[relative]['sha256'])))
  return digest.hexdigest()


def directory_digest(paths: Iterable[str]) -> str:
  digest = hashlib.sha256()
  for relative in sorted(paths):
    digest.update(b'D\0')
    digest.update(relative.encode())
    digest.update(b'\0')
  return digest.hexdigest()


def tree_binding(
    root: Path, *, expected_files: set[str] | None = None,
    expected_directories: set[str] | None = None,
    label: str = 'tree',
) -> dict[str, Any]:
  status = root.lstat()
  if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
    raise RuntimeError(f'Unsafe tree root: {root}')
  directories = ['.']
  files: dict[str, dict[str, Any]] = {}
  for entry in sorted(root.rglob('*')):
    relative = entry.relative_to(root).as_posix()
    observed = entry.lstat()
    if stat.S_ISLNK(observed.st_mode):
      raise RuntimeError(f'Symlink in tree: {entry}')
    if stat.S_ISDIR(observed.st_mode):
      if expected_directories is not None and relative not in expected_directories:
        raise RuntimeError(f'Unexpected directory in {label}: {relative}')
      directories.append(relative)
    elif stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1:
      if expected_files is not None and relative not in expected_files:
        raise RuntimeError(f'Unexpected file in {label}: {relative}')
      files[relative] = {
          'sha256': sha256_no_follow(entry, observed),
          'size_bytes': observed.st_size,
          'mode': f'{stat.S_IMODE(observed.st_mode):04o}',
          'st_dev': observed.st_dev, 'st_ino': observed.st_ino,
          'st_nlink': observed.st_nlink,
      }
    else:
      raise RuntimeError(f'Unsafe tree entry: {entry}')
  if (
      expected_files is not None and set(files) != expected_files
      or expected_directories is not None
      and set(directories) != expected_directories
  ):
    raise RuntimeError(f'Exact {label} membership changed during hashing.')
  if expected_files is not None and expected_directories is not None:
    assert_tree_membership(root, expected_files, expected_directories, label)
  directories = sorted(directories)
  combined = hashlib.sha256()
  for relative in directories:
    combined.update(b'D\0' + relative.encode() + b'\0')
  for relative in sorted(files):
    combined.update(b'F\0' + relative.encode() + b'\0')
    combined.update(bytes.fromhex(files[relative]['sha256']))
  return {
      'root': str(root.resolve()), 'file_count': len(files),
      'directory_count': len(directories), 'file_bindings': files,
      'file_tree_sha256': binding_map_digest(files),
      'directory_paths': directories,
      'directory_tree_sha256': directory_digest(directories),
      'directory_file_tree_sha256': combined.hexdigest(),
  }


def validate_immutable_facts(value: Mapping[str, Any]) -> None:
  """Rejects drift instead of normalizing it into the prospective freeze."""
  summaries = {
      'run_root_binding': (
          12, ['.', 'compiler', 'compiler/eight_row'],
          '960faf1675caaa0f3c9798f7b943998e91650b2a9359b26ca31ce2d417c2ce0b',
          '5331e5041b557a4324ba57d57179dc2bfd8e6ab981f0cdede5d4f126e94c2041',
      ),
      'compiler_tree_binding': (
          5, ['.', 'eight_row'],
          'b1094dfaddb0e8c6672b09a18e124af2a20a1a91c7ca817911c2f3fe4c0220a3',
          'bb042bf9a2cb34c61aae121733edce583cc2d747de1913b1d74f00b7a8de200c',
      ),
      'preflight_tree_binding': (
          5, ['.'],
          'ae277eafa4f7f20bfa74c3a0a1bbaa0f51468cac945d29d0c49cab699738ecfd',
          'cc106b406da58ddd95611aef7e471f5a5cefd96e302ebb91ea4ef9e28a618c87',
      ),
      'external_cache_tree_binding': (
          2, ['.', 'triton', 'xdg'],
          '3bd7b53ba7ab1dae7161999ff907137f82ee6d7f322512a3221646f66bb1e975',
          'd040af81aa50fbe28e0523747355f84d851f36f39e586294b24dd994f69f66a0',
      ),
      'model_cache_tree_binding': (
          1, ['.', 'triton', 'xdg', 'xdg/matplotlib'],
          '487c67a6dbb251aca190ac9eda5d2425c3584febc9ad63e60d0812c7f2fb69ea',
          '51fe59713c301342bf5bb161f26b9e4ee6828e508b96e4dbd21c6efcdde1115e',
      ),
  }
  for name, (file_count, directories, file_digest, combined_digest) in summaries.items():
    binding = value[name]
    if (
        binding['file_count'] != file_count
        or binding['directory_paths'] != directories
        or binding['file_tree_sha256'] != file_digest
        or binding['directory_file_tree_sha256'] != combined_digest
    ):
      raise RuntimeError(f'Immutable model artifact fact changed: {name}')
  expected_modes = {
      'run_root_binding': {
          relative: '0400'
          for relative in value['run_root_binding']['file_bindings']
      },
      'compiler_tree_binding': {
          relative: '0400'
          for relative in value['compiler_tree_binding']['file_bindings']
      },
      'preflight_tree_binding': {
          '.allocation.lock': '0600', '.preflight_0000.reserved': '0400',
          'preflight_0000.json': '0400',
          'preflight_0000.stderr.log': '0400',
          'preflight_0000.stdout.log': '0400',
      },
      'external_cache_tree_binding': {
          relative: '0400'
          for relative in value['external_cache_tree_binding']['file_bindings']
      },
      'model_cache_tree_binding': {
          'xdg/matplotlib/fontlist-v3.11.0.json': '0600',
      },
  }
  for name, modes in expected_modes.items():
    observed = {
        relative: binding['mode']
        for relative, binding in value[name]['file_bindings'].items()
    }
    if observed != modes:
      raise RuntimeError(f'Immutable model artifact modes changed: {name}')
  exact_files = {
      'run_terminal_binding': (
          'fdbd0a1dc7d24145f88c5a009cc80d8904e57920e0c9584426e791373fae6d8f',
          43_760,
      ),
      'raw_manifest_binding': (
          '3ee95b22d483c7c4f234fbb75281e05e84f0be263b1ee670a94b2cd442d61136',
          1_562,
      ),
  }
  for name, (digest, size) in exact_files.items():
    if value[name]['sha256'] != digest or value[name]['size_bytes'] != size:
      raise RuntimeError(f'Immutable model artifact changed: {name}')
  bundle = value['old_analyzer_bundle']
  expected_bundle = {
      'analyzer': '9320184c53ed6bc3b246443314d84c1f1543bbbf77aa10e3fff982bd5c18913a',
      'test': 'dcede28da855e3784a86453fb8f1cdeb3b94d326bc249023f3e72b82316a0fe5',
      'shell': 'cea01ac69a8468f54e4bfb8a453709494449ef886e7dc6ae28e073a79fa2855c',
      'freeze': '2f4eaf1366dcb42b8f89a386e8201b3f2ba0b9f8ae5ef02409436492666d8366',
  }
  bundle_paths = {
      'analyzer': HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_4_5.py',
      'test': HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_4_5_test.py',
      'shell': HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_4_5.sh',
      'freeze': MODEL_FREEZE,
  }
  if bundle.get('git_head') != MODEL_HEAD:
    raise RuntimeError('Immutable old analyzer bundle head changed.')
  for name, digest in expected_bundle.items():
    if (
        bundle.get(name) != file_binding(bundle_paths[name], absolute=True)
        or bundle[name]['sha256'] != digest
    ):
      raise RuntimeError(f'Immutable old analyzer bundle changed: {name}')


def assert_tree_membership(
    root: Path, expected_files: Iterable[str],
    expected_directories: Iterable[str], label: str,
) -> None:
  """Performs an lstat-only allowlist pass before any artifact byte read."""
  status = root.lstat()
  if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
    raise RuntimeError(f'{label} root is unsafe.')
  files: list[str] = []
  directories = ['.']
  for entry in sorted(root.rglob('*')):
    relative = entry.relative_to(root).as_posix()
    observed = entry.lstat()
    if stat.S_ISLNK(observed.st_mode):
      raise RuntimeError(f'{label} contains a symlink: {relative}')
    if stat.S_ISDIR(observed.st_mode):
      directories.append(relative)
    elif stat.S_ISREG(observed.st_mode):
      files.append(relative)
    else:
      raise RuntimeError(f'{label} contains a special entry: {relative}')
  if (
      sorted(files) != sorted(expected_files)
      or sorted(directories) != sorted(expected_directories)
  ):
    raise RuntimeError(f'{label} exact structural membership changed.')


FAILURE_TRACEBACK = '''Traceback (most recent call last):
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_5.py", line 8513, in <module>
    main()
    ~~~~^^
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_5.py", line 8452, in main
    precheck = _analysis_attempt_precheck(run_dir, bundle_root=bundle_root)
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_5.py", line 7854, in _analysis_attempt_precheck
    _validate_freeze_v3345(run_dir, bundle_root=bundle_root)
    ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_5.py", line 5162, in _validate_freeze_v3345
    prior333 = _validate_prior_v3_3_3()
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_5.py", line 1201, in _validate_prior_v3_3_3
    cache_paths = _strict_tree(
        _PRIOR_CACHE_DIR, set(_PRIOR_CACHE_FILES), 'v3.3.3 cache'
    )
  File "/home/degen2/alphafold-stuff/alphagenome_research/experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_5.py", line 1076, in _strict_tree
    raise AnalysisError(f'{label} contains an extra/empty directory.')
AnalysisError: v3.3.3 cache contains an extra/empty directory.
'''


def consumed_failure() -> dict[str, Any]:
  old_shell = HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_4_5.sh'
  destinations = OLD_DESTINATIONS
  return {
      'captured_at_unix_s': None, 'chunk_id': '77e144',
      'command': [str(old_shell.resolve()), '--acknowledge-structural-only-v3-3-4-5'],
      'destination_states': {str(path.resolve()): 'absent' for path in destinations},
      'exit_code': 1, 'failed_before_start': True,
      'failure': {
          'message': 'v3.3.3 cache contains an extra/empty directory.',
          'stage': 'precheck_prior_v3_3_3_cache', 'type': 'AnalysisError',
      },
      'no_jax_model_raw_or_confirmation_access': True,
      'retry_permitted': False, 'session_id': None,
      'status': 'consumed_pre_start_failure',
      'stderr': {
          'final_newline': True, 'persisted_to_filesystem': False,
          'sha256': '0158926b7b41b6636bfacd2acdcf268bae7f7082b9f935edb483aa184bdd6967',
          'size_bytes': 1587,
          'source': 'coordinator_captured_unpersisted_tool_output',
      },
      'stderr_text': FAILURE_TRACEBACK,
      'stdout': {'persisted_to_filesystem': False, 'sha256': EMPTY_SHA256, 'size_bytes': 0},
      'wall_time_seconds': 1.95131362,
  }


def prior_cache_contract() -> dict[str, Any]:
  rows = [
      ('.', 'directory', '0700', 4096, 66307, 140791354, 4, None),
      ('triton', 'directory', '0700', 4096, 66307, 140791357, 2, None),
      ('xdg', 'directory', '0700', 4096, 66307, 140791358, 3, None),
      ('xdg/matplotlib', 'directory', '0700', 4096, 66307, 140791359, 2, None),
      ('xdg/matplotlib/fontlist-v3.11.0.json', 'regular', '0600', 163240, 66307, 140791361, 1, 'a777469f8f54be8cc9107788bce2b3cd23709aa317114392cdc365b3fb127125'),
  ]
  lstat_rows = [{
      'path': p, 'entry_type': t, 'mode': m, 'size_bytes': s,
      'st_dev': d, 'st_ino': i, 'st_nlink': n, 'sha256': h,
  } for p, t, m, s, d, i, n, h in rows]
  file_row = lstat_rows[-1]
  binding = {key: file_row[key] for key in ('sha256', 'size_bytes', 'mode', 'st_dev', 'st_ino', 'st_nlink')}
  result = {
      'root': str(PRIOR_CACHE.resolve()), 'file_count': 1,
      'directory_count': 4,
      'directory_paths': ['.', 'triton', 'xdg', 'xdg/matplotlib'],
      'lstat_rows': lstat_rows,
      'file_bindings': {'xdg/matplotlib/fontlist-v3.11.0.json': binding},
      'file_tree_sha256': 'd1d11bc6dc48b302cf675fb48727bd6ededec09142429eaa9e368f7631463717',
      'directory_file_tree_sha256': 'a1dafb75097282b3d28174e4bb6c79a81d2f1c8c25cd8c64a1ea84642ba7f43a',
      'exact_membership': True, 'no_follow': True,
  }
  descriptors: list[int] = []
  try:
    root_path_status = PRIOR_CACHE.lstat()
    root_fd = os.open(
        PRIOR_CACHE, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    descriptors.append(root_fd)
    triton_fd = os.open(
        'triton', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=root_fd,
    )
    descriptors.append(triton_fd)
    xdg_fd = os.open(
        'xdg', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=root_fd,
    )
    descriptors.append(xdg_fd)
    mpl_fd = os.open(
        'matplotlib', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=xdg_fd,
    )
    descriptors.append(mpl_fd)
    file_fd = os.open(
        'fontlist-v3.11.0.json',
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=mpl_fd,
    )
    descriptors.append(file_fd)
    if (
        sorted(os.listdir(root_fd)) != ['triton', 'xdg']
        or os.listdir(triton_fd) or os.listdir(xdg_fd) != ['matplotlib']
        or os.listdir(mpl_fd) != ['fontlist-v3.11.0.json']
    ):
      raise RuntimeError('Prior cache membership changed.')
    statuses = [
        os.fstat(root_fd), os.fstat(triton_fd), os.fstat(xdg_fd),
        os.fstat(mpl_fd), os.fstat(file_fd),
    ]
    if (
        (root_path_status.st_dev, root_path_status.st_ino)
        != (statuses[0].st_dev, statuses[0].st_ino)
    ):
      raise RuntimeError('Prior cache root inode changed.')
    digest = hashlib.sha256()
    for block in iter(lambda: os.read(file_fd, 1024 * 1024), b''):
      digest.update(block)
    if digest.hexdigest() != lstat_rows[-1]['sha256']:
      raise RuntimeError('Prior cache file bytes changed.')
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_nlink, value.st_mode,
        value.st_size,
    )
    final_root = PRIOR_CACHE.lstat()
    if (
        (final_root.st_dev, final_root.st_ino)
        != (statuses[0].st_dev, statuses[0].st_ino)
        or sorted(os.listdir(root_fd)) != ['triton', 'xdg']
        or os.listdir(triton_fd) or os.listdir(xdg_fd) != ['matplotlib']
        or os.listdir(mpl_fd) != ['fontlist-v3.11.0.json']
    ):
      raise RuntimeError('Prior cache hierarchy changed during read.')
    final_statuses = [
        os.fstat(root_fd), os.fstat(triton_fd), os.fstat(xdg_fd),
        os.fstat(mpl_fd), os.fstat(file_fd),
    ]
    if any(
        identity(before) != identity(after)
        for before, after in zip(statuses, final_statuses, strict=True)
    ):
      raise RuntimeError('Prior cache inode changed during read.')
    linked = [
        os.stat('triton', dir_fd=root_fd, follow_symlinks=False),
        os.stat('xdg', dir_fd=root_fd, follow_symlinks=False),
        os.stat('matplotlib', dir_fd=xdg_fd, follow_symlinks=False),
        os.stat(
            'fontlist-v3.11.0.json', dir_fd=mpl_fd,
            follow_symlinks=False,
        ),
    ]
    if any(
        identity(link) != identity(expected)
        for link, expected in zip(linked, statuses[1:], strict=True)
    ):
      raise RuntimeError('Prior cache pathname inode changed during read.')
    for expected, observed in zip(lstat_rows, statuses, strict=True):
      actual = {
          'path': expected['path'], 'entry_type': expected['entry_type'],
          'mode': f'{stat.S_IMODE(observed.st_mode):04o}',
          'size_bytes': observed.st_size, 'st_dev': observed.st_dev,
          'st_ino': observed.st_ino, 'st_nlink': observed.st_nlink,
          'sha256': digest.hexdigest() if expected['entry_type'] == 'regular' else None,
      }
      if actual != expected:
        raise RuntimeError(f"Prior cache lstat changed: {expected['path']}")
  finally:
    for descriptor in reversed(descriptors):
      os.close(descriptor)
  if (
      binding_map_digest(result['file_bindings']) != result['file_tree_sha256']
  ):
    raise RuntimeError('Prior cache file-tree framing changed.')
  combined = hashlib.sha256()
  for relative in result['directory_paths']:
    combined.update(b'D\0' + relative.encode() + b'\0')
  for relative, value in sorted(result['file_bindings'].items()):
    combined.update(b'F\0' + relative.encode() + b'\0')
    combined.update(bytes.fromhex(value['sha256']))
  if combined.hexdigest() != result['directory_file_tree_sha256']:
    raise RuntimeError('Prior cache D/F framing changed.')
  return result


def source_contract(source_head: str) -> dict[str, Any]:
  old_binding = file_binding(MODEL_FREEZE)
  if old_binding != {
      'sha256': '2f4eaf1366dcb42b8f89a386e8201b3f2ba0b9f8ae5ef02409436492666d8366',
      'size_bytes': 204_697,
  }:
    raise RuntimeError('Authenticated v3.3.4.5 source freeze changed.')
  model_freeze_relative = MODEL_FREEZE.relative_to(REPO).as_posix()
  old_bytes = git_blob(MODEL_HEAD, model_freeze_relative)
  if (
      len(old_bytes) != old_binding['size_bytes']
      or hashlib.sha256(old_bytes).hexdigest() != old_binding['sha256']
  ):
    raise RuntimeError('Historical v3.3.4.5 source freeze changed.')
  old_freeze = json.loads(old_bytes)
  old_contract = old_freeze.get('source_inventory_contract')
  if not isinstance(old_contract, dict) or set(old_contract) != {
      'source_row_count', 'rows', 'prospective_upstream_source_file_count',
      'loaded_scientific_module_contract',
  }:
    raise RuntimeError('Inherited source contract schema changed.')
  old = old_contract['rows']
  if old_contract['source_row_count'] != 132 or len(old) != 132:
    raise RuntimeError('Inherited source inventory is not 132 rows.')
  if [row.get('path') for row in old] != sorted(row.get('path') for row in old):
    raise RuntimeError('Inherited source inventory order changed.')
  rows = []
  for value in old:
    relative = value['path']
    rows.append({**value, 'authority_commit': MODEL_HEAD})
  new_paths = [path.relative_to(REPO).as_posix() for path in NEW_FILES]
  amendment_relative = AMENDMENT.relative_to(REPO).as_posix()
  for path, commit in [(AMENDMENT, DOCS_HEAD), *[(path, source_head) for path in NEW_FILES]]:
    relative = path.relative_to(REPO).as_posix()
    mode = '100755' if relative.endswith('.sh') else '100644'
    observed = path.lstat()
    rows.append({
        'path': relative, 'sha256': sha256_no_follow(path, observed),
        'size_bytes': observed.st_size, 'git_mode': mode,
        'authority_commit': commit,
    })
  rows.sort(key=lambda row: row['path'])
  if len(rows) != 137 or len({row['path'] for row in rows}) != 137:
    raise RuntimeError('Generated source inventory is not 137 unique rows.')
  for row in rows:
    live_path = REPO / row['path']
    observed_live = live_path.lstat()
    if (
        sha256_no_follow(live_path, observed_live) != row['sha256']
        or observed_live.st_size != row['size_bytes']
    ):
      raise RuntimeError(f"Live source changed: {row['path']}")
    if hashlib.sha256(git_blob(row['authority_commit'], row['path'])).hexdigest() != row['sha256']:
      raise RuntimeError(f"Authority blob changed: {row['path']}")
    if hashlib.sha256(git_blob(source_head, row['path'])).hexdigest() != row['sha256']:
      raise RuntimeError(f"Source-head blob changed: {row['path']}")
    authority_line = git_output(
        'ls-tree', row['authority_commit'], '--', row['path']
    ).strip()
    source_line = git_output('ls-tree', source_head, '--', row['path']).strip()
    live_mode = (
        '100755' if stat.S_IMODE(observed_live.st_mode) & 0o111
        else '100644'
    )
    if (
        not authority_line or authority_line.split()[0] != row['git_mode']
        or not source_line or source_line.split()[0] != row['git_mode']
        or live_mode != row['git_mode']
    ):
      raise RuntimeError(f"Source mode changed: {row['path']}")
  if git_output('rev-parse', 'HEAD').strip() != source_head:
    raise RuntimeError('Source HEAD changed during inventory audit.')
  subprocess.check_call(('git', '-C', str(REPO), 'diff', '--quiet', 'HEAD', '--'))
  return {
      'row_count': 137, 'rows': rows,
      'authority_partitions': {
          'inherited_132': {
              'authority_commit': MODEL_HEAD, 'row_count': 132,
              'paths': sorted(row['path'] for row in rows if row['authority_commit'] == MODEL_HEAD),
          },
          'amendment': {
              'authority_commit': DOCS_HEAD, 'row_count': 1,
              'paths': [amendment_relative],
          },
          'new_implementation_4': {
              'authority_commit': source_head, 'row_count': 4,
              'paths': sorted(new_paths),
          },
      },
      'source_authority_head': source_head,
      'source_authority_tree_exact': True,
      'all_rows_authority_exact': True,
      'all_rows_live_at_generation_exact': True,
      'tree_sha256': canonical_binding(rows)['sha256'],
  }


def build_freeze() -> dict[str, Any]:
  if FREEZE.exists() or FREEZE.is_symlink():
    raise FileExistsError(f'Analysis freeze already exists: {FREEZE}')
  for path in (*OLD_DESTINATIONS, ATTEMPT, ANALYSIS):
    if path.exists() or path.is_symlink():
      raise RuntimeError(f'Analyzer destination is not absent: {path}')
  subprocess.check_call(('git', '-C', str(REPO), 'diff', '--quiet', 'HEAD', '--'))
  source_head = git_output('rev-parse', 'HEAD').strip()
  if not source_head or source_head in {MODEL_HEAD, DOCS_HEAD}:
    raise RuntimeError('Generator is not running at the dedicated source-authority commit.')
  parents = git_output('rev-list', '--parents', '-n', '1', source_head).split()
  if parents != [source_head, DOCS_HEAD]:
    raise RuntimeError('Source-authority commit is not the sole child of docs HEAD.')
  expected_delta = [
      f'A\t{path.relative_to(REPO).as_posix()}' for path in sorted(NEW_FILES)
  ]
  observed_delta = git_output(
      'diff', '--name-status', DOCS_HEAD, source_head
  ).splitlines()
  if observed_delta != expected_delta:
    raise RuntimeError(
        'Docs-to-source-authority delta is not exactly the four new files.'
    )
  source = source_contract(source_head)
  prior_cache = prior_cache_contract()
  consumed = consumed_failure()
  old_bundle = {
      'git_head': MODEL_HEAD,
      'analyzer': file_binding(HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_4_5.py', absolute=True),
      'test': file_binding(HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_4_5_test.py', absolute=True),
      'shell': file_binding(HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_4_5.sh', absolute=True),
      'freeze': file_binding(MODEL_FREEZE, absolute=True),
  }
  run_files = {
      'ATTEMPT_STARTED.json', 'IMPORT_PROVENANCE.json',
      'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
      'IMPORT_PROVENANCE_PRE_MODEL.json', 'PROTOBUF_PROVENANCE.json',
      'RAW_MANIFEST.json', 'RUN_COMPLETE.json',
      'compiler/eight_row/COMPILER_DIAGNOSTIC_FAILURE.json',
      'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json',
      'compiler/eight_row/graph.compiled.hlo.txt',
      'compiler/eight_row/graph.pre_backend.hlo.txt',
      'compiler/eight_row/graph.stablehlo.mlir',
  }
  compiler_files = {
      relative.removeprefix('compiler/') for relative in run_files
      if relative.startswith('compiler/')
  }
  assert_tree_membership(
      RUN, run_files, {'.', 'compiler', 'compiler/eight_row'}, 'model run'
  )
  assert_tree_membership(
      RUN / 'compiler', compiler_files, {'.', 'eight_row'}, 'compiler'
  )
  assert_tree_membership(
      PREFLIGHT,
      {
          '.allocation.lock', '.preflight_0000.reserved',
          'preflight_0000.json', 'preflight_0000.stderr.log',
          'preflight_0000.stdout.log',
      },
      {'.'}, 'external preflight',
  )
  assert_tree_membership(
      EXTERNAL_CACHE,
      {
          '.v3345.tmp.2777420.000001.7a795e5eda1e9fcf14f19a8d62c7960f',
          'atomic_publication_probe_v3_3_4_5.txt',
      },
      {'.', 'triton', 'xdg'}, 'external cache',
  )
  assert_tree_membership(
      MODEL_CACHE, {'xdg/matplotlib/fontlist-v3.11.0.json'},
      {'.', 'triton', 'xdg', 'xdg/matplotlib'}, 'model cache',
  )
  immutable = {
      'run_root_binding': tree_binding(
          RUN, expected_files=run_files,
          expected_directories={'.', 'compiler', 'compiler/eight_row'},
          label='model run',
      ),
      'compiler_tree_binding': tree_binding(
          RUN / 'compiler', expected_files=compiler_files,
          expected_directories={'.', 'eight_row'}, label='compiler',
      ),
      'preflight_tree_binding': tree_binding(
          PREFLIGHT,
          expected_files={
              '.allocation.lock', '.preflight_0000.reserved',
              'preflight_0000.json', 'preflight_0000.stderr.log',
              'preflight_0000.stdout.log',
          },
          expected_directories={'.'}, label='external preflight',
      ),
      'external_cache_tree_binding': tree_binding(
          EXTERNAL_CACHE,
          expected_files={
              '.v3345.tmp.2777420.000001.7a795e5eda1e9fcf14f19a8d62c7960f',
              'atomic_publication_probe_v3_3_4_5.txt',
          },
          expected_directories={'.', 'triton', 'xdg'}, label='external cache',
      ),
      'model_cache_tree_binding': tree_binding(
          MODEL_CACHE,
          expected_files={'xdg/matplotlib/fontlist-v3.11.0.json'},
          expected_directories={'.', 'triton', 'xdg', 'xdg/matplotlib'},
          label='model cache',
      ),
      'run_terminal_binding': file_binding(RUN / 'RUN_COMPLETE.json', absolute=True),
      'raw_manifest_binding': file_binding(RUN / 'RAW_MANIFEST.json', absolute=True),
      'old_analyzer_bundle': old_bundle,
  }
  validate_immutable_facts(immutable)
  publication = {
      'schema_version': 'v3.3.4.5.1-named-temp-renameat2-noreplace-v1',
      'method': 'named_temp_renameat2_noreplace',
      'temp_name_regex': r'^\.v33451\.tmp\.[1-9][0-9]*\.[0-9]{6}\.[0-9a-f]{32}$',
      'nonce_bytes': 16,
      'open_flags': ['O_RDWR', 'O_CREAT', 'O_EXCL', 'O_NOFOLLOW', 'O_CLOEXEC'],
      'initial_mode': '0600', 'sealed_mode': '0400',
      'rename_flags': ['RENAME_NOREPLACE'], 'same_directory_required': True,
      'keep_fd_open_through_rename': True, 'file_fsync_count': 2,
      'parent_fsync_required': True,
      'post_publish_inode_revalidation_required': True,
      'no_replace': True, 'no_fallback': True, 'no_retry': True,
      'root_roles': ['analysis_attempt', 'analysis_output'],
      'success_keys': sorted(PUBLICATION_SUCCESS_KEYS),
      'failure_keys': sorted(PUBLICATION_FAILURE_KEYS),
      'audit_keys': sorted(PUBLICATION_AUDIT_KEYS),
      'output_state_keys': sorted(OUTPUT_STATE_KEYS),
      'entry_state_keys': sorted(ENTRY_STATE_KEYS),
      'failure_stages': sorted(FAILURE_STAGES),
  }
  records = {
      'start_keys': sorted(START_KEYS), 'analysis_keys': sorted(ANALYSIS_KEYS),
      'complete_keys': sorted(COMPLETE_KEYS), 'failure_keys': sorted(FAILURE_KEYS),
      'publication_success_keys': sorted(PUBLICATION_SUCCESS_KEYS),
      'publication_failure_keys': sorted(PUBLICATION_FAILURE_KEYS),
      'publication_audit_keys': sorted(PUBLICATION_AUDIT_KEYS),
      'output_state_keys': sorted(OUTPUT_STATE_KEYS),
      'failure_phase_values': sorted(FAILURE_PHASES),
  }
  claim = {
      'structural_only': True, 'no_biological_claim': True,
      'no_scientific_summary': True, 'no_normalization': True,
      'no_shapley': True, 'no_interaction': True, 'no_resolution': True,
      'no_nomination': True, 'combined_analysis_permitted': False,
      'future_protocol_required': True,
  }
  result = {
      'schema_version': SCHEMA_VERSION, 'analysis_version': ANALYSIS_VERSION,
      'attempt_id': ATTEMPT_ID, 'acknowledgement_token': ACK,
      'amendment_path': str(AMENDMENT.resolve()),
      'amendment_sha256': AMENDMENT_SHA256,
      'amendment_commit': DOCS_HEAD, 'prior_model_head': MODEL_HEAD,
      'prior_model_freeze_binding': file_binding(MODEL_FREEZE, absolute=True),
      'source_inventory_contract': source,
      'immutable_model_artifact_contract': immutable,
      'consumed_analyzer_failure': consumed,
      'consumed_analyzer_failure_content_binding': canonical_binding(consumed),
      'prior_cache_contract': prior_cache,
      'prior_cache_contract_content_binding': canonical_binding(prior_cache),
      'analysis_attempt_dir': str(ATTEMPT.resolve()),
      'analysis_dir': str(ANALYSIS.resolve()),
      'publication_contract': publication, 'record_contracts': records,
      'claim_boundary': claim,
  }
  if len(result) != 20:
    raise RuntimeError('Generated freeze is not exactly 20 keys.')
  return result


def main() -> None:
  value = build_freeze()
  payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
  descriptor = os.open(
      FREEZE, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
  )
  try:
    view = memoryview(payload)
    while view:
      written = os.write(descriptor, view)
      if written <= 0:
        raise OSError('Short freeze write.')
      view = view[written:]
    os.fchmod(descriptor, 0o644)
    os.fsync(descriptor)
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o644
        or before.st_nlink != 1 or before.st_size != len(payload)
    ):
      raise RuntimeError('Generated freeze inode contract changed.')
    os.lseek(descriptor, 0, os.SEEK_SET)
    observed = bytearray()
    while len(observed) < len(payload):
      block = os.read(descriptor, min(1024 * 1024, len(payload) - len(observed)))
      if not block:
        break
      observed.extend(block)
    after = os.fstat(descriptor)
    if (
        bytes(observed) != payload
        or hashlib.sha256(observed).hexdigest()
        != hashlib.sha256(payload).hexdigest()
        or (after.st_dev, after.st_ino, after.st_nlink, after.st_mode,
            after.st_size)
        != (before.st_dev, before.st_ino, before.st_nlink, before.st_mode,
            before.st_size)
    ):
      raise RuntimeError('Generated freeze readback contract changed.')
  finally:
    os.close(descriptor)
  parent_fd = os.open(
      FREEZE.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
  )
  try:
    os.fsync(parent_fd)
  finally:
    os.close(parent_fd)


if __name__ == '__main__':
  main()
