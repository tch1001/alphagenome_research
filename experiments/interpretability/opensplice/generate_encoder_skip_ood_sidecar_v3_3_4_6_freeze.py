#!/usr/bin/env python3
"""Generate the prospective, acyclic v3.3.4.6 machine freeze.

This module is standard-library-only. It never imports the runner, JAX, the
AlphaGenome model, or an older analyzer.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping


_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

# pylint: disable=g-import-not-at-top
import validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_6 as bootstrap


DOCS_HEAD = '686b67bd772a7771a6c4540c5b188b6497dbdec7'
MODEL_HEAD = '0da8f47ea6e576a72a1cda204ce868ef79cc2ce5'
BASE_FREEZE_PATH = _HERE / 'encoder_skip_ood_sidecar_v3_3_4_5_freeze.json'
BASE_FREEZE_SHA256 = (
    '2f4eaf1366dcb42b8f89a386e8201b3f2ba0b9f8ae5ef02409436492666d8366'
)
BASE_FREEZE_SIZE_BYTES = 204697


_EXTRA_FILES = (
    'experiments/interpretability/opensplice/'
    'analyze_encoder_skip_ood_sidecar_v3_3_4_6.py',
    'experiments/interpretability/opensplice/'
    'analyze_encoder_skip_ood_sidecar_v3_3_4_6.sh',
    'experiments/interpretability/opensplice/'
    'analyze_encoder_skip_ood_sidecar_v3_3_4_6_test.py',
    'experiments/interpretability/opensplice/'
    'generate_encoder_skip_ood_sidecar_v3_3_4_6_freeze.py',
    'experiments/interpretability/opensplice/'
    'validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_6.py',
    'experiments/interpretability/opensplice/'
    'launch_encoder_skip_ood_sidecar_v3_3_4_6.py',
    'experiments/interpretability/opensplice/run_device_preflight_v3_3_4_6.py',
    'experiments/interpretability/opensplice/'
    'run_device_preflight_v3_3_4_6_test.py',
    'experiments/interpretability/opensplice/'
    'run_encoder_skip_ood_sidecar_v3_3_4_6.py',
    'experiments/interpretability/opensplice/'
    'run_encoder_skip_ood_sidecar_v3_3_4_6_test.py',
    'experiments/interpretability/opensplice/'
    'run_encoder_skip_ood_sidecar_v3_3_4_6.sh',
    'experiments/interpretability/opensplice/v3_wider_mechanism/'
    'encoder_skip_ood_sidecar_compiler_record_amendment_v3_3_4_6.md',
)


def _read_bytes(path: Path) -> bytes:
  observed = path.lstat()
  if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
    raise RuntimeError(f'Unsafe source path: {path}.')
  descriptor = os.open(
      path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
  )
  try:
    before = os.fstat(descriptor)
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_nlink, value.st_mode,
        value.st_size,
    )
    if identity(before) != identity(observed):
      raise RuntimeError(f'Source inode changed before read: {path}.')
    payload = bytearray()
    for block in iter(lambda: os.read(descriptor, 1024 * 1024), b''):
      payload.extend(block)
    after = os.fstat(descriptor)
    final_path = path.lstat()
    if identity(after) != identity(before) or identity(final_path) != identity(before):
      raise RuntimeError(f'Source inode changed during read: {path}.')
    return bytes(payload)
  finally:
    os.close(descriptor)


def _sha256(path: Path) -> str:
  return hashlib.sha256(_read_bytes(path)).hexdigest()


def _canonical_binding(value: Any) -> dict[str, Any]:
  payload = json.dumps(
      value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
      allow_nan=False,
  ).encode('utf-8')
  return {'sha256': hashlib.sha256(payload).hexdigest(), 'size_bytes': len(payload)}


def _git_output(*args: str, binary: bool = False) -> Any:
  return subprocess.check_output(
      ('git', '-C', str(_REPO), *args), text=not binary
  )


def _git_mode(commit: str, relative: str) -> str:
  line = _git_output('ls-tree', commit, '--', relative).strip()
  fields = line.split()
  if len(fields) < 4 or fields[1] != 'blob':
    raise RuntimeError(f'Git mode is unavailable: {commit}:{relative}.')
  return fields[0]


def _source_authority_head() -> str:
  subprocess.check_call(('git', '-C', str(_REPO), 'diff', '--quiet', 'HEAD', '--'))
  head = _git_output('rev-parse', 'HEAD').strip()
  parents = _git_output('rev-list', '--parents', '-n', '1', head).split()
  if parents != [head, DOCS_HEAD]:
    raise RuntimeError('Source-authority HEAD is not the sole child of docs HEAD.')
  expected = [
      f'A\t{relative}' for relative in sorted(
          path for path in _EXTRA_FILES
          if not path.endswith(
              'encoder_skip_ood_sidecar_compiler_record_amendment_v3_3_4_6.md'
          )
      )
  ]
  observed = _git_output('diff', '--name-status', DOCS_HEAD, head).splitlines()
  if observed != expected:
    raise RuntimeError('Docs-to-source delta is not exactly eleven additions.')
  return head


def build_freeze() -> dict[str, object]:
  base_status = BASE_FREEZE_PATH.lstat()
  if (
      stat.S_ISLNK(base_status.st_mode)
      or not stat.S_ISREG(base_status.st_mode)
      or stat.S_IMODE(base_status.st_mode) != 0o644
      or base_status.st_size != BASE_FREEZE_SIZE_BYTES
      or _sha256(BASE_FREEZE_PATH) != BASE_FREEZE_SHA256
  ):
    raise RuntimeError('Inherited v3.3.4.5 freeze binding changed.')
  base = json.loads(_read_bytes(BASE_FREEZE_PATH).decode('utf-8'))
  inherited_source = base.get('source_inventory_contract')
  inherited_top_level_keys = set(base)
  inherited_rows = (
      inherited_source.get('rows') if isinstance(inherited_source, Mapping)
      else None
  )
  if (
      len(base) != 86 or len(base.get('file_sha256', {})) != 132
      or not isinstance(inherited_rows, list) or len(inherited_rows) != 132
      or base.pop('nonpublication_terminal_contract_v3_3_4_5', None) is None
  ):
    raise RuntimeError('Inherited v3.3.4.5 freeze is not exact 86/132/132.')
  source_head = _source_authority_head()
  prior_archive = bootstrap.validate_prior_v3_3_4_5_controlled_stop_archive()
  prior_archive_binding = _canonical_binding(prior_archive)
  base.update({
      'amendment_commit': bootstrap.AMENDMENT_COMMIT,
      'amendment_path': str(bootstrap.AMENDMENT_PATH.resolve()),
      'amendment_sha256': bootstrap.AMENDMENT_SHA256,
      'analysis_attempt_dir': str(bootstrap.ANALYSIS_ATTEMPT_DIR.resolve()),
      'analysis_dir': str(bootstrap.ANALYSIS_DIR.resolve()),
      'attempt_id': 'v3.3.4.6-development-ood-sidecar-one-shot',
      'cache_isolation_contract': bootstrap.CACHE_ISOLATION_CONTRACT,
      'compiled_backend_equality_is_a_gate': False,
      'denied_cache_environment_names': list(
          bootstrap.DENIED_CACHE_ENVIRONMENT_NAMES
      ),
      'denied_cache_environment_prefixes': list(
          bootstrap.DENIED_CACHE_ENVIRONMENT_PREFIXES
      ),
      'model_kernel_cache_dir': str(
          bootstrap.MODEL_KERNEL_CACHE_DIR.resolve()
      ),
      'output_dir': str(bootstrap.OUTPUT_DIR.resolve()),
      'preflight_dir': str(bootstrap.PREFLIGHT_DIR.resolve()),
      'preflight_kernel_cache_dir': str(
          bootstrap.PREFLIGHT_KERNEL_CACHE_DIR.resolve()
      ),
      'preflight_script_version': 'opensplice-device-preflight-v3.3.4.6',
      'script_version': 'v3.3.4.6',
      'source_program_contract': bootstrap.SOURCE_PROGRAM_CONTRACT,
      'external_freeze_authorization_contract': (
          bootstrap.EXTERNAL_FREEZE_AUTHORIZATION_CONTRACT
      ),
      'program_signature_attestation_contract': {
          **bootstrap.PROGRAM_SIGNATURE_ATTESTATION_CONTRACT,
          'literal_program_signatures': base['program_signatures'],
      },
      'source_input_audit_contract': bootstrap.SOURCE_INPUT_AUDIT_CONTRACT,
      'same_object_attestation_contract': (
          bootstrap.SAME_OBJECT_ATTESTATION_CONTRACT
      ),
      'dispatch_journal_contract': bootstrap.DISPATCH_JOURNAL_CONTRACT,
      'failed_current_contract': bootstrap.FAILED_CURRENT_CONTRACT,
      'raw_record_contract': bootstrap.RAW_RECORD_CONTRACT,
      'raw_manifest_contract': bootstrap.RAW_MANIFEST_CONTRACT,
      'terminal_contract': {
          **bootstrap.TERMINAL_CONTRACT,
          'execution_contract': base['terminal_contract']['execution_contract'],
      },
      'preflight_contract': bootstrap.PREFLIGHT_CONTRACT,
      'compiled_diagnostics_contract': (
          bootstrap.COMPILED_DIAGNOSTICS_CONTRACT
      ),
      'publication_contract_v3_3_4_1': (
          bootstrap.PUBLICATION_CONTRACT_V3_3_4_1
      ),
      'nonpublication_terminal_contract_v3_3_4_6': (
          bootstrap.NONPUBLICATION_TERMINAL_CONTRACT_V3_3_4_6
      ),
      'prior_v3_3_4_3_consumed_preflight_prefix': (
          bootstrap.validate_prior_v3_3_4_3_consumed_preflight_prefix()
      ),
      'prior_v3_3_4_4_consumed_preflight_prefix': (
          bootstrap.validate_prior_v3_3_4_4_consumed_preflight_prefix()
      ),
      'prior_v3_3_4_5_controlled_stop_archive': prior_archive,
      'prior_v3_3_4_5_controlled_stop_archive_content_binding': (
          prior_archive_binding
      ),
      'source_inventory_contract': {},
      'v3_3_3_1_archive': bootstrap.validate_v3_3_3_1_archive(),
  })
  rows = []
  for inherited in inherited_rows:
    if not isinstance(inherited, Mapping):
      raise RuntimeError('Inherited source row is not an object.')
    relative = str(inherited.get('path', ''))
    path = _REPO / relative
    payload = _read_bytes(path)
    digest = hashlib.sha256(payload).hexdigest()
    if (
        digest != inherited.get('sha256')
        or len(payload) != inherited.get('size_bytes')
        or _git_mode(MODEL_HEAD, relative) != inherited.get('git_mode')
        or hashlib.sha256(
            _git_output('show', f'{MODEL_HEAD}:{relative}', binary=True)
        ).hexdigest() != digest
    ):
      raise RuntimeError(f'Inherited source authority changed: {relative}.')
    rows.append({**dict(inherited), 'authority_commit': MODEL_HEAD})
  for relative in sorted(_EXTRA_FILES):
    authority = DOCS_HEAD if relative.endswith(
        'encoder_skip_ood_sidecar_compiler_record_amendment_v3_3_4_6.md'
    ) else source_head
    path = _REPO / relative
    payload = _read_bytes(path)
    digest = hashlib.sha256(payload).hexdigest()
    mode = _git_mode(authority, relative)
    if (
        hashlib.sha256(
            _git_output('show', f'{authority}:{relative}', binary=True)
        ).hexdigest() != digest
        or mode != ('100755' if relative.endswith('.sh') else '100644')
    ):
      raise RuntimeError(f'New source authority changed: {relative}.')
    rows.append({
        'path': relative, 'sha256': digest, 'size_bytes': len(payload),
        'git_mode': mode, 'authority_commit': authority,
    })
  rows.sort(key=lambda item: item['path'])
  if len({row['path'] for row in rows}) != len(rows):
    raise RuntimeError('Source inventory contains duplicate paths.')
  base['file_sha256'] = {
      row['path']: row['sha256'] for row in rows
  }
  base['source_inventory_contract'] = {
      'source_row_count': 144,
      'rows': rows,
      'prospective_upstream_source_file_count': inherited_source[
          'prospective_upstream_source_file_count'
      ],
      'loaded_scientific_module_contract': inherited_source[
          'loaded_scientific_module_contract'
      ],
      'inherited_source_authority_commit': MODEL_HEAD,
      'amendment_authority_commit': DOCS_HEAD,
      'implementation_source_authority_commit': source_head,
  }
  if (
      set(base) != (
          inherited_top_level_keys
          - {'nonpublication_terminal_contract_v3_3_4_5'}
          | {
              'nonpublication_terminal_contract_v3_3_4_6',
              'prior_v3_3_4_5_controlled_stop_archive',
              'prior_v3_3_4_5_controlled_stop_archive_content_binding',
          }
      )
      or len(base) != 88 or len(base['file_sha256']) != 144
      or len(rows) != 144
  ):
    raise RuntimeError('v3.3.4.6 freeze is not exactly 88/144/144.')
  return base


def main() -> None:
  value = build_freeze()
  payload = (
      json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
  ).encode('utf-8')
  descriptor = os.open(
      bootstrap.FREEZE_PATH,
      os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
      0o600,
  )
  try:
    offset = 0
    while offset < len(payload):
      written = os.write(descriptor, payload[offset:])
      if written <= 0:
        raise RuntimeError('Freeze writer made no progress.')
      offset += written
    os.fsync(descriptor)
    os.fchmod(descriptor, 0o644)
    os.fsync(descriptor)
    observed = os.fstat(descriptor)
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o644
        or observed.st_nlink != 1 or observed.st_size != len(payload)
    ):
      raise RuntimeError('Published freeze metadata is not exact.')
    os.lseek(descriptor, 0, os.SEEK_SET)
    readback = bytearray()
    while True:
      block = os.read(descriptor, 1024 * 1024)
      if not block:
        break
      readback.extend(block)
    if bytes(readback) != payload:
      raise RuntimeError('Published freeze readback differs from payload.')
  finally:
    os.close(descriptor)
  parent = os.open(
      bootstrap.FREEZE_PATH.parent,
      os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
  )
  try:
    os.fsync(parent)
  finally:
    os.close(parent)
  print(bootstrap.FREEZE_PATH)
  print(_sha256(bootstrap.FREEZE_PATH))
  print(len(value['file_sha256']))


if __name__ == '__main__':
  main()
