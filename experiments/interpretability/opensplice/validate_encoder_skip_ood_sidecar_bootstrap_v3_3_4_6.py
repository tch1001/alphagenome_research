#!/usr/bin/env python3
"""Standard-library-only pre-import gate for the v3.3.4.6 OOD sidecar."""

from __future__ import annotations

import argparse
import ast
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))
# pylint: disable=g-import-not-at-top
import validate_encoder_skip_bootstrap_v3_3 as v33_bootstrap


AMENDMENT_PATH = (
    _HERE / 'v3_wider_mechanism'
    / 'encoder_skip_ood_sidecar_compiler_record_amendment_v3_3_4_6.md'
)
AMENDMENT_SHA256 = (
    '729127abe69838a7cacb0619774fa93a2b24b00faf7ed144916b2bedfa10b3b2'
)
AMENDMENT_COMMIT = '686b67bd772a7771a6c4540c5b188b6497dbdec7'
PREFLIGHT_SCRIPT_VERSION = 'opensplice-device-preflight-v3.3.4.6'
V3_3_4_AMENDMENT_PATH = (
    _HERE / 'v3_wider_mechanism'
    / 'encoder_skip_ood_sidecar_infrastructure_amendment_v3_3_4.md'
)
V3_3_4_AMENDMENT_SHA256 = (
    '38d07c0b612e50aadc64ba18537561cbdb0489b67fd0824cae749bba6214207b'
)
V3_3_4_AMENDMENT_COMMIT = 'f833a8d2108636871abfce8b4cbabe4255536974'
V3_3_4_1_AMENDMENT_PATH = (
    _HERE / 'v3_wider_mechanism'
    / 'encoder_skip_ood_sidecar_publication_amendment_v3_3_4_1.md'
)
V3_3_4_1_AMENDMENT_SHA256 = (
    '6abc470f6fb14b70c8930195bb8f26ce730b8c07c636cd842d5451f37d8eb55c'
)
V3_3_4_1_AMENDMENT_COMMIT = (
    '2b5e3e93a9961ac7cb12c088f6922acc9fdc5dde'
)
V3_3_4_2_AMENDMENT_PATH = (
    _HERE / 'v3_wider_mechanism'
    / 'encoder_skip_ood_sidecar_nonpublication_terminal_amendment_v3_3_4_2.md'
)
V3_3_4_2_AMENDMENT_SHA256 = (
    '1d2109e58d11cb07e99490bfde5fbb5d5ab43bd12e429c28b4ca9dfc0656fb87'
)
V3_3_4_2_AMENDMENT_COMMIT = (
    'f48d6b73839b428fe950b00696548b6410a52659'
)
ORIGINAL_PROTOCOL_SHA256 = (
    '85151029297cf4b4c1a07dd2a0f47b10b0663acd08ddfe25ba72b9898b1beea0'
)
ORIGINAL_FREEZE_PATH = _HERE / 'encoder_skip_factorial_v3_3_freeze.json'
ORIGINAL_FREEZE_SHA256 = (
    '98860ed4e60c427a76ac05879d800f36b65c10a310f4b2b981819fa48af767b3'
)
ORIGINAL_RUN_DIR = (
    _HERE / 'results' / 'v3_3_development_encoder_skip_factorial_one_shot'
)
FREEZE_PATH = _HERE / 'encoder_skip_ood_sidecar_v3_3_4_6_freeze.json'
OUTPUT_DIR = (
    _HERE / 'results' / 'v3_3_4_6_development_ood_sidecar_one_shot'
)
ANALYSIS_DIR = (
    _HERE / 'results' / 'v3_3_4_6_development_ood_sidecar_analysis'
)
ANALYSIS_ATTEMPT_DIR = (
    _HERE / 'results' / 'v3_3_4_6_development_ood_sidecar_analysis_attempt'
)
PREFLIGHT_DIR = _HERE / 'results' / 'v3_3_4_6_device_preflight'
PREFLIGHT_KERNEL_CACHE_DIR = (
    _HERE / 'results' / 'v3_3_4_6_preflight_kernel_cache'
)
MODEL_KERNEL_CACHE_DIR = _HERE / 'results' / 'v3_3_4_6_model_kernel_cache'

V3_3_4_3_COMMIT = 'ea486661ffe64d5640485ebb2a3ca297e128530a'
V3_3_4_3_FREEZE_PATH = (
    _HERE / 'encoder_skip_ood_sidecar_v3_3_4_3_freeze.json'
)
V3_3_4_3_FREEZE_SHA256 = (
    '713790306dd3d88d735229f497587ab6fe611e435eee3f4ef6b862f7baa3cedc'
)
V3_3_4_3_FREEZE_SIZE_BYTES = 174545
V3_3_4_3_SOURCE_BINDINGS = {
    'experiments/interpretability/opensplice/run_device_preflight_v3_3_4_3.py': (
        '35025f6a03ce0e6c8f260706c286545590fe8fc4135b0bd8f2aa7fe84c538810'
    ),
    'experiments/interpretability/opensplice/launch_encoder_skip_ood_sidecar_v3_3_4_3.py': (
        '459d3fd7646f495f4fc39e788d35f04d16dc517ee4541229307a749e62c58458'
    ),
    'experiments/interpretability/opensplice/validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_3.py': (
        '35f6e5d35e7a2698da65cdc931e07d2a334ab07d22a2d46837f762b0ba5b9a79'
    ),
    'experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_4_3.sh': (
        'a6fffa175bad23089dca62570a9217cbf23a6eaa90afa0cc6bb85b57ce94b4e2'
    ),
}
V3_3_4_3_PREDECESSOR_PATHS = {
    'model_run': (
        _HERE / 'results' / 'v3_3_4_3_development_ood_sidecar_one_shot'
    ),
    'device_preflight': _HERE / 'results' / 'v3_3_4_3_device_preflight',
    'external_cache': _HERE / 'results' / 'v3_3_4_3_preflight_kernel_cache',
    'model_cache': _HERE / 'results' / 'v3_3_4_3_model_kernel_cache',
    'analysis_output': (
        _HERE / 'results' / 'v3_3_4_3_development_ood_sidecar_analysis'
    ),
    'analysis_attempt': (
        _HERE / 'results'
        / 'v3_3_4_3_development_ood_sidecar_analysis_attempt'
    ),
}

V3_3_4_4_COMMIT = '6858bbcdd869ac9ae93064910227003a911d0bd1'
V3_3_4_4_FREEZE_PATH = (
    _HERE / 'encoder_skip_ood_sidecar_v3_3_4_4_freeze.json'
)
V3_3_4_4_FREEZE_SHA256 = (
    '73b26eddf5578ef0847ac69c279c262e6f43102127bbe4299bbdab7e52227e30'
)
V3_3_4_4_FREEZE_SIZE_BYTES = 187923
V3_3_4_4_PREFLIGHT_DIR = _HERE / 'results' / 'v3_3_4_4_device_preflight'
V3_3_4_4_EXTERNAL_CACHE_DIR = (
    _HERE / 'results' / 'v3_3_4_4_preflight_kernel_cache'
)
V3_3_4_4_OTHER_PATHS = {
    'analysis_attempt': (
        _HERE / 'results'
        / 'v3_3_4_4_development_ood_sidecar_analysis_attempt'
    ),
    'analysis_output': (
        _HERE / 'results' / 'v3_3_4_4_development_ood_sidecar_analysis'
    ),
    'model_cache': _HERE / 'results' / 'v3_3_4_4_model_kernel_cache',
    'model_run': (
        _HERE / 'results' / 'v3_3_4_4_development_ood_sidecar_one_shot'
    ),
}
V3_3_4_4_LAUNCHER_BINDING = {
    'path': str(
        (_HERE / 'launch_encoder_skip_ood_sidecar_v3_3_4_4.py').absolute()
    ),
    'sha256': (
        '4cdeee9df8b14043633383b99cdf8d88bdf1c3a0a3e4146a3c40adf9e78991f9'
    ),
    'size_bytes': 27372,
}
V3_3_4_4_BOOTSTRAP_BINDING = {
    'path': str(
        (_HERE / 'validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_4.py')
        .absolute()
    ),
    'sha256': (
        '8e2559d5dae96f6d9ddaa752e2aa4de3829ec85115e78fba356e6fe8c8abccb8'
    ),
    'size_bytes': 143561,
}
V3_3_4_4_CONSUMED_PREFIX_BINDING = {
    'sha256': (
        'efcb6d8946666d104d7458c0f13cc8f53e6dfaa1a30a2e83744f48641978f3c7'
    ),
    'size_bytes': 8653,
}

V3_3_4_5_COMMIT = '0da8f47ea6e576a72a1cda204ce868ef79cc2ce5'
V3_3_4_5_FREEZE_PATH = (
    _HERE / 'encoder_skip_ood_sidecar_v3_3_4_5_freeze.json'
)
V3_3_4_5_FREEZE_SHA256 = (
    '2f4eaf1366dcb42b8f89a386e8201b3f2ba0b9f8ae5ef02409436492666d8366'
)
V3_3_4_5_FREEZE_SIZE_BYTES = 204697
V3_3_4_5_RUN_DIR = (
    _HERE / 'results/v3_3_4_5_development_ood_sidecar_one_shot'
)
V3_3_4_5_PREFLIGHT_DIR = _HERE / 'results/v3_3_4_5_device_preflight'
V3_3_4_5_EXTERNAL_CACHE_DIR = (
    _HERE / 'results/v3_3_4_5_preflight_kernel_cache'
)
V3_3_4_5_MODEL_CACHE_DIR = _HERE / 'results/v3_3_4_5_model_kernel_cache'
V3_3_4_5_RUN_TREE_SHA256 = (
    '960faf1675caaa0f3c9798f7b943998e91650b2a9359b26ca31ce2d417c2ce0b'
)
V3_3_4_5_RUN_DIRECTORY_FILE_TREE_SHA256 = (
    '5331e5041b557a4324ba57d57179dc2bfd8e6ab981f0cdede5d4f126e94c2041'
)
V3_3_4_5_RUN_FILES = {
    'ATTEMPT_STARTED.json': (116707, 'c211bf46f9fd55689da21d02d3b7859f08cd14f27a0419c872047f4a1f3f3f13'),
    'IMPORT_PROVENANCE.json': (59596, 'c6cdd83cf263c4a0c5e745dfb2ea2b0163593fc9dc41c7c9bc543dad5c836d9f'),
    'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json': (59609, '506500e21750e242917ffa15e7ca8187ba20da18555ce3d0030f41a7c2a19147'),
    'IMPORT_PROVENANCE_PRE_MODEL.json': (59597, '4f23789f2817ea0af5d8dffc9b21bc9c3aadd2a0835ce747d8ac599c887e3174'),
    'PROTOBUF_PROVENANCE.json': (3839, 'f5270aef8e2e71e06a66ec310b90c82efae17a7a418fc8713443760f98605880'),
    'RAW_MANIFEST.json': (1562, '3ee95b22d483c7c4f234fbb75281e05e84f0be263b1ee670a94b2cd442d61136'),
    'RUN_COMPLETE.json': (43760, 'fdbd0a1dc7d24145f88c5a009cc80d8904e57920e0c9584426e791373fae6d8f'),
    'compiler/eight_row/COMPILER_DIAGNOSTIC_FAILURE.json': (6018, 'ec8f2d39297e6ea3c0ed633afe58bbf8252114ae0c29de5fc3a70cfd93131881'),
    'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json': (17455, '441b152ff23f802a7adde0a9b53301b44dbebc9f385406eb8775fa06d41bf8ec'),
    'compiler/eight_row/graph.compiled.hlo.txt': (16601615, '524ae897733d9b4b88a9a5572767810166ba7e3c6e91efd82ed370d24de42d99'),
    'compiler/eight_row/graph.pre_backend.hlo.txt': (1829833, '675b9cf26fe8f59a1d138e856ee84f48edb20d8c896ac9e539b6b064dcef2750'),
    'compiler/eight_row/graph.stablehlo.mlir': (3196162, '69dbf2a054cf89e56a9000dc0c04e5cd4ba425f016885ddc127d22c01661bddd'),
}
V3_3_4_5_ANALYSIS_AMENDMENT_COMMIT = (
    '564a01dc2981d57c8f8298f3efca5b22fcb381e0'
)
V3_3_4_5_ANALYSIS_SOURCE_COMMIT = (
    'dfa56d90c035c3aa370c65b79197820dd5787c92'
)
V3_3_4_5_ANALYSIS_FREEZE_COMMIT = (
    'eeadee88b747acc75e9437b5f2d1e7e3aab9701c'
)
V3_3_4_5_ANALYSIS_ARCHIVE_COMMIT = (
    'c292622e5732329cbee50575381682519017ac68'
)
V3_3_4_5_ANALYSIS_AMENDMENT_PATH = (
    _HERE / 'v3_wider_mechanism'
    / 'encoder_skip_ood_sidecar_analysis_amendment_v3_3_4_5_1.md'
)
V3_3_4_5_ANALYSIS_AMENDMENT_SHA256 = (
    '16af8ccb65f3e08739c3792c5c9ab3affcb19a3ca9993904260729a898afd5c4'
)
V3_3_4_5_ANALYSIS_FREEZE_PATH = (
    _HERE / 'encoder_skip_ood_sidecar_analysis_v3_3_4_5_1_freeze.json'
)
V3_3_4_5_ANALYSIS_FREEZE_SHA256 = (
    '3c5405e8d9aadbe8f594fbc262a155a669bfbf67301898dee687aeeb2e286d9f'
)
V3_3_4_5_ANALYSIS_FREEZE_SIZE_BYTES = 85875
V3_3_4_5_ANALYSIS_ATTEMPT_DIR = (
    _HERE / 'results/v3_3_4_5_development_ood_sidecar_analysis_v3_3_4_5_1_attempt'
)
V3_3_4_5_ANALYSIS_DIR = (
    _HERE / 'results/v3_3_4_5_development_ood_sidecar_analysis_v3_3_4_5_1'
)
V3_3_4_5_OLD_ANALYSIS_ATTEMPT_DIR = (
    _HERE / 'results/v3_3_4_5_development_ood_sidecar_analysis_attempt'
)
V3_3_4_5_OLD_ANALYSIS_DIR = (
    _HERE / 'results/v3_3_4_5_development_ood_sidecar_analysis'
)
V3_3_4_5_ANALYSIS_ATTEMPT_FILES = {
    'ANALYSIS_ATTEMPT_STARTED.json': (67913, 'ada53945670529d24c396514e31ed5155c4c3da0c2d88d2d86d93cfc4bbfc9c1'),
    'ANALYSIS_COMPLETE.json': (4672, '1512aefe1613a81ed7532a1c66cb270a005aeb3f1d9a6c82a0558bd702b78277'),
}
V3_3_4_5_ANALYSIS_FILES = {
    'ANALYSIS.json': (18633, '5090675905789aff6a290dcbeeabefe3a8a2938ebc95ecdddd296a3f0ca31a6f'),
    'RESULT.md': (504, '5d2f30d217c3324d79097e256ffa4ab52d3469ba6453c59ac866191ab8bfdffd'),
}

V3_3_4_PREDECESSOR_PATHS = {
    'model_run': _HERE / 'results' / 'v3_3_4_development_ood_sidecar_one_shot',
    'external_preflight': _HERE / 'results' / 'v3_3_4_device_preflight',
    'external_cache': _HERE / 'results' / 'v3_3_4_preflight_kernel_cache',
    'model_cache': _HERE / 'results' / 'v3_3_4_model_kernel_cache',
    'analysis_output': (
        _HERE / 'results' / 'v3_3_4_development_ood_sidecar_analysis'
    ),
    'analysis_attempt': (
        _HERE / 'results' / 'v3_3_4_development_ood_sidecar_analysis_attempt'
    ),
}
V3_3_4_1_PREDECESSOR_PATHS = {
    'model_run': (
        _HERE / 'results' / 'v3_3_4_1_development_ood_sidecar_one_shot'
    ),
    'external_preflight': _HERE / 'results' / 'v3_3_4_1_device_preflight',
    'external_cache': (
        _HERE / 'results' / 'v3_3_4_1_preflight_kernel_cache'
    ),
    'model_cache': _HERE / 'results' / 'v3_3_4_1_model_kernel_cache',
    'analysis_output': (
        _HERE / 'results' / 'v3_3_4_1_development_ood_sidecar_analysis'
    ),
    'analysis_attempt': (
        _HERE / 'results'
        / 'v3_3_4_1_development_ood_sidecar_analysis_attempt'
    ),
}
V3_3_4_2_PREDECESSOR_PATHS = {
    'model_run': (
        _HERE / 'results' / 'v3_3_4_2_development_ood_sidecar_one_shot'
    ),
    'external_preflight': _HERE / 'results' / 'v3_3_4_2_device_preflight',
    'external_cache': (
        _HERE / 'results' / 'v3_3_4_2_preflight_kernel_cache'
    ),
    'model_cache': _HERE / 'results' / 'v3_3_4_2_model_kernel_cache',
    'analysis_output': (
        _HERE / 'results' / 'v3_3_4_2_development_ood_sidecar_analysis'
    ),
    'analysis_attempt': (
        _HERE / 'results'
        / 'v3_3_4_2_development_ood_sidecar_analysis_attempt'
    ),
}

PUBLICATION_SCHEMA_VERSION = 'v3.3.4.6-named-temp-renameat2-noreplace-v1'
PUBLICATION_METHOD = 'named_temp_renameat2_noreplace'
RENAME_NOREPLACE = 1
PUBLICATION_ROOTS = {
    'model_run': OUTPUT_DIR,
    'external_preflight': PREFLIGHT_DIR,
    'external_cache': PREFLIGHT_KERNEL_CACHE_DIR,
    'model_cache': MODEL_KERNEL_CACHE_DIR,
    'analysis_output': ANALYSIS_DIR,
    'analysis_attempt': ANALYSIS_ATTEMPT_DIR,
}
SUCCESSFUL_PUBLICATION_OBJECT_KEYS = (
    'schema_version', 'method', 'root_role', 'final_relative_path',
    'temp_basename', 'publication_ordinal', 'runner_pid', 'nonce_hex',
    'sha256', 'size_bytes', 'mode', 'st_dev', 'st_ino', 'st_nlink',
    'file_fsync_before_rename', 'file_fsync_after_fchmod',
    'rename_noreplace_succeeded', 'parent_fsync_succeeded',
    'post_publish_revalidation_exact',
)
PUBLICATION_FAILURE_OBJECT_KEYS = (
    'schema_version', 'method', 'root_role', 'artifact_role',
    'final_relative_path', 'temp_relative_path', 'publication_ordinal',
    'runner_pid', 'failure_stage', 'errno', 'error_type', 'message',
    'rename_noreplace_attempted', 'rename_noreplace_succeeded',
    'parent_fsync_attempted', 'parent_fsync_succeeded', 'temp_state',
    'final_state', 'created_at_unix_s',
)
ENTRY_STATE_OBJECT_KEYS = (
    'state', 'entry_type', 'mode', 'size_bytes', 'sha256', 'st_dev',
    'st_ino', 'st_nlink',
)
PUBLICATION_CONTRACT_V3_3_4_1 = {
    'schema_version': PUBLICATION_SCHEMA_VERSION,
    'method': PUBLICATION_METHOD,
    'temp_name_regex': (
        r'^\.v3345\.tmp\.[1-9][0-9]*\.[0-9]{6}\.[0-9a-f]{32}$'
    ),
    'nonce_bytes': 16,
    'open_flags': [
        'O_RDWR', 'O_CREAT', 'O_EXCL', 'O_NOFOLLOW', 'O_CLOEXEC'
    ],
    'initial_mode': '0600',
    'sealed_mode': '0400',
    'rename_flags': ['RENAME_NOREPLACE'],
    'same_directory_required': True,
    'keep_fd_open_through_rename': True,
    'file_fsync_count': 2,
    'parent_fsync_required': True,
    'post_publish_inode_revalidation_required': True,
    'no_replace': True,
    'no_fallback': True,
    'no_retry': True,
    'temporary_orphan_preservation_required': True,
    'durability_uncertain_final_preservation_required': True,
    'successful_publication_object_keys': list(
        SUCCESSFUL_PUBLICATION_OBJECT_KEYS
    ),
    'publication_failure_object_keys': list(
        PUBLICATION_FAILURE_OBJECT_KEYS
    ),
    'entry_state_object_keys': list(ENTRY_STATE_OBJECT_KEYS),
    'external_preflight_probe_contract': {
        'final_basename': 'atomic_publication_probe_v3_3_4_6.txt',
        'final_sha256': (
            'db249bd851ab5028d95d71a10d6edad91f15b18e023722a3086383f7aab04a65'
        ),
        'final_size_bytes': 49,
        'collision_sha256': (
            'd9eaea234b10d919daaabddf3d560d34c946ba3259bb1e16cfd9a005a05d85b2'
        ),
        'collision_size_bytes': 39,
        'collision_errno': errno.EEXIST,
        'collision_temp_preserved': True,
        'parent_fsync_exact_required': True,
    },
}

_PUBLICATION_ORDINAL = 0
_PUBLICATION_SUCCESS: dict[str, dict[str, dict[str, Any]]] = {}
_PUBLICATION_TEMP_ORPHANS: dict[str, dict[str, dict[str, Any]]] = {}
_PUBLICATION_UNCERTAIN_FINALS: dict[str, dict[str, dict[str, Any]]] = {}
_PUBLICATION_PREEXISTING: dict[str, dict[str, dict[str, Any]]] = {}
_PUBLICATION_DIRECTORIES: dict[tuple[str, str], tuple[int, int]] = {}
_PUBLICATION_UNBINDABLE_FAILURES: dict[str, dict[str, Any]] = {}


class PublicationError(RuntimeError):
  """Exact preserved no-replace publication failure."""

  def __init__(self, publication_failure: Mapping[str, Any]):
    self.publication_failure = dict(publication_failure)
    super().__init__(self.publication_failure['message'])


def _publication_mode(mode: int) -> str:
  return f'{stat.S_IMODE(mode):04o}'


def _entry_type(mode: int) -> str:
  if stat.S_ISREG(mode):
    return 'regular'
  if stat.S_ISDIR(mode):
    return 'directory'
  if stat.S_ISLNK(mode):
    return 'symlink'
  if stat.S_ISFIFO(mode):
    return 'fifo'
  if stat.S_ISSOCK(mode):
    return 'socket'
  if stat.S_ISBLK(mode):
    return 'block'
  if stat.S_ISCHR(mode):
    return 'character'
  return 'other'


def _read_fd_bytes(fd: int) -> bytes:
  os.lseek(fd, 0, os.SEEK_SET)
  result = bytearray()
  while True:
    block = os.read(fd, 1024 * 1024)
    if not block:
      return bytes(result)
    result.extend(block)


def _entry_state(parent_fd: int, name: str) -> dict[str, Any]:
  try:
    observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
  except FileNotFoundError:
    return {
        'state': 'absent', 'entry_type': None, 'mode': None,
        'size_bytes': None, 'sha256': None, 'st_dev': None,
        'st_ino': None, 'st_nlink': None,
    }
  except OSError:
    return {
        'state': 'unreadable', 'entry_type': None, 'mode': None,
        'size_bytes': None, 'sha256': None, 'st_dev': None,
        'st_ino': None, 'st_nlink': None,
    }
  kind = _entry_type(observed.st_mode)
  digest = None
  size = None
  if kind == 'regular':
    try:
      fd = os.open(
          name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
          dir_fd=parent_fd,
      )
      try:
        opened = os.fstat(fd)
        if (
            opened.st_dev != observed.st_dev
            or opened.st_ino != observed.st_ino
            or not stat.S_ISREG(opened.st_mode)
        ):
          return {
              'state': 'unreadable', 'entry_type': None, 'mode': None,
              'size_bytes': None, 'sha256': None, 'st_dev': None,
              'st_ino': None, 'st_nlink': None,
          }
        payload = _read_fd_bytes(fd)
      finally:
        os.close(fd)
      digest = hashlib.sha256(payload).hexdigest()
      size = len(payload)
    except OSError:
      return {
          'state': 'unreadable', 'entry_type': None, 'mode': None,
          'size_bytes': None, 'sha256': None, 'st_dev': None,
          'st_ino': None, 'st_nlink': None,
      }
  return {
      'state': 'present', 'entry_type': kind,
      'mode': _publication_mode(observed.st_mode), 'size_bytes': size,
      'sha256': digest, 'st_dev': observed.st_dev,
      'st_ino': observed.st_ino, 'st_nlink': observed.st_nlink,
  }


def _entry_state_path(path: Path) -> dict[str, Any]:
  try:
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
  except OSError:
    return {
        'state': 'unreadable', 'entry_type': None, 'mode': None,
        'size_bytes': None, 'sha256': None, 'st_dev': None,
        'st_ino': None, 'st_nlink': None,
    }
  try:
    return _entry_state(parent_fd, path.name)
  finally:
    os.close(parent_fd)


def _next_publication_identity() -> tuple[int, str, str]:
  global _PUBLICATION_ORDINAL  # pylint: disable=global-statement
  ordinal = _PUBLICATION_ORDINAL
  _PUBLICATION_ORDINAL += 1
  nonce = secrets.token_hex(16)
  temporary = f'.v3345.tmp.{os.getpid()}.{ordinal:06d}.{nonce}'
  return ordinal, nonce, temporary


def _rename_noreplace(
    parent_fd: int, temporary_basename: str, final_basename: str
) -> tuple[int, int]:
  """Issues the sole renameat2 call and captures errno immediately."""
  libc = ctypes.CDLL(None, use_errno=True)
  libc.renameat2.argtypes = (
      ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
      ctypes.c_uint,
  )
  libc.renameat2.restype = ctypes.c_int
  ctypes.set_errno(0)
  result = libc.renameat2(
      parent_fd, temporary_basename.encode('ascii'),
      parent_fd, final_basename.encode('ascii'), RENAME_NOREPLACE,
  )
  return result, ctypes.get_errno()


def _publication_directory(root_role: str, relative: str) -> Path:
  if root_role not in PUBLICATION_ROOTS:
    raise ValueError(f'Unknown publication root role: {root_role}.')
  root = PUBLICATION_ROOTS[root_role]
  target = root if relative == '.' else root / relative
  if target.resolve().is_relative_to(root.resolve()) is not True:
    raise ValueError('Publication directory escaped its frozen root.')
  if any(part in ('', '.', '..') for part in Path(relative).parts):
    if relative != '.':
      raise ValueError('Invalid publication directory component.')
  return target


def require_publication_directory(root_role: str, relative: str = '.') -> Path:
  target = _publication_directory(root_role, relative)
  observed = target.lstat()
  if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
    raise ValueError('Publication path is not a regular directory.')
  if _publication_mode(observed.st_mode) != '0700':
    raise ValueError('Publication directory mode is not 0700.')
  if observed.st_dev != PUBLICATION_ROOTS[root_role].lstat().st_dev:
    raise ValueError('Publication directory device changed.')
  registered = _PUBLICATION_DIRECTORIES.get((root_role, relative))
  if registered is not None and registered != (observed.st_dev, observed.st_ino):
    raise ValueError('Registered publication directory inode changed.')
  return target


def allocate_publication_directory(
    root_role: str, relative: str = '.', *, register_existing: bool = False
) -> Path:
  """Allocates once or registers a lifecycle-authorized existing directory."""
  target = _publication_directory(root_role, relative)
  key = (root_role, relative)
  if key in _PUBLICATION_DIRECTORIES:
    return require_publication_directory(root_role, relative)
  final_identity = None
  if target.exists() or target.is_symlink():
    if not register_existing:
      raise FileExistsError(
          f'Publication directory already exists before allocation: {target}.'
      )
    target_fd = os.open(
        target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    try:
      observed = os.fstat(target_fd)
      root_device = PUBLICATION_ROOTS[root_role].lstat().st_dev
      if (
          not stat.S_ISDIR(observed.st_mode)
          or _publication_mode(observed.st_mode) != '0700'
          or observed.st_dev != root_device
      ):
        raise RuntimeError(
            'Existing publication directory validation failed.'
        )
      final_identity = (observed.st_dev, observed.st_ino)
    finally:
      os.close(target_fd)
  else:
    parent = target.parent
    if relative == '.':
      parent_observed = parent.lstat()
      if not stat.S_ISDIR(parent_observed.st_mode) or stat.S_ISLNK(
          parent_observed.st_mode
      ):
        raise RuntimeError('Publication root parent is not a directory.')
      expected_parent = (parent_observed.st_dev, parent_observed.st_ino)
    else:
      parent_relative = Path(relative).parent.as_posix()
      require_publication_directory(root_role, parent_relative)
      expected_parent = _PUBLICATION_DIRECTORIES.get(
          (root_role, parent_relative)
      )
      if expected_parent is None:
        raise RuntimeError('Publication parent was not registered.')
    parent_fd = os.open(
        parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    try:
      opened_parent = os.fstat(parent_fd)
      if (
          not stat.S_ISDIR(opened_parent.st_mode)
          or (
              relative != '.'
              and _publication_mode(opened_parent.st_mode) != '0700'
          )
          or (opened_parent.st_dev, opened_parent.st_ino) != expected_parent
      ):
        raise RuntimeError(
            'Opened publication parent does not match its registered inode.'
        )
      os.mkdir(target.name, 0o700, dir_fd=parent_fd)
      os.fsync(parent_fd)
      observed = os.stat(
          target.name, dir_fd=parent_fd, follow_symlinks=False
      )
      if (
          not stat.S_ISDIR(observed.st_mode)
          or _publication_mode(observed.st_mode) != '0700'
      ):
        raise RuntimeError('New publication directory validation failed.')
      final_identity = (observed.st_dev, observed.st_ino)
    finally:
      os.close(parent_fd)
  if final_identity is None:
    raise RuntimeError('Publication directory identity was not captured.')
  _PUBLICATION_DIRECTORIES[key] = final_identity
  return require_publication_directory(root_role, relative)


def ensure_publication_directory(root_role: str, relative: str = '.') -> Path:
  """Allocates a never-seen directory and never recreates a consumed one."""
  key = (root_role, relative)
  if key in _PUBLICATION_DIRECTORIES:
    return require_publication_directory(root_role, relative)
  target = _publication_directory(root_role, relative)
  if target.exists() or target.is_symlink():
    raise FileExistsError(
        f'Unregistered publication directory already exists: {target}.'
    )
  if relative != '.':
    parent_relative = Path(relative).parent.as_posix()
    require_publication_directory(root_role, parent_relative)
  return allocate_publication_directory(root_role, relative)


def ensure_publication_parent(
    root_role: str, final_relative_path: str
) -> Path:
  """Allocates every parent once, starting from an allocated root."""
  relative = Path(final_relative_path)
  if relative.is_absolute() or '..' in relative.parts or not relative.parts:
    raise ValueError('Publication final path must be root-relative.')
  require_publication_directory(root_role, '.')
  current = Path('.')
  for part in relative.parent.parts:
    if part in ('', '.'):
      continue
    current /= part
    key = (root_role, current.as_posix())
    if key in _PUBLICATION_DIRECTORIES:
      require_publication_directory(root_role, current.as_posix())
    else:
      ensure_publication_directory(root_role, current.as_posix())
  return _publication_directory(root_role, relative.parent.as_posix())


class PublicationHandle:
  """One named temporary held open through the sole renameat2 call."""

  def __init__(
      self, root_role: str, final_relative_path: str, artifact_role: str,
      *, allow_existing_final_for_probe: bool = False,
  ):
    if root_role not in PUBLICATION_ROOTS:
      raise ValueError(f'Unknown publication root role: {root_role}.')
    self.root_role = root_role
    self.root = PUBLICATION_ROOTS[root_role].resolve()
    relative = Path(final_relative_path)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
      raise ValueError('Publication final path must be root-relative.')
    self.final_relative_path = relative.as_posix()
    self.final_basename = relative.name
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', self.final_basename) or (
        self.final_basename in ('.', '..')
    ):
      raise ValueError('Publication final basename is invalid.')
    self.artifact_role = artifact_role
    self.ordinal, self.nonce, self.temp_basename = (
        _next_publication_identity()
    )
    self.parent = self.root / relative.parent
    self.parent_fd = -1
    self.temp_fd = -1
    self.rename_attempted = False
    self.rename_succeeded = False
    self.parent_fsync_attempted = False
    self.parent_fsync_succeeded = False
    self.created_temp = False
    self.stage = 'parent_open'
    self._allow_existing_final = allow_existing_final_for_probe
    try:
      parent_relative = relative.parent.as_posix()
      require_publication_directory(root_role, parent_relative)
      expected_parent = _PUBLICATION_DIRECTORIES.get(
          (root_role, parent_relative)
      )
      if expected_parent is None:
        raise RuntimeError('Publication parent was not registered.')
      self.parent_fd = os.open(
          self.parent,
          os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
      )
      self.stage = 'parent_validation'
      parent_stat = os.fstat(self.parent_fd)
      root_stat = os.stat(self.root, follow_symlinks=False)
      if (
          not stat.S_ISDIR(parent_stat.st_mode)
          or _publication_mode(parent_stat.st_mode) != '0700'
          or parent_stat.st_dev != root_stat.st_dev
          or (parent_stat.st_dev, parent_stat.st_ino) != expected_parent
      ):
        raise RuntimeError('Publication parent validation failed.')
      self.stage = 'final_preexistence'
      final_state = _entry_state(self.parent_fd, self.final_basename)
      if final_state['state'] != 'absent' and not self._allow_existing_final:
        raise FileExistsError(errno.EEXIST, 'Final entry already exists.')
      self.stage = 'temp_open'
      self.temp_fd = os.open(
          self.temp_basename,
          os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
          0o600,
          dir_fd=self.parent_fd,
      )
      self.created_temp = True
      self.stage = 'temp_validation'
      opened = os.fstat(self.temp_fd)
      if (
          not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
          or _publication_mode(opened.st_mode) != '0600'
          or opened.st_dev != parent_stat.st_dev
      ):
        raise RuntimeError('Publication temporary validation failed.')
    except Exception as error:
      self._raise(error)

  def write(self, payload: bytes) -> None:
    self.stage = 'write'
    view = memoryview(payload)
    try:
      while view:
        written = os.write(self.temp_fd, view)
        if written < 1:
          raise OSError('Publication write made no progress.')
        view = view[written:]
    except Exception as error:
      self._raise(error)

  def finalize(self, expected_payload: bytes | None = None) -> dict[str, Any]:
    try:
      self.stage = 'first_file_fsync'
      os.fsync(self.temp_fd)
      self.stage = 'fchmod'
      os.fchmod(self.temp_fd, 0o400)
      self.stage = 'second_file_fsync'
      os.fsync(self.temp_fd)
      self.stage = 'readback'
      payload = _read_fd_bytes(self.temp_fd)
      if expected_payload is not None and payload != expected_payload:
        raise RuntimeError('Publication readback differs from payload.')
      digest = hashlib.sha256(payload).hexdigest()
      temp_stat = os.fstat(self.temp_fd)
      if (
          not stat.S_ISREG(temp_stat.st_mode) or temp_stat.st_nlink != 1
          or _publication_mode(temp_stat.st_mode) != '0400'
          or temp_stat.st_dev != os.fstat(self.parent_fd).st_dev
      ):
        raise RuntimeError('Sealed publication temporary validation failed.')
      self.stage = 'rename_noreplace'
      self.rename_attempted = True
      result, rename_errno = _rename_noreplace(
          self.parent_fd, self.temp_basename, self.final_basename
      )
      if result != 0:
        raise OSError(
            rename_errno, os.strerror(rename_errno), self.final_basename
        )
      self.rename_succeeded = True
      self.stage = 'post_rename_validation'
      self._validate_final(temp_stat, payload, digest)
      self.stage = 'parent_fsync'
      self.parent_fsync_attempted = True
      os.fsync(self.parent_fd)
      self.parent_fsync_succeeded = True
      self.stage = 'final_revalidation'
      observed = self._validate_final(temp_stat, payload, digest)
      success = {
          'schema_version': PUBLICATION_SCHEMA_VERSION,
          'method': PUBLICATION_METHOD,
          'root_role': self.root_role,
          'final_relative_path': self.final_relative_path,
          'temp_basename': self.temp_basename,
          'publication_ordinal': self.ordinal,
          'runner_pid': os.getpid(),
          'nonce_hex': self.nonce,
          'sha256': digest,
          'size_bytes': len(payload),
          'mode': '0400',
          'st_dev': observed.st_dev,
          'st_ino': observed.st_ino,
          'st_nlink': observed.st_nlink,
          'file_fsync_before_rename': True,
          'file_fsync_after_fchmod': True,
          'rename_noreplace_succeeded': True,
          'parent_fsync_succeeded': True,
          'post_publish_revalidation_exact': True,
      }
      if set(success) != set(SUCCESSFUL_PUBLICATION_OBJECT_KEYS):
        raise RuntimeError('Successful publication schema changed.')
      _PUBLICATION_SUCCESS.setdefault(self.root_role, {})[
          self.final_relative_path
      ] = publication_file_binding(success)
      self.close()
      return success
    except Exception as error:
      self._raise(error)
    raise AssertionError('Unreachable publication finalizer.')

  def _validate_final(
      self, temp_stat: os.stat_result, payload: bytes, digest: str
  ) -> os.stat_result:
    if _entry_state(self.parent_fd, self.temp_basename)['state'] != 'absent':
      raise RuntimeError('Publication temporary remained after rename.')
    final_fd = os.open(
        self.final_basename,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=self.parent_fd,
    )
    try:
      observed = os.fstat(final_fd)
      final_payload = _read_fd_bytes(final_fd)
    finally:
      os.close(final_fd)
    if (
        not stat.S_ISREG(observed.st_mode)
        or _publication_mode(observed.st_mode) != '0400'
        or len(final_payload) != len(payload)
        or hashlib.sha256(final_payload).hexdigest() != digest
        or observed.st_dev != temp_stat.st_dev
        or observed.st_ino != temp_stat.st_ino or observed.st_nlink != 1
    ):
      raise RuntimeError('Published final revalidation failed.')
    return observed

  def _raise(self, error: Exception) -> None:
    number = error.errno if isinstance(error, OSError) else None
    temp_state = (
        _entry_state(self.parent_fd, self.temp_basename)
        if self.parent_fd >= 0 else _entry_state_path(
            self.parent / self.temp_basename
        )
    )
    final_state = (
        _entry_state(self.parent_fd, self.final_basename)
        if self.parent_fd >= 0 else _entry_state_path(
            self.parent / self.final_basename
        )
    )
    failure = {
        'schema_version': PUBLICATION_SCHEMA_VERSION,
        'method': PUBLICATION_METHOD,
        'root_role': self.root_role,
        'artifact_role': self.artifact_role,
        'final_relative_path': self.final_relative_path,
        'temp_relative_path': (
            (Path(self.final_relative_path).parent / self.temp_basename)
            .as_posix()
        ),
        'publication_ordinal': self.ordinal,
        'runner_pid': os.getpid(),
        'failure_stage': self.stage,
        'errno': number,
        'error_type': type(error).__name__,
        'message': str(error),
        'rename_noreplace_attempted': self.rename_attempted,
        'rename_noreplace_succeeded': self.rename_succeeded,
        'parent_fsync_attempted': self.parent_fsync_attempted,
        'parent_fsync_succeeded': self.parent_fsync_succeeded,
        'temp_state': temp_state,
        'final_state': final_state,
        'created_at_unix_s': time.time(),
    }
    if set(failure) != set(PUBLICATION_FAILURE_OBJECT_KEYS):
      raise RuntimeError('Publication failure schema changed.') from error
    self._record_failure(failure)
    self.close()
    raise PublicationError(failure) from error

  def _record_failure(self, failure: Mapping[str, Any]) -> None:
    parent = Path(self.final_relative_path).parent
    temp_path = (parent / self.temp_basename).as_posix()
    if self.rename_succeeded:
      binding = _state_file_binding(failure['final_state'])
      if binding is not None:
        _PUBLICATION_UNCERTAIN_FINALS.setdefault(self.root_role, {})[
            self.final_relative_path
        ] = binding
      else:
        _PUBLICATION_UNBINDABLE_FAILURES[self.root_role] = dict(failure)
      if failure['temp_state']['state'] != 'absent':
        _PUBLICATION_PREEXISTING.setdefault(self.root_role, {})[
            temp_path
        ] = dict(failure['temp_state'])
    elif self.created_temp and failure['temp_state']['state'] == 'present':
      binding = _state_file_binding(failure['temp_state'])
      if binding is not None:
        _PUBLICATION_TEMP_ORPHANS.setdefault(self.root_role, {})[
            temp_path
        ] = binding
      else:
        _PUBLICATION_UNBINDABLE_FAILURES[self.root_role] = dict(failure)
      if failure['final_state']['state'] != 'absent':
        _PUBLICATION_PREEXISTING.setdefault(self.root_role, {})[
            self.final_relative_path
        ] = dict(failure['final_state'])
    else:
      for path, state_value in (
          (temp_path, failure['temp_state']),
          (self.final_relative_path, failure['final_state']),
      ):
        if state_value['state'] != 'absent':
          _PUBLICATION_PREEXISTING.setdefault(self.root_role, {})[path] = (
              dict(state_value)
          )

  def close(self) -> None:
    if self.temp_fd >= 0:
      os.close(self.temp_fd)
      self.temp_fd = -1
    if self.parent_fd >= 0:
      os.close(self.parent_fd)
      self.parent_fd = -1


def _absent_entry_state() -> dict[str, Any]:
  return {
      'state': 'absent', 'entry_type': None, 'mode': None,
      'size_bytes': None, 'sha256': None, 'st_dev': None,
      'st_ino': None, 'st_nlink': None,
  }


def _state_file_binding(
    state_value: Mapping[str, Any]
) -> dict[str, Any] | None:
  if (
      state_value.get('state') != 'present'
      or state_value.get('entry_type') != 'regular'
      or any(state_value.get(name) is None for name in (
          'sha256', 'size_bytes', 'mode', 'st_dev', 'st_ino', 'st_nlink'
      ))
  ):
    return None
  return {
      'sha256': state_value['sha256'],
      'size_bytes': state_value['size_bytes'],
      'mode': state_value['mode'],
      'st_dev': state_value['st_dev'],
      'st_ino': state_value['st_ino'],
      'st_nlink': state_value['st_nlink'],
  }


def publication_file_binding(
    success: Mapping[str, Any]
) -> dict[str, Any]:
  return {
      'sha256': success['sha256'], 'size_bytes': success['size_bytes'],
      'mode': success['mode'], 'st_dev': success['st_dev'],
      'st_ino': success['st_ino'], 'st_nlink': success['st_nlink'],
  }


def publish_bytes(
    root_role: str, final_relative_path: str, payload: bytes,
    *, artifact_role: str,
    allow_existing_final_for_probe: bool = False,
) -> dict[str, Any]:
  if root_role in _PUBLICATION_UNBINDABLE_FAILURES:
    raise RuntimeError(
        'Publication root has an unbindable created entry; terminal '
        'publication is prohibited.'
    )
  handle = PublicationHandle(
      root_role, final_relative_path, artifact_role,
      allow_existing_final_for_probe=allow_existing_final_for_probe,
  )
  handle.write(payload)
  return handle.finalize(payload)


def open_publication_stream(
    root_role: str, final_relative_path: str, *, artifact_role: str
) -> PublicationHandle:
  if root_role in _PUBLICATION_UNBINDABLE_FAILURES:
    raise RuntimeError(
        'Publication root has an unbindable created entry; terminal '
        'publication is prohibited.'
    )
  return PublicationHandle(root_role, final_relative_path, artifact_role)


def terminal_publication_available(root_role: str) -> bool:
  """Whether every created entry can still be represented losslessly."""
  return root_role not in _PUBLICATION_UNBINDABLE_FAILURES


def create_empty_lifecycle_file(
    root_role: str, relative_path: str, *, mode: int
) -> dict[str, Any]:
  """Creates one frozen empty allocation marker with no-replace semantics."""
  relative = Path(relative_path)
  if relative.is_absolute() or '..' in relative.parts or not relative.parts:
    raise ValueError('Lifecycle-file path must be root-relative.')
  ensure_publication_parent(root_role, relative.as_posix())
  parent = _publication_directory(root_role, relative.parent.as_posix())
  parent_relative = relative.parent.as_posix()
  expected_parent = _PUBLICATION_DIRECTORIES.get(
      (root_role, parent_relative)
  )
  if expected_parent is None:
    raise RuntimeError('Lifecycle publication parent was not registered.')
  parent_fd = os.open(
      parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
  )
  fd = -1
  try:
    opened_parent = os.fstat(parent_fd)
    if (
        not stat.S_ISDIR(opened_parent.st_mode)
        or _publication_mode(opened_parent.st_mode) != '0700'
        or (opened_parent.st_dev, opened_parent.st_ino) != expected_parent
    ):
      raise RuntimeError(
          'Opened lifecycle parent does not match its registered inode.'
      )
    fd = os.open(
        relative.name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        mode,
        dir_fd=parent_fd,
    )
    observed = os.fstat(fd)
    if (
        not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1
        or stat.S_IMODE(observed.st_mode) != mode or observed.st_size != 0
    ):
      raise RuntimeError('Lifecycle allocation-file validation failed.')
    os.fsync(fd)
    os.fsync(parent_fd)
  finally:
    if fd >= 0:
      os.close(fd)
    os.close(parent_fd)
  return {
      'path': relative.as_posix(), 'sha256': hashlib.sha256(b'').hexdigest(),
      'size_bytes': 0, 'mode': f'{mode:04o}',
  }


def publication_audit(
    root_role: str, publication_failure: Mapping[str, Any] | None = None
) -> dict[str, Any]:
  success = dict(sorted(_PUBLICATION_SUCCESS.get(root_role, {}).items()))
  temporary = dict(sorted(
      _PUBLICATION_TEMP_ORPHANS.get(root_role, {}).items()
  ))
  uncertain = dict(sorted(
      _PUBLICATION_UNCERTAIN_FINALS.get(root_role, {}).items()
  ))
  preexisting = dict(sorted(
      _PUBLICATION_PREEXISTING.get(root_role, {}).items()
  ))
  return {
      'schema_version': PUBLICATION_SCHEMA_VERSION,
      'method': PUBLICATION_METHOD,
      'successful_final_count_before_terminal': len(success),
      'successful_final_bindings_before_terminal': success,
      'temporary_orphan_count': len(temporary),
      'temporary_orphan_bindings': temporary,
      'durability_uncertain_final_count': len(uncertain),
      'durability_uncertain_final_bindings': uncertain,
      'preexisting_entry_count': len(preexisting),
      'preexisting_entry_states': preexisting,
      'no_new_entry_failure': bool(
          publication_failure is not None
          and not temporary and not uncertain
      ),
      'publication_failure': (
          None if publication_failure is None else dict(publication_failure)
      ),
      'no_published_final_deleted': True,
      'no_temp_or_final_reused': True,
      'no_publication_retry': True,
  }

V3_3_3_FREEZE_PATH = _HERE / 'encoder_skip_ood_sidecar_v3_3_3_freeze.json'
V3_3_3_FREEZE_SHA256 = (
    '0e4c16a306f734e016c64509a3b7f0d76f26baf399ee0b1d41c6fb073203741b'
)
V3_3_3_RUN_DIR = (
    _HERE / 'results' / 'v3_3_3_development_ood_sidecar_one_shot'
)
V3_3_3_RUN_COMMIT = '228083b931dbc62d4a283e68df01011f5ef4bff9'
V3_3_3_ANALYSIS_ATTEMPT_DIR = (
    _HERE / 'results' / 'v3_3_3_development_ood_sidecar_analysis_attempt'
)
V3_3_3_ANALYSIS_DIR = (
    _HERE / 'results' / 'v3_3_3_development_ood_sidecar_analysis'
)
V3_3_3_1_ATTEMPT_DIR = (
    _HERE / 'results'
    / 'v3_3_3_development_ood_sidecar_analysis_v3_3_3_1_attempt'
)
V3_3_3_1_ANALYSIS_DIR = (
    _HERE / 'results' / 'v3_3_3_development_ood_sidecar_analysis_v3_3_3_1'
)
V3_3_3_1_AMENDMENT_COMMIT = 'd2a013944a399ddac59a023d7d84ea5a7c23e9f4'
V3_3_3_1_IMPLEMENTATION_COMMIT = (
    '98c467ae16200071d110c9d73520e35e5e6d7bbf'
)
V3_3_3_1_ARCHIVE_COMMIT = '37bd58e88e1814f9a67bfbaaaad66d0a2b77f242'
V3_3_3_RUN_FILES = {
    'ATTEMPT_STARTED.json': (871020, 'e5f7c33f2e8c82af51ed98a3884d7df83e1828e92e322df8aa8a054ec7464c65'),
    'IMPORT_PROVENANCE.json': (41572, 'aa5072c505ebe54d0a7812a7fa3e6bda249a74c17d2b90df3f235c2f4cd6bb4b'),
    'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json': (41572, 'aa5072c505ebe54d0a7812a7fa3e6bda249a74c17d2b90df3f235c2f4cd6bb4b'),
    'IMPORT_PROVENANCE_PRE_MODEL.json': (41572, 'aa5072c505ebe54d0a7812a7fa3e6bda249a74c17d2b90df3f235c2f4cd6bb4b'),
    'PROTOBUF_PROVENANCE.json': (3339, '2498a940f6ee15e54e72e8f51587d4c42ffc1b49851873c31ad09085315d0ba8'),
    'RAW_MANIFEST.json': (145, 'fadbff5ab512dea3d25edb39dc9a91ffe25473b73f1e33fd53f97b60fc8436fd'),
    'RUN_COMPLETE.json': (227159, '43e0ff055e9f7fa4032a75120c551a2b5762e4fbd85119e80e3694f8b9f54bba'),
    'compiler/eight_row/COMPILER_PROVENANCE.json': (102245, 'ae07b0f10784ea3c6dd26d2b87eb718c5e28d3834112ae4f0566d1c4fb7e3125'),
    'compiler/eight_row/graph.compiled.hlo.txt': (16603075, 'f0fe2fa0b7e8326390c8f2ed38ce52ef6c64c355bc462450d85d3b2f040645f4'),
    'compiler/eight_row/graph.pre_backend.hlo.txt': (1829833, '675b9cf26fe8f59a1d138e856ee84f48edb20d8c896ac9e539b6b064dcef2750'),
    'compiler/eight_row/graph.stablehlo.mlir': (3196162, '69dbf2a054cf89e56a9000dc0c04e5cd4ba425f016885ddc127d22c01661bddd'),
}
V3_3_3_RUN_TREE_SHA256 = 'bb13aa4de212c3896781401374057bc0cdfc0c7527772cc36b08b57c70451805'
V3_3_3_COMPILER_TREE_SHA256 = '7ee5ad1bb94ecbd97606fcccae3abcad6b0ebec74dd9f983d81b4fc179142ef0'
V3_3_3_1_ATTEMPT_FILES = {
    'ANALYSIS_ATTEMPT_STARTED.json': (6512, '497374d68c245c30fb0a54968859b9066d1bc16085146b978070bb092ff23bda'),
    'ANALYSIS_COMPLETE.json': (1179, 'e050e091743262e989693c59f5e1fcb2939190a71ee4851c5d2a345c1827c4be'),
}
V3_3_3_1_ANALYSIS_FILES = {
    'ANALYSIS.json': (10060, 'f1e20b3ca4f111854b22eff1e2cd2ffdb05796d800d2831eedcc6caa1a3b7245'),
    'RESULT.md': (695, '8ba2721c8bc350a564f4d5ffdabd65b118f60d92cbdb8ea00a8d040842012e65'),
}
V3_3_3_1_SOURCE_BINDINGS = {
    'experiments/interpretability/opensplice/v3_wider_mechanism/encoder_skip_ood_sidecar_analysis_amendment_v3_3_3_1.md': '4d2957d144e56e58c5b2058076bbcdb7f1495f3172e1b8829a0affa10a0ea4a9',
    'experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_3_1.py': 'f433221f38408ee06d3bdb2c1119ae050720652ee4ec513a0b91f2d7814da063',
    'experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_3_1_test.py': '4f2e70a8f61bb1b9af7b2b98ef8f450d0937855a69d1bc83fbec9d06f21dd971',
    'experiments/interpretability/opensplice/encoder_skip_ood_sidecar_analysis_v3_3_3_1_freeze.json': '96c599f3c607107b8c7ab235d7c8cef7aa1bc544189b44b15b6f3fbf1a8b3291',
    'experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_analysis_v3_3_3_1.sh': '63a0cc95596d47ee5900fe928e1bb42115b18157f87bdae45000e5bb7ccef5c9',
}

V3_3_2_FREEZE_PATH = _HERE / 'encoder_skip_ood_sidecar_v3_3_2_freeze.json'
V3_3_2_FREEZE_SHA256 = (
    'baa257dfa8d0bd1afca40b11038d71c9157499eeb654523ad59df119eb2c0295'
)
V3_3_2_RUN_DIR = (
    _HERE / 'results' / 'v3_3_2_development_ood_sidecar_one_shot'
)
V3_3_2_RUN_COMMIT = '24e2214168eeca41d4f3b60b62094b6befcadcc1'
V3_3_2_RUN_BINDING = {
    'attempt_started_sha256': (
        'd1bb85b3c35d892f10f67a5510b85784eb1b6b989c9ae32290b8c0fdc3c8d235'
    ),
    'run_complete_sha256': (
        'd88c8becc96a24f1f56a1cee82da35fdd331ee3b40ba4108ae98f44ae24b4ad7'
    ),
    'raw_manifest_sha256': (
        'fadbff5ab512dea3d25edb39dc9a91ffe25473b73f1e33fd53f97b60fc8436fd'
    ),
    'import_provenance_sha256': (
        'a74f3c9658e9d2286724680b52f4ea788d492f4fa9d7c52b20a53c90d57edc99'
    ),
    'protobuf_provenance_sha256': (
        '2498a940f6ee15e54e72e8f51587d4c42ffc1b49851873c31ad09085315d0ba8'
    ),
    'compiler_provenance_sha256': (
        'bd20e21a56a9ca5498d7119771bb1da9ac2e156ed3190d3ce3aa09ff2d2e312c'
    ),
    'whole_run_file_count': 11,
    'whole_run_tree_sha256': (
        '4ac66e45a4d7d65af2785904d11b23bf7e809e07f3f617e190772242b2e7a4ab'
    ),
    'compiler_file_count': 4,
    'compiler_tree_sha256': (
        '4378048568ff58a2bbee55ba9da750498b89fdef72c97911815cf895c8a8b7d1'
    ),
}

V3_3_2_1_COMMIT = 'b43051aa4a893e24a38e932900d349278c9ead88'
V3_3_2_1_ATTEMPT_DIR = (
    _HERE / 'results'
    / 'v3_3_2_development_ood_sidecar_analysis_v3_3_2_1_attempt'
)
V3_3_2_1_ANALYSIS_DIR = (
    _HERE / 'results' / 'v3_3_2_development_ood_sidecar_analysis_v3_3_2_1'
)
EXPECTED_V3_3_2_1_FAILURE_STATUS = {
    'implementation_commit': V3_3_2_1_COMMIT,
    'amendment_sha256': (
        '81a4f4c126b83225b02c7de5cf0dc6fd0baf6085b84b9ed5dd7a3677744090ba'
    ),
    'analyzer_sha256': (
        '35db9ca198cb5d7f03621ccf322ea116f98cea3bfdc711006dfd20bc809e8048'
    ),
    'analyzer_test_sha256': (
        'a8733f3ffb35920dda2f6a856076cbe82a9e90aa2fe483169150dbad4421a1b8'
    ),
    'freeze_sha256': (
        '3871ab41b16105a94673e89381d32d7253b014c64eda5c6789eaecf16477c061'
    ),
    'wrapper_sha256': (
        'ea5cce6ae631ba3fa2bf0082d691d0896ef0fed7b20f0d908034e17775060caa'
    ),
    'attempt_dir': str(V3_3_2_1_ATTEMPT_DIR.resolve()),
    'analysis_dir': str(V3_3_2_1_ANALYSIS_DIR.resolve()),
    'attempt_files': {
        'ANALYSIS_ATTEMPT_STARTED.json': {
            'size_bytes': 8616,
            'sha256': (
                'a87c4e15ed67a363d07c434ca232540687950d145e67492b9ed9c17d9adebf1d'
            ),
        },
        'ANALYSIS_FAILURE.json': {
            'size_bytes': 2163,
            'sha256': (
                '1cd933623ecdfb328d5db458b16df909e632a560361dbb547f83c22cf13ab7c7'
            ),
        },
    },
    'attempt_file_count': 2,
    'attempt_tree_sha256': (
        '5e97b191e781c5141d2f308deefacfa8f6a196449fd7e11f36c22828a13f036a'
    ),
    'state': 'failed_consumed_no_retry',
    'error_type': 'RecursionError',
    'model_apply_count': 0,
    'scientific_summary_computed': False,
    'shapley_or_nomination_computed': False,
    'combined_analysis_permitted': False,
    'analysis_dir_absent': True,
}

V3_3_2_2_AMENDMENT_COMMIT = '2a2cc59136f5b83f3a7c265b5197e30cdecd7c11'
V3_3_2_2_IMPLEMENTATION_COMMIT = (
    '67abe303082c62fd925c3c23d9a23b3e0f4526f6'
)
V3_3_2_2_ARCHIVE_COMMIT = '2f73f8750384c7fc5c73bded379e667a642c5d0a'
V3_3_2_2_ATTEMPT_DIR = (
    _HERE / 'results'
    / 'v3_3_2_development_ood_sidecar_analysis_v3_3_2_2_attempt'
)
V3_3_2_2_ANALYSIS_DIR = (
    _HERE / 'results' / 'v3_3_2_development_ood_sidecar_analysis_v3_3_2_2'
)
EXPECTED_V3_3_2_2_ARCHIVE_STATUS = {
    'amendment_commit': V3_3_2_2_AMENDMENT_COMMIT,
    'implementation_commit': V3_3_2_2_IMPLEMENTATION_COMMIT,
    'archive_commit': V3_3_2_2_ARCHIVE_COMMIT,
    'amendment_sha256': (
        '3188b44f85d8315eb4a099b42930b2d08f76074ffb190ef11b67a9f39e788a3d'
    ),
    'analyzer_sha256': (
        '70be2f80e598f6c307511dfcad1a550a2438766b3348014be1e6ff25c9b99221'
    ),
    'analyzer_test_sha256': (
        'cb7541e75dd65130f7f123643d88adac4f82e984afce4856babfc58284723a4d'
    ),
    'freeze_sha256': (
        'e7c3fe72c9d9ca5b23299dfbc2a2991643b19722cf9a5c386debf73f367fa520'
    ),
    'wrapper_sha256': (
        '7eb4d6b8dda1a415881a6934d6cf9e30cbfc707fd531eb1a1d19df3b90b1a2f9'
    ),
    'attempt_dir': str(V3_3_2_2_ATTEMPT_DIR.resolve()),
    'analysis_dir': str(V3_3_2_2_ANALYSIS_DIR.resolve()),
    'attempt_files': {
        'ANALYSIS_ATTEMPT_STARTED.json': {
            'size_bytes': 11042,
            'sha256': (
                'bdbcc6f37093924ddc79ed38e674d084761e044aefeec065d25c6692605028af'
            ),
        },
        'ANALYSIS_COMPLETE.json': {
            'size_bytes': 789,
            'sha256': (
                '5bf9d45c3b890fff7653b5d9ee57ffa959ac09933c3620069edf513eca3473f5'
            ),
        },
    },
    'attempt_file_count': 2,
    'attempt_tree_sha256': (
        'bc703c43e8afe4a01b18621180bb5d3e90c2a49c4902d175e622e0c48eeea29d'
    ),
    'analysis_files': {
        'ANALYSIS.json': {
            'size_bytes': 60844,
            'sha256': (
                '9af699921344ff8528260f7b6b2d2d57a529b9863c40edf99757929934e44b61'
            ),
        },
        'RESULT.md': {
            'size_bytes': 811,
            'sha256': (
                'be9f9c7fe31363f926999de78085b3d10c5552a80e14d0b5432deb9fb7adfc03'
            ),
        },
    },
    'analysis_file_count': 2,
    'analysis_tree_sha256': (
        '581392f933c909fe4a56d51cd03089f6c506bdc058dfdbd902abfc49c8332a0c'
    ),
    'state': 'complete_controlled_stop_audited',
    'decision': 'controlled_stop_compiler_graph_mismatch',
    'model_apply_count': 0,
    'scientific_summary_computed': False,
    'shapley_or_nomination_computed': False,
    'combined_analysis_permitted': False,
}

SOURCE_PROGRAM_CONTRACT = {
    'stablehlo_sha256': (
        '69dbf2a054cf89e56a9000dc0c04e5cd4ba425f016885ddc127d22c01661bddd'
    ),
    'stablehlo_size_bytes': 3196162,
    'pre_backend_hlo_sha256': (
        '675b9cf26fe8f59a1d138e856ee84f48edb20d8c896ac9e539b6b064dcef2750'
    ),
    'pre_backend_hlo_size_bytes': 1829833,
    'program_signatures_sha256': (
        'd8f95fb9d3637fd263cc3da0f6a33409d5fb2a5cf37e348723ecc89b3224c300'
    ),
    'entry_abi_sha256': (
        'ebf900771a87775a5fb657b90131fd884d68ce4725245defa164bf5066c74a80'
    ),
}

DENIED_CACHE_ENVIRONMENT_NAMES = (
    'XLA_FLAGS',
    'JAX_COMPILATION_CACHE_DIR',
    'CUDA_CACHE_PATH',
    'CUDA_CACHE_MAXSIZE',
    'TRITON_DUMP_DIR',
    'TRITON_OVERRIDE_DIR',
)
DENIED_CACHE_ENVIRONMENT_PREFIXES = ('JAX_PERSISTENT_CACHE_',)
CACHE_ROLE_ENVIRONMENT = 'ALPHAGENOME_V3_3_4_6_CACHE_ROLE'
CACHE_ROOT_ENVIRONMENT = 'ALPHAGENOME_V3_3_4_6_CACHE_ROOT'
CACHE_ISOLATION_CONTRACT = {
    'CUDA_CACHE_DISABLE': '1',
    'JAX_ENABLE_COMPILATION_CACHE': 'false',
    'external_preflight': {
        'cache_role': 'external_preflight',
        'cache_root': str(PREFLIGHT_KERNEL_CACHE_DIR.resolve()),
        'triton_cache_dir': str(
            (PREFLIGHT_KERNEL_CACHE_DIR / 'triton').resolve()
        ),
        'xdg_cache_home': str(
            (PREFLIGHT_KERNEL_CACHE_DIR / 'xdg').resolve()
        ),
    },
    'model': {
        'cache_role': 'model',
        'cache_root': str(MODEL_KERNEL_CACHE_DIR.resolve()),
        'triton_cache_dir': str(
            (MODEL_KERNEL_CACHE_DIR / 'triton').resolve()
        ),
        'xdg_cache_home': str(
            (MODEL_KERNEL_CACHE_DIR / 'xdg').resolve()
        ),
    },
    'pre_import_file_count': 0,
    'pre_import_tree_sha256': hashlib.sha256(b'').hexdigest(),
    'roots_distinct': True,
    'default_user_cache_paths_eligible': False,
    'cache_output_equality_is_a_gate': False,
    'postcompile_historical_snapshot_reauthenticated': False,
    'terminal_live_tree_rehashed': True,
}

AUTHORIZED_GIT_HEAD_ENV = 'V3345_AUTHORIZED_GIT_HEAD'
AUTHORIZED_FREEZE_SHA256_ENV = 'V3345_AUTHORIZED_FREEZE_SHA256'
AUTHORIZED_FREEZE_SIZE_ENV = 'V3345_AUTHORIZED_FREEZE_SIZE_BYTES'
EXTERNAL_FREEZE_AUTHORIZATION_CONTRACT = {
    'git_head_environment_name': AUTHORIZED_GIT_HEAD_ENV,
    'freeze_sha256_environment_name': AUTHORIZED_FREEZE_SHA256_ENV,
    'freeze_size_environment_name': AUTHORIZED_FREEZE_SIZE_ENV,
    'freeze_path': str(FREEZE_PATH.resolve()),
    'required_for_dry_run': True,
    'required_for_production': True,
    'source_files_must_not_embed_final_freeze_digest': True,
}
SOURCE_INPUT_AUDIT_KEYS = (
    'bootstrap_sources_and_prior_trees_exact',
    'tracked_head_and_frozen_inventory_exact',
    'external_device_runtime_environment_exact',
    'same_process_device_runtime_environment_exact',
    'checkpoint_exact', 'reference_object_and_sequences_exact',
    'protobuf_binding_exact', 'three_import_inventories_stable_exact',
)
SOURCE_INPUT_AUDIT_CONTRACT = {
    'keys': list(SOURCE_INPUT_AUDIT_KEYS),
    'start_values': [True, True, True, True, None, None, None, None],
    'dispatch_values': [True] * 8,
    'null_is_not_false': True,
}
START_RECORD_KEYS = (
    'status', 'attempt_id', 'script_version', 'amendment_sha256',
    'amendment_commit', 'original_protocol_sha256', 'freeze_path',
    'freeze_sha256', 'git_head', 'external_freeze_authorization',
    'runner_pid', 'parent_pid', 'started_at_unix_s', 'successful_preflight',
    'same_process_preflight', 'same_process_preflight_content_binding',
    'fresh_paths', 'budgets', 'execution_contract',
    'source_inventory_attestation', 'prior_v3_3_3_binding',
    'prior_v3_3_3_1_archive_binding', 'source_input_audit',
    'source_input_audit_content_binding', 'program_signature_contract',
    'cache_isolation_contract', 'confirmation_scope_disclosure',
    'confirmation_model_calls', 'scientific_summary_computed',
    'donor_normalization_computed', 'shapley_or_nomination_computed',
    'interaction_or_resolution_computed', 'nomination_performed',
    'combined_analysis_permitted',
    'prior_v3_3_4_3_consumed_preflight_prefix',
    'prior_v3_3_4_3_consumed_preflight_prefix_content_binding',
    'prior_v3_3_4_4_consumed_preflight_prefix',
    'prior_v3_3_4_4_consumed_preflight_prefix_content_binding',
    'prior_v3_3_4_5_controlled_stop_archive_content_binding',
)
POST_START_PROVENANCE_FAILURE_KEYS = (
    'status', 'stop_reason', 'message', 'failure', 'attempt_id',
    'script_version', 'amendment_sha256', 'freeze_sha256', 'git_head',
    'external_freeze_authorization', 'runner_pid',
    'source_inventory_failure', 'model_constructed', 'model_apply_count',
    'source_input_audit', 'source_input_audit_content_binding',
    'confirmation_model_calls', 'scientific_summary_computed',
    'combined_analysis_permitted', 'failed_at_unix_s',
    'prior_v3_3_4_3_consumed_preflight_prefix',
    'prior_v3_3_4_3_consumed_preflight_prefix_content_binding',
    'prior_v3_3_4_4_consumed_preflight_prefix',
    'prior_v3_3_4_4_consumed_preflight_prefix_content_binding',
    'prior_v3_3_4_5_controlled_stop_archive_content_binding',
)
PUBLICATION_TERMINAL_FAILURE_KEYS = (
    'schema_version', 'status', 'stop_reason', 'attempt_id',
    'script_version', 'external_freeze_authorization', 'runner_pid',
    'publication_failure', 'preterminal_tree_binding', 'source_input_audit',
    'source_input_audit_content_binding', 'same_object_attestation',
    'same_object_attestation_content_binding', 'phase_state',
    'model_apply_attempt_count', 'model_apply_success_count',
    'valid_record_count', 'failed_current_binding',
    'temporary_orphan_bindings', 'durability_uncertain_final_bindings',
    'preexisting_entry_states', 'no_new_entry_failure',
    'confirmation_model_calls', 'scientific_summary_computed',
    'donor_normalization_computed', 'shapley_or_nomination_computed',
    'interaction_or_resolution_computed', 'nomination_performed',
    'combined_analysis_permitted', 'no_retry', 'created_at_unix_s',
    'prior_v3_3_4_3_consumed_preflight_prefix',
    'prior_v3_3_4_3_consumed_preflight_prefix_content_binding',
    'prior_v3_3_4_4_consumed_preflight_prefix',
    'prior_v3_3_4_4_consumed_preflight_prefix_content_binding',
    'prior_v3_3_4_5_controlled_stop_archive_content_binding',
)
SAME_OBJECT_ATTESTATION_KEYS = (
    'lower_call_count', 'compile_call_count',
    'stablehlo_read_from_lowered_object',
    'pre_backend_hlo_read_from_lowered_object',
    'compile_argument_is_lowered_object',
    'compiled_hlo_read_from_compiled_object',
    'signature_attestation_from_apply_arguments',
    'apply_callable_is_compiled_object', 'compiler_record_is_gate_record',
    'lowered_python_id', 'compiled_python_id',
)
SAME_OBJECT_ATTESTATION_CONTRACT = {
    'keys': list(SAME_OBJECT_ATTESTATION_KEYS),
    'lower_budget': 1, 'compile_budget': 1,
    'identity_checks_use_python_is': True,
}
PROGRAM_SIGNATURE_ATTESTATION_CONTRACT = {
    'schema_version': 'v3.3.4.6-program-signature-attestation-v1',
    'object_order': ['eight_interventions', 'selection', 'target'],
    'runtime_container_count': 32, 'frozen_container_count': 32,
    'runtime_leaves_tuple_count': 3, 'runtime_shape_tuple_count': 29,
    'frozen_leaves_list_count': 3, 'frozen_shape_list_count': 29,
    'canonical_sha256': SOURCE_PROGRAM_CONTRACT['program_signatures_sha256'],
    'canonical_size_bytes': 2877,
    'declared_paths': [
        '/eight_interventions/leaves',
        *[f'/eight_interventions/leaves/{index}/shape' for index in range(17)],
        '/selection/leaves',
        *[f'/selection/leaves/{index}/shape' for index in range(9)],
        '/target/leaves',
        *[f'/target/leaves/{index}/shape' for index in range(3)],
    ],
    'runtime_kind': 'tuple',
    'frozen_kind': 'list',
    'narrow_adapter_only': True,
}
DISPATCH_JOURNAL_CONTRACT = {
    'call_roles': ['intended', 'intended_repeat', 'unrelated', 'unrelated_repeat'],
    'started_path': 'dispatch_journal/started/{global_dispatch_index:03d}.json',
    'completed_path': 'dispatch_journal/completed/{global_dispatch_index:03d}.json',
    'expected_started_count': 320, 'expected_completed_count': 320,
    'publication': 'named_temp_renameat2_noreplace',
    'started_keys': [
        'schema_version', 'event', 'attempt_id', 'script_version',
        'execution_index', 'recipient_order', 'recipient_variant_id',
        'anchor_id', 'call_index_within_record', 'call_role',
        'global_dispatch_index', 'runner_pid',
        'source_input_audit_sha256', 'same_object_attestation_sha256',
        'started_at_unix_s',
    ],
    'completed_keys': [
        'schema_version', 'event', 'attempt_id', 'script_version',
        'execution_index', 'recipient_order', 'recipient_variant_id',
        'anchor_id', 'call_index_within_record', 'call_role',
        'global_dispatch_index', 'runner_pid',
        'source_input_audit_sha256', 'same_object_attestation_sha256',
        'started_event_sha256', 'returned', 'completed_at_unix_s',
    ],
}
FAILED_CURRENT_CONTRACT = {
    'd_values': [0, 1, 2, 3, 4],
    'failure_phases': ['record_setup', 'model_dispatch', 'record_validation', 'record_serialization'],
    'lossless_encoding': 'base64_c_order_raw_bytes',
    'keys': [
        'schema_version', 'status', 'attempt_id', 'script_version',
        'external_freeze_authorization', 'execution_index',
        'recipient_order', 'recipient_variant_id', 'anchor_id',
        'failed_or_next_call_role', 'd_completed', 'started_count',
        'completed_count', 'started_event_bindings',
        'completed_event_bindings', 'partial_call_outputs', 'failure_phase',
        'failure', 'source_input_audit_content_binding',
        'same_object_attestation_content_binding',
        'confirmation_scope_disclosure', 'created_at_unix_s',
    ],
    'partial_call_output_keys': ['status', 'treedef', 'leaf_count', 'leaves'],
    'leaf_keys': [
        'path', 'dtype_name', 'byte_order', 'shape', 'encoding',
        'data_base64', 'sha256', 'size_bytes',
    ],
    'treedef_node_keys': ['kind', 'metadata', 'children'],
}
RAW_RECORD_CONTRACT = {
    'record_count': 80, 'applies_per_record': 4,
    'family': 'v3_3_4_6_unrelated_donor_sidecar_anchor',
    'top_level_keys': [
        'status', 'family', 'script_version', 'amendment_sha256',
        'amendment_commit', 'original_protocol_sha256', 'freeze_sha256',
        'external_freeze_authorization', 'execution_index',
        'sidecar_execution_index', 'execution_order',
        'eight_row_executable_fingerprint',
        'same_eight_row_compiled_executable', 'six_row_executable_used',
        'recipient_case', 'donor_case', 'coalition', 'batch_roles',
        'natural_identity_rows', 'intended_donor_rows',
        'unrelated_donor_rows', 'invariant_rows_between_calls',
        'active_recipient_rows',
        'active_recipient_cross_call_equality_gate',
        'active_recipient_cross_call_inequality_gate',
        'original_artifact_bindings', 'original_ood_records_used_as_data',
        'recipient_sequence_sha256', 'donor_sequence_sha256',
        'runtime_interventions', 'intended_target_readout',
        'intended_repeat_target_readout', 'unrelated_target_readout',
        'unrelated_repeat_target_readout', 'intended_trace_fingerprint',
        'intended_repeat_trace_fingerprint', 'unrelated_trace_fingerprint',
        'unrelated_repeat_trace_fingerprint', 'rowwise_trace_fingerprints',
        'raw_movement', 'model_apply_count_through_record', 'checks',
        'failure', 'seconds', 'dispatch_started_bindings',
        'dispatch_completed_bindings', 'source_input_audit',
        'source_input_audit_content_binding', 'same_object_attestation',
        'same_object_attestation_content_binding',
        'confirmation_scope_disclosure', 'created_at_unix_s',
    ],
}
RAW_MANIFEST_CONTRACT = {
    'path_base': 'model_run_root',
    'statuses': ['complete80', 'controlled_prefix', 'empty_controlled_stop'],
    'keys': [
        'schema_version', 'status', 'attempt_id',
        'external_freeze_authorization', 'valid_artifact_count',
        'artifact_bindings', 'artifact_tree_sha256',
        'valid_recipient_anchor_pairs', 'failed_current_binding',
        'dispatch_started_count', 'dispatch_completed_count',
        'dispatch_started_bindings', 'dispatch_started_tree_sha256',
        'dispatch_completed_bindings', 'dispatch_completed_tree_sha256',
        'source_input_audit_content_binding',
        'same_object_attestation_content_binding', 'created_at_unix_s',
    ],
}
TERMINAL_CONTRACT = {
    'expected_records': 80, 'expected_model_applies': 320,
    'confirmation_model_calls': 0, 'combined_analysis_permitted': False,
    'run_complete_size_cap_bytes': 16777216,
    'run_complete_keys': [
        'status', 'stop_reason', 'message', 'failure', 'attempt_id',
        'script_version', 'amendment_sha256', 'amendment_commit',
        'original_protocol_sha256', 'freeze_sha256', 'git_head',
        'external_freeze_authorization', 'runner_pid',
        'started_at_unix_s', 'completed_at_unix_s', 'phase_state',
        'terminal_detail', 'budgets', 'source_input_audit',
        'source_input_audit_content_binding', 'same_object_attestation',
        'same_object_attestation_content_binding',
        'program_signature_attestation_binding', 'source_program_gate',
        'compiler_binding', 'compiler_artifact_bindings',
        'attempt_budget_audit', 'diagnostic_provenance_complete',
        'compiled_backend_diagnostic_only', 'backend_diagnostics',
        'diagnostic_comparisons', 'dispatch_journal', 'raw_manifest',
        'preterminal_tree_binding', 'valid_record_count',
        'failed_current_binding', 'model_apply_attempt_count',
        'model_apply_success_count', 'expected_model_apply_count',
        'eight_row_lower_attempt_count', 'eight_row_compile_attempt_count',
        'eight_row_successful_compile_count', 'six_row_compile_count',
        'identity_rerun_count', 'main_cube_rerun_count',
        'old_ood_records_reused', 'confirmation_model_calls',
        'all_80_recipient_anchors_complete', 'id0_all20', 'id255_all20',
        'import_provenance_phases', 'protobuf_provenance_sha256',
        'model_kernel_cache_final', 'prior_v3_3_3_binding',
        'prior_v3_3_3_1_archive_binding', 'confirmation_scope_disclosure',
        'publication_audit',
        'scientific_summary_computed', 'donor_normalization_computed',
        'shapley_or_nomination_computed',
        'interaction_or_resolution_computed', 'nomination_performed',
        'combined_analysis_permitted', 'no_retry',
        'prior_v3_3_4_3_consumed_preflight_prefix',
        'prior_v3_3_4_3_consumed_preflight_prefix_content_binding',
        'prior_v3_3_4_4_consumed_preflight_prefix',
        'prior_v3_3_4_4_consumed_preflight_prefix_content_binding',
        'prior_v3_3_4_5_controlled_stop_archive_content_binding',
    ],
    'phase_state_keys': list((
        'preflight_passed', 'start_persisted',
        'post_start_source_gate_passed', 'protobuf_persisted',
        'pre_model_import_inventory_persisted',
        'model_construction_attempted', 'model_constructed',
        'reference_cases_loaded', 'signatures_captured',
        'signature_attestation_persisted',
        'post_model_import_inventory_persisted', 'lower_attempted',
        'lower_succeeded', 'compile_attempted', 'compile_succeeded',
        'terminal_import_inventory_persisted',
        'source_program_gate_passed', 'diagnostic_provenance_passed',
        'dispatch_begun',
    )),
    'terminal_detail_keys': [
        'k_valid_records', 'd_completed', 'failed_execution_index',
        'failed_call_role', 'failure_phase', 'forbidden_operation',
        'provenance_artifact_role',
    ],
    'statuses': [
        'controlled_stop_import_provenance_failure',
        'controlled_stop_protobuf_provenance_failure',
        'controlled_stop_cache_hit', 'controlled_stop_model_setup_failure',
        'controlled_stop_signature_attestation_failure',
        'controlled_stop_lower_failure', 'controlled_stop_compile_failure',
        'controlled_stop_attempt_budget_violation',
        'controlled_stop_same_object_provenance_failure',
        'controlled_stop_source_program_mismatch',
        'controlled_stop_diagnostic_provenance_failure',
        'controlled_stop_partial_dispatch',
        'controlled_stop_four_call_invalid', 'complete_structural_sidecar',
    ],
}
NONPUBLICATION_TERMINAL_CONTRACT_V3_3_4_6 = {
    'artifact_path': 'NONPUBLICATION_TERMINAL_FAILURE.json',
    'schema_version': 'v3.3.4.6-nonpublication-terminal-v1',
    'status': 'incomplete_nonpublication_infrastructure_failure',
    'stop_reason': 'post_compile_nonpublication_infrastructure_failure',
    'artifact_role': 'nonpublication_terminal_failure',
    'predecessor_amendments': {
        'v3_3_4': {
            'commit': V3_3_4_AMENDMENT_COMMIT,
            'path': str(V3_3_4_AMENDMENT_PATH.resolve()),
            'sha256': V3_3_4_AMENDMENT_SHA256,
        },
        'v3_3_4_1': {
            'commit': V3_3_4_1_AMENDMENT_COMMIT,
            'path': str(V3_3_4_1_AMENDMENT_PATH.resolve()),
            'sha256': V3_3_4_1_AMENDMENT_SHA256,
        },
        'v3_3_4_2': {
            'commit': V3_3_4_2_AMENDMENT_COMMIT,
            'path': str(V3_3_4_2_AMENDMENT_PATH.resolve()),
            'sha256': V3_3_4_2_AMENDMENT_SHA256,
        },
    },
    'predecessor_path_contract': {
        version: {
            role: str(path.resolve()) for role, path in sorted(paths.items())
        }
        for version, paths in (
            ('v3_3_4', V3_3_4_PREDECESSOR_PATHS),
            ('v3_3_4_1', V3_3_4_1_PREDECESSOR_PATHS),
            ('v3_3_4_2', V3_3_4_2_PREDECESSOR_PATHS),
        )
    },
    'failure_stages': [
        'stablehlo_text_extraction',
        'pre_backend_hlo_text_extraction',
        'compiled_hlo_text_extraction',
        'source_program_gate_derivation_for_diagnostic_failure',
        'diagnostic_failure_record_construction',
    ],
    'triggering_diagnostic_stop_reasons': [
        'diagnostic_parser_failure', 'diagnostic_persistence_failure',
        'cache_signal_unavailable', 'fingerprint_formula_mismatch',
    ],
    'keys': [
        'schema_version', 'status', 'stop_reason', 'attempt_id',
        'script_version', 'amendment_commit', 'amendment_sha256',
        'inherited_v3_3_4_commit', 'inherited_v3_3_4_sha256',
        'inherited_v3_3_4_1_commit', 'inherited_v3_3_4_1_sha256',
        'freeze_sha256', 'git_head', 'external_freeze_authorization',
        'runner_pid', 'started_at_unix_s', 'created_at_unix_s',
        'failure_stage', 'failure', 'triggering_diagnostic_failure',
        'triggering_diagnostic_stop_reason', 'phase_state',
        'source_input_audit', 'source_input_audit_content_binding',
        'program_signature_attestation_binding',
        'same_object_attestation',
        'same_object_attestation_content_binding', 'attempt_budget_audit',
        'compiler_counts', 'graph_artifact_bindings',
        'import_provenance_phases', 'protobuf_provenance_sha256',
        'model_kernel_cache_state',
        'source_program_gate_without_backend_diagnostics',
        'source_program_gate_without_backend_diagnostics_content_binding',
        'prior_v3_3_3_binding', 'prior_v3_3_3_1_archive_binding',
        'preterminal_tree_binding', 'publication_audit',
        'model_apply_attempt_count', 'model_apply_success_count',
        'valid_record_count', 'raw_record_count',
        'dispatch_started_count', 'dispatch_completed_count',
        'six_row_compile_count', 'identity_rerun_count',
        'main_cube_rerun_count', 'old_ood_records_reused',
        'confirmation_model_calls', 'confirmation_scope_disclosure',
        'scientific_summary_computed', 'donor_normalization_computed',
        'shapley_or_nomination_computed',
        'interaction_or_resolution_computed', 'nomination_performed',
        'combined_analysis_permitted', 'no_retry',
        'prior_v3_3_4_3_consumed_preflight_prefix',
        'prior_v3_3_4_3_consumed_preflight_prefix_content_binding',
        'prior_v3_3_4_4_consumed_preflight_prefix',
        'prior_v3_3_4_4_consumed_preflight_prefix_content_binding',
        'prior_v3_3_4_5_controlled_stop_archive_content_binding',
    ],
    'extraction_preterminal_membership': [
        'ATTEMPT_STARTED.json', 'IMPORT_PROVENANCE_PRE_MODEL.json',
        'PROTOBUF_PROVENANCE.json',
        'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
        'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json',
    ],
    'diagnostic_construction_preterminal_membership': [
        'ATTEMPT_STARTED.json', 'IMPORT_PROVENANCE.json',
        'IMPORT_PROVENANCE_PRE_MODEL.json',
        'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
        'PROTOBUF_PROVENANCE.json',
        'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json',
        'compiler/eight_row/graph.compiled.hlo.txt',
        'compiler/eight_row/graph.pre_backend.hlo.txt',
        'compiler/eight_row/graph.stablehlo.mlir',
    ],
    'compiler_counts': {
        'lower_attempt_count': 1, 'compile_attempt_count': 1,
        'successful_compile_count': 1,
    },
    'stage_semantics': {
        'extraction_source_gate_is_null': True,
        'source_gate_derivation_failure_source_gate_is_null': True,
        'diagnostic_record_construction_source_gate_is_nonnull': True,
        'diagnostic_record_construction_phase_equals_source_program_exact': (
            True
        ),
        'diagnostic_stage_applicable_same_object_primitives_are_true': True,
        'compiler_record_identity_means_in_memory_gate_object': True,
    },
    'zero_count_keys': [
        'model_apply_attempt_count', 'model_apply_success_count',
        'valid_record_count', 'raw_record_count',
        'dispatch_started_count', 'dispatch_completed_count',
        'six_row_compile_count', 'identity_rerun_count',
        'main_cube_rerun_count', 'old_ood_records_reused',
        'confirmation_model_calls',
    ],
    'science_flag_keys': [
        'scientific_summary_computed', 'donor_normalization_computed',
        'shapley_or_nomination_computed',
        'interaction_or_resolution_computed', 'nomination_performed',
        'combined_analysis_permitted',
    ],
    'analyzer_outcome': {
        'status': (
            'complete_incomplete_nonpublication_infrastructure_archive'
        ),
        'decision': (
            'post_compile_nonpublication_failure_no_scientific_analysis'
        ),
        'compiler_state': 'compiled_without_legal_graph_gate_record',
        'terminal_kind': 'nonpublication_terminal_failure',
        'control_state_eligible': False,
    },
    'publication_error_fallback': 'v3.3.4.1-TERMINAL_FAILURE-only',
    'ordinary_construction_error_fallback': 'terminal_less_consumed_prefix',
}
PREFLIGHT_CONTRACT = {
    'attempt_number': 0, 'file_count': 5,
    'no_jit_no_array_no_model': True,
    'atomic_publication_probe_required': True,
    'root_membership': [
        '.allocation.lock', '.preflight_0000.reserved',
        'preflight_0000.json', 'preflight_0000.stdout.log',
        'preflight_0000.stderr.log',
    ],
    'record_keys': [
        'amendment_sha256', 'atomic_publication_probe', 'created_at_unix_s',
        'external_freeze_authorization', 'external_cache_post_observation',
        'external_cache_hit_evidence', 'failure', 'freeze',
        'freeze_sha256', 'logs', 'no_jit_or_array_kernel',
        'no_model_or_biological_access', 'observation',
        'original_protocol_sha256', 'preflight_attempt_number',
        'script_version', 'status', 'warnings',
        'prior_v3_3_4_3_consumed_preflight_prefix',
        'prior_v3_3_4_3_consumed_preflight_prefix_content_binding',
        'prior_v3_3_4_4_consumed_preflight_prefix',
        'prior_v3_3_4_4_consumed_preflight_prefix_content_binding',
        'prior_v3_3_4_5_controlled_stop_archive_content_binding',
    ],
}
COMPILED_DIAGNOSTICS_CONTRACT = {
    'compiled_backend_equality_is_a_gate': False,
    'source_program_equality_is_a_gate': True,
    'backend_diagnostics_keys': [
        'descriptive_only_not_an_equality_gate', 'computation_count',
        'instruction_count_excluding_computation_headers',
        'instruction_record_count', 'fusion_kind_counts',
        'triton_configuration_count', 'triton_configurations',
        'cublas_call_count', 'cublas_algorithms', 'cudnn_call_count',
        'cudnn_algorithms_workspaces',
    ],
}
SOURCE_INVENTORY_CONTRACT = {
    'source_row_count': 144,
    'rows': None,
    'prospective_upstream_source_file_count': 26,
    'loaded_scientific_module_contract': None,
    'inherited_source_authority_commit': V3_3_4_5_COMMIT,
    'amendment_authority_commit': AMENDMENT_COMMIT,
    'implementation_source_authority_commit': None,
}


def validate_external_freeze_authorization() -> dict[str, Any]:
  """Validate the acyclic post-commit authorization supplied by the auditor."""
  head = os.environ.get(AUTHORIZED_GIT_HEAD_ENV, '')
  digest = os.environ.get(AUTHORIZED_FREEZE_SHA256_ENV, '')
  size_text = os.environ.get(AUTHORIZED_FREEZE_SIZE_ENV, '')
  if not re.fullmatch(r'[0-9a-f]{40}', head):
    raise RuntimeError('Missing or malformed authorized Git HEAD.')
  if not re.fullmatch(r'[0-9a-f]{64}', digest):
    raise RuntimeError('Missing or malformed authorized freeze SHA-256.')
  if not size_text.isascii() or not size_text.isdigit():
    raise RuntimeError('Missing or malformed authorized freeze size.')
  size = int(size_text)
  if size < 1 or not FREEZE_PATH.is_file() or FREEZE_PATH.is_symlink():
    raise RuntimeError('Authorized freeze is absent or not a regular file.')
  live_head = subprocess.run(
      ['git', 'rev-parse', 'HEAD'], cwd=_REPO, check=True,
      capture_output=True, text=True,
  ).stdout.strip()
  if live_head != head:
    raise RuntimeError('Authorized Git HEAD differs from live HEAD.')
  if FREEZE_PATH.stat().st_size != size or _sha256(FREEZE_PATH) != digest:
    raise RuntimeError('Authorized freeze bytes differ from live freeze.')
  relative = FREEZE_PATH.relative_to(_REPO).as_posix()
  blob = subprocess.run(
      ['git', 'show', f'HEAD:{relative}'], cwd=_REPO, check=True,
      capture_output=True,
  ).stdout
  if hashlib.sha256(blob).hexdigest() != digest or len(blob) != size:
    raise RuntimeError('Authorized freeze differs from committed HEAD blob.')
  tracked_diff = subprocess.run(
      ['git', 'diff', '--quiet', 'HEAD', '--'], cwd=_REPO, check=False,
  )
  if tracked_diff.returncode != 0:
    raise RuntimeError('Tracked repository bytes differ from HEAD.')
  return {
      'git_head': head,
      'freeze_path': str(FREEZE_PATH.resolve()),
      'freeze_sha256': digest,
      'freeze_size_bytes': size,
      'live_equals_git_show': True,
      'tracked_clean': True,
      'authorization_source': 'external_post_commit_audit',
  }

V3_3_2_1_SOURCE_BINDINGS = {
    'experiments/interpretability/opensplice/v3_wider_mechanism/'
    'encoder_skip_ood_sidecar_analysis_amendment_v3_3_2_1.md': (
        '81a4f4c126b83225b02c7de5cf0dc6fd0baf6085b84b9ed5dd7a3677744090ba'
    ),
    'experiments/interpretability/opensplice/'
    'analyze_encoder_skip_ood_sidecar_v3_3_2_1.py': (
        '35db9ca198cb5d7f03621ccf322ea116f98cea3bfdc711006dfd20bc809e8048'
    ),
    'experiments/interpretability/opensplice/'
    'analyze_encoder_skip_ood_sidecar_v3_3_2_1_test.py': (
        'a8733f3ffb35920dda2f6a856076cbe82a9e90aa2fe483169150dbad4421a1b8'
    ),
    'experiments/interpretability/opensplice/'
    'encoder_skip_ood_sidecar_analysis_v3_3_2_1_freeze.json': (
        '3871ab41b16105a94673e89381d32d7253b014c64eda5c6789eaecf16477c061'
    ),
    'experiments/interpretability/opensplice/'
    'run_encoder_skip_ood_sidecar_analysis_v3_3_2_1.sh': (
        'ea5cce6ae631ba3fa2bf0082d691d0896ef0fed7b20f0d908034e17775060caa'
    ),
}

V3_3_2_2_SOURCE_BINDINGS = {
    'experiments/interpretability/opensplice/v3_wider_mechanism/'
    'encoder_skip_ood_sidecar_analysis_amendment_v3_3_2_2.md': (
        '3188b44f85d8315eb4a099b42930b2d08f76074ffb190ef11b67a9f39e788a3d'
    ),
    'experiments/interpretability/opensplice/'
    'analyze_encoder_skip_ood_sidecar_v3_3_2_2.py': (
        '70be2f80e598f6c307511dfcad1a550a2438766b3348014be1e6ff25c9b99221'
    ),
    'experiments/interpretability/opensplice/'
    'analyze_encoder_skip_ood_sidecar_v3_3_2_2_test.py': (
        'cb7541e75dd65130f7f123643d88adac4f82e984afce4856babfc58284723a4d'
    ),
    'experiments/interpretability/opensplice/'
    'encoder_skip_ood_sidecar_analysis_v3_3_2_2_freeze.json': (
        'e7c3fe72c9d9ca5b23299dfbc2a2991643b19722cf9a5c386debf73f367fa520'
    ),
    'experiments/interpretability/opensplice/'
    'run_encoder_skip_ood_sidecar_analysis_v3_3_2_2.sh': (
        '7eb4d6b8dda1a415881a6934d6cf9e30cbfc707fd531eb1a1d19df3b90b1a2f9'
    ),
}
V3_3_1_AMENDMENT_PATH = (
    _HERE / 'v3_wider_mechanism'
    / 'encoder_skip_analysis_amendment_v3_3_1.md'
)
V3_3_1_AMENDMENT_SHA256 = (
    '37e23b251f53ab87bae99b63024a381c367ce33bbc950a2227b3267fbc9668d1'
)
V3_3_1_AMENDMENT_COMMIT = '186c25f'
V3_3_1_ATTEMPT_DIR = (
    _HERE / 'results'
    / 'v3_3_development_encoder_skip_factorial_analysis_v3_3_1_attempt'
)
V3_3_1_ANALYSIS_DIR = (
    _HERE / 'results' / 'v3_3_development_encoder_skip_factorial_analysis'
)

EXPECTED_V3_3_1_COMPLETED_STATUS = {
    'amendment_path': str(V3_3_1_AMENDMENT_PATH.resolve()),
    'amendment_sha256': V3_3_1_AMENDMENT_SHA256,
    'amendment_commit': V3_3_1_AMENDMENT_COMMIT,
    'attempt_dir': str(V3_3_1_ATTEMPT_DIR.resolve()),
    'analysis_dir': str(V3_3_1_ANALYSIS_DIR.resolve()),
    'state': 'completed',
    'attempt_file_count': 2,
    'attempt_tree_sha256': (
        'b0e788f0df3db1678ca410da7b0c409a18ceeaa6ddcbb97c61a155188a6e719f'
    ),
    'analysis_file_count': 2,
    'analysis_tree_sha256': (
        'f3e6eee31c3fc978356a5766c190061ae3f8fd709da6c5c0836f7ce3d47de8f0'
    ),
    'attempt_files': {
        'ANALYSIS_ATTEMPT_STARTED.json': {
            'sha256': (
                '1c4738026210ddd7f4d62b21f04eb1305cc86041daadb61cb3cfe0e549af8922'
            ),
            'size_bytes': 16116,
        },
        'ANALYSIS_COMPLETE.json': {
            'sha256': (
                'ee7d9fa0d0d06abbc52beda8801f411b8725d59e7d5683c256bd51010d732e99'
            ),
            'size_bytes': 574,
        },
    },
    'analysis_files': {
        'ANALYSIS.json': {
            'sha256': (
                'ed18cce580c578d3cd750756d882a7b120a87736849077abffaee5781c09dd6b'
            ),
            'size_bytes': 653362,
        },
        'RESULT.md': {
            'sha256': (
                '6a4884040677e194c9b43c115af8c045cd67d2ffa857088710ab852399d2a440'
            ),
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

EXPECTED_ORIGINAL_BINDING = {
    'git_commit': '9e8324434a3c0b2d5c53d209ad04c2d88b99f4bc',
    'protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
    'freeze_sha256': ORIGINAL_FREEZE_SHA256,
    'runner_sha256': (
        '56eef2cc5b87f3ff9ad5837d19b891b98bbb4a7e126e20713ea9bc8b21c409c5'
    ),
    'analyzer_sha256': (
        '0a65a27a5c424bb9dddacb5475e02d28a0999fc7cd593d4dd63ba4be06c39a46'
    ),
    'analyzer_test_sha256': (
        'd027f73fb07682e8cb54d46653a5bd9aa900aaeac433b1748dbdf4886c6d5034'
    ),
    'model_sha256': (
        '7aee357d776f1f10f9ef04b1602103496ad543d89f49d5e59af459afca217ea1'
    ),
    'interpretability_sha256': (
        'd00a4dd8a4e62c2d8a7d583a74cbf5632121f98892e901c7f8927539ee156500'
    ),
    'attempt_started_sha256': (
        'b74081fd0cbd1c8d6ec5445b3b71661f40ac4d47dd77fd2d9bd3675b4cf9c3c3'
    ),
    'run_complete_sha256': (
        'ddc8350361ae9091ac47878a2c2d043897c46ef1d7722a401869d8d69e4be463'
    ),
    'raw_manifest_sha256': (
        '6c50c86153fbce5136ed99205ca4726f87a00ef56216f1205dba5c25d3d27cd7'
    ),
    'raw_artifact_count': 5142,
    'raw_artifact_tree_sha256': (
        'e7376062ce31090b349e88b91bd41700caf4e690511c15993e50f2bd0d47f770'
    ),
    'whole_run_file_count': 5158,
    'whole_run_tree_sha256': (
        '2d8125fe6d13773ba9621e527870361b6a195c516c5b4f044c7dad64c9310aaa'
    ),
    'compiler_file_count': 8,
    'compiler_tree_sha256': (
        '9a03dcbc9d439cb9bf197941af3bbdb3e6bda067cf661b90de6d7eab1f4d87eb'
    ),
    'import_provenance_sha256': (
        '64a5538499e5b06e29cb506a2b08585bb002b3766bd1be210d1a568b9ec5110e'
    ),
    'protobuf_provenance_sha256': (
        '2498a940f6ee15e54e72e8f51587d4c42ffc1b49851873c31ad09085315d0ba8'
    ),
    'target_eligibility_sha256': (
        'b216692d8028faab09b5f6590e3e68d9c8805d3c715ddedd99e8956019cedcf0'
    ),
    'device_preflight_sha256': (
        'b983c7f4910ef4fc5f68bc72486552063f4497f90bba64497eb29a09d3d1809d'
    ),
    'preflight_stdout_sha256': (
        'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    ),
    'preflight_stderr_sha256': (
        'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    ),
}

EXPECTED_ORIGINAL_STATUS = {
    'status': 'controlled_stop',
    'stop_reason': 'ood_tooling_failure',
    'identity_count': 20,
    'identity_invalid_count': 0,
    'eligible_effect_count': 12,
    'all_effects_target_eligible': True,
    'all_neutrals_retained': True,
    'coalition_record_count': 5120,
    'coalition_invalid_count': 0,
    'ood_anchor_record_count': 2,
    'ood_invalid_count': 1,
    'scientific_record_count': 5142,
    'model_apply_count': 10288,
    'compile_count': 2,
    'id0_noop_all20': True,
    'id255_closure_all20': True,
    'confirmation_model_calls': 0,
}

EXPECTED_ORIGINAL_OOD = {
    'raw/ood_anchors/000_BRAF_e14_A117G/000.json': (
        '97917119318b21e679bb0c2d11f40937f1e0d8b2ec41c20275dc9f9305d0e680'
    ),
    'raw/ood_anchors/000_BRAF_e14_A117G/127.json': (
        '4245778e3c5edca8075b8e0a703cea470d6567e8083c369370b735c390397998'
    ),
}


def _sha256(path: Path) -> str:
  expected = path.lstat()
  if stat.S_ISLNK(expected.st_mode) or not stat.S_ISREG(expected.st_mode):
    raise ValueError(f'Hash target is not a safe regular file: {path}.')
  descriptor = os.open(
      path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
  )
  try:
    before = os.fstat(descriptor)
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_nlink, value.st_mode,
        value.st_size,
    )
    if identity(before) != identity(expected):
      raise ValueError(f'Hash target inode changed before read: {path}.')
    digest = hashlib.sha256()
    for block in iter(lambda: os.read(descriptor, 1024 * 1024), b''):
      digest.update(block)
    after = os.fstat(descriptor)
    final_path = path.lstat()
    if identity(after) != identity(before) or identity(final_path) != identity(before):
      raise ValueError(f'Hash target inode changed during read: {path}.')
    return digest.hexdigest()
  finally:
    os.close(descriptor)


def _canonical_sha256(value: Any) -> str:
  return hashlib.sha256(json.dumps(
      value,
      sort_keys=True,
      separators=(',', ':'),
      ensure_ascii=False,
      allow_nan=False,
  ).encode('utf-8')).hexdigest()


def canonical_content_binding(value: Any) -> dict[str, Any]:
  payload = json.dumps(
      value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
      allow_nan=False,
  ).encode('utf-8')
  return {
      'sha256': hashlib.sha256(payload).hexdigest(),
      'size_bytes': len(payload),
  }


def _reject_confirmation_path(path: Path) -> None:
  if any('confirm' in part.lower() for part in path.resolve().parts):
    raise ValueError(f'Confirmation-named path is forbidden: {path}.')


def assert_cache_environment_sanitized() -> dict[str, Any]:
  """Rejects persistent/compiler/kernel/autotune cache inputs pre-import."""
  present = sorted(
      name
      for name in os.environ
      if name in DENIED_CACHE_ENVIRONMENT_NAMES
      or any(name.startswith(prefix) for prefix in DENIED_CACHE_ENVIRONMENT_PREFIXES)
      or (
          'AUTOTUNE' in name.upper()
          and any(
              term in name.upper() for term in ('LOAD', 'DUMP', 'CACHE')
          )
      )
  )
  if present:
    raise ValueError(f'Forbidden cache environment input is present: {present}.')
  if os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE') != 'false':
    raise ValueError('XLA_PYTHON_CLIENT_PREALLOCATE must be literal false.')
  if os.environ.get('JAX_ENABLE_COMPILATION_CACHE') != 'false':
    raise ValueError('JAX_ENABLE_COMPILATION_CACHE must be literal false.')
  if os.environ.get('CUDA_CACHE_DISABLE') != '1':
    raise ValueError('CUDA_CACHE_DISABLE must be literal 1.')
  role = os.environ.get(CACHE_ROLE_ENVIRONMENT)
  root_text = os.environ.get(CACHE_ROOT_ENVIRONMENT)
  if role not in ('external_preflight', 'model', 'dry_run') or not root_text:
    raise ValueError('A frozen v3.3.4.6 cache role/root is required.')
  root = Path(root_text).resolve()
  if role == 'external_preflight' and root != PREFLIGHT_KERNEL_CACHE_DIR.resolve():
    raise ValueError('External-preflight cache root changed.')
  if role == 'model' and root != MODEL_KERNEL_CACHE_DIR.resolve():
    raise ValueError('Model cache root changed.')
  if role == 'dry_run' and not root.name.startswith(
      'alphagenome-v3.3.4.6-dry-cache.'
  ):
    raise ValueError('Dry-run cache root is not a fresh temporary path.')
  if role == 'external_preflight' and (
      MODEL_KERNEL_CACHE_DIR.exists() or MODEL_KERNEL_CACHE_DIR.is_symlink()
  ):
    raise ValueError('Model cache root exists before external preflight.')
  if role == 'dry_run':
    for production_root in (
        PREFLIGHT_KERNEL_CACHE_DIR, MODEL_KERNEL_CACHE_DIR
    ):
      if production_root.exists() or production_root.is_symlink():
        raise ValueError('Dry-run found a production cache root.')
  triton = root / 'triton'
  xdg = root / 'xdg'
  if os.environ.get('TRITON_CACHE_DIR') != str(triton):
    raise ValueError('TRITON_CACHE_DIR is not the fresh role-specific path.')
  if os.environ.get('XDG_CACHE_HOME') != str(xdg):
    raise ValueError('XDG_CACHE_HOME is not the fresh role-specific path.')
  for path in (root, triton, xdg):
    if path.is_symlink() or not path.is_dir():
      raise ValueError(f'Fresh cache path is absent/non-directory: {path}.')
  entries = sorted(
      path
      for path in root.rglob('*')
      if path not in (triton, xdg)
  )
  if entries:
    raise ValueError(f'Fresh cache tree is not empty: {entries[:3]}.')
  return {
      'denied_exact_names': list(DENIED_CACHE_ENVIRONMENT_NAMES),
      'denied_prefixes': list(DENIED_CACHE_ENVIRONMENT_PREFIXES),
      'present_forbidden_names': [],
      'autotune_load_dump_cache_inputs_absent': True,
      'kernel_cache_inputs_absent': True,
      'persistent_compilation_cache_inputs_absent': True,
      'cuda_kernel_cache_disabled': True,
      'cache_role': role,
      'cache_root': str(root),
      'triton_cache_dir': str(triton),
      'xdg_cache_home': str(xdg),
      'pre_import_file_count': 0,
      'pre_import_tree_sha256': hashlib.sha256(b'').hexdigest(),
      'default_user_cache_paths_eligible': False,
  }


def assert_live_cache_environment_matches(
    pre_import: Mapping[str, Any],
) -> dict[str, Any]:
  """Rechecks cache-routing environment without reusing cache claims.

  This is safe after JAX import: it deliberately does not require the fresh
  cache tree to remain empty, because diagnostic compiler outputs may have
  appeared.  It does require every routing/policy input to remain identical
  to the independently captured pre-import attestation.
  """
  present = sorted(
      name
      for name in os.environ
      if name in DENIED_CACHE_ENVIRONMENT_NAMES
      or any(
          name.startswith(prefix)
          for prefix in DENIED_CACHE_ENVIRONMENT_PREFIXES
      )
      or (
          'AUTOTUNE' in name.upper()
          and any(term in name.upper() for term in ('LOAD', 'DUMP', 'CACHE'))
      )
  )
  if present:
    raise ValueError(f'Forbidden live cache input is present: {present}.')
  live = {
      'XLA_PYTHON_CLIENT_PREALLOCATE': os.environ.get(
          'XLA_PYTHON_CLIENT_PREALLOCATE'
      ),
      'JAX_ENABLE_COMPILATION_CACHE': os.environ.get(
          'JAX_ENABLE_COMPILATION_CACHE'
      ),
      'CUDA_CACHE_DISABLE': os.environ.get('CUDA_CACHE_DISABLE'),
      'cache_role': os.environ.get(CACHE_ROLE_ENVIRONMENT),
      'cache_root': os.environ.get(CACHE_ROOT_ENVIRONMENT),
      'triton_cache_dir': os.environ.get('TRITON_CACHE_DIR'),
      'xdg_cache_home': os.environ.get('XDG_CACHE_HOME'),
  }
  expected = {
      'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
      'JAX_ENABLE_COMPILATION_CACHE': 'false',
      'CUDA_CACHE_DISABLE': '1',
      'cache_role': pre_import.get('cache_role'),
      'cache_root': pre_import.get('cache_root'),
      'triton_cache_dir': pre_import.get('triton_cache_dir'),
      'xdg_cache_home': pre_import.get('xdg_cache_home'),
  }
  if live != expected:
    raise ValueError(
        f'Live cache routing changed after pre-import gate: {live!r}.'
    )
  if pre_import.get('present_forbidden_names') != []:
    raise ValueError('Pre-import cache attestation contained forbidden names.')
  return {
      **live,
      'present_forbidden_names': [],
      'exact_to_pre_import_routing': True,
  }


def cache_output_tree_binding(root: Path) -> dict[str, Any]:
  """Hashes only the isolated role cache, allowing diagnostic outputs."""
  if root.is_symlink() or not root.is_dir():
    raise ValueError(f'Isolated cache root is absent/non-directory: {root}.')
  root = root.resolve()
  files = []
  directories = []
  pending = [root]
  while pending:
    directory = pending.pop()
    directories.append(directory)
    for entry in sorted(directory.iterdir()):
      mode = entry.lstat().st_mode
      if stat.S_ISLNK(mode):
        raise ValueError(f'Isolated cache contains a symlink: {entry}.')
      if stat.S_ISDIR(mode):
        pending.append(entry)
      elif stat.S_ISREG(mode):
        files.append(entry)
      else:
        raise ValueError(f'Isolated cache contains a special entry: {entry}.')
  digest = hashlib.sha256()
  directory_paths = sorted(
      '.' if directory == root else directory.relative_to(root).as_posix()
      for directory in directories
  )
  for relative in directory_paths:
    digest.update(b'D\0')
    digest.update(relative.encode('utf-8'))
    digest.update(b'\0')
  records = {}
  for path in sorted(files):
    relative = path.relative_to(root).as_posix()
    binding = {'sha256': _sha256(path), 'size_bytes': path.stat().st_size}
    records[relative] = binding
    digest.update(b'F\0')
    digest.update(relative.encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(binding['sha256']))
  if root == PREFLIGHT_KERNEL_CACHE_DIR.resolve():
    cache_role = 'external_preflight'
  elif root == V3_3_4_5_EXTERNAL_CACHE_DIR.resolve():
    cache_role = 'external_preflight'
  elif root == V3_3_4_4_EXTERNAL_CACHE_DIR.resolve():
    cache_role = 'external_preflight'
  elif root == V3_3_4_3_PREDECESSOR_PATHS['external_cache'].resolve():
    cache_role = 'external_preflight'
  elif root == MODEL_KERNEL_CACHE_DIR.resolve():
    cache_role = 'model'
  elif root == V3_3_4_5_MODEL_CACHE_DIR.resolve():
    cache_role = 'model'
  else:
    raise ValueError('Isolated cache root is not a frozen role root.')
  return {
      'cache_role': cache_role,
      'cache_root': str(root),
      'triton_cache_dir': str((root / 'triton').resolve()),
      'xdg_cache_home': str((root / 'xdg').resolve()),
      'directory_count': len(directories),
      'directory_paths': directory_paths,
      'file_count': len(files),
      'files': records,
      'tree_sha256': digest.hexdigest(),
      'default_user_cache_paths_eligible': False,
      'diagnostic_outputs_only_no_cache_input': True,
  }


def _tree_digest(paths: Sequence[Path], root: Path) -> str:
  digest = hashlib.sha256()
  for path in sorted(paths):
    relative = str(path.relative_to(root))
    digest.update(relative.encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(_sha256(path)))
  return digest.hexdigest()


def _directory_file_tree_evidence(root: Path) -> dict[str, Any]:
  """Returns exact file-only and directory/file framed tree evidence."""
  if root.is_symlink() or not root.is_dir():
    raise ValueError(f'Archive tree root is absent or unsafe: {root}.')
  root = root.resolve()
  directories = [root]
  files = []
  pending = [root]
  while pending:
    directory = pending.pop()
    for entry in sorted(directory.iterdir()):
      mode = entry.lstat().st_mode
      if stat.S_ISLNK(mode):
        raise ValueError(f'Archive tree contains a symlink: {entry}.')
      if stat.S_ISDIR(mode):
        directories.append(entry)
        pending.append(entry)
      elif stat.S_ISREG(mode):
        files.append(entry)
      else:
        raise ValueError(f'Archive tree contains a special entry: {entry}.')
  directory_paths = sorted(
      '.' if item == root else item.relative_to(root).as_posix()
      for item in directories
  )
  digest = hashlib.sha256()
  for relative in directory_paths:
    digest.update(b'D\0')
    digest.update(relative.encode('utf-8'))
    digest.update(b'\0')
  for path in sorted(files):
    relative = path.relative_to(root).as_posix()
    digest.update(b'F\0')
    digest.update(relative.encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(_sha256(path)))
  return {
      'root': str(root), 'directory_count': len(directory_paths),
      'directory_paths': directory_paths, 'file_count': len(files),
      'file_tree_sha256': _tree_digest(files, root),
      'directory_file_tree_sha256': digest.hexdigest(),
  }


def _lstat_tree_rows(root: Path) -> list[dict[str, Any]]:
  rows = []
  for path in [root, *sorted(root.rglob('*'))]:
    observed = path.lstat()
    relative = '.' if path == root else path.relative_to(root).as_posix()
    if stat.S_ISLNK(observed.st_mode):
      entry_type = 'symlink'
    elif stat.S_ISDIR(observed.st_mode):
      entry_type = 'directory'
    elif stat.S_ISREG(observed.st_mode):
      entry_type = 'regular'
    elif stat.S_ISFIFO(observed.st_mode):
      entry_type = 'fifo'
    elif stat.S_ISSOCK(observed.st_mode):
      entry_type = 'socket'
    else:
      entry_type = 'other'
    if entry_type not in {'directory', 'regular'}:
      raise ValueError(f'Archive tree has unsafe entry: {path}.')
    rows.append({
        'path': relative, 'entry_type': entry_type,
        'mode': f'{stat.S_IMODE(observed.st_mode):04o}',
        'size_bytes': observed.st_size, 'st_dev': observed.st_dev,
        'st_ino': observed.st_ino, 'st_nlink': observed.st_nlink,
    })
  return rows


def _archive_file_binding(
    path: Path, *, include_path: bool = False,
) -> dict[str, Any]:
  observed = path.lstat()
  if (
      stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode)
      or observed.st_nlink != 1
  ):
    raise ValueError(f'Archive file is absent or unsafe: {path}.')
  digest = _sha256(path)
  final = path.lstat()
  if (
      final.st_dev, final.st_ino, final.st_nlink, final.st_mode, final.st_size
  ) != (
      observed.st_dev, observed.st_ino, observed.st_nlink,
      observed.st_mode, observed.st_size,
  ):
    raise ValueError(f'Archive file identity changed: {path}.')
  result = {
      'sha256': digest, 'size_bytes': observed.st_size,
      'mode': f'{stat.S_IMODE(observed.st_mode):04o}',
      'st_dev': observed.st_dev, 'st_ino': observed.st_ino,
      'st_nlink': observed.st_nlink,
  }
  if include_path:
    result = {'path': str(path.resolve()), **result}
  return result


def _archive_tree_binding(root: Path) -> dict[str, Any]:
  evidence = _directory_file_tree_evidence(root)
  root = root.resolve()
  files = {}
  for path in sorted(
      item for item in root.rglob('*') if stat.S_ISREG(item.lstat().st_mode)
  ):
    relative = path.relative_to(root).as_posix()
    files[relative] = _archive_file_binding(path)
  directory_digest = hashlib.sha256()
  for relative in evidence['directory_paths']:
    directory_digest.update(b'D\0')
    directory_digest.update(relative.encode('utf-8'))
    directory_digest.update(b'\0')
  result = {
      'root': str(root), 'file_count': evidence['file_count'],
      'directory_count': evidence['directory_count'],
      'file_bindings': files,
      'file_tree_sha256': evidence['file_tree_sha256'],
      'directory_paths': evidence['directory_paths'],
      'directory_tree_sha256': directory_digest.hexdigest(),
      'directory_file_tree_sha256': evidence['directory_file_tree_sha256'],
  }
  final_rows = _lstat_tree_rows(root)
  return {'tree': result, 'lstat_rows': final_rows}


def _git_blob_binding(
    path: Path, *, commit: str, expected_sha256: str,
    expected_size_bytes: int, expected_mode: str,
    require_live_mode: bool = True,
) -> dict[str, Any]:
  relative = path.resolve().relative_to(_REPO.resolve()).as_posix()
  live = _archive_file_binding(path)
  blob = subprocess.check_output(
      ('git', '-C', str(_REPO), 'show', f'{commit}:{relative}')
  )
  tree_line = subprocess.check_output(
      ('git', '-C', str(_REPO), 'ls-tree', commit, '--', relative),
      text=True,
  ).strip()
  if (
      live['sha256'] != expected_sha256
      or live['size_bytes'] != expected_size_bytes
      or require_live_mode and live['mode'] != expected_mode[-4:]
      or hashlib.sha256(blob).hexdigest() != expected_sha256
      or len(blob) != expected_size_bytes
      or not tree_line.startswith(f'{expected_mode} blob ')
      or not tree_line.endswith(f'\t{relative}')
  ):
    raise ValueError(f'Historical Git blob changed: {commit}:{relative}.')
  return {
      'path': str(path.resolve()), 'sha256': expected_sha256,
      'size_bytes': expected_size_bytes, 'git_mode': expected_mode,
      'authority_commit': commit,
  }


def _require_commit_edge(
    child: str, parent: str, expected_delta: Sequence[str],
) -> None:
  parents = subprocess.check_output(
      ('git', '-C', str(_REPO), 'rev-list', '--parents', '-n', '1', child),
      text=True,
  ).split()
  delta = subprocess.check_output(
      ('git', '-C', str(_REPO), 'diff', '--name-status', parent, child),
      text=True,
  ).splitlines()
  if parents != [child, parent] or delta != list(expected_delta):
    raise ValueError(f'Historical commit edge changed: {parent} -> {child}.')


def _archive_tree_exact(
    root: Path, *, files: Mapping[str, tuple[int, str]],
    directory_paths: Sequence[str], file_tree_sha256: str,
    directory_file_tree_sha256: str,
) -> dict[str, Any]:
  observed = _archive_tree_binding(root)
  tree = observed['tree']
  expected_names = sorted(files)
  if (
      tree['file_count'] != len(files)
      or sorted(tree['file_bindings']) != expected_names
      or tree['directory_paths'] != list(directory_paths)
      or tree['file_tree_sha256'] != file_tree_sha256
      or tree['directory_file_tree_sha256'] != directory_file_tree_sha256
  ):
    raise ValueError(f'Historical archive tree changed: {root}.')
  for relative, (size_bytes, digest) in files.items():
    binding = tree['file_bindings'][relative]
    if (
        binding['size_bytes'] != size_bytes or binding['sha256'] != digest
        or binding['mode'] not in {'0400', '0600'}
        or binding['st_nlink'] != 1
    ):
      raise ValueError(f'Historical archive file changed: {root / relative}.')
  return observed


def validate_prior_v3_3_4_5_controlled_stop_archive() -> dict[str, Any]:
  """Reauthenticates the immutable failed attempt and structural archive."""
  for path in (
      V3_3_4_5_FREEZE_PATH, V3_3_4_5_RUN_DIR,
      V3_3_4_5_PREFLIGHT_DIR, V3_3_4_5_EXTERNAL_CACHE_DIR,
      V3_3_4_5_MODEL_CACHE_DIR, V3_3_4_5_ANALYSIS_AMENDMENT_PATH,
      V3_3_4_5_ANALYSIS_FREEZE_PATH, V3_3_4_5_ANALYSIS_ATTEMPT_DIR,
      V3_3_4_5_ANALYSIS_DIR, V3_3_4_5_OLD_ANALYSIS_ATTEMPT_DIR,
      V3_3_4_5_OLD_ANALYSIS_DIR,
  ):
    _reject_confirmation_path(path)

  model_freeze_binding = _git_blob_binding(
      V3_3_4_5_FREEZE_PATH, commit=V3_3_4_5_COMMIT,
      expected_sha256=V3_3_4_5_FREEZE_SHA256,
      expected_size_bytes=V3_3_4_5_FREEZE_SIZE_BYTES,
      expected_mode='100644',
  )
  model_freeze = json.loads(
      V3_3_4_5_FREEZE_PATH.read_text(encoding='utf-8')
  )
  source_contract = model_freeze.get('source_inventory_contract')
  source_rows = (
      source_contract.get('rows')
      if isinstance(source_contract, Mapping) else None
  )
  if (
      len(model_freeze) != 86
      or len(model_freeze.get('file_sha256', {})) != 132
      or not isinstance(source_rows, list) or len(source_rows) != 132
  ):
    raise ValueError('Historical v3.3.4.5 freeze is not exact 86/132/132.')
  for row in source_rows:
    if not isinstance(row, Mapping) or set(row) != {
        'path', 'sha256', 'size_bytes', 'git_mode'
    }:
      raise ValueError('Historical v3.3.4.5 source row is malformed.')
    relative = row['path']
    path = (_REPO / relative).resolve()
    if (
        not path.is_relative_to(_REPO.resolve())
        or path.relative_to(_REPO.resolve()).as_posix() != relative
    ):
      raise ValueError('Historical v3.3.4.5 source path escaped the repo.')
    _git_blob_binding(
        path, commit=V3_3_4_5_COMMIT,
        expected_sha256=row['sha256'],
        expected_size_bytes=row['size_bytes'],
        expected_mode=row['git_mode'], require_live_mode=False,
    )

  _require_commit_edge(
      V3_3_4_5_ANALYSIS_AMENDMENT_COMMIT, V3_3_4_5_COMMIT,
      [
          'A\texperiments/interpretability/opensplice/v3_wider_mechanism/'
          'encoder_skip_ood_sidecar_analysis_amendment_v3_3_4_5_1.md'
      ],
  )
  _require_commit_edge(
      V3_3_4_5_ANALYSIS_SOURCE_COMMIT,
      V3_3_4_5_ANALYSIS_AMENDMENT_COMMIT,
      [
          'A\texperiments/interpretability/opensplice/'
          'analyze_encoder_skip_ood_sidecar_v3_3_4_5_1.py',
          'A\texperiments/interpretability/opensplice/'
          'analyze_encoder_skip_ood_sidecar_v3_3_4_5_1.sh',
          'A\texperiments/interpretability/opensplice/'
          'analyze_encoder_skip_ood_sidecar_v3_3_4_5_1_test.py',
          'A\texperiments/interpretability/opensplice/'
          'generate_encoder_skip_ood_sidecar_analysis_v3_3_4_5_1_freeze.py',
      ],
  )
  _require_commit_edge(
      V3_3_4_5_ANALYSIS_FREEZE_COMMIT, V3_3_4_5_ANALYSIS_SOURCE_COMMIT,
      [
          'A\texperiments/interpretability/opensplice/'
          'encoder_skip_ood_sidecar_analysis_v3_3_4_5_1_freeze.json'
      ],
  )
  _require_commit_edge(
      V3_3_4_5_ANALYSIS_ARCHIVE_COMMIT, V3_3_4_5_ANALYSIS_FREEZE_COMMIT,
      [
          'A\texperiments/interpretability/opensplice/results/'
          'v3_3_4_5_development_ood_sidecar_analysis_v3_3_4_5_1/ANALYSIS.json',
          'A\texperiments/interpretability/opensplice/results/'
          'v3_3_4_5_development_ood_sidecar_analysis_v3_3_4_5_1/RESULT.md',
          'A\texperiments/interpretability/opensplice/results/'
          'v3_3_4_5_development_ood_sidecar_analysis_v3_3_4_5_1_attempt/'
          'ANALYSIS_ATTEMPT_STARTED.json',
          'A\texperiments/interpretability/opensplice/results/'
          'v3_3_4_5_development_ood_sidecar_analysis_v3_3_4_5_1_attempt/'
          'ANALYSIS_COMPLETE.json',
      ],
  )
  analysis_amendment_binding = _git_blob_binding(
      V3_3_4_5_ANALYSIS_AMENDMENT_PATH,
      commit=V3_3_4_5_ANALYSIS_AMENDMENT_COMMIT,
      expected_sha256=V3_3_4_5_ANALYSIS_AMENDMENT_SHA256,
      expected_size_bytes=V3_3_4_5_ANALYSIS_AMENDMENT_PATH.stat().st_size,
      expected_mode='100644',
  )
  analysis_freeze_binding = _git_blob_binding(
      V3_3_4_5_ANALYSIS_FREEZE_PATH,
      commit=V3_3_4_5_ANALYSIS_FREEZE_COMMIT,
      expected_sha256=V3_3_4_5_ANALYSIS_FREEZE_SHA256,
      expected_size_bytes=V3_3_4_5_ANALYSIS_FREEZE_SIZE_BYTES,
      expected_mode='100644',
  )
  analysis_freeze = json.loads(
      V3_3_4_5_ANALYSIS_FREEZE_PATH.read_text(encoding='utf-8')
  )
  if (
      len(analysis_freeze) != 20
      or analysis_freeze.get('source_inventory_contract', {}).get(
          'row_count'
      ) != 137
      or len(analysis_freeze.get('source_inventory_contract', {}).get(
          'rows', []
      )) != 137
  ):
    raise ValueError('Historical v3.3.4.5.1 freeze is not exact 20/137.')

  preflight_files = {
      '.allocation.lock': (0, hashlib.sha256(b'').hexdigest()),
      '.preflight_0000.reserved': (0, hashlib.sha256(b'').hexdigest()),
      'preflight_0000.json': (
          50472, 'cefe7d95c67868668f912575c8e00fecc9813ec011d395b8332f65d0a2c7d785'
      ),
      'preflight_0000.stderr.log': (0, hashlib.sha256(b'').hexdigest()),
      'preflight_0000.stdout.log': (0, hashlib.sha256(b'').hexdigest()),
  }
  external_cache_files = {
      '.v3345.tmp.2777420.000001.7a795e5eda1e9fcf14f19a8d62c7960f': (
          39, 'd7e55ae0ed0453b3d29f92731588b9626f10d5814b0f0ecd3198ced485940d44'
      ),
      'atomic_publication_probe_v3_3_4_5.txt': (
          49, '7ffb46419c01255944db76c4530e7943574212aa4c4595fa85254bc9d21d6bd1'
      ),
  }
  model_cache_files = {
      'xdg/matplotlib/fontlist-v3.11.0.json': (
          163240,
          'a933e64e6dbc737661309fbe5a1a8402aa192976c6758f6f4a8ac2f37c5010f9',
      ),
  }
  roots = {
      'model_run': _archive_tree_exact(
          V3_3_4_5_RUN_DIR, files=V3_3_4_5_RUN_FILES,
          directory_paths=['.', 'compiler', 'compiler/eight_row'],
          file_tree_sha256=V3_3_4_5_RUN_TREE_SHA256,
          directory_file_tree_sha256=V3_3_4_5_RUN_DIRECTORY_FILE_TREE_SHA256,
      ),
      'device_preflight': _archive_tree_exact(
          V3_3_4_5_PREFLIGHT_DIR, files=preflight_files,
          directory_paths=['.'],
          file_tree_sha256=(
              'ae277eafa4f7f20bfa74c3a0a1bbaa0f51468cac945d29d0c49cab699738ecfd'
          ),
          directory_file_tree_sha256=(
              'cc106b406da58ddd95611aef7e471f5a5cefd96e302ebb91ea4ef9e28a618c87'
          ),
      ),
      'external_cache': _archive_tree_exact(
          V3_3_4_5_EXTERNAL_CACHE_DIR, files=external_cache_files,
          directory_paths=['.', 'triton', 'xdg'],
          file_tree_sha256=(
              '3bd7b53ba7ab1dae7161999ff907137f82ee6d7f322512a3221646f66bb1e975'
          ),
          directory_file_tree_sha256=(
              'd040af81aa50fbe28e0523747355f84d851f36f39e586294b24dd994f69f66a0'
          ),
      ),
      'model_cache': _archive_tree_exact(
          V3_3_4_5_MODEL_CACHE_DIR, files=model_cache_files,
          directory_paths=['.', 'triton', 'xdg', 'xdg/matplotlib'],
          file_tree_sha256=(
              '487c67a6dbb251aca190ac9eda5d2425c3584febc9ad63e60d0812c7f2fb69ea'
          ),
          directory_file_tree_sha256=(
              '51fe59713c301342bf5bb161f26b9e4ee6828e508b96e4dbd21c6efcdde1115e'
          ),
      ),
      'analysis_attempt': _archive_tree_exact(
          V3_3_4_5_ANALYSIS_ATTEMPT_DIR,
          files=V3_3_4_5_ANALYSIS_ATTEMPT_FILES, directory_paths=['.'],
          file_tree_sha256=(
              '58a2a295adb013cc41fc3dd4b611d647d21638de259563ba0a45c597f1c2d847'
          ),
          directory_file_tree_sha256=(
              '48bceef0155a57d541ec958a920b915bce4de550be555e315ccbc2da8d566f26'
          ),
      ),
      'analysis_output': _archive_tree_exact(
          V3_3_4_5_ANALYSIS_DIR, files=V3_3_4_5_ANALYSIS_FILES,
          directory_paths=['.'],
          file_tree_sha256=(
              '221df53d78db5fedfdde1f1003a80ed7d2a7135c27c03644e6aa751d7256ac57'
          ),
          directory_file_tree_sha256=(
              '2fba6fe13289faf20702e48e3dd6b572c1597bb78342a4986509c4fbefd304ea'
          ),
      ),
  }

  start_path = V3_3_4_5_RUN_DIR / 'ATTEMPT_STARTED.json'
  terminal_path = V3_3_4_5_RUN_DIR / 'RUN_COMPLETE.json'
  manifest_path = V3_3_4_5_RUN_DIR / 'RAW_MANIFEST.json'
  diagnostic_path = (
      V3_3_4_5_RUN_DIR / 'compiler/eight_row/COMPILER_DIAGNOSTIC_FAILURE.json'
  )
  start = json.loads(start_path.read_text(encoding='utf-8'))
  terminal = json.loads(terminal_path.read_text(encoding='utf-8'))
  manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
  diagnostic = json.loads(diagnostic_path.read_text(encoding='utf-8'))
  no_science_keys = (
      'scientific_summary_computed', 'donor_normalization_computed',
      'shapley_or_nomination_computed',
      'interaction_or_resolution_computed', 'nomination_performed',
      'combined_analysis_permitted',
  )
  expected_fresh_paths = {
      'analysis_attempt': str(V3_3_4_5_OLD_ANALYSIS_ATTEMPT_DIR.resolve()),
      'analysis_output': str(V3_3_4_5_OLD_ANALYSIS_DIR.resolve()),
      'device_preflight': str(V3_3_4_5_PREFLIGHT_DIR.resolve()),
      'model_kernel_cache': str(V3_3_4_5_MODEL_CACHE_DIR.resolve()),
      'model_run': str(V3_3_4_5_RUN_DIR.resolve()),
      'preflight_kernel_cache': str(V3_3_4_5_EXTERNAL_CACHE_DIR.resolve()),
  }
  old_analysis_absence = {}
  for role, path in (
      ('analysis_attempt', V3_3_4_5_OLD_ANALYSIS_ATTEMPT_DIR),
      ('analysis_output', V3_3_4_5_OLD_ANALYSIS_DIR),
  ):
    if path.exists() or path.is_symlink():
      raise ValueError(f'Permanently absent old v3.3.4.5 path appeared: {path}.')
    old_analysis_absence[role] = {'path': str(path.resolve()), 'absent': True}
  empty_sha = hashlib.sha256(b'').hexdigest()
  if (
      start.get('status') != 'attempt_started'
      or start.get('fresh_paths') != expected_fresh_paths
      or any(start.get(key) is not False for key in no_science_keys)
      or start.get('confirmation_model_calls') != 0
      or terminal.get('status')
      != 'controlled_stop_diagnostic_provenance_failure'
      or terminal.get('stop_reason') != 'diagnostic_persistence_failure'
      or terminal.get('failure') != {
          'message': "'eight_row_compiler'",
          'traceback': "DiagnosticPersistenceFailure: 'eight_row_compiler'\n",
          'type': 'DiagnosticPersistenceFailure',
      }
      or terminal.get('eight_row_lower_attempt_count') != 1
      or terminal.get('eight_row_compile_attempt_count') != 1
      or terminal.get('eight_row_successful_compile_count') != 1
      or any(terminal.get(key) != 0 for key in (
          'six_row_compile_count', 'identity_rerun_count',
          'main_cube_rerun_count', 'old_ood_records_reused',
          'model_apply_attempt_count', 'model_apply_success_count',
          'valid_record_count', 'confirmation_model_calls',
      ))
      or any(terminal.get(key) is not False for key in no_science_keys)
      or terminal.get('dispatch_journal') != {
          'completed_bindings': {}, 'completed_count': 0,
          'completed_prefix_exact': True, 'completed_tree_sha256': empty_sha,
          'started_bindings': {}, 'started_count': 0,
          'started_prefix_exact': True, 'started_tree_sha256': empty_sha,
      }
      or manifest.get('status') != 'empty_controlled_stop'
      or manifest.get('valid_artifact_count') != 0
      or manifest.get('valid_recipient_anchor_pairs') != []
      or diagnostic.get('status') != 'diagnostic_provenance_failure'
      or diagnostic.get('failure') != terminal.get('failure')
      or diagnostic.get('no_dispatch') is not True
  ):
    raise ValueError('Historical v3.3.4.5 controlled-stop semantics changed.')

  archive_start_path = (
      V3_3_4_5_ANALYSIS_ATTEMPT_DIR / 'ANALYSIS_ATTEMPT_STARTED.json'
  )
  archive_complete_path = (
      V3_3_4_5_ANALYSIS_ATTEMPT_DIR / 'ANALYSIS_COMPLETE.json'
  )
  analysis_path = V3_3_4_5_ANALYSIS_DIR / 'ANALYSIS.json'
  result_path = V3_3_4_5_ANALYSIS_DIR / 'RESULT.md'
  archive_start = json.loads(archive_start_path.read_text(encoding='utf-8'))
  analysis = json.loads(analysis_path.read_text(encoding='utf-8'))
  archive_complete = json.loads(
      archive_complete_path.read_text(encoding='utf-8')
  )
  archive_bindings = {
      'analysis_attempt_start': _archive_file_binding(
          archive_start_path, include_path=True
      ),
      'analysis_complete': _archive_file_binding(
          archive_complete_path, include_path=True
      ),
      'analysis': _archive_file_binding(analysis_path, include_path=True),
      'result': _archive_file_binding(result_path, include_path=True),
  }
  linkage_bindings = {
      name: {key: value for key, value in binding.items() if key in {
          'path', 'sha256', 'size_bytes'
      }} for name, binding in archive_bindings.items()
  }
  if (
      archive_start.get('status') != 'started'
      or analysis.get('status')
      != 'complete_controlled_stop_structural_archive'
      or analysis.get('decision')
      != 'controlled_stop_diagnostic_provenance_failure'
      or any(analysis.get(key) is not False for key in no_science_keys)
      or analysis.get('run_binding') != roots['model_run']['tree']
      or analysis.get('preflight_binding') != roots['device_preflight']['tree']
      or analysis.get('external_cache_binding')
      != roots['external_cache']['tree']
      or analysis.get('model_cache_binding') != roots['model_cache']['tree']
      or analysis.get('analysis_attempt_start_binding')
      != linkage_bindings['analysis_attempt_start']
      or archive_complete.get('status') != 'complete'
      or archive_complete.get('start_binding')
      != linkage_bindings['analysis_attempt_start']
      or archive_complete.get('analysis_binding') != linkage_bindings['analysis']
      or archive_complete.get('result_binding') != linkage_bindings['result']
  ):
    raise ValueError('Historical v3.3.4.5.1 archive linkage changed.')

  return {
      'model_attempt': {
          'implementation_commit': V3_3_4_5_COMMIT,
          'freeze_binding': model_freeze_binding,
          'freeze_top_level_key_count': 86,
          'source_row_count': 132,
          'source_rows_content_binding': canonical_content_binding(source_rows),
          'six_root_filesystem_state': roots,
          'old_analysis_path_absence': old_analysis_absence,
          'start_binding': _archive_file_binding(start_path, include_path=True),
          'terminal_binding': _archive_file_binding(
              terminal_path, include_path=True
          ),
          'raw_manifest_binding': _archive_file_binding(
              manifest_path, include_path=True
          ),
          'compiler_diagnostic_binding': _archive_file_binding(
              diagnostic_path, include_path=True
          ),
          'status': terminal['status'], 'stop_reason': terminal['stop_reason'],
          'failure': terminal['failure'], 'dispatch_count': 0,
          'model_apply_count': 0, 'raw_record_count': 0,
          'eight_row_lower_count': 1, 'eight_row_compile_count': 1,
          'six_row_compile_count': 0, 'identity_rerun_count': 0,
          'main_cube_rerun_count': 0, 'confirmation_model_calls': 0,
          'no_science_flags': {key: False for key in no_science_keys},
      },
      'structural_archive': {
          'amendment_commit': V3_3_4_5_ANALYSIS_AMENDMENT_COMMIT,
          'source_commit': V3_3_4_5_ANALYSIS_SOURCE_COMMIT,
          'freeze_commit': V3_3_4_5_ANALYSIS_FREEZE_COMMIT,
          'archive_commit': V3_3_4_5_ANALYSIS_ARCHIVE_COMMIT,
          'amendment_binding': analysis_amendment_binding,
          'freeze_binding': analysis_freeze_binding,
          'freeze_top_level_key_count': 20, 'source_row_count': 137,
          'artifact_bindings': archive_bindings,
          'status': analysis['status'], 'decision': analysis['decision'],
          'no_science_flags': {key: False for key in no_science_keys},
          'start_analysis_complete_linkage_exact': True,
      },
      'old_model_or_analyzer_retry_permitted': False,
      'old_paths_are_cache_inputs': False,
  }


def validate_preflight_version_contract(
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
  """Proves the repaired preflight literal before any production allocation."""
  source = _HERE / 'run_device_preflight_v3_3_4_6.py'
  relative = source.relative_to(_REPO).as_posix()
  file_sha = frozen.get('file_sha256')
  if not isinstance(file_sha, Mapping) or relative not in file_sha:
    raise ValueError('Frozen preflight producer binding is absent.')
  _validate_file(source, str(file_sha[relative]), 'v3.3.4.6 preflight source')
  try:
    tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
  except (OSError, SyntaxError, UnicodeError) as error:
    raise ValueError('Preflight producer source cannot be parsed.') from error
  assignments = []
  for node in tree.body:
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
      targets = node.targets if isinstance(node, ast.Assign) else [node.target]
      if any(
          isinstance(target, ast.Name)
          and target.id == 'PREFLIGHT_SCRIPT_VERSION'
          for target in targets
      ):
        assignments.append(ast.literal_eval(node.value))
  expected = 'opensplice-device-preflight-v3.3.4.6'
  observed = {
      'freeze': frozen.get('preflight_script_version'),
      'bootstrap': PREFLIGHT_SCRIPT_VERSION,
      'producer': assignments[0] if len(assignments) == 1 else None,
  }
  if len(assignments) != 1 or any(value != expected for value in observed.values()):
    raise ValueError(
        'v3.3.4.6 preflight version three-way proof failed before allocation.'
    )
  return {
      'preflight_script_version': expected,
      'freeze_equals_bootstrap': True,
      'bootstrap_equals_producer_literal': True,
      'producer_source_binding': {
          'path': str(source.resolve()),
          'sha256': str(file_sha[relative]),
          'size_bytes': source.stat().st_size,
      },
      'validated_before_allocation_or_registration': True,
  }


def _validate_predecessor_git_blob(
    relative: str, expected_sha256: str,
) -> None:
  path = (_REPO / relative).resolve()
  _validate_file(path, expected_sha256, f'v3.3.4.3 predecessor {relative}')
  payload = subprocess.run(
      ('git', '-C', str(_REPO), 'show', f'{V3_3_4_3_COMMIT}:{relative}'),
      check=True, capture_output=True,
  ).stdout
  if hashlib.sha256(payload).hexdigest() != expected_sha256:
    raise ValueError(f'v3.3.4.3 committed predecessor bytes changed: {relative}.')


def _predecessor_absence_map() -> dict[str, dict[str, Any]]:
  result = {}
  for version, paths in (
      ('v3_3_4', V3_3_4_PREDECESSOR_PATHS),
      ('v3_3_4_1', V3_3_4_1_PREDECESSOR_PATHS),
      ('v3_3_4_2', V3_3_4_2_PREDECESSOR_PATHS),
  ):
    for role, path in sorted(paths.items()):
      name = 'device_preflight' if role == 'external_preflight' else role
      key = f'{version}.{name}'
      if path.exists() or path.is_symlink():
        raise FileExistsError(f'Consumed-prefix absent path appeared: {path}.')
      result[key] = {'path': str(path.resolve()), 'absent': True}
  for role, path in sorted(V3_3_4_3_PREDECESSOR_PATHS.items()):
    if role == 'external_cache':
      continue
    key = f'v3_3_4_3.{role}'
    if path.exists() or path.is_symlink():
      raise FileExistsError(f'Consumed v3.3.4.3 sibling path appeared: {path}.')
    result[key] = {'path': str(path.resolve()), 'absent': True}
  if len(result) != 23:
    raise RuntimeError('Consumed-prefix absence map is not exactly 23 paths.')
  return dict(sorted(result.items()))


def validate_prior_v3_3_4_3_consumed_preflight_prefix() -> dict[str, Any]:
  """Reauthenticates the immutable directory-only v3.3.4.3 prefix."""
  freeze_relative = V3_3_4_3_FREEZE_PATH.relative_to(_REPO).as_posix()
  _validate_predecessor_git_blob(freeze_relative, V3_3_4_3_FREEZE_SHA256)
  if (
      V3_3_4_3_FREEZE_PATH.stat().st_size != V3_3_4_3_FREEZE_SIZE_BYTES
      or stat.S_IMODE(V3_3_4_3_FREEZE_PATH.stat().st_mode) != 0o644
  ):
    raise ValueError('v3.3.4.3 predecessor freeze size/mode changed.')
  predecessor_freeze = json.loads(
      V3_3_4_3_FREEZE_PATH.read_text(encoding='utf-8')
  )
  if (
      len(predecessor_freeze) != 84
      or len(predecessor_freeze.get('file_sha256', {})) != 108
      or len(predecessor_freeze.get('source_inventory_contract', {}).get(
          'rows', []
      )) != 108
  ):
    raise ValueError('v3.3.4.3 predecessor freeze counts changed.')
  for relative, digest in sorted(V3_3_4_3_SOURCE_BINDINGS.items()):
    _validate_predecessor_git_blob(relative, digest)

  root = V3_3_4_3_PREDECESSOR_PATHS['external_cache']
  expected_rows = (
      ('.', 66307, 140791433, 4, 4096),
      ('triton', 66307, 140791434, 2, 4096),
      ('xdg', 66307, 140791435, 2, 4096),
  )
  rows = []
  for relative, expected_dev, expected_ino, expected_nlink, expected_size in expected_rows:
    path = root if relative == '.' else root / relative
    observed = path.lstat()
    row = {
        'path': relative, 'entry_type': 'directory', 'mode': '0700',
        'st_dev': observed.st_dev, 'st_ino': observed.st_ino,
        'st_nlink': observed.st_nlink, 'size_bytes': observed.st_size,
    }
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o700
        or (
            observed.st_dev, observed.st_ino, observed.st_nlink,
            observed.st_size,
        ) != (expected_dev, expected_ino, expected_nlink, expected_size)
    ):
      raise ValueError(f'Consumed v3.3.4.3 lstat row changed: {relative}.')
    rows.append(row)
  cache_binding = cache_output_tree_binding(root)
  if (
      cache_binding.get('directory_paths') != ['.', 'triton', 'xdg']
      or cache_binding.get('directory_count') != 3
      or cache_binding.get('files') != {}
      or cache_binding.get('file_count') != 0
      or cache_binding.get('tree_sha256')
      != '9162636192082efbef80c9b37dd3ebc138aa094f70111874b9dad70e468af1af'
      or canonical_content_binding(cache_binding) != {
          'sha256': 'd53f56fabd83cb43b79d7a7c73c5b56727d846713e96ea73ef9b26360e18bdea',
          'size_bytes': 745,
      }
  ):
    raise ValueError('Consumed v3.3.4.3 cache binding changed.')
  prefix = {
      'status': 'consumed_external_preflight_freeze_validation_failure',
      'predecessor_commit': V3_3_4_3_COMMIT,
      'predecessor_freeze': {
          'path': str(V3_3_4_3_FREEZE_PATH.resolve()),
          'sha256': V3_3_4_3_FREEZE_SHA256,
          'size_bytes': V3_3_4_3_FREEZE_SIZE_BYTES,
          'git_mode': '100644',
          'top_level_key_count': 84,
          'file_sha256_count': 108,
          'source_row_count': 108,
      },
      'failure_stage': 'preflight_freeze_validation',
      'failure_type': 'ValueError',
      'failure_message': (
          'v3.3.4.3 preflight freeze mismatch: preflight_script_version.'
      ),
      'traceback_provenance': {
          'storage': 'coordinator_captured_not_persisted',
          'sha256': '8b6d0f7575adfc66032ba56d3ab5373f05b0bb1e85b2e39c94e2a82a356e39e9',
          'size_bytes': 953, 'session_id': None,
          'captured_at_unix_s': None,
          'wall_clock_timestamp_available': False,
      },
      'cache_tree_binding': cache_binding,
      'directory_lstat_rows': rows,
      'other_predecessor_paths_absent': _predecessor_absence_map(),
      'no_jax_or_model_access': True,
      'no_gpu_or_confirmation_access': True,
      'immutable_and_not_cache_input': True,
  }
  if set(prefix) != {
      'status', 'predecessor_commit', 'predecessor_freeze',
      'failure_stage', 'failure_type', 'failure_message',
      'traceback_provenance', 'cache_tree_binding', 'directory_lstat_rows',
      'other_predecessor_paths_absent', 'no_jax_or_model_access',
      'no_gpu_or_confirmation_access', 'immutable_and_not_cache_input',
  }:
    raise RuntimeError('Consumed-prefix outer schema changed.')
  return prefix


def validate_recorded_prior_v3_3_4_3_consumed_preflight_prefix(
    recorded: Any, binding: Any,
) -> dict[str, Any]:
  """Reauthenticates an embedded consumed-prefix object and its exact bytes."""
  live = validate_prior_v3_3_4_3_consumed_preflight_prefix()
  expected_binding = canonical_content_binding(live)
  if not isinstance(recorded, Mapping) or dict(recorded) != live:
    raise ValueError('Recorded v3.3.4.3 consumed-prefix object changed.')
  if not isinstance(binding, Mapping) or dict(binding) != expected_binding:
    raise ValueError('Recorded v3.3.4.3 consumed-prefix binding changed.')
  return live


def _validate_v3_3_4_4_git_blob(binding: Mapping[str, Any]) -> None:
  path = Path(str(binding['path'])).absolute()
  if not path.is_relative_to(_REPO.resolve()):
    raise ValueError('v3.3.4.4 source binding escaped the repository.')
  relative = path.relative_to(_REPO.resolve()).as_posix()
  _validate_file(path, str(binding['sha256']), f'v3.3.4.4 {relative}')
  if path.stat().st_size != binding['size_bytes']:
    raise ValueError(f'v3.3.4.4 source size changed: {relative}.')
  committed = subprocess.run(
      ('git', '-C', str(_REPO), 'show', f'{V3_3_4_4_COMMIT}:{relative}'),
      check=True, capture_output=True,
  ).stdout
  if (
      hashlib.sha256(committed).hexdigest() != binding['sha256']
      or len(committed) != binding['size_bytes']
  ):
    raise ValueError(f'v3.3.4.4 committed source changed: {relative}.')


def _exact_directory_lstat_row(
    path: Path, *, relative: str,
    expected: tuple[int, int, int, int],
) -> dict[str, Any]:
  observed = path.lstat()
  if (
      stat.S_ISLNK(observed.st_mode)
      or not stat.S_ISDIR(observed.st_mode)
      or stat.S_IMODE(observed.st_mode) != 0o700
      or (
          observed.st_dev, observed.st_ino, observed.st_nlink,
          observed.st_size,
      ) != expected
  ):
    raise ValueError(f'v3.3.4.4 directory lstat row changed: {relative}.')
  fd = os.open(
      path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
  )
  try:
    opened = os.fstat(fd)
    if (
        opened.st_dev != observed.st_dev
        or opened.st_ino != observed.st_ino
        or not stat.S_ISDIR(opened.st_mode)
        or stat.S_IMODE(opened.st_mode) != 0o700
        or opened.st_nlink != expected[2]
        or opened.st_size != expected[3]
    ):
      raise ValueError(f'v3.3.4.4 directory changed while opened: {relative}.')
  finally:
    os.close(fd)
  return {
      'path': relative, 'entry_type': 'directory', 'mode': '0700',
      'st_dev': observed.st_dev, 'st_ino': observed.st_ino,
      'st_nlink': observed.st_nlink, 'size_bytes': observed.st_size,
  }


def _exact_regular_lstat_binding(
    path: Path, *, relative: str, mode: str, st_dev: int, st_ino: int,
    st_nlink: int, size_bytes: int, sha256: str,
) -> dict[str, Any]:
  observed = path.lstat()
  if (
      stat.S_ISLNK(observed.st_mode)
      or not stat.S_ISREG(observed.st_mode)
      or f'{stat.S_IMODE(observed.st_mode):04o}' != mode
      or (
          observed.st_dev, observed.st_ino, observed.st_nlink,
          observed.st_size,
      ) != (st_dev, st_ino, st_nlink, size_bytes)
  ):
    raise ValueError(f'v3.3.4.4 file lstat row changed: {relative}.')
  fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
  try:
    opened = os.fstat(fd)
    payload = _read_fd_bytes(fd)
    if (
        opened.st_dev != observed.st_dev
        or opened.st_ino != observed.st_ino
        or not stat.S_ISREG(opened.st_mode)
        or f'{stat.S_IMODE(opened.st_mode):04o}' != mode
        or opened.st_nlink != st_nlink
        or opened.st_size != size_bytes
        or hashlib.sha256(payload).hexdigest() != sha256
        or len(payload) != size_bytes
    ):
      raise ValueError(f'v3.3.4.4 file bytes changed: {relative}.')
  finally:
    os.close(fd)
  return {
      'path': relative, 'sha256': sha256, 'size_bytes': size_bytes,
      'mode': mode, 'st_dev': st_dev, 'st_ino': st_ino,
      'st_nlink': st_nlink,
  }


def _v3_3_4_4_other_paths_absent() -> dict[str, dict[str, Any]]:
  result = {}
  for role, path in sorted(V3_3_4_4_OTHER_PATHS.items()):
    _reject_confirmation_path(path)
    if path.exists() or path.is_symlink():
      raise FileExistsError(f'Consumed v3.3.4.4 sibling path appeared: {path}.')
    result[role] = {'path': str(path.resolve()), 'absent': True}
  return result


def validate_prior_v3_3_4_4_consumed_preflight_prefix() -> dict[str, Any]:
  """Reauthenticates the exact successful-preflight/failing-parent prefix."""
  freeze_binding = {
      'path': str(V3_3_4_4_FREEZE_PATH.absolute()),
      'sha256': V3_3_4_4_FREEZE_SHA256,
      'size_bytes': V3_3_4_4_FREEZE_SIZE_BYTES,
      'git_mode': '100644', 'top_level_key_count': 85,
      'file_sha256_count': 120, 'source_row_count': 120,
  }
  _validate_v3_3_4_4_git_blob({
      'path': freeze_binding['path'], 'sha256': freeze_binding['sha256'],
      'size_bytes': freeze_binding['size_bytes'],
  })
  frozen = json.loads(V3_3_4_4_FREEZE_PATH.read_text(encoding='utf-8'))
  if (
      stat.S_IMODE(V3_3_4_4_FREEZE_PATH.stat().st_mode) != 0o644
      or len(frozen) != 85
      or len(frozen.get('file_sha256', {})) != 120
      or len(frozen.get('source_inventory_contract', {}).get('rows', []))
      != 120
  ):
    raise ValueError('v3.3.4.4 predecessor freeze contract changed.')
  _validate_v3_3_4_4_git_blob(V3_3_4_4_LAUNCHER_BINDING)
  _validate_v3_3_4_4_git_blob(V3_3_4_4_BOOTSTRAP_BINDING)

  preflight_root_row = _exact_directory_lstat_row(
      V3_3_4_4_PREFLIGHT_DIR, relative='.',
      expected=(66307, 140791442, 2, 4096),
  )
  file_contract = {
      '.allocation.lock': (
          '0600', 66307, 140791443, 1, 0,
          'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
      ),
      '.preflight_0000.reserved': (
          '0400', 66307, 140791444, 1, 0,
          'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
      ),
      'preflight_0000.json': (
          '0400', 66307, 140791447, 1, 27062,
          'a240bf223dd62ebc53b84da35bb614df7987254c3694d7f07aae9785adec3801',
      ),
      'preflight_0000.stderr.log': (
          '0400', 66307, 140791446, 1, 0,
          'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
      ),
      'preflight_0000.stdout.log': (
          '0400', 66307, 140791445, 1, 0,
          'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
      ),
  }
  observed_paths = _strict_file_tree(V3_3_4_4_PREFLIGHT_DIR)
  if [path.name for path in observed_paths] != sorted(file_contract):
    raise ValueError('v3.3.4.4 preflight membership changed.')
  preflight_files = {}
  for relative, values in sorted(file_contract.items()):
    preflight_files[relative] = _exact_regular_lstat_binding(
        V3_3_4_4_PREFLIGHT_DIR / relative, relative=relative,
        mode=values[0], st_dev=values[1], st_ino=values[2],
        st_nlink=values[3], size_bytes=values[4], sha256=values[5],
    )
  if (
      _tree_digest(observed_paths, V3_3_4_4_PREFLIGHT_DIR)
      != 'f009ba6fe67a715301b443940876be8f85998a50f71f320d0dc5e3dd52dfd6e5'
  ):
    raise ValueError('v3.3.4.4 preflight file tree changed.')
  directory_aware = hashlib.sha256()
  directory_aware.update(b'D\0.\0')
  for relative, binding in sorted(preflight_files.items()):
    directory_aware.update(b'F\0')
    directory_aware.update(relative.encode('utf-8'))
    directory_aware.update(b'\0')
    directory_aware.update(bytes.fromhex(binding['sha256']))
  if directory_aware.hexdigest() != (
      '1a343cacb96cc1a1c88735c6a3bb8edfb0b71c1df89e924f52e099c30e3217f5'
  ):
    raise ValueError('v3.3.4.4 directory-aware preflight tree changed.')
  record_path = V3_3_4_4_PREFLIGHT_DIR / 'preflight_0000.json'
  record = json.loads(record_path.read_text(encoding='utf-8'))
  if len(record) != 20 or canonical_content_binding(record) != {
      'sha256': '9b1a5e3bbc9845d04430c259ed39db0f39f31b56128f022443382b91e6027285',
      'size_bytes': 22193,
  }:
    raise ValueError('v3.3.4.4 parsed preflight record changed.')
  observation = record['observation']
  nvidia = observation['nvidia_smi']['parsed_single_gpu']
  record_semantics = {
      'status': record['status'], 'failure': record['failure'],
      'preflight_attempt_number': record['preflight_attempt_number'],
      'script_version': record['script_version'],
      'freeze_sha256': record['freeze_sha256'],
      'external_pid': observation['pid'],
      'jax_default_backend': observation['jax_default_backend'],
      'jax_gpu_device_count': len(observation['jax_gpu_devices']),
      'device_kind': observation['jax_gpu_devices'][0]['device_kind'],
      'device_uuid': nvidia['uuid'],
      'compute_capability': nvidia['compute_capability'],
      'no_jit_or_array_kernel': record['no_jit_or_array_kernel'],
      'no_model_or_biological_access': record['no_model_or_biological_access'],
      'external_cache_hit': record['external_cache_hit_evidence']['cache_hit'],
  }
  expected_record_semantics = {
      'status': 'pass', 'failure': None, 'preflight_attempt_number': 0,
      'script_version': 'opensplice-device-preflight-v3.3.4.4',
      'freeze_sha256': V3_3_4_4_FREEZE_SHA256,
      'external_pid': 2696297, 'jax_default_backend': 'gpu',
      'jax_gpu_device_count': 1, 'device_kind': 'NVIDIA GeForce RTX 3090',
      'device_uuid': 'GPU-64111645-1e42-a96d-f192-4abbec4b8090',
      'compute_capability': '8.6', 'no_jit_or_array_kernel': True,
      'no_model_or_biological_access': True, 'external_cache_hit': False,
  }
  if record_semantics != expected_record_semantics:
    raise ValueError('v3.3.4.4 preflight semantics changed.')

  cache_rows = [
      _exact_directory_lstat_row(
          V3_3_4_4_EXTERNAL_CACHE_DIR, relative='.',
          expected=(66307, 140791437, 4, 4096),
      ),
      _exact_directory_lstat_row(
          V3_3_4_4_EXTERNAL_CACHE_DIR / 'triton', relative='triton',
          expected=(66307, 140791438, 2, 4096),
      ),
      _exact_directory_lstat_row(
          V3_3_4_4_EXTERNAL_CACHE_DIR / 'xdg', relative='xdg',
          expected=(66307, 140791439, 2, 4096),
      ),
  ]
  cache_file_contract = {
      '.v3344.tmp.2696297.000001.55167cfd266423a5ba861df8ca40686d': (
          '0400', 66307, 140791441, 1, 39,
          'a1e62f4f34497aa5e72ece0670f1d865cd6eaacdcdfbacb00c39648d9e83f14f',
      ),
      'atomic_publication_probe_v3_3_4_4.txt': (
          '0400', 66307, 140791440, 1, 49,
          '47efa8c868d4d9455730ad1e89d6e44afee44172f0d2af7521d8574b7d85ecc9',
      ),
  }
  cache_file_lstats = {
      relative: _exact_regular_lstat_binding(
          V3_3_4_4_EXTERNAL_CACHE_DIR / relative, relative=relative,
          mode=values[0], st_dev=values[1], st_ino=values[2],
          st_nlink=values[3], size_bytes=values[4], sha256=values[5],
      )
      for relative, values in sorted(cache_file_contract.items())
  }
  cache_binding = cache_output_tree_binding(V3_3_4_4_EXTERNAL_CACHE_DIR)
  if (
      cache_binding.get('directory_paths') != ['.', 'triton', 'xdg']
      or cache_binding.get('directory_count') != 3
      or cache_binding.get('file_count') != 2
      or cache_binding.get('files') != {
          relative: {
              'sha256': binding['sha256'],
              'size_bytes': binding['size_bytes'],
          }
          for relative, binding in cache_file_lstats.items()
      }
      or cache_binding.get('tree_sha256')
      != '3a294e09038311b8bad85836c6983da31f50fdeef3365b844e8842922d33acba'
      or canonical_content_binding(cache_binding) != {
          'sha256': '88c2a4cde3a9881f76dc719b48dbd7f051b5841843dc5439a8e7d2349aabbc46',
          'size_bytes': 1033,
      }
  ):
    raise ValueError('v3.3.4.4 external-cache binding changed.')
  probe = record['atomic_publication_probe']
  if (
      probe.get('successful_final_binding')
      != cache_file_lstats['atomic_publication_probe_v3_3_4_4.txt']
      or probe.get('collision_temp_binding') != cache_file_lstats[
          '.v3344.tmp.2696297.000001.55167cfd266423a5ba861df8ca40686d'
      ]
      or canonical_content_binding(probe) != {
      'sha256': 'a25798b7c788ce614d10a7cc0d07f1795ebaf6fd9e928dcef78e096323d9bf70',
      'size_bytes': 738,
      }
  ):
    raise ValueError('v3.3.4.4 publication probe changed.')

  prefix = {
      'status': (
          'consumed_successful_external_preflight_then_parent_role_routing_failure'
      ),
      'predecessor_commit': V3_3_4_4_COMMIT,
      'predecessor_freeze': freeze_binding,
      'failure_stage': 'parent_completed_external_preflight_validation',
      'failure_type': 'FileExistsError',
      'failure_message': (
          'Preflight directory exists before the external_preflight process.'
      ),
      'traceback_provenance': {
          'storage': 'coordinator_captured_not_persisted',
          'sha256': '03cf721c145a2d70764455c8ab197482aed52a062d10c1ca29818bb0c1c8c3d3',
          'size_bytes': 1168, 'session_id': None,
          'captured_at_unix_s': None,
          'wall_clock_timestamp_available': False,
      },
      'root_cause': {
          'parent_ambient_cache_role': 'external_preflight',
          'parent_ambient_cache_root': str(
              V3_3_4_4_EXTERNAL_CACHE_DIR.resolve()
          ),
          'called_validator': 'validate_preflight_state_for_role',
          'selected_branch': 'external_preflight_entry_absence',
          'rejected_state': 'completed_preflight_directory_present',
          'required_validator': 'validate_completed_external_preflight_state',
          'failure_before_model_cache_allocation': True,
          'failure_before_model_start': True,
          'launcher_source_binding': dict(V3_3_4_4_LAUNCHER_BINDING),
          'bootstrap_source_binding': dict(V3_3_4_4_BOOTSTRAP_BINDING),
      },
      'external_preflight_archive': {
          'root': str(V3_3_4_4_PREFLIGHT_DIR.resolve()),
          'directory_lstat_rows': [preflight_root_row],
          'directory_count': 1, 'directory_paths': ['.'],
          'file_count': 5, 'files': preflight_files,
          'file_tree_sha256': (
              'f009ba6fe67a715301b443940876be8f85998a50f71f320d0dc5e3dd52dfd6e5'
          ),
          'record_binding': {
              'path': str(record_path.resolve()),
              'sha256': file_contract['preflight_0000.json'][5],
              'size_bytes': 27062, 'mode': '0400',
          },
          'directory_aware_tree_sha256': (
              '1a343cacb96cc1a1c88735c6a3bb8edfb0b71c1df89e924f52e099c30e3217f5'
          ),
          'record_canonical_binding': {
              'sha256': '9b1a5e3bbc9845d04430c259ed39db0f39f31b56128f022443382b91e6027285',
              'size_bytes': 22193,
          },
          'record_semantics': record_semantics,
      },
      'external_cache_archive': {
          'root': str(V3_3_4_4_EXTERNAL_CACHE_DIR.resolve()),
          'directory_lstat_rows': cache_rows,
          'cache_tree_binding': cache_binding,
          'cache_tree_content_binding': {
              'sha256': '88c2a4cde3a9881f76dc719b48dbd7f051b5841843dc5439a8e7d2349aabbc46',
              'size_bytes': 1033,
          },
          'atomic_publication_probe': probe,
          'atomic_publication_probe_content_binding': {
              'sha256': 'a25798b7c788ce614d10a7cc0d07f1795ebaf6fd9e928dcef78e096323d9bf70',
              'size_bytes': 738,
          },
      },
      'other_v3_3_4_4_paths_absent': _v3_3_4_4_other_paths_absent(),
      'no_model_cache_or_start': True,
      'no_model_or_biological_access': True,
      'no_array_jit_or_model_kernel': True,
      'no_scientific_or_confirmation_access': True,
      'immutable_and_not_cache_input': True,
      'claim_boundary': (
          'A JAX-only external GPU/device preflight passed; no model, model '
          'cache, START, apply, raw scientific record, analysis, or '
          'confirmation access occurred.'
      ),
      'access_boundary': {
          'external_preflight_device_observation_only': True,
          'external_gpu_device_observation_occurred': True,
          'no_jit_or_array_kernel': True,
          'no_model_or_biological_access': True,
          'model_cache_allocated': False,
          'same_process_preflight_reached': False,
          'model_constructed': False, 'model_apply_count': 0,
          'scientific_raw_record_count': 0, 'confirmation_model_calls': 0,
      },
  }
  if set(prefix) != {
      'status', 'predecessor_commit', 'predecessor_freeze',
      'failure_stage', 'failure_type', 'failure_message',
      'traceback_provenance', 'root_cause', 'external_preflight_archive',
      'external_cache_archive', 'other_v3_3_4_4_paths_absent',
      'no_model_cache_or_start', 'no_model_or_biological_access',
      'no_array_jit_or_model_kernel',
      'no_scientific_or_confirmation_access',
      'immutable_and_not_cache_input', 'claim_boundary', 'access_boundary',
  } or canonical_content_binding(prefix) != V3_3_4_4_CONSUMED_PREFIX_BINDING:
    raise RuntimeError('v3.3.4.4 consumed-prefix object/binding changed.')
  return prefix


def validate_recorded_prior_v3_3_4_4_consumed_preflight_prefix(
    recorded: Any, binding: Any,
) -> dict[str, Any]:
  """Reauthenticates an embedded v3.3.4.4 prefix and canonical binding."""
  live = validate_prior_v3_3_4_4_consumed_preflight_prefix()
  if not isinstance(recorded, Mapping) or dict(recorded) != live:
    raise ValueError('Recorded v3.3.4.4 consumed-prefix object changed.')
  if (
      not isinstance(binding, Mapping)
      or dict(binding) != V3_3_4_4_CONSUMED_PREFIX_BINDING
  ):
    raise ValueError('Recorded v3.3.4.4 consumed-prefix binding changed.')
  return live


def validate_recorded_prior_v3_3_4_5_controlled_stop_archive(
    binding: Any,
) -> dict[str, Any]:
  """Reauthenticates the frozen predecessor archive and copied binding."""
  live = validate_prior_v3_3_4_5_controlled_stop_archive()
  expected = canonical_content_binding(live)
  if not isinstance(binding, Mapping) or dict(binding) != expected:
    raise ValueError('Recorded v3.3.4.5 controlled-stop binding changed.')
  return live


def _strict_file_tree(root: Path) -> list[Path]:
  """Returns every regular file and rejects symlinks/special/empty entries."""
  if root.is_symlink():
    raise ValueError(f'Strict tree root is a symlink: {root}.')
  root = root.resolve()
  if not root.is_dir():
    raise ValueError(f'Strict tree root is absent, non-directory, or symlink: {root}.')
  files = []
  pending = [root]
  while pending:
    directory = pending.pop()
    entries = sorted(directory.iterdir())
    if not entries:
      raise ValueError(f'Strict tree contains an empty directory: {directory}.')
    for entry in entries:
      mode = entry.lstat().st_mode
      if stat.S_ISLNK(mode):
        raise ValueError(f'Strict tree contains a symlink: {entry}.')
      if stat.S_ISREG(mode):
        files.append(entry)
      elif stat.S_ISDIR(mode):
        pending.append(entry)
      else:
        raise ValueError(f'Strict tree contains a special entry: {entry}.')
  return sorted(files)


def _validate_file(path: Path, expected_sha256: str, label: str) -> None:
  _reject_confirmation_path(path)
  if not path.is_file() or path.is_symlink():
    raise ValueError(f'{label} is absent or not a regular file.')
  if _sha256(path) != expected_sha256:
    raise ValueError(f'{label} bytes changed.')


def validate_one_shot_output_absence(
    output_dir: Path = OUTPUT_DIR,
    analysis_dir: Path = ANALYSIS_DIR,
    analysis_attempt_dir: Path = ANALYSIS_ATTEMPT_DIR,
) -> dict[str, Any]:
  """Fail-closes before model import if scientific sidecar output exists."""
  for path, label in (
      (output_dir, 'v3.3.4.6 sidecar output'),
      (analysis_dir, 'v3.3.4.6 sidecar analysis'),
      (analysis_attempt_dir, 'v3.3.4.6 sidecar analysis attempt'),
  ):
    _reject_confirmation_path(path)
    if path.exists() or path.is_symlink():
      raise FileExistsError(f'{label} already exists; never resume or retry.')
  return {
      'output_dir': str(output_dir.resolve()),
      'output_dir_absent': True,
      'analysis_dir': str(analysis_dir.resolve()),
      'analysis_dir_absent': True,
      'analysis_attempt_dir': str(analysis_attempt_dir.resolve()),
      'analysis_attempt_dir_absent': True,
      'preflight_dir_may_exist': True,
  }


def validate_predecessor_path_absence() -> dict[str, Any]:
  """Requires all absent predecessor roots to remain absent."""
  result = {}
  for version, paths in (
      ('v3_3_4', V3_3_4_PREDECESSOR_PATHS),
      ('v3_3_4_1', V3_3_4_1_PREDECESSOR_PATHS),
      ('v3_3_4_2', V3_3_4_2_PREDECESSOR_PATHS),
  ):
    version_result = {}
    for role, path in sorted(paths.items()):
      _reject_confirmation_path(path)
      if path.exists() or path.is_symlink():
        raise FileExistsError(
            f'Never-launched {version} predecessor path exists: {path}.'
        )
      version_result[role] = {
          'path': str(path.resolve()), 'absent': True,
      }
    result[version] = version_result
  v3_3_4_3 = {}
  for role, path in sorted(V3_3_4_3_PREDECESSOR_PATHS.items()):
    if role == 'external_cache':
      continue
    _reject_confirmation_path(path)
    if path.exists() or path.is_symlink():
      raise FileExistsError(
          f'Consumed v3_3_4_3 sibling path exists: {path}.'
      )
    v3_3_4_3[role] = {'path': str(path.resolve()), 'absent': True}
  result['v3_3_4_3'] = v3_3_4_3
  result['v3_3_4_4'] = _v3_3_4_4_other_paths_absent()
  return result


def validate_gate_a_path_absence() -> dict[str, Any]:
  """Binds all six predecessor and all six fresh v3.3.4.6 paths."""
  current = {}
  for role, path in sorted(PUBLICATION_ROOTS.items()):
    _reject_confirmation_path(path)
    if path.exists() or path.is_symlink():
      raise FileExistsError(
          f'Fresh v3.3.4.6 publication root already exists: {path}.'
      )
    current[role] = {'path': str(path.resolve()), 'absent': True}
  return {
      'predecessors': validate_predecessor_path_absence(),
      'v3_3_4_6_current': current,
  }


def validate_publication_probe_live_bindings(
    probe: Mapping[str, Any],
) -> None:
  """Reauthenticates both no-follow probe entries including inode/link state."""
  root = require_publication_directory('external_cache', '.')
  expected_root = _PUBLICATION_DIRECTORIES.get(('external_cache', '.'))
  if expected_root is None:
    raise RuntimeError('External-cache publication root was not registered.')
  root_fd = os.open(
      root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
  )
  try:
    opened_root = os.fstat(root_fd)
    if (
        not stat.S_ISDIR(opened_root.st_mode)
        or _publication_mode(opened_root.st_mode) != '0700'
        or (opened_root.st_dev, opened_root.st_ino) != expected_root
    ):
      raise RuntimeError(
          'Opened external-cache root does not match its registered inode.'
      )
    for binding_name in (
        'successful_final_binding', 'collision_temp_binding'
    ):
      binding = probe.get(binding_name)
      if not isinstance(binding, Mapping):
        raise ValueError('Publication probe binding is missing.')
      relative = Path(str(binding.get('path', '')))
      if (
          relative.is_absolute() or not relative.parts
          or '..' in relative.parts or relative.parent != Path('.')
      ):
        raise ValueError('Publication probe binding path is unsafe.')
      state = _entry_state(root_fd, relative.name)
      live = {
          'path': relative.as_posix(), 'sha256': state.get('sha256'),
          'size_bytes': state.get('size_bytes'), 'mode': state.get('mode'),
          'st_dev': state.get('st_dev'), 'st_ino': state.get('st_ino'),
          'st_nlink': state.get('st_nlink'),
      }
      if state.get('state') != 'present' or state.get(
          'entry_type'
      ) != 'regular' or live != dict(binding):
        raise ValueError('Live publication probe binding changed.')
  finally:
    os.close(root_fd)


def _validate_completed_preflight_state(
    *, require_parent_external_routing: bool,
) -> dict[str, Any]:
  """Validates the completed five-file preflight without phase dispatch."""
  role = os.environ.get(CACHE_ROLE_ENVIRONMENT)
  root = os.environ.get(CACHE_ROOT_ENVIRONMENT)
  if require_parent_external_routing:
    if (
        role != 'external_preflight'
        or root != str(PREFLIGHT_KERNEL_CACHE_DIR.resolve())
        or os.environ.get('TRITON_CACHE_DIR')
        != str((PREFLIGHT_KERNEL_CACHE_DIR / 'triton').resolve())
        or os.environ.get('XDG_CACHE_HOME')
        != str((PREFLIGHT_KERNEL_CACHE_DIR / 'xdg').resolve())
    ):
      raise ValueError('Completed-preflight parent routing is not exact.')
  elif (
      role != 'model'
      or root != str(MODEL_KERNEL_CACHE_DIR.resolve())
  ):
    raise ValueError('Model process cache routing is not exact.')
  preflight_root_stat = PREFLIGHT_DIR.lstat()
  if (
      stat.S_ISLNK(preflight_root_stat.st_mode)
      or not stat.S_ISDIR(preflight_root_stat.st_mode)
      or stat.S_IMODE(preflight_root_stat.st_mode) != 0o700
  ):
    raise ValueError('Completed preflight root is not an exact 0700 directory.')
  files = _strict_file_tree(PREFLIGHT_DIR)
  expected_names = {
      '.allocation.lock',
      '.preflight_0000.reserved',
      'preflight_0000.json',
      'preflight_0000.stdout.log',
      'preflight_0000.stderr.log',
  }
  if (
      {path.name for path in files} != expected_names or len(files) != 5
      or any(path.parent != PREFLIGHT_DIR.resolve() for path in files)
  ):
    raise ValueError('Model process requires one exact preflight attempt tree.')
  record_path = PREFLIGHT_DIR / 'preflight_0000.json'
  record = json.loads(record_path.read_text(encoding='utf-8'))
  if set(record) != set(PREFLIGHT_CONTRACT['record_keys']):
    raise ValueError('Sole preflight record key set changed.')
  expected = {
      'script_version': 'opensplice-device-preflight-v3.3.4.6',
      'status': 'pass',
      'preflight_attempt_number': 0,
      'amendment_sha256': AMENDMENT_SHA256,
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'freeze_sha256': _sha256(FREEZE_PATH),
      'failure': None,
      'no_model_or_biological_access': True,
      'no_jit_or_array_kernel': True,
  }
  for name, value in expected.items():
    if record.get(name) != value:
      raise ValueError(f'Sole preflight record changed: {name}.')
  validate_recorded_prior_v3_3_4_3_consumed_preflight_prefix(
      record.get('prior_v3_3_4_3_consumed_preflight_prefix'),
      record.get(
          'prior_v3_3_4_3_consumed_preflight_prefix_content_binding'
      ),
  )
  validate_recorded_prior_v3_3_4_4_consumed_preflight_prefix(
      record.get('prior_v3_3_4_4_consumed_preflight_prefix'),
      record.get(
          'prior_v3_3_4_4_consumed_preflight_prefix_content_binding'
      ),
  )
  validate_recorded_prior_v3_3_4_5_controlled_stop_archive(
      record.get('prior_v3_3_4_5_controlled_stop_archive_content_binding')
  )
  freeze_record = record.get('freeze')
  if not isinstance(freeze_record, Mapping) or set(freeze_record) != {
      'path', 'sha256', 'size_bytes', 'external_freeze_authorization',
      'preflight_version_proof',
      'prior_v3_3_4_3_consumed_preflight_prefix',
      'prior_v3_3_4_3_consumed_preflight_prefix_content_binding',
      'prior_v3_3_4_4_consumed_preflight_prefix',
      'prior_v3_3_4_4_consumed_preflight_prefix_content_binding',
      'prior_v3_3_4_5_controlled_stop_archive_content_binding',
  }:
    raise ValueError('Sole preflight freeze record is missing.')
  validate_recorded_prior_v3_3_4_3_consumed_preflight_prefix(
      freeze_record.get('prior_v3_3_4_3_consumed_preflight_prefix'),
      freeze_record.get(
          'prior_v3_3_4_3_consumed_preflight_prefix_content_binding'
      ),
  )
  validate_recorded_prior_v3_3_4_4_consumed_preflight_prefix(
      freeze_record.get('prior_v3_3_4_4_consumed_preflight_prefix'),
      freeze_record.get(
          'prior_v3_3_4_4_consumed_preflight_prefix_content_binding'
      ),
  )
  validate_recorded_prior_v3_3_4_5_controlled_stop_archive(
      freeze_record.get(
          'prior_v3_3_4_5_controlled_stop_archive_content_binding'
      )
  )
  live_authorization = validate_external_freeze_authorization()
  if record.get('external_freeze_authorization') != live_authorization:
    raise ValueError('Sole preflight external authorization changed.')
  if (
      freeze_record.get('path') != live_authorization['freeze_path']
      or freeze_record.get('sha256') != live_authorization['freeze_sha256']
      or freeze_record.get('size_bytes')
      != live_authorization['freeze_size_bytes']
      or freeze_record.get('external_freeze_authorization')
      != live_authorization
  ):
    raise ValueError('Sole preflight nested freeze binding changed.')
  live_frozen = json.loads(FREEZE_PATH.read_text(encoding='utf-8'))
  if freeze_record.get('preflight_version_proof') != (
      validate_preflight_version_contract(live_frozen)
  ):
    raise ValueError('Sole preflight nested version proof changed.')
  probe = record.get('atomic_publication_probe')
  if not isinstance(probe, dict) or set(probe) != {
      'schema_version', 'method', 'supported',
      'successful_final_binding', 'collision_errno',
      'collision_no_replace_exact', 'collision_temp_binding',
      'destination_unchanged', 'temp_orphan_preserved',
      'parent_fsync_exact',
  }:
    raise ValueError('Sole preflight publication-probe schema changed.')
  probe_contract = PUBLICATION_CONTRACT_V3_3_4_1[
      'external_preflight_probe_contract'
  ]
  success_binding = probe['successful_final_binding']
  collision_binding = probe['collision_temp_binding']
  binding_keys = {
      'path', 'sha256', 'size_bytes', 'mode', 'st_dev', 'st_ino', 'st_nlink'
  }
  if (
      probe['schema_version'] != PUBLICATION_SCHEMA_VERSION
      or probe['method'] != PUBLICATION_METHOD
      or probe['supported'] is not True
      or probe['collision_errno'] != probe_contract['collision_errno']
      or any(probe[name] is not True for name in (
          'collision_no_replace_exact', 'destination_unchanged',
          'temp_orphan_preserved', 'parent_fsync_exact'
      ))
      or set(success_binding) != binding_keys
      or set(collision_binding) != binding_keys
      or success_binding.get('path') != probe_contract['final_basename']
      or success_binding.get('sha256') != probe_contract['final_sha256']
      or success_binding.get('size_bytes') != probe_contract['final_size_bytes']
      or collision_binding.get('sha256') != probe_contract['collision_sha256']
      or collision_binding.get('size_bytes')
      != probe_contract['collision_size_bytes']
      or not re.fullmatch(
          r'\.v3345\.tmp\.[1-9][0-9]*\.[0-9]{6}\.[0-9a-f]{32}',
          str(collision_binding.get('path')),
      )
      or any(binding.get('mode') != '0400' for binding in (
          success_binding, collision_binding
      ))
      or any(binding.get('st_nlink') != 1 for binding in (
          success_binding, collision_binding
      ))
      or any(
          not isinstance(binding.get(name), int)
          or isinstance(binding.get(name), bool)
          or binding[name] < 0
          for binding in (success_binding, collision_binding)
          for name in ('size_bytes', 'st_dev', 'st_ino', 'st_nlink')
      )
  ):
    raise ValueError('Sole preflight publication probe changed.')
  validate_publication_probe_live_bindings(probe)
  for stream in ('stdout', 'stderr'):
    binding = record.get('logs', {}).get(stream, {})
    expected_path = PREFLIGHT_DIR / f'preflight_0000.{stream}.log'
    if (
        binding.get('path') != str(expected_path.resolve())
        or binding.get('sha256') != _sha256(expected_path)
        or binding.get('size_bytes') != expected_path.stat().st_size
    ):
      raise ValueError(f'Sole preflight {stream} binding changed.')
  cache = record.get('observation', {}).get(
      'v3_3_4_6_runtime_environment', {}
  ).get('cache_environment', {})
  if (
      cache.get('cache_role') != 'external_preflight'
      or cache.get('cache_root')
      != str(PREFLIGHT_KERNEL_CACHE_DIR.resolve())
      or cache.get('pre_import_file_count') != 0
  ):
    raise ValueError('Sole preflight cache attestation changed.')
  post_cache = record.get('external_cache_post_observation', {})
  live_post_cache = cache_output_tree_binding(PREFLIGHT_KERNEL_CACHE_DIR)
  if (
      post_cache.get('cache_root')
      != str(PREFLIGHT_KERNEL_CACHE_DIR.resolve())
      or post_cache.get('diagnostic_outputs_only_no_cache_input') is not True
      or post_cache != live_post_cache
  ):
    raise ValueError('Sole preflight cache-output binding changed.')
  for relative in live_post_cache.get('directory_paths', []):
    directory = (
        PREFLIGHT_KERNEL_CACHE_DIR
        if relative == '.'
        else PREFLIGHT_KERNEL_CACHE_DIR / relative
    )
    observed = directory.lstat()
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o700
    ):
      raise ValueError('Sole preflight cache-directory mode changed.')
  if record.get('external_cache_hit_evidence') != {
      'pre_import_files_present': False,
      'default_user_cache_path_eligible': False,
      'persistent_compilation_cache_hit_reported': False,
      'executable_deserialized': False,
      'compile_skipped': None,
      'compile_stage_not_applicable': True,
      'old_cache_input_opened': False,
      'routing_exact': True,
      'cache_hit': False,
  }:
    raise ValueError('External preflight cache-hit evidence changed.')
  external_pid = record.get('observation', {}).get('pid')
  if (
      not isinstance(external_pid, int) or isinstance(external_pid, bool)
      or external_pid <= 0 or external_pid == os.getpid()
  ):
    raise ValueError('External preflight PID is not distinct from the parent.')
  for path in files:
    expected_mode = 0o600 if path.name == '.allocation.lock' else 0o400
    if (path.stat().st_mode & 0o777) != expected_mode:
      raise ValueError(f'Preflight file mode changed: {path.name}.')
  return {
      'cache_role': role,
      'preflight_dir': str(PREFLIGHT_DIR.resolve()),
      'preflight_dir_absent': False,
      'file_count': 5,
      'file_sha256': {
          path.name: {'sha256': _sha256(path), 'size_bytes': path.stat().st_size}
          for path in sorted(files)
      },
      'tree_sha256': _tree_digest(files, PREFLIGHT_DIR),
      'record_sha256': _sha256(record_path),
      'sole_successful_preflight_exact': True,
  }


def validate_completed_external_preflight_state() -> dict[str, Any]:
  """Parent-only completed phase; ambient role is evidence, never dispatch."""
  return _validate_completed_preflight_state(
      require_parent_external_routing=True
  )


def validate_preflight_state_for_role() -> dict[str, Any]:
  """Validates entry absence or the later model-role completed state."""
  role = os.environ.get(CACHE_ROLE_ENVIRONMENT)
  if role in ('external_preflight', 'dry_run'):
    if PREFLIGHT_DIR.exists() or PREFLIGHT_DIR.is_symlink():
      raise FileExistsError(
          f'Preflight directory exists before the {role} process.'
      )
    return {
        'cache_role': role,
        'preflight_dir': str(PREFLIGHT_DIR.resolve()),
        'preflight_dir_absent': True,
    }
  if role != 'model':
    raise ValueError('Preflight-state validation requires a known cache role.')
  return _validate_completed_preflight_state(
      require_parent_external_routing=False
  )


def validate_original_run() -> dict[str, Any]:
  """Rehashes the consumed v3.3 attempt without reading scientific values."""
  binding = dict(EXPECTED_ORIGINAL_BINDING)
  _validate_file(ORIGINAL_FREEZE_PATH, binding['freeze_sha256'], 'v3.3 freeze')
  fixed_files = {
      _HERE / 'run_encoder_skip_factorial_v3_3.py': binding['runner_sha256'],
      _HERE / 'analyze_encoder_skip_localization_v3_3.py': (
          binding['analyzer_sha256']
      ),
      _HERE / 'analyze_encoder_skip_localization_v3_3_test.py': (
          binding['analyzer_test_sha256']
      ),
      _REPO / 'src/alphagenome_research/model/model.py': (
          binding['model_sha256']
      ),
      _REPO / 'src/alphagenome_research/model/interpretability.py': (
          binding['interpretability_sha256']
      ),
      ORIGINAL_RUN_DIR / 'ATTEMPT_STARTED.json': (
          binding['attempt_started_sha256']
      ),
      ORIGINAL_RUN_DIR / 'RUN_COMPLETE.json': binding['run_complete_sha256'],
      ORIGINAL_RUN_DIR / 'RAW_MANIFEST.json': binding['raw_manifest_sha256'],
      ORIGINAL_RUN_DIR / 'PROTOBUF_PROVENANCE.json': (
          binding['protobuf_provenance_sha256']
      ),
      ORIGINAL_RUN_DIR / 'TARGET_ELIGIBILITY.json': (
          binding['target_eligibility_sha256']
      ),
  }
  for path, expected_sha in fixed_files.items():
    _validate_file(path, expected_sha, str(path.relative_to(_REPO)))
  for name in (
      'IMPORT_PROVENANCE_PRE_MODEL.json',
      'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
      'IMPORT_PROVENANCE.json',
  ):
    _validate_file(
        ORIGINAL_RUN_DIR / name,
        binding['import_provenance_sha256'],
        f'original {name}',
    )

  start = json.loads(
      (ORIGINAL_RUN_DIR / 'ATTEMPT_STARTED.json').read_text(encoding='utf-8')
  )
  external = start.get('external_preflight', {})
  preflight_path = Path(str(external.get('path', ''))).resolve()
  if (
      external.get('sha256') != binding['device_preflight_sha256']
      or external.get('status') != 'pass'
      or external.get('no_model_or_biological_access') is not True
      or external.get('no_jit_or_array_kernel') is not True
      or external.get('failure') is not None
  ):
    raise ValueError('Original v3.3 external-preflight binding changed.')
  _validate_file(
      preflight_path,
      binding['device_preflight_sha256'],
      'original v3.3 device preflight',
  )
  for stream, expected_sha in (
      ('stdout', binding['preflight_stdout_sha256']),
      ('stderr', binding['preflight_stderr_sha256']),
  ):
    item = external.get('logs', {}).get(stream, {})
    log_path = Path(str(item.get('path', ''))).resolve()
    if item.get('sha256') != expected_sha:
      raise ValueError(f'Original preflight {stream} binding changed.')
    _validate_file(log_path, expected_sha, f'original preflight {stream}')

  manifest_path = ORIGINAL_RUN_DIR / 'RAW_MANIFEST.json'
  manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
  artifact_sha = manifest.get('artifact_sha256')
  if (
      manifest.get('artifact_count') != binding['raw_artifact_count']
      or manifest.get('artifact_tree_sha256')
      != binding['raw_artifact_tree_sha256']
      or not isinstance(artifact_sha, dict)
      or len(artifact_sha) != binding['raw_artifact_count']
  ):
    raise ValueError('Original v3.3 raw-manifest structure changed.')
  raw_paths = []
  for relative, expected_sha in sorted(artifact_sha.items()):
    path = (ORIGINAL_RUN_DIR / relative).resolve()
    if not path.is_relative_to(ORIGINAL_RUN_DIR.resolve()):
      raise ValueError('Original raw path escaped the run directory.')
    _validate_file(path, expected_sha, f'original raw {relative}')
    raw_paths.append(path)
  if _tree_digest(raw_paths, ORIGINAL_RUN_DIR) != binding[
      'raw_artifact_tree_sha256'
  ]:
    raise ValueError('Original v3.3 raw tree digest changed.')
  for relative, expected_sha in EXPECTED_ORIGINAL_OOD.items():
    if artifact_sha.get(relative) != expected_sha:
      raise ValueError(f'Original OOD failure-boundary binding changed: {relative}.')

  all_paths = _strict_file_tree(ORIGINAL_RUN_DIR)
  if (
      len(all_paths) != binding['whole_run_file_count']
      or _tree_digest(all_paths, ORIGINAL_RUN_DIR)
      != binding['whole_run_tree_sha256']
  ):
    raise ValueError('Original v3.3 whole-run file set or tree changed.')
  compiler_paths = _strict_file_tree(ORIGINAL_RUN_DIR / 'compiler')
  if (
      len(compiler_paths) != binding['compiler_file_count']
      or _tree_digest(compiler_paths, ORIGINAL_RUN_DIR)
      != binding['compiler_tree_sha256']
  ):
    raise ValueError('Original v3.3 compiler tree changed.')

  completion = json.loads(
      (ORIGINAL_RUN_DIR / 'RUN_COMPLETE.json').read_text(encoding='utf-8')
  )
  if {
      name: completion.get(name) for name in EXPECTED_ORIGINAL_STATUS
  } != EXPECTED_ORIGINAL_STATUS:
    raise ValueError('Original v3.3 structural completion status changed.')
  if (
      completion.get('raw_manifest') != manifest
      or completion.get('eight_row_executable_fingerprint')
      != '12283496a0987eec942bd8f9b7bbb86a9d9d676b13bee1956b30da933a4e9967'
  ):
    raise ValueError('Original v3.3 completion linkage changed.')
  compiler = json.loads(
      (ORIGINAL_RUN_DIR / 'compiler/eight_row/COMPILER_PROVENANCE.json')
      .read_text(encoding='utf-8')
  )
  return {
      **binding,
      'path': str(ORIGINAL_RUN_DIR.resolve()),
      'raw_manifest': manifest,
      'status_predicates': EXPECTED_ORIGINAL_STATUS,
      'original_ood_boundary': dict(EXPECTED_ORIGINAL_OOD),
      'eight_row_compiler': compiler,
  }


def validate_v3_3_1_status(frozen: Mapping[str, Any]) -> dict[str, Any]:
  """Validates the prospectively frozen structural state of v3.3.1."""
  expected = frozen.get('v3_3_1_status')
  if not isinstance(expected, dict):
    raise ValueError('v3.3.1 structural status is absent from the freeze.')
  _validate_file(
      V3_3_1_AMENDMENT_PATH,
      V3_3_1_AMENDMENT_SHA256,
      'v3.3.1 analyzer amendment',
  )
  base = {
      'amendment_path': str(V3_3_1_AMENDMENT_PATH.resolve()),
      'amendment_sha256': V3_3_1_AMENDMENT_SHA256,
      'amendment_commit': V3_3_1_AMENDMENT_COMMIT,
      'attempt_dir': str(V3_3_1_ATTEMPT_DIR.resolve()),
      'analysis_dir': str(V3_3_1_ANALYSIS_DIR.resolve()),
  }
  if expected.get('state') == 'unconsumed':
    observed = {
        **base,
        'state': 'unconsumed',
        'attempt_dir_absent': not V3_3_1_ATTEMPT_DIR.exists(),
        'analysis_dir_absent': not V3_3_1_ANALYSIS_DIR.exists(),
    }
    if not observed['attempt_dir_absent'] or not observed['analysis_dir_absent']:
      raise ValueError('Frozen unconsumed v3.3.1 state changed.')
  elif expected.get('state') in ('terminal', 'completed'):
    if not V3_3_1_ATTEMPT_DIR.is_dir():
      raise ValueError('Frozen terminal v3.3.1 attempt is absent.')
    paths = _strict_file_tree(V3_3_1_ATTEMPT_DIR)
    observed = {
        **base,
        'state': expected['state'],
        'attempt_file_count': len(paths),
        'attempt_tree_sha256': _tree_digest(paths, V3_3_1_ATTEMPT_DIR),
    }
    if expected['state'] == 'completed':
      if expected != EXPECTED_V3_3_1_COMPLETED_STATUS:
        raise ValueError(
            'Frozen completed v3.3.1 status differs from exact audit bytes.'
        )
      if not V3_3_1_ANALYSIS_DIR.is_dir():
        raise ValueError('Frozen completed v3.3.1 analysis is absent.')
      analysis_paths = _strict_file_tree(V3_3_1_ANALYSIS_DIR)
      observed.update({
          'analysis_file_count': len(analysis_paths),
          'analysis_tree_sha256': _tree_digest(
              analysis_paths, V3_3_1_ANALYSIS_DIR
          ),
          'attempt_files': {},
          'analysis_files': {},
      })
      for name, binding in expected['attempt_files'].items():
        path = V3_3_1_ATTEMPT_DIR / name
        _validate_file(path, binding['sha256'], f'v3.3.1 attempt {name}')
        if path.stat().st_size != binding['size_bytes']:
          raise ValueError(f'v3.3.1 attempt size changed: {name}.')
        observed['attempt_files'][name] = dict(binding)
      for name, binding in expected['analysis_files'].items():
        path = V3_3_1_ANALYSIS_DIR / name
        _validate_file(path, binding['sha256'], f'v3.3.1 analysis {name}')
        if path.stat().st_size != binding['size_bytes']:
          raise ValueError(f'v3.3.1 analysis size changed: {name}.')
        observed['analysis_files'][name] = dict(binding)
      completion = json.loads(
          (V3_3_1_ATTEMPT_DIR / 'ANALYSIS_COMPLETE.json').read_text(
              encoding='utf-8'
          )
      )
      predicates = {
          name: completion.get(name)
          for name in expected['structural_predicates']
      }
      if predicates != expected['structural_predicates']:
        raise ValueError('v3.3.1 controlled-stop structural predicates changed.')
      observed['structural_predicates'] = predicates
    else:
      observed['analysis_dir_absent'] = not V3_3_1_ANALYSIS_DIR.exists()
  else:
    raise ValueError('Unknown frozen v3.3.1 structural state.')
  if observed != expected:
    raise ValueError('v3.3.1 structural status differs from the freeze.')
  return observed


def _git_blob_sha256(commit: str, relative: str) -> str:
  payload = subprocess.check_output(
      ('git', '-C', str(_REPO), 'show', f'{commit}:{relative}')
  )
  return hashlib.sha256(payload).hexdigest()


def _validate_source_bindings(
    bindings: Mapping[str, str], commit_by_path: Mapping[str, str]
) -> None:
  for relative, expected_sha in sorted(bindings.items()):
    path = (_REPO / relative).resolve()
    _validate_file(path, expected_sha, f'frozen source {relative}')
    commit = commit_by_path[relative]
    if _git_blob_sha256(commit, relative) != expected_sha:
      raise ValueError(f'Committed source blob changed: {relative}.')


def _validate_exact_tree(
    root: Path,
    expected_files: Mapping[str, Mapping[str, Any]],
    expected_tree_sha256: str,
    *,
    archive_commit: str | None = None,
) -> None:
  paths = _strict_file_tree(root)
  if {str(path.relative_to(root)) for path in paths} != set(expected_files):
    raise ValueError(f'Exact tree membership changed: {root}.')
  for relative, binding in sorted(expected_files.items()):
    path = root / relative
    _validate_file(path, binding['sha256'], f'exact tree {relative}')
    if path.stat().st_size != binding['size_bytes']:
      raise ValueError(f'Exact tree size changed: {path}.')
    if archive_commit is not None:
      repo_relative = str(path.relative_to(_REPO))
      if _git_blob_sha256(archive_commit, repo_relative) != binding['sha256']:
        raise ValueError(f'Archived result blob changed: {repo_relative}.')
  if _tree_digest(paths, root) != expected_tree_sha256:
    raise ValueError(f'Exact tree digest changed: {root}.')


def validate_v3_3_2_1_failure() -> dict[str, Any]:
  """Revalidates the consumed recursion failure without normalizing it."""
  _validate_source_bindings(
      V3_3_2_1_SOURCE_BINDINGS,
      {name: V3_3_2_1_COMMIT for name in V3_3_2_1_SOURCE_BINDINGS},
  )
  expected = EXPECTED_V3_3_2_1_FAILURE_STATUS
  _validate_exact_tree(
      V3_3_2_1_ATTEMPT_DIR,
      expected['attempt_files'],
      expected['attempt_tree_sha256'],
  )
  if V3_3_2_1_ANALYSIS_DIR.exists() or V3_3_2_1_ANALYSIS_DIR.is_symlink():
    raise ValueError('Consumed v3.3.2.1 failure gained an analysis output.')
  failure = json.loads(
      (V3_3_2_1_ATTEMPT_DIR / 'ANALYSIS_FAILURE.json').read_text(
          encoding='utf-8'
      )
  )
  predicates = {
      'state': failure.get('status'),
      'error_type': failure.get('failure', {}).get('type'),
      'model_apply_count': failure.get('model_apply_count'),
      'scientific_summary_computed': failure.get(
          'scientific_summary_computed'
      ),
      'shapley_or_nomination_computed': failure.get(
          'shapley_or_nomination_computed'
      ),
      'combined_analysis_permitted': failure.get(
          'combined_analysis_permitted'
      ),
      'analysis_dir_absent': failure.get('analysis_dir_exists') is False,
  }
  for name, value in predicates.items():
    if expected[name] != value:
      raise ValueError(f'v3.3.2.1 failure predicate changed: {name}.')
  if failure.get('attempt_started_sha256') != expected['attempt_files'][
      'ANALYSIS_ATTEMPT_STARTED.json'
  ]['sha256']:
    raise ValueError('v3.3.2.1 START-to-failure linkage changed.')
  return dict(expected)


def validate_v3_3_2_2_archive() -> dict[str, Any]:
  """Revalidates the sole successful CPU structural archive."""
  commits = {
      name: (
          V3_3_2_2_AMENDMENT_COMMIT
          if name.endswith(
              'encoder_skip_ood_sidecar_analysis_amendment_v3_3_2_2.md'
          )
          else V3_3_2_2_IMPLEMENTATION_COMMIT
      )
      for name in V3_3_2_2_SOURCE_BINDINGS
  }
  _validate_source_bindings(V3_3_2_2_SOURCE_BINDINGS, commits)
  expected = EXPECTED_V3_3_2_2_ARCHIVE_STATUS
  _validate_exact_tree(
      V3_3_2_2_ATTEMPT_DIR,
      expected['attempt_files'],
      expected['attempt_tree_sha256'],
      archive_commit=V3_3_2_2_ARCHIVE_COMMIT,
  )
  _validate_exact_tree(
      V3_3_2_2_ANALYSIS_DIR,
      expected['analysis_files'],
      expected['analysis_tree_sha256'],
      archive_commit=V3_3_2_2_ARCHIVE_COMMIT,
  )
  complete = json.loads(
      (V3_3_2_2_ATTEMPT_DIR / 'ANALYSIS_COMPLETE.json').read_text(
          encoding='utf-8'
      )
  )
  predicates = {
      'state': complete.get('status'),
      'decision': complete.get('decision'),
      'model_apply_count': complete.get('model_apply_count'),
      'scientific_summary_computed': complete.get(
          'scientific_summary_computed'
      ),
      'shapley_or_nomination_computed': complete.get(
          'shapley_or_nomination_computed'
      ),
      'combined_analysis_permitted': complete.get(
          'combined_analysis_permitted'
      ),
  }
  for name, value in predicates.items():
    if expected[name] != value:
      raise ValueError(f'v3.3.2.2 archive predicate changed: {name}.')
  links = {
      'analysis_file_count': expected['analysis_file_count'],
      'analysis_json_sha256': expected['analysis_files']['ANALYSIS.json'][
          'sha256'
      ],
      'analysis_markdown_sha256': expected['analysis_files']['RESULT.md'][
          'sha256'
      ],
      'analysis_tree_sha256': expected['analysis_tree_sha256'],
      'attempt_started_sha256': expected['attempt_files'][
          'ANALYSIS_ATTEMPT_STARTED.json'
      ]['sha256'],
  }
  for name, value in links.items():
    if complete.get(name) != value:
      raise ValueError(f'v3.3.2.2 terminal linkage changed: {name}.')
  analysis = json.loads(
      (V3_3_2_2_ANALYSIS_DIR / 'ANALYSIS.json').read_text(encoding='utf-8')
  )
  for name, value in predicates.items():
    if name == 'state':
      actual = analysis.get('status')
    elif name == 'model_apply_count':
      actual = analysis.get('sidecar_audit', {}).get(
          'audited_model_apply_count'
      )
    else:
      actual = analysis.get(name)
    if actual != value:
      raise ValueError(f'v3.3.2.2 analysis predicate changed: {name}.')
  if analysis.get('nomination') is not None or analysis.get(
      'resolution_analysis'
  ) is not None:
    raise ValueError('v3.3.2.2 archive unexpectedly contains science output.')
  return dict(expected)


def validate_v3_3_2_run() -> dict[str, Any]:
  """Revalidates the immutable apply-zero v3.3.2 model-run stop."""
  _validate_file(
      V3_3_2_FREEZE_PATH, V3_3_2_FREEZE_SHA256, 'v3.3.2 freeze'
  )
  frozen = json.loads(V3_3_2_FREEZE_PATH.read_text(encoding='utf-8'))
  file_sha = frozen.get('file_sha256')
  if not isinstance(file_sha, dict) or len(file_sha) != 75:
    raise ValueError('v3.3.2 frozen 75-file source inventory changed.')
  for relative, expected_sha in sorted(file_sha.items()):
    path = (_REPO / relative).resolve()
    _validate_file(path, expected_sha, f'v3.3.2 bundle {relative}')
    if _git_blob_sha256(V3_3_2_RUN_COMMIT, relative) != expected_sha:
      raise ValueError(f'v3.3.2 committed source changed: {relative}.')

  binding = V3_3_2_RUN_BINDING
  fixed = {
      'ATTEMPT_STARTED.json': binding['attempt_started_sha256'],
      'RUN_COMPLETE.json': binding['run_complete_sha256'],
      'RAW_MANIFEST.json': binding['raw_manifest_sha256'],
      'PROTOBUF_PROVENANCE.json': binding['protobuf_provenance_sha256'],
      'compiler/eight_row/COMPILER_PROVENANCE.json': (
          binding['compiler_provenance_sha256']
      ),
  }
  for name, expected_sha in fixed.items():
    _validate_file(V3_3_2_RUN_DIR / name, expected_sha, f'v3.3.2 {name}')
  for name in (
      'IMPORT_PROVENANCE_PRE_MODEL.json',
      'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
      'IMPORT_PROVENANCE.json',
  ):
    _validate_file(
        V3_3_2_RUN_DIR / name,
        binding['import_provenance_sha256'],
        f'v3.3.2 {name}',
    )
  paths = _strict_file_tree(V3_3_2_RUN_DIR)
  if (
      len(paths) != binding['whole_run_file_count']
      or _tree_digest(paths, V3_3_2_RUN_DIR)
      != binding['whole_run_tree_sha256']
  ):
    raise ValueError('v3.3.2 whole-run tree changed.')
  compiler_paths = _strict_file_tree(V3_3_2_RUN_DIR / 'compiler')
  if (
      len(compiler_paths) != binding['compiler_file_count']
      or _tree_digest(compiler_paths, V3_3_2_RUN_DIR)
      != binding['compiler_tree_sha256']
  ):
    raise ValueError('v3.3.2 compiler tree changed.')
  if (V3_3_2_RUN_DIR / 'raw').exists():
    raise ValueError('v3.3.2 zero-apply stop gained a raw directory.')
  manifest = json.loads(
      (V3_3_2_RUN_DIR / 'RAW_MANIFEST.json').read_text(encoding='utf-8')
  )
  expected_manifest = {
      'artifact_count': 0,
      'artifact_sha256': {},
      'artifact_tree_sha256': hashlib.sha256(b'').hexdigest(),
  }
  if manifest != expected_manifest:
    raise ValueError('v3.3.2 empty raw manifest changed.')
  complete = json.loads(
      (V3_3_2_RUN_DIR / 'RUN_COMPLETE.json').read_text(encoding='utf-8')
  )
  status = {
      'status': 'controlled_stop',
      'stop_reason': 'compiler_graph_mismatch',
      'eight_row_compile_count': 1,
      'six_row_compile_count': 0,
      'model_apply_count': 0,
      'ood_anchor_record_count': 0,
      'ood_invalid_count': 0,
      'identity_rerun_count': 0,
      'main_cube_rerun_count': 0,
      'confirmation_model_calls': 0,
      'scientific_summary_computed': False,
      'shapley_or_nomination_computed': False,
  }
  if {name: complete.get(name) for name in status} != status:
    raise ValueError('v3.3.2 terminal predicates changed.')
  if complete.get('raw_manifest') != expected_manifest:
    raise ValueError('v3.3.2 completion-to-manifest linkage changed.')
  compiler = json.loads(
      (V3_3_2_RUN_DIR / 'compiler/eight_row/COMPILER_PROVENANCE.json')
      .read_text(encoding='utf-8')
  )
  return {
      'path': str(V3_3_2_RUN_DIR.resolve()),
      'model_run_commit': V3_3_2_RUN_COMMIT,
      'freeze_sha256': V3_3_2_FREEZE_SHA256,
      **binding,
      'status_predicates': status,
      'empty_raw_manifest': expected_manifest,
      'eight_row_compiler': compiler,
  }


def _binding_rows(rows: Mapping[str, tuple[int, str]]) -> dict[str, Any]:
  return {
      name: {'size_bytes': size, 'sha256': digest}
      for name, (size, digest) in rows.items()
  }


def validate_v3_3_3_run() -> dict[str, Any]:
  """Rehash the immutable apply-zero v3.3.3 source-program stop."""
  _validate_file(V3_3_3_FREEZE_PATH, V3_3_3_FREEZE_SHA256, 'v3.3.3 freeze')
  _validate_exact_tree(
      V3_3_3_RUN_DIR,
      _binding_rows(V3_3_3_RUN_FILES),
      V3_3_3_RUN_TREE_SHA256,
  )
  compiler_rows = {
      name: value for name, value in V3_3_3_RUN_FILES.items()
      if name.startswith('compiler/')
  }
  compiler_paths = [V3_3_3_RUN_DIR / name for name in compiler_rows]
  if _tree_digest(compiler_paths, V3_3_3_RUN_DIR) != V3_3_3_COMPILER_TREE_SHA256:
    raise ValueError('v3.3.3 compiler tree changed.')
  if (V3_3_3_RUN_DIR / 'raw').exists():
    raise ValueError('v3.3.3 apply-zero stop gained raw artifacts.')
  complete = json.loads(
      (V3_3_3_RUN_DIR / 'RUN_COMPLETE.json').read_text(encoding='utf-8')
  )
  predicates = {
      'status': 'controlled_stop',
      'stop_reason': 'source_program_mismatch',
      'model_apply_count': 0,
      'ood_anchor_record_count': 0,
      'confirmation_model_calls': 0,
      'scientific_summary_computed': False,
      'combined_analysis_permitted': False,
  }
  if {name: complete.get(name) for name in predicates} != predicates:
    raise ValueError('v3.3.3 terminal predicates changed.')
  return {
      'path': str(V3_3_3_RUN_DIR.resolve()),
      'model_run_commit': V3_3_3_RUN_COMMIT,
      'freeze_sha256': V3_3_3_FREEZE_SHA256,
      'file_count': 11,
      'file_tree_sha256': V3_3_3_RUN_TREE_SHA256,
      'compiler_file_count': 4,
      'compiler_tree_sha256': V3_3_3_COMPILER_TREE_SHA256,
      'status_predicates': predicates,
  }


def validate_v3_3_3_1_archive() -> dict[str, Any]:
  """Rehash the representation-only v3.3.3.1 structural archive."""
  commits = {
      path: (
          V3_3_3_1_AMENDMENT_COMMIT
          if path.endswith('encoder_skip_ood_sidecar_analysis_amendment_v3_3_3_1.md')
          else V3_3_3_1_IMPLEMENTATION_COMMIT
      )
      for path in V3_3_3_1_SOURCE_BINDINGS
  }
  _validate_source_bindings(V3_3_3_1_SOURCE_BINDINGS, commits)
  _validate_exact_tree(
      V3_3_3_1_ATTEMPT_DIR,
      _binding_rows(V3_3_3_1_ATTEMPT_FILES),
      'cff8dd5418405dd1acef9c6de1d1e2688e63a6807b1ff4e1ef0c8b8908229307',
      archive_commit=V3_3_3_1_ARCHIVE_COMMIT,
  )
  _validate_exact_tree(
      V3_3_3_1_ANALYSIS_DIR,
      _binding_rows(V3_3_3_1_ANALYSIS_FILES),
      '4dcbaa9069b130d160efbde95b1f82b3561ea90d2a38923d259978126e889b2c',
      archive_commit=V3_3_3_1_ARCHIVE_COMMIT,
  )
  analysis = json.loads(
      (V3_3_3_1_ANALYSIS_DIR / 'ANALYSIS.json').read_text(encoding='utf-8')
  )
  if (
      analysis.get('status') != 'complete_controlled_stop_structural_archive'
      or analysis.get('decision')
      != 'controlled_stop_source_program_mismatch_representation_only'
  ):
    raise ValueError('v3.3.3.1 structural archive predicates changed.')
  return {
      'amendment_commit': V3_3_3_1_AMENDMENT_COMMIT,
      'implementation_commit': V3_3_3_1_IMPLEMENTATION_COMMIT,
      'archive_commit': V3_3_3_1_ARCHIVE_COMMIT,
      'attempt_dir': str(V3_3_3_1_ATTEMPT_DIR.resolve()),
      'analysis_dir': str(V3_3_3_1_ANALYSIS_DIR.resolve()),
      'attempt_files': _binding_rows(V3_3_3_1_ATTEMPT_FILES),
      'analysis_files': _binding_rows(V3_3_3_1_ANALYSIS_FILES),
      'attempt_tree_sha256': 'cff8dd5418405dd1acef9c6de1d1e2688e63a6807b1ff4e1ef0c8b8908229307',
      'analysis_tree_sha256': '4dcbaa9069b130d160efbde95b1f82b3561ea90d2a38923d259978126e889b2c',
      'status': analysis['status'],
      'decision': analysis['decision'],
  }


def _validate_started_output_prefix() -> dict[str, Any]:
  """Requires the exact post-START/pre-scientific filesystem prefix."""
  for path, label in (
      (ANALYSIS_DIR, 'v3.3.4.6 analysis output'),
      (ANALYSIS_ATTEMPT_DIR, 'v3.3.4.6 analysis attempt'),
  ):
    if path.exists() or path.is_symlink():
      raise FileExistsError(f'{label} exists during Gate B.')
  if not OUTPUT_DIR.is_dir() or OUTPUT_DIR.is_symlink():
    raise ValueError('Gate B requires the freshly created model-run root.')
  paths = _strict_file_tree(OUTPUT_DIR)
  if [path.name for path in paths] != ['ATTEMPT_STARTED.json']:
    raise ValueError('Gate B permits exactly ATTEMPT_STARTED.json in the run.')
  start = OUTPUT_DIR / 'ATTEMPT_STARTED.json'
  if (start.stat().st_mode & 0o777) != 0o400:
    raise ValueError('ATTEMPT_STARTED.json mode changed before Gate B.')
  return {
      'output_dir': str(OUTPUT_DIR.resolve()),
      'start_path': str(start.resolve()),
      'start_sha256': _sha256(start),
      'start_size_bytes': start.stat().st_size,
      'exact_start_only_prefix': True,
      'analysis_dir_absent': True,
      'analysis_attempt_dir_absent': True,
  }


def validate_freeze(*, allow_started_output: bool = False) -> dict[str, Any]:
  for path in (
      FREEZE_PATH, AMENDMENT_PATH, ORIGINAL_FREEZE_PATH, OUTPUT_DIR,
      ANALYSIS_DIR, ANALYSIS_ATTEMPT_DIR, PREFLIGHT_DIR,
      V3_3_2_FREEZE_PATH, V3_3_2_RUN_DIR,
      V3_3_2_1_ATTEMPT_DIR, V3_3_2_1_ANALYSIS_DIR,
      V3_3_2_2_ATTEMPT_DIR, V3_3_2_2_ANALYSIS_DIR,
      V3_3_3_FREEZE_PATH, V3_3_3_RUN_DIR,
      V3_3_3_1_ATTEMPT_DIR, V3_3_3_1_ANALYSIS_DIR,
      V3_3_3_ANALYSIS_ATTEMPT_DIR, V3_3_3_ANALYSIS_DIR,
      V3_3_4_3_FREEZE_PATH, V3_3_4_4_FREEZE_PATH,
      V3_3_4_4_PREFLIGHT_DIR, V3_3_4_4_EXTERNAL_CACHE_DIR,
  ):
    _reject_confirmation_path(path)
  _validate_file(AMENDMENT_PATH, AMENDMENT_SHA256, 'v3.3.4.6 amendment')
  _validate_file(
      V3_3_4_AMENDMENT_PATH, V3_3_4_AMENDMENT_SHA256,
      'v3.3.4 predecessor amendment',
  )
  _validate_file(
      V3_3_4_1_AMENDMENT_PATH, V3_3_4_1_AMENDMENT_SHA256,
      'v3.3.4.1 predecessor amendment',
  )
  _validate_file(
      V3_3_4_2_AMENDMENT_PATH, V3_3_4_2_AMENDMENT_SHA256,
      'v3.3.4.2 predecessor amendment',
  )
  _validate_file(ORIGINAL_FREEZE_PATH, ORIGINAL_FREEZE_SHA256, 'v3.3 freeze')
  frozen = json.loads(FREEZE_PATH.read_text(encoding='utf-8'))
  version_proof = validate_preflight_version_contract(frozen)
  prior_prefix = validate_prior_v3_3_4_3_consumed_preflight_prefix()
  prior_prefix_binding = canonical_content_binding(prior_prefix)
  prior_v3_3_4_4_prefix = (
      validate_prior_v3_3_4_4_consumed_preflight_prefix()
  )
  prior_v3_3_4_5_archive = (
      validate_prior_v3_3_4_5_controlled_stop_archive()
  )
  prior_v3_3_4_5_archive_binding = canonical_content_binding(
      prior_v3_3_4_5_archive
  )
  expected = {
      'amendment_sha256': AMENDMENT_SHA256,
      'amendment_commit': AMENDMENT_COMMIT,
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'original_freeze_sha256': ORIGINAL_FREEZE_SHA256,
      'v3_3_2_freeze_sha256': V3_3_2_FREEZE_SHA256,
      'v3_3_2_freeze_path': str(V3_3_2_FREEZE_PATH.resolve()),
      'output_dir': str(OUTPUT_DIR.resolve()),
      'analysis_dir': str(ANALYSIS_DIR.resolve()),
      'analysis_attempt_dir': str(ANALYSIS_ATTEMPT_DIR.resolve()),
      'preflight_dir': str(PREFLIGHT_DIR.resolve()),
      'preflight_kernel_cache_dir': str(
          PREFLIGHT_KERNEL_CACHE_DIR.resolve()
      ),
      'model_kernel_cache_dir': str(MODEL_KERNEL_CACHE_DIR.resolve()),
      'ood_anchor_ids': [0, 127, 128, 255],
      'recipient_orders': list(range(20)),
      'ood_record_count': 80,
      'model_apply_count': 320,
      'eight_row_compile_count': 1,
      'six_row_compile_count': 0,
      'identity_rerun_count': 0,
      'main_cube_rerun_count': 0,
      'invariant_rows_between_calls': [0, 1, 3, 5, 6, 7],
      'source_program_contract': SOURCE_PROGRAM_CONTRACT,
      'compiled_backend_equality_is_a_gate': False,
      'denied_cache_environment_names': list(DENIED_CACHE_ENVIRONMENT_NAMES),
      'denied_cache_environment_prefixes': list(
          DENIED_CACHE_ENVIRONMENT_PREFIXES
      ),
      'cache_isolation_contract': CACHE_ISOLATION_CONTRACT,
      'external_freeze_authorization_contract': (
          EXTERNAL_FREEZE_AUTHORIZATION_CONTRACT
      ),
      'source_input_audit_contract': SOURCE_INPUT_AUDIT_CONTRACT,
      'same_object_attestation_contract': SAME_OBJECT_ATTESTATION_CONTRACT,
      'dispatch_journal_contract': DISPATCH_JOURNAL_CONTRACT,
      'failed_current_contract': FAILED_CURRENT_CONTRACT,
      'raw_record_contract': RAW_RECORD_CONTRACT,
      'raw_manifest_contract': RAW_MANIFEST_CONTRACT,
      'terminal_contract': {
          **TERMINAL_CONTRACT,
          'execution_contract': frozen.get('terminal_contract', {}).get(
              'execution_contract'
          ),
      },
      'preflight_contract': PREFLIGHT_CONTRACT,
      'compiled_diagnostics_contract': COMPILED_DIAGNOSTICS_CONTRACT,
      'publication_contract_v3_3_4_1': PUBLICATION_CONTRACT_V3_3_4_1,
      'nonpublication_terminal_contract_v3_3_4_6': (
          NONPUBLICATION_TERMINAL_CONTRACT_V3_3_4_6
      ),
      'prior_v3_3_4_3_consumed_preflight_prefix': prior_prefix,
      'prior_v3_3_4_4_consumed_preflight_prefix': prior_v3_3_4_4_prefix,
      'prior_v3_3_4_5_controlled_stop_archive': prior_v3_3_4_5_archive,
      'prior_v3_3_4_5_controlled_stop_archive_content_binding': (
          prior_v3_3_4_5_archive_binding
      ),
      'v3_3_2_1_failure_status': EXPECTED_V3_3_2_1_FAILURE_STATUS,
      'v3_3_2_2_archive_status': EXPECTED_V3_3_2_2_ARCHIVE_STATUS,
  }
  for name, value in expected.items():
    if frozen.get(name) != value:
      raise ValueError(f'v3.3.4.6 freeze mismatch: {name}.')
  inherited_freeze = json.loads(
      V3_3_4_5_FREEZE_PATH.read_text(encoding='utf-8')
  )
  expected_freeze_keys = (
      set(inherited_freeze)
      - {'nonpublication_terminal_contract_v3_3_4_5'}
      | {
          'nonpublication_terminal_contract_v3_3_4_6',
          'prior_v3_3_4_5_controlled_stop_archive',
          'prior_v3_3_4_5_controlled_stop_archive_content_binding',
      }
  )
  if (
      set(frozen) != expected_freeze_keys or len(frozen) != 88
      or len(frozen.get('file_sha256', {})) != 144
  ):
    raise ValueError('v3.3.4.6 freeze key/source counts changed.')
  literal_signatures = frozen.get('program_signatures')
  if (
      not isinstance(literal_signatures, dict)
      or _canonical_sha256(literal_signatures)
      != SOURCE_PROGRAM_CONTRACT['program_signatures_sha256']
  ):
    raise ValueError('Frozen literal program-signature object changed.')
  signature_contract = frozen.get('program_signature_attestation_contract')
  if signature_contract != {
      **PROGRAM_SIGNATURE_ATTESTATION_CONTRACT,
      'literal_program_signatures': literal_signatures,
  }:
    raise ValueError('Program-signature attestation contract changed.')
  inventory_contract = frozen.get('source_inventory_contract')
  if not isinstance(inventory_contract, dict):
    raise ValueError('Source-inventory contract is absent.')
  if set(inventory_contract) != set(SOURCE_INVENTORY_CONTRACT):
    raise ValueError('Source-inventory contract key set changed.')
  if (
      inventory_contract.get('source_row_count') != 144
      or inventory_contract.get('prospective_upstream_source_file_count') != 26
      or inventory_contract.get('inherited_source_authority_commit')
      != V3_3_4_5_COMMIT
      or inventory_contract.get('amendment_authority_commit')
      != AMENDMENT_COMMIT
  ):
    raise ValueError('Source-inventory scalar contract changed.')
  loaded_contract = inventory_contract.get('loaded_scientific_module_contract')
  if not isinstance(loaded_contract, list) or not loaded_contract:
    raise ValueError('Loaded scientific-module contract is empty.')
  source_rows = inventory_contract.get('rows')
  if not isinstance(source_rows, list) or len(source_rows) != 144:
    raise ValueError('Frozen 144-row source inventory changed.')
  if frozen.get('original_run') != EXPECTED_ORIGINAL_BINDING:
    raise ValueError('Frozen original-v3.3 binding differs from the amendment.')

  tracked = {str(FREEZE_PATH.relative_to(_REPO))}
  file_sha = frozen.get('file_sha256')
  if not isinstance(file_sha, dict) or not file_sha:
    raise ValueError('v3.3.4.6 frozen file inventory is absent.')
  for relative, expected_sha in sorted(file_sha.items()):
    path = (_REPO / relative).resolve()
    if not path.is_relative_to(_REPO.resolve()) or str(path.relative_to(_REPO)) != relative:
      raise ValueError(f'Frozen path escaped or is not normalized: {relative}.')
    _validate_file(path, expected_sha, f'v3.3.4.6 bundle {relative}')
    tracked.add(relative)
  implementation_paths = sorted({
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
      'experiments/interpretability/opensplice/'
      'run_device_preflight_v3_3_4_6.py',
      'experiments/interpretability/opensplice/'
      'run_device_preflight_v3_3_4_6_test.py',
      'experiments/interpretability/opensplice/'
      'run_encoder_skip_ood_sidecar_v3_3_4_6.py',
      'experiments/interpretability/opensplice/'
      'run_encoder_skip_ood_sidecar_v3_3_4_6_test.py',
      'experiments/interpretability/opensplice/'
      'run_encoder_skip_ood_sidecar_v3_3_4_6.sh',
  })
  amendment_relative = AMENDMENT_PATH.relative_to(_REPO).as_posix()
  freeze_relative = FREEZE_PATH.relative_to(_REPO).as_posix()
  source_head = inventory_contract.get(
      'implementation_source_authority_commit'
  )
  launch_head = subprocess.check_output(
      ('git', '-C', str(_REPO), 'rev-parse', 'HEAD'), text=True
  ).strip()
  if not isinstance(source_head, str) or not re.fullmatch(
      r'[0-9a-f]{40}', source_head
  ):
    raise ValueError('Implementation source authority commit is malformed.')
  _require_commit_edge(
      source_head, AMENDMENT_COMMIT,
      [f'A\t{relative}' for relative in implementation_paths],
  )
  _require_commit_edge(
      launch_head, source_head, [f'A\t{freeze_relative}']
  )
  expected_rows = []
  for row, (relative, expected_sha) in zip(
      source_rows, sorted(file_sha.items()), strict=True
  ):
    if not isinstance(row, Mapping) or set(row) != {
        'path', 'sha256', 'size_bytes', 'git_mode', 'authority_commit'
    }:
      raise ValueError('Frozen source-inventory row schema changed.')
    expected_authority = (
        AMENDMENT_COMMIT if relative == amendment_relative
        else source_head if relative in implementation_paths
        else V3_3_4_5_COMMIT
    )
    path = (_REPO / relative).resolve()
    index_line = subprocess.run(
        ('git', '-C', str(_REPO), 'ls-files', '-s', '--', relative),
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    mode = index_line.split()[0]
    authority_blob = subprocess.check_output(
        ('git', '-C', str(_REPO), 'show', f'{expected_authority}:{relative}')
    )
    launch_blob = subprocess.check_output(
        ('git', '-C', str(_REPO), 'show', f'{launch_head}:{relative}')
    )
    authority_line = subprocess.check_output(
        ('git', '-C', str(_REPO), 'ls-tree', expected_authority, '--', relative),
        text=True,
    ).strip()
    if (
        hashlib.sha256(authority_blob).hexdigest() != expected_sha
        or hashlib.sha256(launch_blob).hexdigest() != expected_sha
        or len(authority_blob) != path.stat().st_size
        or len(launch_blob) != path.stat().st_size
        or not authority_line.startswith(f'{mode} blob ')
    ):
      raise ValueError(f'Frozen source authority changed: {relative}.')
    expected_rows.append({
        'path': relative, 'sha256': expected_sha,
        'size_bytes': path.stat().st_size, 'git_mode': mode,
        'authority_commit': expected_authority,
    })
  if source_rows != expected_rows:
    raise ValueError(
        'Frozen source-inventory rows differ from live/authority Git bytes.'
    )
  for relative in sorted(tracked):
    subprocess.run(
        ('git', '-C', str(_REPO), 'ls-files', '--error-unmatch', relative),
        check=True,
        capture_output=True,
    )
  if subprocess.check_output(
      ('git', '-C', str(_REPO), 'diff', '--binary', 'HEAD', '--')
  ):
    raise ValueError('v3.3.4.6 requires globally tracked-clean HEAD before import.')

  original_bundle = v33_bootstrap.validate_freeze()
  original_run = validate_original_run()
  v3_3_2_run = validate_v3_3_2_run()
  if frozen.get('v3_3_2_run') != v3_3_2_run:
    raise ValueError('Frozen v3.3.2 apply-zero run binding changed.')
  if literal_signatures != v3_3_2_run['eight_row_compiler'].get(
      'program_signatures'
  ):
    raise ValueError('Frozen literal signatures differ from exact v3.3.2.')
  v3_3_2_frozen = json.loads(
      V3_3_2_FREEZE_PATH.read_text(encoding='utf-8')
  )
  v3_3_1 = validate_v3_3_1_status(v3_3_2_frozen)
  v3_3_2_1_failure = validate_v3_3_2_1_failure()
  v3_3_2_2_archive = validate_v3_3_2_2_archive()
  v3_3_3_run = validate_v3_3_3_run()
  v3_3_3_1_archive = validate_v3_3_3_1_archive()
  if frozen.get('v3_3_3_1_archive') != v3_3_3_1_archive:
    raise ValueError('Frozen v3.3.3.1 archive binding changed.')
  authorization = validate_external_freeze_authorization()
  predecessor_absence = validate_predecessor_path_absence()
  one_shot_absence = (
      _validate_started_output_prefix()
      if allow_started_output
      else validate_one_shot_output_absence()
  )
  preflight_state = validate_preflight_state_for_role()
  return {
      'path': str(FREEZE_PATH.resolve()),
      'sha256': _sha256(FREEZE_PATH),
      'git_head': subprocess.check_output(
          ('git', '-C', str(_REPO), 'rev-parse', 'HEAD'), text=True
      ).strip(),
      'tracked_head_clean': True,
      'tracked_paths': sorted(tracked),
      'original_bundle': original_bundle,
      'original_run': original_run,
      'v3_3_2_run': v3_3_2_run,
      'v3_3_1_status': v3_3_1,
      'v3_3_2_1_failure_status': v3_3_2_1_failure,
      'v3_3_2_2_archive_status': v3_3_2_2_archive,
      'v3_3_3_run': v3_3_3_run,
      'v3_3_3_1_archive': v3_3_3_1_archive,
      'external_freeze_authorization': authorization,
      'preflight_version_proof': version_proof,
      'prior_v3_3_4_3_consumed_preflight_prefix': prior_prefix,
      'prior_v3_3_4_3_consumed_preflight_prefix_content_binding': (
          prior_prefix_binding
      ),
      'prior_v3_3_4_4_consumed_preflight_prefix': prior_v3_3_4_4_prefix,
      'prior_v3_3_4_4_consumed_preflight_prefix_content_binding': dict(
          V3_3_4_4_CONSUMED_PREFIX_BINDING
      ),
      'prior_v3_3_4_5_controlled_stop_archive': prior_v3_3_4_5_archive,
      'prior_v3_3_4_5_controlled_stop_archive_content_binding': (
          prior_v3_3_4_5_archive_binding
      ),
      'v3_3_4_predecessor_path_absence': predecessor_absence,
      'one_shot_output_absence': one_shot_absence,
      'preflight_state': preflight_state,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--check', action='store_true', required=True)
  record = {
      'cache_environment': assert_cache_environment_sanitized(),
      'generated_bindings': (
          v33_bootstrap.proto_gate.validate_generated_bindings_before_import()
      ),
      'freeze': validate_freeze(),
      'model_or_jax_imported': False,
  }
  print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == '__main__':
  main()
