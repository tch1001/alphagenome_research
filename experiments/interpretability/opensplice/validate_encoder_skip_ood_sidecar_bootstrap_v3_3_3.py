#!/usr/bin/env python3
"""Standard-library-only pre-import gate for the v3.3.3 OOD sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))
# pylint: disable=g-import-not-at-top
import validate_encoder_skip_bootstrap_v3_3 as v33_bootstrap


AMENDMENT_PATH = (
    _HERE / 'v3_wider_mechanism'
    / 'encoder_skip_ood_sidecar_compiler_gate_amendment_v3_3_3.md'
)
AMENDMENT_SHA256 = (
    'c9b00398296e683ac6e1c321fd8c4302f96b2e62bb23828e8b5ef2fe9de3f70b'
)
AMENDMENT_COMMIT = '783a7d0dfbd5f26e22152d1201dacf82f2b01d15'
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
FREEZE_PATH = _HERE / 'encoder_skip_ood_sidecar_v3_3_3_freeze.json'
OUTPUT_DIR = (
    _HERE / 'results' / 'v3_3_3_development_ood_sidecar_one_shot'
)
ANALYSIS_DIR = (
    _HERE / 'results' / 'v3_3_3_development_ood_sidecar_analysis'
)
ANALYSIS_ATTEMPT_DIR = (
    _HERE / 'results' / 'v3_3_3_development_ood_sidecar_analysis_attempt'
)
PREFLIGHT_DIR = _HERE / 'results' / 'v3_3_3_device_preflight'
PREFLIGHT_KERNEL_CACHE_DIR = (
    _HERE / 'results' / 'v3_3_3_preflight_kernel_cache'
)
MODEL_KERNEL_CACHE_DIR = _HERE / 'results' / 'v3_3_3_model_kernel_cache'

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
CACHE_ROLE_ENVIRONMENT = 'ALPHAGENOME_V3_3_3_CACHE_ROLE'
CACHE_ROOT_ENVIRONMENT = 'ALPHAGENOME_V3_3_3_CACHE_ROOT'
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
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
  return hashlib.sha256(json.dumps(
      value,
      sort_keys=True,
      separators=(',', ':'),
      ensure_ascii=False,
      allow_nan=False,
  ).encode('utf-8')).hexdigest()


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
    raise ValueError('A frozen v3.3.3 cache role/root is required.')
  root = Path(root_text).resolve()
  if role == 'external_preflight' and root != PREFLIGHT_KERNEL_CACHE_DIR.resolve():
    raise ValueError('External-preflight cache root changed.')
  if role == 'model' and root != MODEL_KERNEL_CACHE_DIR.resolve():
    raise ValueError('Model cache root changed.')
  if role == 'dry_run' and not root.name.startswith(
      'alphagenome-v3.3.3-dry-cache.'
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
  for directory in sorted(directories):
    relative = '.' if directory == root else str(directory.relative_to(root))
    digest.update(b'D\0')
    digest.update(relative.encode('utf-8'))
    digest.update(b'\0')
  records = {}
  for path in sorted(files):
    relative = str(path.relative_to(root))
    binding = {'sha256': _sha256(path), 'size_bytes': path.stat().st_size}
    records[relative] = binding
    digest.update(b'F\0')
    digest.update(relative.encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(binding['sha256']))
  return {
      'root': str(root),
      'file_count': len(files),
      'directory_count': len(directories),
      'files': records,
      'tree_sha256': digest.hexdigest(),
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
      (output_dir, 'v3.3.3 sidecar output'),
      (analysis_dir, 'v3.3.3 sidecar analysis'),
      (analysis_attempt_dir, 'v3.3.3 sidecar analysis attempt'),
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


def validate_preflight_state_for_role() -> dict[str, Any]:
  """Binds sole preflight absence/completion before model/JAX import."""
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
  files = _strict_file_tree(PREFLIGHT_DIR)
  expected_names = {
      '.allocation.lock',
      '.preflight_0000.reserved',
      'preflight_0000.json',
      'preflight_0000.stdout.log',
      'preflight_0000.stderr.log',
  }
  if {path.name for path in files} != expected_names or len(files) != 5:
    raise ValueError('Model process requires one exact preflight attempt tree.')
  record_path = PREFLIGHT_DIR / 'preflight_0000.json'
  record = json.loads(record_path.read_text(encoding='utf-8'))
  expected = {
      'script_version': 'opensplice-device-preflight-v3.3.3',
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
  for stream in ('stdout', 'stderr'):
    binding = record.get('logs', {}).get(stream, {})
    expected_path = PREFLIGHT_DIR / f'preflight_0000.{stream}.log'
    if (
        binding.get('path') != str(expected_path.resolve())
        or binding.get('sha256') != _sha256(expected_path)
    ):
      raise ValueError(f'Sole preflight {stream} binding changed.')
  cache = record.get('observation', {}).get(
      'v3_3_3_runtime_environment', {}
  ).get('cache_environment', {})
  if (
      cache.get('cache_role') != 'external_preflight'
      or cache.get('cache_root')
      != str(PREFLIGHT_KERNEL_CACHE_DIR.resolve())
      or cache.get('pre_import_file_count') != 0
  ):
    raise ValueError('Sole preflight cache attestation changed.')
  post_cache = record.get('external_cache_post_observation', {})
  if (
      post_cache.get('root')
      != str(PREFLIGHT_KERNEL_CACHE_DIR.resolve())
      or post_cache.get('diagnostic_outputs_only_no_cache_input') is not True
      or post_cache != cache_output_tree_binding(PREFLIGHT_KERNEL_CACHE_DIR)
  ):
    raise ValueError('Sole preflight cache-output binding changed.')
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


def validate_freeze() -> dict[str, Any]:
  for path in (
      FREEZE_PATH, AMENDMENT_PATH, ORIGINAL_FREEZE_PATH, OUTPUT_DIR,
      ANALYSIS_DIR, ANALYSIS_ATTEMPT_DIR, PREFLIGHT_DIR,
      V3_3_2_FREEZE_PATH, V3_3_2_RUN_DIR,
      V3_3_2_1_ATTEMPT_DIR, V3_3_2_1_ANALYSIS_DIR,
      V3_3_2_2_ATTEMPT_DIR, V3_3_2_2_ANALYSIS_DIR,
  ):
    _reject_confirmation_path(path)
  _validate_file(AMENDMENT_PATH, AMENDMENT_SHA256, 'v3.3.3 amendment')
  _validate_file(ORIGINAL_FREEZE_PATH, ORIGINAL_FREEZE_SHA256, 'v3.3 freeze')
  frozen = json.loads(FREEZE_PATH.read_text(encoding='utf-8'))
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
      'v3_3_2_1_failure_status': EXPECTED_V3_3_2_1_FAILURE_STATUS,
      'v3_3_2_2_archive_status': EXPECTED_V3_3_2_2_ARCHIVE_STATUS,
  }
  for name, value in expected.items():
    if frozen.get(name) != value:
      raise ValueError(f'v3.3.3 freeze mismatch: {name}.')
  literal_signatures = frozen.get('program_signatures')
  if (
      not isinstance(literal_signatures, dict)
      or _canonical_sha256(literal_signatures)
      != SOURCE_PROGRAM_CONTRACT['program_signatures_sha256']
  ):
    raise ValueError('Frozen literal program-signature object changed.')
  if frozen.get('original_run') != EXPECTED_ORIGINAL_BINDING:
    raise ValueError('Frozen original-v3.3 binding differs from the amendment.')

  tracked = {str(FREEZE_PATH.relative_to(_REPO))}
  file_sha = frozen.get('file_sha256')
  if not isinstance(file_sha, dict) or not file_sha:
    raise ValueError('v3.3.3 frozen file inventory is absent.')
  for relative, expected_sha in sorted(file_sha.items()):
    path = (_REPO / relative).resolve()
    if not path.is_relative_to(_REPO.resolve()) or str(path.relative_to(_REPO)) != relative:
      raise ValueError(f'Frozen path escaped or is not normalized: {relative}.')
    _validate_file(path, expected_sha, f'v3.3.3 bundle {relative}')
    tracked.add(relative)
  for relative in sorted(tracked):
    subprocess.run(
        ('git', '-C', str(_REPO), 'ls-files', '--error-unmatch', relative),
        check=True,
        capture_output=True,
    )
  if subprocess.check_output(
      ('git', '-C', str(_REPO), 'diff', '--binary', 'HEAD', '--')
  ):
    raise ValueError('v3.3.3 requires globally tracked-clean HEAD before import.')

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
  one_shot_absence = validate_one_shot_output_absence()
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
