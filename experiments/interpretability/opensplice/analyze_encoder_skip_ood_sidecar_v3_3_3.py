#!/usr/bin/env python3
"""CPU-only structural audit for the prospective v3.3.3 OOD sidecar.

This module intentionally does not import JAX, AlphaGenome, or model code.  It
audits provenance, the frozen source-program boundary, append-only execution
prefixes, and (when present) all raw structural controls.  Compiled backend
HLO is retained as diagnostic provenance and is never an equality gate.  No
normalization, Shapley value, resolution result, rank, or nomination is
computed here.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence


ANALYSIS_VERSION = 'opensplice-encoder-skip-ood-sidecar-analysis-v3.3.3'
SCRIPT_VERSION = 'opensplice-encoder-skip-ood-sidecar-v3.3.3'
ATTEMPT_ID = 'opensplice-v3.3.3-development-ood-sidecar-one-shot'
AMENDMENT_SHA256 = (
    'c9b00398296e683ac6e1c321fd8c4302f96b2e62bb23828e8b5ef2fe9de3f70b'
)
AMENDMENT_COMMIT = '783a7d0dfbd5f26e22152d1201dacf82f2b01d15'
ORIGINAL_PROTOCOL_SHA256 = (
    '85151029297cf4b4c1a07dd2a0f47b10b0663acd08ddfe25ba72b9898b1beea0'
)
V3_3_2_AMENDMENT_SHA256 = (
    '42cd43bbc25517d35b3e21dd5df7bf68a37ee46c51cb71bcdf363ac4de3b19e3'
)
EXPECTED_RECORD_COUNT = 80
EXPECTED_APPLY_COUNT = 320
ANCHOR_IDS = (0, 127, 128, 255)
RECIPIENT_ORDERS = tuple(range(20))
INVARIANT_ROWS = (0, 1, 3, 5, 6, 7)
ACTIVE_ROWS = (2, 4)
CONFIRMATION_DISCLOSURE = (
    'Later-exon metadata/labels were exposed after protocol freeze; '
    'no later-exon model outputs, activations, or interventions are used.'
)
EMPTY_SHA256 = (
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
)
SOURCE_STABLEHLO = {
    'sha256': '69dbf2a054cf89e56a9000dc0c04e5cd4ba425f016885ddc127d22c01661bddd',
    'size_bytes': 3_196_162,
}
SOURCE_PRE_BACKEND_HLO = {
    'sha256': '675b9cf26fe8f59a1d138e856ee84f48edb20d8c896ac9e539b6b064dcef2750',
    'size_bytes': 1_829_833,
}
PROGRAM_SIGNATURES_SHA256 = (
    'd8f95fb9d3637fd263cc3da0f6a33409d5fb2a5cf37e348723ecc89b3224c300'
)
ENTRY_ABI_SHA256 = (
    'ebf900771a87775a5fb657b90131fd884d68ce4725245defa164bf5066c74a80'
)

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
_AMENDMENT_PATH = (
    _HERE / 'v3_wider_mechanism'
    / 'encoder_skip_ood_sidecar_compiler_gate_amendment_v3_3_3.md'
)
_FREEZE_PATH = _HERE / 'encoder_skip_ood_sidecar_v3_3_3_freeze.json'
_RUN_DIR = _HERE / 'results/v3_3_3_development_ood_sidecar_one_shot'
_ANALYSIS_DIR = _HERE / 'results/v3_3_3_development_ood_sidecar_analysis'
_ANALYSIS_ATTEMPT_DIR = (
    _HERE / 'results/v3_3_3_development_ood_sidecar_analysis_attempt'
)
_PREFLIGHT_DIR = _HERE / 'results/v3_3_3_device_preflight'
_TEST_PATH = _HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_3_test.py'
_V3_3_2_ANALYZER_PATH = _HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_2.py'
_V3_3_2_RUN_DIR = _HERE / 'results/v3_3_2_development_ood_sidecar_one_shot'
_V3_3_2_1_ATTEMPT_DIR = (
    _HERE / 'results/v3_3_2_development_ood_sidecar_analysis_v3_3_2_1_attempt'
)
_V3_3_2_1_ANALYSIS_DIR = (
    _HERE / 'results/v3_3_2_development_ood_sidecar_analysis_v3_3_2_1'
)
_V3_3_2_2_ATTEMPT_DIR = (
    _HERE / 'results/v3_3_2_development_ood_sidecar_analysis_v3_3_2_2_attempt'
)
_V3_3_2_2_ANALYSIS_DIR = (
    _HERE / 'results/v3_3_2_development_ood_sidecar_analysis_v3_3_2_2'
)
_ANALYSIS_ATTEMPT_TOKEN = object()

V3_3_2_ANALYZER_SHA256 = (
    '90f00a6d51f33ac456a0fd799f2b9caf456b58944928886dc1577731707f205e'
)

_V3_3_2_RUN_FILES = {
    'ATTEMPT_STARTED.json': 'd1bb85b3c35d892f10f67a5510b85784eb1b6b989c9ae32290b8c0fdc3c8d235',
    'RUN_COMPLETE.json': 'd88c8becc96a24f1f56a1cee82da35fdd331ee3b40ba4108ae98f44ae24b4ad7',
    'RAW_MANIFEST.json': 'fadbff5ab512dea3d25edb39dc9a91ffe25473b73f1e33fd53f97b60fc8436fd',
    'IMPORT_PROVENANCE_PRE_MODEL.json': 'a74f3c9658e9d2286724680b52f4ea788d492f4fa9d7c52b20a53c90d57edc99',
    'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json': 'a74f3c9658e9d2286724680b52f4ea788d492f4fa9d7c52b20a53c90d57edc99',
    'IMPORT_PROVENANCE.json': 'a74f3c9658e9d2286724680b52f4ea788d492f4fa9d7c52b20a53c90d57edc99',
    'PROTOBUF_PROVENANCE.json': '2498a940f6ee15e54e72e8f51587d4c42ffc1b49851873c31ad09085315d0ba8',
    'compiler/eight_row/COMPILER_PROVENANCE.json': 'bd20e21a56a9ca5498d7119771bb1da9ac2e156ed3190d3ce3aa09ff2d2e312c',
    'compiler/eight_row/graph.stablehlo.mlir': SOURCE_STABLEHLO['sha256'],
    'compiler/eight_row/graph.pre_backend.hlo.txt': SOURCE_PRE_BACKEND_HLO['sha256'],
    'compiler/eight_row/graph.compiled.hlo.txt': 'b436435ebb14b87cf9929ee9b16fc2c74d1764460c701f8160f1dc092687b718',
}
V3_3_2_RUN_TREE_SHA256 = (
    '4ac66e45a4d7d65af2785904d11b23bf7e809e07f3f617e190772242b2e7a4ab'
)
V3_3_2_COMPILER_TREE_SHA256 = (
    '4378048568ff58a2bbee55ba9da750498b89fdef72c97911815cf895c8a8b7d1'
)

_V3_3_2_1_SOURCE_COMMIT = 'b43051aa4a893e24a38e932900d349278c9ead88'
_V3_3_2_1_SOURCES = {
    'experiments/interpretability/opensplice/v3_wider_mechanism/encoder_skip_ood_sidecar_analysis_amendment_v3_3_2_1.md': '81a4f4c126b83225b02c7de5cf0dc6fd0baf6085b84b9ed5dd7a3677744090ba',
    'experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_2_1.py': '35db9ca198cb5d7f03621ccf322ea116f98cea3bfdc711006dfd20bc809e8048',
    'experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_2_1_test.py': 'a8733f3ffb35920dda2f6a856076cbe82a9e90aa2fe483169150dbad4421a1b8',
    'experiments/interpretability/opensplice/encoder_skip_ood_sidecar_analysis_v3_3_2_1_freeze.json': '3871ab41b16105a94673e89381d32d7253b014c64eda5c6789eaecf16477c061',
    'experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_analysis_v3_3_2_1.sh': 'ea5cce6ae631ba3fa2bf0082d691d0896ef0fed7b20f0d908034e17775060caa',
}
_V3_3_2_1_ATTEMPT_FILES = {
    'ANALYSIS_ATTEMPT_STARTED.json': {
        'size_bytes': 8_616,
        'sha256': 'a87c4e15ed67a363d07c434ca232540687950d145e67492b9ed9c17d9adebf1d',
    },
    'ANALYSIS_FAILURE.json': {
        'size_bytes': 2_163,
        'sha256': '1cd933623ecdfb328d5db458b16df909e632a560361dbb547f83c22cf13ab7c7',
    },
}
V3_3_2_1_ATTEMPT_TREE_SHA256 = (
    '5e97b191e781c5141d2f308deefacfa8f6a196449fd7e11f36c22828a13f036a'
)

_V3_3_2_2_AMENDMENT_COMMIT = '2a2cc59136f5b83f3a7c265b5197e30cdecd7c11'
_V3_3_2_2_IMPLEMENTATION_COMMIT = '67abe303082c62fd925c3c23d9a23b3e0f4526f6'
_V3_3_2_2_ARCHIVE_COMMIT = '2f73f8750384c7fc5c73bded379e667a642c5d0a'
_V3_3_2_2_SOURCES = {
    'experiments/interpretability/opensplice/v3_wider_mechanism/encoder_skip_ood_sidecar_analysis_amendment_v3_3_2_2.md': '3188b44f85d8315eb4a099b42930b2d08f76074ffb190ef11b67a9f39e788a3d',
    'experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_2_2.py': '70be2f80e598f6c307511dfcad1a550a2438766b3348014be1e6ff25c9b99221',
    'experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_2_2_test.py': 'cb7541e75dd65130f7f123643d88adac4f82e984afce4856babfc58284723a4d',
    'experiments/interpretability/opensplice/encoder_skip_ood_sidecar_analysis_v3_3_2_2_freeze.json': 'e7c3fe72c9d9ca5b23299dfbc2a2991643b19722cf9a5c386debf73f367fa520',
    'experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_analysis_v3_3_2_2.sh': '7eb4d6b8dda1a415881a6934d6cf9e30cbfc707fd531eb1a1d19df3b90b1a2f9',
}
_V3_3_2_2_ATTEMPT_FILES = {
    'ANALYSIS_ATTEMPT_STARTED.json': {
        'size_bytes': 11_042,
        'sha256': 'bdbcc6f37093924ddc79ed38e674d084761e044aefeec065d25c6692605028af',
    },
    'ANALYSIS_COMPLETE.json': {
        'size_bytes': 789,
        'sha256': '5bf9d45c3b890fff7653b5d9ee57ffa959ac09933c3620069edf513eca3473f5',
    },
}
_V3_3_2_2_OUTPUT_FILES = {
    'ANALYSIS.json': {
        'size_bytes': 60_844,
        'sha256': '9af699921344ff8528260f7b6b2d2d57a529b9863c40edf99757929934e44b61',
    },
    'RESULT.md': {
        'size_bytes': 811,
        'sha256': 'be9f9c7fe31363f926999de78085b3d10c5552a80e14d0b5432deb9fb7adfc03',
    },
}
V3_3_2_2_ATTEMPT_TREE_SHA256 = (
    'bc703c43e8afe4a01b18621180bb5d3e90c2a49c4902d175e622e0c48eeea29d'
)
V3_3_2_2_OUTPUT_TREE_SHA256 = (
    '581392f933c909fe4a56d51cd03089f6c506bdc058dfdbd902abfc49c8332a0c'
)


class AnalysisError(RuntimeError):
  """Raised when a structural/provenance gate fails closed."""


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


_v332 = _load_cpu_module(
    _V3_3_2_ANALYZER_PATH, V3_3_2_ANALYZER_SHA256,
    '_opensplice_frozen_analyzer_v3_3_2_for_v3_3_3',
)


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


def _exact_keys(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
  if not isinstance(value, Mapping) or set(value) != keys:
    raise AnalysisError(f'{label} key set changed.')
  return value


def _tree_digest(paths: Iterable[Path], root: Path) -> str:
  digest = hashlib.sha256()
  for path in sorted(paths):
    digest.update(path.relative_to(root).as_posix().encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(_sha256(path)))
  return digest.hexdigest()


def _strict_tree(root: Path, expected_relatives: set[str], label: str) -> list[Path]:
  _guard_path(root)
  if root.is_symlink() or not root.is_dir():
    raise AnalysisError(f'{label} root is absent or unsafe.')
  expected_files = {(root / relative).resolve() for relative in expected_relatives}
  expected_dirs = {root.resolve()}
  for path in expected_files:
    expected_dirs.update(parent for parent in path.parents if parent == root.resolve() or root.resolve() in parent.parents)
  observed_files: set[Path] = set()
  for lexical in root.rglob('*'):
    mode = lexical.lstat().st_mode
    if stat.S_ISLNK(mode):
      raise AnalysisError(f'{label} contains a symlink.')
    if stat.S_ISREG(mode):
      observed_files.add(lexical.resolve())
    elif stat.S_ISDIR(mode):
      if lexical.resolve() not in expected_dirs:
        raise AnalysisError(f'{label} contains an extra/empty directory.')
    else:
      raise AnalysisError(f'{label} contains a special entry.')
  if observed_files != expected_files:
    raise AnalysisError(f'{label} membership changed.')
  return sorted(observed_files)


def _validate_bound_tree(
    root: Path, files: Mapping[str, Mapping[str, Any]], tree_sha256: str,
    label: str,
) -> dict[str, Any]:
  paths = _strict_tree(root, set(files), label)
  for relative, binding in files.items():
    path = root / relative
    _strict_regular(path, f'{label}.{relative}')
    if (
        path.stat().st_size != binding['size_bytes']
        or _sha256(path) != binding['sha256']
    ):
      raise AnalysisError(f'{label}.{relative} binding changed.')
  if _tree_digest(paths, root) != tree_sha256:
    raise AnalysisError(f'{label} tree digest changed.')
  return {
      'path': str(root.resolve()), 'file_count': len(paths),
      'tree_sha256': tree_sha256,
      'files': copy.deepcopy(dict(files)),
  }


def _git_blob_sha256(commit: str, relative: str) -> str:
  try:
    value = subprocess.check_output(
        ('git', '-C', str(_REPO_ROOT), 'show', f'{commit}:{relative}')
    )
  except subprocess.CalledProcessError as error:
    raise AnalysisError(f'Historical source is absent: {commit}:{relative}.') from error
  return hashlib.sha256(value).hexdigest()


def _validate_source_bundle(
    sources: Mapping[str, str], *, implementation_commit: str,
    amendment_commit: str | None = None,
) -> dict[str, Any]:
  observed = {}
  for relative, expected_sha in sources.items():
    path = _REPO_ROOT / relative
    _strict_regular(path, f'bound source {relative}')
    if _sha256(path) != expected_sha:
      raise AnalysisError(f'Live bound source changed: {relative}.')
    commit = (
        amendment_commit
        if amendment_commit is not None and 'amendment_' in relative
        else implementation_commit
    )
    if _git_blob_sha256(commit, relative) != expected_sha:
      raise AnalysisError(f'Historical bound source changed: {commit}:{relative}.')
    observed[relative] = expected_sha
  return {'source_count': len(observed), 'source_sha256': observed}


def _validate_v3_3_2_run() -> dict[str, Any]:
  bindings = {relative: {'sha256': digest} for relative, digest in _V3_3_2_RUN_FILES.items()}
  paths = _strict_tree(_V3_3_2_RUN_DIR, set(bindings), 'consumed v3.3.2 run')
  for relative, digest in _V3_3_2_RUN_FILES.items():
    if _sha256(_V3_3_2_RUN_DIR / relative) != digest:
      raise AnalysisError(f'Consumed v3.3.2 file changed: {relative}.')
  if len(paths) != 11 or _tree_digest(paths, _V3_3_2_RUN_DIR) != V3_3_2_RUN_TREE_SHA256:
    raise AnalysisError('Consumed v3.3.2 whole-run tree changed.')
  compiler_paths = [path for path in paths if path.relative_to(_V3_3_2_RUN_DIR).parts[0] == 'compiler']
  if len(compiler_paths) != 4 or _tree_digest(compiler_paths, _V3_3_2_RUN_DIR) != V3_3_2_COMPILER_TREE_SHA256:
    raise AnalysisError('Consumed v3.3.2 compiler tree changed.')
  manifest = _read_json(_V3_3_2_RUN_DIR / 'RAW_MANIFEST.json', 'v3.3.2 RAW_MANIFEST')
  if manifest != {'artifact_count': 0, 'artifact_sha256': {}, 'artifact_tree_sha256': EMPTY_SHA256}:
    raise AnalysisError('Consumed v3.3.2 raw manifest is not exact empty state.')
  completion = _read_json(_V3_3_2_RUN_DIR / 'RUN_COMPLETE.json', 'v3.3.2 RUN_COMPLETE')
  predicates = {
      'status': 'controlled_stop', 'stop_reason': 'compiler_graph_mismatch',
      'eight_row_compile_count': 1, 'six_row_compile_count': 0,
      'model_apply_count': 0,
      'ood_anchor_record_count': 0, 'ood_invalid_count': 0,
      'identity_rerun_count': 0, 'main_cube_rerun_count': 0,
      'confirmation_model_calls': 0,
      'scientific_summary_computed': False,
      'shapley_or_nomination_computed': False,
  }
  if {key: completion.get(key) for key in predicates} != predicates:
    raise AnalysisError('Consumed v3.3.2 terminal predicates changed.')
  additional = {
      'unique_recipient_anchor_count': 0,
      'all_80_recipient_anchors_complete': False,
      'old_ood_records_reused': 0,
      'id0_all20': False,
      'id255_all20': False,
  }
  if {key: completion.get(key) for key in additional} != additional:
    raise AnalysisError('Consumed v3.3.2 extended terminal predicates changed.')
  if completion.get('raw_manifest') != manifest:
    raise AnalysisError('Consumed v3.3.2 completion/raw linkage changed.')
  compiler = _read_json(
      _V3_3_2_RUN_DIR / 'compiler/eight_row/COMPILER_PROVENANCE.json',
      'v3.3.2 COMPILER_PROVENANCE',
  )
  binding = {
      'attempt_started_sha256': _V3_3_2_RUN_FILES['ATTEMPT_STARTED.json'],
      'run_complete_sha256': _V3_3_2_RUN_FILES['RUN_COMPLETE.json'],
      'raw_manifest_sha256': _V3_3_2_RUN_FILES['RAW_MANIFEST.json'],
      'import_provenance_sha256': _V3_3_2_RUN_FILES['IMPORT_PROVENANCE.json'],
      'protobuf_provenance_sha256': _V3_3_2_RUN_FILES['PROTOBUF_PROVENANCE.json'],
      'compiler_provenance_sha256': _V3_3_2_RUN_FILES[
          'compiler/eight_row/COMPILER_PROVENANCE.json'
      ],
      'whole_run_file_count': 11,
      'whole_run_tree_sha256': V3_3_2_RUN_TREE_SHA256,
      'compiler_file_count': 4,
      'compiler_tree_sha256': V3_3_2_COMPILER_TREE_SHA256,
  }
  return {
      'path': str(_V3_3_2_RUN_DIR.resolve()),
      'model_run_commit': '24e2214168eeca41d4f3b60b62094b6befcadcc1',
      'freeze_sha256': 'baa257dfa8d0bd1afca40b11038d71c9157499eeb654523ad59df119eb2c0295',
      **binding,
      'status_predicates': predicates,
      'empty_raw_manifest': manifest,
      'eight_row_compiler': compiler,
  }


def _validate_v3_3_2_1_failure() -> dict[str, Any]:
  sources = _validate_source_bundle(
      _V3_3_2_1_SOURCES, implementation_commit=_V3_3_2_1_SOURCE_COMMIT
  )
  attempt = _validate_bound_tree(
      _V3_3_2_1_ATTEMPT_DIR, _V3_3_2_1_ATTEMPT_FILES,
      V3_3_2_1_ATTEMPT_TREE_SHA256, 'failed v3.3.2.1 attempt',
  )
  if _V3_3_2_1_ANALYSIS_DIR.exists() or _V3_3_2_1_ANALYSIS_DIR.is_symlink():
    raise AnalysisError('Failed v3.3.2.1 unexpectedly has an analysis output.')
  failure = _read_json(
      _V3_3_2_1_ATTEMPT_DIR / 'ANALYSIS_FAILURE.json',
      'v3.3.2.1 ANALYSIS_FAILURE',
  )
  error = failure.get('failure')
  if (
      not isinstance(error, Mapping)
      or error.get('type') != 'RecursionError'
      or error.get('message') != 'maximum recursion depth exceeded'
      or 'RecursionError: maximum recursion depth exceeded' not in str(
          error.get('traceback', '')
      )
      or failure.get('status') != 'failed_consumed_no_retry'
      or failure.get('model_apply_count') != 0
      or failure.get('analysis_dir_exists') is not False
      or failure.get('scientific_summary_computed') is not False
      or failure.get('shapley_or_nomination_computed') is not False
      or failure.get('combined_analysis_permitted') is not False
  ):
    raise AnalysisError('Failed v3.3.2.1 terminal predicates changed.')
  del sources, attempt
  return {
      'implementation_commit': _V3_3_2_1_SOURCE_COMMIT,
      'amendment_sha256': _V3_3_2_1_SOURCES[next(
          key for key in _V3_3_2_1_SOURCES if 'amendment_' in key
      )],
      'analyzer_sha256': _V3_3_2_1_SOURCES[next(
          key for key in _V3_3_2_1_SOURCES if key.endswith('_v3_3_2_1.py')
      )],
      'analyzer_test_sha256': _V3_3_2_1_SOURCES[next(
          key for key in _V3_3_2_1_SOURCES if key.endswith('_v3_3_2_1_test.py')
      )],
      'freeze_sha256': _V3_3_2_1_SOURCES[next(
          key for key in _V3_3_2_1_SOURCES if key.endswith('_freeze.json')
      )],
      'wrapper_sha256': _V3_3_2_1_SOURCES[next(
          key for key in _V3_3_2_1_SOURCES if key.endswith('.sh')
      )],
      'attempt_dir': str(_V3_3_2_1_ATTEMPT_DIR.resolve()),
      'analysis_dir': str(_V3_3_2_1_ANALYSIS_DIR.resolve()),
      'attempt_files': copy.deepcopy(_V3_3_2_1_ATTEMPT_FILES),
      'attempt_file_count': 2,
      'attempt_tree_sha256': V3_3_2_1_ATTEMPT_TREE_SHA256,
      'state': 'failed_consumed_no_retry',
      'error_type': 'RecursionError',
      'model_apply_count': 0,
      'scientific_summary_computed': False,
      'shapley_or_nomination_computed': False,
      'combined_analysis_permitted': False,
      'analysis_dir_absent': True,
  }


def _validate_v3_3_2_2_archive() -> dict[str, Any]:
  sources = _validate_source_bundle(
      _V3_3_2_2_SOURCES,
      implementation_commit=_V3_3_2_2_IMPLEMENTATION_COMMIT,
      amendment_commit=_V3_3_2_2_AMENDMENT_COMMIT,
  )
  attempt = _validate_bound_tree(
      _V3_3_2_2_ATTEMPT_DIR, _V3_3_2_2_ATTEMPT_FILES,
      V3_3_2_2_ATTEMPT_TREE_SHA256, 'completed v3.3.2.2 attempt',
  )
  output = _validate_bound_tree(
      _V3_3_2_2_ANALYSIS_DIR, _V3_3_2_2_OUTPUT_FILES,
      V3_3_2_2_OUTPUT_TREE_SHA256, 'completed v3.3.2.2 output',
  )
  for root, files in (
      (_V3_3_2_2_ATTEMPT_DIR, _V3_3_2_2_ATTEMPT_FILES),
      (_V3_3_2_2_ANALYSIS_DIR, _V3_3_2_2_OUTPUT_FILES),
  ):
    for name, binding in files.items():
      relative = str((root / name).relative_to(_REPO_ROOT))
      if _git_blob_sha256(_V3_3_2_2_ARCHIVE_COMMIT, relative) != binding['sha256']:
        raise AnalysisError(f'Archived v3.3.2.2 blob changed: {relative}.')
  complete = _read_json(
      _V3_3_2_2_ATTEMPT_DIR / 'ANALYSIS_COMPLETE.json',
      'v3.3.2.2 ANALYSIS_COMPLETE',
  )
  analysis = _read_json(
      _V3_3_2_2_ANALYSIS_DIR / 'ANALYSIS.json', 'v3.3.2.2 ANALYSIS'
  )
  predicates = {
      'decision': 'controlled_stop_compiler_graph_mismatch',
      'scientific_summary_computed': False,
      'shapley_or_nomination_computed': False,
      'nomination_performed': False,
      'combined_analysis_permitted': False,
  }
  if {key: analysis.get(key) for key in predicates} != predicates:
    raise AnalysisError('Completed v3.3.2.2 no-science predicates changed.')
  if (
      complete.get('analysis_json_sha256')
      != _V3_3_2_2_OUTPUT_FILES['ANALYSIS.json']['sha256']
      or complete.get('analysis_markdown_sha256')
      != _V3_3_2_2_OUTPUT_FILES['RESULT.md']['sha256']
      or complete.get('analysis_tree_sha256') != V3_3_2_2_OUTPUT_TREE_SHA256
      or complete.get('analysis_file_count') != 2
      or complete.get('model_apply_count') != 0
      or complete.get('scientific_summary_computed') is not False
      or complete.get('shapley_or_nomination_computed') is not False
      or complete.get('combined_analysis_permitted') is not False
  ):
    raise AnalysisError('Completed v3.3.2.2 attempt/output linkage changed.')
  del sources, attempt, output
  return {
      'amendment_commit': _V3_3_2_2_AMENDMENT_COMMIT,
      'implementation_commit': _V3_3_2_2_IMPLEMENTATION_COMMIT,
      'archive_commit': _V3_3_2_2_ARCHIVE_COMMIT,
      'amendment_sha256': _V3_3_2_2_SOURCES[next(
          key for key in _V3_3_2_2_SOURCES if 'amendment_' in key
      )],
      'analyzer_sha256': _V3_3_2_2_SOURCES[next(
          key for key in _V3_3_2_2_SOURCES if key.endswith('_v3_3_2_2.py')
      )],
      'analyzer_test_sha256': _V3_3_2_2_SOURCES[next(
          key for key in _V3_3_2_2_SOURCES if key.endswith('_v3_3_2_2_test.py')
      )],
      'freeze_sha256': _V3_3_2_2_SOURCES[next(
          key for key in _V3_3_2_2_SOURCES if key.endswith('_freeze.json')
      )],
      'wrapper_sha256': _V3_3_2_2_SOURCES[next(
          key for key in _V3_3_2_2_SOURCES if key.endswith('.sh')
      )],
      'attempt_dir': str(_V3_3_2_2_ATTEMPT_DIR.resolve()),
      'analysis_dir': str(_V3_3_2_2_ANALYSIS_DIR.resolve()),
      'attempt_files': copy.deepcopy(_V3_3_2_2_ATTEMPT_FILES),
      'attempt_file_count': 2,
      'attempt_tree_sha256': V3_3_2_2_ATTEMPT_TREE_SHA256,
      'analysis_files': copy.deepcopy(_V3_3_2_2_OUTPUT_FILES),
      'analysis_file_count': 2,
      'analysis_tree_sha256': V3_3_2_2_OUTPUT_TREE_SHA256,
      'state': 'complete_controlled_stop_audited',
      'decision': predicates['decision'],
      'scientific_summary_computed': False,
      'shapley_or_nomination_computed': False,
      'combined_analysis_permitted': False,
      'model_apply_count': 0,
  }


def _canonical_json_sha256(value: Any) -> str:
  encoded = json.dumps(
      value, sort_keys=True, separators=(',', ':'), ensure_ascii=False
  ).encode('utf-8')
  return hashlib.sha256(encoded).hexdigest()


def _normalized_entry_abi(compiled_hlo: str) -> tuple[str, str]:
  """Returns the single allowed first-line normalization and its digest."""
  first_line = compiled_hlo.splitlines()[0] if compiled_hlo else ''
  marker = 'fingerprint_before_lhs="'
  start = first_line.find(marker)
  if start < 0:
    raise AnalysisError('Compiled HLO first line lacks fingerprint_before_lhs.')
  value_start = start + len(marker)
  value_end = first_line.find('"', value_start)
  if value_end < 0:
    raise AnalysisError('Compiled HLO entry fingerprint is unterminated.')
  fingerprint = first_line[value_start:value_end]
  if (
      not fingerprint
      or any(character not in '0123456789abcdefABCDEF' for character in fingerprint)
  ):
    raise AnalysisError('Compiled HLO entry fingerprint is not hexadecimal.')
  normalized = (
      first_line[:value_start] + '<backend-generated>' + first_line[value_end:]
  )
  return normalized, hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def _validate_source_program_gate(
    compiler: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]],
    *, expected_signatures: Mapping[str, Any],
    expected_source_input_audit: Mapping[str, Any],
) -> dict[str, Any]:
  signatures = compiler.get('program_signatures')
  if not isinstance(signatures, Mapping):
    raise AnalysisError('Compiler program signatures are absent.')
  signature_sha = _canonical_json_sha256(signatures)
  if compiler.get('program_signatures_sha256') != signature_sha:
    raise AnalysisError('Compiler program-signature linkage changed.')

  compiled_path = Path(str(artifacts['compiled_hlo']['path'])).resolve()
  try:
    compiled_text = compiled_path.read_text(encoding='utf-8')
  except (OSError, UnicodeDecodeError) as error:
    raise AnalysisError('Compiled HLO cannot be read for the entry-ABI gate.') from error
  normalized_line, entry_sha = _normalized_entry_abi(compiled_text)
  entry = _exact_keys(
      compiler.get('entry_abi'),
      {
          'normalization', 'normalized_line_sha256',
          'normalized_line_size_bytes',
          'backend_fingerprint_substitution_count',
      },
      'compiler.entry_abi',
  )
  if (
      entry.get('normalized_line_sha256') != entry_sha
      or entry.get('normalized_line_size_bytes')
      != len(normalized_line.encode('utf-8'))
      or entry.get('backend_fingerprint_substitution_count') != 1
      or entry.get('normalization') != (
          'first HloModule line; replace only fingerprint_before_lhs value; '
          'omit line-ending newline'
      )
  ):
    raise AnalysisError('Compiler entry-ABI record changed.')

  observed = {
      'stablehlo_sha256': artifacts['stablehlo']['sha256'],
      'stablehlo_size_bytes': artifacts['stablehlo']['size_bytes'],
      'pre_backend_hlo_sha256': artifacts['hlo']['sha256'],
      'pre_backend_hlo_size_bytes': artifacts['hlo']['size_bytes'],
      'program_signatures_sha256': signature_sha,
      'entry_abi_sha256': entry_sha,
  }
  contract = {
      'stablehlo_sha256': SOURCE_STABLEHLO['sha256'],
      'stablehlo_size_bytes': SOURCE_STABLEHLO['size_bytes'],
      'pre_backend_hlo_sha256': SOURCE_PRE_BACKEND_HLO['sha256'],
      'pre_backend_hlo_size_bytes': SOURCE_PRE_BACKEND_HLO['size_bytes'],
      'program_signatures_sha256': PROGRAM_SIGNATURES_SHA256,
      'entry_abi_sha256': ENTRY_ABI_SHA256,
  }
  gate = _exact_keys(
      compiler.get('source_program_gate'),
      {
          'contract', 'observed', 'stablehlo_exact',
          'pre_backend_hlo_exact', 'program_signatures_exact',
          'entry_abi_exact',
          'source_runtime_device_toolchain_checkpoint_reference_exact',
          'source_input_audit', 'same_lowered_compiled_object',
          'source_program_exact',
      },
      'compiler.source_program_gate',
  )
  expected_flags = {
      'stablehlo_exact': observed['stablehlo_sha256'] == contract['stablehlo_sha256']
      and observed['stablehlo_size_bytes'] == contract['stablehlo_size_bytes'],
      'pre_backend_hlo_exact': observed['pre_backend_hlo_sha256']
      == contract['pre_backend_hlo_sha256']
      and observed['pre_backend_hlo_size_bytes']
      == contract['pre_backend_hlo_size_bytes'],
      'program_signatures_exact': (
          signature_sha == PROGRAM_SIGNATURES_SHA256
          and dict(signatures) == dict(expected_signatures)
      ),
      'entry_abi_exact': entry_sha == ENTRY_ABI_SHA256,
  }
  if gate.get('source_input_audit') != dict(expected_source_input_audit):
    raise AnalysisError('Compiler source_input_audit differs from independent audits.')
  audit_bools = (
      'bootstrap_sources_and_prior_trees_exact',
      'tracked_head_and_frozen_inventory_exact',
      'external_device_runtime_environment_exact',
      'same_process_device_runtime_environment_exact',
      'checkpoint_exact', 'reference_object_and_sequences_exact',
      'protobuf_binding_exact', 'three_import_inventories_stable_exact',
  )
  independently_provenance_exact = all(
      expected_source_input_audit.get(name) is True for name in audit_bools
  )
  for key, expected in (
      ('contract', contract), ('observed', observed), *expected_flags.items()
  ):
    if gate.get(key) != expected:
      raise AnalysisError(f'Compiler source-program gate changed at {key}.')
  if gate.get(
      'source_runtime_device_toolchain_checkpoint_reference_exact'
  ) is not independently_provenance_exact:
    raise AnalysisError('Compiler source/input/runtime aggregate is incorrect.')
  if gate.get('same_lowered_compiled_object') is not True:
    raise AnalysisError('Compiler did not bind one lowered/compiled object.')
  expected_exact = all(expected_flags.values())
  expected_exact = expected_exact and bool(
      independently_provenance_exact
      and gate['same_lowered_compiled_object']
  )
  if gate.get('source_program_exact') is not expected_exact:
    raise AnalysisError('Compiler source_program_exact aggregate is incorrect.')
  return {
      'source_program_exact': expected_exact,
      'observed': observed,
      'entry_abi': dict(entry),
  }


def _backend_config_from_instruction(line: str) -> dict[str, Any] | None:
  marker = 'backend_config='
  start = line.find(marker)
  if start < 0:
    return None
  start += len(marker)
  if start >= len(line) or line[start] != '{':
    raise AnalysisError('Backend config is not a JSON object.')
  depth = 0
  in_string = False
  escaped = False
  for index in range(start, len(line)):
    character = line[index]
    if in_string:
      if escaped:
        escaped = False
      elif character == '\\':
        escaped = True
      elif character == '"':
        in_string = False
      continue
    if character == '"':
      in_string = True
    elif character == '{':
      depth += 1
    elif character == '}':
      depth -= 1
      if depth == 0:
        try:
          value = json.loads(line[start:index + 1])
        except json.JSONDecodeError as error:
          raise AnalysisError('Backend config JSON is malformed.') from error
        if not isinstance(value, dict):
          raise AnalysisError('Backend config decoded to a non-object.')
        return value
  raise AnalysisError('Backend config JSON object is unterminated.')


def _recompute_backend_diagnostics(compiled_hlo: str) -> dict[str, Any]:
  """Mirrors the frozen descriptive summary without making it a gate."""
  from collections import Counter  # Local stdlib import keeps module lean.

  lines = compiled_hlo.splitlines()
  computation_count = sum(
      bool(re.match(r'^(?:ENTRY )?%[^ ]+ \(', line)) for line in lines
  )
  instruction_count = sum(line.startswith('  %') for line in lines)
  fusion_kinds = Counter(re.findall(r'kind=(k[A-Za-z_]+)', compiled_hlo))
  triton, cublas, cudnn = [], [], []
  for line in lines:
    backend = _backend_config_from_instruction(line)
    if '"kind":"__triton"' in line:
      block = (
          (backend or {}).get('fusion_backend_config', {})
          .get('block_level_fusion_config')
      )
      if not isinstance(block, dict):
        raise AnalysisError('Triton instruction lacks block-level settings.')
      triton.append({
          'block_level_fusion_config': block,
          'instruction_line_sha256': hashlib.sha256(
              line.encode('utf-8')
          ).hexdigest(),
      })
    if 'custom_call_target="__cublas$' in line:
      target = re.search(r'custom_call_target="([^"]+)"', line)
      gemm = (backend or {}).get('gemm_backend_config')
      cublas.append({
          'target': target.group(1) if target else None,
          'gemm_backend_config': gemm,
          'instruction_line_sha256': hashlib.sha256(
              line.encode('utf-8')
          ).hexdigest(),
      })
    if 'custom_call_target="__cudnn$' in line:
      target = re.search(r'custom_call_target="([^"]+)"', line)
      convolution = (backend or {}).get('cudnn_conv_backend_config', {})
      algorithm = convolution.get('algorithm')
      cudnn.append({
          'target': target.group(1) if target else None,
          'algorithm': algorithm,
          'workspace_size_bytes': (
              None if not isinstance(algorithm, dict)
              else int(algorithm.get('workspace_size', 0))
          ),
          'instruction_line_sha256': hashlib.sha256(
              line.encode('utf-8')
          ).hexdigest(),
      })
  return {
      'descriptive_only_not_an_equality_gate': True,
      'computation_count': computation_count,
      'instruction_count_excluding_computation_headers': instruction_count,
      'instruction_record_count': computation_count + instruction_count,
      'fusion_kind_counts': dict(sorted(fusion_kinds.items())),
      'triton_configuration_count': len(triton),
      'triton_configurations': triton,
      'cublas_call_count': len(cublas),
      'cublas_algorithms': cublas,
      'cudnn_call_count': len(cudnn),
      'cudnn_algorithms_workspaces': cudnn,
  }


def _execution_order() -> tuple[tuple[int, int], ...]:
  return tuple(
      (order, anchor)
      for order in RECIPIENT_ORDERS
      for anchor in ANCHOR_IDS
  )


def _artifact_relative(case: Mapping[str, Any], anchor: int) -> str:
  return _v332._artifact_relative(case, anchor)  # pylint: disable=protected-access


def _donor_order(order: int) -> int:
  return _v332._donor_order(order)  # pylint: disable=protected-access


def _validate_record(
    record: Mapping[str, Any], *, case: Mapping[str, Any],
    donor_case: Mapping[str, Any], anchor: int, execution_index: int,
    freeze_sha256: str, executable_fingerprint: str,
    original_manifest: Mapping[str, str], sequence_bindings: Mapping[int, Any],
    allow_invalid: bool,
) -> dict[str, Any]:
  """Reuses frozen v3.3.2 row semantics after version-only normalization."""
  normalized = copy.deepcopy(dict(record))
  expected_version_fields = {
      'family': 'v3_3_3_unrelated_donor_sidecar_anchor',
      'script_version': SCRIPT_VERSION,
      'amendment_sha256': AMENDMENT_SHA256,
      'amendment_commit': AMENDMENT_COMMIT,
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
  }
  for key, expected in expected_version_fields.items():
    if normalized.get(key) != expected:
      raise AnalysisError(
          f'order={case["order"]},anchor={anchor}.{key} changed.'
      )
  normalized.update({
      'family': 'v3_3_2_unrelated_donor_sidecar_anchor',
      'script_version': _v332.SCRIPT_VERSION,
      'amendment_sha256': _v332.AMENDMENT_SHA256,
      'amendment_commit': _v332.AMENDMENT_COMMIT,
  })
  try:
    result = _v332._validate_record(  # pylint: disable=protected-access
        normalized, case=case, donor_case=donor_case, anchor=anchor,
        execution_index=execution_index, freeze_sha256=freeze_sha256,
        executable_fingerprint=executable_fingerprint,
        original_manifest=original_manifest,
        sequence_bindings=sequence_bindings, allow_invalid=allow_invalid,
    )
  except _v332.AnalysisError as error:
    raise AnalysisError(str(error)) from error
  return dict(result)


def _artifact_comparison(
    artifacts: Mapping[str, Any], compiler: Mapping[str, Any]
) -> dict[str, Any]:
  return {
      name: {
          'sha256_exact': artifacts[name]['sha256']
          == compiler['artifacts'][name]['sha256'],
          'size_exact': artifacts[name]['size_bytes']
          == compiler['artifacts'][name]['size_bytes'],
      }
      for name in ('stablehlo', 'hlo', 'compiled_hlo')
  }


def _cache_tree_binding(root: Path, label: str) -> dict[str, Any]:
  """Recomputes the runner's diagnostic-only kernel-cache tree binding."""
  _guard_path(root)
  if root.is_symlink() or not root.is_dir():
    raise AnalysisError(f'{label} root is absent or unsafe.')
  root = root.resolve()
  files: list[Path] = []
  directories: list[Path] = []
  pending = [root]
  while pending:
    directory = pending.pop()
    directories.append(directory)
    for entry in sorted(directory.iterdir()):
      mode = entry.lstat().st_mode
      if stat.S_ISLNK(mode):
        raise AnalysisError(f'{label} contains a symlink.')
      if stat.S_ISDIR(mode):
        pending.append(entry)
      elif stat.S_ISREG(mode):
        files.append(entry)
      else:
        raise AnalysisError(f'{label} contains a special entry.')
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
      'root': str(root), 'file_count': len(files),
      'directory_count': len(directories), 'files': records,
      'tree_sha256': digest.hexdigest(),
      'diagnostic_outputs_only_no_cache_input': True,
  }


def _validate_kernel_cache_provenance(
    value: Any, *, expected_preimport: Mapping[str, Any], phase: str,
) -> dict[str, Any]:
  if phase not in {'post_compile', 'post_failure'}:
    raise AnalysisError('Unknown kernel-cache provenance phase.')
  record = _exact_keys(
      value,
      {
          'pre_import', phase, 'default_user_cache_paths_eligible',
          'cache_outputs_are_diagnostic_only',
      },
      'compiler.kernel_cache_provenance',
  )
  if (
      record.get('pre_import') != dict(expected_preimport)
      or record.get('default_user_cache_paths_eligible') is not False
      or record.get('cache_outputs_are_diagnostic_only') is not True
  ):
    raise AnalysisError('Kernel-cache input/output semantics changed.')
  root = Path(str(expected_preimport.get('cache_root'))).resolve()
  historical = _exact_keys(record.get(phase), {
      'root', 'file_count', 'directory_count', 'files', 'tree_sha256',
      'diagnostic_outputs_only_no_cache_input',
  }, f'compiler.kernel_cache_provenance.{phase}')
  files = historical.get('files')
  if (
      historical.get('root') != str(root)
      or isinstance(historical.get('file_count'), bool)
      or not isinstance(historical.get('file_count'), int)
      or historical['file_count'] < 0
      or isinstance(historical.get('directory_count'), bool)
      or not isinstance(historical.get('directory_count'), int)
      or historical['directory_count'] < 1
      or not isinstance(files, Mapping)
      or historical['file_count'] != len(files)
      or not _is_sha256(historical.get('tree_sha256'))
      or historical.get('diagnostic_outputs_only_no_cache_input') is not True
  ):
    raise AnalysisError('Historical kernel-cache binding is malformed.')
  for relative, binding in files.items():
    if (
        not isinstance(relative, str) or not relative
        or Path(relative).is_absolute() or '..' in Path(relative).parts
    ):
      raise AnalysisError('Historical kernel-cache path is malformed.')
    row = _exact_keys(
        binding, {'sha256', 'size_bytes'},
        f'historical kernel-cache file {relative}',
    )
    if (
        not _is_sha256(row.get('sha256'))
        or isinstance(row.get('size_bytes'), bool)
        or not isinstance(row.get('size_bytes'), int)
        or row['size_bytes'] < 0
    ):
      raise AnalysisError('Historical kernel-cache file binding is malformed.')
  return {
      'pre_import_inputs_absent': True, 'post_phase': phase,
      'post_tree_sha256': historical['tree_sha256'],
      'historical_binding': copy.deepcopy(dict(historical)),
      'diagnostic_outputs_only': True,
  }


def _validate_final_model_cache(
    value: Any, freeze: Mapping[str, Any], *,
    compiler_cache_audit: Mapping[str, Any],
) -> dict[str, Any]:
  terminal = _cache_tree_binding(
      Path(str(freeze['model_kernel_cache_dir'])),
      'final model kernel-cache output tree',
  )
  record = _exact_keys(value, {
      'pre_import', 'historical_stage', 'historical_binding', 'terminal',
      'historical_to_terminal_tree_exact',
      'historical_to_terminal_equality_is_a_gate',
      'historical_snapshot_not_reauthenticated_as_live_files',
      'default_user_cache_paths_eligible',
      'cache_outputs_are_diagnostic_only',
  }, 'RUN_COMPLETE.model_kernel_cache_final')
  historical = compiler_cache_audit['historical_binding']
  expected_exact = historical['tree_sha256'] == terminal['tree_sha256']
  if (
      record.get('pre_import')
      != _expected_cache_environment(freeze, 'model')
      or record.get('historical_stage') != compiler_cache_audit['post_phase']
      or record.get('historical_binding') != historical
      or record.get('terminal') != terminal
      or record.get('historical_to_terminal_tree_exact') is not expected_exact
      or record.get('historical_to_terminal_equality_is_a_gate') is not False
      or record.get('historical_snapshot_not_reauthenticated_as_live_files')
      is not True
      or record.get('default_user_cache_paths_eligible') is not False
      or record.get('cache_outputs_are_diagnostic_only') is not True
  ):
    raise AnalysisError('Final model kernel-cache diagnostic binding changed.')
  return {
      'historical_stage': compiler_cache_audit['post_phase'],
      'historical_tree_sha256': historical['tree_sha256'],
      'terminal_tree_sha256': terminal['tree_sha256'],
      'historical_to_terminal_tree_exact': expected_exact,
      'equality_is_a_gate': False,
  }


def _validate_compiler(
    run_dir: Path, value: Any, *, original_v3_3: Mapping[str, Any],
    consumed_v3_3_2: Mapping[str, Any],
    expected_source_input_audit: Mapping[str, Any],
    expected_kernel_cache_preimport: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
  compiler = _exact_keys(
      value,
      {
          'executable_name', 'compile_count', 'compile_seconds',
          'lower_attempt_count', 'compile_attempt_count',
          'successful_compile_count',
          'executable_fingerprint', 'artifacts', 'program_signatures',
          'program_signatures_sha256', 'entry_abi', 'source_program_gate',
          'backend_diagnostics', 'diagnostic_comparisons',
          'kernel_cache_provenance',
      },
      'eight-row compiler',
  )
  if (
      compiler.get('executable_name') != 'eight_row'
      or compiler.get('compile_count') != 1
      or compiler.get('lower_attempt_count') != 1
      or compiler.get('compile_attempt_count') != 1
      or compiler.get('successful_compile_count') != 1
      or _finite(compiler.get('compile_seconds'), 'compiler.compile_seconds') < 0
  ):
    raise AnalysisError('Eight-row compiler identity/count/timing changed.')
  artifacts = _exact_keys(
      compiler.get('artifacts'), {'stablehlo', 'hlo', 'compiled_hlo'},
      'compiler.artifacts',
  )
  filenames = {
      'stablehlo': 'graph.stablehlo.mlir',
      'hlo': 'graph.pre_backend.hlo.txt',
      'compiled_hlo': 'graph.compiled.hlo.txt',
  }
  for name, filename in filenames.items():
    binding = _exact_keys(
        artifacts[name], {'path', 'sha256', 'size_bytes'},
        f'compiler.artifacts.{name}',
    )
    path = (run_dir / 'compiler/eight_row' / filename).resolve()
    if binding.get('path') != str(path):
      raise AnalysisError(f'Compiler {name} path changed.')
    _strict_regular(path, f'compiler {name}')
    if (
        not _is_sha256(binding.get('sha256'))
        or _sha256(path) != binding['sha256']
        or path.stat().st_size != binding.get('size_bytes')
    ):
      raise AnalysisError(f'Compiler {name} hash/size changed.')
  compiler_path = run_dir / 'compiler/eight_row/COMPILER_PROVENANCE.json'
  expected_tree = {compiler_path.resolve()} | {
      (run_dir / 'compiler/eight_row' / filename).resolve()
      for filename in filenames.values()
  }
  observed_tree = set(_strict_tree(
      run_dir / 'compiler',
      {
          'eight_row/COMPILER_PROVENANCE.json',
          *(f'eight_row/{filename}' for filename in filenames.values()),
      },
      'compiler tree',
  ))
  if observed_tree != expected_tree:
    raise AnalysisError('Compiler tree membership changed.')
  persisted = _read_json(compiler_path, 'COMPILER_PROVENANCE')
  if persisted != dict(compiler):
    raise AnalysisError('Embedded compiler record differs from current bytes.')

  fingerprint = hashlib.sha256(
      bytes.fromhex(artifacts['compiled_hlo']['sha256'])
  ).hexdigest()
  if compiler.get('executable_fingerprint') != fingerprint:
    raise AnalysisError('Compiler executable fingerprint changed.')
  expected_signatures = consumed_v3_3_2.get('program_signatures')
  if not isinstance(expected_signatures, Mapping):
    raise AnalysisError('Consumed v3.3.2 program-signature literal is absent.')
  source_gate = _validate_source_program_gate(
      compiler, artifacts, expected_signatures=expected_signatures,
      expected_source_input_audit=expected_source_input_audit,
  )
  cache_audit = _validate_kernel_cache_provenance(
      compiler.get('kernel_cache_provenance'),
      expected_preimport=expected_kernel_cache_preimport,
      phase='post_compile',
  )
  compiled_text = Path(artifacts['compiled_hlo']['path']).read_text(
      encoding='utf-8'
  )
  diagnostics = _recompute_backend_diagnostics(compiled_text)
  if compiler.get('backend_diagnostics') != diagnostics:
    raise AnalysisError('Compiled-backend descriptive diagnostics changed.')

  comparisons = _exact_keys(
      compiler.get('diagnostic_comparisons'),
      {
          'v3_3', 'v3_3_2',
          'compiled_backend_differences_are_diagnostic_only',
      },
      'compiler.diagnostic_comparisons',
  )
  for label, expected in (
      ('v3_3', original_v3_3), ('v3_3_2', consumed_v3_3_2)
  ):
    row = _exact_keys(
        comparisons.get(label), {'artifacts', 'executable_fingerprint_exact'},
        f'compiler.diagnostic_comparisons.{label}',
    )
    computed = _artifact_comparison(artifacts, expected)
    if (
        row.get('artifacts') != computed
        or row.get('executable_fingerprint_exact') is not (
            fingerprint == expected.get('executable_fingerprint')
        )
    ):
      raise AnalysisError(f'Compiler diagnostic comparison changed: {label}.')
  if comparisons.get('compiled_backend_differences_are_diagnostic_only') is not True:
    raise AnalysisError('Compiled backend was incorrectly promoted to an equality gate.')
  return fingerprint, {
      **source_gate,
      'executable_fingerprint': fingerprint,
      'backend_diagnostics': diagnostics,
      'diagnostic_comparisons': copy.deepcopy(dict(comparisons)),
      'kernel_cache_provenance': cache_audit,
      'compiled_backend_equality_gate': False,
  }


_COMPLETION_KEYS = {
    'status', 'stop_reason', 'message', 'attempt_id', 'script_version',
    'amendment_sha256', 'amendment_commit', 'original_protocol_sha256',
    'freeze_sha256', 'ood_anchor_record_count', 'ood_invalid_count',
    'unique_recipient_anchor_count', 'all_80_recipient_anchors_complete',
    'model_apply_count', 'expected_model_apply_count',
    'eight_row_compile_count', 'eight_row_compile_attempt_count',
    'eight_row_successful_compile_count', 'six_row_compile_count',
    'identity_rerun_count', 'main_cube_rerun_count',
    'old_ood_records_reused', 'one_fixed_eight_row_executable',
    'eight_row_compiler', 'eight_row_executable_fingerprint',
    'source_program_gate', 'source_program_exact',
    'compiled_backend_diagnostic_only', 'backend_diagnostics',
    'diagnostic_comparisons', 'id0_all20', 'id255_all20',
    'invariant_rows_between_calls',
    'active_rows_have_no_forced_cross_call_predicate',
    'original_run_binding', 'original_run_revalidated_in_full',
    'original_ood_records_provenance_only', 'v3_3_1_status',
    'v3_3_2_run_binding', 'v3_3_2_1_failure_status',
    'v3_3_2_2_archive_status', 'import_provenance_phases',
    'import_provenance_sha256', 'protobuf_provenance_sha256',
    'raw_manifest', 'confirmation_model_calls',
    'confirmation_scope_disclosure', 'scientific_summary_computed',
    'shapley_or_nomination_computed', 'interaction_or_resolution_computed',
    'combined_analysis_permitted', 'completed_at_unix_s',
    'model_kernel_cache_final',
}


def _completion_prefix(
    completion: Mapping[str, Any], *, freeze_sha256: str,
    original_run: Mapping[str, Any], v3_3_1_status: Mapping[str, Any],
    v3_3_2_run: Mapping[str, Any], v3_3_2_1_failure: Mapping[str, Any],
    v3_3_2_2_archive: Mapping[str, Any],
) -> tuple[tuple[tuple[int, int], ...], bool]:
  _exact_keys(completion, _COMPLETION_KEYS, 'RUN_COMPLETE')
  common = {
      'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION,
      'amendment_sha256': AMENDMENT_SHA256,
      'amendment_commit': AMENDMENT_COMMIT,
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'freeze_sha256': freeze_sha256,
      'expected_model_apply_count': EXPECTED_APPLY_COUNT,
      'eight_row_compile_count': 1,
      'eight_row_compile_attempt_count': 1,
      'eight_row_successful_compile_count': 1,
      'six_row_compile_count': 0,
      'identity_rerun_count': 0,
      'main_cube_rerun_count': 0,
      'old_ood_records_reused': 0,
      'one_fixed_eight_row_executable': True,
      'compiled_backend_diagnostic_only': True,
      'invariant_rows_between_calls': list(INVARIANT_ROWS),
      'active_rows_have_no_forced_cross_call_predicate': True,
      'original_run_binding': dict(original_run),
      'original_run_revalidated_in_full': True,
      'original_ood_records_provenance_only': True,
      'v3_3_1_status': dict(v3_3_1_status),
      'v3_3_2_run_binding': dict(v3_3_2_run),
      'v3_3_2_1_failure_status': dict(v3_3_2_1_failure),
      'v3_3_2_2_archive_status': dict(v3_3_2_2_archive),
      'confirmation_model_calls': 0,
      'confirmation_scope_disclosure': CONFIRMATION_DISCLOSURE,
      'scientific_summary_computed': False,
      'shapley_or_nomination_computed': False,
      'interaction_or_resolution_computed': False,
      'combined_analysis_permitted': False,
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
    raise AnalysisError('RUN_COMPLETE prefix/count/apply arithmetic changed.')
  source_exact = completion.get('source_program_exact')
  if not isinstance(source_exact, bool):
    raise AnalysisError('RUN_COMPLETE.source_program_exact is not boolean.')
  status, reason = completion.get('status'), completion.get('stop_reason')
  full = status == 'complete' and reason is None
  if full:
    valid = (
        count == 80 and completion['ood_invalid_count'] == 0
        and completion.get('all_80_recipient_anchors_complete') is True
        and completion.get('id0_all20') is True
        and completion.get('id255_all20') is True
        and source_exact
        and completion.get('message')
        == 'All 80 frozen v3.3.3 OOD sidecar records completed.'
    )
  elif status == 'controlled_stop' and reason == 'source_program_mismatch':
    valid = (
        count == completion['ood_invalid_count'] == 0
        and completion.get('all_80_recipient_anchors_complete') is False
        and completion.get('id0_all20') is False
        and completion.get('id255_all20') is False
        and not source_exact
        and completion.get('message') == (
            'The mandatory StableHLO/pre-backend/signature/ABI/source '
            'program gate failed before model apply zero.'
        )
    )
  elif status == 'controlled_stop' and reason == 'ood_tooling_failure':
    if not (
        1 <= count <= 80 and completion['ood_invalid_count'] == 1
        and source_exact
    ):
      valid = False
    else:
      order, anchor = _execution_order()[count - 1]
      valid = (
          completion.get('message')
          == f'OOD sidecar audit failed at order={order}, anchor_id={anchor}.'
          and completion.get('all_80_recipient_anchors_complete') is False
      )
  else:
    valid = False
  if not valid:
    raise AnalysisError('RUN_COMPLETE status/count/message is not an allowed prefix.')
  if not isinstance(completion.get('id0_all20'), bool) or not isinstance(
      completion.get('id255_all20'), bool
  ):
    raise AnalysisError('RUN_COMPLETE ID closure flags are not booleans.')
  return _execution_order()[:count], full


def _sidecar_source_paths() -> tuple[Path, ...]:
  return (
      _HERE / 'run_encoder_skip_ood_sidecar_v3_3_3.py',
      _HERE / 'launch_encoder_skip_ood_sidecar_v3_3_3.py',
      _HERE / 'validate_encoder_skip_ood_sidecar_bootstrap_v3_3_3.py',
      _HERE / 'run_device_preflight_v3_3_3.py',
      _AMENDMENT_PATH, _FREEZE_PATH,
      _HERE / 'run_encoder_skip_factorial_v3_3.py',
  )


def _validate_import_file(
    path: Path, expected_sha256: Any, *, bundle_root: Path,
    freeze: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
  if not _is_sha256(expected_sha256) or _sha256(path) != expected_sha256:
    raise AnalysisError(f'Import-provenance hash mismatch: {path.name}.')
  value = _read_json(path, path.name)
  _exact_keys(value, {
      'module_count', 'modules', 'upstream_source_attestation',
      'v3_3_3_sidecar_sources',
  }, path.name)
  expected_sources = {
      str(source.resolve()): {
          'sha256': _sha256(source.resolve()),
          'size_bytes': source.resolve().stat().st_size,
      }
      for source in _sidecar_source_paths()
  }
  if len(expected_sources) != 7 or value.get(
      'v3_3_3_sidecar_sources'
  ) != expected_sources:
    raise AnalysisError(f'{path.name} seven-source binding changed.')
  inventory = freeze.get('upstream_imported_modules')
  exception = freeze.get('upstream_generated_binding_exception')
  upstream_head = freeze.get('upstream_alphagenome_git_head')
  if not isinstance(inventory, Mapping) or len(inventory) != 26:
    raise AnalysisError('Frozen upstream source inventory is incomplete.')
  upstream_root = (bundle_root.parent / 'alphagenome').resolve()
  generated_names = set(_v332._v33.UPSTREAM_GENERATED_MODULE_NAMES)  # pylint: disable=protected-access
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
    row = _exact_keys(
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
      key: names for key, names in by_path.items() if len(names) > 1
  }.items():
    if (
        set(names) != {'__main__', '__mp_main__'}
        or Path(duplicate_path).name
        != 'run_encoder_skip_ood_sidecar_v3_3_3.py'
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
  bindings = _exact_keys(
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
  try:
    return _v332._validate_protobuf(  # pylint: disable=protected-access
        run_dir, completion, freeze
    )
  except _v332.AnalysisError as error:
    raise AnalysisError(str(error)) from error


_START_KEYS = {
    'attempt_id', 'script_version', 'status', 'amendment',
    'original_protocol_sha256', 'freeze', 'freeze_sha256', 'bootstrap',
    'external_preflight', 'same_process_preflight', 'cache_role_transition',
    'runtime_environment',
    'runtime_version_binding', 'checkpoint_path', 'checkpoint_binding',
    'reference_object_binding', 'reference_sequence_bindings',
    'original_run_binding', 'original_run_revalidated_in_full',
    'v3_3_1_status', 'v3_3_2_run_binding',
    'v3_3_2_1_failure_status', 'v3_3_2_2_archive_status',
    'source_program_contract', 'program_signatures',
    'compiled_backend_equality_is_a_gate', 'record_count_contract',
    'model_apply_count_contract', 'compile_count_contract',
    'rerun_count_contract', 'execution_order_contract',
    'invariant_rows_between_calls',
    'active_recipient_rows_without_cross_call_predicate',
    'max_wall_time_seconds', 'max_output_bytes', 'confirmation_model_calls',
    'scientific_summary_computed', 'shapley_or_nomination_computed',
    'combined_analysis_permitted', 'confirmation_scope_disclosure',
    'started_at_unix_s',
}

_FREEZE_KEYS = set(_v332._FREEZE_KEYS) | {  # pylint: disable=protected-access
    'v3_3_2_freeze_path', 'v3_3_2_freeze_sha256', 'v3_3_2_run',
    'v3_3_2_1_failure_status', 'v3_3_2_2_archive_status',
    'source_program_contract', 'program_signatures',
    'compiled_backend_equality_is_a_gate',
    'preflight_kernel_cache_dir', 'model_kernel_cache_dir',
    'analysis_attempt_dir',
    'denied_cache_environment_names',
    'denied_cache_environment_prefixes',
    'cache_isolation_contract',
}


def _validate_freeze(
    run_dir: Path, *, bundle_root: Path,
) -> tuple[
    dict[str, Any], str, dict[str, Any], dict[str, str], dict[int, Any],
    dict[str, Any], dict[str, Any], dict[str, Any],
]:
  if run_dir.resolve() != _RUN_DIR.resolve() or bundle_root.resolve() != _REPO_ROOT:
    raise AnalysisError('v3.3.3 run/repository path changed.')
  if _ANALYSIS_DIR.exists() or _ANALYSIS_DIR.is_symlink():
    raise AnalysisError('v3.3.3 analysis destination is not fresh/append-only.')
  if _sha256(_AMENDMENT_PATH) != AMENDMENT_SHA256:
    raise AnalysisError('v3.3.3 amendment bytes changed.')
  freeze = _read_json(_FREEZE_PATH, 'v3.3.3 freeze')
  _exact_keys(freeze, _FREEZE_KEYS, 'v3.3.3 freeze')
  freeze_sha = _sha256(_FREEZE_PATH)
  v332_freeze = _read_json(_v332._FREEZE_PATH, 'v3.3.2 freeze')  # pylint: disable=protected-access
  v3_3_2_run = _validate_v3_3_2_run()
  v3_3_2_1_failure = _validate_v3_3_2_1_failure()
  v3_3_2_2_archive = _validate_v3_3_2_2_archive()
  expected_contract = {
      'stablehlo_sha256': SOURCE_STABLEHLO['sha256'],
      'stablehlo_size_bytes': SOURCE_STABLEHLO['size_bytes'],
      'pre_backend_hlo_sha256': SOURCE_PRE_BACKEND_HLO['sha256'],
      'pre_backend_hlo_size_bytes': SOURCE_PRE_BACKEND_HLO['size_bytes'],
      'program_signatures_sha256': PROGRAM_SIGNATURES_SHA256,
      'entry_abi_sha256': ENTRY_ABI_SHA256,
  }
  required = {
      'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION,
      'amendment_path': str(_AMENDMENT_PATH.resolve()),
      'amendment_sha256': AMENDMENT_SHA256,
      'amendment_commit': AMENDMENT_COMMIT,
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'original_protocol_path': str(_v332._ORIGINAL_PROTOCOL_PATH.resolve()),  # pylint: disable=protected-access
      'original_freeze_sha256': _v332.ORIGINAL_FREEZE_SHA256,
      'original_freeze_path': str(_v332._ORIGINAL_FREEZE_PATH.resolve()),  # pylint: disable=protected-access
      'v3_3_2_freeze_path': str(_v332._FREEZE_PATH.resolve()),  # pylint: disable=protected-access
      'v3_3_2_freeze_sha256': 'baa257dfa8d0bd1afca40b11038d71c9157499eeb654523ad59df119eb2c0295',
      'output_dir': str(_RUN_DIR.resolve()),
      'analysis_dir': str(_ANALYSIS_DIR.resolve()),
      'analysis_attempt_dir': str(_ANALYSIS_ATTEMPT_DIR.resolve()),
      'preflight_dir': str(_PREFLIGHT_DIR.resolve()),
      'preflight_script_version': 'opensplice-device-preflight-v3.3.3',
      'preflight_kernel_cache_dir': str(
          (_HERE / 'results/v3_3_3_preflight_kernel_cache').resolve()
      ),
      'model_kernel_cache_dir': str(
          (_HERE / 'results/v3_3_3_model_kernel_cache').resolve()
      ),
      'ood_anchor_ids': list(ANCHOR_IDS),
      'recipient_orders': list(RECIPIENT_ORDERS),
      'ood_record_count': 80, 'model_apply_count': 320,
      'eight_row_compile_count': 1, 'six_row_compile_count': 0,
      'identity_rerun_count': 0, 'main_cube_rerun_count': 0,
      'old_ood_records_reused': 0,
      'invariant_rows_between_calls': list(INVARIANT_ROWS),
      'max_wall_time_seconds': 7_200,
      'max_output_bytes': 1_073_741_824,
      'original_run': dict(_v332._ORIGINAL_BINDING),  # pylint: disable=protected-access
      'v3_3_2_run': v3_3_2_run,
      'v3_3_2_1_failure_status': v3_3_2_1_failure,
      'v3_3_2_2_archive_status': v3_3_2_2_archive,
      'source_program_contract': expected_contract,
      'program_signatures': v3_3_2_run['eight_row_compiler']['program_signatures'],
      'compiled_backend_equality_is_a_gate': False,
      'denied_cache_environment_names': [
          'XLA_FLAGS', 'JAX_COMPILATION_CACHE_DIR', 'CUDA_CACHE_PATH',
          'CUDA_CACHE_MAXSIZE', 'TRITON_DUMP_DIR', 'TRITON_OVERRIDE_DIR',
      ],
      'denied_cache_environment_prefixes': ['JAX_PERSISTENT_CACHE_'],
      'cache_isolation_contract': {
          'CUDA_CACHE_DISABLE': '1',
          'JAX_ENABLE_COMPILATION_CACHE': 'false',
          'external_preflight': {
              'cache_role': 'external_preflight',
              'cache_root': str(
                  (_HERE / 'results/v3_3_3_preflight_kernel_cache').resolve()
              ),
              'triton_cache_dir': str((
                  _HERE / 'results/v3_3_3_preflight_kernel_cache/triton'
              ).resolve()),
              'xdg_cache_home': str((
                  _HERE / 'results/v3_3_3_preflight_kernel_cache/xdg'
              ).resolve()),
          },
          'model': {
              'cache_role': 'model',
              'cache_root': str(
                  (_HERE / 'results/v3_3_3_model_kernel_cache').resolve()
              ),
              'triton_cache_dir': str((
                  _HERE / 'results/v3_3_3_model_kernel_cache/triton'
              ).resolve()),
              'xdg_cache_home': str((
                  _HERE / 'results/v3_3_3_model_kernel_cache/xdg'
              ).resolve()),
          },
          'pre_import_file_count': 0,
          'pre_import_tree_sha256': EMPTY_SHA256,
          'roots_distinct': True,
          'default_user_cache_paths_eligible': False,
          'cache_output_equality_is_a_gate': False,
          'postcompile_historical_snapshot_reauthenticated': False,
          'terminal_live_tree_rehashed': True,
      },
  }
  for key, expected in required.items():
    if freeze.get(key) != expected:
      raise AnalysisError(f'v3.3.3 freeze changed at {key}.')
  inherited = set(_v332._FREEZE_KEYS) - {  # pylint: disable=protected-access
      'analysis_dir', 'amendment_commit', 'amendment_path', 'amendment_sha256',
      'attempt_id', 'file_sha256', 'output_dir', 'preflight_dir',
      'preflight_script_version', 'script_version', 'original_run',
  }
  for key in inherited:
    if key not in required and freeze.get(key) != v332_freeze.get(key):
      raise AnalysisError(f'v3.3.3 inherited freeze binding changed at {key}.')
  inventory = freeze.get('file_sha256')
  if not isinstance(inventory, Mapping) or not inventory:
    raise AnalysisError('v3.3.3 freeze source inventory is absent.')
  required_files = {
      str(_AMENDMENT_PATH.relative_to(bundle_root)),
      str(Path(__file__).resolve().relative_to(bundle_root)),
      str(_TEST_PATH.relative_to(bundle_root)),
      'experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_3.py',
      'experiments/interpretability/opensplice/launch_encoder_skip_ood_sidecar_v3_3_3.py',
      'experiments/interpretability/opensplice/validate_encoder_skip_ood_sidecar_bootstrap_v3_3_3.py',
      'experiments/interpretability/opensplice/run_device_preflight_v3_3_3.py',
      'experiments/interpretability/opensplice/run_device_preflight_v3_3_3_test.py',
      'experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_3.sh',
      'experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_3_test.py',
      str(_V3_3_2_ANALYZER_PATH.relative_to(bundle_root)),
  }
  if not required_files.issubset(inventory):
    raise AnalysisError('v3.3.3 freeze misses required source files.')
  for relative, digest in inventory.items():
    if (
        not isinstance(relative, str) or Path(relative).is_absolute()
        or '..' in Path(relative).parts or not _is_sha256(digest)
    ):
      raise AnalysisError('v3.3.3 file inventory is malformed.')
    path = (bundle_root / relative).resolve()
    try:
      path.relative_to(bundle_root)
    except ValueError as error:
      raise AnalysisError('v3.3.3 source path escaped repository.') from error
    _strict_regular(path, f'v3.3.3 source {relative}')
    if _sha256(path) != digest:
      raise AnalysisError(f'v3.3.3 source bytes changed: {relative}.')
    subprocess.run(
        ('git', '-C', str(bundle_root), 'ls-files', '--error-unmatch', relative),
        check=True, capture_output=True,
    )
  freeze_relative = str(_FREEZE_PATH.relative_to(bundle_root))
  subprocess.run(
      ('git', '-C', str(bundle_root), 'ls-files', '--error-unmatch', freeze_relative),
      check=True, capture_output=True,
  )
  if subprocess.check_output(
      ('git', '-C', str(bundle_root), 'diff', '--binary', 'HEAD', '--')
  ):
    raise AnalysisError('v3.3.3 requires a globally tracked-clean HEAD.')
  original_audit, original_manifest, sequence_bindings = (
      _v332._validate_original_v3_3()  # pylint: disable=protected-access
  )
  return (
      dict(freeze), freeze_sha, original_audit, original_manifest,
      sequence_bindings, v3_3_2_run, v3_3_2_1_failure,
      v3_3_2_2_archive,
  )


def _validate_external_and_same_process(
    start: Mapping[str, Any], freeze: Mapping[str, Any], freeze_sha256: str,
) -> dict[str, Any]:
  same = start.get('same_process_preflight')
  try:
    _v332._v33._validate_device_observation(  # pylint: disable=protected-access
        same, 'same-process preflight'
    )
  except _v332._v33.AnalysisError as error:  # pylint: disable=protected-access
    raise AnalysisError(str(error)) from error
  bootstrap = start['bootstrap']
  if same.get('pid') != bootstrap.get('pid'):
    raise AnalysisError('Same-process preflight PID differs from bootstrap.')
  cache = same.get('v3_3_3_cache_environment')
  model_cache = bootstrap.get('sanitized_environment', {}).get(
      'cache_environment'
  )
  if cache != model_cache:
    raise AnalysisError('Same-process cache-environment attestation changed.')
  external = start.get('external_preflight')
  if not isinstance(external, Mapping):
    raise AnalysisError('External preflight binding is absent.')
  path_value, digest = external.get('path'), external.get('sha256')
  if not isinstance(path_value, str) or not _is_sha256(digest):
    raise AnalysisError('External preflight path/hash is malformed.')
  path = Path(path_value).resolve()
  _guard_path(path)
  if (
      path != (_PREFLIGHT_DIR / 'preflight_0000.json').resolve()
      or _sha256(path) != digest
  ):
    raise AnalysisError('External preflight path/hash changed.')
  raw = _read_json(path, 'external preflight')
  embedded = {
      key: value for key, value in external.items()
      if key not in {'path', 'sha256', 'validated_logs', 'directory_binding'}
  }
  if raw != embedded:
    raise AnalysisError('Embedded external preflight differs from artifact.')
  if (
      raw.get('script_version') != 'opensplice-device-preflight-v3.3.3'
      or raw.get('status') != 'pass'
      or raw.get('amendment_sha256') != AMENDMENT_SHA256
      or raw.get('original_protocol_sha256') != ORIGINAL_PROTOCOL_SHA256
      or raw.get('freeze_sha256') != freeze_sha256
      or raw.get('failure') is not None
      or raw.get('no_model_or_biological_access') is not True
      or raw.get('no_jit_or_array_kernel') is not True
  ):
    raise AnalysisError('External preflight contract did not pass exactly.')
  logs, validated = raw.get('logs'), external.get('validated_logs')
  if not isinstance(logs, Mapping) or set(logs) != {'stdout', 'stderr'}:
    raise AnalysisError('External preflight logs are incomplete.')
  expected_validated = {}
  for stream in ('stdout', 'stderr'):
    binding = _exact_keys(
        logs[stream], {'path', 'sha256'}, f'external preflight {stream}'
    )
    log_path = Path(str(binding['path'])).resolve()
    if log_path.parent != _PREFLIGHT_DIR.resolve() or _sha256(log_path) != binding['sha256']:
      raise AnalysisError(f'External preflight {stream} log changed.')
    expected_validated[stream] = {
        'path': str(log_path), 'sha256': binding['sha256']
    }
  if validated != expected_validated:
    raise AnalysisError('External preflight validated-log binding changed.')
  preflight_files = {
      '.allocation.lock', '.preflight_0000.reserved',
      'preflight_0000.json', 'preflight_0000.stdout.log',
      'preflight_0000.stderr.log',
  }
  preflight_paths = _strict_tree(
      _PREFLIGHT_DIR, preflight_files, 'external preflight append-only tree'
  )
  expected_directory_binding = {
      'path': str(_PREFLIGHT_DIR.resolve()),
      'file_count': 5,
      'file_sha256': {
          item.name: {
              'sha256': _sha256(item), 'size_bytes': item.stat().st_size,
          }
          for item in sorted(preflight_paths)
      },
      'tree_sha256': _tree_digest(preflight_paths, _PREFLIGHT_DIR.resolve()),
      'sole_preflight_attempt_exact': True,
  }
  if external.get('directory_binding') != expected_directory_binding:
    raise AnalysisError('External preflight directory binding changed.')
  for name in ('.allocation.lock', '.preflight_0000.reserved'):
    item = _PREFLIGHT_DIR / name
    if item.stat().st_size != 0 or _sha256(item) != EMPTY_SHA256:
      raise AnalysisError(f'External preflight control file changed: {name}.')
  if raw.get('preflight_attempt_number') != 0:
    raise AnalysisError('External preflight is not the sole attempt zero.')
  observation = raw.get('observation')
  try:
    _v332._v33._validate_device_observation(  # pylint: disable=protected-access
        observation, 'external preflight'
    )
  except _v332._v33.AnalysisError as error:  # pylint: disable=protected-access
    raise AnalysisError(str(error)) from error
  external_runtime = observation.get('v3_3_3_runtime_environment')
  expected_external_cache = _expected_cache_environment(
      freeze, 'external_preflight'
  )
  expected_live_cache = {
      'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
      'JAX_ENABLE_COMPILATION_CACHE': 'false',
      'CUDA_CACHE_DISABLE': '1',
      'cache_role': expected_external_cache['cache_role'],
      'cache_root': expected_external_cache['cache_root'],
      'triton_cache_dir': expected_external_cache['triton_cache_dir'],
      'xdg_cache_home': expected_external_cache['xdg_cache_home'],
      'present_forbidden_names': [],
      'exact_to_pre_import_routing': True,
  }
  if (
      not isinstance(external_runtime, Mapping)
      or external_runtime.get('cache_environment')
      != expected_external_cache
      or external_runtime.get('live_cache_environment')
      != expected_live_cache
  ):
    raise AnalysisError('External preflight cache-environment attestation changed.')
  expected_transition = {
      'external_preflight': expected_external_cache,
      'model': _expected_cache_environment(freeze, 'model'),
      'roles_and_roots_distinct': True,
      'shared_policy_exact': True,
      'default_user_cache_paths_eligible': False,
  }
  if start.get('cache_role_transition') != expected_transition:
    raise AnalysisError('External/model cache-role transition changed.')
  return {
      'external_preflight_sha256': digest,
      'external_preflight_file_count': 5,
      'external_preflight_tree_sha256': _tree_digest(
          preflight_paths, _PREFLIGHT_DIR
      ),
      'external_exact_rtx3090_uuid_gate': True,
      'same_process_exact_rtx3090_uuid_gate': True,
      'same_process_pid_bound_to_bootstrap': True,
      'cache_inputs_absent': True,
      'external_cache_tree': _audit_cache_tree(
          Path(str(freeze['preflight_kernel_cache_dir'])),
          'external-preflight cache tree',
      ),
      'model_cache_tree': _audit_cache_tree(
          Path(str(freeze['model_kernel_cache_dir'])), 'model cache tree'
      ),
  }


def _expected_cache_environment(
    freeze: Mapping[str, Any], role: str,
) -> dict[str, Any]:
  key = {
      'external_preflight': 'preflight_kernel_cache_dir',
      'model': 'model_kernel_cache_dir',
  }.get(role)
  if key is None:
    raise AnalysisError('Unknown frozen cache role.')
  root = Path(str(freeze.get(key))).resolve()
  return {
      'denied_exact_names': list(freeze['denied_cache_environment_names']),
      'denied_prefixes': list(freeze['denied_cache_environment_prefixes']),
      'present_forbidden_names': [],
      'autotune_load_dump_cache_inputs_absent': True,
      'kernel_cache_inputs_absent': True,
      'persistent_compilation_cache_inputs_absent': True,
      'cuda_kernel_cache_disabled': True,
      'cache_role': role,
      'cache_root': str(root),
      'triton_cache_dir': str(root / 'triton'),
      'xdg_cache_home': str(root / 'xdg'),
      'pre_import_file_count': 0,
      'pre_import_tree_sha256': EMPTY_SHA256,
      'default_user_cache_paths_eligible': False,
  }


def _audit_cache_tree(root: Path, label: str) -> dict[str, Any]:
  _guard_path(root)
  if root.is_symlink() or not root.is_dir():
    raise AnalysisError(f'{label} root is absent or unsafe.')
  files = []
  directory_count = 1
  for lexical in root.rglob('*'):
    mode = lexical.lstat().st_mode
    if stat.S_ISLNK(mode):
      raise AnalysisError(f'{label} contains a symlink.')
    if stat.S_ISREG(mode):
      files.append(lexical)
    elif stat.S_ISDIR(mode):
      directory_count += 1
    else:
      raise AnalysisError(f'{label} contains a special entry.')
  required = {root / 'triton', root / 'xdg'}
  if any(path.is_symlink() or not path.is_dir() for path in required):
    raise AnalysisError(f'{label} lacks exact role-specific child roots.')
  return {
      'path': str(root.resolve()),
      'regular_file_count': len(files),
      'directory_count': directory_count,
      'regular_file_tree_sha256': _tree_digest(files, root),
      'diagnostic_only_not_an_execution_input_equality_gate': True,
  }


def _validate_start(
    run_dir: Path, freeze: Mapping[str, Any], freeze_sha256: str,
    *, bundle_root: Path, original_audit: Mapping[str, Any],
    v3_3_2_run: Mapping[str, Any], v3_3_2_1_failure: Mapping[str, Any],
    v3_3_2_2_archive: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[int, Any]]:
  start = _read_json(run_dir / 'ATTEMPT_STARTED.json', 'ATTEMPT_STARTED')
  _exact_keys(start, _START_KEYS, 'ATTEMPT_STARTED')
  expected_signatures = v3_3_2_run['eight_row_compiler']['program_signatures']
  required = {
      'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION,
      'status': 'started_append_only_one_shot',
      'amendment': {
          'path': str(_AMENDMENT_PATH.resolve()),
          'sha256': AMENDMENT_SHA256, 'commit': AMENDMENT_COMMIT,
      },
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'freeze': dict(freeze), 'freeze_sha256': freeze_sha256,
      'original_run_binding': dict(_v332._ORIGINAL_BINDING),  # pylint: disable=protected-access
      'original_run_revalidated_in_full': True,
      'v3_3_1_status': freeze['v3_3_1_status'],
      'v3_3_2_run_binding': dict(v3_3_2_run),
      'v3_3_2_1_failure_status': dict(v3_3_2_1_failure),
      'v3_3_2_2_archive_status': dict(v3_3_2_2_archive),
      'source_program_contract': {
          'stablehlo_sha256': SOURCE_STABLEHLO['sha256'],
          'stablehlo_size_bytes': SOURCE_STABLEHLO['size_bytes'],
          'pre_backend_hlo_sha256': SOURCE_PRE_BACKEND_HLO['sha256'],
          'pre_backend_hlo_size_bytes': SOURCE_PRE_BACKEND_HLO['size_bytes'],
          'program_signatures_sha256': PROGRAM_SIGNATURES_SHA256,
          'entry_abi_sha256': ENTRY_ABI_SHA256,
      },
      'program_signatures': expected_signatures,
      'compiled_backend_equality_is_a_gate': False,
      'record_count_contract': 80, 'model_apply_count_contract': 320,
      'compile_count_contract': {'eight_row': 1, 'six_row': 0},
      'rerun_count_contract': {'identity': 0, 'main_cube': 0},
      'execution_order_contract': {
          'recipient_orders': list(RECIPIENT_ORDERS),
          'anchor_ids': list(ANCHOR_IDS), 'major': 'recipient',
          'minor': 'anchor', 'indices': [0, 79],
      },
      'invariant_rows_between_calls': list(INVARIANT_ROWS),
      'active_recipient_rows_without_cross_call_predicate': list(ACTIVE_ROWS),
      'max_wall_time_seconds': 7_200,
      'max_output_bytes': 1_073_741_824,
      'confirmation_model_calls': 0,
      'scientific_summary_computed': False,
      'shapley_or_nomination_computed': False,
      'combined_analysis_permitted': False,
      'confirmation_scope_disclosure': CONFIRMATION_DISCLOSURE,
  }
  for key, expected in required.items():
    if start.get(key) != expected:
      raise AnalysisError(f'ATTEMPT_STARTED.{key} changed.')
  _finite(start.get('started_at_unix_s'), 'ATTEMPT_STARTED.started_at_unix_s')
  bootstrap = _exact_keys(start.get('bootstrap'), {
      'pid', 'created_at_unix_s', 'sanitized_environment',
      'generated_bindings', 'launcher_path', 'launcher_sha256',
      'bootstrap_path', 'bootstrap_sha256', 'freeze_path', 'freeze_sha256',
      'git_head', 'tracked_head_clean', 'original_run',
      'original_eight_row_compiler', 'original_run_all_5158_files_rehashed',
      'v3_3_1_status', 'v3_3_2_run', 'v3_3_2_1_failure_status',
      'v3_3_2_2_archive_status', 'original_bundle',
  }, 'ATTEMPT_STARTED.bootstrap')
  original_complete = _read_json(
      _v332._ORIGINAL_RUN_DIR / 'RUN_COMPLETE.json',  # pylint: disable=protected-access
      'original v3.3 RUN_COMPLETE',
  )
  original_compiler = original_complete.get('eight_row_compiler')
  if not isinstance(original_compiler, Mapping):
    raise AnalysisError('Original v3.3 compiler binding is absent.')
  original_freeze = _read_json(
      _v332._ORIGINAL_FREEZE_PATH,  # pylint: disable=protected-access
      'original v3.3 freeze',
  )
  expected_original_bundle = copy.deepcopy(
      original_audit['original_bundle_binding']
  )
  expected_original_bundle['git_head'] = bootstrap.get('git_head')
  expected_original_bundle['tracked_paths'] = sorted({
      str(_v332._ORIGINAL_FREEZE_PATH.relative_to(bundle_root)),  # pylint: disable=protected-access
      str(Path(original_freeze['protocol_path']).resolve().relative_to(bundle_root)),
      str(Path(original_freeze['supersession_path']).resolve().relative_to(bundle_root)),
      *original_freeze['file_sha256'].keys(),
  })
  if (
      not isinstance(bootstrap.get('pid'), int) or bootstrap['pid'] <= 0
      or bootstrap.get('freeze_path') != str(_FREEZE_PATH.resolve())
      or bootstrap.get('freeze_sha256') != freeze_sha256
      or bootstrap.get('tracked_head_clean') is not True
      or bootstrap.get('original_run') != _v332._ORIGINAL_BINDING  # pylint: disable=protected-access
      or bootstrap.get('original_eight_row_compiler')
      != original_compiler
      or bootstrap.get('original_run_all_5158_files_rehashed') is not True
      or bootstrap.get('v3_3_1_status') != freeze['v3_3_1_status']
      or bootstrap.get('v3_3_2_run') != v3_3_2_run
      or bootstrap.get('v3_3_2_1_failure_status') != v3_3_2_1_failure
      or bootstrap.get('v3_3_2_2_archive_status') != v3_3_2_2_archive
      or bootstrap.get('original_bundle')
      != expected_original_bundle
  ):
    raise AnalysisError('START bootstrap archive/bundle binding changed.')
  _finite(bootstrap.get('created_at_unix_s'), 'bootstrap.created_at_unix_s')
  expected_sanitized = {
      'LD_LIBRARY_PATH': 'absent',
      'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
      'JAX_ENABLE_COMPILATION_CACHE': 'false',
      'cache_environment': _expected_cache_environment(freeze, 'model'),
  }
  if bootstrap.get('sanitized_environment') != expected_sanitized:
    raise AnalysisError('Bootstrap sanitized/cache environment changed.')
  for field, path in (
      ('launcher', _HERE / 'launch_encoder_skip_ood_sidecar_v3_3_3.py'),
      ('bootstrap', _HERE / 'validate_encoder_skip_ood_sidecar_bootstrap_v3_3_3.py'),
  ):
    if (
        bootstrap.get(f'{field}_path') != str(path.resolve())
        or bootstrap.get(f'{field}_sha256') != _sha256(path)
    ):
      raise AnalysisError(f'Bootstrap {field} binding changed.')

  preflight_audit = _validate_external_and_same_process(
      start, freeze, freeze_sha256
  )
  cases = _v332._v33._load_cases()  # pylint: disable=protected-access
  try:
    checkpoint_audit = _v332._v33._validate_checkpoint_reference(  # pylint: disable=protected-access
        start, freeze, cases
    )
    runtime_audit = _v332._v33._validate_runtime_manifest(  # pylint: disable=protected-access
        start, freeze
    )
    normalized_start = dict(start)
    normalized_start['same_process_pre_import_bootstrap'] = {
        'freeze': bootstrap['original_bundle']
    }
    _, upstream_audit = _v332._v33._validate_upstream_checkout(  # pylint: disable=protected-access
        normalized_start, freeze, bundle_root=bundle_root
    )
  except _v332._v33.AnalysisError as error:  # pylint: disable=protected-access
    raise AnalysisError(str(error)) from error
  return {
      'git_head': bootstrap['git_head'],
      'v3_3_1_status': freeze['v3_3_1_status'],
      'original_bundle_binding': bootstrap['original_bundle'],
      'checkpoint_binding': copy.deepcopy(start['checkpoint_binding']),
      'reference_object_binding': copy.deepcopy(
          start['reference_object_binding']
      ),
      **preflight_audit, **checkpoint_audit, **runtime_audit,
      **upstream_audit,
  }, checkpoint_audit['sequence_bindings']


def _validate_manifest(
    run_dir: Path, cases: Mapping[int, Mapping[str, Any]],
    prefix: Sequence[tuple[int, int]],
) -> tuple[dict[str, Any], dict[str, str]]:
  try:
    return _v332._validate_manifest(  # pylint: disable=protected-access
        run_dir, cases, prefix
    )
  except _v332.AnalysisError as error:
    raise AnalysisError(str(error)) from error


def _validate_top_level_tree(run_dir: Path, *, has_raw: bool) -> None:
  try:
    _v332._validate_top_level_tree(  # pylint: disable=protected-access
        run_dir, has_raw=has_raw
    )
  except _v332.AnalysisError as error:
    raise AnalysisError(str(error)) from error


def analyze(
    run_dir: Path, *, bundle_root: Path | None = None,
    _raw_access_marker: Callable[[], None] | None = None,
    _attempt_token: object | None = None,
    _attempt_started_sha256: str | None = None,
) -> dict[str, Any]:
  """Audits only provenance/structure; no scientific estimator is run."""
  _assert_cpu_only('v3.3.3 analyzer entry')
  run_dir = run_dir.resolve()
  bundle_root = _REPO_ROOT if bundle_root is None else bundle_root.resolve()
  if run_dir == _RUN_DIR.resolve():
    _validate_active_analysis_attempt(
        run_dir, token=_attempt_token,
        started_sha256=_attempt_started_sha256,
    )
  _guard_path(run_dir)
  _guard_path(bundle_root)
  if not run_dir.is_dir() or run_dir.is_symlink():
    raise AnalysisError('v3.3.3 run directory is absent or unsafe.')
  if (run_dir / 'TERMINAL_FAILURE.json').exists():
    raise AnalysisError('Unexpected TERMINAL_FAILURE is not auditable.')
  (
      freeze, freeze_sha, original_audit, original_manifest,
      sequence_bindings, v3_3_2_run, v3_3_2_1_failure,
      v3_3_2_2_archive,
  ) = _validate_freeze(run_dir, bundle_root=bundle_root)
  start_audit, start_sequence_bindings = _validate_start(
      run_dir, freeze, freeze_sha, bundle_root=bundle_root,
      original_audit=original_audit, v3_3_2_run=v3_3_2_run,
      v3_3_2_1_failure=v3_3_2_1_failure,
      v3_3_2_2_archive=v3_3_2_2_archive,
  )
  if start_sequence_bindings != sequence_bindings:
    raise AnalysisError('Independent START/original sequence bindings differ.')
  completion = _read_json(run_dir / 'RUN_COMPLETE.json', 'RUN_COMPLETE')

  # Separate compiler-failure and partial-apply schemas are handled below once
  # their frozen serializers are selected.  Never reinterpret them as a full
  # record prefix.
  if completion.get('stop_reason') == 'compiler_failure':
    return _analyze_compiler_failure(
        run_dir, freeze=freeze, freeze_sha=freeze_sha,
        completion=completion, start_audit=start_audit,
        original_audit=original_audit, v3_3_2_run=v3_3_2_run,
        v3_3_2_1_failure=v3_3_2_1_failure,
        v3_3_2_2_archive=v3_3_2_2_archive, bundle_root=bundle_root,
    )
  if completion.get('failed_current_record') is not None:
    return _analyze_partial_apply_stop(
        run_dir, freeze=freeze, freeze_sha=freeze_sha,
        completion=completion, start_audit=start_audit,
        original_audit=original_audit, original_manifest=original_manifest,
        sequence_bindings=sequence_bindings, v3_3_2_run=v3_3_2_run,
        v3_3_2_1_failure=v3_3_2_1_failure,
        v3_3_2_2_archive=v3_3_2_2_archive, bundle_root=bundle_root,
        raw_access_marker=_raw_access_marker,
    )

  prefix, fully_complete = _completion_prefix(
      completion, freeze_sha256=freeze_sha,
      original_run=_v332._ORIGINAL_BINDING,  # pylint: disable=protected-access
      v3_3_1_status=freeze['v3_3_1_status'],
      v3_3_2_run=v3_3_2_run, v3_3_2_1_failure=v3_3_2_1_failure,
      v3_3_2_2_archive=v3_3_2_2_archive,
  )
  _validate_top_level_tree(run_dir, has_raw=bool(prefix))
  cases = _v332._v33._load_cases()  # pylint: disable=protected-access
  manifest, raw_hashes = _validate_manifest(run_dir, cases, prefix)
  if completion.get('raw_manifest') != manifest:
    raise AnalysisError('RUN_COMPLETE embedded raw manifest changed.')
  imports_audit = _validate_imports(
      run_dir, completion, bundle_root=bundle_root, freeze=freeze
  )
  protobuf_audit = _validate_protobuf(run_dir, completion, freeze)
  original_complete = _read_json(
      _v332._ORIGINAL_RUN_DIR / 'RUN_COMPLETE.json',  # pylint: disable=protected-access
      'original v3.3 RUN_COMPLETE',
  )
  original_compiler = original_complete.get('eight_row_compiler')
  v332_compiler = v3_3_2_run['eight_row_compiler']
  expected_source_input_audit = {
      'bootstrap_sources_and_prior_trees_exact': True,
      'tracked_head_and_frozen_inventory_exact': True,
      'external_device_runtime_environment_exact': True,
      'same_process_device_runtime_environment_exact': True,
      'checkpoint_exact': True,
      'reference_object_and_sequences_exact': True,
      'protobuf_binding_exact': protobuf_audit['seven_role_two_generated_output_repair_exact'],
      'three_import_inventories_stable_exact': imports_audit['stable_shared_module_bytes'],
      'freeze_sha256': freeze_sha,
      'external_preflight_sha256': start_audit['external_preflight_sha256'],
      'checkpoint_binding': start_audit['checkpoint_binding'],
      'reference_object_binding': start_audit['reference_object_binding'],
  }
  fingerprint, compiler_audit = _validate_compiler(
      run_dir, completion.get('eight_row_compiler'),
      original_v3_3=original_compiler, consumed_v3_3_2=v332_compiler,
      expected_source_input_audit=expected_source_input_audit,
      expected_kernel_cache_preimport=_expected_cache_environment(
          freeze, 'model'
      ),
  )
  if (
      completion.get('eight_row_executable_fingerprint') != fingerprint
      or completion.get('source_program_gate')
      != completion['eight_row_compiler']['source_program_gate']
      or completion.get('source_program_exact')
      is not compiler_audit['source_program_exact']
      or completion.get('backend_diagnostics')
      != compiler_audit['backend_diagnostics']
      or completion.get('diagnostic_comparisons')
      != compiler_audit['diagnostic_comparisons']
  ):
    raise AnalysisError('RUN_COMPLETE compiler/source-program linkage changed.')
  _validate_final_model_cache(
      completion.get('model_kernel_cache_final'), freeze,
      compiler_cache_audit=compiler_audit['kernel_cache_provenance'],
  )

  record_audits = []
  if prefix and _raw_access_marker is not None:
    _raw_access_marker()
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
  invalid_indices = [
      index for index, row in enumerate(record_audits)
      if row['status'] != 'complete'
  ]
  expected_invalid = (
      [len(record_audits) - 1]
      if completion.get('stop_reason') == 'ood_tooling_failure'
      and record_audits else []
  )
  if invalid_indices != expected_invalid:
    raise AnalysisError('Invalid artifact is not the exact final prefix row.')
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
      or not compiler_audit['source_program_exact']
  ):
    raise AnalysisError('Complete sidecar lacks exact 80-record/source closure.')
  decision = (
      'sidecar_complete_structural_audit' if fully_complete
      else f"controlled_stop_{completion['stop_reason']}"
  )
  result = _structural_result(
      decision=decision, fully_complete=fully_complete,
      controlled_stop=(None if fully_complete else {
          'reason': completion['stop_reason'],
          'message': completion['message'],
          'audited_record_count': len(prefix),
      }),
      start_audit=start_audit, compiler_audit=compiler_audit,
      imports_audit=imports_audit, protobuf_audit=protobuf_audit,
      manifest=manifest, raw_hash_count=len(raw_hashes),
      valid_count=sum(row['status'] == 'complete' for row in record_audits),
      invalid_count=sum(row['status'] != 'complete' for row in record_audits),
      apply_count=completion['model_apply_count'], id0=observed_id0,
      id255=observed_id255, v3_3_2_run=v3_3_2_run,
      v3_3_2_1_failure=v3_3_2_1_failure,
      v3_3_2_2_archive=v3_3_2_2_archive,
  )
  _assert_cpu_only('v3.3.3 analyzer exit')
  return result


def _structural_result(
    *, decision: str, fully_complete: bool,
    controlled_stop: Mapping[str, Any] | None,
    start_audit: Mapping[str, Any], compiler_audit: Mapping[str, Any],
    imports_audit: Mapping[str, Any], protobuf_audit: Mapping[str, Any],
    manifest: Mapping[str, Any], raw_hash_count: int, valid_count: int,
    invalid_count: int, apply_count: int, id0: bool, id255: bool,
    v3_3_2_run: Mapping[str, Any], v3_3_2_1_failure: Mapping[str, Any],
    v3_3_2_2_archive: Mapping[str, Any],
) -> dict[str, Any]:
  return {
      'analysis_version': ANALYSIS_VERSION,
      'status': (
          'complete_structural_audit' if fully_complete
          else 'complete_controlled_stop_audited'
      ),
      'decision': decision,
      'controlled_stop': controlled_stop,
      'scientific_summary_computed': False,
      'donor_normalization_computed': False,
      'shapley_or_nomination_computed': False,
      'interaction_or_resolution_computed': False,
      'nomination_performed': False,
      'nomination': None,
      'resolution_analysis': None,
      'combined_analysis_permitted': False,
      'combined_analysis_requirement': (
          'A later separately prospective CPU scientific analyzer is required.'
      ),
      'sidecar_audit': {
          'expected_record_count': 80,
          'audited_record_count': manifest['artifact_count'],
          'valid_record_count': valid_count,
          'invalid_record_count': invalid_count,
          'expected_model_apply_count': 320,
          'audited_model_apply_count': apply_count,
          'execution_order_exact': True,
          'raw_artifact_tree_sha256': manifest['artifact_tree_sha256'],
          'raw_artifact_count': manifest['artifact_count'],
          'raw_hash_count': raw_hash_count,
          'raw_endpoint_evidence_recomputed_where_emitted': True,
          'all_four_readouts_present_for_every_record': fully_complete,
          'old_ood_records_used_as_data': False,
          'id0_all20': id0, 'id255_all20': id255,
      },
      'provenance_audit': {
          **dict(start_audit),
          'amendment_sha256': AMENDMENT_SHA256,
          'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
          'v3_3_2_run': dict(v3_3_2_run),
          'v3_3_2_1_failure_status': dict(v3_3_2_1_failure),
          'v3_3_2_2_archive_status': dict(v3_3_2_2_archive),
          'source_program_gate': dict(compiler_audit),
          'compiled_backend_equality_gate': False,
          'imports': dict(imports_audit),
          'protobuf': dict(protobuf_audit),
      },
      'confirmation_model_outputs_activations_interventions_unopened': True,
      'confirmation_scope_disclosure': CONFIRMATION_DISCLOSURE,
  }


def _validate_compiler_failure_artifact(
    run_dir: Path, value: Any, *, expected_signatures: Mapping[str, Any],
    expected_kernel_cache_preimport: Mapping[str, Any],
) -> dict[str, Any]:
  compiler = _exact_keys(value, {
      'status', 'failure_stage', 'compile_count', 'compile_seconds',
      'lower_attempt_count', 'compile_attempt_count',
      'successful_compile_count', 'lower_or_compile_pipeline_attempt_count',
      'artifacts', 'program_signatures', 'program_signatures_sha256',
      'source_program_gate', 'compiled_backend_diagnostic_only', 'failure',
      'no_compile_retry', 'model_apply_count',
      'kernel_cache_provenance',
  }, 'compiler failure artifact')
  stage = compiler.get('failure_stage')
  count = compiler.get('compile_count')
  if (
      compiler.get('status') != 'compiler_failure'
      or stage not in {'lower', 'compile'}
      or count != (0 if stage == 'lower' else 1)
      or compiler.get('lower_attempt_count') != 1
      or compiler.get('compile_attempt_count') != count
      or compiler.get('successful_compile_count') != 0
      or compiler.get('lower_or_compile_pipeline_attempt_count') != 1
      or _finite(compiler.get('compile_seconds'), 'compiler failure seconds') < 0
      or compiler.get('program_signatures') != dict(expected_signatures)
      or compiler.get('program_signatures_sha256')
      != _canonical_json_sha256(expected_signatures)
      or compiler.get('source_program_gate') is not None
      or compiler.get('compiled_backend_diagnostic_only') is not True
      or compiler.get('no_compile_retry') is not True
      or compiler.get('model_apply_count') != 0
  ):
    raise AnalysisError('Compiler failure artifact contract changed.')
  failure = _exact_keys(
      compiler.get('failure'), {'type', 'message', 'traceback'},
      'compiler failure detail',
  )
  if (
      not isinstance(failure.get('type'), str)
      or not failure['type'].isidentifier()
      or not isinstance(failure.get('message'), str) or not failure['message']
      or not isinstance(failure.get('traceback'), str)
      or failure['type'] not in failure['traceback']
  ):
    raise AnalysisError('Compiler failure detail is malformed.')
  expected_names = set() if stage == 'lower' else {'stablehlo', 'hlo'}
  artifacts = compiler.get('artifacts')
  if not isinstance(artifacts, Mapping) or set(artifacts) != expected_names:
    raise AnalysisError('Compiler failure partial artifact set changed.')
  filenames = {
      'stablehlo': 'graph.stablehlo.mlir',
      'hlo': 'graph.pre_backend.hlo.txt',
  }
  expected_files = {'eight_row/COMPILER_PROVENANCE.json'}
  for name in expected_names:
    binding = _exact_keys(
        artifacts[name], {'path', 'sha256', 'size_bytes'},
        f'compiler failure {name}',
    )
    path = (run_dir / 'compiler/eight_row' / filenames[name]).resolve()
    expected_files.add(f'eight_row/{filenames[name]}')
    if (
        binding.get('path') != str(path) or not _is_sha256(binding.get('sha256'))
        or _sha256(path) != binding['sha256']
        or path.stat().st_size != binding.get('size_bytes')
    ):
      raise AnalysisError(f'Compiler failure {name} binding changed.')
  _strict_tree(run_dir / 'compiler', expected_files, 'compiler failure tree')
  persisted = _read_json(
      run_dir / 'compiler/eight_row/COMPILER_PROVENANCE.json',
      'compiler failure provenance',
  )
  if persisted != dict(compiler):
    raise AnalysisError('Compiler failure embedded/persisted record differs.')
  cache_audit = _validate_kernel_cache_provenance(
      compiler.get('kernel_cache_provenance'),
      expected_preimport=expected_kernel_cache_preimport,
      phase='post_failure',
  )
  return {
      'failure_stage': stage, 'compile_count': count,
      'failure_type': failure['type'], 'source_program_gate': None,
      'compiled_backend_equality_gate': False,
      'kernel_cache_provenance': cache_audit,
  }


def _validate_compiler_failure_imports(
    run_dir: Path, completion: Mapping[str, Any], *, bundle_root: Path,
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
  names = {
      'pre_model': 'IMPORT_PROVENANCE_PRE_MODEL.json',
      'post_model_precompile': 'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
      'postcompile_or_failure': 'IMPORT_PROVENANCE.json',
  }
  bindings = _exact_keys(
      completion.get('import_provenance_phases'), set(names),
      'compiler failure import phases',
  )
  phases = {
      phase: _validate_import_file(
          run_dir / filename, bindings[phase], bundle_root=bundle_root,
          freeze=freeze,
      )
      for phase, filename in names.items()
  }
  for earlier, later in (
      ('pre_model', 'post_model_precompile'),
      ('post_model_precompile', 'postcompile_or_failure'),
  ):
    if set(phases[earlier]) - set(phases[later]) or any(
        phases[earlier][name] != phases[later][name]
        for name in phases[earlier] if name in phases[later]
    ):
      raise AnalysisError('Compiler-failure shared import bytes changed.')
  return {
      'phase_sha256': dict(bindings),
      'module_counts': {name: len(rows) for name, rows in phases.items()},
      'stable_shared_module_bytes': True,
  }


def _analyze_compiler_failure(
    run_dir: Path, *, freeze: Mapping[str, Any], freeze_sha: str,
    completion: Mapping[str, Any], start_audit: Mapping[str, Any],
    original_audit: Mapping[str, Any], v3_3_2_run: Mapping[str, Any],
    v3_3_2_1_failure: Mapping[str, Any],
    v3_3_2_2_archive: Mapping[str, Any], bundle_root: Path,
) -> dict[str, Any]:
  del original_audit
  keys = {
      'status', 'stop_reason', 'message', 'attempt_id', 'script_version',
      'amendment_sha256', 'amendment_commit', 'original_protocol_sha256',
      'freeze_sha256', 'ood_anchor_record_count', 'ood_invalid_count',
      'unique_recipient_anchor_count', 'model_apply_count',
      'expected_model_apply_count', 'eight_row_compile_count',
      'eight_row_compile_attempt_count',
      'eight_row_successful_compile_count',
      'six_row_compile_count', 'identity_rerun_count',
      'main_cube_rerun_count', 'old_ood_records_reused', 'compiler',
      'source_program_gate', 'compiled_backend_diagnostic_only',
      'no_compile_retry', 'original_run_binding',
      'original_run_revalidated_in_full',
      'original_ood_records_provenance_only', 'v3_3_1_status',
      'v3_3_2_run_binding',
      'v3_3_2_1_failure_status', 'v3_3_2_2_archive_status',
      'import_provenance_phases', 'protobuf_provenance_sha256',
      'raw_manifest', 'confirmation_model_calls',
      'scientific_summary_computed', 'shapley_or_nomination_computed',
      'interaction_or_resolution_computed', 'combined_analysis_permitted',
      'confirmation_scope_disclosure',
      'model_kernel_cache_final',
      'completed_at_unix_s',
  }
  _exact_keys(completion, keys, 'compiler-failure RUN_COMPLETE')
  common = {
      'status': 'controlled_stop', 'stop_reason': 'compiler_failure',
      'message': 'The sole lowering/compiler attempt failed; no retry.',
      'attempt_id': ATTEMPT_ID, 'script_version': SCRIPT_VERSION,
      'amendment_sha256': AMENDMENT_SHA256,
      'amendment_commit': AMENDMENT_COMMIT,
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'freeze_sha256': freeze_sha, 'ood_anchor_record_count': 0,
      'ood_invalid_count': 0, 'unique_recipient_anchor_count': 0,
      'model_apply_count': 0, 'expected_model_apply_count': 320,
      'eight_row_compile_attempt_count': completion.get(
          'eight_row_compile_count'
      ),
      'eight_row_successful_compile_count': 0,
      'six_row_compile_count': 0, 'identity_rerun_count': 0,
      'main_cube_rerun_count': 0, 'old_ood_records_reused': 0,
      'source_program_gate': None,
      'compiled_backend_diagnostic_only': True, 'no_compile_retry': True,
      'original_run_binding': dict(_v332._ORIGINAL_BINDING),  # pylint: disable=protected-access
      'original_run_revalidated_in_full': True,
      'original_ood_records_provenance_only': True,
      'v3_3_1_status': freeze['v3_3_1_status'],
      'v3_3_2_run_binding': dict(v3_3_2_run),
      'v3_3_2_1_failure_status': dict(v3_3_2_1_failure),
      'v3_3_2_2_archive_status': dict(v3_3_2_2_archive),
      'confirmation_model_calls': 0,
      'scientific_summary_computed': False,
      'shapley_or_nomination_computed': False,
      'interaction_or_resolution_computed': False,
      'combined_analysis_permitted': False,
      'confirmation_scope_disclosure': CONFIRMATION_DISCLOSURE,
  }
  for key, expected in common.items():
    if completion.get(key) != expected:
      raise AnalysisError(f'Compiler-failure RUN_COMPLETE.{key} changed.')
  _finite(completion.get('completed_at_unix_s'), 'compiler failure completed_at')
  compiler_audit = _validate_compiler_failure_artifact(
      run_dir, completion.get('compiler'),
      expected_signatures=freeze['program_signatures'],
      expected_kernel_cache_preimport=_expected_cache_environment(
          freeze, 'model'
      ),
  )
  if completion.get('eight_row_compile_count') != compiler_audit['compile_count']:
    raise AnalysisError('Compiler-failure compile count linkage changed.')
  _validate_final_model_cache(
      completion.get('model_kernel_cache_final'), freeze,
      compiler_cache_audit=compiler_audit['kernel_cache_provenance'],
  )
  _validate_top_level_tree(run_dir, has_raw=False)
  cases = _v332._v33._load_cases()  # pylint: disable=protected-access
  manifest, raw_hashes = _validate_manifest(run_dir, cases, ())
  if completion.get('raw_manifest') != manifest:
    raise AnalysisError('Compiler-failure raw-manifest linkage changed.')
  imports_audit = _validate_compiler_failure_imports(
      run_dir, completion, bundle_root=bundle_root, freeze=freeze
  )
  protobuf_audit = _validate_protobuf(run_dir, completion, freeze)
  return _structural_result(
      decision='controlled_stop_compiler_failure', fully_complete=False,
      controlled_stop={
          'reason': 'compiler_failure', 'message': completion['message'],
          'audited_record_count': 0,
      },
      start_audit=start_audit, compiler_audit=compiler_audit,
      imports_audit=imports_audit, protobuf_audit=protobuf_audit,
      manifest=manifest, raw_hash_count=len(raw_hashes), valid_count=0,
      invalid_count=0, apply_count=0, id0=False, id255=False,
      v3_3_2_run=v3_3_2_run, v3_3_2_1_failure=v3_3_2_1_failure,
      v3_3_2_2_archive=v3_3_2_2_archive,
  )


def _analyze_partial_apply_stop(
    run_dir: Path, *, freeze: Mapping[str, Any], freeze_sha: str,
    completion: Mapping[str, Any], start_audit: Mapping[str, Any],
    original_audit: Mapping[str, Any], original_manifest: Mapping[str, str],
    sequence_bindings: Mapping[int, Any], v3_3_2_run: Mapping[str, Any],
    v3_3_2_1_failure: Mapping[str, Any],
    v3_3_2_2_archive: Mapping[str, Any], bundle_root: Path,
    raw_access_marker: Callable[[], None] | None = None,
) -> dict[str, Any]:
  del original_audit
  keys = {
      'status', 'stop_reason', 'message', 'attempt_id', 'script_version',
      'amendment_sha256', 'amendment_commit', 'original_protocol_sha256',
      'freeze_sha256', 'ood_anchor_record_count', 'ood_invalid_count',
      'incomplete_record_count', 'unique_recipient_anchor_count',
      'model_apply_count', 'expected_model_apply_count',
      'eight_row_compile_count', 'eight_row_compile_attempt_count',
      'eight_row_successful_compile_count', 'six_row_compile_count',
      'identity_rerun_count', 'main_cube_rerun_count',
      'old_ood_records_reused', 'failed_current_record',
      'eight_row_compiler', 'eight_row_executable_fingerprint',
      'source_program_gate', 'source_program_exact',
      'compiled_backend_diagnostic_only', 'backend_diagnostics',
      'diagnostic_comparisons', 'original_run_binding',
      'original_run_revalidated_in_full',
      'original_ood_records_provenance_only', 'v3_3_1_status',
      'v3_3_2_run_binding', 'v3_3_2_1_failure_status',
      'v3_3_2_2_archive_status', 'import_provenance_phases',
      'protobuf_provenance_sha256', 'raw_manifest',
      'confirmation_model_calls', 'scientific_summary_computed',
      'shapley_or_nomination_computed', 'interaction_or_resolution_computed',
      'combined_analysis_permitted', 'no_retry', 'completed_at_unix_s',
      'confirmation_scope_disclosure',
      'model_kernel_cache_final',
  }
  _exact_keys(completion, keys, 'partial-apply RUN_COMPLETE')
  count = completion.get('ood_anchor_record_count')
  if isinstance(count, bool) or not isinstance(count, int) or not (0 <= count < 80):
    raise AnalysisError('Partial-apply completed-record count is invalid.')
  failed = _exact_keys(completion.get('failed_current_record'), {
      'execution_index', 'recipient_order', 'anchor_id', 'call_label',
      'dispatched_apply_count_for_current_record', 'error_type',
      'error_message',
  }, 'partial-apply failed_current_record')
  expected_order, expected_anchor = _execution_order()[count]
  current_count = failed.get('dispatched_apply_count_for_current_record')
  call_labels = ('intended', 'intended_repeat', 'unrelated', 'unrelated_repeat')
  allowed_labels = (
      {'record_setup_validation_or_persistence'} if current_count == 0
      else (
          {call_labels[current_count - 1]}
          if current_count in (1, 2, 3)
          else {
              'unrelated_repeat', 'record_setup_validation_or_persistence'
          }
      )
  ) if current_count in (0, 1, 2, 3, 4) else set()
  if (
      failed.get('execution_index') != count
      or failed.get('recipient_order') != expected_order
      or failed.get('anchor_id') != expected_anchor
      or current_count not in (0, 1, 2, 3, 4)
      or failed.get('call_label') not in allowed_labels
      or not isinstance(failed.get('error_type'), str)
      or not failed['error_type'].isidentifier()
      or not isinstance(failed.get('error_message'), str)
      or not failed['error_message']
  ):
    raise AnalysisError('Partial-apply failed-current-record binding changed.')
  common = {
      'status': 'controlled_stop', 'stop_reason': 'ood_tooling_failure',
      'message': 'A dispatched OOD call failed; exact prefix preserved; no retry.',
      'attempt_id': ATTEMPT_ID, 'script_version': SCRIPT_VERSION,
      'amendment_sha256': AMENDMENT_SHA256,
      'amendment_commit': AMENDMENT_COMMIT,
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'freeze_sha256': freeze_sha, 'ood_invalid_count': 0,
      'incomplete_record_count': 1,
      'unique_recipient_anchor_count': count,
      'model_apply_count': 4 * count + current_count,
      'expected_model_apply_count': 320,
      'eight_row_compile_count': 1, 'six_row_compile_count': 0,
      'eight_row_compile_attempt_count': 1,
      'eight_row_successful_compile_count': 1,
      'identity_rerun_count': 0, 'main_cube_rerun_count': 0,
      'old_ood_records_reused': 0, 'source_program_exact': True,
      'compiled_backend_diagnostic_only': True,
      'original_run_binding': dict(_v332._ORIGINAL_BINDING),  # pylint: disable=protected-access
      'original_run_revalidated_in_full': True,
      'original_ood_records_provenance_only': True,
      'v3_3_1_status': freeze['v3_3_1_status'],
      'v3_3_2_run_binding': dict(v3_3_2_run),
      'v3_3_2_1_failure_status': dict(v3_3_2_1_failure),
      'v3_3_2_2_archive_status': dict(v3_3_2_2_archive),
      'confirmation_model_calls': 0,
      'scientific_summary_computed': False,
      'shapley_or_nomination_computed': False,
      'interaction_or_resolution_computed': False,
      'combined_analysis_permitted': False, 'no_retry': True,
      'confirmation_scope_disclosure': CONFIRMATION_DISCLOSURE,
  }
  for key, expected in common.items():
    if completion.get(key) != expected:
      raise AnalysisError(f'Partial-apply RUN_COMPLETE.{key} changed.')
  _finite(completion.get('completed_at_unix_s'), 'partial apply completed_at')
  prefix = _execution_order()[:count]
  _validate_top_level_tree(run_dir, has_raw=bool(prefix))
  cases = _v332._v33._load_cases()  # pylint: disable=protected-access
  manifest, raw_hashes = _validate_manifest(run_dir, cases, prefix)
  if completion.get('raw_manifest') != manifest:
    raise AnalysisError('Partial-apply raw-manifest linkage changed.')
  imports_audit = _validate_imports(
      run_dir, completion, bundle_root=bundle_root, freeze=freeze
  )
  protobuf_audit = _validate_protobuf(run_dir, completion, freeze)
  original_complete = _read_json(
      _v332._ORIGINAL_RUN_DIR / 'RUN_COMPLETE.json',  # pylint: disable=protected-access
      'original v3.3 RUN_COMPLETE',
  )
  expected_source_input_audit = {
      'bootstrap_sources_and_prior_trees_exact': True,
      'tracked_head_and_frozen_inventory_exact': True,
      'external_device_runtime_environment_exact': True,
      'same_process_device_runtime_environment_exact': True,
      'checkpoint_exact': True, 'reference_object_and_sequences_exact': True,
      'protobuf_binding_exact': True,
      'three_import_inventories_stable_exact': True,
      'freeze_sha256': freeze_sha,
      'external_preflight_sha256': start_audit['external_preflight_sha256'],
      'checkpoint_binding': start_audit['checkpoint_binding'],
      'reference_object_binding': start_audit['reference_object_binding'],
  }
  fingerprint, compiler_audit = _validate_compiler(
      run_dir, completion.get('eight_row_compiler'),
      original_v3_3=original_complete['eight_row_compiler'],
      consumed_v3_3_2=v3_3_2_run['eight_row_compiler'],
      expected_source_input_audit=expected_source_input_audit,
      expected_kernel_cache_preimport=_expected_cache_environment(
          freeze, 'model'
      ),
  )
  if (
      completion.get('eight_row_executable_fingerprint') != fingerprint
      or completion.get('source_program_gate')
      != completion['eight_row_compiler']['source_program_gate']
      or completion.get('backend_diagnostics')
      != compiler_audit['backend_diagnostics']
      or completion.get('diagnostic_comparisons')
      != compiler_audit['diagnostic_comparisons']
  ):
    raise AnalysisError('Partial-apply compiler linkage changed.')
  _validate_final_model_cache(
      completion.get('model_kernel_cache_final'), freeze,
      compiler_cache_audit=compiler_audit['kernel_cache_provenance'],
  )
  record_audits = []
  if prefix and raw_access_marker is not None:
    raw_access_marker()
  for index, (order, anchor) in enumerate(prefix):
    case = cases[order]
    donor_case = cases[_donor_order(order)]
    relative = _artifact_relative(case, anchor)
    record_audits.append(_validate_record(
        _read_json(run_dir / relative, relative), case=case,
        donor_case=donor_case, anchor=anchor, execution_index=index,
        freeze_sha256=freeze_sha, executable_fingerprint=fingerprint,
        original_manifest=original_manifest,
        sequence_bindings=sequence_bindings, allow_invalid=False,
    ))
  if any(row['status'] != 'complete' for row in record_audits):
    raise AnalysisError('Partial-apply prefix contains an invalid persisted record.')
  id0 = sum(row['anchor'] == 0 for row in record_audits) == 20
  id255 = sum(row['anchor'] == 255 for row in record_audits) == 20
  return _structural_result(
      decision='controlled_stop_ood_tooling_failure_partial_apply',
      fully_complete=False,
      controlled_stop={
          'reason': 'ood_tooling_failure', 'message': completion['message'],
          'audited_record_count': count,
          'failed_current_record': dict(failed),
      },
      start_audit=start_audit, compiler_audit=compiler_audit,
      imports_audit=imports_audit, protobuf_audit=protobuf_audit,
      manifest=manifest, raw_hash_count=len(raw_hashes),
      valid_count=count, invalid_count=0,
      apply_count=completion['model_apply_count'], id0=id0, id255=id255,
      v3_3_2_run=v3_3_2_run, v3_3_2_1_failure=v3_3_2_1_failure,
      v3_3_2_2_archive=v3_3_2_2_archive,
  )


def render_markdown(result: Mapping[str, Any]) -> str:
  lines = [
      '# OpenSplice v3.3.3 OOD sidecar structural audit', '',
      f"**Decision:** `{result['decision']}`", '',
  ]
  if result.get('controlled_stop') is None:
    lines.extend([
        'All 80 OOD sidecar records passed the frozen structural, repeat,',
        'route, source-program, and provenance gates.', '',
    ])
  else:
    lines.extend([
        'The exact append-only controlled-stop prefix was audited. No',
        'scientific or biological conclusion is authorized from this stop.', '',
    ])
  lines.extend([
      'Compiled backend HLO and its Triton/cuBLAS/cuDNN choices are retained',
      'as descriptive provenance only; compiled-byte equality is not a gate.',
      '',
      'This CPU analyzer computed no donor normalization, Shapley value,',
      'interaction, resolution result, rank, or nomination. A later,',
      'separately prospective CPU scientific analyzer is required.', '',
      'The earlier v3.3 OOD tooling stop, v3.3.2 zero-apply compiler-byte',
      'stop, failed v3.3.2.1 recursion attempt, and successful no-science',
      'v3.3.2.2 archive were independently bound and rehashed.', '',
      'Later-exon metadata/labels were exposed after protocol freeze;',
      'confirmation model outputs, activations, and interventions remain',
      'unopened.', '',
  ])
  return '\n'.join(lines)


def _write_outputs(
    result: Mapping[str, Any], *, output_json: Path, output_markdown: Path,
) -> None:
  expected_json = (_ANALYSIS_DIR / 'ANALYSIS.json').resolve()
  expected_markdown = (_ANALYSIS_DIR / 'RESULT.md').resolve()
  if (
      output_json.resolve() != expected_json
      or output_markdown.resolve() != expected_markdown
  ):
    raise AnalysisError('Analysis output paths differ from the frozen destinations.')
  if _ANALYSIS_DIR.exists() or _ANALYSIS_DIR.is_symlink():
    raise FileExistsError('v3.3.3 analysis output already exists; never overwrite.')
  _ANALYSIS_DIR.mkdir(parents=True, exist_ok=False)
  for path, payload in (
      (output_json, json.dumps(
          result, indent=2, sort_keys=True, allow_nan=False
      ) + '\n'),
      (output_markdown, render_markdown(result)),
  ):
    descriptor = path.open('x', encoding='utf-8')
    with descriptor:
      descriptor.write(payload)


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
  payload = json.dumps(
      value, indent=2, sort_keys=True, allow_nan=False
  ) + '\n'
  with path.open('x', encoding='utf-8') as handle:
    handle.write(payload)


def _analysis_attempt_precheck(
    run_dir: Path, *, bundle_root: Path,
) -> dict[str, Any]:
  """Completes provenance-only gates before consuming the analysis attempt."""
  if (
      _ANALYSIS_ATTEMPT_DIR.exists() or _ANALYSIS_ATTEMPT_DIR.is_symlink()
      or _ANALYSIS_DIR.exists() or _ANALYSIS_DIR.is_symlink()
  ):
    raise FileExistsError('v3.3.3 analysis/attempt exists; never resume or retry.')
  (
      freeze, freeze_sha, original_audit, _original_manifest,
      _sequence_bindings, v3_3_2_run, v3_3_2_1_failure,
      v3_3_2_2_archive,
  ) = _validate_freeze(run_dir, bundle_root=bundle_root)
  _validate_start(
      run_dir, freeze, freeze_sha, bundle_root=bundle_root,
      original_audit=original_audit, v3_3_2_run=v3_3_2_run,
      v3_3_2_1_failure=v3_3_2_1_failure,
      v3_3_2_2_archive=v3_3_2_2_archive,
  )
  bindings = {}
  for name in ('ATTEMPT_STARTED.json', 'RUN_COMPLETE.json', 'RAW_MANIFEST.json'):
    path = run_dir / name
    _strict_regular(path, f'analysis precheck {name}')
    bindings[name] = {
        'path': str(path.resolve()), 'sha256': _sha256(path),
        'size_bytes': path.stat().st_size,
    }
  return {
      'analysis_version': ANALYSIS_VERSION,
      'status': 'analysis_started_append_only_one_shot',
      'amendment': {
          'path': str(_AMENDMENT_PATH.resolve()),
          'sha256': AMENDMENT_SHA256, 'commit': AMENDMENT_COMMIT,
      },
      'analyzer': {
          'path': str(Path(__file__).resolve()),
          'sha256': _sha256(Path(__file__).resolve()),
      },
      'freeze': {'path': str(_FREEZE_PATH.resolve()), 'sha256': freeze_sha},
      'run_dir': str(run_dir.resolve()),
      'analysis_attempt_dir': str(_ANALYSIS_ATTEMPT_DIR.resolve()),
      'analysis_dir': str(_ANALYSIS_DIR.resolve()),
      'output_json': str((_ANALYSIS_DIR / 'ANALYSIS.json').resolve()),
      'output_markdown': str((_ANALYSIS_DIR / 'RESULT.md').resolve()),
      'run_artifacts': bindings,
      'raw_scientific_endpoint_evidence_reached': False,
      'scientific_summary_computed': False,
      'donor_normalization_computed': False,
      'shapley_or_nomination_computed': False,
      'interaction_or_resolution_computed': False,
      'nomination_performed': False,
      'confirmation_model_outputs_activations_interventions_unopened': True,
      'started_at_unix_s': time.time(),
  }


_ANALYSIS_STARTED_KEYS = {
    'analysis_version', 'status', 'amendment', 'analyzer', 'freeze',
    'run_dir', 'analysis_attempt_dir', 'analysis_dir', 'output_json',
    'output_markdown', 'run_artifacts',
    'raw_scientific_endpoint_evidence_reached',
    'scientific_summary_computed', 'donor_normalization_computed',
    'shapley_or_nomination_computed', 'interaction_or_resolution_computed',
    'nomination_performed',
    'confirmation_model_outputs_activations_interventions_unopened',
    'started_at_unix_s',
}


def _validate_active_analysis_attempt(
    run_dir: Path, *, token: object | None, started_sha256: str | None,
) -> dict[str, Any]:
  if token is not _ANALYSIS_ATTEMPT_TOKEN or not _is_sha256(started_sha256):
    raise AnalysisError(
        'Production raw audit requires the internal post-START attempt gate.'
    )
  paths = _strict_tree(
      _ANALYSIS_ATTEMPT_DIR, {'ANALYSIS_ATTEMPT_STARTED.json'},
      'active analysis-attempt tree',
  )
  path = paths[0]
  if _sha256(path) != started_sha256:
    raise AnalysisError('Active analysis START hash changed.')
  value = _read_json(path, 'ANALYSIS_ATTEMPT_STARTED')
  _exact_keys(value, _ANALYSIS_STARTED_KEYS, 'ANALYSIS_ATTEMPT_STARTED')
  expected = {
      'analysis_version': ANALYSIS_VERSION,
      'status': 'analysis_started_append_only_one_shot',
      'amendment': {
          'path': str(_AMENDMENT_PATH.resolve()),
          'sha256': AMENDMENT_SHA256, 'commit': AMENDMENT_COMMIT,
      },
      'analyzer': {
          'path': str(Path(__file__).resolve()),
          'sha256': _sha256(Path(__file__).resolve()),
      },
      'freeze': {
          'path': str(_FREEZE_PATH.resolve()), 'sha256': _sha256(_FREEZE_PATH),
      },
      'run_dir': str(run_dir.resolve()),
      'analysis_attempt_dir': str(_ANALYSIS_ATTEMPT_DIR.resolve()),
      'analysis_dir': str(_ANALYSIS_DIR.resolve()),
      'output_json': str((_ANALYSIS_DIR / 'ANALYSIS.json').resolve()),
      'output_markdown': str((_ANALYSIS_DIR / 'RESULT.md').resolve()),
      'raw_scientific_endpoint_evidence_reached': False,
      'scientific_summary_computed': False,
      'donor_normalization_computed': False,
      'shapley_or_nomination_computed': False,
      'interaction_or_resolution_computed': False,
      'nomination_performed': False,
      'confirmation_model_outputs_activations_interventions_unopened': True,
  }
  for key, expected_value in expected.items():
    if value.get(key) != expected_value:
      raise AnalysisError(f'Active analysis START changed at {key}.')
  _finite(value.get('started_at_unix_s'), 'analysis START.started_at_unix_s')
  bindings = _exact_keys(
      value.get('run_artifacts'),
      {'ATTEMPT_STARTED.json', 'RUN_COMPLETE.json', 'RAW_MANIFEST.json'},
      'analysis START run artifacts',
  )
  for name, binding in bindings.items():
    row = _exact_keys(
        binding, {'path', 'sha256', 'size_bytes'},
        f'analysis START {name}',
    )
    current = run_dir / name
    if (
        row.get('path') != str(current.resolve())
        or row.get('sha256') != _sha256(current)
        or row.get('size_bytes') != current.stat().st_size
    ):
      raise AnalysisError(f'Analysis START run binding changed: {name}.')
  return dict(value)


def _analysis_complete_record(
    started_sha256: str, *, raw_reached: bool,
) -> dict[str, Any]:
  outputs = {}
  for name in ('ANALYSIS.json', 'RESULT.md'):
    path = _ANALYSIS_DIR / name
    _strict_regular(path, f'analysis output {name}')
    outputs[name] = {
        'path': str(path.resolve()), 'sha256': _sha256(path),
        'size_bytes': path.stat().st_size,
    }
  return {
      'analysis_version': ANALYSIS_VERSION,
      'status': 'analysis_complete_structural_only',
      'attempt_started_sha256': started_sha256,
      'outputs': outputs,
      'raw_scientific_endpoint_evidence_reached': raw_reached,
      'scientific_summary_computed': False,
      'donor_normalization_computed': False,
      'shapley_or_nomination_computed': False,
      'interaction_or_resolution_computed': False,
      'nomination_performed': False,
      'confirmation_model_outputs_activations_interventions_unopened': True,
      'completed_at_unix_s': time.time(),
  }


def _analysis_failure_record(
    error: BaseException, started_sha256: str, *, raw_reached: bool,
) -> dict[str, Any]:
  return {
      'analysis_version': ANALYSIS_VERSION,
      'status': 'analysis_failed_consumed_no_retry',
      'attempt_started_sha256': started_sha256,
      'error': {
          'type': type(error).__name__, 'message': str(error),
          'traceback': ''.join(traceback.format_exception(error)),
      },
      'analysis_output_state': _analysis_output_state(),
      'raw_scientific_endpoint_evidence_reached': raw_reached,
      'scientific_summary_computed': False,
      'donor_normalization_computed': False,
      'shapley_or_nomination_computed': False,
      'interaction_or_resolution_computed': False,
      'nomination_performed': False,
      'confirmation_model_outputs_activations_interventions_unopened': True,
      'failed_at_unix_s': time.time(),
  }


def _analysis_output_state() -> dict[str, Any]:
  if not _ANALYSIS_DIR.exists() and not _ANALYSIS_DIR.is_symlink():
    return {
        'state': 'absent', 'file_count': 0, 'files': {},
        'tree_sha256': EMPTY_SHA256,
    }
  if _ANALYSIS_DIR.is_symlink() or not _ANALYSIS_DIR.is_dir():
    raise AnalysisError('Failed-analysis output root is unsafe.')
  expected_names = {'ANALYSIS.json', 'RESULT.md'}
  files = []
  for entry in _ANALYSIS_DIR.iterdir():
    mode = entry.lstat().st_mode
    if entry.is_symlink() or not stat.S_ISREG(mode):
      raise AnalysisError('Failed-analysis output contains an unsafe entry.')
    if entry.name not in expected_names:
      raise AnalysisError('Failed-analysis output contains an extra file.')
    files.append(entry)
  bindings = {
      path.name: {
          'sha256': _sha256(path), 'size_bytes': path.stat().st_size,
      }
      for path in sorted(files)
  }
  return {
      'state': 'complete' if set(bindings) == expected_names else 'partial',
      'file_count': len(bindings), 'files': bindings,
      'tree_sha256': _tree_digest(files, _ANALYSIS_DIR),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run-dir', type=Path, required=True)
  parser.add_argument('--bundle-root', type=Path, default=_REPO_ROOT)
  parser.add_argument('--output-json', type=Path, required=True)
  parser.add_argument('--output-markdown', type=Path, required=True)
  args = parser.parse_args()
  for path in (
      args.run_dir, args.bundle_root, args.output_json, args.output_markdown,
      _FREEZE_PATH, _AMENDMENT_PATH,
  ):
    _guard_path(path)
  if (
      args.output_json.resolve() != (_ANALYSIS_DIR / 'ANALYSIS.json').resolve()
      or args.output_markdown.resolve()
      != (_ANALYSIS_DIR / 'RESULT.md').resolve()
  ):
    raise AnalysisError('CLI output paths differ from the frozen analysis paths.')
  run_dir = args.run_dir.resolve()
  bundle_root = args.bundle_root.resolve()
  started = _analysis_attempt_precheck(run_dir, bundle_root=bundle_root)
  _ANALYSIS_ATTEMPT_DIR.mkdir(parents=True, exist_ok=False)
  started_path = _ANALYSIS_ATTEMPT_DIR / 'ANALYSIS_ATTEMPT_STARTED.json'
  _write_json_new(started_path, started)
  started_sha = _sha256(started_path)
  raw_state = {'reached': False}

  def mark_raw_reached() -> None:
    raw_state['reached'] = True

  try:
    result = analyze(
        run_dir, bundle_root=bundle_root,
        _raw_access_marker=mark_raw_reached,
        _attempt_token=_ANALYSIS_ATTEMPT_TOKEN,
        _attempt_started_sha256=started_sha,
    )
    _write_outputs(
        result, output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    _write_json_new(
        _ANALYSIS_ATTEMPT_DIR / 'ANALYSIS_COMPLETE.json',
        _analysis_complete_record(
            started_sha, raw_reached=raw_state['reached']
        ),
    )
  except BaseException as error:
    _write_json_new(
        _ANALYSIS_ATTEMPT_DIR / 'ANALYSIS_FAILURE.json',
        _analysis_failure_record(
            error, started_sha, raw_reached=raw_state['reached']
        ),
    )
    raise


if __name__ == '__main__':
  main()
