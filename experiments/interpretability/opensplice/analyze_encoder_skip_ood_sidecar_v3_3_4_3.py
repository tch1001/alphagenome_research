#!/usr/bin/env python3
"""CPU-only structural audit for the prospective v3.3.4.3 OOD sidecar.

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
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence


ANALYSIS_VERSION = 'v3.3.4.3-structural-analyzer-v1'
SCRIPT_VERSION = 'v3.3.4.3'
ATTEMPT_ID = 'v3.3.4.3-development-ood-sidecar-one-shot'
AMENDMENT_SHA256 = (
    'e1eb3418a17f2a784c6cf5389f1e9bb7858125a514cd661546ce67ab154cbf93'
)
AMENDMENT_COMMIT = '90ba822a4e6f24514f1f20515cc4f4cf4fb84aa3'
V3342_AMENDMENT_SHA256 = (
    '1d2109e58d11cb07e99490bfde5fbb5d5ab43bd12e429c28b4ca9dfc0656fb87'
)
V3342_AMENDMENT_COMMIT = 'f48d6b73839b428fe950b00696548b6410a52659'
PREDECESSOR_AMENDMENT_SHA256 = (
    '38d07c0b612e50aadc64ba18537561cbdb0489b67fd0824cae749bba6214207b'
)
PREDECESSOR_AMENDMENT_COMMIT = 'f833a8d2108636871abfce8b4cbabe4255536974'
PUBLICATION_AMENDMENT_SHA256 = (
    '6abc470f6fb14b70c8930195bb8f26ce730b8c07c636cd842d5451f37d8eb55c'
)
PUBLICATION_AMENDMENT_COMMIT = '2b5e3e93a9961ac7cb12c088f6922acc9fdc5dde'
PUBLICATION_SCHEMA_VERSION = 'v3.3.4.3-named-temp-renameat2-noreplace-v1'
PUBLICATION_METHOD = 'named_temp_renameat2_noreplace'
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
ENTRY_STATE_KEYS = (
    'state', 'entry_type', 'mode', 'size_bytes', 'sha256', 'st_dev',
    'st_ino', 'st_nlink',
)
PUBLICATION_AUDIT_KEYS = (
    'schema_version', 'method', 'successful_final_count_before_terminal',
    'successful_final_bindings_before_terminal', 'temporary_orphan_count',
    'temporary_orphan_bindings', 'durability_uncertain_final_count',
    'durability_uncertain_final_bindings', 'preexisting_entry_count',
    'preexisting_entry_states', 'no_new_entry_failure',
    'publication_failure', 'no_published_final_deleted',
    'no_temp_or_final_reused', 'no_publication_retry',
)
PUBLICATION_FAILURE_STAGES = frozenset({
    'parent_open', 'parent_validation', 'final_preexistence', 'temp_open',
    'temp_validation', 'write', 'first_file_fsync', 'fchmod',
    'second_file_fsync', 'readback', 'rename_noreplace',
    'post_rename_validation', 'parent_fsync', 'final_revalidation',
})
TERMINAL_FAILURE_KEYS = frozenset({
    'schema_version', 'status', 'stop_reason', 'attempt_id', 'script_version',
    'external_freeze_authorization', 'runner_pid', 'publication_failure',
    'preterminal_tree_binding', 'source_input_audit',
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
})
NONPUBLICATION_TERMINAL_KEYS = frozenset({
    'schema_version', 'status', 'stop_reason', 'attempt_id', 'script_version',
    'amendment_commit', 'amendment_sha256', 'inherited_v3_3_4_commit',
    'inherited_v3_3_4_sha256', 'inherited_v3_3_4_1_commit',
    'inherited_v3_3_4_1_sha256', 'freeze_sha256', 'git_head',
    'external_freeze_authorization', 'runner_pid', 'started_at_unix_s',
    'created_at_unix_s', 'failure_stage', 'failure',
    'triggering_diagnostic_failure', 'triggering_diagnostic_stop_reason',
    'phase_state', 'source_input_audit',
    'source_input_audit_content_binding',
    'program_signature_attestation_binding', 'same_object_attestation',
    'same_object_attestation_content_binding', 'attempt_budget_audit',
    'compiler_counts', 'graph_artifact_bindings',
    'import_provenance_phases', 'protobuf_provenance_sha256',
    'model_kernel_cache_state',
    'source_program_gate_without_backend_diagnostics',
    'source_program_gate_without_backend_diagnostics_content_binding',
    'prior_v3_3_3_binding', 'prior_v3_3_3_1_archive_binding',
    'preterminal_tree_binding', 'publication_audit',
    'model_apply_attempt_count', 'model_apply_success_count',
    'valid_record_count', 'raw_record_count', 'dispatch_started_count',
    'dispatch_completed_count', 'six_row_compile_count',
    'identity_rerun_count', 'main_cube_rerun_count',
    'old_ood_records_reused', 'confirmation_model_calls',
    'confirmation_scope_disclosure', 'scientific_summary_computed',
    'donor_normalization_computed', 'shapley_or_nomination_computed',
    'interaction_or_resolution_computed', 'nomination_performed',
    'combined_analysis_permitted', 'no_retry',
})
NONPUBLICATION_TERMINAL_KEY_ORDER = (
    'schema_version', 'status', 'stop_reason', 'attempt_id',
    'script_version', 'amendment_commit', 'amendment_sha256',
    'inherited_v3_3_4_commit', 'inherited_v3_3_4_sha256',
    'inherited_v3_3_4_1_commit', 'inherited_v3_3_4_1_sha256',
    'freeze_sha256', 'git_head', 'external_freeze_authorization',
    'runner_pid', 'started_at_unix_s', 'created_at_unix_s',
    'failure_stage', 'failure', 'triggering_diagnostic_failure',
    'triggering_diagnostic_stop_reason', 'phase_state',
    'source_input_audit', 'source_input_audit_content_binding',
    'program_signature_attestation_binding', 'same_object_attestation',
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
)
if frozenset(NONPUBLICATION_TERMINAL_KEY_ORDER) != NONPUBLICATION_TERMINAL_KEYS:
  raise RuntimeError('NONPUBLICATION terminal key order/set differs.')
NONPUBLICATION_FAILURE_STAGES = frozenset({
    'stablehlo_text_extraction', 'pre_backend_hlo_text_extraction',
    'compiled_hlo_text_extraction',
    'source_program_gate_derivation_for_diagnostic_failure',
    'diagnostic_failure_record_construction',
})
DIAGNOSTIC_STOP_REASONS = frozenset({
    'diagnostic_parser_failure', 'diagnostic_persistence_failure',
    'cache_signal_unavailable', 'fingerprint_formula_mismatch',
})
DIAGNOSTIC_TRIGGER_TYPE_TO_REASON = {
    'EntryAbiParserFailure': 'diagnostic_parser_failure',
    'BackendDiagnosticParserFailure': 'diagnostic_parser_failure',
    'DiagnosticPersistenceFailure': 'diagnostic_persistence_failure',
    'CacheSignalUnavailable': 'cache_signal_unavailable',
    'FingerprintFormulaMismatch': 'fingerprint_formula_mismatch',
}
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
    / 'encoder_skip_ood_sidecar_stage_semantics_amendment_v3_3_4_3.md'
)
_V3342_AMENDMENT_PATH = (
    _HERE / 'v3_wider_mechanism'
    / 'encoder_skip_ood_sidecar_nonpublication_terminal_amendment_v3_3_4_2.md'
)
_PREDECESSOR_AMENDMENT_PATH = (
    _HERE / 'v3_wider_mechanism'
    / 'encoder_skip_ood_sidecar_infrastructure_amendment_v3_3_4.md'
)
_PUBLICATION_AMENDMENT_PATH = (
    _HERE / 'v3_wider_mechanism'
    / 'encoder_skip_ood_sidecar_publication_amendment_v3_3_4_1.md'
)
_FREEZE_PATH = _HERE / 'encoder_skip_ood_sidecar_v3_3_4_3_freeze.json'
_RUN_DIR = _HERE / 'results/v3_3_4_3_development_ood_sidecar_one_shot'
_ANALYSIS_DIR = _HERE / 'results/v3_3_4_3_development_ood_sidecar_analysis'
_ANALYSIS_ATTEMPT_DIR = (
    _HERE / 'results/v3_3_4_3_development_ood_sidecar_analysis_attempt'
)
_PREFLIGHT_DIR = _HERE / 'results/v3_3_4_3_device_preflight'
_PREFLIGHT_CACHE_DIR = _HERE / 'results/v3_3_4_3_preflight_kernel_cache'
_MODEL_CACHE_DIR = _HERE / 'results/v3_3_4_3_model_kernel_cache'
_TEST_PATH = _HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_4_3_test.py'
_PRIOR_RUN_DIR = _HERE / 'results/v3_3_3_development_ood_sidecar_one_shot'
_ORIGINAL_CUBE_DIR = (
    _HERE / 'results/v3_3_development_encoder_skip_factorial_one_shot'
)
_PRIOR_PREFLIGHT_DIR = _HERE / 'results/v3_3_3_device_preflight'
_PRIOR_CACHE_DIR = _HERE / 'results/v3_3_3_model_kernel_cache'
_PRIOR_FREEZE_PATH = _HERE / 'encoder_skip_ood_sidecar_v3_3_3_freeze.json'
_PRIOR_ANALYZER_ATTEMPT_DIR = (
    _HERE / 'results/v3_3_3_development_ood_sidecar_analysis_attempt'
)
_PRIOR_ANALYZER_OUTPUT_DIR = (
    _HERE / 'results/v3_3_3_development_ood_sidecar_analysis'
)
_PRIOR_331_ATTEMPT_DIR = (
    _HERE / 'results/v3_3_3_development_ood_sidecar_analysis_v3_3_3_1_attempt'
)
_PRIOR_331_OUTPUT_DIR = (
    _HERE / 'results/v3_3_3_development_ood_sidecar_analysis_v3_3_3_1'
)
_CASES_PATH = _HERE / 'superset_graph_v3_2_development_variants.tsv'
_EXONS_PATH = _HERE / 'superset_graph_v3_2_development_exons.tsv'
_ANALYSIS_ATTEMPT_TOKEN = object()

_V334_PREDECESSOR_PRODUCTION_PATHS = (
    _HERE / 'results/v3_3_4_development_ood_sidecar_one_shot',
    _HERE / 'results/v3_3_4_device_preflight',
    _HERE / 'results/v3_3_4_preflight_kernel_cache',
    _HERE / 'results/v3_3_4_model_kernel_cache',
    _HERE / 'results/v3_3_4_development_ood_sidecar_analysis',
    _HERE / 'results/v3_3_4_development_ood_sidecar_analysis_attempt',
    _HERE / 'results/v3_3_4_1_development_ood_sidecar_one_shot',
    _HERE / 'results/v3_3_4_1_device_preflight',
    _HERE / 'results/v3_3_4_1_preflight_kernel_cache',
    _HERE / 'results/v3_3_4_1_model_kernel_cache',
    _HERE / 'results/v3_3_4_1_development_ood_sidecar_analysis',
    _HERE / 'results/v3_3_4_1_development_ood_sidecar_analysis_attempt',
    _HERE / 'results/v3_3_4_2_development_ood_sidecar_one_shot',
    _HERE / 'results/v3_3_4_2_device_preflight',
    _HERE / 'results/v3_3_4_2_preflight_kernel_cache',
    _HERE / 'results/v3_3_4_2_model_kernel_cache',
    _HERE / 'results/v3_3_4_2_development_ood_sidecar_analysis',
    _HERE / 'results/v3_3_4_2_development_ood_sidecar_analysis_attempt',
)

_V3343_SOURCE_PATHS = (
    'experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_3.py',
    'experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_3.sh',
    'experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_3_test.py',
    'experiments/interpretability/opensplice/generate_encoder_skip_ood_sidecar_v3_3_4_3_freeze.py',
    'experiments/interpretability/opensplice/launch_encoder_skip_ood_sidecar_v3_3_4_3.py',
    'experiments/interpretability/opensplice/run_device_preflight_v3_3_4_3.py',
    'experiments/interpretability/opensplice/run_device_preflight_v3_3_4_3_test.py',
    'experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_4_3.py',
    'experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_4_3.sh',
    'experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_4_3_test.py',
    'experiments/interpretability/opensplice/v3_wider_mechanism/encoder_skip_ood_sidecar_stage_semantics_amendment_v3_3_4_3.md',
    'experiments/interpretability/opensplice/validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_3.py',
)

PRIOR_SOURCE_COMMIT = '228083b931dbc62d4a283e68df01011f5ef4bff9'
PRIOR_FREEZE_SHA256 = (
    '0e4c16a306f734e016c64509a3b7f0d76f26baf399ee0b1d41c6fb073203741b'
)
PRIOR_RUN_TREE_SHA256 = (
    'bb13aa4de212c3896781401374057bc0cdfc0c7527772cc36b08b57c70451805'
)
PRIOR_COMPILER_TREE_SHA256 = (
    '7ee5ad1bb94ecbd97606fcccae3abcad6b0ebec74dd9f983d81b4fc179142ef0'
)
PRIOR_PREFLIGHT_TREE_SHA256 = (
    'f2bae99e3b0a59a50419e0507146e26f4eea1c67f2595ddccec4e8d5aef7a0e1'
)
PRIOR_CACHE_TREE_SHA256 = (
    'a1dafb75097282b3d28174e4bb6c79a81d2f1c8c25cd8c64a1ea84642ba7f43a'
)
PRIOR_331_AMENDMENT_COMMIT = 'd2a013944a399ddac59a023d7d84ea5a7c23e9f4'
PRIOR_331_IMPLEMENTATION_COMMIT = '98c467ae16200071d110c9d73520e35e5e6d7bbf'
PRIOR_331_ARCHIVE_COMMIT = '37bd58e88e1814f9a67bfbaaaad66d0a2b77f242'

_PRIOR_RUN_FILES = {
    'ATTEMPT_STARTED.json': (871_020, 'e5f7c33f2e8c82af51ed98a3884d7df83e1828e92e322df8aa8a054ec7464c65'),
    'IMPORT_PROVENANCE.json': (41_572, 'aa5072c505ebe54d0a7812a7fa3e6bda249a74c17d2b90df3f235c2f4cd6bb4b'),
    'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json': (41_572, 'aa5072c505ebe54d0a7812a7fa3e6bda249a74c17d2b90df3f235c2f4cd6bb4b'),
    'IMPORT_PROVENANCE_PRE_MODEL.json': (41_572, 'aa5072c505ebe54d0a7812a7fa3e6bda249a74c17d2b90df3f235c2f4cd6bb4b'),
    'PROTOBUF_PROVENANCE.json': (3_339, '2498a940f6ee15e54e72e8f51587d4c42ffc1b49851873c31ad09085315d0ba8'),
    'RAW_MANIFEST.json': (145, 'fadbff5ab512dea3d25edb39dc9a91ffe25473b73f1e33fd53f97b60fc8436fd'),
    'RUN_COMPLETE.json': (227_159, '43e0ff055e9f7fa4032a75120c551a2b5762e4fbd85119e80e3694f8b9f54bba'),
    'compiler/eight_row/COMPILER_PROVENANCE.json': (102_245, 'ae07b0f10784ea3c6dd26d2b87eb718c5e28d3834112ae4f0566d1c4fb7e3125'),
    'compiler/eight_row/graph.compiled.hlo.txt': (16_603_075, 'f0fe2fa0b7e8326390c8f2ed38ce52ef6c64c355bc462450d85d3b2f040645f4'),
    'compiler/eight_row/graph.pre_backend.hlo.txt': (1_829_833, SOURCE_PRE_BACKEND_HLO['sha256']),
    'compiler/eight_row/graph.stablehlo.mlir': (3_196_162, SOURCE_STABLEHLO['sha256']),
}
_PRIOR_PREFLIGHT_FILES = {
    '.allocation.lock': (0, EMPTY_SHA256),
    '.preflight_0000.reserved': (0, EMPTY_SHA256),
    'preflight_0000.json': (704_213, '79e2c9937025830b309854cff4f5c93c607b7574fb44a9d51f45564b14246224'),
    'preflight_0000.stderr.log': (0, EMPTY_SHA256),
    'preflight_0000.stdout.log': (0, EMPTY_SHA256),
}
_PRIOR_CACHE_FILES = {
    'xdg/matplotlib/fontlist-v3.11.0.json': (
        163_240, 'a777469f8f54be8cc9107788bce2b3cd23709aa317114392cdc365b3fb127125'
    ),
}
_PRIOR_331_SOURCES = {
    'experiments/interpretability/opensplice/v3_wider_mechanism/encoder_skip_ood_sidecar_analysis_amendment_v3_3_3_1.md': '4d2957d144e56e58c5b2058076bbcdb7f1495f3172e1b8829a0affa10a0ea4a9',
    'experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_3_1.py': 'f433221f38408ee06d3bdb2c1119ae050720652ee4ec513a0b91f2d7814da063',
    'experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_3_1_test.py': '4f2e70a8f61bb1b9af7b2b98ef8f450d0937855a69d1bc83fbec9d06f21dd971',
    'experiments/interpretability/opensplice/encoder_skip_ood_sidecar_analysis_v3_3_3_1_freeze.json': '96c599f3c607107b8c7ab235d7c8cef7aa1bc544189b44b15b6f3fbf1a8b3291',
    'experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_analysis_v3_3_3_1.sh': '63a0cc95596d47ee5900fe928e1bb42115b18157f87bdae45000e5bb7ccef5c9',
}
_PRIOR_331_ATTEMPT_FILES = {
    'ANALYSIS_ATTEMPT_STARTED.json': (6_512, '497374d68c245c30fb0a54968859b9066d1bc16085146b978070bb092ff23bda'),
    'ANALYSIS_COMPLETE.json': (1_179, 'e050e091743262e989693c59f5e1fcb2939190a71ee4851c5d2a345c1827c4be'),
}
_PRIOR_331_OUTPUT_FILES = {
    'ANALYSIS.json': (10_060, 'f1e20b3ca4f111854b22eff1e2cd2ffdb05796d800d2831eedcc6caa1a3b7245'),
    'RESULT.md': (695, '8ba2721c8bc350a564f4d5ffdabd65b118f60d92cbdb8ea00a8d040842012e65'),
}
PRIOR_331_ATTEMPT_TREE_SHA256 = 'cff8dd5418405dd1acef9c6de1d1e2688e63a6807b1ff4e1ef0c8b8908229307'
PRIOR_331_OUTPUT_TREE_SHA256 = '4dcbaa9069b130d160efbde95b1f82b3561ea90d2a38923d259978126e889b2c'


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


def _guard_path(path: Path) -> None:
  for part in path.resolve().parts:
    lowered = part.lower()
    if 'confirm' in lowered or lowered in {'eln', 'eif4a2', 'dmd'}:
      raise AnalysisError(f'Refusing confirmation path: {path}.')


def _assert_predecessor_v334_paths_absent(label: str) -> None:
  """Proves that the never-launched v3.3.4 namespace stays untouched."""
  for path in _V334_PREDECESSOR_PRODUCTION_PATHS:
    if path.exists() or path.is_symlink():
      raise AnalysisError(
          f'{label}: predecessor v3.3.4 production path appeared: {path}.'
      )


def _validate_analysis_destination_state(
    active_started_sha256: str | None,
) -> None:
  """Allows only a fresh destination or the exact active START singleton."""
  if _ANALYSIS_DIR.exists() or _ANALYSIS_DIR.is_symlink():
    raise AnalysisError('v3.3.4.3 analysis output destination is not fresh.')
  attempt_exists = (
      _ANALYSIS_ATTEMPT_DIR.exists() or _ANALYSIS_ATTEMPT_DIR.is_symlink()
  )
  if active_started_sha256 is None:
    if attempt_exists:
      raise AnalysisError('v3.3.4.3 analysis attempt destination is not fresh.')
    return
  if not _is_sha256(active_started_sha256) or not attempt_exists:
    raise AnalysisError('v3.3.4.3 active analysis attempt is absent.')
  paths = _strict_tree(
      _ANALYSIS_ATTEMPT_DIR, {'ANALYSIS_ATTEMPT_STARTED.json'},
      'active analysis-attempt tree during freeze validation',
  )
  if (
      stat.S_IMODE(paths[0].lstat().st_mode) != 0o400
      or _sha256(paths[0]) != active_started_sha256
  ):
    raise AnalysisError('v3.3.4.3 active analysis START changed.')


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


def _relative_publication_path(value: Any, label: str) -> str:
  if not isinstance(value, str) or not value or '\x00' in value:
    raise AnalysisError(f'{label} is not a nonempty relative path.')
  path = Path(value)
  if path.is_absolute() or value != path.as_posix() or '..' in path.parts:
    raise AnalysisError(f'{label} escaped its frozen root.')
  return value


def _validate_file_publication_binding(
    value: Any, label: str, *, expected_mode: str | None = None,
) -> dict[str, Any]:
  node = _exact_keys(
      value, {'sha256', 'size_bytes', 'mode', 'st_dev', 'st_ino', 'st_nlink'},
      label,
  )
  if not _is_sha256(node.get('sha256')):
    raise AnalysisError(f'{label}.sha256 is malformed.')
  for key in ('size_bytes', 'st_dev', 'st_ino', 'st_nlink'):
    item = node.get(key)
    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
      raise AnalysisError(f'{label}.{key} is malformed.')
  if node['st_nlink'] != 1:
    raise AnalysisError(f'{label}.st_nlink is not one.')
  mode = node.get('mode')
  if not isinstance(mode, str) or re.fullmatch(r'[0-7]{4}', mode) is None:
    raise AnalysisError(f'{label}.mode is malformed.')
  if expected_mode is not None and mode != expected_mode:
    raise AnalysisError(f'{label}.mode changed.')
  return dict(node)


def _validate_entry_state(value: Any, label: str) -> dict[str, Any]:
  node = _exact_keys(value, set(ENTRY_STATE_KEYS), label)
  state = node.get('state')
  if state not in {'absent', 'present', 'unreadable'}:
    raise AnalysisError(f'{label}.state changed.')
  detail_keys = set(ENTRY_STATE_KEYS) - {'state'}
  if state in {'absent', 'unreadable'}:
    if any(node.get(key) is not None for key in detail_keys):
      raise AnalysisError(f'{label} has details for {state} state.')
    return dict(node)
  entry_type = node.get('entry_type')
  if entry_type not in {
      'regular', 'directory', 'symlink', 'fifo', 'socket', 'block',
      'character', 'other',
  }:
    raise AnalysisError(f'{label}.entry_type changed.')
  if (
      not isinstance(node.get('mode'), str)
      or re.fullmatch(r'[0-7]{4}', node['mode']) is None
  ):
    raise AnalysisError(f'{label}.mode is malformed.')
  for key in ('st_dev', 'st_ino', 'st_nlink'):
    item = node.get(key)
    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
      raise AnalysisError(f'{label}.{key} is malformed.')
  if entry_type == 'regular':
    if (
        isinstance(node.get('size_bytes'), bool)
        or not isinstance(node.get('size_bytes'), int)
        or node['size_bytes'] < 0
        or not _is_sha256(node.get('sha256'))
    ):
      raise AnalysisError(f'{label} regular-file evidence is malformed.')
  elif node.get('size_bytes') is not None or node.get('sha256') is not None:
    raise AnalysisError(f'{label} non-regular entry has file-byte evidence.')
  return dict(node)


def _observe_entry_state(path: Path) -> dict[str, Any]:
  try:
    status = path.lstat()
  except FileNotFoundError:
    return {'state': 'absent', **{key: None for key in ENTRY_STATE_KEYS[1:]}}
  except OSError:
    return {'state': 'unreadable', **{key: None for key in ENTRY_STATE_KEYS[1:]}}
  mode = status.st_mode
  if stat.S_ISREG(mode):
    entry_type = 'regular'
  elif stat.S_ISDIR(mode):
    entry_type = 'directory'
  elif stat.S_ISLNK(mode):
    entry_type = 'symlink'
  elif stat.S_ISFIFO(mode):
    entry_type = 'fifo'
  elif stat.S_ISSOCK(mode):
    entry_type = 'socket'
  elif stat.S_ISBLK(mode):
    entry_type = 'block'
  elif stat.S_ISCHR(mode):
    entry_type = 'character'
  else:
    entry_type = 'other'
  return {
      'state': 'present', 'entry_type': entry_type,
      'mode': f'{stat.S_IMODE(mode):04o}',
      'size_bytes': status.st_size if entry_type == 'regular' else None,
      'sha256': _sha256(path) if entry_type == 'regular' else None,
      'st_dev': status.st_dev, 'st_ino': status.st_ino,
      'st_nlink': status.st_nlink,
  }


def _validate_publication_failure(value: Any, label: str) -> dict[str, Any]:
  node = _exact_keys(value, set(PUBLICATION_FAILURE_KEYS), label)
  if (
      node.get('schema_version') != PUBLICATION_SCHEMA_VERSION
      or node.get('method') != PUBLICATION_METHOD
      or node.get('failure_stage') not in PUBLICATION_FAILURE_STAGES
  ):
    raise AnalysisError(f'{label} contract changed.')
  _relative_publication_path(node.get('final_relative_path'), f'{label}.final')
  temporary = node.get('temp_relative_path')
  _relative_publication_path(temporary, f'{label}.temporary')
  for key in ('publication_ordinal', 'runner_pid'):
    item = node.get(key)
    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
      raise AnalysisError(f'{label}.{key} is malformed.')
  if node['runner_pid'] < 1:
    raise AnalysisError(f'{label}.runner_pid is malformed.')
  if node['publication_ordinal'] >= 1_000_000:
    raise AnalysisError(f'{label}.publication_ordinal exceeds six digits.')
  final_path = Path(node['final_relative_path'])
  temporary_path = Path(temporary)
  if (
      final_path.parent != temporary_path.parent
      or final_path == temporary_path
      or re.fullmatch(
          rf'\.v3343\.tmp\.{node["runner_pid"]}\.'
          rf'{node["publication_ordinal"]:06d}\.[0-9a-f]{{32}}',
          temporary_path.name,
      ) is None
  ):
    raise AnalysisError(f'{label} temporary name/directory changed.')
  if node.get('errno') is not None and (
      isinstance(node['errno'], bool) or not isinstance(node['errno'], int)
  ):
    raise AnalysisError(f'{label}.errno is malformed.')
  for key in (
      'rename_noreplace_attempted', 'rename_noreplace_succeeded',
      'parent_fsync_attempted', 'parent_fsync_succeeded',
  ):
    if not isinstance(node.get(key), bool):
      raise AnalysisError(f'{label}.{key} is not boolean.')
  for key in ('root_role', 'artifact_role', 'error_type', 'message'):
    if not isinstance(node.get(key), str) or not node[key]:
      raise AnalysisError(f'{label}.{key} is malformed.')
  if node['root_role'] not in {
      'model_run', 'external_preflight', 'external_cache', 'model_cache',
      'analysis_output', 'analysis_attempt',
  }:
    raise AnalysisError(f'{label}.root_role changed.')
  _finite(node.get('created_at_unix_s'), f'{label}.created_at_unix_s')
  _validate_entry_state(node.get('temp_state'), f'{label}.temp_state')
  _validate_entry_state(node.get('final_state'), f'{label}.final_state')
  if node['rename_noreplace_succeeded'] and not node['rename_noreplace_attempted']:
    raise AnalysisError(f'{label} claims rename success without an attempt.')
  if node['parent_fsync_succeeded'] and not node['parent_fsync_attempted']:
    raise AnalysisError(f'{label} claims parent fsync success without an attempt.')
  return dict(node)


def _validate_publication_binding_map(
    value: Any, label: str, *, expected_mode: str | None = None,
) -> dict[str, dict[str, Any]]:
  if not isinstance(value, Mapping):
    raise AnalysisError(f'{label} is not a binding map.')
  if list(value) != sorted(value):
    raise AnalysisError(f'{label} is not POSIX-sorted.')
  result = {}
  for relative, binding in value.items():
    _relative_publication_path(relative, f'{label}.path')
    result[relative] = _validate_file_publication_binding(
        binding, f'{label}[{relative}]', expected_mode=expected_mode,
    )
  return result


def _validate_entry_state_map(value: Any, label: str) -> dict[str, Any]:
  if not isinstance(value, Mapping) or list(value) != sorted(value):
    raise AnalysisError(f'{label} is not a sorted state map.')
  result = {}
  for relative, state in value.items():
    _relative_publication_path(relative, f'{label}.path')
    result[relative] = _validate_entry_state(state, f'{label}[{relative}]')
  return result


def _validate_live_publication_file(
    root: Path, value: Any, label: str, *, expected_path: str | None = None,
) -> dict[str, Any]:
  node = _exact_keys(
      value,
      {'path', 'sha256', 'size_bytes', 'mode', 'st_dev', 'st_ino', 'st_nlink'},
      label,
  )
  relative = _relative_publication_path(node.get('path'), f'{label}.path')
  if expected_path is not None and relative != expected_path:
    raise AnalysisError(f'{label}.path changed.')
  binding = _validate_file_publication_binding(
      {key: node[key] for key in node if key != 'path'}, label,
      expected_mode='0400',
  )
  path = root / relative
  _strict_regular(path, label)
  status = path.lstat()
  observed = {
      'sha256': _sha256(path), 'size_bytes': status.st_size,
      'mode': f'{stat.S_IMODE(status.st_mode):04o}', 'st_dev': status.st_dev,
      'st_ino': status.st_ino, 'st_nlink': status.st_nlink,
  }
  if binding != observed:
    raise AnalysisError(f'{label} differs from the live no-follow file.')
  return {'path': relative, **binding}


def _validate_atomic_publication_probe(
    value: Any, *, external_pid: int,
) -> dict[str, Any]:
  node = _exact_keys(value, {
      'schema_version', 'method', 'supported', 'successful_final_binding',
      'collision_errno', 'collision_no_replace_exact',
      'collision_temp_binding', 'destination_unchanged',
      'temp_orphan_preserved', 'parent_fsync_exact',
  }, 'external preflight.atomic_publication_probe')
  if (
      node.get('schema_version') != PUBLICATION_SCHEMA_VERSION
      or node.get('method') != PUBLICATION_METHOD
      or node.get('supported') is not True
      or node.get('collision_errno') != 17
      or any(node.get(key) is not True for key in (
          'collision_no_replace_exact', 'destination_unchanged',
          'temp_orphan_preserved', 'parent_fsync_exact'
      ))
  ):
    raise AnalysisError('External atomic-publication probe did not pass exactly.')
  final = _validate_live_publication_file(
      _PREFLIGHT_CACHE_DIR, node.get('successful_final_binding'),
      'atomic-publication final',
      expected_path='atomic_publication_probe_v3_3_4_3.txt',
  )
  collision = _validate_live_publication_file(
      _PREFLIGHT_CACHE_DIR, node.get('collision_temp_binding'),
      'atomic-publication collision temporary',
  )
  if (
      final['sha256']
      != '29c4509405ae8afa833da03d2ff5824fc2b842c6bc56df4d0aac2f7838685ea1'
      or final['size_bytes'] != 49
      or collision['sha256']
      != 'c9ceb22bdeed1f370e6ca33814313f3a3d5007e286954aa3e4f31e5d041b8ba0'
      or collision['size_bytes'] != 39
      or re.fullmatch(
          rf'\.v3343\.tmp\.{external_pid}\.[0-9]{{6}}\.[0-9a-f]{{32}}',
          collision['path'],
      ) is None
  ):
    raise AnalysisError('External atomic-publication probe bytes/name changed.')
  return dict(node)


def _publication_tree_binding(
    root: Path, *, role: str, expected_files: set[str],
) -> dict[str, Any]:
  """Builds the exact no-follow file/directory tree used in publications."""
  if root.is_symlink() or not root.is_dir():
    raise AnalysisError(f'{role} root is absent or unsafe.')
  if stat.S_IMODE(root.lstat().st_mode) != 0o700:
    raise AnalysisError(f'{role} root mode changed.')
  files: dict[str, dict[str, Any]] = {}
  directories = ['.']
  for entry in sorted(root.rglob('*')):
    relative = entry.relative_to(root).as_posix()
    mode = entry.lstat().st_mode
    if stat.S_ISLNK(mode):
      raise AnalysisError(f'{role} contains a symlink: {relative}.')
    if stat.S_ISDIR(mode):
      if stat.S_IMODE(mode) != 0o700:
        raise AnalysisError(f'{role} directory mode changed: {relative}.')
      directories.append(relative)
      continue
    if not stat.S_ISREG(mode) or stat.S_IMODE(mode) != 0o400:
      raise AnalysisError(f'{role} contains an unsafe file: {relative}.')
    status = entry.lstat()
    files[relative] = {
        'sha256': _sha256(entry), 'size_bytes': status.st_size,
        'mode': f'{stat.S_IMODE(status.st_mode):04o}',
        'st_dev': status.st_dev, 'st_ino': status.st_ino,
        'st_nlink': status.st_nlink,
    }
    _validate_file_publication_binding(
        files[relative], f'{role}[{relative}]', expected_mode='0400'
    )
  if set(files) != expected_files:
    raise AnalysisError(f'{role} exact file membership changed.')
  directories = sorted(directories)
  directory_digest = hashlib.sha256()
  for relative in directories:
    directory_digest.update(relative.encode('utf-8'))
    directory_digest.update(b'\0')
    directory_digest.update(b'0700')
  return {
      'root_role': role, 'file_count': len(files),
      'directory_count': len(directories), 'file_bindings': files,
      'file_tree_sha256': _binding_map_digest(files),
      'directory_paths': directories,
      'directory_tree_sha256': directory_digest.hexdigest(),
  }


def _validate_publication_tree_binding(value: Any, label: str) -> dict[str, Any]:
  node = _exact_keys(value, {
      'root_role', 'file_count', 'directory_count', 'file_bindings',
      'file_tree_sha256', 'directory_paths', 'directory_tree_sha256',
  }, label)
  if not isinstance(node.get('root_role'), str) or not node['root_role']:
    raise AnalysisError(f'{label}.root_role is malformed.')
  bindings = _validate_publication_binding_map(
      node.get('file_bindings'), f'{label}.file_bindings',
      expected_mode='0400',
  )
  directories = node.get('directory_paths')
  if (
      node.get('file_count') != len(bindings)
      or not isinstance(directories, list)
      or directories != sorted(set(directories))
      or not directories or directories[0] != '.'
      or node.get('directory_count') != len(directories)
      or node.get('file_tree_sha256') != _binding_map_digest(bindings)
  ):
    raise AnalysisError(f'{label} counts/tree changed.')
  digest = hashlib.sha256()
  for relative in directories:
    _relative_publication_path(relative, f'{label}.directory') if relative != '.' else None
    digest.update(relative.encode('utf-8'))
    digest.update(b'\0')
    digest.update(b'0700')
  if node.get('directory_tree_sha256') != digest.hexdigest():
    raise AnalysisError(f'{label}.directory_tree_sha256 changed.')
  return dict(node)


def _analysis_root_maps(
    attempt: Mapping[str, Mapping[str, Any]],
    output: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
  return {
      'analysis_attempt': copy.deepcopy(dict(attempt)),
      'analysis_output': copy.deepcopy(dict(output)),
  }


def _analysis_publication_audit(
    *, attempt_tree: Mapping[str, Any], output_tree: Mapping[str, Any],
    publication_failure: Mapping[str, Any] | None = None,
    temporary_orphans: Mapping[str, Mapping[str, Any]] | None = None,
    uncertain_finals: Mapping[str, Mapping[str, Any]] | None = None,
    preexisting: Mapping[str, Mapping[str, Any]] | None = None,
    no_new_entry_failure: bool = False,
) -> dict[str, Any]:
  attempt_tree = _validate_publication_tree_binding(
      attempt_tree, 'analysis publication attempt tree'
  )
  output_tree = _validate_publication_tree_binding(
      output_tree, 'analysis publication output tree'
  )
  successful = _analysis_root_maps(
      attempt_tree['file_bindings'], output_tree['file_bindings']
  )
  temporary = _analysis_root_maps({}, {}) if temporary_orphans is None else copy.deepcopy(dict(temporary_orphans))
  uncertain = _analysis_root_maps({}, {}) if uncertain_finals is None else copy.deepcopy(dict(uncertain_finals))
  existing = _analysis_root_maps({}, {}) if preexisting is None else copy.deepcopy(dict(preexisting))
  for label, value, state_map in (
      ('temporary_orphan_bindings', temporary, False),
      ('durability_uncertain_final_bindings', uncertain, False),
      ('preexisting_entry_states', existing, True),
  ):
    _exact_keys(value, {'analysis_attempt', 'analysis_output'}, label)
    for root_role in ('analysis_attempt', 'analysis_output'):
      if state_map:
        _validate_entry_state_map(value[root_role], f'{label}.{root_role}')
      else:
        _validate_publication_binding_map(value[root_role], f'{label}.{root_role}')
  failure = (
      None if publication_failure is None
      else _validate_publication_failure(publication_failure, 'analysis publication failure')
  )
  result = {
      'schema_version': PUBLICATION_SCHEMA_VERSION,
      'method': PUBLICATION_METHOD,
      'successful_final_count_before_terminal': sum(
          len(item) for item in successful.values()
      ),
      'successful_final_bindings_before_terminal': successful,
      'temporary_orphan_count': sum(len(item) for item in temporary.values()),
      'temporary_orphan_bindings': temporary,
      'durability_uncertain_final_count': sum(len(item) for item in uncertain.values()),
      'durability_uncertain_final_bindings': uncertain,
      'preexisting_entry_count': sum(len(item) for item in existing.values()),
      'preexisting_entry_states': existing,
      'no_new_entry_failure': no_new_entry_failure,
      'publication_failure': failure,
      'no_published_final_deleted': True, 'no_temp_or_final_reused': True,
      'no_publication_retry': True,
      'analysis_attempt_tree_binding': copy.deepcopy(dict(attempt_tree)),
      'analysis_output_tree_binding': copy.deepcopy(dict(output_tree)),
  }
  return _validate_analysis_publication_audit(result)


def _validate_analysis_publication_audit(value: Any) -> dict[str, Any]:
  node = _exact_keys(
      value,
      set(PUBLICATION_AUDIT_KEYS)
      | {'analysis_attempt_tree_binding', 'analysis_output_tree_binding'},
      'ANALYSIS.publication_audit',
  )
  if (
      node.get('schema_version') != PUBLICATION_SCHEMA_VERSION
      or node.get('method') != PUBLICATION_METHOD
      or node.get('no_published_final_deleted') is not True
      or node.get('no_temp_or_final_reused') is not True
      or node.get('no_publication_retry') is not True
      or not isinstance(node.get('no_new_entry_failure'), bool)
  ):
    raise AnalysisError('ANALYSIS publication fixed predicates changed.')
  for key, states in (
      ('successful_final_bindings_before_terminal', False),
      ('temporary_orphan_bindings', False),
      ('durability_uncertain_final_bindings', False),
      ('preexisting_entry_states', True),
  ):
    roots = _exact_keys(
        node.get(key), {'analysis_attempt', 'analysis_output'},
        f'ANALYSIS.publication_audit.{key}',
    )
    for role, mapping in roots.items():
      if states:
        _validate_entry_state_map(mapping, f'ANALYSIS publication {key}.{role}')
      else:
        _validate_publication_binding_map(
            mapping, f'ANALYSIS publication {key}.{role}'
        )
  counts = {
      'successful_final_count_before_terminal': 'successful_final_bindings_before_terminal',
      'temporary_orphan_count': 'temporary_orphan_bindings',
      'durability_uncertain_final_count': 'durability_uncertain_final_bindings',
      'preexisting_entry_count': 'preexisting_entry_states',
  }
  for count_key, map_key in counts.items():
    expected = sum(len(item) for item in node[map_key].values())
    if node.get(count_key) != expected:
      raise AnalysisError(f'ANALYSIS publication {count_key} changed.')
  if node.get('publication_failure') is None:
    if any(node[key] for key in (
        'temporary_orphan_count', 'durability_uncertain_final_count',
        'preexisting_entry_count'
    )) or node.get('no_new_entry_failure') is not False:
      raise AnalysisError('Successful ANALYSIS publication has failure state.')
  else:
    _validate_publication_failure(
        node['publication_failure'], 'ANALYSIS publication failure'
    )
  _validate_publication_tree_binding(
      node.get('analysis_attempt_tree_binding'),
      'ANALYSIS publication attempt tree',
  )
  _validate_publication_tree_binding(
      node.get('analysis_output_tree_binding'),
      'ANALYSIS publication output tree',
  )
  return dict(node)


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








def _binding_map(rows: Mapping[str, tuple[int, str]]) -> dict[str, dict[str, Any]]:
  return {
      relative: {'size_bytes': size, 'sha256': digest}
      for relative, (size, digest) in rows.items()
  }


def _directory_tree_digest(
    root: Path, relative_dirs: Sequence[str], files: Mapping[str, Any]
) -> str:
  digest = hashlib.sha256()
  for relative in sorted(relative_dirs):
    digest.update(b'D\0')
    digest.update(relative.encode('utf-8'))
    digest.update(b'\0')
  for relative in sorted(files):
    digest.update(b'F\0')
    digest.update(relative.encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(_sha256(root / relative)))
  return digest.hexdigest()


def _validate_prior_v3_3_3() -> dict[str, Any]:
  if _sha256(_PRIOR_FREEZE_PATH) != PRIOR_FREEZE_SHA256:
    raise AnalysisError('Immutable v3.3.3 freeze changed.')
  freeze = _read_json(_PRIOR_FREEZE_PATH, 'v3.3.3 freeze')
  sources = freeze.get('file_sha256')
  if not isinstance(sources, Mapping) or len(sources) != 96:
    raise AnalysisError('Immutable v3.3.3 source inventory is not 96 rows.')
  for relative, digest in sources.items():
    if not isinstance(relative, str) or not _is_sha256(digest):
      raise AnalysisError('Immutable v3.3.3 source inventory is malformed.')
    path = _REPO_ROOT / relative
    _strict_regular(path, f'v3.3.3 source {relative}')
    if (
        _sha256(path) != digest
        or _git_blob_sha256(PRIOR_SOURCE_COMMIT, relative) != digest
    ):
      raise AnalysisError(f'Immutable v3.3.3 source changed: {relative}.')
  run_files = _binding_map(_PRIOR_RUN_FILES)
  run = _validate_bound_tree(
      _PRIOR_RUN_DIR, run_files, PRIOR_RUN_TREE_SHA256, 'v3.3.3 run'
  )
  compiler_paths = [
      _PRIOR_RUN_DIR / relative for relative in run_files
      if relative.startswith('compiler/')
  ]
  if (
      len(compiler_paths) != 4
      or _tree_digest(compiler_paths, _PRIOR_RUN_DIR)
      != PRIOR_COMPILER_TREE_SHA256
  ):
    raise AnalysisError('Immutable v3.3.3 compiler tree changed.')
  preflight = _validate_bound_tree(
      _PRIOR_PREFLIGHT_DIR, _binding_map(_PRIOR_PREFLIGHT_FILES),
      PRIOR_PREFLIGHT_TREE_SHA256, 'v3.3.3 preflight',
  )
  cache_paths = _strict_tree(
      _PRIOR_CACHE_DIR, set(_PRIOR_CACHE_FILES), 'v3.3.3 cache'
  )
  for relative, (size, digest) in _PRIOR_CACHE_FILES.items():
    path = _PRIOR_CACHE_DIR / relative
    if path.stat().st_size != size or _sha256(path) != digest:
      raise AnalysisError(f'Immutable v3.3.3 cache changed: {relative}.')
  if (
      len(cache_paths) != 1
      or _directory_tree_digest(
          _PRIOR_CACHE_DIR, ['.', 'triton', 'xdg', 'xdg/matplotlib'],
          _PRIOR_CACHE_FILES,
      ) != PRIOR_CACHE_TREE_SHA256
  ):
    raise AnalysisError('Immutable v3.3.3 cache tree changed.')
  manifest = _read_json(
      _PRIOR_RUN_DIR / 'RAW_MANIFEST.json', 'v3.3.3 RAW_MANIFEST'
  )
  if manifest != {
      'artifact_count': 0, 'artifact_sha256': {},
      'artifact_tree_sha256': EMPTY_SHA256,
  }:
    raise AnalysisError('Immutable v3.3.3 raw manifest changed.')
  completion = _read_json(
      _PRIOR_RUN_DIR / 'RUN_COMPLETE.json', 'v3.3.3 RUN_COMPLETE'
  )
  expected = {
      'status': 'controlled_stop',
      'stop_reason': 'source_program_mismatch',
      'model_apply_count': 0, 'six_row_compile_count': 0,
      'identity_rerun_count': 0, 'main_cube_rerun_count': 0,
      'old_ood_records_reused': 0, 'confirmation_model_calls': 0,
      'scientific_summary_computed': False,
      'shapley_or_nomination_computed': False,
      'nomination_performed': False, 'combined_analysis_permitted': False,
  }
  for key, value in expected.items():
    if completion.get(key) != value:
      raise AnalysisError(f'Immutable v3.3.3 terminal changed: {key}.')
  return {
      'source_commit': PRIOR_SOURCE_COMMIT,
      'freeze_sha256': PRIOR_FREEZE_SHA256, 'source_count': 96,
      'run': run, 'preflight': preflight,
      'cache_file_count': 1, 'cache_tree_sha256': PRIOR_CACHE_TREE_SHA256,
      'terminal_status': completion['status'],
      'terminal_stop_reason': completion['stop_reason'],
  }


def _validate_prior_v3_3_3_1() -> dict[str, Any]:
  sources = _validate_source_bundle(
      _PRIOR_331_SOURCES,
      implementation_commit=PRIOR_331_IMPLEMENTATION_COMMIT,
      amendment_commit=PRIOR_331_AMENDMENT_COMMIT,
  )
  attempt = _validate_bound_tree(
      _PRIOR_331_ATTEMPT_DIR, _binding_map(_PRIOR_331_ATTEMPT_FILES),
      PRIOR_331_ATTEMPT_TREE_SHA256, 'v3.3.3.1 attempt',
  )
  output = _validate_bound_tree(
      _PRIOR_331_OUTPUT_DIR, _binding_map(_PRIOR_331_OUTPUT_FILES),
      PRIOR_331_OUTPUT_TREE_SHA256, 'v3.3.3.1 output',
  )
  for root, rows in (
      (_PRIOR_331_ATTEMPT_DIR, _PRIOR_331_ATTEMPT_FILES),
      (_PRIOR_331_OUTPUT_DIR, _PRIOR_331_OUTPUT_FILES),
  ):
    for relative, (_size, digest) in rows.items():
      archive_relative = (root / relative).relative_to(_REPO_ROOT).as_posix()
      if _git_blob_sha256(PRIOR_331_ARCHIVE_COMMIT, archive_relative) != digest:
        raise AnalysisError(
            f'Immutable v3.3.3.1 archive blob changed: {archive_relative}.'
        )
  analysis = _read_json(
      _PRIOR_331_OUTPUT_DIR / 'ANALYSIS.json', 'v3.3.3.1 ANALYSIS'
  )
  expected = {
      'status': 'complete_controlled_stop_structural_archive',
      'decision': 'controlled_stop_source_program_mismatch_representation_only',
      'scientific_summary_computed': False,
      'donor_normalization_computed': False,
      'shapley_or_nomination_computed': False,
      'interaction_or_resolution_computed': False,
      'nomination_performed': False, 'combined_analysis_permitted': False,
  }
  for key, value in expected.items():
    if analysis.get(key) != value:
      raise AnalysisError(f'Immutable v3.3.3.1 archive changed: {key}.')
  return {
      'amendment_commit': PRIOR_331_AMENDMENT_COMMIT,
      'implementation_commit': PRIOR_331_IMPLEMENTATION_COMMIT,
      'archive_commit': PRIOR_331_ARCHIVE_COMMIT,
      'source_audit': sources, 'attempt': attempt, 'output': output,
      'terminal_status': analysis['status'], 'decision': analysis['decision'],
  }


def _canonical_json_sha256(value: Any) -> str:
  encoded = json.dumps(
      value, sort_keys=True, separators=(',', ':'), ensure_ascii=False
  ).encode('utf-8')
  return hashlib.sha256(encoded).hexdigest()


_SOURCE_AUDIT_KEYS = {
    'bootstrap_sources_and_prior_trees_exact',
    'tracked_head_and_frozen_inventory_exact',
    'external_device_runtime_environment_exact',
    'same_process_device_runtime_environment_exact', 'checkpoint_exact',
    'reference_object_and_sequences_exact', 'protobuf_binding_exact',
    'three_import_inventories_stable_exact',
}


def _content_binding(value: Any) -> dict[str, Any]:
  payload = json.dumps(
      value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
      allow_nan=False,
  ).encode('utf-8')
  return {'sha256': hashlib.sha256(payload).hexdigest(), 'size_bytes': len(payload)}


def _validate_source_audit(
    value: Any, binding: Any, expected: Sequence[bool | None], label: str,
) -> dict[str, Any]:
  node = _exact_keys(value, _SOURCE_AUDIT_KEYS, label)
  ordered = (
      'bootstrap_sources_and_prior_trees_exact',
      'tracked_head_and_frozen_inventory_exact',
      'external_device_runtime_environment_exact',
      'same_process_device_runtime_environment_exact', 'checkpoint_exact',
      'reference_object_and_sequences_exact', 'protobuf_binding_exact',
      'three_import_inventories_stable_exact',
  )
  if tuple(node[name] for name in ordered) != tuple(expected):
    raise AnalysisError(f'{label} phase matrix changed.')
  if binding != _content_binding(node):
    raise AnalysisError(f'{label} canonical content binding changed.')
  return dict(node)


def _normalized_entry_abi(compiled_hlo: str) -> tuple[str, str]:
  """Returns the single allowed first-line normalization and its digest."""
  lines = compiled_hlo.splitlines()
  if not lines or not lines[0].startswith('HloModule '):
    raise AnalysisError('Compiled HLO has no entry module line.')
  fingerprint_values = re.findall(
      r'fingerprint_before_lhs="([0-9A-Fa-f]+)"', lines[0]
  )
  if (
      len(fingerprint_values) != 1
      or lines[0].count('fingerprint_before_lhs=') != 1
  ):
    raise AnalysisError(
        'Entry ABI requires one nonempty hexadecimal backend fingerprint.'
    )
  normalized, substitutions = re.subn(
      r'fingerprint_before_lhs="[0-9A-Fa-f]+"',
      'fingerprint_before_lhs="<backend-generated>"',
      lines[0],
  )
  if substitutions != 1:
    raise AnalysisError(
        'Entry ABI must contain exactly one backend fingerprint.'
    )
  return normalized, hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def _diagnostic_entry_abi_exact(
    compiled_hlo: str, *, reason: str, failure: Mapping[str, Any],
    observed_sha256: Any,
) -> bool:
  """Replays the entry parser and binds its operation-typed failure."""
  trigger_type = failure.get('type')
  if DIAGNOSTIC_TRIGGER_TYPE_TO_REASON.get(trigger_type) != reason:
    raise AnalysisError('Diagnostic entry trigger type/reason changed.')
  if trigger_type == 'CacheSignalUnavailable':
    if observed_sha256 != '':
      raise AnalysisError('Unavailable cache trigger invented entry evidence.')
    return False
  try:
    _, recomputed = _normalized_entry_abi(compiled_hlo)
  except AnalysisError as error:
    expected = {
        'EntryAbiParserFailure': 'Compiled HLO has no entry module line.',
        'FingerprintFormulaMismatch': (
            'Entry ABI requires one nonempty hexadecimal backend fingerprint.'
        ),
    }.get(str(trigger_type))
    if expected is None or str(error) != expected or observed_sha256 != '':
      raise AnalysisError(
          'Diagnostic entry-ABI failure does not match its exact caught phase.'
      )
    return False
  if trigger_type in {'EntryAbiParserFailure', 'FingerprintFormulaMismatch'}:
    raise AnalysisError('Typed entry-ABI failure did not replay.')
  if observed_sha256 != recomputed:
    raise AnalysisError('Diagnostic failure entry-ABI evidence changed.')
  return observed_sha256 == ENTRY_ABI_SHA256




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


def _validate_triggering_diagnostic_operation(
    failure: Mapping[str, Any], reason: str, compiled_hlo: str,
) -> dict[str, Any]:
  """Validates the operation-typed trigger without reading its message."""
  trigger_type = failure.get('type')
  expected_reason = DIAGNOSTIC_TRIGGER_TYPE_TO_REASON.get(trigger_type)
  if expected_reason is None or reason != expected_reason:
    raise AnalysisError('Diagnostic trigger type/reason mapping changed.')
  if trigger_type == 'CacheSignalUnavailable':
    return {
        'trigger_type': trigger_type, 'reason': reason,
        'operation_replayed': 'cache_evidence_nullability',
    }
  entry_failure = None
  try:
    _normalized_entry_abi(compiled_hlo)
  except AnalysisError as error:
    entry_failure = str(error)
  expected_entry_failure = {
      'EntryAbiParserFailure': 'Compiled HLO has no entry module line.',
      'FingerprintFormulaMismatch': (
          'Entry ABI requires one nonempty hexadecimal backend fingerprint.'
      ),
  }.get(str(trigger_type))
  if expected_entry_failure is not None:
    if entry_failure != expected_entry_failure:
      raise AnalysisError('Typed entry-ABI diagnostic trigger did not replay.')
    return {
        'trigger_type': trigger_type, 'reason': reason,
        'operation_replayed': 'entry_abi',
    }
  if entry_failure is not None:
    raise AnalysisError('Non-entry diagnostic trigger has invalid entry ABI.')
  backend_failure = None
  try:
    _recompute_backend_diagnostics(compiled_hlo)
  except AnalysisError as error:
    backend_failure = str(error)
  if trigger_type == 'BackendDiagnosticParserFailure':
    if backend_failure is None:
      raise AnalysisError('Typed backend diagnostic trigger did not replay.')
    return {
        'trigger_type': trigger_type, 'reason': reason,
        'operation_replayed': 'backend_diagnostics',
    }
  if trigger_type == 'DiagnosticPersistenceFailure':
    if backend_failure is not None:
      raise AnalysisError('Persistence trigger has a prior parser failure.')
    return {
        'trigger_type': trigger_type, 'reason': reason,
        'operation_replayed': 'diagnostic_persistence_residual',
    }
  raise AnalysisError('Diagnostic trigger operation is not attributable.')


def _execution_order() -> tuple[tuple[int, int], ...]:
  return tuple(
      (order, anchor)
      for order in RECIPIENT_ORDERS
      for anchor in ANCHOR_IDS
  )


def _slug(value: str) -> str:
  return ''.join(
      character if character.isalnum() else '_' for character in value
  ).strip('_')


def _artifact_relative(case: Mapping[str, Any], anchor: int) -> str:
  slug = _slug(str(case['variant_id']))
  return f"raw/ood_anchors/{case['order']:03d}_{slug}/{anchor:03d}.json"


def _failed_current_relative(case: Mapping[str, Any], anchor: int) -> str:
  slug = _slug(str(case['variant_id']))
  return f"raw/failed_current/{case['order']:03d}_{slug}/{anchor:03d}.json"


def _donor_order(order: int) -> int:
  if 0 <= order < 10:
    return order + 10
  if 10 <= order < 20:
    return order - 10
  raise AnalysisError(f'Invalid development order {order}.')


def _load_cases() -> dict[int, dict[str, Any]]:
  exons: dict[str, dict[str, str]] = {}
  with _EXONS_PATH.open('r', encoding='utf-8', newline='') as handle:
    for row in csv.DictReader(handle, delimiter='\t'):
      if row.get('gene') not in {'BRAF', 'SLC25A48'}:
        raise AnalysisError('Development exon projection changed.')
      exons[row['ensembl_exon_id']] = dict(row)
  with _CASES_PATH.open('r', encoding='utf-8', newline='') as handle:
    rows = list(csv.DictReader(handle, delimiter='\t'))
  if len(rows) != 20 or len(exons) != 2:
    raise AnalysisError('Development projection must be exactly 20 rows/two exons.')
  result = {}
  for order, row in enumerate(rows):
    exon = exons.get(row['ensembl_exon_id'])
    if exon is None:
      raise AnalysisError('Development row references an unknown frozen exon.')
    chromosome = exon['chromosome']
    result[order] = {
        'order': order, 'selection_version': row['selection_version'],
        'selection_class': row['selection_class'],
        'observed_effect_sign': row['observed_effect_sign'].strip().lower(),
        'gene': exon['gene'], 'exon_id': exon['exon_id'],
        'ensembl_exon_id': exon['ensembl_exon_id'],
        'chromosome': chromosome if chromosome.startswith('chr') else f'chr{chromosome}',
        'strand': exon['strand'],
        'exon_start_1based': int(exon['exon_start_1based']),
        'exon_end_1based': int(exon['exon_end_1based']),
        'variant_id': row['variant_id'],
        'position_1based': int(row['position_1based']),
        'reference_bases': row['reference_bases'].upper(),
        'alternate_bases': row['alternate_bases'].upper(),
        'region': row['region'], 'mut_type': row['mut_type'],
        'delta_psi': float(row['delta_psi']),
        'delta_logit': float(row['delta_logit']),
    }
  if tuple(result) != RECIPIENT_ORDERS:
    raise AnalysisError('Development order changed.')
  return result


def _f32(value: Any, label: str) -> float:
  number = _finite(value, label)
  try:
    return struct.unpack('<f', struct.pack('<f', number))[0]
  except OverflowError as error:
    raise AnalysisError(f'{label} is outside float32 range.') from error


def _readout(record: Mapping[str, Any], field: str, label: str) -> dict[str, Any]:
  value = _exact_keys(
      record.get(field),
      {
          'endpoint_axis', 'selected_logit_axis', 'selected_logits',
          'endpoint_margins', 'means', 'totals', 'num_values',
      }, f'{label}.{field}',
  )
  if (
      value.get('endpoint_axis') != ['acceptor', 'donor']
      or value.get('selected_logit_axis')
      != ['relevant_class', 'padding_class']
      or value.get('num_values') != 2
  ):
    raise AnalysisError(f'{label}.{field} axes/reducer changed.')
  logits = value.get('selected_logits')
  margins = value.get('endpoint_margins')
  totals = value.get('totals')
  means = value.get('means')
  if not all(isinstance(node, list) and len(node) == 8 for node in (logits, margins, totals, means)):
    raise AnalysisError(f'{label}.{field} row count changed.')
  clean = {'selected_logits': [], 'endpoint_margins': [], 'totals': [], 'means': []}
  for row in range(8):
    if (
        not isinstance(logits[row], list) or len(logits[row]) != 2
        or not isinstance(margins[row], list) or len(margins[row]) != 2
    ):
      raise AnalysisError(f'{label}.{field}[{row}] shape changed.')
    clean_logits, clean_margins = [], []
    for endpoint in range(2):
      pair = logits[row][endpoint]
      if not isinstance(pair, list) or len(pair) != 2:
        raise AnalysisError(f'{label}.{field}[{row},{endpoint}] shape changed.')
      relevant = _f32(pair[0], f'{label}.{field}.relevant')
      padding = _f32(pair[1], f'{label}.{field}.padding')
      margin = _f32(relevant - padding, f'{label}.{field}.margin')
      if _f32(margins[row][endpoint], f'{label}.{field}.emitted_margin') != margin:
        raise AnalysisError(f'{label}.{field} margin arithmetic changed.')
      clean_logits.append([relevant, padding])
      clean_margins.append(margin)
    total = _f32(sum(clean_margins), f'{label}.{field}.total')
    mean = _f32(total / 2.0, f'{label}.{field}.mean')
    if (
        _f32(totals[row], f'{label}.{field}.emitted_total') != total
        or _f32(means[row], f'{label}.{field}.emitted_mean') != mean
    ):
      raise AnalysisError(f'{label}.{field} reducer arithmetic changed.')
    clean['selected_logits'].append(clean_logits)
    clean['endpoint_margins'].append(clean_margins)
    clean['totals'].append(total)
    clean['means'].append(mean)
  return clean


def _row_bytes(readout: Mapping[str, Any], row: int) -> bytes:
  numbers: list[float] = []
  for endpoint in readout['selected_logits'][row]:
    numbers.extend(endpoint)
  numbers.extend(readout['endpoint_margins'][row])
  numbers.extend((readout['totals'][row], readout['means'][row]))
  return b''.join(struct.pack('<f', number) for number in numbers)


def _array_shape(value: Any) -> tuple[int, ...]:
  if not isinstance(value, list):
    return ()
  if not value:
    return (0,)
  shapes = {_array_shape(item) for item in value}
  if len(shapes) != 1:
    raise AnalysisError('Runtime array is ragged.')
  return (len(value),) + shapes.pop()


def _array_leaves(value: Any) -> Iterable[Any]:
  if isinstance(value, list):
    for item in value:
      yield from _array_leaves(item)
  else:
    yield value


def _runtime_route(
    value: Any, *, coalition_id: int, donor_rows: Sequence[int], label: str,
) -> None:
  node = _exact_keys(
      value,
      {'transformer_output', 'encoder_skips', 'final_embedding', 'phase_r_residuals'},
      label,
  )
  t, e_mask = divmod(coalition_id, 128)
  active = [False, False, True, True, True, True, False, False]

  def whole(raw: Any, components: int, enabled: Sequence[bool], name: str) -> None:
    route = _exact_keys(
        raw, {'donor_batch_indices', 'natural_identity_batch_indices', 'transfer_mask'},
        f'{label}.{name}',
    )
    if (
        _array_shape(route['donor_batch_indices']) != (components, 8)
        or _array_shape(route['natural_identity_batch_indices']) != (components, 8)
        or _array_shape(route['transfer_mask']) != (components, 8)
        or any(not isinstance(item, int) or isinstance(item, bool)
               for item in _array_leaves(route['donor_batch_indices']))
        or any(not isinstance(item, int) or isinstance(item, bool)
               for item in _array_leaves(route['natural_identity_batch_indices']))
        or any(not isinstance(item, bool) for item in _array_leaves(route['transfer_mask']))
    ):
      raise AnalysisError(f'{label}.{name} route tensor changed.')
    if route['donor_batch_indices'] != [list(donor_rows) for _ in range(components)]:
      raise AnalysisError(f'{label}.{name} donor rows changed.')
    if route['natural_identity_batch_indices'] != [list(IDENTITY_ROWS) for _ in range(components)]:
      raise AnalysisError(f'{label}.{name} natural rows changed.')
    expected_mask = [
        [bool(flag and row_active) for row_active in active] for flag in enabled
    ]
    if route['transfer_mask'] != expected_mask:
      raise AnalysisError(f'{label}.{name} transfer mask changed.')

  whole(node['transformer_output'], 1, [bool(t)], 'transformer_output')
  whole(
      node['encoder_skips'], 7,
      [bool(e_mask & (1 << index)) for index in range(7)], 'encoder_skips',
  )
  final = _exact_keys(
      node['final_embedding'], {'donor_batch_indices', 'transfer_mask'},
      f'{label}.final_embedding',
  )
  if (
      final['donor_batch_indices'] != [[[row, row] for row in range(8)]]
      or _array_shape(final['transfer_mask']) != (1, 8, 2)
      or any(_array_leaves(final['transfer_mask']))
  ):
    raise AnalysisError(f'{label}.final_embedding is not exact disabled self-map.')
  residuals = _exact_keys(
      node['phase_r_residuals'],
      {
          'pre_attention_residual_transfer', 'post_attention_residual_transfer',
          'post_mlp_residual_transfer',
      }, f'{label}.phase_r_residuals',
  )
  expected_donors = [[[row] * 24 for row in range(8)] for _ in range(9)]
  for name, raw in residuals.items():
    route = _exact_keys(raw, {'donor_batch_indices', 'transfer_mask'}, f'{label}.{name}')
    if (
        route['donor_batch_indices'] != expected_donors
        or _array_shape(route['transfer_mask']) != (9, 8, 24)
        or any(_array_leaves(route['transfer_mask']))
    ):
      raise AnalysisError(f'{label}.{name} is not exact disabled self-map.')


_SAME_OBJECT_KEYS = {
    'lower_call_count', 'compile_call_count',
    'stablehlo_read_from_lowered_object',
    'pre_backend_hlo_read_from_lowered_object',
    'compile_argument_is_lowered_object',
    'compiled_hlo_read_from_compiled_object',
    'signature_attestation_from_apply_arguments',
    'apply_callable_is_compiled_object', 'compiler_record_is_gate_record',
    'lowered_python_id', 'compiled_python_id',
}


def _validate_content_bound_object(
    value: Any, binding: Any, label: str, *, keys: set[str] | None = None,
) -> dict[str, Any]:
  if not isinstance(value, Mapping):
    raise AnalysisError(f'{label} object is absent.')
  if keys is not None:
    _exact_keys(value, keys, label)
  expected = _content_binding(value)
  if binding != expected:
    raise AnalysisError(f'{label} canonical content binding changed.')
  return dict(value)


def _validate_same_object_success(value: Any, binding: Any, label: str) -> dict[str, Any]:
  node = _validate_content_bound_object(
      value, binding, label, keys=_SAME_OBJECT_KEYS
  )
  if (
      node['lower_call_count'] != 1
      or node['compile_call_count'] != 1
      or any(node[name] is not True for name in (
          'stablehlo_read_from_lowered_object',
          'pre_backend_hlo_read_from_lowered_object',
          'compile_argument_is_lowered_object',
          'compiled_hlo_read_from_compiled_object',
          'signature_attestation_from_apply_arguments',
          'apply_callable_is_compiled_object',
          'compiler_record_is_gate_record',
      ))
      or any(
          isinstance(node[name], bool) or not isinstance(node[name], int)
          or node[name] < 0
          for name in ('lowered_python_id', 'compiled_python_id')
      )
  ):
    raise AnalysisError(f'{label} is not the exact successful object flow.')
  return node


def _original_relative(
    case: Mapping[str, Any], family: str, anchor: int | None,
) -> str:
  key = f"{case['order']:03d}_{_slug(str(case['variant_id']))}"
  if family == 'identity' and anchor is None:
    return f'raw/identity/{key}.json'
  if family == 'coalition' and anchor is not None:
    return f'raw/coalitions/{key}/{anchor:03d}.json'
  raise AnalysisError('Invalid original-artifact family/anchor request.')


def _validate_original_links(
    value: Any, *, case: Mapping[str, Any], donor_case: Mapping[str, Any],
    anchor: int, original_manifest: Mapping[str, str], label: str,
) -> None:
  node = _exact_keys(
      value,
      {'recipient_identity', 'donor_identity', 'recipient_six_row_coalition'},
      f'{label}.original_artifact_bindings',
  )
  for name, linked_case, family, linked_anchor in (
      ('recipient_identity', case, 'identity', None),
      ('donor_identity', donor_case, 'identity', None),
      ('recipient_six_row_coalition', case, 'coalition', anchor),
  ):
    relative = _original_relative(linked_case, family, linked_anchor)
    row = _exact_keys(node[name], {'path', 'sha256'}, f'{label}.{name}')
    expected_sha = original_manifest.get(relative)
    if row != {'path': relative, 'sha256': expected_sha} or not _is_sha256(expected_sha):
      raise AnalysisError(f'{label}.{name} differs from the frozen v3.3 cube.')
    path = (_ORIGINAL_CUBE_DIR / relative).resolve()
    try:
      path.relative_to(_ORIGINAL_CUBE_DIR.resolve())
    except ValueError as error:
      raise AnalysisError(f'{label}.{name} escaped the original cube.') from error
    _strict_regular(path, f'{label}.{name}')
    if _sha256(path) != expected_sha:
      raise AnalysisError(f'{label}.{name} live bytes changed.')


def _validate_trace_fingerprint(value: Any, label: str) -> dict[str, Any]:
  node = _exact_keys(value, {'sha256', 'leaves'}, label)
  if not _is_sha256(node.get('sha256')) or not isinstance(node.get('leaves'), list):
    raise AnalysisError(f'{label} trace fingerprint is malformed.')
  for index, raw in enumerate(node['leaves']):
    leaf = _exact_keys(raw, {'shape', 'dtype'}, f'{label}.leaves[{index}]')
    if (
        not isinstance(leaf.get('shape'), list)
        or any(isinstance(size, bool) or not isinstance(size, int) or size < 0
               for size in leaf['shape'])
        or not isinstance(leaf.get('dtype'), str) or not leaf['dtype']
    ):
      raise AnalysisError(f'{label}.leaves[{index}] is malformed.')
  return dict(node)


def _fingerprint_rows(value: Any, label: str) -> list[dict[str, Any]]:
  node = _exact_keys(
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
  bytes_per_item = {
      'float16': 2, 'bfloat16': 2, 'float32': 4, 'float64': 8,
  }
  result = []
  row_shape: list[int] | None = None
  for index, raw in enumerate(node['rows']):
    row = _exact_keys(
        raw, {'row', 'shape', 'dtype', 'size_bytes', 'sha256'},
        f'{label}.rows[{index}]',
    )
    if (
        row.get('row') != index
        or not isinstance(row.get('shape'), list)
        or any(isinstance(size, bool) or not isinstance(size, int) or size < 0
               for size in row['shape'])
        or row.get('dtype') != node['dtype']
        or row['dtype'] not in bytes_per_item
        or isinstance(row.get('size_bytes'), bool)
        or not isinstance(row.get('size_bytes'), int)
        or row['size_bytes'] != math.prod(row['shape']) * bytes_per_item[row['dtype']]
        or not _is_sha256(row.get('sha256'))
    ):
      raise AnalysisError(f'{label}.rows[{index}] is malformed.')
    if row_shape is None:
      row_shape = list(row['shape'])
    elif row['shape'] != row_shape:
      raise AnalysisError(f'{label} row shapes differ.')
    result.append(dict(row))
  if node['full_shape'] != [8, *(row_shape or [])]:
    raise AnalysisError(f'{label} full shape differs from its row shapes.')
  return result


def _nested_shape(value: Any) -> tuple[int, ...]:
  if not isinstance(value, list):
    return ()
  if not value:
    return (0,)
  shapes = {_nested_shape(item) for item in value}
  if len(shapes) != 1:
    raise AnalysisError('Compact trace values are ragged.')
  return (len(value),) + shapes.pop()


def _upstream_compact(value: Any, label: str, shape: tuple[int, ...]) -> dict[str, Any]:
  node = _exact_keys(value, {'shape', 'dtype', 'values'}, label)
  if (
      node.get('shape') != list(shape)
      or _nested_shape(node.get('values')) != shape
      or node.get('dtype') not in {'float16', 'bfloat16', 'float32', 'float64'}
  ):
    raise AnalysisError(f'{label} compact upstream schema changed.')
  for leaf in _array_leaves(node['values']):
    _finite(leaf, f'{label}.value')
  return dict(node)


def _validate_rowwise(value: Any, anchor: int, label: str) -> None:
  calls = _exact_keys(
      value, {'intended', 'intended_repeat', 'unrelated', 'unrelated_repeat'},
      f'{label}.rowwise_trace_fingerprints',
  )
  parsed = {}
  for call, raw in calls.items():
    node = _exact_keys(raw, {
        'natural_final_embeddings', 'effective_final_embeddings',
        'transformer_output_natural_fingerprint',
        'encoder_skips_natural_fingerprints',
    }, f'{label}.rowwise.{call}')
    parsed[call] = {
        'natural': _fingerprint_rows(
            node['natural_final_embeddings'], f'{label}.{call}.natural'
        ),
        'effective': _fingerprint_rows(
            node['effective_final_embeddings'], f'{label}.{call}.effective'
        ),
        'T': _upstream_compact(
            node['transformer_output_natural_fingerprint'],
            f'{label}.{call}.T', (8, 4),
        ),
        'E': _upstream_compact(
            node['encoder_skips_natural_fingerprints'],
            f'{label}.{call}.E', (7, 8, 4),
        ),
    }
  for call in ('intended', 'unrelated'):
    if parsed[call] != parsed[f'{call}_repeat']:
      raise AnalysisError(f'{label}.{call} rowwise repeat changed.')
    if parsed[call]['natural'] != parsed[call]['effective']:
      raise AnalysisError(f'{label}.{call} final seam was not disabled.')
  for field in ('T', 'E'):
    if parsed['intended'][field] != parsed['unrelated'][field]:
      raise AnalysisError(f'{label} natural upstream {field} differs.')
  rows = range(8) if anchor == 0 else INVARIANT_ROWS
  if any(
      parsed['intended']['natural'][row]
      != parsed['unrelated']['natural'][row] for row in rows
  ):
    raise AnalysisError(f'{label} rowwise invariant rows differ.')


_CHECK_KEYS = {
    'passed', 'corrected_host_assertion_version',
    'upstream_transformer_natural_tensors_all8_exact_between_calls',
    'upstream_T_E_natural_fingerprints_all8_exact_between_calls',
    'natural_final_invariant_rows_exact_between_calls',
    'natural_final_invariant_rows',
    'active_rows_cross_call_equality_not_required',
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
  node = _exact_keys(value, _CHECK_KEYS, f'{label}.checks')
  expected_nonbool = {
      'corrected_host_assertion_version': 'v3.3.4.3',
      'natural_final_invariant_rows': list(INVARIANT_ROWS),
      'active_rows_cross_call_equality_not_required': list(ACTIVE_ROWS),
  }
  for key, expected in expected_nonbool.items():
    if node.get(key) != expected:
      raise AnalysisError(f'{label}.checks.{key} changed.')
  conditional = {
      'id0_all8_natural_final_exact_between_calls': anchor == 0,
      'id0_within_call_natural_final_recipient_noop_exact': anchor == 0,
      'id0_all8_endpoint_exact_between_calls': anchor == 0,
      'id0_recipient_noop_exact': anchor == 0,
      'id255_intended_endpoint_closure_exact': anchor == 255,
      'id255_unrelated_endpoint_closure_exact': anchor == 255,
  }
  for key in _CHECK_KEYS - set(expected_nonbool) - set(conditional) - {'normalization_computed'}:
    if node.get(key) is not True:
      raise AnalysisError(f'{label}.checks.{key} is not true.')
  for key, expected in conditional.items():
    if node.get(key) is not expected:
      raise AnalysisError(f'{label}.checks.{key} applicability changed.')
  if node.get('normalization_computed') is not False:
    raise AnalysisError(f'{label} computed forbidden normalization.')


def _validate_record(
    record: Mapping[str, Any], *, case: Mapping[str, Any],
    donor_case: Mapping[str, Any], anchor: int, execution_index: int,
    freeze_sha256: str, executable_fingerprint: str,
    original_manifest: Mapping[str, str], sequence_bindings: Mapping[int, Any],
    authorization: Mapping[str, Any], source_audit: Mapping[str, Any],
    same_object: Mapping[str, Any], started_bindings: Mapping[str, Any],
    completed_bindings: Mapping[str, Any], allow_invalid: bool,
) -> dict[str, Any]:
  del allow_invalid  # Invalid current work is represented separately in v3.3.4.3.
  label = f'order={case["order"]},anchor={anchor}'
  required = {
      'status', 'family', 'script_version', 'amendment_sha256',
      'amendment_commit', 'original_protocol_sha256', 'freeze_sha256',
      'external_freeze_authorization', 'execution_index',
      'sidecar_execution_index', 'execution_order',
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
      'raw_movement', 'model_apply_count_through_record', 'checks', 'failure',
      'seconds', 'dispatch_started_bindings', 'dispatch_completed_bindings',
      'source_input_audit', 'source_input_audit_content_binding',
      'same_object_attestation', 'same_object_attestation_content_binding',
      'confirmation_scope_disclosure', 'created_at_unix_s',
  }
  _exact_keys(record, required, label)
  e_players = ['E64', 'E32', 'E16', 'E8', 'E4', 'E2', 'E1']
  t, e_mask = divmod(anchor, 128)
  e_bits = [bool(e_mask & (1 << index)) for index in range(7)]
  expected_coalition = {
      'coalition_id': anchor, 't': t, 'e_mask': e_mask,
      'e_bits': e_bits, 'e_bits_binary': format(e_mask, '07b'),
      'enabled_players': (['T'] if t else []) + [
          player for player, enabled in zip(e_players, e_bits, strict=True)
          if enabled
      ],
      'coalition_bit_order': [*e_players, 'T'],
      'shapley_player_order': ['T', *e_players],
  }
  expected_common = {
      'status': 'complete', 'family': 'v3_3_4_3_unrelated_donor_sidecar_anchor',
      'script_version': SCRIPT_VERSION, 'amendment_sha256': AMENDMENT_SHA256,
      'amendment_commit': AMENDMENT_COMMIT,
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'freeze_sha256': freeze_sha256, 'execution_index': execution_index,
      'sidecar_execution_index': execution_index,
      'execution_order': 'recipient-major, anchor-minor',
      'eight_row_executable_fingerprint': executable_fingerprint,
      'same_eight_row_compiled_executable': True,
      'six_row_executable_used': False, 'recipient_case': dict(case),
      'donor_case': dict(donor_case), 'coalition': expected_coalition,
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
      'failure': None, 'confirmation_scope_disclosure': CONFIRMATION_DISCLOSURE,
  }
  for key, expected in expected_common.items():
    if record.get(key) != expected:
      raise AnalysisError(f'{label}.{key} changed.')
  if not _is_sha256(executable_fingerprint):
    raise AnalysisError(f'{label} executable fingerprint is malformed.')
  _finite(record.get('created_at_unix_s'), f'{label}.created_at_unix_s')
  if record.get('external_freeze_authorization') != authorization:
    raise AnalysisError(f'{label} authorization binding changed.')
  audited_source = _validate_content_bound_object(
      record.get('source_input_audit'),
      record.get('source_input_audit_content_binding'),
      f'{label}.source_input_audit', keys=_SOURCE_AUDIT_KEYS,
  )
  if audited_source != source_audit or any(value is not True for value in audited_source.values()):
    raise AnalysisError(f'{label} source-input audit is not all true.')
  audited_object = _validate_same_object_success(
      record.get('same_object_attestation'),
      record.get('same_object_attestation_content_binding'),
      f'{label}.same_object_attestation',
  )
  if audited_object != same_object:
    raise AnalysisError(f'{label} same-object evidence differs from compiler.')
  _validate_original_links(
      record.get('original_artifact_bindings'), case=case,
      donor_case=donor_case, anchor=anchor,
      original_manifest=original_manifest, label=label,
  )
  runtime = _exact_keys(
      record.get('runtime_interventions'), {'intended', 'unrelated'},
      f'{label}.runtime_interventions',
  )
  _runtime_route(runtime['intended'], coalition_id=anchor, donor_rows=INTENDED_DONOR_ROWS, label=f'{label}.intended')
  _runtime_route(runtime['unrelated'], coalition_id=anchor, donor_rows=UNRELATED_DONOR_ROWS, label=f'{label}.unrelated')
  readouts = {
      name: _readout(record, field, label) for name, field in (
          ('intended', 'intended_target_readout'),
          ('intended_repeat', 'intended_repeat_target_readout'),
          ('unrelated', 'unrelated_target_readout'),
          ('unrelated_repeat', 'unrelated_repeat_target_readout'),
      )
  }
  for call in ('intended', 'unrelated'):
    if any(
        _row_bytes(readouts[call], row)
        != _row_bytes(readouts[f'{call}_repeat'], row) for row in range(8)
    ):
      raise AnalysisError(f'{label}.{call} repeat changed.')
    if (
        _row_bytes(readouts[call], 3) != _row_bytes(readouts[call], 1)
        or _row_bytes(readouts[call], 5) != _row_bytes(readouts[call], 0)
    ):
      raise AnalysisError(f'{label}.{call} self control changed.')
  if any(
      _row_bytes(readouts['intended'], row)
      != _row_bytes(readouts['unrelated'], row) for row in INVARIANT_ROWS
  ):
    raise AnalysisError(f'{label} invariant endpoints changed between calls.')
  if anchor == 0:
    for call in ('intended', 'unrelated'):
      if (
          _row_bytes(readouts[call], 2) != _row_bytes(readouts[call], 1)
          or _row_bytes(readouts[call], 4) != _row_bytes(readouts[call], 0)
      ):
        raise AnalysisError(f'{label} ID0 no-op closure failed.')
  if anchor == 255 and (
      _row_bytes(readouts['intended'], 2) != _row_bytes(readouts['intended'], 0)
      or _row_bytes(readouts['intended'], 4) != _row_bytes(readouts['intended'], 1)
      or _row_bytes(readouts['unrelated'], 2) != _row_bytes(readouts['unrelated'], 6)
      or _row_bytes(readouts['unrelated'], 4) != _row_bytes(readouts['unrelated'], 7)
  ):
    raise AnalysisError(f'{label} ID255 donor closure failed.')
  movements = _exact_keys(record.get('raw_movement'), {'intended', 'unrelated'}, f'{label}.raw_movement')
  for call in ('intended', 'unrelated'):
    emitted = _exact_keys(
        movements[call], {'reference_into_alternate', 'alternate_into_reference'},
        f'{label}.raw_movement.{call}',
    )
    expected = {
        'reference_into_alternate': _f32(
            readouts[call]['means'][2] - readouts[call]['means'][3],
            f'{label}.{call}.movement.forward',
        ),
        'alternate_into_reference': _f32(
            readouts[call]['means'][4] - readouts[call]['means'][5],
            f'{label}.{call}.movement.reverse',
        ),
    }
    for direction, number in expected.items():
      if _f32(emitted.get(direction), f'{label}.{call}.{direction}') != number:
        raise AnalysisError(f'{label}.{call}.{direction} changed.')
  seconds = _exact_keys(
      record.get('seconds'),
      {'intended', 'intended_repeat', 'unrelated', 'unrelated_repeat'},
      f'{label}.seconds',
  )
  if any(_finite(value, f'{label}.seconds') < 0 for value in seconds.values()):
    raise AnalysisError(f'{label} has negative timing.')
  for key in ('dispatch_started_bindings', 'dispatch_completed_bindings'):
    bindings = record.get(key)
    if not isinstance(bindings, list) or len(bindings) != 4:
      raise AnalysisError(f'{label}.{key} must bind four events.')
    expected_map = started_bindings if key == 'dispatch_started_bindings' else completed_bindings
    call_base = execution_index * 4
    for call_index, binding in enumerate(bindings):
      row = _exact_keys(binding, {'path', 'sha256', 'size_bytes'}, f'{label}.{key}')
      expected_path = (
          f'dispatch_journal/{"started" if key == "dispatch_started_bindings" else "completed"}/'
          f'{call_base + call_index:03d}.json'
      )
      if dict(row) != {'path': expected_path, **dict(expected_map.get(expected_path, {}))}:
        raise AnalysisError(f'{label}.{key} binding is malformed.')
  _validate_rowwise(record.get('rowwise_trace_fingerprints'), anchor, label)
  for field in (
      'intended_trace_fingerprint', 'intended_repeat_trace_fingerprint',
      'unrelated_trace_fingerprint', 'unrelated_repeat_trace_fingerprint',
  ):
    _validate_trace_fingerprint(record.get(field), f'{label}.{field}')
  if record['intended_trace_fingerprint'] != record['intended_repeat_trace_fingerprint']:
    raise AnalysisError(f'{label} intended trace repeat changed.')
  if record['unrelated_trace_fingerprint'] != record['unrelated_repeat_trace_fingerprint']:
    raise AnalysisError(f'{label} unrelated trace repeat changed.')
  _validate_checks(record.get('checks'), anchor, label)
  return {'status': 'complete', 'anchor': anchor, 'execution_index': execution_index}




def _sidecar_source_paths() -> tuple[Path, ...]:
  return tuple(_REPO_ROOT / relative for relative in sorted(_V3343_SOURCE_PATHS))


def _validate_import_file(
    path: Path, binding: Any, *, phase: str, bundle_root: Path,
    freeze: Mapping[str, Any], authorization: Mapping[str, Any],
) -> dict[str, Any]:
  row_binding = _validate_file_binding(
      binding, f'{phase} import binding', with_path=True
  )
  if row_binding['path'] != path.relative_to(_RUN_DIR).as_posix():
    raise AnalysisError(f'{phase} import-provenance path changed.')
  if (
      _sha256(path) != row_binding['sha256']
      or path.stat().st_size != row_binding['size_bytes']
  ):
    raise AnalysisError(f'Import-provenance hash mismatch: {path.name}.')
  value = _read_json(path, path.name)
  _exact_keys(value, {
      'schema_version', 'phase', 'external_freeze_authorization',
      'prospective_upstream_source_file_count',
      'prospective_upstream_source_files',
      'loaded_scientific_module_count', 'loaded_scientific_modules',
      'upstream_source_attestation', 'v3_3_4_3_sidecar_sources',
      'created_at_unix_s',
  }, path.name)
  if (
      value.get('schema_version') != 'v3.3.4.3-import-provenance-v1'
      or value.get('phase') != phase
      or value.get('external_freeze_authorization') != authorization
  ):
    raise AnalysisError(f'{path.name} identity/authorization changed.')
  _finite(value.get('created_at_unix_s'), f'{path.name}.created_at_unix_s')
  inventory = freeze.get('upstream_imported_modules')
  if not isinstance(inventory, Mapping) or len(inventory) != 26:
    raise AnalysisError('Frozen upstream source inventory is incomplete.')
  upstream_root = (bundle_root.parent / 'alphagenome').resolve()
  generated_names = {
      'alphagenome.protos.dna_model_pb2',
      'alphagenome.protos.dna_model_service_pb2',
      'alphagenome.protos.dna_model_service_pb2_grpc',
      'alphagenome.protos.tensor_pb2',
  }
  prospective = value.get('prospective_upstream_source_files')
  if (
      value.get('prospective_upstream_source_file_count') != 26
      or not isinstance(prospective, list) or len(prospective) != 26
  ):
    raise AnalysisError(f'{path.name} prospective inventory count changed.')
  expected_names = sorted(inventory)
  if [item.get('module_name') for item in prospective] != expected_names:
    raise AnalysisError(f'{path.name} prospective module order changed.')
  for raw, name in zip(prospective, expected_names, strict=True):
    item = _exact_keys(raw, {
        'module_name', 'path', 'declared_root', 'relative_path', 'sha256',
        'size_bytes', 'source_kind', 'git_mode', 'filesystem_mode',
    }, f'{path.name}.prospective.{name}')
    frozen = inventory[name]
    module_path = (upstream_root / str(frozen['relative_path'])).resolve()
    source_kind = (
        'generated_untracked_exception' if name in generated_names
        else 'git_tracked'
    )
    expected_row = {
        'module_name': name, 'path': str(module_path),
        'declared_root': 'upstream_alphagenome_checkout',
        'relative_path': frozen['relative_path'], 'sha256': frozen['sha256'],
        'size_bytes': frozen['size_bytes'], 'source_kind': source_kind,
        'git_mode': None if name in generated_names else '100644',
        'filesystem_mode': '0664',
    }
    if item != expected_row:
      raise AnalysisError(f'{path.name} prospective row changed: {name}.')
    _strict_regular(module_path, f'{path.name}.{name}')
    if (
        _sha256(module_path) != frozen['sha256']
        or module_path.stat().st_size != frozen['size_bytes']
        or f'{stat.S_IMODE(module_path.stat().st_mode):04o}' != '0664'
    ):
      raise AnalysisError(f'{path.name} prospective bytes/mode changed: {name}.')
  expected_attestation = {
      'git_head': freeze['upstream_alphagenome_git_head'],
      'tracked_head_clean': True,
      'imported_module_count': 26,
      'imported_modules': {
          name: {
              **dict(inventory[name]),
              'path': str((upstream_root / inventory[name]['relative_path']).resolve()),
              'source_kind': (
                  'generated_exact_byte_exception'
                  if name in generated_names else 'tracked'
              ),
          }
          for name in expected_names
      },
      'tracked_imported_module_count': 22,
      'generated_imported_module_count': 4,
      'generated_binding_exception': freeze[
          'upstream_generated_binding_exception'
      ],
  }
  if value.get('upstream_source_attestation') != expected_attestation:
    raise AnalysisError(f'{path.name} upstream attestation changed.')
  modules = value.get('loaded_scientific_modules')
  if not isinstance(modules, list) or value.get(
      'loaded_scientific_module_count'
  ) != len(modules):
    raise AnalysisError(f'{path.name} loaded-module list/count is invalid.')
  contract = freeze.get('source_inventory_contract', {}).get(
      'loaded_scientific_module_contract'
  )
  if not isinstance(contract, list) or modules != contract:
    raise AnalysisError(f'{path.name} loaded-module contract changed.')
  by_name: dict[str, Mapping[str, Any]] = {}
  by_path: dict[str, list[str]] = defaultdict(list)
  for raw in modules:
    row = _exact_keys(
        raw, {'name', 'path', 'root', 'sha256', 'size_bytes', 'filesystem_mode'},
        f'{path.name}.module',
    )
    name = row.get('name')
    if not isinstance(name, str) or not name or name in by_name:
      raise AnalysisError(f'{path.name} has a duplicate/malformed module name.')
    root = {
        'alphagenome_research_checkout': bundle_root.resolve(),
        'upstream_alphagenome_checkout': upstream_root,
        'locked_opensplice_checkout': _HERE.resolve(),
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
        or f'{stat.S_IMODE(module_path.stat().st_mode):04o}'
        != row.get('filesystem_mode')
    ):
      raise AnalysisError(f'{path.name} module bytes changed: {name}.')
    by_name[name] = dict(row)
    by_path[str(module_path)].append(name)
  for duplicate_path, names in {
      key: names for key, names in by_path.items() if len(names) > 1
  }.items():
    if (
        set(names) != {'__main__', '__mp_main__'}
        or Path(duplicate_path).name
        != 'run_encoder_skip_ood_sidecar_v3_3_4_3.py'
    ):
      raise AnalysisError(f'{path.name} has an unapproved duplicate path alias.')
    rows = [by_name[name] for name in names]
    if any(
        (row['sha256'], row['size_bytes'], row['root'], row['filesystem_mode'])
        != (rows[0]['sha256'], rows[0]['size_bytes'], rows[0]['root'],
            rows[0]['filesystem_mode'])
        for row in rows[1:]
    ):
      raise AnalysisError(f'{path.name} approved alias bytes differ.')
  expected_sidecar = {
      row['path']: {'sha256': row['sha256'], 'size_bytes': row['size_bytes']}
      for row in modules if row['root'] == 'locked_opensplice_checkout'
  }
  if value.get('v3_3_4_3_sidecar_sources') != expected_sidecar:
    raise AnalysisError(f'{path.name} sidecar-source binding changed.')
  return {'value': value, 'modules': by_name}


def _validate_imports(
    run_dir: Path, completion: Mapping[str, Any], *, bundle_root: Path,
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
  filenames = {
      'pre_model': 'IMPORT_PROVENANCE_PRE_MODEL.json',
      'post_model_precompile': 'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
      'terminal': 'IMPORT_PROVENANCE.json',
  }
  bindings = _exact_keys(
      completion.get('import_provenance_phases'), set(filenames),
      'RUN_COMPLETE.import_provenance_phases',
  )
  authorization = completion['external_freeze_authorization']
  phases = {}
  for phase, filename in filenames.items():
    binding = bindings[phase]
    path = run_dir / filename
    if binding is None:
      if path.exists() or path.is_symlink():
        raise AnalysisError(f'Null {phase} import phase has a visible file.')
      phases[phase] = None
    else:
      phases[phase] = _validate_import_file(
          path, binding, phase=phase, bundle_root=bundle_root, freeze=freeze,
          authorization=authorization,
      )
  present = [phase for phase in filenames if phases[phase] is not None]
  if present != list(filenames)[:len(present)]:
    raise AnalysisError('Import provenance is not an exact phase prefix.')
  if len(present) == 3:
    canonical = []
    for phase in present:
      item = dict(phases[phase]['value'])
      item.pop('phase')
      item.pop('created_at_unix_s')
      canonical.append(json.dumps(
          item, sort_keys=True, separators=(',', ':'), allow_nan=False
      ))
    if len(set(canonical)) != 1:
      raise AnalysisError('Three import inventories are not byte-stable.')
  return {
      'phase_bindings': dict(bindings), 'present_phases': present,
      'three_inventories_stable_exact': len(present) == 3,
  }


def _validate_protobuf(
    run_dir: Path, completion: Mapping[str, Any], freeze: Mapping[str, Any],
) -> dict[str, Any]:
  digest = completion.get('protobuf_provenance_sha256')
  path = run_dir / 'PROTOBUF_PROVENANCE.json'
  if digest is None:
    if path.exists() or path.is_symlink():
      raise AnalysisError('Null protobuf provenance has a visible artifact.')
    return {'status': 'not_reached'}
  if not _is_sha256(digest):
    raise AnalysisError('RUN_COMPLETE protobuf digest is malformed.')
  value = _read_json(path, 'PROTOBUF_PROVENANCE')
  keys = {
      'byte_level_reproducibility',
      'current_protoc_was_used_to_generate_frozen_outputs',
      'current_standalone_protoc', 'dependency_pb2', 'dependency_proto',
      'embedded_generated_header', 'generated_outputs',
      'historical_generation_provenance', 'imported_dependency_pb2',
      'imported_pb2', 'protobuf_runtime_version', 'regeneration_claim',
      'source_proto', 'tensor_pb2', 'tensor_proto',
      'external_freeze_authorization',
  }
  _exact_keys(value, keys, 'PROTOBUF_PROVENANCE')
  if _sha256(path) != digest:
    raise AnalysisError('PROTOBUF_PROVENANCE hash binding changed.')
  expected = freeze.get('protobuf_binding')
  if not isinstance(expected, Mapping):
    raise AnalysisError('Freeze protobuf binding is malformed.')
  plain = dict(value)
  authorization = plain.pop('external_freeze_authorization')
  if (
      plain != expected
      or authorization != completion.get('external_freeze_authorization')
  ):
    raise AnalysisError('PROTOBUF_PROVENANCE differs from frozen exact bytes.')
  def validate_paths(node: Any, label: str) -> None:
    if isinstance(node, Mapping):
      if 'path' in node and 'sha256' in node:
        source = Path(str(node['path'])).resolve()
        _strict_regular(source, label)
        if not _is_sha256(node['sha256']) or _sha256(source) != node['sha256']:
          raise AnalysisError(f'{label} live SHA-256 changed.')
        if 'size_bytes' in node and source.stat().st_size != node['size_bytes']:
          raise AnalysisError(f'{label} live size changed.')
      for key, child in node.items():
        validate_paths(child, f'{label}.{key}')
    elif isinstance(node, list):
      for index, child in enumerate(node):
        validate_paths(child, f'{label}[{index}]')
  validate_paths(plain, 'PROTOBUF_PROVENANCE')
  if value.get('current_protoc_was_used_to_generate_frozen_outputs') is not False:
    raise AnalysisError('Analyzer cannot certify a new protobuf regeneration.')
  return {'status': 'validated', 'sha256': digest, 'key_count': len(value)}


_START_KEYS = {
    'status', 'attempt_id', 'script_version', 'amendment_sha256',
    'amendment_commit', 'original_protocol_sha256', 'freeze_path',
    'freeze_sha256', 'git_head', 'external_freeze_authorization', 'runner_pid',
    'parent_pid', 'started_at_unix_s', 'successful_preflight',
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
}

_FREEZE_KEYS = {
    'amendment_commit', 'amendment_path', 'amendment_sha256',
    'analysis_attempt_dir', 'analysis_dir', 'attempt_id', 'attention_backend',
    'cache_isolation_contract', 'checkpoint_manifest_path',
    'checkpoint_manifest_sha256', 'checkpoint_snapshot',
    'compiled_backend_equality_is_a_gate', 'context_bp',
    'denied_cache_environment_names', 'denied_cache_environment_prefixes',
    'development_exons_path', 'development_exons_sha256',
    'development_variants_path', 'development_variants_sha256',
    'eight_row_compile_count', 'eight_row_intended_donor_rows',
    'eight_row_natural_identity_rows', 'eight_row_roles',
    'eight_row_unrelated_donor_rows', 'environment_contract',
    'expected_compute_capability', 'expected_device_kind', 'expected_gpu_uuid',
    'file_sha256', 'identity_rerun_count', 'invariant_rows_between_calls',
    'main_cube_rerun_count', 'max_output_bytes', 'max_wall_time_seconds',
    'mixed_precision_policy', 'model_apply_count', 'model_kernel_cache_dir',
    'old_ood_records_reused', 'ood_anchor_ids', 'ood_record_count',
    'original_freeze_path', 'original_freeze_sha256',
    'original_protocol_path', 'original_protocol_sha256', 'original_run',
    'output_dir', 'preflight_dir', 'preflight_kernel_cache_dir',
    'preflight_script_version', 'program_signatures', 'protobuf_binding',
    'recipient_orders', 'reference_bindings_path',
    'reference_bindings_sha256', 'reference_object', 'reference_url',
    'runtime_version_manifest', 'script_version', 'six_row_compile_count',
    'source_program_contract', 'upstream_alphagenome_git_head',
    'upstream_generated_binding_exception', 'upstream_imported_modules',
    'v3_3_1_status', 'v3_3_2_1_failure_status', 'v3_3_2_2_archive_status',
    'v3_3_2_freeze_path', 'v3_3_2_freeze_sha256', 'v3_3_2_run',
    'v3_3_3_1_archive', 'program_signature_attestation_contract',
    'source_input_audit_contract', 'same_object_attestation_contract',
    'dispatch_journal_contract', 'failed_current_contract',
    'raw_record_contract', 'raw_manifest_contract', 'terminal_contract',
    'preflight_contract', 'compiled_diagnostics_contract',
    'source_inventory_contract', 'external_freeze_authorization_contract',
    'publication_contract_v3_3_4_1',
    'nonpublication_terminal_contract_v3_3_4_3',
}

_RUN_COMPLETE_KEYS = {
    'status', 'stop_reason', 'message', 'failure', 'attempt_id',
    'script_version', 'amendment_sha256', 'amendment_commit',
    'original_protocol_sha256', 'freeze_sha256', 'git_head',
    'external_freeze_authorization', 'runner_pid', 'started_at_unix_s',
    'completed_at_unix_s', 'phase_state', 'terminal_detail', 'budgets',
    'source_input_audit', 'source_input_audit_content_binding',
    'same_object_attestation', 'same_object_attestation_content_binding',
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
    'shapley_or_nomination_computed', 'interaction_or_resolution_computed',
    'nomination_performed', 'combined_analysis_permitted', 'no_retry',
}

_PHASE_STATE_KEYS = {
    'preflight_passed', 'start_persisted', 'post_start_source_gate_passed',
    'protobuf_persisted', 'pre_model_import_inventory_persisted',
    'model_construction_attempted', 'model_constructed',
    'reference_cases_loaded', 'signatures_captured',
    'signature_attestation_persisted',
    'post_model_import_inventory_persisted', 'lower_attempted',
    'lower_succeeded', 'compile_attempted', 'compile_succeeded',
    'terminal_import_inventory_persisted', 'source_program_gate_passed',
    'diagnostic_provenance_passed', 'dispatch_begun',
}


def _validate_run_publication_audit(
    value: Any, *, run_dir: Path, preterminal: Mapping[str, Any],
) -> dict[str, Any]:
  node = _exact_keys(value, set(PUBLICATION_AUDIT_KEYS), 'publication_audit')
  if (
      node.get('schema_version') != PUBLICATION_SCHEMA_VERSION
      or node.get('method') != PUBLICATION_METHOD
      or node.get('temporary_orphan_count') != 0
      or node.get('durability_uncertain_final_count') != 0
      or node.get('preexisting_entry_count') != 0
      or node.get('temporary_orphan_bindings') != {}
      or node.get('durability_uncertain_final_bindings') != {}
      or node.get('preexisting_entry_states') != {}
      or node.get('no_new_entry_failure') is not False
      or node.get('publication_failure') is not None
      or node.get('no_published_final_deleted') is not True
      or node.get('no_temp_or_final_reused') is not True
      or node.get('no_publication_retry') is not True
  ):
    raise AnalysisError('RUN_COMPLETE publication-audit terminal state changed.')
  bindings = _validate_publication_binding_map(
      node.get('successful_final_bindings_before_terminal'),
      'publication_audit.successful_final_bindings_before_terminal',
      expected_mode='0400',
  )
  if node.get('successful_final_count_before_terminal') != len(bindings):
    raise AnalysisError('RUN_COMPLETE publication successful count changed.')
  preterminal_files = preterminal.get('file_bindings')
  if not isinstance(preterminal_files, Mapping) or set(bindings) != set(preterminal_files):
    raise AnalysisError('RUN_COMPLETE publication map differs from preterminal tree.')
  for relative, binding in bindings.items():
    path = run_dir / relative
    _strict_regular(path, f'published final {relative}')
    status = path.lstat()
    observed = {
        'sha256': _sha256(path), 'size_bytes': status.st_size,
        'mode': f'{stat.S_IMODE(status.st_mode):04o}',
        'st_dev': status.st_dev, 'st_ino': status.st_ino,
        'st_nlink': status.st_nlink,
    }
    if binding != observed:
      raise AnalysisError(f'Published final binding changed: {relative}.')
    terminal_binding = preterminal_files[relative]
    if (
        not isinstance(terminal_binding, Mapping)
        or terminal_binding.get('sha256') != binding['sha256']
        or terminal_binding.get('size_bytes') != binding['size_bytes']
    ):
      raise AnalysisError(f'Preterminal publication linkage changed: {relative}.')
  return dict(node)


def _validate_root_publication_audit(
    value: Any, label: str, *, root: Path,
) -> dict[str, Any]:
  """Validates a helper-produced 15-key publication audit by itself."""
  node = _exact_keys(value, set(PUBLICATION_AUDIT_KEYS), label)
  successful = _validate_publication_binding_map(
      node.get('successful_final_bindings_before_terminal'),
      f'{label}.successful_final_bindings_before_terminal',
      expected_mode='0400',
  )
  temporary = _validate_publication_binding_map(
      node.get('temporary_orphan_bindings'),
      f'{label}.temporary_orphan_bindings',
  )
  uncertain = _validate_publication_binding_map(
      node.get('durability_uncertain_final_bindings'),
      f'{label}.durability_uncertain_final_bindings',
      expected_mode='0400',
  )
  preexisting = _validate_entry_state_map(
      node.get('preexisting_entry_states'),
      f'{label}.preexisting_entry_states',
  )
  if any(binding['mode'] not in {'0600', '0400'} for binding in temporary.values()):
    raise AnalysisError(f'{label} temporary orphan mode changed.')
  failure_raw = node.get('publication_failure')
  failure = (
      None if failure_raw is None
      else _validate_publication_failure(failure_raw, f'{label}.publication_failure')
  )
  if (
      node.get('schema_version') != PUBLICATION_SCHEMA_VERSION
      or node.get('method') != PUBLICATION_METHOD
      or node.get('successful_final_count_before_terminal') != len(successful)
      or node.get('temporary_orphan_count') != len(temporary)
      or node.get('durability_uncertain_final_count') != len(uncertain)
      or node.get('preexisting_entry_count') != len(preexisting)
      or not isinstance(node.get('no_new_entry_failure'), bool)
      or node.get('no_published_final_deleted') is not True
      or node.get('no_temp_or_final_reused') is not True
      or node.get('no_publication_retry') is not True
  ):
    raise AnalysisError(f'{label} count/contract changed.')
  if failure is None:
    if temporary or uncertain or preexisting or node['no_new_entry_failure']:
      raise AnalysisError(f'{label} has failure state without a failure.')
  elif node['no_new_entry_failure']:
    if temporary or uncertain:
      raise AnalysisError(f'{label} no-new failure binds a created entry.')
  elif not temporary and not uncertain:
    raise AnalysisError(f'{label} failure lacks a preserved created entry.')
  maps = (set(successful), set(temporary), set(uncertain), set(preexisting))
  if any(maps[left] & maps[right] for left in range(4) for right in range(left + 1, 4)):
    raise AnalysisError(f'{label} publication-state maps overlap.')
  for relative, binding in {**successful, **temporary, **uncertain}.items():
    observed = _observe_entry_state(root / relative)
    expected_binding = {
        key: observed[key] for key in
        ('sha256', 'size_bytes', 'mode', 'st_dev', 'st_ino', 'st_nlink')
    } if observed['state'] == 'present' and observed['entry_type'] == 'regular' else None
    if expected_binding != binding:
      raise AnalysisError(f'{label} live regular binding changed: {relative}.')
  for relative, state in preexisting.items():
    if _observe_entry_state(root / relative) != state:
      raise AnalysisError(f'{label} live pre-existing state changed: {relative}.')
  if failure is not None:
    expected_temporary: dict[str, Any] = {}
    expected_uncertain: dict[str, Any] = {}
    expected_preexisting: dict[str, Any] = {}
    if (
        not failure['rename_noreplace_succeeded']
        and failure['temp_state']['state'] == 'present'
        and not node['no_new_entry_failure']
    ):
      if failure['temp_state']['entry_type'] != 'regular':
        raise AnalysisError(f'{label} created temporary is not regular.')
      expected_temporary[failure['temp_relative_path']] = {
          key: failure['temp_state'][key] for key in
          ('sha256', 'size_bytes', 'mode', 'st_dev', 'st_ino', 'st_nlink')
      }
    if failure['rename_noreplace_succeeded']:
      if (
          failure['final_state']['state'] != 'present'
          or failure['final_state']['entry_type'] != 'regular'
      ):
        raise AnalysisError(f'{label} uncertain final is not regular.')
      expected_uncertain[failure['final_relative_path']] = {
          key: failure['final_state'][key] for key in
          ('sha256', 'size_bytes', 'mode', 'st_dev', 'st_ino', 'st_nlink')
      }
    if not failure['rename_noreplace_succeeded']:
      for relative, state in (
          (failure['temp_relative_path'], failure['temp_state']),
          (failure['final_relative_path'], failure['final_state']),
      ):
        if state['state'] != 'absent' and relative not in expected_temporary:
          expected_preexisting[relative] = state
    if (
        temporary != expected_temporary or uncertain != expected_uncertain
        or preexisting != expected_preexisting
    ):
      raise AnalysisError(f'{label} maps differ from publication failure states.')
  return dict(node)


def _validate_terminal_failure_archive(
    run_dir: Path, value: Any, *, start: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
  node = _exact_keys(value, set(TERMINAL_FAILURE_KEYS), 'TERMINAL_FAILURE')
  if (
      node.get('schema_version') != PUBLICATION_SCHEMA_VERSION
      or node.get('status') != 'incomplete_publication_failure'
      or node.get('stop_reason') != 'artifact_publication_failure'
      or node.get('attempt_id') != ATTEMPT_ID
      or node.get('script_version') != SCRIPT_VERSION
      or node.get('external_freeze_authorization')
      != start.get('external_freeze_authorization')
      or node.get('runner_pid') != start.get('runner_pid')
      or node.get('confirmation_model_calls') != 0
      or node.get('scientific_summary_computed') is not False
      or node.get('donor_normalization_computed') is not False
      or node.get('shapley_or_nomination_computed') is not False
      or node.get('interaction_or_resolution_computed') is not False
      or node.get('nomination_performed') is not False
      or node.get('combined_analysis_permitted') is not False
      or node.get('no_retry') is not True
  ):
    raise AnalysisError('TERMINAL_FAILURE fixed boundary changed.')
  for key in (
      'model_apply_attempt_count', 'model_apply_success_count',
      'valid_record_count',
  ):
    item = node.get(key)
    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
      raise AnalysisError(f'TERMINAL_FAILURE.{key} is malformed.')
  if not (
      node['valid_record_count'] <= EXPECTED_RECORD_COUNT
      and node['model_apply_success_count'] <= node['model_apply_attempt_count']
      <= EXPECTED_APPLY_COUNT
  ):
    raise AnalysisError('TERMINAL_FAILURE prefix counts are impossible.')
  _finite(node.get('created_at_unix_s'), 'TERMINAL_FAILURE.created_at_unix_s')
  failure = _validate_publication_failure(
      node.get('publication_failure'), 'TERMINAL_FAILURE.publication_failure'
  )
  if failure['root_role'] != 'model_run' or failure['runner_pid'] != node['runner_pid']:
    raise AnalysisError('TERMINAL_FAILURE publication failure escaped model run.')
  source = _validate_content_bound_object(
      node.get('source_input_audit'),
      node.get('source_input_audit_content_binding'),
      'TERMINAL_FAILURE.source_input_audit', keys=_SOURCE_AUDIT_KEYS,
  )
  same_raw = node.get('same_object_attestation')
  same_binding = node.get('same_object_attestation_content_binding')
  if same_raw is None:
    if same_binding is not None:
      raise AnalysisError('TERMINAL_FAILURE has a dangling same-object binding.')
  else:
    _validate_content_bound_object(
        same_raw, same_binding, 'TERMINAL_FAILURE.same_object_attestation',
        keys=_SAME_OBJECT_KEYS,
    )
  _validate_phase_state(
      node.get('phase_state'), 'incomplete_publication_failure',
      'artifact_publication_failure',
  )
  temporary = _validate_publication_binding_map(
      node.get('temporary_orphan_bindings'),
      'TERMINAL_FAILURE.temporary_orphan_bindings',
  )
  uncertain = _validate_publication_binding_map(
      node.get('durability_uncertain_final_bindings'),
      'TERMINAL_FAILURE.durability_uncertain_final_bindings',
      expected_mode='0400',
  )
  preexisting = _validate_entry_state_map(
      node.get('preexisting_entry_states'),
      'TERMINAL_FAILURE.preexisting_entry_states',
  )
  if any(binding['mode'] not in {'0600', '0400'} for binding in temporary.values()):
    raise AnalysisError('TERMINAL_FAILURE temporary orphan mode changed.')
  maps = (set(temporary), set(uncertain), set(preexisting))
  if maps[0] & maps[1] or maps[0] & maps[2] or maps[1] & maps[2]:
    raise AnalysisError('TERMINAL_FAILURE publication maps overlap.')
  for relative, binding in {**temporary, **uncertain}.items():
    path = run_dir / relative
    observed = _observe_entry_state(path)
    if observed['state'] != 'present' or observed['entry_type'] != 'regular':
      raise AnalysisError(f'TERMINAL_FAILURE bound file is not regular: {relative}.')
    expected = {
        key: observed[key]
        for key in ('sha256', 'size_bytes', 'mode', 'st_dev', 'st_ino', 'st_nlink')
    }
    if binding != expected:
      raise AnalysisError(f'TERMINAL_FAILURE bound file changed: {relative}.')
  for relative, state in preexisting.items():
    if _observe_entry_state(run_dir / relative) != state:
      raise AnalysisError(f'TERMINAL_FAILURE pre-existing state changed: {relative}.')
  no_new = node.get('no_new_entry_failure')
  if not isinstance(no_new, bool):
    raise AnalysisError('TERMINAL_FAILURE.no_new_entry_failure is not boolean.')
  if no_new:
    if temporary or uncertain:
      raise AnalysisError('No-new-entry failure binds a new artifact.')
    decision = (
        'preexisting_entry_preserved_no_scientific_analysis'
        if preexisting else 'publication_failed_no_new_entry_no_scientific_analysis'
    )
  elif temporary and not uncertain:
    decision = 'temporary_orphan_preserved_no_scientific_analysis'
  elif uncertain and not temporary:
    decision = 'durability_uncertain_final_preserved_no_scientific_analysis'
  else:
    raise AnalysisError('TERMINAL_FAILURE publication class is ambiguous.')
  if failure['rename_noreplace_succeeded'] and not uncertain:
    raise AnalysisError('Successful rename lacks a durability-uncertain final.')
  if temporary and failure['rename_noreplace_succeeded']:
    raise AnalysisError('Temporary orphan contradicts rename success.')
  expected_temporary = {}
  expected_uncertain = {}
  expected_preexisting = {}
  if (
      not failure['rename_noreplace_succeeded']
      and failure['temp_state']['state'] == 'present'
      and not no_new
  ):
    if failure['temp_state']['entry_type'] != 'regular':
      raise AnalysisError('Created publication temporary is not regular.')
    expected_temporary[failure['temp_relative_path']] = {
        key: failure['temp_state'][key] for key in
        ('sha256', 'size_bytes', 'mode', 'st_dev', 'st_ino', 'st_nlink')
    }
  if failure['rename_noreplace_succeeded']:
    if failure['final_state']['state'] != 'present' or failure['final_state']['entry_type'] != 'regular':
      raise AnalysisError('Renamed publication final is not preserved.')
    expected_uncertain[failure['final_relative_path']] = {
        key: failure['final_state'][key] for key in
        ('sha256', 'size_bytes', 'mode', 'st_dev', 'st_ino', 'st_nlink')
    }
  if not failure['rename_noreplace_succeeded']:
    for relative, state_value in (
        (failure['temp_relative_path'], failure['temp_state']),
        (failure['final_relative_path'], failure['final_state']),
    ):
      if state_value['state'] != 'absent' and relative not in expected_temporary:
        expected_preexisting[relative] = state_value
  if (
      temporary != expected_temporary or uncertain != expected_uncertain
      or preexisting != expected_preexisting
  ):
    raise AnalysisError('TERMINAL_FAILURE publication maps differ from failure states.')
  preterminal = _exact_keys(node.get('preterminal_tree_binding'), {
      'file_count', 'directory_count', 'file_bindings', 'file_tree_sha256',
      'directory_paths', 'directory_tree_sha256',
  }, 'TERMINAL_FAILURE.preterminal_tree_binding')
  preterminal_files = _validate_binding_map(
      preterminal.get('file_bindings'), run_dir,
      'TERMINAL_FAILURE.preterminal_tree_binding.file_bindings',
  )
  if (
      preterminal.get('file_count') != len(preterminal_files)
      or preterminal.get('file_tree_sha256')
      != _binding_map_digest(preterminal_files)
  ):
    raise AnalysisError('TERMINAL_FAILURE preterminal tree changed.')
  preterminal_directories = preterminal.get('directory_paths')
  if (
      not isinstance(preterminal_directories, list)
      or preterminal_directories != sorted(set(preterminal_directories))
      or not preterminal_directories or preterminal_directories[0] != '.'
      or preterminal.get('directory_count') != len(preterminal_directories)
      or preterminal.get('directory_tree_sha256')
      != _directory_digest(preterminal_directories)
      or preterminal_directories != _live_directory_paths(
          run_dir, 'TERMINAL_FAILURE preterminal tree',
          opaque_directories={
              relative for relative, state in preexisting.items()
              if state['state'] == 'present'
              and state['entry_type'] == 'directory'
          },
      )
  ):
    raise AnalysisError('TERMINAL_FAILURE preterminal directories changed.')
  if set(preterminal_files) & (set(temporary) | set(uncertain) | set(preexisting)):
    raise AnalysisError('Failed publication entry was counted as a successful final.')
  failed_current_binding = node.get('failed_current_binding')
  if failed_current_binding is not None:
    failed_current_row = _validate_file_binding(
        failed_current_binding,
        'TERMINAL_FAILURE.failed_current_binding',
        with_path=True,
    )
    if preterminal_files.get(failed_current_row['path']) != {
        'sha256': failed_current_row['sha256'],
        'size_bytes': failed_current_row['size_bytes'],
    }:
      raise AnalysisError(
          'TERMINAL_FAILURE failed-current is not a successful preterminal final.'
      )
  failed_current_audit = _validate_terminal_failure_prefix(
      run_dir, node, source_binding=node['source_input_audit_content_binding'],
      object_binding=node['same_object_attestation_content_binding'],
  )
  terminal_path = run_dir / 'TERMINAL_FAILURE.json'
  _strict_regular(terminal_path, 'TERMINAL_FAILURE final')
  if stat.S_IMODE(terminal_path.lstat().st_mode) != 0o400:
    raise AnalysisError('TERMINAL_FAILURE final mode changed.')
  # The terminal itself is intentionally outside its own preterminal binding.
  model_publication = {
      'schema_version': PUBLICATION_SCHEMA_VERSION,
      'method': PUBLICATION_METHOD,
      'successful_final_count_before_terminal': len(preterminal_files),
      'successful_final_bindings_before_terminal': {
          relative: _live_file_publication_binding(run_dir / relative)
          for relative in sorted(preterminal_files)
      },
      'temporary_orphan_count': len(temporary),
      'temporary_orphan_bindings': temporary,
      'durability_uncertain_final_count': len(uncertain),
      'durability_uncertain_final_bindings': uncertain,
      'preexisting_entry_count': len(preexisting),
      'preexisting_entry_states': preexisting,
      'no_new_entry_failure': no_new,
      'publication_failure': failure,
      'no_published_final_deleted': True,
      'no_temp_or_final_reused': True,
      'no_publication_retry': True,
  }
  return dict(node), {
      'decision': decision, 'source_audit': source,
      'failed_current_audit': failed_current_audit,
  }, model_publication


def _validate_terminal_failure_prefix(
    run_dir: Path, terminal: Mapping[str, Any], *,
    source_binding: Mapping[str, Any], object_binding: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
  """Audits the durable structural prefix without opening raw score JSON."""
  cases = _load_cases()
  k = terminal['valid_record_count']
  started_count = terminal['model_apply_attempt_count']
  completed_count = terminal['model_apply_success_count']
  d = completed_count - 4 * k
  if (
      d < 0 or d > 4 or started_count < completed_count
      or started_count > completed_count + 1
  ):
    raise AnalysisError('TERMINAL_FAILURE durable dispatch arithmetic changed.')
  expected_raw = {
      _artifact_relative(cases[order], anchor)
      for order, anchor in _execution_order()[:k]
  }
  raw_root = run_dir / 'raw/ood_anchors'
  observed_raw = set()
  if raw_root.exists() or raw_root.is_symlink():
    if raw_root.is_symlink() or not raw_root.is_dir():
      raise AnalysisError('TERMINAL_FAILURE raw prefix root is unsafe.')
    for entry in raw_root.rglob('*'):
      mode = entry.lstat().st_mode
      if stat.S_ISLNK(mode):
        raise AnalysisError('TERMINAL_FAILURE raw prefix contains a symlink.')
      if stat.S_ISREG(mode):
        if stat.S_IMODE(mode) != 0o400:
          raise AnalysisError('TERMINAL_FAILURE raw prefix file mode changed.')
        observed_raw.add(entry.relative_to(run_dir).as_posix())
      elif not stat.S_ISDIR(mode):
        raise AnalysisError('TERMINAL_FAILURE raw prefix has a special entry.')
  if observed_raw != expected_raw:
    raise AnalysisError('TERMINAL_FAILURE raw prefix membership changed.')
  source_sha = source_binding['sha256']
  object_sha = object_binding['sha256'] if object_binding is not None else ''
  for completed, count in ((False, started_count), (True, completed_count)):
    role = 'completed' if completed else 'started'
    directory = run_dir / f'dispatch_journal/{role}'
    expected = {f'{index:03d}.json' for index in range(count)}
    observed = set()
    if directory.exists() or directory.is_symlink():
      if directory.is_symlink() or not directory.is_dir():
        raise AnalysisError(f'TERMINAL_FAILURE {role} journal is unsafe.')
      for entry in directory.iterdir():
        _strict_regular(entry, f'TERMINAL_FAILURE {role} journal event')
        if stat.S_IMODE(entry.lstat().st_mode) != 0o400:
          raise AnalysisError(f'TERMINAL_FAILURE {role} journal mode changed.')
        observed.add(entry.name)
    if observed != expected:
      raise AnalysisError(f'TERMINAL_FAILURE {role} journal prefix changed.')
    for index in range(count):
      path = directory / f'{index:03d}.json'
      started_sha = (
          _sha256(run_dir / f'dispatch_journal/started/{index:03d}.json')
          if completed else None
      )
      _validate_dispatch_event(
          _read_json(path, f'TERMINAL_FAILURE {role} event'),
          global_index=index, completed=completed, cases=cases,
          runner_pid=terminal['runner_pid'], expected_source_sha=source_sha,
          expected_object_sha=object_sha, started_sha=started_sha,
      )
  failed = terminal.get('failed_current_binding')
  failed_audit = None
  if failed is not None:
    binding = _validate_file_binding(
        failed, 'TERMINAL_FAILURE.failed_current_binding', with_path=True,
    )
    if k >= EXPECTED_RECORD_COUNT:
      raise AnalysisError('TERMINAL_FAILURE completed prefix has failed-current binding.')
    order, anchor = _execution_order()[k]
    expected_failed_path = _failed_current_relative(cases[order], anchor)
    if binding['path'] != expected_failed_path:
      raise AnalysisError('TERMINAL_FAILURE failed-current path changed.')
    path = run_dir / binding['path']
    _strict_regular(path, 'TERMINAL_FAILURE failed-current')
    if path.stat().st_size != binding['size_bytes'] or _sha256(path) != binding['sha256']:
      raise AnalysisError('TERMINAL_FAILURE failed-current bytes changed.')
    started_map = {
        f'dispatch_journal/started/{index:03d}.json': {
            'sha256': _sha256(
                run_dir / f'dispatch_journal/started/{index:03d}.json'
            ),
            'size_bytes': (
                run_dir / f'dispatch_journal/started/{index:03d}.json'
            ).stat().st_size,
        }
        for index in range(started_count)
    }
    completed_map = {
        f'dispatch_journal/completed/{index:03d}.json': {
            'sha256': _sha256(
                run_dir / f'dispatch_journal/completed/{index:03d}.json'
            ),
            'size_bytes': (
                run_dir / f'dispatch_journal/completed/{index:03d}.json'
            ).stat().st_size,
        }
        for index in range(completed_count)
    }
    if object_binding is None:
      raise AnalysisError('TERMINAL_FAILURE failed-current lacks object provenance.')
    failed_audit = _validate_failed_current(
        _read_json(path, 'TERMINAL_FAILURE failed-current'),
        k=k, cases=cases, source_binding=source_binding,
        object_binding=object_binding,
        authorization=terminal['external_freeze_authorization'],
        started_map=started_map, completed_map=completed_map,
    )
    if failed_audit['d'] != d:
      raise AnalysisError('TERMINAL_FAILURE failed-current/journal prefix differs.')
  return failed_audit


def _live_file_publication_binding(path: Path) -> dict[str, Any]:
  _strict_regular(path, f'published artifact {path.name}')
  status = path.lstat()
  value = {
      'sha256': _sha256(path), 'size_bytes': status.st_size,
      'mode': f'{stat.S_IMODE(status.st_mode):04o}',
      'st_dev': status.st_dev, 'st_ino': status.st_ino,
      'st_nlink': status.st_nlink,
  }
  return _validate_file_publication_binding(
      value, f'published artifact {path.name}', expected_mode='0400'
  )


def _model_publication_audit_without_failure(
    run_dir: Path, *, terminal_name: str,
) -> dict[str, Any]:
  terminal = run_dir / terminal_name
  bindings = {}
  for entry in sorted(run_dir.rglob('*')):
    mode = entry.lstat().st_mode
    relative = entry.relative_to(run_dir).as_posix()
    if stat.S_ISLNK(mode):
      raise AnalysisError(f'Model publication archive contains a symlink: {relative}.')
    if stat.S_ISDIR(mode):
      if stat.S_IMODE(mode) != 0o700:
        raise AnalysisError(f'Model publication directory mode changed: {relative}.')
      continue
    if not stat.S_ISREG(mode) or stat.S_IMODE(mode) != 0o400:
      raise AnalysisError(f'Model publication archive entry is unsafe: {relative}.')
    if entry == terminal:
      continue
    bindings[relative] = _live_file_publication_binding(entry)
  return {
      'schema_version': PUBLICATION_SCHEMA_VERSION,
      'method': PUBLICATION_METHOD,
      'successful_final_count_before_terminal': len(bindings),
      'successful_final_bindings_before_terminal': bindings,
      'temporary_orphan_count': 0, 'temporary_orphan_bindings': {},
      'durability_uncertain_final_count': 0,
      'durability_uncertain_final_bindings': {},
      'preexisting_entry_count': 0, 'preexisting_entry_states': {},
      'no_new_entry_failure': False, 'publication_failure': None,
      'no_published_final_deleted': True,
      'no_temp_or_final_reused': True, 'no_publication_retry': True,
  }


def _terminal_failure_run_binding(
    run_dir: Path, terminal: Mapping[str, Any],
) -> dict[str, Any]:
  preterminal = terminal['preterminal_tree_binding']
  expected_regular = set(preterminal['file_bindings']) | {
      'TERMINAL_FAILURE.json'
  }
  expected_regular.update(terminal['temporary_orphan_bindings'])
  expected_regular.update(terminal['durability_uncertain_final_bindings'])
  expected_special = set()
  opaque_directories = set()
  for relative, state in terminal['preexisting_entry_states'].items():
    if state['state'] == 'present':
      if state['entry_type'] == 'regular':
        expected_regular.add(relative)
      elif state['entry_type'] == 'directory':
        opaque_directories.add(relative)
      else:
        expected_special.add(relative)
  files = {}
  special = set()
  directories = set()
  pending = [run_dir]
  while pending:
    directory = pending.pop()
    relative_directory = (
        '.' if directory == run_dir
        else directory.relative_to(run_dir).as_posix()
    )
    mode = directory.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
      raise AnalysisError('TERMINAL_FAILURE traversal reached a non-directory.')
    if (
        relative_directory not in opaque_directories
        and stat.S_IMODE(mode) != 0o700
    ):
      raise AnalysisError(
          f'TERMINAL_FAILURE directory mode changed: {relative_directory}.'
      )
    directories.add(relative_directory)
    if relative_directory in opaque_directories:
      continue
    for entry in directory.iterdir():
      relative = entry.relative_to(run_dir).as_posix()
      entry_mode = entry.lstat().st_mode
      if stat.S_ISDIR(entry_mode) and not stat.S_ISLNK(entry_mode):
        pending.append(entry)
      elif stat.S_ISREG(entry_mode):
        files[relative] = {
            'sha256': _sha256(entry), 'size_bytes': entry.lstat().st_size,
        }
      else:
        special.add(relative)
  if set(files) != expected_regular or special != expected_special:
    raise AnalysisError('TERMINAL_FAILURE whole-tree membership changed.')
  if not opaque_directories <= directories:
    raise AnalysisError('TERMINAL_FAILURE pre-existing directory disappeared.')
  if directories != set(terminal['preterminal_tree_binding']['directory_paths']):
    raise AnalysisError('TERMINAL_FAILURE directory membership changed.')
  terminal_path = run_dir / 'TERMINAL_FAILURE.json'
  return {
      'path': str(run_dir.resolve()), 'file_count': len(files),
      'directory_count': len(directories), 'file_bindings': files,
      'file_tree_sha256': _binding_map_digest(files),
      'directory_paths': sorted(directories),
      'directory_tree_sha256': _directory_digest(sorted(directories)),
      'terminal_kind': 'terminal_failure',
      'terminal_binding': _absolute_binding(terminal_path),
      'start_binding': _absolute_binding(run_dir / 'ATTEMPT_STARTED.json'),
      'strict_membership_exact': True,
  }

_TERMINAL_DETAIL_KEYS = {
    'k_valid_records', 'd_completed', 'failed_execution_index',
    'failed_call_role', 'failure_phase', 'forbidden_operation',
    'provenance_artifact_role',
}

_TERMINAL_STATUS_REASONS = {
    'controlled_stop_import_provenance_failure': {
        'pre_model_import_inventory_mismatch',
        'post_model_import_inventory_mismatch',
        'terminal_import_inventory_mismatch',
    },
    'controlled_stop_protobuf_provenance_failure': {'protobuf_binding_mismatch'},
    'controlled_stop_cache_hit': {
        'model_cache_pre_import_hit', 'model_cache_post_compile_hit',
    },
    'controlled_stop_model_setup_failure': {'model_setup_failure'},
    'controlled_stop_signature_attestation_failure': {
        'signature_attestation_failure'
    },
    'controlled_stop_lower_failure': {'lower_failure'},
    'controlled_stop_compile_failure': {'compile_failure'},
    'controlled_stop_attempt_budget_violation': {
        'second_lower_attempt_forbidden', 'second_compile_attempt_forbidden',
    },
    'controlled_stop_same_object_provenance_failure': {
        'lowered_object_identity_lost', 'compile_argument_identity_lost',
        'compiled_object_identity_lost', 'apply_callable_identity_lost',
    },
    'controlled_stop_source_program_mismatch': {'source_program_mismatch'},
    'controlled_stop_diagnostic_provenance_failure': {
        'diagnostic_parser_failure', 'diagnostic_persistence_failure',
        'cache_signal_unavailable', 'fingerprint_formula_mismatch',
    },
    'controlled_stop_partial_dispatch': {
        'record_setup_failure', 'model_dispatch_failure'
    },
    'controlled_stop_four_call_invalid': {
        'record_validation_or_serialization_failure'
    },
    'complete_structural_sidecar': {None},
}


def _validate_failure(value: Any, label: str, *, nullable: bool = False) -> None:
  if value is None and nullable:
    return
  node = _exact_keys(value, {'type', 'message', 'traceback'}, label)
  if any(not isinstance(node.get(key), str) or not node[key] for key in node):
    raise AnalysisError(f'{label} is malformed.')


def _validate_relative_path(value: Any, label: str) -> str:
  if not isinstance(value, str) or not value:
    raise AnalysisError(f'{label} path is absent.')
  path = Path(value)
  if (
      path.is_absolute() or '..' in path.parts or '.' in path.parts
      or '\\' in value or '//' in value or path.as_posix() != value
  ):
    raise AnalysisError(f'{label} path is not canonical run-relative POSIX.')
  return value


def _validate_file_binding(value: Any, label: str, *, with_path: bool) -> dict[str, Any]:
  keys = {'sha256', 'size_bytes'} | ({'path'} if with_path else set())
  node = _exact_keys(value, keys, label)
  if (
      not _is_sha256(node.get('sha256'))
      or isinstance(node.get('size_bytes'), bool)
      or not isinstance(node.get('size_bytes'), int)
      or node['size_bytes'] < 0
  ):
    raise AnalysisError(f'{label} binding is malformed.')
  if with_path:
    _validate_relative_path(node.get('path'), label)
  return dict(node)


def _validate_binding_map(
    value: Any, run_dir: Path, label: str, *, expected_paths: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
  if not isinstance(value, Mapping):
    raise AnalysisError(f'{label} must be a binding map.')
  if expected_paths is not None and set(value) != set(expected_paths):
    raise AnalysisError(f'{label} ordered path set changed.')
  result = {}
  for relative, raw in value.items():
    path_text = _validate_relative_path(relative, f'{label}.path')
    binding = _validate_file_binding(raw, f'{label}.{relative}', with_path=False)
    path = run_dir / path_text
    _strict_regular(path, f'{label}.{relative}')
    if path.stat().st_size != binding['size_bytes'] or _sha256(path) != binding['sha256']:
      raise AnalysisError(f'{label}.{relative} current bytes changed.')
    result[path_text] = binding
  return result


def _binding_map_digest(value: Mapping[str, Mapping[str, Any]]) -> str:
  digest = hashlib.sha256()
  for relative in sorted(value):
    digest.update(relative.encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(str(value[relative]['sha256'])))
  return digest.hexdigest()


def _directory_digest(paths: Sequence[str]) -> str:
  digest = hashlib.sha256()
  for relative in paths:
    digest.update(b'D\0')
    digest.update(relative.encode('utf-8'))
    digest.update(b'\0')
  return digest.hexdigest()


def _parent_directories(paths: Iterable[str]) -> list[str]:
  directories = {'.'}
  for relative in paths:
    parent = Path(relative).parent
    while parent.as_posix() != '.':
      directories.add(parent.as_posix())
      parent = parent.parent
  return sorted(directories)


def _live_directory_paths(
    root: Path, label: str, *, opaque_directories: Iterable[str] = (),
) -> list[str]:
  """Returns every physical directory without following symlinks."""
  try:
    root_mode = root.lstat().st_mode
  except OSError as error:
    raise AnalysisError(f'{label} root cannot be statted.') from error
  if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
    raise AnalysisError(f'{label} root is not a physical directory.')
  opaque = set(opaque_directories)
  pending = [root]
  observed = []
  while pending:
    directory = pending.pop()
    status = directory.lstat()
    relative = '.' if directory == root else directory.relative_to(root).as_posix()
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
      raise AnalysisError(f'{label} contains a non-directory traversal node.')
    if relative not in opaque and stat.S_IMODE(status.st_mode) != 0o700:
      raise AnalysisError(f'{label} directory mode changed.')
    observed.append(relative)
    if relative in opaque:
      continue
    for entry in directory.iterdir():
      mode = entry.lstat().st_mode
      if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
        pending.append(entry)
  return sorted(observed)


def _special_terminal_artifact(status: str, reason: str | None) -> str | None:
  if (
      status == 'controlled_stop_protobuf_provenance_failure'
      or status == 'controlled_stop_import_provenance_failure'
      and reason != 'terminal_import_inventory_mismatch'
  ):
    return 'PROVENANCE_VALIDATION_FAILURE.json'
  if reason == 'model_cache_pre_import_hit':
    return 'MODEL_CACHE_PRE_IMPORT_HIT.json'
  return None


def _validate_run_membership(
    run_dir: Path, completion: Mapping[str, Any], manifest: Mapping[str, Any],
) -> dict[str, Any]:
  """Reconstruct the exact append-only terminal membership from bindings."""
  expected = {'ATTEMPT_STARTED.json', 'RAW_MANIFEST.json'}
  import_bindings = _exact_keys(
      completion.get('import_provenance_phases'),
      {'pre_model', 'post_model_precompile', 'terminal'},
      'RUN_COMPLETE.import_provenance_phases',
  )
  for phase, raw in import_bindings.items():
    if raw is None:
      continue
    binding = _validate_file_binding(
        raw, f'RUN_COMPLETE.import_provenance_phases.{phase}', with_path=True
    )
    expected.add(binding['path'])
  protobuf_digest = completion.get('protobuf_provenance_sha256')
  if protobuf_digest is not None:
    if not _is_sha256(protobuf_digest):
      raise AnalysisError('RUN_COMPLETE protobuf SHA-256 is malformed.')
    expected.add('PROTOBUF_PROVENANCE.json')
  compiler_bindings = _validate_binding_map(
      completion.get('compiler_artifact_bindings'), run_dir,
      'RUN_COMPLETE.compiler_artifact_bindings',
  )
  expected.update(compiler_bindings)
  for key in (
      'artifact_bindings', 'dispatch_started_bindings',
      'dispatch_completed_bindings',
  ):
    raw_map = manifest.get(key)
    if not isinstance(raw_map, Mapping):
      raise AnalysisError(f'RAW_MANIFEST.{key} is not a binding map.')
    expected.update(raw_map)
  failed = manifest.get('failed_current_binding')
  if failed is not None:
    expected.add(_validate_file_binding(
        failed, 'RAW_MANIFEST.failed_current_binding', with_path=True
    )['path'])
  special = _special_terminal_artifact(
      str(completion.get('status')), completion.get('stop_reason')
  )
  if special is not None:
    expected.add(special)
  preterminal = _exact_keys(
      completion.get('preterminal_tree_binding'),
      {'file_count', 'directory_count', 'file_bindings', 'file_tree_sha256',
       'directory_paths', 'directory_tree_sha256'},
      'RUN_COMPLETE.preterminal_tree_binding',
  )
  file_bindings = _validate_binding_map(
      preterminal.get('file_bindings'), run_dir, 'preterminal file bindings',
      expected_paths=sorted(expected),
  )
  directories = _parent_directories(expected)
  if (
      preterminal.get('file_count') != len(expected)
      or preterminal.get('file_tree_sha256') != _binding_map_digest(file_bindings)
      or preterminal.get('directory_count') != len(directories)
      or preterminal.get('directory_paths') != directories
      or preterminal.get('directory_tree_sha256') != _directory_digest(directories)
  ):
    raise AnalysisError('Preterminal tree binding differs from exact membership.')
  expected_with_terminal = set(expected) | {'RUN_COMPLETE.json'}
  actual_files: set[str] = set()
  actual_directories = {'.'}
  for entry in run_dir.rglob('*'):
    mode = entry.lstat().st_mode
    relative = entry.relative_to(run_dir).as_posix()
    if stat.S_ISLNK(mode):
      raise AnalysisError('Run terminal tree contains a symlink.')
    if stat.S_ISREG(mode):
      if stat.S_IMODE(mode) != 0o400:
        raise AnalysisError(f'Run artifact mode changed: {relative}.')
      actual_files.add(relative)
    elif stat.S_ISDIR(mode):
      if stat.S_IMODE(mode) != 0o700:
        raise AnalysisError(f'Run directory mode changed: {relative}.')
      actual_directories.add(relative)
    else:
      raise AnalysisError(f'Run terminal tree contains a special entry: {relative}.')
  if actual_files != expected_with_terminal:
    raise AnalysisError('Whole-run terminal file membership changed.')
  if actual_directories != set(_parent_directories(expected_with_terminal)):
    raise AnalysisError('Whole-run terminal directory membership changed.')
  run_complete = run_dir / 'RUN_COMPLETE.json'
  full_map = dict(file_bindings)
  full_map['RUN_COMPLETE.json'] = {
      'sha256': _sha256(run_complete), 'size_bytes': run_complete.stat().st_size,
  }
  full_size = sum(row['size_bytes'] for row in full_map.values())
  if full_size > 1_073_741_824:
    raise AnalysisError('Whole-run immutable tree exceeds the frozen budget.')
  return {
      'path': str(run_dir.resolve()), 'file_count': len(full_map),
      'directory_count': len(actual_directories),
      'file_bindings': full_map,
      'file_tree_sha256': _binding_map_digest(full_map),
      'directory_paths': sorted(actual_directories),
      'directory_tree_sha256': _directory_digest(sorted(actual_directories)),
      'terminal_kind': 'run_complete',
      'terminal_binding': _absolute_binding(run_complete),
      'start_binding': _absolute_binding(run_dir / 'ATTEMPT_STARTED.json'),
      'strict_membership_exact': True,
  }


def _validate_phase_state(
    value: Any, status: str, reason: str | None,
) -> dict[str, bool]:
  phase = _exact_keys(value, _PHASE_STATE_KEYS, 'RUN_COMPLETE.phase_state')
  if any(not isinstance(item, bool) for item in phase.values()):
    raise AnalysisError('RUN_COMPLETE.phase_state must contain only booleans.')
  if not phase['preflight_passed'] or not phase['start_persisted'] or not phase['post_start_source_gate_passed']:
    raise AnalysisError('Caught RUN_COMPLETE did not pass the first three phases.')
  if reason == 'model_cache_pre_import_hit':
    if any(phase[name] for name in _PHASE_STATE_KEYS - {
        'preflight_passed', 'start_persisted', 'post_start_source_gate_passed'
    }):
      raise AnalysisError('Pre-import cache stop advanced a scientific phase.')
  if phase['dispatch_begun'] and not (
      phase['source_program_gate_passed'] and phase['diagnostic_provenance_passed']
  ):
    raise AnalysisError('Dispatch began before both compiler gates passed.')
  if phase['compile_succeeded'] and not phase['compile_attempted']:
    raise AnalysisError('Compile succeeded without an attempt.')
  if phase['lower_succeeded'] and not phase['lower_attempted']:
    raise AnalysisError('Lower succeeded without an attempt.')
  if phase['compile_attempted'] and not phase['lower_succeeded']:
    raise AnalysisError('Compile was attempted without successful lowering.')
  if phase['signatures_captured'] and not phase['model_constructed']:
    raise AnalysisError('Signatures were captured without a model.')
  if phase['signature_attestation_persisted'] and not phase['signatures_captured']:
      raise AnalysisError('Signature attestation lacks captured signatures.')
  dependencies = {
      'start_persisted': 'preflight_passed',
      'post_start_source_gate_passed': 'start_persisted',
      'pre_model_import_inventory_persisted': 'post_start_source_gate_passed',
      'protobuf_persisted': 'pre_model_import_inventory_persisted',
      'model_construction_attempted': 'protobuf_persisted',
      'model_constructed': 'model_construction_attempted',
      'reference_cases_loaded': 'model_constructed',
      'signatures_captured': 'reference_cases_loaded',
      'signature_attestation_persisted': 'signatures_captured',
      'post_model_import_inventory_persisted': 'model_construction_attempted',
      'lower_attempted': 'signature_attestation_persisted',
      'lower_succeeded': 'lower_attempted',
      'compile_attempted': 'lower_succeeded',
      'compile_succeeded': 'compile_attempted',
      'terminal_import_inventory_persisted': 'post_model_import_inventory_persisted',
      'source_program_gate_passed': 'compile_succeeded',
      'diagnostic_provenance_passed': 'compile_succeeded',
  }
  for child, parent in dependencies.items():
    if phase[child] and not phase[parent]:
      raise AnalysisError(f'Phase {child} is true before parent {parent}.')
  return dict(phase)


def _validate_terminal_common(
    completion: Any, *, freeze_sha: str, start: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
  node = _exact_keys(completion, _RUN_COMPLETE_KEYS, 'RUN_COMPLETE')
  status, reason = node.get('status'), node.get('stop_reason')
  if status not in _TERMINAL_STATUS_REASONS or reason not in _TERMINAL_STATUS_REASONS[status]:
    raise AnalysisError('RUN_COMPLETE status/reason pair changed.')
  expected = {
      'attempt_id': ATTEMPT_ID, 'script_version': SCRIPT_VERSION,
      'amendment_sha256': AMENDMENT_SHA256,
      'amendment_commit': AMENDMENT_COMMIT,
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'freeze_sha256': freeze_sha,
      'git_head': start['external_freeze_authorization']['git_head'],
      'external_freeze_authorization': start['external_freeze_authorization'],
      'runner_pid': start['runner_pid'],
      'expected_model_apply_count': EXPECTED_APPLY_COUNT,
      'six_row_compile_count': 0, 'identity_rerun_count': 0,
      'main_cube_rerun_count': 0, 'old_ood_records_reused': 0,
      'confirmation_model_calls': 0,
      'confirmation_scope_disclosure': CONFIRMATION_DISCLOSURE,
      'scientific_summary_computed': False,
      'donor_normalization_computed': False,
      'shapley_or_nomination_computed': False,
      'interaction_or_resolution_computed': False,
      'nomination_performed': False, 'combined_analysis_permitted': False,
      'no_retry': True,
  }
  for key, value in expected.items():
    if node.get(key) != value:
      raise AnalysisError(f'RUN_COMPLETE.{key} changed.')
  for key in ('runner_pid', 'valid_record_count', 'model_apply_attempt_count',
              'model_apply_success_count', 'eight_row_lower_attempt_count',
              'eight_row_compile_attempt_count',
              'eight_row_successful_compile_count'):
    if isinstance(node.get(key), bool) or not isinstance(node.get(key), int) or node[key] < 0:
      raise AnalysisError(f'RUN_COMPLETE.{key} is not a non-negative integer.')
  for key in ('started_at_unix_s', 'completed_at_unix_s'):
    _finite(node.get(key), f'RUN_COMPLETE.{key}')
  if node['started_at_unix_s'] != start['started_at_unix_s']:
    raise AnalysisError('RUN_COMPLETE START timestamp linkage changed.')
  if status == 'complete_structural_sidecar':
    _validate_failure(node.get('failure'), 'RUN_COMPLETE.failure', nullable=True)
    if node.get('failure') is not None:
      raise AnalysisError('Complete terminal has a failure.')
  else:
    _validate_failure(node.get('failure'), 'RUN_COMPLETE.failure')
  phase = _validate_phase_state(node.get('phase_state'), status, reason)
  detail = _exact_keys(node.get('terminal_detail'), _TERMINAL_DETAIL_KEYS, 'RUN_COMPLETE.terminal_detail')
  if detail.get('k_valid_records') != node['valid_record_count']:
    raise AnalysisError('Terminal valid-prefix count changed.')
  if reason == 'model_cache_pre_import_hit' and detail.get('failure_phase') != 'cache_pre_import':
    raise AnalysisError('Pre-import cache reason/phase mapping changed.')
  if reason == 'model_cache_post_compile_hit' and detail.get('failure_phase') != 'cache_post_compile':
    raise AnalysisError('Post-compile cache reason/phase mapping changed.')
  if detail.get('failure_phase') in {'cache_pre_import', 'cache_post_compile'} and status != 'controlled_stop_cache_hit':
    raise AnalysisError('Cache failure phase used by another status.')
  k, d = detail['k_valid_records'], detail['d_completed']
  if (
      isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= 80
      or isinstance(d, bool) or not isinstance(d, int) or not 0 <= d <= 4
  ):
    raise AnalysisError('Terminal k,d domain changed.')
  dispatch_status = status in {
      'controlled_stop_partial_dispatch',
      'controlled_stop_four_call_invalid', 'complete_structural_sidecar',
  }
  if not dispatch_status and (k != 0 or d != 0):
    raise AnalysisError('Predispatch terminal has nonzero k,d.')
  if status == 'complete_structural_sidecar':
    if k != 80 or d != 0 or any(detail[name] is not None for name in (
        'failed_execution_index', 'failed_call_role', 'failure_phase',
        'forbidden_operation', 'provenance_artifact_role',
    )):
      raise AnalysisError('Full completion terminal detail changed.')
  elif status == 'controlled_stop_four_call_invalid':
    if k >= 80 or d != 4 or detail.get('failed_execution_index') != k:
      raise AnalysisError('Four-call invalid terminal arithmetic changed.')
  elif status == 'controlled_stop_partial_dispatch':
    if k >= 80 or not 0 <= d < 4 or detail.get('failed_execution_index') != k:
      raise AnalysisError('Partial-dispatch terminal arithmetic changed.')
  elif detail.get('failed_execution_index') is not None or detail.get('failed_call_role') is not None:
    raise AnalysisError('Predispatch terminal invented a current-record identity.')
  if status == 'complete_structural_sidecar':
    expected_attempted = expected_completed = 320
  elif status == 'controlled_stop_four_call_invalid':
    expected_attempted = expected_completed = 4 * k + 4
  elif status == 'controlled_stop_partial_dispatch':
    expected_completed = 4 * k + d
    expected_attempted = expected_completed + int(
        detail.get('failure_phase') == 'model_dispatch'
    )
  else:
    expected_attempted = expected_completed = 0
  if (
      node['model_apply_attempt_count'] != expected_attempted
      or node['model_apply_success_count'] != expected_completed
  ):
    raise AnalysisError('RUN_COMPLETE model-apply prefix arithmetic changed.')
  full = status == 'complete_structural_sidecar'
  if any(node.get(name) is not full for name in (
      'all_80_recipient_anchors_complete', 'id0_all20', 'id255_all20'
  )):
    raise AnalysisError('RUN_COMPLETE closure flags changed for terminal state.')
  source_audit = _validate_content_bound_object(
      node.get('source_input_audit'),
      node.get('source_input_audit_content_binding'),
      'RUN_COMPLETE.source_input_audit', keys=_SOURCE_AUDIT_KEYS,
  )
  for name in (
      'bootstrap_sources_and_prior_trees_exact',
      'tracked_head_and_frozen_inventory_exact',
      'external_device_runtime_environment_exact',
      'same_process_device_runtime_environment_exact',
  ):
    if source_audit[name] is not True:
      raise AnalysisError('Caught RUN_COMPLETE failed a required pre-model source gate.')
  budgets = _exact_keys(node.get('budgets'), {
      'max_wall_time_seconds', 'elapsed_wall_time_seconds',
      'wall_time_within_budget', 'max_output_bytes',
      'preterminal_output_bytes', 'run_complete_size_cap_bytes',
      'preterminal_plus_terminal_cap_within_budget',
  }, 'RUN_COMPLETE.budgets')
  elapsed = _finite(budgets.get('elapsed_wall_time_seconds'), 'RUN_COMPLETE elapsed time')
  if (
      budgets.get('max_wall_time_seconds') != 7200 or elapsed < 0
      or budgets.get('wall_time_within_budget') is not (elapsed <= 7200)
      or budgets.get('max_output_bytes') != 1_073_741_824
      or budgets.get('run_complete_size_cap_bytes') != 16_777_216
      or isinstance(budgets.get('preterminal_output_bytes'), bool)
      or not isinstance(budgets.get('preterminal_output_bytes'), int)
      or budgets['preterminal_output_bytes'] < 0
      or budgets.get('preterminal_plus_terminal_cap_within_budget')
      is not (
          budgets['preterminal_output_bytes'] + 16_777_216
          <= 1_073_741_824
      )
  ):
    raise AnalysisError('RUN_COMPLETE budget arithmetic changed.')
  journal = _exact_keys(node.get('dispatch_journal'), {
      'started_count', 'completed_count', 'started_bindings',
      'completed_bindings', 'started_tree_sha256',
      'completed_tree_sha256', 'started_prefix_exact',
      'completed_prefix_exact',
  }, 'RUN_COMPLETE.dispatch_journal')
  if (
      journal.get('started_count') != node['model_apply_attempt_count']
      or journal.get('completed_count') != node['model_apply_success_count']
      or journal.get('started_prefix_exact') is not True
      or journal.get('completed_prefix_exact') is not True
  ):
    raise AnalysisError('RUN_COMPLETE dispatch journal/count linkage changed.')
  return dict(node), {
      'phase_state': phase, 'terminal_detail': dict(detail),
      'source_input_audit': source_audit, 'budgets': dict(budgets),
      'dispatch_journal': dict(journal),
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


_CACHE_BINDING_KEYS = {
    'cache_role', 'cache_root', 'triton_cache_dir', 'xdg_cache_home',
    'directory_count', 'directory_paths', 'file_count', 'files',
    'tree_sha256', 'default_user_cache_paths_eligible',
    'diagnostic_outputs_only_no_cache_input',
}
_CACHE_HIT_KEYS = {
    'pre_import_files_present', 'default_user_cache_path_eligible',
    'persistent_compilation_cache_hit_reported', 'executable_deserialized',
    'compile_skipped', 'compile_stage_not_applicable',
    'old_cache_input_opened', 'routing_exact', 'cache_hit',
}

_SAME_PROCESS_PREFLIGHT_KEYS = {
    'pid', 'parent_pid', 'external_preflight_pid', 'default_backend',
    'jax_gpu_devices', 'nvidia_smi', 'runtime_environment',
    'runtime_versions', 'freeze_sha256', 'external_freeze_authorization',
    'external_preflight_binding', 'external_preflight_tree_sha256',
    'model_cache_pre_import', 'current_source_inventory_exact',
    'prior_artifacts_exact', 'no_model_constructed',
    'no_jit_or_array_kernel', 'created_at_unix_s',
}


def _cache_binding_digest(
    directories: Sequence[str], files: Mapping[str, Mapping[str, Any]],
) -> str:
  digest = hashlib.sha256()
  for relative in directories:
    digest.update(b'D\0' + relative.encode('utf-8') + b'\0')
  for relative in sorted(files):
    digest.update(b'F\0' + relative.encode('utf-8') + b'\0')
    digest.update(bytes.fromhex(str(files[relative]['sha256'])))
  return digest.hexdigest()


def _live_cache_binding(root: Path, role: str, label: str) -> dict[str, Any]:
  if root.is_symlink() or not root.is_dir():
    raise AnalysisError(f'{label} cache root is absent or unsafe.')
  directories = ['.']
  files: dict[str, dict[str, Any]] = {}
  for entry in root.rglob('*'):
    mode = entry.lstat().st_mode
    relative = entry.relative_to(root).as_posix()
    if stat.S_ISLNK(mode):
      raise AnalysisError(f'{label} cache contains a symlink.')
    if stat.S_ISDIR(mode):
      if stat.S_IMODE(mode) != 0o700:
        raise AnalysisError(f'{label} cache directory mode changed.')
      directories.append(relative)
    elif stat.S_ISREG(mode):
      if stat.S_IMODE(mode) not in {0o400, 0o600, 0o644, 0o664}:
        raise AnalysisError(f'{label} cache file has an unsafe mode.')
      files[relative] = {
          'sha256': _sha256(entry), 'size_bytes': entry.stat().st_size,
      }
    else:
      raise AnalysisError(f'{label} cache contains a special entry.')
  directories.sort()
  files = dict(sorted(files.items()))
  return {
      'cache_role': role, 'cache_root': str(root.resolve()),
      'triton_cache_dir': str((root / 'triton').resolve()),
      'xdg_cache_home': str((root / 'xdg').resolve()),
      'directory_count': len(directories), 'directory_paths': directories,
      'file_count': len(files), 'files': files,
      'tree_sha256': _cache_binding_digest(directories, files),
      'default_user_cache_paths_eligible': False,
      'diagnostic_outputs_only_no_cache_input': True,
  }


def _validate_cache_binding(
    value: Any, *, root: Path, role: str, label: str,
    compare_live: bool,
) -> dict[str, Any]:
  node = _exact_keys(value, _CACHE_BINDING_KEYS, label)
  directories = node.get('directory_paths')
  files = node.get('files')
  if (
      not isinstance(directories, list)
      or directories != sorted(directories)
      or not directories or directories[0] != '.'
      or len(directories) != len(set(directories))
      or node.get('directory_count') != len(directories)
      or not isinstance(files, Mapping)
      or list(files) != sorted(files)
      or node.get('file_count') != len(files)
  ):
    raise AnalysisError(f'{label} directory/file ordering changed.')
  checked_files = {}
  for relative, binding in files.items():
    _validate_relative_path(relative, f'{label}.files')
    checked_files[relative] = _validate_file_binding(
        binding, f'{label}.files.{relative}', with_path=False
    )
  if (
      node.get('cache_role') != role
      or node.get('cache_root') != str(root.resolve())
      or node.get('triton_cache_dir') != str((root / 'triton').resolve())
      or node.get('xdg_cache_home') != str((root / 'xdg').resolve())
      or node.get('default_user_cache_paths_eligible') is not False
      or node.get('diagnostic_outputs_only_no_cache_input') is not True
      or node.get('tree_sha256')
      != _cache_binding_digest(directories, checked_files)
  ):
    raise AnalysisError(f'{label} cache contract/digest changed.')
  result = dict(node)
  if compare_live and result != _live_cache_binding(root, role, label):
    raise AnalysisError(f'{label} live cache tree differs from its binding.')
  return result


def _validate_cache_hit_evidence(
    value: Any, *, phase: str, expected_hit: bool, label: str,
) -> dict[str, Any]:
  node = _exact_keys(value, _CACHE_HIT_KEYS, label)
  if phase not in {
      'external_preflight', 'model_precompile', 'model_lower_failure',
      'model_compile_failure', 'model_post_compile',
  }:
    raise AnalysisError(f'{label} cache-evidence phase is unknown.')
  external = phase == 'external_preflight'
  compile_stage_not_applicable = phase in {
      'external_preflight', 'model_precompile', 'model_lower_failure',
  }
  expected = {
      'pre_import_files_present': expected_hit,
      'default_user_cache_path_eligible': False,
      'persistent_compilation_cache_hit_reported': None if external else False,
      'executable_deserialized': None if external else False,
      'compile_skipped': None if compile_stage_not_applicable else False,
      'compile_stage_not_applicable': compile_stage_not_applicable,
      'old_cache_input_opened': False, 'routing_exact': True,
      'cache_hit': expected_hit,
  }
  if node != expected:
    raise AnalysisError(f'{label} cache-hit formula changed.')
  return dict(node)


def _validate_model_cache_final(
    value: Any, *, compiler: Mapping[str, Any] | None, status: str,
    reason: str | None,
) -> dict[str, Any]:
  node = _exact_keys(value, {
      'pre_import', 'historical_stage', 'historical_binding', 'terminal',
      'cache_hit_evidence', 'historical_to_terminal_tree_exact',
      'historical_to_terminal_equality_is_a_gate',
      'historical_snapshot_not_reauthenticated_as_live_files',
      'default_user_cache_paths_eligible',
      'cache_outputs_are_diagnostic_only',
  }, 'RUN_COMPLETE.model_kernel_cache_final')
  pre_import = _validate_cache_binding(
      node['pre_import'], root=_MODEL_CACHE_DIR, role='model',
      label='terminal model-cache pre-import', compare_live=False,
  )
  if pre_import['directory_paths'] != ['.', 'triton', 'xdg'] or pre_import['file_count'] != 0:
    raise AnalysisError('Terminal model-cache pre-import binding is not fresh.')
  terminal = _validate_cache_binding(
      node['terminal'], root=_MODEL_CACHE_DIR, role='model',
      label='terminal live model cache', compare_live=True,
  )
  historical_stage = node['historical_stage']
  historical = node['historical_binding']
  evidence = node['cache_hit_evidence']
  if compiler is None:
    pre_import_hit = (
        status == 'controlled_stop_cache_hit'
        and reason == 'model_cache_pre_import_hit'
    )
    if historical_stage is not None or historical is not None or (
        evidence is None
    ) is pre_import_hit:
      raise AnalysisError('Precompiler terminal invented historical cache evidence.')
    if pre_import_hit:
      _validate_cache_hit_evidence(
          evidence, phase='model_precompile', expected_hit=True,
          label='pre-import model cache hit evidence',
      )
    equality = None
  else:
    diagnostic_failure = compiler.get('status') == 'diagnostic_provenance_failure'
    expected_stage = (
        'post_compile' if compiler.get('successful_compile_count') == 1
        else 'post_failure'
    )
    if historical_stage != expected_stage:
      raise AnalysisError('Compiler/terminal cache phase linkage changed.')
    historical = _validate_cache_binding(
        historical, root=_MODEL_CACHE_DIR, role='model',
        label=f'compiler cache {expected_stage}', compare_live=False,
    )
    if not diagnostic_failure:
      provenance = _exact_keys(
          compiler.get('kernel_cache_provenance'),
          {
              'pre_import', expected_stage, 'cache_hit_evidence',
              'default_user_cache_paths_eligible',
              'cache_outputs_are_diagnostic_only',
          }, 'compiler.kernel_cache_provenance',
      )
      if (
          provenance['pre_import'] != pre_import
          or historical != provenance[expected_stage]
          or evidence != provenance['cache_hit_evidence']
      ):
        raise AnalysisError('Compiler/terminal historical cache evidence differs.')
    elif reason not in {
        'diagnostic_parser_failure', 'diagnostic_persistence_failure',
        'cache_signal_unavailable', 'fingerprint_formula_mismatch',
    }:
      raise AnalysisError('Diagnostic cache evidence used by another terminal.')
    expected_hit = reason == 'model_cache_post_compile_hit'
    if compiler.get('successful_compile_count') == 1:
      evidence_phase = 'model_post_compile'
    elif compiler.get('failure_stage') == 'lower':
      evidence_phase = 'model_lower_failure'
    elif compiler.get('failure_stage') == 'compile':
      evidence_phase = 'model_compile_failure'
    else:
      raise AnalysisError('Compiler cache-evidence phase is not attributable.')
    _validate_cache_hit_evidence(
        evidence, phase=evidence_phase, expected_hit=expected_hit,
        label='terminal model cache hit evidence',
    )
    equality = historical['tree_sha256'] == terminal['tree_sha256']
  if (
      node['historical_to_terminal_tree_exact'] is not equality
      or node['historical_to_terminal_equality_is_a_gate'] is not False
      or node['historical_snapshot_not_reauthenticated_as_live_files'] is not True
      or node['default_user_cache_paths_eligible'] is not False
      or node['cache_outputs_are_diagnostic_only'] is not True
  ):
    raise AnalysisError('Terminal cache diagnostic boundary changed.')
  if status == 'controlled_stop_cache_hit' and not (
      reason in {'model_cache_pre_import_hit', 'model_cache_post_compile_hit'}
  ):
    raise AnalysisError('Cache-hit terminal reason changed.')
  return {
      'path': str(_MODEL_CACHE_DIR.resolve()),
      'pre_import_binding': pre_import,
      'historical_binding': historical,
      'terminal_live_binding': terminal,
      'directory_paths_exact': True,
      'cache_hit': reason in {
          'model_cache_pre_import_hit', 'model_cache_post_compile_hit'
      },
      'cache_hit_evidence': evidence,
      'historical_to_terminal_equality_is_a_gate': False,
  }


def _terminal_failure_model_cache(start: Mapping[str, Any]) -> dict[str, Any]:
  """Binds the live diagnostic cache without inventing a terminal snapshot."""
  same = _exact_keys(
      start.get('same_process_preflight'), _SAME_PROCESS_PREFLIGHT_KEYS,
      'START.same_process_preflight for publication failure',
  )
  pre_import = _validate_cache_binding(
      same.get('model_cache_pre_import'), root=_MODEL_CACHE_DIR, role='model',
      label='publication-failure model-cache pre-import', compare_live=False,
  )
  if pre_import['directory_paths'] != ['.', 'triton', 'xdg'] or pre_import['file_count'] != 0:
    raise AnalysisError('Publication-failure model cache was not fresh pre-import.')
  terminal = _live_cache_binding(
      _MODEL_CACHE_DIR, 'model', 'publication-failure live model cache'
  )
  return {
      'path': str(_MODEL_CACHE_DIR.resolve()),
      'pre_import_binding': pre_import,
      'historical_binding': None,
      'terminal_live_binding': terminal,
      'directory_paths_exact': True,
      'cache_hit': False,
      'cache_hit_evidence': None,
      'historical_to_terminal_equality_is_a_gate': False,
  }


def _predecessor_path_contract(version: str) -> dict[str, str]:
  prefix = f'v3_3_4{version}'
  return {
      'analysis_attempt': str((
          _HERE / 'results'
          / f'{prefix}_development_ood_sidecar_analysis_attempt'
      ).resolve()),
      'analysis_output': str((
          _HERE / 'results' / f'{prefix}_development_ood_sidecar_analysis'
      ).resolve()),
      'external_cache': str((
          _HERE / 'results' / f'{prefix}_preflight_kernel_cache'
      ).resolve()),
      'external_preflight': str((
          _HERE / 'results' / f'{prefix}_device_preflight'
      ).resolve()),
      'model_cache': str((
          _HERE / 'results' / f'{prefix}_model_kernel_cache'
      ).resolve()),
      'model_run': str((
          _HERE / 'results' / f'{prefix}_development_ood_sidecar_one_shot'
      ).resolve()),
  }


def _expected_nonpublication_terminal_contract() -> dict[str, Any]:
  stages = (
      'stablehlo_text_extraction', 'pre_backend_hlo_text_extraction',
      'compiled_hlo_text_extraction',
      'source_program_gate_derivation_for_diagnostic_failure',
      'diagnostic_failure_record_construction',
  )
  reasons = (
      'diagnostic_parser_failure', 'diagnostic_persistence_failure',
      'cache_signal_unavailable', 'fingerprint_formula_mismatch',
  )
  return {
      'artifact_path': 'NONPUBLICATION_TERMINAL_FAILURE.json',
      'schema_version': 'v3.3.4.3-nonpublication-terminal-v1',
      'status': 'incomplete_nonpublication_infrastructure_failure',
      'stop_reason': 'post_compile_nonpublication_infrastructure_failure',
      'artifact_role': 'nonpublication_terminal_failure',
      'predecessor_amendments': {
          'v3_3_4': {
              'commit': PREDECESSOR_AMENDMENT_COMMIT,
              'path': str(_PREDECESSOR_AMENDMENT_PATH.resolve()),
              'sha256': PREDECESSOR_AMENDMENT_SHA256,
          },
          'v3_3_4_1': {
              'commit': PUBLICATION_AMENDMENT_COMMIT,
              'path': str(_PUBLICATION_AMENDMENT_PATH.resolve()),
              'sha256': PUBLICATION_AMENDMENT_SHA256,
          },
          'v3_3_4_2': {
              'commit': V3342_AMENDMENT_COMMIT,
              'path': str(_V3342_AMENDMENT_PATH.resolve()),
              'sha256': V3342_AMENDMENT_SHA256,
          },
      },
      'predecessor_path_contract': {
          'v3_3_4': _predecessor_path_contract(''),
          'v3_3_4_1': _predecessor_path_contract('_1'),
          'v3_3_4_2': _predecessor_path_contract('_2'),
      },
      'failure_stages': list(stages),
      'triggering_diagnostic_stop_reasons': list(reasons),
      'keys': list(NONPUBLICATION_TERMINAL_KEY_ORDER),
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
          'diagnostic_record_construction_phase_equals_source_program_exact': True,
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
          'status': 'complete_incomplete_nonpublication_infrastructure_archive',
          'decision': 'post_compile_nonpublication_failure_no_scientific_analysis',
          'compiler_state': 'compiled_without_legal_graph_gate_record',
          'terminal_kind': 'nonpublication_terminal_failure',
          'control_state_eligible': False,
      },
      'publication_error_fallback': 'v3.3.4.1-TERMINAL_FAILURE-only',
      'ordinary_construction_error_fallback': 'terminal_less_consumed_prefix',
  }


def _validate_nonpublication_terminal_contract(value: Any) -> dict[str, Any]:
  expected = _expected_nonpublication_terminal_contract()
  if value != expected:
    raise AnalysisError('Frozen v3.3.4.3 nonpublication contract changed.')
  return copy.deepcopy(expected)


def _validate_publication_contract(value: Any) -> dict[str, Any]:
  keys = {
      'schema_version', 'method', 'temp_name_regex', 'nonce_bytes',
      'open_flags', 'initial_mode', 'sealed_mode', 'rename_flags',
      'same_directory_required', 'keep_fd_open_through_rename',
      'file_fsync_count', 'parent_fsync_required',
      'post_publish_inode_revalidation_required', 'no_replace',
      'no_fallback', 'no_retry', 'temporary_orphan_preservation_required',
      'durability_uncertain_final_preservation_required',
      'successful_publication_object_keys',
      'publication_failure_object_keys', 'entry_state_object_keys',
      'external_preflight_probe_contract',
  }
  node = _exact_keys(value, keys, 'freeze.publication_contract_v3_3_4_1')
  expected = {
      'schema_version': PUBLICATION_SCHEMA_VERSION,
      'method': PUBLICATION_METHOD,
      'temp_name_regex': (
          r'^\.v3343\.tmp\.[1-9][0-9]*\.[0-9]{6}\.[0-9a-f]{32}$'
      ),
      'nonce_bytes': 16,
      'open_flags': ['O_RDWR', 'O_CREAT', 'O_EXCL', 'O_NOFOLLOW', 'O_CLOEXEC'],
      'initial_mode': '0600', 'sealed_mode': '0400',
      'rename_flags': ['RENAME_NOREPLACE'],
      'same_directory_required': True,
      'keep_fd_open_through_rename': True,
      'file_fsync_count': 2, 'parent_fsync_required': True,
      'post_publish_inode_revalidation_required': True,
      'no_replace': True, 'no_fallback': True, 'no_retry': True,
      'temporary_orphan_preservation_required': True,
      'durability_uncertain_final_preservation_required': True,
      'successful_publication_object_keys': list(PUBLICATION_SUCCESS_KEYS),
      'publication_failure_object_keys': list(PUBLICATION_FAILURE_KEYS),
      'entry_state_object_keys': list(ENTRY_STATE_KEYS),
      'external_preflight_probe_contract': {
          'final_basename': 'atomic_publication_probe_v3_3_4_3.txt',
          'final_sha256': (
              '29c4509405ae8afa833da03d2ff5824fc2b842c6bc56df4d0aac2f7838685ea1'
          ),
          'final_size_bytes': 49,
          'collision_sha256': (
              'c9ceb22bdeed1f370e6ca33814313f3a3d5007e286954aa3e4f31e5d041b8ba0'
          ),
          'collision_size_bytes': 39, 'collision_errno': 17,
          'collision_temp_preserved': True,
          'parent_fsync_exact_required': True,
      },
  }
  if dict(node) != expected:
    raise AnalysisError('Frozen v3.3.4.3 publication contract changed.')
  return dict(node)








def _validate_freeze_v3343(
    run_dir: Path, *, bundle_root: Path,
    active_started_sha256: str | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, str], dict[int, Any], dict[str, Any]]:
  if run_dir.resolve() != _RUN_DIR.resolve() or bundle_root.resolve() != _REPO_ROOT:
    raise AnalysisError('v3.3.4.3 production run/repository path changed.')
  _assert_predecessor_v334_paths_absent('freeze validation')
  _validate_analysis_destination_state(active_started_sha256)
  if _sha256(_AMENDMENT_PATH) != AMENDMENT_SHA256:
    raise AnalysisError('v3.3.4.3 amendment bytes changed.')
  amendment_relative = _AMENDMENT_PATH.relative_to(_REPO_ROOT).as_posix()
  if _git_blob_sha256(AMENDMENT_COMMIT, amendment_relative) != AMENDMENT_SHA256:
    raise AnalysisError('Bound v3.3.4.3 amendment Git blob changed.')
  if _sha256(_V3342_AMENDMENT_PATH) != V3342_AMENDMENT_SHA256:
    raise AnalysisError('Bound predecessor v3.3.4.2 amendment bytes changed.')
  v3342_relative = _V3342_AMENDMENT_PATH.relative_to(_REPO_ROOT).as_posix()
  if (
      _git_blob_sha256(V3342_AMENDMENT_COMMIT, v3342_relative)
      != V3342_AMENDMENT_SHA256
  ):
    raise AnalysisError('Bound predecessor v3.3.4.2 amendment Git blob changed.')
  if _sha256(_PREDECESSOR_AMENDMENT_PATH) != PREDECESSOR_AMENDMENT_SHA256:
    raise AnalysisError('Bound predecessor v3.3.4 amendment bytes changed.')
  if _sha256(_PUBLICATION_AMENDMENT_PATH) != PUBLICATION_AMENDMENT_SHA256:
    raise AnalysisError('Bound predecessor v3.3.4.1 amendment bytes changed.')
  predecessor_relative = _PREDECESSOR_AMENDMENT_PATH.relative_to(
      _REPO_ROOT
  ).as_posix()
  if (
      _git_blob_sha256(PREDECESSOR_AMENDMENT_COMMIT, predecessor_relative)
      != PREDECESSOR_AMENDMENT_SHA256
  ):
    raise AnalysisError('Bound predecessor v3.3.4 amendment Git blob changed.')
  publication_relative = _PUBLICATION_AMENDMENT_PATH.relative_to(
      _REPO_ROOT
  ).as_posix()
  if (
      _git_blob_sha256(PUBLICATION_AMENDMENT_COMMIT, publication_relative)
      != PUBLICATION_AMENDMENT_SHA256
  ):
    raise AnalysisError('Bound predecessor v3.3.4.1 amendment Git blob changed.')
  freeze = _read_json(_FREEZE_PATH, 'v3.3.4.3 freeze')
  _exact_keys(freeze, _FREEZE_KEYS, 'v3.3.4.3 freeze')
  _validate_publication_contract(freeze.get('publication_contract_v3_3_4_1'))
  _validate_nonpublication_terminal_contract(
      freeze.get('nonpublication_terminal_contract_v3_3_4_3')
  )
  freeze_sha = _sha256(_FREEZE_PATH)
  expected_scalars = {
      'attempt_id': ATTEMPT_ID, 'script_version': SCRIPT_VERSION,
      'amendment_path': str(_AMENDMENT_PATH.resolve()),
      'amendment_sha256': AMENDMENT_SHA256,
      'amendment_commit': AMENDMENT_COMMIT,
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'output_dir': str(_RUN_DIR.resolve()),
      'analysis_dir': str(_ANALYSIS_DIR.resolve()),
      'analysis_attempt_dir': str(_ANALYSIS_ATTEMPT_DIR.resolve()),
      'preflight_dir': str(_PREFLIGHT_DIR.resolve()),
      'preflight_kernel_cache_dir': str(_PREFLIGHT_CACHE_DIR.resolve()),
      'model_kernel_cache_dir': str(_MODEL_CACHE_DIR.resolve()),
      'ood_record_count': 80, 'model_apply_count': 320,
      'eight_row_compile_count': 1, 'six_row_compile_count': 0,
      'identity_rerun_count': 0, 'main_cube_rerun_count': 0,
      'old_ood_records_reused': 0, 'max_wall_time_seconds': 7200,
      'max_output_bytes': 1_073_741_824,
      'compiled_backend_equality_is_a_gate': False,
  }
  for key, expected in expected_scalars.items():
    if freeze.get(key) != expected:
      raise AnalysisError(f'v3.3.4.3 freeze.{key} changed.')
  if (
      freeze.get('recipient_orders') != list(RECIPIENT_ORDERS)
      or freeze.get('ood_anchor_ids') != list(ANCHOR_IDS)
      or freeze.get('eight_row_roles') != list(EIGHT_ROLES)
      or freeze.get('eight_row_natural_identity_rows') != list(IDENTITY_ROWS)
      or freeze.get('eight_row_intended_donor_rows') != list(INTENDED_DONOR_ROWS)
      or freeze.get('eight_row_unrelated_donor_rows') != list(UNRELATED_DONOR_ROWS)
      or freeze.get('invariant_rows_between_calls') != list(INVARIANT_ROWS)
  ):
    raise AnalysisError('v3.3.4.3 freeze scientific order/role contract changed.')
  inventory = freeze.get('file_sha256')
  source_contract = freeze.get('source_inventory_contract')
  if not isinstance(inventory, Mapping) or len(inventory) != 108:
    raise AnalysisError('v3.3.4.3 freeze source inventory is not 108 rows.')
  if not isinstance(source_contract, Mapping):
    raise AnalysisError('v3.3.4.3 source inventory contract is absent.')
  _exact_keys(source_contract, {
      'source_row_count', 'rows', 'prospective_upstream_source_file_count',
      'loaded_scientific_module_contract',
  }, 'freeze.source_inventory_contract')
  rows = source_contract.get('rows')
  if (
      source_contract.get('source_row_count') != 108
      or source_contract.get('prospective_upstream_source_file_count') != 26
      or not isinstance(rows, list) or len(rows) != 108
      or not isinstance(
          source_contract.get('loaded_scientific_module_contract'), list
      )
  ):
    raise AnalysisError('v3.3.4.3 source inventory contract counts changed.')
  new_shells = {_V3343_SOURCE_PATHS[1], _V3343_SOURCE_PATHS[8]}
  row_map = {}
  for raw in rows:
    row = _exact_keys(
        raw, {'path', 'sha256', 'size_bytes', 'git_mode'},
        'freeze.source_inventory_contract.row',
    )
    relative = row.get('path')
    if not isinstance(relative, str) or relative in row_map:
      raise AnalysisError('Freeze source inventory has a duplicate path.')
    if (
        row.get('sha256') != inventory.get(relative)
        or row.get('git_mode') not in {'100644', '100755'}
        or relative in new_shells and row.get('git_mode') != '100755'
        or relative in set(_V3343_SOURCE_PATHS) - new_shells
        and row.get('git_mode') != '100644'
        or isinstance(row.get('size_bytes'), bool)
        or not isinstance(row.get('size_bytes'), int)
        or row['size_bytes'] < 1
    ):
      raise AnalysisError(f'Freeze source inventory row changed: {relative}.')
    row_map[relative] = dict(row)
  if list(row_map) != sorted(inventory) or set(row_map) != set(inventory):
    raise AnalysisError('Freeze source rows are not the exact sorted inventory.')
  try:
    head = subprocess.check_output(
        ('git', '-C', str(bundle_root), 'rev-parse', 'HEAD'), text=True
    ).strip()
    subprocess.check_call(
        ('git', '-C', str(bundle_root), 'diff', '--quiet', 'HEAD', '--')
    )
  except subprocess.CalledProcessError as error:
    raise AnalysisError('v3.3.4.3 tracked repository is not clean.') from error
  for relative, digest in inventory.items():
    if not isinstance(relative, str) or not _is_sha256(digest):
      raise AnalysisError('v3.3.4.3 source inventory row is malformed.')
    path = bundle_root / relative
    _strict_regular(path, f'v3.3.4.3 source {relative}')
    mode_line = subprocess.check_output(
        ('git', '-C', str(bundle_root), 'ls-tree', head, '--', relative),
        text=True,
    ).strip()
    git_mode = mode_line.split()[0] if mode_line else None
    if (
        _sha256(path) != digest or _git_blob_sha256(head, relative) != digest
        or path.stat().st_size != row_map[relative]['size_bytes']
        or git_mode != row_map[relative]['git_mode']
    ):
      raise AnalysisError(f'v3.3.4.3 source differs from launch HEAD: {relative}.')
  prior333 = _validate_prior_v3_3_3()
  prior331 = _validate_prior_v3_3_3_1()
  for path in (_PRIOR_ANALYZER_ATTEMPT_DIR, _PRIOR_ANALYZER_OUTPUT_DIR):
    if path.exists() or path.is_symlink():
      raise AnalysisError('Original v3.3.3 analyzer destination appeared.')
  original_manifest_value = _read_json(
      _ORIGINAL_CUBE_DIR / 'RAW_MANIFEST.json', 'original v3.3 cube manifest'
  )
  original_manifest = original_manifest_value.get('artifact_sha256')
  if not isinstance(original_manifest, Mapping) or len(original_manifest) != 5_142:
    raise AnalysisError('Original v3.3 cube manifest changed.')
  return (
      freeze, freeze_sha, prior333, dict(original_manifest), {}, prior331,
  )


def _validate_start_v3343(
    run_dir: Path, freeze: Mapping[str, Any], freeze_sha: str, *,
    prior333: Mapping[str, Any], prior331: Mapping[str, Any],
) -> dict[str, Any]:
  start = _read_json(run_dir / 'ATTEMPT_STARTED.json', 'ATTEMPT_STARTED')
  _exact_keys(start, _START_KEYS, 'ATTEMPT_STARTED')
  authorization = _exact_keys(
      start.get('external_freeze_authorization'),
      {
          'git_head', 'freeze_path', 'freeze_sha256', 'freeze_size_bytes',
          'live_equals_git_show', 'tracked_clean', 'authorization_source',
      }, 'START.external_freeze_authorization',
  )
  current_head = subprocess.check_output(
      ('git', '-C', str(_REPO_ROOT), 'rev-parse', 'HEAD'), text=True
  ).strip()
  freeze_relative = _FREEZE_PATH.relative_to(_REPO_ROOT).as_posix()
  try:
    frozen_bytes = subprocess.check_output(
        ('git', '-C', str(_REPO_ROOT), 'show',
         f'{current_head}:{freeze_relative}')
    )
  except subprocess.CalledProcessError as error:
    raise AnalysisError('The v3.3.4.3 freeze is not tracked at launch HEAD.') from error
  if (
      authorization.get('freeze_path') != str(_FREEZE_PATH.resolve())
      or authorization.get('freeze_sha256') != freeze_sha
      or authorization.get('freeze_size_bytes') != _FREEZE_PATH.stat().st_size
      or authorization.get('live_equals_git_show') is not True
      or authorization.get('tracked_clean') is not True
      or authorization.get('authorization_source')
      != 'external_post_commit_audit'
      or start.get('git_head') != authorization.get('git_head')
      or current_head != authorization.get('git_head')
      or hashlib.sha256(frozen_bytes).hexdigest() != freeze_sha
      or frozen_bytes != _FREEZE_PATH.read_bytes()
  ):
    raise AnalysisError('START external freeze authorization changed.')
  expected = {
      'status': 'attempt_started', 'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION, 'amendment_sha256': AMENDMENT_SHA256,
      'amendment_commit': AMENDMENT_COMMIT,
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'freeze_path': str(_FREEZE_PATH.resolve()), 'freeze_sha256': freeze_sha,
      'confirmation_scope_disclosure': CONFIRMATION_DISCLOSURE,
      'confirmation_model_calls': 0, 'scientific_summary_computed': False,
      'donor_normalization_computed': False,
      'shapley_or_nomination_computed': False,
      'interaction_or_resolution_computed': False,
      'nomination_performed': False, 'combined_analysis_permitted': False,
      'prior_v3_3_3_binding': prior333,
      'prior_v3_3_3_1_archive_binding': prior331,
      'program_signature_contract': freeze[
          'program_signature_attestation_contract'
      ],
      'cache_isolation_contract': freeze['cache_isolation_contract'],
  }
  for key, value in expected.items():
    if start.get(key) != value:
      raise AnalysisError(f'ATTEMPT_STARTED.{key} changed.')
  if (
      isinstance(start.get('runner_pid'), bool)
      or not isinstance(start.get('runner_pid'), int)
      or start['runner_pid'] < 1
      or isinstance(start.get('parent_pid'), bool)
      or not isinstance(start.get('parent_pid'), int)
      or start['parent_pid'] < 1
  ):
    raise AnalysisError('START PID fields are malformed.')
  _finite(start.get('started_at_unix_s'), 'START.started_at_unix_s')
  _validate_source_audit(
      start.get('source_input_audit'),
      start.get('source_input_audit_content_binding'),
      (True, True, True, True, None, None, None, None),
      'START.source_input_audit',
  )
  fresh = _exact_keys(
      start.get('fresh_paths'),
      {
          'device_preflight', 'preflight_kernel_cache', 'model_kernel_cache',
          'model_run', 'analysis_attempt', 'analysis_output',
      }, 'START.fresh_paths',
  )
  if fresh != {
      'device_preflight': str(_PREFLIGHT_DIR.resolve()),
      'preflight_kernel_cache': str(_PREFLIGHT_CACHE_DIR.resolve()),
      'model_kernel_cache': str(_MODEL_CACHE_DIR.resolve()),
      'model_run': str(_RUN_DIR.resolve()),
      'analysis_attempt': str(_ANALYSIS_ATTEMPT_DIR.resolve()),
      'analysis_output': str(_ANALYSIS_DIR.resolve()),
  }:
    raise AnalysisError('START fresh-path binding changed.')
  budgets = _exact_keys(
      start.get('budgets'),
      {
          'max_wall_time_seconds', 'max_output_bytes', 'expected_records',
          'expected_model_applies', 'lower_attempt_budget',
          'compile_attempt_budget', 'run_complete_size_cap_bytes',
      }, 'START.budgets',
  )
  if budgets != {
      'max_wall_time_seconds': 7200, 'max_output_bytes': 1_073_741_824,
      'expected_records': 80, 'expected_model_applies': 320,
      'lower_attempt_budget': 1, 'compile_attempt_budget': 1,
      'run_complete_size_cap_bytes': 16_777_216,
  }:
    raise AnalysisError('START budget contract changed.')
  if start.get('execution_contract') != freeze['terminal_contract'][
      'execution_contract'
  ]:
    raise AnalysisError('START execution contract changed.')
  source_contract = freeze['source_inventory_contract']
  source_rows = source_contract['rows']
  digest = hashlib.sha256()
  for row in source_rows:
    digest.update(row['path'].encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(row['sha256']))
  inventory_attestation = _exact_keys(
      start.get('source_inventory_attestation'),
      {'row_count', 'rows', 'tree_sha256', 'git_head', 'tracked_clean',
       'live_equals_head'},
      'START.source_inventory_attestation',
  )
  if inventory_attestation != {
      'row_count': 108, 'rows': source_rows,
      'tree_sha256': digest.hexdigest(),
      'git_head': start['git_head'], 'tracked_clean': True,
      'live_equals_head': True,
  }:
    raise AnalysisError('START source-inventory attestation changed.')
  _validate_preflight_and_same_process(start, freeze)
  return dict(start)


def _validate_preflight_and_same_process(
    start: Mapping[str, Any], freeze: Mapping[str, Any],
) -> dict[str, Any]:
  expected_names = [
      '.allocation.lock', '.preflight_0000.reserved',
      'preflight_0000.json', 'preflight_0000.stderr.log',
      'preflight_0000.stdout.log',
  ]
  files = []
  if _PREFLIGHT_DIR.is_symlink() or not _PREFLIGHT_DIR.is_dir():
    raise AnalysisError('External preflight directory is absent or unsafe.')
  for entry in _PREFLIGHT_DIR.iterdir():
    mode = entry.lstat().st_mode
    if entry.is_symlink() or not stat.S_ISREG(mode):
      raise AnalysisError('External preflight tree contains an unsafe entry.')
    expected_mode = 0o600 if entry.name == '.allocation.lock' else 0o400
    if stat.S_IMODE(mode) != expected_mode:
      raise AnalysisError(f'External preflight mode changed: {entry.name}.')
    files.append(entry)
  if sorted(path.name for path in files) != expected_names:
    raise AnalysisError('External preflight exact five-file membership changed.')
  record_path = _PREFLIGHT_DIR / 'preflight_0000.json'
  record = _read_json(record_path, 'external preflight')
  _exact_keys(record, {
      'amendment_sha256', 'atomic_publication_probe', 'created_at_unix_s',
      'external_freeze_authorization', 'external_cache_post_observation',
      'external_cache_hit_evidence', 'failure', 'freeze', 'freeze_sha256',
      'logs', 'no_jit_or_array_kernel', 'no_model_or_biological_access',
      'observation', 'original_protocol_sha256', 'preflight_attempt_number',
      'script_version', 'status', 'warnings',
  }, 'external preflight')
  if (
      record.get('status') != 'pass' or record.get('failure') is not None
      or record.get('preflight_attempt_number') != 0
      or record.get('amendment_sha256') != AMENDMENT_SHA256
      or record.get('freeze_sha256') != start['freeze_sha256']
      or record.get('external_freeze_authorization')
      != start['external_freeze_authorization']
      or record.get('original_protocol_sha256') != ORIGINAL_PROTOCOL_SHA256
      or record.get('no_jit_or_array_kernel') is not True
      or record.get('no_model_or_biological_access') is not True
      or not isinstance(record.get('warnings'), list)
      or any(not isinstance(item, str) for item in record['warnings'])
  ):
    raise AnalysisError('External preflight pass predicates changed.')
  _finite(record.get('created_at_unix_s'), 'external preflight timestamp')
  observation = _exact_keys(record.get('observation'), {
      'atomic_publication_supported', 'environment', 'hostname',
      'jax_default_backend', 'jax_enable_compilation_cache',
      'jax_gpu_devices', 'jax_module_version', 'jaxlib_module_version',
      'kernel', 'no_jit_no_array_no_model', 'nvidia_smi', 'packages', 'pid',
      'platform', 'python_executable', 'python_version', 'runtime_environment',
      'v3_3_4_3_runtime_environment',
  }, 'external preflight observation')
  if (
      isinstance(observation.get('pid'), bool)
      or not isinstance(observation.get('pid'), int)
      or observation['pid'] < 1
      or observation.get('jax_default_backend') != 'gpu'
      or observation.get('no_jit_no_array_no_model') is not True
      or observation.get('atomic_publication_supported') is not True
  ):
    raise AnalysisError('External preflight observation changed.')
  publication_probe = _validate_atomic_publication_probe(
      record.get('atomic_publication_probe'), external_pid=observation['pid'],
  )
  logs = _exact_keys(record.get('logs'), {'stdout', 'stderr'}, 'preflight.logs')
  for name, raw in logs.items():
    binding = _exact_keys(
        raw, {'path', 'sha256', 'size_bytes'}, f'preflight.logs.{name}'
    )
    path = _PREFLIGHT_DIR / f'preflight_0000.{name}.log'
    if (
        binding.get('path') != str(path.resolve())
        or binding.get('sha256') != _sha256(path)
        or binding.get('size_bytes') != path.stat().st_size
    ):
      raise AnalysisError(f'External preflight {name} log binding changed.')
  cache_post = _validate_cache_binding(
      record.get('external_cache_post_observation'),
      root=_PREFLIGHT_CACHE_DIR, role='external_preflight',
      label='external preflight terminal cache', compare_live=True,
  )
  cache_hit = _validate_cache_hit_evidence(
      record.get('external_cache_hit_evidence'), phase='external_preflight',
      expected_hit=False, label='external preflight cache evidence',
  )
  successful = _exact_keys(start.get('successful_preflight'), {
      'artifact_binding', 'root_file_count', 'root_file_tree_sha256',
      'external_pid', 'status', 'external_freeze_authorization',
      'external_cache_post_observation', 'external_cache_hit_evidence',
  }, 'START.successful_preflight')
  artifact_binding = _exact_keys(
      successful.get('artifact_binding'), {'path', 'sha256', 'size_bytes'},
      'START successful preflight binding',
  )
  file_map = {
      path.name: {'sha256': _sha256(path), 'size_bytes': path.stat().st_size}
      for path in files
  }
  if (
      artifact_binding != _absolute_binding(record_path)
      or successful.get('root_file_count') != 5
      or successful.get('root_file_tree_sha256')
      != _binding_map_digest(file_map)
      or successful.get('external_pid') != observation['pid']
      or successful.get('status') != 'pass'
      or successful.get('external_freeze_authorization')
      != start['external_freeze_authorization']
      or successful.get('external_cache_post_observation') != cache_post
      or successful.get('external_cache_hit_evidence') != cache_hit
  ):
    raise AnalysisError('START successful-preflight binding changed.')
  same = _exact_keys(
      start.get('same_process_preflight'), _SAME_PROCESS_PREFLIGHT_KEYS,
      'START.same_process_preflight',
  )
  if (
      same.get('pid') != start['runner_pid']
      or same.get('parent_pid') != start['parent_pid']
      or same.get('external_preflight_pid') != observation['pid']
      or same.get('pid') == observation['pid']
      or same.get('default_backend') != 'gpu'
      or same.get('freeze_sha256') != start['freeze_sha256']
      or same.get('external_freeze_authorization')
      != start['external_freeze_authorization']
      or same.get('external_preflight_binding') != artifact_binding
      or same.get('external_preflight_tree_sha256')
      != successful['root_file_tree_sha256']
      or same.get('current_source_inventory_exact') is not True
      or same.get('prior_artifacts_exact') is not True
      or same.get('no_model_constructed') is not True
      or same.get('no_jit_or_array_kernel') is not True
      or start.get('same_process_preflight_content_binding')
      != _content_binding(same)
  ):
    raise AnalysisError('START same-process preflight changed.')
  _finite(same.get('created_at_unix_s'), 'same-process preflight timestamp')
  model_pre = _validate_cache_binding(
      same.get('model_cache_pre_import'), root=_MODEL_CACHE_DIR, role='model',
      label='model pre-import cache', compare_live=False,
  )
  if model_pre['directory_paths'] != ['.', 'triton', 'xdg'] or model_pre['file_count'] != 0:
    raise AnalysisError('Model pre-import cache was not exact/fresh.')
  return {
      'path': str(_PREFLIGHT_DIR.resolve()), 'file_count': 5,
      'directory_count': 1, 'directory_paths': ['.'],
      'file_bindings': file_map,
      'file_tree_sha256': _binding_map_digest(file_map),
      'directory_tree_sha256': _directory_digest(['.']),
      'status': 'pass', 'external_pid': observation['pid'],
      'runner_pid': start['runner_pid'], 'pids_distinct': True,
      'device_exact': True, 'cache_role_exact': True,
  }


_MANIFEST_KEYS = {
    'schema_version', 'status', 'attempt_id', 'external_freeze_authorization',
    'valid_artifact_count', 'artifact_bindings', 'artifact_tree_sha256',
    'valid_recipient_anchor_pairs', 'failed_current_binding',
    'dispatch_started_count', 'dispatch_completed_count',
    'dispatch_started_bindings', 'dispatch_started_tree_sha256',
    'dispatch_completed_bindings', 'dispatch_completed_tree_sha256',
    'source_input_audit_content_binding',
    'same_object_attestation_content_binding', 'created_at_unix_s',
}

_CALL_ROLES = ('intended', 'intended_repeat', 'unrelated', 'unrelated_repeat')


def _event_identity(global_index: int, cases: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
  execution_index, call_index = divmod(global_index, 4)
  recipient_order, anchor_minor = divmod(execution_index, 4)
  anchor = ANCHOR_IDS[anchor_minor]
  return {
      'execution_index': execution_index, 'recipient_order': recipient_order,
      'recipient_variant_id': cases[recipient_order]['variant_id'],
      'anchor_id': anchor, 'call_index_within_record': call_index,
      'call_role': _CALL_ROLES[call_index], 'global_dispatch_index': global_index,
  }


def _validate_dispatch_event(
    value: Any, *, global_index: int, completed: bool,
    cases: Mapping[int, Mapping[str, Any]], runner_pid: int,
    expected_source_sha: str, expected_object_sha: str,
    started_sha: str | None = None,
) -> None:
  common = {
      'schema_version', 'event', 'attempt_id', 'script_version',
      'execution_index', 'recipient_order', 'recipient_variant_id', 'anchor_id',
      'call_index_within_record', 'call_role', 'global_dispatch_index',
      'runner_pid', 'source_input_audit_sha256',
      'same_object_attestation_sha256',
  }
  keys = common | (
      {'started_event_sha256', 'returned', 'completed_at_unix_s'}
      if completed else {'started_at_unix_s'}
  )
  node = _exact_keys(value, keys, 'dispatch event')
  expected = {
      'attempt_id': ATTEMPT_ID, 'script_version': SCRIPT_VERSION,
      'runner_pid': runner_pid,
      'source_input_audit_sha256': expected_source_sha,
      'same_object_attestation_sha256': expected_object_sha,
      **_event_identity(global_index, cases),
      'event': 'dispatch_completed' if completed else 'dispatch_started',
  }
  for key, item in expected.items():
    if node.get(key) != item:
      raise AnalysisError(f'Dispatch event changed at {key}.')
  timestamp = 'completed_at_unix_s' if completed else 'started_at_unix_s'
  _finite(node.get(timestamp), f'dispatch event.{timestamp}')
  if completed and (
      node.get('returned') is not True
      or node.get('started_event_sha256') != started_sha
  ):
    raise AnalysisError('Completed event lacks its exact started event.')


def _validate_manifest_v3343(
    run_dir: Path, value: Any, *, cases: Mapping[int, Mapping[str, Any]],
    runner_pid: int, source_binding: Mapping[str, Any], object_binding: Mapping[str, Any] | None,
) -> tuple[list[tuple[int, int]], dict[str, Any] | None]:
  manifest = _exact_keys(value, _MANIFEST_KEYS, 'RAW_MANIFEST')
  _finite(manifest.get('created_at_unix_s'), 'RAW_MANIFEST.created_at_unix_s')
  if (
      manifest.get('schema_version') != 'v3.3.4.3-raw-manifest-v1'
      or manifest.get('attempt_id') != ATTEMPT_ID
      or manifest.get('external_freeze_authorization')
      != _read_json(run_dir / 'ATTEMPT_STARTED.json', 'START for manifest')[
          'external_freeze_authorization'
      ]
  ):
    raise AnalysisError('RAW_MANIFEST attempt changed.')
  k = manifest.get('valid_artifact_count')
  started_count = manifest.get('dispatch_started_count')
  completed_count = manifest.get('dispatch_completed_count')
  for name, count, limit in (
      ('valid_artifact_count', k, 80), ('dispatch_started_count', started_count, 320),
      ('dispatch_completed_count', completed_count, 320),
  ):
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= limit:
      raise AnalysisError(f'RAW_MANIFEST.{name} changed.')
  expected_pairs = list(_execution_order())[:k]
  pair_rows = manifest.get('valid_recipient_anchor_pairs')
  if not isinstance(pair_rows, list) or len(pair_rows) != k:
    raise AnalysisError('RAW_MANIFEST valid pair count changed.')
  for index, ((order, anchor), row) in enumerate(zip(expected_pairs, pair_rows, strict=True)):
    if row != {'execution_index': index, 'recipient_order': order, 'anchor_id': anchor}:
      raise AnalysisError('RAW_MANIFEST valid-pair order changed.')
  raw_paths = [_artifact_relative(cases[order], anchor) for order, anchor in expected_pairs]
  raw_bindings = _validate_binding_map(
      manifest.get('artifact_bindings'), run_dir, 'RAW_MANIFEST.artifacts',
      expected_paths=raw_paths,
  )
  if (
      len(raw_bindings) != k
      or manifest.get('artifact_tree_sha256') != _binding_map_digest(raw_bindings)
  ):
    raise AnalysisError('RAW_MANIFEST raw tree binding changed.')
  started_paths = [f'dispatch_journal/started/{index:03d}.json' for index in range(started_count)]
  completed_paths = [f'dispatch_journal/completed/{index:03d}.json' for index in range(completed_count)]
  started = _validate_binding_map(
      manifest.get('dispatch_started_bindings'), run_dir,
      'RAW_MANIFEST.started', expected_paths=started_paths,
  )
  completed = _validate_binding_map(
      manifest.get('dispatch_completed_bindings'), run_dir,
      'RAW_MANIFEST.completed', expected_paths=completed_paths,
  )
  if (
      manifest.get('dispatch_started_tree_sha256') != _binding_map_digest(started)
      or manifest.get('dispatch_completed_tree_sha256') != _binding_map_digest(completed)
  ):
    raise AnalysisError('Dispatch journal tree digest changed.')
  if manifest.get('source_input_audit_content_binding') != source_binding:
    raise AnalysisError('RAW_MANIFEST source-audit binding changed.')
  if manifest.get('same_object_attestation_content_binding') != object_binding:
    raise AnalysisError('RAW_MANIFEST same-object binding changed.')
  source_sha = source_binding['sha256']
  object_sha = object_binding['sha256'] if object_binding else ''
  for index, relative in enumerate(started_paths):
    _validate_dispatch_event(
        _read_json(run_dir / relative, relative), global_index=index,
        completed=False, cases=cases, runner_pid=runner_pid,
        expected_source_sha=source_sha, expected_object_sha=object_sha,
    )
  for index, relative in enumerate(completed_paths):
    _validate_dispatch_event(
        _read_json(run_dir / relative, relative), global_index=index,
        completed=True, cases=cases, runner_pid=runner_pid,
        expected_source_sha=source_sha, expected_object_sha=object_sha,
        started_sha=started[started_paths[index]]['sha256'],
    )
  failed = manifest.get('failed_current_binding')
  if failed is not None:
    failed = _validate_file_binding(failed, 'RAW_MANIFEST.failed_current', with_path=True)
    path = run_dir / failed['path']
    _strict_regular(path, 'failed-current artifact')
    if path.stat().st_size != failed['size_bytes'] or _sha256(path) != failed['sha256']:
      raise AnalysisError('Failed-current artifact binding changed.')
  status = manifest.get('status')
  expected_status = 'complete80' if k == 80 and failed is None else (
      'empty_controlled_stop' if k == 0 and failed is None else 'controlled_prefix'
  )
  if status != expected_status:
    raise AnalysisError('RAW_MANIFEST status differs from exact prefix.')
  return expected_pairs, failed


def _sequence_bindings_from_original(
    cases: Mapping[int, Mapping[str, Any]], original_manifest: Mapping[str, str],
) -> dict[int, Any]:
  result = {}
  for order, case in cases.items():
    slug = ''.join(
        character if character.isalnum() else '_'
        for character in case['variant_id']
    ).strip('_')
    relative = f'raw/identity/{order:03d}_{slug}.json'
    expected_sha = original_manifest.get(relative)
    path = _ORIGINAL_CUBE_DIR / relative
    if not _is_sha256(expected_sha) or _sha256(path) != expected_sha:
      raise AnalysisError(f'Original identity binding changed: {relative}.')
    record = _read_json(path, f'original identity {order}')
    sequence = _exact_keys(
        record.get('sequence_sha256'), {'reference', 'alternate'},
        f'original identity {order}.sequence_sha256',
    )
    if any(not _is_sha256(value) for value in sequence.values()):
      raise AnalysisError(f'Original identity {order} sequence hash changed.')
    result[order] = dict(sequence)
  return result


_FAILED_CURRENT_KEYS = {
    'schema_version', 'status', 'attempt_id', 'script_version',
    'external_freeze_authorization', 'execution_index', 'recipient_order',
    'recipient_variant_id', 'anchor_id', 'failed_or_next_call_role',
    'd_completed', 'started_count', 'completed_count',
    'started_event_bindings', 'completed_event_bindings',
    'partial_call_outputs', 'failure_phase', 'failure',
    'source_input_audit_content_binding',
    'same_object_attestation_content_binding',
    'confirmation_scope_disclosure', 'created_at_unix_s',
}


def _validate_treedef(node: Any, label: str, path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
  value = _exact_keys(node, {'kind', 'metadata', 'children'}, label)
  kind, metadata, children = value.get('kind'), value.get('metadata'), value.get('children')
  if kind not in {'dict', 'list', 'tuple', 'leaf'} or not isinstance(children, list):
    raise AnalysisError(f'{label} treedef node changed.')
  if kind == 'leaf':
    if metadata is not None or children:
      raise AnalysisError(f'{label} leaf treedef changed.')
    return [path]
  if kind == 'dict':
    if (
        not isinstance(metadata, list) or len(metadata) != len(children)
        or any(not isinstance(key, str) for key in metadata)
        or len(set(metadata)) != len(metadata)
    ):
      raise AnalysisError(f'{label} dict treedef changed.')
    result = []
    for key, child in zip(metadata, children, strict=True):
      result.extend(_validate_treedef(child, label, (*path, ('dict_key', key))))
    return result
  if isinstance(metadata, bool) or not isinstance(metadata, int) or metadata != len(children):
    raise AnalysisError(f'{label} sequence treedef changed.')
  result = []
  for index, child in enumerate(children):
    result.extend(_validate_treedef(child, label, (*path, ('sequence_index', index))))
  return result


def _validate_partial_output(value: Any, label: str) -> None:
  node = _exact_keys(value, {'status', 'treedef', 'leaf_count', 'leaves'}, label)
  if node.get('status') != 'returned' or not isinstance(node.get('leaves'), list):
    raise AnalysisError(f'{label} returned output changed.')
  paths = _validate_treedef(node.get('treedef'), f'{label}.treedef')
  if node.get('leaf_count') != len(paths) or len(node['leaves']) != len(paths):
    raise AnalysisError(f'{label} leaf count differs from treedef.')
  item_sizes = {'bfloat16': 2, 'float16': 2, 'float32': 4, 'float64': 8,
                'int32': 4, 'int64': 8, 'bool': 1}
  import base64  # stdlib, deliberately local to the rare failed-current path.
  for index, (leaf, expected_path) in enumerate(zip(node['leaves'], paths, strict=True)):
    row = _exact_keys(
        leaf,
        {
            'path', 'dtype_name', 'byte_order', 'shape', 'encoding',
            'data_base64', 'sha256', 'size_bytes',
        }, f'{label}.leaves[{index}]',
    )
    emitted_path = []
    if not isinstance(row.get('path'), list):
      raise AnalysisError(f'{label} leaf path changed.')
    for token in row['path']:
      if isinstance(token, Mapping) and set(token) == {'kind', 'key'} and token['kind'] == 'dict_key' and isinstance(token['key'], str):
        emitted_path.append(('dict_key', token['key']))
      elif isinstance(token, Mapping) and set(token) == {'kind', 'index'} and token['kind'] == 'sequence_index' and isinstance(token['index'], int) and not isinstance(token['index'], bool) and token['index'] >= 0:
        emitted_path.append(('sequence_index', token['index']))
      else:
        raise AnalysisError(f'{label} leaf path token changed.')
    dtype = row.get('dtype_name')
    shape = row.get('shape')
    if (
        tuple(emitted_path) != expected_path or dtype not in item_sizes
        or not isinstance(shape, list)
        or any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape)
        or row.get('encoding') != 'base64_c_order_raw_bytes'
        or row.get('byte_order') != ('not_applicable' if item_sizes[dtype] == 1 else 'little')
    ):
      raise AnalysisError(f'{label} leaf semantic schema changed.')
    encoded = row.get('data_base64')
    if not isinstance(encoded, str) or len(encoded) % 4:
      raise AnalysisError(f'{label} leaf base64 framing changed.')
    try:
      raw = base64.b64decode(encoded, validate=True)
    except Exception as error:  # binascii.Error is intentionally wrapped.
      raise AnalysisError(f'{label} leaf base64 is invalid.') from error
    if base64.b64encode(raw).decode('ascii') != encoded:
      raise AnalysisError(f'{label} leaf base64 is not canonical RFC 4648.')
    expected_size = math.prod(shape) * item_sizes[dtype]
    if (
        row.get('size_bytes') != expected_size or len(raw) != expected_size
        or row.get('sha256') != hashlib.sha256(raw).hexdigest()
    ):
      raise AnalysisError(f'{label} leaf bytes/hash/shape changed.')


def _validate_failed_current(
    value: Any, *, k: int, cases: Mapping[int, Mapping[str, Any]],
    source_binding: Mapping[str, Any], object_binding: Mapping[str, Any],
    authorization: Mapping[str, Any], started_map: Mapping[str, Any],
    completed_map: Mapping[str, Any],
) -> dict[str, Any]:
  node = _exact_keys(value, _FAILED_CURRENT_KEYS, 'failed_current')
  order, anchor = _execution_order()[k]
  d = node.get('d_completed')
  if isinstance(d, bool) or not isinstance(d, int) or d not in range(5):
    raise AnalysisError('failed_current d_completed changed.')
  expected = {
      'schema_version': 'v3.3.4.3-failed-current-v1',
      'status': 'failed_current', 'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION, 'execution_index': k,
      'external_freeze_authorization': authorization,
      'recipient_order': order, 'recipient_variant_id': cases[order]['variant_id'],
      'anchor_id': anchor, 'source_input_audit_content_binding': source_binding,
      'same_object_attestation_content_binding': object_binding,
      'confirmation_scope_disclosure': CONFIRMATION_DISCLOSURE,
  }
  for key, item in expected.items():
    if node.get(key) != item:
      raise AnalysisError(f'failed_current.{key} changed.')
  phase = node.get('failure_phase')
  if phase not in {'record_setup', 'model_dispatch', 'record_validation', 'record_serialization'}:
    raise AnalysisError('failed_current failure phase changed.')
  expected_completed = 4 * k + d
  expected_started = expected_completed if d == 4 or phase == 'record_setup' else expected_completed + 1
  if node.get('started_count') != expected_started or node.get('completed_count') != expected_completed:
    raise AnalysisError('failed_current call arithmetic changed.')
  role = node.get('failed_or_next_call_role')
  if phase == 'record_setup':
    if d != 0 or role != 'intended':
      raise AnalysisError('failed_current setup boundary changed.')
  elif d == 4:
    if role is not None or phase not in {'record_validation', 'record_serialization'}:
      raise AnalysisError('failed_current post-four boundary changed.')
  elif role != _CALL_ROLES[d] or phase != 'model_dispatch':
    raise AnalysisError('failed_current dispatch boundary changed.')
  _validate_failure(node.get('failure'), 'failed_current.failure')
  _finite(node.get('created_at_unix_s'), 'failed_current.created_at_unix_s')
  outputs = _exact_keys(node.get('partial_call_outputs'), set(_CALL_ROLES), 'failed_current.outputs')
  for index, call in enumerate(_CALL_ROLES):
    if index < d:
      _validate_partial_output(outputs[call], f'failed_current.outputs.{call}')
    elif outputs[call] is not None:
      raise AnalysisError('failed_current output prefix changed.')
  for name, expected_count, event_kind, event_map in (
      ('started_event_bindings', expected_started - 4 * k, 'started', started_map),
      ('completed_event_bindings', d, 'completed', completed_map),
  ):
    bindings = node.get(name)
    if not isinstance(bindings, list) or len(bindings) != expected_count:
      raise AnalysisError(f'failed_current.{name} count changed.')
    for index, binding in enumerate(bindings):
      row = _validate_file_binding(binding, f'failed_current.{name}', with_path=True)
      relative = f'dispatch_journal/{event_kind}/{4 * k + index:03d}.json'
      if row != {'path': relative, **dict(event_map.get(relative, {}))}:
        raise AnalysisError(f'failed_current.{name} binding prefix changed.')
  return {'k': k, 'd': d, 'failure_phase': phase}














def _validate_signature_attestation(
    run_dir: Path, binding: Any, freeze: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
  row = _validate_file_binding(binding, 'signature attestation binding', with_path=True)
  if row['path'] != 'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json':
    raise AnalysisError('Signature attestation path changed.')
  path = run_dir / row['path']
  _strict_regular(path, 'signature attestation')
  if path.stat().st_size != row['size_bytes'] or _sha256(path) != row['sha256']:
    raise AnalysisError('Signature attestation current bytes changed.')
  value = _read_json(path, 'PROGRAM_SIGNATURE_ATTESTATION')
  keys = {
      'schema_version', 'script_version', 'attempt_id',
      'external_freeze_authorization', 'object_order',
      'runtime_container_tags', 'frozen_container_tags',
      'runtime_semantic_mapping', 'frozen_semantic_mapping',
      'runtime_canonical', 'frozen_canonical', 'comparisons',
      'created_at_unix_s',
  }
  _exact_keys(value, keys, 'PROGRAM_SIGNATURE_ATTESTATION')
  expected = {
      'schema_version': 'v3.3.4.3-program-signature-attestation-v1',
      'script_version': SCRIPT_VERSION, 'attempt_id': ATTEMPT_ID,
      'external_freeze_authorization': authorization,
      'object_order': ['eight_interventions', 'selection', 'target'],
      'frozen_semantic_mapping': freeze['program_signatures'],
  }
  for key, item in expected.items():
    if value.get(key) != item:
      raise AnalysisError(f'Signature attestation changed at {key}.')
  expected_paths = (
      ['/eight_interventions/leaves']
      + [f'/eight_interventions/leaves/{index}/shape' for index in range(17)]
      + ['/selection/leaves']
      + [f'/selection/leaves/{index}/shape' for index in range(9)]
      + ['/target/leaves']
      + [f'/target/leaves/{index}/shape' for index in range(3)]
  )
  for field, kind in (
      ('runtime_container_tags', 'tuple'), ('frozen_container_tags', 'list')
  ):
    tags = value.get(field)
    if (
        not isinstance(tags, list) or len(tags) != 32
        or tags != [{'path': path, 'kind': kind} for path in expected_paths]
    ):
      raise AnalysisError(f'Signature attestation {field} changed.')
  canonical = {'sha256': PROGRAM_SIGNATURES_SHA256, 'size_bytes': 2877}
  if (
      value.get('runtime_semantic_mapping') != freeze['program_signatures']
      or value.get('runtime_canonical') != canonical
      or value.get('frozen_canonical') != canonical
      or _content_binding(value['runtime_semantic_mapping'])['sha256']
      != PROGRAM_SIGNATURES_SHA256
      or _content_binding(value['runtime_semantic_mapping'])['size_bytes'] != 2877
  ):
    raise AnalysisError('Signature semantic/canonical mapping changed.')
  comparisons = _exact_keys(
      value.get('comparisons'),
      {
          'direct_python_equality', 'runtime_tuple_container_count',
          'runtime_leaves_tuple_count', 'runtime_shape_tuple_count',
          'frozen_list_container_count', 'frozen_leaves_list_count',
          'frozen_shape_list_count', 'declared_paths_exact',
          'container_kinds_exact', 'treedefs_exact',
          'leaf_order_counts_dtypes_shapes_exact', 'canonical_bytes_exact',
          'canonical_hash_and_size_exact',
      }, 'signature comparisons',
  )
  expected_comparisons = {
      'direct_python_equality': False, 'runtime_tuple_container_count': 32,
      'runtime_leaves_tuple_count': 3, 'runtime_shape_tuple_count': 29,
      'frozen_list_container_count': 32, 'frozen_leaves_list_count': 3,
      'frozen_shape_list_count': 29, 'declared_paths_exact': True,
      'container_kinds_exact': True, 'treedefs_exact': True,
      'leaf_order_counts_dtypes_shapes_exact': True,
      'canonical_bytes_exact': True, 'canonical_hash_and_size_exact': True,
  }
  if comparisons != expected_comparisons:
    raise AnalysisError('Signature comparison matrix changed.')
  _finite(value.get('created_at_unix_s'), 'signature attestation timestamp')
  return {'binding': row, 'canonical_sha256': PROGRAM_SIGNATURES_SHA256}


def _validate_signature_failure(
    run_dir: Path, binding: Any, authorization: Mapping[str, Any],
) -> dict[str, Any]:
  row = _validate_file_binding(
      binding, 'signature failure binding', with_path=True
  )
  expected = (
      'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION_FAILURE.json'
  )
  if row['path'] != expected:
    raise AnalysisError('Signature-failure artifact path changed.')
  path = run_dir / expected
  _strict_regular(path, 'signature failure artifact')
  if path.stat().st_size != row['size_bytes'] or _sha256(path) != row['sha256']:
    raise AnalysisError('Signature-failure artifact bytes changed.')
  value = _read_json(path, 'signature failure artifact')
  _exact_keys(value, {
      'schema_version', 'script_version', 'attempt_id',
      'external_freeze_authorization', 'status', 'partial_runtime_tags',
      'partial_frozen_tags', 'failure', 'created_at_unix_s',
  }, 'signature failure artifact')
  if (
      value.get('schema_version')
      != 'v3.3.4.3-program-signature-attestation-v1'
      or value.get('script_version') != SCRIPT_VERSION
      or value.get('attempt_id') != ATTEMPT_ID
      or value.get('external_freeze_authorization') != authorization
      or value.get('status') != 'failure'
  ):
    raise AnalysisError('Signature-failure identity changed.')
  _validate_failure(value.get('failure'), 'signature failure.failure')
  _finite(value.get('created_at_unix_s'), 'signature failure timestamp')
  declared = [
      '/eight_interventions/leaves',
      *[f'/eight_interventions/leaves/{index}/shape' for index in range(17)],
      '/selection/leaves',
      *[f'/selection/leaves/{index}/shape' for index in range(9)],
      '/target/leaves',
      *[f'/target/leaves/{index}/shape' for index in range(3)],
  ]
  runtime = value.get('partial_runtime_tags')
  frozen = value.get('partial_frozen_tags')
  if not isinstance(runtime, list) or not isinstance(frozen, list):
    raise AnalysisError('Signature failure partial tag arrays changed.')
  for tags, kind, label in (
      (runtime, 'tuple', 'runtime'), (frozen, 'list', 'frozen'),
  ):
    if len(tags) > 32:
      raise AnalysisError(f'Signature failure {label} prefix is too long.')
    for index, raw in enumerate(tags):
      if dict(_exact_keys(raw, {'path', 'kind'}, f'{label} partial tag')) != {
          'path': declared[index], 'kind': kind,
      }:
        raise AnalysisError(f'Signature failure {label} tag prefix changed.')
  return {'binding': row, 'state': 'failure'}


def _validate_compiler_failure_record(
    compiler: Mapping[str, Any], *, run_dir: Path, freeze: Mapping[str, Any],
    start: Mapping[str, Any], completion: Mapping[str, Any],
) -> dict[str, Any]:
  _exact_keys(compiler, {
      'status', 'failure_stage', 'compile_count', 'lower_attempt_count',
      'compile_attempt_count', 'successful_compile_count',
      'lower_or_compile_pipeline_attempt_count', 'compile_seconds',
      'artifacts', 'program_signatures', 'program_signatures_sha256',
      'program_signature_attestation', 'external_freeze_authorization',
      'source_input_audit', 'source_input_audit_content_binding',
      'same_object_attestation', 'same_object_attestation_content_binding',
      'source_program_gate', 'compiled_backend_diagnostic_only', 'failure',
      'no_compile_retry', 'model_apply_count', 'attempt_budget_audit',
      'diagnostic_provenance_complete', 'kernel_cache_provenance',
  }, 'compiler failure record')
  stage = compiler.get('failure_stage')
  if stage not in {'lower', 'compile'}:
    raise AnalysisError('Compiler failure stage changed.')
  compile_count = int(stage == 'compile')
  if (
      compiler.get('status') != 'compiler_failure'
      or compiler.get('compile_count') != compile_count
      or compiler.get('lower_attempt_count') != 1
      or compiler.get('compile_attempt_count') != compile_count
      or compiler.get('successful_compile_count') != 0
      or compiler.get('lower_or_compile_pipeline_attempt_count') != 1
      or _finite(compiler.get('compile_seconds'), 'compiler failure seconds') < 0
      or compiler.get('program_signatures') != freeze['program_signatures']
      or compiler.get('program_signatures_sha256') != PROGRAM_SIGNATURES_SHA256
      or compiler.get('external_freeze_authorization')
      != start['external_freeze_authorization']
      or compiler.get('source_program_gate') is not None
      or compiler.get('compiled_backend_diagnostic_only') is not True
      or compiler.get('no_compile_retry') is not True
      or compiler.get('model_apply_count') != 0
      or compiler.get('diagnostic_provenance_complete') is not None
  ):
    raise AnalysisError('Compiler failure scalar contract changed.')
  _validate_failure(compiler.get('failure'), 'compiler failure.failure')
  _validate_signature_attestation(
      run_dir, compiler.get('program_signature_attestation'), freeze,
      start['external_freeze_authorization'],
  )
  audit = _validate_content_bound_object(
      compiler.get('source_input_audit'),
      compiler.get('source_input_audit_content_binding'),
      'compiler failure source audit', keys=_SOURCE_AUDIT_KEYS,
  )
  if any(value is not True for value in audit.values()):
    raise AnalysisError('Compiler failure source audit is not final/all true.')
  same = _validate_content_bound_object(
      compiler.get('same_object_attestation'),
      compiler.get('same_object_attestation_content_binding'),
      'compiler failure same-object', keys=_SAME_OBJECT_KEYS,
  )
  expected_same = {
      'lower_call_count': 1, 'compile_call_count': compile_count,
      'stablehlo_read_from_lowered_object': None if stage == 'lower' else True,
      'pre_backend_hlo_read_from_lowered_object': None if stage == 'lower' else True,
      'compile_argument_is_lowered_object': None if stage == 'lower' else True,
      'compiled_hlo_read_from_compiled_object': None,
      'signature_attestation_from_apply_arguments': True,
      'apply_callable_is_compiled_object': None,
      'compiler_record_is_gate_record': True,
      'lowered_python_id': None if stage == 'lower' else same['lowered_python_id'],
      'compiled_python_id': None,
  }
  if same != expected_same or (
      stage == 'compile' and (
          isinstance(same['lowered_python_id'], bool)
          or not isinstance(same['lowered_python_id'], int)
          or same['lowered_python_id'] < 0
      )
  ):
    raise AnalysisError('Compiler failure same-object phase changed.')
  artifact_names = set() if stage == 'lower' else {'stablehlo', 'hlo'}
  artifacts = _exact_keys(compiler.get('artifacts'), artifact_names, 'compiler failure artifacts')
  paths = {
      'stablehlo': 'compiler/eight_row/graph.stablehlo.mlir',
      'hlo': 'compiler/eight_row/graph.pre_backend.hlo.txt',
  }
  for name in artifact_names:
    row = _validate_file_binding(artifacts[name], f'compiler failure {name}', with_path=True)
    if row['path'] != paths[name]:
      raise AnalysisError('Compiler failure graph path changed.')
    live = run_dir / row['path']
    _strict_regular(live, f'compiler failure {name}')
    if live.stat().st_size != row['size_bytes'] or _sha256(live) != row['sha256']:
      raise AnalysisError('Compiler failure graph bytes changed.')
  budget = _exact_keys(compiler.get('attempt_budget_audit'), {
      'lower_budget', 'compile_budget', 'lower_invocations',
      'compile_invocations', 'forbidden_request',
      'forbidden_request_detected_before_invocation',
  }, 'compiler failure attempt budget')
  if budget != {
      'lower_budget': 1, 'compile_budget': 1, 'lower_invocations': 1,
      'compile_invocations': compile_count, 'forbidden_request': None,
      'forbidden_request_detected_before_invocation': False,
  }:
    raise AnalysisError('Compiler failure attempt budget changed.')
  if completion.get('same_object_attestation') != same:
    raise AnalysisError('Terminal/compiler failure same-object evidence differs.')
  return {
      'state': 'lower_failed' if stage == 'lower' else 'compile_failed',
      'record': dict(compiler), 'source_program_exact': None,
      'audit': {
          'signature_attestation_state': 'validated',
          'source_input_audit_exact': True,
          'same_object_attestation_exact': True,
          'stablehlo_exact': None if stage == 'lower' else (
              artifacts['stablehlo']['sha256'] == SOURCE_STABLEHLO['sha256']
          ),
          'pre_backend_exact': None if stage == 'lower' else (
              artifacts['hlo']['sha256'] == SOURCE_PRE_BACKEND_HLO['sha256']
          ),
          'entry_abi_exact': None, 'source_program_exact': None,
          'compiled_backend_diagnostic_only': True,
          'diagnostic_provenance_complete': None,
      },
  }


def _validate_diagnostic_failure_record(
    compiler: Mapping[str, Any], *, run_dir: Path, freeze: Mapping[str, Any],
    start: Mapping[str, Any], completion: Mapping[str, Any],
) -> dict[str, Any]:
  """Validates the compiled-but-diagnostics-incomplete runner artifact."""
  _exact_keys(compiler, {
      'status', 'executable_name', 'lower_attempt_count',
      'compile_attempt_count', 'successful_compile_count', 'artifacts',
      'program_signature_attestation_binding',
      'external_freeze_authorization', 'source_input_audit',
      'source_input_audit_content_binding', 'same_object_attestation',
      'same_object_attestation_content_binding',
      'source_program_gate_without_backend_diagnostics', 'failure',
      'attempt_budget_audit', 'diagnostic_provenance_complete',
      'compiled_backend_diagnostic_only', 'no_dispatch',
      'created_at_unix_s',
  }, 'diagnostic failure record')
  reason = completion.get('stop_reason')
  if (
      compiler.get('status') != 'diagnostic_provenance_failure'
      or completion.get('status')
      != 'controlled_stop_diagnostic_provenance_failure'
      or reason not in {
          'diagnostic_parser_failure', 'diagnostic_persistence_failure',
          'cache_signal_unavailable', 'fingerprint_formula_mismatch',
      }
      or compiler.get('executable_name') != 'eight_row'
      or compiler.get('lower_attempt_count') != 1
      or compiler.get('compile_attempt_count') != 1
      or compiler.get('successful_compile_count') != 1
      or compiler.get('external_freeze_authorization')
      != start['external_freeze_authorization']
      or compiler.get('diagnostic_provenance_complete') is not False
      or compiler.get('compiled_backend_diagnostic_only') is not True
      or compiler.get('no_dispatch') is not True
  ):
    raise AnalysisError('Diagnostic failure scalar contract changed.')
  _validate_failure(compiler.get('failure'), 'diagnostic failure.failure')
  _finite(
      compiler.get('created_at_unix_s'),
      'diagnostic failure.created_at_unix_s',
  )
  signature = _validate_signature_attestation(
      run_dir, compiler.get('program_signature_attestation_binding'), freeze,
      start['external_freeze_authorization'],
  )
  if (
      completion.get('program_signature_attestation_binding')
      != compiler.get('program_signature_attestation_binding')
  ):
    raise AnalysisError('Diagnostic failure signature binding changed.')
  source_audit = _validate_content_bound_object(
      compiler.get('source_input_audit'),
      compiler.get('source_input_audit_content_binding'),
      'diagnostic failure source audit', keys=_SOURCE_AUDIT_KEYS,
  )
  if any(value is not True for value in source_audit.values()):
    raise AnalysisError('Diagnostic failure source audit is not all true.')
  same_object = _validate_same_object_success(
      compiler.get('same_object_attestation'),
      compiler.get('same_object_attestation_content_binding'),
      'diagnostic failure same-object attestation',
  )
  if (
      completion.get('same_object_attestation') != same_object
      or completion.get('same_object_attestation_content_binding')
      != _content_binding(same_object)
  ):
    raise AnalysisError('Diagnostic failure terminal object binding changed.')
  artifacts = _exact_keys(
      compiler.get('artifacts'), {'stablehlo', 'hlo', 'compiled_hlo'},
      'diagnostic failure artifacts',
  )
  expected_paths = {
      'stablehlo': 'compiler/eight_row/graph.stablehlo.mlir',
      'hlo': 'compiler/eight_row/graph.pre_backend.hlo.txt',
      'compiled_hlo': 'compiler/eight_row/graph.compiled.hlo.txt',
  }
  checked_artifacts = {}
  for name, relative in expected_paths.items():
    row = _validate_file_binding(
        artifacts[name], f'diagnostic failure {name}', with_path=True
    )
    live = run_dir / relative
    _strict_regular(live, f'diagnostic failure {name}')
    if (
        row['path'] != relative or live.stat().st_size != row['size_bytes']
        or _sha256(live) != row['sha256']
    ):
      raise AnalysisError(f'Diagnostic failure {name} bytes changed.')
    checked_artifacts[name] = row
  compiler_binding = _validate_file_binding(
      completion.get('compiler_binding'),
      'diagnostic failure terminal compiler binding', with_path=True,
  )
  signature_binding = _validate_file_binding(
      compiler.get('program_signature_attestation_binding'),
      'diagnostic failure signature binding', with_path=True,
  )
  if (
      compiler_binding['path']
      != 'compiler/eight_row/COMPILER_DIAGNOSTIC_FAILURE.json'
      or signature_binding['path']
      != 'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json'
  ):
    raise AnalysisError('Diagnostic failure compiler path grammar changed.')
  expected_compiler_bindings = {
      signature_binding['path']: {
          'sha256': signature_binding['sha256'],
          'size_bytes': signature_binding['size_bytes'],
      },
      compiler_binding['path']: {
          'sha256': compiler_binding['sha256'],
          'size_bytes': compiler_binding['size_bytes'],
      },
      **{
          row['path']: {
              'sha256': row['sha256'], 'size_bytes': row['size_bytes'],
          }
          for row in checked_artifacts.values()
      },
  }
  actual_compiler_bindings = completion.get('compiler_artifact_bindings')
  if actual_compiler_bindings != dict(sorted(expected_compiler_bindings.items())):
    raise AnalysisError('Diagnostic failure compiler membership changed.')
  compiled_text = (run_dir / expected_paths['compiled_hlo']).read_text(
      encoding='utf-8'
  )
  gate = _exact_keys(
      compiler.get('source_program_gate_without_backend_diagnostics'), {
          'contract', 'observed', 'stablehlo_exact',
          'pre_backend_hlo_exact', 'program_signature_structure_exact',
          'program_signatures_canonical_exact', 'entry_abi_exact',
          'source_runtime_device_toolchain_checkpoint_reference_exact',
          'source_input_audit', 'source_input_audit_content_binding',
          'same_object_attestation', 'same_object_attestation_content_binding',
          'same_lowered_compiled_object', 'source_program_exact',
      }, 'diagnostic failure source gate',
  )
  observed = _exact_keys(gate.get('observed'), {
      'stablehlo_sha256', 'stablehlo_size_bytes',
      'pre_backend_hlo_sha256', 'pre_backend_hlo_size_bytes',
      'program_signatures_sha256', 'entry_abi_sha256',
  }, 'diagnostic failure source gate observed')
  stable_exact = (
      checked_artifacts['stablehlo']['sha256'] == SOURCE_STABLEHLO['sha256']
      and checked_artifacts['stablehlo']['size_bytes']
      == SOURCE_STABLEHLO['size_bytes']
  )
  pre_backend_exact = (
      checked_artifacts['hlo']['sha256'] == SOURCE_PRE_BACKEND_HLO['sha256']
      and checked_artifacts['hlo']['size_bytes']
      == SOURCE_PRE_BACKEND_HLO['size_bytes']
  )
  signatures_exact = (
      _canonical_json_sha256(freeze['program_signatures'])
      == PROGRAM_SIGNATURES_SHA256
  )
  observed_entry_sha = observed.get('entry_abi_sha256')
  entry_exact = _diagnostic_entry_abi_exact(
      compiled_text, reason=str(reason), failure=compiler['failure'],
      observed_sha256=observed_entry_sha,
  )
  trigger_operation = _validate_triggering_diagnostic_operation(
      compiler['failure'], str(reason), compiled_text,
  )
  expected_observed = {
      'stablehlo_sha256': checked_artifacts['stablehlo']['sha256'],
      'stablehlo_size_bytes': checked_artifacts['stablehlo']['size_bytes'],
      'pre_backend_hlo_sha256': checked_artifacts['hlo']['sha256'],
      'pre_backend_hlo_size_bytes': checked_artifacts['hlo']['size_bytes'],
      'program_signatures_sha256': PROGRAM_SIGNATURES_SHA256,
      'entry_abi_sha256': observed_entry_sha,
  }
  expected_primitives = {
      'stablehlo_exact': stable_exact,
      'pre_backend_hlo_exact': pre_backend_exact,
      'program_signature_structure_exact': signatures_exact,
      'program_signatures_canonical_exact': signatures_exact,
      'entry_abi_exact': entry_exact,
      'source_runtime_device_toolchain_checkpoint_reference_exact': True,
      'same_lowered_compiled_object': True,
  }
  if (
      gate.get('contract') != freeze['source_program_contract']
      or dict(observed) != expected_observed
      or gate.get('source_input_audit') != source_audit
      or gate.get('source_input_audit_content_binding')
      != _content_binding(source_audit)
      or gate.get('same_object_attestation') != same_object
      or gate.get('same_object_attestation_content_binding')
      != _content_binding(same_object)
      or any(
          gate.get(name) is not expected
          for name, expected in expected_primitives.items()
      )
  ):
    raise AnalysisError('Diagnostic failure source-program evidence changed.')
  source_exact = all(expected_primitives.values())
  if (
      gate.get('source_program_exact') is not source_exact
      or completion.get('source_program_gate') != gate
      or completion.get('diagnostic_provenance_complete') is not False
      or completion.get('compiled_backend_diagnostic_only') is not True
      or completion.get('backend_diagnostics') is not None
      or completion.get('diagnostic_comparisons') is not None
  ):
    raise AnalysisError('Diagnostic failure terminal boundary changed.')
  budget = _exact_keys(compiler.get('attempt_budget_audit'), {
      'lower_budget', 'compile_budget', 'lower_invocations',
      'compile_invocations', 'forbidden_request',
      'forbidden_request_detected_before_invocation',
  }, 'diagnostic failure attempt budget')
  if budget != {
      'lower_budget': 1, 'compile_budget': 1, 'lower_invocations': 1,
      'compile_invocations': 1, 'forbidden_request': None,
      'forbidden_request_detected_before_invocation': False,
  }:
    raise AnalysisError('Diagnostic failure attempt budget changed.')
  return {
      'state': 'diagnostic_provenance_failed', 'record': dict(compiler),
      'source_program_exact': source_exact,
      'executable_fingerprint': None,
      'audit': {
          'signature_attestation_state': 'validated',
          'canonical_sha256': signature['canonical_sha256'],
          'source_input_audit_exact': True,
          'same_object_attestation_exact': True,
          'stablehlo_exact': stable_exact,
          'pre_backend_exact': pre_backend_exact,
          'entry_abi_exact': entry_exact,
          'source_program_exact': source_exact,
          'trigger_operation': trigger_operation,
          'compiled_backend_diagnostic_only': True,
          'diagnostic_provenance_complete': False,
      },
  }


def _validate_compiler_v3343(
    run_dir: Path, completion: Mapping[str, Any], freeze: Mapping[str, Any],
    start: Mapping[str, Any],
) -> dict[str, Any]:
  binding = completion.get('compiler_binding')
  if binding is None:
    if completion.get('compiler_artifact_bindings') != {}:
      raise AnalysisError('Null compiler has nonempty artifact bindings.')
    if completion.get('status') == 'controlled_stop_signature_attestation_failure':
      _validate_signature_failure(
          run_dir, completion.get('program_signature_attestation_binding'),
          start['external_freeze_authorization'],
      )
      return {
          'state': 'signature_attestation_failed',
          'source_program_exact': None, 'record': None,
          'audit': {
              'signature_attestation_state': 'failure',
              'source_program_exact': None,
              'diagnostic_provenance_complete': None,
          },
      }
    if completion.get('program_signature_attestation_binding') is not None:
      raise AnalysisError('Precompiler terminal has an unexpected signature artifact.')
    return {
        'state': 'not_reached', 'source_program_exact': None,
        'record': None, 'audit': {
            'signature_attestation_state': 'not_reached',
            'source_program_exact': None,
            'diagnostic_provenance_complete': None,
        },
    }
  binding = _validate_file_binding(binding, 'RUN_COMPLETE.compiler_binding', with_path=True)
  path = run_dir / binding['path']
  _strict_regular(path, 'compiler record')
  if path.stat().st_size != binding['size_bytes'] or _sha256(path) != binding['sha256']:
    raise AnalysisError('Compiler record binding changed.')
  compiler = _read_json(path, 'compiler record')
  if compiler.get('status') == 'diagnostic_provenance_failure':
    return _validate_diagnostic_failure_record(
        compiler, run_dir=run_dir, freeze=freeze, start=start,
        completion=completion,
    )
  success_keys = {
      'executable_name', 'compile_count', 'lower_attempt_count',
      'compile_attempt_count', 'successful_compile_count', 'compile_seconds',
      'executable_fingerprint', 'artifacts', 'program_signatures',
      'program_signatures_sha256', 'entry_abi', 'source_program_gate',
      'backend_diagnostics', 'diagnostic_comparisons',
      'kernel_cache_provenance', 'program_signature_attestation',
      'external_freeze_authorization', 'source_input_audit',
      'source_input_audit_content_binding', 'same_object_attestation',
      'same_object_attestation_content_binding', 'attempt_budget_audit',
      'diagnostic_provenance_complete',
  }
  if set(compiler) != success_keys:
    if completion.get('eight_row_successful_compile_count') == 0:
      return _validate_compiler_failure_record(
          compiler, run_dir=run_dir, freeze=freeze, start=start,
          completion=completion,
      )
    raise AnalysisError('Successful compiler record schema changed.')
  if compiler.get('external_freeze_authorization') != start['external_freeze_authorization']:
    raise AnalysisError('Compiler authorization changed.')
  _validate_source_audit(
      compiler.get('source_input_audit'),
      compiler.get('source_input_audit_content_binding'),
      (True,) * 8, 'compiler.source_input_audit',
  )
  signature = _validate_signature_attestation(
      run_dir, compiler.get('program_signature_attestation'), freeze,
      start['external_freeze_authorization'],
  )
  artifacts = _exact_keys(compiler.get('artifacts'), {'stablehlo', 'hlo', 'compiled_hlo'}, 'compiler.artifacts')
  artifact_bindings = completion.get('compiler_artifact_bindings')
  if not isinstance(artifact_bindings, Mapping):
    raise AnalysisError('RUN_COMPLETE compiler artifact map is absent.')
  expected_paths = {
      'stablehlo': 'compiler/eight_row/graph.stablehlo.mlir',
      'hlo': 'compiler/eight_row/graph.pre_backend.hlo.txt',
      'compiled_hlo': 'compiler/eight_row/graph.compiled.hlo.txt',
  }
  for name, relative in expected_paths.items():
    row = _exact_keys(artifacts[name], {'path', 'sha256', 'size_bytes'}, f'compiler.{name}')
    if row.get('path') != relative:
      raise AnalysisError(f'Compiler {name} path changed.')
    live = run_dir / relative
    _strict_regular(live, f'compiler {name}')
    if live.stat().st_size != row['size_bytes'] or _sha256(live) != row['sha256']:
      raise AnalysisError(f'Compiler {name} bytes changed.')
  stable_exact = (
      artifacts['stablehlo']['sha256'] == SOURCE_STABLEHLO['sha256']
      and artifacts['stablehlo']['size_bytes'] == SOURCE_STABLEHLO['size_bytes']
  )
  pre_backend_exact = (
      artifacts['hlo']['sha256'] == SOURCE_PRE_BACKEND_HLO['sha256']
      and artifacts['hlo']['size_bytes'] == SOURCE_PRE_BACKEND_HLO['size_bytes']
  )
  signatures_exact = (
      compiler.get('program_signatures') == freeze['program_signatures']
      and compiler.get('program_signatures_sha256')
      == _canonical_json_sha256(compiler.get('program_signatures'))
      == PROGRAM_SIGNATURES_SHA256
  )
  if (
      compiler.get('executable_name') != 'eight_row'
      or compiler.get('compile_count') != 1
      or compiler.get('lower_attempt_count') != 1
      or compiler.get('compile_attempt_count') != 1
      or compiler.get('successful_compile_count') != 1
      or _finite(compiler.get('compile_seconds'), 'compiler.compile_seconds') < 0
      or compiler.get('external_freeze_authorization')
      != start['external_freeze_authorization']
  ):
    raise AnalysisError('Successful compiler counters/identity changed.')
  source_audit = _validate_content_bound_object(
      compiler.get('source_input_audit'),
      compiler.get('source_input_audit_content_binding'),
      'compiler.source_input_audit', keys=_SOURCE_AUDIT_KEYS,
  )
  if any(value is not True for value in source_audit.values()):
    raise AnalysisError('Successful compiler source-input audit is not all true.')
  same_object = _validate_same_object_success(
      compiler.get('same_object_attestation'),
      compiler.get('same_object_attestation_content_binding'),
      'compiler.same_object_attestation',
  )
  compiled_text = (run_dir / expected_paths['compiled_hlo']).read_text(encoding='utf-8')
  normalized, entry_sha = _normalized_entry_abi(compiled_text)
  entry = compiler.get('entry_abi')
  if not isinstance(entry, Mapping) or entry.get('normalized_line_sha256') != entry_sha:
    raise AnalysisError('Compiler entry ABI changed.')
  entry_exact = entry_sha == ENTRY_ABI_SHA256
  expected_fingerprint = hashlib.sha256(bytes.fromhex(artifacts['compiled_hlo']['sha256'])).hexdigest()
  if compiler.get('executable_fingerprint') != expected_fingerprint:
    raise AnalysisError('Executable fingerprint formula changed.')
  diagnostics = _recompute_backend_diagnostics(compiled_text)
  if compiler.get('backend_diagnostics') != diagnostics:
    raise AnalysisError('Compiled backend diagnostics changed from raw HLO.')
  gate = compiler.get('source_program_gate')
  gate = _exact_keys(gate, {
      'contract', 'observed', 'stablehlo_exact', 'pre_backend_hlo_exact',
      'program_signature_structure_exact',
      'program_signatures_canonical_exact', 'entry_abi_exact',
      'source_runtime_device_toolchain_checkpoint_reference_exact',
      'source_input_audit', 'source_input_audit_content_binding',
      'same_object_attestation', 'same_object_attestation_content_binding',
      'same_lowered_compiled_object', 'source_program_exact',
  }, 'compiler.source_program_gate')
  observed = _exact_keys(gate.get('observed'), {
      'stablehlo_sha256', 'stablehlo_size_bytes',
      'pre_backend_hlo_sha256', 'pre_backend_hlo_size_bytes',
      'program_signatures_sha256', 'entry_abi_sha256',
  }, 'compiler.source_program_gate.observed')
  expected_observed = {
      'stablehlo_sha256': artifacts['stablehlo']['sha256'],
      'stablehlo_size_bytes': artifacts['stablehlo']['size_bytes'],
      'pre_backend_hlo_sha256': artifacts['hlo']['sha256'],
      'pre_backend_hlo_size_bytes': artifacts['hlo']['size_bytes'],
      'program_signatures_sha256': compiler.get('program_signatures_sha256'),
      'entry_abi_sha256': entry_sha,
  }
  if (
      gate.get('contract') != freeze['source_program_contract']
      or dict(observed) != expected_observed
      or gate.get('source_input_audit') != source_audit
      or gate.get('source_input_audit_content_binding')
      != _content_binding(source_audit)
      or gate.get('same_object_attestation') != same_object
      or gate.get('same_object_attestation_content_binding')
      != _content_binding(same_object)
  ):
    raise AnalysisError('Compiler source-program primitive evidence changed.')
  primitive_flags = (
      'stablehlo_exact', 'pre_backend_hlo_exact',
      'program_signature_structure_exact',
      'program_signatures_canonical_exact', 'entry_abi_exact',
      'source_runtime_device_toolchain_checkpoint_reference_exact',
      'same_lowered_compiled_object',
  )
  if any(not isinstance(gate.get(name), bool) for name in primitive_flags):
    raise AnalysisError('Compiler source-program primitive is not boolean.')
  expected_primitives = {
      'stablehlo_exact': stable_exact,
      'pre_backend_hlo_exact': pre_backend_exact,
      'program_signature_structure_exact': signatures_exact,
      'program_signatures_canonical_exact': signatures_exact,
      'entry_abi_exact': entry_exact,
      'source_runtime_device_toolchain_checkpoint_reference_exact': all(
          value is True for value in source_audit.values()
      ),
      'same_lowered_compiled_object': True,
  }
  if any(gate[name] is not expected for name, expected in expected_primitives.items()):
    raise AnalysisError('Compiler source-program primitive disagrees with raw evidence.')
  source_exact = all(gate[name] for name in (
      'stablehlo_exact', 'pre_backend_hlo_exact',
      'program_signatures_canonical_exact', 'entry_abi_exact',
      'source_runtime_device_toolchain_checkpoint_reference_exact',
      'same_lowered_compiled_object',
  ))
  if gate.get('source_program_exact') is not source_exact:
    raise AnalysisError('Compiler source-program aggregate changed.')
  if completion.get('source_program_gate') != gate:
    raise AnalysisError('RUN_COMPLETE source-program gate differs from compiler.')
  budget = _exact_keys(compiler.get('attempt_budget_audit'), {
      'lower_budget', 'compile_budget', 'lower_invocations',
      'compile_invocations', 'forbidden_request',
      'forbidden_request_detected_before_invocation',
  }, 'compiler.attempt_budget_audit')
  if budget != {
      'lower_budget': 1, 'compile_budget': 1, 'lower_invocations': 1,
      'compile_invocations': 1, 'forbidden_request': None,
      'forbidden_request_detected_before_invocation': False,
  }:
    raise AnalysisError('Successful compiler attempt budget changed.')
  if (
      compiler.get('diagnostic_provenance_complete') is not True
      or completion.get('diagnostic_provenance_complete') is not True
      or completion.get('compiled_backend_diagnostic_only') is not True
  ):
    raise AnalysisError('Successful compiler diagnostic boundary changed.')
  del normalized
  return {
      'state': 'compiled', 'source_program_exact': source_exact,
      'executable_fingerprint': expected_fingerprint,
      'signature': signature, 'diagnostics': diagnostics, 'record': compiler,
      'audit': {
          'signature_attestation_state': 'validated',
          'type_tag_paths_exact': True, 'semantic_mapping_exact': True,
          'canonical_sha256': signature['canonical_sha256'],
          'canonical_size_bytes': 2877,
          'source_input_audit_exact': True,
          'same_object_attestation_exact': True,
          'stablehlo_exact': stable_exact,
          'pre_backend_exact': pre_backend_exact,
          'entry_abi_exact': entry_exact,
          'source_program_exact': source_exact,
          'compiled_backend_diagnostic_only': True,
          'diagnostic_provenance_complete': True,
      },
  }


def _validate_nonpublication_same_object(
    value: Any, binding: Any, stage: str,
) -> dict[str, Any]:
  node = _validate_content_bound_object(
      value, binding, 'nonpublication same-object attestation',
      keys=_SAME_OBJECT_KEYS,
  )
  expected_reads = {
      'stablehlo_text_extraction': (None, None, None),
      'pre_backend_hlo_text_extraction': (True, None, None),
      'compiled_hlo_text_extraction': (True, True, None),
      'source_program_gate_derivation_for_diagnostic_failure': (
          True, True, True
      ),
      'diagnostic_failure_record_construction': (True, True, True),
  }[stage]
  lowered_id = node.get('lowered_python_id')
  compiled_id = node.get('compiled_python_id')
  if (
      node.get('lower_call_count') != 1
      or node.get('compile_call_count') != 1
      or tuple(node.get(name) for name in (
          'stablehlo_read_from_lowered_object',
          'pre_backend_hlo_read_from_lowered_object',
          'compiled_hlo_read_from_compiled_object',
      )) != expected_reads
      or node.get('compile_argument_is_lowered_object') is not True
      or node.get('signature_attestation_from_apply_arguments') is not True
      or node.get('apply_callable_is_compiled_object') is not True
      or node.get('compiler_record_is_gate_record') is not (
          True if stage in {
              'source_program_gate_derivation_for_diagnostic_failure',
              'diagnostic_failure_record_construction',
          } else None
      )
      or isinstance(lowered_id, bool) or not isinstance(lowered_id, int)
      or isinstance(compiled_id, bool) or not isinstance(compiled_id, int)
      or lowered_id < 0 or compiled_id < 0 or lowered_id == compiled_id
  ):
    raise AnalysisError('Nonpublication same-object phase matrix changed.')
  return node


def _validate_nonpublication_cache(
    value: Any, *, triggering_reason: str | None,
) -> dict[str, Any]:
  node = _exact_keys(value, {
      'pre_import', 'historical_stage', 'historical_binding',
      'terminal_live_binding', 'cache_hit_evidence',
      'historical_to_terminal_tree_exact',
      'historical_to_terminal_equality_is_a_gate',
      'historical_snapshot_not_reauthenticated_as_live_files',
      'default_user_cache_paths_eligible',
      'cache_outputs_are_diagnostic_only',
  }, 'NONPUBLICATION_TERMINAL_FAILURE.model_kernel_cache_state')
  pre_import = _validate_cache_binding(
      node.get('pre_import'), root=_MODEL_CACHE_DIR, role='model',
      label='nonpublication model-cache pre-import', compare_live=False,
  )
  if (
      pre_import['directory_paths'] != ['.', 'triton', 'xdg']
      or pre_import['file_count'] != 0
  ):
    raise AnalysisError('Nonpublication pre-import cache was not empty.')
  if node.get('historical_stage') != 'post_compile':
    raise AnalysisError('Nonpublication historical cache stage changed.')
  historical = _validate_cache_binding(
      node.get('historical_binding'), root=_MODEL_CACHE_DIR, role='model',
      label='nonpublication post-compile cache', compare_live=False,
  )
  terminal = _validate_cache_binding(
      node.get('terminal_live_binding'), root=_MODEL_CACHE_DIR, role='model',
      label='nonpublication terminal-live cache', compare_live=True,
  )
  evidence = node.get('cache_hit_evidence')
  if triggering_reason == 'cache_signal_unavailable':
    if evidence is not None:
      raise AnalysisError('Unavailable cache signal synthesized evidence.')
  else:
    _validate_cache_hit_evidence(
        evidence, phase='model_post_compile', expected_hit=False,
        label='nonpublication cache-hit evidence',
    )
  equality = historical['tree_sha256'] == terminal['tree_sha256']
  if (
      node.get('historical_to_terminal_tree_exact') is not equality
      or node.get('historical_to_terminal_equality_is_a_gate') is not False
      or node.get('historical_snapshot_not_reauthenticated_as_live_files')
      is not True
      or node.get('default_user_cache_paths_eligible') is not False
      or node.get('cache_outputs_are_diagnostic_only') is not True
  ):
    raise AnalysisError('Nonpublication cache diagnostic boundary changed.')
  return {
      'path': str(_MODEL_CACHE_DIR.resolve()),
      'pre_import_binding': pre_import, 'historical_binding': historical,
      'terminal_live_binding': terminal, 'directory_paths_exact': True,
      'cache_hit': False, 'cache_hit_evidence': evidence,
      'historical_to_terminal_equality_is_a_gate': False,
  }


def _validate_nonpublication_tree(
    run_dir: Path, terminal: Mapping[str, Any], expected_preterminal: set[str],
) -> dict[str, Any]:
  preterminal = _exact_keys(
      terminal.get('preterminal_tree_binding'),
      {'file_count', 'directory_count', 'file_bindings', 'file_tree_sha256',
       'directory_paths', 'directory_tree_sha256'},
      'NONPUBLICATION_TERMINAL_FAILURE.preterminal_tree_binding',
  )
  bindings = _validate_binding_map(
      preterminal.get('file_bindings'), run_dir,
      'nonpublication preterminal bindings',
      expected_paths=sorted(expected_preterminal),
  )
  directories = _parent_directories(expected_preterminal)
  if (
      preterminal.get('file_count') != len(expected_preterminal)
      or preterminal.get('file_tree_sha256') != _binding_map_digest(bindings)
      or preterminal.get('directory_count') != len(directories)
      or preterminal.get('directory_paths') != directories
      or preterminal.get('directory_tree_sha256')
      != _directory_digest(directories)
  ):
    raise AnalysisError('Nonpublication preterminal tree binding changed.')
  terminal_name = 'NONPUBLICATION_TERMINAL_FAILURE.json'
  expected_files = expected_preterminal | {terminal_name}
  actual_files: set[str] = set()
  actual_directories = {'.'}
  for entry in run_dir.rglob('*'):
    mode = entry.lstat().st_mode
    relative = entry.relative_to(run_dir).as_posix()
    if stat.S_ISLNK(mode):
      raise AnalysisError('Nonpublication run contains a symlink.')
    if stat.S_ISREG(mode):
      if stat.S_IMODE(mode) != 0o400:
        raise AnalysisError(f'Nonpublication artifact mode changed: {relative}.')
      actual_files.add(relative)
    elif stat.S_ISDIR(mode):
      if stat.S_IMODE(mode) != 0o700:
        raise AnalysisError(f'Nonpublication directory mode changed: {relative}.')
      actual_directories.add(relative)
    else:
      raise AnalysisError(f'Nonpublication run has a special entry: {relative}.')
  if actual_files != expected_files:
    raise AnalysisError('Nonpublication whole-run file membership changed.')
  expected_directories = set(_parent_directories(expected_files))
  if actual_directories != expected_directories:
    raise AnalysisError('Nonpublication whole-run directory membership changed.')
  terminal_path = run_dir / terminal_name
  full = dict(bindings)
  full[terminal_name] = {
      'sha256': _sha256(terminal_path),
      'size_bytes': terminal_path.stat().st_size,
  }
  return {
      'path': str(run_dir.resolve()), 'file_count': len(full),
      'directory_count': len(actual_directories), 'file_bindings': full,
      'file_tree_sha256': _binding_map_digest(full),
      'directory_paths': sorted(actual_directories),
      'directory_tree_sha256': _directory_digest(sorted(actual_directories)),
      'terminal_kind': 'nonpublication_terminal_failure',
      'terminal_binding': _absolute_binding(terminal_path),
      'start_binding': _absolute_binding(run_dir / 'ATTEMPT_STARTED.json'),
      'strict_membership_exact': True,
  }


def _validate_nonpublication_source_gate(
    value: Any, binding: Any, *, artifacts: Mapping[str, Mapping[str, Any]],
    source_audit: Mapping[str, Any], same_object: Mapping[str, Any],
    freeze: Mapping[str, Any], run_dir: Path, reason: str,
    triggering_failure: Mapping[str, Any],
) -> dict[str, Any]:
  gate = _validate_content_bound_object(
      value, binding, 'nonpublication source-program gate', keys={
          'contract', 'observed', 'stablehlo_exact', 'pre_backend_hlo_exact',
          'program_signature_structure_exact',
          'program_signatures_canonical_exact', 'entry_abi_exact',
          'source_runtime_device_toolchain_checkpoint_reference_exact',
          'source_input_audit', 'source_input_audit_content_binding',
          'same_object_attestation', 'same_object_attestation_content_binding',
          'same_lowered_compiled_object', 'source_program_exact',
      },
  )
  observed = _exact_keys(gate.get('observed'), {
      'stablehlo_sha256', 'stablehlo_size_bytes',
      'pre_backend_hlo_sha256', 'pre_backend_hlo_size_bytes',
      'program_signatures_sha256', 'entry_abi_sha256',
  }, 'nonpublication source-program observed')
  stable = artifacts['compiler/eight_row/graph.stablehlo.mlir']
  pre_backend = artifacts['compiler/eight_row/graph.pre_backend.hlo.txt']
  compiled_text = (
      run_dir / 'compiler/eight_row/graph.compiled.hlo.txt'
  ).read_text(encoding='utf-8')
  entry_exact = _diagnostic_entry_abi_exact(
      compiled_text, reason=reason, failure=triggering_failure,
      observed_sha256=observed.get('entry_abi_sha256'),
  )
  stable_exact = stable == SOURCE_STABLEHLO
  pre_backend_exact = pre_backend == SOURCE_PRE_BACKEND_HLO
  signatures_exact = (
      _canonical_json_sha256(freeze['program_signatures'])
      == PROGRAM_SIGNATURES_SHA256
  )
  expected_observed = {
      'stablehlo_sha256': stable['sha256'],
      'stablehlo_size_bytes': stable['size_bytes'],
      'pre_backend_hlo_sha256': pre_backend['sha256'],
      'pre_backend_hlo_size_bytes': pre_backend['size_bytes'],
      'program_signatures_sha256': PROGRAM_SIGNATURES_SHA256,
      'entry_abi_sha256': observed.get('entry_abi_sha256'),
  }
  primitives = {
      'stablehlo_exact': stable_exact,
      'pre_backend_hlo_exact': pre_backend_exact,
      'program_signature_structure_exact': signatures_exact,
      'program_signatures_canonical_exact': signatures_exact,
      'entry_abi_exact': entry_exact,
      'source_runtime_device_toolchain_checkpoint_reference_exact': all(
          item is True for item in source_audit.values()
      ),
      'same_lowered_compiled_object': True,
  }
  if (
      gate.get('contract') != freeze['source_program_contract']
      or dict(observed) != expected_observed
      or gate.get('source_input_audit') != source_audit
      or gate.get('source_input_audit_content_binding')
      != _content_binding(source_audit)
      or gate.get('same_object_attestation') != same_object
      or gate.get('same_object_attestation_content_binding')
      != _content_binding(same_object)
      or any(gate.get(name) is not expected for name, expected in primitives.items())
  ):
    raise AnalysisError('Nonpublication source-program evidence changed.')
  expected_source_exact = all(primitives.values())
  if gate.get('source_program_exact') is not expected_source_exact:
    raise AnalysisError('Nonpublication source-program aggregate changed.')
  return gate


def _validate_nonpublication_terminal_archive(
    run_dir: Path, terminal: Any, *, start: Mapping[str, Any],
    freeze: Mapping[str, Any], freeze_sha: str, bundle_root: Path,
    prior333: Mapping[str, Any], prior331: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
  node = _exact_keys(
      terminal, set(NONPUBLICATION_TERMINAL_KEYS),
      'NONPUBLICATION_TERMINAL_FAILURE',
  )
  if len(node) != 58:
    raise AnalysisError('NONPUBLICATION terminal is not exactly 58 keys.')
  stage = node.get('failure_stage')
  if stage not in NONPUBLICATION_FAILURE_STAGES:
    raise AnalysisError('NONPUBLICATION terminal failure stage changed.')
  diagnostic_stage = stage in {
      'source_program_gate_derivation_for_diagnostic_failure',
      'diagnostic_failure_record_construction',
  }
  expected = {
      'schema_version': 'v3.3.4.3-nonpublication-terminal-v1',
      'status': 'incomplete_nonpublication_infrastructure_failure',
      'stop_reason': 'post_compile_nonpublication_infrastructure_failure',
      'attempt_id': ATTEMPT_ID, 'script_version': 'v3.3.4.3',
      'amendment_commit': AMENDMENT_COMMIT,
      'amendment_sha256': AMENDMENT_SHA256,
      'inherited_v3_3_4_commit': PREDECESSOR_AMENDMENT_COMMIT,
      'inherited_v3_3_4_sha256': PREDECESSOR_AMENDMENT_SHA256,
      'inherited_v3_3_4_1_commit': PUBLICATION_AMENDMENT_COMMIT,
      'inherited_v3_3_4_1_sha256': PUBLICATION_AMENDMENT_SHA256,
      'freeze_sha256': freeze_sha,
      'git_head': start['git_head'],
      'external_freeze_authorization': start['external_freeze_authorization'],
      'runner_pid': start['runner_pid'],
      'started_at_unix_s': start['started_at_unix_s'],
      'prior_v3_3_3_binding': prior333,
      'prior_v3_3_3_1_archive_binding': prior331,
      'model_apply_attempt_count': 0, 'model_apply_success_count': 0,
      'valid_record_count': 0, 'raw_record_count': 0,
      'dispatch_started_count': 0, 'dispatch_completed_count': 0,
      'six_row_compile_count': 0, 'identity_rerun_count': 0,
      'main_cube_rerun_count': 0, 'old_ood_records_reused': 0,
      'confirmation_model_calls': 0,
      'confirmation_scope_disclosure': CONFIRMATION_DISCLOSURE,
      'scientific_summary_computed': False,
      'donor_normalization_computed': False,
      'shapley_or_nomination_computed': False,
      'interaction_or_resolution_computed': False,
      'nomination_performed': False, 'combined_analysis_permitted': False,
      'no_retry': True,
  }
  for key, expected_value in expected.items():
    if node.get(key) != expected_value:
      raise AnalysisError(f'NONPUBLICATION terminal changed at {key}.')
  _finite(node.get('created_at_unix_s'), 'NONPUBLICATION created_at_unix_s')
  _validate_failure(node.get('failure'), 'NONPUBLICATION failure')
  triggering_failure = node.get('triggering_diagnostic_failure')
  triggering_reason = node.get('triggering_diagnostic_stop_reason')
  if diagnostic_stage:
    _validate_failure(triggering_failure, 'NONPUBLICATION triggering failure')
    if triggering_reason not in DIAGNOSTIC_STOP_REASONS:
      raise AnalysisError('NONPUBLICATION triggering reason changed.')
    if (
        DIAGNOSTIC_TRIGGER_TYPE_TO_REASON.get(triggering_failure.get('type'))
        != triggering_reason
    ):
      raise AnalysisError('NONPUBLICATION trigger type/reason changed.')
  elif triggering_failure is not None or triggering_reason is not None:
    raise AnalysisError('Extraction terminal invented a triggering diagnostic.')
  expected_source = (True,) * 7 + ((True,) if diagnostic_stage else (None,))
  source_audit = _validate_source_audit(
      node.get('source_input_audit'),
      node.get('source_input_audit_content_binding'), expected_source,
      'NONPUBLICATION source_input_audit',
  )
  phase = _exact_keys(node.get('phase_state'), _PHASE_STATE_KEYS,
                      'NONPUBLICATION phase_state')
  expected_phase = {
      name: name in {
          'preflight_passed', 'start_persisted',
          'post_start_source_gate_passed', 'protobuf_persisted',
          'pre_model_import_inventory_persisted',
          'model_construction_attempted', 'model_constructed',
          'reference_cases_loaded', 'signatures_captured',
          'signature_attestation_persisted',
          'post_model_import_inventory_persisted', 'lower_attempted',
          'lower_succeeded', 'compile_attempted', 'compile_succeeded',
          *({'terminal_import_inventory_persisted'} if diagnostic_stage else set()),
      }
      for name in _PHASE_STATE_KEYS
  }
  signature = _validate_signature_attestation(
      run_dir, node.get('program_signature_attestation_binding'), freeze,
      start['external_freeze_authorization'],
  )
  same_object = _validate_nonpublication_same_object(
      node.get('same_object_attestation'),
      node.get('same_object_attestation_content_binding'), stage,
  )
  if node.get('attempt_budget_audit') != {
      'lower_budget': 1, 'compile_budget': 1, 'lower_invocations': 1,
      'compile_invocations': 1, 'forbidden_request': None,
      'forbidden_request_detected_before_invocation': False,
  }:
    raise AnalysisError('NONPUBLICATION attempt-budget audit changed.')
  if node.get('compiler_counts') != {
      'lower_attempt_count': 1, 'compile_attempt_count': 1,
      'successful_compile_count': 1,
  }:
    raise AnalysisError('NONPUBLICATION compiler counts changed.')
  graph_paths = {
      'compiler/eight_row/graph.stablehlo.mlir',
      'compiler/eight_row/graph.pre_backend.hlo.txt',
      'compiler/eight_row/graph.compiled.hlo.txt',
  }
  graph_bindings = _validate_binding_map(
      node.get('graph_artifact_bindings'), run_dir,
      'NONPUBLICATION graph bindings',
      expected_paths=sorted(graph_paths if diagnostic_stage else set()),
  )
  trigger_operation = None
  if diagnostic_stage:
    compiled_hlo = (
        run_dir / 'compiler/eight_row/graph.compiled.hlo.txt'
    ).read_text(encoding='utf-8')
    trigger_operation = _validate_triggering_diagnostic_operation(
        triggering_failure, str(triggering_reason), compiled_hlo,
    )
  expected_preterminal = {
      'ATTEMPT_STARTED.json', 'IMPORT_PROVENANCE_PRE_MODEL.json',
      'PROTOBUF_PROVENANCE.json',
      'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
      'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json',
  }
  if diagnostic_stage:
    expected_preterminal |= graph_paths | {'IMPORT_PROVENANCE.json'}
  imports = _exact_keys(
      node.get('import_provenance_phases'),
      {'pre_model', 'post_model_precompile', 'terminal'},
      'NONPUBLICATION import_provenance_phases',
  )
  if (imports.get('terminal') is None) is diagnostic_stage:
    raise AnalysisError('NONPUBLICATION terminal import nullability changed.')
  import_audit = _validate_imports(
      run_dir, {
          'import_provenance_phases': dict(imports),
          'external_freeze_authorization': start['external_freeze_authorization'],
      }, bundle_root=bundle_root, freeze=freeze,
  )
  protobuf_audit = _validate_protobuf(
      run_dir, {
          'protobuf_provenance_sha256': node.get('protobuf_provenance_sha256'),
          'external_freeze_authorization': start['external_freeze_authorization'],
      }, freeze,
  )
  source_gate = node.get('source_program_gate_without_backend_diagnostics')
  source_gate_binding = node.get(
      'source_program_gate_without_backend_diagnostics_content_binding'
  )
  if stage == 'diagnostic_failure_record_construction':
    source_gate = _validate_nonpublication_source_gate(
        source_gate, source_gate_binding, artifacts=graph_bindings,
        source_audit=source_audit, same_object=same_object, freeze=freeze,
        run_dir=run_dir, reason=str(triggering_reason),
        triggering_failure=triggering_failure,
    )
  elif source_gate is not None or source_gate_binding is not None:
    raise AnalysisError('NONPUBLICATION source gate nullability changed.')
  if stage == 'diagnostic_failure_record_construction':
    expected_phase['source_program_gate_passed'] = source_gate[
        'source_program_exact'
    ]
  if dict(phase) != expected_phase:
    raise AnalysisError('NONPUBLICATION phase-state matrix changed.')
  cache = _validate_nonpublication_cache(
      node.get('model_kernel_cache_state'),
      triggering_reason=triggering_reason,
  )
  run_binding = _validate_nonpublication_tree(
      run_dir, node, expected_preterminal,
  )
  publication = _validate_run_publication_audit(
      node.get('publication_audit'), run_dir=run_dir,
      preterminal=node['preterminal_tree_binding'],
  )
  return dict(node), {
      'signature_attestation_state': 'validated',
      'canonical_sha256': signature['canonical_sha256'],
      'source_input_audit_exact': True,
      'same_object_attestation_exact': True,
      'stablehlo_exact': (
          None if not diagnostic_stage else
          graph_bindings['compiler/eight_row/graph.stablehlo.mlir']
          == SOURCE_STABLEHLO
      ),
      'pre_backend_exact': (
          None if not diagnostic_stage else
          graph_bindings['compiler/eight_row/graph.pre_backend.hlo.txt']
          == SOURCE_PRE_BACKEND_HLO
      ),
      'entry_abi_exact': (
          None if source_gate is None else source_gate['entry_abi_exact']
      ),
      'source_program_exact': (
          None if source_gate is None else source_gate['source_program_exact']
      ),
      'compiled_backend_diagnostic_only': True,
      'diagnostic_provenance_complete': False,
      'import_audit': import_audit, 'protobuf_audit': protobuf_audit,
      'failure_stage': stage, 'triggering_reason': triggering_reason,
      'trigger_operation': trigger_operation,
  }, run_binding, {'publication': publication, 'cache': cache}


def analyze(
    run_dir: Path, *, bundle_root: Path | None = None,
    _raw_access_marker: Callable[[], None] | None = None,
    _attempt_token: object | None = None,
    _attempt_started_sha256: str | None = None,
) -> dict[str, Any]:
  """Fail-closed CPU structural audit; computes no scientific statistic."""
  _assert_cpu_only('v3.3.4.3 analyzer entry')
  run_dir = run_dir.resolve()
  bundle_root = _REPO_ROOT if bundle_root is None else bundle_root.resolve()
  if run_dir == _RUN_DIR.resolve():
    _validate_active_analysis_attempt(
        run_dir, token=_attempt_token, started_sha256=_attempt_started_sha256,
        rehash_run_artifacts=False,
    )
  if not run_dir.is_dir() or run_dir.is_symlink():
    raise AnalysisError('v3.3.4.3 run directory is absent or unsafe.')
  freeze, freeze_sha, prior333, original_manifest, _unused, prior331 = (
      _validate_freeze_v3343(
          run_dir, bundle_root=bundle_root,
          active_started_sha256=(
              _attempt_started_sha256
              if run_dir == _RUN_DIR.resolve() else None
          ),
      )
  )
  if run_dir == _RUN_DIR.resolve():
    _validate_active_analysis_attempt(
        run_dir, token=_attempt_token, started_sha256=_attempt_started_sha256,
        rehash_run_artifacts=True,
    )
  start = _validate_start_v3343(
      run_dir, freeze, freeze_sha, prior333=prior333, prior331=prior331
  )
  preflight_audit = _validate_preflight_and_same_process(start, freeze)
  nonpublication_path = run_dir / 'NONPUBLICATION_TERMINAL_FAILURE.json'
  if nonpublication_path.exists():
    if any((run_dir / name).exists() for name in (
        'RUN_COMPLETE.json', 'POST_START_PROVENANCE_FAILURE.json',
        'TERMINAL_FAILURE.json',
    )):
      raise AnalysisError('NONPUBLICATION terminal coexists with another terminal.')
    terminal, compiler_audit, run_binding, auxiliary = (
        _validate_nonpublication_terminal_archive(
            run_dir,
            _read_json(nonpublication_path, 'NONPUBLICATION_TERMINAL_FAILURE'),
            start=start, freeze=freeze, freeze_sha=freeze_sha,
            bundle_root=bundle_root, prior333=prior333, prior331=prior331,
        )
    )
    _assert_cpu_only('v3.3.4.3 nonpublication archive exit')
    return _result_v3343(
        status='complete_incomplete_nonpublication_infrastructure_archive',
        decision='post_compile_nonpublication_failure_no_scientific_analysis',
        terminal_kind='nonpublication_terminal_failure',
        compiler_state='compiled_without_legal_graph_gate_record',
        k=0, d=0, started=0, completed=0, id0=False, id255=False,
        prior333=prior333, prior331=prior331,
        start_binding=_absolute_binding(
            _ANALYSIS_ATTEMPT_DIR / 'ANALYSIS_ATTEMPT_STARTED.json'
        ),
        run_binding=run_binding, preflight_binding=preflight_audit,
        model_publication_audit=auxiliary['publication'],
        model_cache_binding=auxiliary['cache'],
        extra={
            'model_terminal_status': terminal['status'],
            'model_stop_reason': terminal['stop_reason'],
            'compiler_audit': compiler_audit,
            'terminal_audit': {
                'terminal_detail': {
                    'failure_stage': terminal['failure_stage'],
                    'triggering_diagnostic_stop_reason': terminal[
                        'triggering_diagnostic_stop_reason'
                    ],
                },
            },
        },
    )
  terminal_failure_path = run_dir / 'TERMINAL_FAILURE.json'
  if terminal_failure_path.exists():
    if (
        (run_dir / 'RUN_COMPLETE.json').exists()
        or (run_dir / 'POST_START_PROVENANCE_FAILURE.json').exists()
    ):
      raise AnalysisError('TERMINAL_FAILURE coexists with another terminal.')
    terminal_failure, terminal_detail, model_publication = (
        _validate_terminal_failure_archive(
            run_dir,
            _read_json(terminal_failure_path, 'TERMINAL_FAILURE'),
            start=start,
        )
    )
    k = terminal_failure['valid_record_count']
    started = terminal_failure['model_apply_attempt_count']
    completed = terminal_failure['model_apply_success_count']
    d = completed - 4 * k
    if d < 0 or d > 4 or started < completed or started > 4 * k + 4:
      raise AnalysisError('TERMINAL_FAILURE journal arithmetic changed.')
    phase = terminal_failure['phase_state']
    compiler_state = (
        'compiled_ready_controlled_stop'
        if phase['compile_succeeded'] else
        'compile_failed' if phase['compile_attempted'] else
        'lower_failed' if phase['lower_attempted'] else
        'signature_attestation_failed'
        if phase['signatures_captured'] else 'not_reached'
    )
    run_binding = _terminal_failure_run_binding(run_dir, terminal_failure)
    model_cache = _terminal_failure_model_cache(start)
    _assert_cpu_only('v3.3.4.3 terminal-failure archive exit')
    return _result_v3343(
        status='complete_incomplete_publication_archive',
        decision=terminal_detail['decision'], terminal_kind='terminal_failure',
        compiler_state=compiler_state, k=k, d=d, started=started,
        completed=completed, id0=False, id255=False,
        prior333=prior333, prior331=prior331,
        start_binding=_absolute_binding(
            _ANALYSIS_ATTEMPT_DIR / 'ANALYSIS_ATTEMPT_STARTED.json'
        ),
        run_binding=run_binding, preflight_binding=preflight_audit,
        model_publication_audit=model_publication,
        model_cache_binding=model_cache,
        extra={
            'model_terminal_status': terminal_failure['status'],
            'model_stop_reason': terminal_failure['stop_reason'],
            'terminal_audit': {
                'terminal_detail': {
                    'k_valid_records': k, 'd_completed': d,
                },
            },
            'failed_current_audit': terminal_detail.get(
                'failed_current_audit'
            ),
        },
    )
  post_start_failure = run_dir / 'POST_START_PROVENANCE_FAILURE.json'
  completion_path = run_dir / 'RUN_COMPLETE.json'
  if post_start_failure.exists():
    _strict_tree(
        run_dir, {'ATTEMPT_STARTED.json', 'POST_START_PROVENANCE_FAILURE.json'},
        'post-START provenance-failure archive',
    )
    terminal = _read_json(post_start_failure, 'POST_START_PROVENANCE_FAILURE')
    keys = {
        'status', 'stop_reason', 'message', 'failure', 'attempt_id',
        'script_version', 'amendment_sha256', 'freeze_sha256', 'git_head',
        'external_freeze_authorization', 'runner_pid',
        'source_inventory_failure', 'model_constructed', 'model_apply_count',
        'source_input_audit', 'source_input_audit_content_binding',
        'confirmation_model_calls', 'scientific_summary_computed',
        'combined_analysis_permitted', 'failed_at_unix_s',
    }
    _exact_keys(terminal, keys, 'POST_START_PROVENANCE_FAILURE')
    if (
        terminal.get('status') != 'controlled_stop_post_start_provenance_failure'
        or terminal.get('stop_reason') != 'post_start_provenance_failure'
        or terminal.get('model_constructed') is not False
        or terminal.get('model_apply_count') != 0
        or terminal.get('confirmation_model_calls') != 0
        or terminal.get('scientific_summary_computed') is not False
        or terminal.get('combined_analysis_permitted') is not False
    ):
      raise AnalysisError('POST_START provenance terminal changed.')
    audit = _exact_keys(terminal.get('source_input_audit'), _SOURCE_AUDIT_KEYS, 'POST_START source audit')
    first = list(audit.values())[:4]
    if not all(isinstance(item, bool) for item in first) or all(first) or list(audit.values())[4:] != [None] * 4:
      raise AnalysisError('POST_START source-audit failure matrix changed.')
    if terminal.get('source_input_audit_content_binding') != _content_binding(audit):
      raise AnalysisError('POST_START source-audit binding changed.')
    _assert_cpu_only('v3.3.4.3 post-START archive exit')
    return _result_v3343(
        status='complete_controlled_stop_structural_archive',
        decision='controlled_stop_post_start_provenance_failure',
        terminal_kind='post_start_provenance_failure', compiler_state='not_reached',
        k=0, d=0, started=0, completed=0, id0=False, id255=False,
        prior333=prior333, prior331=prior331,
        start_binding=_absolute_binding(run_dir / 'ATTEMPT_STARTED.json'),
        run_binding={
            'path': str(run_dir), 'file_count': 2, 'directory_count': 1,
            'file_bindings': {
                path.name: {'sha256': _sha256(path), 'size_bytes': path.stat().st_size}
                for path in sorted((run_dir / 'ATTEMPT_STARTED.json', post_start_failure))
            },
            'file_tree_sha256': _tree_digest(
                [run_dir / 'ATTEMPT_STARTED.json', post_start_failure], run_dir
            ),
            'directory_paths': ['.'],
            'directory_tree_sha256': _directory_digest(['.']),
            'terminal_kind': 'post_start_provenance_failure',
            'terminal_binding': _absolute_binding(post_start_failure),
            'start_binding': _absolute_binding(run_dir / 'ATTEMPT_STARTED.json'),
            'strict_membership_exact': True,
        },
        preflight_binding=preflight_audit,
        model_publication_audit=_model_publication_audit_without_failure(
            run_dir, terminal_name='POST_START_PROVENANCE_FAILURE.json'
        ),
        model_cache_binding=_terminal_failure_model_cache(start),
    )
  if not completion_path.exists():
    raise AnalysisError('Run has no auditable terminal.')
  completion = _read_json(completion_path, 'RUN_COMPLETE')
  completion, terminal_audit = _validate_terminal_common(
      completion, freeze_sha=freeze_sha, start=start
  )
  run_publication_audit = _validate_run_publication_audit(
      completion.get('publication_audit'), run_dir=run_dir,
      preterminal=completion['preterminal_tree_binding'],
  )
  cases = _load_cases()
  sequence_bindings = _sequence_bindings_from_original(cases, original_manifest)
  manifest = _read_json(run_dir / 'RAW_MANIFEST.json', 'RAW_MANIFEST')
  pairs, failed_binding = _validate_manifest_v3343(
      run_dir, manifest, cases=cases, runner_pid=start['runner_pid'],
      source_binding=completion['source_input_audit_content_binding'],
      object_binding=completion['same_object_attestation_content_binding'],
  )
  if completion.get('raw_manifest') != manifest:
    raise AnalysisError('RUN_COMPLETE embedded RAW_MANIFEST changed.')
  k = len(pairs)
  detail = completion['terminal_detail']
  d = detail['d_completed']
  if completion['valid_record_count'] != k:
    raise AnalysisError('RUN_COMPLETE valid-record count differs from manifest.')
  import_audit = _validate_imports(
      run_dir, completion, bundle_root=bundle_root, freeze=freeze
  )
  protobuf_audit = _validate_protobuf(run_dir, completion, freeze)
  compiler = _validate_compiler_v3343(run_dir, completion, freeze, start)
  run_binding = _validate_run_membership(run_dir, completion, manifest)
  compiler_record = compiler.get('record')
  model_cache = _validate_model_cache_final(
      completion['model_kernel_cache_final'], compiler=compiler_record,
      status=completion['status'], reason=completion['stop_reason'],
  )
  # All non-raw provenance, hashes, and exact path membership are established
  # before opening the failed-current payload or a valid development record.
  failed_audit = None
  if (failed_binding is not None or pairs) and _raw_access_marker is not None:
    _raw_access_marker()
  if failed_binding is not None:
    failed_audit = _validate_failed_current(
        _read_json(run_dir / failed_binding['path'], 'failed_current'),
        k=k, cases=cases,
        source_binding=completion['source_input_audit_content_binding'],
        object_binding=completion['same_object_attestation_content_binding'],
        authorization=completion['external_freeze_authorization'],
        started_map=manifest['dispatch_started_bindings'],
        completed_map=manifest['dispatch_completed_bindings'],
    )
    if failed_audit['d'] != d:
      raise AnalysisError('Terminal d differs from failed-current evidence.')
  elif d != 0:
    raise AnalysisError('Terminal has d>0 without failed-current artifact.')
  if completion['model_apply_attempt_count'] != manifest['dispatch_started_count'] or completion['model_apply_success_count'] != manifest['dispatch_completed_count']:
    raise AnalysisError('RUN_COMPLETE apply counts differ from journal.')
  audits = []
  fingerprint = compiler.get('executable_fingerprint')
  source_audit = _validate_content_bound_object(
      completion['source_input_audit'],
      completion['source_input_audit_content_binding'],
      'RUN_COMPLETE.source_input_audit', keys=_SOURCE_AUDIT_KEYS,
  )
  same_object = (
      {} if completion['same_object_attestation'] is None else
      _validate_content_bound_object(
          completion['same_object_attestation'],
          completion['same_object_attestation_content_binding'],
          'RUN_COMPLETE.same_object_attestation', keys=_SAME_OBJECT_KEYS,
      )
  )
  for index, (order, anchor) in enumerate(pairs):
    record = _read_json(run_dir / _artifact_relative(cases[order], anchor), 'raw record')
    audits.append(_validate_record(
        record, case=cases[order], donor_case=cases[_donor_order(order)],
        anchor=anchor, execution_index=index, freeze_sha256=freeze_sha,
        executable_fingerprint=fingerprint, original_manifest=original_manifest,
        sequence_bindings=sequence_bindings,
        authorization=completion['external_freeze_authorization'],
        source_audit=source_audit, same_object=same_object,
        started_bindings=manifest['dispatch_started_bindings'],
        completed_bindings=manifest['dispatch_completed_bindings'],
        allow_invalid=False,
    ))
  full = completion['status'] == 'complete_structural_sidecar'
  id0 = len(audits) == 80 and sum(row['anchor'] == 0 for row in audits) == 20
  id255 = len(audits) == 80 and sum(row['anchor'] == 255 for row in audits) == 20
  if (
      completion.get('all_80_recipient_anchors_complete') is not full
      or completion.get('id0_all20') is not (id0 if full else False)
      or completion.get('id255_all20') is not (id255 if full else False)
  ):
    raise AnalysisError('RUN_COMPLETE closure flags differ from audited prefix.')
  if full and (k != 80 or d != 0 or compiler.get('source_program_exact') is not True):
    raise AnalysisError('Complete terminal lacks 80 records/source closure.')
  _assert_cpu_only('v3.3.4.3 analyzer exit')
  if full:
    compiler_state = 'compiled_ready_complete'
  else:
    compiler_state = {
        'not_reached': 'not_reached',
        'signature_attestation_failed': 'signature_attestation_failed',
        'lower_failed': 'lower_failed',
        'compile_failed': 'compile_failed',
        'compiled_artifacts_no_gate_record': 'compiled_artifacts_no_gate_record',
    }.get(compiler['state'])
    if compiler_state is None:
      compiler_state = {
          'controlled_stop_source_program_mismatch': 'compiled_source_mismatch',
          'controlled_stop_diagnostic_provenance_failure': 'compiled_diagnostic_failure',
          'controlled_stop_partial_dispatch': 'compiled_ready_controlled_prefix',
          'controlled_stop_four_call_invalid': 'compiled_ready_controlled_prefix',
      }.get(completion['status'], 'compiled_ready_controlled_stop')
  return _result_v3343(
      status='complete_structural_sidecar_audit' if full else 'complete_controlled_stop_structural_archive',
      decision='structurally_complete_no_scientific_analysis' if full else completion['status'],
      terminal_kind='run_complete',
      compiler_state=compiler_state,
      k=k, d=d, started=manifest['dispatch_started_count'],
      completed=manifest['dispatch_completed_count'], id0=id0, id255=id255,
      prior333=prior333, prior331=prior331,
      start_binding=_absolute_binding(
          _ANALYSIS_ATTEMPT_DIR / 'ANALYSIS_ATTEMPT_STARTED.json'
      ),
      run_binding=run_binding, preflight_binding=preflight_audit,
      model_publication_audit=run_publication_audit,
      model_cache_binding=model_cache,
      extra={
          'terminal_audit': terminal_audit,
          'model_terminal_status': completion['status'],
          'model_stop_reason': completion['stop_reason'],
          'failed_current_audit': failed_audit,
          'run_binding': run_binding,
          'import_audit': import_audit,
          'protobuf_audit': protobuf_audit,
          'compiler_audit': compiler.get('audit', {}),
          'run_publication_audit': run_publication_audit,
      },
  )


def _result_v3343(
    *, status: str, decision: str, terminal_kind: str, compiler_state: str,
    k: int, d: int, started: int, completed: int, id0: bool, id255: bool,
    prior333: Mapping[str, Any], prior331: Mapping[str, Any],
    start_binding: Mapping[str, Any], run_binding: Mapping[str, Any],
    preflight_binding: Mapping[str, Any],
    model_publication_audit: Mapping[str, Any],
    model_cache_binding: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  detail = {} if extra is None else dict(extra)
  model_publication = _exact_keys(
      model_publication_audit, set(PUBLICATION_AUDIT_KEYS),
      'model publication audit for ANALYSIS',
  )
  terminal_audit_detail = detail.get('terminal_audit', {})
  terminal_detail = terminal_audit_detail.get('terminal_detail', {})
  run_complete = terminal_kind == 'run_complete'
  valid_pairs = [list(pair) for pair in _execution_order()[:k]]
  expected_next = (
      None if k == EXPECTED_RECORD_COUNT else list(_execution_order()[k])
  )
  control_eligible = status == 'complete_structural_sidecar_audit'
  compiler_detail = detail.get('compiler_audit', {})
  result = {
      'status': status, 'decision': decision,
      'analysis_version': ANALYSIS_VERSION,
      'analysis_attempt_start_binding': dict(start_binding),
      'run_binding': dict(run_binding),
      'preflight_binding': dict(preflight_binding),
      'model_cache_binding': (
          {
              'path': str(_MODEL_CACHE_DIR.resolve()),
              'pre_import_binding': None, 'historical_binding': None,
              'terminal_live_binding': None, 'directory_paths_exact': True,
              'cache_hit': False, 'cache_hit_evidence': None,
              'historical_to_terminal_equality_is_a_gate': False,
          } if model_cache_binding is None else dict(model_cache_binding)
      ),
      'source_and_prior_audit': {
          'current_108_source_rows_exact': True,
          'historical_96_source_rows_exact': True,
          'git_head_exact': True, 'tracked_clean': True,
          'external_freeze_authorization_exact': True,
          'prior_v3_3_3_exact': bool(prior333),
          'prior_v3_3_3_1_exact': bool(prior331),
          'old_analyzer_paths_absent': True,
          'pre_start_exact': True, 'post_start_exact': True,
          'final_exact': True,
      },
      'compiler_and_signature_audit': {
          'compiler_state': compiler_state,
          'artifact_membership_exact': True,
          'signature_attestation_state': compiler_detail.get(
              'signature_attestation_state',
              'not_reached' if compiler_state == 'not_reached' else 'validated',
          ),
          'type_tag_paths_exact': compiler_detail.get('type_tag_paths_exact'),
          'semantic_mapping_exact': compiler_detail.get('semantic_mapping_exact'),
          'canonical_sha256': compiler_detail.get('canonical_sha256'),
          'canonical_size_bytes': compiler_detail.get('canonical_size_bytes'),
          'source_input_audit_exact': compiler_detail.get(
              'source_input_audit_exact', compiler_state != 'not_reached'
          ),
          'same_object_attestation_exact': compiler_detail.get(
              'same_object_attestation_exact', compiler_state not in {
                  'not_reached', 'signature_attestation_failed'
              }
          ),
          'stablehlo_exact': compiler_detail.get('stablehlo_exact'),
          'pre_backend_exact': compiler_detail.get('pre_backend_exact'),
          'entry_abi_exact': compiler_detail.get('entry_abi_exact'),
          'source_program_exact': compiler_detail.get('source_program_exact'),
          'compiled_backend_diagnostic_only': compiler_detail.get(
              'compiled_backend_diagnostic_only', compiler_state.startswith('compiled')
          ),
          'diagnostic_provenance_complete': compiler_detail.get(
              'diagnostic_provenance_complete'
          ),
          'compile_counts_exact': True,
      },
      'dispatch_journal_audit': {
          'started_count': started, 'completed_count': completed,
          'started_prefix_exact': True, 'completed_prefix_exact': True,
          'event_schemas_exact': True, 'event_hash_links_exact': True,
          'call_order_exact': True, 'pid_exact': True,
          'publication_membership_exact': True,
      },
      'raw_prefix_audit': {
          'valid_record_count': k,
          'failed_current_count': int(
              detail.get('failed_current_audit') is not None
          ),
          'valid_pairs': valid_pairs, 'expected_next_pair': expected_next,
          'manifest_exact': run_complete, 'raw_paths_exact': run_complete,
          'raw_schemas_exact': run_complete,
          'failed_current_schema_exact': (
              detail.get('failed_current_audit') is not None
              or terminal_kind != 'run_complete' or d == 0
          ),
          'k': k, 'd': d, 'started_completed_arithmetic_exact': True,
          'lossless_partial_encoding_exact': (
              detail.get('failed_current_audit') is not None
              or d == 0
          ),
      },
      'control_audit': {
          'all_80_complete': k == 80 and d == 0,
          'id0_all20': id0, 'id255_all20': id255,
          'invariant_rows_exact': control_eligible,
          'repeat_fingerprints_exact': control_eligible,
          'donor_maps_exact': control_eligible,
          'sequence_bindings_exact': control_eligible,
          'finiteness_exact': control_eligible,
          'control_state_eligible': control_eligible,
      },
      'terminal_audit': {
          'status': detail.get('model_terminal_status', decision),
          'stop_reason': detail.get('model_stop_reason'),
          'phase_state_exact': True, 'membership_exact': True,
          'count_arithmetic_exact': True, 'budgets_exact': True,
          'disclosure_exact': True, 'no_retry': True,
          'no_forbidden_calls': True, 'terminal_linkage_exact': True,
      },
      # Filled only after RESULT.md has been durably published.  This draft is
      # never serialized as ANALYSIS.json.
      'publication_audit': None,
      'confirmation_boundary': {
          'confirmation_paths_opened': False,
          'confirmation_model_calls': 0,
          'later_exon_metadata_label_exposure_disclosed': True,
          'model_outputs_activations_interventions_blind': True,
      },
      'claim_boundary': {
          'structural_only': True, 'no_biological_claim': True,
          'no_scientific_summary': True, 'no_normalization': True,
          'no_shapley': True, 'no_interaction': True,
          'no_resolution': True, 'no_nomination': True,
          'combined_analysis_permitted': False,
          'future_protocol_required': True,
      },
      'scientific_summary_computed': False,
      'donor_normalization_computed': False,
      'shapley_or_nomination_computed': False,
      'interaction_or_resolution_computed': False,
      'nomination_performed': False, 'combined_analysis_permitted': False,
      'completed_at_unix_s': time.time(),
  }
  _exact_keys(result, {
      'status', 'decision', 'analysis_version',
      'analysis_attempt_start_binding', 'run_binding', 'preflight_binding',
      'model_cache_binding', 'source_and_prior_audit',
      'compiler_and_signature_audit', 'dispatch_journal_audit',
      'raw_prefix_audit', 'control_audit', 'terminal_audit',
      'publication_audit',
      'confirmation_boundary', 'claim_boundary',
      'scientific_summary_computed', 'donor_normalization_computed',
      'shapley_or_nomination_computed',
      'interaction_or_resolution_computed', 'nomination_performed',
      'combined_analysis_permitted', 'completed_at_unix_s',
  }, 'ANALYSIS result')
  return result


_ANALYSIS_KEYS = {
    'status', 'decision', 'analysis_version',
    'analysis_attempt_start_binding', 'run_binding', 'preflight_binding',
    'model_cache_binding', 'source_and_prior_audit',
    'compiler_and_signature_audit', 'dispatch_journal_audit',
    'raw_prefix_audit', 'control_audit', 'terminal_audit',
    'publication_audit', 'confirmation_boundary', 'claim_boundary',
    'scientific_summary_computed', 'donor_normalization_computed',
    'shapley_or_nomination_computed',
    'interaction_or_resolution_computed', 'nomination_performed',
    'combined_analysis_permitted', 'completed_at_unix_s',
}


def _validate_final_analysis(value: Any) -> dict[str, Any]:
  node = _exact_keys(value, _ANALYSIS_KEYS, 'ANALYSIS result')
  if len(node) != 23:
    raise AnalysisError('ANALYSIS does not have the exact 23-key schema.')
  if (
      node.get('analysis_version') != ANALYSIS_VERSION
      or node.get('scientific_summary_computed') is not False
      or node.get('donor_normalization_computed') is not False
      or node.get('shapley_or_nomination_computed') is not False
      or node.get('interaction_or_resolution_computed') is not False
      or node.get('nomination_performed') is not False
      or node.get('combined_analysis_permitted') is not False
  ):
    raise AnalysisError('ANALYSIS scientific boundary changed.')
  _validate_analysis_publication_audit(node.get('publication_audit'))
  return dict(node)


def render_markdown(result: Mapping[str, Any]) -> str:
  lines = [
      '# OpenSplice v3.3.4.3 OOD sidecar structural audit', '',
      f"**Decision:** `{result['decision']}`", '',
  ]
  if result.get('status') == 'complete_structural_sidecar_audit':
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
      'The immutable v3.3.3 source-program stop and v3.3.3.1',
      'representation-only structural archive were independently rebound',
      'and rehashed.', '',
      'Later-exon metadata/labels were exposed after protocol freeze;',
      'confirmation model outputs, activations, and interventions remain',
      'unopened.', '',
  ])
  return '\n'.join(lines)


def _write_outputs(
    result: Mapping[str, Any], *, output_json: Path, output_markdown: Path,
    pre_publish_check: Callable[[str], None],
) -> tuple[dict[str, Any], dict[str, Any]]:
  expected_json = (_ANALYSIS_DIR / 'ANALYSIS.json').resolve()
  expected_markdown = (_ANALYSIS_DIR / 'RESULT.md').resolve()
  if (
      output_json.resolve() != expected_json
      or output_markdown.resolve() != expected_markdown
  ):
    raise AnalysisError('Analysis output paths differ from the frozen destinations.')
  if _ANALYSIS_DIR.exists() or _ANALYSIS_DIR.is_symlink():
    raise FileExistsError('v3.3.4.3 analysis output already exists; never overwrite.')
  _create_append_only_directory(_ANALYSIS_DIR, root_role='analysis_output')
  pre_publish_check('before RESULT')
  result_success = _publish_new_bytes(
      output_markdown, render_markdown(result).encode('utf-8'),
      root_role='analysis_output', root=_ANALYSIS_DIR,
      artifact_role='analysis_result_markdown',
  )
  pre_publish_check('before ANALYSIS')
  attempt_tree = _publication_tree_binding(
      _ANALYSIS_ATTEMPT_DIR, role='analysis_attempt',
      expected_files={'ANALYSIS_ATTEMPT_STARTED.json'},
  )
  output_tree = _publication_tree_binding(
      _ANALYSIS_DIR, role='analysis_output', expected_files={'RESULT.md'},
  )
  final_result = copy.deepcopy(dict(result))
  final_result['publication_audit'] = _analysis_publication_audit(
      attempt_tree=attempt_tree, output_tree=output_tree,
  )
  final_result = _validate_final_analysis(final_result)
  analysis_success = _publish_new_bytes(
      output_json,
      (json.dumps(
          final_result, indent=2, sort_keys=True, allow_nan=False
      ) + '\n').encode('utf-8'),
      root_role='analysis_output', root=_ANALYSIS_DIR,
      artifact_role='analysis_json',
  )
  return final_result, {
      'result': result_success, 'analysis': analysis_success,
  }


def _create_append_only_directory(path: Path, *, root_role: str) -> None:
  if path.exists() or path.is_symlink():
    raise FileExistsError(f'Append-only directory already exists: {path}.')
  import validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_3 as bootstrap  # pylint: disable=g-import-not-at-top
  _assert_cpu_only('v3.3.4.3 publication-directory helper import')
  observed = bootstrap.ensure_publication_directory(root_role)
  if observed.resolve() != path.resolve():
    raise AnalysisError('Publication helper returned the wrong frozen root.')


def _publish_new_bytes(
    path: Path, payload: bytes, *, root_role: str, root: Path,
    artifact_role: str,
) -> dict[str, Any]:
  """Delegates to the sole frozen v3.3.4.3 publication implementation."""
  if path.parent.is_symlink() or not path.parent.is_dir():
    raise AnalysisError(f'Append-only parent is unsafe: {path.parent}.')
  if root.is_symlink() or not root.is_dir():
    raise AnalysisError(f'Append-only root is unsafe: {root}.')
  if path.exists() or path.is_symlink():
    raise FileExistsError(f'Append-only artifact already exists: {path}.')
  # Imported only after source/freeze/HEAD gates prove these exact bytes.
  import validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_3 as bootstrap  # pylint: disable=g-import-not-at-top
  _assert_cpu_only('v3.3.4.3 publication helper import')
  result = bootstrap.publish_bytes(
      root_role, path.relative_to(root).as_posix(), payload,
      artifact_role=artifact_role,
  )
  node = _exact_keys(result, set(PUBLICATION_SUCCESS_KEYS), artifact_role)
  ordinal = node.get('publication_ordinal')
  runner_pid = node.get('runner_pid')
  nonce = node.get('nonce_hex')
  if (
      node.get('schema_version') != PUBLICATION_SCHEMA_VERSION
      or node.get('method') != PUBLICATION_METHOD
      or node.get('root_role') != root_role
      or node.get('final_relative_path') != path.relative_to(root).as_posix()
      or node.get('sha256') != hashlib.sha256(payload).hexdigest()
      or node.get('size_bytes') != len(payload)
      or node.get('mode') != '0400'
      or node.get('st_nlink') != 1
      or isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0
      or runner_pid != os.getpid()
      or not isinstance(nonce, str)
      or re.fullmatch(r'[0-9a-f]{32}', nonce) is None
      or node.get('temp_basename')
      != f'.v3343.tmp.{runner_pid}.{ordinal:06d}.{nonce}'
      or any(node.get(key) is not True for key in (
          'file_fsync_before_rename', 'file_fsync_after_fchmod',
          'rename_noreplace_succeeded', 'parent_fsync_succeeded',
          'post_publish_revalidation_exact',
      ))
  ):
    raise AnalysisError(f'{artifact_role} publication evidence changed.')
  observed = _live_file_publication_binding(path)
  if any(node.get(key) != observed[key] for key in observed):
    raise AnalysisError(f'{artifact_role} publication/live binding differs.')
  return dict(node)


def _write_json_new(
    path: Path, value: Mapping[str, Any], *, root_role: str, root: Path,
    artifact_role: str,
) -> dict[str, Any]:
  payload = json.dumps(
      value, indent=2, sort_keys=True, allow_nan=False
  ) + '\n'
  return _publish_new_bytes(
      path, payload.encode('utf-8'), root_role=root_role, root=root,
      artifact_role=artifact_role,
  )


def _analysis_attempt_precheck(
    run_dir: Path, *, bundle_root: Path,
) -> dict[str, Any]:
  """Completes provenance-only gates before consuming the analysis attempt."""
  if (
      _ANALYSIS_ATTEMPT_DIR.exists() or _ANALYSIS_ATTEMPT_DIR.is_symlink()
      or _ANALYSIS_DIR.exists() or _ANALYSIS_DIR.is_symlink()
  ):
    raise FileExistsError('v3.3.4.3 analysis/attempt exists; never resume or retry.')
  _assert_predecessor_v334_paths_absent('analysis precheck')
  freeze, freeze_sha, prior333, _manifest, _unused, prior331 = (
      _validate_freeze_v3343(run_dir, bundle_root=bundle_root)
  )
  start = _validate_start_v3343(
      run_dir, freeze, freeze_sha, prior333=prior333, prior331=prior331
  )
  terminal_names = [
      name for name in (
          'POST_START_PROVENANCE_FAILURE.json', 'RUN_COMPLETE.json',
          'TERMINAL_FAILURE.json', 'NONPUBLICATION_TERMINAL_FAILURE.json',
      ) if (run_dir / name).exists()
  ]
  if len(terminal_names) != 1:
    raise AnalysisError('Analysis requires exactly one eligible model terminal.')
  terminal_name = terminal_names[0]
  terminal_path = run_dir / terminal_name
  _strict_regular(terminal_path, f'analysis precheck {terminal_name}')
  if stat.S_IMODE(terminal_path.lstat().st_mode) != 0o400:
    raise AnalysisError('Analysis precheck terminal mode changed.')
  analyzer_path = Path(__file__).resolve()
  _strict_regular(analyzer_path, 'v3.3.4.3 analyzer')
  _strict_regular(_TEST_PATH, 'v3.3.4.3 analyzer test')
  return {
      'freeze_sha256': freeze_sha,
      'git_head': start['git_head'],
      'external_freeze_authorization': dict(
          start['external_freeze_authorization']
      ),
      'analyzer_binding': _absolute_binding(analyzer_path),
      'test_binding': _absolute_binding(_TEST_PATH),
      'run_terminal_binding': _absolute_binding(terminal_path),
  }


_ANALYSIS_STARTED_KEYS = {
    'status', 'analysis_version', 'attempt_id', 'acknowledgement', 'git_head',
    'freeze_sha256', 'external_freeze_authorization', 'analyzer_binding',
    'test_binding', 'run_root', 'run_terminal_binding', 'fresh_output_dir',
    'old_analyzer_destinations_absent', 'started_at_unix_s',
}


def _absolute_binding(path: Path) -> dict[str, Any]:
  _strict_regular(path, f'absolute binding {path.name}')
  return {
      'path': str(path.resolve()), 'sha256': _sha256(path),
      'size_bytes': path.stat().st_size,
  }


def _analysis_started_record(precheck: Mapping[str, Any]) -> dict[str, Any]:
  return {
      'status': 'analysis_attempt_started',
      'analysis_version': ANALYSIS_VERSION,
      'attempt_id': ATTEMPT_ID,
      'acknowledgement': '--acknowledge-structural-only-v3-3-4-3',
      'git_head': precheck['git_head'],
      'freeze_sha256': precheck['freeze_sha256'],
      'external_freeze_authorization': dict(
          precheck['external_freeze_authorization']
      ),
      'analyzer_binding': dict(precheck['analyzer_binding']),
      'test_binding': dict(precheck['test_binding']),
      'run_root': str(_RUN_DIR.resolve()),
      'run_terminal_binding': dict(precheck['run_terminal_binding']),
      'fresh_output_dir': str(_ANALYSIS_DIR.resolve()),
      'old_analyzer_destinations_absent': True,
      'started_at_unix_s': time.time(),
  }


def _validate_active_analysis_attempt(
    run_dir: Path, *, token: object | None, started_sha256: str | None,
    rehash_run_artifacts: bool = True,
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
  if stat.S_IMODE(path.lstat().st_mode) != 0o400:
    raise AnalysisError('Active analysis START mode changed.')
  if _sha256(path) != started_sha256:
    raise AnalysisError('Active analysis START hash changed.')
  value = _read_json(path, 'ANALYSIS_ATTEMPT_STARTED')
  _exact_keys(value, _ANALYSIS_STARTED_KEYS, 'ANALYSIS_ATTEMPT_STARTED')
  expected = {
      'status': 'analysis_attempt_started',
      'analysis_version': ANALYSIS_VERSION,
      'attempt_id': ATTEMPT_ID,
      'acknowledgement': '--acknowledge-structural-only-v3-3-4-3',
      'freeze_sha256': _sha256(_FREEZE_PATH),
      'analyzer_binding': _absolute_binding(Path(__file__).resolve()),
      'test_binding': _absolute_binding(_TEST_PATH),
      'run_root': str(run_dir.resolve()),
      'fresh_output_dir': str(_ANALYSIS_DIR.resolve()),
      'old_analyzer_destinations_absent': True,
  }
  for key, expected_value in expected.items():
    if value.get(key) != expected_value:
      raise AnalysisError(f'Active analysis START changed at {key}.')
  _finite(value.get('started_at_unix_s'), 'analysis START.started_at_unix_s')
  authorization = _exact_keys(
      value.get('external_freeze_authorization'),
      {'git_head', 'freeze_path', 'freeze_sha256', 'freeze_size_bytes',
       'live_equals_git_show', 'tracked_clean', 'authorization_source'},
      'analysis START authorization',
  )
  if (
      authorization.get('git_head') != value.get('git_head')
      or authorization.get('freeze_sha256') != value.get('freeze_sha256')
      or authorization.get('live_equals_git_show') is not True
      or authorization.get('tracked_clean') is not True
      or authorization.get('authorization_source')
      != 'external_post_commit_audit'
  ):
    raise AnalysisError('Analysis START authorization changed.')
  terminal = _exact_keys(
      value.get('run_terminal_binding'), {'path', 'sha256', 'size_bytes'},
      'analysis START run terminal',
  )
  if (
      not _is_sha256(terminal.get('sha256'))
      or isinstance(terminal.get('size_bytes'), bool)
      or not isinstance(terminal.get('size_bytes'), int)
      or terminal['size_bytes'] < 0
  ):
    raise AnalysisError('Analysis START terminal binding is malformed.')
  terminal_path = Path(terminal['path'])
  if terminal_path.parent.resolve() != run_dir:
    raise AnalysisError('Analysis START terminal escaped the run root.')
  if terminal_path.name not in {
      'RUN_COMPLETE.json', 'POST_START_PROVENANCE_FAILURE.json',
      'TERMINAL_FAILURE.json', 'NONPUBLICATION_TERMINAL_FAILURE.json',
  }:
    raise AnalysisError('Analysis START terminal filename changed.')
  if rehash_run_artifacts and (
      terminal_path.stat().st_size != terminal['size_bytes']
      or _sha256(terminal_path) != terminal['sha256']
  ):
    raise AnalysisError('Analysis START terminal bytes changed.')
  return dict(value)


def _analysis_toctou_check(
    *, run_dir: Path, started_sha256: str, result: Mapping[str, Any],
    label: str,
) -> None:
  _assert_cpu_only(f'analysis TOCTOU {label}')
  _assert_predecessor_v334_paths_absent(f'analysis TOCTOU {label}')
  _validate_active_analysis_attempt(
      run_dir, token=_ANALYSIS_ATTEMPT_TOKEN,
      started_sha256=started_sha256, rehash_run_artifacts=True,
  )
  freeze = _read_json(_FREEZE_PATH, f'analysis TOCTOU freeze {label}')
  inventory = freeze.get('file_sha256')
  rows = freeze.get('source_inventory_contract', {}).get('rows')
  if not isinstance(inventory, Mapping) or not isinstance(rows, list):
    raise AnalysisError(f'Analysis TOCTOU source inventory absent at {label}.')
  row_map = {
      row['path']: row for row in rows
      if isinstance(row, Mapping) and set(row) == {
          'path', 'sha256', 'size_bytes', 'git_mode'
      }
  }
  if set(row_map) != set(inventory):
    raise AnalysisError(f'Analysis TOCTOU source rows changed at {label}.')
  try:
    head = subprocess.check_output(
        ('git', '-C', str(_REPO_ROOT), 'rev-parse', 'HEAD'), text=True
    ).strip()
    subprocess.check_call(
        ('git', '-C', str(_REPO_ROOT), 'diff', '--quiet', 'HEAD', '--')
    )
  except subprocess.CalledProcessError as error:
    raise AnalysisError(f'Analysis TOCTOU tracked tree changed at {label}.') from error
  for relative, expected_sha in inventory.items():
    path = _REPO_ROOT / relative
    row = row_map[relative]
    _strict_regular(path, f'analysis TOCTOU source {relative}')
    if (
        _sha256(path) != expected_sha
        or path.stat().st_size != row['size_bytes']
        or _git_blob_sha256(head, relative) != expected_sha
    ):
      raise AnalysisError(f'Analysis TOCTOU source changed: {relative}.')
  run_binding = result.get('run_binding')
  if not isinstance(run_binding, Mapping):
    raise AnalysisError('Analysis TOCTOU result lacks a run binding.')
  file_bindings = run_binding.get('file_bindings')
  if not isinstance(file_bindings, Mapping):
    raise AnalysisError('Analysis TOCTOU run file map is malformed.')
  for relative, binding in file_bindings.items():
    path = run_dir / _validate_relative_path(relative, 'TOCTOU run path')
    _strict_regular(path, f'analysis TOCTOU run {relative}')
    if (
        path.stat().st_size != binding.get('size_bytes')
        or _sha256(path) != binding.get('sha256')
    ):
      raise AnalysisError(f'Analysis TOCTOU run file changed: {relative}.')
  terminal_kind = run_binding.get('terminal_kind')
  if terminal_kind == 'terminal_failure':
    start = _read_json(run_dir / 'ATTEMPT_STARTED.json', 'TOCTOU START')
    terminal, _, _ = _validate_terminal_failure_archive(
        run_dir,
        _read_json(run_dir / 'TERMINAL_FAILURE.json', 'TOCTOU terminal failure'),
        start=start,
    )
    if _terminal_failure_run_binding(run_dir, terminal) != run_binding:
      raise AnalysisError('Analysis TOCTOU terminal-failure tree changed.')
  else:
    actual_files = set()
    actual_directories = {'.'}
    for entry in run_dir.rglob('*'):
      relative = entry.relative_to(run_dir).as_posix()
      mode = entry.lstat().st_mode
      if stat.S_ISLNK(mode):
        raise AnalysisError(f'Analysis TOCTOU run symlink appeared: {relative}.')
      if stat.S_ISREG(mode):
        actual_files.add(relative)
      elif stat.S_ISDIR(mode):
        actual_directories.add(relative)
      else:
        raise AnalysisError(f'Analysis TOCTOU run special entry appeared: {relative}.')
    if (
        actual_files != set(file_bindings)
        or actual_directories != set(run_binding.get('directory_paths', []))
        or _binding_map_digest(file_bindings) != run_binding.get('file_tree_sha256')
        or _directory_digest(sorted(actual_directories))
        != run_binding.get('directory_tree_sha256')
    ):
      raise AnalysisError(f'Analysis TOCTOU run membership changed at {label}.')
    if terminal_kind == 'run_complete':
      completion = _read_json(run_dir / 'RUN_COMPLETE.json', 'TOCTOU RUN_COMPLETE')
      _validate_run_publication_audit(
          completion.get('publication_audit'), run_dir=run_dir,
          preterminal=completion['preterminal_tree_binding'],
      )
    elif terminal_kind == 'post_start_provenance_failure':
      _model_publication_audit_without_failure(
          run_dir, terminal_name='POST_START_PROVENANCE_FAILURE.json'
      )
    elif terminal_kind == 'nonpublication_terminal_failure':
      terminal = _read_json(
          run_dir / 'NONPUBLICATION_TERMINAL_FAILURE.json',
          'TOCTOU NONPUBLICATION terminal',
      )
      _exact_keys(
          terminal, set(NONPUBLICATION_TERMINAL_KEYS),
          'TOCTOU NONPUBLICATION terminal',
      )
      _validate_run_publication_audit(
          terminal.get('publication_audit'), run_dir=run_dir,
          preterminal=terminal['preterminal_tree_binding'],
      )
  _validate_prior_v3_3_3()
  _validate_prior_v3_3_3_1()
  start = _read_json(run_dir / 'ATTEMPT_STARTED.json', 'TOCTOU START')
  _validate_preflight_and_same_process(start, freeze)
  model_cache = result.get('model_cache_binding')
  if isinstance(model_cache, Mapping):
    terminal_cache = model_cache.get('terminal_live_binding')
    if terminal_cache is not None:
      _validate_cache_binding(
          terminal_cache, root=_MODEL_CACHE_DIR, role='model',
          label=f'analysis TOCTOU model cache {label}', compare_live=True,
      )
  expected_output = {
      'before RESULT': set(),
      'before ANALYSIS': {'RESULT.md'},
      'before COMPLETE': {'RESULT.md', 'ANALYSIS.json'},
  }.get(label)
  if expected_output is None:
    raise AnalysisError(f'Unknown analysis TOCTOU phase: {label}.')
  _publication_tree_binding(
      _ANALYSIS_DIR, role='analysis_output', expected_files=expected_output,
  )


def _analysis_complete_record(started_sha256: str) -> dict[str, Any]:
  start_binding = _absolute_binding(
      _ANALYSIS_ATTEMPT_DIR / 'ANALYSIS_ATTEMPT_STARTED.json'
  )
  if start_binding['sha256'] != started_sha256:
    raise AnalysisError('Analysis START changed before COMPLETE.')
  analysis_binding = _absolute_binding(_ANALYSIS_DIR / 'ANALYSIS.json')
  result_binding = _absolute_binding(_ANALYSIS_DIR / 'RESULT.md')
  attempt_tree = _publication_tree_binding(
      _ANALYSIS_ATTEMPT_DIR, role='analysis_attempt',
      expected_files={'ANALYSIS_ATTEMPT_STARTED.json'},
  )
  output_tree = _publication_tree_binding(
      _ANALYSIS_DIR, role='analysis_output',
      expected_files={'ANALYSIS.json', 'RESULT.md'},
  )
  result = {
      'status': 'analysis_complete', 'attempt_id': ATTEMPT_ID,
      'analysis_attempt_start_binding': start_binding,
      'analysis_binding': analysis_binding,
      'result_binding': result_binding,
      'output_tree_sha256': _tree_digest(
          [_ANALYSIS_DIR / 'ANALYSIS.json', _ANALYSIS_DIR / 'RESULT.md'],
          _ANALYSIS_DIR,
      ),
      'run_terminal_binding': _read_json(
          _ANALYSIS_ATTEMPT_DIR / 'ANALYSIS_ATTEMPT_STARTED.json',
          'analysis START for COMPLETE',
      )['run_terminal_binding'],
      'publication_audit': _analysis_publication_audit(
          attempt_tree=attempt_tree, output_tree=output_tree,
      ),
      'completed_at_unix_s': time.time(),
  }
  _exact_keys(result, {
      'status', 'attempt_id', 'analysis_attempt_start_binding',
      'analysis_binding', 'result_binding', 'output_tree_sha256',
      'run_terminal_binding', 'publication_audit', 'completed_at_unix_s',
  }, 'ANALYSIS_COMPLETE')
  return result


def _analysis_failure_record(
    error: BaseException, started_sha256: str, *, raw_reached: bool,
) -> dict[str, Any]:
  start_binding = _absolute_binding(
      _ANALYSIS_ATTEMPT_DIR / 'ANALYSIS_ATTEMPT_STARTED.json'
  )
  if start_binding['sha256'] != started_sha256:
    raise AnalysisError('Analysis START changed before FAILURE.')
  publication_failure = getattr(error, 'publication_failure', None)
  if publication_failure is not None:
    publication_failure = _validate_publication_failure(
        publication_failure, 'ANALYSIS_FAILURE.publication_failure'
    )
  attempt_audit, output_audit = _current_analysis_publication_audits(
      publication_failure
  )
  temporary = _analysis_root_maps(
      attempt_audit['temporary_orphan_bindings'],
      output_audit['temporary_orphan_bindings'],
  )
  uncertain = _analysis_root_maps(
      attempt_audit['durability_uncertain_final_bindings'],
      output_audit['durability_uncertain_final_bindings'],
  )
  preexisting = _analysis_root_maps(
      attempt_audit['preexisting_entry_states'],
      output_audit['preexisting_entry_states'],
  )
  result = {
      'status': 'analysis_failure', 'attempt_id': ATTEMPT_ID,
      'analysis_attempt_start_binding': start_binding,
      'type': type(error).__name__, 'message': str(error),
      'traceback': ''.join(traceback.format_exception(error)),
      'raw_values_read': raw_reached,
      'scientific_analysis_performed': False,
      'output_dir_state': _analysis_output_state(output_audit),
      'publication_failure': publication_failure,
      'temporary_orphan_bindings': temporary,
      'durability_uncertain_final_bindings': uncertain,
      'preexisting_entry_states': preexisting,
      'no_new_entry_failure': bool(
          publication_failure is not None
          and not any(temporary.values()) and not any(uncertain.values())
      ),
      'failed_at_unix_s': time.time(),
  }
  _exact_keys(result, {
      'status', 'attempt_id', 'analysis_attempt_start_binding', 'type',
      'message', 'traceback', 'raw_values_read',
      'scientific_analysis_performed', 'output_dir_state',
      'publication_failure', 'temporary_orphan_bindings',
      'durability_uncertain_final_bindings', 'preexisting_entry_states',
      'no_new_entry_failure', 'failed_at_unix_s',
  }, 'ANALYSIS_FAILURE')
  return result


def _current_analysis_publication_audits(
    publication_failure: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
  import validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_3 as bootstrap  # pylint: disable=g-import-not-at-top
  _assert_cpu_only('analysis failure publication audit')
  failure_root = (
      None if publication_failure is None else publication_failure['root_role']
  )
  attempt = bootstrap.publication_audit(
      'analysis_attempt',
      publication_failure if failure_root == 'analysis_attempt' else None,
  )
  output = bootstrap.publication_audit(
      'analysis_output',
      publication_failure if failure_root == 'analysis_output' else None,
  )
  checked_attempt = _validate_root_publication_audit(
      attempt, 'analysis_attempt publication audit',
      root=_ANALYSIS_ATTEMPT_DIR,
  )
  checked_output = _validate_root_publication_audit(
      output, 'analysis_output publication audit', root=_ANALYSIS_DIR,
  )
  for role, audit in (
      ('analysis_attempt', checked_attempt),
      ('analysis_output', checked_output),
  ):
    expected_failure = publication_failure if failure_root == role else None
    if audit['publication_failure'] != expected_failure:
      raise AnalysisError(f'{role} publication failure linkage changed.')
  return checked_attempt, checked_output


def _entry_state_digest(states: Mapping[str, Mapping[str, Any]]) -> str:
  digest = hashlib.sha256()
  for relative in sorted(states):
    payload = json.dumps(
        states[relative], sort_keys=True, separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    digest.update(relative.encode('utf-8'))
    digest.update(b'\0')
    digest.update(hashlib.sha256(payload).digest())
  return digest.hexdigest()


def _analysis_output_state(output_audit: Mapping[str, Any]) -> dict[str, Any]:
  _exact_keys(output_audit, set(PUBLICATION_AUDIT_KEYS), 'analysis output audit')
  if not _ANALYSIS_DIR.exists() and not _ANALYSIS_DIR.is_symlink():
    return {
        'state': 'absent', 'published_prefix': [],
        'published_final_bindings': {}, 'temporary_orphan_bindings': {},
        'durability_uncertain_final_bindings': {},
        'preexisting_entry_states': {}, 'file_tree_sha256': EMPTY_SHA256,
        'entry_state_tree_sha256': EMPTY_SHA256, 'directory_paths': [],
        'directory_tree_sha256': EMPTY_SHA256,
    }
  if _ANALYSIS_DIR.is_symlink() or not _ANALYSIS_DIR.is_dir():
    raise AnalysisError('Failed-analysis output root is unsafe.')
  published = _validate_publication_binding_map(
      output_audit['successful_final_bindings_before_terminal'],
      'analysis output published finals', expected_mode='0400',
  )
  temporary = _validate_publication_binding_map(
      output_audit['temporary_orphan_bindings'],
      'analysis output temporary orphans',
  )
  uncertain = _validate_publication_binding_map(
      output_audit['durability_uncertain_final_bindings'],
      'analysis output uncertain finals',
  )
  preexisting = _validate_entry_state_map(
      output_audit['preexisting_entry_states'],
      'analysis output pre-existing states',
  )
  final_names = set(published)
  if final_names == set():
    prefix = []
  elif final_names == {'RESULT.md'}:
    prefix = ['RESULT.md']
  elif final_names == {'RESULT.md', 'ANALYSIS.json'}:
    prefix = ['RESULT.md', 'ANALYSIS.json']
  else:
    raise AnalysisError('Failed-analysis published-final prefix changed.')
  regular = {**published, **temporary, **uncertain}
  observed_entries = {
      entry.name: _observe_entry_state(entry) for entry in _ANALYSIS_DIR.iterdir()
  }
  expected_names = set(regular) | {
      name for name, state in preexisting.items() if state['state'] == 'present'
  }
  if set(observed_entries) != expected_names:
    raise AnalysisError('Failed-analysis output exact entry membership changed.')
  states = {}
  for relative, binding in regular.items():
    observed = observed_entries[relative]
    if observed['state'] != 'present' or observed['entry_type'] != 'regular':
      raise AnalysisError('Failed-analysis regular publication changed type.')
    expected_binding = {
        key: observed[key] for key in
        ('sha256', 'size_bytes', 'mode', 'st_dev', 'st_ino', 'st_nlink')
    }
    if binding != expected_binding:
      raise AnalysisError('Failed-analysis regular publication bytes changed.')
    states[f'analysis_output/{relative}'] = observed
  for relative, expected in preexisting.items():
    if expected['state'] == 'present':
      if observed_entries[relative] != expected:
        raise AnalysisError('Failed-analysis pre-existing state changed.')
    states[f'analysis_output/{relative}'] = expected
  has_failure_state = bool(
      output_audit.get('publication_failure') is not None
      or temporary or uncertain or preexisting
  )
  directory_digest = hashlib.sha256()
  directory_digest.update(b'.\0')
  directory_digest.update(b'0700')
  return {
      'state': 'publication_failure_prefix' if has_failure_state else 'published_prefix',
      'published_prefix': prefix, 'published_final_bindings': published,
      'temporary_orphan_bindings': temporary,
      'durability_uncertain_final_bindings': uncertain,
      'preexisting_entry_states': preexisting,
      'file_tree_sha256': _binding_map_digest(regular),
      'entry_state_tree_sha256': _entry_state_digest(states),
      'directory_paths': ['.'],
      'directory_tree_sha256': directory_digest.hexdigest(),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--acknowledge-structural-only-v3-3-4-3', action='store_true',
      help='Acknowledge the frozen structural-only/no-science boundary.',
  )
  args = parser.parse_args()
  if not args.acknowledge_structural_only_v3_3_4_3:
    raise AnalysisError('The literal structural-only acknowledgement is required.')
  run_dir = _RUN_DIR.resolve()
  bundle_root = _REPO_ROOT.resolve()
  output_json = (_ANALYSIS_DIR / 'ANALYSIS.json').resolve()
  output_markdown = (_ANALYSIS_DIR / 'RESULT.md').resolve()
  for path in (
      run_dir, bundle_root, output_json, output_markdown,
      _FREEZE_PATH, _AMENDMENT_PATH,
  ):
    _guard_path(path)
  precheck = _analysis_attempt_precheck(run_dir, bundle_root=bundle_root)
  started = _analysis_started_record(precheck)
  _create_append_only_directory(
      _ANALYSIS_ATTEMPT_DIR, root_role='analysis_attempt'
  )
  started_path = _ANALYSIS_ATTEMPT_DIR / 'ANALYSIS_ATTEMPT_STARTED.json'
  _write_json_new(
      started_path, started, root_role='analysis_attempt',
      root=_ANALYSIS_ATTEMPT_DIR, artifact_role='analysis_attempt_start',
  )
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
    for old_path in (_PRIOR_ANALYZER_ATTEMPT_DIR, _PRIOR_ANALYZER_OUTPUT_DIR):
      if old_path.exists() or old_path.is_symlink():
        raise AnalysisError('Original v3.3.3 analyzer destination appeared.')
    def pre_publish_check(label: str) -> None:
      _analysis_toctou_check(
          run_dir=run_dir, started_sha256=started_sha, result=result,
          label=label,
      )

    final_result, _publication_successes = _write_outputs(
        result, output_json=output_json,
        output_markdown=output_markdown,
        pre_publish_check=pre_publish_check,
    )
    del final_result, _publication_successes
    for old_path in (_PRIOR_ANALYZER_ATTEMPT_DIR, _PRIOR_ANALYZER_OUTPUT_DIR):
      if old_path.exists() or old_path.is_symlink():
        raise AnalysisError('Original v3.3.3 analyzer destination appeared.')
    pre_publish_check('before COMPLETE')
    _write_json_new(
        _ANALYSIS_ATTEMPT_DIR / 'ANALYSIS_COMPLETE.json',
        _analysis_complete_record(started_sha),
        root_role='analysis_attempt', root=_ANALYSIS_ATTEMPT_DIR,
        artifact_role='analysis_complete',
    )
  except BaseException as error:
    _write_json_new(
        _ANALYSIS_ATTEMPT_DIR / 'ANALYSIS_FAILURE.json',
        _analysis_failure_record(
            error, started_sha, raw_reached=raw_state['reached']
        ),
        root_role='analysis_attempt', root=_ANALYSIS_ATTEMPT_DIR,
        artifact_role='analysis_failure',
    )
    raise


if __name__ == '__main__':
  main()
