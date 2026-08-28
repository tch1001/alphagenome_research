#!/usr/bin/env python3
"""Standalone CPU-only v3.3.4.5.1 structural archive analyzer.

This module intentionally does not import JAX, AlphaGenome, or model code.  It
audits provenance, the frozen source-program boundary, append-only execution
prefixes, and (when present) all raw structural controls.  Compiled backend
HLO is retained as diagnostic provenance and is never an equality gate.  No
normalization, Shapley value, resolution result, rank, or nomination is
computed here.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import errno as errno_module
from collections import defaultdict
import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import struct
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence


ANALYSIS_VERSION = 'v3.3.4.5.1-structural-analyzer-v1'
ANALYSIS_ATTEMPT_ID = (
    'v3.3.4.5.1-development-ood-sidecar-structural-analysis'
)
ANALYSIS_ACKNOWLEDGEMENT = '--acknowledge-structural-only-v3-3-4-5-1'
ANALYSIS_SCHEMA_VERSION = 'v3.3.4.5.1-analysis-freeze-v1'
ANALYSIS_AMENDMENT_SHA256 = (
    '16af8ccb65f3e08739c3792c5c9ab3affcb19a3ca9993904260729a898afd5c4'
)
ANALYSIS_AMENDMENT_COMMIT = '564a01dc2981d57c8f8298f3efca5b22fcb381e0'
MODEL_SOURCE_COMMIT = '0da8f47ea6e576a72a1cda204ce868ef79cc2ce5'
SCRIPT_VERSION = 'v3.3.4.5'
PREFLIGHT_SCRIPT_VERSION = 'opensplice-device-preflight-v3.3.4.5'
ATTEMPT_ID = 'v3.3.4.5-development-ood-sidecar-one-shot'
AMENDMENT_SHA256 = (
    'e64af0ba8ad6436530a1bd0da2807f3a9bd6ef874306255e1f9683e1731574c8'
)
AMENDMENT_COMMIT = '001b4453833cb3c57991187c96416fccd22e4928'
CONSUMED_V3344_COMMIT = '6858bbcdd869ac9ae93064910227003a911d0bd1'
CONSUMED_V3344_FREEZE_SHA256 = (
    '73b26eddf5578ef0847ac69c279c262e6f43102127bbe4299bbdab7e52227e30'
)
CONSUMED_V3344_PREFIX_SHA256 = (
    'efcb6d8946666d104d7458c0f13cc8f53e6dfaa1a30a2e83744f48641978f3c7'
)
CONSUMED_V3344_TRACEBACK_SHA256 = (
    '03cf721c145a2d70764455c8ab197482aed52a062d10c1ca29818bb0c1c8c3d3'
)
CONSUMED_V3343_COMMIT = 'ea486661ffe64d5640485ebb2a3ca297e128530a'
CONSUMED_V3343_FREEZE_SHA256 = (
    '713790306dd3d88d735229f497587ab6fe611e435eee3f4ef6b862f7baa3cedc'
)
CONSUMED_V3343_CACHE_TREE_SHA256 = (
    '9162636192082efbef80c9b37dd3ebc138aa094f70111874b9dad70e468af1af'
)
CONSUMED_V3343_CACHE_BINDING_SHA256 = (
    'd53f56fabd83cb43b79d7a7c73c5b56727d846713e96ea73ef9b26360e18bdea'
)
CONSUMED_V3343_TRACEBACK_SHA256 = (
    '8b6d0f7575adfc66032ba56d3ab5373f05b0bb1e85b2e39c94e2a82a356e39e9'
)
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
PUBLICATION_SCHEMA_VERSION = 'v3.3.4.5-named-temp-renameat2-noreplace-v1'
_V33451_PUBLICATION_SCHEMA_VERSION = (
    'v3.3.4.5.1-named-temp-renameat2-noreplace-v1'
)
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
    'root_parent_open', 'root_parent_validation',
    'root_final_preexistence', 'root_mkdir', 'root_parent_fsync',
    'root_revalidation', 'parent_open', 'parent_validation',
    'final_preexistence', 'temp_open', 'temp_write',
    'file_fsync_before_rename', 'fchmod', 'file_fsync_after_fchmod',
    'readback', 'rename_noreplace', 'parent_fsync',
    'post_publish_revalidation',
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
    'prior_v3_3_4_3_consumed_preflight_prefix',
    'prior_v3_3_4_3_consumed_preflight_prefix_content_binding',
    'prior_v3_3_4_4_consumed_preflight_prefix',
    'prior_v3_3_4_4_consumed_preflight_prefix_content_binding',
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
    'prior_v3_3_4_3_consumed_preflight_prefix',
    'prior_v3_3_4_3_consumed_preflight_prefix_content_binding',
    'prior_v3_3_4_4_consumed_preflight_prefix',
    'prior_v3_3_4_4_consumed_preflight_prefix_content_binding',
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
    'prior_v3_3_4_3_consumed_preflight_prefix',
    'prior_v3_3_4_3_consumed_preflight_prefix_content_binding',
    'prior_v3_3_4_4_consumed_preflight_prefix',
    'prior_v3_3_4_4_consumed_preflight_prefix_content_binding',
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
    / 'encoder_skip_ood_sidecar_preflight_phase_amendment_v3_3_4_5.md'
)
_CONSUMED_V3344_FREEZE_PATH = (
    _HERE / 'encoder_skip_ood_sidecar_v3_3_4_4_freeze.json'
)
_CONSUMED_V3344_PREFLIGHT_DIR = (
    _HERE / 'results/v3_3_4_4_device_preflight'
)
_CONSUMED_V3344_CACHE_DIR = (
    _HERE / 'results/v3_3_4_4_preflight_kernel_cache'
)
_CONSUMED_V3344_LAUNCHER_PATH = (
    _HERE / 'launch_encoder_skip_ood_sidecar_v3_3_4_4.py'
)
_CONSUMED_V3344_BOOTSTRAP_PATH = (
    _HERE / 'validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_4.py'
)
_CONSUMED_V3343_FREEZE_PATH = (
    _HERE / 'encoder_skip_ood_sidecar_v3_3_4_3_freeze.json'
)
_CONSUMED_V3343_CACHE_DIR = (
    _HERE / 'results/v3_3_4_3_preflight_kernel_cache'
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
_FREEZE_PATH = _HERE / 'encoder_skip_ood_sidecar_v3_3_4_5_freeze.json'
_ANALYSIS_AMENDMENT_PATH = (
    _HERE / 'v3_wider_mechanism'
    / 'encoder_skip_ood_sidecar_analysis_amendment_v3_3_4_5_1.md'
)
_ANALYSIS_FREEZE_PATH = (
    _HERE / 'encoder_skip_ood_sidecar_analysis_v3_3_4_5_1_freeze.json'
)
_RUN_DIR = _HERE / 'results/v3_3_4_5_development_ood_sidecar_one_shot'
_OLD_V3345_ANALYSIS_DIR = (
    _HERE / 'results/v3_3_4_5_development_ood_sidecar_analysis'
)
_OLD_V3345_ANALYSIS_ATTEMPT_DIR = (
    _HERE / 'results/v3_3_4_5_development_ood_sidecar_analysis_attempt'
)
_ANALYSIS_DIR = (
    _HERE
    / 'results/v3_3_4_5_development_ood_sidecar_analysis_v3_3_4_5_1'
)
_ANALYSIS_ATTEMPT_DIR = (
    _HERE
    / 'results/v3_3_4_5_development_ood_sidecar_analysis_v3_3_4_5_1_attempt'
)
_PREFLIGHT_DIR = _HERE / 'results/v3_3_4_5_device_preflight'
_PREFLIGHT_CACHE_DIR = _HERE / 'results/v3_3_4_5_preflight_kernel_cache'
_MODEL_CACHE_DIR = _HERE / 'results/v3_3_4_5_model_kernel_cache'
_TEST_PATH = _HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_4_5_1_test.py'
_SHELL_PATH = _HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_4_5_1.sh'
_GENERATOR_PATH = (
    _HERE / 'generate_encoder_skip_ood_sidecar_analysis_v3_3_4_5_1_freeze.py'
)
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

_CONSUMED_V3343_OTHER_ABSENT_PATHS = {
    **{
        f'v3_3_4{suffix}.{role}': _HERE / 'results' / name
        for suffix in ('', '_1', '_2')
        for role, name in (
            ('device_preflight', f'v3_3_4{suffix}_device_preflight'),
            ('external_cache', f'v3_3_4{suffix}_preflight_kernel_cache'),
            ('model_cache', f'v3_3_4{suffix}_model_kernel_cache'),
            ('model_run', f'v3_3_4{suffix}_development_ood_sidecar_one_shot'),
            ('analysis_attempt', f'v3_3_4{suffix}_development_ood_sidecar_analysis_attempt'),
            ('analysis_output', f'v3_3_4{suffix}_development_ood_sidecar_analysis'),
        )
    },
    'v3_3_4_3.device_preflight': _HERE / 'results/v3_3_4_3_device_preflight',
    'v3_3_4_3.model_cache': _HERE / 'results/v3_3_4_3_model_kernel_cache',
    'v3_3_4_3.model_run': _HERE / 'results/v3_3_4_3_development_ood_sidecar_one_shot',
    'v3_3_4_3.analysis_attempt': _HERE / 'results/v3_3_4_3_development_ood_sidecar_analysis_attempt',
    'v3_3_4_3.analysis_output': _HERE / 'results/v3_3_4_3_development_ood_sidecar_analysis',
}

_CONSUMED_V3344_OTHER_ABSENT_PATHS = {
    'analysis_attempt': (
        _HERE / 'results/v3_3_4_4_development_ood_sidecar_analysis_attempt'
    ),
    'analysis_output': (
        _HERE / 'results/v3_3_4_4_development_ood_sidecar_analysis'
    ),
    'model_cache': _HERE / 'results/v3_3_4_4_model_kernel_cache',
    'model_run': (
        _HERE / 'results/v3_3_4_4_development_ood_sidecar_one_shot'
    ),
}

_V3345_SOURCE_PATHS = (
    'experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_5.py',
    'experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_5.sh',
    'experiments/interpretability/opensplice/analyze_encoder_skip_ood_sidecar_v3_3_4_5_test.py',
    'experiments/interpretability/opensplice/generate_encoder_skip_ood_sidecar_v3_3_4_5_freeze.py',
    'experiments/interpretability/opensplice/launch_encoder_skip_ood_sidecar_v3_3_4_5.py',
    'experiments/interpretability/opensplice/run_device_preflight_v3_3_4_5.py',
    'experiments/interpretability/opensplice/run_device_preflight_v3_3_4_5_test.py',
    'experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_4_5.py',
    'experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_4_5.sh',
    'experiments/interpretability/opensplice/run_encoder_skip_ood_sidecar_v3_3_4_5_test.py',
    'experiments/interpretability/opensplice/v3_wider_mechanism/encoder_skip_ood_sidecar_preflight_phase_amendment_v3_3_4_5.md',
    'experiments/interpretability/opensplice/validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_5.py',
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
PRIOR_CACHE_FILE_TREE_SHA256 = (
    'd1d11bc6dc48b302cf675fb48727bd6ededec09142429eaa9e368f7631463717'
)
_PRIOR_CACHE_LSTAT_ROWS = (
    ('.', 'directory', '0700', 4096, 66307, 140791354, 4, None),
    ('triton', 'directory', '0700', 4096, 66307, 140791357, 2, None),
    ('xdg', 'directory', '0700', 4096, 66307, 140791358, 3, None),
    ('xdg/matplotlib', 'directory', '0700', 4096, 66307, 140791359, 2, None),
    (
        'xdg/matplotlib/fontlist-v3.11.0.json', 'regular', '0600',
        163_240, 66307, 140791361, 1,
        'a777469f8f54be8cc9107788bce2b3cd23709aa317114392cdc365b3fb127125',
    ),
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


class PublicationError(RuntimeError):
  """A non-retriable failure of the local append-only publisher."""

  def __init__(self, message: str, publication_failure: Mapping[str, Any]):
    super().__init__(message)
    self.publication_failure = dict(publication_failure)


_PUBLICATION_ROOTS = {
    'analysis_attempt': _ANALYSIS_ATTEMPT_DIR,
    'analysis_output': _ANALYSIS_DIR,
}
_PUBLICATION_DIRECTORIES: dict[str, tuple[int, int, int]] = {}
_PUBLICATION_SUCCESSES: dict[str, dict[str, dict[str, Any]]] = {}
_PUBLICATION_TEMP_ORPHANS: dict[str, dict[str, dict[str, Any]]] = {}
_PUBLICATION_UNCERTAIN_FINALS: dict[str, dict[str, dict[str, Any]]] = {}
_PUBLICATION_PREEXISTING: dict[str, dict[str, dict[str, Any]]] = {}
_PUBLICATION_FAILURE: dict[str, dict[str, Any]] = {}
_PUBLICATION_FAILURE_TERMINAL_USED: set[str] = set()
_PUBLICATION_UNBINDABLE_ROOTS: set[str] = set()
_PUBLICATION_ORDINAL = 0
_PUBLICATION_TEST_FAIL_STAGE: str | None = None


def _publication_mode(mode: int) -> str:
  return f'{stat.S_IMODE(mode):04o}'


def _absent_publication_entry() -> dict[str, Any]:
  return {
      'state': 'absent', 'entry_type': None, 'mode': None,
      'size_bytes': None, 'sha256': None, 'st_dev': None,
      'st_ino': None, 'st_nlink': None,
  }


def _publication_entry(path: Path) -> dict[str, Any]:
  try:
    observed = path.lstat()
  except FileNotFoundError:
    return _absent_publication_entry()
  mode = observed.st_mode
  if stat.S_ISREG(mode):
    entry_type = 'regular'
    sha256 = _sha256_no_follow(path, observed)
  elif stat.S_ISDIR(mode):
    entry_type = 'directory'
    sha256 = None
  elif stat.S_ISLNK(mode):
    entry_type = 'symlink'
    sha256 = None
  elif stat.S_ISFIFO(mode):
    entry_type = 'fifo'
    sha256 = None
  elif stat.S_ISSOCK(mode):
    entry_type = 'socket'
    sha256 = None
  elif stat.S_ISCHR(mode):
    entry_type = 'character'
    sha256 = None
  elif stat.S_ISBLK(mode):
    entry_type = 'block'
    sha256 = None
  else:
    entry_type = 'other'
    sha256 = None
  return {
      'state': 'present', 'entry_type': entry_type,
      'mode': _publication_mode(mode),
      'size_bytes': observed.st_size if entry_type == 'regular' else None,
      'sha256': sha256, 'st_dev': observed.st_dev,
      'st_ino': observed.st_ino, 'st_nlink': observed.st_nlink,
  }


def _publication_entry_at(directory_fd: int, relative: str) -> dict[str, Any]:
  """Observes one entry relative to an already bound no-follow directory."""
  try:
    observed = os.stat(relative, dir_fd=directory_fd, follow_symlinks=False)
  except FileNotFoundError:
    return _absent_publication_entry()
  mode = observed.st_mode
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
  sha256 = None
  size_bytes = None
  if entry_type == 'regular':
    fd = os.open(
        relative, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=directory_fd,
    )
    try:
      before = os.fstat(fd)
      if (
          (before.st_dev, before.st_ino, before.st_nlink, before.st_mode,
           before.st_size)
          != (observed.st_dev, observed.st_ino, observed.st_nlink,
              observed.st_mode, observed.st_size)
      ):
        raise AnalysisError('Publication entry changed before fd read.')
      digest = hashlib.sha256()
      for block in iter(lambda: os.read(fd, 1024 * 1024), b''):
        digest.update(block)
      after = os.fstat(fd)
      if (
          (after.st_dev, after.st_ino, after.st_nlink, after.st_mode,
           after.st_size)
          != (before.st_dev, before.st_ino, before.st_nlink, before.st_mode,
              before.st_size)
      ):
        raise AnalysisError('Publication entry changed during fd read.')
      try:
        final_path = os.stat(
            relative, dir_fd=directory_fd, follow_symlinks=False
        )
      except FileNotFoundError as error:
        raise AnalysisError(
            'Publication entry pathname disappeared during fd read.'
        ) from error
      if (
          final_path.st_dev, final_path.st_ino, final_path.st_nlink,
          final_path.st_mode, final_path.st_size,
      ) != (
          before.st_dev, before.st_ino, before.st_nlink, before.st_mode,
          before.st_size,
      ):
        raise AnalysisError('Publication entry pathname changed during fd read.')
      sha256 = digest.hexdigest()
      size_bytes = before.st_size
    finally:
      os.close(fd)
  else:
    try:
      final_path = os.stat(
          relative, dir_fd=directory_fd, follow_symlinks=False
      )
    except FileNotFoundError as error:
      raise AnalysisError('Publication entry pathname disappeared.') from error
    if (
        final_path.st_dev, final_path.st_ino, final_path.st_nlink,
        final_path.st_mode,
    ) != (
        observed.st_dev, observed.st_ino, observed.st_nlink, observed.st_mode,
    ):
      raise AnalysisError('Publication entry pathname changed during observation.')
  return {
      'state': 'present', 'entry_type': entry_type,
      'mode': _publication_mode(mode), 'size_bytes': size_bytes,
      'sha256': sha256, 'st_dev': observed.st_dev,
      'st_ino': observed.st_ino, 'st_nlink': observed.st_nlink,
  }


def _publication_binding_from_state(value: Mapping[str, Any]) -> dict[str, Any]:
  if value.get('state') != 'present' or value.get('entry_type') != 'regular':
    raise AnalysisError('Publication entry is not a regular-file binding.')
  return {
      key: value[key] for key in
      ('sha256', 'size_bytes', 'mode', 'st_dev', 'st_ino', 'st_nlink')
  }


def _publication_failure_object(
    *, root_role: str, artifact_role: str, final_relative_path: str,
    temp_relative_path: str | None, publication_ordinal: int | None,
    stage: str, error: BaseException, rename_attempted: bool,
    rename_succeeded: bool, parent_fsync_attempted: bool,
    parent_fsync_succeeded: bool,
) -> dict[str, Any]:
  root = _PUBLICATION_ROOTS[root_role]
  registered = _PUBLICATION_DIRECTORIES.get(root_role)
  root_fd = None if registered is None else registered[0]
  if stage.startswith('root_'):
    # The root itself may be a preexisting non-directory or symlink.  Root
    # allocation failures bind that root separately in output_state and must
    # never probe a planned child through it.
    temp_state = _absent_publication_entry()
    final_state = _absent_publication_entry()
  else:
    temp_state = (
        _publication_entry(root / temp_relative_path)
        if root_fd is None else _publication_entry_at(root_fd, temp_relative_path)
    )
    final_state = (
        _publication_entry(root / final_relative_path)
        if root_fd is None else _publication_entry_at(root_fd, final_relative_path)
    )
  errno_value = getattr(error, 'errno', None)
  if isinstance(errno_value, bool) or not isinstance(errno_value, int):
    errno_value = None
  result = {
      'schema_version': _V33451_PUBLICATION_SCHEMA_VERSION,
      'method': PUBLICATION_METHOD, 'root_role': root_role,
      'artifact_role': artifact_role,
      'final_relative_path': final_relative_path,
      'temp_relative_path': temp_relative_path,
      'publication_ordinal': publication_ordinal,
      'runner_pid': os.getpid(), 'failure_stage': stage,
      'errno': errno_value, 'error_type': type(error).__name__,
      'message': str(error),
      'rename_noreplace_attempted': rename_attempted,
      'rename_noreplace_succeeded': rename_succeeded,
      'parent_fsync_attempted': parent_fsync_attempted,
      'parent_fsync_succeeded': parent_fsync_succeeded,
      'temp_state': temp_state, 'final_state': final_state,
      'created_at_unix_s': time.time(),
  }
  _exact_keys(result, set(PUBLICATION_FAILURE_KEYS), 'publication failure')
  return result


def _v33451_validate_publication_failure(
    value: Any, label: str,
) -> dict[str, Any]:
  node = _exact_keys(value, set(PUBLICATION_FAILURE_KEYS), label)
  stage = node.get('failure_stage')
  root_stage = isinstance(stage, str) and stage.startswith('root_')
  temp = node.get('temp_relative_path')
  ordinal = node.get('publication_ordinal')
  if (
      node.get('schema_version') != _V33451_PUBLICATION_SCHEMA_VERSION
      or node.get('method') != PUBLICATION_METHOD
      or node.get('root_role') not in _PUBLICATION_ROOTS
      or stage not in PUBLICATION_FAILURE_STAGES
      or not isinstance(node.get('artifact_role'), str)
      or not node['artifact_role']
      or not isinstance(node.get('final_relative_path'), str)
      or '/' in node['final_relative_path']
      or node['final_relative_path'] in {'', '.', '..'}
      or isinstance(node.get('runner_pid'), bool)
      or not isinstance(node.get('runner_pid'), int)
      or node['runner_pid'] < 1
      or not isinstance(node.get('error_type'), str)
      or not node['error_type']
      or not isinstance(node.get('message'), str)
  ):
    raise AnalysisError(f'{label} contract changed.')
  _finite(node.get('created_at_unix_s'), f'{label}.created_at_unix_s')
  if root_stage:
    if temp is not None or ordinal is not None:
      raise AnalysisError(f'{label} root-stage nullability changed.')
  elif (
      not isinstance(temp, str)
      or isinstance(ordinal, bool) or not isinstance(ordinal, int)
      or ordinal < 0 or ordinal >= 1_000_000
      or re.fullmatch(
          rf'\.v33451\.tmp\.{node["runner_pid"]}\.{ordinal:06d}\.[0-9a-f]{{32}}',
          temp,
      ) is None
  ):
    raise AnalysisError(f'{label} temp/ordinal changed.')
  if node.get('errno') is not None and (
      isinstance(node['errno'], bool) or not isinstance(node['errno'], int)
      or node['errno'] < 0
  ):
    raise AnalysisError(f'{label} errno changed.')
  for name in (
      'rename_noreplace_attempted', 'rename_noreplace_succeeded',
      'parent_fsync_attempted', 'parent_fsync_succeeded',
  ):
    if not isinstance(node.get(name), bool):
      raise AnalysisError(f'{label}.{name} changed.')
  if (
      node['rename_noreplace_succeeded']
      and not node['rename_noreplace_attempted']
      or node['parent_fsync_succeeded'] and not node['parent_fsync_attempted']
  ):
    raise AnalysisError(f'{label} operation flag ordering changed.')
  expected_flags = {
      **{
          name: (False, False, False, False)
          for name in (
              'root_parent_open', 'root_parent_validation',
              'root_final_preexistence', 'root_mkdir', 'parent_open',
              'parent_validation', 'final_preexistence', 'temp_open',
              'temp_write', 'file_fsync_before_rename', 'fchmod',
              'file_fsync_after_fchmod', 'readback',
          )
      },
      'root_parent_fsync': (False, False, True, False),
      'root_revalidation': (False, False, True, True),
      'rename_noreplace': (True, False, False, False),
      'parent_fsync': (True, True, True, False),
      'post_publish_revalidation': (True, True, True, True),
  }
  observed_flags = tuple(node[name] for name in (
      'rename_noreplace_attempted', 'rename_noreplace_succeeded',
      'parent_fsync_attempted', 'parent_fsync_succeeded',
  ))
  if observed_flags != expected_flags[stage]:
    raise AnalysisError(f'{label} stage/operation flags changed.')
  _v33451_validate_entry_state(node.get('temp_state'), f'{label}.temp_state')
  _v33451_validate_entry_state(node.get('final_state'), f'{label}.final_state')
  if root_stage and (
      node['temp_state'] != _absent_publication_entry()
      or node['final_state'] != _absent_publication_entry()
  ):
    raise AnalysisError(f'{label} root-stage entry states changed.')
  return dict(node)


def _v33451_validate_entry_state(value: Any, label: str) -> dict[str, Any]:
  node = _validate_entry_state(value, label)
  if node.get('state') not in {'absent', 'present'}:
    raise AnalysisError(f'{label}.state is not frozen for v3.3.4.5.1.')
  return node


def _injected_publication_failure(stage: str) -> None:
  if _PUBLICATION_TEST_FAIL_STAGE == stage:
    raise OSError(errno_module.EIO, f'injected publication failure at {stage}')


def ensure_publication_directory(
    root_role: str, first_final_relative_path: str,
    first_artifact_role: str,
) -> Path:
  """Allocates exactly one fresh mode-0700 root without following links."""
  if root_role not in _PUBLICATION_ROOTS:
    raise AnalysisError('Unknown publication root role.')
  if root_role in _PUBLICATION_DIRECTORIES:
    raise AnalysisError('Publication root was registered more than once.')
  root = _PUBLICATION_ROOTS[root_role]
  parent = root.parent
  stage = 'root_parent_open'
  parent_fd = -1
  parent_fsync_attempted = False
  parent_fsync_succeeded = False
  root_created = False
  created_root_identity: tuple[int, int] | None = None
  validated_parent_identity: tuple[int, int] | None = None
  try:
    _injected_publication_failure(stage)
    parent_fd = os.open(
        parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    stage = 'root_parent_validation'
    _injected_publication_failure(stage)
    parent_status = os.fstat(parent_fd)
    parent_path_status = parent.lstat()
    if (
        not stat.S_ISDIR(parent_status.st_mode)
        or stat.S_ISLNK(parent_path_status.st_mode)
        or (parent_path_status.st_dev, parent_path_status.st_ino)
        != (parent_status.st_dev, parent_status.st_ino)
    ):
      raise AnalysisError('Publication parent is not a directory.')
    validated_parent_identity = (
        parent_status.st_dev, parent_status.st_ino
    )
    stage = 'root_final_preexistence'
    _injected_publication_failure(stage)
    if _publication_entry_at(parent_fd, root.name)['state'] != 'absent':
      raise FileExistsError(errno_module.EEXIST, 'Publication root exists.')
    stage = 'root_mkdir'
    _injected_publication_failure(stage)
    os.mkdir(root.name, 0o700, dir_fd=parent_fd)
    root_created = True
    created_root_status = os.stat(
        root.name, dir_fd=parent_fd, follow_symlinks=False
    )
    created_root_identity = (
        created_root_status.st_dev, created_root_status.st_ino
    )
    stage = 'root_parent_fsync'
    parent_fsync_attempted = True
    _injected_publication_failure(stage)
    os.fsync(parent_fd)
    parent_fsync_succeeded = True
    stage = 'root_revalidation'
    _injected_publication_failure(stage)
    root_fd = os.open(
        root.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=parent_fd,
    )
    observed = os.fstat(root_fd)
    live_root = root.lstat()
    live_parent = parent.lstat()
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o700
        or os.listdir(root_fd)
        or (live_root.st_dev, live_root.st_ino)
        != (observed.st_dev, observed.st_ino)
        or (live_parent.st_dev, live_parent.st_ino)
        != (parent_status.st_dev, parent_status.st_ino)
    ):
      os.close(root_fd)
      raise AnalysisError('Fresh publication root validation failed.')
    _PUBLICATION_DIRECTORIES[root_role] = (
        root_fd, observed.st_dev, observed.st_ino
    )
    return root
  except BaseException as error:
    if root_created:
      fixed_namespace_exact = True
      try:
        current_root_status = os.stat(
            root.name, dir_fd=parent_fd, follow_symlinks=False
        )
      except OSError:
        fixed_namespace_exact = False
      else:
        if (
            current_root_status.st_dev, current_root_status.st_ino
        ) != created_root_identity:
          fixed_namespace_exact = False
      try:
        fixed_parent_status = parent.lstat()
        fixed_root_status = root.lstat()
      except OSError:
        fixed_namespace_exact = False
      else:
        if (
            stat.S_ISLNK(fixed_parent_status.st_mode)
            or not stat.S_ISDIR(fixed_parent_status.st_mode)
            or (fixed_parent_status.st_dev, fixed_parent_status.st_ino)
            != validated_parent_identity
            or stat.S_ISLNK(fixed_root_status.st_mode)
            or not stat.S_ISDIR(fixed_root_status.st_mode)
            or (fixed_root_status.st_dev, fixed_root_status.st_ino)
            != created_root_identity
        ):
          fixed_namespace_exact = False
      if not fixed_namespace_exact:
        _PUBLICATION_UNBINDABLE_ROOTS.add(root_role)
    failure = _publication_failure_object(
        root_role=root_role, artifact_role=first_artifact_role,
        final_relative_path=first_final_relative_path,
        temp_relative_path=None, publication_ordinal=None, stage=stage,
        error=error, rename_attempted=False, rename_succeeded=False,
        parent_fsync_attempted=parent_fsync_attempted,
        parent_fsync_succeeded=parent_fsync_succeeded,
    )
    _PUBLICATION_FAILURE[root_role] = failure
    raise PublicationError(str(error), failure) from error
  finally:
    if parent_fd >= 0:
      os.close(parent_fd)


def _rename_noreplace(parent_fd: int, old_name: str, new_name: str) -> None:
  libc = ctypes.CDLL(None, use_errno=True)
  renameat2 = getattr(libc, 'renameat2', None)
  if renameat2 is None:
    raise OSError(errno_module.ENOSYS, 'renameat2 is unavailable')
  renameat2.argtypes = (
      ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
      ctypes.c_uint,
  )
  renameat2.restype = ctypes.c_int
  if renameat2(
      parent_fd, old_name.encode(), parent_fd, new_name.encode(), 1
  ) != 0:
    value = ctypes.get_errno()
    raise OSError(value, os.strerror(value))


def publish_bytes(
    root_role: str, final_relative_path: str, payload: bytes,
    artifact_role: str,
) -> dict[str, Any]:
  """Publishes one immutable file with named-temp renameat2(NOREPLACE)."""
  global _PUBLICATION_ORDINAL  # pylint: disable=global-statement
  if root_role not in _PUBLICATION_DIRECTORIES:
    raise AnalysisError('Publication root is not registered.')
  if _PUBLICATION_FAILURE.get(root_role) is not None:
    prior_failure = _PUBLICATION_FAILURE[root_role]
    if (
        root_role != 'analysis_attempt' or artifact_role != 'analysis_failure'
        or prior_failure.get('artifact_role') == 'analysis_failure'
        or root_role in _PUBLICATION_FAILURE_TERMINAL_USED
    ):
      raise AnalysisError('Publication root is already terminally consumed.')
    _PUBLICATION_FAILURE_TERMINAL_USED.add(root_role)
  relative = _validate_relative_path(
      final_relative_path, 'publication final relative path'
  )
  if '/' in relative:
    raise AnalysisError('Analysis publications must be root-level files.')
  root = _PUBLICATION_ROOTS[root_role]
  root_fd, expected_dev, expected_ino = _PUBLICATION_DIRECTORIES[root_role]
  ordinal = _PUBLICATION_ORDINAL
  _PUBLICATION_ORDINAL += 1
  nonce = secrets.token_hex(16)
  temp = f'.v33451.tmp.{os.getpid()}.{ordinal:06d}.{nonce}'
  stage = 'parent_open'
  fd = -1
  rename_attempted = False
  rename_succeeded = False
  parent_fsync_attempted = False
  parent_fsync_succeeded = False
  created_temp = False
  try:
    _injected_publication_failure(stage)
    opened = os.dup(root_fd)
    try:
      stage = 'parent_validation'
      _injected_publication_failure(stage)
      observed_root = os.fstat(opened)
      try:
        live_root = root.lstat()
      except FileNotFoundError as error:
        raise AnalysisError('Publication root pathname disappeared.') from error
      if (
          not stat.S_ISDIR(observed_root.st_mode)
          or stat.S_IMODE(observed_root.st_mode) != 0o700
          or (observed_root.st_dev, observed_root.st_ino)
          != (expected_dev, expected_ino)
          or stat.S_ISLNK(live_root.st_mode)
          or not stat.S_ISDIR(live_root.st_mode)
          or (live_root.st_dev, live_root.st_ino)
          != (expected_dev, expected_ino)
      ):
        raise AnalysisError('Publication root inode changed.')
      stage = 'final_preexistence'
      _injected_publication_failure(stage)
      if _publication_entry_at(opened, relative)['state'] != 'absent':
        raise FileExistsError(errno_module.EEXIST, 'Publication final exists.')
      stage = 'temp_open'
      _injected_publication_failure(stage)
      fd = os.open(
          temp,
          os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
          0o600, dir_fd=opened,
      )
      created_temp = True
      initial = os.fstat(fd)
      if (
          not stat.S_ISREG(initial.st_mode)
          or stat.S_IMODE(initial.st_mode) != 0o600
          or initial.st_nlink != 1 or initial.st_size != 0
      ):
        raise AnalysisError('Initial publication temp state changed.')
      stage = 'temp_write'
      _injected_publication_failure(stage)
      view = memoryview(payload)
      while view:
        written = os.write(fd, view)
        if written <= 0:
          raise OSError(errno_module.EIO, 'short publication write')
        view = view[written:]
      stage = 'file_fsync_before_rename'
      _injected_publication_failure(stage)
      os.fsync(fd)
      stage = 'fchmod'
      _injected_publication_failure(stage)
      os.fchmod(fd, 0o400)
      stage = 'file_fsync_after_fchmod'
      _injected_publication_failure(stage)
      os.fsync(fd)
      stage = 'readback'
      _injected_publication_failure(stage)
      os.lseek(fd, 0, os.SEEK_SET)
      readback = bytearray()
      while len(readback) < len(payload):
        block = os.read(fd, min(1024 * 1024, len(payload) - len(readback)))
        if not block:
          break
        readback.extend(block)
      if bytes(readback) != payload:
        raise OSError(errno_module.EIO, 'publication readback changed')
      stage = 'rename_noreplace'
      rename_attempted = True
      _injected_publication_failure(stage)
      _rename_noreplace(opened, temp, relative)
      rename_succeeded = True
      stage = 'parent_fsync'
      parent_fsync_attempted = True
      _injected_publication_failure(stage)
      os.fsync(opened)
      parent_fsync_succeeded = True
      stage = 'post_publish_revalidation'
      _injected_publication_failure(stage)
      fd_status = os.fstat(fd)
      final_state = _publication_entry_at(opened, relative)
      final_status = os.stat(relative, dir_fd=opened, follow_symlinks=False)
      temp_state = _publication_entry_at(opened, temp)
      live_root = root.lstat()
      final_identity = (
          final_status.st_dev, final_status.st_ino, final_status.st_nlink,
          _publication_mode(final_status.st_mode), final_status.st_size,
      )
      if (
          (fd_status.st_dev, fd_status.st_ino, fd_status.st_nlink,
           _publication_mode(fd_status.st_mode), fd_status.st_size)
          != final_identity
          or not stat.S_ISREG(final_status.st_mode)
          or stat.S_IMODE(final_status.st_mode) != 0o400
          or final_status.st_nlink != 1
          or final_status.st_size != len(payload)
          or final_state['state'] != 'present'
          or final_state['entry_type'] != 'regular'
          or (
              final_state['st_dev'], final_state['st_ino'],
              final_state['st_nlink'], final_state['mode'],
              final_state['size_bytes'],
          ) != final_identity
          or final_state['sha256'] != hashlib.sha256(payload).hexdigest()
          or temp_state['state'] != 'absent'
          or stat.S_ISLNK(live_root.st_mode)
          or not stat.S_ISDIR(live_root.st_mode)
          or (live_root.st_dev, live_root.st_ino)
          != (expected_dev, expected_ino)
      ):
        raise AnalysisError('Post-publication revalidation failed.')
      success = {
          'schema_version': _V33451_PUBLICATION_SCHEMA_VERSION,
          'method': PUBLICATION_METHOD, 'root_role': root_role,
          'final_relative_path': relative, 'temp_basename': temp,
          'publication_ordinal': ordinal, 'runner_pid': os.getpid(),
          'nonce_hex': nonce, 'sha256': hashlib.sha256(payload).hexdigest(),
          'size_bytes': len(payload), 'mode': '0400',
          'st_dev': final_status.st_dev, 'st_ino': final_status.st_ino,
          'st_nlink': final_status.st_nlink,
          'file_fsync_before_rename': True,
          'file_fsync_after_fchmod': True,
          'rename_noreplace_succeeded': True,
          'parent_fsync_succeeded': True,
          'post_publish_revalidation_exact': True,
      }
      _exact_keys(success, set(PUBLICATION_SUCCESS_KEYS), 'publication success')
      _PUBLICATION_SUCCESSES.setdefault(root_role, {})[relative] = (
          _publication_binding_from_state(final_state)
      )
      return success
    finally:
      os.close(opened)
  except BaseException as error:
    failure = _publication_failure_object(
        root_role=root_role, artifact_role=artifact_role,
        final_relative_path=relative, temp_relative_path=temp,
        publication_ordinal=ordinal, stage=stage, error=error,
        rename_attempted=rename_attempted, rename_succeeded=rename_succeeded,
        parent_fsync_attempted=parent_fsync_attempted,
        parent_fsync_succeeded=parent_fsync_succeeded,
    )
    temp_state = failure['temp_state']
    final_state = failure['final_state']
    if (
        created_temp and not rename_succeeded and temp_state['state'] == 'absent'
        or rename_succeeded and final_state['state'] == 'absent'
    ):
      # The invocation-created entry was removed before it could be bound.
      # No audit may claim the append-only/no-deletion predicates after that
      # namespace loss, so this root has no serializable terminal archive.
      _PUBLICATION_UNBINDABLE_ROOTS.add(root_role)
    if rename_succeeded and final_state['state'] == 'present':
      _PUBLICATION_UNCERTAIN_FINALS.setdefault(root_role, {})[relative] = (
          _publication_binding_from_state(final_state)
      )
      if temp_state['state'] == 'present':
        _PUBLICATION_PREEXISTING.setdefault(root_role, {})[temp] = temp_state
    elif created_temp and temp_state['state'] == 'present':
      _PUBLICATION_TEMP_ORPHANS.setdefault(root_role, {})[temp] = (
          _publication_binding_from_state(temp_state)
      )
      if final_state['state'] == 'present':
        _PUBLICATION_PREEXISTING.setdefault(root_role, {})[relative] = final_state
    else:
      for name, state_value in ((temp, temp_state), (relative, final_state)):
        if state_value['state'] == 'present':
          _PUBLICATION_PREEXISTING.setdefault(root_role, {})[name] = state_value
    _PUBLICATION_FAILURE[root_role] = failure
    raise PublicationError(str(error), failure) from error
  finally:
    if fd >= 0:
      os.close(fd)


def publication_audit(
    root_role: str, publication_failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  if root_role in _PUBLICATION_UNBINDABLE_ROOTS:
    raise AnalysisError('Publication root lost an invocation-created entry.')
  success = dict(sorted(_PUBLICATION_SUCCESSES.get(root_role, {}).items()))
  temporary = dict(sorted(_PUBLICATION_TEMP_ORPHANS.get(root_role, {}).items()))
  uncertain = dict(sorted(_PUBLICATION_UNCERTAIN_FINALS.get(root_role, {}).items()))
  preexisting = dict(sorted(_PUBLICATION_PREEXISTING.get(root_role, {}).items()))
  return {
      'schema_version': _V33451_PUBLICATION_SCHEMA_VERSION,
      'method': PUBLICATION_METHOD,
      'successful_final_count_before_terminal': len(success),
      'successful_final_bindings_before_terminal': success,
      'temporary_orphan_count': len(temporary),
      'temporary_orphan_bindings': temporary,
      'durability_uncertain_final_count': len(uncertain),
      'durability_uncertain_final_bindings': uncertain,
      'preexisting_entry_count': len(preexisting),
      'preexisting_entry_states': preexisting,
      'publication_failure': (
          None if publication_failure is None else dict(publication_failure)
      ),
      'no_new_entry_failure': bool(
          publication_failure is not None and not temporary and not uncertain
      ),
      'no_publication_retry': True,
      'no_published_final_deleted': True,
      'no_temp_or_final_reused': True,
  }


def _sha256(path: Path) -> str:
  return hashlib.sha256(
      _read_bytes_no_follow(path, f'hash target {path}')
  ).hexdigest()


def _sha256_no_follow(path: Path, expected: os.stat_result | None = None) -> str:
  fd = os.open(
      path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
  )
  try:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
      raise AnalysisError(f'No-follow hash target is not regular: {path}.')
    if expected is not None and (
        before.st_dev, before.st_ino, before.st_nlink, before.st_mode,
        before.st_size,
    ) != (
        expected.st_dev, expected.st_ino, expected.st_nlink, expected.st_mode,
        expected.st_size,
    ):
      raise AnalysisError(f'No-follow hash target inode changed: {path}.')
    digest = hashlib.sha256()
    for block in iter(lambda: os.read(fd, 1024 * 1024), b''):
      digest.update(block)
    after = os.fstat(fd)
    if (
        after.st_dev, after.st_ino, after.st_nlink, after.st_mode,
        after.st_size,
    ) != (
        before.st_dev, before.st_ino, before.st_nlink, before.st_mode,
        before.st_size,
    ):
      raise AnalysisError(f'No-follow hash target changed during read: {path}.')
    try:
      final_path = path.lstat()
    except FileNotFoundError as error:
      raise AnalysisError(
          f'No-follow hash target pathname disappeared: {path}.'
      ) from error
    if (
        final_path.st_dev, final_path.st_ino, final_path.st_nlink,
        final_path.st_mode, final_path.st_size,
    ) != (
        before.st_dev, before.st_ino, before.st_nlink, before.st_mode,
        before.st_size,
    ):
      raise AnalysisError(f'No-follow hash target pathname changed: {path}.')
    return digest.hexdigest()
  finally:
    os.close(fd)


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
    raise AnalysisError('v3.3.4.5 analysis output destination is not fresh.')
  attempt_exists = (
      _ANALYSIS_ATTEMPT_DIR.exists() or _ANALYSIS_ATTEMPT_DIR.is_symlink()
  )
  if active_started_sha256 is None:
    if attempt_exists:
      raise AnalysisError('v3.3.4.5 analysis attempt destination is not fresh.')
    return
  if not _is_sha256(active_started_sha256) or not attempt_exists:
    raise AnalysisError('v3.3.4.5 active analysis attempt is absent.')
  paths = _strict_tree(
      _ANALYSIS_ATTEMPT_DIR, {'ANALYSIS_ATTEMPT_STARTED.json'},
      'active analysis-attempt tree during freeze validation',
  )
  if (
      stat.S_IMODE(paths[0].lstat().st_mode) != 0o400
      or _sha256(paths[0]) != active_started_sha256
  ):
    raise AnalysisError('v3.3.4.5 active analysis START changed.')


def _strict_regular(path: Path, label: str) -> None:
  _guard_path(path)
  try:
    mode = path.lstat().st_mode
  except OSError as error:
    raise AnalysisError(f'{label} cannot be statted.') from error
  if path.is_symlink() or not stat.S_ISREG(mode):
    raise AnalysisError(f'{label} is symlinked or not a regular file.')


def _read_bytes_no_follow(path: Path, label: str) -> bytes:
  _guard_path(path)
  try:
    expected = path.lstat()
  except OSError as error:
    raise AnalysisError(f'{label} cannot be statted.') from error
  if stat.S_ISLNK(expected.st_mode) or not stat.S_ISREG(expected.st_mode):
    raise AnalysisError(f'{label} is symlinked or not a regular file.')
  descriptor = -1
  try:
    descriptor = os.open(
        path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    before = os.fstat(descriptor)
    identity = lambda observed: (
        observed.st_dev, observed.st_ino, observed.st_nlink,
        observed.st_mode, observed.st_size,
    )
    if not stat.S_ISREG(before.st_mode) or identity(before) != identity(expected):
      raise AnalysisError(f'{label} inode changed before read.')
    payload = bytearray()
    for block in iter(lambda: os.read(descriptor, 1024 * 1024), b''):
      payload.extend(block)
    after = os.fstat(descriptor)
    final_path = path.lstat()
    if identity(after) != identity(before) or identity(final_path) != identity(before):
      raise AnalysisError(f'{label} changed during read.')
    return bytes(payload)
  except OSError as error:
    raise AnalysisError(f'{label} is not safely readable.') from error
  finally:
    if descriptor >= 0:
      os.close(descriptor)


def _read_text_no_follow(path: Path, label: str) -> str:
  try:
    return _read_bytes_no_follow(path, label).decode('utf-8')
  except UnicodeDecodeError as error:
    raise AnalysisError(f'{label} is not UTF-8 text.') from error


def _read_json(path: Path, label: str) -> dict[str, Any]:
  try:
    value = json.loads(_read_text_no_follow(path, label))
  except json.JSONDecodeError as error:
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
          rf'\.v3345\.tmp\.{node["runner_pid"]}\.'
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
      expected_path='atomic_publication_probe_v3_3_4_5.txt',
  )
  collision = _validate_live_publication_file(
      _PREFLIGHT_CACHE_DIR, node.get('collision_temp_binding'),
      'atomic-publication collision temporary',
  )
  if (
      final['sha256']
      != '7ffb46419c01255944db76c4530e7943574212aa4c4595fa85254bc9d21d6bd1'
      or final['size_bytes'] != 49
      or collision['sha256']
      != 'd7e55ae0ed0453b3d29f92731588b9626f10d5814b0f0ecd3198ced485940d44'
      or collision['size_bytes'] != 39
      or re.fullmatch(
          rf'\.v3345\.tmp\.{external_pid}\.[0-9]{{6}}\.[0-9a-f]{{32}}',
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


def _validate_prior_cache_directory_aware() -> dict[str, Any]:
  """Validates the one explicitly frozen empty-directory-aware cache tree."""
  expected_rows = [
      {
          'path': relative, 'entry_type': entry_type, 'mode': mode,
          'size_bytes': size_bytes, 'st_dev': st_dev, 'st_ino': st_ino,
          'st_nlink': st_nlink, 'sha256': sha256,
      }
      for (
          relative, entry_type, mode, size_bytes, st_dev, st_ino, st_nlink,
          sha256,
      ) in _PRIOR_CACHE_LSTAT_ROWS
  ]
  root = _PRIOR_CACHE_DIR
  descriptors: list[int] = []
  try:
    root_path_status = root.lstat()
    root_fd = os.open(
        root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
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
    matplotlib_fd = os.open(
        'matplotlib',
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=xdg_fd,
    )
    descriptors.append(matplotlib_fd)
    if (
        sorted(os.listdir(root_fd)) != ['triton', 'xdg']
        or os.listdir(triton_fd) != []
        or os.listdir(xdg_fd) != ['matplotlib']
        or os.listdir(matplotlib_fd) != ['fontlist-v3.11.0.json']
    ):
      raise AnalysisError('Immutable v3.3.3 cache exact membership changed.')
    directory_statuses = {
        '.': os.fstat(root_fd), 'triton': os.fstat(triton_fd),
        'xdg': os.fstat(xdg_fd), 'xdg/matplotlib': os.fstat(matplotlib_fd),
    }
    if (
        (root_path_status.st_dev, root_path_status.st_ino)
        != (directory_statuses['.'].st_dev, directory_statuses['.'].st_ino)
    ):
      raise AnalysisError('Immutable v3.3.3 cache root inode changed.')
    file_fd = os.open(
        'fontlist-v3.11.0.json',
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=matplotlib_fd,
    )
    descriptors.append(file_fd)
    before = os.fstat(file_fd)
    digest = hashlib.sha256()
    for block in iter(lambda: os.read(file_fd, 1024 * 1024), b''):
      digest.update(block)
    after = os.fstat(file_fd)
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_nlink, value.st_mode,
        value.st_size,
    )
    if identity(before) != identity(after):
      raise AnalysisError('Immutable v3.3.3 cache file changed during read.')
    # Reauthenticate the live pathname hierarchy, the held descriptors, and
    # all directory memberships after the file read.  A renamed replacement
    # root or child must not be accepted merely because the old inode remains
    # reachable through an already-open descriptor.
    final_root_path = root.lstat()
    if (
        (final_root_path.st_dev, final_root_path.st_ino)
        != (directory_statuses['.'].st_dev, directory_statuses['.'].st_ino)
        or sorted(os.listdir(root_fd)) != ['triton', 'xdg']
        or os.listdir(triton_fd) != []
        or os.listdir(xdg_fd) != ['matplotlib']
        or os.listdir(matplotlib_fd) != ['fontlist-v3.11.0.json']
    ):
      raise AnalysisError('Immutable v3.3.3 cache hierarchy changed during read.')
    final_directory_statuses = {
        '.': os.fstat(root_fd), 'triton': os.fstat(triton_fd),
        'xdg': os.fstat(xdg_fd), 'xdg/matplotlib': os.fstat(matplotlib_fd),
    }
    for relative, initial in directory_statuses.items():
      final = final_directory_statuses[relative]
      if identity(initial) != identity(final):
        raise AnalysisError(
            f'Immutable v3.3.3 cache directory changed: {relative}.'
        )
    live_children = {
        'triton': os.stat('triton', dir_fd=root_fd, follow_symlinks=False),
        'xdg': os.stat('xdg', dir_fd=root_fd, follow_symlinks=False),
        'xdg/matplotlib': os.stat(
            'matplotlib', dir_fd=xdg_fd, follow_symlinks=False
        ),
    }
    for relative, live in live_children.items():
      if identity(live) != identity(directory_statuses[relative]):
        raise AnalysisError(
            f'Immutable v3.3.3 cache child inode changed: {relative}.'
        )
    live_file = os.stat(
        'fontlist-v3.11.0.json', dir_fd=matplotlib_fd,
        follow_symlinks=False,
    )
    if identity(live_file) != identity(before) or identity(os.fstat(file_fd)) != identity(before):
      raise AnalysisError('Immutable v3.3.3 cache file inode changed.')
    observed_rows = []
    for expected in expected_rows:
      observed = (
          before if expected['entry_type'] == 'regular'
          else directory_statuses[expected['path']]
      )
      row = {
          'path': expected['path'], 'entry_type': expected['entry_type'],
          'mode': _publication_mode(observed.st_mode),
          'size_bytes': observed.st_size, 'st_dev': observed.st_dev,
          'st_ino': observed.st_ino, 'st_nlink': observed.st_nlink,
          'sha256': digest.hexdigest() if expected['entry_type'] == 'regular' else None,
      }
      if (
          row != expected
          or expected['entry_type'] == 'directory'
          and not stat.S_ISDIR(observed.st_mode)
          or expected['entry_type'] == 'regular'
          and not stat.S_ISREG(observed.st_mode)
      ):
        raise AnalysisError(
            f"Immutable v3.3.3 cache lstat row changed: {expected['path']}."
        )
      observed_rows.append(row)
  except OSError as error:
    raise AnalysisError('Immutable v3.3.3 cache no-follow open failed.') from error
  finally:
    for descriptor in reversed(descriptors):
      os.close(descriptor)
  file_row = observed_rows[-1]
  file_bindings = {
      file_row['path']: {
          key: file_row[key] for key in (
              'sha256', 'size_bytes', 'mode', 'st_dev', 'st_ino', 'st_nlink'
          )
      }
  }
  file_tree = _binding_map_digest(file_bindings)
  directories = [
      row['path'] for row in observed_rows if row['entry_type'] == 'directory'
  ]
  combined = hashlib.sha256()
  for relative in directories:
    combined.update(b'D\0' + relative.encode() + b'\0')
  for relative, binding in sorted(file_bindings.items()):
    combined.update(b'F\0' + relative.encode() + b'\0')
    combined.update(bytes.fromhex(binding['sha256']))
  directory_file_tree = combined.hexdigest()
  if (
      file_tree != PRIOR_CACHE_FILE_TREE_SHA256
      or directory_file_tree != PRIOR_CACHE_TREE_SHA256
  ):
    raise AnalysisError('Immutable v3.3.3 cache tree framing changed.')
  result = {
      'root': str(root.resolve()), 'file_count': 1, 'directory_count': 4,
      'directory_paths': directories, 'lstat_rows': observed_rows,
      'file_bindings': dict(sorted(file_bindings.items())),
      'file_tree_sha256': file_tree,
      'directory_file_tree_sha256': directory_file_tree,
      'exact_membership': True, 'no_follow': True,
  }
  _exact_keys(result, {
      'root', 'file_count', 'directory_count', 'directory_paths',
      'lstat_rows', 'file_bindings', 'file_tree_sha256',
      'directory_file_tree_sha256', 'exact_membership', 'no_follow',
  }, 'prior cache contract')
  return result


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
  cache = _validate_prior_cache_directory_aware()
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
      'cache': cache,
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
  with io.StringIO(
      _read_text_no_follow(_EXONS_PATH, 'development exon projection'),
      newline='',
  ) as handle:
    for row in csv.DictReader(handle, delimiter='\t'):
      if row.get('gene') not in {'BRAF', 'SLC25A48'}:
        raise AnalysisError('Development exon projection changed.')
      exons[row['ensembl_exon_id']] = dict(row)
  with io.StringIO(
      _read_text_no_follow(_CASES_PATH, 'development case projection'),
      newline='',
  ) as handle:
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
      'corrected_host_assertion_version': 'v3.3.4.5',
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
  del allow_invalid  # Invalid current work is represented separately in v3.3.4.5.
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
      'status': 'complete', 'family': 'v3_3_4_5_unrelated_donor_sidecar_anchor',
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
  return tuple(_REPO_ROOT / relative for relative in sorted(_V3345_SOURCE_PATHS))


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
      'upstream_source_attestation', 'v3_3_4_5_sidecar_sources',
      'created_at_unix_s',
  }, path.name)
  if (
      value.get('schema_version') != 'v3.3.4.5-import-provenance-v1'
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
        != 'run_encoder_skip_ood_sidecar_v3_3_4_5.py'
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
  if value.get('v3_3_4_5_sidecar_sources') != expected_sidecar:
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
    'prior_v3_3_4_3_consumed_preflight_prefix',
    'prior_v3_3_4_3_consumed_preflight_prefix_content_binding',
    'prior_v3_3_4_4_consumed_preflight_prefix',
    'prior_v3_3_4_4_consumed_preflight_prefix_content_binding',
}

_PREFLIGHT_RECORD_KEYS = {
    'amendment_sha256', 'atomic_publication_probe', 'created_at_unix_s',
    'external_freeze_authorization', 'external_cache_post_observation',
    'external_cache_hit_evidence', 'failure', 'freeze', 'freeze_sha256',
    'logs', 'no_jit_or_array_kernel', 'no_model_or_biological_access',
    'observation', 'original_protocol_sha256', 'preflight_attempt_number',
    'script_version', 'status', 'warnings',
    'prior_v3_3_4_3_consumed_preflight_prefix',
    'prior_v3_3_4_3_consumed_preflight_prefix_content_binding',
    'prior_v3_3_4_4_consumed_preflight_prefix',
    'prior_v3_3_4_4_consumed_preflight_prefix_content_binding',
}

_POST_START_FAILURE_KEYS = {
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
    'nonpublication_terminal_contract_v3_3_4_5',
    'prior_v3_3_4_3_consumed_preflight_prefix',
    'prior_v3_3_4_4_consumed_preflight_prefix',
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
    'prior_v3_3_4_3_consumed_preflight_prefix',
    'prior_v3_3_4_3_consumed_preflight_prefix_content_binding',
    'prior_v3_3_4_4_consumed_preflight_prefix',
    'prior_v3_3_4_4_consumed_preflight_prefix_content_binding',
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
  _validate_embedded_consumed_prefix(node, start, label='TERMINAL_FAILURE')
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
  _validate_embedded_consumed_prefix(node, start, label='RUN_COMPLETE')
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
  if stat.S_IMODE(root.lstat().st_mode) != 0o700:
    raise AnalysisError(f'{label} cache root mode changed.')
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


_CONSUMED_PREFIX_KEYS = {
    'status', 'predecessor_commit', 'predecessor_freeze', 'failure_stage',
    'failure_type', 'failure_message', 'traceback_provenance',
    'cache_tree_binding', 'directory_lstat_rows',
    'other_predecessor_paths_absent', 'no_jax_or_model_access',
    'no_gpu_or_confirmation_access', 'immutable_and_not_cache_input',
}
_CONSUMED_PREFIX_BINDING_KEYS = {'sha256', 'size_bytes'}

_CONSUMED_V3344_PREFIX_KEYS = {
    'status', 'predecessor_commit', 'predecessor_freeze', 'failure_stage',
    'failure_type', 'failure_message', 'traceback_provenance', 'root_cause',
    'external_preflight_archive', 'external_cache_archive',
    'other_v3_3_4_4_paths_absent', 'no_model_cache_or_start',
    'no_model_or_biological_access', 'no_array_jit_or_model_kernel',
    'no_scientific_or_confirmation_access',
    'immutable_and_not_cache_input', 'claim_boundary', 'access_boundary',
}
_CONSUMED_V3344_PREFLIGHT_KEYS = {
    'amendment_sha256', 'atomic_publication_probe', 'created_at_unix_s',
    'external_freeze_authorization', 'external_cache_post_observation',
    'external_cache_hit_evidence', 'failure', 'freeze', 'freeze_sha256',
    'logs', 'no_jit_or_array_kernel', 'no_model_or_biological_access',
    'observation', 'original_protocol_sha256', 'preflight_attempt_number',
    'script_version', 'status', 'warnings',
    'prior_v3_3_4_3_consumed_preflight_prefix',
    'prior_v3_3_4_3_consumed_preflight_prefix_content_binding',
}


def _expected_consumed_v3344_prefix() -> dict[str, Any]:
  """Returns the literal successful-preflight/parent-failure archive."""
  preflight_root = _CONSUMED_V3344_PREFLIGHT_DIR.resolve()
  cache_root = _CONSUMED_V3344_CACHE_DIR.resolve()
  preflight_files = {
      '.allocation.lock': {
          'path': '.allocation.lock', 'sha256': EMPTY_SHA256,
          'size_bytes': 0, 'mode': '0600', 'st_dev': 66307,
          'st_ino': 140791443, 'st_nlink': 1,
      },
      '.preflight_0000.reserved': {
          'path': '.preflight_0000.reserved', 'sha256': EMPTY_SHA256,
          'size_bytes': 0, 'mode': '0400', 'st_dev': 66307,
          'st_ino': 140791444, 'st_nlink': 1,
      },
      'preflight_0000.json': {
          'path': 'preflight_0000.json',
          'sha256': (
              'a240bf223dd62ebc53b84da35bb614df7987254c3694d7f07aae9785adec3801'
          ),
          'size_bytes': 27_062, 'mode': '0400', 'st_dev': 66307,
          'st_ino': 140791447, 'st_nlink': 1,
      },
      'preflight_0000.stderr.log': {
          'path': 'preflight_0000.stderr.log', 'sha256': EMPTY_SHA256,
          'size_bytes': 0, 'mode': '0400', 'st_dev': 66307,
          'st_ino': 140791446, 'st_nlink': 1,
      },
      'preflight_0000.stdout.log': {
          'path': 'preflight_0000.stdout.log', 'sha256': EMPTY_SHA256,
          'size_bytes': 0, 'mode': '0400', 'st_dev': 66307,
          'st_ino': 140791445, 'st_nlink': 1,
      },
  }
  cache_files = {
      '.v3344.tmp.2696297.000001.55167cfd266423a5ba861df8ca40686d': {
          'sha256': (
              'a1e62f4f34497aa5e72ece0670f1d865cd6eaacdcdfbacb00c39648d9e83f14f'
          ),
          'size_bytes': 39,
      },
      'atomic_publication_probe_v3_3_4_4.txt': {
          'sha256': (
              '47efa8c868d4d9455730ad1e89d6e44afee44172f0d2af7521d8574b7d85ecc9'
          ),
          'size_bytes': 49,
      },
  }
  final_binding = {
      'path': 'atomic_publication_probe_v3_3_4_4.txt',
      **cache_files['atomic_publication_probe_v3_3_4_4.txt'],
      'mode': '0400', 'st_dev': 66307, 'st_ino': 140791440,
      'st_nlink': 1,
  }
  collision_name = next(name for name in cache_files if name.startswith('.'))
  collision_binding = {
      'path': collision_name, **cache_files[collision_name],
      'mode': '0400', 'st_dev': 66307, 'st_ino': 140791441,
      'st_nlink': 1,
  }
  cache_binding = {
      'cache_role': 'external_preflight', 'cache_root': str(cache_root),
      'triton_cache_dir': str((cache_root / 'triton').resolve()),
      'xdg_cache_home': str((cache_root / 'xdg').resolve()),
      'directory_count': 3, 'directory_paths': ['.', 'triton', 'xdg'],
      'file_count': 2, 'files': cache_files,
      'tree_sha256': (
          '3a294e09038311b8bad85836c6983da31f50fdeef3365b844e8842922d33acba'
      ),
      'default_user_cache_paths_eligible': False,
      'diagnostic_outputs_only_no_cache_input': True,
  }
  probe = {
      'schema_version': 'v3.3.4.4-named-temp-renameat2-noreplace-v1',
      'method': 'named_temp_renameat2_noreplace', 'supported': True,
      'successful_final_binding': final_binding, 'collision_errno': 17,
      'collision_no_replace_exact': True,
      'collision_temp_binding': collision_binding,
      'destination_unchanged': True, 'temp_orphan_preserved': True,
      'parent_fsync_exact': True,
  }
  absent = {
      role: {'path': str(path.resolve()), 'absent': True}
      for role, path in sorted(_CONSUMED_V3344_OTHER_ABSENT_PATHS.items())
  }
  return {
      'status': (
          'consumed_successful_external_preflight_then_parent_role_routing_failure'
      ),
      'predecessor_commit': CONSUMED_V3344_COMMIT,
      'predecessor_freeze': {
          'path': str(_CONSUMED_V3344_FREEZE_PATH.resolve()),
          'sha256': CONSUMED_V3344_FREEZE_SHA256,
          'size_bytes': 187_923, 'git_mode': '100644',
          'top_level_key_count': 85, 'file_sha256_count': 120,
          'source_row_count': 120,
      },
      'failure_stage': 'parent_completed_external_preflight_validation',
      'failure_type': 'FileExistsError',
      'failure_message': (
          'Preflight directory exists before the external_preflight process.'
      ),
      'traceback_provenance': {
          'storage': 'coordinator_captured_not_persisted',
          'sha256': CONSUMED_V3344_TRACEBACK_SHA256,
          'size_bytes': 1_168, 'session_id': None,
          'captured_at_unix_s': None,
          'wall_clock_timestamp_available': False,
      },
      'root_cause': {
          'parent_ambient_cache_role': 'external_preflight',
          'parent_ambient_cache_root': str(cache_root),
          'called_validator': 'validate_preflight_state_for_role',
          'selected_branch': 'external_preflight_entry_absence',
          'rejected_state': 'completed_preflight_directory_present',
          'required_validator': 'validate_completed_external_preflight_state',
          'failure_before_model_cache_allocation': True,
          'failure_before_model_start': True,
          'launcher_source_binding': {
              'path': str(_CONSUMED_V3344_LAUNCHER_PATH.resolve()),
              'sha256': (
                  '4cdeee9df8b14043633383b99cdf8d88bdf1c3a0a3e4146a3c40adf9e78991f9'
              ),
              'size_bytes': 27_372,
          },
          'bootstrap_source_binding': {
              'path': str(_CONSUMED_V3344_BOOTSTRAP_PATH.resolve()),
              'sha256': (
                  '8e2559d5dae96f6d9ddaa752e2aa4de3829ec85115e78fba356e6fe8c8abccb8'
              ),
              'size_bytes': 143_561,
          },
      },
      'external_preflight_archive': {
          'root': str(preflight_root),
          'directory_lstat_rows': [{
              'path': '.', 'entry_type': 'directory', 'mode': '0700',
              'st_dev': 66307, 'st_ino': 140791442, 'st_nlink': 2,
              'size_bytes': 4096,
          }],
          'directory_count': 1, 'directory_paths': ['.'],
          'file_count': 5, 'files': preflight_files,
          'file_tree_sha256': (
              'f009ba6fe67a715301b443940876be8f85998a50f71f320d0dc5e3dd52dfd6e5'
          ),
          'record_binding': {
              'path': str((preflight_root / 'preflight_0000.json').resolve()),
              'sha256': preflight_files['preflight_0000.json']['sha256'],
              'size_bytes': 27_062, 'mode': '0400',
          },
          'directory_aware_tree_sha256': (
              '1a343cacb96cc1a1c88735c6a3bb8edfb0b71c1df89e924f52e099c30e3217f5'
          ),
          'record_canonical_binding': {
              'sha256': (
                  '9b1a5e3bbc9845d04430c259ed39db0f39f31b56128f022443382b91e6027285'
              ),
              'size_bytes': 22_193,
          },
          'record_semantics': {
              'status': 'pass', 'failure': None,
              'preflight_attempt_number': 0,
              'script_version': 'opensplice-device-preflight-v3.3.4.4',
              'freeze_sha256': CONSUMED_V3344_FREEZE_SHA256,
              'external_pid': 2_696_297, 'jax_default_backend': 'gpu',
              'jax_gpu_device_count': 1,
              'device_kind': 'NVIDIA GeForce RTX 3090',
              'device_uuid': 'GPU-64111645-1e42-a96d-f192-4abbec4b8090',
              'compute_capability': '8.6',
              'no_jit_or_array_kernel': True,
              'no_model_or_biological_access': True,
              'external_cache_hit': False,
          },
      },
      'external_cache_archive': {
          'root': str(cache_root),
          'directory_lstat_rows': [
              {'path': '.', 'entry_type': 'directory', 'mode': '0700',
               'st_dev': 66307, 'st_ino': 140791437, 'st_nlink': 4,
               'size_bytes': 4096},
              {'path': 'triton', 'entry_type': 'directory', 'mode': '0700',
               'st_dev': 66307, 'st_ino': 140791438, 'st_nlink': 2,
               'size_bytes': 4096},
              {'path': 'xdg', 'entry_type': 'directory', 'mode': '0700',
               'st_dev': 66307, 'st_ino': 140791439, 'st_nlink': 2,
               'size_bytes': 4096},
          ],
          'cache_tree_binding': cache_binding,
          'cache_tree_content_binding': {
              'sha256': (
                  '88c2a4cde3a9881f76dc719b48dbd7f051b5841843dc5439a8e7d2349aabbc46'
              ),
              'size_bytes': 1_033,
          },
          'atomic_publication_probe': probe,
          'atomic_publication_probe_content_binding': {
              'sha256': (
                  'a25798b7c788ce614d10a7cc0d07f1795ebaf6fd9e928dcef78e096323d9bf70'
              ),
              'size_bytes': 738,
          },
      },
      'other_v3_3_4_4_paths_absent': absent,
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


def _expected_consumed_v3343_prefix() -> dict[str, Any]:
  absent = {
      key: {'path': str(path.resolve()), 'absent': True}
      for key, path in _CONSUMED_V3343_OTHER_ABSENT_PATHS.items()
  }
  cache_root = _CONSUMED_V3343_CACHE_DIR.resolve()
  return {
      'status': 'consumed_external_preflight_freeze_validation_failure',
      'predecessor_commit': CONSUMED_V3343_COMMIT,
      'predecessor_freeze': {
          'path': str(_CONSUMED_V3343_FREEZE_PATH.resolve()),
          'sha256': CONSUMED_V3343_FREEZE_SHA256,
          'size_bytes': 174_545, 'git_mode': '100644',
          'top_level_key_count': 84, 'file_sha256_count': 108,
          'source_row_count': 108,
      },
      'failure_stage': 'preflight_freeze_validation',
      'failure_type': 'ValueError',
      'failure_message': (
          'v3.3.4.3 preflight freeze mismatch: preflight_script_version.'
      ),
      'traceback_provenance': {
          'storage': 'coordinator_captured_not_persisted',
          'sha256': CONSUMED_V3343_TRACEBACK_SHA256,
          'size_bytes': 953, 'session_id': None,
          'captured_at_unix_s': None,
          'wall_clock_timestamp_available': False,
      },
      'cache_tree_binding': {
          'cache_role': 'external_preflight',
          'cache_root': str(cache_root),
          'triton_cache_dir': str((cache_root / 'triton').resolve()),
          'xdg_cache_home': str((cache_root / 'xdg').resolve()),
          'directory_paths': ['.', 'triton', 'xdg'], 'directory_count': 3,
          'files': {}, 'file_count': 0,
          'tree_sha256': CONSUMED_V3343_CACHE_TREE_SHA256,
          'default_user_cache_paths_eligible': False,
          'diagnostic_outputs_only_no_cache_input': True,
      },
      'directory_lstat_rows': [
          {'path': '.', 'entry_type': 'directory', 'mode': '0700',
           'st_dev': 66307, 'st_ino': 140791433, 'st_nlink': 4,
           'size_bytes': 4096},
          {'path': 'triton', 'entry_type': 'directory', 'mode': '0700',
           'st_dev': 66307, 'st_ino': 140791434, 'st_nlink': 2,
           'size_bytes': 4096},
          {'path': 'xdg', 'entry_type': 'directory', 'mode': '0700',
           'st_dev': 66307, 'st_ino': 140791435, 'st_nlink': 2,
           'size_bytes': 4096},
      ],
      'other_predecessor_paths_absent': absent,
      'no_jax_or_model_access': True,
      'no_gpu_or_confirmation_access': True,
      'immutable_and_not_cache_input': True,
  }


def _validate_consumed_v3343_prefix(
    freeze: Mapping[str, Any], *, label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Replays the immutable, directory-only consumed preflight prefix."""
  expected = _expected_consumed_v3343_prefix()
  prefix = _exact_keys(
      freeze.get('prior_v3_3_4_3_consumed_preflight_prefix'),
      _CONSUMED_PREFIX_KEYS, f'{label}.prefix',
  )
  if prefix != expected:
    raise AnalysisError(f'{label} consumed v3.3.4.3 prefix changed.')
  expected_binding = _content_binding(expected)

  _strict_regular(
      _CONSUMED_V3343_FREEZE_PATH, f'{label} predecessor v3.3.4.3 freeze'
  )
  freeze_status = _CONSUMED_V3343_FREEZE_PATH.lstat()
  freeze_relative = _CONSUMED_V3343_FREEZE_PATH.relative_to(
      _REPO_ROOT
  ).as_posix()
  try:
    mode_line = subprocess.check_output(
        ('git', '-C', str(_REPO_ROOT), 'ls-tree', CONSUMED_V3343_COMMIT,
         '--', freeze_relative), text=True,
    ).strip()
  except subprocess.CalledProcessError as error:
    raise AnalysisError(f'{label} predecessor commit is unavailable.') from error
  if (
      _sha256(_CONSUMED_V3343_FREEZE_PATH) != CONSUMED_V3343_FREEZE_SHA256
      or freeze_status.st_size != 174_545
      or stat.S_IMODE(freeze_status.st_mode) != 0o644
      or not mode_line or mode_line.split()[0] != '100644'
      or _git_blob_sha256(CONSUMED_V3343_COMMIT, freeze_relative)
      != CONSUMED_V3343_FREEZE_SHA256
  ):
    raise AnalysisError(f'{label} predecessor freeze bytes/mode changed.')
  predecessor_freeze = _read_json(
      _CONSUMED_V3343_FREEZE_PATH, f'{label} predecessor freeze'
  )
  inventory = predecessor_freeze.get('file_sha256')
  source_contract = predecessor_freeze.get('source_inventory_contract')
  rows = source_contract.get('rows') if isinstance(source_contract, Mapping) else None
  if (
      len(predecessor_freeze) != 84 or not isinstance(inventory, Mapping)
      or len(inventory) != 108 or not isinstance(rows, list)
      or len(rows) != 108
      or source_contract.get('source_row_count') != 108
  ):
    raise AnalysisError(f'{label} predecessor freeze inventory changed.')
  row_map: dict[str, Mapping[str, Any]] = {}
  for raw in rows:
    row = _exact_keys(
        raw, {'path', 'sha256', 'size_bytes', 'git_mode'},
        f'{label}.predecessor_source_row',
    )
    relative = row.get('path')
    if not isinstance(relative, str) or relative in row_map:
      raise AnalysisError(f'{label} predecessor source path changed.')
    row_map[relative] = row
  if list(row_map) != sorted(inventory) or set(row_map) != set(inventory):
    raise AnalysisError(f'{label} predecessor source ordering changed.')
  for relative, digest in inventory.items():
    row = row_map[relative]
    path = _REPO_ROOT / relative
    _strict_regular(path, f'{label} predecessor source {relative}')
    try:
      source_mode_line = subprocess.check_output(
          ('git', '-C', str(_REPO_ROOT), 'ls-tree', CONSUMED_V3343_COMMIT,
           '--', relative), text=True,
      ).strip()
    except subprocess.CalledProcessError as error:
      raise AnalysisError(
          f'{label} predecessor source is absent from its commit: {relative}.'
      ) from error
    if (
        not _is_sha256(digest) or row.get('sha256') != digest
        or path.stat().st_size != row.get('size_bytes')
        or _sha256(path) != digest
        or _git_blob_sha256(CONSUMED_V3343_COMMIT, relative) != digest
        or not source_mode_line
        or source_mode_line.split()[0] != row.get('git_mode')
    ):
      raise AnalysisError(f'{label} predecessor source changed: {relative}.')

  live_cache = _live_cache_binding(
      _CONSUMED_V3343_CACHE_DIR, 'external_preflight',
      f'{label} consumed predecessor cache',
  )
  if live_cache != expected['cache_tree_binding']:
    raise AnalysisError(f'{label} consumed predecessor cache tree changed.')
  if _content_binding(live_cache) != {
      'sha256': CONSUMED_V3343_CACHE_BINDING_SHA256, 'size_bytes': 745,
  }:
    raise AnalysisError(f'{label} consumed cache canonical binding changed.')
  observed_rows = []
  for relative in ('.', 'triton', 'xdg'):
    path = (
        _CONSUMED_V3343_CACHE_DIR if relative == '.'
        else _CONSUMED_V3343_CACHE_DIR / relative
    )
    status = path.lstat()
    if path.is_symlink() or not stat.S_ISDIR(status.st_mode):
      raise AnalysisError(f'{label} consumed cache directory changed type.')
    observed_rows.append({
        'path': relative, 'entry_type': 'directory',
        'mode': f'{stat.S_IMODE(status.st_mode):04o}',
        'st_dev': status.st_dev, 'st_ino': status.st_ino,
        'st_nlink': status.st_nlink, 'size_bytes': status.st_size,
    })
  if observed_rows != expected['directory_lstat_rows']:
    raise AnalysisError(f'{label} consumed cache lstat rows changed.')
  for key, path in _CONSUMED_V3343_OTHER_ABSENT_PATHS.items():
    if path.exists() or path.is_symlink():
      raise AnalysisError(f'{label} consumed-prefix absence changed: {key}.')
  return copy.deepcopy(expected), dict(expected_binding)


def _lstat_archive_row(path: Path, root: Path) -> dict[str, Any]:
  status = path.lstat()
  if path.is_symlink():
    raise AnalysisError(f'Consumed archive entry is symlinked: {path}.')
  if stat.S_ISDIR(status.st_mode):
    entry_type = 'directory'
  elif stat.S_ISREG(status.st_mode):
    entry_type = 'regular'
  else:
    raise AnalysisError(f'Consumed archive entry is special: {path}.')
  return {
      'path': '.' if path == root else path.relative_to(root).as_posix(),
      'entry_type': entry_type,
      'mode': f'{stat.S_IMODE(status.st_mode):04o}',
      'st_dev': status.st_dev, 'st_ino': status.st_ino,
      'st_nlink': status.st_nlink, 'size_bytes': status.st_size,
  }


def _validate_consumed_v3344_prefix(
    freeze: Mapping[str, Any], *, label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Replays the immutable successful-preflight/parent-failure prefix."""
  expected = _expected_consumed_v3344_prefix()
  prefix = _exact_keys(
      freeze.get('prior_v3_3_4_4_consumed_preflight_prefix'),
      _CONSUMED_V3344_PREFIX_KEYS, f'{label}.v3344_prefix',
  )
  expected_binding = _content_binding(expected)
  if prefix != expected or expected_binding != {
      'sha256': CONSUMED_V3344_PREFIX_SHA256, 'size_bytes': 8_653,
  }:
    raise AnalysisError(f'{label} consumed v3.3.4.4 prefix changed.')

  _strict_regular(
      _CONSUMED_V3344_FREEZE_PATH, f'{label} predecessor v3.3.4.4 freeze'
  )
  freeze_status = _CONSUMED_V3344_FREEZE_PATH.lstat()
  freeze_relative = _CONSUMED_V3344_FREEZE_PATH.relative_to(
      _REPO_ROOT
  ).as_posix()
  try:
    mode_line = subprocess.check_output(
        ('git', '-C', str(_REPO_ROOT), 'ls-tree', CONSUMED_V3344_COMMIT,
         '--', freeze_relative), text=True,
    ).strip()
  except subprocess.CalledProcessError as error:
    raise AnalysisError(f'{label} v3.3.4.4 commit is unavailable.') from error
  if (
      _sha256(_CONSUMED_V3344_FREEZE_PATH) != CONSUMED_V3344_FREEZE_SHA256
      or freeze_status.st_size != 187_923
      or stat.S_IMODE(freeze_status.st_mode) != 0o644
      or not mode_line or mode_line.split()[0] != '100644'
      or _git_blob_sha256(CONSUMED_V3344_COMMIT, freeze_relative)
      != CONSUMED_V3344_FREEZE_SHA256
  ):
    raise AnalysisError(f'{label} v3.3.4.4 freeze bytes/mode changed.')
  predecessor_freeze = _read_json(
      _CONSUMED_V3344_FREEZE_PATH, f'{label} v3.3.4.4 freeze'
  )
  inventory = predecessor_freeze.get('file_sha256')
  source_contract = predecessor_freeze.get('source_inventory_contract')
  rows = source_contract.get('rows') if isinstance(source_contract, Mapping) else None
  if (
      len(predecessor_freeze) != 85 or not isinstance(inventory, Mapping)
      or len(inventory) != 120 or not isinstance(rows, list)
      or len(rows) != 120 or source_contract.get('source_row_count') != 120
  ):
    raise AnalysisError(f'{label} v3.3.4.4 freeze inventory changed.')
  row_map: dict[str, Mapping[str, Any]] = {}
  for raw in rows:
    row = _exact_keys(
        raw, {'path', 'sha256', 'size_bytes', 'git_mode'},
        f'{label}.v3344_source_row',
    )
    relative = row.get('path')
    if not isinstance(relative, str) or relative in row_map:
      raise AnalysisError(f'{label} v3.3.4.4 source path changed.')
    row_map[relative] = row
  if list(row_map) != sorted(inventory) or set(row_map) != set(inventory):
    raise AnalysisError(f'{label} v3.3.4.4 source ordering changed.')
  for relative, digest in inventory.items():
    row = row_map[relative]
    path = _REPO_ROOT / relative
    _strict_regular(path, f'{label} v3.3.4.4 source {relative}')
    try:
      source_mode_line = subprocess.check_output(
          ('git', '-C', str(_REPO_ROOT), 'ls-tree', CONSUMED_V3344_COMMIT,
           '--', relative), text=True,
      ).strip()
    except subprocess.CalledProcessError as error:
      raise AnalysisError(
          f'{label} v3.3.4.4 source missing from commit: {relative}.'
      ) from error
    if (
        not _is_sha256(digest) or row.get('sha256') != digest
        or path.stat().st_size != row.get('size_bytes')
        or _sha256(path) != digest
        or _git_blob_sha256(CONSUMED_V3344_COMMIT, relative) != digest
        or not source_mode_line
        or source_mode_line.split()[0] != row.get('git_mode')
    ):
      raise AnalysisError(f'{label} v3.3.4.4 source changed: {relative}.')

  archive = expected['external_preflight_archive']
  file_bindings = archive['files']
  paths = _strict_tree(
      _CONSUMED_V3344_PREFLIGHT_DIR, set(file_bindings),
      f'{label} consumed v3.3.4.4 preflight',
  )
  root_row = _lstat_archive_row(
      _CONSUMED_V3344_PREFLIGHT_DIR, _CONSUMED_V3344_PREFLIGHT_DIR
  )
  if [root_row] != archive['directory_lstat_rows']:
    raise AnalysisError(f'{label} consumed preflight root lstat changed.')
  simple_file_bindings = {}
  for relative, expected_row in file_bindings.items():
    path = _CONSUMED_V3344_PREFLIGHT_DIR / relative
    observed = _lstat_archive_row(path, _CONSUMED_V3344_PREFLIGHT_DIR)
    observed.pop('entry_type')
    observed['sha256'] = _sha256(path)
    # Keep the protocol's exact key order irrelevant while comparing values.
    if observed != expected_row:
      raise AnalysisError(f'{label} consumed preflight file changed: {relative}.')
    simple_file_bindings[relative] = {
        'sha256': observed['sha256'], 'size_bytes': observed['size_bytes'],
    }
  if (
      _tree_digest(paths, _CONSUMED_V3344_PREFLIGHT_DIR)
      != archive['file_tree_sha256']
      or _cache_binding_digest(['.'], simple_file_bindings)
      != archive['directory_aware_tree_sha256']
  ):
    raise AnalysisError(f'{label} consumed preflight tree digest changed.')
  record_path = _CONSUMED_V3344_PREFLIGHT_DIR / 'preflight_0000.json'
  record = _read_json(record_path, f'{label} consumed preflight record')
  _exact_keys(record, _CONSUMED_V3344_PREFLIGHT_KEYS, 'consumed preflight')
  if _content_binding(record) != archive['record_canonical_binding']:
    raise AnalysisError(f'{label} consumed preflight canonical bytes changed.')
  semantics = archive['record_semantics']
  gpu_devices = record.get('observation', {}).get('jax_gpu_devices')
  parsed_gpu = record.get('observation', {}).get('nvidia_smi', {}).get(
      'parsed_single_gpu'
  )
  observed_semantics = {
      'status': record.get('status'), 'failure': record.get('failure'),
      'preflight_attempt_number': record.get('preflight_attempt_number'),
      'script_version': record.get('script_version'),
      'freeze_sha256': record.get('freeze_sha256'),
      'external_pid': record.get('observation', {}).get('pid'),
      'jax_default_backend': record.get('observation', {}).get(
          'jax_default_backend'
      ),
      'jax_gpu_device_count': len(gpu_devices) if isinstance(gpu_devices, list) else None,
      'device_kind': (
          gpu_devices[0].get('device_kind')
          if isinstance(gpu_devices, list) and len(gpu_devices) == 1
          else None
      ),
      'device_uuid': parsed_gpu.get('uuid') if isinstance(parsed_gpu, Mapping) else None,
      'compute_capability': (
          parsed_gpu.get('compute_capability')
          if isinstance(parsed_gpu, Mapping) else None
      ),
      'no_jit_or_array_kernel': record.get('no_jit_or_array_kernel'),
      'no_model_or_biological_access': record.get(
          'no_model_or_biological_access'
      ),
      'external_cache_hit': record.get('external_cache_hit_evidence', {}).get(
          'cache_hit'
      ),
  }
  if observed_semantics != semantics:
    raise AnalysisError(f'{label} consumed preflight semantics changed.')
  expected_v3343 = _expected_consumed_v3343_prefix()
  if (
      record.get('prior_v3_3_4_3_consumed_preflight_prefix')
      != expected_v3343
      or record.get(
          'prior_v3_3_4_3_consumed_preflight_prefix_content_binding'
      ) != _content_binding(expected_v3343)
  ):
    raise AnalysisError(f'{label} consumed preflight transitive prefix changed.')

  cache_archive = expected['external_cache_archive']
  live_cache = _live_cache_binding(
      _CONSUMED_V3344_CACHE_DIR, 'external_preflight',
      f'{label} consumed v3.3.4.4 cache',
  )
  if (
      live_cache != cache_archive['cache_tree_binding']
      or _content_binding(live_cache)
      != cache_archive['cache_tree_content_binding']
      or record.get('external_cache_post_observation') != live_cache
      or record.get('atomic_publication_probe')
      != cache_archive['atomic_publication_probe']
      or _content_binding(record['atomic_publication_probe'])
      != cache_archive['atomic_publication_probe_content_binding']
  ):
    raise AnalysisError(f'{label} consumed cache/probe archive changed.')
  observed_cache_rows = [
      _lstat_archive_row(
          _CONSUMED_V3344_CACHE_DIR if relative == '.'
          else _CONSUMED_V3344_CACHE_DIR / relative,
          _CONSUMED_V3344_CACHE_DIR,
      )
      for relative in ('.', 'triton', 'xdg')
  ]
  if observed_cache_rows != cache_archive['directory_lstat_rows']:
    raise AnalysisError(f'{label} consumed cache lstat rows changed.')
  for binding_name in (
      'successful_final_binding', 'collision_temp_binding'
  ):
    expected_file = cache_archive['atomic_publication_probe'][binding_name]
    cache_path = _CONSUMED_V3344_CACHE_DIR / expected_file['path']
    observed_file = _lstat_archive_row(
        cache_path, _CONSUMED_V3344_CACHE_DIR
    )
    observed_file.pop('entry_type')
    observed_file['sha256'] = _sha256(cache_path)
    if observed_file != expected_file:
      raise AnalysisError(
          f'{label} consumed cache file lstat changed: {binding_name}.'
      )
  for role, path in _CONSUMED_V3344_OTHER_ABSENT_PATHS.items():
    if path.exists() or path.is_symlink():
      raise AnalysisError(f'{label} v3.3.4.4 absence changed: {role}.')
  return copy.deepcopy(expected), dict(expected_binding)


def _validate_embedded_consumed_prefix(
    value: Mapping[str, Any], freeze: Mapping[str, Any], *, label: str,
) -> None:
  if (
      value.get('prior_v3_3_4_3_consumed_preflight_prefix')
      != freeze.get('prior_v3_3_4_3_consumed_preflight_prefix')
      or value.get(
          'prior_v3_3_4_3_consumed_preflight_prefix_content_binding'
      ) != _content_binding(
          freeze.get('prior_v3_3_4_3_consumed_preflight_prefix')
      )
  ):
    raise AnalysisError(f'{label} consumed v3.3.4.3 prefix binding changed.')
  if (
      value.get('prior_v3_3_4_4_consumed_preflight_prefix')
      != freeze.get('prior_v3_3_4_4_consumed_preflight_prefix')
      or value.get(
          'prior_v3_3_4_4_consumed_preflight_prefix_content_binding'
      ) != _content_binding(
          freeze.get('prior_v3_3_4_4_consumed_preflight_prefix')
      )
  ):
    raise AnalysisError(f'{label} consumed v3.3.4.4 prefix binding changed.')


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
      # The external CPU preflight has no compile stage, but its serializer
      # records the two observed cache-hit signals as literal false booleans.
      'persistent_compilation_cache_hit_reported': False,
      'executable_deserialized': False,
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
      'schema_version': 'v3.3.4.5-nonpublication-terminal-v1',
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
    raise AnalysisError('Frozen v3.3.4.5 nonpublication contract changed.')
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
          r'^\.v3345\.tmp\.[1-9][0-9]*\.[0-9]{6}\.[0-9a-f]{32}$'
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
          'final_basename': 'atomic_publication_probe_v3_3_4_5.txt',
              'final_sha256': (
                      '7ffb46419c01255944db76c4530e7943574212aa4c4595fa85254bc9d21d6bd1'
          ),
          'final_size_bytes': 49,
              'collision_sha256': (
              'd7e55ae0ed0453b3d29f92731588b9626f10d5814b0f0ecd3198ced485940d44'
          ),
          'collision_size_bytes': 39, 'collision_errno': 17,
          'collision_temp_preserved': True,
          'parent_fsync_exact_required': True,
      },
  }
  if dict(node) != expected:
    raise AnalysisError('Frozen v3.3.4.5 publication contract changed.')
  return dict(node)








def _validate_freeze_v3345(
    run_dir: Path, *, bundle_root: Path,
    active_started_sha256: str | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, str], dict[int, Any], dict[str, Any]]:
  if run_dir.resolve() != _RUN_DIR.resolve() or bundle_root.resolve() != _REPO_ROOT:
    raise AnalysisError('v3.3.4.5 production run/repository path changed.')
  _assert_predecessor_v334_paths_absent('freeze validation')
  _validate_analysis_destination_state(active_started_sha256)
  if _sha256(_AMENDMENT_PATH) != AMENDMENT_SHA256:
    raise AnalysisError('v3.3.4.5 amendment bytes changed.')
  amendment_relative = _AMENDMENT_PATH.relative_to(_REPO_ROOT).as_posix()
  if _git_blob_sha256(AMENDMENT_COMMIT, amendment_relative) != AMENDMENT_SHA256:
    raise AnalysisError('Bound v3.3.4.5 amendment Git blob changed.')
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
  freeze = _read_json(_FREEZE_PATH, 'v3.3.4.5 freeze')
  _exact_keys(freeze, _FREEZE_KEYS, 'v3.3.4.5 freeze')
  _validate_consumed_v3343_prefix(freeze, label='freeze validation')
  _validate_consumed_v3344_prefix(freeze, label='freeze validation')
  _validate_publication_contract(freeze.get('publication_contract_v3_3_4_1'))
  _validate_nonpublication_terminal_contract(
      freeze.get('nonpublication_terminal_contract_v3_3_4_5')
  )
  freeze_sha = _sha256(_FREEZE_PATH)
  expected_scalars = {
      'attempt_id': ATTEMPT_ID, 'script_version': SCRIPT_VERSION,
      'amendment_path': str(_AMENDMENT_PATH.resolve()),
      'amendment_sha256': AMENDMENT_SHA256,
      'amendment_commit': AMENDMENT_COMMIT,
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'output_dir': str(_RUN_DIR.resolve()),
      'analysis_dir': str(_OLD_V3345_ANALYSIS_DIR.resolve()),
      'analysis_attempt_dir': str(_OLD_V3345_ANALYSIS_ATTEMPT_DIR.resolve()),
      'preflight_dir': str(_PREFLIGHT_DIR.resolve()),
      'preflight_script_version': PREFLIGHT_SCRIPT_VERSION,
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
      raise AnalysisError(f'v3.3.4.5 freeze.{key} changed.')
  if (
      freeze.get('recipient_orders') != list(RECIPIENT_ORDERS)
      or freeze.get('ood_anchor_ids') != list(ANCHOR_IDS)
      or freeze.get('eight_row_roles') != list(EIGHT_ROLES)
      or freeze.get('eight_row_natural_identity_rows') != list(IDENTITY_ROWS)
      or freeze.get('eight_row_intended_donor_rows') != list(INTENDED_DONOR_ROWS)
      or freeze.get('eight_row_unrelated_donor_rows') != list(UNRELATED_DONOR_ROWS)
      or freeze.get('invariant_rows_between_calls') != list(INVARIANT_ROWS)
  ):
    raise AnalysisError('v3.3.4.5 freeze scientific order/role contract changed.')
  inventory = freeze.get('file_sha256')
  source_contract = freeze.get('source_inventory_contract')
  if not isinstance(inventory, Mapping) or len(inventory) != 132:
    raise AnalysisError('v3.3.4.5 freeze source inventory is not 132 rows.')
  if not isinstance(source_contract, Mapping):
    raise AnalysisError('v3.3.4.5 source inventory contract is absent.')
  _exact_keys(source_contract, {
      'source_row_count', 'rows', 'prospective_upstream_source_file_count',
      'loaded_scientific_module_contract',
  }, 'freeze.source_inventory_contract')
  rows = source_contract.get('rows')
  if (
      source_contract.get('source_row_count') != 132
      or source_contract.get('prospective_upstream_source_file_count') != 26
      or not isinstance(rows, list) or len(rows) != 132
      or not isinstance(
          source_contract.get('loaded_scientific_module_contract'), list
      )
  ):
    raise AnalysisError('v3.3.4.5 source inventory contract counts changed.')
  new_shells = {_V3345_SOURCE_PATHS[1], _V3345_SOURCE_PATHS[8]}
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
        or relative in set(_V3345_SOURCE_PATHS) - new_shells
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
    raise AnalysisError('v3.3.4.5 tracked repository is not clean.') from error
  for relative, digest in inventory.items():
    if not isinstance(relative, str) or not _is_sha256(digest):
      raise AnalysisError('v3.3.4.5 source inventory row is malformed.')
    path = bundle_root / relative
    _strict_regular(path, f'v3.3.4.5 source {relative}')
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
      raise AnalysisError(f'v3.3.4.5 source differs from launch HEAD: {relative}.')
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


def _validate_start_v3345(
    run_dir: Path, freeze: Mapping[str, Any], freeze_sha: str, *,
    prior333: Mapping[str, Any], prior331: Mapping[str, Any],
) -> dict[str, Any]:
  start = _read_json(run_dir / 'ATTEMPT_STARTED.json', 'ATTEMPT_STARTED')
  _exact_keys(start, _START_KEYS, 'ATTEMPT_STARTED')
  _validate_embedded_consumed_prefix(start, freeze, label='ATTEMPT_STARTED')
  authorization = _exact_keys(
      start.get('external_freeze_authorization'),
      {
          'git_head', 'freeze_path', 'freeze_sha256', 'freeze_size_bytes',
          'live_equals_git_show', 'tracked_clean', 'authorization_source',
      }, 'START.external_freeze_authorization',
  )
  model_head = MODEL_SOURCE_COMMIT
  freeze_relative = _FREEZE_PATH.relative_to(_REPO_ROOT).as_posix()
  try:
    frozen_bytes = subprocess.check_output(
        ('git', '-C', str(_REPO_ROOT), 'show',
         f'{model_head}:{freeze_relative}')
    )
  except subprocess.CalledProcessError as error:
    raise AnalysisError('The v3.3.4.5 freeze is not tracked at launch HEAD.') from error
  if (
      authorization.get('freeze_path') != str(_FREEZE_PATH.resolve())
      or authorization.get('freeze_sha256') != freeze_sha
      or authorization.get('freeze_size_bytes') != _FREEZE_PATH.stat().st_size
      or authorization.get('live_equals_git_show') is not True
      or authorization.get('tracked_clean') is not True
      or authorization.get('authorization_source')
      != 'external_post_commit_audit'
      or start.get('git_head') != authorization.get('git_head')
      or model_head != authorization.get('git_head')
      or hashlib.sha256(frozen_bytes).hexdigest() != freeze_sha
      or frozen_bytes
      != _read_bytes_no_follow(_FREEZE_PATH, 'v3.3.4.5 freeze')
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
      'analysis_attempt': str(_OLD_V3345_ANALYSIS_ATTEMPT_DIR.resolve()),
      'analysis_output': str(_OLD_V3345_ANALYSIS_DIR.resolve()),
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
      'row_count': 132, 'rows': source_rows,
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
  if stat.S_IMODE(_PREFLIGHT_DIR.lstat().st_mode) != 0o700:
    raise AnalysisError('External preflight root mode changed.')
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
  _exact_keys(record, _PREFLIGHT_RECORD_KEYS, 'external preflight')
  _validate_embedded_consumed_prefix(
      record, freeze, label='external preflight'
  )
  freeze_evidence = _exact_keys(record.get('freeze'), {
      'path', 'sha256', 'size_bytes', 'external_freeze_authorization',
      'preflight_version_proof',
      'prior_v3_3_4_3_consumed_preflight_prefix',
      'prior_v3_3_4_3_consumed_preflight_prefix_content_binding',
      'prior_v3_3_4_4_consumed_preflight_prefix',
      'prior_v3_3_4_4_consumed_preflight_prefix_content_binding',
  }, 'external preflight.freeze')
  version_proof = _exact_keys(
      freeze_evidence.get('preflight_version_proof'), {
          'preflight_script_version', 'freeze_equals_bootstrap',
          'bootstrap_equals_producer_literal', 'producer_source_binding',
          'validated_before_allocation_or_registration',
      }, 'external preflight.freeze.preflight_version_proof',
  )
  producer_path = _HERE / 'run_device_preflight_v3_3_4_5.py'
  producer_relative = producer_path.relative_to(_REPO_ROOT).as_posix()
  producer_binding = _exact_keys(
      version_proof.get('producer_source_binding'),
      {'path', 'sha256', 'size_bytes'},
      'external preflight producer binding',
  )
  try:
    producer_tree = ast.parse(
        _read_text_no_follow(producer_path, 'external preflight producer'),
        filename=str(producer_path),
    )
  except (OSError, SyntaxError, UnicodeError) as error:
    raise AnalysisError('External preflight producer cannot be parsed.') from error
  assignments = []
  for item in producer_tree.body:
    if isinstance(item, (ast.Assign, ast.AnnAssign)):
      targets = item.targets if isinstance(item, ast.Assign) else [item.target]
      if any(
          isinstance(target, ast.Name)
          and target.id == 'PREFLIGHT_SCRIPT_VERSION' for target in targets
      ):
        try:
          assignments.append(ast.literal_eval(item.value))
        except (ValueError, TypeError) as error:
          raise AnalysisError(
              'External preflight version assignment is not literal.'
          ) from error
  if (
      freeze_evidence.get('path') != str(_FREEZE_PATH.resolve())
      or freeze_evidence.get('sha256') != start['freeze_sha256']
      or freeze_evidence.get('size_bytes') != _FREEZE_PATH.stat().st_size
      or freeze_evidence.get('external_freeze_authorization')
      != start['external_freeze_authorization']
      or freeze_evidence.get('prior_v3_3_4_3_consumed_preflight_prefix')
      != start['prior_v3_3_4_3_consumed_preflight_prefix']
      or freeze_evidence.get(
          'prior_v3_3_4_3_consumed_preflight_prefix_content_binding'
      ) != start[
          'prior_v3_3_4_3_consumed_preflight_prefix_content_binding'
      ]
      or freeze_evidence.get('prior_v3_3_4_4_consumed_preflight_prefix')
      != start['prior_v3_3_4_4_consumed_preflight_prefix']
      or freeze_evidence.get(
          'prior_v3_3_4_4_consumed_preflight_prefix_content_binding'
      ) != start[
          'prior_v3_3_4_4_consumed_preflight_prefix_content_binding'
      ]
      or version_proof.get('preflight_script_version')
      != PREFLIGHT_SCRIPT_VERSION
      or version_proof.get('freeze_equals_bootstrap') is not True
      or version_proof.get('bootstrap_equals_producer_literal') is not True
      or version_proof.get(
          'validated_before_allocation_or_registration'
      ) is not True
      or assignments != [PREFLIGHT_SCRIPT_VERSION]
      or freeze.get('preflight_script_version') != PREFLIGHT_SCRIPT_VERSION
      or producer_binding != {
          'path': str(producer_path.resolve()),
          'sha256': freeze['file_sha256'][producer_relative],
          'size_bytes': producer_path.stat().st_size,
      }
      or record.get('script_version') != PREFLIGHT_SCRIPT_VERSION
  ):
    raise AnalysisError('External preflight version/freeze proof changed.')
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
      'v3_3_4_5_runtime_environment',
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
      'prior_v3_3_4_3_consumed_preflight_prefix',
      'prior_v3_3_4_3_consumed_preflight_prefix_content_binding',
      'prior_v3_3_4_4_consumed_preflight_prefix',
      'prior_v3_3_4_4_consumed_preflight_prefix_content_binding',
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
      or successful.get('prior_v3_3_4_3_consumed_preflight_prefix')
      != start['prior_v3_3_4_3_consumed_preflight_prefix']
      or successful.get(
          'prior_v3_3_4_3_consumed_preflight_prefix_content_binding'
      ) != start[
          'prior_v3_3_4_3_consumed_preflight_prefix_content_binding'
      ]
      or successful.get('prior_v3_3_4_4_consumed_preflight_prefix')
      != start['prior_v3_3_4_4_consumed_preflight_prefix']
      or successful.get(
          'prior_v3_3_4_4_consumed_preflight_prefix_content_binding'
      ) != start[
          'prior_v3_3_4_4_consumed_preflight_prefix_content_binding'
      ]
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


def _validate_manifest_v3345(
    run_dir: Path, value: Any, *, cases: Mapping[int, Mapping[str, Any]],
    runner_pid: int, source_binding: Mapping[str, Any], object_binding: Mapping[str, Any] | None,
) -> tuple[list[tuple[int, int]], dict[str, Any] | None]:
  manifest = _exact_keys(value, _MANIFEST_KEYS, 'RAW_MANIFEST')
  _finite(manifest.get('created_at_unix_s'), 'RAW_MANIFEST.created_at_unix_s')
  if (
      manifest.get('schema_version') != 'v3.3.4.5-raw-manifest-v1'
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
      'schema_version': 'v3.3.4.5-failed-current-v1',
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
      'schema_version': 'v3.3.4.5-program-signature-attestation-v1',
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
      != 'v3.3.4.5-program-signature-attestation-v1'
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
  compiled_text = _read_text_no_follow(
      run_dir / expected_paths['compiled_hlo'], 'diagnostic compiled HLO',
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


def _validate_compiler_v3345(
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
  compiled_text = _read_text_no_follow(
      run_dir / expected_paths['compiled_hlo'], 'compiled HLO',
  )
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
  compiled_text = _read_text_no_follow(
      run_dir / 'compiler/eight_row/graph.compiled.hlo.txt',
      'nonpublication compiled HLO',
  )
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
  if len(node) != 62:
    raise AnalysisError('NONPUBLICATION terminal is not exactly 62 keys.')
  _validate_embedded_consumed_prefix(
      node, start, label='NONPUBLICATION_TERMINAL_FAILURE'
  )
  stage = node.get('failure_stage')
  if stage not in NONPUBLICATION_FAILURE_STAGES:
    raise AnalysisError('NONPUBLICATION terminal failure stage changed.')
  diagnostic_stage = stage in {
      'source_program_gate_derivation_for_diagnostic_failure',
      'diagnostic_failure_record_construction',
  }
  expected = {
      'schema_version': 'v3.3.4.5-nonpublication-terminal-v1',
      'status': 'incomplete_nonpublication_infrastructure_failure',
      'stop_reason': 'post_compile_nonpublication_infrastructure_failure',
      'attempt_id': ATTEMPT_ID, 'script_version': 'v3.3.4.5',
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
    compiled_hlo = _read_text_no_follow(
        run_dir / 'compiler/eight_row/graph.compiled.hlo.txt',
        'nonpublication triggering compiled HLO',
    )
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
  _assert_cpu_only('v3.3.4.5 analyzer entry')
  run_dir = run_dir.resolve()
  bundle_root = _REPO_ROOT if bundle_root is None else bundle_root.resolve()
  if run_dir == _RUN_DIR.resolve():
    _validate_active_analysis_attempt(
        run_dir, token=_attempt_token, started_sha256=_attempt_started_sha256,
        rehash_run_artifacts=False,
    )
  if not run_dir.is_dir() or run_dir.is_symlink():
    raise AnalysisError('v3.3.4.5 run directory is absent or unsafe.')
  freeze, freeze_sha, prior333, original_manifest, _unused, prior331 = (
      _validate_freeze_v3345(
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
  start = _validate_start_v3345(
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
    _assert_cpu_only('v3.3.4.5 nonpublication archive exit')
    return _result_v3345(
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
    _assert_cpu_only('v3.3.4.5 terminal-failure archive exit')
    return _result_v3345(
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
    _exact_keys(terminal, _POST_START_FAILURE_KEYS, 'POST_START_PROVENANCE_FAILURE')
    _validate_embedded_consumed_prefix(
        terminal, start, label='POST_START_PROVENANCE_FAILURE'
    )
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
    _assert_cpu_only('v3.3.4.5 post-START archive exit')
    return _result_v3345(
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
  pairs, failed_binding = _validate_manifest_v3345(
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
  compiler = _validate_compiler_v3345(run_dir, completion, freeze, start)
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
  _assert_cpu_only('v3.3.4.5 analyzer exit')
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
  return _result_v3345(
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


def _result_v3345(
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
  consumed_prefix = _expected_consumed_v3343_prefix()
  consumed_prefix_binding = _content_binding(consumed_prefix)
  consumed_v3344_prefix = _expected_consumed_v3344_prefix()
  consumed_v3344_prefix_binding = _content_binding(consumed_v3344_prefix)
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
          'current_132_source_rows_exact': True,
          'historical_96_source_rows_exact': True,
          'git_head_exact': True, 'tracked_clean': True,
          'external_freeze_authorization_exact': True,
          'prior_v3_3_3_exact': bool(prior333),
          'prior_v3_3_3_1_exact': bool(prior331),
          'old_analyzer_paths_absent': True,
          'pre_start_exact': True, 'post_start_exact': True,
          'final_exact': True,
          'prior_v3_3_4_3_consumed_preflight_prefix_exact': True,
          'prior_v3_3_4_4_consumed_preflight_prefix_exact': True,
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
      'prior_v3_3_4_3_consumed_preflight_prefix': consumed_prefix,
      'prior_v3_3_4_3_consumed_preflight_prefix_content_binding': (
          consumed_prefix_binding
      ),
      'prior_v3_3_4_4_consumed_preflight_prefix': consumed_v3344_prefix,
      'prior_v3_3_4_4_consumed_preflight_prefix_content_binding': (
          consumed_v3344_prefix_binding
      ),
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
      'prior_v3_3_4_3_consumed_preflight_prefix',
      'prior_v3_3_4_3_consumed_preflight_prefix_content_binding',
      'prior_v3_3_4_4_consumed_preflight_prefix',
      'prior_v3_3_4_4_consumed_preflight_prefix_content_binding',
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
    'prior_v3_3_4_3_consumed_preflight_prefix',
    'prior_v3_3_4_3_consumed_preflight_prefix_content_binding',
    'prior_v3_3_4_4_consumed_preflight_prefix',
    'prior_v3_3_4_4_consumed_preflight_prefix_content_binding',
}


def _validate_final_analysis(value: Any) -> dict[str, Any]:
  node = _exact_keys(value, _ANALYSIS_KEYS, 'ANALYSIS result')
  if len(node) != 27:
    raise AnalysisError('ANALYSIS does not have the exact 27-key schema.')
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
  source = _exact_keys(node.get('source_and_prior_audit'), {
      'current_132_source_rows_exact', 'historical_96_source_rows_exact',
      'git_head_exact', 'tracked_clean',
      'external_freeze_authorization_exact', 'prior_v3_3_3_exact',
      'prior_v3_3_3_1_exact', 'old_analyzer_paths_absent',
      'pre_start_exact', 'post_start_exact', 'final_exact',
      'prior_v3_3_4_3_consumed_preflight_prefix_exact',
      'prior_v3_3_4_4_consumed_preflight_prefix_exact',
  }, 'ANALYSIS.source_and_prior_audit')
  if len(source) != 13 or any(value is not True for value in source.values()):
    raise AnalysisError('ANALYSIS source/prior audit changed.')
  freeze = _read_json(_FREEZE_PATH, 'ANALYSIS result prefix freeze')
  _validate_embedded_consumed_prefix(node, freeze, label='ANALYSIS result')
  _validate_analysis_publication_audit(node.get('publication_audit'))
  return dict(node)


def render_markdown(result: Mapping[str, Any]) -> str:
  lines = [
      '# OpenSplice v3.3.4.5 OOD sidecar structural audit', '',
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
      'The consumed v3.3.4.3 external-preflight version-mismatch prefix was',
      'independently revalidated as an immutable, directory-only, non-cache',
      'input before this structural archive was read.', '',
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
    raise FileExistsError('v3.3.4.5 analysis output already exists; never overwrite.')
  _create_append_only_directory(
      _ANALYSIS_DIR, root_role='analysis_output',
      first_final_relative_path='RESULT.md',
      first_artifact_role='analysis_result_markdown',
  )
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


def _create_append_only_directory(
    path: Path, *, root_role: str, first_final_relative_path: str,
    first_artifact_role: str,
) -> None:
  if path.exists() or path.is_symlink():
    raise FileExistsError(f'Append-only directory already exists: {path}.')
  observed = ensure_publication_directory(
      root_role, first_final_relative_path, first_artifact_role
  )
  if observed.resolve() != path.resolve():
    raise AnalysisError('Publication helper returned the wrong frozen root.')


def _publish_new_bytes(
    path: Path, payload: bytes, *, root_role: str, root: Path,
    artifact_role: str,
) -> dict[str, Any]:
  """Delegates to the sole frozen v3.3.4.5 publication implementation."""
  if path.parent.is_symlink() or not path.parent.is_dir():
    raise AnalysisError(f'Append-only parent is unsafe: {path.parent}.')
  if root.is_symlink() or not root.is_dir():
    raise AnalysisError(f'Append-only root is unsafe: {root}.')
  if path.exists() or path.is_symlink():
    raise FileExistsError(f'Append-only artifact already exists: {path}.')
  result = publish_bytes(
      root_role, path.relative_to(root).as_posix(), payload, artifact_role,
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
      != f'.v33451.tmp.{runner_pid}.{ordinal:06d}.{nonce}'
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
    raise FileExistsError('v3.3.4.5 analysis/attempt exists; never resume or retry.')
  _assert_predecessor_v334_paths_absent('analysis precheck')
  freeze, freeze_sha, prior333, _manifest, _unused, prior331 = (
      _validate_freeze_v3345(run_dir, bundle_root=bundle_root)
  )
  start = _validate_start_v3345(
      run_dir, freeze, freeze_sha, prior333=prior333, prior331=prior331
  )
  consumed_prefix, consumed_prefix_binding = _validate_consumed_v3343_prefix(
      freeze, label='analysis precheck'
  )
  consumed_v3344_prefix, consumed_v3344_prefix_binding = (
      _validate_consumed_v3344_prefix(freeze, label='analysis precheck')
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
  _strict_regular(analyzer_path, 'v3.3.4.5 analyzer')
  _strict_regular(_TEST_PATH, 'v3.3.4.5 analyzer test')
  return {
      'freeze_sha256': freeze_sha,
      'git_head': start['git_head'],
      'external_freeze_authorization': dict(
          start['external_freeze_authorization']
      ),
      'analyzer_binding': _absolute_binding(analyzer_path),
      'test_binding': _absolute_binding(_TEST_PATH),
      'run_terminal_binding': _absolute_binding(terminal_path),
      'prior_v3_3_4_3_consumed_preflight_prefix': consumed_prefix,
      'prior_v3_3_4_3_consumed_preflight_prefix_content_binding': (
          consumed_prefix_binding
      ),
      'prior_v3_3_4_4_consumed_preflight_prefix': consumed_v3344_prefix,
      'prior_v3_3_4_4_consumed_preflight_prefix_content_binding': (
          consumed_v3344_prefix_binding
      ),
  }


_ANALYSIS_STARTED_KEYS = {
    'status', 'analysis_version', 'attempt_id', 'acknowledgement', 'git_head',
    'freeze_sha256', 'external_freeze_authorization', 'analyzer_binding',
    'test_binding', 'run_root', 'run_terminal_binding', 'fresh_output_dir',
    'old_analyzer_destinations_absent', 'started_at_unix_s',
    'prior_v3_3_4_3_consumed_preflight_prefix',
    'prior_v3_3_4_3_consumed_preflight_prefix_content_binding',
    'prior_v3_3_4_4_consumed_preflight_prefix',
    'prior_v3_3_4_4_consumed_preflight_prefix_content_binding',
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
      'acknowledgement': '--acknowledge-structural-only-v3-3-4-5',
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
      'prior_v3_3_4_3_consumed_preflight_prefix': copy.deepcopy(
          precheck['prior_v3_3_4_3_consumed_preflight_prefix']
      ),
      'prior_v3_3_4_3_consumed_preflight_prefix_content_binding': dict(
          precheck[
              'prior_v3_3_4_3_consumed_preflight_prefix_content_binding'
          ]
      ),
      'prior_v3_3_4_4_consumed_preflight_prefix': copy.deepcopy(
          precheck['prior_v3_3_4_4_consumed_preflight_prefix']
      ),
      'prior_v3_3_4_4_consumed_preflight_prefix_content_binding': dict(
          precheck[
              'prior_v3_3_4_4_consumed_preflight_prefix_content_binding'
          ]
      ),
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
      'acknowledgement': '--acknowledge-structural-only-v3-3-4-5',
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
  freeze = _read_json(_FREEZE_PATH, 'active analysis START prefix freeze')
  _validate_embedded_consumed_prefix(
      value, freeze, label='ANALYSIS_ATTEMPT_STARTED'
  )
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
  _validate_consumed_v3343_prefix(
      freeze, label=f'analysis TOCTOU {label}'
  )
  _validate_consumed_v3344_prefix(
      freeze, label=f'analysis TOCTOU {label}'
  )
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
  started = _read_json(
      _ANALYSIS_ATTEMPT_DIR / 'ANALYSIS_ATTEMPT_STARTED.json',
      'analysis START for FAILURE',
  )
  _exact_keys(started, _ANALYSIS_STARTED_KEYS, 'analysis START for FAILURE')
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
      'prior_v3_3_4_3_consumed_preflight_prefix': copy.deepcopy(
          started['prior_v3_3_4_3_consumed_preflight_prefix']
      ),
      'prior_v3_3_4_3_consumed_preflight_prefix_content_binding': dict(
          started[
              'prior_v3_3_4_3_consumed_preflight_prefix_content_binding'
          ]
      ),
      'prior_v3_3_4_4_consumed_preflight_prefix': copy.deepcopy(
          started['prior_v3_3_4_4_consumed_preflight_prefix']
      ),
      'prior_v3_3_4_4_consumed_preflight_prefix_content_binding': dict(
          started[
              'prior_v3_3_4_4_consumed_preflight_prefix_content_binding'
          ]
      ),
  }
  _exact_keys(result, {
      'status', 'attempt_id', 'analysis_attempt_start_binding', 'type',
      'message', 'traceback', 'raw_values_read',
      'scientific_analysis_performed', 'output_dir_state',
      'publication_failure', 'temporary_orphan_bindings',
      'durability_uncertain_final_bindings', 'preexisting_entry_states',
      'no_new_entry_failure', 'failed_at_unix_s',
      'prior_v3_3_4_3_consumed_preflight_prefix',
      'prior_v3_3_4_3_consumed_preflight_prefix_content_binding',
      'prior_v3_3_4_4_consumed_preflight_prefix',
      'prior_v3_3_4_4_consumed_preflight_prefix_content_binding',
  }, 'ANALYSIS_FAILURE')
  return result


def _current_analysis_publication_audits(
    publication_failure: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
  failure_root = (
      None if publication_failure is None else publication_failure['root_role']
  )
  attempt = publication_audit(
      'analysis_attempt',
      publication_failure if failure_root == 'analysis_attempt' else None,
  )
  output = publication_audit(
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
      '--acknowledge-structural-only-v3-3-4-5', action='store_true',
      help='Acknowledge the frozen structural-only/no-science boundary.',
  )
  args = parser.parse_args()
  if not args.acknowledge_structural_only_v3_3_4_5:
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
      _ANALYSIS_ATTEMPT_DIR, root_role='analysis_attempt',
      first_final_relative_path='ANALYSIS_ATTEMPT_STARTED.json',
      first_artifact_role='analysis_attempt_start',
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


_V33451_FREEZE_KEYS = frozenset({
    'schema_version', 'analysis_version', 'attempt_id',
    'acknowledgement_token', 'amendment_path', 'amendment_sha256',
    'amendment_commit', 'prior_model_head', 'prior_model_freeze_binding',
    'source_inventory_contract', 'immutable_model_artifact_contract',
    'consumed_analyzer_failure', 'consumed_analyzer_failure_content_binding',
    'prior_cache_contract', 'prior_cache_contract_content_binding',
    'analysis_attempt_dir', 'analysis_dir', 'publication_contract',
    'record_contracts', 'claim_boundary',
})
_V33451_START_KEYS = frozenset({
    'status', 'schema_version', 'analysis_version', 'attempt_id',
    'acknowledgement', 'git_head', 'external_freeze_authorization',
    'freeze_binding', 'analyzer_binding', 'test_binding', 'shell_binding',
    'generator_binding', 'amendment_binding', 'run_terminal_binding',
    'source_inventory_attestation', 'immutable_input_audit',
    'consumed_analyzer_failure',
    'consumed_analyzer_failure_content_binding', 'prior_cache_audit',
    'prior_cache_audit_content_binding', 'fresh_paths', 'started_at_unix_s',
})
_V33451_ANALYSIS_KEYS = frozenset({
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
})
_V33451_COMPLETE_KEYS = frozenset({
    'status', 'schema_version', 'analysis_version', 'attempt_id',
    'start_binding', 'analysis_binding', 'result_binding',
    'attempt_tree_before_complete', 'output_tree_complete',
    'publication_audit', 'completed_at_unix_s',
})
_V33451_FAILURE_KEYS = frozenset({
    'status', 'schema_version', 'analysis_version', 'attempt_id',
    'start_binding', 'failure', 'failure_phase', 'raw_access_reached',
    'analysis_output_state', 'attempt_output_state', 'publication_audit',
    'old_destinations_absent', 'failed_at_unix_s',
})
_V33451_OUTPUT_STATE_KEYS = frozenset({
    'state', 'root_role', 'root_lstat', 'regular_final_bindings',
    'temporary_orphan_bindings', 'durability_uncertain_final_bindings',
    'preexisting_entry_states', 'directory_paths', 'directory_tree_sha256',
    'directory_file_tree_sha256', 'file_tree_sha256',
    'entry_state_tree_sha256', 'publication_failure',
})
_V33451_FAILURE_PHASES = (
    'post_start_source_gate', 'post_start_prior_cache_gate',
    'model_input_rehash', 'structural_terminal_audit',
    'result_publication', 'analysis_publication', 'final_toctou',
    'complete_publication',
)
_V33451_CLAIM_BOUNDARY = {
    'structural_only': True, 'no_biological_claim': True,
    'no_scientific_summary': True, 'no_normalization': True,
    'no_shapley': True, 'no_interaction': True, 'no_resolution': True,
    'no_nomination': True, 'combined_analysis_permitted': False,
    'future_protocol_required': True,
}
_V33451_ACTIVE_TOKEN = object()


def _v33451_canonical_binding(value: Any) -> dict[str, Any]:
  payload = json.dumps(
      value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
      allow_nan=False,
  ).encode('utf-8')
  return {'sha256': hashlib.sha256(payload).hexdigest(), 'size_bytes': len(payload)}


def _v33451_file_binding(path: Path, *, absolute: bool = False) -> dict[str, Any]:
  _strict_regular(path, f'v3.3.4.5.1 bound file {path}')
  observed = path.lstat()
  result = {
      'sha256': _sha256_no_follow(path, observed),
      'size_bytes': observed.st_size,
  }
  if absolute:
    result = {'path': str(path.resolve()), **result}
  return result


def _v33451_tree_binding(
    root: Path, *, expected_files: set[str] | None = None,
    expected_directories: set[str] | None = None,
    label: str = 'immutable tree',
) -> dict[str, Any]:
  try:
    root_status = root.lstat()
  except FileNotFoundError as error:
    raise AnalysisError(f'Immutable tree root is absent: {root}.') from error
  if stat.S_ISLNK(root_status.st_mode) or not stat.S_ISDIR(root_status.st_mode):
    raise AnalysisError(f'Immutable tree root is unsafe: {root}.')
  directories = ['.']
  files: dict[str, dict[str, Any]] = {}
  registered_role = next((
      role for role, publication_root in _PUBLICATION_ROOTS.items()
      if publication_root == root and role in _PUBLICATION_DIRECTORIES
  ), None)
  if registered_role is not None:
    root_fd, expected_dev, expected_ino = _PUBLICATION_DIRECTORIES[
        registered_role
    ]
    descriptor_status = os.fstat(root_fd)
    if (
        (descriptor_status.st_dev, descriptor_status.st_ino)
        != (expected_dev, expected_ino)
        or (root_status.st_dev, root_status.st_ino)
        != (expected_dev, expected_ino)
        or stat.S_IMODE(descriptor_status.st_mode) != 0o700
    ):
      raise AnalysisError('Registered publication root identity changed.')

    def walk(directory_fd: int, prefix: str = '') -> None:
      for basename in sorted(os.listdir(directory_fd)):
        if '/' in basename or basename in {'.', '..'}:
          raise AnalysisError('Unsafe publication tree entry name.')
        relative = basename if not prefix else f'{prefix}/{basename}'
        state_value = _publication_entry_at(directory_fd, basename)
        if state_value['entry_type'] == 'directory':
          if expected_directories is not None and relative not in expected_directories:
            raise AnalysisError(f'{label} contains an unexpected directory.')
          directories.append(relative)
          child_fd = os.open(
              basename,
              os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
              dir_fd=directory_fd,
          )
          try:
            child_status = os.fstat(child_fd)
            if (
                (child_status.st_dev, child_status.st_ino)
                != (state_value['st_dev'], state_value['st_ino'])
            ):
              raise AnalysisError('Publication directory inode changed.')
            walk(child_fd, relative)
          finally:
            os.close(child_fd)
        elif state_value['entry_type'] == 'regular':
          if expected_files is not None and relative not in expected_files:
            raise AnalysisError(f'{label} contains an unexpected file.')
          files[relative] = _publication_binding_from_state(state_value)
        else:
          raise AnalysisError(
              f'Immutable tree contains unsafe entry: {relative}.'
          )
    walk(root_fd)
  else:
    for entry in sorted(root.rglob('*')):
      relative = entry.relative_to(root).as_posix()
      status = entry.lstat()
      if stat.S_ISLNK(status.st_mode):
        raise AnalysisError(f'Immutable tree contains symlink: {relative}.')
      if stat.S_ISDIR(status.st_mode):
        if expected_directories is not None and relative not in expected_directories:
          raise AnalysisError(f'{label} contains an unexpected directory.')
        directories.append(relative)
        continue
      if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
        raise AnalysisError(f'Immutable tree contains unsafe entry: {relative}.')
      if expected_files is not None and relative not in expected_files:
        raise AnalysisError(f'{label} contains an unexpected file.')
      files[relative] = {
          'sha256': _sha256_no_follow(entry, status),
          'size_bytes': status.st_size,
          'mode': _publication_mode(status.st_mode), 'st_dev': status.st_dev,
          'st_ino': status.st_ino, 'st_nlink': status.st_nlink,
      }
  if (
      expected_files is not None and set(files) != expected_files
      or expected_directories is not None
      and set(directories) != expected_directories
  ):
    raise AnalysisError(f'{label} exact membership changed during hashing.')
  if expected_files is not None and expected_directories is not None:
    _v33451_assert_tree_membership(
        root, {
            'directory_paths': sorted(expected_directories),
            'file_bindings': {relative: {} for relative in expected_files},
        }, label,
    )
  directories = sorted(directories)
  directory_file = hashlib.sha256()
  for relative in directories:
    directory_file.update(b'D\0')
    directory_file.update(relative.encode('utf-8'))
    directory_file.update(b'\0')
  for relative in sorted(files):
    directory_file.update(b'F\0')
    directory_file.update(relative.encode('utf-8'))
    directory_file.update(b'\0')
    directory_file.update(bytes.fromhex(files[relative]['sha256']))
  result = {
      'root': str(root.resolve()), 'file_count': len(files),
      'directory_count': len(directories), 'file_bindings': files,
      'file_tree_sha256': _binding_map_digest(files),
      'directory_paths': directories,
      'directory_tree_sha256': _directory_digest(directories),
      'directory_file_tree_sha256': directory_file.hexdigest(),
  }
  _exact_keys(result, {
      'root', 'file_count', 'directory_count', 'file_bindings',
      'file_tree_sha256', 'directory_paths', 'directory_tree_sha256',
      'directory_file_tree_sha256',
  }, 'immutable tree binding')
  return result


def _v33451_old_destinations() -> tuple[Path, ...]:
  return (
      _OLD_V3345_ANALYSIS_ATTEMPT_DIR, _OLD_V3345_ANALYSIS_DIR,
      _PRIOR_ANALYZER_ATTEMPT_DIR, _PRIOR_ANALYZER_OUTPUT_DIR,
  )


def _v33451_fresh_paths() -> dict[str, str]:
  targets = {
      'old_v3345_attempt': _OLD_V3345_ANALYSIS_ATTEMPT_DIR,
      'old_v3345_output': _OLD_V3345_ANALYSIS_DIR,
      'old_v333_attempt': _PRIOR_ANALYZER_ATTEMPT_DIR,
      'old_v333_output': _PRIOR_ANALYZER_OUTPUT_DIR,
      'new_attempt': _ANALYSIS_ATTEMPT_DIR,
      'new_output': _ANALYSIS_DIR,
  }
  result = {}
  for name, path in targets.items():
    try:
      os.lstat(path)
    except FileNotFoundError:
      result[name] = 'absent'
    else:
      raise AnalysisError(f'Analyzer destination is not absent: {path}.')
  return result


def _v33451_require_old_absent() -> None:
  for path in _v33451_old_destinations():
    if path.exists() or path.is_symlink():
      raise AnalysisError(f'Permanently absent analyzer path appeared: {path}.')


_V33451_FAILURE_TRACEBACK = '''Traceback (most recent call last):
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


def _v33451_expected_consumed_failure() -> dict[str, Any]:
  old_shell = _HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_4_5.sh'
  return {
      'captured_at_unix_s': None, 'chunk_id': '77e144',
      'command': [
          str(old_shell.resolve()),
          '--acknowledge-structural-only-v3-3-4-5',
      ],
      'destination_states': {
          str(path.resolve()): 'absent' for path in _v33451_old_destinations()
      },
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
      'stderr_text': _V33451_FAILURE_TRACEBACK,
      'stdout': {
          'persisted_to_filesystem': False, 'sha256': EMPTY_SHA256,
          'size_bytes': 0,
      },
      'wall_time_seconds': 1.95131362,
  }


def _v33451_validate_consumed_failure(
    value: Any, binding: Any,
) -> dict[str, Any]:
  expected = _v33451_expected_consumed_failure()
  if value != expected or binding != _v33451_canonical_binding(expected):
    raise AnalysisError('Consumed pre-START analyzer failure changed.')
  return copy.deepcopy(expected)


def _v33451_publication_contract() -> dict[str, Any]:
  return {
      'schema_version': _V33451_PUBLICATION_SCHEMA_VERSION,
      'method': PUBLICATION_METHOD,
      'temp_name_regex': (
          r'^\.v33451\.tmp\.[1-9][0-9]*\.[0-9]{6}\.[0-9a-f]{32}$'
      ),
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
      'output_state_keys': sorted(_V33451_OUTPUT_STATE_KEYS),
      'entry_state_keys': sorted(ENTRY_STATE_KEYS),
      'failure_stages': sorted(PUBLICATION_FAILURE_STAGES),
  }


def _v33451_record_contracts() -> dict[str, Any]:
  return {
      'start_keys': sorted(_V33451_START_KEYS),
      'analysis_keys': sorted(_V33451_ANALYSIS_KEYS),
      'complete_keys': sorted(_V33451_COMPLETE_KEYS),
      'failure_keys': sorted(_V33451_FAILURE_KEYS),
      'publication_success_keys': sorted(PUBLICATION_SUCCESS_KEYS),
      'publication_failure_keys': sorted(PUBLICATION_FAILURE_KEYS),
      'publication_audit_keys': sorted(PUBLICATION_AUDIT_KEYS),
      'output_state_keys': sorted(_V33451_OUTPUT_STATE_KEYS),
      'failure_phase_values': sorted(_V33451_FAILURE_PHASES),
  }


def _v33451_validate_authorization(
    authorization: Mapping[str, Any], freeze: Mapping[str, Any],
) -> dict[str, Any]:
  node = _exact_keys(authorization, {
      'git_head', 'freeze_path', 'freeze_sha256', 'freeze_size_bytes',
      'live_equals_git_show', 'tracked_clean', 'authorization_source',
  }, 'external freeze authorization')
  head = subprocess.check_output(
      ('git', '-C', str(_REPO_ROOT), 'rev-parse', 'HEAD'), text=True
  ).strip()
  try:
    subprocess.check_call(
        ('git', '-C', str(_REPO_ROOT), 'diff', '--quiet', 'HEAD', '--')
    )
    relative = _ANALYSIS_FREEZE_PATH.relative_to(_REPO_ROOT).as_posix()
    head_bytes = subprocess.check_output(
        ('git', '-C', str(_REPO_ROOT), 'show', f'{head}:{relative}')
    )
    mode_line = subprocess.check_output(
        ('git', '-C', str(_REPO_ROOT), 'ls-tree', head, '--', relative),
        text=True,
    ).strip()
  except subprocess.CalledProcessError as error:
    raise AnalysisError('Launch HEAD/freeze is not tracked-clean/exact.') from error
  live_binding = _v33451_file_binding(_ANALYSIS_FREEZE_PATH)
  live_status = _ANALYSIS_FREEZE_PATH.lstat()
  if (
      not re.fullmatch(r'[0-9a-f]{40}', head)
      or node.get('git_head') != head
      or node.get('freeze_path') != str(_ANALYSIS_FREEZE_PATH.resolve())
      or node.get('freeze_sha256') != live_binding['sha256']
      or node.get('freeze_size_bytes') != live_binding['size_bytes']
      or node.get('live_equals_git_show') is not True
      or node.get('tracked_clean') is not True
      or node.get('authorization_source') != 'external_post_commit_audit'
      or len(head_bytes) != live_binding['size_bytes']
      or hashlib.sha256(head_bytes).hexdigest() != live_binding['sha256']
      or not stat.S_ISREG(live_status.st_mode)
      or stat.S_IMODE(live_status.st_mode) != 0o644
      or live_status.st_nlink != 1
      or not mode_line.startswith(f'100644 blob ')
      or not mode_line.endswith(f'\t{relative}')
  ):
    raise AnalysisError('External analysis-freeze authorization changed.')
  del freeze
  return dict(node)


def _v33451_expected_inherited_source_rows() -> list[dict[str, Any]]:
  if _v33451_file_binding(_FREEZE_PATH)['sha256'] != (
      '2f4eaf1366dcb42b8f89a386e8201b3f2ba0b9f8ae5ef02409436492666d8366'
  ):
    raise AnalysisError('Authenticated inherited source freeze changed.')
  inherited_freeze = _read_json(
      _FREEZE_PATH, 'authenticated v3.3.4.5 source freeze'
  )
  inherited_contract = _exact_keys(
      inherited_freeze.get('source_inventory_contract'), {
          'source_row_count', 'rows', 'prospective_upstream_source_file_count',
          'loaded_scientific_module_contract',
      }, 'inherited source inventory',
  )
  inherited_rows = inherited_contract.get('rows')
  if (
      inherited_contract.get('source_row_count') != 132
      or not isinstance(inherited_rows, list) or len(inherited_rows) != 132
  ):
    raise AnalysisError('Inherited source inventory is not exactly 132 rows.')
  return [
      {**copy.deepcopy(row), 'authority_commit': MODEL_SOURCE_COMMIT}
      for row in inherited_rows
  ]


def _v33451_require_inherited_source_rows(rows: Sequence[Any]) -> None:
  expected = _v33451_expected_inherited_source_rows()
  observed = [
      copy.deepcopy(row) for row in rows
      if isinstance(row, Mapping)
      and row.get('authority_commit') == MODEL_SOURCE_COMMIT
  ]
  if observed != expected:
    raise AnalysisError('Inherited 132-row source inventory changed.')


def _v33451_validate_source_inventory(
    contract: Any, authorization: Mapping[str, Any],
) -> dict[str, Any]:
  node = _exact_keys(contract, {
      'row_count', 'rows', 'authority_partitions', 'source_authority_head',
      'source_authority_tree_exact', 'all_rows_authority_exact',
      'all_rows_live_at_generation_exact', 'tree_sha256',
  }, 'analysis freeze source inventory')
  rows = node.get('rows')
  if node.get('row_count') != 137 or not isinstance(rows, list) or len(rows) != 137:
    raise AnalysisError('Analysis source inventory is not exactly 137 rows.')
  row_paths = [row.get('path') for row in rows if isinstance(row, Mapping)]
  if len(row_paths) != 137 or row_paths != sorted(row_paths):
    raise AnalysisError('Analysis source inventory rows are not POSIX-sorted.')
  expected_inherited_rows = _v33451_expected_inherited_source_rows()
  _v33451_require_inherited_source_rows(rows)
  partitions = _exact_keys(node.get('authority_partitions'), {
      'inherited_132', 'amendment', 'new_implementation_4',
  }, 'source authority partitions')
  expected_partition = {
      'inherited_132': (MODEL_SOURCE_COMMIT, 132),
      'amendment': (ANALYSIS_AMENDMENT_COMMIT, 1),
      'new_implementation_4': (node.get('source_authority_head'), 4),
  }
  partition_paths: dict[str, str] = {}
  for name, (commit, count) in expected_partition.items():
    part = _exact_keys(
        partitions.get(name), {'authority_commit', 'row_count', 'paths'},
        f'source partition {name}',
    )
    if (
        part.get('authority_commit') != commit
        or part.get('row_count') != count
        or not isinstance(part.get('paths'), list)
        or part['paths'] != sorted(part['paths'])
        or len(part['paths']) != count
    ):
      raise AnalysisError(f'Source authority partition changed: {name}.')
    for relative in part['paths']:
      if relative in partition_paths:
        raise AnalysisError('Source authority paths overlap.')
      partition_paths[relative] = commit
  if set(partition_paths) != {row.get('path') for row in rows if isinstance(row, Mapping)}:
    raise AnalysisError('Source authority partition membership changed.')
  if partitions['inherited_132']['paths'] != [
      row['path'] for row in expected_inherited_rows
  ]:
    raise AnalysisError('Inherited 132-row source partition changed.')
  if node.get('source_authority_head') != partitions['new_implementation_4']['authority_commit']:
    raise AnalysisError('Source authority head changed.')
  if any(node.get(name) is not True for name in (
      'source_authority_tree_exact', 'all_rows_authority_exact',
      'all_rows_live_at_generation_exact',
  )):
    raise AnalysisError('Frozen source authority audit is not true.')
  if _v33451_canonical_binding(rows)['sha256'] != node.get('tree_sha256'):
    raise AnalysisError('Frozen source inventory tree digest changed.')
  launch_head = authorization['git_head']
  implementation_paths = sorted(
      path.relative_to(_REPO_ROOT).as_posix()
      for path in (Path(__file__), _TEST_PATH, _SHELL_PATH, _GENERATOR_PATH)
  )
  freeze_relative = _ANALYSIS_FREEZE_PATH.relative_to(_REPO_ROOT).as_posix()
  source_delta = subprocess.check_output(
      ('git', '-C', str(_REPO_ROOT), 'diff', '--name-status',
       ANALYSIS_AMENDMENT_COMMIT, node['source_authority_head']),
      text=True,
  ).splitlines()
  launch_delta = subprocess.check_output(
      ('git', '-C', str(_REPO_ROOT), 'diff', '--name-status',
       node['source_authority_head'], launch_head),
      text=True,
  ).splitlines()
  source_parents = subprocess.check_output(
      ('git', '-C', str(_REPO_ROOT), 'rev-list', '--parents', '-n', '1',
       node['source_authority_head']), text=True,
  ).split()
  launch_parents = subprocess.check_output(
      ('git', '-C', str(_REPO_ROOT), 'rev-list', '--parents', '-n', '1',
       launch_head), text=True,
  ).split()
  if (
      source_delta != [f'A\t{path}' for path in implementation_paths]
      or launch_delta != [f'A\t{freeze_relative}']
      or source_parents != [node['source_authority_head'], ANALYSIS_AMENDMENT_COMMIT]
      or launch_parents != [launch_head, node['source_authority_head']]
  ):
    raise AnalysisError('Three-commit source/freeze authority delta changed.')
  seen = set()
  for row_value in rows:
    row = _exact_keys(
        row_value, {'path', 'sha256', 'size_bytes', 'git_mode', 'authority_commit'},
        'analysis source row',
    )
    relative = row.get('path')
    if (
        not isinstance(relative, str) or relative in seen
        or row.get('authority_commit') != partition_paths.get(relative)
        or not _is_sha256(row.get('sha256'))
        or isinstance(row.get('size_bytes'), bool)
        or not isinstance(row.get('size_bytes'), int) or row['size_bytes'] < 0
        or row.get('git_mode') not in {'100644', '100755'}
    ):
      raise AnalysisError('Analysis source row is malformed.')
    seen.add(relative)
    path = _REPO_ROOT / relative
    _strict_regular(path, f'analysis source {relative}')
    observed_live = path.lstat()
    authority_sha = _git_blob_sha256(row['authority_commit'], relative)
    launch_sha = _git_blob_sha256(launch_head, relative)
    line = subprocess.check_output(
        ('git', '-C', str(_REPO_ROOT), 'ls-tree', launch_head, '--', relative),
        text=True,
    ).strip()
    mode = line.split()[0] if line else None
    authority_line = subprocess.check_output(
        ('git', '-C', str(_REPO_ROOT), 'ls-tree', row['authority_commit'],
         '--', relative), text=True,
    ).strip()
    authority_mode = authority_line.split()[0] if authority_line else None
    live_mode = (
        '100755' if stat.S_IMODE(observed_live.st_mode) & 0o111 else '100644'
    )
    if (
        _sha256_no_follow(path, observed_live) != row['sha256']
        or observed_live.st_size != row['size_bytes']
        or authority_sha != row['sha256'] or launch_sha != row['sha256']
        or mode != row['git_mode'] or authority_mode != row['git_mode']
        or live_mode != row['git_mode']
    ):
      raise AnalysisError(f'Analysis source authority changed: {relative}.')
  final_head = subprocess.check_output(
      ('git', '-C', str(_REPO_ROOT), 'rev-parse', 'HEAD'), text=True,
  ).strip()
  try:
    subprocess.check_call(
        ('git', '-C', str(_REPO_ROOT), 'diff', '--quiet', 'HEAD', '--')
    )
  except subprocess.CalledProcessError as error:
    raise AnalysisError('Launch HEAD changed during source audit.') from error
  if final_head != launch_head:
    raise AnalysisError('Launch HEAD changed during source audit.')
  return {
      'row_count': 137, 'rows': copy.deepcopy(rows),
      'authority_partitions': copy.deepcopy(partitions),
      'source_authority_head': node['source_authority_head'],
      'launch_git_head': launch_head, 'source_authority_tree_exact': True,
      'all_rows_authority_exact': True, 'all_rows_live_exact': True,
      'all_rows_launch_head_exact': True, 'launch_head_tracked_clean': True,
      'tree_sha256': node['tree_sha256'],
  }


def _v33451_assert_tree_membership(
    root: Path, expected: Mapping[str, Any], label: str,
) -> None:
  """Rejects every unlisted path before opening any artifact bytes."""
  try:
    root_status = root.lstat()
  except FileNotFoundError as error:
    raise AnalysisError(f'{label} root is absent.') from error
  if stat.S_ISLNK(root_status.st_mode) or not stat.S_ISDIR(root_status.st_mode):
    raise AnalysisError(f'{label} root is unsafe.')
  directories = ['.']
  files = []
  for entry in sorted(root.rglob('*')):
    relative = entry.relative_to(root).as_posix()
    observed_status = entry.lstat()
    if stat.S_ISLNK(observed_status.st_mode):
      raise AnalysisError(f'{label} contains a symlink: {relative}.')
    if stat.S_ISDIR(observed_status.st_mode):
      directories.append(relative)
    elif stat.S_ISREG(observed_status.st_mode):
      files.append(relative)
    else:
      raise AnalysisError(f'{label} contains a special entry: {relative}.')
  if (
      sorted(directories) != expected.get('directory_paths')
      or sorted(files) != sorted(expected.get('file_bindings', {}))
  ):
    raise AnalysisError(f'{label} exact structural membership changed.')


def _v33451_validate_immutable_contract(value: Any) -> dict[str, Any]:
  node = _exact_keys(value, {
      'run_root_binding', 'compiler_tree_binding', 'preflight_tree_binding',
      'external_cache_tree_binding', 'model_cache_tree_binding',
      'run_terminal_binding', 'raw_manifest_binding', 'old_analyzer_bundle',
  }, 'immutable model artifact contract')

  # This lstat-only pass is deliberately before every run-tree byte hash.
  # Unexpected raw/journal/confirmation files therefore fail without ever
  # being opened by the structural analyzer.
  literal_memberships = {
      'run_root_binding': (
          _RUN_DIR,
          {
              'ATTEMPT_STARTED.json', 'IMPORT_PROVENANCE.json',
              'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
              'IMPORT_PROVENANCE_PRE_MODEL.json', 'PROTOBUF_PROVENANCE.json',
              'RAW_MANIFEST.json', 'RUN_COMPLETE.json',
              'compiler/eight_row/COMPILER_DIAGNOSTIC_FAILURE.json',
              'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json',
              'compiler/eight_row/graph.compiled.hlo.txt',
              'compiler/eight_row/graph.pre_backend.hlo.txt',
              'compiler/eight_row/graph.stablehlo.mlir',
          },
          ['.', 'compiler', 'compiler/eight_row'],
          '960faf1675caaa0f3c9798f7b943998e91650b2a9359b26ca31ce2d417c2ce0b',
          '5331e5041b557a4324ba57d57179dc2bfd8e6ab981f0cdede5d4f126e94c2041',
      ),
      'compiler_tree_binding': (
          _RUN_DIR / 'compiler',
          {
              'eight_row/COMPILER_DIAGNOSTIC_FAILURE.json',
              'eight_row/PROGRAM_SIGNATURE_ATTESTATION.json',
              'eight_row/graph.compiled.hlo.txt',
              'eight_row/graph.pre_backend.hlo.txt',
              'eight_row/graph.stablehlo.mlir',
          },
          ['.', 'eight_row'],
          'b1094dfaddb0e8c6672b09a18e124af2a20a1a91c7ca817911c2f3fe4c0220a3',
          'bb042bf9a2cb34c61aae121733edce583cc2d747de1913b1d74f00b7a8de200c',
      ),
      'preflight_tree_binding': (
          _PREFLIGHT_DIR,
          {
              '.allocation.lock', '.preflight_0000.reserved',
              'preflight_0000.json', 'preflight_0000.stderr.log',
              'preflight_0000.stdout.log',
          },
          ['.'],
          'ae277eafa4f7f20bfa74c3a0a1bbaa0f51468cac945d29d0c49cab699738ecfd',
          'cc106b406da58ddd95611aef7e471f5a5cefd96e302ebb91ea4ef9e28a618c87',
      ),
      'external_cache_tree_binding': (
          _PREFLIGHT_CACHE_DIR,
          {
              '.v3345.tmp.2777420.000001.7a795e5eda1e9fcf14f19a8d62c7960f',
              'atomic_publication_probe_v3_3_4_5.txt',
          },
          ['.', 'triton', 'xdg'],
          '3bd7b53ba7ab1dae7161999ff907137f82ee6d7f322512a3221646f66bb1e975',
          'd040af81aa50fbe28e0523747355f84d851f36f39e586294b24dd994f69f66a0',
      ),
      'model_cache_tree_binding': (
          _MODEL_CACHE_DIR,
          {'xdg/matplotlib/fontlist-v3.11.0.json'},
          ['.', 'triton', 'xdg', 'xdg/matplotlib'],
          '487c67a6dbb251aca190ac9eda5d2425c3584febc9ad63e60d0812c7f2fb69ea',
          '51fe59713c301342bf5bb161f26b9e4ee6828e508b96e4dbd21c6efcdde1115e',
      ),
  }
  for tree_name, (
      root, files, directories, file_digest, directory_file_digest,
  ) in literal_memberships.items():
    frozen = node[tree_name]
    if (
        frozen.get('file_count') != len(files)
        or set(frozen.get('file_bindings', {})) != files
        or frozen.get('directory_paths') != directories
        or frozen.get('file_tree_sha256') != file_digest
        or frozen.get('directory_file_tree_sha256') != directory_file_digest
    ):
      raise AnalysisError(
          f'Frozen immutable allowlist changed before byte access: {tree_name}.'
      )
    _v33451_assert_tree_membership(
        root, {'directory_paths': directories,
               'file_bindings': {relative: {} for relative in files}},
        tree_name,
    )
  observed = {}
  for tree_name, (
      root, files, directories, _file_digest, _directory_file_digest,
  ) in literal_memberships.items():
    observed[tree_name] = _v33451_tree_binding(
        root, expected_files=set(files), expected_directories=set(directories),
        label=tree_name,
    )
  for name, binding in observed.items():
    if node.get(name) != binding:
      raise AnalysisError(f'Immutable artifact tree changed: {name}.')
  run = observed['run_root_binding']
  compiler = observed['compiler_tree_binding']
  preflight = observed['preflight_tree_binding']
  external_cache = observed['external_cache_tree_binding']
  model_cache = observed['model_cache_tree_binding']
  if (
      run['file_count'] != 12
      or run['directory_paths'] != ['.', 'compiler', 'compiler/eight_row']
      or run['file_tree_sha256']
      != '960faf1675caaa0f3c9798f7b943998e91650b2a9359b26ca31ce2d417c2ce0b'
      or run['directory_file_tree_sha256']
      != '5331e5041b557a4324ba57d57179dc2bfd8e6ab981f0cdede5d4f126e94c2041'
      or compiler['file_count'] != 5
      or compiler['directory_paths'] != ['.', 'eight_row']
      or compiler['file_tree_sha256']
      != 'b1094dfaddb0e8c6672b09a18e124af2a20a1a91c7ca817911c2f3fe4c0220a3'
      or compiler['directory_file_tree_sha256']
      != 'bb042bf9a2cb34c61aae121733edce583cc2d747de1913b1d74f00b7a8de200c'
      or preflight['file_count'] != 5
      or preflight['file_tree_sha256']
      != 'ae277eafa4f7f20bfa74c3a0a1bbaa0f51468cac945d29d0c49cab699738ecfd'
      or preflight['directory_file_tree_sha256']
      != 'cc106b406da58ddd95611aef7e471f5a5cefd96e302ebb91ea4ef9e28a618c87'
      or external_cache['file_count'] != 2
      or external_cache['directory_count'] != 3
      or external_cache['file_tree_sha256']
      != '3bd7b53ba7ab1dae7161999ff907137f82ee6d7f322512a3221646f66bb1e975'
      or external_cache['directory_file_tree_sha256']
      != 'd040af81aa50fbe28e0523747355f84d851f36f39e586294b24dd994f69f66a0'
      or model_cache['file_count'] != 1
      or model_cache['directory_count'] != 4
      or model_cache['file_tree_sha256']
      != '487c67a6dbb251aca190ac9eda5d2425c3584febc9ad63e60d0812c7f2fb69ea'
      or model_cache['directory_file_tree_sha256']
      != '51fe59713c301342bf5bb161f26b9e4ee6828e508b96e4dbd21c6efcdde1115e'
  ):
    raise AnalysisError('Immutable model artifact literal binding changed.')
  terminal = _v33451_file_binding(_RUN_DIR / 'RUN_COMPLETE.json', absolute=True)
  manifest = _v33451_file_binding(_RUN_DIR / 'RAW_MANIFEST.json', absolute=True)
  if (
      terminal != node.get('run_terminal_binding')
      or terminal['sha256']
      != 'fdbd0a1dc7d24145f88c5a009cc80d8904e57920e0c9584426e791373fae6d8f'
      or terminal['size_bytes'] != 43_760
      or manifest != node.get('raw_manifest_binding')
      or manifest['sha256']
      != '3ee95b22d483c7c4f234fbb75281e05e84f0be263b1ee670a94b2cd442d61136'
      or manifest['size_bytes'] != 1_562
  ):
    raise AnalysisError('Immutable model terminal/manifest binding changed.')
  bundle = _exact_keys(node.get('old_analyzer_bundle'), {
      'git_head', 'analyzer', 'test', 'shell', 'freeze',
  }, 'old analyzer bundle')
  paths = {
      'analyzer': _HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_4_5.py',
      'test': _HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_4_5_test.py',
      'shell': _HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_4_5.sh',
      'freeze': _FREEZE_PATH,
  }
  expected_sha = {
      'analyzer': '9320184c53ed6bc3b246443314d84c1f1543bbbf77aa10e3fff982bd5c18913a',
      'test': 'dcede28da855e3784a86453fb8f1cdeb3b94d326bc249023f3e72b82316a0fe5',
      'shell': 'cea01ac69a8468f54e4bfb8a453709494449ef886e7dc6ae28e073a79fa2855c',
      'freeze': '2f4eaf1366dcb42b8f89a386e8201b3f2ba0b9f8ae5ef02409436492666d8366',
  }
  if bundle.get('git_head') != MODEL_SOURCE_COMMIT:
    raise AnalysisError('Old analyzer bundle commit changed.')
  for name, path in paths.items():
    binding = _v33451_file_binding(path, absolute=True)
    relative = path.relative_to(_REPO_ROOT).as_posix()
    if (
        bundle.get(name) != binding or binding['sha256'] != expected_sha[name]
        or _git_blob_sha256(MODEL_SOURCE_COMMIT, relative) != expected_sha[name]
    ):
      raise AnalysisError(f'Old analyzer bundle changed: {name}.')
  return dict(node)


def _v33451_validate_analysis_freeze(
    authorization: Mapping[str, Any], *, phase: str,
    started_sha256: str | None = None,
    phase_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
  _assert_cpu_only(f'v3.3.4.5.1 {phase} source gate')
  if _ANALYSIS_FREEZE_PATH.is_symlink():
    raise AnalysisError('Analysis freeze is a symlink.')
  freeze = _read_json(_ANALYSIS_FREEZE_PATH, 'v3.3.4.5.1 freeze')
  _exact_keys(freeze, _V33451_FREEZE_KEYS, 'v3.3.4.5.1 freeze')
  expected = {
      'schema_version': ANALYSIS_SCHEMA_VERSION,
      'analysis_version': ANALYSIS_VERSION, 'attempt_id': ANALYSIS_ATTEMPT_ID,
      'acknowledgement_token': ANALYSIS_ACKNOWLEDGEMENT,
      'amendment_path': str(_ANALYSIS_AMENDMENT_PATH.resolve()),
      'amendment_sha256': ANALYSIS_AMENDMENT_SHA256,
      'amendment_commit': ANALYSIS_AMENDMENT_COMMIT,
      'prior_model_head': MODEL_SOURCE_COMMIT,
      'analysis_attempt_dir': str(_ANALYSIS_ATTEMPT_DIR.resolve()),
      'analysis_dir': str(_ANALYSIS_DIR.resolve()),
  }
  for key, value in expected.items():
    if freeze.get(key) != value:
      raise AnalysisError(f'Analysis freeze literal changed: {key}.')
  if _v33451_file_binding(_ANALYSIS_AMENDMENT_PATH, absolute=True) != {
      'path': str(_ANALYSIS_AMENDMENT_PATH.resolve()),
      'sha256': ANALYSIS_AMENDMENT_SHA256, 'size_bytes': 41_952,
  }:
    raise AnalysisError('Analysis amendment bytes changed.')
  authorization_node = _v33451_validate_authorization(authorization, freeze)
  source = _v33451_validate_source_inventory(
      freeze.get('source_inventory_contract'), authorization_node
  )
  if phase_callback is not None:
    phase_callback('model_input_rehash')
  # No run/compiler/preflight/cache byte is opened before the source gate above.
  if freeze.get('prior_model_freeze_binding') != _v33451_file_binding(
      _FREEZE_PATH, absolute=True
  ):
    raise AnalysisError('Prior model freeze binding changed.')
  immutable = _v33451_validate_immutable_contract(
      freeze.get('immutable_model_artifact_contract')
  )
  if phase_callback is not None:
    phase_callback('post_start_prior_cache_gate')
  consumed = _v33451_validate_consumed_failure(
      freeze.get('consumed_analyzer_failure'),
      freeze.get('consumed_analyzer_failure_content_binding'),
  )
  prior_cache = _validate_prior_cache_directory_aware()
  if (
      freeze.get('prior_cache_contract') != prior_cache
      or freeze.get('prior_cache_contract_content_binding')
      != _v33451_canonical_binding(prior_cache)
  ):
    raise AnalysisError('Frozen prior cache audit changed.')
  if freeze.get('publication_contract') != _v33451_publication_contract():
    raise AnalysisError('Analysis publication contract changed.')
  if freeze.get('record_contracts') != _v33451_record_contracts():
    raise AnalysisError('Analysis record contracts changed.')
  if freeze.get('claim_boundary') != _V33451_CLAIM_BOUNDARY:
    raise AnalysisError('Analysis claim boundary changed.')
  _v33451_require_old_absent()
  if phase == 'pre_start':
    if any(
        path.exists() or path.is_symlink()
        for path in (_ANALYSIS_ATTEMPT_DIR, _ANALYSIS_DIR)
    ):
      raise AnalysisError('Fresh v3.3.4.5.1 analysis path already exists.')
  elif phase == 'post_start':
    if not _is_sha256(started_sha256):
      raise AnalysisError('Active analysis START SHA is absent.')
    tree = _v33451_tree_binding(_ANALYSIS_ATTEMPT_DIR)
    binding = tree['file_bindings'].get('ANALYSIS_ATTEMPT_STARTED.json')
    if (
        tree['directory_paths'] != ['.']
        or set(tree['file_bindings']) != {'ANALYSIS_ATTEMPT_STARTED.json'}
        or binding is None or binding['mode'] != '0400'
        or binding['sha256'] != started_sha256
        or _ANALYSIS_DIR.exists() or _ANALYSIS_DIR.is_symlink()
    ):
      raise AnalysisError('Active analysis START singleton changed.')
  elif phase not in {'before_result', 'before_analysis', 'before_complete'}:
    raise AnalysisError('Unknown analysis validation phase.')
  return {
      'freeze': freeze, 'authorization': authorization_node,
      'source': source, 'immutable': immutable,
      'consumed': consumed, 'prior_cache': prior_cache,
  }


def _v33451_authorization(head: str, digest: str, size: int) -> dict[str, Any]:
  return {
      'git_head': head, 'freeze_path': str(_ANALYSIS_FREEZE_PATH.resolve()),
      'freeze_sha256': digest, 'freeze_size_bytes': size,
      'live_equals_git_show': True, 'tracked_clean': True,
      'authorization_source': 'external_post_commit_audit',
  }


def _v33451_read_publication_json(
    root_role: str, relative: str, label: str,
) -> dict[str, Any]:
  root_fd = _PUBLICATION_DIRECTORIES[root_role][0]
  state_value = _publication_entry_at(root_fd, relative)
  if state_value['entry_type'] != 'regular':
    raise AnalysisError(f'{label} is not a regular publication.')
  fd = os.open(
      relative,
      os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
      dir_fd=root_fd,
  )
  try:
    before = os.fstat(fd)
    if (
        (before.st_dev, before.st_ino, before.st_nlink,
         stat.S_IMODE(before.st_mode),
         before.st_size)
        != (state_value['st_dev'], state_value['st_ino'],
            state_value['st_nlink'], int(state_value['mode'], 8),
            state_value['size_bytes'])
    ):
      raise AnalysisError(f'{label} inode changed before read.')
    payload = bytearray()
    for block in iter(lambda: os.read(fd, 1024 * 1024), b''):
      payload.extend(block)
    after = os.fstat(fd)
    final_path = os.stat(
        relative, dir_fd=root_fd, follow_symlinks=False
    )
    if (
        (after.st_dev, after.st_ino, after.st_nlink, after.st_mode,
         after.st_size)
        != (before.st_dev, before.st_ino, before.st_nlink, before.st_mode,
            before.st_size)
        or (final_path.st_dev, final_path.st_ino, final_path.st_nlink,
            final_path.st_mode, final_path.st_size)
        != (before.st_dev, before.st_ino, before.st_nlink, before.st_mode,
            before.st_size)
        or hashlib.sha256(payload).hexdigest() != state_value['sha256']
    ):
      raise AnalysisError(f'{label} changed during read.')
  finally:
    os.close(fd)
  try:
    value = json.loads(payload)
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise AnalysisError(f'{label} is not valid JSON.') from error
  if not isinstance(value, dict):
    raise AnalysisError(f'{label} must be a JSON object.')
  return value


def _v33451_publication_file_binding(
    root_role: str, relative: str, *, absolute: bool = False,
) -> dict[str, Any]:
  state_value = _publication_entry_at(
      _PUBLICATION_DIRECTORIES[root_role][0], relative
  )
  result = _publication_binding_from_state(state_value)
  compact = {
      'sha256': result['sha256'], 'size_bytes': result['size_bytes'],
  }
  if absolute:
    compact = {
        'path': str(_PUBLICATION_ROOTS[root_role] / relative),
        **compact,
    }
  return compact


def _v33451_start_record(precheck: Mapping[str, Any]) -> dict[str, Any]:
  freeze = precheck['freeze']
  source = precheck['source']
  fresh_paths = _v33451_fresh_paths()
  if set(fresh_paths.values()) != {'absent'}:
    raise AnalysisError('Fresh path evidence changed.')
  result = {
      'status': 'started', 'schema_version': 'v3.3.4.5.1-analysis-start-v1',
      'analysis_version': ANALYSIS_VERSION, 'attempt_id': ANALYSIS_ATTEMPT_ID,
      'acknowledgement': ANALYSIS_ACKNOWLEDGEMENT,
      'git_head': precheck['authorization']['git_head'],
      'external_freeze_authorization': dict(precheck['authorization']),
      'freeze_binding': _v33451_file_binding(
          _ANALYSIS_FREEZE_PATH, absolute=True
      ),
      'analyzer_binding': _v33451_file_binding(Path(__file__), absolute=True),
      'test_binding': _v33451_file_binding(_TEST_PATH, absolute=True),
      'shell_binding': _v33451_file_binding(_SHELL_PATH, absolute=True),
      'generator_binding': _v33451_file_binding(_GENERATOR_PATH, absolute=True),
      'amendment_binding': _v33451_file_binding(
          _ANALYSIS_AMENDMENT_PATH, absolute=True
      ),
      'run_terminal_binding': copy.deepcopy(
          freeze['immutable_model_artifact_contract']['run_terminal_binding']
      ),
      'source_inventory_attestation': copy.deepcopy(source),
      'immutable_input_audit': {
          'inherited_132_live_exact': True,
          'inherited_132_historical_exact': True,
          'amendment_live_historical_exact': True,
          'new_implementation_4_live_historical_exact': True,
          'source_authority_exact': True, 'launch_head_clean': True,
          'current_137_live_launch_exact': True,
          'immutable_model_run_exact': True,
          'immutable_preflight_cache_exact': True,
          'consumed_failure_exact': True, 'prior_cache_exact': True,
          'old_destinations_absent': True, 'new_destinations_fresh': True,
      },
      'consumed_analyzer_failure': copy.deepcopy(precheck['consumed']),
      'consumed_analyzer_failure_content_binding': _v33451_canonical_binding(
          precheck['consumed']
      ),
      'prior_cache_audit': copy.deepcopy(precheck['prior_cache']),
      'prior_cache_audit_content_binding': _v33451_canonical_binding(
          precheck['prior_cache']
      ),
      'fresh_paths': fresh_paths, 'started_at_unix_s': time.time(),
  }
  _exact_keys(result, _V33451_START_KEYS, 'v3.3.4.5.1 START')
  if len(result) != 22:
    raise AnalysisError('v3.3.4.5.1 START is not exactly 22 keys.')
  _finite(result['started_at_unix_s'], 'analysis START timestamp')
  return result


def _v33451_validate_active_start(
    started_sha256: str, authorization: Mapping[str, Any],
    *, output_prefix: set[str] | None,
) -> dict[str, Any]:
  if not _is_sha256(started_sha256):
    raise AnalysisError('Active analysis START SHA is malformed.')
  attempt = _v33451_tree_binding(_ANALYSIS_ATTEMPT_DIR)
  if (
      attempt['directory_paths'] != ['.']
      or set(attempt['file_bindings']) != {'ANALYSIS_ATTEMPT_STARTED.json'}
      or attempt['file_bindings']['ANALYSIS_ATTEMPT_STARTED.json']['mode']
      != '0400'
      or attempt['file_bindings']['ANALYSIS_ATTEMPT_STARTED.json']['sha256']
      != started_sha256
  ):
    raise AnalysisError('Active analysis START singleton changed.')
  started = _v33451_read_publication_json(
      'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json',
      'active analysis START',
  )
  _exact_keys(started, _V33451_START_KEYS, 'active analysis START')
  if (
      started.get('attempt_id') != ANALYSIS_ATTEMPT_ID
      or started.get('acknowledgement') != ANALYSIS_ACKNOWLEDGEMENT
      or started.get('external_freeze_authorization') != authorization
  ):
    raise AnalysisError('Active analysis START envelope changed.')
  if output_prefix is None:
    if _ANALYSIS_DIR.exists() or _ANALYSIS_DIR.is_symlink():
      raise AnalysisError('Analysis output appeared before allocation.')
  else:
    output = _v33451_tree_binding(_ANALYSIS_DIR)
    if output['directory_paths'] != ['.'] or set(output['file_bindings']) != output_prefix:
      raise AnalysisError('Analysis output publication prefix changed.')
    if any(binding['mode'] != '0400' for binding in output['file_bindings'].values()):
      raise AnalysisError('Analysis output prefix mode changed.')
  _v33451_require_old_absent()
  return started


def _v33451_combined_publication_audit(
    publication_failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  failure_role = (
      None if publication_failure is None else publication_failure['root_role']
  )
  audits = {
      role: publication_audit(
          role, publication_failure if failure_role == role else None
      ) for role in ('analysis_attempt', 'analysis_output')
  }
  map_names = (
      'successful_final_bindings_before_terminal',
      'temporary_orphan_bindings',
      'durability_uncertain_final_bindings', 'preexisting_entry_states',
  )
  combined: dict[str, dict[str, Any]] = {name: {} for name in map_names}
  for role, audit in audits.items():
    for name in map_names:
      for relative, value in audit[name].items():
        combined[name][f'{role}/{relative}'] = copy.deepcopy(value)
  result = {
      'schema_version': _V33451_PUBLICATION_SCHEMA_VERSION,
      'method': PUBLICATION_METHOD,
      'successful_final_count_before_terminal': len(
          combined['successful_final_bindings_before_terminal']
      ),
      'successful_final_bindings_before_terminal': dict(sorted(
          combined['successful_final_bindings_before_terminal'].items()
      )),
      'temporary_orphan_count': len(combined['temporary_orphan_bindings']),
      'temporary_orphan_bindings': dict(sorted(
          combined['temporary_orphan_bindings'].items()
      )),
      'durability_uncertain_final_count': len(
          combined['durability_uncertain_final_bindings']
      ),
      'durability_uncertain_final_bindings': dict(sorted(
          combined['durability_uncertain_final_bindings'].items()
      )),
      'preexisting_entry_count': len(combined['preexisting_entry_states']),
      'preexisting_entry_states': dict(sorted(
          combined['preexisting_entry_states'].items()
      )),
      'publication_failure': (
          None if publication_failure is None else dict(publication_failure)
      ),
      'no_new_entry_failure': bool(
          publication_failure is not None
          and not combined['temporary_orphan_bindings']
          and not combined['durability_uncertain_final_bindings']
      ),
      'no_publication_retry': True, 'no_published_final_deleted': True,
      'no_temp_or_final_reused': True,
  }
  _exact_keys(result, set(PUBLICATION_AUDIT_KEYS), 'combined publication audit')
  return result


def _v33451_validate_success_audit(
    audit: Mapping[str, Any], expected: Mapping[str, Mapping[str, Any]],
) -> None:
  _exact_keys(audit, set(PUBLICATION_AUDIT_KEYS), 'successful publication audit')
  if (
      audit.get('successful_final_count_before_terminal') != len(expected)
      or audit.get('successful_final_bindings_before_terminal') != dict(expected)
      or audit.get('temporary_orphan_count') != 0
      or audit.get('temporary_orphan_bindings') != {}
      or audit.get('durability_uncertain_final_count') != 0
      or audit.get('durability_uncertain_final_bindings') != {}
      or audit.get('preexisting_entry_count') != 0
      or audit.get('preexisting_entry_states') != {}
      or audit.get('publication_failure') is not None
      or audit.get('no_new_entry_failure') is not False
      or any(audit.get(name) is not True for name in (
          'no_publication_retry', 'no_published_final_deleted',
          'no_temp_or_final_reused',
      ))
  ):
    raise AnalysisError('Successful publication audit/linkage changed.')


def _v33451_prior333_binding() -> dict[str, Any]:
  if (
      _v33451_file_binding(_PRIOR_FREEZE_PATH)['sha256'] != PRIOR_FREEZE_SHA256
  ):
    raise AnalysisError('Immutable v3.3.3 freeze changed.')
  run = _validate_bound_tree(
      _PRIOR_RUN_DIR, _binding_map(_PRIOR_RUN_FILES),
      PRIOR_RUN_TREE_SHA256, 'v3.3.3 run',
  )
  compiler_paths = [
      _PRIOR_RUN_DIR / relative for relative in _PRIOR_RUN_FILES
      if relative.startswith('compiler/')
  ]
  if (
      len(compiler_paths) != 4
      or _tree_digest(compiler_paths, _PRIOR_RUN_DIR)
      != PRIOR_COMPILER_TREE_SHA256
      or (_PRIOR_RUN_DIR / 'raw').exists()
      or (_PRIOR_RUN_DIR / 'raw').is_symlink()
  ):
    raise AnalysisError('Immutable v3.3.3 compiler/raw prefix changed.')
  completion = _read_json(
      _PRIOR_RUN_DIR / 'RUN_COMPLETE.json', 'v3.3.3 RUN_COMPLETE'
  )
  predicates = {
      'status': 'controlled_stop', 'stop_reason': 'source_program_mismatch',
      'model_apply_count': 0, 'ood_anchor_record_count': 0,
      'confirmation_model_calls': 0, 'scientific_summary_computed': False,
      'combined_analysis_permitted': False,
  }
  if {key: completion.get(key) for key in predicates} != predicates:
    raise AnalysisError('Immutable v3.3.3 terminal predicates changed.')
  return {
      'path': str(_PRIOR_RUN_DIR.resolve()),
      'model_run_commit': PRIOR_SOURCE_COMMIT,
      'freeze_sha256': PRIOR_FREEZE_SHA256, 'file_count': 11,
      'file_tree_sha256': run['tree_sha256'], 'compiler_file_count': 4,
      'compiler_tree_sha256': PRIOR_COMPILER_TREE_SHA256,
      'status_predicates': predicates,
  }


def _v33451_prior331_binding() -> dict[str, Any]:
  _validate_source_bundle(
      _PRIOR_331_SOURCES,
      implementation_commit=PRIOR_331_IMPLEMENTATION_COMMIT,
      amendment_commit=PRIOR_331_AMENDMENT_COMMIT,
  )
  _validate_bound_tree(
      _PRIOR_331_ATTEMPT_DIR, _binding_map(_PRIOR_331_ATTEMPT_FILES),
      PRIOR_331_ATTEMPT_TREE_SHA256, 'v3.3.3.1 attempt',
  )
  _validate_bound_tree(
      _PRIOR_331_OUTPUT_DIR, _binding_map(_PRIOR_331_OUTPUT_FILES),
      PRIOR_331_OUTPUT_TREE_SHA256, 'v3.3.3.1 output',
  )
  for root, files in (
      (_PRIOR_331_ATTEMPT_DIR, _PRIOR_331_ATTEMPT_FILES),
      (_PRIOR_331_OUTPUT_DIR, _PRIOR_331_OUTPUT_FILES),
  ):
    for relative, (_size, digest) in files.items():
      archive_relative = (root / relative).relative_to(_REPO_ROOT).as_posix()
      if _git_blob_sha256(PRIOR_331_ARCHIVE_COMMIT, archive_relative) != digest:
        raise AnalysisError('Immutable v3.3.3.1 archive blob changed.')
  analysis = _read_json(
      _PRIOR_331_OUTPUT_DIR / 'ANALYSIS.json', 'v3.3.3.1 ANALYSIS'
  )
  if (
      analysis.get('status') != 'complete_controlled_stop_structural_archive'
      or analysis.get('decision')
      != 'controlled_stop_source_program_mismatch_representation_only'
  ):
    raise AnalysisError('Immutable v3.3.3.1 decision changed.')
  return {
      'amendment_commit': PRIOR_331_AMENDMENT_COMMIT,
      'implementation_commit': PRIOR_331_IMPLEMENTATION_COMMIT,
      'archive_commit': PRIOR_331_ARCHIVE_COMMIT,
      'attempt_dir': str(_PRIOR_331_ATTEMPT_DIR.resolve()),
      'analysis_dir': str(_PRIOR_331_OUTPUT_DIR.resolve()),
      'attempt_files': _binding_map(_PRIOR_331_ATTEMPT_FILES),
      'analysis_files': _binding_map(_PRIOR_331_OUTPUT_FILES),
      'attempt_tree_sha256': PRIOR_331_ATTEMPT_TREE_SHA256,
      'analysis_tree_sha256': PRIOR_331_OUTPUT_TREE_SHA256,
      'status': analysis['status'], 'decision': analysis['decision'],
  }


def _v33451_live_success_bindings(
    *, include_analysis: bool,
) -> dict[str, dict[str, Any]]:
  paths = {
      'analysis_attempt/ANALYSIS_ATTEMPT_STARTED.json': (
          'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json'
      ),
      'analysis_output/RESULT.md': ('analysis_output', 'RESULT.md'),
  }
  if include_analysis:
    paths['analysis_output/ANALYSIS.json'] = (
        'analysis_output', 'ANALYSIS.json'
    )
  return {
      key: _publication_binding_from_state(_publication_entry_at(
          _PUBLICATION_DIRECTORIES[role][0], relative
      )) for key, (role, relative) in sorted(paths.items())
  }


def _v33451_structural_analyze(
    *, token: object, started_sha256: str,
    authorization: Mapping[str, Any],
    phase_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
  if token is not _V33451_ACTIVE_TOKEN:
    raise AnalysisError('Direct structural analysis invocation is forbidden.')
  _assert_cpu_only('v3.3.4.5.1 structural analyzer entry')
  _v33451_validate_active_start(started_sha256, authorization, output_prefix=None)
  precheck = _v33451_validate_analysis_freeze(
      authorization, phase='post_start', started_sha256=started_sha256,
      phase_callback=phase_callback,
  )
  # Reuse only side-effect-free local validators copied into this standalone
  # module.  No prior analyzer/helper is imported or invoked.
  if phase_callback is not None:
    phase_callback('model_input_rehash')
  # Do not call the copied v3.3.4.5 freeze entrypoint here: that entrypoint
  # contains the directory-blind v3.3.3 cache walk that consumed the sole
  # predecessor invocation.  The analysis-freeze gate above has already
  # authenticated the immutable model freeze and all source bytes; replay the
  # remaining model START inputs with the corrected, standalone bindings.
  model_freeze_sha = _v33451_file_binding(_FREEZE_PATH)['sha256']
  if model_freeze_sha != (
      '2f4eaf1366dcb42b8f89a386e8201b3f2ba0b9f8ae5ef02409436492666d8366'
  ):
    raise AnalysisError('Immutable v3.3.4.5 model freeze changed.')
  model_freeze = _read_json(_FREEZE_PATH, 'v3.3.4.5 model freeze')
  _exact_keys(model_freeze, _FREEZE_KEYS, 'v3.3.4.5 model freeze')
  prior333 = _v33451_prior333_binding()
  prior331 = _v33451_prior331_binding()
  model_start = _validate_start_v3345(
      _RUN_DIR, model_freeze, model_freeze_sha,
      prior333=prior333, prior331=prior331,
  )
  _validate_preflight_and_same_process(model_start, model_freeze)
  if phase_callback is not None:
    phase_callback('structural_terminal_audit')
  completion = _read_json(_RUN_DIR / 'RUN_COMPLETE.json', 'RUN_COMPLETE')
  completion, _terminal_detail = _validate_terminal_common(
      completion, freeze_sha=model_freeze_sha, start=model_start
  )
  if (
      completion.get('status')
      != 'controlled_stop_diagnostic_provenance_failure'
      or completion.get('stop_reason') != 'diagnostic_persistence_failure'
      or completion.get('failure') != {
          'type': 'DiagnosticPersistenceFailure',
          'message': "'eight_row_compiler'",
          'traceback': "DiagnosticPersistenceFailure: 'eight_row_compiler'\n",
      }
      or completion.get('eight_row_lower_attempt_count') != 1
      or completion.get('eight_row_compile_attempt_count') != 1
      or completion.get('eight_row_successful_compile_count') != 1
      or completion.get('model_apply_attempt_count') != 0
      or completion.get('model_apply_success_count') != 0
      or completion.get('valid_record_count') != 0
  ):
    raise AnalysisError('Exact diagnostic-persistence terminal changed.')
  manifest = _read_json(_RUN_DIR / 'RAW_MANIFEST.json', 'RAW_MANIFEST')
  _exact_keys(manifest, _MANIFEST_KEYS, 'RAW_MANIFEST')
  if (
      manifest.get('status') != 'empty_controlled_stop'
      or manifest.get('valid_artifact_count') != 0
      or manifest.get('artifact_bindings') != {}
      or manifest.get('artifact_tree_sha256') != EMPTY_SHA256
      or manifest.get('valid_recipient_anchor_pairs') != []
      or manifest.get('failed_current_binding') is not None
      or manifest.get('dispatch_started_count') != 0
      or manifest.get('dispatch_completed_count') != 0
      or manifest.get('dispatch_started_bindings') != {}
      or manifest.get('dispatch_completed_bindings') != {}
      or manifest.get('dispatch_started_tree_sha256') != EMPTY_SHA256
      or manifest.get('dispatch_completed_tree_sha256') != EMPTY_SHA256
      or completion.get('raw_manifest') != manifest
  ):
    raise AnalysisError('Exact empty raw manifest changed.')
  for relative in ('raw', 'dispatch_started', 'dispatch_completed'):
    path = _RUN_DIR / relative
    if path.exists() or path.is_symlink():
      raise AnalysisError(f'Forbidden nonempty scientific prefix appeared: {relative}.')
  imports = _validate_imports(
      _RUN_DIR, completion, bundle_root=_REPO_ROOT, freeze=model_freeze
  )
  protobuf = _validate_protobuf(_RUN_DIR, completion, model_freeze)
  compiler = _validate_compiler_v3345(
      _RUN_DIR, completion, model_freeze, model_start
  )
  if (
      compiler.get('state') != 'diagnostic_provenance_failed'
      or compiler.get('audit', {}).get('source_program_exact') is not True
      or compiler.get('audit', {}).get('stablehlo_exact') is not True
      or compiler.get('audit', {}).get('pre_backend_exact') is not True
      or compiler.get('audit', {}).get('entry_abi_exact') is not True
      or compiler.get('audit', {}).get('diagnostic_provenance_complete')
      is not False
  ):
    raise AnalysisError('Compiler/source-program diagnostic gate changed.')
  _validate_run_publication_audit(
      completion.get('publication_audit'), run_dir=_RUN_DIR,
      preterminal=completion['preterminal_tree_binding'],
  )
  _validate_run_membership(_RUN_DIR, completion, manifest)
  _validate_model_cache_final(
      completion['model_kernel_cache_final'], compiler=compiler.get('record'),
      status=completion['status'], reason=completion['stop_reason'],
  )
  start_binding = _v33451_publication_file_binding(
      'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json', absolute=True
  )
  result = {
      'status': 'complete_controlled_stop_structural_archive',
      'decision': 'controlled_stop_diagnostic_provenance_failure',
      'analysis_version': ANALYSIS_VERSION,
      'analysis_attempt_start_binding': start_binding,
      'run_binding': copy.deepcopy(
          precheck['immutable']['run_root_binding']
      ),
      'preflight_binding': copy.deepcopy(
          precheck['immutable']['preflight_tree_binding']
      ),
      'external_cache_binding': copy.deepcopy(
          precheck['immutable']['external_cache_tree_binding']
      ),
      'model_cache_binding': copy.deepcopy(
          precheck['immutable']['model_cache_tree_binding']
      ),
      'source_and_prior_audit': {
          'inherited_132_live_exact': True,
          'inherited_132_historical_exact': True,
          'amendment_live_historical_exact': True,
          'new_implementation_4_live_historical_exact': True,
          'source_authority_exact': True, 'launch_head_clean': True,
          'current_137_live_launch_exact': True,
          'immutable_model_run_exact': True,
          'immutable_preflight_cache_exact': True,
          'consumed_failure_exact': True, 'prior_cache_exact': True,
          'old_destinations_absent': True, 'active_attempt_exact': True,
      },
      'consumed_analyzer_failure_audit': copy.deepcopy(precheck['consumed']),
      'prior_cache_audit': copy.deepcopy(precheck['prior_cache']),
      'compiler_and_signature_audit': {
          'terminal_status': completion['status'],
          'stop_reason': completion['stop_reason'],
          'source_program_exact': True, 'signature_adapter_exact': True,
          'stablehlo_exact': True, 'prebackend_exact': True,
          'entry_abi_exact': True, 'successful_compile_count': 1,
          'diagnostic_provenance_complete': False,
          'compiled_backend_diagnostic_only': True,
      },
      'dispatch_journal_audit': {
          'expected_record_count': 80, 'expected_apply_count': 320,
          'valid_record_count': 0, 'started_count': 0,
          'completed_count': 0, 'raw_tree_sha256': EMPTY_SHA256,
          'started_tree_sha256': EMPTY_SHA256,
          'completed_tree_sha256': EMPTY_SHA256,
      },
      'raw_prefix_audit': {
          'manifest_binding': _v33451_file_binding(
              _RUN_DIR / 'RAW_MANIFEST.json', absolute=True
          ),
          'status': 'empty_controlled_stop', 'artifact_count': 0,
          'failed_current_binding': None, 'raw_directory_absent': True,
          'journal_directories_absent': True,
      },
      'control_audit': {
          'control_state_eligible': False, 'all_80_complete': False,
          'id0_all20': False, 'id255_all20': False,
          'six_row_compile_count': 0, 'identity_rerun_count': 0,
          'main_cube_rerun_count': 0, 'old_records_reused': 0,
      },
      'terminal_audit': {
          'status': completion['status'], 'stop_reason': completion['stop_reason'],
          'terminal_linkage_exact': True, 'count_arithmetic_exact': True,
          'no_retry': True, 'failure_exact': True,
          'publication_exact': True,
          'imports_protobuf_exact': bool(imports and protobuf),
      },
      'publication_audit': None,
      'confirmation_boundary': {
          'confirmation_paths_opened': False, 'confirmation_model_calls': 0,
          'later_exon_metadata_label_exposure_disclosed': True,
          'model_outputs_activations_interventions_blind': True,
          'no_confirmation_scientific_access': True,
      },
      'claim_boundary': copy.deepcopy(_V33451_CLAIM_BOUNDARY),
      'scientific_summary_computed': False,
      'donor_normalization_computed': False,
      'shapley_or_nomination_computed': False,
      'interaction_or_resolution_computed': False,
      'nomination_performed': False, 'combined_analysis_permitted': False,
      'completed_at_unix_s': time.time(),
  }
  _exact_keys(result, _V33451_ANALYSIS_KEYS, 'v3.3.4.5.1 ANALYSIS draft')
  _assert_cpu_only('v3.3.4.5.1 structural analyzer exit')
  return result


def _v33451_render_markdown(result: Mapping[str, Any]) -> str:
  if (
      result.get('status') != 'complete_controlled_stop_structural_archive'
      or result.get('decision')
      != 'controlled_stop_diagnostic_provenance_failure'
  ):
    raise AnalysisError('Structural result is not the frozen controlled stop.')
  return '\n'.join((
      '# OpenSplice v3.3.4.5.1 structural archive', '',
      '**Decision:** `controlled_stop_diagnostic_provenance_failure`', '',
      'The immutable v3.3.4.5 run stopped before every model apply and raw',
      'record. Its source-program gate passed; diagnostic compiler-record',
      'construction failed with `DiagnosticPersistenceFailure`.', '',
      'This CPU-only archive opened no raw record and computed no scientific',
      'summary, donor normalization, Shapley value, interaction, resolution,',
      'rank, or nomination. A future prospective protocol is required.', '',
  ))


def analyze(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
  """Rejects the copied predecessor's public analysis entrypoint."""
  raise AnalysisError(
      'Direct/legacy analyze() is forbidden; only the append-only '
      'v3.3.4.5.1 main lifecycle may enter the structural audit.'
  )


def _analysis_attempt_precheck(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
  """Rejects the copied predecessor's known-bad precheck entrypoint."""
  raise AnalysisError('The predecessor analysis precheck is disabled.')


def _v33451_entry_state_tree(states: Mapping[str, Mapping[str, Any]]) -> str:
  digest = hashlib.sha256()
  for relative, value in sorted(states.items()):
    digest.update(relative.encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(_v33451_canonical_binding(value)['sha256']))
  return digest.hexdigest()


def _v33451_output_state(
    root_role: str, publication_failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  if root_role in _PUBLICATION_UNBINDABLE_ROOTS:
    raise AnalysisError('Publication root lost an invocation-created entry.')
  root = _PUBLICATION_ROOTS[root_role]
  if not root.exists() and not root.is_symlink():
    result = {
        'state': 'absent', 'root_role': root_role, 'root_lstat': None,
        'regular_final_bindings': {}, 'temporary_orphan_bindings': {},
        'durability_uncertain_final_bindings': {},
        'preexisting_entry_states': {}, 'directory_paths': [],
        'directory_tree_sha256': EMPTY_SHA256,
        'directory_file_tree_sha256': EMPTY_SHA256,
        'file_tree_sha256': EMPTY_SHA256,
        'entry_state_tree_sha256': EMPTY_SHA256,
        'publication_failure': publication_failure,
    }
    _exact_keys(result, _V33451_OUTPUT_STATE_KEYS, 'absent output state')
    return result
  audit = publication_audit(root_role, publication_failure)
  successful = dict(audit['successful_final_bindings_before_terminal'])
  temporary = dict(audit['temporary_orphan_bindings'])
  uncertain = dict(audit['durability_uncertain_final_bindings'])
  preexisting = dict(audit['preexisting_entry_states'])
  physical: dict[str, dict[str, Any]] = {}
  directories = ['.']
  registered = _PUBLICATION_DIRECTORIES.get(root_role)
  owned_root_fd = False
  if registered is None:
    root_entry = _publication_entry(root)
    if root_entry['entry_type'] != 'directory':
      if publication_failure is None:
        raise AnalysisError('Unregistered non-directory output root appeared.')
      if any((successful, temporary, uncertain, preexisting)):
        raise AnalysisError('Non-directory output root has internal maps.')
      result = {
          'state': 'publication_failure_prefix', 'root_role': root_role,
          'root_lstat': root_entry, 'regular_final_bindings': {},
          'temporary_orphan_bindings': {},
          'durability_uncertain_final_bindings': {},
          'preexisting_entry_states': {}, 'directory_paths': [],
          'directory_tree_sha256': EMPTY_SHA256,
          'directory_file_tree_sha256': EMPTY_SHA256,
          'file_tree_sha256': EMPTY_SHA256,
          'entry_state_tree_sha256': EMPTY_SHA256,
          'publication_failure': publication_failure,
      }
      _exact_keys(
          result, _V33451_OUTPUT_STATE_KEYS,
          'non-directory analysis output state',
      )
      return result
    root_fd = os.open(
        root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    owned_root_fd = True
    opened = os.fstat(root_fd)
    expected_dev, expected_ino = opened.st_dev, opened.st_ino
  else:
    root_fd, expected_dev, expected_ino = registered
  root_status = os.fstat(root_fd)
  path_status = root.lstat()
  if (
      (root_status.st_dev, root_status.st_ino) != (expected_dev, expected_ino)
      or (path_status.st_dev, path_status.st_ino) != (expected_dev, expected_ino)
  ):
    raise AnalysisError('Analysis publication root pathname/inode changed.')
  root_state = {
      'state': 'present', 'entry_type': 'directory',
      'mode': _publication_mode(root_status.st_mode), 'size_bytes': None,
      'sha256': None, 'st_dev': root_status.st_dev,
      'st_ino': root_status.st_ino, 'st_nlink': root_status.st_nlink,
  }
  if (
      root_state['entry_type'] != 'directory' or root_state['mode'] != '0700'
  ):
    raise AnalysisError('Analysis publication root state changed.')
  def walk(directory_fd: int, prefix: str = '') -> None:
    for basename in sorted(os.listdir(directory_fd)):
      if '/' in basename or basename in {'.', '..'}:
        raise AnalysisError('Analysis publication entry name changed.')
      relative = basename if not prefix else f'{prefix}/{basename}'
      state_value = _publication_entry_at(directory_fd, basename)
      physical[relative] = state_value
      if state_value['entry_type'] == 'directory':
        if state_value['mode'] != '0700':
          raise AnalysisError('Analysis publication directory mode changed.')
        directories.append(relative)
        child_fd = os.open(
            basename,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_fd,
        )
        try:
          walk(child_fd, relative)
        finally:
          os.close(child_fd)
  try:
    walk(root_fd)
  finally:
    if owned_root_fd:
      os.close(root_fd)
  if (
      registered is None and publication_failure is not None
      and str(publication_failure.get('failure_stage', '')).startswith('root_')
  ):
    # A root allocation collision did not create any child.  Every physical
    # child of a preexisting directory belongs in the disjoint preexisting
    # map so the failure archive binds the complete preserved namespace.
    preexisting.update(physical)
    _PUBLICATION_PREEXISTING[root_role] = dict(sorted(preexisting.items()))
  classified = set(successful) | set(temporary) | set(uncertain) | {
      path for path, state_value in preexisting.items()
      if state_value.get('state') == 'present'
  }
  groups = (set(successful), set(temporary), set(uncertain), set(preexisting))
  if any(groups[i] & groups[j] for i in range(4) for j in range(i + 1, 4)):
    raise AnalysisError('Analysis publication state maps overlap.')
  if set(physical) != classified:
    raise AnalysisError('Analysis publication state does not bind whole tree.')
  regular_union = {**successful, **temporary, **uncertain}
  for relative, binding in regular_union.items():
    if _publication_binding_from_state(physical[relative]) != binding:
      raise AnalysisError('Analysis publication regular binding changed.')
  for relative, state_value in preexisting.items():
    if state_value.get('state') == 'present' and physical[relative] != state_value:
      raise AnalysisError('Analysis publication preexisting state changed.')
  physical_regular = {
      relative: _publication_binding_from_state(state_value)
      for relative, state_value in physical.items()
      if state_value['entry_type'] == 'regular'
  }
  directories = sorted(directories)
  df = hashlib.sha256()
  for relative in directories:
    df.update(b'D\0')
    df.update(relative.encode())
    df.update(b'\0')
  for relative in sorted(physical_regular):
    df.update(b'F\0')
    df.update(relative.encode())
    df.update(b'\0')
    df.update(bytes.fromhex(physical_regular[relative]['sha256']))
  entry_states = {
      relative: state_value for relative, state_value in physical.items()
  }
  result = {
      'state': (
          'publication_failure_prefix' if publication_failure is not None
          else 'published_prefix'
      ),
      'root_role': root_role, 'root_lstat': root_state,
      'regular_final_bindings': successful,
      'temporary_orphan_bindings': temporary,
      'durability_uncertain_final_bindings': uncertain,
      'preexisting_entry_states': preexisting,
      'directory_paths': directories,
      'directory_tree_sha256': _directory_digest(directories),
      'directory_file_tree_sha256': df.hexdigest(),
      'file_tree_sha256': _binding_map_digest(physical_regular),
      'entry_state_tree_sha256': _v33451_entry_state_tree(entry_states),
      'publication_failure': publication_failure,
  }
  _exact_keys(result, _V33451_OUTPUT_STATE_KEYS, 'analysis output state')
  return result


def _v33451_phase_recheck(
    authorization: Mapping[str, Any], started_sha256: str, *, phase: str,
    output_prefix: set[str] | None,
) -> dict[str, Any]:
  _assert_cpu_only(f'analysis {phase} prepublication')
  _v33451_validate_active_start(
      started_sha256, authorization, output_prefix=output_prefix
  )
  checked = _v33451_validate_analysis_freeze(
      authorization, phase=phase, started_sha256=started_sha256
  )
  return checked


def _v33451_complete_record(started_sha256: str) -> dict[str, Any]:
  start_binding = _v33451_publication_file_binding(
      'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json', absolute=True
  )
  if start_binding['sha256'] != started_sha256:
    raise AnalysisError('Analysis START changed before COMPLETE.')
  attempt_tree = _v33451_tree_binding(_ANALYSIS_ATTEMPT_DIR)
  output_tree = _v33451_tree_binding(_ANALYSIS_DIR)
  if (
      set(attempt_tree['file_bindings']) != {'ANALYSIS_ATTEMPT_STARTED.json'}
      or set(output_tree['file_bindings']) != {'RESULT.md', 'ANALYSIS.json'}
  ):
    raise AnalysisError('Successful analysis prefix changed before COMPLETE.')
  result = {
      'status': 'complete',
      'schema_version': 'v3.3.4.5.1-analysis-complete-v1',
      'analysis_version': ANALYSIS_VERSION, 'attempt_id': ANALYSIS_ATTEMPT_ID,
      'start_binding': start_binding,
      'analysis_binding': _v33451_publication_file_binding(
          'analysis_output', 'ANALYSIS.json', absolute=True
      ),
      'result_binding': _v33451_publication_file_binding(
          'analysis_output', 'RESULT.md', absolute=True
      ),
      'attempt_tree_before_complete': attempt_tree,
      'output_tree_complete': output_tree,
      'publication_audit': _v33451_combined_publication_audit(),
      'completed_at_unix_s': time.time(),
  }
  _exact_keys(result, _V33451_COMPLETE_KEYS, 'ANALYSIS_COMPLETE')
  _v33451_validate_success_audit(
      result['publication_audit'],
      _v33451_live_success_bindings(include_analysis=True),
  )
  return result


def _v33451_failure_record(
    error: BaseException, started_sha256: str, *, phase: str,
) -> dict[str, Any]:
  if phase not in _V33451_FAILURE_PHASES:
    raise AnalysisError('Analysis failure phase is not frozen.')
  failure = getattr(error, 'publication_failure', None)
  if failure is not None:
    failure = _v33451_validate_publication_failure(
        failure, 'analysis publication failure'
    )
  result = {
      'status': 'failure',
      'schema_version': 'v3.3.4.5.1-analysis-failure-v1',
      'analysis_version': ANALYSIS_VERSION, 'attempt_id': ANALYSIS_ATTEMPT_ID,
      'start_binding': _v33451_publication_file_binding(
          'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json', absolute=True
      ),
      'failure': {
          'type': type(error).__name__, 'message': str(error),
          'traceback': ''.join(traceback.format_exception(error)),
      },
      'failure_phase': phase, 'raw_access_reached': False,
      'analysis_output_state': _v33451_output_state(
          'analysis_output', failure if failure and failure['root_role'] == 'analysis_output' else None
      ),
      'attempt_output_state': _v33451_output_state(
          'analysis_attempt', failure if failure and failure['root_role'] == 'analysis_attempt' else None
      ),
      'publication_audit': _v33451_combined_publication_audit(failure),
      'old_destinations_absent': True, 'failed_at_unix_s': time.time(),
  }
  if result['start_binding']['sha256'] != started_sha256:
    raise AnalysisError('Analysis START changed before FAILURE.')
  _v33451_require_old_absent()
  _exact_keys(result, _V33451_FAILURE_KEYS, 'ANALYSIS_FAILURE')
  return result


def _v33451_write_json(
    role: str, relative: str, value: Mapping[str, Any], artifact_role: str,
) -> dict[str, Any]:
  payload = (
      json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
  ).encode('utf-8')
  return publish_bytes(role, relative, payload, artifact_role)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(ANALYSIS_ACKNOWLEDGEMENT, action='store_true')
  parser.add_argument('--authorized-git-head', required=True)
  parser.add_argument('--authorized-freeze-sha256', required=True)
  parser.add_argument('--authorized-freeze-size-bytes', required=True, type=int)
  args = parser.parse_args()
  if not getattr(args, 'acknowledge_structural_only_v3_3_4_5_1'):
    raise AnalysisError('The literal v3.3.4.5.1 acknowledgement is required.')
  if (
      re.fullmatch(r'[0-9a-f]{40}', args.authorized_git_head) is None
      or re.fullmatch(r'[0-9a-f]{64}', args.authorized_freeze_sha256) is None
      or args.authorized_freeze_size_bytes < 0
  ):
    raise AnalysisError('External freeze authorization arguments are malformed.')
  authorization = _v33451_authorization(
      args.authorized_git_head, args.authorized_freeze_sha256,
      args.authorized_freeze_size_bytes,
  )
  # All provenance-only gates precede allocation; failure here persists no file.
  precheck = _v33451_validate_analysis_freeze(
      authorization, phase='pre_start'
  )
  started = _v33451_start_record(precheck)
  if _v33451_fresh_paths() != started['fresh_paths']:
    raise AnalysisError('Fresh analysis paths changed before START allocation.')
  ensure_publication_directory(
      'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json',
      'analysis_attempt_start',
  )
  _v33451_write_json(
      'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json', started,
      'analysis_attempt_start',
  )
  started_sha256 = _v33451_publication_file_binding(
      'analysis_attempt', 'ANALYSIS_ATTEMPT_STARTED.json'
  )['sha256']
  phase_state = {'value': 'post_start_source_gate'}
  def advance_phase(value: str) -> None:
    if value not in _V33451_FAILURE_PHASES:
      raise AnalysisError('Internal analysis failure phase changed.')
    phase_state['value'] = value
  try:
    result = _v33451_structural_analyze(
        token=_V33451_ACTIVE_TOKEN, started_sha256=started_sha256,
        authorization=authorization, phase_callback=advance_phase,
    )
    advance_phase('result_publication')
    _v33451_phase_recheck(
        authorization, started_sha256, phase='before_result',
        output_prefix=None,
    )
    ensure_publication_directory(
        'analysis_output', 'RESULT.md', 'analysis_result_markdown'
    )
    publish_bytes(
        'analysis_output', 'RESULT.md',
        _v33451_render_markdown(result).encode('utf-8'),
        'analysis_result_markdown',
    )
    advance_phase('analysis_publication')
    _v33451_phase_recheck(
        authorization, started_sha256, phase='before_analysis',
        output_prefix={'RESULT.md'},
    )
    result['publication_audit'] = _v33451_combined_publication_audit()
    _v33451_validate_success_audit(
        result['publication_audit'],
        _v33451_live_success_bindings(include_analysis=False),
    )
    _exact_keys(result, _V33451_ANALYSIS_KEYS, 'final ANALYSIS')
    _v33451_write_json(
        'analysis_output', 'ANALYSIS.json', result, 'analysis_json'
    )
    advance_phase('final_toctou')
    _v33451_phase_recheck(
        authorization, started_sha256, phase='before_complete',
        output_prefix={'RESULT.md', 'ANALYSIS.json'},
    )
    advance_phase('complete_publication')
    _v33451_write_json(
        'analysis_attempt', 'ANALYSIS_COMPLETE.json',
        _v33451_complete_record(started_sha256), 'analysis_complete',
    )
  except BaseException as error:
    # Exactly one failure-terminal attempt; a failure here is terminal-less.
    failure_record = _v33451_failure_record(
        error, started_sha256, phase=phase_state['value']
    )
    _v33451_write_json(
        'analysis_attempt', 'ANALYSIS_FAILURE.json', failure_record,
        'analysis_failure',
    )
    raise
  finally:
    _assert_cpu_only('v3.3.4.5.1 analyzer final boundary')


if __name__ == '__main__':
  main()
