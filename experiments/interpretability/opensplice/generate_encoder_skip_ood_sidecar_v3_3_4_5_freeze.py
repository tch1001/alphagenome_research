#!/usr/bin/env python3
"""Generate the prospective v3.3.4.5 machine freeze from frozen predecessors."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

# pylint: disable=g-import-not-at-top
import validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_5 as bootstrap
import run_encoder_skip_ood_sidecar_v3_3_4_5 as runner


_EXTRA_FILES = (
    'experiments/interpretability/opensplice/'
    'analyze_encoder_skip_ood_sidecar_v3_3_4_5.py',
    'experiments/interpretability/opensplice/'
    'analyze_encoder_skip_ood_sidecar_v3_3_4_5.sh',
    'experiments/interpretability/opensplice/'
    'analyze_encoder_skip_ood_sidecar_v3_3_4_5_test.py',
    'experiments/interpretability/opensplice/'
    'generate_encoder_skip_ood_sidecar_v3_3_4_5_freeze.py',
    'experiments/interpretability/opensplice/'
    'validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_5.py',
    'experiments/interpretability/opensplice/'
    'launch_encoder_skip_ood_sidecar_v3_3_4_5.py',
    'experiments/interpretability/opensplice/run_device_preflight_v3_3_4_5.py',
    'experiments/interpretability/opensplice/'
    'run_device_preflight_v3_3_4_5_test.py',
    'experiments/interpretability/opensplice/'
    'run_encoder_skip_ood_sidecar_v3_3_4_5.py',
    'experiments/interpretability/opensplice/'
    'run_encoder_skip_ood_sidecar_v3_3_4_5_test.py',
    'experiments/interpretability/opensplice/'
    'run_encoder_skip_ood_sidecar_v3_3_4_5.sh',
    'experiments/interpretability/opensplice/v3_wider_mechanism/'
    'encoder_skip_ood_sidecar_preflight_phase_amendment_v3_3_4_5.md',
)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def build_freeze() -> dict[str, object]:
  base = json.loads(
      bootstrap.V3_3_4_4_FREEZE_PATH.read_text(encoding='utf-8')
  )
  if base.pop('nonpublication_terminal_contract_v3_3_4_4', None) is None:
    raise RuntimeError('Inherited v3.3.4.4 nonpublication contract is absent.')
  base.update({
      'amendment_commit': bootstrap.AMENDMENT_COMMIT,
      'amendment_path': str(bootstrap.AMENDMENT_PATH.resolve()),
      'amendment_sha256': bootstrap.AMENDMENT_SHA256,
      'analysis_attempt_dir': str(bootstrap.ANALYSIS_ATTEMPT_DIR.resolve()),
      'analysis_dir': str(bootstrap.ANALYSIS_DIR.resolve()),
      'attempt_id': 'v3.3.4.5-development-ood-sidecar-one-shot',
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
      'preflight_script_version': 'opensplice-device-preflight-v3.3.4.5',
      'script_version': 'v3.3.4.5',
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
          'execution_contract': _execution_contract(),
      },
      'preflight_contract': bootstrap.PREFLIGHT_CONTRACT,
      'compiled_diagnostics_contract': (
          bootstrap.COMPILED_DIAGNOSTICS_CONTRACT
      ),
      'publication_contract_v3_3_4_1': (
          bootstrap.PUBLICATION_CONTRACT_V3_3_4_1
      ),
      'nonpublication_terminal_contract_v3_3_4_5': (
          bootstrap.NONPUBLICATION_TERMINAL_CONTRACT_V3_3_4_5
      ),
      'prior_v3_3_4_3_consumed_preflight_prefix': (
          bootstrap.validate_prior_v3_3_4_3_consumed_preflight_prefix()
      ),
      'prior_v3_3_4_4_consumed_preflight_prefix': (
          bootstrap.validate_prior_v3_3_4_4_consumed_preflight_prefix()
      ),
      'source_inventory_contract': {},
      'v3_3_3_1_archive': bootstrap.validate_v3_3_3_1_archive(),
  })
  base['program_signatures'] = json.loads(
      bootstrap.V3_3_4_4_FREEZE_PATH.read_text(encoding='utf-8')
  )['program_signatures']
  files = set(base['file_sha256']) | set(_EXTRA_FILES)
  base['file_sha256'] = {
      relative: _sha256(_REPO / relative) for relative in sorted(files)
  }
  rows = []
  for relative, digest in base['file_sha256'].items():
    path = _REPO / relative
    mode_result = subprocess.run(
        ('git', '-C', str(_REPO), 'ls-files', '-s', '--', relative),
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if mode_result:
      git_mode = mode_result.split()[0]
    else:
      git_mode = '100755' if relative.endswith('.sh') else '100644'
    rows.append({
        'path': relative,
        'sha256': digest,
        'size_bytes': path.stat().st_size,
        'git_mode': git_mode,
    })
  base['source_inventory_contract'] = {
      'source_row_count': 132,
      'rows': rows,
      'prospective_upstream_source_file_count': 26,
      'loaded_scientific_module_contract': runner.loaded_scientific_modules(),
  }
  if (
      len(base) != 86 or len(base['file_sha256']) != 132
      or len(rows) != 132
  ):
    raise RuntimeError('v3.3.4.5 freeze is not exactly 86/132/132.')
  return base


def _execution_contract() -> dict[str, object]:
  cases = runner.v32.load_development_cases()
  recipient_cases = [runner._case_record(case) for case in cases]  # pylint: disable=protected-access
  order = [[recipient, anchor] for recipient in range(20)
           for anchor in runner.ANCHOR_IDS]
  return {
      'recipient_cases': recipient_cases,
      'donor_order': [runner.v33.OOD_DONOR_ORDER[index] for index in range(20)],
      'recipient_orders': list(range(20)),
      'anchor_ids': list(runner.ANCHOR_IDS),
      'call_roles': list(runner.CALL_ROLES),
      'execution_order': order,
      'record_count': 80,
      'applies_per_record': 4,
      'expected_model_apply_count': 320,
      'eight_row_roles': list(runner.v33.EIGHT_ROLES),
      'natural_identity_rows': list(runner.v33.EIGHT_IDENTITY_ROWS),
      'intended_donor_rows': list(runner.v33.EIGHT_INTENDED_DONOR_ROWS),
      'unrelated_donor_rows': list(runner.v33.EIGHT_UNRELATED_DONOR_ROWS),
      'invariant_rows': list(runner.INVARIANT_ROWS),
      'active_recipient_rows': list(runner.ACTIVE_RECIPIENT_ROWS),
      'eight_row_compile_count': 1,
      'six_row_compile_count': 0,
      'identity_rerun_count': 0,
      'main_cube_rerun_count': 0,
      'old_ood_records_reused': 0,
      'confirmation_model_calls': 0,
      'raw_path_template': (
          'raw/ood_anchors/{recipient_order:03d}_{slug}/'
          '{anchor_id:03d}.json'
      ),
      'started_event_path_template': (
          'dispatch_journal/started/{global_dispatch_index:03d}.json'
      ),
      'completed_event_path_template': (
          'dispatch_journal/completed/{global_dispatch_index:03d}.json'
      ),
      'failed_current_path_template': (
          'raw/failed_current/{execution_index:03d}_{slug}/'
          '{anchor_id:03d}.json'
      ),
  }


def main() -> None:
  value = build_freeze()
  payload = (
      json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
  )
  descriptor = os.open(
      bootstrap.FREEZE_PATH,
      os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
      0o600,
  )
  os.fchmod(descriptor, 0o644)
  with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
    handle.write(payload)
  print(bootstrap.FREEZE_PATH)
  print(_sha256(bootstrap.FREEZE_PATH))
  print(len(value['file_sha256']))


if __name__ == '__main__':
  main()
