#!/usr/bin/env python3
"""Generate the prospective v3.3.3 machine freeze from frozen predecessors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

# pylint: disable=g-import-not-at-top
import validate_encoder_skip_ood_sidecar_bootstrap_v3_3_3 as bootstrap


_EXTRA_FILES = (
    'experiments/interpretability/opensplice/'
    'analyze_encoder_skip_ood_sidecar_v3_3_2_1.py',
    'experiments/interpretability/opensplice/'
    'analyze_encoder_skip_ood_sidecar_v3_3_2_1_test.py',
    'experiments/interpretability/opensplice/'
    'encoder_skip_ood_sidecar_analysis_v3_3_2_1_freeze.json',
    'experiments/interpretability/opensplice/'
    'run_encoder_skip_ood_sidecar_analysis_v3_3_2_1.sh',
    'experiments/interpretability/opensplice/v3_wider_mechanism/'
    'encoder_skip_ood_sidecar_analysis_amendment_v3_3_2_1.md',
    'experiments/interpretability/opensplice/'
    'analyze_encoder_skip_ood_sidecar_v3_3_2_2.py',
    'experiments/interpretability/opensplice/'
    'analyze_encoder_skip_ood_sidecar_v3_3_2_2_test.py',
    'experiments/interpretability/opensplice/'
    'encoder_skip_ood_sidecar_analysis_v3_3_2_2_freeze.json',
    'experiments/interpretability/opensplice/'
    'run_encoder_skip_ood_sidecar_analysis_v3_3_2_2.sh',
    'experiments/interpretability/opensplice/v3_wider_mechanism/'
    'encoder_skip_ood_sidecar_analysis_amendment_v3_3_2_2.md',
    'experiments/interpretability/opensplice/v3_wider_mechanism/'
    'encoder_skip_ood_sidecar_compiler_gate_amendment_v3_3_3.md',
    'experiments/interpretability/opensplice/'
    'validate_encoder_skip_ood_sidecar_bootstrap_v3_3_3.py',
    'experiments/interpretability/opensplice/'
    'launch_encoder_skip_ood_sidecar_v3_3_3.py',
    'experiments/interpretability/opensplice/run_device_preflight_v3_3_3.py',
    'experiments/interpretability/opensplice/'
    'run_device_preflight_v3_3_3_test.py',
    'experiments/interpretability/opensplice/'
    'run_encoder_skip_ood_sidecar_v3_3_3.py',
    'experiments/interpretability/opensplice/'
    'run_encoder_skip_ood_sidecar_v3_3_3_test.py',
    'experiments/interpretability/opensplice/'
    'run_encoder_skip_ood_sidecar_v3_3_3.sh',
    'experiments/interpretability/opensplice/'
    'analyze_encoder_skip_ood_sidecar_v3_3_3.py',
    'experiments/interpretability/opensplice/'
    'analyze_encoder_skip_ood_sidecar_v3_3_3_test.py',
    'experiments/interpretability/opensplice/'
    'generate_encoder_skip_ood_sidecar_freeze_v3_3_3.py',
)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def build_freeze() -> dict[str, object]:
  base = json.loads(
      bootstrap.V3_3_2_FREEZE_PATH.read_text(encoding='utf-8')
  )
  base.update({
      'amendment_commit': bootstrap.AMENDMENT_COMMIT,
      'amendment_path': str(bootstrap.AMENDMENT_PATH.resolve()),
      'amendment_sha256': bootstrap.AMENDMENT_SHA256,
      'analysis_attempt_dir': str(bootstrap.ANALYSIS_ATTEMPT_DIR.resolve()),
      'analysis_dir': str(bootstrap.ANALYSIS_DIR.resolve()),
      'attempt_id': 'opensplice-v3.3.3-development-ood-sidecar-one-shot',
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
      'preflight_script_version': 'opensplice-device-preflight-v3.3.3',
      'script_version': 'opensplice-encoder-skip-ood-sidecar-v3.3.3',
      'source_program_contract': bootstrap.SOURCE_PROGRAM_CONTRACT,
      'v3_3_2_1_failure_status': (
          bootstrap.EXPECTED_V3_3_2_1_FAILURE_STATUS
      ),
      'v3_3_2_2_archive_status': (
          bootstrap.EXPECTED_V3_3_2_2_ARCHIVE_STATUS
      ),
      'v3_3_2_freeze_path': str(bootstrap.V3_3_2_FREEZE_PATH.resolve()),
      'v3_3_2_freeze_sha256': bootstrap.V3_3_2_FREEZE_SHA256,
  })
  v3_3_2_run = bootstrap.validate_v3_3_2_run()
  base['v3_3_2_run'] = v3_3_2_run
  base['program_signatures'] = v3_3_2_run['eight_row_compiler'][
      'program_signatures'
  ]
  files = set(base['file_sha256']) | set(_EXTRA_FILES)
  base['file_sha256'] = {
      relative: _sha256(_REPO / relative) for relative in sorted(files)
  }
  return base


def main() -> None:
  value = build_freeze()
  payload = (
      json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
  )
  with bootstrap.FREEZE_PATH.open('x', encoding='utf-8') as handle:
    handle.write(payload)
  print(bootstrap.FREEZE_PATH)
  print(_sha256(bootstrap.FREEZE_PATH))
  print(len(value['file_sha256']))


if __name__ == '__main__':
  main()
