#!/usr/bin/env python3
"""Standard-library-only pre-import gate for the v3.3.2 OOD sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    / 'encoder_skip_ood_sidecar_amendment_v3_3_2.md'
)
AMENDMENT_SHA256 = (
    '42cd43bbc25517d35b3e21dd5df7bf68a37ee46c51cb71bcdf363ac4de3b19e3'
)
AMENDMENT_COMMIT = '95d028f'
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
FREEZE_PATH = _HERE / 'encoder_skip_ood_sidecar_v3_3_2_freeze.json'
OUTPUT_DIR = (
    _HERE / 'results' / 'v3_3_2_development_ood_sidecar_one_shot'
)
ANALYSIS_DIR = (
    _HERE / 'results' / 'v3_3_2_development_ood_sidecar_analysis'
)
PREFLIGHT_DIR = _HERE / 'results' / 'v3_3_2_device_preflight'
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


def _reject_confirmation_path(path: Path) -> None:
  if any('confirm' in part.lower() for part in path.resolve().parts):
    raise ValueError(f'Confirmation-named path is forbidden: {path}.')


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
) -> dict[str, Any]:
  """Fail-closes before model import if scientific sidecar output exists."""
  for path, label in (
      (output_dir, 'v3.3.2 sidecar output'),
      (analysis_dir, 'v3.3.2 sidecar analysis'),
  ):
    _reject_confirmation_path(path)
    if path.exists() or path.is_symlink():
      raise FileExistsError(f'{label} already exists; never resume or retry.')
  return {
      'output_dir': str(output_dir.resolve()),
      'output_dir_absent': True,
      'analysis_dir': str(analysis_dir.resolve()),
      'analysis_dir_absent': True,
      'preflight_dir_may_exist': True,
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


def validate_freeze() -> dict[str, Any]:
  for path in (
      FREEZE_PATH, AMENDMENT_PATH, ORIGINAL_FREEZE_PATH, OUTPUT_DIR,
      ANALYSIS_DIR, PREFLIGHT_DIR,
  ):
    _reject_confirmation_path(path)
  _validate_file(AMENDMENT_PATH, AMENDMENT_SHA256, 'v3.3.2 amendment')
  _validate_file(ORIGINAL_FREEZE_PATH, ORIGINAL_FREEZE_SHA256, 'v3.3 freeze')
  frozen = json.loads(FREEZE_PATH.read_text(encoding='utf-8'))
  expected = {
      'amendment_sha256': AMENDMENT_SHA256,
      'amendment_commit': AMENDMENT_COMMIT,
      'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
      'original_freeze_sha256': ORIGINAL_FREEZE_SHA256,
      'output_dir': str(OUTPUT_DIR.resolve()),
      'analysis_dir': str(ANALYSIS_DIR.resolve()),
      'preflight_dir': str(PREFLIGHT_DIR.resolve()),
      'ood_anchor_ids': [0, 127, 128, 255],
      'recipient_orders': list(range(20)),
      'ood_record_count': 80,
      'model_apply_count': 320,
      'eight_row_compile_count': 1,
      'six_row_compile_count': 0,
      'identity_rerun_count': 0,
      'main_cube_rerun_count': 0,
      'invariant_rows_between_calls': [0, 1, 3, 5, 6, 7],
  }
  for name, value in expected.items():
    if frozen.get(name) != value:
      raise ValueError(f'v3.3.2 freeze mismatch: {name}.')
  if frozen.get('original_run') != EXPECTED_ORIGINAL_BINDING:
    raise ValueError('Frozen original-v3.3 binding differs from the amendment.')

  tracked = {str(FREEZE_PATH.relative_to(_REPO))}
  file_sha = frozen.get('file_sha256')
  if not isinstance(file_sha, dict) or not file_sha:
    raise ValueError('v3.3.2 frozen file inventory is absent.')
  for relative, expected_sha in sorted(file_sha.items()):
    path = (_REPO / relative).resolve()
    if not path.is_relative_to(_REPO.resolve()) or str(path.relative_to(_REPO)) != relative:
      raise ValueError(f'Frozen path escaped or is not normalized: {relative}.')
    _validate_file(path, expected_sha, f'v3.3.2 bundle {relative}')
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
    raise ValueError('v3.3.2 requires globally tracked-clean HEAD before import.')

  original_bundle = v33_bootstrap.validate_freeze()
  original_run = validate_original_run()
  v3_3_1 = validate_v3_3_1_status(frozen)
  one_shot_absence = validate_one_shot_output_absence()
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
      'v3_3_1_status': v3_3_1,
      'one_shot_output_absence': one_shot_absence,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--check', action='store_true', required=True)
  record = {
      'generated_bindings': (
          v33_bootstrap.proto_gate.validate_generated_bindings_before_import()
      ),
      'freeze': validate_freeze(),
      'model_or_jax_imported': False,
  }
  print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == '__main__':
  main()
