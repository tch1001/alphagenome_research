#!/usr/bin/env python3
"""Standard-library-only pre-import source/protobuf gate for v3.2."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
from typing import Any


_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
PROTOCOL_PATH = (
    _HERE / 'v3_wider_mechanism' / 'superset_graph_protocol_v3_2.md'
)
PROTOCOL_SHA256 = (
    '1e87839250e838d1d9aa95162e76a3acb6b1eaab59b481c60dab943b62f6caea'
)
FREEZE_PATH = _HERE / 'superset_graph_v3_2_freeze.json'
EXPECTED = {
    'source_proto': (
        _REPO / 'src/alphagenome_research/protos/calibration_scores.proto',
        '356f08689a4bafa0761f88f08dac08468a2de2c8aef38dcef093457eceee2f34',
        None,
    ),
    'generated_pb2': (
        _REPO / 'src/alphagenome_research/protos/calibration_scores_pb2.py',
        '4673289dd481fd8c4976f602ab36b07646304107e352e3e6d27b2abe4f9e9ebc',
        2794,
    ),
    'generated_pyi': (
        _REPO / 'src/alphagenome_research/protos/calibration_scores_pb2.pyi',
        '329dc390abeb187084fff28fbe6cb6d9868aa8867326bf53f9a52d4c83f527f9',
        1815,
    ),
    'dependency_proto': (
        _REPO.parent / 'alphagenome/src/alphagenome/protos/dna_model.proto',
        'd19a7208ec34953ca021efbff32516f1aa277f0477276f7699d9567fd616329a',
        None,
    ),
    'dependency_pb2': (
        _REPO.parent / 'alphagenome/src/alphagenome/protos/dna_model_pb2.py',
        'd97564536e77ec09bdf144ba1204d4e08f79095fb9ed6c0cba7b065dc6f252ee',
        None,
    ),
    'tensor_proto': (
        _REPO.parent / 'alphagenome/src/alphagenome/protos/tensor.proto',
        '07779023b2868377cbfc3c2ce96cd266ae425a0a1116aea755691c263d6238f7',
        None,
    ),
    'tensor_pb2': (
        _REPO.parent / 'alphagenome/src/alphagenome/protos/tensor_pb2.py',
        'dea7a5207e82601b6763e95ee4b69356345e95bc2de91632928d6161f873cdb8',
        None,
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


def validate_generated_bindings_before_import() -> dict[str, Any]:
  records = {}
  for name, (path, expected_sha, expected_size) in EXPECTED.items():
    _reject_confirmation_path(path)
    if not path.is_file():
      raise ValueError(f'Frozen protobuf path is absent: {path}.')
    observed_sha = _sha256(path)
    observed_size = path.stat().st_size
    if observed_sha != expected_sha:
      raise ValueError(f'Frozen protobuf hash changed: {name}.')
    if expected_size is not None and observed_size != expected_size:
      raise ValueError(f'Frozen protobuf size changed: {name}.')
    records[name] = {
        'path': str(path.resolve()), 'sha256': observed_sha,
        'size_bytes': observed_size,
    }
  header = EXPECTED['generated_pb2'][0].read_text(
      encoding='utf-8'
  ).splitlines()[:8]
  if not any('Protobuf Python Version: 7.35.1' in line for line in header):
    raise ValueError('Frozen generated protobuf version header changed.')
  if importlib.metadata.version('protobuf') != '7.35.1':
    raise ValueError('Runtime protobuf distribution is not frozen 7.35.1.')
  status = subprocess.check_output(
      ('git', '-C', str(_REPO), 'status', '--porcelain=v1',
       '--untracked-files=all', '--', 'src/alphagenome_research/protos'),
      text=True,
  ).splitlines()
  expected_status = {
      '?? src/alphagenome_research/protos/calibration_scores_pb2.py',
      '?? src/alphagenome_research/protos/calibration_scores_pb2.pyi',
  }
  if set(status) != expected_status:
    raise ValueError(f'Generated-binding untracked allowlist changed: {status}.')
  return {
      'pre_import_gate': True,
      'historical_generator_argv': 'unknown',
      'exact_regeneration_claim': False,
      'generated_artifact_exception': sorted(expected_status),
      'artifacts': records,
      'embedded_header': header,
      'protobuf_runtime_version': '7.35.1',
  }


def validate_freeze() -> dict[str, Any]:
  _reject_confirmation_path(FREEZE_PATH)
  _reject_confirmation_path(PROTOCOL_PATH)
  freeze_relative = str(FREEZE_PATH.relative_to(_REPO))
  protocol_relative = str(PROTOCOL_PATH.relative_to(_REPO))
  for relative in (freeze_relative, protocol_relative):
    subprocess.run(
        ('git', '-C', str(_REPO), 'ls-files', '--error-unmatch', relative),
        check=True,
        capture_output=True,
    )
  if subprocess.check_output(
      (
          'git', '-C', str(_REPO), 'diff', '--binary', 'HEAD', '--',
          freeze_relative, protocol_relative,
      )
  ):
    raise ValueError('v3.2 freeze/protocol differs from committed HEAD.')
  if _sha256(PROTOCOL_PATH) != PROTOCOL_SHA256:
    raise ValueError('v3.2 protocol hash changed.')
  frozen = json.loads(FREEZE_PATH.read_text(encoding='utf-8'))
  if frozen.get('protocol_sha256') != PROTOCOL_SHA256:
    raise ValueError('v3.2 freeze protocol binding changed.')
  tracked_relatives = []
  for relative, expected_sha in frozen['file_sha256'].items():
    path = (_REPO / relative).resolve()
    try:
      normalized_relative = str(path.relative_to(_REPO))
    except ValueError as error:
      raise ValueError(
          f'Frozen bundle path escapes the repository: {relative}.'
      ) from error
    if normalized_relative != relative:
      raise ValueError(f'Frozen bundle path is not normalized: {relative}.')
    _reject_confirmation_path(path)
    subprocess.run(
        (
            'git', '-C', str(_REPO), 'ls-files', '--error-unmatch',
            normalized_relative,
        ),
        check=True,
        capture_output=True,
    )
    if _sha256(path) != expected_sha:
      raise ValueError(f'v3.2 bundle file changed: {relative}.')
    tracked_relatives.append(normalized_relative)
  if subprocess.check_output(
      (
          'git', '-C', str(_REPO), 'diff', '--binary', 'HEAD', '--',
          *tracked_relatives,
      )
  ):
    raise ValueError('Committed v3.2 bundle differs from the working tree.')
  return {
      'path': str(FREEZE_PATH.resolve()),
      'sha256': _sha256(FREEZE_PATH),
      'git_head': subprocess.check_output(
          ('git', '-C', str(_REPO), 'rev-parse', 'HEAD'), text=True
      ).strip(),
      'tracked_head_clean': True,
      'tracked_paths': sorted(
          {freeze_relative, protocol_relative, *tracked_relatives}
      ),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--check', action='store_true', required=True)
  result = {
      'generated_bindings': validate_generated_bindings_before_import(),
      'freeze': validate_freeze(),
      'model_or_jax_imported': False,
  }
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
  main()
