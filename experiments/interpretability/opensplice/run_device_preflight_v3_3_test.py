#!/usr/bin/env python3
"""Standard-library/CPU tests for the v3.3 preflight contract."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest import mock

import run_device_preflight_v3_3 as preflight
import launch_encoder_skip_factorial_v3_3 as launcher


class DevicePreflightV33Test(unittest.TestCase):

  def test_environment_rejects_cache_inputs(self):
    clean = {
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
        'JAX_ENABLE_COMPILATION_CACHE': 'false',
    }
    with mock.patch.dict(os.environ, clean, clear=True):
      observed = preflight.assert_environment()
    self.assertEqual(observed['JAX_ENABLE_COMPILATION_CACHE'], 'false')
    with mock.patch.dict(
        os.environ,
        {**clean, 'JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES': '1'},
        clear=True,
    ):
      with self.assertRaisesRegex(ValueError, 'Forbidden'):
        preflight.assert_environment()

  def test_dry_plan_has_no_model_jit_or_output(self):
    plan = preflight.build_dry_run_plan()
    self.assertEqual(plan['model_calls'], 0)
    self.assertEqual(plan['jit_calls'], 0)
    self.assertFalse(plan['scientific_output_created'])
    self.assertEqual(
        plan['required_device']['device_kind'], 'NVIDIA GeForce RTX 3090'
    )

  def test_launcher_sanitizes_before_model_import(self):
    dirty = {
        'LD_LIBRARY_PATH': '/bad',
        'XLA_FLAGS': '--bad',
        'JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES': '1',
    }
    with mock.patch.dict(os.environ, dirty, clear=True):
      record = launcher._sanitize_environment()  # pylint: disable=protected-access
      self.assertNotIn('LD_LIBRARY_PATH', os.environ)
      self.assertNotIn('XLA_FLAGS', os.environ)
      self.assertEqual(os.environ['JAX_ENABLE_COMPILATION_CACHE'], 'false')
    self.assertEqual(record['LD_LIBRARY_PATH'], 'absent')

  def test_prior_v3_2_evidence_is_hardcoded_and_byte_validated(self):
    expected = preflight.bootstrap.EXPECTED_PRIOR_V3_2_EVIDENCE
    observed = preflight.bootstrap.validate_prior_v3_2_evidence({
        'prior_v3_2_evidence': expected
    })
    self.assertEqual(observed, expected)
    tampered = copy.deepcopy(expected)
    tampered['raw_manifest']['artifact_tree_sha256'] = '0' * 64
    with self.assertRaisesRegex(ValueError, 'protocol'):
      preflight.bootstrap.validate_prior_v3_2_evidence({
          'prior_v3_2_evidence': tampered
      })

  def test_upstream_import_inventory_is_tracked_and_byte_exact(self):
    source = (
        preflight._HERE / 'results'  # pylint: disable=protected-access
        / 'v3_2_development_superset_graph_one_shot'
        / 'IMPORT_PROVENANCE.json'
    )
    provenance = json.loads(source.read_text(encoding='utf-8'))
    root = preflight.bootstrap._UPSTREAM  # pylint: disable=protected-access
    inventory = {}
    for item in provenance['modules']:
      if item['root'] != 'upstream_alphagenome_checkout':
        continue
      inventory[item['name']] = {
          'relative_path': str(Path(item['path']).resolve().relative_to(root)),
          'sha256': item['sha256'],
          'size_bytes': item['size_bytes'],
      }
    frozen = {
        'upstream_alphagenome_git_head': subprocess.check_output(
            ('git', '-C', str(root), 'rev-parse', 'HEAD'), text=True
        ).strip(),
        'upstream_imported_modules': inventory,
        'upstream_generated_binding_exception': (
            preflight.bootstrap.EXPECTED_UPSTREAM_GENERATED_BINDING_EXCEPTION
        ),
    }
    observed = preflight.bootstrap.validate_upstream_checkout(frozen)
    self.assertEqual(observed['imported_module_count'], 26)
    self.assertEqual(observed['tracked_imported_module_count'], 22)
    self.assertEqual(observed['generated_imported_module_count'], 4)
    self.assertTrue(observed['tracked_head_clean'])
    tampered = copy.deepcopy(frozen)
    tampered['upstream_generated_binding_exception'][
        'generated_outputs'
    ]['alphagenome.protos.tensor_pb2']['size_bytes'] += 1
    with self.assertRaisesRegex(ValueError, 'protocol §2.3'):
      preflight.bootstrap.validate_upstream_checkout(tampered)


if __name__ == '__main__':
  unittest.main()
