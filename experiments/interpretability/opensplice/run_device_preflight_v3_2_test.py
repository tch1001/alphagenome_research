#!/usr/bin/env python3
"""Standard-library/CPU tests for the v3.2 pre-import and device gates."""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import run_device_preflight_v3_2 as preflight
import validate_superset_graph_bootstrap_v3_2 as bootstrap


class DevicePreflightV32Test(unittest.TestCase):

  def test_generated_bindings_are_exact_before_model_import(self):
    record = bootstrap.validate_generated_bindings_before_import()
    self.assertTrue(record['pre_import_gate'])
    self.assertFalse(record['exact_regeneration_claim'])
    self.assertEqual(record['protobuf_runtime_version'], '7.35.1')
    self.assertNotIn('alphagenome_research.model.dna_model', sys.modules)

  def test_environment_requires_no_cache_inputs(self):
    clean = {
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
        'JAX_ENABLE_COMPILATION_CACHE': 'false',
    }
    with mock.patch.dict(os.environ, clean, clear=True):
      observed = preflight.assert_environment()
    self.assertEqual(observed['JAX_ENABLE_COMPILATION_CACHE'], 'false')
    with mock.patch.dict(
        os.environ, {**clean, 'JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES': '1'},
        clear=True,
    ):
      with self.assertRaisesRegex(ValueError, 'Forbidden'):
        preflight.assert_environment()

  def test_dry_plan_has_no_model_jit_or_scientific_output(self):
    plan = preflight.build_dry_run_plan()
    self.assertEqual(plan['model_calls'], 0)
    self.assertEqual(plan['jit_calls'], 0)
    self.assertFalse(plan['scientific_output_created'])
    self.assertEqual(plan['required_device']['device_kind'], 'NVIDIA GeForce RTX 3090')


if __name__ == '__main__':
  unittest.main()
