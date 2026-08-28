#!/usr/bin/env python3
"""Standard-library/CPU tests for the v3.3.2 sidecar preflight."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import launch_encoder_skip_ood_sidecar_v3_3_2 as launcher
import run_device_preflight_v3_3_2 as preflight


class DevicePreflightV332Test(unittest.TestCase):

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

  def test_dry_plan_is_jax_only_and_creates_no_output(self):
    plan = preflight.build_dry_run_plan()
    self.assertEqual(plan['model_calls'], 0)
    self.assertEqual(plan['jit_calls'], 0)
    self.assertFalse(plan['scientific_output_created'])
    self.assertEqual(
        plan['required_device']['device_kind'], 'NVIDIA GeForce RTX 3090'
    )
    self.assertEqual(plan['amendment_sha256'], preflight.bootstrap.AMENDMENT_SHA256)

  def test_launcher_sanitizes_before_any_model_import(self):
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
      self.assertEqual(os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'], 'false')
    self.assertEqual(record['LD_LIBRARY_PATH'], 'absent')

  def test_consumed_original_run_is_rehashed_without_score_parsing(self):
    observed = preflight.bootstrap.validate_original_run()
    self.assertEqual(observed['raw_artifact_count'], 5142)
    self.assertEqual(observed['whole_run_file_count'], 5158)
    self.assertEqual(observed['compiler_file_count'], 8)
    self.assertEqual(
        observed['status_predicates']['stop_reason'], 'ood_tooling_failure'
    )
    self.assertEqual(
        observed['eight_row_compiler']['executable_fingerprint'],
        '12283496a0987eec942bd8f9b7bbb86a9d9d676b13bee1956b30da933a4e9967',
    )

  def test_completed_v3_3_1_attempt_and_analysis_are_exactly_bound(self):
    expected = preflight.bootstrap.EXPECTED_V3_3_1_COMPLETED_STATUS
    observed = preflight.bootstrap.validate_v3_3_1_status({
        'v3_3_1_status': expected
    })
    self.assertEqual(observed, expected)
    changed = dict(expected)
    changed['analysis_tree_sha256'] = '0' * 64
    with self.assertRaisesRegex(ValueError, 'exact audit bytes'):
      preflight.bootstrap.validate_v3_3_1_status({
          'v3_3_1_status': changed
      })

  def test_strict_tree_rejects_symlink_special_and_empty_directory(self):
    strict = preflight.bootstrap._strict_file_tree  # pylint: disable=protected-access
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      (root / 'file').write_bytes(b'x')
      self.assertEqual(strict(root), [root / 'file'])
      (root / 'link').symlink_to(root / 'file')
      with self.assertRaisesRegex(ValueError, 'symlink'):
        strict(root)
      (root / 'link').unlink()
      (root / 'empty').mkdir()
      with self.assertRaisesRegex(ValueError, 'empty directory'):
        strict(root)
      (root / 'empty').rmdir()
      os.mkfifo(root / 'fifo')
      with self.assertRaisesRegex(ValueError, 'special entry'):
        strict(root)

  def test_preexisting_scientific_output_blocks_before_model_import(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      output = root / 'output'
      analysis = root / 'analysis'
      observed = preflight.bootstrap.validate_one_shot_output_absence(
          output, analysis
      )
      self.assertTrue(observed['output_dir_absent'])
      output.mkdir()
      (output / 'sentinel').write_text('consumed', encoding='utf-8')
      with self.assertRaisesRegex(FileExistsError, 'never resume'):
        preflight.bootstrap.validate_one_shot_output_absence(output, analysis)


if __name__ == '__main__':
  unittest.main()
