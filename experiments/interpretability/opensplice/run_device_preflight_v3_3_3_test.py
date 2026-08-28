#!/usr/bin/env python3
"""Standard-library/CPU tests for the v3.3.3 sidecar preflight."""

from __future__ import annotations

import os
from pathlib import Path
import json
import subprocess
import tempfile
import unittest
from unittest import mock

_MODULE_CACHE = tempfile.TemporaryDirectory(
    prefix='alphagenome-v3.3.3-dry-cache.'
)
_MODULE_CACHE_ROOT = Path(_MODULE_CACHE.name).resolve()
(_MODULE_CACHE_ROOT / 'triton').mkdir()
(_MODULE_CACHE_ROOT / 'xdg').mkdir()
os.environ.update({
    'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
    'JAX_ENABLE_COMPILATION_CACHE': 'false',
    'CUDA_CACHE_DISABLE': '1',
    'ALPHAGENOME_V3_3_3_CACHE_ROLE': 'dry_run',
    'ALPHAGENOME_V3_3_3_CACHE_ROOT': str(_MODULE_CACHE_ROOT),
    'TRITON_CACHE_DIR': str(_MODULE_CACHE_ROOT / 'triton'),
    'XDG_CACHE_HOME': str(_MODULE_CACHE_ROOT / 'xdg'),
})

import launch_encoder_skip_ood_sidecar_v3_3_3 as launcher
import run_device_preflight_v3_3_3 as preflight


class DevicePreflightV333Test(unittest.TestCase):

  def _module_live_environment(self) -> dict[str, str]:
    return {
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
        'JAX_ENABLE_COMPILATION_CACHE': 'false',
        'CUDA_CACHE_DISABLE': '1',
        'ALPHAGENOME_V3_3_3_CACHE_ROLE': 'dry_run',
        'ALPHAGENOME_V3_3_3_CACHE_ROOT': str(_MODULE_CACHE_ROOT),
        'TRITON_CACHE_DIR': str(_MODULE_CACHE_ROOT / 'triton'),
        'XDG_CACHE_HOME': str(_MODULE_CACHE_ROOT / 'xdg'),
    }

  def _fresh_cache_environment(self, root: Path) -> dict[str, str]:
    (root / 'triton').mkdir()
    (root / 'xdg').mkdir()
    return {
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
        'JAX_ENABLE_COMPILATION_CACHE': 'false',
        'CUDA_CACHE_DISABLE': '1',
        'ALPHAGENOME_V3_3_3_CACHE_ROLE': 'dry_run',
        'ALPHAGENOME_V3_3_3_CACHE_ROOT': str(root),
        'TRITON_CACHE_DIR': str(root / 'triton'),
        'XDG_CACHE_HOME': str(root / 'xdg'),
    }

  def test_cuda_triton_and_autotune_cache_inputs_fail_closed(self):
    for mutation, pattern in (
        ({'CUDA_CACHE_DISABLE': '0'}, 'CUDA_CACHE_DISABLE'),
        ({'TRITON_CACHE_DIR': '/home/degen2/.triton/cache'}, 'TRITON'),
        ({'XLA_AUTOTUNE_CACHE_DIR': '/tmp/old'}, 'Forbidden'),
    ):
      with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
          prefix='alphagenome-v3.3.3-dry-cache.'
      ) as directory:
        root = Path(directory)
        environment = self._fresh_cache_environment(root)
        environment.update(mutation)
        with mock.patch.dict(os.environ, environment, clear=True):
          with self.assertRaisesRegex(ValueError, pattern):
            preflight.bootstrap.assert_cache_environment_sanitized()

  def test_nonempty_or_symlink_cache_tree_fails_closed(self):
    with tempfile.TemporaryDirectory(
        prefix='alphagenome-v3.3.3-dry-cache.'
    ) as directory:
      root = Path(directory)
      environment = self._fresh_cache_environment(root)
      (root / 'triton' / 'old.bin').write_bytes(b'prior-cache')
      with mock.patch.dict(os.environ, environment, clear=True):
        with self.assertRaisesRegex(ValueError, 'not empty'):
          preflight.bootstrap.assert_cache_environment_sanitized()

  def test_environment_rejects_cache_inputs(self):
    clean = self._module_live_environment()
    with mock.patch.dict(os.environ, clean, clear=True):
      observed = preflight.assert_environment()
    self.assertEqual(observed['JAX_ENABLE_COMPILATION_CACHE'], 'false')
    self.assertTrue(
        observed['live_cache_environment']['exact_to_pre_import_routing']
    )
    with mock.patch.dict(
        os.environ,
        {**clean, 'JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES': '1'},
        clear=True,
    ):
      with self.assertRaisesRegex(ValueError, 'Forbidden'):
        preflight.assert_environment()

  def test_post_import_cache_routing_mutation_fails_closed(self):
    clean = self._module_live_environment()
    for name, value in (
        ('CUDA_CACHE_DISABLE', '0'),
        ('TRITON_CACHE_DIR', '/tmp/wrong-triton'),
        ('XDG_CACHE_HOME', '/tmp/wrong-xdg'),
        ('ALPHAGENOME_V3_3_3_CACHE_ROLE', 'model'),
        ('ALPHAGENOME_V3_3_3_CACHE_ROOT', '/tmp/wrong-root'),
    ):
      with self.subTest(name=name), mock.patch.dict(
          os.environ, {**clean, name: value}, clear=True
      ):
        with self.assertRaisesRegex(ValueError, 'Live cache routing changed'):
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
    with tempfile.TemporaryDirectory(
        prefix='alphagenome-v3.3.3-dry-cache.'
    ) as directory:
      root = Path(directory)
      (root / 'triton').mkdir()
      (root / 'xdg').mkdir()
      dirty = {
          'LD_LIBRARY_PATH': '/bad',
          'XLA_FLAGS': '--bad',
          'JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES': '1',
          'CUDA_CACHE_DISABLE': '1',
          'ALPHAGENOME_V3_3_3_CACHE_ROLE': 'dry_run',
          'ALPHAGENOME_V3_3_3_CACHE_ROOT': str(root),
          'TRITON_CACHE_DIR': str(root / 'triton'),
          'XDG_CACHE_HOME': str(root / 'xdg'),
      }
      with mock.patch.dict(os.environ, dirty, clear=True):
        record = launcher._sanitize_environment()  # pylint: disable=protected-access
        self.assertNotIn('LD_LIBRARY_PATH', os.environ)
        self.assertNotIn('XLA_FLAGS', os.environ)
        self.assertEqual(os.environ['JAX_ENABLE_COMPILATION_CACHE'], 'false')
        self.assertEqual(os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'], 'false')
        self.assertEqual(os.environ['CUDA_CACHE_DISABLE'], '1')
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
      analysis_attempt = root / 'analysis-attempt'
      observed = preflight.bootstrap.validate_one_shot_output_absence(
          output, analysis, analysis_attempt
      )
      self.assertTrue(observed['output_dir_absent'])
      output.mkdir()
      (output / 'sentinel').write_text('consumed', encoding='utf-8')
      with self.assertRaisesRegex(FileExistsError, 'never resume'):
        preflight.bootstrap.validate_one_shot_output_absence(
            output, analysis, analysis_attempt
        )
      (output / 'sentinel').unlink()
      output.rmdir()
      analysis_attempt.mkdir()
      (analysis_attempt / 'sentinel').write_text('consumed', encoding='utf-8')
      with self.assertRaisesRegex(FileExistsError, 'analysis attempt'):
        preflight.bootstrap.validate_one_shot_output_absence(
            output, analysis, analysis_attempt
        )

  def test_cache_root_reservation_is_atomic_and_wrapper_uses_it(self):
    wrapper = (
        Path(__file__).resolve().parent
        / 'run_encoder_skip_ood_sidecar_v3_3_3.sh'
    )
    source = wrapper.read_text(encoding='utf-8')
    self.assertIn('if ! mkdir -- "$root"; then', source)
    self.assertNotIn('mkdir -p -- "$root/triton"', source)
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / 'reserved'
      processes = [
          subprocess.Popen(  # pylint: disable=consider-using-with
              ('mkdir', '--', str(root)),
              stdout=subprocess.DEVNULL,
              stderr=subprocess.DEVNULL,
          )
          for _ in range(2)
      ]
      returncodes = sorted(process.wait() for process in processes)
      self.assertEqual(returncodes, [0, 1])

  def test_model_bootstrap_binds_the_sole_preflight_before_import(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      preflight_dir = root / 'preflight'
      preflight_dir.mkdir()
      cache_root = root / 'external-cache'
      (cache_root / 'triton').mkdir(parents=True)
      (cache_root / 'xdg').mkdir()
      freeze = root / 'freeze.json'
      freeze.write_text('{}\n', encoding='utf-8')
      for name in (
          '.allocation.lock',
          '.preflight_0000.reserved',
          'preflight_0000.stdout.log',
          'preflight_0000.stderr.log',
      ):
        (preflight_dir / name).write_bytes(b'')
      record = {
          'script_version': preflight.SCRIPT_VERSION,
          'status': 'pass',
          'preflight_attempt_number': 0,
          'amendment_sha256': preflight.bootstrap.AMENDMENT_SHA256,
          'original_protocol_sha256': (
              preflight.bootstrap.ORIGINAL_PROTOCOL_SHA256
          ),
          'freeze_sha256': preflight.bootstrap._sha256(freeze),  # pylint: disable=protected-access
          'failure': None,
          'no_model_or_biological_access': True,
          'no_jit_or_array_kernel': True,
          'logs': {
              stream: {
                  'path': str(
                      (preflight_dir / f'preflight_0000.{stream}.log')
                      .resolve()
                  ),
                  'sha256': preflight.bootstrap._sha256(  # pylint: disable=protected-access
                      preflight_dir / f'preflight_0000.{stream}.log'
                  ),
              }
              for stream in ('stdout', 'stderr')
          },
          'observation': {
              'v3_3_3_runtime_environment': {
                  'cache_environment': {
                      'cache_role': 'external_preflight',
                      'cache_root': str(cache_root.resolve()),
                      'pre_import_file_count': 0,
                  }
              }
          },
          'external_cache_post_observation': (
              preflight.bootstrap.cache_output_tree_binding(cache_root)
          ),
      }
      (preflight_dir / 'preflight_0000.json').write_text(
          json.dumps(record) + '\n', encoding='utf-8'
      )
      with (
          mock.patch.object(preflight.bootstrap, 'PREFLIGHT_DIR', preflight_dir),
          mock.patch.object(
              preflight.bootstrap, 'PREFLIGHT_KERNEL_CACHE_DIR', cache_root
          ),
          mock.patch.object(preflight.bootstrap, 'FREEZE_PATH', freeze),
          mock.patch.dict(
              os.environ,
              {preflight.bootstrap.CACHE_ROLE_ENVIRONMENT: 'model'},
              clear=False,
          ),
      ):
        binding = preflight.bootstrap.validate_preflight_state_for_role()
        self.assertEqual(binding['file_count'], 5)
        self.assertTrue(binding['sole_successful_preflight_exact'])
        (preflight_dir / 'preflight_0001.json').write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'one exact preflight'):
          preflight.bootstrap.validate_preflight_state_for_role()


if __name__ == '__main__':
  unittest.main()
