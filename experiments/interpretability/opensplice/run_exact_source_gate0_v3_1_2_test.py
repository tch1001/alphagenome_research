"""CPU-only tests for the versioned v3.1.2 device-gated diagnostic."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


_HERE = Path(__file__).resolve().parent
_RUNNER_PATH = _HERE / 'run_exact_source_gate0_v3_1_2.py'
_SPEC = importlib.util.spec_from_file_location(
    'run_exact_source_gate0_v3_1_2_tested', _RUNNER_PATH
)
runner = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)
preflight = runner.device_preflight


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_observation() -> dict:
  return {
      'environment': {
          'LD_LIBRARY_PATH': {'present': False, 'value': None},
          'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
      },
      'jax_default_backend': 'gpu',
      'jax_gpu_devices': ({
          'id': 0,
          'platform': 'gpu',
          'device_kind': preflight.EXPECTED_DEVICE_KIND,
          'client_platform': 'gpu',
      },),
      'nvidia_smi': {
          'returncode': 0,
          'lines': ('one device',),
          'parsed_single_gpu': {
              'index': '0',
              'name': preflight.EXPECTED_DEVICE_KIND,
              'uuid': preflight.EXPECTED_GPU_UUID,
              'compute_capability': preflight.EXPECTED_COMPUTE_CAPABILITY,
              'vbios': 'test',
              'driver': 'test',
          },
      },
      'no_jit_no_array_no_model': True,
  }


class ExactSourceGate0V312Test(unittest.TestCase):

  def test_protocol_and_partial_failure_are_hash_bound(self):
    self.assertEqual(_sha(runner.PROTOCOL_PATH), runner.PROTOCOL_SHA256)
    failure = runner.validate_v3_1_1_partial_failure()
    self.assertEqual(failure['identity_calls'], 0)
    self.assertEqual(failure['compiled_model_executables'], 0)
    self.assertEqual(failure['confirmation_calls'], 0)
    self.assertEqual(
        failure['partial_failure_sha256'],
        runner.V3_1_1_PARTIAL_FAILURE_SHA256,
    )
    self.assertNotEqual(runner.ATTEMPT_ID, runner.V3_1_1_ATTEMPT_ID)
    self.assertNotEqual(runner.OUTPUT_DIR, runner.V3_1_1_RESULT_DIR)

  def test_environment_contract_is_literal_and_fail_closed(self):
    with mock.patch.dict(
        os.environ,
        {'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'},
        clear=True,
    ):
      self.assertFalse(
          preflight.assert_sanitized_environment()['LD_LIBRARY_PATH'][
              'present'
          ]
      )
    with mock.patch.dict(
        os.environ,
        {
            'LD_LIBRARY_PATH': '',
            'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
        },
        clear=True,
    ):
      with self.assertRaisesRegex(ValueError, 'must be absent'):
        preflight.assert_sanitized_environment()
    with mock.patch.dict(
        os.environ,
        {'XLA_PYTHON_CLIENT_PREALLOCATE': 'False'},
        clear=True,
    ):
      with self.assertRaisesRegex(ValueError, 'lowercase'):
        preflight.assert_sanitized_environment()

  def test_device_validation_requires_exact_single_rtx3090_and_uuid(self):
    valid = _valid_observation()
    preflight.validate_device_observation(valid)
    for mutation, message in (
        (('jax_default_backend', 'cpu'), 'default backend'),
        (('jax_gpu_devices', ()), 'exactly one'),
    ):
      changed = dict(valid)
      changed[mutation[0]] = mutation[1]
      with self.assertRaisesRegex(ValueError, message):
        preflight.validate_device_observation(changed)
    changed = json.loads(json.dumps(valid))
    changed['nvidia_smi']['parsed_single_gpu']['uuid'] = 'wrong'
    with self.assertRaisesRegex(ValueError, 'uuid'):
      preflight.validate_device_observation(changed)

  def test_preflight_dry_plan_has_no_model_jit_or_array_work(self):
    plan = preflight.build_dry_run_plan()
    self.assertEqual(plan['imports_if_run'], ('jax', 'jaxlib'))
    self.assertEqual(plan['jit_calls'], 0)
    self.assertEqual(plan['array_kernel_calls'], 0)
    self.assertEqual(plan['model_calls'], 0)
    self.assertFalse(plan['scientific_output_created'])

  def test_committed_bundle_includes_frozen_v311_start_and_disclosure(self):
    names = {path.name for path in preflight._bundle_paths()}  # pylint: disable=protected-access
    self.assertIn('ATTEMPT_STARTED.json', names)
    self.assertIn('PARTIAL_FAILURE.md', names)

  def test_runtime_environment_records_all_compiler_device_prefixes(self):
    with mock.patch.dict(os.environ, {
        'XLA_FLAGS': '--example',
        'JAX_EXAMPLE': 'j',
        'CUDA_EXAMPLE': 'c',
        'NVIDIA_EXAMPLE': 'n',
        'UNRELATED_SECRET': 'not-recorded',
    }, clear=True):
      observed = preflight._runtime_environment()  # pylint: disable=protected-access
    self.assertEqual(observed['XLA_FLAGS'], '--example')
    self.assertEqual(observed['JAX_EXAMPLE'], 'j')
    self.assertEqual(observed['CUDA_EXAMPLE'], 'c')
    self.assertEqual(observed['NVIDIA_EXAMPLE'], 'n')
    self.assertNotIn('UNRELATED_SECRET', observed)

  def test_reservations_are_monotonic_even_after_interrupted_preflight(self):
    with tempfile.TemporaryDirectory() as directory, mock.patch.object(
        preflight, 'PREFLIGHT_DIR', Path(directory)
    ):
      self.assertEqual(preflight._reserve_attempt_number(), 0)  # pylint: disable=protected-access
      self.assertEqual(preflight._reserve_attempt_number(), 1)  # pylint: disable=protected-access
      (Path(directory) / 'preflight_0004.json').write_text(
          '{}\n', encoding='utf-8'
      )
      self.assertEqual(preflight._reserve_attempt_number(), 5)  # pylint: disable=protected-access

  def test_failed_device_check_is_durable_and_repeatable(self):
    with tempfile.TemporaryDirectory() as directory, \
         mock.patch.object(preflight, 'PREFLIGHT_DIR', Path(directory)), \
         mock.patch.object(preflight, 'validate_committed_bundle',
                           return_value={'git_head': 'committed'}), \
         mock.patch.object(preflight, 'validate_freeze',
                           return_value={'sha256': 'frozen'}), \
         mock.patch.object(preflight, 'collect_device_observation',
                           side_effect=RuntimeError('device unavailable')):
      first_path, first_passed = preflight.run_external_preflight()
      second_path, second_passed = preflight.run_external_preflight()
      self.assertFalse(first_passed)
      self.assertFalse(second_passed)
      self.assertNotEqual(first_path, second_path)
      for path in (first_path, second_path):
        record = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(record['status'], 'failure')
        self.assertEqual(record['failure']['message'], 'device unavailable')
        for stream in ('stdout', 'stderr'):
          log = record['logs'][stream]
          self.assertEqual(_sha(Path(log['path'])), log['sha256'])

  def test_external_success_record_is_bound_to_logs_freeze_and_device(self):
    with tempfile.TemporaryDirectory() as directory, mock.patch.object(
        preflight, 'PREFLIGHT_DIR', Path(directory)
    ):
      root = Path(directory)
      logs = {}
      for stream in ('stdout', 'stderr'):
        path = root / f'preflight_0000.{stream}.log'
        path.write_text(f'{stream}\n', encoding='utf-8')
        logs[stream] = {'path': str(path), 'sha256': _sha(path)}
      record = {
          'script_version': preflight.SCRIPT_VERSION,
          'preflight_attempt_number': 0,
          'status': 'pass',
          'no_model_or_biological_access': True,
          'no_jit_or_array_kernel': True,
          'freeze': {'sha256': _sha(runner.FREEZE_PATH)},
          'observation': _valid_observation(),
          'logs': logs,
          'bundle': {'git_head': 'test-head'},
      }
      path = root / 'preflight_0000.json'
      path.write_text(json.dumps(record), encoding='utf-8')
      validated = runner.validate_external_preflight(path)
      self.assertTrue(validated['passed'])
      self.assertEqual(validated['bundle_git_head'], 'test-head')
      record['status'] = 'failure'
      path.write_text(json.dumps(record), encoding='utf-8')
      with self.assertRaisesRegex(ValueError, 'did not pass'):
        runner.validate_external_preflight(path)

  def test_wrapper_sanitizes_before_every_python_process(self):
    text = _RUNNER_PATH.with_suffix('.sh').read_text(encoding='utf-8')
    first_python = text.index('"$PYTHON_BIN"')
    self.assertLess(text.index('unset LD_LIBRARY_PATH'), first_python)
    self.assertLess(
        text.index('export XLA_PYTHON_CLIENT_PREALLOCATE=false'),
        first_python,
    )
    self.assertIn('run_device_preflight_v3_1_2.py" --run', text)
    self.assertIn('--successful-preflight "$SUCCESSFUL_PREFLIGHT"', text)

  def test_same_process_gate_precedes_start_and_model(self):
    text = _RUNNER_PATH.read_text(encoding='utf-8')
    gate = text.index(
        'same_process_preflight = device_preflight.collect_device_observation()'
    )
    start = text.index('base._ensure_fresh_attempt(start_record)')
    model = text.index('model_instance = modules.dna_model.create(')
    self.assertLess(gate, start)
    self.assertLess(start, model)

  def test_launch_freeze_binds_all_versioned_files_and_prior_attempt(self):
    frozen = json.loads(runner.FREEZE_PATH.read_text(encoding='utf-8'))
    validated = runner.validate_freeze(
        proto_build={
            'manifest_sha256': frozen['proto_build_manifest_sha256']
        },
        import_provenance={
            'module_tree_sha256': frozen[
                'initial_transitive_import_tree_sha256'
            ]
        },
        mixed_precision=frozen['mixed_precision'],
        v3_1_1_failure=runner.validate_v3_1_1_partial_failure(),
    )
    self.assertEqual(validated['runner_sha256'], _sha(_RUNNER_PATH))
    self.assertEqual(
        validated['preflight_sha256'],
        _sha(Path(preflight.__file__)),
    )
    self.assertEqual(
        validated['v3_1_1_base_runner_sha256'],
        _sha(Path(runner.base.__file__)),
    )
    self.assertEqual(
        preflight.validate_freeze()['sha256'], _sha(runner.FREEZE_PATH)
    )

  def test_importing_successor_does_not_mutate_v311_output_globals(self):
    self.assertEqual(runner.base.SCRIPT_VERSION,
                     'opensplice-exact-source-gate0-v3.1.1')
    self.assertEqual(runner.base.ATTEMPT_ID, runner.V3_1_1_ATTEMPT_ID)
    self.assertEqual(runner.base.OUTPUT_DIR, runner.V3_1_1_RESULT_DIR)


if __name__ == '__main__':
  unittest.main()
