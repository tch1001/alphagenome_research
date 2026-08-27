"""CPU tests for the one-shot exact-source Gate-0 diagnostic."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


_MODULE_PATH = Path(__file__).with_name('run_exact_source_gate0_v3_1_1.py')
_SPEC = importlib.util.spec_from_file_location(
    'run_exact_source_gate0_v3_1_1', _MODULE_PATH
)
runner = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)


def _sha(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _minimal_record(root: Path):
  five = (
      'experiments/interpretability/opensplice/run_route_census_v3.py',
      'experiments/interpretability/opensplice/target_reducers_v3.py',
      'src/alphagenome_research/model/interpretability.py',
      'src/alphagenome_research/model/model.py',
      'src/alphagenome_research/model/dna_model.py',
  )
  file_hashes = {}
  for index, relative in enumerate(five):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    data = f'locked-{index}\n'.encode()
    path.write_bytes(data)
    file_hashes[relative] = _sha(data)
  helpers = {}
  for index, relative in enumerate((
      'experiments/interpretability/opensplice/run_phase_r_v3.py',
      'experiments/interpretability/opensplice/run_inference_trace.py',
  )):
    path = root / relative
    data = f'helper-{index}\n'.encode()
    path.write_bytes(data)
    helpers[relative] = _sha(data)
  return {
      'configuration': {
          'code': {
              'git_head': runner.LOCK_COMMIT,
              'tracked_dirty_diff_sha256': hashlib.sha256(b'').hexdigest(),
              'file_sha256': file_hashes,
          },
          'phase_runner_sha256': helpers[
              'experiments/interpretability/opensplice/run_phase_r_v3.py'
          ],
          'v2_runner_sha256': helpers[
              'experiments/interpretability/opensplice/run_inference_trace.py'
          ],
      }
  }


@dataclasses.dataclass(frozen=True)
class _Endpoint:
  role: str
  position_1based: int
  position_index: int
  track_index: int


@dataclasses.dataclass(frozen=True)
class _PositionSet:
  name: str
  tokens: tuple[int, ...]


class ExactSourceGate0V311Test(unittest.TestCase):

  def test_protocol_and_historical_tree_are_hash_bound(self):
    self.assertEqual(runner._sha256(runner._PROTOCOL_PATH),  # pylint: disable=protected-access
                     runner.PROTOCOL_SHA256)
    records = runner.load_locked_identity_records()
    self.assertEqual(len(records), 20)
    self.assertEqual(
        [record['configuration']['case']['order'] for record in records],
        list(range(20)),
    )
    self.assertEqual(
        {record['configuration']['case']['gene'] for record in records},
        {'BRAF', 'SLC25A48'},
    )

  def test_exact_five_file_map_and_helpers_fail_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      record = _minimal_record(root)
      result = runner.validate_locked_sources(root, (record, record))
      self.assertEqual(len(result['locked_five_file_sha256']), 5)
      changed = root / 'src/alphagenome_research/model/model.py'
      changed.write_text('drift\n', encoding='utf-8')
      with self.assertRaisesRegex(ValueError, 'source hash mismatch'):
        runner.validate_locked_sources(root, (record, record))

  def test_worktree_requires_exact_detached_clean_commit(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      subprocess.run(('git', 'init', '-q', str(root)), check=True)
      (root / 'tracked.txt').write_text('frozen\n', encoding='utf-8')
      subprocess.run(('git', '-C', str(root), 'add', 'tracked.txt'), check=True)
      subprocess.run((
          'git', '-C', str(root), '-c', 'user.name=Test',
          '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'lock'
      ), check=True)
      head = subprocess.check_output(
          ('git', '-C', str(root), 'rev-parse', 'HEAD'), text=True
      ).strip()
      subprocess.run(('git', '-C', str(root), 'checkout', '-q', '--detach'),
                     check=True)
      result = runner.validate_locked_worktree(root, expected_commit=head)
      self.assertTrue(result['detached'])
      (root / 'tracked.txt').write_text('changed\n', encoding='utf-8')
      with self.assertRaisesRegex(ValueError, 'tracked changes'):
        runner.validate_locked_worktree(root, expected_commit=head)

  def test_locked_target_rule_is_inclusive_and_keeps_direction_gate(self):
    locked = (1.0, 1.5, 1.5, 1.5, 1.0, 1.0)
    boundary = list(locked)
    boundary[1] += runner.TARGET_TOLERANCE
    passed = runner.compare_locked_target(
        boundary, locked, is_effect=True, experimental_delta_logit=2.0
    )
    self.assertTrue(passed['passed'])
    beyond = list(boundary)
    beyond[1] += 1 / 1024
    failed = runner.compare_locked_target(
        beyond, locked, is_effect=True, experimental_delta_logit=2.0
    )
    self.assertFalse(failed['passed'])
    self.assertFalse(failed['all_six_within_tolerance'])
    wrong_sign = runner.compare_locked_target(
        (1.0, 0.5, 0.5, 0.5, 1.0, 1.0),
        (1.0, 0.5, 0.5, 0.5, 1.0, 1.0),
        is_effect=True,
        experimental_delta_logit=2.0,
    )
    self.assertFalse(wrong_sign['passed'])

  def test_live_linkage_covers_sequence_target_and_position_sets(self):
    case_record = {'variant_id': 'development_only'}
    case = SimpleNamespace(variant_id='development_only')
    interval = SimpleNamespace(chromosome='chr1', start=10, end=20)
    endpoint = _Endpoint('acceptor', 12, 1, 3)
    resolved = SimpleNamespace(endpoints=(endpoint,), padding_track_index=4)
    position = _PositionSet('S', (0,))
    modules = SimpleNamespace(
        v2=SimpleNamespace(
            _case_record=lambda _: case_record,
            trace_position_sets=lambda *_: (position,),
        )
    )
    record = {'configuration': {
        'case': case_record,
        'interval': {
            'chromosome': 'chr1', 'start_0based': 10,
            'end_0based_exclusive': 20,
        },
        'sequence_sha256': {'reference': 'r', 'alternate': 'a'},
        'canonical_target': {
            'endpoints': [dataclasses.asdict(endpoint)],
            'padding_track_index': 4,
        },
        'resolved_position_sets': [dataclasses.asdict(position)],
    }}
    result = runner.validate_live_case_linkage(
        modules, case, interval, resolved,
        {'reference': 'r', 'alternate': 'a'}, record
    )
    self.assertTrue(result['passed'])
    with self.assertRaisesRegex(ValueError, 'sequence_sha256'):
      runner.validate_live_case_linkage(
          modules, case, interval, resolved,
          {'reference': 'changed', 'alternate': 'a'}, record
      )

  def test_append_only_artifact_is_durable_before_failure_decision(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'raw' / 'case.json'
      failure = {
          'status': 'numerical_failure',
          'current_target_means': {'reference_baseline': 2.5},
      }
      expected_hash = runner._write_new(path, failure)  # pylint: disable=protected-access
      self.assertTrue(path.exists())
      self.assertEqual(runner._sha256(path), expected_hash)  # pylint: disable=protected-access
      self.assertEqual(json.loads(path.read_text()), failure)
      with self.assertRaises(FileExistsError):
        runner._write_new(path, {'status': 'pass'})  # pylint: disable=protected-access

  def test_existing_attempt_directory_refuses_resume_or_retry(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'one-shot'
      start = output / 'ATTEMPT_STARTED.json'
      with mock.patch.object(runner, 'OUTPUT_DIR', output), \
           mock.patch.object(runner, 'START_PATH', start):
        runner._ensure_fresh_attempt({'attempt_id': 'once'})  # pylint: disable=protected-access
        self.assertTrue(start.exists())
        with self.assertRaisesRegex(FileExistsError, 'cannot resume or retry'):
          runner._ensure_fresh_attempt({'attempt_id': 'twice'})  # pylint: disable=protected-access

  def test_dry_plan_is_exactly_20_identities_and_zero_active_calls(self):
    records = runner.load_locked_identity_records()
    plan = runner.build_dry_run_plan(
        Path('/locked'),
        {'head': runner.LOCK_COMMIT, 'tracked_clean': True},
        {'locked_five_file_sha256': {}},
        records,
    )
    self.assertEqual(plan['identity_units'], 20)
    self.assertEqual(plan['all_false_apply_calls'], 40)
    self.assertEqual(plan['active_intervention_calls'], 0)
    self.assertEqual(plan['confirmation_model_calls'], 0)
    self.assertFalse(plan['resume_after_attempt_start'])
    self.assertEqual(set(plan['development_genes']), {'BRAF', 'SLC25A48'})

  def test_autotune_is_honestly_unavailable_when_pre_run_flags_are_unset(self):
    with mock.patch.dict(os.environ, {'XLA_FLAGS': ''}, clear=False):
      result = runner.autotune_provenance()
    for value in result.values():
      self.assertEqual(value['status'], 'unavailable')
      self.assertIn('out-of-band', value['reason'])

  def test_launch_freeze_binds_runner_wrapper_protocol_and_proto_build(self):
    configured = json.loads(
        runner._FREEZE_CONFIG_PATH.read_text(encoding='utf-8')  # pylint: disable=protected-access
    )
    provenance = {
        'manifest_sha256': configured['proto_build_manifest_sha256'],
        'generated_outputs': {
            name: {'sha256': digest}
            for name, digest in configured['generated_output_sha256'].items()
        },
        'tool': {
            'grpcio_tools_version': configured['proto_generation'][
                'grpcio_tools_version'
            ],
            'protoc_path': configured['proto_generation']['protoc_path'],
            'protoc_version': configured['proto_generation'][
                'protoc_version'
            ],
        },
        'command_argv': configured['proto_generation']['command_argv'],
        'input_protos': configured['proto_generation']['input_protos'],
        'build_recipe': configured['proto_generation']['build_recipe'],
    }
    import_provenance = {
        'module_tree_sha256': configured[
            'initial_transitive_import_tree_sha256'
        ],
        'alphagenome_dependency_binding': configured[
            'alphagenome_dependency_binding'
        ],
    }
    frozen = runner.validate_freeze_configuration(
        provenance, import_provenance, configured['mixed_precision']
    )
    self.assertEqual(frozen['launcher_sha256'], runner._sha256(_MODULE_PATH))  # pylint: disable=protected-access
    self.assertEqual(frozen['protocol_sha256'], runner.PROTOCOL_SHA256)

  def test_proto_disclosure_does_not_claim_grpc_tools_or_hook_identity(self):
    disclosed = runner.disclose_proto_generation({
        'tool': {'grpcio_tools_version': None},
    })
    interpretation = disclosed['generation_disclosure']
    self.assertTrue(interpretation['standalone_protoc_used'])
    self.assertFalse(interpretation['grpcio_tools_used'])
    self.assertFalse(interpretation['hatch_build_hook_invoked'])
    self.assertFalse(
        interpretation['historical_generated_bytes_or_toolchain_reproduced']
    )

  def test_mixed_precision_source_is_exact_fd4_policy(self):
    locked_root = Path(
        '/home/degen2/alphafold-stuff/alphagenome_research_exact_fd4dc6913335'
    )
    provenance = runner.mixed_precision_provenance(locked_root)
    self.assertEqual(
        provenance['policy'],
        'params=float32,compute=bfloat16,output=bfloat16',
    )
    self.assertEqual(
        provenance['source_sha256'],
        '5d11021edd88f38ef55e6a385c2e20b1c39e4dd6c48139041477e5033b78b2bb',
    )

  def test_runtime_mixed_precision_dtypes_fail_closed(self):
    float32 = object()
    bfloat16 = object()
    jnp = SimpleNamespace(float32=float32, bfloat16=bfloat16)
    valid = SimpleNamespace(
        param_dtype=float32,
        compute_dtype=bfloat16,
        output_dtype=bfloat16,
    )
    runner._assert_mixed_precision_dtypes(valid, jnp)  # pylint: disable=protected-access
    invalid = SimpleNamespace(
        param_dtype=float32,
        compute_dtype=float32,
        output_dtype=bfloat16,
    )
    with self.assertRaisesRegex(ValueError, 'Runtime mixed-precision'):
      runner._assert_mixed_precision_dtypes(  # pylint: disable=protected-access
          invalid, jnp
      )


if __name__ == '__main__':
  unittest.main()
