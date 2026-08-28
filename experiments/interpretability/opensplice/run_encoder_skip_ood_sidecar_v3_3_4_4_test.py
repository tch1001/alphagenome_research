#!/usr/bin/env python3
"""CPU-only contract tests for the v3.3.4.4 OOD sidecar runner."""

from __future__ import annotations

import copy
from contextlib import contextmanager, ExitStack
import functools
import hashlib
import json
import errno
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import jax.numpy as jnp
import numpy as np


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))
import run_encoder_skip_ood_sidecar_v3_3_4_4 as runner  # pylint: disable=g-import-not-at-top


def _frozen_signatures() -> dict:
  return json.loads(
      runner.bootstrap.V3_3_3_FREEZE_PATH.read_text(encoding='utf-8')
  )['program_signatures']


def _runtime_signatures() -> dict:
  value = copy.deepcopy(_frozen_signatures())
  for record in value.values():
    record['leaves'] = tuple(
        {**leaf, 'shape': tuple(leaf['shape'])} for leaf in record['leaves']
    )
  return value


def _signature_attestation() -> dict:
  _, detail = runner.canonicalize_program_signatures(
      _runtime_signatures(), _frozen_signatures()
  )
  return {
      'external_freeze_authorization': {'freeze_sha256': 'f' * 64},
      **detail,
  }


def _source_audit() -> dict[str, bool]:
  return {name: True for name in runner.bootstrap.SOURCE_INPUT_AUDIT_KEYS}


def _same_object() -> dict:
  return {
      'lower_call_count': 1,
      'compile_call_count': 1,
      'stablehlo_read_from_lowered_object': True,
      'pre_backend_hlo_read_from_lowered_object': True,
      'compile_argument_is_lowered_object': True,
      'compiled_hlo_read_from_compiled_object': True,
      'signature_attestation_from_apply_arguments': True,
      'apply_callable_is_compiled_object': True,
      'compiler_record_is_gate_record': True,
      'lowered_python_id': 1,
      'compiled_python_id': 2,
  }


def _observed() -> dict:
  contract = runner.SOURCE_PROGRAM_CONTRACT
  return {
      'stablehlo_sha256': contract['stablehlo_sha256'],
      'stablehlo_size_bytes': contract['stablehlo_size_bytes'],
      'pre_backend_hlo_sha256': contract['pre_backend_hlo_sha256'],
      'pre_backend_hlo_size_bytes': contract['pre_backend_hlo_size_bytes'],
      'program_signatures_sha256': contract['program_signatures_sha256'],
      'entry_abi_sha256': contract['entry_abi_sha256'],
  }


def _authorization() -> dict:
  return {
      'git_head': 'a' * 40,
      'freeze_path': '/fixture/freeze.json',
      'freeze_sha256': 'f' * 64,
      'freeze_size_bytes': 1,
      'live_equals_git_show': True,
      'tracked_clean': True,
      'authorization_source': 'external_post_commit_audit',
  }


@functools.cache
def _prior_prefix_fixture() -> tuple[dict, dict]:
  prefix = runner.bootstrap.validate_prior_v3_3_4_3_consumed_preflight_prefix()
  return prefix, runner.bootstrap.canonical_content_binding(prefix)


def _prior_prefix_fields() -> dict:
  prefix, binding = _prior_prefix_fixture()
  return {
      'prior_v3_3_4_3_consumed_preflight_prefix': copy.deepcopy(prefix),
      'prior_v3_3_4_3_consumed_preflight_prefix_content_binding': (
          copy.deepcopy(binding)
      ),
  }


def _minimal_start() -> dict:
  return {
      'freeze_sha256': 'f' * 64,
      'git_head': 'a' * 40,
      'started_at_unix_s': 1.0,
      **_prior_prefix_fields(),
  }


def _terminal_phase(
    status: str, reason: str | None, compiler_state: str, *, started: int = 0,
) -> dict[str, bool]:
  phase = {name: False for name in runner._PHASE_STATE_KEYS}  # pylint: disable=protected-access
  for name in (
      'preflight_passed', 'start_persisted',
      'post_start_source_gate_passed',
  ):
    phase[name] = True
  early = {
      'pre_model_import_inventory_mismatch': (
          'pre_model_import_inventory_persisted',
      ),
      'protobuf_binding_mismatch': (
          'pre_model_import_inventory_persisted', 'protobuf_persisted',
      ),
      'post_model_import_inventory_mismatch': (
          'pre_model_import_inventory_persisted', 'protobuf_persisted',
          'model_construction_attempted', 'model_constructed',
          'reference_cases_loaded', 'post_model_import_inventory_persisted',
      ),
      'model_setup_failure': (
          'pre_model_import_inventory_persisted', 'protobuf_persisted',
          'model_construction_attempted',
          'post_model_import_inventory_persisted',
          'terminal_import_inventory_persisted',
      ),
  }
  for name in early.get(reason, ()):
    phase[name] = True
  requirements = {
      'signature_failure': (
          'pre_model_import_inventory_persisted', 'protobuf_persisted',
          'model_construction_attempted', 'model_constructed',
          'reference_cases_loaded', 'post_model_import_inventory_persisted',
          'terminal_import_inventory_persisted',
      ),
      'lowered': (
          'pre_model_import_inventory_persisted', 'protobuf_persisted',
          'model_construction_attempted', 'model_constructed',
          'reference_cases_loaded', 'signatures_captured',
          'signature_attestation_persisted',
          'post_model_import_inventory_persisted', 'lower_attempted',
          'terminal_import_inventory_persisted',
      ),
      'precompiled': (
          'pre_model_import_inventory_persisted', 'protobuf_persisted',
          'model_construction_attempted', 'model_constructed',
          'reference_cases_loaded', 'signatures_captured',
          'signature_attestation_persisted',
          'post_model_import_inventory_persisted', 'lower_attempted',
          'lower_succeeded', 'compile_attempted',
          'terminal_import_inventory_persisted',
      ),
      'compiled': (
          'pre_model_import_inventory_persisted', 'protobuf_persisted',
          'model_construction_attempted', 'model_constructed',
          'reference_cases_loaded', 'signatures_captured',
          'signature_attestation_persisted',
          'post_model_import_inventory_persisted', 'lower_attempted',
          'lower_succeeded', 'compile_attempted', 'compile_succeeded',
          'terminal_import_inventory_persisted',
      ),
      'compiled_guarded': (
          'pre_model_import_inventory_persisted', 'protobuf_persisted',
          'model_construction_attempted', 'model_constructed',
          'reference_cases_loaded', 'signatures_captured',
          'signature_attestation_persisted',
          'post_model_import_inventory_persisted', 'lower_attempted',
          'lower_succeeded', 'compile_attempted', 'compile_succeeded',
          'terminal_import_inventory_persisted',
      ),
      'diagnostic_failure': (
          'pre_model_import_inventory_persisted', 'protobuf_persisted',
          'model_construction_attempted', 'model_constructed',
          'reference_cases_loaded', 'signatures_captured',
          'signature_attestation_persisted',
          'post_model_import_inventory_persisted', 'lower_attempted',
          'lower_succeeded', 'compile_attempted', 'compile_succeeded',
          'terminal_import_inventory_persisted',
      ),
  }
  for name in requirements.get(compiler_state, ()):
    phase[name] = True
  if compiler_state == 'compiled':
    phase['diagnostic_provenance_passed'] = True
    phase['source_program_gate_passed'] = (
        status != 'controlled_stop_source_program_mismatch'
    )
  phase['dispatch_begun'] = started > 0
  return phase


class _Case:
  order = 0
  variant_id = 'chr1:1:A:G'


class _FakeHlo:

  def as_hlo_text(self):
    return 'pre-backend-hlo'


class _FakeLowered:

  def __init__(self, failure_stage: str | None = None):
    self.failure_stage = failure_stage

  def compiler_ir(self, *, dialect: str):
    if self.failure_stage == 'stablehlo_text_extraction' and dialect == 'stablehlo':
      raise RuntimeError('injected stablehlo extraction failure')
    if self.failure_stage == 'pre_backend_hlo_text_extraction' and dialect == 'hlo':
      raise RuntimeError('injected pre-backend extraction failure')
    if dialect == 'stablehlo':
      return 'stablehlo'
    if dialect == 'hlo':
      return _FakeHlo()
    raise ValueError(f'Unexpected compiler IR dialect: {dialect}.')


class _FakeCompiled:

  def __init__(self, failure_stage: str | None = None):
    self.failure_stage = failure_stage

  def as_text(self):
    if self.failure_stage == 'compiled_hlo_text_extraction':
      raise RuntimeError('injected compiled HLO extraction failure')
    return 'HloModule main, entry_computation_layout={(f32[])->f32[]}'


class EncoderSkipOodSidecarV334Test(unittest.TestCase):

  @contextmanager
  def _publication_root(self, directory: str):
    root = Path(directory)
    old_root = runner.bootstrap.PUBLICATION_ROOTS['model_run']
    runner.bootstrap.PUBLICATION_ROOTS['model_run'] = root
    runner.bootstrap._PUBLICATION_DIRECTORIES.clear()  # pylint: disable=protected-access
    runner.bootstrap._PUBLICATION_SUCCESS.clear()  # pylint: disable=protected-access
    runner.bootstrap._PUBLICATION_TEMP_ORPHANS.clear()  # pylint: disable=protected-access
    runner.bootstrap._PUBLICATION_UNCERTAIN_FINALS.clear()  # pylint: disable=protected-access
    runner.bootstrap._PUBLICATION_PREEXISTING.clear()  # pylint: disable=protected-access
    runner.bootstrap._PUBLICATION_UNBINDABLE_FAILURES.clear()  # pylint: disable=protected-access
    runner.bootstrap._PUBLICATION_ORDINAL = 0  # pylint: disable=protected-access
    try:
      runner.bootstrap.allocate_publication_directory(
          'model_run', register_existing=True
      )
      with mock.patch.object(runner, 'OUTPUT_DIR', root):
        yield root
    finally:
      runner.bootstrap.PUBLICATION_ROOTS['model_run'] = old_root
      runner.bootstrap._PUBLICATION_DIRECTORIES.clear()  # pylint: disable=protected-access

  def test_full_order_and_apply_indices_are_exact(self):
    cases = [types.SimpleNamespace(order=index) for index in range(20)]
    order = runner.sidecar_execution_order(cases)
    self.assertEqual(len(order), 80)
    self.assertEqual(order[:4], ((0, 0), (0, 127), (0, 128), (0, 255)))
    self.assertEqual(order[-1], (19, 255))
    self.assertEqual(
        [4 * index + call for index in range(80) for call in range(4)],
        list(range(320)),
    )

  def test_signature_adapter_is_exact_at_all_32_paths(self):
    adapted, audit = runner.canonicalize_program_signatures(
        _runtime_signatures(), _frozen_signatures()
    )
    self.assertEqual(adapted, _frozen_signatures())
    self.assertEqual(len(audit['runtime_container_tags']), 32)
    self.assertEqual(audit['runtime_canonical'], {
        'sha256': runner.SOURCE_PROGRAM_CONTRACT['program_signatures_sha256'],
        'size_bytes': 2877,
    })
    self.assertFalse(audit['comparisons']['direct_python_equality'])

  def test_signature_adapter_rejects_each_declared_kind_drift(self):
    for row in _signature_attestation()['runtime_container_tags']:
      with self.subTest(path=row['path']):
        runtime = _runtime_signatures()
        tokens = row['path'].strip('/').split('/')
        object_name = tokens[0]
        if len(tokens) == 2:
          runtime[object_name]['leaves'] = list(
              runtime[object_name]['leaves']
          )
        else:
          runtime[object_name]['leaves'][int(tokens[2])]['shape'] = list(
              runtime[object_name]['leaves'][int(tokens[2])]['shape']
          )
        with self.assertRaises(TypeError):
          runner.canonicalize_program_signatures(runtime, _frozen_signatures())

  def test_signature_adapter_rejects_tuple_elsewhere(self):
    runtime = _runtime_signatures()
    runtime['target']['treedef'] = ('unexpected',)
    with self.assertRaisesRegex(ValueError, 'outside 32 paths'):
      runner.canonicalize_program_signatures(runtime, _frozen_signatures())

  def test_every_source_program_primitive_fails_independently(self):
    compiler = {'program_signatures': _frozen_signatures()}
    passing = runner.evaluate_source_program_gate(
        _observed(), _frozen_signatures(), compiler, _source_audit(),
        _signature_attestation(), _same_object(),
    )
    self.assertTrue(passing['source_program_exact'])
    observed_fields = (
        'stablehlo_sha256', 'stablehlo_size_bytes',
        'pre_backend_hlo_sha256', 'pre_backend_hlo_size_bytes',
        'program_signatures_sha256', 'entry_abi_sha256',
    )
    for name in observed_fields:
      with self.subTest(observed=name):
        observed = _observed()
        observed[name] = 0 if name.endswith('bytes') else '0' * 64
        result = runner.evaluate_source_program_gate(
            observed, _frozen_signatures(), compiler, _source_audit(),
            _signature_attestation(), _same_object(),
        )
        self.assertFalse(result['source_program_exact'])
    for name in runner.bootstrap.SOURCE_INPUT_AUDIT_KEYS:
      with self.subTest(source=name):
        source = _source_audit()
        source[name] = False
        result = runner.evaluate_source_program_gate(
            _observed(), _frozen_signatures(), compiler, source,
            _signature_attestation(), _same_object(),
        )
        self.assertFalse(result['source_program_exact'])
    for name in (
        'stablehlo_read_from_lowered_object',
        'pre_backend_hlo_read_from_lowered_object',
        'compile_argument_is_lowered_object',
        'compiled_hlo_read_from_compiled_object',
        'signature_attestation_from_apply_arguments',
        'apply_callable_is_compiled_object',
        'compiler_record_is_gate_record',
    ):
      with self.subTest(same_object=name):
        same = _same_object()
        same[name] = False
        result = runner.evaluate_source_program_gate(
            _observed(), _frozen_signatures(), compiler, _source_audit(),
            _signature_attestation(), same,
        )
        self.assertFalse(result['source_program_exact'])

  def test_compiled_backend_is_not_a_source_program_term(self):
    gate = runner.evaluate_source_program_gate(
        _observed(), _frozen_signatures(),
        {'program_signatures': _frozen_signatures()}, _source_audit(),
        _signature_attestation(), _same_object(),
    )
    self.assertNotIn('compiled_hlo_sha256', gate['contract'])
    self.assertNotIn('compiled_hlo_sha256', gate['observed'])
    self.assertTrue(gate['source_program_exact'])

  def test_graph_texts_are_all_captured_before_first_publication(self):
    events = []

    class Hlo:
      def as_hlo_text(self):
        events.append('capture_pre_backend')
        return 'pre-backend'

    class Lowered:
      def compiler_ir(self, *, dialect):
        if dialect == 'stablehlo':
          events.append('capture_stablehlo')
          return 'stablehlo'
        events.append('request_pre_backend')
        return Hlo()

    class Compiled:
      def as_text(self):
        events.append('capture_compiled')
        return 'compiled'

    texts = runner._extract_compiler_graph_texts(  # pylint: disable=protected-access
        Lowered(), Compiled()
    )
    self.assertEqual(events, [
        'capture_stablehlo', 'request_pre_backend',
        'capture_pre_backend', 'capture_compiled',
    ])
    with tempfile.TemporaryDirectory() as directory:
      with self._publication_root(directory):
        runner._publish_compiler_graphs(texts)  # pylint: disable=protected-access
    self.assertEqual(events, [
        'capture_stablehlo', 'request_pre_backend',
        'capture_pre_backend', 'capture_compiled',
    ])

  def test_each_graph_extraction_failure_has_zero_publication(self):
    for failed_stage in (
        'stablehlo_text_extraction',
        'pre_backend_hlo_text_extraction',
        'compiled_hlo_text_extraction',
    ):
      with self.subTest(stage=failed_stage):
        class Hlo:
          def as_hlo_text(self):
            if failed_stage == 'pre_backend_hlo_text_extraction':
              raise RuntimeError('pre-backend failure')
            return 'pre-backend'

        class Lowered:
          def compiler_ir(self, *, dialect):
            if (
                dialect == 'stablehlo'
                and failed_stage == 'stablehlo_text_extraction'
            ):
              raise RuntimeError('stablehlo failure')
            return 'stablehlo' if dialect == 'stablehlo' else Hlo()

        class Compiled:
          def as_text(self):
            if failed_stage == 'compiled_hlo_text_extraction':
              raise RuntimeError('compiled failure')
            return 'compiled'

        with mock.patch.object(
            runner, '_publish_compiler_graphs'
        ) as publish:
          with self.assertRaises(runner.CompilerGraphExtractionError) as caught:
            runner._extract_compiler_graph_texts(  # pylint: disable=protected-access
                Lowered(), Compiled()
            )
        self.assertEqual(caught.exception.stage, failed_stage)
        publish.assert_not_called()
        reads = tuple(
            caught.exception.same_object_attestation[name]
            for name in (
                'stablehlo_read_from_lowered_object',
                'pre_backend_hlo_read_from_lowered_object',
                'compiled_hlo_read_from_compiled_object',
            )
        )
        self.assertEqual(reads, {
            'stablehlo_text_extraction': (None, None, None),
            'pre_backend_hlo_text_extraction': (True, None, None),
            'compiled_hlo_text_extraction': (True, True, None),
        }[failed_stage])

  def test_extraction_nonpublication_terminal_has_exact_six_file_archive(self):
    source = _source_audit()
    source['three_import_inventories_stable_exact'] = None
    budget = {
        'lower_budget': 1, 'compile_budget': 1,
        'lower_invocations': 1, 'compile_invocations': 1,
        'forbidden_request': None,
        'forbidden_request_detected_before_invocation': False,
    }
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / 'run'
      cache = Path(directory) / 'cache'
      root.mkdir(mode=0o700)
      cache.mkdir(mode=0o700)
      (cache / 'triton').mkdir(mode=0o700)
      (cache / 'xdg').mkdir(mode=0o700)
      with self._publication_root(str(root)), mock.patch.object(
          runner.bootstrap, 'MODEL_KERNEL_CACHE_DIR', cache
      ):
        for relative in (
            'ATTEMPT_STARTED.json', 'IMPORT_PROVENANCE_PRE_MODEL.json',
            'PROTOBUF_PROVENANCE.json',
            'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
            'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json',
        ):
          runner._write_new(root / relative, {'path': relative})  # pylint: disable=protected-access
        signature = runner._relative_file_binding(  # pylint: disable=protected-access
            root / 'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json'
        )
        start = {
            'freeze_sha256': 'f' * 64, 'git_head': 'a' * 40,
            'started_at_unix_s': 1.0,
            **_prior_prefix_fields(),
            'same_process_preflight': {
                'model_cache_pre_import': {
                    **runner.bootstrap.cache_output_tree_binding(cache),
                    'cache_role': 'model', 'cache_root': str(cache.resolve()),
                    'triton_cache_dir': str((cache / 'triton').resolve()),
                    'xdg_cache_home': str((cache / 'xdg').resolve()),
                    'cuda_kernel_cache_disabled': True,
                    'jax_compilation_cache_disabled': True,
                    'present_forbidden_names': [],
                },
            },
        }
        launcher = {
            'start': start, 'v3_3_3_run': {'prior': 'run'},
            'v3_3_3_1_archive': {'prior': 'archive'},
        }
        same = {
            **_same_object(),
            'stablehlo_read_from_lowered_object': None,
            'pre_backend_hlo_read_from_lowered_object': None,
            'compiled_hlo_read_from_compiled_object': None,
            'compiler_record_is_gate_record': None,
        }
        historical = runner.bootstrap.cache_output_tree_binding(cache)
        evidence = {
            'pre_import_files_present': False,
            'default_user_cache_path_eligible': False,
            'persistent_compilation_cache_hit_reported': False,
            'executable_deserialized': False, 'compile_skipped': False,
            'compile_stage_not_applicable': False,
            'old_cache_input_opened': False, 'routing_exact': True,
            'cache_hit': False,
        }
        with (
            mock.patch.object(runner, '_launcher_record', return_value=launcher),
            mock.patch.object(
                runner, '_external_authorization', return_value=_authorization()
            ),
        ):
          terminal = runner._write_nonpublication_terminal(  # pylint: disable=protected-access
              failure_stage='stablehlo_text_extraction',
              failure=RuntimeError('graph extraction failed'),
              triggering_diagnostic_failure=None,
              triggering_diagnostic_stop_reason=None,
              source_input_audit=source,
              same_object_attestation=same,
              program_signature_attestation_binding=signature,
              attempt_budget_audit=budget,
              historical_cache_binding=historical,
              cache_hit_evidence=evidence, published_graphs=None,
              source_program_gate_without_backend_diagnostics=None,
          )
      self.assertEqual(len(terminal), 60)
      self.assertEqual(terminal['preterminal_tree_binding']['file_count'], 5)
      self.assertEqual(
          {path.relative_to(root).as_posix() for path in root.rglob('*')
           if path.is_file()},
          set(runner.bootstrap.NONPUBLICATION_TERMINAL_CONTRACT_V3_3_4_3[
              'extraction_preterminal_membership'
          ]) | {'NONPUBLICATION_TERMINAL_FAILURE.json'},
      )
      self.assertEqual(terminal['graph_artifact_bindings'], {})
      self.assertIsNone(terminal['import_provenance_phases']['terminal'])

  def test_diagnostic_nonpublication_router_covers_both_stages_and_gate_bits(self):
    common = {
        'lowered': object(), 'compiled': object(),
        'published_graphs': {
            'artifacts': {
                name: {'path': f'compiler/eight_row/{name}.txt',
                       'sha256': char * 64, 'size_bytes': 1}
                for name, char in (
                    ('stablehlo', 'a'), ('hlo', 'b'),
                    ('compiled_hlo', 'c')
                )
            },
            'compiled_hlo_text': 'compiled',
        },
        'program_signatures': _frozen_signatures(),
        'program_signature_attestation': _signature_attestation(),
        'program_signature_attestation_binding': {
            'path': 'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json',
            'sha256': 'd' * 64, 'size_bytes': 1,
        },
        'source_input_audit': _source_audit(), 'entry_abi': None,
        'attempt_budget_audit': {
            'lower_budget': 1, 'compile_budget': 1,
            'lower_invocations': 1, 'compile_invocations': 1,
            'forbidden_request': None,
            'forbidden_request_detected_before_invocation': False,
        },
        'historical_cache_binding': {'tree_sha256': 'e' * 64},
        'cache_hit_evidence': {'cache_hit': False},
    }
    trigger = runner.BackendDiagnosticParserFailure(
        ValueError('diagnostic parser failed')
    )
    with (
        mock.patch.object(
            runner, '_derive_diagnostic_failure_source_gate',
            side_effect=RuntimeError('source gate failed'),
        ),
        mock.patch.object(runner, '_write_nonpublication_terminal') as write,
        mock.patch.object(runner, '_compiler_diagnostic_failure_artifact') as artifact,
    ):
      self.assertIsNone(runner._route_diagnostic_failure(  # pylint: disable=protected-access
          trigger, 'diagnostic_parser_failure', **common
      ))
    artifact.assert_not_called()
    self.assertEqual(
        write.call_args.kwargs['failure_stage'],
        'source_program_gate_derivation_for_diagnostic_failure',
    )
    self.assertIsNone(
        write.call_args.kwargs[
            'source_program_gate_without_backend_diagnostics'
        ]
    )
    for source_exact in (False, True):
      with self.subTest(source_program_exact=source_exact):
        gate = {'source_program_exact': source_exact}
        with (
            mock.patch.object(
                runner, '_derive_diagnostic_failure_source_gate',
                return_value=gate,
            ),
            mock.patch.object(
                runner, '_compiler_diagnostic_failure_artifact',
                side_effect=RuntimeError('record construction failed'),
            ),
            mock.patch.object(
                runner, '_write_nonpublication_terminal'
            ) as write,
        ):
          self.assertIsNone(runner._route_diagnostic_failure(  # pylint: disable=protected-access
              trigger, 'diagnostic_parser_failure', **common
          ))
        self.assertEqual(
            write.call_args.kwargs['failure_stage'],
            'diagnostic_failure_record_construction',
        )
        self.assertEqual(
            write.call_args.kwargs[
                'source_program_gate_without_backend_diagnostics'
            ], gate,
        )
        self.assertIs(
            runner._nonpublication_phase_state(  # pylint: disable=protected-access
                diagnostic_stage=True,
                source_program_gate_passed=source_exact,
            )['source_program_gate_passed'], source_exact,
        )

  def test_diagnostic_nonpublication_terminal_has_exact_ten_file_archives(self):
    stage_cases = (
        ('source_program_gate_derivation_for_diagnostic_failure', None),
        ('diagnostic_failure_record_construction', False),
        ('diagnostic_failure_record_construction', True),
    )
    for stage, gate_exact in stage_cases:
      with self.subTest(stage=stage, source_program_exact=gate_exact):
        with tempfile.TemporaryDirectory() as directory:
          root = Path(directory) / 'run'
          cache = Path(directory) / 'cache'
          root.mkdir(mode=0o700)
          cache.mkdir(mode=0o700)
          (cache / 'triton').mkdir(mode=0o700)
          (cache / 'xdg').mkdir(mode=0o700)
          with self._publication_root(str(root)), mock.patch.object(
              runner.bootstrap, 'MODEL_KERNEL_CACHE_DIR', cache
          ):
            for relative in (
                'ATTEMPT_STARTED.json', 'IMPORT_PROVENANCE_PRE_MODEL.json',
                'PROTOBUF_PROVENANCE.json',
                'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
                'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json',
                'IMPORT_PROVENANCE.json',
            ):
              runner._write_new(root / relative, {'path': relative})  # pylint: disable=protected-access
            prior_graph_root = (
                runner.bootstrap.V3_3_3_RUN_DIR / 'compiler/eight_row'
            )
            graph_texts = {
                'stablehlo': 'stable', 'hlo': 'pre-backend',
                'compiled_hlo': 'compiled',
            }
            if gate_exact is True:
              graph_texts['stablehlo'] = (
                  prior_graph_root / 'graph.stablehlo.mlir'
              ).read_text(encoding='utf-8')
              graph_texts['hlo'] = (
                  prior_graph_root / 'graph.pre_backend.hlo.txt'
              ).read_text(encoding='utf-8')
            published = runner._publish_compiler_graphs(  # pylint: disable=protected-access
                graph_texts
            )
            signature = runner._relative_file_binding(  # pylint: disable=protected-access
                root / 'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json'
            )
            cache_binding = runner.bootstrap.cache_output_tree_binding(cache)
            start = {
                'freeze_sha256': 'f' * 64, 'git_head': 'a' * 40,
                'started_at_unix_s': 1.0,
                **_prior_prefix_fields(),
                'same_process_preflight': {
                    'model_cache_pre_import': cache_binding,
                },
            }
            launcher = {
                'start': start, 'v3_3_3_run': {'prior': 'run'},
                'v3_3_3_1_archive': {'prior': 'archive'},
            }
            gate = None
            if gate_exact is not None:
              observed = {
                  'stablehlo_sha256': published['artifacts'][
                      'stablehlo'
                  ]['sha256'],
                  'stablehlo_size_bytes': published['artifacts'][
                      'stablehlo'
                  ]['size_bytes'],
                  'pre_backend_hlo_sha256': published['artifacts'][
                      'hlo'
                  ]['sha256'],
                  'pre_backend_hlo_size_bytes': published['artifacts'][
                      'hlo'
                  ]['size_bytes'],
                  'program_signatures_sha256': runner.SOURCE_PROGRAM_CONTRACT[
                      'program_signatures_sha256'
                  ],
                  'entry_abi_sha256': runner.SOURCE_PROGRAM_CONTRACT[
                      'entry_abi_sha256'
                  ],
              }
              gate = runner.evaluate_source_program_gate(
                  observed, _frozen_signatures(),
                  {'program_signatures': _frozen_signatures()},
                  _source_audit(), _signature_attestation(), _same_object(),
              )
              self.assertIs(gate['source_program_exact'], gate_exact)
            evidence = {
                'pre_import_files_present': False,
                'default_user_cache_path_eligible': False,
                'persistent_compilation_cache_hit_reported': False,
                'executable_deserialized': False,
                'compile_skipped': False,
                'compile_stage_not_applicable': False,
                'old_cache_input_opened': False, 'routing_exact': True,
                'cache_hit': False,
            }
            with (
                mock.patch.object(
                    runner, '_launcher_record', return_value=launcher
                ),
                mock.patch.object(
                    runner, '_external_authorization',
                    return_value=_authorization(),
                ),
            ):
              terminal = runner._write_nonpublication_terminal(  # pylint: disable=protected-access
                  failure_stage=stage,
                  failure=RuntimeError('construction boundary failed'),
                  triggering_diagnostic_failure=(
                      runner.BackendDiagnosticParserFailure(
                          ValueError('diagnostic parser failed')
                      )
                  ),
                  triggering_diagnostic_stop_reason=(
                      'diagnostic_parser_failure'
                  ),
                  source_input_audit=_source_audit(),
                  same_object_attestation=_same_object(),
                  program_signature_attestation_binding=signature,
                  attempt_budget_audit={
                      'lower_budget': 1, 'compile_budget': 1,
                      'lower_invocations': 1, 'compile_invocations': 1,
                      'forbidden_request': None,
                      'forbidden_request_detected_before_invocation': False,
                  },
                  historical_cache_binding=cache_binding,
                  cache_hit_evidence=evidence,
                  published_graphs=published,
                  source_program_gate_without_backend_diagnostics=gate,
              )
          self.assertEqual(len(terminal), 60)
          self.assertEqual(
              terminal['preterminal_tree_binding']['file_count'], 9
          )
          self.assertEqual(len(terminal['graph_artifact_bindings']), 3)
          self.assertEqual(
              len([path for path in root.rglob('*') if path.is_file()]), 10
          )
          self.assertIs(
              terminal['phase_state']['source_program_gate_passed'],
              False if gate is None else gate_exact,
          )

  def test_diagnostic_router_never_relabels_publication_error(self):
    publication = runner.bootstrap.PublicationError({
        'message': 'publication failed',
    })
    with (
        mock.patch.object(
            runner, '_derive_diagnostic_failure_source_gate',
            side_effect=publication,
        ),
        mock.patch.object(runner, '_write_nonpublication_terminal') as write,
    ):
      with self.assertRaises(runner.bootstrap.PublicationError):
        runner._route_diagnostic_failure(  # pylint: disable=protected-access
            runner.EntryAbiParserFailure(ValueError('parser failed')),
            'diagnostic_parser_failure',
            lowered=object(), compiled=object(), published_graphs={},
            program_signatures={}, program_signature_attestation={},
            program_signature_attestation_binding={},
            source_input_audit=_source_audit(), entry_abi=None,
            attempt_budget_audit={}, historical_cache_binding={},
            cache_hit_evidence={'cache_hit': False},
        )
    write.assert_not_called()

  def test_postcompile_orchestration_reaches_all_five_exact_stages(self):
    stages = tuple(
        runner.bootstrap.NONPUBLICATION_TERMINAL_CONTRACT_V3_3_4_3[
            'failure_stages'
        ]
    )
    for stage in stages:
      with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / 'run'
        cache = Path(directory) / 'cache'
        root.mkdir(mode=0o700)
        cache.mkdir(mode=0o700)
        (cache / 'triton').mkdir(mode=0o700)
        (cache / 'xdg').mkdir(mode=0o700)
        with self._publication_root(str(root)), mock.patch.object(
            runner.bootstrap, 'MODEL_KERNEL_CACHE_DIR', cache
        ):
          for relative in (
              'ATTEMPT_STARTED.json', 'IMPORT_PROVENANCE_PRE_MODEL.json',
              'PROTOBUF_PROVENANCE.json',
              'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
              'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json',
          ):
            runner._write_new(root / relative, {'path': relative})  # pylint: disable=protected-access
          signature_binding = runner._relative_file_binding(  # pylint: disable=protected-access
              root / 'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json'
          )
          pre_import = runner.bootstrap.cache_output_tree_binding(cache)
          start = {
              'freeze_sha256': 'f' * 64, 'git_head': 'a' * 40,
              'started_at_unix_s': 1.0,
              **_prior_prefix_fields(),
              'same_process_preflight': {
                  'model_cache_pre_import': pre_import,
              },
          }
          launch = {
              'start': start,
              'v3_3_3_run': {'prior': 'run'},
              'v3_3_3_1_archive': {'prior': 'archive'},
              'gate_a': {
                  'v3_3_2_run': {
                      'eight_row_compiler': {
                          'program_signatures': _frozen_signatures(),
                      },
                  },
              },
          }
          budget = runner.OneShotCompilerBudget()
          budget.request('lower')
          budget.request('compile')
          source_prefix = _source_audit()
          source_prefix[runner.bootstrap.SOURCE_INPUT_AUDIT_KEYS[-1]] = None
          evidence = {
              'pre_import_files_present': False,
              'default_user_cache_path_eligible': False,
              'persistent_compilation_cache_hit_reported': False,
              'executable_deserialized': False,
              'compile_skipped': False,
              'compile_stage_not_applicable': False,
              'old_cache_input_opened': False,
              'routing_exact': True, 'cache_hit': False,
          }

          def persist_terminal(*unused_args, **unused_kwargs):
            runner._write_new(  # pylint: disable=protected-access
                root / 'IMPORT_PROVENANCE.json', {'phase': 'terminal'}
            )
            return {'phase': 'terminal'}, {'all_exact': True}

          def gate_for_entry_failure(**kwargs):
            artifacts = kwargs['published_graphs']['artifacts']
            observed = {
                'stablehlo_sha256': artifacts['stablehlo']['sha256'],
                'stablehlo_size_bytes': artifacts['stablehlo']['size_bytes'],
                'pre_backend_hlo_sha256': artifacts['hlo']['sha256'],
                'pre_backend_hlo_size_bytes': artifacts['hlo']['size_bytes'],
                'program_signatures_sha256': (
                    runner.SOURCE_PROGRAM_CONTRACT[
                        'program_signatures_sha256'
                    ]
                ),
                'entry_abi_sha256': '',
            }
            return runner.evaluate_source_program_gate(
                observed, kwargs['program_signatures'],
                launch['gate_a']['v3_3_2_run']['eight_row_compiler'],
                kwargs['source_input_audit'],
                kwargs['program_signature_attestation'],
                runner._successful_same_object_attestation(  # pylint: disable=protected-access
                    kwargs['lowered'], kwargs['compiled']
                ),
            )

          lowered = _FakeLowered(
              stage if stage.endswith('_text_extraction') else None
          )
          compiled = _FakeCompiled(
              stage if stage == 'compiled_hlo_text_extraction' else None
          )
          with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                runner, '_launcher_record', return_value=launch
            ))
            stack.enter_context(mock.patch.object(
                runner, '_external_authorization', return_value=_authorization()
            ))
            stack.enter_context(mock.patch.object(
                runner, '_model_cache_hit_evidence', return_value=evidence
            ))
            stack.enter_context(mock.patch.object(
                runner, '_persist_terminal_import_inventory',
                side_effect=persist_terminal,
            ))
            stack.enter_context(mock.patch.object(
                runner, 'derive_source_input_audit',
                return_value=_source_audit(),
            ))
            if stage in stages[3:]:
              stack.enter_context(mock.patch.object(
                  runner, '_entry_abi_binding',
                  side_effect=runner.EntryAbiParserFailure(
                      RuntimeError('injected entry ABI parser failure')
                  ),
              ))
            if stage == stages[3]:
              stack.enter_context(mock.patch.object(
                  runner, '_derive_diagnostic_failure_source_gate',
                  side_effect=RuntimeError('injected gate derivation failure'),
              ))
            elif stage == stages[4]:
              stack.enter_context(mock.patch.object(
                  runner, '_derive_diagnostic_failure_source_gate',
                  side_effect=gate_for_entry_failure,
              ))
              stack.enter_context(mock.patch.object(
                  runner, '_compiler_diagnostic_failure_artifact',
                  side_effect=RuntimeError(
                      'injected diagnostic record construction failure'
                  ),
              ))
            result = runner._orchestrate_postcompile_provenance(  # pylint: disable=protected-access
                lowered=lowered, compiled=compiled, compile_start=0.0,
                compiler_budget=budget, launch=launch, start=start,
                original_frozen={}, frozen={}, imports_pre={},
                imports_post={}, checkpoint_binding={},
                reference_object_binding={}, protobuf_record={},
                adapted_signatures=_frozen_signatures(),
                signature_attestation=_signature_attestation(),
                signature_binding=signature_binding,
                source_input_audit_prefix=source_prefix,
                current_phase={}, results=(),
            )
          self.assertIsNone(result)
          terminal_path = root / 'NONPUBLICATION_TERMINAL_FAILURE.json'
          terminal = json.loads(terminal_path.read_text(encoding='utf-8'))
          self.assertEqual(terminal['failure_stage'], stage)
          expected_files = 6 if stage in stages[:3] else 10
          self.assertEqual(
              len([path for path in root.rglob('*') if path.is_file()]),
              expected_files,
          )
          self.assertEqual(
              terminal['preterminal_tree_binding']['file_count'],
              expected_files - 1,
          )

  def test_postcompile_publication_failures_fall_back_to_terminal_archive(self):
    targets = (
        'graph.stablehlo.mlir', 'graph.pre_backend.hlo.txt',
        'graph.compiled.hlo.txt', 'NONPUBLICATION_TERMINAL_FAILURE.json',
    )
    for target in targets:
      with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / 'run'
        cache = Path(directory) / 'cache'
        root.mkdir(mode=0o700)
        cache.mkdir(mode=0o700)
        (cache / 'triton').mkdir(mode=0o700)
        (cache / 'xdg').mkdir(mode=0o700)
        with self._publication_root(str(root)), mock.patch.object(
            runner.bootstrap, 'MODEL_KERNEL_CACHE_DIR', cache
        ):
          for relative in (
              'ATTEMPT_STARTED.json', 'IMPORT_PROVENANCE_PRE_MODEL.json',
              'PROTOBUF_PROVENANCE.json',
              'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
              'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json',
          ):
            runner._write_new(root / relative, {'path': relative})  # pylint: disable=protected-access
          signature_binding = runner._relative_file_binding(  # pylint: disable=protected-access
              root / 'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json'
          )
          pre_import = runner.bootstrap.cache_output_tree_binding(cache)
          start = {
              'freeze_sha256': 'f' * 64, 'git_head': 'a' * 40,
              'started_at_unix_s': 1.0,
              **_prior_prefix_fields(),
              'same_process_preflight': {
                  'model_cache_pre_import': pre_import,
              },
          }
          launch = {
              'start': start, 'v3_3_3_run': {'prior': 'run'},
              'v3_3_3_1_archive': {'prior': 'archive'},
              'gate_a': {'v3_3_2_run': {'eight_row_compiler': {
                  'program_signatures': _frozen_signatures(),
              }}},
          }
          budget = runner.OneShotCompilerBudget()
          budget.request('lower')
          budget.request('compile')
          source_prefix = _source_audit()
          source_prefix[runner.bootstrap.SOURCE_INPUT_AUDIT_KEYS[-1]] = None
          evidence = {
              'pre_import_files_present': False,
              'default_user_cache_path_eligible': False,
              'persistent_compilation_cache_hit_reported': False,
              'executable_deserialized': False,
              'compile_skipped': False,
              'compile_stage_not_applicable': False,
              'old_cache_input_opened': False,
              'routing_exact': True, 'cache_hit': False,
          }
          real_rename = runner.bootstrap._rename_noreplace  # pylint: disable=protected-access

          def fail_target(parent_fd, temporary, final):
            if final == target:
              return -1, errno.EIO
            return real_rename(parent_fd, temporary, final)

          lowered = _FakeLowered(
              'stablehlo_text_extraction'
              if target == 'NONPUBLICATION_TERMINAL_FAILURE.json' else None
          )
          compiled = _FakeCompiled()
          with (
              mock.patch.object(runner, '_launcher_record', return_value=launch),
              mock.patch.object(
                  runner, '_external_authorization', return_value=_authorization()
              ),
              mock.patch.object(
                  runner, '_model_cache_hit_evidence', return_value=evidence
              ),
              mock.patch.object(
                  runner.bootstrap, '_rename_noreplace', side_effect=fail_target
              ),
          ):
            with self.assertRaises(runner.bootstrap.PublicationError) as caught:
              runner._orchestrate_postcompile_provenance(  # pylint: disable=protected-access
                  lowered=lowered, compiled=compiled, compile_start=0.0,
                  compiler_budget=budget, launch=launch, start=start,
                  original_frozen={}, frozen={}, imports_pre={},
                  imports_post={}, checkpoint_binding={},
                  reference_object_binding={}, protobuf_record={},
                  adapted_signatures=_frozen_signatures(),
                  signature_attestation=_signature_attestation(),
                  signature_binding=signature_binding,
                  source_input_audit_prefix=source_prefix,
                  current_phase={}, results=(),
              )
            self.assertEqual(
                Path(caught.exception.publication_failure[
                    'final_relative_path'
                ]).name,
                target,
            )
            same_object = (
                runner._partial_same_object_attestation(  # pylint: disable=protected-access
                    lowered, compiled, stage='stablehlo_text_extraction'
                )
                if target == 'NONPUBLICATION_TERMINAL_FAILURE.json'
                else runner._successful_same_object_attestation(  # pylint: disable=protected-access
                    lowered, compiled
                )
            )
            runner._write_terminal_failure(  # pylint: disable=protected-access
                caught.exception, completed_record_count=0,
                source_input_audit=source_prefix,
                same_object_attestation=same_object,
                phase_state=runner._phase_state(  # pylint: disable=protected-access
                    preflight_passed=True, start_persisted=True,
                    post_start_source_gate_passed=True,
                    pre_model_import_inventory_persisted=True,
                    protobuf_persisted=True,
                    model_construction_attempted=True,
                    model_constructed=True, reference_cases_loaded=True,
                    signatures_captured=True,
                    signature_attestation_persisted=True,
                    post_model_import_inventory_persisted=True,
                    lower_attempted=True, lower_succeeded=True,
                    compile_attempted=True, compile_succeeded=True,
                ),
                failed_current_binding=None,
            )
          self.assertTrue((root / 'TERMINAL_FAILURE.json').is_file())
          self.assertFalse(
              (root / 'NONPUBLICATION_TERMINAL_FAILURE.json').exists()
          )

  def test_postdiagnostic_record_construction_is_persistence_operation(self):
    lowered = object()
    compiled = object()
    gate = {'source_program_exact': True}
    diagnostic_compiler = {
        'same_object_attestation': _same_object(),
        'source_program_gate_without_backend_diagnostics': gate,
    }
    budget = runner.OneShotCompilerBudget()
    budget.request('lower')
    budget.request('compile')
    with (
        mock.patch.object(
            runner, '_cache_tree_binding', return_value={'tree_sha256': 'a'}
        ),
        mock.patch.object(
            runner, '_model_cache_hit_evidence', return_value={'cache_hit': False}
        ),
        mock.patch.object(
            runner, '_extract_compiler_graph_texts', return_value={
                'stablehlo': 's', 'hlo': 'h', 'compiled_hlo': 'c',
            }
        ),
        mock.patch.object(
            runner, '_publish_compiler_graphs', return_value={
                'artifacts': {}, 'compiled_hlo_text': 'c',
            }
        ),
        mock.patch.object(
            runner, '_persist_terminal_import_inventory',
            return_value=({}, {'all_exact': True}),
        ),
        mock.patch.object(
            runner, 'derive_source_input_audit', return_value=_source_audit()
        ),
        mock.patch.object(
            runner, '_entry_abi_binding', return_value={
                'normalized_line_sha256': 'e' * 64,
            }
        ),
        mock.patch.object(runner, '_backend_diagnostics', return_value={}),
        mock.patch.object(
            runner, '_compiler_artifacts',
            side_effect=TypeError('injected diagnostic record JSON failure'),
        ),
        mock.patch.object(
            runner, '_route_diagnostic_failure',
            return_value=diagnostic_compiler,
        ) as route,
        mock.patch.object(
            runner, '_relative_file_binding', return_value={
                'path': 'compiler/eight_row/COMPILER_DIAGNOSTIC_FAILURE.json',
                'sha256': 'f' * 64, 'size_bytes': 1,
            }
        ),
        mock.patch.object(runner, '_write_common_terminal') as terminal,
    ):
      result = runner._orchestrate_postcompile_provenance(  # pylint: disable=protected-access
          lowered=lowered, compiled=compiled, compile_start=0.0,
          compiler_budget=budget,
          launch={
              'v3_3_3_run': {'eight_row_compiler': {}},
              'gate_a': {'v3_3_2_run': {'eight_row_compiler': {}}},
          },
          start={'same_process_preflight': {
              'model_cache_pre_import': {},
          }},
          original_frozen={}, frozen={}, imports_pre={}, imports_post={},
          checkpoint_binding={}, reference_object_binding={},
          protobuf_record={}, adapted_signatures={},
          signature_attestation={}, signature_binding={},
          source_input_audit_prefix={name: True for name in (
              runner.bootstrap.SOURCE_INPUT_AUDIT_KEYS[:-1]
          )} | {runner.bootstrap.SOURCE_INPUT_AUDIT_KEYS[-1]: None},
          current_phase={}, results=(),
      )
    self.assertIsNone(result)
    trigger, reason = route.call_args.args[:2]
    self.assertIsInstance(trigger, runner.DiagnosticPersistenceFailure)
    self.assertEqual(reason, 'diagnostic_persistence_failure')
    self.assertEqual(
        terminal.call_args.kwargs['stop_reason'],
        'diagnostic_persistence_failure',
    )

  def test_diagnostic_trigger_reason_pair_is_operation_typed(self):
    cases = (
        (runner.EntryAbiParserFailure, 'diagnostic_parser_failure'),
        (runner.BackendDiagnosticParserFailure, 'diagnostic_parser_failure'),
        (runner.DiagnosticPersistenceFailure,
         'diagnostic_persistence_failure'),
        (runner.CacheSignalUnavailable, 'cache_signal_unavailable'),
        (runner.FingerprintFormulaMismatch, 'fingerprint_formula_mismatch'),
    )
    reasons = {
        'diagnostic_parser_failure', 'diagnostic_persistence_failure',
        'cache_signal_unavailable', 'fingerprint_formula_mismatch',
    }
    for error_type, reason in cases:
      trigger = error_type(ValueError('misleading free-form message'))
      self.assertEqual(
          runner._diagnostic_stop_reason(trigger),  # pylint: disable=protected-access
          reason,
      )
      for wrong in reasons - {reason}:
        with self.subTest(error=error_type.__name__, wrong=wrong):
          with self.assertRaisesRegex(ValueError, 'disagree'):
            runner._route_diagnostic_failure(  # pylint: disable=protected-access
                trigger, wrong, lowered=object(), compiled=object(),
                published_graphs={}, program_signatures={},
                program_signature_attestation={},
                program_signature_attestation_binding={},
                source_input_audit=_source_audit(), entry_abi=None,
                attempt_budget_audit={}, historical_cache_binding={},
                cache_hit_evidence={'cache_hit': False},
            )

  def test_nonpublication_source_gate_rejects_every_leaf_and_key_tamper(self):
    graph_bindings = {
        'compiler/eight_row/graph.stablehlo.mlir': {
            'sha256': 'a' * 64, 'size_bytes': 11,
        },
        'compiler/eight_row/graph.pre_backend.hlo.txt': {
            'sha256': 'b' * 64, 'size_bytes': 12,
        },
        'compiler/eight_row/graph.compiled.hlo.txt': {
            'sha256': 'c' * 64, 'size_bytes': 13,
        },
    }
    observed = {
        'stablehlo_sha256': 'a' * 64, 'stablehlo_size_bytes': 11,
        'pre_backend_hlo_sha256': 'b' * 64,
        'pre_backend_hlo_size_bytes': 12,
        'program_signatures_sha256': runner.SOURCE_PROGRAM_CONTRACT[
            'program_signatures_sha256'
        ],
        'entry_abi_sha256': runner.SOURCE_PROGRAM_CONTRACT[
            'entry_abi_sha256'
        ],
    }
    source = _source_audit()
    same = _same_object()
    gate = runner.evaluate_source_program_gate(
        observed, _frozen_signatures(),
        {'program_signatures': _frozen_signatures()}, source,
        _signature_attestation(), same,
    )
    validate = lambda value, entry=observed['entry_abi_sha256']: (
        runner._validate_nonpublication_source_gate(  # pylint: disable=protected-access
            value, graph_bindings=graph_bindings,
            source_input_audit=source, same_object_attestation=same,
            expected_entry_abi_sha256=entry,
        )
    )
    validate(gate)
    tampered = copy.deepcopy(gate)
    tampered['extra'] = None
    with self.assertRaises(ValueError):
      validate(tampered)
    for key in tuple(gate):
      with self.subTest(top_level_key=key):
        tampered = copy.deepcopy(gate)
        tampered.pop(key)
        with self.assertRaises(ValueError):
          validate(tampered)
    for key in tuple(observed):
      with self.subTest(observed_leaf=key):
        tampered = copy.deepcopy(gate)
        tampered['observed'][key] = (
            999 if key.endswith('bytes') else '9' * 64
        )
        with self.assertRaises(ValueError):
          validate(tampered)
    for key in (
        'stablehlo_exact', 'pre_backend_hlo_exact',
        'program_signature_structure_exact',
        'program_signatures_canonical_exact', 'entry_abi_exact',
        'source_runtime_device_toolchain_checkpoint_reference_exact',
        'same_lowered_compiled_object', 'source_program_exact',
    ):
      with self.subTest(primitive=key):
        tampered = copy.deepcopy(gate)
        tampered[key] = not tampered[key]
        with self.assertRaises(ValueError):
          validate(tampered)
    for key in (
        'source_input_audit', 'source_input_audit_content_binding',
        'same_object_attestation',
        'same_object_attestation_content_binding', 'contract',
    ):
      with self.subTest(bound_object=key):
        tampered = copy.deepcopy(gate)
        if key in {'source_input_audit', 'same_object_attestation'}:
          first = next(iter(tampered[key]))
          tampered[key][first] = not tampered[key][first]
        else:
          tampered[key]['sha256' if 'binding' in key else 'tamper'] = '0' * 64
        with self.assertRaises(ValueError):
          validate(tampered)
    entry_failure_gate = runner.evaluate_source_program_gate(
        {**observed, 'entry_abi_sha256': ''}, _frozen_signatures(),
        {'program_signatures': _frozen_signatures()}, source,
        _signature_attestation(), same,
    )
    validate(entry_failure_gate, '')
    forged = copy.deepcopy(entry_failure_gate)
    forged['observed']['entry_abi_sha256'] = (
        runner.SOURCE_PROGRAM_CONTRACT['entry_abi_sha256']
    )
    forged['entry_abi_exact'] = True
    forged['source_program_exact'] = all(
        forged[name] for name in (
            'stablehlo_exact', 'pre_backend_hlo_exact',
            'program_signatures_canonical_exact', 'entry_abi_exact',
            'source_runtime_device_toolchain_checkpoint_reference_exact',
            'same_lowered_compiled_object',
        )
    )
    with self.assertRaises(ValueError):
      validate(forged, '')

  def test_nonpublication_same_object_rejects_missing_and_extra_keys(self):
    stages = runner.bootstrap.NONPUBLICATION_TERMINAL_CONTRACT_V3_3_4_3[
        'failure_stages'
    ]
    valid = _same_object()
    runner._validate_nonpublication_same_object(  # pylint: disable=protected-access
        valid, failure_stage=stages[3], stages=stages
    )
    for mutation in ('missing', 'extra'):
      tampered = copy.deepcopy(valid)
      if mutation == 'missing':
        tampered.pop('compiled_python_id')
      else:
        tampered['extra'] = True
      with self.subTest(mutation=mutation), self.assertRaises(ValueError):
        runner._validate_nonpublication_same_object(  # pylint: disable=protected-access
            tampered, failure_stage=stages[3], stages=stages
        )

  def test_predispatch_terminal_router_covers_every_compiled_gate(self):
    base = {
        'kernel_cache_provenance': {'cache_hit_evidence': {'cache_hit': False}},
        'attempt_budget_audit': {'forbidden_request': None},
        'same_object_attestation': _same_object(),
        'diagnostic_provenance_complete': True,
        'source_program_gate': {'source_program_exact': True},
    }
    self.assertIsNone(runner._predispatch_controlled_stop(base))  # pylint: disable=protected-access
    cases = []
    cache = copy.deepcopy(base)
    cache['kernel_cache_provenance']['cache_hit_evidence']['cache_hit'] = True
    cases.append((cache, 'controlled_stop_cache_hit',
                  'model_cache_post_compile_hit'))
    for operation in ('lower', 'compile'):
      value = copy.deepcopy(base)
      value['attempt_budget_audit']['forbidden_request'] = operation
      cases.append((value, 'controlled_stop_attempt_budget_violation',
                    f'second_{operation}_attempt_forbidden'))
    for field, reason in (
        ('stablehlo_read_from_lowered_object', 'lowered_object_identity_lost'),
        ('compile_argument_is_lowered_object', 'compile_argument_identity_lost'),
        ('compiled_hlo_read_from_compiled_object', 'compiled_object_identity_lost'),
        ('apply_callable_is_compiled_object', 'apply_callable_identity_lost'),
    ):
      value = copy.deepcopy(base)
      value['same_object_attestation'][field] = False
      cases.append((value, 'controlled_stop_same_object_provenance_failure', reason))
    diagnostic = copy.deepcopy(base)
    diagnostic['diagnostic_provenance_complete'] = False
    cases.append((diagnostic, 'controlled_stop_diagnostic_provenance_failure',
                  'diagnostic_parser_failure'))
    source = copy.deepcopy(base)
    source['source_program_gate']['source_program_exact'] = False
    cases.append((source, 'controlled_stop_source_program_mismatch',
                  'source_program_mismatch'))
    for value, status, reason in cases:
      with self.subTest(status=status, reason=reason):
        decision = runner._predispatch_controlled_stop(value)  # pylint: disable=protected-access
        self.assertEqual(decision[:2], (status, reason))

  def test_one_shot_compiler_budget_blocks_second_request_before_counting(self):
    for operation in ('lower', 'compile'):
      with self.subTest(operation=operation):
        budget = runner.OneShotCompilerBudget()
        budget.request(operation)
        with self.assertRaises(runner.AttemptBudgetViolation):
          budget.request(operation)
        audit = budget.audit()
        self.assertEqual(audit[f'{operation}_invocations'], 1)
        self.assertEqual(audit['forbidden_request'], operation)
        self.assertTrue(
            audit['forbidden_request_detected_before_invocation']
        )

  def test_second_compile_uses_failure_record_without_gate_or_diagnostics(self):
    lowered = object()
    compiled = object()
    artifacts = {
        name: {
            'path': f'compiler/eight_row/{name}.txt',
            'sha256': char * 64, 'size_bytes': 1,
        }
        for name, char in (
            ('stablehlo', 'a'), ('hlo', 'b'), ('compiled_hlo', 'c')
        )
    }
    error = runner.AttemptBudgetViolation('compile')
    budget = {
        'lower_budget': 1, 'compile_budget': 1,
        'lower_invocations': 1, 'compile_invocations': 1,
        'forbidden_request': 'compile',
        'forbidden_request_detected_before_invocation': True,
    }
    with (
        mock.patch.object(runner, '_write_new'),
        mock.patch.object(runner, '_relative_file_binding', return_value={
            'path': 'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json',
            'sha256': 'd' * 64, 'size_bytes': 1,
        }),
        mock.patch.object(runner, '_external_authorization', return_value=_authorization()),
        mock.patch.object(runner, '_cache_tree_binding', return_value={
            'cache_root': '/fixture/model-cache', 'tree_sha256': 'e' * 64,
        }),
        mock.patch.object(runner, '_model_cache_hit_evidence', return_value={
            'cache_hit': False,
        }),
    ):
      record = runner._attempt_budget_failure_artifact(  # pylint: disable=protected-access
          error, lowered=lowered, compiled=compiled,
          published_graphs={
              'artifacts': artifacts, 'compiled_hlo_text': 'diagnostic-only',
          },
          seconds=1.0, program_signatures=_frozen_signatures(),
          kernel_cache_preimport_attestation={
              'cache_root': '/fixture/model-cache',
          },
          program_signature_attestation=_signature_attestation(),
          source_input_audit=_source_audit(), attempt_budget_audit=budget,
      )
    self.assertEqual(record['status'], 'attempt_budget_failure')
    self.assertEqual(record['failure_stage'], 'second_compile_guarded')
    self.assertEqual(record['successful_compile_count'], 1)
    self.assertEqual(set(record['artifacts']), set(artifacts))
    self.assertIsNone(record['source_program_gate'])
    self.assertIsNone(record['diagnostic_provenance_complete'])
    self.assertNotIn('backend_diagnostics', record)
    self.assertNotIn('diagnostic_comparisons', record)

  def test_signature_failure_prefix_stops_at_first_invalid_fixed_path(self):
    runtime = _runtime_signatures()
    frozen = _frozen_signatures()
    full_runtime = runner._validated_signature_tag_prefix(  # pylint: disable=protected-access
        runtime, runtime=True
    )
    full_frozen = runner._validated_signature_tag_prefix(  # pylint: disable=protected-access
        frozen, runtime=False
    )
    self.assertEqual(len(full_runtime), 32)
    self.assertEqual(len(full_frozen), 32)
    for index, expected in enumerate(full_runtime):
      with self.subTest(index=index, path=expected['path']):
        drifted = _runtime_signatures()
        tokens = expected['path'].strip('/').split('/')
        if len(tokens) == 2:
          drifted[tokens[0]]['leaves'] = list(
              drifted[tokens[0]]['leaves']
          )
        else:
          drifted[tokens[0]]['leaves'][int(tokens[2])]['shape'] = list(
              drifted[tokens[0]]['leaves'][int(tokens[2])]['shape']
          )
        prefix = runner._validated_signature_tag_prefix(  # pylint: disable=protected-access
            drifted, runtime=True
        )
        self.assertEqual(prefix, full_runtime[:index])

  def test_terminal_status_reason_inventory_is_literal_and_exhaustive(self):
    self.assertEqual(
        set(runner._TERMINAL_STATUS_REASONS),  # pylint: disable=protected-access
        set(runner.bootstrap.TERMINAL_CONTRACT['statuses']),
    )
    for status, reasons in runner._TERMINAL_STATUS_REASONS.items():  # pylint: disable=protected-access
      for reason in reasons:
        with self.subTest(status=status, reason=reason):
          runner._validate_terminal_identity(status, reason)  # pylint: disable=protected-access
    with self.assertRaises(ValueError):
      runner._validate_terminal_identity(  # pylint: disable=protected-access
          'controlled_stop_cache_hit', 'source_program_mismatch'
      )

  def test_literal_terminal_matrix_accepts_every_status_reason_row(self):
    keys = runner.bootstrap.TERMINAL_CONTRACT['run_complete_keys']
    for status, reasons in runner._TERMINAL_STATUS_REASONS.items():  # pylint: disable=protected-access
      for reason in reasons:
        with self.subTest(status=status, reason=reason):
          compiler_state = runner._terminal_compiler_state(  # pylint: disable=protected-access
              status, reason
          )
          k, d = (80, 0) if status == 'complete_structural_sidecar' else (0, 0)
          if status == 'controlled_stop_four_call_invalid':
            d = 4
          started = (
              320 if status == 'complete_structural_sidecar'
              else 4 if status == 'controlled_stop_four_call_invalid'
              else 1 if reason == 'model_dispatch_failure'
              else 0
          )
          completed = (
              320 if status == 'complete_structural_sidecar'
              else 4 if status == 'controlled_stop_four_call_invalid'
              else 0
          )
          phase = _terminal_phase(
              status, reason, compiler_state, started=started
          )
          if status == 'complete_structural_sidecar':
            detail_contract = (None, None, None)
          elif status == 'controlled_stop_four_call_invalid':
            detail_contract = ('record_validation', None, None)
          else:
            detail_contract = runner._TERMINAL_DETAIL_BY_REASON[reason]  # pylint: disable=protected-access
          detail = {
              'k_valid_records': k, 'd_completed': d,
              'failed_execution_index': None, 'failed_call_role': None,
              'failure_phase': detail_contract[0],
              'forbidden_operation': detail_contract[1],
              'provenance_artifact_role': detail_contract[2],
          }
          if compiler_state in {
              'none', 'signature_failure', 'lowered', 'precompiled',
              'compiled_guarded',
          }:
            source_gate = None
            diagnostics = None
          elif compiler_state == 'diagnostic_failure':
            source_gate = {'source_program_exact': False}
            diagnostics = False
          else:
            source_gate = {'source_program_exact': (
                status != 'controlled_stop_source_program_mismatch'
            )}
            diagnostics = True
          signature = None if compiler_state == 'none' else {'path': 'sig'}
          compiler_binding = (
              None if compiler_state in {'none', 'signature_failure'}
              else {'path': 'compiler'}
          )
          record = {key: None for key in keys}
          record.update({
              'status': status, 'stop_reason': reason,
              'failure': (
                  None if status == 'complete_structural_sidecar'
                  else {'type': 'Fixture', 'message': 'fixture', 'traceback': ''}
              ),
              'phase_state': phase, 'terminal_detail': detail,
              'source_input_audit': _source_audit(),
              'program_signature_attestation_binding': signature,
              'source_program_gate': source_gate,
              'compiler_binding': compiler_binding,
              'compiler_artifact_bindings': {
                  path: {'sha256': 'a' * 64, 'size_bytes': 1}
                  for path in runner._COMPILER_MEMBERSHIP_BY_STATE[  # pylint: disable=protected-access
                      compiler_state
                  ]
              },
              'diagnostic_provenance_complete': diagnostics,
              'valid_record_count': k,
              'dispatch_journal': {
                  'started_count': started, 'completed_count': completed,
              },
              'model_apply_attempt_count': started,
              'model_apply_success_count': completed,
              **_prior_prefix_fields(),
          })
          with mock.patch.object(
              runner, '_launcher_record',
              return_value={'start': _minimal_start()},
          ):
            runner._validate_common_terminal_semantics(record)  # pylint: disable=protected-access

          drifted = copy.deepcopy(record)
          drifted['compiler_artifact_bindings']['unexpected'] = {
              'sha256': 'b' * 64, 'size_bytes': 1,
          }
          with mock.patch.object(
              runner, '_launcher_record',
              return_value={'start': _minimal_start()},
          ), self.assertRaises(RuntimeError):
            runner._validate_common_terminal_semantics(drifted)  # pylint: disable=protected-access

  def test_common_terminal_serializer_emits_every_literal_matrix_row(self):
    start = {
        'freeze_sha256': 'f' * 64, 'git_head': 'a' * 40,
        'started_at_unix_s': 0.0,
        **_prior_prefix_fields(),
        'same_process_preflight': {'model_cache_pre_import': {
            'cache_root': '/fixture/model-cache', 'tree_sha256': '0' * 64,
        }},
    }
    empty_manifest = {
        key: None for key in runner.bootstrap.RAW_MANIFEST_CONTRACT['keys']
    }
    for status, reasons in runner._TERMINAL_STATUS_REASONS.items():  # pylint: disable=protected-access
      for reason in reasons:
        with self.subTest(status=status, reason=reason):
          state = runner._terminal_compiler_state(status, reason)  # pylint: disable=protected-access
          compiler_files = {
              path: {'sha256': 'a' * 64, 'size_bytes': 1}
              for path in runner._COMPILER_MEMBERSHIP_BY_STATE[state]  # pylint: disable=protected-access
          }
          if state in {'none', 'signature_failure'}:
            compiler = None
          elif state == 'diagnostic_failure':
            compiler = {
                'status': 'diagnostic_provenance_failure',
                'source_program_gate_without_backend_diagnostics': {
                    'source_program_exact': False,
                },
                'attempt_budget_audit': {},
                'diagnostic_provenance_complete': False,
                'lower_attempt_count': 1, 'compile_attempt_count': 1,
                'successful_compile_count': 1,
            }
          else:
            compiled_gate = state == 'compiled'
            compiler = {
                'source_program_gate': (
                    {'source_program_exact': (
                        status != 'controlled_stop_source_program_mismatch'
                    )} if compiled_gate else None
                ),
                'attempt_budget_audit': {},
                'diagnostic_provenance_complete': (
                    True if compiled_gate else None
                ),
                'backend_diagnostics': {} if compiled_gate else None,
                'diagnostic_comparisons': {} if compiled_gate else None,
                'lower_attempt_count': 1,
                'compile_attempt_count': int(
                    state in {'precompiled', 'compiled', 'compiled_guarded'}
                ),
                'successful_compile_count': int(
                    state in {'compiled', 'compiled_guarded'}
                ),
            }
          results = ([{}] * 80 if status == 'complete_structural_sidecar' else [])
          completed_count = (
              320 if status == 'complete_structural_sidecar'
              else 4 if status == 'controlled_stop_four_call_invalid'
              else 0
          )
          started_count = (
              1 if reason == 'model_dispatch_failure' else completed_count
          )
          journal = {
              'started_count': started_count,
              'completed_count': completed_count,
              'started_bindings': {}, 'completed_bindings': {},
              'started_tree_sha256': '0' * 64,
              'completed_tree_sha256': '0' * 64,
              'started_prefix_exact': True, 'completed_prefix_exact': True,
          }
          if status == 'complete_structural_sidecar':
            detail = (None, None, None)
          elif status == 'controlled_stop_four_call_invalid':
            detail = ('record_validation', None, None)
          else:
            detail = runner._TERMINAL_DETAIL_BY_REASON[reason]  # pylint: disable=protected-access
          captured = {}
          def capture(path, value):
            captured[path.name] = copy.deepcopy(value)
            return 'f' * 64
          phase = _terminal_phase(
              status, reason, state, started=started_count
          )
          signature_binding = (
              None if state == 'none' else {'path': 'signature'}
          )
          compiler_binding = (
              None if state in {'none', 'signature_failure'}
              else {'path': 'compiler'}
          )
          with (
              mock.patch.object(runner, '_raw_manifest', return_value=empty_manifest),
              mock.patch.object(runner, '_write_new', side_effect=capture),
              mock.patch.object(runner, '_journal_summary', return_value=journal),
              mock.patch.object(runner, '_launcher_record', return_value={
                  'start': start, 'v3_3_3_run': {}, 'v3_3_3_1_archive': {},
              }),
              mock.patch.object(runner, '_run_tree_binding', return_value={
                  'file_count': 0, 'directory_count': 1,
                  'file_bindings': {}, 'file_tree_sha256': '0' * 64,
                  'directory_paths': ['.'], 'directory_tree_sha256': '0' * 64,
              }),
              mock.patch.object(runner, '_compiler_artifact_bindings', return_value=compiler_files),
              mock.patch.object(runner, '_import_phase_bindings', return_value={
                  'pre_model': None, 'post_model_precompile': None,
                  'terminal': None,
              }),
              mock.patch.object(runner, '_final_model_cache_binding', return_value={}),
              mock.patch.object(runner, '_cache_tree_binding', return_value={
                  'tree_sha256': '0' * 64,
              }),
              mock.patch.object(runner, '_external_authorization', return_value=_authorization()),
              mock.patch.object(runner.bootstrap, 'publication_audit', return_value={}),
          ):
            runner._write_common_terminal(  # pylint: disable=protected-access
                status=status, stop_reason=reason, message='fixture',
                failure=(
                    None if status == 'complete_structural_sidecar'
                    else {'type': 'Fixture', 'message': 'fixture', 'traceback': ''}
                ),
                results=results, failed_current_binding=None,
                compiler=compiler, source_input_audit=_source_audit(),
                same_object_attestation=(
                    None if state in {'none', 'signature_failure'}
                    else _same_object()
                ),
                phase_state=phase, failure_phase=detail[0],
                forbidden_operation=detail[1],
                provenance_artifact_role=detail[2],
                program_signature_attestation_binding=signature_binding,
                compiler_binding=compiler_binding,
            )
          self.assertEqual(
              set(captured['RUN_COMPLETE.json']),
              set(runner.bootstrap.TERMINAL_CONTRACT['run_complete_keys']),
          )
          self.assertEqual(captured['RUN_COMPLETE.json']['status'], status)

  def test_diagnostic_failure_serializer_does_not_repeat_failed_entry_abi(self):
    artifacts = {
        'stablehlo': {'path': 'compiler/eight_row/graph.stablehlo.mlir',
                      'sha256': 'a' * 64, 'size_bytes': 1},
        'hlo': {'path': 'compiler/eight_row/graph.pre_backend.hlo.txt',
                'sha256': 'b' * 64, 'size_bytes': 1},
        'compiled_hlo': {'path': 'compiler/eight_row/graph.compiled.hlo.txt',
                         'sha256': 'c' * 64, 'size_bytes': 1},
    }
    fake_gate = {'source_program_exact': False}
    with (
        mock.patch.object(runner, '_entry_abi_binding', side_effect=AssertionError(
            'entry ABI must not be retried'
        )),
        mock.patch.object(runner, 'evaluate_source_program_gate', return_value=fake_gate),
        mock.patch.object(runner, '_launcher_record', return_value={
            'gate_a': {'v3_3_2_run': {'eight_row_compiler': {}}}
        }),
        mock.patch.object(runner, '_external_authorization', return_value=_authorization()),
        mock.patch.object(runner, '_relative_file_binding', return_value={
            'path': 'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json',
            'sha256': 'd' * 64, 'size_bytes': 1,
        }),
        mock.patch.object(runner, '_write_new'),
    ):
      record = runner._compiler_diagnostic_failure_artifact(  # pylint: disable=protected-access
          ValueError('fingerprint formula mismatch'),
          lowered=object(), compiled=object(),
          published_graphs={'artifacts': artifacts, 'compiled_hlo_text': 'bad'},
          program_signatures=_frozen_signatures(),
          program_signature_attestation=_signature_attestation(),
          source_input_audit=_source_audit(), entry_abi=None,
      )
    self.assertFalse(record['diagnostic_provenance_complete'])
    self.assertEqual(record['source_program_gate_without_backend_diagnostics'], fake_gate)
    self.assertEqual(
        runner._diagnostic_stop_reason(  # pylint: disable=protected-access
            runner.FingerprintFormulaMismatch(
                ValueError('message deliberately says cache parser persistence')
            )
        ),
        'fingerprint_formula_mismatch',
    )
    with self.assertRaisesRegex(ValueError, 'not captured'):
      runner._diagnostic_stop_reason(  # pylint: disable=protected-access
          ValueError('fingerprint formula mismatch')
      )

  def test_entry_abi_requires_one_nonempty_hex_fingerprint(self):
    valid = (
        'HloModule main, fingerprint_before_lhs="0aFE19", '
        'entry_computation_layout={(f32[])->f32[]}'
    )
    binding = runner._entry_abi_binding(valid)  # pylint: disable=protected-access
    self.assertEqual(binding['normalized_line_size_bytes'], len(
        valid.replace('"0aFE19"', '"<backend-generated>"').encode('utf-8')
    ))
    invalid = (
        'HloModule main, entry_computation_layout={(f32[])->f32[]}',
        'HloModule main, fingerprint_before_lhs=""',
        'HloModule main, fingerprint_before_lhs="not-hex"',
        'HloModule main, fingerprint_before_lhs=abcd',
        ('HloModule main, fingerprint_before_lhs="aa", '
         'fingerprint_before_lhs="bb"'),
        ('HloModule main, fingerprint_before_lhs="aa", '
         'fingerprint_before_lhs=malformed'),
    )
    for line in invalid:
      with self.subTest(line=line), self.assertRaises(
          runner.FingerprintFormulaMismatch
      ):
        runner._entry_abi_binding(line)  # pylint: disable=protected-access

  def test_real_compiler_serializer_allows_compiled_backend_difference(self):
    compiled_digest = '9' * 64
    artifacts = {
        'stablehlo': {
            'path': 'compiler/eight_row/graph.stablehlo.mlir',
            'sha256': runner.SOURCE_PROGRAM_CONTRACT['stablehlo_sha256'],
            'size_bytes': runner.SOURCE_PROGRAM_CONTRACT[
                'stablehlo_size_bytes'
            ],
        },
        'hlo': {
            'path': 'compiler/eight_row/graph.pre_backend.hlo.txt',
            'sha256': runner.SOURCE_PROGRAM_CONTRACT['pre_backend_hlo_sha256'],
            'size_bytes': runner.SOURCE_PROGRAM_CONTRACT[
                'pre_backend_hlo_size_bytes'
            ],
        },
        'compiled_hlo': {
            'path': 'compiler/eight_row/graph.compiled.hlo.txt',
            'sha256': compiled_digest,
            'size_bytes': 7,
        },
    }
    prior = {
        'artifacts': copy.deepcopy(artifacts),
        'executable_fingerprint': '0' * 64,
        'program_signatures': _frozen_signatures(),
    }
    attestation = _signature_attestation()
    with (
        mock.patch.object(runner, '_write_new', return_value='a' * 64),
        mock.patch.object(
            runner, '_relative_file_binding', return_value={
                'path': 'compiler/eight_row/PROGRAM_SIGNATURE_ATTESTATION.json',
                'sha256': 'a' * 64, 'size_bytes': 1,
            }
        ),
        mock.patch.object(
            runner, '_entry_abi_binding', return_value={
                'normalized_line_sha256': runner.SOURCE_PROGRAM_CONTRACT[
                    'entry_abi_sha256'
                ]
            }
        ),
        mock.patch.object(
            runner, '_backend_diagnostics', return_value={'changed': True}
        ),
        mock.patch.object(
            runner, '_cache_tree_binding', return_value={'tree': 'post'}
        ),
        mock.patch.object(
            runner, '_model_cache_hit_evidence', return_value={
                'cache_hit': False
            }
        ),
    ):
      record = runner._compiler_artifacts(  # pylint: disable=protected-access
          object(), object(), 1.0, prior, prior, _frozen_signatures(),
          program_signature_attestation=attestation,
          source_input_audit=_source_audit(),
          kernel_cache_preimport_attestation={'cache_root': '/fixture/cache'},
          published_graphs={
              'artifacts': artifacts,
              'compiled_hlo_text': 'changed',
          },
      )
    self.assertTrue(record['source_program_gate']['source_program_exact'])
    self.assertFalse(
        record['diagnostic_comparisons']['v3_3'][
            'executable_fingerprint_exact'
        ]
    )
    self.assertTrue(record[
        'diagnostic_comparisons'
    ]['compiled_backend_differences_are_diagnostic_only'])

  def test_loaded_scientific_module_contract_has_exact_root_roles(self):
    rows = runner.loaded_scientific_modules()
    self.assertEqual(
        sum(row['root'] == 'upstream_alphagenome_checkout' for row in rows),
        26,
    )
    locked = {
        row['name'] for row in rows
        if row['root'] == 'locked_opensplice_checkout'
    }
    self.assertTrue({
        'v3_3_4_4_runner', 'v3_3_4_4_launcher',
        'validate_encoder_skip_ood_sidecar_bootstrap_v3_3_4_4',
        'run_device_preflight_v3_3',
    }.issubset(locked))
    self.assertEqual(rows, sorted(rows, key=lambda row: (
        row['name'], row['path']
    )))

  def test_model_setup_boundary_catches_each_checkpoint_reference_stage(self):
    cases = [types.SimpleNamespace(order=index) for index in range(20)]
    model = types.SimpleNamespace(_params=object(), _state=object())
    common = (None, None, object(), object(), None, object(), 'a' * 64)
    stages = ('load_cases', 'checkpoint_path', 'checkpoint_validation',
              'model_create', 'reference_validation', 'case_inputs')
    for stage in stages:
      with self.subTest(stage=stage):
        patches = [
            mock.patch.object(runner.v32, 'load_development_cases', return_value=cases),
            mock.patch.object(runner.v32.v2, '_checkpoint_path', return_value=Path('/x')),
            mock.patch.object(runner.v32, 'validate_checkpoint', return_value={'ok': True}),
            mock.patch.object(runner.dna_model, 'create', return_value=model),
            mock.patch.object(runner.v32, 'validate_reference_object', return_value={'ok': True}),
            mock.patch.object(runner.v32, '_case_inputs', return_value=common),
        ]
        index = stages.index(stage)
        patches[index] = mock.patch.object(
            (
                runner.v32 if stage not in {'checkpoint_path', 'model_create'}
                else runner.v32.v2 if stage == 'checkpoint_path'
                else runner.dna_model
            ),
            {
                'load_cases': 'load_development_cases',
                'checkpoint_path': '_checkpoint_path',
                'checkpoint_validation': 'validate_checkpoint',
                'model_create': 'create',
                'reference_validation': 'validate_reference_object',
                'case_inputs': '_case_inputs',
            }[stage],
            side_effect=RuntimeError(stage),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
          with self.assertRaises(runner.ModelSetupError) as caught:
            runner._construct_model_and_inputs()  # pylint: disable=protected-access
        if stage in {'model_create', 'reference_validation', 'case_inputs'}:
          self.assertIsNotNone(caught.exception.checkpoint_binding)
        if stage in {'reference_validation', 'case_inputs'}:
          self.assertIsNone(caught.exception.reference_binding)

  def test_signature_boundary_catches_prototype_and_signature_drift(self):
    common = {index: (object(), object(), object(), 'a' * 64)
              for index in range(20)}
    for name in ('batch', 'interventions', 'pytree', 'equality'):
      effects = {
          'batch': RuntimeError(name) if name == 'batch' else object(),
          'interventions': (
              RuntimeError(name) if name == 'interventions' else object()
          ),
          'pytree': RuntimeError(name) if name == 'pytree' else {},
          'equality': RuntimeError(name) if name == 'equality' else None,
      }
      def side(value):
        return {'side_effect': value} if isinstance(value, Exception) else {
            'return_value': value
        }
      with (
          self.subTest(stage=name),
          mock.patch.object(runner, '_eight_row_batch', **side(effects['batch'])),
          mock.patch.object(
              runner.v33, 'eight_row_interventions',
              **side(effects['interventions'])
          ),
          mock.patch.object(runner.v32, 'pytree_signature', **side(effects['pytree'])),
          mock.patch.object(
              runner.v32, 'assert_same_program_signature',
              **side(effects['equality'])
          ),
      ):
        with self.assertRaisesRegex(RuntimeError, name):
          runner._capture_signature_inputs(common)  # pylint: disable=protected-access

  def test_reference_sequence_audit_stays_unavailable_at_each_failed_case(self):
    cases = [types.SimpleNamespace(order=index) for index in range(20)]
    model = types.SimpleNamespace(_params=object(), _state=object())
    success = (None, None, object(), object(), None, object(), 'a' * 64)
    for failed_index in range(20):
      calls = [0]
      def case_inputs(*unused):
        index = calls[0]
        calls[0] += 1
        if index == failed_index:
          raise RuntimeError(f'case-{failed_index}')
        return success
      with (
          self.subTest(failed_index=failed_index),
          mock.patch.object(runner.v32, 'load_development_cases', return_value=cases),
          mock.patch.object(runner.v32.v2, '_checkpoint_path', return_value=Path('/x')),
          mock.patch.object(runner.v32, 'validate_checkpoint', return_value={'ok': True}),
          mock.patch.object(runner.dna_model, 'create', return_value=model),
          mock.patch.object(runner.v32, 'validate_reference_object', return_value={'ok': True}),
          mock.patch.object(runner.v32, '_case_inputs', side_effect=case_inputs),
      ):
        with self.assertRaises(runner.ModelSetupError) as caught:
          runner._construct_model_and_inputs()  # pylint: disable=protected-access
      self.assertIsNone(caught.exception.reference_binding)

  def test_atomic_publication_is_no_replace_and_mode_0400(self):
    with tempfile.TemporaryDirectory() as directory, self._publication_root(
        directory
    ):
      path = Path(directory) / 'sealed.json'
      runner._publish_new_bytes(path, b'one')  # pylint: disable=protected-access
      self.assertEqual(path.read_bytes(), b'one')
      self.assertEqual(path.stat().st_mode & 0o777, 0o400)
      with self.assertRaises(runner.bootstrap.PublicationError) as caught:
        runner._publish_new_bytes(path, b'two')  # pylint: disable=protected-access
      self.assertEqual(
          caught.exception.publication_failure['failure_stage'],
          'final_preexistence',
      )
      self.assertEqual(path.read_bytes(), b'one')

  def test_journal_success_has_exact_started_completed_prefix(self):
    with tempfile.TemporaryDirectory() as directory, self._publication_root(
        directory
    ), mock.patch.object(
        runner.v32, '_timed_apply', return_value=('returned', 1.25)
    ):
      count = [0]
      result = runner._counted_apply(  # pylint: disable=protected-access
          object(), (), count, call_label='intended', execution_index=0,
          recipient=_Case(), anchor_id=0, call_index=0,
          source_input_audit_binding={'sha256': 'a' * 64},
          same_object_attestation_binding={'sha256': 'b' * 64},
      )
      self.assertEqual(result[:2], ('returned', 1.25))
      self.assertEqual(count, [1])
      self.assertTrue(Path(directory, 'dispatch_journal/started/000.json').is_file())
      self.assertTrue(Path(directory, 'dispatch_journal/completed/000.json').is_file())

  def test_journal_failure_keeps_started_without_completion(self):
    with tempfile.TemporaryDirectory() as directory, self._publication_root(
        directory
    ), mock.patch.object(
        runner.v32, '_timed_apply', side_effect=RuntimeError('dispatch failed')
    ):
      count = [0]
      with self.assertRaises(runner.CountedApplyError):
        runner._counted_apply(  # pylint: disable=protected-access
            object(), (), count, call_label='intended', execution_index=0,
            recipient=_Case(), anchor_id=0, call_index=0,
            source_input_audit_binding={'sha256': 'a' * 64},
            same_object_attestation_binding={'sha256': 'b' * 64},
        )
      self.assertEqual(count, [1])
      self.assertTrue(Path(directory, 'dispatch_journal/started/000.json').is_file())
      self.assertFalse(Path(directory, 'dispatch_journal/completed/000.json').exists())

  def test_run_anchor_dispatches_four_calls_with_one_counter_argument(self):
    recipient = types.SimpleNamespace(order=0, variant_id='recipient')
    donor = types.SimpleNamespace(order=1, variant_id='donor')
    common = (None, object(), object(), 'a' * 64)
    binding = {'path': 'event.json', 'sha256': 'b' * 64, 'size_bytes': 1}
    with (
        mock.patch.object(runner, '_eight_row_batch', return_value=object()),
        mock.patch.object(
            runner.v33, 'eight_row_interventions', return_value=object()
        ),
        mock.patch.object(runner.v32, 'assert_same_program_signature'),
        mock.patch.object(
            runner, '_counted_apply',
            return_value=(object(), 0.1, binding, binding),
        ) as counted,
    ):
      with self.assertRaises(runner.CurrentRecordStop) as caught:
        runner._run_anchor(  # pylint: disable=protected-access
            object(), recipient, donor, common, common,
            object(), object(), 0,
            {'eight_interventions': {}, 'selection': {}, 'target': {}},
            'c' * 64, 'd' * 64, 0, {}, [0],
            _authorization(), _source_audit(), _same_object(),
        )
    self.assertEqual(counted.call_count, 4)
    self.assertEqual(caught.exception.failure_phase, 'record_validation')
    for call in counted.call_args_list:
      self.assertEqual(len(call.args), 3)

  def test_lossless_failed_output_preserves_nonfinite_bits(self):
    value = (jnp.asarray([np.nan, np.inf], dtype=jnp.float32),)
    encoded = runner._lossless_returned_output(value)  # pylint: disable=protected-access
    self.assertEqual(encoded['status'], 'returned')
    self.assertEqual(encoded['leaf_count'], 1)
    leaf = encoded['leaves'][0]
    self.assertEqual(leaf['encoding'], 'base64_c_order_raw_bytes')
    self.assertEqual(leaf['size_bytes'], 8)
    self.assertNotIn('NaN', json.dumps(encoded))

  def test_failed_current_global_arithmetic_for_d_zero_to_four(self):
    for d_completed in range(5):
      with self.subTest(d=d_completed), tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        stop = runner.CurrentRecordStop(
            failure_phase=(
                'record_validation' if d_completed == 4 else 'model_dispatch'
            ),
            failed_or_next_call_role=(None if d_completed == 4 else CALL_ROLE(d_completed)),
            returned_outputs=[None] * 4,
            started=[{'path': f's{index}', 'sha256': 'a' * 64, 'size_bytes': 1}
                     for index in range(d_completed + (d_completed < 4))],
            completed=[{'path': f'c{index}', 'sha256': 'b' * 64, 'size_bytes': 1}
                       for index in range(d_completed)],
            original_error=RuntimeError('x'),
        )
        module = types.SimpleNamespace(record={
            'external_freeze_authorization': _authorization()
        })
        with self._publication_root(directory), mock.patch.dict(
            sys.modules, {runner.ATTESTATION_MODULE: module}
        ):
          binding = runner._write_failed_current(  # pylint: disable=protected-access
              stop, execution_index=2, recipient=_Case(), anchor_id=127,
              source_input_audit_binding={'sha256': 'c' * 64, 'size_bytes': 1},
              same_object_attestation_binding={'sha256': 'd' * 64, 'size_bytes': 1},
          )
        record = json.loads((output / binding['path']).read_text())
        self.assertEqual(record['completed_count'], 8 + d_completed)
        self.assertEqual(
            record['started_count'], 8 + d_completed + (d_completed < 4)
        )

  def test_manifest_uses_exact_run_relative_maps(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory)
      module = types.SimpleNamespace(record={
          'external_freeze_authorization': _authorization()
      })
      with mock.patch.object(runner, 'OUTPUT_DIR', output), mock.patch.dict(
          sys.modules, {runner.ATTESTATION_MODULE: module}
      ):
        manifest = runner._raw_manifest(  # pylint: disable=protected-access
            [], source_input_audit_binding={'sha256': 'a' * 64, 'size_bytes': 1}
        )
      self.assertEqual(set(manifest), set(
          runner.bootstrap.RAW_MANIFEST_CONTRACT['keys']
      ))
      self.assertEqual(manifest['status'], 'empty_controlled_stop')
      self.assertEqual(manifest['artifact_tree_sha256'], hashlib.sha256(b'').hexdigest())

  def test_publication_terminal_excludes_failed_entries_and_keeps_dirs(self):
    creators = {
        'regular': lambda path: path.write_bytes(b'preexisting'),
        'symlink': lambda path: path.symlink_to('nowhere'),
        'fifo': lambda path: os.mkfifo(path),
        'directory': lambda path: path.mkdir(mode=0o700),
    }
    for kind, create in creators.items():
      with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
        with self._publication_root(directory) as output:
          blocked = output / 'blocked.json'
          create(blocked)
          with self.assertRaises(runner.bootstrap.PublicationError) as caught:
            runner._publish_new_bytes(  # pylint: disable=protected-access
                blocked, b'new'
            )
          module = types.SimpleNamespace(record={
              'external_freeze_authorization': _authorization(),
              'start': _minimal_start(),
          })
          with mock.patch.dict(sys.modules, {runner.ATTESTATION_MODULE: module}):
            runner._write_terminal_failure(  # pylint: disable=protected-access
                caught.exception,
                completed_record_count=0,
                source_input_audit=_source_audit(),
                same_object_attestation=None,
                phase_state=runner._phase_state(  # pylint: disable=protected-access
                    preflight_passed=True, start_persisted=True
                ),
                failed_current_binding=None,
            )
          terminal = json.loads(
              (output / 'TERMINAL_FAILURE.json').read_text()
          )
          self.assertEqual(terminal['status'], 'incomplete_publication_failure')
          self.assertIn('blocked.json', terminal['preexisting_entry_states'])
          self.assertNotIn(
              'blocked.json',
              terminal['preterminal_tree_binding']['file_bindings'],
          )
          if kind == 'directory':
            self.assertIn(
                'blocked.json',
                terminal['preterminal_tree_binding']['directory_paths'],
            )

  def test_publication_terminal_binds_orphan_and_uncertain_final(self):
    with tempfile.TemporaryDirectory() as directory, self._publication_root(
        directory
    ) as output:
      with mock.patch.object(runner.bootstrap.os, 'write', return_value=0):
        with self.assertRaises(runner.bootstrap.PublicationError) as caught:
          runner._publish_new_bytes(output / 'orphan.json', b'x')  # pylint: disable=protected-access
      module = types.SimpleNamespace(record={
          'external_freeze_authorization': _authorization(),
          'start': _minimal_start(),
      })
      with mock.patch.dict(sys.modules, {runner.ATTESTATION_MODULE: module}):
        runner._write_terminal_failure(  # pylint: disable=protected-access
            caught.exception, completed_record_count=0,
            source_input_audit=_source_audit(), same_object_attestation=None,
            phase_state=runner._phase_state(  # pylint: disable=protected-access
                preflight_passed=True, start_persisted=True
            ), failed_current_binding=None,
        )
      terminal = json.loads((output / 'TERMINAL_FAILURE.json').read_text())
      self.assertEqual(len(terminal['temporary_orphan_bindings']), 1)
      self.assertFalse(
          set(terminal['temporary_orphan_bindings'])
          & set(terminal['preterminal_tree_binding']['file_bindings'])
      )

    with tempfile.TemporaryDirectory() as directory, self._publication_root(
        directory
    ) as output:
      handle = runner.bootstrap.PublicationHandle(
          'model_run', 'uncertain.json', 'uncertain'
      )
      handle.write(b'x')
      real_fsync = runner.bootstrap.os.fsync

      def fail_parent_fsync(fd):
        if fd == handle.parent_fd and handle.rename_succeeded:
          raise OSError(errno.EIO, 'injected')
        return real_fsync(fd)

      with mock.patch.object(
          runner.bootstrap.os, 'fsync', side_effect=fail_parent_fsync
      ):
        with self.assertRaises(runner.bootstrap.PublicationError) as caught:
          handle.finalize(b'x')
      module = types.SimpleNamespace(record={
          'external_freeze_authorization': _authorization(),
          'start': _minimal_start(),
      })
      with mock.patch.dict(sys.modules, {runner.ATTESTATION_MODULE: module}):
        runner._write_terminal_failure(  # pylint: disable=protected-access
            caught.exception, completed_record_count=0,
            source_input_audit=_source_audit(), same_object_attestation=None,
            phase_state=runner._phase_state(  # pylint: disable=protected-access
                preflight_passed=True, start_persisted=True
            ), failed_current_binding=None,
        )
      terminal = json.loads((output / 'TERMINAL_FAILURE.json').read_text())
      self.assertEqual(
          set(terminal['durability_uncertain_final_bindings']),
          {'uncertain.json'},
      )
      self.assertNotIn(
          'uncertain.json', terminal['preterminal_tree_binding']['file_bindings']
      )

  def test_phase_and_terminal_key_contracts_match_implementation(self):
    self.assertEqual(
        set(runner._PHASE_STATE_KEYS),  # pylint: disable=protected-access
        set(runner.bootstrap.TERMINAL_CONTRACT['phase_state_keys']),
    )
    source = Path(runner.__file__).read_text(encoding='utf-8')
    self.assertNotIn("'status': 'controlled_stop'", source)
    self.assertNotIn("'stop_reason': 'ood_tooling_failure'", source)
    self.assertNotIn("'artifact_count': len(paths)", source)

  def test_wrapper_and_launcher_freeze_lifecycle(self):
    shell = (_HERE / 'run_encoder_skip_ood_sidecar_v3_3_4_4.sh').read_text()
    launcher = (_HERE / 'launch_encoder_skip_ood_sidecar_v3_3_4_4.py').read_text()
    self.assertIn('--dry-run is the sole option', shell)
    self.assertNotIn('prepare_cache_root', shell)
    for token in (
        'gate_a = bootstrap.validate_freeze()',
        "run_device_preflight_v3_3_4_4.py'), '--run'",
        "'ATTEMPT_STARTED.json'",
        'allow_started_output=True',
        'runpy.run_path',
    ):
      self.assertIn(token, launcher)
    self.assertLess(launcher.index('gate_a ='), launcher.index("'--run'"))
    self.assertLess(launcher.index("'ATTEMPT_STARTED.json'"),
                    launcher.index('allow_started_output=True'))

  def test_no_forbidden_scientific_paths(self):
    source = Path(runner.__file__).read_text(encoding='utf-8')
    self.assertEqual(source.count(
        'create_splice_classification_logit_margin_eight_row_superset_graph_apply'
    ), 1)
    self.assertNotIn('create_splice_classification_logit_margin_superset_graph_apply(', source)
    self.assertNotIn('shapley', source.lower().replace('shapley_or_nomination_computed', ''))


def CALL_ROLE(d_completed: int) -> str:
  return runner.CALL_ROLES[min(d_completed, 3)]


if __name__ == '__main__':
  unittest.main()
