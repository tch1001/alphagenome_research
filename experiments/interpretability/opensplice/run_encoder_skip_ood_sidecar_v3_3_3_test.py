#!/usr/bin/env python3
"""CPU-only tests for the OpenSplice v3.3.3 OOD sidecar."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

import run_encoder_skip_factorial_v3_3_test as fixtures
import run_encoder_skip_ood_sidecar_v3_3_3 as runner


def _source_audit() -> dict[str, object]:
  return {
      'bootstrap_sources_and_prior_trees_exact': True,
      'tracked_head_and_frozen_inventory_exact': True,
      'external_device_runtime_environment_exact': True,
      'same_process_device_runtime_environment_exact': True,
      'checkpoint_exact': True,
      'reference_object_and_sequences_exact': True,
      'protobuf_binding_exact': True,
      'three_import_inventories_stable_exact': True,
  }


def _cache_attestation(role: str, root: Path) -> dict[str, object]:
  return {
      'denied_exact_names': list(
          runner.bootstrap.DENIED_CACHE_ENVIRONMENT_NAMES
      ),
      'denied_prefixes': list(
          runner.bootstrap.DENIED_CACHE_ENVIRONMENT_PREFIXES
      ),
      'present_forbidden_names': [],
      'autotune_load_dump_cache_inputs_absent': True,
      'kernel_cache_inputs_absent': True,
      'persistent_compilation_cache_inputs_absent': True,
      'cuda_kernel_cache_disabled': True,
      'cache_role': role,
      'cache_root': str(root.resolve()),
      'triton_cache_dir': str(root.resolve() / 'triton'),
      'xdg_cache_home': str(root.resolve() / 'xdg'),
      'pre_import_file_count': 0,
      'pre_import_tree_sha256': runner.hashlib.sha256(b'').hexdigest(),
      'default_user_cache_paths_eligible': False,
  }


def _frozen_v332_compiler() -> dict[str, object]:
  path = (
      runner.bootstrap.V3_3_2_RUN_DIR
      / 'compiler/eight_row/COMPILER_PROVENANCE.json'
  )
  return json.loads(path.read_text(encoding='utf-8'))


def _complete_compiler_fixture(cache_root: Path) -> dict[str, object]:
  return {
      'executable_fingerprint': 'e' * 64,
      'source_program_gate': {
          'source_program_exact': True,
          'stablehlo_exact': True,
          'pre_backend_hlo_exact': True,
          'program_signatures_exact': True,
          'entry_abi_exact': True,
          'source_runtime_device_toolchain_checkpoint_reference_exact': True,
          'same_lowered_compiled_object': True,
      },
      'backend_diagnostics': {
          'descriptive_only_not_an_equality_gate': True
      },
      'diagnostic_comparisons': {
          'compiled_backend_differences_are_diagnostic_only': True
      },
      'kernel_cache_provenance': {
          'pre_import': _cache_attestation('model', cache_root),
          'post_compile': runner._cache_tree_binding(cache_root),  # pylint: disable=protected-access
          'default_user_cache_paths_eligible': False,
          'cache_outputs_are_diagnostic_only': True,
      },
  }


def _trace_with_active_rows(trace, *, row2: float, row4: float):
  natural = np.asarray(trace.stage_a.natural_final_embeddings).copy()
  natural[2] = row2
  natural[4] = row4
  natural = jnp.asarray(natural, jnp.bfloat16)
  return dataclasses.replace(
      trace,
      stage_a=dataclasses.replace(
          trace.stage_a,
          natural_final_embeddings=natural,
          effective_final_embeddings=natural,
      ),
  )


def _anchor_inputs(anchor_id: int, *, active_rows_differ: bool = True):
  if anchor_id == 0:
    intended_values = unrelated_values = (1, 3, 3, 3, 1, 1, 7, 9)
  elif anchor_id == 255:
    intended_values = (1, 3, 1, 3, 3, 1, 7, 9)
    unrelated_values = (1, 3, 7, 3, 9, 1, 7, 9)
  else:
    intended_values = (1, 3, 5, 3, 6, 1, 7, 9)
    unrelated_values = (
        (1, 3, 7, 3, 9, 1, 7, 9)
        if active_rows_differ else intended_values
    )
  intended_evidence = fixtures._evidence(intended_values)  # pylint: disable=protected-access
  unrelated_evidence = fixtures._evidence(unrelated_values)  # pylint: disable=protected-access
  intended_trace = fixtures._trace(8)  # pylint: disable=protected-access
  unrelated_trace = fixtures._trace(8)  # pylint: disable=protected-access
  if anchor_id != 0 and active_rows_differ:
    unrelated_trace = _trace_with_active_rows(
        unrelated_trace, row2=6.0, row4=7.0
    )
  selection = fixtures._selection()  # pylint: disable=protected-access
  intended_interventions = runner.v33.eight_row_interventions(
      selection, anchor_id, unrelated=False
  )
  unrelated_interventions = runner.v33.eight_row_interventions(
      selection, anchor_id, unrelated=True
  )
  return (
      (intended_evidence, intended_trace),
      (intended_evidence, intended_trace),
      (unrelated_evidence, unrelated_trace),
      (unrelated_evidence, unrelated_trace),
      anchor_id,
      intended_interventions,
      unrelated_interventions,
  )


class EncoderSkipOodSidecarV333Test(unittest.TestCase):

  def test_external_and_model_cache_roles_use_distinct_roots(self):
    external = _cache_attestation(
        'external_preflight', runner.bootstrap.PREFLIGHT_KERNEL_CACHE_DIR
    )
    model = _cache_attestation(
        'model', runner.bootstrap.MODEL_KERNEL_CACHE_DIR
    )
    observed = runner.validate_cache_role_transition(external, model)
    self.assertTrue(observed['roles_and_roots_distinct'])
    self.assertTrue(observed['shared_policy_exact'])
    self.assertEqual(observed['contract'], runner.bootstrap.CACHE_ISOLATION_CONTRACT)
    self.assertFalse(observed['cache_output_equality_is_a_gate'])
    self.assertNotEqual(
        observed['external_preflight']['cache_root'],
        observed['model']['cache_root'],
    )

  def test_external_and_model_cache_role_swaps_fail_closed(self):
    external = _cache_attestation(
        'external_preflight', runner.bootstrap.PREFLIGHT_KERNEL_CACHE_DIR
    )
    model = _cache_attestation(
        'model', runner.bootstrap.MODEL_KERNEL_CACHE_DIR
    )
    with self.assertRaisesRegex(ValueError, 'external_preflight cache role'):
      runner.validate_cache_role_transition(model, external)
    same_root_model = dict(model)
    same_root_model.update({
        'cache_root': external['cache_root'],
        'triton_cache_dir': external['triton_cache_dir'],
        'xdg_cache_home': external['xdg_cache_home'],
    })
    with self.assertRaisesRegex(ValueError, 'model cache root'):
      runner.validate_cache_role_transition(external, same_root_model)

  def test_external_and_model_shared_cache_policy_tamper_fails_closed(self):
    external = _cache_attestation(
        'external_preflight', runner.bootstrap.PREFLIGHT_KERNEL_CACHE_DIR
    )
    model = _cache_attestation(
        'model', runner.bootstrap.MODEL_KERNEL_CACHE_DIR
    )
    model['cuda_kernel_cache_disabled'] = False
    with self.assertRaisesRegex(ValueError, 'model cache policy'):
      runner.validate_cache_role_transition(external, model)

  def test_lazy_cache_outputs_are_terminal_diagnostics_not_a_gate(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      (root / 'triton').mkdir()
      (root / 'xdg').mkdir()
      pre_import = _cache_attestation('model', root)
      historical = runner._cache_tree_binding(root)  # pylint: disable=protected-access
      (root / 'triton/lazy-kernel.bin').write_bytes(b'lazy output')
      final = runner._final_model_cache_binding({  # pylint: disable=protected-access
          'kernel_cache_provenance': {
              'pre_import': pre_import,
              'post_compile': historical,
          }
      })
      self.assertFalse(final['historical_to_terminal_tree_exact'])
      self.assertFalse(final['historical_to_terminal_equality_is_a_gate'])
      self.assertTrue(final['cache_outputs_are_diagnostic_only'])

  def test_preflight_directory_is_exactly_one_attempt(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for name in (
          '.allocation.lock',
          '.preflight_0000.reserved',
          'preflight_0000.json',
          'preflight_0000.stdout.log',
          'preflight_0000.stderr.log',
      ):
        (root / name).write_bytes(b'')
      with mock.patch.object(runner, 'PREFLIGHT_DIR', root):
        binding = runner._strict_preflight_directory(  # pylint: disable=protected-access
            root / 'preflight_0000.json'
        )
        self.assertEqual(binding['file_count'], 5)
        self.assertTrue(binding['sole_preflight_attempt_exact'])
        (root / 'preflight_0001.json').write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'membership changed'):
          runner._strict_preflight_directory(  # pylint: disable=protected-access
              root / 'preflight_0000.json'
          )

  def test_preflight_directory_rejects_nonzero_selected_attempt(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      with mock.patch.object(runner, 'PREFLIGHT_DIR', root):
        with self.assertRaisesRegex(ValueError, 'sole preflight_0000'):
          runner._strict_preflight_directory(  # pylint: disable=protected-access
              root / 'preflight_0001.json'
          )

  def test_source_program_gate_terms_fail_independently(self):
    compiler = _frozen_v332_compiler()
    signatures = compiler['program_signatures']
    observed = dict(runner.SOURCE_PROGRAM_CONTRACT)
    passing = runner.evaluate_source_program_gate(
        observed, signatures, compiler, _source_audit()
    )
    self.assertTrue(passing['source_program_exact'])

    mutations = {
        'stablehlo_sha256': '0' * 64,
        'stablehlo_size_bytes': observed['stablehlo_size_bytes'] + 1,
        'pre_backend_hlo_sha256': '1' * 64,
        'pre_backend_hlo_size_bytes': (
            observed['pre_backend_hlo_size_bytes'] + 1
        ),
        'program_signatures_sha256': '2' * 64,
        'entry_abi_sha256': '3' * 64,
    }
    for name, value in mutations.items():
      with self.subTest(name=name):
        changed = dict(observed)
        changed[name] = value
        gate = runner.evaluate_source_program_gate(
            changed, signatures, compiler, _source_audit()
        )
        self.assertFalse(gate['source_program_exact'])

    literal_mutations = {}
    literal_mutations['shape'] = json.loads(json.dumps(signatures))
    literal_mutations['shape']['selection']['leaves'][0]['shape'][0] += 1
    literal_mutations['dtype'] = json.loads(json.dumps(signatures))
    literal_mutations['dtype']['target']['leaves'][0]['dtype'] = 'int64'
    literal_mutations['treedef'] = json.loads(json.dumps(signatures))
    literal_mutations['treedef']['target']['treedef'] += ' changed'
    literal_mutations['leaf_order'] = json.loads(json.dumps(signatures))
    leaves = literal_mutations['leaf_order']['selection']['leaves']
    leaves[0], leaves[1] = leaves[1], leaves[0]
    for name, literal in literal_mutations.items():
      with self.subTest(signature_mutation=name):
        gate = runner.evaluate_source_program_gate(
            observed, literal, compiler, _source_audit()
        )
        self.assertFalse(gate['program_signatures_exact'])
        self.assertFalse(gate['source_program_exact'])

    for name in _source_audit():
      with self.subTest(source_input=name):
        audit = _source_audit()
        audit[name] = False
        gate = runner.evaluate_source_program_gate(
            observed, signatures, compiler, audit
        )
        self.assertFalse(gate[
            'source_runtime_device_toolchain_checkpoint_reference_exact'
        ])
        self.assertFalse(gate['source_program_exact'])

  def test_compiled_backend_difference_does_not_fail_source_program(self):
    compiler = _frozen_v332_compiler()
    stable_path = runner.bootstrap.V3_3_2_RUN_DIR / (
        'compiler/eight_row/graph.stablehlo.mlir'
    )
    hlo_path = runner.bootstrap.V3_3_2_RUN_DIR / (
        'compiler/eight_row/graph.pre_backend.hlo.txt'
    )
    compiled_path = runner.bootstrap.V3_3_2_RUN_DIR / (
        'compiler/eight_row/graph.compiled.hlo.txt'
    )
    stable = stable_path.read_text(encoding='utf-8')
    hlo = hlo_path.read_text(encoding='utf-8')
    entry = compiled_path.read_text(encoding='utf-8').splitlines()[0]

    class FakeHlo:
      def as_hlo_text(self):
        return hlo

    class FakeLowered:
      def compiler_ir(self, *, dialect):
        return stable if dialect == 'stablehlo' else FakeHlo()

    class FakeCompiled:
      def as_text(self):
        return entry + '\n%diagnostic_only_backend_change () {\n}\n'

    original_v3_3 = json.loads(
        (runner.ORIGINAL_RUN_DIR / 'compiler/eight_row/COMPILER_PROVENANCE.json')
        .read_text(encoding='utf-8')
    )
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      cache_root = root / 'cache'
      (cache_root / 'triton').mkdir(parents=True)
      (cache_root / 'xdg').mkdir()
      cache = _cache_attestation('model', cache_root)
      with mock.patch.object(runner, 'OUTPUT_DIR', root):
        record = runner._compiler_artifacts(  # pylint: disable=protected-access
            FakeLowered(), FakeCompiled(), 1.0, original_v3_3, compiler,
            compiler['program_signatures'], source_input_audit=_source_audit(),
            kernel_cache_preimport_attestation=cache,
        )
    self.assertTrue(record['source_program_gate']['source_program_exact'])
    self.assertFalse(record['diagnostic_comparisons']['v3_3_2'][
        'artifacts'
    ]['compiled_hlo']['sha256_exact'])
    self.assertTrue(record['diagnostic_comparisons'][
        'compiled_backend_differences_are_diagnostic_only'
    ])
    self.assertFalse(record['kernel_cache_provenance'][
        'default_user_cache_paths_eligible'
    ])

  def test_entry_abi_and_nested_triton_diagnostics_are_exact(self):
    compiled_path = runner.bootstrap.V3_3_2_RUN_DIR / (
        'compiler/eight_row/graph.compiled.hlo.txt'
    )
    entry = compiled_path.read_text(encoding='utf-8').splitlines()[0]
    binding = runner._entry_abi_binding(entry)  # pylint: disable=protected-access
    self.assertEqual(
        binding['normalized_line_sha256'],
        runner.SOURCE_PROGRAM_CONTRACT['entry_abi_sha256'],
    )
    line = (
        '  %x = f32[] fusion(), kind=kCustom, '
        'backend_config={"fusion_backend_config":{'
        '"block_level_fusion_config":{"num_ctas":1,"num_stages":3,'
        '"num_warps":"4","output_tiles":[{"sizes":["16","32"]}]},'
        '"kind":"__triton"}}'
    )
    diagnostics = runner._backend_diagnostics(  # pylint: disable=protected-access
        'HloModule x\n%f () {\n' + line + '\n}\n'
    )
    block = diagnostics['triton_configurations'][0][
        'block_level_fusion_config'
    ]
    self.assertEqual(block['output_tiles'], [{'sizes': ['16', '32']}])
    self.assertEqual(block['num_stages'], 3)

  def test_exact_failed_and_archived_prerequisites_revalidate(self):
    self.assertEqual(
        runner.bootstrap.validate_v3_3_2_1_failure(),
        runner.bootstrap.EXPECTED_V3_3_2_1_FAILURE_STATUS,
    )
    self.assertEqual(
        runner.bootstrap.validate_v3_3_2_2_archive(),
        runner.bootstrap.EXPECTED_V3_3_2_2_ARCHIVE_STATUS,
    )

  def test_order_counts_and_dry_run_are_exact(self):
    cases = tuple(SimpleNamespace(order=index) for index in range(20))
    order = runner.sidecar_execution_order(cases)
    self.assertEqual(len(order), 80)
    self.assertEqual(len(set(order)), 80)
    self.assertEqual(order[:5], (
        (0, 0), (0, 127), (0, 128), (0, 255), (1, 0)
    ))
    self.assertEqual(order[-1], (19, 255))
    plan = runner.build_dry_run_plan(
        cases, max_variants=1, max_anchors=2
    )
    self.assertEqual(plan['ood_record_count'], 80)
    self.assertEqual(plan['model_apply_count'], 320)
    self.assertEqual(plan['eight_row_compile_count'], 1)
    self.assertEqual(plan['six_row_compile_count'], 0)
    self.assertEqual(plan['identity_rerun_count'], 0)
    self.assertEqual(plan['main_cube_rerun_count'], 0)
    self.assertEqual(plan['old_ood_records_reused'], 0)
    self.assertEqual(plan['confirmation_model_calls'], 0)
    self.assertEqual(
        plan['analysis_attempt_dir'], str(runner.ANALYSIS_ATTEMPT_DIR)
    )

  def test_all_anchor_interventions_share_fixed_pytree_and_maps(self):
    selection = fixtures._selection()  # pylint: disable=protected-access
    reference = None
    for anchor_id in runner.ANCHOR_IDS:
      for unrelated, donors in (
          (False, runner.v33.EIGHT_INTENDED_DONOR_ROWS),
          (True, runner.v33.EIGHT_UNRELATED_DONOR_ROWS),
      ):
        interventions = runner.v33.eight_row_interventions(
            selection, anchor_id, unrelated=unrelated
        )
        tree = jax.tree_util.tree_structure(interventions)
        reference = reference or tree
        self.assertEqual(tree, reference)
        runner.v33._assert_runtime_transfer_contract(  # pylint: disable=protected-access
            interventions,
            anchor_id,
            batch_size=8,
            donor_rows=donors,
            identity_rows=runner.v33.EIGHT_IDENTITY_ROWS,
        )

  def test_corrected_validator_accepts_active_row_difference(self):
    checks = runner.validate_ood_sidecar_anchor(
        *_anchor_inputs(127, active_rows_differ=True)
    )
    self.assertTrue(checks[
        'natural_final_invariant_rows_exact_between_calls'
    ])
    self.assertEqual(
        checks['active_rows_cross_call_equality_not_required'], [2, 4]
    )
    self.assertTrue(checks['active_rows_forced_difference_not_required'])

  def test_corrected_validator_does_not_force_active_row_difference(self):
    checks = runner.validate_ood_sidecar_anchor(
        *_anchor_inputs(128, active_rows_differ=False)
    )
    self.assertTrue(checks['passed'])

  def test_invariant_row_and_upstream_drift_fail_closed(self):
    for invariant_row in runner.INVARIANT_ROWS:
      with self.subTest(invariant_row=invariant_row):
        args = list(_anchor_inputs(127, active_rows_differ=True))
        unrelated = args[2]
        natural = np.asarray(
            unrelated[1].stage_a.natural_final_embeddings
        ).copy()
        natural[invariant_row] = np.float32(100 + invariant_row)
        natural = jnp.asarray(natural, jnp.bfloat16)
        drifted_trace = dataclasses.replace(
            unrelated[1],
            stage_a=dataclasses.replace(
                unrelated[1].stage_a,
                natural_final_embeddings=natural,
                effective_final_embeddings=natural,
            ),
        )
        args[2] = (unrelated[0], drifted_trace)
        args[3] = (unrelated[0], drifted_trace)
        with self.assertRaisesRegex(
            ValueError, f'invariant row {invariant_row}'
        ):
          runner.validate_ood_sidecar_anchor(*args)

    for fingerprint_field in (
        'transformer_output_natural_fingerprint',
        'encoder_skips_natural_fingerprints',
    ):
      with self.subTest(fingerprint_field=fingerprint_field):
        args = list(_anchor_inputs(127, active_rows_differ=True))
        unrelated = args[2]
        branch = unrelated[1].stage_a
        value = getattr(branch, fingerprint_field)
        index = (0, 0) if value.ndim == 2 else (0, 0, 0)
        bad_fingerprint = value.at[index].set(1)
        drifted_trace = dataclasses.replace(
            unrelated[1],
            stage_a=dataclasses.replace(
                branch, **{fingerprint_field: bad_fingerprint}
            ),
        )
        args[2] = (unrelated[0], drifted_trace)
        args[3] = (unrelated[0], drifted_trace)
        with self.assertRaisesRegex(ValueError, 'upstream natural route'):
          runner.validate_ood_sidecar_anchor(*args)

    for natural_name, effective_name in runner.v32._TRANSFORMER_PAIRS:  # pylint: disable=protected-access
      with self.subTest(transformer_field=natural_name):
        args = list(_anchor_inputs(127, active_rows_differ=True))
        unrelated = args[2]
        transformer = unrelated[1].transformer
        natural = getattr(transformer, natural_name)
        changed = natural.at[(0,) * natural.ndim].set(1)
        drifted_trace = dataclasses.replace(
            unrelated[1],
            transformer=dataclasses.replace(
                transformer,
                **{natural_name: changed, effective_name: changed},
            ),
        )
        args[2] = (unrelated[0], drifted_trace)
        args[3] = (unrelated[0], drifted_trace)
        with self.assertRaisesRegex(ValueError, 'upstream transformer seam'):
          runner.validate_ood_sidecar_anchor(*args)

  def test_id0_and_id255_closures_remain_strong(self):
    id0 = runner.validate_ood_sidecar_anchor(*_anchor_inputs(0))
    self.assertTrue(id0['id0_all8_natural_final_exact_between_calls'])
    self.assertTrue(id0['id0_all8_endpoint_exact_between_calls'])
    id255 = runner.validate_ood_sidecar_anchor(*_anchor_inputs(255))
    self.assertTrue(id255['id255_intended_endpoint_closure_exact'])
    self.assertTrue(id255['id255_unrelated_endpoint_closure_exact'])

    args = list(_anchor_inputs(0))
    intended = args[0]
    unrelated = args[2]
    bad_intended = _trace_with_active_rows(
        intended[1], row2=5.0, row4=6.0
    )
    bad_unrelated = _trace_with_active_rows(
        unrelated[1], row2=5.0, row4=6.0
    )
    args[0] = (intended[0], bad_intended)
    args[1] = (intended[0], bad_intended)
    args[2] = (unrelated[0], bad_unrelated)
    args[3] = (unrelated[0], bad_unrelated)
    with self.assertRaisesRegex(ValueError, 'natural-final recipient'):
      runner.validate_ood_sidecar_anchor(*args)

    args = list(_anchor_inputs(255))
    intended_bad = fixtures._evidence((1, 3, 2, 3, 3, 1, 7, 9))  # pylint: disable=protected-access
    args[0] = (intended_bad, args[0][1])
    args[1] = (intended_bad, args[1][1])
    with self.assertRaisesRegex(ValueError, 'Endpoint readout differs'):
      runner.validate_ood_sidecar_anchor(*args)

    args = list(_anchor_inputs(255))
    unrelated_bad = fixtures._evidence((1, 3, 8, 3, 9, 1, 7, 9))  # pylint: disable=protected-access
    args[2] = (unrelated_bad, args[2][1])
    args[3] = (unrelated_bad, args[3][1])
    with self.assertRaisesRegex(ValueError, 'Endpoint readout differs'):
      runner.validate_ood_sidecar_anchor(*args)

  def test_repeat_donor_and_final_seam_tampering_fails(self):
    args = list(_anchor_inputs(255))
    changed_repeat = fixtures._evidence((1, 3, 1, 3, 3, 1, 7, 10))  # pylint: disable=protected-access
    args[1] = (changed_repeat, args[1][1])
    with self.assertRaisesRegex(ValueError, 'repeat'):
      runner.validate_ood_sidecar_anchor(*args)

    args = list(_anchor_inputs(255))
    corrupted = dataclasses.replace(
        args[5],
        stage_a=dataclasses.replace(
            args[5].stage_a,
            encoder_skips=dataclasses.replace(
                args[5].stage_a.encoder_skips,
                donor_batch_indices=(
                    args[5].stage_a.encoder_skips.donor_batch_indices
                    .at[0, 2].set(5)
                ),
            ),
        ),
    )
    args[5] = corrupted
    with self.assertRaisesRegex(ValueError, 'donor map'):
      runner.validate_ood_sidecar_anchor(*args)

    args = list(_anchor_inputs(255))
    intended = args[0]
    changed_effective = intended[1].stage_a.effective_final_embeddings.at[0, 0, 0].set(5)
    bad_trace = dataclasses.replace(
        intended[1],
        stage_a=dataclasses.replace(
            intended[1].stage_a,
            effective_final_embeddings=changed_effective,
        ),
    )
    args[0] = (intended[0], bad_trace)
    args[1] = (intended[0], bad_trace)
    with self.assertRaisesRegex(ValueError, 'final seam'):
      runner.validate_ood_sidecar_anchor(*args)

  def test_compact_rowwise_fingerprint_is_exact_and_row_local(self):
    values = np.arange(8 * 2 * 3, dtype=np.uint16).reshape(8, 2, 3)
    first = runner.compact_rowwise_fingerprint(values)
    second = runner.compact_rowwise_fingerprint(values.copy())
    self.assertEqual(first, second)
    changed = values.copy()
    changed[2, 0, 0] += 1
    third = runner.compact_rowwise_fingerprint(changed)
    for row in range(8):
      equal = first['rows'][row]['sha256'] == third['rows'][row]['sha256']
      self.assertEqual(equal, row != 2)

  def test_original_binding_uses_manifest_hash_without_opening_json(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      relative = 'raw/identity/000_variant.json'
      path = root / relative
      path.parent.mkdir(parents=True)
      path.write_bytes(b'opaque scientific bytes')
      digest = runner._sha256(path)  # pylint: disable=protected-access
      case = SimpleNamespace(order=0, variant_id='variant')
      with mock.patch.object(runner, 'ORIGINAL_RUN_DIR', root):
        binding = runner.original_artifact_binding(
            {'artifact_sha256': {relative: digest}}, case, 'identity'
        )
        self.assertEqual(binding, {'path': relative, 'sha256': digest})
        path.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'changed'):
          runner.original_artifact_binding(
              {'artifact_sha256': {relative: digest}}, case, 'identity'
          )

  def test_runner_source_has_no_six_row_or_cube_apply_path(self):
    text = Path(runner.__file__).read_text(encoding='utf-8')
    self.assertNotIn(
        'create_splice_classification_logit_margin_superset_graph_apply(', text
    )
    self.assertNotIn('def _run_identity(', text)
    self.assertNotIn('def _run_coalition(', text)
    self.assertEqual(
        text.count(
            'create_splice_classification_logit_margin_eight_row_superset_graph_apply('
        ),
        1,
    )
    start_index = text.index('_write_new(START_PATH, start)')
    try_index = text.index('  try:', start_index)
    protobuf_index = text.index("'PROTOBUF_PROVENANCE.json'", start_index)
    self.assertLess(start_index, try_index)
    self.assertLess(try_index, protobuf_index)

  def test_apply_counter_increments_before_failed_dispatch(self):
    counter = [0]
    with mock.patch.object(
        runner.v32, '_timed_apply', side_effect=RuntimeError('dispatch failed')
    ):
      with self.assertRaisesRegex(runner.CountedApplyError, 'dispatch failed'):
        runner._counted_apply(  # pylint: disable=protected-access
            object(), (), counter, call_label='intended'
        )
    self.assertEqual(counter, [1])

  def test_four_completed_calls_persist_postprocessing_failure(self):
    selection = fixtures._selection()  # pylint: disable=protected-access
    trace = fixtures._trace(8)  # pylint: disable=protected-access
    evidence = fixtures._evidence((1, 3, 3, 3, 1, 1, 7, 9))  # pylint: disable=protected-access
    calls = [((evidence, trace), 0.1)] * 4
    call_iterator = iter(calls)
    def counted_apply(_compiled, _args, apply_counter, *, call_label):
      del call_label
      apply_counter[0] += 1
      return next(call_iterator)
    recipient = SimpleNamespace(order=0, variant_id='recipient')
    donor = SimpleNamespace(order=10, variant_id='donor')
    common = (
        np.zeros((6, 2, 4), np.float32), selection, object(), 'a' * 64
    )
    donor_common = (
        np.zeros((6, 2, 4), np.float32), selection, object(), 'b' * 64
    )
    signatures = {
        'selection': runner.v32.pytree_signature(selection),
        'target': None,
        'eight_interventions': runner.v32.pytree_signature(
            runner.v33.eight_row_interventions(
                selection, 0, unrelated=False
            )
        ),
    }
    with tempfile.TemporaryDirectory() as directory:
      counter = [0]
      with (
          mock.patch.object(runner, 'OUTPUT_DIR', Path(directory)),
          mock.patch.object(runner, '_counted_apply', side_effect=counted_apply),
          mock.patch.object(
              runner.v32, 'assert_same_program_signature', return_value=None
          ),
          mock.patch.object(
              runner, 'original_artifact_binding',
              side_effect=ValueError('linked artifact tampered'),
          ),
          mock.patch.object(runner, '_case_record', return_value={'order': 0}),
      ):
        result = runner._run_anchor(  # pylint: disable=protected-access
            object(), recipient, donor, common, donor_common,
            object(), object(), 0, signatures, 'f' * 64,
            'e' * 64, 0, {'artifact_sha256': {}}, counter,
        )
        self.assertEqual(result['status'], 'invalid')
        self.assertEqual(result['failure']['message'], 'linked artifact tampered')
        self.assertEqual(counter, [4])
        self.assertEqual(len(list(Path(directory).rglob('*.json'))), 1)

  def test_wrapper_rejects_caller_preflight_override_before_bootstrap(self):
    wrapper = Path(runner.__file__).with_name(
        'run_encoder_skip_ood_sidecar_v3_3_3.sh'
    )
    result = subprocess.run(
        ('bash', str(wrapper), '--dry-run', '--successful-preflight=/tmp/x'),
        text=True,
        capture_output=True,
        check=False,
    )
    self.assertEqual(result.returncode, 64)
    self.assertIn('reserved', result.stderr)

  def test_controlled_prefix_requires_exact_incremental_apply_count(self):
    result = {
        'status': 'invalid',
        'recipient_order': 0,
        'anchor_id': 0,
        'checks': None,
    }
    compiler = {
        'executable_fingerprint': 'e' * 64,
        'source_program_gate': {'source_program_exact': True},
        'backend_diagnostics': {'descriptive_only_not_an_equality_gate': True},
        'diagnostic_comparisons': {
            'compiled_backend_differences_are_diagnostic_only': True
        },
    }
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      raw = root / 'raw/ood_anchors/000_case/000.json'
      raw.parent.mkdir(parents=True)
      raw.write_text('{}\n', encoding='utf-8')
      for name in (
          'IMPORT_PROVENANCE_PRE_MODEL.json',
          'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
          'IMPORT_PROVENANCE.json',
          'PROTOBUF_PROVENANCE.json',
      ):
        (root / name).write_text('{}\n', encoding='utf-8')
      freeze = root / 'freeze.json'
      freeze.write_text('{}\n', encoding='utf-8')
      cache_root = root / 'cache'
      (cache_root / 'triton').mkdir(parents=True)
      (cache_root / 'xdg').mkdir()
      compiler['kernel_cache_provenance'] = {
          'pre_import': _cache_attestation('model', cache_root),
          'post_compile': runner._cache_tree_binding(cache_root),  # pylint: disable=protected-access
          'default_user_cache_paths_eligible': False,
          'cache_outputs_are_diagnostic_only': True,
      }
      with (
          mock.patch.object(runner, 'OUTPUT_DIR', root),
          mock.patch.object(runner, 'FREEZE_PATH', freeze),
      ):
        with self.assertRaisesRegex(RuntimeError, 'four applies'):
          runner._write_completion(  # pylint: disable=protected-access
              stop_reason='ood_tooling_failure',
              message='test',
              results=[result],
              apply_count=3,
              compiler=compiler,
              original_run_binding={},
              v3_3_1_status={},
              v3_3_2_run_binding={},
              v3_3_2_1_failure_status={},
              v3_3_2_2_archive_status={},
          )
        completion = runner._write_completion(  # pylint: disable=protected-access
            stop_reason='ood_tooling_failure',
            message='test',
            results=[result],
            apply_count=4,
            compiler=compiler,
            original_run_binding={},
            v3_3_1_status={},
            v3_3_2_run_binding={},
            v3_3_2_1_failure_status={},
            v3_3_2_2_archive_status={},
        )
      self.assertEqual(completion['model_apply_count'], 4)
      self.assertEqual(completion['ood_anchor_record_count'], 1)
      self.assertEqual(completion['status'], 'controlled_stop')

  def test_compile_failure_is_controlled_apply_zero_without_retry(self):
    compiler_signatures = _frozen_v332_compiler()['program_signatures']
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for name in (
          'IMPORT_PROVENANCE_PRE_MODEL.json',
          'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
          'IMPORT_PROVENANCE.json',
          'PROTOBUF_PROVENANCE.json',
      ):
        (root / name).write_text('{}\n', encoding='utf-8')
      freeze = root / 'freeze.json'
      freeze.write_text('{}\n', encoding='utf-8')
      cache_root = root / 'cache'
      (cache_root / 'triton').mkdir(parents=True)
      (cache_root / 'xdg').mkdir()
      attestation = {
          'v3_3_1_status': {},
          'v3_3_2_run': {},
          'v3_3_2_1_failure_status': {},
          'v3_3_2_2_archive_status': {},
      }
      with (
          mock.patch.object(runner, 'OUTPUT_DIR', root),
          mock.patch.object(runner, 'FREEZE_PATH', freeze),
      ):
        compiler = runner._compiler_failure_artifact(  # pylint: disable=protected-access
            RuntimeError('lower failed'),
            stage='lower',
            compile_count=0,
            seconds=1.0,
            lowered=None,
            program_signatures=compiler_signatures,
            kernel_cache_preimport_attestation=_cache_attestation(
                'model', cache_root
            ),
        )
        runner._write_compiler_failure_completion(  # pylint: disable=protected-access
            compiler,
            bootstrap_attestation=attestation,
            original_run_binding={},
        )
      completion = json.loads(
          (root / 'RUN_COMPLETE.json').read_text(encoding='utf-8')
      )
      self.assertEqual(completion['status'], 'controlled_stop')
      self.assertEqual(completion['stop_reason'], 'compiler_failure')
      self.assertEqual(completion['eight_row_compile_count'], 0)
      self.assertEqual(completion['model_apply_count'], 0)
      self.assertEqual(completion['raw_manifest']['artifact_count'], 0)
      self.assertTrue(completion['no_compile_retry'])
      self.assertFalse((root / 'TERMINAL_FAILURE.json').exists())

  def test_compiler_failure_stage_and_artifacts_are_consistent(self):
    class FakeHlo:
      def as_hlo_text(self):
        return 'HloModule fake\n'

    class FakeLowered:
      def compiler_ir(self, *, dialect):
        return 'module {}' if dialect == 'stablehlo' else FakeHlo()

    signatures = _frozen_v332_compiler()['program_signatures']
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      cache_root = root / 'cache'
      (cache_root / 'triton').mkdir(parents=True)
      (cache_root / 'xdg').mkdir()
      with mock.patch.object(runner, 'OUTPUT_DIR', root):
        record = runner._compiler_failure_artifact(  # pylint: disable=protected-access
            RuntimeError('compile failed'),
            stage='compile',
            compile_count=1,
            seconds=1.0,
            lowered=FakeLowered(),
            program_signatures=signatures,
            kernel_cache_preimport_attestation=_cache_attestation(
                'model', cache_root
            ),
        )
        self.assertEqual(set(record['artifacts']), {'stablehlo', 'hlo'})
        self.assertEqual(record['compile_attempt_count'], 1)
        self.assertEqual(record['successful_compile_count'], 0)
    for stage, count, lowered in (
        ('lower', 1, None),
        ('lower', 0, FakeLowered()),
        ('compile', 0, FakeLowered()),
        ('compile', 1, None),
    ):
      with self.subTest(stage=stage, count=count, lowered=lowered):
        with self.assertRaisesRegex(ValueError, 'failure requires'):
          runner._compiler_failure_artifact(  # pylint: disable=protected-access
              RuntimeError('bad tuple'),
              stage=stage,
              compile_count=count,
              seconds=0.0,
              lowered=lowered,
              program_signatures=signatures,
              kernel_cache_preimport_attestation={},
          )

  def test_mid_record_apply_failure_is_controlled_exact_prefix(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for name in (
          'IMPORT_PROVENANCE_PRE_MODEL.json',
          'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
          'IMPORT_PROVENANCE.json',
          'PROTOBUF_PROVENANCE.json',
      ):
        (root / name).write_text('{}\n', encoding='utf-8')
      freeze = root / 'freeze.json'
      freeze.write_text('{}\n', encoding='utf-8')
      cache_root = root / 'cache'
      (cache_root / 'triton').mkdir(parents=True)
      (cache_root / 'xdg').mkdir()
      error = runner.CountedApplyError(
          'unrelated', 3, RuntimeError('device dispatch failed')
      )
      attestation = {
          'v3_3_1_status': {},
          'v3_3_2_run': {},
          'v3_3_2_1_failure_status': {},
          'v3_3_2_2_archive_status': {},
      }
      with (
          mock.patch.object(runner, 'OUTPUT_DIR', root),
          mock.patch.object(runner, 'FREEZE_PATH', freeze),
      ):
        runner._write_partial_apply_completion(  # pylint: disable=protected-access
            error,
            results=[],
            compiler=_complete_compiler_fixture(cache_root),
            execution_index=0,
            recipient_order=0,
            anchor_id=0,
            bootstrap_attestation=attestation,
            original_run_binding={},
        )
      completion = json.loads(
          (root / 'RUN_COMPLETE.json').read_text(encoding='utf-8')
      )
      self.assertEqual(completion['status'], 'controlled_stop')
      self.assertEqual(completion['stop_reason'], 'ood_tooling_failure')
      self.assertEqual(completion['model_apply_count'], 3)
      self.assertEqual(completion['ood_anchor_record_count'], 0)
      self.assertEqual(
          completion['failed_current_record'][
              'dispatched_apply_count_for_current_record'
          ],
          3,
      )
      self.assertEqual(
          completion['failed_current_record']['call_label'], 'unrelated'
      )
      self.assertEqual(completion['raw_manifest']['artifact_count'], 0)
      self.assertFalse((root / 'TERMINAL_FAILURE.json').exists())

  def test_zero_call_setup_and_four_call_persistence_failures_are_controlled(self):
    for count, label in (
        (0, 'record_setup'),
        (4, 'record_persistence'),
    ):
      with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name in (
            'IMPORT_PROVENANCE_PRE_MODEL.json',
            'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
            'IMPORT_PROVENANCE.json',
            'PROTOBUF_PROVENANCE.json',
        ):
          (root / name).write_text('{}\n', encoding='utf-8')
        freeze = root / 'freeze.json'
        freeze.write_text('{}\n', encoding='utf-8')
        cache_root = root / 'cache'
        (cache_root / 'triton').mkdir(parents=True)
        (cache_root / 'xdg').mkdir()
        error = runner.IncompleteRecordError(
            label, count, OSError('synthetic tooling failure')
        )
        attestation = {
            'v3_3_1_status': {},
            'v3_3_2_run': {},
            'v3_3_2_1_failure_status': {},
            'v3_3_2_2_archive_status': {},
        }
        with (
            mock.patch.object(runner, 'OUTPUT_DIR', root),
            mock.patch.object(runner, 'FREEZE_PATH', freeze),
        ):
          runner._write_partial_apply_completion(  # pylint: disable=protected-access
              error,
              results=[],
              compiler=_complete_compiler_fixture(cache_root),
              execution_index=0,
              recipient_order=0,
              anchor_id=0,
              bootstrap_attestation=attestation,
              original_run_binding={},
          )
        completion = json.loads(
            (root / 'RUN_COMPLETE.json').read_text(encoding='utf-8')
        )
        self.assertEqual(completion['model_apply_count'], count)
        self.assertEqual(
            completion['failed_current_record'][
                'dispatched_apply_count_for_current_record'
            ], count,
        )
        self.assertFalse((root / 'TERMINAL_FAILURE.json').exists())

  def test_post_start_failure_persists_exact_zero_apply_count(self):
    with tempfile.TemporaryDirectory() as directory:
      with mock.patch.object(runner, 'OUTPUT_DIR', Path(directory)):
        runner._write_terminal_failure(  # pylint: disable=protected-access
            RuntimeError('pre-model provenance failed'),
            completed_record_count=0,
            apply_count=0,
            compiler_created=False,
        )
      record = json.loads(
          (Path(directory) / 'TERMINAL_FAILURE.json').read_text(
              encoding='utf-8'
          )
      )
    self.assertEqual(record['model_apply_count'], 0)
    self.assertEqual(record['completed_record_count'], 0)
    self.assertEqual(record['eight_row_compile_count'], 0)
    self.assertEqual(record['confirmation_model_calls'], 0)

  def test_direct_main_requires_same_process_attestation(self):
    sys.modules.pop(runner.ATTESTATION_MODULE, None)
    with self.assertRaisesRegex(RuntimeError, 'launcher'):
      runner.consume_bootstrap_attestation()

  def test_direct_script_stops_before_jax_import(self):
    result = subprocess.run(
        (sys.executable, runner.__file__, '--dry-run'),
        text=True,
        capture_output=True,
        check=False,
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn('before pre-import bootstrap', result.stderr)
    self.assertNotIn('Jax plugin configuration error', result.stderr)


if __name__ == '__main__':
  unittest.main()
