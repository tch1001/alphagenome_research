#!/usr/bin/env python3
"""CPU-only tests for the v3.3.3 structural analyzer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


_HERE = Path(__file__).resolve().parent
_ANALYZER_PATH = _HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_3.py'
_SPEC = importlib.util.spec_from_file_location(
    '_test_analyze_encoder_skip_ood_sidecar_v3_3_3', _ANALYZER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
analyzer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analyzer)

_V332_TEST_PATH = _HERE / 'analyze_encoder_skip_ood_sidecar_v3_3_2_test.py'
_V332_TEST_SPEC = importlib.util.spec_from_file_location(
    '_v333_runner_shaped_v332_fixture', _V332_TEST_PATH
)
assert _V332_TEST_SPEC is not None and _V332_TEST_SPEC.loader is not None
v332_fixture = importlib.util.module_from_spec(_V332_TEST_SPEC)
sys.modules[_V332_TEST_SPEC.name] = v332_fixture
_V332_TEST_SPEC.loader.exec_module(v332_fixture)


class V333AnalyzerTest(unittest.TestCase):

  def test_canonical_json_hash_is_order_independent_but_leaf_order_sensitive(self):
    left = {'z': [1, 2], 'a': {'dtype': 'f32', 'shape': [8, 2]}}
    right = {'a': {'shape': [8, 2], 'dtype': 'f32'}, 'z': [1, 2]}
    self.assertEqual(
        analyzer._canonical_json_sha256(left),  # pylint: disable=protected-access
        analyzer._canonical_json_sha256(right),  # pylint: disable=protected-access
    )
    changed = {'z': [2, 1], 'a': {'dtype': 'f32', 'shape': [8, 2]}}
    self.assertNotEqual(
        analyzer._canonical_json_sha256(left),  # pylint: disable=protected-access
        analyzer._canonical_json_sha256(changed),  # pylint: disable=protected-access
    )

  def test_entry_abi_normalizes_only_backend_fingerprint(self):
    first = (
        'HloModule x, entry_computation_layout={(f32[8]{0})->f32[8]{0}}, '
        'fingerprint_before_lhs="1a2B", other="literal"'
    )
    normalized, digest = analyzer._normalized_entry_abi(  # pylint: disable=protected-access
        first + '\nignored body'
    )
    expected = first.replace('"1a2B"', '"<backend-generated>"')
    self.assertEqual(normalized, expected)
    self.assertEqual(digest, hashlib.sha256(expected.encode()).hexdigest())
    changed, changed_digest = analyzer._normalized_entry_abi(  # pylint: disable=protected-access
        first.replace('other="literal"', 'other="changed"')
    )
    self.assertNotEqual(changed, normalized)
    self.assertNotEqual(changed_digest, digest)

  def test_entry_abi_rejects_missing_or_nonhex_fingerprint(self):
    with self.assertRaisesRegex(analyzer.AnalysisError, 'lacks'):
      analyzer._normalized_entry_abi('HloModule x')  # pylint: disable=protected-access
    with self.assertRaisesRegex(analyzer.AnalysisError, 'hexadecimal'):
      analyzer._normalized_entry_abi(  # pylint: disable=protected-access
          'HloModule x, fingerprint_before_lhs="not-hex"'
      )

  def test_backend_diagnostics_preserve_nested_output_tiles(self):
    backend = {
        'fusion_backend_config': {
            'kind': '__triton',
            'block_level_fusion_config': {
                'output_tiles': [[1, 2], [3, 4]],
                'num_warps': '8',
                'num_stages': '3',
            },
        }
    }
    line = (
        '  %fusion = f32[] fusion(), kind=kCustom, '
        f'backend_config={json.dumps(backend, separators=(",", ":"))} '
        'metadata={"kind":"__triton"}'
    )
    parsed = analyzer._backend_config_from_instruction(  # pylint: disable=protected-access
        line
    )
    self.assertEqual(parsed, backend)
    diagnostics = analyzer._recompute_backend_diagnostics(  # pylint: disable=protected-access
        'HloModule x\n' + line
    )
    self.assertEqual(diagnostics['triton_configuration_count'], 1)
    self.assertEqual(
        diagnostics['triton_configurations'][0][
            'block_level_fusion_config'
        ]['output_tiles'],
        [[1, 2], [3, 4]],
    )

  def test_backend_diagnostics_reject_unterminated_nested_json(self):
    with self.assertRaisesRegex(analyzer.AnalysisError, 'unterminated'):
      analyzer._backend_config_from_instruction(  # pylint: disable=protected-access
          '  %x backend_config={"a":{"b":1}'
      )

  def test_runner_shaped_success_compiler_rehashes_cache_and_source_gate(self):
    with tempfile.TemporaryDirectory() as directory:
      run = Path(directory) / 'run'
      compiler_dir = run / 'compiler/eight_row'
      compiler_dir.mkdir(parents=True)
      cache = Path(directory) / 'cache'
      (cache / 'triton').mkdir(parents=True)
      (cache / 'xdg').mkdir()
      stable = compiler_dir / 'graph.stablehlo.mlir'
      hlo = compiler_dir / 'graph.pre_backend.hlo.txt'
      compiled = compiler_dir / 'graph.compiled.hlo.txt'
      stable.write_text('stable source fixture', encoding='utf-8')
      hlo.write_text('pre-backend source fixture', encoding='utf-8')
      compiled.write_text(
          'HloModule fixture, fingerprint_before_lhs="abcd"\n'
          '  %x = f32[] constant(0)\n', encoding='utf-8'
      )
      artifacts = {
          name: {
              'path': str(path.resolve()),
              'sha256': analyzer._sha256(path),  # pylint: disable=protected-access
              'size_bytes': path.stat().st_size,
          }
          for name, path in (
              ('stablehlo', stable), ('hlo', hlo), ('compiled_hlo', compiled)
          )
      }
      signatures = {'fixture': {'shape': [8, 2], 'dtype': 'float32'}}
      signature_sha = analyzer._canonical_json_sha256(signatures)  # pylint: disable=protected-access
      normalized, entry_sha = analyzer._normalized_entry_abi(  # pylint: disable=protected-access
          compiled.read_text(encoding='utf-8')
      )
      fingerprint = hashlib.sha256(
          bytes.fromhex(artifacts['compiled_hlo']['sha256'])
      ).hexdigest()
      source_inputs = {
          name: True for name in (
              'bootstrap_sources_and_prior_trees_exact',
              'tracked_head_and_frozen_inventory_exact',
              'external_device_runtime_environment_exact',
              'same_process_device_runtime_environment_exact',
              'checkpoint_exact', 'reference_object_and_sequences_exact',
              'protobuf_binding_exact',
              'three_import_inventories_stable_exact',
          )
      }
      source_inputs.update({
          'freeze_sha256': 'f' * 64,
          'external_preflight_sha256': 'e' * 64,
          'checkpoint_binding': {'fixture': True},
          'reference_object_binding': {'fixture': True},
      })
      contract = {
          'stablehlo_sha256': artifacts['stablehlo']['sha256'],
          'stablehlo_size_bytes': artifacts['stablehlo']['size_bytes'],
          'pre_backend_hlo_sha256': artifacts['hlo']['sha256'],
          'pre_backend_hlo_size_bytes': artifacts['hlo']['size_bytes'],
          'program_signatures_sha256': signature_sha,
          'entry_abi_sha256': entry_sha,
      }
      observed = dict(contract)
      preimport = {
          'cache_root': str(cache.resolve()), 'fixture_preimport': True,
      }
      original = {
          'artifacts': artifacts, 'executable_fingerprint': fingerprint,
      }
      compiler = {
          'executable_name': 'eight_row', 'compile_count': 1,
          'lower_attempt_count': 1, 'compile_attempt_count': 1,
          'successful_compile_count': 1, 'compile_seconds': 0.25,
          'executable_fingerprint': fingerprint, 'artifacts': artifacts,
          'program_signatures': signatures,
          'program_signatures_sha256': signature_sha,
          'entry_abi': {
              'normalization': (
                  'first HloModule line; replace only '
                  'fingerprint_before_lhs value; omit line-ending newline'
              ),
              'normalized_line_sha256': entry_sha,
              'normalized_line_size_bytes': len(normalized.encode('utf-8')),
              'backend_fingerprint_substitution_count': 1,
          },
          'source_program_gate': {
              'contract': contract, 'observed': observed,
              'stablehlo_exact': True, 'pre_backend_hlo_exact': True,
              'program_signatures_exact': True, 'entry_abi_exact': True,
              'source_runtime_device_toolchain_checkpoint_reference_exact': True,
              'source_input_audit': source_inputs,
              'same_lowered_compiled_object': True,
              'source_program_exact': True,
          },
          'backend_diagnostics': analyzer._recompute_backend_diagnostics(  # pylint: disable=protected-access
              compiled.read_text(encoding='utf-8')
          ),
          'diagnostic_comparisons': {
              'v3_3': {
                  'artifacts': analyzer._artifact_comparison(artifacts, original),  # pylint: disable=protected-access
                  'executable_fingerprint_exact': True,
              },
              'v3_3_2': {
                  'artifacts': analyzer._artifact_comparison(artifacts, original),  # pylint: disable=protected-access
                  'executable_fingerprint_exact': True,
              },
              'compiled_backend_differences_are_diagnostic_only': True,
          },
          'kernel_cache_provenance': {
              'pre_import': preimport,
              'post_compile': analyzer._cache_tree_binding(  # pylint: disable=protected-access
                  cache, 'fixture cache'
              ),
              'default_user_cache_paths_eligible': False,
              'cache_outputs_are_diagnostic_only': True,
          },
      }
      (compiler_dir / 'COMPILER_PROVENANCE.json').write_text(
          json.dumps(compiler, sort_keys=True) + '\n', encoding='utf-8'
      )
      with (
          mock.patch.object(analyzer, 'SOURCE_STABLEHLO', {
              'sha256': artifacts['stablehlo']['sha256'],
              'size_bytes': artifacts['stablehlo']['size_bytes'],
          }),
          mock.patch.object(analyzer, 'SOURCE_PRE_BACKEND_HLO', {
              'sha256': artifacts['hlo']['sha256'],
              'size_bytes': artifacts['hlo']['size_bytes'],
          }),
          mock.patch.object(analyzer, 'PROGRAM_SIGNATURES_SHA256', signature_sha),
          mock.patch.object(analyzer, 'ENTRY_ABI_SHA256', entry_sha),
      ):
        observed_fingerprint, audit = analyzer._validate_compiler(  # pylint: disable=protected-access
            run, compiler, original_v3_3=original,
            consumed_v3_3_2={**original, 'program_signatures': signatures},
            expected_source_input_audit=source_inputs,
            expected_kernel_cache_preimport=preimport,
        )
        self.assertEqual(observed_fingerprint, fingerprint)
        self.assertTrue(audit['source_program_exact'])
        self.assertTrue(
            audit['kernel_cache_provenance']['diagnostic_outputs_only']
        )
        (cache / 'triton/lazy-output').write_text('later', encoding='utf-8')
        compiler['kernel_cache_provenance']['pre_import'] = {
            **preimport, 'fixture_preimport': False,
        }
        (compiler_dir / 'COMPILER_PROVENANCE.json').write_text(
            json.dumps(compiler, sort_keys=True) + '\n', encoding='utf-8'
        )
        with self.assertRaisesRegex(analyzer.AnalysisError, 'cache'):
          analyzer._validate_compiler(  # pylint: disable=protected-access
              run, compiler, original_v3_3=original,
              consumed_v3_3_2={**original, 'program_signatures': signatures},
              expected_source_input_audit=source_inputs,
              expected_kernel_cache_preimport=preimport,
          )

  def test_compiled_backend_difference_does_not_change_source_gate(self):
    signatures = {'selection': {'leaves': [1, 2]}, 'target': {'dtype': 'f32'}}
    signature_sha = analyzer._canonical_json_sha256(signatures)  # pylint: disable=protected-access
    first_template = (
        'HloModule x, entry_computation_layout={{(f32[8]{{0}})->f32[8]{{0}}}}, '
        'fingerprint_before_lhs="{}"'
    )
    normalized, entry_sha = analyzer._normalized_entry_abi(  # pylint: disable=protected-access
        first_template.format('abcd')
    )
    del normalized
    with tempfile.TemporaryDirectory() as directory, mock.patch.object(
        analyzer, 'PROGRAM_SIGNATURES_SHA256', signature_sha
    ), mock.patch.object(analyzer, 'ENTRY_ABI_SHA256', entry_sha):
      root = Path(directory)
      audits = []
      for index, body in enumerate(('backend choice A', 'backend choice B')):
        compiled = root / f'compiled-{index}.hlo'
        compiled.write_text(
            first_template.format('abcd' if index == 0 else '1234')
            + '\n' + body,
            encoding='utf-8',
        )
        artifacts = {
            'stablehlo': dict(analyzer.SOURCE_STABLEHLO),
            'hlo': dict(analyzer.SOURCE_PRE_BACKEND_HLO),
            'compiled_hlo': {
                'path': str(compiled),
                'sha256': analyzer._sha256(compiled),  # pylint: disable=protected-access
                'size_bytes': compiled.stat().st_size,
            },
        }
        observed = {
            'stablehlo_sha256': analyzer.SOURCE_STABLEHLO['sha256'],
            'stablehlo_size_bytes': analyzer.SOURCE_STABLEHLO['size_bytes'],
            'pre_backend_hlo_sha256': analyzer.SOURCE_PRE_BACKEND_HLO['sha256'],
            'pre_backend_hlo_size_bytes': analyzer.SOURCE_PRE_BACKEND_HLO['size_bytes'],
            'program_signatures_sha256': signature_sha,
            'entry_abi_sha256': entry_sha,
        }
        source_inputs = {
            name: True for name in (
                'bootstrap_sources_and_prior_trees_exact',
                'tracked_head_and_frozen_inventory_exact',
                'external_device_runtime_environment_exact',
                'same_process_device_runtime_environment_exact',
                'checkpoint_exact', 'reference_object_and_sequences_exact',
                'protobuf_binding_exact',
                'three_import_inventories_stable_exact',
            )
        }
        source_inputs.update({
            'freeze_sha256': 'f' * 64,
            'external_preflight_sha256': 'e' * 64,
            'checkpoint_binding': {'exact': True},
            'reference_object_binding': {'exact': True},
        })
        compiler = {
            'program_signatures': signatures,
            'program_signatures_sha256': signature_sha,
            'entry_abi': {
                'normalization': (
                    'first HloModule line; replace only '
                    'fingerprint_before_lhs value; omit line-ending newline'
                ),
                'normalized_line_sha256': entry_sha,
                'normalized_line_size_bytes': len(
                    first_template.format('<backend-generated>').encode()
                ),
                'backend_fingerprint_substitution_count': 1,
            },
            'source_program_gate': {
                'contract': {
                    **{
                        'stablehlo_sha256': analyzer.SOURCE_STABLEHLO['sha256'],
                        'stablehlo_size_bytes': analyzer.SOURCE_STABLEHLO['size_bytes'],
                        'pre_backend_hlo_sha256': analyzer.SOURCE_PRE_BACKEND_HLO['sha256'],
                        'pre_backend_hlo_size_bytes': analyzer.SOURCE_PRE_BACKEND_HLO['size_bytes'],
                    },
                    'program_signatures_sha256': signature_sha,
                    'entry_abi_sha256': entry_sha,
                },
                'observed': observed,
                'stablehlo_exact': True, 'pre_backend_hlo_exact': True,
                'program_signatures_exact': True, 'entry_abi_exact': True,
                'source_runtime_device_toolchain_checkpoint_reference_exact': True,
                'source_input_audit': source_inputs,
                'same_lowered_compiled_object': True,
                'source_program_exact': True,
            },
        }
        with mock.patch.object(
            analyzer, 'SOURCE_STABLEHLO', {
                'sha256': analyzer.SOURCE_STABLEHLO['sha256'],
                'size_bytes': analyzer.SOURCE_STABLEHLO['size_bytes'],
            }
        ), mock.patch.object(
            analyzer, 'SOURCE_PRE_BACKEND_HLO', {
                'sha256': analyzer.SOURCE_PRE_BACKEND_HLO['sha256'],
                'size_bytes': analyzer.SOURCE_PRE_BACKEND_HLO['size_bytes'],
            }
        ):
          audits.append(analyzer._validate_source_program_gate(  # pylint: disable=protected-access
              compiler, artifacts, expected_signatures=signatures,
              expected_source_input_audit=source_inputs,
          ))
      self.assertTrue(all(row['source_program_exact'] for row in audits))
      self.assertNotEqual(
          analyzer._sha256(root / 'compiled-0.hlo'),  # pylint: disable=protected-access
          analyzer._sha256(root / 'compiled-1.hlo'),  # pylint: disable=protected-access
      )

  def test_strict_tree_rejects_extra_empty_directory_and_symlink(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      (root / 'a.json').write_text('{}', encoding='utf-8')
      analyzer._strict_tree(root, {'a.json'}, 'fixture')  # pylint: disable=protected-access
      (root / 'empty').mkdir()
      with self.assertRaisesRegex(analyzer.AnalysisError, 'extra/empty'):
        analyzer._strict_tree(root, {'a.json'}, 'fixture')  # pylint: disable=protected-access
      (root / 'empty').rmdir()
      (root / 'link').symlink_to(root / 'a.json')
      with self.assertRaisesRegex(analyzer.AnalysisError, 'symlink'):
        analyzer._strict_tree(root, {'a.json'}, 'fixture')  # pylint: disable=protected-access

  def test_bound_tree_rejects_hash_and_size_tamper(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      path = root / 'record.json'
      path.write_text('{"x":1}\n', encoding='utf-8')
      binding = {
          'record.json': {
              'size_bytes': path.stat().st_size,
              'sha256': analyzer._sha256(path),  # pylint: disable=protected-access
          }
      }
      tree = analyzer._tree_digest([path], root)  # pylint: disable=protected-access
      analyzer._validate_bound_tree(  # pylint: disable=protected-access
          root, binding, tree, 'fixture'
      )
      path.write_text('{"x":2}\n', encoding='utf-8')
      with self.assertRaisesRegex(analyzer.AnalysisError, 'binding changed'):
        analyzer._validate_bound_tree(  # pylint: disable=protected-access
            root, binding, tree, 'fixture'
        )

  def _preflight_fixture(self, root: Path):
    preflight = root / 'preflight'
    preflight.mkdir()
    external_cache = root / 'external-cache'
    model_cache = root / 'model-cache'
    for cache in (external_cache, model_cache):
      (cache / 'triton').mkdir(parents=True)
      (cache / 'xdg').mkdir()
    freeze = {
        'preflight_kernel_cache_dir': str(external_cache),
        'model_kernel_cache_dir': str(model_cache),
        'denied_cache_environment_names': [
            'XLA_FLAGS', 'JAX_COMPILATION_CACHE_DIR', 'CUDA_CACHE_PATH',
            'CUDA_CACHE_MAXSIZE', 'TRITON_DUMP_DIR', 'TRITON_OVERRIDE_DIR',
        ],
        'denied_cache_environment_prefixes': ['JAX_PERSISTENT_CACHE_'],
    }
    external_cache_record = analyzer._expected_cache_environment(  # pylint: disable=protected-access
        freeze, 'external_preflight'
    )
    model_cache_record = analyzer._expected_cache_environment(  # pylint: disable=protected-access
        freeze, 'model'
    )
    for name in ('.allocation.lock', '.preflight_0000.reserved'):
      (preflight / name).write_bytes(b'')
    logs = {}
    for stream in ('stdout', 'stderr'):
      path = preflight / f'preflight_0000.{stream}.log'
      path.write_bytes(b'')
      logs[stream] = {
          'path': str(path.resolve()),
          'sha256': analyzer.EMPTY_SHA256,
      }
    observation = {
        'v3_3_3_runtime_environment': {
            'cache_environment': external_cache_record,
            'live_cache_environment': {
                'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
                'JAX_ENABLE_COMPILATION_CACHE': 'false',
                'CUDA_CACHE_DISABLE': '1',
                'cache_role': 'external_preflight',
                'cache_root': str(external_cache.resolve()),
                'triton_cache_dir': str((external_cache / 'triton').resolve()),
                'xdg_cache_home': str((external_cache / 'xdg').resolve()),
                'present_forbidden_names': [],
                'exact_to_pre_import_routing': True,
            },
        }
    }
    raw = {
        'script_version': 'opensplice-device-preflight-v3.3.3',
        'status': 'pass', 'preflight_attempt_number': 0,
        'amendment_sha256': analyzer.AMENDMENT_SHA256,
        'original_protocol_sha256': analyzer.ORIGINAL_PROTOCOL_SHA256,
        'freeze_sha256': 'f' * 64, 'failure': None,
        'no_model_or_biological_access': True,
        'no_jit_or_array_kernel': True, 'logs': logs,
        'observation': observation,
    }
    json_path = preflight / 'preflight_0000.json'
    json_path.write_text(json.dumps(raw), encoding='utf-8')
    external = {
        'path': str(json_path.resolve()),
        'sha256': analyzer._sha256(json_path),  # pylint: disable=protected-access
        **raw,
        'validated_logs': logs,
    }
    paths = sorted(preflight.iterdir())
    external['directory_binding'] = {
        'path': str(preflight.resolve()), 'file_count': 5,
        'file_sha256': {
            path.name: {
                'sha256': analyzer._sha256(path),  # pylint: disable=protected-access
                'size_bytes': path.stat().st_size,
            }
            for path in paths
        },
        'tree_sha256': analyzer._tree_digest(paths, preflight),  # pylint: disable=protected-access
        'sole_preflight_attempt_exact': True,
    }
    start = {
        'bootstrap': {
            'pid': 7,
            'sanitized_environment': {'cache_environment': model_cache_record},
        },
        'same_process_preflight': {
            'pid': 7, 'v3_3_3_cache_environment': model_cache_record
        },
        'external_preflight': external,
        'cache_role_transition': {
            'external_preflight': external_cache_record,
            'model': model_cache_record,
            'roles_and_roots_distinct': True,
            'shared_policy_exact': True,
            'default_user_cache_paths_eligible': False,
        },
    }
    return preflight, freeze, start

  def test_external_cache_schema_and_exact_preflight_tree(self):
    with tempfile.TemporaryDirectory() as directory:
      preflight, freeze, start = self._preflight_fixture(Path(directory))
      with mock.patch.object(analyzer, '_PREFLIGHT_DIR', preflight), mock.patch.object(
          analyzer._v332._v33, '_validate_device_observation', return_value=None  # pylint: disable=protected-access
      ):
        audit = analyzer._validate_external_and_same_process(  # pylint: disable=protected-access
            start, freeze, 'f' * 64
        )
        self.assertEqual(audit['external_preflight_file_count'], 5)
        (preflight / 'extra').write_bytes(b'')
        with self.assertRaisesRegex(analyzer.AnalysisError, 'membership'):
          analyzer._validate_external_and_same_process(  # pylint: disable=protected-access
              start, freeze, 'f' * 64
          )

  def test_external_cache_must_use_nested_external_role(self):
    with tempfile.TemporaryDirectory() as directory:
      preflight, freeze, start = self._preflight_fixture(Path(directory))
      start['external_preflight']['observation'][
          'v3_3_3_runtime_environment'
      ]['cache_environment']['cache_role'] = 'model'
      raw = {
          key: value for key, value in start['external_preflight'].items()
          if key not in {
              'path', 'sha256', 'validated_logs', 'directory_binding'
          }
      }
      json_path = preflight / 'preflight_0000.json'
      json_path.write_text(json.dumps(raw), encoding='utf-8')
      start['external_preflight']['sha256'] = analyzer._sha256(  # pylint: disable=protected-access
          json_path
      )
      paths = sorted(preflight.iterdir())
      start['external_preflight']['directory_binding'] = {
          'path': str(preflight.resolve()), 'file_count': 5,
          'file_sha256': {
              path.name: {
                  'sha256': analyzer._sha256(path),  # pylint: disable=protected-access
                  'size_bytes': path.stat().st_size,
              }
              for path in paths
          },
          'tree_sha256': analyzer._tree_digest(paths, preflight),  # pylint: disable=protected-access
          'sole_preflight_attempt_exact': True,
      }
      with mock.patch.object(analyzer, '_PREFLIGHT_DIR', preflight), mock.patch.object(
          analyzer._v332._v33, '_validate_device_observation', return_value=None  # pylint: disable=protected-access
      ):
        with self.assertRaisesRegex(analyzer.AnalysisError, 'cache-environment'):
          analyzer._validate_external_and_same_process(  # pylint: disable=protected-access
              start, freeze, 'f' * 64
          )

  def test_external_live_routing_and_directory_binding_are_fail_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      preflight, freeze, start = self._preflight_fixture(Path(directory))
      with mock.patch.object(analyzer, '_PREFLIGHT_DIR', preflight), mock.patch.object(
          analyzer._v332._v33, '_validate_device_observation', return_value=None  # pylint: disable=protected-access
      ):
        start['external_preflight']['directory_binding'][
            'tree_sha256'
        ] = '0' * 64
        with self.assertRaisesRegex(analyzer.AnalysisError, 'directory binding'):
          analyzer._validate_external_and_same_process(  # pylint: disable=protected-access
              start, freeze, 'f' * 64
          )
        (Path(directory) / 'second').mkdir()
        _, freeze, start = self._preflight_fixture(Path(directory) / 'second')
      second = Path(directory) / 'second/preflight'
      start['external_preflight']['observation'][
          'v3_3_3_runtime_environment'
      ]['live_cache_environment']['CUDA_CACHE_DISABLE'] = '0'
      raw = {
          key: value for key, value in start['external_preflight'].items()
          if key not in {
              'path', 'sha256', 'validated_logs', 'directory_binding'
          }
      }
      json_path = second / 'preflight_0000.json'
      json_path.write_text(json.dumps(raw), encoding='utf-8')
      start['external_preflight']['sha256'] = analyzer._sha256(json_path)  # pylint: disable=protected-access
      paths = sorted(second.iterdir())
      start['external_preflight']['directory_binding'] = {
          'path': str(second.resolve()), 'file_count': 5,
          'file_sha256': {
              path.name: {
                  'sha256': analyzer._sha256(path),  # pylint: disable=protected-access
                  'size_bytes': path.stat().st_size,
              } for path in paths
          },
          'tree_sha256': analyzer._tree_digest(paths, second),  # pylint: disable=protected-access
          'sole_preflight_attempt_exact': True,
      }
      with mock.patch.object(analyzer, '_PREFLIGHT_DIR', second), mock.patch.object(
          analyzer._v332._v33, '_validate_device_observation', return_value=None  # pylint: disable=protected-access
      ):
        with self.assertRaisesRegex(analyzer.AnalysisError, 'cache-environment'):
          analyzer._validate_external_and_same_process(  # pylint: disable=protected-access
              start, freeze, 'f' * 64
          )

  def test_consumed_structural_archives_validate_without_scientific_access(self):
    consumed = analyzer._validate_v3_3_2_run()  # pylint: disable=protected-access
    failed = analyzer._validate_v3_3_2_1_failure()  # pylint: disable=protected-access
    completed = analyzer._validate_v3_3_2_2_archive()  # pylint: disable=protected-access
    self.assertEqual(consumed['whole_run_file_count'], 11)
    self.assertEqual(failed['state'], 'failed_consumed_no_retry')
    self.assertEqual(completed['state'], 'complete_controlled_stop_audited')
    self.assertFalse(completed['scientific_summary_computed'])
    self.assertFalse(completed['combined_analysis_permitted'])

  def test_historical_blob_tamper_fails_closed(self):
    with mock.patch.object(
        analyzer, '_git_blob_sha256', return_value='0' * 64
    ):
      with self.assertRaisesRegex(analyzer.AnalysisError, 'Historical'):
        analyzer._validate_source_bundle(  # pylint: disable=protected-access
            analyzer._V3_3_2_1_SOURCES,  # pylint: disable=protected-access
            implementation_commit=analyzer._V3_3_2_1_SOURCE_COMMIT,  # pylint: disable=protected-access
        )

  def test_no_model_or_jax_modules_are_imported(self):
    analyzer._assert_cpu_only('test')  # pylint: disable=protected-access


class RunnerShapedTerminalIntegrationTest(unittest.TestCase):
  """Exercises each frozen terminal shape through the public analyze path."""

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.run = self.root / 'run'
    self.original = self.root / 'original'
    self.run.mkdir()
    (self.run / 'compiler/eight_row').mkdir(parents=True)
    self.cases = {
        order: v332_fixture._case(order) for order in range(20)  # pylint: disable=protected-access
    }
    self.sequences = {
        order: {
            'reference': f'{order:064x}',
            'alternate': f'{order + 100:064x}',
        }
        for order in range(20)
    }
    self.original_manifest = {}
    self.v331 = {'state': 'completed'}
    self.v332run = {'eight_row_compiler': {'program_signatures': {}}}
    self.failed = {'state': 'failed_consumed_no_retry'}
    self.archived = {'state': 'complete_controlled_stop_audited'}
    self.freeze = {
        'v3_3_1_status': self.v331, 'program_signatures': {},
        'model_kernel_cache_dir': str(self.root / 'model-cache'),
        'preflight_kernel_cache_dir': str(self.root / 'preflight-cache'),
        'denied_cache_environment_names': [
            'XLA_FLAGS', 'JAX_COMPILATION_CACHE_DIR', 'CUDA_CACHE_PATH',
            'CUDA_CACHE_MAXSIZE', 'TRITON_DUMP_DIR', 'TRITON_OVERRIDE_DIR',
        ],
        'denied_cache_environment_prefixes': ['JAX_PERSISTENT_CACHE_'],
    }
    for cache in ('model-cache', 'preflight-cache'):
      (self.root / cache / 'triton').mkdir(parents=True)
      (self.root / cache / 'xdg').mkdir()
    self.start_audit = {
        'external_preflight_sha256': 'a' * 64,
        'checkpoint_binding': {'fixture': 'checkpoint'},
        'reference_object_binding': {'fixture': 'reference'},
    }
    (self.original / 'RUN_COMPLETE.json').parent.mkdir(parents=True)
    self._write_json(
        self.original / 'RUN_COMPLETE.json', {'eight_row_compiler': {}}
    )
    for name in (
        'ATTEMPT_STARTED.json', 'IMPORT_PROVENANCE_PRE_MODEL.json',
        'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
        'IMPORT_PROVENANCE.json', 'PROTOBUF_PROVENANCE.json',
    ):
      self._write_json(self.run / name, {'fixture': name})

  def tearDown(self):
    self.temporary.cleanup()

  @staticmethod
  def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )

  def _record(self, order: int, anchor: int, index: int, *, invalid=False):
    record = v332_fixture._record(  # pylint: disable=protected-access
        original_root=self.original,
        original_manifest=self.original_manifest,
        cases=self.cases, sequence_bindings=self.sequences,
        order=order, anchor=anchor, execution_index=index,
        status='invalid' if invalid else 'complete',
    )
    record.update({
        'family': 'v3_3_3_unrelated_donor_sidecar_anchor',
        'script_version': analyzer.SCRIPT_VERSION,
        'amendment_sha256': analyzer.AMENDMENT_SHA256,
        'amendment_commit': analyzer.AMENDMENT_COMMIT,
    })
    return record

  def _compiler_stub(self, *, source_exact=True):
    gate = {'source_program_exact': source_exact}
    preimport = analyzer._expected_cache_environment(  # pylint: disable=protected-access
        self.freeze, 'model'
    )
    historical = analyzer._cache_tree_binding(  # pylint: disable=protected-access
        self.root / 'model-cache', 'fixture model cache'
    )
    cache_audit = {
        'pre_import_inputs_absent': True, 'post_phase': 'post_compile',
        'post_tree_sha256': historical['tree_sha256'],
        'historical_binding': historical, 'diagnostic_outputs_only': True,
    }
    return {
        'source_program_gate': gate,
        'backend_diagnostics': {'fixture': True},
        'diagnostic_comparisons': {'fixture': True},
    }, {
        'source_program_exact': source_exact,
        'backend_diagnostics': {'fixture': True},
        'diagnostic_comparisons': {'fixture': True},
        'compiled_backend_equality_gate': False,
        'kernel_cache_provenance': cache_audit,
    }

  def _completion(self, *, reason=None, count=80, partial_calls=None):
    compiler, _ = self._compiler_stub(
        source_exact=reason != 'source_program_mismatch'
    )
    full = reason is None
    source_exact = reason != 'source_program_mismatch'
    result = {
        'status': 'complete' if full else 'controlled_stop',
        'stop_reason': reason,
        'message': (
            'All 80 frozen v3.3.3 OOD sidecar records completed.' if full
            else (
                'The mandatory StableHLO/pre-backend/signature/ABI/source '
                'program gate failed before model apply zero.'
                if reason == 'source_program_mismatch' else
                'fixture temporary message'
            )
        ),
        'attempt_id': analyzer.ATTEMPT_ID,
        'script_version': analyzer.SCRIPT_VERSION,
        'amendment_sha256': analyzer.AMENDMENT_SHA256,
        'amendment_commit': analyzer.AMENDMENT_COMMIT,
        'original_protocol_sha256': analyzer.ORIGINAL_PROTOCOL_SHA256,
        'freeze_sha256': '9' * 64,
        'ood_anchor_record_count': count, 'ood_invalid_count': (
            1 if reason == 'ood_tooling_failure' and partial_calls is None else 0
        ),
        'unique_recipient_anchor_count': count,
        'all_80_recipient_anchors_complete': full,
        'model_apply_count': 4 * count,
        'expected_model_apply_count': 320,
        'eight_row_compile_count': 1,
        'eight_row_compile_attempt_count': 1,
        'eight_row_successful_compile_count': 1,
        'six_row_compile_count': 0,
        'identity_rerun_count': 0, 'main_cube_rerun_count': 0,
        'old_ood_records_reused': 0,
        'one_fixed_eight_row_executable': True,
        'eight_row_compiler': compiler,
        'eight_row_executable_fingerprint': 'e' * 64,
        'source_program_gate': compiler['source_program_gate'],
        'source_program_exact': source_exact,
        'compiled_backend_diagnostic_only': True,
        'backend_diagnostics': compiler['backend_diagnostics'],
        'diagnostic_comparisons': compiler['diagnostic_comparisons'],
        'id0_all20': full, 'id255_all20': full,
        'invariant_rows_between_calls': list(analyzer.INVARIANT_ROWS),
        'active_rows_have_no_forced_cross_call_predicate': True,
        'original_run_binding': dict(analyzer._v332._ORIGINAL_BINDING),  # pylint: disable=protected-access
        'original_run_revalidated_in_full': True,
        'original_ood_records_provenance_only': True,
        'v3_3_1_status': self.v331,
        'v3_3_2_run_binding': self.v332run,
        'v3_3_2_1_failure_status': self.failed,
        'v3_3_2_2_archive_status': self.archived,
        'import_provenance_phases': {},
        'import_provenance_sha256': 'a' * 64,
        'protobuf_provenance_sha256': 'b' * 64,
        'raw_manifest': {}, 'confirmation_model_calls': 0,
        'confirmation_scope_disclosure': analyzer.CONFIRMATION_DISCLOSURE,
        'scientific_summary_computed': False,
        'shapley_or_nomination_computed': False,
        'interaction_or_resolution_computed': False,
        'combined_analysis_permitted': False,
        'completed_at_unix_s': 1.0,
    }
    historical = analyzer._cache_tree_binding(  # pylint: disable=protected-access
        self.root / 'model-cache', 'fixture model cache'
    )
    result['model_kernel_cache_final'] = {
        'pre_import': analyzer._expected_cache_environment(  # pylint: disable=protected-access
            self.freeze, 'model'
        ),
        'historical_stage': 'post_compile',
        'historical_binding': historical,
        'terminal': historical,
        'historical_to_terminal_tree_exact': True,
        'historical_to_terminal_equality_is_a_gate': False,
        'historical_snapshot_not_reauthenticated_as_live_files': True,
        'default_user_cache_paths_eligible': False,
        'cache_outputs_are_diagnostic_only': True,
    }
    if reason == 'ood_tooling_failure' and partial_calls is None:
      order, anchor = analyzer._execution_order()[count - 1]  # pylint: disable=protected-access
      result['message'] = (
          f'OOD sidecar audit failed at order={order}, anchor_id={anchor}.'
      )
      complete_prefix = analyzer._execution_order()[:count - 1]  # pylint: disable=protected-access
      result['id0_all20'] = sum(a == 0 for _, a in complete_prefix) == 20
      result['id255_all20'] = sum(a == 255 for _, a in complete_prefix) == 20
    if partial_calls is not None:
      result.pop('all_80_recipient_anchors_complete')
      result.pop('one_fixed_eight_row_executable')
      result.pop('id0_all20')
      result.pop('id255_all20')
      result.pop('import_provenance_sha256')
      result.pop('invariant_rows_between_calls')
      result.pop('active_rows_have_no_forced_cross_call_predicate')
      result.update({
          'message': (
              'A dispatched OOD call failed; exact prefix preserved; no retry.'
          ),
          'ood_invalid_count': 0, 'incomplete_record_count': 1,
          'model_apply_count': 4 * count + partial_calls,
          'eight_row_compile_attempt_count': 1,
          'eight_row_successful_compile_count': 1,
          'failed_current_record': {
              'execution_index': count,
              'recipient_order': analyzer._execution_order()[count][0],  # pylint: disable=protected-access
              'anchor_id': analyzer._execution_order()[count][1],  # pylint: disable=protected-access
              'call_label': (
                  'record_setup_validation_or_persistence'
                  if partial_calls == 0 else (
                      'intended', 'intended_repeat', 'unrelated',
                      'unrelated_repeat'
                  )[partial_calls - 1]
              ),
              'dispatched_apply_count_for_current_record': partial_calls,
              'error_type': 'RuntimeError', 'error_message': 'fixture call failed',
          },
          'no_retry': True,
      })
    return result

  def _write_normal_tree(self, *, reason=None, count=80, partial_calls=None):
    mapping = {}
    for index, (order, anchor) in enumerate(
        analyzer._execution_order()[:count]  # pylint: disable=protected-access
    ):
      invalid = reason == 'ood_tooling_failure' and partial_calls is None and index == count - 1
      record = self._record(order, anchor, index, invalid=invalid)
      relative = analyzer._artifact_relative(self.cases[order], anchor)  # pylint: disable=protected-access
      path = self.run / relative
      self._write_json(path, record)
      mapping[relative] = analyzer._sha256(path)  # pylint: disable=protected-access
    manifest = {
        'artifact_count': count, 'artifact_sha256': mapping,
        'artifact_tree_sha256': analyzer._tree_digest(  # pylint: disable=protected-access
            (self.run / name for name in mapping), self.run
        ),
    }
    completion = self._completion(
        reason=reason, count=count, partial_calls=partial_calls
    )
    completion['raw_manifest'] = manifest
    self._write_json(self.run / 'RAW_MANIFEST.json', manifest)
    self._write_json(self.run / 'RUN_COMPLETE.json', completion)
    for filename in (
        'graph.stablehlo.mlir', 'graph.pre_backend.hlo.txt',
        'graph.compiled.hlo.txt',
    ):
      (self.run / 'compiler/eight_row' / filename).write_text(
          'fixture', encoding='utf-8'
      )
    self._write_json(
        self.run / 'compiler/eight_row/COMPILER_PROVENANCE.json',
        completion['eight_row_compiler'],
    )

  def _analyze_normal(self):
    _, compiler_audit = self._compiler_stub(
        source_exact=json.loads(
            (self.run / 'RUN_COMPLETE.json').read_text(encoding='utf-8')
        ).get('source_program_exact', True)
    )
    freeze_result = (
        self.freeze, '9' * 64, {}, self.original_manifest,
        self.sequences, self.v332run, self.failed, self.archived,
    )
    with (
        mock.patch.object(analyzer, '_validate_freeze', return_value=freeze_result),
        mock.patch.object(
            analyzer, '_validate_start',
            return_value=(self.start_audit, self.sequences),
        ),
        mock.patch.object(analyzer._v332, '_ORIGINAL_RUN_DIR', self.original),  # pylint: disable=protected-access
        mock.patch.object(
            analyzer._v332._v33, '_load_cases', return_value=self.cases  # pylint: disable=protected-access
        ),
        mock.patch.object(
            analyzer, '_validate_compiler',
            return_value=('e' * 64, compiler_audit),
        ),
        mock.patch.object(analyzer, '_validate_imports', return_value={
            'stable_shared_module_bytes': True,
        }),
        mock.patch.object(analyzer, '_validate_protobuf', return_value={
            'seven_role_two_generated_output_repair_exact': True,
        }),
    ):
      return analyzer.analyze(self.run, bundle_root=analyzer._REPO_ROOT)  # pylint: disable=protected-access

  def test_full_80_runner_tree_is_structural_only(self):
    self._write_normal_tree()
    result = self._analyze_normal()
    self.assertEqual(result['decision'], 'sidecar_complete_structural_audit')
    self.assertEqual(result['sidecar_audit']['audited_record_count'], 80)
    self.assertFalse(result['combined_analysis_permitted'])
    self.assertFalse(result['scientific_summary_computed'])
    self.assertFalse(result['shapley_or_nomination_computed'])
    self.assertFalse(result['nomination_performed'])

  def test_source_program_mismatch_zero_apply_is_structural_only(self):
    self._write_normal_tree(reason='source_program_mismatch', count=0)
    result = self._analyze_normal()
    self.assertEqual(result['decision'], 'controlled_stop_source_program_mismatch')
    self.assertEqual(result['sidecar_audit']['audited_model_apply_count'], 0)
    self.assertIsNone(result['nomination'])

  def test_invalid_persisted_record_is_exact_final_prefix(self):
    self._write_normal_tree(reason='ood_tooling_failure', count=2)
    result = self._analyze_normal()
    self.assertEqual(result['decision'], 'controlled_stop_ood_tooling_failure')
    self.assertEqual(result['sidecar_audit']['invalid_record_count'], 1)
    path = self.run / analyzer._artifact_relative(self.cases[0], 127)  # pylint: disable=protected-access
    value = json.loads(path.read_text(encoding='utf-8'))
    value['execution_index'] = 7
    self._write_json(path, value)
    with self.assertRaises(analyzer.AnalysisError):
      self._analyze_normal()

  def test_partial_apply_one_to_four_calls_are_audited_without_science(self):
    for calls in (0, 1, 2, 3, 4):
      with self.subTest(calls=calls):
        if self.run.exists():
          for child in list(self.run.iterdir()):
            if child.is_dir():
              import shutil
              shutil.rmtree(child)
            else:
              child.unlink()
        self.run.mkdir(exist_ok=True)
        (self.run / 'compiler/eight_row').mkdir(parents=True)
        for name in (
            'ATTEMPT_STARTED.json', 'IMPORT_PROVENANCE_PRE_MODEL.json',
            'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
            'IMPORT_PROVENANCE.json', 'PROTOBUF_PROVENANCE.json',
        ):
          self._write_json(self.run / name, {'fixture': name})
        self._write_normal_tree(
            reason='ood_tooling_failure', count=1, partial_calls=calls
        )
        completion = json.loads(
            (self.run / 'RUN_COMPLETE.json').read_text(encoding='utf-8')
        )
        _, compiler_audit = self._compiler_stub()
        with (
            mock.patch.object(analyzer._v332, '_ORIGINAL_RUN_DIR', self.original),  # pylint: disable=protected-access
            mock.patch.object(
                analyzer._v332._v33, '_load_cases', return_value=self.cases  # pylint: disable=protected-access
            ),
            mock.patch.object(
                analyzer, '_validate_compiler',
                return_value=('e' * 64, compiler_audit),
            ),
            mock.patch.object(analyzer, '_validate_imports', return_value={
                'stable_shared_module_bytes': True,
            }),
            mock.patch.object(analyzer, '_validate_protobuf', return_value={
                'seven_role_two_generated_output_repair_exact': True,
            }),
        ):
          result = analyzer._analyze_partial_apply_stop(  # pylint: disable=protected-access
              self.run, freeze=self.freeze, freeze_sha='9' * 64,
              completion=completion, start_audit=self.start_audit,
              original_audit={}, original_manifest=self.original_manifest,
              sequence_bindings=self.sequences,
              v3_3_2_run=self.v332run,
              v3_3_2_1_failure=self.failed,
              v3_3_2_2_archive=self.archived,
              bundle_root=analyzer._REPO_ROOT,  # pylint: disable=protected-access
          )
        self.assertFalse(result['scientific_summary_computed'])
        self.assertFalse(result['shapley_or_nomination_computed'])

  def test_lower_and_compile_failures_are_exact_zero_apply_stops(self):
    for stage, compile_count in (('lower', 0), ('compile', 1)):
      with self.subTest(stage=stage):
        import shutil
        shutil.rmtree(self.run)
        self.run.mkdir()
        directory = self.run / 'compiler/eight_row'
        directory.mkdir(parents=True)
        for name in (
            'ATTEMPT_STARTED.json', 'IMPORT_PROVENANCE_PRE_MODEL.json',
            'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
            'IMPORT_PROVENANCE.json', 'PROTOBUF_PROVENANCE.json',
        ):
          self._write_json(self.run / name, {'fixture': name})
        artifacts = {}
        if stage == 'compile':
          for key, filename in (
              ('stablehlo', 'graph.stablehlo.mlir'),
              ('hlo', 'graph.pre_backend.hlo.txt'),
          ):
            path = directory / filename
            path.write_text(f'{key} fixture', encoding='utf-8')
            artifacts[key] = {
                'path': str(path.resolve()),
                'sha256': analyzer._sha256(path),  # pylint: disable=protected-access
                'size_bytes': path.stat().st_size,
            }
        preimport = analyzer._expected_cache_environment(  # pylint: disable=protected-access
            self.freeze, 'model'
        )
        compiler = {
            'status': 'compiler_failure', 'failure_stage': stage,
            'compile_count': compile_count, 'lower_attempt_count': 1,
            'compile_attempt_count': compile_count,
            'successful_compile_count': 0,
            'lower_or_compile_pipeline_attempt_count': 1,
            'compile_seconds': 0.5, 'artifacts': artifacts,
            'program_signatures': {},
            'program_signatures_sha256': analyzer._canonical_json_sha256({}),  # pylint: disable=protected-access
            'source_program_gate': None,
            'compiled_backend_diagnostic_only': True,
            'failure': {
                'type': 'RuntimeError', 'message': 'fixture compiler failed',
                'traceback': 'Traceback\nRuntimeError: fixture compiler failed',
            },
            'no_compile_retry': True, 'model_apply_count': 0,
            'kernel_cache_provenance': {
                'pre_import': preimport,
                'post_failure': analyzer._cache_tree_binding(  # pylint: disable=protected-access
                    self.root / 'model-cache', 'fixture model cache'
                ),
                'default_user_cache_paths_eligible': False,
                'cache_outputs_are_diagnostic_only': True,
            },
        }
        self._write_json(directory / 'COMPILER_PROVENANCE.json', compiler)
        manifest = {
            'artifact_count': 0, 'artifact_sha256': {},
            'artifact_tree_sha256': analyzer.EMPTY_SHA256,
        }
        completion = {
            'status': 'controlled_stop', 'stop_reason': 'compiler_failure',
            'message': 'The sole lowering/compiler attempt failed; no retry.',
            'attempt_id': analyzer.ATTEMPT_ID,
            'script_version': analyzer.SCRIPT_VERSION,
            'amendment_sha256': analyzer.AMENDMENT_SHA256,
            'amendment_commit': analyzer.AMENDMENT_COMMIT,
            'original_protocol_sha256': analyzer.ORIGINAL_PROTOCOL_SHA256,
            'freeze_sha256': '9' * 64,
            'ood_anchor_record_count': 0, 'ood_invalid_count': 0,
            'unique_recipient_anchor_count': 0, 'model_apply_count': 0,
            'expected_model_apply_count': 320,
            'eight_row_compile_count': compile_count,
            'eight_row_compile_attempt_count': compile_count,
            'eight_row_successful_compile_count': 0,
            'six_row_compile_count': 0, 'identity_rerun_count': 0,
            'main_cube_rerun_count': 0, 'old_ood_records_reused': 0,
            'compiler': compiler, 'source_program_gate': None,
            'compiled_backend_diagnostic_only': True,
            'no_compile_retry': True,
            'original_run_binding': dict(analyzer._v332._ORIGINAL_BINDING),  # pylint: disable=protected-access
            'original_run_revalidated_in_full': True,
            'original_ood_records_provenance_only': True,
            'v3_3_1_status': self.v331,
            'v3_3_2_run_binding': self.v332run,
            'v3_3_2_1_failure_status': self.failed,
            'v3_3_2_2_archive_status': self.archived,
            'import_provenance_phases': {},
            'protobuf_provenance_sha256': 'b' * 64,
            'raw_manifest': manifest, 'confirmation_model_calls': 0,
            'scientific_summary_computed': False,
            'shapley_or_nomination_computed': False,
            'interaction_or_resolution_computed': False,
            'combined_analysis_permitted': False,
            'confirmation_scope_disclosure': analyzer.CONFIRMATION_DISCLOSURE,
            'completed_at_unix_s': 1.0,
        }
        terminal = analyzer._cache_tree_binding(  # pylint: disable=protected-access
            self.root / 'model-cache', 'fixture model cache'
        )
        completion['model_kernel_cache_final'] = {
            'pre_import': preimport, 'historical_stage': 'post_failure',
            'historical_binding': compiler['kernel_cache_provenance'][
                'post_failure'
            ],
            'terminal': terminal,
            'historical_to_terminal_tree_exact': True,
            'historical_to_terminal_equality_is_a_gate': False,
            'historical_snapshot_not_reauthenticated_as_live_files': True,
            'default_user_cache_paths_eligible': False,
            'cache_outputs_are_diagnostic_only': True,
        }
        self._write_json(self.run / 'RAW_MANIFEST.json', manifest)
        self._write_json(self.run / 'RUN_COMPLETE.json', completion)
        with (
            mock.patch.object(
                analyzer, '_validate_compiler_failure_imports',
                return_value={'stable_shared_module_bytes': True},
            ),
            mock.patch.object(analyzer, '_validate_protobuf', return_value={
                'seven_role_two_generated_output_repair_exact': True,
            }),
        ):
          result = analyzer._analyze_compiler_failure(  # pylint: disable=protected-access
              self.run, freeze=self.freeze, freeze_sha='9' * 64,
              completion=completion, start_audit=self.start_audit,
              original_audit={}, v3_3_2_run=self.v332run,
              v3_3_2_1_failure=self.failed,
              v3_3_2_2_archive=self.archived,
              bundle_root=analyzer._REPO_ROOT,  # pylint: disable=protected-access
          )
        self.assertEqual(result['decision'], 'controlled_stop_compiler_failure')
        self.assertEqual(result['sidecar_audit']['audited_model_apply_count'], 0)
        self.assertFalse(result['shapley_or_nomination_computed'])


class AppendOnlyAnalysisAttemptTest(unittest.TestCase):

  def _argv(self, run: Path, analysis: Path):
    return [
        str(analyzer.__file__), '--run-dir', str(run), '--bundle-root',
        str(analyzer._REPO_ROOT),  # pylint: disable=protected-access
        '--output-json', str(analysis / 'ANALYSIS.json'),
        '--output-markdown', str(analysis / 'RESULT.md'),
    ]

  @staticmethod
  def _started(attempt: Path, analysis: Path, run: Path):
    return {
        'analysis_version': analyzer.ANALYSIS_VERSION,
        'status': 'analysis_started_append_only_one_shot',
        'analysis_attempt_dir': str(attempt), 'analysis_dir': str(analysis),
        'run_dir': str(run),
        'raw_scientific_endpoint_evidence_reached': False,
        'scientific_summary_computed': False,
        'shapley_or_nomination_computed': False,
        'nomination_performed': False,
    }

  def test_mid_raw_failure_is_consumed_and_rerun_is_refused(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      run, attempt, analysis = root / 'run', root / 'attempt', root / 'analysis'
      run.mkdir()

      def precheck(*_args, **_kwargs):
        if attempt.exists() or analysis.exists():
          raise FileExistsError('analysis/attempt exists; never resume or retry')
        return self._started(attempt, analysis, run)

      def fail_after_raw(*_args, _raw_access_marker=None, **_kwargs):
        self.assertIsNotNone(_raw_access_marker)
        _raw_access_marker()
        raise analyzer.AnalysisError('synthetic mid-raw structural failure')

      with (
          mock.patch.object(analyzer, '_ANALYSIS_ATTEMPT_DIR', attempt),
          mock.patch.object(analyzer, '_ANALYSIS_DIR', analysis),
          mock.patch.object(analyzer, '_analysis_attempt_precheck', side_effect=precheck),
          mock.patch.object(analyzer, 'analyze', side_effect=fail_after_raw),
          mock.patch.object(sys, 'argv', self._argv(run, analysis)),
      ):
        with self.assertRaisesRegex(analyzer.AnalysisError, 'mid-raw'):
          analyzer.main()
        self.assertEqual(
            {path.name for path in attempt.iterdir()},
            {'ANALYSIS_ATTEMPT_STARTED.json', 'ANALYSIS_FAILURE.json'},
        )
        failure = json.loads(
            (attempt / 'ANALYSIS_FAILURE.json').read_text(encoding='utf-8')
        )
        self.assertTrue(failure['raw_scientific_endpoint_evidence_reached'])
        self.assertEqual(failure['analysis_output_state']['state'], 'absent')
        self.assertFalse(failure['scientific_summary_computed'])
        self.assertFalse(failure['shapley_or_nomination_computed'])
        started_hash = analyzer._sha256(  # pylint: disable=protected-access
            attempt / 'ANALYSIS_ATTEMPT_STARTED.json'
        )
        self.assertEqual(failure['attempt_started_sha256'], started_hash)
        with self.assertRaisesRegex(FileExistsError, 'never resume'):
          analyzer.main()
        self.assertEqual(
            {path.name for path in attempt.iterdir()},
            {'ANALYSIS_ATTEMPT_STARTED.json', 'ANALYSIS_FAILURE.json'},
        )

  def test_partial_output_failure_is_bound_without_cleanup_or_retry(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      run, attempt, analysis = root / 'run', root / 'attempt', root / 'analysis'
      run.mkdir()

      def partial_write(*_args, **_kwargs):
        analysis.mkdir()
        (analysis / 'ANALYSIS.json').write_text(
            '{"structural":true}\n', encoding='utf-8'
        )
        raise OSError('synthetic second-output write failure')

      with (
          mock.patch.object(analyzer, '_ANALYSIS_ATTEMPT_DIR', attempt),
          mock.patch.object(analyzer, '_ANALYSIS_DIR', analysis),
          mock.patch.object(
              analyzer, '_analysis_attempt_precheck',
              return_value=self._started(attempt, analysis, run),
          ),
          mock.patch.object(analyzer, 'analyze', return_value={}),
          mock.patch.object(analyzer, '_write_outputs', side_effect=partial_write),
          mock.patch.object(sys, 'argv', self._argv(run, analysis)),
      ):
        with self.assertRaisesRegex(OSError, 'second-output'):
          analyzer.main()
      failure = json.loads(
          (attempt / 'ANALYSIS_FAILURE.json').read_text(encoding='utf-8')
      )
      state = failure['analysis_output_state']
      self.assertEqual(state['state'], 'partial')
      self.assertEqual(set(state['files']), {'ANALYSIS.json'})
      self.assertEqual(
          state['files']['ANALYSIS.json']['sha256'],
          analyzer._sha256(analysis / 'ANALYSIS.json'),  # pylint: disable=protected-access
      )
      self.assertFalse(failure['scientific_summary_computed'])

  def test_direct_production_analyze_cannot_bypass_attempt_start(self):
    with tempfile.TemporaryDirectory() as directory:
      run = Path(directory) / 'run'
      run.mkdir()
      with mock.patch.object(analyzer, '_RUN_DIR', run):
        with self.assertRaisesRegex(
            analyzer.AnalysisError, 'internal post-START'
        ):
          analyzer.analyze(run, bundle_root=analyzer._REPO_ROOT)  # pylint: disable=protected-access

  def test_internal_gate_revalidates_exact_singleton_start_and_run_hashes(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      run, attempt, analysis = root / 'run', root / 'attempt', root / 'analysis'
      run.mkdir()
      attempt.mkdir()
      freeze = root / 'freeze.json'
      freeze.write_text('{}\n', encoding='utf-8')
      for name in ('ATTEMPT_STARTED.json', 'RUN_COMPLETE.json', 'RAW_MANIFEST.json'):
        (run / name).write_text(f'{{"name":"{name}"}}\n', encoding='utf-8')
      start = {
          'analysis_version': analyzer.ANALYSIS_VERSION,
          'status': 'analysis_started_append_only_one_shot',
          'amendment': {
              'path': str(analyzer._AMENDMENT_PATH.resolve()),  # pylint: disable=protected-access
              'sha256': analyzer.AMENDMENT_SHA256,
              'commit': analyzer.AMENDMENT_COMMIT,
          },
          'analyzer': {
              'path': str(Path(analyzer.__file__).resolve()),
              'sha256': analyzer._sha256(Path(analyzer.__file__)),  # pylint: disable=protected-access
          },
          'freeze': {
              'path': str(freeze.resolve()),
              'sha256': analyzer._sha256(freeze),  # pylint: disable=protected-access
          },
          'run_dir': str(run.resolve()),
          'analysis_attempt_dir': str(attempt.resolve()),
          'analysis_dir': str(analysis.resolve()),
          'output_json': str((analysis / 'ANALYSIS.json').resolve()),
          'output_markdown': str((analysis / 'RESULT.md').resolve()),
          'run_artifacts': {
              name: {
                  'path': str((run / name).resolve()),
                  'sha256': analyzer._sha256(run / name),  # pylint: disable=protected-access
                  'size_bytes': (run / name).stat().st_size,
              }
              for name in (
                  'ATTEMPT_STARTED.json', 'RUN_COMPLETE.json',
                  'RAW_MANIFEST.json',
              )
          },
          'raw_scientific_endpoint_evidence_reached': False,
          'scientific_summary_computed': False,
          'donor_normalization_computed': False,
          'shapley_or_nomination_computed': False,
          'interaction_or_resolution_computed': False,
          'nomination_performed': False,
          'confirmation_model_outputs_activations_interventions_unopened': True,
          'started_at_unix_s': 1.0,
      }
      path = attempt / 'ANALYSIS_ATTEMPT_STARTED.json'
      path.write_text(json.dumps(start, sort_keys=True) + '\n', encoding='utf-8')
      with (
          mock.patch.object(analyzer, '_ANALYSIS_ATTEMPT_DIR', attempt),
          mock.patch.object(analyzer, '_ANALYSIS_DIR', analysis),
          mock.patch.object(analyzer, '_FREEZE_PATH', freeze),
      ):
        audit = analyzer._validate_active_analysis_attempt(  # pylint: disable=protected-access
            run, token=analyzer._ANALYSIS_ATTEMPT_TOKEN,  # pylint: disable=protected-access
            started_sha256=analyzer._sha256(path),  # pylint: disable=protected-access
        )
        self.assertEqual(audit['run_dir'], str(run.resolve()))
        (run / 'RAW_MANIFEST.json').write_text('{}\n', encoding='utf-8')
        with self.assertRaisesRegex(analyzer.AnalysisError, 'run binding'):
          analyzer._validate_active_analysis_attempt(  # pylint: disable=protected-access
              run, token=analyzer._ANALYSIS_ATTEMPT_TOKEN,  # pylint: disable=protected-access
              started_sha256=analyzer._sha256(path),  # pylint: disable=protected-access
          )

  def test_success_binds_both_outputs_and_remains_structural_only(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      run, attempt, analysis = root / 'run', root / 'attempt', root / 'analysis'
      run.mkdir()
      result = {
          'analysis_version': analyzer.ANALYSIS_VERSION,
          'decision': 'controlled_stop_compiler_failure',
          'controlled_stop': {'reason': 'compiler_failure'},
          'scientific_summary_computed': False,
          'donor_normalization_computed': False,
          'shapley_or_nomination_computed': False,
          'interaction_or_resolution_computed': False,
          'nomination_performed': False, 'nomination': None,
          'resolution_analysis': None, 'combined_analysis_permitted': False,
      }
      with (
          mock.patch.object(analyzer, '_ANALYSIS_ATTEMPT_DIR', attempt),
          mock.patch.object(analyzer, '_ANALYSIS_DIR', analysis),
          mock.patch.object(
              analyzer, '_analysis_attempt_precheck',
              return_value=self._started(attempt, analysis, run),
          ),
          mock.patch.object(analyzer, 'analyze', return_value=result),
          mock.patch.object(sys, 'argv', self._argv(run, analysis)),
      ):
        analyzer.main()
      self.assertEqual(
          {path.name for path in attempt.iterdir()},
          {'ANALYSIS_ATTEMPT_STARTED.json', 'ANALYSIS_COMPLETE.json'},
      )
      complete = json.loads(
          (attempt / 'ANALYSIS_COMPLETE.json').read_text(encoding='utf-8')
      )
      self.assertEqual(set(complete['outputs']), {'ANALYSIS.json', 'RESULT.md'})
      self.assertFalse(complete['raw_scientific_endpoint_evidence_reached'])
      self.assertFalse(complete['scientific_summary_computed'])
      self.assertFalse(complete['shapley_or_nomination_computed'])
      for name, binding in complete['outputs'].items():
        path = analysis / name
        self.assertEqual(binding['sha256'], analyzer._sha256(path))  # pylint: disable=protected-access
        self.assertEqual(binding['size_bytes'], path.stat().st_size)


if __name__ == '__main__':
  unittest.main()
