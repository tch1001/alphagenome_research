"""Runner-shaped CPU integration tests for the v3.2 offline analyzer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


_MODULE_PATH = Path(__file__).with_name('analyze_superset_graph_v3_2.py')
_SPEC = importlib.util.spec_from_file_location(
    'analyze_superset_graph_v3_2', _MODULE_PATH
)
analyzer = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = analyzer
_SPEC.loader.exec_module(analyzer)


def _write(path: Path, value) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n',
      encoding='utf-8',
  )


def _write_text(path: Path, value: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(value, encoding='utf-8')


def _fingerprint(seed: str = 'trace') -> dict:
  return {
      'sha256': __import__('hashlib').sha256(seed.encode()).hexdigest(),
      'leaves': [{'shape': [1], 'dtype': 'float32'}],
  }


def _readout(values) -> dict:
  logits, margins, totals, means = [], [], [], []
  for index, raw in enumerate(values):
    value = analyzer._f32(raw, f'synthetic[{index}]')
    row_logits = [[value, 0.0], [value, 0.0]]
    row_margins = [value, value]
    total = analyzer._f32(value + value)
    mean = analyzer._f32(total / 2.0)
    logits.append(row_logits)
    margins.append(row_margins)
    totals.append(total)
    means.append(mean)
  return {
      'endpoint_axis': ['acceptor', 'donor'],
      'selected_logit_axis': ['relevant_class', 'padding_class'],
      'selected_logits': logits,
      'endpoint_margins': margins,
      'means': means,
      'totals': totals,
      'num_values': 2,
  }


def _values(ref: float, alt: float, recovery: float | None = None):
  ref, alt = analyzer._f32(ref), analyzer._f32(alt)
  if recovery is None:
    ref_alt, alt_ref = alt, ref
  else:
    ref_alt = analyzer._f32(alt + recovery * (ref - alt))
    alt_ref = analyzer._f32(ref + recovery * (alt - ref))
  return [ref, alt, ref_alt, alt, alt_ref, ref]


def _target_map(readout: dict) -> dict:
  return dict(zip(analyzer.TRACE_ROLES, readout['means'], strict=True))


def _metrics(readout: dict) -> tuple[dict, dict]:
  ref, alt, ref_alt, alt_alt, alt_ref, ref_ref = readout['means']
  raw = {
      'reference_into_alternate': ref_alt - alt_alt,
      'alternate_into_reference': alt_ref - ref_ref,
  }
  if ref == alt:
    recovery = {
        'reference_into_alternate': None,
        'alternate_into_reference': None,
        'bidirectional_bottleneck': None,
    }
  else:
    forward = raw['reference_into_alternate'] / (ref - alt)
    reciprocal = raw['alternate_into_reference'] / (alt - ref)
    recovery = {
        'reference_into_alternate': forward,
        'alternate_into_reference': reciprocal,
        'bidirectional_bottleneck': min(forward, reciprocal),
    }
  return raw, recovery


def _position_set(name: str) -> dict:
  candidate = name.split('_control_', maxsplit=1)[0]
  control = '_control_' in name
  count = 2 if candidate == 'S' else 1
  return {
      'name': name,
      'tokens': list(range(10, 10 + count)),
      'slots': list(range(count)),
      'role': 'width_matched_control' if control else 'candidate',
      'matched_candidate': candidate if control else None,
      'genomic_intervals': [[100, 228]],
  }


def _observation(with_v3_environment: bool = False) -> dict:
  result = {
      'jax_default_backend': 'gpu',
      'jax_gpu_devices': [{
          'device_kind': analyzer.EXPECTED_DEVICE_KIND,
          'platform': 'gpu',
          'client_platform': 'gpu',
          'id': 0,
      }],
      'nvidia_smi': {'parsed_single_gpu': {
          'name': analyzer.EXPECTED_DEVICE_KIND,
          'uuid': analyzer.EXPECTED_GPU_UUID,
          'compute_capability': analyzer.EXPECTED_COMPUTE_CAPABILITY,
      }},
      'environment': {
          'LD_LIBRARY_PATH': {'present': False, 'value': None},
          'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
      },
  }
  if with_v3_environment:
    result['v3_2_runtime_environment'] = {
        'JAX_ENABLE_COMPILATION_CACHE': 'false',
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
    }
    result['jax_enable_compilation_cache'] = False
  return result


def _identity_record(case: dict, common: dict, *, valid: bool = True) -> dict:
  effect = 'neutral' not in case['selection_class'].lower()
  ref = 0.0
  alt = 1.0 if case['delta_logit'] > 0 else -1.0 if effect else 0.0
  readout = _readout(_values(ref, alt))
  center = (case['exon_start_1based'] + case['exon_end_1based']) // 2
  start = center - 1 - 8192
  if case['strand'] == '+':
    endpoints = (
        ('acceptor', case['exon_start_1based'], 1),
        ('donor', case['exon_end_1based'], 0),
    )
  else:
    endpoints = (
        ('acceptor', case['exon_end_1based'], 3),
        ('donor', case['exon_start_1based'], 2),
    )
  checks = {
      'passed': True,
      'target_means': _target_map(readout),
      'target_repeat_exact': True,
      'target_duplicates_exact': True,
      'trace_repeat_exact': True,
      'natural_duplicates_exact': True,
      'num_values': 2,
      'all_false_natural_effective_exact': True,
      'target_total_equals_two_times_mean': True,
  } if valid else None
  delta = analyzer._f32(alt - ref)
  same_sign = effect and ((delta > 0) == (case['delta_logit'] > 0))
  return {
      **common,
      'status': 'complete' if valid else 'invalid',
      'script_version': analyzer.SOURCE_SCRIPT_VERSION,
      'case': case,
      'interval': {
          'chromosome': case['chromosome'],
          'start_0based': start,
          'end_0based_exclusive': start + 16_384,
      },
      'sequence_sha256': {'reference': 'a' * 64, 'alternate': 'b' * 64},
      'resolved_position_sets': [
          _position_set(name) for name in analyzer.POSITION_SETS
      ],
      'canonical_target': {
          'endpoints': [{
              'role': role,
              'position_1based': position,
              'position_index': position - 1 - start,
              'track_index': track,
          } for role, position, track in endpoints],
          'padding_track_index': 4,
      },
      'target_readout': readout,
      'repeat_target_readout': readout,
      'trace_fingerprint': _fingerprint(case['variant_id']),
      'repeat_trace_fingerprint': _fingerprint(case['variant_id']),
      'checks': checks,
      'failure': None if valid else {
          'type': 'ValueError', 'message': 'synthetic identity tensor failure'
      },
      'direction_gate': {
          'predicted_alt_minus_ref_logit_margin': delta,
          'experimental_delta_logit': case['delta_logit'],
          'minimum_absolute_predicted_effect': analyzer.EFFECT_THRESHOLD,
          'direction_matches_delta_logit': same_sign if effect else None,
          'eligible_for_causal_census': bool(same_sign and abs(delta) >= 0.01),
      } if valid else None,
      'program_signatures': {
          'selection': {'shape': 'fixed'},
          'interventions': {'shape': 'fixed'},
          'target': {'shape': 'fixed'},
      },
  }


def _active_record(
    case: dict, common: dict, identity_binding: dict, readout: dict, *,
    family: str, group: dict | None = None, component: dict | None = None,
    valid: bool = True,
) -> dict:
  raw, recovery = _metrics(readout)
  checks = {
      'passed': True,
      'target_means': _target_map(readout),
      'baseline_targets_exact_from_identity': True,
      'self_targets_exact': True,
      'target_repeat_exact': True,
      'trace_repeat_exact': True,
      'raw_movement': raw,
      'recovery': recovery,
  }
  if family == 'phase_r':
    checks.update({
        'selected_donor_vectors_exact': True,
        'active_seam_natural_same_allele_exact': True,
        'baseline_rows_active_seam_natural_effective_exact': True,
        'disabled_seams_exact': True,
    })
  else:
    assert component is not None
    checks['transformer_residual_seams_disabled_exact'] = True
    checks['baseline_rows_T_natural_effective_exact'] = True
    checks['baseline_rows_E_natural_effective_exact'] = True
    checks['baseline_rows_final_A_D_natural_effective_exact'] = True
    if component['transformer_output']:
      checks['transformer_natural_self_tensors_exact'] = True
      checks['transformer_effective_donor_tensors_exact'] = True
    else:
      checks['transformer_disabled_natural_effective_exact'] = True
    if component['encoder_skips']:
      checks['all_seven_skip_natural_self_tensors_exact'] = True
      checks['all_seven_skip_effective_donor_tensors_exact'] = True
    else:
      checks['all_seven_skips_disabled_natural_effective_exact'] = True
    checks['final_embedding_donor_vectors_exact'] = (
        True if component['final_embedding'] else None
    )
    checks['closure'] = ({
        'reference_into_alternate_target_equals_donor': True,
        'alternate_into_reference_target_equals_donor': True,
        'passed': True,
    } if component['closure_required'] else None)
  trace = _fingerprint(f"{case['variant_id']}_{family}_{group or component}")
  return {
      **common,
      'status': 'complete' if valid else 'invalid',
      'script_version': analyzer.SOURCE_SCRIPT_VERSION,
      'identity_binding': identity_binding,
      'case': case,
      'family': family,
      'group': group,
      'component': component,
      'target_readout': readout,
      'repeat_target_readout': readout,
      'trace_fingerprint': trace,
      'repeat_trace_fingerprint': trace,
      'checks': checks if valid else None,
      'failure': None if valid else {
          'type': 'ValueError', 'message': 'synthetic donor tensor failure'
      },
      'same_compiled_executable': True,
  }


class SyntheticRun:

  def __init__(
      self, root: Path, *, identity_stop: bool = False,
      closure_stop: str | None = None,
  ):
    self.root = root
    self.bundle = root.parent / 'bundle'
    self.analysis_dir = root.parent / 'analysis'
    self.preflight_dir = root.parent / 'preflight'
    self.executable = 'e' * 64
    self.cases = analyzer._load_frozen_cases()
    self._make_provenance()
    self._make_raw(identity_stop=identity_stop, closure_stop=closure_stop)
    self.refresh(
        status='controlled_stop' if (identity_stop or closure_stop) else 'complete',
        stop_reason=(
            'identity_tooling_failure' if identity_stop else
            'closure_tooling_failure' if closure_stop else None
        ),
    )

  def _make_provenance(self):
    bound = self.bundle / 'bound.txt'
    _write_text(bound, 'bound\n')
    checkpoint = self.bundle / analyzer.CHECKPOINT_SNAPSHOT
    checkpoint_records = []
    for index in range(12):
      relative = f'data/part_{index:02d}.bin'
      path = checkpoint / relative
      _write_text(path, f'synthetic checkpoint part {index}\n')
      checkpoint_records.append({
          'relative_path': relative,
          'size_bytes': path.stat().st_size,
          'sha256': analyzer._sha256(path),
      })
    checkpoint_manifest = self.bundle / 'checkpoint_manifest_v3_2.tsv'
    _write_text(
        checkpoint_manifest,
        ''.join(
            f"{row['relative_path']}\t{row['size_bytes']}\t{row['sha256']}\n"
            for row in checkpoint_records
        ),
    )
    reference_cases = {}
    for case in self.cases:
      center = (case['exon_start_1based'] + case['exon_end_1based']) // 2
      start = center - 1 - 8192
      reference_cases[case['variant_id']] = [
          case['order'], case['chromosome'], start, start + 16_384,
          'a' * 64, 'b' * 64,
      ]
    reference_binding = {
        'reference_url': analyzer._REFERENCE_URL,
        'context_bp': 16_384,
        'cases': reference_cases,
    }
    reference_binding_path = self.bundle / 'reference_bindings_v3_2.json'
    _write(reference_binding_path, reference_binding)
    launcher = (
        self.bundle / 'experiments/interpretability/opensplice/'
        'launch_superset_graph_v3_2.py'
    )
    bootstrap = (
        self.bundle / 'experiments/interpretability/opensplice/'
        'validate_superset_graph_bootstrap_v3_2.py'
    )
    _write_text(launcher, '# synthetic launcher\n')
    _write_text(bootstrap, '# synthetic bootstrap\n')
    module_specs = (
        ('alphagenome_research.model.model', 'src/model.py'),
        ('alphagenome_research.model.dna_model', 'src/dna_model.py'),
        ('target_reducers_v3', 'experiments/target_reducers_v3.py'),
        ('__main__', 'experiments/run_superset_graph_v3_2.py'),
    )
    modules = []
    for name, relative in module_specs:
      path = self.bundle / relative
      _write_text(path, f'# {name}\n')
      modules.append({
          'name': name,
          'path': str(path.resolve()),
          'root': 'alphagenome_research_checkout',
          'sha256': analyzer._sha256(path),
          'size_bytes': path.stat().st_size,
      })
    imports = {'module_count': len(modules), 'modules': modules}
    _write(self.root / 'IMPORT_PROVENANCE.json', imports)
    _write(self.root / 'IMPORT_PROVENANCE_PRE_MODEL.json', imports)
    _write(self.root / 'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json', imports)

    upstream = self.bundle.parent / 'alphagenome'
    artifact_paths = {
        'source_proto': self.bundle / 'src' / 'binding.proto',
        'generated_pb2': self.bundle / 'src' / 'binding_pb2.py',
        'generated_pyi': self.bundle / 'src' / 'binding_pb2.pyi',
        'dependency_proto': upstream / 'src' / 'dependency.proto',
        'dependency_pb2': upstream / 'src' / 'dependency_pb2.py',
        'tensor_proto': upstream / 'src' / 'tensor.proto',
        'tensor_pb2': upstream / 'src' / 'tensor_pb2.py',
    }
    for name, path in artifact_paths.items():
      _write_text(
          path,
          '# Protobuf Python Version: 7.35.1\n' if name == 'generated_pb2'
          else f'# {name}\n',
      )
    generated_records = {
        name: {
            'path': str(path.resolve()),
            'sha256': analyzer._sha256(path),
            'size_bytes': path.stat().st_size,
        }
        for name, path in artifact_paths.items()
    }
    protobuf = {
        'historical_generation_provenance': 'unknown_not_reconstructed',
        'regeneration_claim': False,
        'current_protoc_was_used_to_generate_frozen_outputs': False,
        'source_proto': {
            **generated_records['source_proto']
        },
        'dependency_proto': {**generated_records['dependency_proto']},
        'dependency_pb2': {**generated_records['dependency_pb2']},
        'generated_outputs': {
            str(artifact_paths['generated_pb2'].resolve()): {
                **generated_records['generated_pb2']
            },
            str(artifact_paths['generated_pyi'].resolve()): {
                **generated_records['generated_pyi']
            },
        },
        'imported_pb2': {**generated_records['generated_pb2']},
    }
    _write(self.root / 'PROTOBUF_PROVENANCE.json', protobuf)

    compiler_artifacts = {}
    for name, filename in (
        ('stablehlo', 'superset.stablehlo.mlir'),
        ('hlo', 'superset.pre_backend.hlo.txt'),
        ('compiled_hlo', 'superset.compiled.hlo.txt'),
    ):
      path = self.root / 'compiler' / filename
      _write_text(path, f'synthetic {name}\n')
      compiler_artifacts[name] = {
          'path': str(path.resolve()),
          'sha256': analyzer._sha256(path),
          'size_bytes': path.stat().st_size,
      }
    self.executable = __import__('hashlib').sha256(
        bytes.fromhex(compiler_artifacts['compiled_hlo']['sha256'])
    ).hexdigest()
    compiler = {
        'compile_count': 1,
        'compile_seconds': 1.0,
        'executable_fingerprint': self.executable,
        'artifacts': compiler_artifacts,
    }
    _write(self.root / 'compiler' / 'COMPILER_PROVENANCE.json', compiler)

    freeze = {
        'script_version': analyzer.SOURCE_SCRIPT_VERSION,
        'attempt_id': analyzer.ATTEMPT_ID,
        'protocol_sha256': analyzer.PROTOCOL_SHA256,
        'development_variants_sha256': analyzer.DEVELOPMENT_VARIANTS_SHA256,
        'development_exons_sha256': analyzer.DEVELOPMENT_EXONS_SHA256,
        'selected_variants_sha256': analyzer.SELECTED_VARIANTS_SHA256,
        'frozen_exons_sha256': analyzer.FROZEN_EXONS_SHA256,
        'development_variants_path': str(analyzer._SELECTED_PATH.resolve()),
        'development_exons_path': str(analyzer._EXONS_PATH.resolve()),
        'checkpoint_snapshot': analyzer.CHECKPOINT_SNAPSHOT,
        'context_bp': 16_384,
        'attention_backend': 'dense',
        'preflight_script_version': 'opensplice-device-preflight-v3.2.0',
        'reference_url': analyzer._REFERENCE_URL,
        'reference_object': analyzer._REFERENCE_OBJECT,
        'reference_bindings_path': str(reference_binding_path.resolve()),
        'reference_bindings_sha256': analyzer._sha256(reference_binding_path),
        'checkpoint_manifest_path': str(checkpoint_manifest.resolve()),
        'checkpoint_manifest_sha256': analyzer._sha256(checkpoint_manifest),
        'expected_device_kind': analyzer.EXPECTED_DEVICE_KIND,
        'expected_gpu_uuid': analyzer.EXPECTED_GPU_UUID,
        'expected_compute_capability': analyzer.EXPECTED_COMPUTE_CAPABILITY,
        'upstream_alphagenome_git_head': '2' * 40,
        'mixed_precision_policy': (
            'params=float32,compute=bfloat16,output=bfloat16'
        ),
        'environment_contract': {
            'LD_LIBRARY_PATH': 'absent',
            'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
            'JAX_ENABLE_COMPILATION_CACHE': 'false',
            'compiler_and_autotune_cache_inputs': 'absent',
        },
        'paired_batch_roles': list(analyzer.TRACE_ROLES),
        'paired_batch_donor_rows': [0, 1, 0, 1, 1, 0],
        'phase_r_groups_per_eligible_effect': 216,
        'phase_r_candidate_count': 72,
        'stage_a_component_keys': list(analyzer.STAGE_COMPONENTS),
        'output_dir': str(self.root.resolve()),
        'analysis_dir': str(self.analysis_dir.resolve()),
        'preflight_dir': str(self.preflight_dir.resolve()),
        'protobuf_binding': protobuf,
        'file_sha256': {
            'bound.txt': analyzer._sha256(bound),
            str(launcher.relative_to(self.bundle)): analyzer._sha256(launcher),
            str(bootstrap.relative_to(self.bundle)): analyzer._sha256(bootstrap),
        },
    }
    freeze_path = self.bundle / 'freeze.json'
    _write(freeze_path, freeze)
    freeze_sha = analyzer._sha256(freeze_path)
    git_head = '1' * 40
    tracked_paths = sorted({
        str(freeze_path.relative_to(self.bundle)),
        str(analyzer._PROTOCOL_PATH.relative_to(analyzer._REPO_ROOT)),
        *freeze['file_sha256'].keys(),
    })

    stdout = self.preflight_dir / 'preflight_0000.stdout.log'
    stderr = self.preflight_dir / 'preflight_0000.stderr.log'
    _write_text(stdout, '')
    _write_text(stderr, '')
    preflight = {
        'script_version': 'opensplice-device-preflight-v3.2.0',
        'status': 'pass',
        'protocol_sha256': analyzer.PROTOCOL_SHA256,
        'freeze_sha256': freeze_sha,
        'freeze': {
            'path': str(freeze_path.resolve()),
            'sha256': freeze_sha,
            'git_head': git_head,
            'tracked_clean': True,
        },
        'observation': _observation(with_v3_environment=True),
        'failure': None,
        'warnings': [],
        'logs': {
            'stdout': {'path': str(stdout), 'sha256': analyzer._sha256(stdout)},
            'stderr': {'path': str(stderr), 'sha256': analyzer._sha256(stderr)},
        },
        'no_model_or_biological_access': True,
        'no_jit_or_array_kernel': True,
    }
    preflight_path = self.preflight_dir / 'preflight_0000.json'
    _write(preflight_path, preflight)
    start = {
        'attempt_id': analyzer.ATTEMPT_ID,
        'script_version': analyzer.SOURCE_SCRIPT_VERSION,
        'status': 'started_append_only_one_shot',
        'protocol_sha256': analyzer.PROTOCOL_SHA256,
        'freeze': {**freeze, 'path': str(freeze_path), 'sha256': freeze_sha},
        'bundle': {'git_head': git_head, 'tracked_clean': True},
        'external_preflight': {
            'path': str(preflight_path),
            'sha256': analyzer._sha256(preflight_path),
            **preflight,
        },
        'same_process_preflight': _observation(),
        'runtime_environment': {
            'JAX_ENABLE_COMPILATION_CACHE': 'false',
            'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
        },
        'same_process_pre_import_bootstrap': {
            'pid': 12345,
            'created_at_unix_s': 1.0,
            'generated_bindings': {
                'pre_import_gate': True,
                'historical_generator_argv': 'unknown',
                'exact_regeneration_claim': False,
                'generated_artifact_exception': [
                    '?? src/alphagenome_research/protos/calibration_scores_pb2.py',
                    '?? src/alphagenome_research/protos/calibration_scores_pb2.pyi',
                ],
                'artifacts': generated_records,
                'embedded_header': ['# Protobuf Python Version: 7.35.1'],
                'protobuf_runtime_version': '7.35.1',
            },
            'freeze': {
                'path': str(freeze_path),
                'sha256': freeze_sha,
                'git_head': git_head,
                'tracked_head_clean': True,
                'tracked_paths': tracked_paths,
            },
            'launcher_path': str(launcher.resolve()),
            'launcher_sha256': analyzer._sha256(launcher),
            'bootstrap_path': str(bootstrap.resolve()),
            'bootstrap_sha256': analyzer._sha256(bootstrap),
        },
        'checkpoint_path': str(checkpoint.resolve()),
        'checkpoint_binding': {
            'snapshot_path': str(checkpoint.resolve()),
            'snapshot_name': analyzer.CHECKPOINT_SNAPSHOT,
            'manifest_path': str(checkpoint_manifest.resolve()),
            'manifest_sha256': analyzer._sha256(checkpoint_manifest),
            'file_count': 12,
            'files': checkpoint_records,
        },
        'reference_object_binding': {
            **analyzer._REFERENCE_OBJECT,
            'observed_generation': analyzer._REFERENCE_OBJECT['generation'],
            'observed_size_bytes': analyzer._REFERENCE_OBJECT['size_bytes'],
            'observed_etag': analyzer._REFERENCE_OBJECT['etag'],
            'observed_md5_base64': analyzer._REFERENCE_OBJECT['md5_base64'],
            'observed_crc32c_base64': analyzer._REFERENCE_OBJECT['crc32c_base64'],
        },
        'reference_sequence_bindings': {
            'path': str(reference_binding_path.resolve()),
            'sha256': analyzer._sha256(reference_binding_path),
        },
        'compile_count_contract': 1,
        'confirmation_model_calls': 0,
        'confirmation_scope_disclosure': analyzer._DISCLOSURE,
    }
    _write(self.root / 'ATTEMPT_STARTED.json', start)
    self.freeze_sha = freeze_sha
    self.common = {
        'protocol_sha256': analyzer.PROTOCOL_SHA256,
        'freeze_sha256': freeze_sha,
        'executable_fingerprint': self.executable,
    }

  def _make_raw(self, *, identity_stop: bool, closure_stop: str | None):
    self.identities = {}
    for case in self.cases:
      valid = not identity_stop or case['order'] != 0
      record = _identity_record(case, self.common, valid=valid)
      path = self.root / 'raw' / 'identity' / (
          f"{case['order']:03d}_{analyzer._slug(case['variant_id'])}.json"
      )
      _write(path, record)
      self.identities[case['variant_id']] = (record, path)
    if identity_stop:
      return
    eligible = []
    for case in self.cases:
      record = self.identities[case['variant_id']][0]
      if record['direction_gate']['eligible_for_causal_census']:
        eligible.append(case)
    _write(self.root / 'TARGET_ELIGIBILITY.json', {
        'eligible_effects': [case['variant_id'] for case in eligible],
        'ineligible_effects': [],
        'neutral_controls': [
            case['variant_id'] for case in self.cases
            if 'neutral' in case['selection_class'].lower()
        ],
        'eligible_effects_per_gene': {'BRAF': 6, 'SLC25A48': 6},
    })
    for case in eligible:
      identity, identity_path = self.identities[case['variant_id']]
      binding = {
          'path': str(identity_path.relative_to(self.root)),
          'sha256': analyzer._sha256(identity_path),
      }
      ref, alt = identity['target_readout']['means'][:2]
      case_dir = self.root / 'raw' / 'phase_r' / (
          f"{case['order']:03d}_{analyzer._slug(case['variant_id'])}"
      )
      order = 0
      for stage in analyzer.STAGES:
        for layer in analyzer.LAYERS:
          for name in analyzer.POSITION_SETS:
            recovery = (
                0.1 if '_control_' in name else
                0.6 if (stage, layer, name) == ('pre_attention', 0, 'V')
                else 0.3
            )
            position = _position_set(name)
            group = {
                'order': order,
                'stage': stage,
                'layer': layer,
                'position_set': position,
                'is_candidate': name in analyzer.CANDIDATES,
            }
            record = _active_record(
                case, self.common, binding,
                _readout(_values(ref, alt, recovery)),
                family='phase_r', group=group,
            )
            _write(
                case_dir / f'{order:03d}_{stage}_layer{layer:02d}_{name}.json',
                record,
            )
            order += 1
    effects = [
        case for case in self.cases
        if 'neutral' not in case['selection_class'].lower()
    ]
    for component_key in analyzer.STAGE_COMPONENTS:
      if closure_stop == 'final' and component_key != analyzer.STAGE_COMPONENTS[0]:
        continue
      if closure_stop == 'joint' and component_key in analyzer.STAGE_COMPONENTS[2:]:
        continue
      component = analyzer._expected_component(component_key)
      recovery = {
          analyzer.STAGE_COMPONENTS[0]: 1.0,
          analyzer.STAGE_COMPONENTS[1]: 1.0,
          analyzer.STAGE_COMPONENTS[2]: 0.7,
          analyzer.STAGE_COMPONENTS[3]: 0.3,
      }[component_key]
      selected = effects if component['closure_required'] else eligible
      for selected_index, case in enumerate(selected):
        identity, identity_path = self.identities[case['variant_id']]
        binding = {
            'path': str(identity_path.relative_to(self.root)),
            'sha256': analyzer._sha256(identity_path),
        }
        ref, alt = identity['target_readout']['means'][:2]
        record = _active_record(
            case, self.common, binding,
            _readout(_values(ref, alt, recovery)),
            family='stage_a', component=component,
            valid=not (
                selected_index == 0
                and (
                    closure_stop == 'final'
                    and component_key == analyzer.STAGE_COMPONENTS[0]
                    or closure_stop == 'joint'
                    and component_key == analyzer.STAGE_COMPONENTS[1]
                )
            ),
        )
        _write(
            self.root / 'raw' / 'stage_a' / component_key
            / f"{case['order']:03d}_{analyzer._slug(case['variant_id'])}.json",
            record,
        )

  def refresh(self, *, status: str | None = None, stop_reason=...):
    raw_paths = sorted((self.root / 'raw').rglob('*.json'))
    manifest = {
        'artifact_count': len(raw_paths),
        'artifact_sha256': {
            str(path.relative_to(self.root)): analyzer._sha256(path)
            for path in raw_paths
        },
        'artifact_tree_sha256': analyzer._tree_digest(raw_paths, self.root),
    }
    _write(self.root / 'RAW_MANIFEST.json', manifest)
    complete_path = self.root / 'RUN_COMPLETE.json'
    old = json.loads(complete_path.read_text()) if complete_path.exists() else {}
    if status is None:
      status = old.get('status', 'complete')
    if stop_reason is ...:
      stop_reason = old.get('stop_reason')
    phase = list((self.root / 'raw' / 'phase_r').glob('*/*.json'))
    stage = list((self.root / 'raw' / 'stage_a').glob('*/*.json'))
    phase_invalid = sum(json.loads(path.read_text())['status'] != 'complete' for path in phase)
    stage_invalid = sum(json.loads(path.read_text())['status'] != 'complete' for path in stage)
    eligible = 0 if stop_reason == 'identity_tooling_failure' else 12
    record = {
        'status': status,
        'stop_reason': stop_reason,
        'message': 'synthetic complete/controlled stop',
        'attempt_id': analyzer.ATTEMPT_ID,
        'script_version': analyzer.SOURCE_SCRIPT_VERSION,
        'protocol_sha256': analyzer.PROTOCOL_SHA256,
        'identity_count': 20,
        'eligible_effect_count': eligible,
        'phase_r_group_count': len(phase),
        'phase_r_invalid_count': phase_invalid,
        'stage_a_group_count': len(stage),
        'stage_a_invalid_count': stage_invalid,
        'closures_passed': None if stop_reason in {
            'identity_tooling_failure', 'target_predictive_failure'
        } else stage_invalid == 0,
        'single_executable': json.loads(
            (self.root / 'compiler' / 'COMPILER_PROVENANCE.json').read_text()
        ),
        'import_provenance_sha256': analyzer._sha256(
            self.root / 'IMPORT_PROVENANCE.json'
        ),
        'import_provenance_phases': {
            'pre_model': analyzer._sha256(
                self.root / 'IMPORT_PROVENANCE_PRE_MODEL.json'
            ),
            'post_model_precompile': analyzer._sha256(
                self.root / 'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json'
            ),
            'postcompile': analyzer._sha256(
                self.root / 'IMPORT_PROVENANCE.json'
            ),
        },
        'protobuf_provenance_sha256': analyzer._sha256(
            self.root / 'PROTOBUF_PROVENANCE.json'
        ),
        'raw_manifest': manifest,
        'confirmation_model_calls': 0,
        'confirmation_scope_disclosure': analyzer._DISCLOSURE,
    }
    _write(complete_path, record)


class SupersetAnalyzerIntegrationTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.temporary = tempfile.TemporaryDirectory(prefix='v3_2_analyzer_')
    cls.fixture = SyntheticRun(Path(cls.temporary.name) / 'development_run')

  @classmethod
  def tearDownClass(cls):
    cls.temporary.cleanup()
    super().tearDownClass()

  def analyze(self):
    return analyzer.analyze(
        self.fixture.root, bundle_root=self.fixture.bundle,
        enforce_standard_locations=False,
    )

  def test_complete_runner_shaped_tree_recomputes_all_estimands(self):
    result = self.analyze()
    self.assertEqual(result['audit']['identity_count'], 20)
    self.assertEqual(result['audit']['eligible_effect_count'], 12)
    self.assertEqual(result['phase_r']['group_count'], 12 * 216)
    self.assertFalse(result['phase_r']['family_tooling_failure'])
    first = result['phase_r']['first_passing_candidate']
    self.assertEqual(
        (first['stage'], first['layer'], first['position_set']),
        ('pre_attention', 0, 'V'),
    )
    self.assertAlmostEqual(first['Q'], 0.5)
    self.assertTrue(result['stage_a']['closures_pass'])
    self.assertTrue(result['stage_a']['isolated_route_complete'])
    self.assertEqual(len(result['target_behavior']['all_twenty_identities']), 20)
    self.assertEqual(len(result['target_behavior']['neutral_behavior_controls']), 8)
    self.assertTrue(result['audit']['external_preflight_logs_verified'])
    self.assertTrue(result['audit']['protobuf_binding_verified'])
    markdown = analyzer.render_markdown(result)
    self.assertIn('metadata/labels had been exposed post-freeze', markdown)

  def test_raw_endpoint_tampering_fails_independent_reducer(self):
    path = next((self.fixture.root / 'raw' / 'phase_r').glob('*/*.json'))
    saved = path.read_bytes()
    value = json.loads(saved)
    value['target_readout']['endpoint_margins'][2][0] += 0.5
    _write(path, value)
    self.fixture.refresh()
    try:
      with self.assertRaisesRegex(ValueError, 'recomputed value'):
        self.analyze()
    finally:
      path.write_bytes(saved)
      self.fixture.refresh()

  def test_missing_group_is_never_dropped(self):
    path = next((self.fixture.root / 'raw' / 'phase_r').glob('*/*.json'))
    saved = path.read_bytes()
    path.unlink()
    self.fixture.refresh()
    try:
      with self.assertRaisesRegex(ValueError, 'grid is incomplete'):
        self.analyze()
    finally:
      path.write_bytes(saved)
      self.fixture.refresh()

  def test_scalar_equal_but_endpoint_unequal_closure_fails(self):
    path = sorted(
        (self.fixture.root / 'raw' / 'stage_a' / analyzer.STAGE_COMPONENTS[0]).glob('*.json')
    )[0]
    saved = path.read_bytes()
    value = json.loads(saved)
    for field in ('target_readout', 'repeat_target_readout'):
      value[field]['selected_logits'][2] = [[0.5, 0.0], [-0.5, 0.0]]
      value[field]['endpoint_margins'][2] = [0.5, -0.5]
      value[field]['totals'][2] = 0.0
      value[field]['means'][2] = 0.0
    _write(path, value)
    self.fixture.refresh()
    try:
      with self.assertRaisesRegex(ValueError, 'endpoint_level_closure_failed'):
        self.analyze()
    finally:
      path.write_bytes(saved)
      self.fixture.refresh()

  def test_invalid_candidate_group_is_not_dropped_from_candidate(self):
    path = next(
        (self.fixture.root / 'raw' / 'phase_r').glob(
            '*/000_pre_attention_layer00_V.json'
        )
    )
    saved = path.read_bytes()
    value = json.loads(saved)
    value['status'] = 'invalid'
    value['checks'] = None
    value['failure'] = {'type': 'ValueError', 'message': 'synthetic'}
    _write(path, value)
    self.fixture.refresh()
    try:
      result = self.analyze()
      affected = next(
          row for row in result['phase_r']['rankings']
          if (row['stage'], row['layer'], row['position_set'])
          == ('pre_attention', 0, 'V')
      )
      self.assertFalse(affected['selectable'])
      self.assertIsNone(affected['Q'])
      self.assertEqual(len(affected['invalid_or_missing_variants']), 1)
    finally:
      path.write_bytes(saved)
      self.fixture.refresh()

  def test_false_active_baseline_tensor_noop_attestation_fails_closed(self):
    path = next((self.fixture.root / 'raw' / 'phase_r').glob('*/*.json'))
    saved = path.read_bytes()
    value = json.loads(saved)
    value['checks']['baseline_rows_active_seam_natural_effective_exact'] = False
    _write(path, value)
    self.fixture.refresh()
    try:
      with self.assertRaisesRegex(ValueError, 'marked complete'):
        self.analyze()
    finally:
      path.write_bytes(saved)
      self.fixture.refresh()

  def test_checkpoint_tree_byte_tampering_fails_closed(self):
    checkpoint = self.fixture.bundle / analyzer.CHECKPOINT_SNAPSHOT
    path = next(checkpoint.rglob('*.bin'))
    saved = path.read_bytes()
    path.write_bytes(saved + b'tamper')
    try:
      with self.assertRaisesRegex(ValueError, 'Checkpoint file content changed'):
        self.analyze()
    finally:
      path.write_bytes(saved)

  def test_identity_sequence_hash_must_match_frozen_reference_binding(self):
    path = sorted((self.fixture.root / 'raw' / 'identity').glob('*.json'))[0]
    saved = path.read_bytes()
    value = json.loads(saved)
    value['sequence_sha256']['alternate'] = 'c' * 64
    _write(path, value)
    self.fixture.refresh()
    try:
      with self.assertRaisesRegex(ValueError, 'frozen reference binding'):
        self.analyze()
    finally:
      path.write_bytes(saved)
      self.fixture.refresh()

  def test_gcs_reference_metadata_mismatch_fails_closed(self):
    path = self.fixture.root / 'ATTEMPT_STARTED.json'
    saved = path.read_bytes()
    value = json.loads(saved)
    value['reference_object_binding']['observed_generation'] = 'different'
    _write(path, value)
    try:
      with self.assertRaisesRegex(ValueError, 'GCS reference-object metadata'):
        self.analyze()
    finally:
      path.write_bytes(saved)

  def test_bootstrap_tracked_head_gate_tampering_fails_closed(self):
    path = self.fixture.root / 'ATTEMPT_STARTED.json'
    saved = path.read_bytes()
    value = json.loads(saved)
    value['same_process_pre_import_bootstrap']['freeze'][
        'tracked_head_clean'
    ] = False
    _write(path, value)
    try:
      with self.assertRaisesRegex(ValueError, 'committed-HEAD gate'):
        self.analyze()
    finally:
      path.write_bytes(saved)

  def test_confirmation_named_path_is_rejected_before_read(self):
    with tempfile.TemporaryDirectory(prefix='confirmation_') as directory:
      with self.assertRaisesRegex(ValueError, 'confirmation-named path'):
        analyzer.analyze(Path(directory), enforce_standard_locations=False)

  def test_controlled_identity_stop_is_audited_without_active_work(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_stop_') as directory:
      run = SyntheticRun(Path(directory) / 'development_run', identity_stop=True)
      result = analyzer.analyze(
          run.root, bundle_root=run.bundle, enforce_standard_locations=False
      )
      self.assertEqual(
          result['decision'], 'identity_tooling_failure_no_mechanism_result'
      )
      self.assertEqual(result['audit']['active_intervention_count'], 0)
      self.assertIsNone(result['phase_r'])

  def test_final_closure_stop_has_12_artifacts_and_no_later_family(self):
    with tempfile.TemporaryDirectory(prefix='v3_2_closure_') as directory:
      run = SyntheticRun(
          Path(directory) / 'development_run', closure_stop='final'
      )
      result = analyzer.analyze(
          run.root, bundle_root=run.bundle, enforce_standard_locations=False
      )
      self.assertEqual(
          result['stage_a']['closure_failure_stage'], 'final_embedding_A_D'
      )
      self.assertEqual(
          result['stage_a']['executed_closure_components'],
          [analyzer.STAGE_COMPONENTS[0]],
      )
      self.assertEqual(
          json.loads((run.root / 'RUN_COMPLETE.json').read_text())[
              'stage_a_group_count'
          ],
          12,
      )


if __name__ == '__main__':
  unittest.main()
