#!/usr/bin/env python3
"""One-shot development-only OpenSplice v3.3 encoder-skip factorial runner."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from alphagenome_research.model import dna_model
from alphagenome_research.model import interpretability


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))
import run_superset_graph_v3_2 as v32  # pylint: disable=g-import-not-at-top
import validate_encoder_skip_bootstrap_v3_3 as bootstrap  # pylint: disable=g-import-not-at-top


SCRIPT_VERSION = 'opensplice-encoder-skip-factorial-v3.3.0'
ATTEMPT_ID = 'opensplice-v3.3-development-encoder-skip-factorial-one-shot'
PROTOCOL_PATH = (
    _HERE / 'v3_wider_mechanism' / 'encoder_skip_localization_protocol_v3_3.md'
)
PROTOCOL_SHA256 = (
    '85151029297cf4b4c1a07dd2a0f47b10b0663acd08ddfe25ba72b9898b1beea0'
)
SUPERSESSION_PATH = (
    _HERE / 'v3_wider_mechanism' / 'seven_skip_factorial_analysis_plan.md'
)
SUPERSESSION_SHA256 = (
    'ca29860eae1d41b9c5c69908b2209c0b5fe06b5d1b9c2f70225ee2d0656fa0dd'
)
SUPERSESSION_COMMIT = 'c64def4'
FREEZE_PATH = _HERE / 'encoder_skip_factorial_v3_3_freeze.json'
OUTPUT_DIR = (
    _HERE / 'results' / 'v3_3_development_encoder_skip_factorial_one_shot'
)
ANALYSIS_DIR = (
    _HERE / 'results' / 'v3_3_development_encoder_skip_factorial_analysis'
)
PREFLIGHT_DIR = _HERE / 'results' / 'v3_3_device_preflight'
START_PATH = OUTPUT_DIR / 'ATTEMPT_STARTED.json'
PREFLIGHT_SCRIPT_VERSION = 'opensplice-device-preflight-v3.3.0'
BOOTSTRAP_ATTESTATION_MODULE = '_opensplice_v3_3_bootstrap_attestation'
EXPECTED_DEVICE_KIND = v32.EXPECTED_DEVICE_KIND
EXPECTED_GPU_UUID = v32.EXPECTED_GPU_UUID
EXPECTED_COMPUTE_CAPABILITY = v32.EXPECTED_COMPUTE_CAPABILITY

E_PLAYERS = ('E64', 'E32', 'E16', 'E8', 'E4', 'E2', 'E1')
COALITION_BIT_ORDER = E_PLAYERS + ('T',)
SHAPLEY_PLAYER_ORDER = ('T',) + E_PLAYERS
NUM_COALITIONS = 256
ANCHOR_IDS = (0, 127, 128, 255)
GATE0_ANCHOR_IDS = (0, 255)
EFFECT_ORDER = (0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15)
NEUTRAL_ORDER = (6, 7, 8, 9, 16, 17, 18, 19)
SHAPLEY_ABSOLUTE_TOLERANCE = 1e-9
SHAPLEY_RELATIVE_TOLERANCE = 1e-9
MAX_WALL_TIME_SECONDS = 4 * 60 * 60
MAX_OUTPUT_BYTES = 8 * 1024 * 1024 * 1024
RUNTIME_PACKAGES = (
    'jax',
    'jaxlib',
    'jax-cuda12-pjrt',
    'jax-cuda12-plugin',
    'nvidia-cublas-cu12',
    'nvidia-cuda-runtime-cu12',
    'nvidia-cudnn-cu12',
    'nvidia-cusparse-cu12',
    'dm-haiku',
    'jmp',
    'protobuf',
    'numpy',
    'orbax-checkpoint',
    'grpcio',
)
SIX_ROLES = v32.TRACE_ROLES
EIGHT_ROLES = SIX_ROLES + ('unrelated_reference_donor', 'unrelated_alternate_donor')
SIX_DONOR_ROWS = (0, 1, 0, 1, 1, 0)
SIX_IDENTITY_ROWS = (0, 1, 1, 1, 0, 0)
EIGHT_IDENTITY_ROWS = (0, 1, 1, 1, 0, 0, 6, 7)
EIGHT_INTENDED_DONOR_ROWS = (0, 1, 0, 1, 1, 0, 6, 7)
EIGHT_UNRELATED_DONOR_ROWS = (0, 1, 6, 1, 7, 0, 6, 7)
ACTIVE_RECIPIENT_ROWS_6 = (False, False, True, True, True, True)
ACTIVE_RECIPIENT_ROWS_8 = (
    False, False, True, True, True, True, False, False
)
OOD_DONOR_ORDER = {
    **{index: index + 10 for index in range(6)},
    **{index + 10: index for index in range(6)},
    **{index: index + 10 for index in range(6, 10)},
    **{index + 10: index for index in range(6, 10)},
}


def _sha256(path: Path) -> str:
  return v32._sha256(path)  # pylint: disable=protected-access


def _write_new(path: Path, value: Any) -> str:
  return v32._write_new(path, value)  # pylint: disable=protected-access


def _write_new_text(path: Path, value: str) -> str:
  return v32._write_new_text(path, value)  # pylint: disable=protected-access


def _slug(value: str) -> str:
  return v32._slug(value)  # pylint: disable=protected-access


def _reject_confirmation_path(path: Path) -> None:
  v32._reject_confirmation_path(path)  # pylint: disable=protected-access


def runtime_version_binding() -> dict[str, Any]:
  return {
      'python_version': platform.python_version(),
      'platform': platform.platform(),
      'kernel': platform.release(),
      'packages': {
          name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES
      },
  }


def validate_device_version_manifest(
    observation: Mapping[str, Any], frozen: Mapping[str, Any]
) -> None:
  expected = frozen['runtime_version_manifest']
  current = runtime_version_binding()
  if {
      key: expected.get(key)
      for key in ('python_version', 'platform', 'kernel', 'packages')
  } != current:
    raise ValueError('Frozen critical numerical runtime manifest changed.')
  packages = observation.get('packages', {})
  for name in RUNTIME_PACKAGES:
    if packages.get(name) != expected['packages'][name]:
      raise ValueError(f'Frozen GPU/JAX package changed: {name}.')
  if (
      observation.get('jax_module_version') != expected['packages']['jax']
      or observation.get('jaxlib_module_version')
      != expected['packages']['jaxlib']
      or observation.get('python_version', '').split()[0]
      != expected['python_version']
      or observation.get('platform') != expected['platform']
      or observation.get('kernel') != expected['kernel']
  ):
    raise ValueError('Frozen Python/JAX/platform version manifest changed.')
  physical = observation.get('nvidia_smi', {}).get('parsed_single_gpu', {})
  if physical != expected['nvidia_smi']:
    raise ValueError('Frozen GPU/driver/VBIOS manifest changed.')


def coalition_metadata(coalition_id: int) -> dict[str, Any]:
  if coalition_id < 0 or coalition_id >= NUM_COALITIONS:
    raise ValueError('v3.3 coalition ID must be in [0, 255].')
  t = coalition_id >> 7
  e_mask = coalition_id & 0x7F
  e_bits = tuple(bool(e_mask & (1 << index)) for index in range(7))
  enabled = tuple(
      name for name in SHAPLEY_PLAYER_ORDER
      if (t if name == 'T' else e_bits[E_PLAYERS.index(name)])
  )
  return {
      'coalition_id': coalition_id,
      't': t,
      'e_mask': e_mask,
      'e_bits': list(e_bits),
      'e_bits_binary': format(e_mask, '07b'),
      'enabled_players': list(enabled),
      'coalition_bit_order': list(COALITION_BIT_ORDER),
      'shapley_player_order': list(SHAPLEY_PLAYER_ORDER),
  }


def _replace_transformer_residual_transfers(
    transformer: interpretability.TransformerInterventions,
    *,
    batch_size: int,
    num_positions: int,
) -> interpretability.TransformerInterventions:
  transfer = interpretability.no_sequence_route_batch_transfer(
      num_stages=interpretability.NUM_TRANSFORMER_LAYERS,
      batch_size=batch_size,
      num_positions=num_positions,
  )
  return dataclasses.replace(
      transformer,
      pre_attention_residual_transfer=transfer,
      post_attention_residual_transfer=transfer,
      post_mlp_residual_transfer=transfer,
  )


def coalition_interventions(
    selection: interpretability.SupersetGraphSelection,
    coalition_id: int,
) -> interpretability.SupersetGraphInterventions:
  """Builds one fixed-shape six-row T+7E coalition intervention."""
  identity = v32.identity_interventions(selection)
  metadata = coalition_metadata(coalition_id)
  stage_a = dataclasses.replace(
      identity.stage_a,
      transformer_output=(
          interpretability.paired_six_row_whole_sequence_transfer(
              jnp.asarray([bool(metadata['t'])], jnp.bool)
          )
      ),
      encoder_skips=interpretability.paired_six_row_whole_sequence_transfer(
          jnp.asarray(metadata['e_bits'], jnp.bool)
      ),
  )
  return dataclasses.replace(identity, stage_a=stage_a)


def _eight_row_whole_transfer(
    component_mask: Sequence[bool], donor_rows: Sequence[int]
) -> interpretability.WholeSequenceBatchTransfer:
  mask = jnp.asarray(component_mask, jnp.bool)
  donors = jnp.broadcast_to(
      jnp.asarray(donor_rows, jnp.int32)[None, :], (mask.shape[0], 8)
  )
  identities = jnp.broadcast_to(
      jnp.asarray(EIGHT_IDENTITY_ROWS, jnp.int32)[None, :],
      (mask.shape[0], 8),
  )
  recipients = jnp.asarray(ACTIVE_RECIPIENT_ROWS_8, jnp.bool)
  return interpretability.WholeSequenceBatchTransfer(
      donor_batch_indices=donors,
      natural_identity_batch_indices=identities,
      transfer_mask=mask[:, None] & recipients[None, :],
  )


def eight_row_interventions(
    selection: interpretability.SupersetGraphSelection,
    coalition_id: int,
    *,
    unrelated: bool,
) -> interpretability.SupersetGraphInterventions:
  """Builds fixed eight-row intended or cross-exon donor interventions."""
  metadata = coalition_metadata(coalition_id)
  residual_positions = selection.transformer.residual_positions
  if residual_positions is None:
    raise ValueError('v3.3 requires the frozen transformer residual selector.')
  transformer = interpretability.no_transformer_interventions(
      batch_size=8,
      num_edges=selection.transformer.pair_bias_edges.query_bins.shape[0],
  )
  transformer = _replace_transformer_residual_transfers(
      transformer,
      batch_size=8,
      num_positions=residual_positions.positions.shape[0],
  )
  donor_rows = (
      EIGHT_UNRELATED_DONOR_ROWS
      if unrelated else EIGHT_INTENDED_DONOR_ROWS
  )
  stage_a = interpretability.StageABranchInterventions(
      transformer_output=_eight_row_whole_transfer(
          [bool(metadata['t'])], donor_rows
      ),
      encoder_skips=_eight_row_whole_transfer(
          metadata['e_bits'], donor_rows
      ),
      final_embedding=interpretability.no_sequence_route_batch_transfer(
          num_stages=1,
          batch_size=8,
          num_positions=selection.stage_a.final_embedding_positions.shape[0],
      ),
  )
  return interpretability.SupersetGraphInterventions(
      transformer=transformer, stage_a=stage_a
  )


def _whole_transfer_record(
    transfer: interpretability.WholeSequenceBatchTransfer,
) -> dict[str, Any]:
  return {
      'donor_batch_indices': np.asarray(
          transfer.donor_batch_indices, np.int32
      ).tolist(),
      'natural_identity_batch_indices': np.asarray(
          transfer.natural_identity_batch_indices, np.int32
      ).tolist(),
      'transfer_mask': np.asarray(transfer.transfer_mask, bool).tolist(),
  }


def _residual_transfer_record(
    transfer: interpretability.SequenceResidualBatchTransfer,
) -> dict[str, Any]:
  return {
      'donor_batch_indices': np.asarray(
          transfer.donor_batch_indices, np.int32
      ).tolist(),
      'transfer_mask': np.asarray(transfer.transfer_mask, bool).tolist(),
  }


def intervention_record(
    interventions: interpretability.SupersetGraphInterventions,
) -> dict[str, Any]:
  """Serializes the actual dynamic transfer arrays passed to the executable."""
  transformer = interventions.transformer
  residual = {
      name: _residual_transfer_record(getattr(transformer, name))
      for name in (
          'pre_attention_residual_transfer',
          'post_attention_residual_transfer',
          'post_mlp_residual_transfer',
      )
  }
  return {
      'transformer_output': _whole_transfer_record(
          interventions.stage_a.transformer_output
      ),
      'encoder_skips': _whole_transfer_record(
          interventions.stage_a.encoder_skips
      ),
      'final_embedding': _residual_transfer_record(
          interventions.stage_a.final_embedding
      ),
      'phase_r_residuals': residual,
  }


def _assert_runtime_transfer_contract(
    interventions: interpretability.SupersetGraphInterventions,
    coalition_id: int,
    *,
    batch_size: int,
    donor_rows: Sequence[int],
    identity_rows: Sequence[int],
) -> None:
  """Fail-closes on any mismatch between frozen masks and runtime arrays."""
  metadata = coalition_metadata(coalition_id)
  expected_recipient = np.asarray(
      ACTIVE_RECIPIENT_ROWS_6 if batch_size == 6 else ACTIVE_RECIPIENT_ROWS_8,
      bool,
  )
  expected_donors = np.asarray(donor_rows, np.int32)
  expected_identity = np.asarray(identity_rows, np.int32)
  for name, transfer, enabled in (
      ('T', interventions.stage_a.transformer_output, [bool(metadata['t'])]),
      ('E', interventions.stage_a.encoder_skips, metadata['e_bits']),
  ):
    donors = np.asarray(transfer.donor_batch_indices, np.int32)
    natural = np.asarray(transfer.natural_identity_batch_indices, np.int32)
    mask = np.asarray(transfer.transfer_mask, bool)
    expected_shape = (len(enabled), batch_size)
    if donors.shape != expected_shape or natural.shape != expected_shape:
      raise ValueError(f'{name} runtime donor-map shape changed.')
    if not np.array_equal(
        donors, np.broadcast_to(expected_donors[None, :], expected_shape)
    ):
      raise ValueError(f'{name} runtime donor map differs from the freeze.')
    if not np.array_equal(
        natural, np.broadcast_to(expected_identity[None, :], expected_shape)
    ):
      raise ValueError(f'{name} natural-identity map differs from the freeze.')
    expected_mask = np.asarray(enabled, bool)[:, None] & expected_recipient[None, :]
    if not np.array_equal(mask, expected_mask):
      raise ValueError(f'{name} runtime route mask differs from coalition bits.')
  final = interventions.stage_a.final_embedding
  final_mask = np.asarray(final.transfer_mask, bool)
  final_donors = np.asarray(final.donor_batch_indices, np.int32)
  expected_final_donors = np.broadcast_to(
      np.arange(batch_size, dtype=np.int32)[None, :, None],
      final_donors.shape,
  )
  if final_mask.any() or not np.array_equal(
      final_donors, expected_final_donors
  ):
    raise ValueError('Final-embedding route must remain all false in v3.3.')
  for name in (
      'pre_attention_residual_transfer',
      'post_attention_residual_transfer',
      'post_mlp_residual_transfer',
  ):
    transfer = getattr(interventions.transformer, name)
    if transfer is None:
      raise ValueError(f'Phase-R route must remain all false: {name}.')
    transfer_mask = np.asarray(transfer.transfer_mask, bool)
    transfer_donors = np.asarray(transfer.donor_batch_indices, np.int32)
    expected_transfer_donors = np.broadcast_to(
        np.arange(batch_size, dtype=np.int32)[None, :, None],
        transfer_donors.shape,
    )
    if transfer_mask.any() or not np.array_equal(
        transfer_donors, expected_transfer_donors
    ):
      raise ValueError(f'Phase-R no-op donor map changed: {name}.')


def target_readout(
    evidence: interpretability.SpliceClassificationLogitMarginEvidence,
    *,
    batch_size: int,
) -> dict[str, Any]:
  selected = np.asarray(evidence.selected_logits, np.float32)
  margins = np.asarray(evidence.margins, np.float32)
  means = np.asarray(evidence.target.mean, np.float32)
  totals = np.asarray(evidence.target.total, np.float32)
  num_values = int(np.asarray(evidence.target.num_values))
  if selected.shape != (batch_size, 2, 2):
    raise ValueError(f'Endpoint logits have wrong shape {selected.shape}.')
  if (
      margins.shape != (batch_size, 2)
      or means.shape != (batch_size,)
      or totals.shape != (batch_size,)
  ):
    raise ValueError('Endpoint reducer shapes changed.')
  if not np.array_equal(selected[..., 0] - selected[..., 1], margins):
    raise ValueError('Endpoint margins do not equal relevant-padding logits.')
  if num_values != 2 or not np.array_equal(margins.sum(1), totals):
    raise ValueError('Endpoint total/denominator algebra changed.')
  if not np.array_equal(totals / np.float32(2), means):
    raise ValueError('Endpoint means do not equal two-site totals/2.')
  if not np.isfinite(selected).all():
    raise ValueError('Endpoint evidence is non-finite.')
  return {
      'endpoint_axis': ['acceptor', 'donor'],
      'selected_logit_axis': ['relevant_class', 'padding_class'],
      'selected_logits': selected.tolist(),
      'endpoint_margins': margins.tolist(),
      'means': means.tolist(),
      'totals': totals.tolist(),
      'num_values': num_values,
  }


def _natural_route_fingerprints(
    trace: interpretability.SupersetGraphTrace,
) -> dict[str, Any]:
  return {
      'T': np.asarray(
          trace.stage_a.transformer_output_natural_fingerprint
      ).tolist(),
      'E': np.asarray(
          trace.stage_a.encoder_skips_natural_fingerprints
      ).tolist(),
  }


def _assert_readout_rows_equal(
    left: Mapping[str, Any], left_row: int,
    right: Mapping[str, Any], right_row: int,
) -> None:
  for field in ('selected_logits', 'endpoint_margins', 'totals', 'means'):
    if not np.array_equal(
        np.asarray(left[field])[left_row], np.asarray(right[field])[right_row]
    ):
      raise ValueError(
          f'Endpoint readout differs at {field}: row {left_row}/{right_row}.'
      )


def raw_bidirectional_movement(readout: Mapping[str, Any]) -> dict[str, float]:
  """Computes only raw OOD movements; never forms a donor denominator."""
  values = np.asarray(readout['means'], np.float32)
  if values.shape[0] < 6:
    raise ValueError('Raw movement requires the six recipient-role rows.')
  return {
      'reference_into_alternate': float(values[2] - values[3]),
      'alternate_into_reference': float(values[4] - values[5]),
  }


def validate_coalition(
    first: tuple[Any, interpretability.SupersetGraphTrace],
    repeated: tuple[Any, interpretability.SupersetGraphTrace],
    identity: Mapping[str, Any],
    coalition_id: int,
    interventions: interpretability.SupersetGraphInterventions,
) -> dict[str, Any]:
  """Performs exact six-row route, target and closure audits."""
  evidence, trace = first
  repeat_evidence, repeat_trace = repeated
  values = v32._validate_common_active(  # pylint: disable=protected-access
      evidence, trace, repeat_evidence, repeat_trace,
      identity['target_readout'],
  )
  v32._assert_transformer_noop(trace.transformer)  # pylint: disable=protected-access
  branch = trace.stage_a
  natural_t = np.asarray(branch.transformer_output_natural_matches_identity)
  natural_e = np.asarray(branch.encoder_skips_natural_match_identity)
  effective_t_natural = np.asarray(
      branch.transformer_output_effective_matches_natural
  )
  effective_t_donor = np.asarray(
      branch.transformer_output_effective_matches_intervention_donor
  )
  effective_e_natural = np.asarray(
      branch.encoder_skips_effective_match_natural
  )
  effective_e_donor = np.asarray(
      branch.encoder_skips_effective_match_intervention_donor
  )
  metadata = coalition_metadata(coalition_id)
  _assert_runtime_transfer_contract(
      interventions,
      coalition_id,
      batch_size=6,
      donor_rows=SIX_DONOR_ROWS,
      identity_rows=SIX_IDENTITY_ROWS,
  )
  e_bits = np.asarray(metadata['e_bits'], bool)
  if not natural_t.all() or not natural_e.all():
    raise ValueError('Natural T/E same-allele tensor audit failed.')
  if not effective_t_natural[:2].all() or not effective_e_natural[:, :2].all():
    raise ValueError('Active coalition changed baseline T/E tensors.')
  if metadata['t']:
    if not effective_t_donor.all():
      raise ValueError('Enabled T does not equal its live donor.')
  elif not effective_t_natural.all():
    raise ValueError('Disabled T changed its natural tensor.')
  for stage, enabled in enumerate(e_bits):
    check = effective_e_donor[stage] if enabled else effective_e_natural[stage]
    if not check.all():
      raise ValueError(f'E{64 >> stage} route tensor audit failed.')
  if not np.array_equal(
      branch.natural_final_embeddings, branch.effective_final_embeddings
  ):
    raise ValueError('Disabled final A/D embedding seam changed.')
  fingerprints = _natural_route_fingerprints(trace)
  if fingerprints != identity['natural_route_fingerprints']:
    raise ValueError('Natural T/E fingerprints differ from identity call.')
  readout = target_readout(evidence, batch_size=6)
  if coalition_id == 0:
    for row in range(6):
      _assert_readout_rows_equal(readout, row, identity['target_readout'], row)
  if coalition_id == 255:
    _assert_readout_rows_equal(readout, 2, readout, 0)
    _assert_readout_rows_equal(readout, 4, readout, 1)
  recovery = v32.recovery_statistics(values)
  return {
      'passed': True,
      'target_means': dict(zip(SIX_ROLES, map(float, values), strict=True)),
      'baseline_targets_exact_from_identity': True,
      'self_targets_exact': True,
      'target_repeat_exact': True,
      'trace_repeat_exact': True,
      'transformer_internal_seams_disabled_exact': True,
      'natural_T_same_allele_exact': True,
      'natural_E_same_allele_exact': True,
      'natural_route_fingerprints_match_identity': True,
      'baseline_rows_T_natural_effective_exact': True,
      'baseline_rows_E_natural_effective_exact': True,
      'T_enabled_donor_exact': bool(metadata['t']),
      'T_disabled_noop_exact': not bool(metadata['t']),
      'E_enabled_donor_exact': True,
      'E_disabled_noop_exact': True,
      'final_embedding_disabled_exact': True,
      'runtime_route_masks_and_maps_exact': True,
      'id0_identity_endpoint_exact': coalition_id == 0,
      'id255_endpoint_closure_exact': coalition_id == 255,
      **recovery,
  }


def _validate_eight_route_trace(
    trace: interpretability.SupersetGraphTrace,
    coalition_id: int,
    *,
    unrelated: bool,
) -> None:
  v32._assert_transformer_noop(trace.transformer)  # pylint: disable=protected-access
  branch = trace.stage_a
  if not np.asarray(branch.transformer_output_natural_matches_identity).all():
    raise ValueError('Eight-row natural T identity audit failed.')
  if not np.asarray(branch.encoder_skips_natural_match_identity).all():
    raise ValueError('Eight-row natural E identity audit failed.')
  t_natural = np.asarray(branch.transformer_output_effective_matches_natural)
  t_donor = np.asarray(
      branch.transformer_output_effective_matches_intervention_donor
  )
  e_natural = np.asarray(branch.encoder_skips_effective_match_natural)
  e_donor = np.asarray(
      branch.encoder_skips_effective_match_intervention_donor
  )
  metadata = coalition_metadata(coalition_id)
  if not t_natural[[0, 1, 6, 7]].all():
    raise ValueError('Eight-row T changed a baseline/donor source row.')
  if not e_natural[:, [0, 1, 6, 7]].all():
    raise ValueError('Eight-row E changed a baseline/donor source row.')
  if metadata['t']:
    if not t_donor.all():
      raise ValueError('Eight-row enabled T donor audit failed.')
  elif not t_natural.all():
    raise ValueError('Eight-row disabled T no-op audit failed.')
  for stage, enabled in enumerate(metadata['e_bits']):
    check = e_donor[stage] if enabled else e_natural[stage]
    if not check.all():
      raise ValueError(f'Eight-row E stage {stage} audit failed.')
  if not np.array_equal(
      branch.natural_final_embeddings, branch.effective_final_embeddings
  ):
    raise ValueError('Eight-row disabled final seam changed.')


def validate_ood_anchor(
    intended: tuple[Any, interpretability.SupersetGraphTrace],
    intended_repeated: tuple[Any, interpretability.SupersetGraphTrace],
    unrelated: tuple[Any, interpretability.SupersetGraphTrace],
    unrelated_repeated: tuple[Any, interpretability.SupersetGraphTrace],
    coalition_id: int,
    intended_interventions: interpretability.SupersetGraphInterventions,
    unrelated_interventions: interpretability.SupersetGraphInterventions,
) -> dict[str, Any]:
  """Audits intended and cross-exon donor calls from one 8-row executable."""
  intended_evidence, intended_trace = intended
  intended_repeat_evidence, intended_repeat_trace = intended_repeated
  unrelated_evidence, unrelated_trace = unrelated
  unrelated_repeat_evidence, unrelated_repeat_trace = unrelated_repeated
  v32._assert_evidence_repeat(  # pylint: disable=protected-access
      intended_evidence, intended_repeat_evidence
  )
  v32._assert_trace_repeat(  # pylint: disable=protected-access
      intended_trace, intended_repeat_trace
  )
  v32._assert_evidence_repeat(  # pylint: disable=protected-access
      unrelated_evidence, unrelated_repeat_evidence
  )
  v32._assert_trace_repeat(  # pylint: disable=protected-access
      unrelated_trace, unrelated_repeat_trace
  )
  _assert_runtime_transfer_contract(
      intended_interventions,
      coalition_id,
      batch_size=8,
      donor_rows=EIGHT_INTENDED_DONOR_ROWS,
      identity_rows=EIGHT_IDENTITY_ROWS,
  )
  _assert_runtime_transfer_contract(
      unrelated_interventions,
      coalition_id,
      batch_size=8,
      donor_rows=EIGHT_UNRELATED_DONOR_ROWS,
      identity_rows=EIGHT_IDENTITY_ROWS,
  )
  _validate_eight_route_trace(intended_trace, coalition_id, unrelated=False)
  _validate_eight_route_trace(unrelated_trace, coalition_id, unrelated=True)
  for natural_name, _ in v32._TRANSFORMER_PAIRS:  # pylint: disable=protected-access
    if not np.array_equal(
        getattr(intended_trace.transformer, natural_name),
        getattr(unrelated_trace.transformer, natural_name),
    ):
      raise ValueError(
          f'Eight-row intended/unrelated natural seam differs: {natural_name}.'
      )
  for field in (
      'transformer_output_natural_fingerprint',
      'encoder_skips_natural_fingerprints',
      'natural_final_embeddings',
  ):
    if not np.array_equal(
        getattr(intended_trace.stage_a, field),
        getattr(unrelated_trace.stage_a, field),
    ):
      raise ValueError(f'Eight-row natural route differs between calls: {field}.')
  intended_readout = target_readout(intended_evidence, batch_size=8)
  unrelated_readout = target_readout(unrelated_evidence, batch_size=8)
  for readout in (intended_readout, unrelated_readout):
    _assert_readout_rows_equal(readout, 3, readout, 1)
    _assert_readout_rows_equal(readout, 5, readout, 0)
  for row in (0, 1, 3, 5, 6, 7):
    _assert_readout_rows_equal(intended_readout, row, unrelated_readout, row)
  if coalition_id == 0:
    for readout in (intended_readout, unrelated_readout):
      _assert_readout_rows_equal(readout, 2, readout, 1)
      _assert_readout_rows_equal(readout, 4, readout, 0)
    for row in range(8):
      _assert_readout_rows_equal(
          intended_readout, row, unrelated_readout, row
      )
  if coalition_id == 255:
    _assert_readout_rows_equal(intended_readout, 2, intended_readout, 0)
    _assert_readout_rows_equal(intended_readout, 4, intended_readout, 1)
    _assert_readout_rows_equal(unrelated_readout, 2, unrelated_readout, 6)
    _assert_readout_rows_equal(unrelated_readout, 4, unrelated_readout, 7)
  return {
      'passed': True,
      'baseline_rows_exact_between_calls': True,
      'self_rows_exact_between_calls': True,
      'donor_source_rows_exact_between_calls': True,
      'natural_route_tensors_exact_between_calls': True,
      'self_targets_exact': True,
      'id0_all_recipient_rows_noop_exact': coalition_id == 0,
      'id0_intended_unrelated_all_rows_exact': coalition_id == 0,
      'id255_intended_endpoint_closure_exact': coalition_id == 255,
      'id255_unrelated_endpoint_closure_exact': coalition_id == 255,
      'intended_route_tensor_donor_exact': True,
      'unrelated_route_tensor_donor_exact': True,
      'enabled_disabled_T_E_exact': True,
      'runtime_route_masks_and_maps_exact': True,
      'intended_target_repeat_exact': True,
      'intended_trace_repeat_exact': True,
      'unrelated_target_repeat_exact': True,
      'unrelated_trace_repeat_exact': True,
      'transformer_internal_seams_disabled_exact': True,
      'final_embedding_disabled_exact': True,
      'normalization_computed': False,
  }


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', type=Path)
  parser.add_argument('--successful-preflight', type=Path)
  parser.add_argument('--dry-run', action='store_true')
  parser.add_argument('--max-variants', type=int, default=0)
  parser.add_argument('--max-coalitions', type=int, default=0)
  args = parser.parse_args()
  if not args.dry_run and (args.max_variants or args.max_coalitions):
    parser.error('Bounded flags are dry-run-only; v3.3 execution is all-or-none.')
  return args


def coalition_execution_order(cases: Sequence[Any]) -> tuple[tuple[int, int], ...]:
  """Returns frozen `(manifest_order, coalition_id)` execution order."""
  by_order = {case.order: case for case in cases}
  if tuple(sorted(by_order)) != tuple(range(20)):
    raise ValueError('v3.3 requires exact development manifest orders 0--19.')
  anchors = tuple(
      (order, coalition_id)
      for order in range(20)
      for coalition_id in (0, 255)
  )
  remainder_ids = tuple(range(1, 255))
  effects = tuple(
      (order, coalition_id)
      for order in EFFECT_ORDER
      for coalition_id in remainder_ids
  )
  neutrals = tuple(
      (order, coalition_id)
      for order in NEUTRAL_ORDER
      for coalition_id in remainder_ids
  )
  result = anchors + effects + neutrals
  if len(result) != 5120 or len(set(result)) != 5120:
    raise ValueError('Frozen v3.3 coalition execution order is incomplete.')
  return result


def build_dry_run_plan(
    cases: Sequence[Any], *, max_variants: int, max_coalitions: int
) -> dict[str, Any]:
  displayed_variants = min(len(cases), max_variants or len(cases))
  displayed_coalitions = min(NUM_COALITIONS, max_coalitions or NUM_COALITIONS)
  return {
      'script_version': SCRIPT_VERSION,
      'protocol_sha256': PROTOCOL_SHA256,
      'supersession_sha256': SUPERSESSION_SHA256,
      'dry_run': True,
      'development_case_count': 20,
      'effect_count': 12,
      'neutral_count': 8,
      'identity_record_count': 20,
      'coalitions_per_variant': NUM_COALITIONS,
      'coalition_record_count': 5120,
      'ood_anchor_ids': list(ANCHOR_IDS),
      'ood_record_count': 80,
      'scientific_record_count': 5220,
      'six_row_compile_count': 1,
      'eight_row_compile_count': 1,
      'total_compile_count': 2,
      'model_apply_count': 10600,
      'max_wall_time_seconds': MAX_WALL_TIME_SECONDS,
      'max_output_bytes': MAX_OUTPUT_BYTES,
      'coalition_execution_order': (
          'ID0/255 all20; IDs1..254 effects; IDs1..254 neutrals'
      ),
      'displayed_variants': displayed_variants,
      'displayed_coalitions': displayed_coalitions,
      'confirmation_model_calls': 0,
      'output_dir': str(OUTPUT_DIR),
      'analysis_dir': str(ANALYSIS_DIR),
  }


def _artifact_path(
    family: str, case: Any, *, coalition_id: int | None = None
) -> Path:
  case_key = f'{case.order:03d}_{_slug(case.variant_id)}'
  if family == 'identity':
    return OUTPUT_DIR / 'raw' / 'identity' / f'{case_key}.json'
  if coalition_id is None:
    raise ValueError(f'{family} requires a coalition ID.')
  if family == 'coalition':
    return (
        OUTPUT_DIR / 'raw' / 'coalitions' / case_key
        / f'{coalition_id:03d}.json'
    )
  if family == 'ood':
    return (
        OUTPUT_DIR / 'raw' / 'ood_anchors' / case_key
        / f'{coalition_id:03d}.json'
    )
  raise ValueError(f'Unknown v3.3 artifact family {family!r}.')


def _case_record(case: Any) -> dict[str, Any]:
  return v32._case_record(case)  # pylint: disable=protected-access


def _timed_apply(compiled: Any, args: Sequence[Any]):
  return v32._timed_apply(compiled, args)  # pylint: disable=protected-access


def _compiler_artifacts(
    lowered: Any, compiled: Any, seconds: float, *, executable_name: str
) -> dict[str, Any]:
  directory = OUTPUT_DIR / 'compiler' / executable_name
  stable = str(lowered.compiler_ir(dialect='stablehlo'))
  hlo_object = lowered.compiler_ir(dialect='hlo')
  hlo = (
      hlo_object.as_hlo_text()
      if hasattr(hlo_object, 'as_hlo_text') else str(hlo_object)
  )
  compiled_hlo = compiled.as_text()
  artifacts = {}
  for name, filename, content in (
      ('stablehlo', 'graph.stablehlo.mlir', stable),
      ('hlo', 'graph.pre_backend.hlo.txt', hlo),
      ('compiled_hlo', 'graph.compiled.hlo.txt', compiled_hlo),
  ):
    path = directory / filename
    digest = _write_new_text(path, content)
    artifacts[name] = {
        'path': str(path),
        'sha256': digest,
        'size_bytes': len(content.encode('utf-8')),
    }
  fingerprint = hashlib.sha256(
      bytes.fromhex(artifacts['compiled_hlo']['sha256'])
  ).hexdigest()
  record = {
      'executable_name': executable_name,
      'compile_count': 1,
      'compile_seconds': seconds,
      'executable_fingerprint': fingerprint,
      'artifacts': artifacts,
  }
  _write_new(directory / 'COMPILER_PROVENANCE.json', record)
  return record


def _identity_natural_fingerprints(trace: Any) -> dict[str, Any]:
  return _natural_route_fingerprints(trace)


def _run_identity(
    compiled: Any,
    model_instance: Any,
    case: Any,
    params: Any,
    state: Any,
    signatures: Mapping[str, Any],
    freeze_sha256: str,
    executable_fingerprint: str,
    execution_index: int,
) -> tuple[dict[str, Any], tuple[Any, ...]]:
  interval, position_sets, selection, target, resolved, dna, sequence_sha = (
      v32._case_inputs(model_instance, case)  # pylint: disable=protected-access
  )
  interventions = coalition_interventions(selection, 0)
  v32.assert_same_program_signature(signatures['selection'], selection)
  v32.assert_same_program_signature(signatures['six_interventions'], interventions)
  v32.assert_same_program_signature(signatures['target'], target)
  args = (
      params, state, dna, jnp.zeros((6,), jnp.int32),
      selection, interventions, target,
  )
  first, first_seconds = _timed_apply(compiled, args)
  repeated, repeat_seconds = _timed_apply(compiled, args)
  readout = target_readout(first[0], batch_size=6)
  repeated_readout = target_readout(repeated[0], batch_size=6)
  try:
    checks = v32.validate_identity(first[0], first[1], repeated[0], repeated[1])
    direction = v32.route_v3.direction_gate(
        case, np.asarray(first[0].target.mean)
    )
    status, failure = 'complete', None
  except ValueError as error:
    checks, direction = None, None
    status = 'invalid'
    failure = {'type': type(error).__name__, 'message': str(error)}
  natural_fingerprints = _identity_natural_fingerprints(first[1])
  artifact = {
      'status': status,
      'family': 'identity',
      'script_version': SCRIPT_VERSION,
      'protocol_sha256': PROTOCOL_SHA256,
      'supersession_sha256': SUPERSESSION_SHA256,
      'supersession_commit': SUPERSESSION_COMMIT,
      'freeze_sha256': freeze_sha256,
      'execution_index': execution_index,
      'six_row_executable_fingerprint': executable_fingerprint,
      'same_six_row_compiled_executable': True,
      'case': _case_record(case),
      'interval': {
          'chromosome': interval.chromosome,
          'start_0based': interval.start,
          'end_0based_exclusive': interval.end,
      },
      'sequence_sha256': sequence_sha,
      'resolved_position_sets': [dataclasses.asdict(x) for x in position_sets],
      'canonical_target': {
          'endpoints': [dataclasses.asdict(x) for x in resolved.endpoints],
          'padding_track_index': resolved.padding_track_index,
      },
      'runtime_interventions': intervention_record(interventions),
      'target_readout': readout,
      'repeat_target_readout': repeated_readout,
      'trace_fingerprint': v32.trace_fingerprint(first[1]),
      'repeat_trace_fingerprint': v32.trace_fingerprint(repeated[1]),
      'natural_route_fingerprints': natural_fingerprints,
      'checks': checks,
      'failure': failure,
      'direction_gate': direction,
      'program_signatures': signatures,
      'seconds': {'first': first_seconds, 'repeat': repeat_seconds},
      'created_at_unix_s': time.time(),
  }
  path = _artifact_path('identity', case)
  digest = _write_new(path, artifact)
  artifact['identity_binding'] = {
      'path': str(path.relative_to(OUTPUT_DIR)), 'sha256': digest
  }
  return artifact, (dna, selection, target, sequence_sha)


def _run_coalition(
    compiled: Any,
    case: Any,
    common: tuple[Any, ...],
    params: Any,
    state: Any,
    identity: Mapping[str, Any],
    coalition_id: int,
    signatures: Mapping[str, Any],
    freeze_sha256: str,
    executable_fingerprint: str,
    execution_index: int,
) -> dict[str, Any]:
  dna, selection, target, _ = common
  interventions = coalition_interventions(selection, coalition_id)
  v32.assert_same_program_signature(
      signatures['six_interventions'], interventions
  )
  args = (
      params, state, dna, jnp.zeros((6,), jnp.int32),
      selection, interventions, target,
  )
  first, first_seconds = _timed_apply(compiled, args)
  repeated, repeat_seconds = _timed_apply(compiled, args)
  readout = target_readout(first[0], batch_size=6)
  repeated_readout = target_readout(repeated[0], batch_size=6)
  try:
    checks = validate_coalition(
        first, repeated, identity, coalition_id, interventions
    )
    status, failure = 'complete', None
  except ValueError as error:
    checks = None
    status = 'invalid'
    failure = {'type': type(error).__name__, 'message': str(error)}
  identity_binding = identity['identity_binding']
  artifact = {
      'status': status,
      'family': 'encoder_skip_coalition',
      'script_version': SCRIPT_VERSION,
      'protocol_sha256': PROTOCOL_SHA256,
      'supersession_sha256': SUPERSESSION_SHA256,
      'supersession_commit': SUPERSESSION_COMMIT,
      'freeze_sha256': freeze_sha256,
      'execution_index': execution_index,
      'six_row_executable_fingerprint': executable_fingerprint,
      'same_six_row_compiled_executable': True,
      'case': _case_record(case),
      'coalition': coalition_metadata(coalition_id),
      'runtime_interventions': intervention_record(interventions),
      'identity_binding': identity_binding,
      'target_readout': readout,
      'repeat_target_readout': repeated_readout,
      'trace_fingerprint': v32.trace_fingerprint(first[1]),
      'repeat_trace_fingerprint': v32.trace_fingerprint(repeated[1]),
      'checks': checks,
      'failure': failure,
      'seconds': {'first': first_seconds, 'repeat': repeat_seconds},
      'created_at_unix_s': time.time(),
  }
  path = _artifact_path('coalition', case, coalition_id=coalition_id)
  digest = _write_new(path, artifact)
  return {
      'status': status,
      'family': artifact['family'],
      'case_order': case.order,
      'coalition': artifact['coalition'],
      'checks': checks,
      'failure': failure,
      'artifact_binding': {
          'path': str(path.relative_to(OUTPUT_DIR)), 'sha256': digest
      },
  }


def _eight_row_batch(
    recipient_common: tuple[Any, ...], donor_common: tuple[Any, ...]
) -> np.ndarray:
  recipient_dna = np.asarray(recipient_common[0])
  donor_dna = np.asarray(donor_common[0])
  if recipient_dna.shape[0] != 6 or donor_dna.shape[0] != 6:
    raise ValueError('v3.3 expected six-row source batches.')
  if recipient_dna.shape[1:] != donor_dna.shape[1:]:
    raise ValueError('Cross-exon OOD donor tensor shapes differ.')
  return np.concatenate((recipient_dna, donor_dna[:2]), axis=0)


def _run_ood_anchor(
    compiled: Any,
    recipient: Any,
    donor: Any,
    recipient_common: tuple[Any, ...],
    donor_common: tuple[Any, ...],
    params: Any,
    state: Any,
    identity: Mapping[str, Any],
    donor_identity: Mapping[str, Any],
    linked_coalition: Mapping[str, Any],
    coalition_id: int,
    signatures: Mapping[str, Any],
    freeze_sha256: str,
    executable_fingerprint: str,
    execution_index: int,
) -> dict[str, Any]:
  _, selection, target, recipient_sequence_sha = recipient_common
  dna = _eight_row_batch(recipient_common, donor_common)
  intended_interventions = eight_row_interventions(
      selection, coalition_id, unrelated=False
  )
  unrelated_interventions = eight_row_interventions(
      selection, coalition_id, unrelated=True
  )
  for interventions in (intended_interventions, unrelated_interventions):
    v32.assert_same_program_signature(
        signatures['eight_interventions'], interventions
    )
  common_args = (
      params, state, dna, jnp.zeros((8,), jnp.int32), selection,
  )
  intended, intended_seconds = _timed_apply(
      compiled, (*common_args, intended_interventions, target)
  )
  intended_repeated, intended_repeat_seconds = _timed_apply(
      compiled, (*common_args, intended_interventions, target)
  )
  unrelated, unrelated_seconds = _timed_apply(
      compiled, (*common_args, unrelated_interventions, target)
  )
  unrelated_repeated, unrelated_repeat_seconds = _timed_apply(
      compiled, (*common_args, unrelated_interventions, target)
  )
  intended_readout = target_readout(intended[0], batch_size=8)
  intended_repeat_readout = target_readout(
      intended_repeated[0], batch_size=8
  )
  unrelated_readout = target_readout(unrelated[0], batch_size=8)
  unrelated_repeat_readout = target_readout(
      unrelated_repeated[0], batch_size=8
  )
  try:
    checks = validate_ood_anchor(
        intended,
        intended_repeated,
        unrelated,
        unrelated_repeated,
        coalition_id,
        intended_interventions,
        unrelated_interventions,
    )
    status, failure = 'complete', None
  except ValueError as error:
    checks = None
    status = 'invalid'
    failure = {'type': type(error).__name__, 'message': str(error)}
  artifact = {
      'status': status,
      'family': 'unrelated_donor_anchor',
      'script_version': SCRIPT_VERSION,
      'protocol_sha256': PROTOCOL_SHA256,
      'supersession_sha256': SUPERSESSION_SHA256,
      'supersession_commit': SUPERSESSION_COMMIT,
      'freeze_sha256': freeze_sha256,
      'execution_index': execution_index,
      'eight_row_executable_fingerprint': executable_fingerprint,
      'same_eight_row_compiled_executable': True,
      'recipient_case': _case_record(recipient),
      'donor_case': _case_record(donor),
      'coalition': coalition_metadata(coalition_id),
      'batch_roles': list(EIGHT_ROLES),
      'natural_identity_rows': list(EIGHT_IDENTITY_ROWS),
      'intended_donor_rows': list(EIGHT_INTENDED_DONOR_ROWS),
      'unrelated_donor_rows': list(EIGHT_UNRELATED_DONOR_ROWS),
      'identity_binding': identity['identity_binding'],
      'donor_identity_binding': donor_identity['identity_binding'],
      'linked_six_row_coalition': linked_coalition['artifact_binding'],
      'recipient_sequence_sha256': recipient_sequence_sha,
      'donor_sequence_sha256': donor_common[3],
      'runtime_interventions': {
          'intended': intervention_record(intended_interventions),
          'unrelated': intervention_record(unrelated_interventions),
      },
      'intended_target_readout': intended_readout,
      'intended_repeat_target_readout': intended_repeat_readout,
      'unrelated_target_readout': unrelated_readout,
      'unrelated_repeat_target_readout': unrelated_repeat_readout,
      'intended_trace_fingerprint': v32.trace_fingerprint(intended[1]),
      'intended_repeat_trace_fingerprint': v32.trace_fingerprint(
          intended_repeated[1]
      ),
      'unrelated_trace_fingerprint': v32.trace_fingerprint(unrelated[1]),
      'unrelated_repeat_trace_fingerprint': v32.trace_fingerprint(
          unrelated_repeated[1]
      ),
      'raw_movement': {
          'intended': raw_bidirectional_movement(intended_readout),
          'unrelated': raw_bidirectional_movement(unrelated_readout),
      },
      'checks': checks,
      'failure': failure,
      'seconds': {
          'intended': intended_seconds,
          'intended_repeat': intended_repeat_seconds,
          'unrelated': unrelated_seconds,
          'unrelated_repeat': unrelated_repeat_seconds,
      },
      'created_at_unix_s': time.time(),
  }
  path = _artifact_path('ood', recipient, coalition_id=coalition_id)
  digest = _write_new(path, artifact)
  return {
      'status': status,
      'family': artifact['family'],
      'recipient_order': recipient.order,
      'coalition': artifact['coalition'],
      'checks': checks,
      'failure': failure,
      'artifact_binding': {
          'path': str(path.relative_to(OUTPUT_DIR)), 'sha256': digest
      },
  }


def consume_bootstrap_attestation() -> dict[str, Any]:
  module = sys.modules.pop(BOOTSTRAP_ATTESTATION_MODULE, None)
  record = getattr(module, 'record', None)
  if not isinstance(record, dict):
    raise RuntimeError(
        'Direct v3.3 runner invocation is forbidden; use its launcher.'
    )
  if record.get('pid') != os.getpid():
    raise RuntimeError('v3.3 bootstrap attestation came from another process.')
  if record.get('freeze', {}).get('sha256') != _sha256(FREEZE_PATH):
    raise RuntimeError('v3.3 bootstrap attestation used another freeze.')
  if record.get('freeze', {}).get('tracked_head_clean') is not True:
    raise RuntimeError('v3.3 bootstrap did not attest tracked-clean HEAD.')
  upstream = record.get('freeze', {}).get('upstream_checkout', {})
  frozen_upstream = json.loads(FREEZE_PATH.read_text(encoding='utf-8'))
  generated_names = set(bootstrap.UPSTREAM_GENERATED_MODULE_NAMES)
  expected_imported_modules = {
      name: {
          **binding,
          'path': str(
              (
                  _HERE.parents[3] / 'alphagenome'
                  / binding['relative_path']
              ).resolve()
          ),
          'source_kind': (
              'generated_exact_byte_exception'
              if name in generated_names else 'tracked'
          ),
      }
      for name, binding in frozen_upstream[  # pylint: disable=unsubscriptable-object
          'upstream_imported_modules'
      ].items()
  }
  if (
      upstream.get('tracked_head_clean') is not True
      or upstream.get('git_head')
      != frozen_upstream.get('upstream_alphagenome_git_head')
      or upstream.get('imported_modules') != expected_imported_modules
      or upstream.get('tracked_imported_module_count') != 22
      or upstream.get('generated_imported_module_count') != 4
      or upstream.get('generated_binding_exception')
      != frozen_upstream.get('upstream_generated_binding_exception')
  ):
    raise RuntimeError('v3.3 upstream checkout attestation changed.')
  if record.get('sanitized_environment') != {
      'LD_LIBRARY_PATH': 'absent',
      'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
      'JAX_ENABLE_COMPILATION_CACHE': 'false',
  }:
    raise RuntimeError('v3.3 launcher environment attestation changed.')
  if (
      record.get('freeze', {}).get('prior_v3_2_evidence')
      != bootstrap.EXPECTED_PRIOR_V3_2_EVIDENCE
      or frozen_upstream.get('prior_v3_2_evidence')
      != bootstrap.EXPECTED_PRIOR_V3_2_EVIDENCE
  ):
    raise RuntimeError('v3.3 prior-v3.2 evidence attestation changed.')
  for field, path in (
      ('launcher', _HERE / 'launch_encoder_skip_factorial_v3_3.py'),
      ('bootstrap', _HERE / 'validate_encoder_skip_bootstrap_v3_3.py'),
  ):
    if record.get(f'{field}_path') != str(path.resolve()):
      raise RuntimeError(f'v3.3 bootstrap {field} path changed.')
    if record.get(f'{field}_sha256') != _sha256(path):
      raise RuntimeError(f'v3.3 bootstrap {field} hash changed.')
  return record


def import_provenance(frozen: Mapping[str, Any]) -> dict[str, Any]:
  """Extends module-byte provenance with exact tracked upstream inventory."""
  record = v32.import_provenance()
  expected = frozen['upstream_imported_modules']
  upstream_root = (_HERE.parents[3] / 'alphagenome').resolve()
  observed = {}
  for item in record['modules']:
    if item['root'] != 'upstream_alphagenome_checkout':
      continue
    path = Path(item['path']).resolve()
    relative = str(path.relative_to(upstream_root))
    observed[item['name']] = {
        'relative_path': relative,
        'sha256': item['sha256'],
        'size_bytes': item['size_bytes'],
    }
  if observed != expected:
    raise ValueError('Loaded upstream module inventory differs from freeze.')
  record['upstream_source_attestation'] = (
      bootstrap.validate_upstream_checkout(dict(frozen))
  )
  return record


def validate_freeze() -> dict[str, Any]:
  frozen = json.loads(FREEZE_PATH.read_text(encoding='utf-8'))
  observed_protobuf = v32.protobuf_provenance()
  upstream_head = subprocess.check_output(
      ('git', '-C', str(_HERE.parents[3] / 'alphagenome'), 'rev-parse', 'HEAD'),
      text=True,
  ).strip()
  expected = {
      'script_version': SCRIPT_VERSION,
      'attempt_id': ATTEMPT_ID,
      'protocol_path': str(PROTOCOL_PATH.resolve()),
      'protocol_sha256': PROTOCOL_SHA256,
      'supersession_path': str(SUPERSESSION_PATH.resolve()),
      'supersession_sha256': SUPERSESSION_SHA256,
      'supersession_commit': SUPERSESSION_COMMIT,
      'selected_variants_sha256': v32.route_v3.SELECTED_SHA256,
      'frozen_exons_sha256': v32.route_v3.EXONS_SHA256,
      'development_variants_path': str(
          v32.DEVELOPMENT_VARIANTS_PATH.resolve()
      ),
      'development_variants_sha256': v32.DEVELOPMENT_VARIANTS_SHA256,
      'development_exons_path': str(v32.DEVELOPMENT_EXONS_PATH.resolve()),
      'development_exons_sha256': v32.DEVELOPMENT_EXONS_SHA256,
      'checkpoint_snapshot': v32.route_v3.CHECKPOINT_SNAPSHOT,
      'checkpoint_manifest_path': str(
          v32.CHECKPOINT_MANIFEST_PATH.resolve()
      ),
      'checkpoint_manifest_sha256': v32.CHECKPOINT_MANIFEST_SHA256,
      'reference_url': v32.REFERENCE_URL,
      'reference_object': v32.REFERENCE_OBJECT,
      'reference_bindings_path': str(v32.REFERENCE_BINDINGS_PATH.resolve()),
      'reference_bindings_sha256': v32.REFERENCE_BINDINGS_SHA256,
      'context_bp': v32.route_v3.CONTEXT_BP,
      'attention_backend': v32.route_v3.ATTENTION_BACKEND,
      'output_dir': str(OUTPUT_DIR.resolve()),
      'analysis_dir': str(ANALYSIS_DIR.resolve()),
      'preflight_dir': str(PREFLIGHT_DIR.resolve()),
      'preflight_script_version': PREFLIGHT_SCRIPT_VERSION,
      'expected_device_kind': EXPECTED_DEVICE_KIND,
      'expected_gpu_uuid': EXPECTED_GPU_UUID,
      'expected_compute_capability': EXPECTED_COMPUTE_CAPABILITY,
      'upstream_alphagenome_git_head': upstream_head,
      'mixed_precision_policy': (
          'params=float32,compute=bfloat16,output=bfloat16'
      ),
      'environment_contract': {
          'LD_LIBRARY_PATH': 'absent',
          'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
          'JAX_ENABLE_COMPILATION_CACHE': 'false',
          'compiler_and_autotune_cache_inputs': 'absent',
      },
      'six_row_roles': list(SIX_ROLES),
      'six_row_natural_identity_rows': list(SIX_IDENTITY_ROWS),
      'six_row_donor_rows': list(SIX_DONOR_ROWS),
      'eight_row_roles': list(EIGHT_ROLES),
      'eight_row_natural_identity_rows': list(EIGHT_IDENTITY_ROWS),
      'eight_row_intended_donor_rows': list(EIGHT_INTENDED_DONOR_ROWS),
      'eight_row_unrelated_donor_rows': list(EIGHT_UNRELATED_DONOR_ROWS),
      'coalition_bit_order': list(COALITION_BIT_ORDER),
      'shapley_player_order': list(SHAPLEY_PLAYER_ORDER),
      'coalition_count_per_variant': 256,
      'identity_record_count': 20,
      'coalition_record_count': 5120,
      'ood_anchor_record_count': 80,
      'effect_order': list(EFFECT_ORDER),
      'neutral_order': list(NEUTRAL_ORDER),
      'gate0_anchor_ids': list(GATE0_ANCHOR_IDS),
      'ood_anchor_ids': list(ANCHOR_IDS),
      'scientific_record_count': 5220,
      'model_apply_count': 10600,
      'six_row_compile_count': 1,
      'eight_row_compile_count': 1,
      'shapley_absolute_tolerance': SHAPLEY_ABSOLUTE_TOLERANCE,
      'shapley_relative_tolerance': SHAPLEY_RELATIVE_TOLERANCE,
      'max_wall_time_seconds': MAX_WALL_TIME_SECONDS,
      'max_output_bytes': MAX_OUTPUT_BYTES,
      'protobuf_binding': observed_protobuf,
  }
  for name, value in expected.items():
    if frozen.get(name) != value:
      raise ValueError(f'v3.3 freeze mismatch: {name}.')
  runtime_manifest = frozen.get('runtime_version_manifest')
  if not isinstance(runtime_manifest, Mapping) or {
      key: runtime_manifest.get(key)
      for key in ('python_version', 'platform', 'kernel', 'packages')
  } != runtime_version_binding():
    raise ValueError('v3.3 frozen runtime version manifest changed.')
  if not isinstance(
      frozen.get('upstream_imported_modules'), Mapping
  ) or len(frozen['upstream_imported_modules']) != 26:
    raise ValueError('v3.3 frozen upstream inventory changed.')
  if (
      frozen.get('upstream_generated_binding_exception')
      != bootstrap.EXPECTED_UPSTREAM_GENERATED_BINDING_EXCEPTION
  ):
    raise ValueError('v3.3 frozen upstream generated exception changed.')
  if (
      frozen.get('prior_v3_2_evidence')
      != bootstrap.EXPECTED_PRIOR_V3_2_EVIDENCE
  ):
    raise ValueError('v3.3 frozen prior-v3.2 evidence changed.')
  repo = _HERE.parents[2]
  for relative, digest in frozen['file_sha256'].items():
    path = repo / relative
    _reject_confirmation_path(path)
    if _sha256(path) != digest:
      raise ValueError(f'v3.3 frozen file changed: {relative}.')
  return {**frozen, 'path': str(FREEZE_PATH), 'sha256': _sha256(FREEZE_PATH)}


def validate_bundle_committed(frozen: Mapping[str, Any]) -> dict[str, Any]:
  repo = _HERE.parents[2]
  relative = tuple(frozen['file_sha256']) + (
      str(FREEZE_PATH.relative_to(repo)),
  )
  for path in relative:
    subprocess.run(
        ('git', '-C', str(repo), 'ls-files', '--error-unmatch', path),
        check=True,
        capture_output=True,
    )
  if subprocess.check_output(
      ('git', '-C', str(repo), 'diff', '--binary', 'HEAD', '--', *relative)
  ):
    raise ValueError('Committed v3.3 bundle differs from the working tree.')
  if subprocess.check_output(
      ('git', '-C', str(repo), 'diff', '--binary', 'HEAD', '--')
  ):
    raise ValueError('v3.3 requires a globally tracked-clean HEAD.')
  proto_status = subprocess.check_output(
      (
          'git', '-C', str(repo), 'status', '--porcelain=v1',
          '--untracked-files=all', '--', 'src/alphagenome_research/protos',
      ),
      text=True,
  ).splitlines()
  expected_generated = {
      '?? src/alphagenome_research/protos/calibration_scores_pb2.py',
      '?? src/alphagenome_research/protos/calibration_scores_pb2.pyi',
  }
  if set(proto_status) != expected_generated:
    raise ValueError('v3.3 generated protobuf allowlist changed.')
  return {
      'git_head': subprocess.check_output(
          ('git', '-C', str(repo), 'rev-parse', 'HEAD'), text=True
      ).strip(),
      'tracked_clean': True,
      'generated_artifact_exception': sorted(expected_generated),
  }


def validate_external_preflight(
    path: Path, frozen: Mapping[str, Any]
) -> dict[str, Any]:
  _reject_confirmation_path(path)
  record = json.loads(path.read_text(encoding='utf-8'))
  if record.get('script_version') != PREFLIGHT_SCRIPT_VERSION:
    raise ValueError('Wrong v3.3 external preflight version.')
  if record.get('status') != 'pass':
    raise ValueError('v3.3 external preflight did not pass.')
  if record.get('protocol_sha256') != PROTOCOL_SHA256:
    raise ValueError('v3.3 external preflight used another protocol.')
  if record.get('freeze_sha256') != frozen['sha256']:
    raise ValueError('v3.3 external preflight used another freeze.')
  if not record.get('no_model_or_biological_access'):
    raise ValueError('v3.3 preflight imported biological/model code.')
  if not record.get('no_jit_or_array_kernel'):
    raise ValueError('v3.3 preflight performed a JIT/array operation.')
  observation = record['observation']
  if observation.get('jax_enable_compilation_cache') is not False:
    raise ValueError('v3.3 preflight observed an enabled JAX cache.')
  current_environment = {
      name: value for name, value in sorted(os.environ.items())
      if name.startswith(v32.ENVIRONMENT_PREFIXES)
  }
  if observation.get('v3_3_runtime_environment') != current_environment:
    raise ValueError('External and same-launch v3.3 environments differ.')
  preflight_root = Path(frozen['preflight_dir']).resolve()
  validated_logs = {}
  for stream in ('stdout', 'stderr'):
    binding = record['logs'][stream]
    log_path = Path(binding['path']).resolve()
    _reject_confirmation_path(log_path)
    if not log_path.is_relative_to(preflight_root):
      raise ValueError(f'External preflight {stream} escaped its directory.')
    if _sha256(log_path) != binding['sha256']:
      raise ValueError(f'External preflight {stream} log changed.')
    validated_logs[stream] = {
        'path': str(log_path), 'sha256': binding['sha256']
    }
  v32.device_gate.validate_device_observation(observation)
  validate_device_version_manifest(observation, frozen)
  return {
      'path': str(path.resolve()),
      'sha256': _sha256(path),
      **record,
      'validated_logs': validated_logs,
  }


def _tree_digest(paths: Sequence[Path], root: Path) -> str:
  digest = hashlib.sha256()
  for path in sorted(paths):
    digest.update(str(path.relative_to(root)).encode('utf-8'))
    digest.update(b'\0')
    digest.update(bytes.fromhex(_sha256(path)))
  return digest.hexdigest()


def _raw_manifest() -> dict[str, Any]:
  paths = sorted(path for path in (OUTPUT_DIR / 'raw').rglob('*.json'))
  return {
      'artifact_count': len(paths),
      'artifact_sha256': {
          str(path.relative_to(OUTPUT_DIR)): _sha256(path) for path in paths
      },
      'artifact_tree_sha256': _tree_digest(paths, OUTPUT_DIR),
  }


def _assert_attempt_budget(started_monotonic: float, *, scan_storage: bool) -> None:
  elapsed = time.monotonic() - started_monotonic
  if elapsed > MAX_WALL_TIME_SECONDS:
    raise RuntimeError('v3.3 frozen wall-time budget was exceeded.')
  if scan_storage:
    size = sum(
        path.stat().st_size for path in OUTPUT_DIR.rglob('*') if path.is_file()
    )
    if size > MAX_OUTPUT_BYTES:
      raise RuntimeError('v3.3 frozen output-storage budget was exceeded.')


def _write_completion(
    *,
    stop_reason: str | None,
    message: str,
    identity_results: Sequence[Mapping[str, Any]],
    eligible_effect_count: int,
    coalition_results: Sequence[Mapping[str, Any]],
    ood_results: Sequence[Mapping[str, Any]],
    compilers: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
  raw = _raw_manifest()
  _write_new(OUTPUT_DIR / 'RAW_MANIFEST.json', raw)
  import_phases = {
      'pre_model': _sha256(OUTPUT_DIR / 'IMPORT_PROVENANCE_PRE_MODEL.json'),
      'post_model_precompile': _sha256(
          OUTPUT_DIR / 'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json'
      ),
      'postcompile': _sha256(OUTPUT_DIR / 'IMPORT_PROVENANCE.json'),
  }
  record = {
      'status': 'complete' if stop_reason is None else 'controlled_stop',
      'stop_reason': stop_reason,
      'message': message,
      'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION,
      'protocol_sha256': PROTOCOL_SHA256,
      'freeze_sha256': _sha256(FREEZE_PATH),
      'supersession_sha256': SUPERSESSION_SHA256,
      'supersession_commit': SUPERSESSION_COMMIT,
      'identity_count': len(identity_results),
      'eligible_effect_count': eligible_effect_count,
      'all_effects_target_eligible': eligible_effect_count == 12,
      'all_neutrals_retained': sum(
          item['status'] == 'complete'
          and 'neutral' in item['case']['selection_class'].lower()
          for item in identity_results
      ) == 8,
      'identity_invalid_count': sum(
          item['status'] != 'complete' for item in identity_results
      ),
      'coalition_record_count': len(coalition_results),
      'coalition_invalid_count': sum(
          item['status'] != 'complete' for item in coalition_results
      ),
      'ood_anchor_record_count': len(ood_results),
      'ood_invalid_count': sum(
          item['status'] != 'complete' for item in ood_results
      ),
      'scientific_record_count': len(identity_results)
          + len(coalition_results) + len(ood_results),
      'model_apply_count': (
          2 * len(identity_results)
          + 2 * len(coalition_results)
          + 4 * len(ood_results)
      ),
      'id0_noop_all20': sum(
          item['coalition']['coalition_id'] == 0
          and item['status'] == 'complete'
          and item['checks']['id0_identity_endpoint_exact']
          for item in coalition_results
      ) == 20,
      'id255_closure_all20': sum(
          item['coalition']['coalition_id'] == 255
          and item['status'] == 'complete'
          and item['checks']['id255_endpoint_closure_exact']
          for item in coalition_results
      ) == 20,
      'six_row_compiler': dict(compilers['six_row']),
      'eight_row_compiler': dict(compilers['eight_row']),
      'six_row_executable_fingerprint': compilers['six_row'][
          'executable_fingerprint'
      ],
      'eight_row_executable_fingerprint': compilers['eight_row'][
          'executable_fingerprint'
      ],
      'compile_count': 2,
      'import_provenance_sha256': import_phases['postcompile'],
      'import_provenance_phases': import_phases,
      'protobuf_provenance_sha256': _sha256(
          OUTPUT_DIR / 'PROTOBUF_PROVENANCE.json'
      ),
      'raw_manifest': raw,
      'confirmation_model_calls': 0,
      'confirmation_scope_disclosure': (
          'Later-exon metadata/labels were exposed after protocol freeze; '
          'no later-exon model outputs, activations, or interventions are used.'
      ),
      'completed_at_unix_s': time.time(),
  }
  _write_new(OUTPUT_DIR / 'RUN_COMPLETE.json', record)
  return record


def _controlled_stop(
    *,
    reason: str,
    message: str,
    identities: Sequence[Mapping[str, Any]],
    eligible_effect_count: int,
    coalitions: Sequence[Mapping[str, Any]],
    ood: Sequence[Mapping[str, Any]],
    compilers: Mapping[str, Mapping[str, Any]],
) -> None:
  _write_completion(
      stop_reason=reason,
      message=message,
      identity_results=identities,
      eligible_effect_count=eligible_effect_count,
      coalition_results=coalitions,
      ood_results=ood,
      compilers=compilers,
  )


def main() -> None:
  args = _parse_args()
  bootstrap_attestation = consume_bootstrap_attestation()
  for path in (
      OUTPUT_DIR,
      ANALYSIS_DIR,
      PREFLIGHT_DIR,
      PROTOCOL_PATH,
      SUPERSESSION_PATH,
      FREEZE_PATH,
      v32.DEVELOPMENT_VARIANTS_PATH,
      v32.DEVELOPMENT_EXONS_PATH,
      v32.CHECKPOINT_MANIFEST_PATH,
      v32.REFERENCE_BINDINGS_PATH,
  ):
    _reject_confirmation_path(path)
  if args.successful_preflight is not None:
    _reject_confirmation_path(args.successful_preflight)
  if args.checkpoint is not None:
    _reject_confirmation_path(args.checkpoint)
  if _sha256(PROTOCOL_PATH) != PROTOCOL_SHA256:
    raise ValueError('v3.3 protocol hash mismatch.')
  if _sha256(SUPERSESSION_PATH) != SUPERSESSION_SHA256:
    raise ValueError('v3.3 supersession-document hash mismatch.')
  cases = v32.load_development_cases()
  execution_order = coalition_execution_order(cases)
  if args.dry_run:
    print(json.dumps(build_dry_run_plan(
        cases,
        max_variants=args.max_variants,
        max_coalitions=args.max_coalitions,
    ), indent=2))
    return

  environment = v32.assert_v3_2_environment()
  frozen = validate_freeze()
  bundle = validate_bundle_committed(frozen)
  if args.successful_preflight is None:
    raise ValueError('v3.3 requires its successful external preflight.')
  external = validate_external_preflight(args.successful_preflight, frozen)
  same_process = v32.device_gate.collect_device_observation()
  same_process['packages'].update(runtime_version_binding()['packages'])
  v32.device_gate.validate_device_observation(same_process)
  validate_device_version_manifest(same_process, frozen)
  if OUTPUT_DIR.exists() or ANALYSIS_DIR.exists():
    raise FileExistsError('v3.3 output/analysis exists; never resume or retry.')
  checkpoint = v32.v2._checkpoint_path(  # pylint: disable=protected-access
      args.checkpoint
  )
  _reject_confirmation_path(checkpoint)
  if checkpoint.name != v32.route_v3.CHECKPOINT_SNAPSHOT:
    raise ValueError('v3.3 checkpoint snapshot changed.')
  checkpoint_binding = v32.validate_checkpoint(checkpoint)
  reference_object_binding = v32.validate_reference_object()
  start = {
      'attempt_id': ATTEMPT_ID,
      'script_version': SCRIPT_VERSION,
      'status': 'started_append_only_one_shot',
      'protocol_sha256': PROTOCOL_SHA256,
      'supersession_sha256': SUPERSESSION_SHA256,
      'supersession_commit': SUPERSESSION_COMMIT,
      'supersession': {
          'path': str(SUPERSESSION_PATH.resolve()),
          'sha256': SUPERSESSION_SHA256,
          'commit': SUPERSESSION_COMMIT,
      },
      'freeze': frozen,
      'bundle': bundle,
      'external_preflight': external,
      'same_process_preflight': same_process,
      'runtime_environment': environment,
      'runtime_version_binding': runtime_version_binding(),
      'prior_v3_2_evidence': frozen['prior_v3_2_evidence'],
      'same_process_pre_import_bootstrap': bootstrap_attestation,
      'checkpoint_path': str(checkpoint),
      'checkpoint_binding': checkpoint_binding,
      'reference_object_binding': reference_object_binding,
      'reference_sequence_bindings': {
          'path': str(v32.REFERENCE_BINDINGS_PATH.resolve()),
          'sha256': v32.REFERENCE_BINDINGS_SHA256,
      },
      'compile_count_contract': 2,
      'compile_count_by_executable': {'six_row': 1, 'eight_row': 1},
      'scientific_record_count_contract': 5220,
      'model_apply_count_contract': 10600,
      'max_wall_time_seconds': MAX_WALL_TIME_SECONDS,
      'max_output_bytes': MAX_OUTPUT_BYTES,
      'execution_order_contract': {
          'identities': 'manifest orders 0..19',
          'coalition_anchors': 'ID0 then ID255 for orders 0..19',
          'remaining_effects': (
              'orders 0..5,10..15; each IDs1..254 increasing'
          ),
          'remaining_neutrals': (
              'orders 6..9,16..19; each IDs1..254 increasing'
          ),
          'ood': 'orders0..19; IDs0,127,128,255',
      },
      'confirmation_model_calls': 0,
      'confirmation_scope_disclosure': (
          'Later-exon metadata/labels were exposed after protocol freeze; '
          'no later-exon model outputs, activations, or interventions are used.'
      ),
      'started_at_unix_s': time.time(),
  }
  _write_new(START_PATH, start)
  _write_new(OUTPUT_DIR / 'PROTOBUF_PROVENANCE.json', frozen['protobuf_binding'])
  imports_pre_model = import_provenance(frozen)
  _write_new(
      OUTPUT_DIR / 'IMPORT_PROVENANCE_PRE_MODEL.json', imports_pre_model
  )

  identities: list[dict[str, Any]] = []
  coalition_results: list[dict[str, Any]] = []
  ood_results: list[dict[str, Any]] = []
  attempt_started_monotonic = time.monotonic()
  try:
    model_instance = dna_model.create(
        checkpoint,
        model_settings=dna_model.ModelSettings(
            attention_backend=v32.route_v3.ATTENTION_BACKEND
        ),
    )
    params = model_instance._params  # pylint: disable=protected-access
    state = model_instance._state  # pylint: disable=protected-access
    prototype_case = cases[0]
    donor_prototype_case = cases[10]
    _, _, prototype_selection, prototype_target, _, prototype_dna, _ = (
        v32._case_inputs(  # pylint: disable=protected-access
            model_instance, prototype_case
        )
    )
    _, _, _, _, _, donor_prototype_dna, _ = (
        v32._case_inputs(  # pylint: disable=protected-access
            model_instance, donor_prototype_case
        )
    )
    six_interventions = coalition_interventions(prototype_selection, 0)
    eight_intended = eight_row_interventions(
        prototype_selection, 0, unrelated=False
    )
    eight_unrelated = eight_row_interventions(
        prototype_selection, 0, unrelated=True
    )
    signatures = {
        'selection': v32.pytree_signature(prototype_selection),
        'target': v32.pytree_signature(prototype_target),
        'six_interventions': v32.pytree_signature(six_interventions),
        'eight_interventions': v32.pytree_signature(eight_intended),
    }
    v32.assert_same_program_signature(
        signatures['eight_interventions'], eight_unrelated
    )
    raw_six_apply = (
        dna_model
        .create_splice_classification_logit_margin_superset_graph_apply(
            model_instance._metadata,  # pylint: disable=protected-access
            attention_backend=v32.route_v3.ATTENTION_BACKEND,
        )
    )
    raw_eight_apply = (
        dna_model
        .create_splice_classification_logit_margin_eight_row_superset_graph_apply(
            model_instance._metadata,  # pylint: disable=protected-access
            attention_backend=v32.route_v3.ATTENTION_BACKEND,
        )
    )
    imports_post_model = import_provenance(frozen)
    v32.assert_import_provenance_stable(imports_pre_model, imports_post_model)
    _write_new(
        OUTPUT_DIR / 'IMPORT_PROVENANCE_POST_MODEL_PRECOMPILE.json',
        imports_post_model,
    )
    six_args = (
        params,
        state,
        prototype_dna,
        jnp.zeros((6,), jnp.int32),
        prototype_selection,
        six_interventions,
        prototype_target,
    )
    eight_dna = np.concatenate(
        (np.asarray(prototype_dna), np.asarray(donor_prototype_dna)[:2]),
        axis=0,
    )
    eight_args = (
        params,
        state,
        eight_dna,
        jnp.zeros((8,), jnp.int32),
        prototype_selection,
        eight_intended,
        prototype_target,
    )
    compile_start = time.perf_counter()
    six_lowered = jax.jit(raw_six_apply).lower(*six_args)
    six_compiled = six_lowered.compile()
    six_compile_seconds = time.perf_counter() - compile_start
    compile_start = time.perf_counter()
    eight_lowered = jax.jit(raw_eight_apply).lower(*eight_args)
    eight_compiled = eight_lowered.compile()
    eight_compile_seconds = time.perf_counter() - compile_start
    compilers = {
        'six_row': _compiler_artifacts(
            six_lowered,
            six_compiled,
            six_compile_seconds,
            executable_name='six_row',
        ),
        'eight_row': _compiler_artifacts(
            eight_lowered,
            eight_compiled,
            eight_compile_seconds,
            executable_name='eight_row',
        ),
    }
    imports = import_provenance(frozen)
    v32.assert_import_provenance_stable(imports_post_model, imports)
    _write_new(OUTPUT_DIR / 'IMPORT_PROVENANCE.json', imports)

    identities_by_id: dict[str, dict[str, Any]] = {}
    common_by_id: dict[str, tuple[Any, ...]] = {}
    for execution_index, case in enumerate(cases):
      artifact, common = _run_identity(
          six_compiled,
          model_instance,
          case,
          params,
          state,
          signatures,
          frozen['sha256'],
          compilers['six_row']['executable_fingerprint'],
          execution_index,
      )
      identities.append(artifact)
      identities_by_id[case.variant_id] = artifact
      common_by_id[case.variant_id] = common
      _assert_attempt_budget(attempt_started_monotonic, scan_storage=True)
    identity_failures = [
        item['case']['variant_id']
        for item in identities if item['status'] != 'complete'
    ]
    if identity_failures:
      _controlled_stop(
          reason='identity_tooling_failure',
          message=f'Identity Gate 0 failed for {identity_failures}.',
          identities=identities,
          eligible_effect_count=0,
          coalitions=coalition_results,
          ood=ood_results,
          compilers=compilers,
      )
      return
    eligible_effects = [
        case for case in cases
        if case.is_effect
        and identities_by_id[case.variant_id]['direction_gate'][
            'eligible_for_causal_census'
        ]
    ]
    _write_new(OUTPUT_DIR / 'TARGET_ELIGIBILITY.json', {
        'eligible_effects': [case.variant_id for case in eligible_effects],
        'ineligible_effects': [
            case.variant_id for case in cases
            if case.is_effect and case not in eligible_effects
        ],
        'neutral_controls': [
            case.variant_id for case in cases if not case.is_effect
        ],
    })
    if len(eligible_effects) != 12:
      _controlled_stop(
          reason='target_comparability_failure',
          message=(
              f'v3.3 requires all 12 eligible effects; got '
              f'{len(eligible_effects)}.'
          ),
          identities=identities,
          eligible_effect_count=len(eligible_effects),
          coalitions=coalition_results,
          ood=ood_results,
          compilers=compilers,
      )
      return

    cases_by_order = {case.order: case for case in cases}
    coalition_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for offset, (order, coalition_id) in enumerate(execution_order, start=20):
      case = cases_by_order[order]
      result = _run_coalition(
          six_compiled,
          case,
          common_by_id[case.variant_id],
          params,
          state,
          identities_by_id[case.variant_id],
          coalition_id,
          signatures,
          frozen['sha256'],
          compilers['six_row']['executable_fingerprint'],
          offset,
      )
      coalition_results.append(result)
      if coalition_id in ANCHOR_IDS:
        coalition_by_key[(order, coalition_id)] = result
      _assert_attempt_budget(
          attempt_started_monotonic,
          scan_storage=(len(coalition_results) % 64 == 0),
      )
      if result['status'] != 'complete':
        reason = (
            'gate0_closure_failure'
            if coalition_id in (0, 255) and len(coalition_results) <= 40
            else 'coalition_tooling_failure'
        )
        _controlled_stop(
            reason=reason,
            message=(
                f'Coalition audit failed at order={order}, '
                f'coalition_id={coalition_id}.'
            ),
            identities=identities,
            eligible_effect_count=len(eligible_effects),
            coalitions=coalition_results,
            ood=ood_results,
            compilers=compilers,
        )
        return

    for order in range(20):
      recipient = cases_by_order[order]
      donor = cases_by_order[OOD_DONOR_ORDER[order]]
      for coalition_id in ANCHOR_IDS:
        execution_index = 20 + 5120 + len(ood_results)
        result = _run_ood_anchor(
            eight_compiled,
            recipient,
            donor,
            common_by_id[recipient.variant_id],
            common_by_id[donor.variant_id],
            params,
            state,
            identities_by_id[recipient.variant_id],
            identities_by_id[donor.variant_id],
            coalition_by_key[(order, coalition_id)],
            coalition_id,
            signatures,
            frozen['sha256'],
            compilers['eight_row']['executable_fingerprint'],
            execution_index,
        )
        ood_results.append(result)
        _assert_attempt_budget(attempt_started_monotonic, scan_storage=True)
        if result['status'] != 'complete':
          _controlled_stop(
              reason='ood_tooling_failure',
              message=(
                  f'OOD anchor audit failed at order={order}, '
                  f'coalition_id={coalition_id}.'
              ),
              identities=identities,
              eligible_effect_count=len(eligible_effects),
              coalitions=coalition_results,
              ood=ood_results,
              compilers=compilers,
          )
          return

    _assert_attempt_budget(attempt_started_monotonic, scan_storage=True)
    _write_completion(
        stop_reason=None,
        message='All frozen v3.3 development raw families completed.',
        identity_results=identities,
        eligible_effect_count=len(eligible_effects),
        coalition_results=coalition_results,
        ood_results=ood_results,
        compilers=compilers,
    )
  except Exception as error:
    _write_new(OUTPUT_DIR / 'TERMINAL_FAILURE.json', {
        'status': 'terminal_failure',
        'type': type(error).__name__,
        'message': str(error),
        'traceback': ''.join(traceback.format_exception(error)),
        'created_at_unix_s': time.time(),
    })
    raise


if __name__ == '__main__':
  main()
