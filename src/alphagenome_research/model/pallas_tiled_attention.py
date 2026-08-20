# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Experimental forward-only Pallas attention kernel for AlphaGenome.

This module provides the opt-in ``pallas_tiled`` backend used by ``MHABlock``.
It implements the online-softmax algorithm in :mod:`tiled_attention`, adapted
to AlphaGenome's multi-query tensor shapes and unequal query/value widths.
"""

import functools
import math

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-importing-member


def _register_rtx_3090_for_jax_pallas() -> None:
  """Works around JAX 0.11's incomplete Triton GPU-name registry.

  JAX 0.11 recognizes A10 (the same SM 8.6 architecture) but omits the RTX
  3090 marketing name, causing lowering to fail before compilation.  Keep this
  narrow workaround local to the prototype until upstream recognizes the GPU.
  """
  # pylint: disable=g-import-not-at-top,protected-access
  from jax._src.pallas.triton import gpu_info
  device = jax.devices()[0]
  if device.device_kind == 'NVIDIA GeForce RTX 3090':
    gpu_info.registry.setdefault(
        device.device_kind,
        lambda: gpu_info.GpuInfo(
            gpu_version=None, arch_name='8.6', compute_capability=86
        ),
    )


def _forward_kernel(
    q_ref,
    k_ref,
    v_ref,
    bias_ref,
    out_ref,
    *,
    sequence_length: int,
    query_width: int,
    value_width: int,
    query_tile_size: int,
    key_tile_size: int,
    bias_bin_size: int,
    logits_soft_cap: float,
):
  """One Triton program computes one (batch, head, query tile)."""
  query_tile_index = pl.program_id(0)
  query_start = query_tile_index * query_tile_size

  q_mask = jnp.arange(q_ref.shape[-1])[None, :] < query_width
  query = plgpu.load(q_ref, mask=q_mask, other=0.0)
  row_max = jnp.full((query_tile_size,), -jnp.inf, jnp.float32)
  denominator = jnp.zeros((query_tile_size,), jnp.float32)
  # The Triton block is power-of-two padded to 256 even though AlphaGenome's
  # logical value width is 192.  Accumulate the padded lanes and mask them only
  # on the final store; this keeps the matmul legal without changing the API.
  numerator = jnp.zeros((query_tile_size, v_ref.shape[-1]), jnp.float32)
  scale = 1.0 / math.sqrt(query_width)
  log2_e = math.log2(math.e)

  query_positions = query_start + jnp.arange(query_tile_size)
  query_bins = query_positions // bias_bin_size

  def process_key_tile(key_tile_index, state):
    row_max, denominator, numerator = state
    key_start = key_tile_index * key_tile_size
    key_slice = pl.dslice(key_start, key_tile_size)
    key_mask = jnp.arange(k_ref.shape[-1])[None, :] < query_width
    key = plgpu.load(k_ref.at[key_slice, :], mask=key_mask, other=0.0)
    value_mask = jnp.arange(v_ref.shape[-1])[None, :] < value_width
    value = plgpu.load(v_ref.at[key_slice, :], mask=value_mask, other=0.0)

    key_positions = key_start + jnp.arange(key_tile_size)
    key_bins = key_positions // bias_bin_size
    bias = plgpu.load(bias_ref.at[query_bins[:, None], key_bins[None, :]])
    logits = plgpu.dot(query, key.T) * scale + bias.astype(jnp.float32)
    logits = jnp.tanh(logits / logits_soft_cap) * logits_soft_cap

    tile_max = jnp.max(logits, axis=1)
    new_max = jnp.maximum(row_max, tile_max)
    # exp2 is the native Triton operation.  Multiplication by log2(e) makes
    # this exactly the base-e online-softmax recurrence up to approximation.
    previous_correction = jnp.exp2((row_max - new_max) * log2_e)
    weights = jnp.exp2((logits - new_max[:, None]) * log2_e)
    denominator = denominator * previous_correction + jnp.sum(weights, axis=1)
    numerator = numerator * previous_correction[:, None] + plgpu.dot(
        weights.astype(value.dtype), value
    )
    return new_max, denominator, numerator

  _, denominator, numerator = lax.fori_loop(
      0,
      sequence_length // key_tile_size,
      process_key_tile,
      (row_max, denominator, numerator),
  )
  result = numerator / denominator[:, None]
  output_mask = jnp.arange(out_ref.shape[-1])[None, :] < value_width
  plgpu.store(out_ref, result.astype(out_ref.dtype), mask=output_mask)


@functools.partial(
    jax.jit,
    static_argnames=(
        'query_tile_size',
        'key_tile_size',
        'bias_bin_size',
        'logits_soft_cap',
        'num_warps',
        'num_stages',
    ),
)
def pallas_tiled_attention(
    q: Float[Array, 'B S H Q'],
    k: Float[Array, 'B S 1 Q'],
    v: Float[Array, 'B S 1 V'],
    coarse_bias: Float[Array, 'B C C H'],
    *,
    query_tile_size: int = 64,
    key_tile_size: int = 64,
    bias_bin_size: int = 16,
    logits_soft_cap: float = 5.0,
    num_warps: int = 8,
    num_stages: int = 2,
) -> Float[Array, 'B S H V']:
  """Runs forward-only fused attention with FP32 online-softmax state.

  The phase-2 prototype intentionally requires full tiles and CUDA.  These
  constraints keep masking and tail behavior out of the first verified kernel.
  ``q``/``k`` widths must be 128 and the value/output width must be 192, which
  are the dimensions used by AlphaGenome's sequence transformer.
  """
  if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or coarse_bias.ndim != 4:
    raise ValueError('q, k, v, and coarse_bias must all have rank 4.')
  batch_size, sequence_length, num_heads, query_width = q.shape
  value_width = v.shape[-1]
  if sequence_length < 1:
    raise ValueError('Sequence length must be positive.')
  if query_width != 128 or value_width != 192:
    raise ValueError(
        'This prototype requires query width 128 and value width 192.'
    )
  if k.shape != (batch_size, sequence_length, 1, query_width):
    raise ValueError('k has an incompatible shape.')
  if v.shape != (batch_size, sequence_length, 1, value_width):
    raise ValueError('v has an incompatible shape.')
  if query_tile_size != 64 or key_tile_size != 64:
    raise ValueError('This prototype currently supports only 64x64 tiles.')
  if sequence_length % query_tile_size or sequence_length % key_tile_size:
    raise ValueError('Sequence length must be a multiple of both tile sizes.')
  if bias_bin_size < 1 or sequence_length % bias_bin_size:
    raise ValueError('Sequence length must be divisible by bias_bin_size.')
  coarse_length = sequence_length // bias_bin_size
  if coarse_bias.shape != (
      batch_size, coarse_length, coarse_length, num_heads
  ):
    raise ValueError('coarse_bias has an incompatible shape.')
  if logits_soft_cap <= 0:
    raise ValueError('logits_soft_cap must be positive.')

  _register_rtx_3090_for_jax_pallas()

  # A separate program per head permits the singleton K/V head to be reused.
  grid = (sequence_length // query_tile_size, batch_size, num_heads)
  padded_value_width = pl.next_power_of_2(value_width)
  kernel = functools.partial(
      _forward_kernel,
      sequence_length=sequence_length,
      query_width=query_width,
      value_width=value_width,
      query_tile_size=query_tile_size,
      key_tile_size=key_tile_size,
      bias_bin_size=bias_bin_size,
      logits_soft_cap=logits_soft_cap,
  )
  return pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=[
          pl.BlockSpec(
              (None, query_tile_size, None, query_width),
              lambda query_tile, batch, head: (batch, query_tile, head, 0),
          ),
          pl.BlockSpec(
              (None, sequence_length, None, query_width),
              lambda _, batch, _head: (batch, 0, 0, 0),
          ),
          pl.BlockSpec(
              (None, sequence_length, None, padded_value_width),
              lambda _, batch, _head: (batch, 0, 0, 0),
          ),
          pl.BlockSpec(
              (None, coarse_length, coarse_length, None),
              lambda _, batch, head: (batch, 0, 0, head),
          ),
      ],
      out_specs=pl.BlockSpec(
          (None, query_tile_size, None, padded_value_width),
          lambda query_tile, batch, head: (batch, query_tile, head, 0),
      ),
      out_shape=jax.ShapeDtypeStruct(
          (batch_size, sequence_length, num_heads, value_width), q.dtype
      ),
      compiler_params=plgpu.CompilerParams(
          num_warps=num_warps, num_stages=num_stages
      ),
      name='alphagenome_pallas_tiled_attention',
  )(q, k, v, coarse_bias)
