# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Memory-bounded exact attention for AlphaGenome's sequence transformer.

This is an experimental, pure-JAX reference implementation of tiled online
softmax.  It intentionally is not connected to ``MHABlock`` yet.  Unlike a
conventional attention implementation, it materializes only one query/key tile
of logits and expands the coarse pair bias only for that tile.
"""

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-importing-member


def tiled_attention(
    q: Float[Array, 'B S H Q'],
    k: Float[Array, 'B S 1 Q'],
    v: Float[Array, 'B S 1 V'],
    coarse_bias: Float[Array, 'B C C H'],
    *,
    query_tile_size: int = 128,
    key_tile_size: int = 128,
    bias_bin_size: int = 16,
    logits_soft_cap: float = 5.0,
    dot_precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
) -> Float[Array, 'B S H V']:
  """Computes exact multi-query attention using bounded temporary storage.

  Args:
    q: Queries with a distinct vector per head.
    k: Keys shared by all heads (multi-query attention).
    v: Values shared by all heads.  Their width may differ from ``q``.
    coarse_bias: Pair bias before nearest-neighbour expansion.  Position ``i``
      uses coarse index ``i // bias_bin_size`` on each sequence axis.
    query_tile_size: Number of query positions processed at once.
    key_tile_size: Number of key positions incorporated per online update.
    bias_bin_size: Sequence positions represented by one coarse bias position.
    logits_soft_cap: Symmetric tanh soft cap applied after adding the bias.
    dot_precision: JAX dot precision or algorithm.  The default matches
      ``MHABlock``.  Since tiled and dense BF16 matmuls have different
      reduction orders, they are numerically, rather than bitwise, equivalent.

  Returns:
    Attention output with the same floating dtype as ``q``.  Softmax state and
    value accumulation are maintained in float32.

  Raises:
    ValueError: If ranks, dimensions, tile sizes, or bias coverage are invalid.
  """
  if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or coarse_bias.ndim != 4:
    raise ValueError('q, k, v, and coarse_bias must all have rank 4.')
  batch_size, seq_len, num_heads, query_width = q.shape
  if seq_len < 1:
    raise ValueError('Sequence length must be positive.')
  if query_tile_size < 1 or key_tile_size < 1 or bias_bin_size < 1:
    raise ValueError('Tile and bias-bin sizes must be positive.')
  if logits_soft_cap <= 0:
    raise ValueError('logits_soft_cap must be positive.')
  if k.shape != (batch_size, seq_len, 1, query_width):
    raise ValueError(
        'k must have shape [batch, sequence, 1, q.shape[-1]].'
    )
  if v.shape[:3] != (batch_size, seq_len, 1):
    raise ValueError('v must have shape [batch, sequence, 1, value_width].')
  required_coarse_len = math.ceil(seq_len / bias_bin_size)
  if coarse_bias.shape != (
      batch_size,
      required_coarse_len,
      required_coarse_len,
      num_heads,
  ):
    raise ValueError(
        'coarse_bias must exactly cover the sequence with shape '
        f'[{batch_size}, {required_coarse_len}, {required_coarse_len}, '
        f'{num_heads}].'
    )

  num_query_tiles = math.ceil(seq_len / query_tile_size)
  num_key_tiles = math.ceil(seq_len / key_tile_size)
  padded_query_len = num_query_tiles * query_tile_size
  padded_key_len = num_key_tiles * key_tile_size
  q = jnp.pad(q, ((0, 0), (0, padded_query_len - seq_len), (0, 0), (0, 0)))
  k = jnp.pad(k, ((0, 0), (0, padded_key_len - seq_len), (0, 0), (0, 0)))
  v = jnp.pad(v, ((0, 0), (0, padded_key_len - seq_len), (0, 0), (0, 0)))
  value_width = v.shape[-1]
  scale = jnp.asarray(1.0 / math.sqrt(query_width), jnp.float32)
  soft_cap = jnp.asarray(logits_soft_cap, jnp.float32)
  output = jnp.zeros(
      (batch_size, padded_query_len, num_heads, value_width), jnp.float32
  )

  def process_query_tile(query_tile_index, output):
    query_start = query_tile_index * query_tile_size
    query_tile = jax.lax.dynamic_slice(
        q, (0, query_start, 0, 0),
        (batch_size, query_tile_size, num_heads, query_width)
    )
    query_positions = query_start + jnp.arange(query_tile_size)
    query_bins = jnp.minimum(
        query_positions // bias_bin_size, required_coarse_len - 1
    )
    row_bias = jnp.take(coarse_bias, query_bins, axis=1)
    initial_state = (
        jnp.full((batch_size, query_tile_size, num_heads), -jnp.inf),
        jnp.zeros((batch_size, query_tile_size, num_heads), jnp.float32),
        jnp.zeros(
            (batch_size, query_tile_size, num_heads, value_width), jnp.float32
        ),
    )

    def process_key_tile(key_tile_index, state):
      row_max, denominator, numerator = state
      key_start = key_tile_index * key_tile_size
      key_tile = jax.lax.dynamic_slice(
          k, (0, key_start, 0, 0),
          (batch_size, key_tile_size, 1, query_width)
      )
      value_tile = jax.lax.dynamic_slice(
          v, (0, key_start, 0, 0),
          (batch_size, key_tile_size, 1, value_width)
      )
      key_positions = key_start + jnp.arange(key_tile_size)
      key_bins = jnp.minimum(
          key_positions // bias_bin_size, required_coarse_len - 1
      )
      bias_tile = jnp.take(row_bias, key_bins, axis=2)
      # [batch, query, key, head] -> [batch, head, query, key].
      bias_tile = jnp.moveaxis(bias_tile, 3, 1)
      logits = jnp.einsum(
          'bqhd,bk1d->bhqk',
          query_tile,
          key_tile,
          precision=dot_precision,
          preferred_element_type=jnp.float32,
      )
      logits = logits * scale + bias_tile.astype(jnp.float32)
      logits = jnp.tanh(logits / soft_cap) * soft_cap
      # Mask after the soft cap: tanh(-inf) would otherwise become finite.
      logits = jnp.where(key_positions < seq_len, logits, -jnp.inf)
      logits = jnp.moveaxis(logits, 1, 2)

      tile_max = jnp.max(logits, axis=-1)
      new_max = jnp.maximum(row_max, tile_max)
      old_scale = jnp.exp(row_max - new_max)
      probabilities = jnp.exp(logits - new_max[..., None])
      denominator = denominator * old_scale + jnp.sum(
          probabilities, axis=-1
      )
      numerator = numerator * old_scale[..., None] + jnp.einsum(
          'bqhk,bk1v->bqhv',
          probabilities,
          value_tile,
          precision=dot_precision,
          preferred_element_type=jnp.float32,
      )
      return new_max, denominator, numerator

    _, denominator, numerator = jax.lax.fori_loop(
        0, num_key_tiles, process_key_tile, initial_state
    )
    result = numerator / denominator[..., None]
    return jax.lax.dynamic_update_slice(output, result, (0, query_start, 0, 0))

  output = jax.lax.fori_loop(
      0, num_query_tiles, process_query_tile, output
  )
  return output[:, :seq_len].astype(q.dtype)
