# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the experimental tiled attention implementation."""

import math

from absl.testing import absltest
from absl.testing import parameterized
from alphagenome_research.model import tiled_attention
import jax
import jax.numpy as jnp
import numpy as np


def _dense_reference(
    q,
    k,
    v,
    coarse_bias,
    bias_bin_size=16,
    soft_cap=5.0,
    dot_precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
):
  seq_len = q.shape[1]
  positions = jnp.arange(seq_len) // bias_bin_size
  bias = jnp.take(jnp.take(coarse_bias, positions, axis=1), positions, axis=2)
  bias = jnp.moveaxis(bias, 3, 1)
  logits = jnp.einsum(
      'bshd,bk1d->bhsk',
      q,
      k,
      precision=dot_precision,
      preferred_element_type=jnp.float32,
  ) / math.sqrt(q.shape[-1])
  logits = logits + bias.astype(jnp.float32)
  logits = jnp.tanh(logits / soft_cap) * soft_cap
  weights = jax.nn.softmax(logits, axis=-1)
  return jnp.einsum(
      'bhsk,bk1v->bshv',
      weights,
      v,
      precision=dot_precision,
      preferred_element_type=jnp.float32,
  ).astype(q.dtype)


def _inputs(seq_len, *, batch=1, heads=3, q_width=8, v_width=7):
  keys = jax.random.split(jax.random.key(seq_len), 4)
  coarse_len = math.ceil(seq_len / 16)
  return (
      jax.random.normal(keys[0], (batch, seq_len, heads, q_width)),
      jax.random.normal(keys[1], (batch, seq_len, 1, q_width)),
      jax.random.normal(keys[2], (batch, seq_len, 1, v_width)),
      jax.random.normal(keys[3], (batch, coarse_len, coarse_len, heads)),
  )


class TiledAttentionTest(parameterized.TestCase):

  def assert_bf16_numerically_equivalent(self, actual, expected):
    """Checks the bounded drift expected from differently tiled BF16 dots."""
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    max_absolute_error = np.max(np.abs(actual - expected))
    cosine_similarity = np.vdot(actual.ravel(), expected.ravel()) / (
        np.linalg.norm(actual) * np.linalg.norm(expected)
    )
    self.assertLessEqual(max_absolute_error, 5e-3)
    self.assertGreaterEqual(cosine_similarity, 0.9999)

  @parameterized.parameters(
      # Both tile tails and a coarse-bin tail.
      dict(seq_len=19, query_tile=7, key_tile=5),
      dict(seq_len=32, query_tile=8, key_tile=16),
      dict(seq_len=33, query_tile=64, key_tile=64),
      dict(seq_len=1, query_tile=4, key_tile=3),
  )
  def test_matches_dense_reference(self, seq_len, query_tile, key_tile):
    inputs = _inputs(seq_len, batch=2)
    expected = _dense_reference(*inputs)
    actual = tiled_attention.tiled_attention(
        *inputs, query_tile_size=query_tile, key_tile_size=key_tile
    )
    self.assert_bf16_numerically_equivalent(actual, expected)

  def test_matches_dense_with_large_logits_and_bias(self):
    q, k, v, bias = _inputs(21)
    q, k, bias = q * 100, k * 100, bias * 100
    expected = _dense_reference(q, k, v, bias)
    actual = tiled_attention.tiled_attention(
        q, k, v, bias, query_tile_size=6, key_tile_size=4
    )
    self.assert_bf16_numerically_equivalent(actual, expected)

  def test_jit_matches_eager(self):
    inputs = _inputs(17)
    function = jax.jit(
        lambda q, k, v, b: tiled_attention.tiled_attention(
            q, k, v, b, query_tile_size=8, key_tile_size=7
        )
    )
    self.assert_bf16_numerically_equivalent(
        function(*inputs), _dense_reference(*inputs)
    )

  def test_alphagenome_tensor_widths(self):
    inputs = _inputs(16, heads=8, q_width=128, v_width=192)
    actual = tiled_attention.tiled_attention(
        *inputs, query_tile_size=8, key_tile_size=8
    )
    self.assertEqual(actual.shape, (1, 16, 8, 192))
    self.assert_bf16_numerically_equivalent(
        actual, _dense_reference(*inputs)
    )

  def test_fp32_online_recurrence_matches_dense(self):
    inputs = _inputs(29, batch=2)
    precision = jax.lax.Precision.HIGHEST
    expected = _dense_reference(*inputs, dot_precision=precision)
    actual = tiled_attention.tiled_attention(
        *inputs,
        query_tile_size=7,
        key_tile_size=6,
        dot_precision=precision,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

  @parameterized.parameters(
      lambda q, k, v, b: (q[:, :0], k[:, :0], v[:, :0], b[:, :0, :0]),
      lambda q, k, v, b: (q, k[:, :-1], v, b),
      lambda q, k, v, b: (q, k, v, b[:, :-1, :-1]),
  )
  def test_rejects_invalid_shapes(self, mutate):
    with self.assertRaises(ValueError):
      tiled_attention.tiled_attention(*mutate(*_inputs(17)))


if __name__ == '__main__':
  absltest.main()
