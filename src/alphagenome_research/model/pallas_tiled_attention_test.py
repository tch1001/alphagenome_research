# Copyright 2026 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License");

"""GPU tests for the experimental Pallas attention kernel."""

import math

from absl.testing import absltest
from alphagenome_research.model import pallas_tiled_attention
from alphagenome_research.model import tiled_attention
import jax
import jax.numpy as jnp
import numpy as np


def _has_cuda():
  return any(device.platform == 'gpu' for device in jax.devices())


class PallasTiledAttentionTest(absltest.TestCase):

  def test_matches_pure_jax_oracle_at_alphagenome_widths(self):
    if not _has_cuda():
      self.skipTest('The Pallas Triton backend requires a CUDA GPU.')
    sequence_length = 128
    batch_size = 1
    heads = 2
    keys = jax.random.split(jax.random.key(2026), 4)
    q = jax.random.normal(
        keys[0], (batch_size, sequence_length, heads, 128), dtype=jnp.bfloat16
    )
    k = jax.random.normal(
        keys[1], (batch_size, sequence_length, 1, 128), dtype=jnp.bfloat16
    )
    v = jax.random.normal(
        keys[2], (batch_size, sequence_length, 1, 192), dtype=jnp.bfloat16
    )
    coarse_length = math.ceil(sequence_length / 16)
    bias = jax.random.normal(
        keys[3], (batch_size, coarse_length, coarse_length, heads)
    )
    expected = tiled_attention.tiled_attention(
        q, k, v, bias, query_tile_size=64, key_tile_size=64
    )
    actual = pallas_tiled_attention.pallas_tiled_attention(q, k, v, bias)
    actual, expected = jax.device_get((actual, expected))
    np.testing.assert_allclose(
        np.asarray(actual, np.float32),
        np.asarray(expected, np.float32),
        rtol=2e-2,
        atol=2e-2,
    )

  def test_rejects_non_alpha_widths(self):
    with self.assertRaises(ValueError):
      pallas_tiled_attention.pallas_tiled_attention(
          jnp.zeros((1, 64, 1, 8)),
          jnp.zeros((1, 64, 1, 8)),
          jnp.zeros((1, 64, 1, 7)),
          jnp.zeros((1, 4, 4, 1)),
      )

  def test_rejects_empty_sequence(self):
    with self.assertRaisesRegex(ValueError, 'Sequence length must be positive'):
      pallas_tiled_attention.pallas_tiled_attention(
          jnp.zeros((1, 0, 1, 128), dtype=jnp.bfloat16),
          jnp.zeros((1, 0, 1, 128), dtype=jnp.bfloat16),
          jnp.zeros((1, 0, 1, 192), dtype=jnp.bfloat16),
          jnp.zeros((1, 0, 0, 1)),
      )


if __name__ == '__main__':
  absltest.main()
