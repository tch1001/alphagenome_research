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

from absl.testing import absltest
from absl.testing import parameterized
from alphagenome_research.model import attention
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from unittest import mock


class ConvolutionsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._batch_size = 4
    self._sequence_length = 4096
    self._pair_sequence_length = self._sequence_length // 16
    self._hidden_size = 64

  def test_mlp_block_output_shape(self):
    """Tests that MLPBlock produces the expected output shape."""
    rng = jax.random.PRNGKey(42)

    def _forward(x):
      module = attention.MLPBlock()
      return module(x, is_training=False)

    init, apply = hk.transform_with_state(_forward)
    x = jnp.zeros((self._batch_size, self._sequence_length, self._hidden_size))
    params, state = init(rng, x)
    output, _ = apply(params, state, rng, x)
    chex.assert_shape(
        output, (self._batch_size, self._sequence_length, self._hidden_size)
    )

  def test_pair_mlp_block_output_shape(self):
    """Tests that PairMLPBlock produces the expected output shape."""
    rng = jax.random.PRNGKey(42)

    def _forward(x):
      module = attention.PairMLPBlock()
      return module(x)

    init, apply = hk.transform_with_state(_forward)
    x = jnp.zeros((
        self._batch_size,
        self._pair_sequence_length,
        self._pair_sequence_length,
        self._hidden_size,
    ))
    params, state = init(rng, x)
    output, _ = apply(params, state, rng, x)
    chex.assert_shape(
        output,
        (
            self._batch_size,
            self._pair_sequence_length,
            self._pair_sequence_length,
            self._hidden_size,
        ),
    )

  def test_attention_bias_backend_preserves_parameters(self):
    pair_x = jnp.zeros((1, 4, 4, 16), jnp.float32)

    def forward(x, backend):
      return attention.AttentionBiasBlock(attention_backend=backend)(
          x, is_training=False
      )

    transformed = hk.transform_with_state(forward)
    rng = jax.random.PRNGKey(7)
    dense_params, dense_state = transformed.init(
        rng, pair_x, attention.ATTENTION_BACKEND_DENSE
    )
    tiled_params, tiled_state = transformed.init(
        rng, pair_x, attention.ATTENTION_BACKEND_PALLAS_TILED
    )
    jax.tree.map(np.testing.assert_array_equal, dense_params, tiled_params)
    jax.tree.map(np.testing.assert_array_equal, dense_state, tiled_state)

    dense, _ = transformed.apply(
        dense_params,
        dense_state,
        None,
        pair_x,
        attention.ATTENTION_BACKEND_DENSE,
    )
    tiled, _ = transformed.apply(
        dense_params,
        dense_state,
        None,
        pair_x,
        attention.ATTENTION_BACKEND_PALLAS_TILED,
    )
    chex.assert_shape(dense, (1, 8, 64, 64))
    chex.assert_shape(tiled, (1, 4, 4, 8))
    np.testing.assert_array_equal(
        dense, jnp.moveaxis(jnp.repeat(jnp.repeat(tiled, 16, 1), 16, 2), 3, 1)
    )

  def test_mha_default_is_explicit_dense_and_tiled_reuses_parameters(self):
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 64, 32))
    coarse_bias = jax.random.normal(jax.random.PRNGKey(2), (1, 4, 4, 8))
    dense_bias = jnp.moveaxis(
        jnp.repeat(jnp.repeat(coarse_bias, 16, 1), 16, 2), 3, 1
    )

    def forward(inputs, bias, backend):
      kwargs = {} if backend == 'default' else {'attention_backend': backend}
      return attention.MHABlock(**kwargs)(inputs, bias, is_training=False)

    transformed = hk.transform_with_state(forward)
    rng = jax.random.PRNGKey(3)
    default_params, default_state = transformed.init(
        rng, x, dense_bias, 'default'
    )
    dense_params, dense_state = transformed.init(
        rng, x, dense_bias, attention.ATTENTION_BACKEND_DENSE
    )

    # A CPU reference for the fused call lets this test validate integration
    # without requiring Triton.  The kernel itself has separate GPU tests.
    def tiled_reference(q, k, v, bias):
      expanded_bias = jnp.moveaxis(
          jnp.repeat(jnp.repeat(bias, 16, 1), 16, 2), 3, 1
      )
      logits = jnp.einsum(
          'bshc,bS1c->bhsS',
          q,
          k,
          precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
          preferred_element_type=jnp.float32,
      ) / jnp.sqrt(128.0)
      logits = (logits + expanded_bias).astype(jnp.float32)
      logits = jnp.tanh(logits / 5.0) * 5.0
      weights = jax.nn.softmax(logits, axis=-1)
      return jnp.einsum(
          'bhsS,bS1c->bshc',
          weights,
          v,
          precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
      ).astype(q.dtype)

    with mock.patch.object(
        attention.pallas_tiled_attention,
        'pallas_tiled_attention',
        side_effect=tiled_reference,
    ):
      tiled_params, tiled_state = transformed.init(
          rng,
          x,
          coarse_bias,
          attention.ATTENTION_BACKEND_PALLAS_TILED,
      )
      tiled, _ = transformed.apply(
          default_params,
          default_state,
          None,
          x,
          coarse_bias,
          attention.ATTENTION_BACKEND_PALLAS_TILED,
      )

    jax.tree.map(np.testing.assert_array_equal, default_params, dense_params)
    jax.tree.map(np.testing.assert_array_equal, default_params, tiled_params)
    jax.tree.map(np.testing.assert_array_equal, default_state, dense_state)
    jax.tree.map(np.testing.assert_array_equal, default_state, tiled_state)
    default, _ = transformed.apply(
        default_params, default_state, None, x, dense_bias, 'default'
    )
    explicit_dense, _ = transformed.apply(
        default_params,
        default_state,
        None,
        x,
        dense_bias,
        attention.ATTENTION_BACKEND_DENSE,
    )
    np.testing.assert_array_equal(default, explicit_dense)
    np.testing.assert_allclose(default, tiled, rtol=2e-5, atol=2e-5)

  def test_invalid_attention_backend_fails_early(self):
    with self.assertRaisesRegex(ValueError, 'Unsupported attention backend'):
      attention.validate_attention_backend('typo')


if __name__ == "__main__":
  absltest.main()
