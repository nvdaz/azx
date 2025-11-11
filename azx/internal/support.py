import jax
import jax.numpy as jnp


class DiscreteSupport:
    def __init__(
        self, min_val: int = -300, max_val: int = 300, eps: float = 0.001
    ) -> None:
        self.eps = eps

        self.min_val = min_val
        self.max_val = max_val
        self.size = max_val - min_val + 1

        self._bucket_values = jnp.arange(min_val, max_val + 1).astype(jnp.float32)

    def _scale(self, x: jax.Array) -> jax.Array:
        return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1 + self.eps * x)

    def _unscale(self, x: jax.Array) -> jax.Array:
        return jnp.sign(x) * (
            (
                (jnp.sqrt(1 + 4 * self.eps * (jnp.abs(x) + 1 + self.eps)) - 1)
                / (2 * self.eps)
            )
            ** 2
            - 1
        )

    def encode(self, x: jax.Array) -> jax.Array:
        x = self._scale(x)
        x = jnp.clip(x, self.min_val, self.max_val)
        x_low = jnp.floor(x)
        x_high = jnp.ceil(x)
        p_high = x - x_low
        p_low = 1.0 - p_high
        idx_low = (x_low - self.min_val).astype(jnp.int32)
        idx_high = (x_high - self.min_val).astype(jnp.int32)
        low_one_hot = jax.nn.one_hot(idx_low, self.size, dtype=x.dtype)
        high_one_hot = jax.nn.one_hot(idx_high, self.size, dtype=x.dtype)

        support = p_low[..., None] * low_one_hot + p_high[..., None] * high_one_hot
        return support

    def decode(self, probs: jax.Array) -> jax.Array:
        x_trans = jnp.sum(probs * self._bucket_values, axis=-1)
        return self._unscale(x_trans)

    def decode_logits(self, logits: jax.Array) -> jax.Array:
        return self.decode(jax.nn.softmax(logits, axis=-1))
