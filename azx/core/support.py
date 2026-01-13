import jax
import jax.numpy as jnp


class DiscreteSupport:
    def __init__(self, min_val: int, max_val: int) -> None:
        self.min_val = min_val
        self.max_val = max_val
        self.size = max_val - min_val + 1
        self.bucket_values = jnp.arange(min_val, max_val + 1).astype(jnp.float32)

    def encode(self, x: jax.Array) -> jax.Array:
        x = jnp.clip(x, self.min_val, self.max_val)
        x_shift = x - self.min_val
        idx_low = jnp.floor(x_shift).astype(jnp.int32)
        idx_high = jnp.minimum(idx_low + 1, self.size - 1)

        p_high = x_shift - idx_low
        p_low = 1.0 - p_high

        low_oh = jax.nn.one_hot(idx_low, self.size, dtype=x.dtype)
        high_oh = jax.nn.one_hot(idx_high, self.size, dtype=x.dtype)

        support = p_low[..., None] * low_oh + p_high[..., None] * high_oh
        return support

    def decode(self, probs: jax.Array) -> jax.Array:
        return jnp.sum(probs * self.bucket_values, axis=-1)

    def decode_logits(self, logits: jax.Array) -> jax.Array:
        return self.decode(jax.nn.softmax(logits, axis=-1))


class ScaledSupport(DiscreteSupport):
    def __init__(self, min_val: int, max_val: int, eps: float) -> None:
        super().__init__(min_val, max_val)
        self.eps = eps

    def encode(self, x: jax.Array) -> jax.Array:
        x = jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + self.eps * x
        return super().encode(x)

    def decode(self, probs: jax.Array) -> jax.Array:
        x = super().decode(probs)
        return jnp.sign(x) * (
            (
                (jnp.sqrt(1 + 4 * self.eps * (jnp.abs(x) + 1 + self.eps)) - 1)
                / (2 * self.eps)
            )
            ** 2
            - 1
        )
