import functools

import jax
import jax.numpy as jnp
import rlax

from azx.core.support import DiscreteSupport


def trajectory_alive_mask(
    terminals: jax.Array,
    unroll_steps: int,
) -> tuple[jax.Array, jax.Array]:
    terminals_all = terminals[:, :unroll_steps]
    done_cumulative = jnp.cumsum(terminals_all.astype(jnp.int32), axis=1) > 0
    done_before_all = jnp.concatenate(
        [
            jnp.zeros_like(done_cumulative[:, :1], dtype=jnp.bool_),
            done_cumulative[:, :-1],
        ],
        axis=1,
    )
    alive_mask = (~done_before_all).astype(jnp.float32)
    alive_sum = alive_mask.sum() + 1e-9
    return alive_mask, alive_sum


def nstep_value_targets(
    rewards: jax.Array,
    discounts: jax.Array,
    target_values: jax.Array,
    n_step: int,
) -> jax.Array:
    nstep_returns = functools.partial(
        rlax.n_step_bootstrapped_returns,
        n=n_step,
        stop_target_gradients=True,
    )
    return jax.vmap(nstep_returns)(rewards, discounts, target_values)


def policy_cross_entropy(
    pred_policy_logits: jax.Array,
    target_pi: jax.Array,
    action_mask: jax.Array,
    alive_mask: jax.Array,
    alive_sum: jax.Array,
    per_step_scale: jax.Array | None = None,
) -> jax.Array:
    masked_logits = jnp.where(action_mask, pred_policy_logits, jnp.array(-1e9))
    masked_target = target_pi * action_mask
    norm = masked_target.sum(-1, keepdims=True) + 1e-9
    target_norm = masked_target / norm

    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
    cross_entropy = -(target_norm * log_probs).sum(axis=-1)
    if per_step_scale is None:
        return (cross_entropy * alive_mask).sum() / alive_sum
    scaled = (cross_entropy * alive_mask) / (per_step_scale + 1e-9)
    return scaled.sum(axis=1).mean()


def support_cross_entropy(
    pred_logits: jax.Array,
    target_values: jax.Array,
    support: DiscreteSupport,
    alive_mask: jax.Array,
    alive_sum: jax.Array,
    per_step_scale: jax.Array | None = None,
) -> jax.Array:
    target_prob = support.encode(target_values)
    log_probs = jax.nn.log_softmax(pred_logits, axis=-1)
    cross_entropy = -(target_prob * log_probs).sum(axis=-1)
    if per_step_scale is None:
        return (cross_entropy * alive_mask).sum() / alive_sum
    scaled = (cross_entropy * alive_mask) / (per_step_scale + 1e-9)
    return scaled.sum(axis=1).mean()
