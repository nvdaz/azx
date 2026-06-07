from typing import Callable, NamedTuple

import chex
import jax
import jax.numpy as jnp
from mctx import PolicyOutput

from azx.core.env import EnvironmentAdapter, TimeStep, step_envs

SearchFn = Callable[[chex.PRNGKey, chex.ArrayTree], PolicyOutput]


class EvalStats(NamedTuple):
    episode_return: jax.Array = jnp.array(jnp.nan)
    episode_steps: jax.Array = jnp.array(jnp.nan)


class RolloutStep(NamedTuple):
    env_states: chex.ArrayTree
    experience: TimeStep
    reward: jax.Array
    terminal: jax.Array
    root_value: jax.Array


def collect_rollout_step(
    adapter: EnvironmentAdapter,
    env_states: chex.ArrayTree,
    search_fn: SearchFn,
    key: chex.PRNGKey,
) -> RolloutStep:
    search_key, step_key = jax.random.split(key)
    obs = jax.vmap(adapter.obs_fn)(env_states)
    action_mask = jax.vmap(adapter.action_mask_fn)(env_states)

    policy_output = search_fn(search_key, env_states)
    root_value = policy_output.search_tree.node_values[
        :, policy_output.search_tree.ROOT_INDEX
    ]

    env_states, reward, terminal = step_envs(
        adapter, env_states, policy_output, step_key
    )

    experience = TimeStep(
        obs=obs[:, None, ...],
        reward=reward[:, None, ...],
        terminal=terminal[:, None, ...],
        policy=policy_output.action_weights[:, None, ...],
        action_mask=action_mask.astype(jnp.bool_)[:, None, ...],
        action=policy_output.action[:, None, ...],
        value=root_value[:, None, ...],
    )

    return RolloutStep(
        env_states=env_states,
        experience=experience,
        reward=reward,
        terminal=terminal,
        root_value=root_value,
    )


def evaluate_rollout(
    adapter: EnvironmentAdapter,
    actor_batch_size: int,
    max_steps: int,
    key: chex.PRNGKey,
    search_fn: SearchFn,
) -> EvalStats:
    def loop_fn(carry):
        env_states, reward_acc, steps_acc, done_mask, key, iter = carry
        key, subkey = jax.random.split(key)

        policy = search_fn(subkey, env_states)
        next_states, steps = jax.vmap(adapter.env.step)(env_states, policy.action)
        r = jax.vmap(lambda ts: ts.reward)(steps)
        done = jax.vmap(lambda ts: ts.last())(steps)

        reward_acc = jnp.where(done_mask, reward_acc, reward_acc + r)
        steps_acc = jnp.where(done_mask, steps_acc, steps_acc + 1)
        done_mask = jnp.logical_or(done_mask, done)

        return next_states, reward_acc, steps_acc, done_mask, key, iter + 1

    key, subkey = jax.random.split(key)
    reset_keys = jax.random.split(subkey, actor_batch_size)
    env_states, _ = jax.vmap(adapter.env.reset)(reset_keys)

    reward_acc = jnp.zeros(actor_batch_size)
    steps_acc = jnp.zeros(actor_batch_size, dtype=jnp.int32)
    done_mask = jnp.zeros(actor_batch_size, dtype=jnp.bool_)

    _, reward_acc, steps_acc, _, _, _ = jax.lax.while_loop(
        lambda carry: jnp.any(~carry[3]) & (carry[5] < max_steps),
        loop_fn,
        (env_states, reward_acc, steps_acc, done_mask, key, 0),
    )

    return EvalStats(
        episode_return=jnp.mean(reward_acc), episode_steps=jnp.mean(steps_acc)
    )
