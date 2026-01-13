from typing import Callable, Generic, NamedTuple, TypeVar

import chex
import jax
import jax.numpy as jnp
from flashbax.buffers.trajectory_buffer import TrajectoryBuffer, TrajectoryBufferState
from jumanji.env import Environment
from mctx import PolicyOutput

FIXED_KEY = jax.random.PRNGKey(0)


TState = TypeVar("TState")


class EnvironmentAdapter(NamedTuple, Generic[TState]):
    env: Environment
    obs_fn: Callable[[TState], jax.Array]
    action_mask_fn: Callable[[TState], jax.Array]


class TimeStep(NamedTuple):
    obs: jax.Array
    reward: jax.Array
    terminal: jax.Array
    policy: jax.Array
    action_mask: jax.Array
    action: jax.Array
    value: jax.Array


def init_buffer(
    adapter: EnvironmentAdapter,
    buffer: TrajectoryBuffer,
) -> TrajectoryBufferState:
    state, _ = adapter.env.reset(FIXED_KEY)
    obs = adapter.obs_fn(state)

    experience = TimeStep(
        obs=obs,
        reward=jnp.zeros((), dtype=jnp.float32),
        terminal=jnp.zeros((), dtype=jnp.bool_),
        policy=jnp.zeros((adapter.env.action_spec.num_values,), dtype=jnp.float32),
        action_mask=jnp.zeros((adapter.env.action_spec.num_values,), dtype=jnp.bool_),
        action=jnp.zeros((), dtype=jnp.int32),
        value=jnp.zeros((), dtype=jnp.float32),
    )

    return buffer.init(experience)


def reset_envs(
    adapter: EnvironmentAdapter,
    batch_size: int,
    key: chex.PRNGKey,
) -> chex.ArrayTree:
    keys = jax.random.split(key, batch_size)
    env_states, _ = jax.vmap(adapter.env.reset)(keys)
    return env_states


def step_envs(
    adapter: EnvironmentAdapter,
    env_states: chex.ArrayTree,
    policy_output: PolicyOutput,
    key: chex.PRNGKey,
) -> tuple[chex.ArrayTree, jax.Array, jax.Array]:
    env_states, steps = jax.vmap(adapter.env.step)(env_states, policy_output.action)
    reward = jax.vmap(lambda step: step.reward)(steps)
    terminal = jax.vmap(lambda step: step.last())(steps)

    subkeys = jax.random.split(key, policy_output.action.shape[0])
    reset_states, _ = jax.vmap(adapter.env.reset)(subkeys)

    env_states = jax.vmap(
        lambda next_state, reset_state, done: jax.lax.cond(
            done, lambda _: reset_state, lambda _: next_state, operand=None
        )
    )(env_states, reset_states, terminal)

    return env_states, reward, terminal
