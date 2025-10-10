import dataclasses
import functools
from typing import Any, Callable

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import mctx
from jumanji.env import Environment


@dataclasses.dataclass
class Config:
    discount: float
    num_simulations: int
    use_mixed_value: bool
    value_scale: float
    gumbel_scale: float


class AlphaZero:
    def __init__(
        self,
        env: Environment,
        config: Config,
        network_fn: Callable[[jax.Array], hk.Module],
        obs_fn: Callable[[Any], chex.Array],
    ):
        self.env = env
        self.config = config
        self.network = hk.transform_with_state(network_fn)
        self.obs_fn = obs_fn

    def _recurrent_fn(
        self,
        params:hk.MutableParams,
        key: chex.PRNGKey,
        actions: chex.Array,
        embedding: tuple,
    ):
        net_state, env_states = embedding
        batch_step = jax.vmap(self.env.step, in_axes=(0, 0))
        batch_obs = jax.vmap(self.obs_fn, in_axes=(0,))
        batch_rewards = jax.vmap(lambda x: x.reward, in_axes=(0,))
        batch_terminals = jax.vmap(lambda x: x.last(), in_axes=(0,))

        env_states, steps = batch_step(env_states, actions)
        obs = batch_obs(env_states)
        rewards = batch_rewards(steps)
        terminals = batch_terminals(steps)

        (pi_logits, value), _ = self.network.apply(params, net_state, key, obs)

        return (
            mctx.RecurrentFnOutput(
                reward=rewards,  # type: ignore
                discount=jnp.where(terminals, 0.0, self.config.discount),  # type: ignore
                prior_logits=pi_logits,  # type: ignore
                value=value,  # type: ignore
            ),
            (net_state, env_states),
        )

    def _policy_output(
        self,
        params: hk.MutableParams,
        net_state: hk.MutableState,
        key: chex.PRNGKey,
        env_states: Any,
        pi_logits: chex.Array,
        value: chex.Array,
    ) -> mctx.PolicyOutput:
        root = mctx.RootFnOutput(
            prior_logits=pi_logits,  # type: ignore
            value=value,  # type: ignore
            embedding=(net_state, env_states),  # type: ignore
        )

        return mctx.gumbel_muzero_policy(
            params=params,
            rng_key=key,
            root=root,
            recurrent_fn=self._recurrent_fn,
            num_simulations=self.config.num_simulations,
            max_num_considered_actions=self.env.action_spec.num_values,
            qtransform=functools.partial(
                mctx.qtransform_completed_by_mix_value,
                use_mixed_value=self.config.use_mixed_value,
                value_scale=self.config.value_scale,
            ),
            gumbel_scale=self.config.gumbel_scale,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        params: hk.MutableParams,
        net_state: hk.MutableState,
        key: chex.PRNGKey,
        env_state: Any,
    ) -> int:
        key, subkey = jax.random.split(key)
        (pi_logits, value), _ = self.network.apply(params, net_state, subkey, self.obs_fn(env_state)[None, ...])

        policy_output = self._policy_output(
            params=params,
            net_state=net_state,
            key=key,
            env_states=jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), env_state),
            pi_logits=pi_logits,
            value=value,
        )

        return policy_output.action[0]  # type: ignore
