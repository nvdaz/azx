import dataclasses
import functools
from typing import Any, Callable

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import optax
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
        self.network = hk.without_apply_rng(hk.transform(network_fn))
        self.obs_fn = obs_fn

    def _recurrent_fn(
        self, params: optax.Params, _: chex.PRNGKey, actions: chex.Array, env_states
    ):
        batch_step = jax.vmap(self.env.step, in_axes=(0, 0))
        batch_obs = jax.vmap(self.obs_fn, in_axes=(0,))
        batch_apply = jax.vmap(self.network.apply, in_axes=(None, 0))
        batch_rewards = jax.vmap(lambda x: x.reward, in_axes=(0,))
        batch_terminals = jax.vmap(lambda x: x.last(), in_axes=(0,))

        env_states, steps = batch_step(env_states, actions)
        obs = batch_obs(env_states)
        rewards = batch_rewards(steps)
        terminals = batch_terminals(steps)

        pi_logits, value = batch_apply(params, obs)

        return (
            mctx.RecurrentFnOutput(
                reward=rewards,  # type: ignore
                discount=jnp.where(terminals, 0.0, self.config.discount),  # type: ignore
                prior_logits=pi_logits,  # type: ignore
                value=value,  # type: ignore
            ),
            env_states,
        )

    def _policy_output(
        self,
        params: optax.Params,
        key: chex.PRNGKey,
        env_states: Any,
        pi_logits: chex.Array,
        value: chex.Array,
    ) -> mctx.PolicyOutput:
        root = mctx.RootFnOutput(
            prior_logits=pi_logits,  # type: ignore
            value=value,  # type: ignore
            embedding=env_states,  # type: ignore
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
    def predict(self, params: optax.Params, key: chex.PRNGKey, env_state: Any) -> int:
        pi_logits, value = self.network.apply(params, self.obs_fn(env_state))

        policy_output = self._policy_output(
            params=params,
            key=key,
            env_states=jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), env_state),
            pi_logits=jnp.expand_dims(pi_logits, axis=0),
            value=jnp.expand_dims(value, axis=0),
        )

        return policy_output.action[0]  # type: ignore
