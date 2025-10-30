import dataclasses
import functools
from typing import Callable, NamedTuple

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


class ModelState(NamedTuple):
    params: hk.MutableParams
    state: hk.MutableState


class AlphaZero:
    def __init__(
        self,
        env: Environment,
        config: Config,
        network_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
        obs_fn: Callable[[chex.ArrayTree], jax.Array],
        action_mask_fn: Callable[[chex.ArrayTree], jax.Array],
    ):
        self.env = env
        self.config = config
        self.network = hk.transform_with_state(network_fn)
        self.obs_fn = obs_fn
        self.action_mask_fn = action_mask_fn

    def _recurrent_fn(
        self,
        model: ModelState,
        key: chex.PRNGKey,
        actions: jax.Array,
        env_states: jax.Array,
    ):
        batch_step = jax.vmap(self.env.step, in_axes=(0, 0))
        batch_obs = jax.vmap(self.obs_fn, in_axes=(0,))
        batch_rewards = jax.vmap(lambda x: x.reward, in_axes=(0,))
        batch_terminals = jax.vmap(lambda x: x.last(), in_axes=(0,))

        env_states, steps = batch_step(env_states, actions)
        obs = batch_obs(env_states)
        rewards = batch_rewards(steps)
        terminals = batch_terminals(steps)

        (pi_logits, value), _ = self.network.apply(model.params, model.state, key, obs)
        mask = jax.vmap(self.action_mask_fn)(env_states)
        pi_logits = jnp.where(mask, pi_logits, -jnp.inf)

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
        model: ModelState,
        key: chex.PRNGKey,
        env_states: chex.ArrayTree,
        pi_logits: jax.Array,
        value: jax.Array,
        eval: bool,
    ) -> mctx.PolicyOutput:
        root = mctx.RootFnOutput(
            prior_logits=pi_logits,  # type: ignore
            value=value,  # type: ignore
            embedding=env_states,  # type: ignore
        )

        return mctx.gumbel_muzero_policy(
            params=model,
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
            gumbel_scale=jnp.where(eval, 0.0, self.config.gumbel_scale),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        model: ModelState,
        key: chex.PRNGKey,
        env_state: chex.ArrayTree,
        eval: bool = False,
    ) -> jax.Array:
        key, subkey = jax.random.split(key)
        (pi_logits, value), _ = self.network.apply(
            model.params, model.state, subkey, self.obs_fn(env_state)[None, ...]
        )  # ignore new state for inference
        mask = self.action_mask_fn(env_state)
        pi_logits = jnp.where(mask, pi_logits, -jnp.inf)

        policy_output = self._policy_output(
            model=model,
            key=key,
            env_states=jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), env_state),
            pi_logits=pi_logits,
            value=value,
            eval=eval,
        )

        return policy_output.action[0]  # type: ignore
