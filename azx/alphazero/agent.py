import dataclasses
import functools
from typing import Callable, NamedTuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import mctx

from azx.core.env import EnvironmentAdapter
from azx.core.support import ScaledSupport


@dataclasses.dataclass
class Config:
    discount: float
    num_simulations: int
    use_mixed_value: bool
    value_scale: float
    support_min: int
    support_max: int
    support_eps: float


class AgentState(NamedTuple):
    params: hk.MutableParams
    state: hk.MutableState


class AlphaZero:
    def __init__(
        self,
        adapter: EnvironmentAdapter,
        config: Config,
        network_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    ):
        self.adapter = adapter
        self.config = config
        self.network = hk.transform_with_state(network_fn)
        self.support = ScaledSupport(
            min_val=config.support_min,
            max_val=config.support_max,
            eps=config.support_eps,
        )

    def init(self, key: chex.PRNGKey) -> AgentState:
        reset_key, net_key = jax.random.split(key)
        state, _ = self.adapter.env.reset(reset_key)
        obs = self.adapter.obs_fn(state)
        params, net_state = self.network.init(net_key, obs)
        return AgentState(params=params, state=net_state)

    def _recurrent_fn(
        self,
        model: AgentState,
        key: chex.PRNGKey,
        actions: jax.Array,
        env_states: jax.Array,
    ):
        env_states, steps = jax.vmap(self.adapter.env.step)(env_states, actions)
        obs = jax.vmap(self.adapter.obs_fn)(env_states)
        rewards = jax.vmap(lambda x: x.reward)(steps)
        terminals = jax.vmap(lambda x: x.last())(steps)

        (pi_logits, value_logits), _ = self.network.apply(
            model.params, model.state, key, obs
        )
        mask = jax.vmap(self.adapter.action_mask_fn)(env_states)
        pi_logits = jnp.where(mask, pi_logits, -jnp.inf)
        value = self.support.decode_logits(value_logits)

        return (
            mctx.RecurrentFnOutput(
                reward=rewards,  # type: ignore
                discount=jnp.where(terminals, 0.0, self.config.discount),  # type: ignore
                prior_logits=pi_logits,  # type: ignore
                value=value,  # type: ignore
            ),
            env_states,
        )

    def search(
        self,
        model: AgentState,
        key: chex.PRNGKey,
        env_states: chex.ArrayTree,
        gumbel_scale: float = 0.0,
    ) -> mctx.PolicyOutput:
        obs = jax.vmap(self.adapter.obs_fn)(env_states)
        key, apply_key, mcts_key = jax.random.split(key, 3)
        (pi_logits, value_logits), _ = self.network.apply(
            model.params, model.state, apply_key, obs
        )
        valid_actions = jax.vmap(self.adapter.action_mask_fn)(env_states)
        value = self.support.decode_logits(value_logits)

        root = mctx.RootFnOutput(
            prior_logits=pi_logits,  # type: ignore
            value=value,  # type: ignore
            embedding=env_states,  # type: ignore
        )

        invalid_actions = 1 - valid_actions

        return mctx.gumbel_muzero_policy(
            params=model,
            rng_key=mcts_key,
            root=root,
            invalid_actions=invalid_actions,
            recurrent_fn=self._recurrent_fn,
            num_simulations=self.config.num_simulations,
            max_num_considered_actions=self.adapter.env.action_spec.num_values,
            qtransform=functools.partial(
                mctx.qtransform_completed_by_mix_value,
                use_mixed_value=self.config.use_mixed_value,
                value_scale=self.config.value_scale,
            ),
            gumbel_scale=gumbel_scale,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self, model: AgentState, key: chex.PRNGKey, env_state: chex.ArrayTree
    ) -> jax.Array:
        env_states = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), env_state)

        policy_output = self.search(model=model, key=key, env_states=env_states)

        return policy_output.action[0]
