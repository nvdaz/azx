import dataclasses
import functools
from typing import Callable, NamedTuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import mctx

from azx.internal.support import DiscreteSupport


@dataclasses.dataclass
class Config:
    discount: float
    num_simulations: int
    use_mixed_value: bool
    value_scale: float
    support_min: int
    support_max: int
    support_eps: float


class ModelParams(NamedTuple):
    rep: hk.MutableParams
    dyn: hk.MutableParams
    pred: hk.MutableParams


class ModelNetState(NamedTuple):
    rep: hk.MutableState
    dyn: hk.MutableState
    pred: hk.MutableState


class ModelState(NamedTuple):
    params: ModelParams
    state: ModelNetState


class MuZero:
    def __init__(
        self,
        config: Config,
        representation_fn: Callable[[jax.Array], jax.Array],
        dynamics_fn: Callable[[jax.Array, jax.Array], jax.Array],
        prediction_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    ):
        self.config = config
        self.rep_net = hk.transform_with_state(representation_fn)
        self.dyn_net = hk.transform_with_state(dynamics_fn)
        self.pred_net = hk.transform_with_state(prediction_fn)
        self.support = DiscreteSupport(
            min_val=config.support_min,
            max_val=config.support_max,
            eps=config.support_eps,
        )

    def _recurrent_fn(
        self,
        model: ModelState,
        key: chex.PRNGKey,
        actions: jax.Array,
        latent_states: jax.Array,
    ):
        key, subkey = jax.random.split(key)
        (next_latent, reward_logits), _ = self.dyn_net.apply(
            model.params.dyn, model.state.dyn, subkey, latent_states, actions
        )
        reward = self.support.decode_logits(reward_logits)
        (pi_logits, value_logits), _ = self.pred_net.apply(
            model.params.pred, model.state.pred, key, next_latent
        )
        value = self.support.decode_logits(value_logits)

        return (
            mctx.RecurrentFnOutput(
                reward=reward,  # type: ignore
                discount=jnp.full_like(reward, self.config.discount),  # type: ignore
                prior_logits=pi_logits,  # type: ignore
                value=value,  # type: ignore
            ),
            next_latent,
        )

    def _muzero_search(
        self,
        model: ModelState,
        key: chex.PRNGKey,
        obs: jax.Array,
        valid_actions: jax.Array,
        gumbel_scale: float = 0.0,
    ) -> mctx.PolicyOutput:
        key, rep_key, pred_key, mcts_key = jax.random.split(key, 4)
        latent, _ = self.rep_net.apply(model.params.rep, model.state.rep, rep_key, obs)

        (pi_logits, value_logits), _ = self.pred_net.apply(
            model.params.pred, model.state.pred, pred_key, latent
        )
        value = self.support.decode_logits(value_logits)

        root = mctx.RootFnOutput(
            prior_logits=pi_logits,  # type: ignore
            value=value,  # type: ignore
            embedding=latent,  # type: ignore
        )
        invalid_actions = 1 - valid_actions

        return mctx.gumbel_muzero_policy(
            params=model,
            rng_key=mcts_key,
            root=root,
            invalid_actions=invalid_actions,
            recurrent_fn=self._recurrent_fn,
            num_simulations=self.config.num_simulations,
            qtransform=functools.partial(
                mctx.qtransform_completed_by_mix_value,
                use_mixed_value=self.config.use_mixed_value,
                value_scale=self.config.value_scale,
            ),
            gumbel_scale=gumbel_scale,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        model: ModelState,
        key: chex.PRNGKey,
        obs: jax.Array,
        valid_actions: jax.Array,
    ) -> jax.Array:
        policy_output = self._muzero_search(
            model=model,
            key=key,
            obs=obs[None, ...],
            valid_actions=valid_actions,
        )

        return policy_output.action[0]
