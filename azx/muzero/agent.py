import dataclasses
import functools
from typing import Callable, NamedTuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import mctx


@dataclasses.dataclass
class Config:
    discount: float
    num_simulations: int
    use_mixed_value: bool
    value_scale: float
    gumbel_scale: float


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
        representation_fn: Callable[[chex.Array], chex.Array],
        dynamics_fn: Callable[[chex.Array, chex.Array], chex.Array],
        prediction_fn: Callable[[chex.Array], tuple[chex.Array, chex.Array]],
    ):
        self.config = config
        self.rep_net = hk.transform_with_state(representation_fn)
        self.dyn_net = hk.transform_with_state(dynamics_fn)
        self.pred_net = hk.transform_with_state(prediction_fn)

    def _recurrent_fn(
        self,
        model: ModelState,
        key: chex.PRNGKey,
        actions: chex.Array,
        latent_states: chex.Array,
    ):
        key, subkey = jax.random.split(key)
        (next_latent, reward, terminal), _ = self.dyn_net.apply(
            model.params.dyn, model.state.dyn, subkey, latent_states, actions
        )
        (pi_logits, value), _ = self.pred_net.apply(
            model.params.pred, model.state.pred, key, next_latent
        )

        return (
            mctx.RecurrentFnOutput(
                reward=reward,  # type: ignore
                discount=jnp.where(terminal, 0.0, self.config.discount),  # type: ignore
                prior_logits=pi_logits,  # type: ignore
                value=value,  # type: ignore
            ),
            next_latent,
        )

    def _policy_output(
        self,
        model: ModelState,
        key: chex.PRNGKey,
        latent: chex.Array,
        pi_logits: chex.Array,
        value: chex.Array,
    ) -> mctx.PolicyOutput:
        root = mctx.RootFnOutput(
            prior_logits=pi_logits,  # type: ignore
            value=value,  # type: ignore
            embedding=latent,  # type: ignore
        )

        return mctx.gumbel_muzero_policy(
            params=model,
            rng_key=key,
            root=root,
            recurrent_fn=self._recurrent_fn,
            num_simulations=self.config.num_simulations,
            qtransform=functools.partial(
                mctx.qtransform_completed_by_mix_value,
                use_mixed_value=self.config.use_mixed_value,
                value_scale=self.config.value_scale,
            ),
            gumbel_scale=self.config.gumbel_scale,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self, model: ModelState, key: chex.PRNGKey, obs: chex.ArrayTree
    ) -> chex.Array:
        key, subkey = jax.random.split(key)
        latent, _ = self.rep_net.apply(
            model.params.rep,
            model.state.rep,
            subkey,
            jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), obs),
        )
        key, subkey = jax.random.split(key)
        (pi_logits, value), _ = self.pred_net.apply(
            model.params.pred, model.state.pred, subkey, latent
        )

        policy_output = self._policy_output(
            model=model,
            key=key,
            latent=latent,
            pi_logits=pi_logits,
            value=value,
        )

        return policy_output.action[0]  # type: ignore
