import dataclasses
import functools
from typing import Any, Callable

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import optax


@dataclasses.dataclass
class Config:
    discount: float
    num_simulations: int
    use_mixed_value: bool
    value_scale: float
    gumbel_scale: float


@chex.dataclass
class Params:
    representation: optax.Params
    dynamics: optax.Params
    prediction: optax.Params


class MuZero:
    def __init__(
        self,
        config: Config,
        representation_fn: Callable[[jax.Array], hk.Module],
        dynamics_fn: Callable[[jax.Array], hk.Module],
        prediction_fn: Callable[[jax.Array], hk.Module],
    ):
        self.config = config
        self.rep_net = hk.without_apply_rng(hk.transform(representation_fn))
        self.dyn_net = hk.without_apply_rng(hk.transform(dynamics_fn))
        self.pred_net = hk.without_apply_rng(hk.transform(prediction_fn))

    def _recurrent_fn(
        self, params: Params, _: chex.PRNGKey, actions: chex.Array, latents: chex.Array
    ):
        next_latent, rewards, terminals = jax.vmap(
            self.dyn_net.apply, in_axes=(None, 0, 0)
        )(params.dynamics, latents, jnp.expand_dims(actions, axis=-1))
        rewards = -jnp.ones_like(rewards)

        pi_logits, value = jax.vmap(self.pred_net.apply, in_axes=(None, 0))(
            params.prediction, next_latent
        )

        return (
            mctx.RecurrentFnOutput(
                reward=rewards,  # type: ignore
                discount=jnp.where(terminals, 0.0, self.config.discount),  # type: ignore
                prior_logits=pi_logits,  # type: ignore
                value=value,  # type: ignore
            ),
            next_latent,
        )

    def _policy_output(
        self,
        params: Params,
        key: chex.PRNGKey,
        latent: Any,
        pi_logits: chex.Array,
        value: chex.Array,
    ) -> mctx.PolicyOutput:
        root = mctx.RootFnOutput(
            prior_logits=pi_logits,  # type: ignore
            value=value,  # type: ignore
            embedding=latent,  # type: ignore
        )

        return mctx.gumbel_muzero_policy(
            params=params,
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
    def predict(self, params: Params, key: chex.PRNGKey, obs: Any) -> int:
        latent = self.rep_net.apply(params.representation, obs)
        pi_logits, value = self.pred_net.apply(params.prediction, latent)

        policy_output = self._policy_output(
            params=params,
            key=key,
            latent=jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), latent),
            pi_logits=jnp.expand_dims(pi_logits, axis=0),
            value=jnp.expand_dims(value, axis=0),
        )

        return policy_output.action[0]  # type: ignore
