import dataclasses
import functools
from pathlib import Path
from typing import Any, Callable, Literal

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import optax
import orbax.checkpoint as ocp
from jumanji.env import Environment

from .agent import AlphaZero, Config


@dataclasses.dataclass
class TrainConfig(Config):
    batch_size: int
    eval_frequency: int
    avg_return_smoothing: float
    value_target: Literal["maxq", "nodev"]
    dirichlet_alpha: float
    dirichlet_mix: float
    checkpoint_frequency: int


@chex.dataclass
class TrainState:
    env_states: chex.Array
    params: optax.Params
    opt_state: optax.OptState
    avg_return: chex.Array
    episode_return: chex.Array
    num_episodes: chex.Array
    key: chex.Array
    eval_episode_return: chex.ArrayTree
    eval_avg_return: chex.ArrayTree


class AlphaZeroTrainer(AlphaZero):
    def __init__(
        self,
        env: Environment,
        config: TrainConfig,
        network_fn: Callable[[jax.Array], hk.Module],
        obs_fn: Callable[[Any], chex.Array],
        opt: optax.GradientTransformation,
    ):
        super().__init__(env, config, network_fn, obs_fn)
        self.opt = opt
        self.config = config

    def init(self, key: chex.PRNGKey) -> TrainState:
        key, subkey = jax.random.split(key)
        state, _ = self.env.reset(subkey)

        key, subkey = jax.random.split(key)
        params = self.network.init(subkey, self.obs_fn(state))

        opt_state = self.opt.init(params)

        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, self.config.batch_size)
        env_states, _ = jax.vmap(self.env.reset)(subkeys)

        return TrainState(
            env_states=env_states,
            params=params,
            opt_state=opt_state,
            avg_return=jnp.zeros(self.config.batch_size),
            episode_return=jnp.zeros(self.config.batch_size),
            num_episodes=jnp.zeros(self.config.batch_size),
            eval_avg_return=jnp.zeros(self.config.batch_size),
            eval_episode_return=jnp.zeros(self.config.batch_size),
            key=key,
        )

    def _loss_fn(
        self,
        params: optax.Params,
        pi_target: chex.Array,
        value_target: chex.Array,
        obs: chex.Array,
    ) -> chex.Array:
        pi_logits, value = self.network.apply(params, obs)
        pi_loss = optax.softmax_cross_entropy(pi_logits, pi_target).mean()
        value_loss = optax.l2_loss(value_target, value).mean()

        return pi_loss + value_loss

    def _step_env(
        self, state: TrainState, actions: chex.Array
    ) -> tuple[TrainState, chex.Array, chex.Array]:
        batch_step = jax.vmap(self.env.step, in_axes=(0, 0))
        batch_reset = jax.vmap(self.env.reset, in_axes=(0,))
        batch_reward = jax.vmap(lambda x: x.reward, in_axes=(0,))
        batch_terminal = jax.vmap(lambda x: x.last(), in_axes=(0,))

        env_states, steps = batch_step(state.env_states, actions)
        reward = batch_reward(steps)
        terminal = batch_terminal(steps)

        key, subkey = jax.random.split(state.key)
        subkeys = jax.random.split(subkey, self.config.batch_size)
        reset_states, _ = batch_reset(subkeys)

        env_states = jax.vmap(
            lambda t, reset_state, env_state: jax.lax.cond(
                t, lambda: reset_state, lambda: env_state
            )
        )(terminal, reset_states, env_states)

        state = state.replace(key=key, env_states=env_states)  # type: ignore

        return state, reward, terminal

    def _alphazero_search(
        self, state: TrainState
    ) -> tuple[TrainState, mctx.PolicyOutput]:
        batch_obs = jax.vmap(self.obs_fn, in_axes=(0,))
        batch_apply = jax.vmap(self.network.apply, in_axes=(None, 0))

        env_obs = batch_obs(state.env_states)
        pi_logits, value = batch_apply(state.params, env_obs)

        key, subkey = jax.random.split(state.key)
        noise_key, mcts_key = jax.random.split(subkey)

        prior_probs = jax.nn.softmax(pi_logits)

        dirichlet_noise = jax.random.dirichlet(
            noise_key,
            jnp.full(self.env.action_spec.num_values, self.config.dirichlet_alpha),
            shape=(self.config.batch_size,),
        )

        action_mask = state.env_states.action_mask  # type: ignore
        noisy_priors = (
            prior_probs * (1 - self.config.dirichlet_mix)
            + dirichlet_noise * self.config.dirichlet_mix
        )
        noisy_logits = jnp.where(
            (noisy_priors > 0) & action_mask, jnp.log(noisy_priors), -1e10
        )

        policy_output = self._policy_output(
            params=state.params,
            key=mcts_key,
            env_states=state.env_states,
            pi_logits=noisy_logits,
            value=value,
        )

        return state.replace(key=key), policy_output  # type: ignore

    def _compute_gradients(
        self,
        params: optax.Params,
        search_policy: chex.Array,
        search_value: chex.Array,
        obs: chex.Array,
    ) -> tuple[chex.Array, chex.Array]:
        batch_loss = jax.vmap(self._loss_fn, in_axes=(None, 0, 0, 0))
        loss_grad = jax.value_and_grad(
            lambda *args: jnp.mean(batch_loss(*args)), argnums=0
        )

        return loss_grad(
            params,
            search_policy,
            search_value,
            obs,
        )

    def _apply_updates(self, state: TrainState, ac_grads: chex.Array) -> TrainState:
        updates, state.opt_state = self.opt.update(
            ac_grads, state.opt_state, state.params
        )
        params = optax.apply_updates(state.params, updates)
        return state.replace(params=params)  # type: ignore

    @functools.partial(jax.jit, static_argnums=(0,))
    def train_step(self, state: TrainState) -> TrainState:
        batch_obs = jax.vmap(self.obs_fn, in_axes=(0,))

        def loop_fn(state: TrainState, _):
            state, policy_output = self._alphazero_search(state)
            search_policy = policy_output.action_weights

            if self.config.value_target == "maxq":
                qvalues = policy_output.search_tree.qvalues(
                    jnp.full(
                        self.config.batch_size, policy_output.search_tree.ROOT_INDEX
                    )
                )

                search_value = qvalues[
                    jnp.arange(self.config.batch_size), policy_output.action
                ]
            elif self.config.value_target == "nodev":
                search_value = policy_output.search_tree.node_values[
                    :, policy_output.search_tree.ROOT_INDEX
                ]
            else:
                raise AssertionError("Unknown value target")

            env_obs = batch_obs(state.env_states)
            loss, grads = self._compute_gradients(
                state.params, search_policy, search_value, env_obs
            )
            state = self._apply_updates(state, grads)
            state, reward, terminals = self._step_env(state, policy_output.action)

            new_return = state.episode_return + reward

            episode_return = jnp.where(terminals, 0, new_return)
            avg_return = jnp.where(
                terminals,
                state.avg_return * self.config.avg_return_smoothing
                + new_return * (1 - self.config.avg_return_smoothing),
                state.avg_return,
            )
            num_episodes = state.num_episodes + terminals.astype(jnp.int32)

            state = state.replace(  # type: ignore
                episode_return=episode_return,
                avg_return=avg_return,
                num_episodes=num_episodes,
            )

            return state, None

        state, _ = jax.lax.scan(loop_fn, state, None, length=self.config.eval_frequency)
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def evaluate(self, state: TrainState) -> chex.Array:
        batch_reset = jax.vmap(self.env.reset, in_axes=(0,))
        batch_step = jax.vmap(self.env.step, in_axes=(0, 0))

        def single_action(env_state, key):
            return self.predict(state.params, key, env_state)

        batch_predict = jax.vmap(single_action, in_axes=(0, 0))

        def loop_fn(carry):
            env_states, reward_acc, done_mask, key = carry
            key, subkey = jax.random.split(key)
            step_keys = jax.random.split(subkey, self.config.batch_size)

            actions = batch_predict(env_states, step_keys)

            # step the envs
            next_states, steps = batch_step(env_states, actions)
            r = jax.vmap(lambda ts: ts.reward, in_axes=(0,))(steps)
            done = jax.vmap(lambda ts: ts.last(), in_axes=(0,))(steps)

            # accumulate only for unfinished envs
            reward_acc = jnp.where(done_mask, reward_acc, reward_acc + r)
            done_mask = jnp.logical_or(done_mask, done)

            return next_states, reward_acc, done_mask, key

        key, subkey = jax.random.split(state.key)
        reset_keys = jax.random.split(subkey, self.config.batch_size)
        env_states, _ = batch_reset(reset_keys)

        reward_acc = jnp.zeros(self.config.batch_size)
        done_mask = jnp.zeros(self.config.batch_size, dtype=jnp.bool_)

        _, reward_acc, _, _ = jax.lax.while_loop(
            lambda carry: jnp.any(~carry[2]),  # while any not done
            loop_fn,
            (env_states, reward_acc, done_mask, key),
        )

        return jnp.mean(reward_acc)

    def learn(
        self, state: TrainState, num_steps: int, checkpoints_dir: str
    ) -> tuple[TrainState, list[float], list[int]]:
        path = ocp.test_utils.erase_and_create_empty(Path(checkpoints_dir).resolve())
        checkpointer = ocp.StandardCheckpointer()
        returns = []
        steps = []
        time_step = 0

        for _ in range(num_steps // self.config.eval_frequency):
            state = self.train_step(state)

            valid_returns = state.avg_return[state.num_episodes > 0]
            valid_episodes = state.num_episodes[state.num_episodes > 0]
            avg_return = jnp.mean(
                valid_returns / (1 - self.config.avg_return_smoothing**valid_episodes),
            )

            returns.append(avg_return)
            time_step += self.config.eval_frequency
            steps.append(time_step)
            ev = self.evaluate(state)

            print(
                f"Step {time_step} | Avg Return: {avg_return:.3f} | Eval: {ev:.3f}",
                flush=True,
            )

            if time_step % self.config.checkpoint_frequency == 0:
                checkpointer.save(path / f"checkpoint-{time_step}", state)

        checkpointer.save(path / "checkpoint-final", state)

        print("Saving final checkpoint...", flush=True)
        checkpointer.wait_until_finished()

        return state, returns, steps
