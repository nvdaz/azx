import dataclasses
import functools
from pathlib import Path
from typing import Callable, NamedTuple

import chex
import flashbax as fbx
import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import optax
import orbax.checkpoint as ocp
import rlax
from flashbax.buffers.prioritised_trajectory_buffer import (
    PrioritisedTrajectoryBufferState,
)
from jumanji.env import Environment

from .agent import AlphaZero, Config, ModelState


@dataclasses.dataclass
class TrainConfig(Config):
    batch_size: int
    eval_frequency: int
    n_step: int
    unroll_steps: int
    max_eval_steps: int
    avg_return_smoothing: float
    dirichlet_alpha: float
    dirichlet_mix: float
    checkpoint_frequency: int
    max_length_buffer: int
    min_length_buffer: int


class TrainState(NamedTuple):
    model: ModelState
    env_states: jax.Array
    buffer_state: PrioritisedTrajectoryBufferState
    opt_state: optax.OptState
    avg_return: jax.Array
    avg_loss: jax.Array
    avg_pi_loss: jax.Array
    avg_value_loss: jax.Array
    episode_return: jax.Array
    num_episodes: jax.Array
    key: jax.Array
    eval_episode_return: chex.ArrayTree
    eval_avg_return: chex.ArrayTree


class TimeStep(NamedTuple):
    obs: jax.Array
    reward: jax.Array
    terminal: jax.Array
    pi: jax.Array
    action_mask: jax.Array


class AlphaZeroTrainer(AlphaZero):
    def __init__(
        self,
        env: Environment,
        config: TrainConfig,
        network_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
        obs_fn: Callable[[chex.ArrayTree], jax.Array],
        action_mask_fn: Callable[[chex.ArrayTree], jax.Array],
        opt: optax.GradientTransformation,
    ):
        super().__init__(env, config, network_fn, obs_fn)
        self.opt = opt
        self.config = config
        self.action_mask_fn = action_mask_fn
        self.train_checkpointer = ocp.StandardCheckpointer()
        self.buffer = fbx.make_prioritised_trajectory_buffer(
            max_length_time_axis=config.max_length_buffer,
            min_length_time_axis=config.min_length_buffer,
            sample_batch_size=config.batch_size,
            add_batch_size=config.batch_size,
            sample_sequence_length=config.n_step + config.unroll_steps,
            period=1,
        )

    def init(self, key: chex.PRNGKey) -> TrainState:
        key, subkey = jax.random.split(key)
        state, _ = self.env.reset(subkey)
        obs = self.obs_fn(state)[None, ...]

        key, subkey = jax.random.split(key)
        params, net_state = self.network.init(subkey, obs)

        experience = TimeStep(
            obs=obs[0],
            reward=jnp.zeros((1,), dtype=jnp.float32),
            terminal=jnp.zeros((1,), dtype=jnp.bool_),
            pi=jnp.zeros((self.env.action_spec.num_values,), dtype=jnp.float32),
            action_mask=jnp.zeros((self.env.action_spec.num_values,), dtype=jnp.bool_),
        )
        buffer_state = self.buffer.init(experience)

        opt_state = self.opt.init(params)

        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, self.config.batch_size)
        env_states, _ = jax.vmap(self.env.reset)(subkeys)

        return TrainState(
            model=ModelState(params, net_state),
            env_states=env_states,
            buffer_state=buffer_state,
            opt_state=opt_state,
            avg_return=jnp.zeros(self.config.batch_size),
            avg_loss=jnp.zeros(self.config.batch_size),
            avg_pi_loss=jnp.zeros(self.config.batch_size),
            avg_value_loss=jnp.zeros(self.config.batch_size),
            episode_return=jnp.zeros(self.config.batch_size),
            num_episodes=jnp.zeros(self.config.batch_size),
            eval_avg_return=jnp.zeros(self.config.batch_size),
            eval_episode_return=jnp.zeros(self.config.batch_size),
            key=key,
        )

    def _loss_fn(
        self,
        params: hk.MutableParams,
        net_state: hk.MutableState,
        key: jax.Array,
        pi_target: jax.Array,
        value_target: jax.Array,
        obs: jax.Array,
    ) -> tuple[jax.Array, tuple[hk.MutableState, jax.Array, jax.Array]]:
        pi_target = jax.lax.stop_gradient(pi_target)
        value_target = jax.lax.stop_gradient(value_target)
        (pi_logits, value), net_state = self.network.apply(params, net_state, key, obs)
        pi_loss = optax.softmax_cross_entropy(pi_logits, pi_target).mean()
        value_loss = optax.l2_loss(value_target, value).mean()

        return pi_loss + value_loss, (
            net_state,
            jax.lax.stop_gradient(pi_loss),
            jax.lax.stop_gradient(value_loss),
        )

    def _step_env(
        self, key: chex.PRNGKey, env_states: chex.ArrayTree, actions: jax.Array
    ) -> tuple[chex.ArrayTree, jax.Array, jax.Array]:
        batch_step = jax.vmap(self.env.step, in_axes=(0, 0))
        batch_reset = jax.vmap(self.env.reset, in_axes=(0,))
        batch_reward = jax.vmap(lambda x: x.reward, in_axes=(0,))
        batch_terminal = jax.vmap(lambda x: x.last(), in_axes=(0,))

        env_states, steps = batch_step(env_states, actions)
        reward = batch_reward(steps)
        terminal = batch_terminal(steps)

        subkeys = jax.random.split(key, self.config.batch_size)
        reset_states, _ = batch_reset(subkeys)

        env_states = jax.vmap(
            lambda t, reset_state, env_state: jax.lax.cond(
                t,
                lambda x: x[0],
                lambda x: x[1],
                (reset_state, env_state),
            )
        )(terminal, reset_states, env_states)

        return env_states, reward, terminal

    def _alphazero_search(
        self, state: TrainState
    ) -> tuple[TrainState, mctx.PolicyOutput]:
        batch_obs = jax.vmap(self.obs_fn, in_axes=(0,))

        env_obs = batch_obs(state.env_states)

        key, subkey = jax.random.split(state.key)
        (pi_logits, value), _ = self.network.apply(
            state.model.params, state.model.state, subkey, env_obs
        )

        key, subkey = jax.random.split(key)
        noise_key, mcts_key = jax.random.split(subkey)

        prior_probs = jax.nn.softmax(pi_logits)

        dirichlet_noise = jax.random.dirichlet(
            noise_key,
            jnp.full(self.env.action_spec.num_values, self.config.dirichlet_alpha),
            shape=(self.config.batch_size,),
        )

        action_mask = jax.vmap(self.action_mask_fn)(state.env_states)
        noisy_priors = (
            prior_probs * (1 - self.config.dirichlet_mix)
            + dirichlet_noise * self.config.dirichlet_mix
        )
        valid = (noisy_priors > 0.0) & action_mask
        noisy_logits = jnp.where(valid, jnp.log(noisy_priors), -1e10)

        policy_output = self._policy_output(
            model=state.model,
            key=mcts_key,
            env_states=state.env_states,
            pi_logits=noisy_logits,
            value=value,
        )

        return state._replace(key=key), policy_output

    def _compute_gradients(
        self,
        model: ModelState,
        key: jax.Array,
        search_policy: jax.Array,
        search_value: jax.Array,
        obs: jax.Array,
        action_mask: jax.Array,
    ) -> tuple[
        tuple[jax.Array, tuple[hk.MutableState, jax.Array, jax.Array]], jax.Array
    ]:
        masked = search_policy * action_mask
        sum_ = jnp.sum(masked, axis=-1, keepdims=True)
        legal_cnt = jnp.sum(action_mask, axis=-1, keepdims=True)
        uniform_legal = jnp.where(action_mask > 0, 1.0 / jnp.maximum(legal_cnt, 1), 0.0)
        pi_target = jnp.where(sum_ > 0, masked / jnp.clip(sum_, 1e-10), uniform_legal)
        (loss, (net_state, pi_loss, value_loss)), grads = jax.value_and_grad(
            self._loss_fn, argnums=0, has_aux=True
        )(model.params, model.state, key, pi_target, search_value, obs)

        return (loss, (net_state, pi_loss, value_loss)), grads

    def _apply_updates(
        self,
        model: ModelState,
        opt_state: optax.OptState,
        grads: optax.Updates,
    ) -> tuple[hk.MutableParams, optax.OptState]:
        updates, opt_state = self.opt.update(grads, opt_state, model.params)
        params = optax.apply_updates(model.params, updates)
        return params, opt_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def train_step(self, state: TrainState) -> TrainState:
        batch_obs = jax.vmap(self.obs_fn, in_axes=(0,))

        def loop_fn(state: TrainState, _):
            state, policy_output = self._alphazero_search(state)

            raw_counts = policy_output.action_weights  # shape: [batch, num_actions]
            count_sums = jnp.sum(raw_counts, axis=-1, keepdims=True)
            count_sums = jnp.maximum(count_sums, 1.0)
            pi_target = raw_counts / count_sums

            obs = batch_obs(state.env_states)

            key, subkey = jax.random.split(state.key)
            env_states, reward, terminal = self._step_env(
                subkey, state.env_states, policy_output.action
            )
            action_mask = jax.vmap(self.action_mask_fn)(env_states)

            experience = TimeStep(
                obs=obs.astype(jnp.float32)[:, None, ...],
                reward=reward[:, None, None, ...],
                terminal=terminal[:, None, None, ...],
                pi=pi_target[:, None, ...],
                action_mask=action_mask.astype(jnp.bool_)[:, None, :],
            )
            buffer_state = self.buffer.add(state.buffer_state, experience)

            key, subkey = jax.random.split(key)
            sample = self.buffer.sample(buffer_state, subkey)

            B = self.config.batch_size
            K = self.config.unroll_steps
            n = self.config.n_step

            obs_seq = sample.experience.obs  # (B, T, ...)
            reward_seq = sample.experience.reward[..., 0]  # (B, T)
            terminal_seq = sample.experience.terminal[..., 0]  # (B, T)
            pi_seq = sample.experience.pi  # (B, T, A)
            mask_seq = sample.experience.action_mask  # (B, T, A)

            disc_seq = jnp.where(terminal_seq, 0.0, self.config.discount)  # (B, T)

            # bootstrap values at indices t+k+n.
            boot_obs = obs_seq[:, n : n + K]  # (B, K, ...)
            flat_boot_obs = boot_obs.reshape(B * K, *boot_obs.shape[2:])
            key, subkey = jax.random.split(key)
            (_, v_boot_flat), _ = self.network.apply(
                state.model.params, state.model.state, subkey, flat_boot_obs
            )
            v_boot = v_boot_flat.reshape(B, K)  # (B, K)

            value_seq = jnp.zeros_like(reward_seq)  # (B, T)
            # put bootstraps for t=n-1..n+K-1
            value_seq = value_seq.at[:, n - 1 : n + K - 1].set(v_boot)

            n_step_returns = functools.partial(
                rlax.n_step_bootstrapped_returns, n=n, stop_target_gradients=True
            )
            z_all = jax.vmap(n_step_returns)(reward_seq, disc_seq, value_seq)  # [B, T]

            z_targets = z_all[:, :K]  # (B, K)
            pi_targets = pi_seq[:, :K, :]  # (B, K, A)
            obs_targets = obs_seq[:, :K, ...]  # (B, K, ...)
            mask_targets = mask_seq[:, :K, :]  # (B, K, A)

            flat_obs = obs_targets.reshape(B * K, *obs_targets.shape[2:])  # (B*K, ...)
            flat_pi = pi_targets.reshape(B * K, pi_targets.shape[-1])  # (B*K, A)
            flat_z = z_targets.reshape(B * K)  # (B*K,)
            flat_mask = mask_targets.reshape(B * K, mask_targets.shape[-1])

            key, subkey = jax.random.split(key)
            (loss, (net_state, pi_loss, value_loss)), grads = self._compute_gradients(
                state.model, subkey, flat_pi, flat_z, flat_obs, flat_mask
            )
            params, opt_state = self._apply_updates(state.model, state.opt_state, grads)

            new_return = state.episode_return + reward

            next_episode_return = jnp.where(terminal, 0, new_return)
            next_avg_return = jnp.where(
                terminal,
                state.avg_return * self.config.avg_return_smoothing
                + new_return * (1 - self.config.avg_return_smoothing),
                state.avg_return,
            )
            next_avg_pi_loss = (
                state.avg_pi_loss * self.config.avg_return_smoothing
                + pi_loss * (1 - self.config.avg_return_smoothing)
            )
            next_avg_value_loss = (
                state.avg_value_loss * self.config.avg_return_smoothing
                + value_loss * (1 - self.config.avg_return_smoothing)
            )
            next_num_episodes = state.num_episodes + terminal.astype(jnp.int32)

            state = state._replace(
                model=ModelState(params=params, state=net_state),
                opt_state=opt_state,
                key=key,
                buffer_state=buffer_state,
                env_states=env_states,
                episode_return=next_episode_return,
                avg_return=next_avg_return,
                num_episodes=next_num_episodes,
                avg_pi_loss=next_avg_pi_loss,
                avg_value_loss=next_avg_value_loss,
            )

            return state, None

        state, _ = jax.lax.scan(loop_fn, state, None, length=self.config.eval_frequency)
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def evaluate(self, state: TrainState, max_steps: int) -> jax.Array:
        batch_reset = jax.vmap(self.env.reset, in_axes=(0,))
        batch_step = jax.vmap(self.env.step, in_axes=(0, 0))

        def single_action(env_state, key):
            return self.predict(state.model, key, env_state)

        batch_predict = jax.vmap(single_action, in_axes=(0, 0))

        def loop_fn(carry):
            env_states, reward_acc, done_mask, key, iter = carry
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

            return next_states, reward_acc, done_mask, key, iter + 1

        key, subkey = jax.random.split(state.key)
        reset_keys = jax.random.split(subkey, self.config.batch_size)
        env_states, _ = batch_reset(reset_keys)

        reward_acc = jnp.zeros(self.config.batch_size)
        done_mask = jnp.zeros(self.config.batch_size, dtype=jnp.bool_)

        _, reward_acc, _, _, _ = jax.lax.while_loop(
            lambda carry: jnp.any(~carry[2])
            & (carry[4] < max_steps),  # while any not done and iters under max steps
            loop_fn,
            (env_states, reward_acc, done_mask, key, 0),
        )

        return jnp.mean(reward_acc)

    def learn(
        self, state: TrainState, num_steps: int, checkpoints_dir: str
    ) -> tuple[TrainState, list[float], list[int]]:
        path = Path(checkpoints_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        returns = []
        steps = []
        time_step = 0

        for _ in range(num_steps // self.config.eval_frequency):
            state = self.train_step(state)

            valid_returns = state.avg_return[state.num_episodes > 0]
            avg_return = jnp.mean(valid_returns) if valid_returns.size > 0 else 0.0
            avg_pi_loss = jnp.mean(state.avg_pi_loss)
            avg_value_loss = jnp.mean(state.avg_value_loss)

            returns.append(avg_return)
            time_step += self.config.eval_frequency
            steps.append(time_step)
            ev = self.evaluate(state, self.config.max_eval_steps)

            print(
                f"Step {time_step} | Avg Return: {avg_return:.3f} | Eval: {ev:.3f} | "
                f"Pi Loss: {avg_pi_loss:.3f} | Value Loss: {avg_value_loss:.3f}",
                flush=True,
            )

            if time_step % self.config.checkpoint_frequency == 0:
                self.save_checkpoint(state, f"checkpoint-{time_step}", path)

        self.save_checkpoint(state, "checkpoint-final", path)

        print("Saving final checkpoint...", flush=True)
        self.train_checkpointer.wait_until_finished()

        return state, returns, steps

    def save_checkpoint(self, state: TrainState, filename: str, directory: Path):
        self.train_checkpointer.save(directory / filename, state)

    def restore_checkpoint(self, filename: str, directory: Path):
        state = self.init(jax.random.PRNGKey(0))  # create dummy state
        self.train_checkpointer.restore(directory / filename, state)
        return state
