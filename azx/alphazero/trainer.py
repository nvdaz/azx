import dataclasses
import functools
from copy import deepcopy
from pathlib import Path
from typing import Callable, NamedTuple

import chex
import flashbax as fbx
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import rlax
from flashbax.buffers.trajectory_buffer import (
    TrajectoryBufferSample,
    TrajectoryBufferState,
)
from jumanji.env import Environment
from tqdm import tqdm

from .agent import AlphaZero, Config, ModelState


@dataclasses.dataclass
class TrainConfig(Config):
    actor_batch_size: int
    train_batch_size: int
    eval_frequency: int
    gumbel_scale: float
    n_step: int
    unroll_steps: int
    max_eval_steps: int
    avg_return_smoothing: float
    checkpoint_frequency: int
    max_length_buffer: int
    min_length_buffer: int


class TrainState(NamedTuple):
    model: ModelState
    target_model: ModelState
    env_states: jax.Array
    buffer_state: TrajectoryBufferState
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
        super().__init__(env, config, network_fn, obs_fn, action_mask_fn)
        self.opt = opt
        self.config = config
        self.train_checkpointer = ocp.StandardCheckpointer()
        self.buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config.max_length_buffer,
            min_length_time_axis=config.min_length_buffer,
            sample_batch_size=config.train_batch_size,
            add_batch_size=config.actor_batch_size,
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
            reward=jnp.zeros((), dtype=jnp.float32),
            terminal=jnp.zeros((), dtype=jnp.bool_),
            pi=jnp.zeros((self.env.action_spec.num_values,), dtype=jnp.float32),
            action_mask=jnp.zeros((self.env.action_spec.num_values,), dtype=jnp.bool_),
        )
        buffer_state = self.buffer.init(experience)

        opt_state = self.opt.init(params)

        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, self.config.actor_batch_size)
        env_states, _ = jax.vmap(self.env.reset)(subkeys)

        model = ModelState(params, net_state)

        return TrainState(
            model=model,
            target_model=deepcopy(model),
            env_states=env_states,
            buffer_state=buffer_state,
            opt_state=opt_state,
            avg_loss=jnp.zeros(self.config.train_batch_size),
            avg_pi_loss=jnp.zeros(self.config.train_batch_size),
            avg_value_loss=jnp.zeros(self.config.train_batch_size),
            avg_return=jnp.zeros(self.config.actor_batch_size),
            episode_return=jnp.zeros(self.config.actor_batch_size),
            num_episodes=jnp.zeros(self.config.actor_batch_size),
            eval_avg_return=jnp.zeros(self.config.actor_batch_size),
            eval_episode_return=jnp.zeros(self.config.actor_batch_size),
            key=key,
        )

    def _compute_target_values(
        self,
        target_model: ModelState,
        key: jax.Array,
        obs: jax.Array,
    ) -> jax.Array:
        B, T = obs.shape[:2]

        flat_obs = obs.reshape(B * T, *obs.shape[2:])

        (_, flat_value_logits), _ = self.network.apply(
            target_model.params, target_model.state, key, flat_obs
        )

        flat_values = self.support.decode_logits(flat_value_logits)
        return flat_values.reshape(B, T)

    def _loss_fn(
        self,
        params: hk.MutableParams,
        net_state: hk.MutableState,
        target: ModelState,
        rng: jax.Array,
        batch: TrajectoryBufferSample,
    ):
        unroll_steps = self.config.unroll_steps
        total_steps = self.config.n_step + self.config.unroll_steps
        n_step = self.config.n_step

        exp = batch.experience
        obs = exp.obs
        rewards = exp.reward
        terminals = exp.terminal
        target_pi = exp.pi
        action_mask = exp.action_mask

        discounts = self.config.discount * (1.0 - terminals)
        batch_size = obs.shape[0]

        rng, target_key = jax.random.split(rng)
        target_values_from_obs = self._compute_target_values(target, target_key, obs)
        target_values_from_obs = jax.lax.stop_gradient(target_values_from_obs)

        rng, step_rng = jax.random.split(rng)
        step_keys = jax.random.split(step_rng, total_steps)

        support_size = self.support.size
        num_actions = self.env.action_spec.num_values

        def step_fn(carry, t):
            pred_state, v_logits, p_logits = carry
            key = step_keys[t]

            obs_t = obs[:, t]
            (policy_logits_t, value_logits_t), pred_state = self.network.apply(
                params, pred_state, key, obs_t
            )

            v_logits = v_logits.at[:, t].set(value_logits_t)
            p_logits = p_logits.at[:, t].set(policy_logits_t)

            return (pred_state, v_logits, p_logits), None

        v_logits_unroll = jnp.zeros(
            (batch_size, total_steps, support_size), dtype=jnp.float32
        )
        p_logits_unroll = jnp.zeros(
            (batch_size, total_steps, num_actions), dtype=jnp.float32
        )

        (pred_state_new, v_logits_unroll, p_logits_unroll), _ = jax.lax.scan(
            step_fn,
            (net_state, v_logits_unroll, p_logits_unroll),
            jnp.arange(total_steps),
        )

        nstep_returns = functools.partial(
            rlax.n_step_bootstrapped_returns,
            n=n_step,
            stop_target_gradients=True,
        )
        target_z = jax.vmap(nstep_returns)(rewards, discounts, target_values_from_obs)

        # Slice the first K steps
        target_value_k = target_z[:, :unroll_steps]  # (B, K)
        pred_value_logits_k = v_logits_unroll[:, :unroll_steps]  # (B, K, S)

        pred_policy_logits_k = p_logits_unroll[:, :unroll_steps]  # (B, K, A)
        target_policy_k = target_pi[:, :unroll_steps]  # (B, K, A)
        mask_k = action_mask[:, :unroll_steps]  # (B, K, A)

        # Masking and normalization in the same style as MuZero
        masked_policy_logits = jnp.where(
            mask_k,
            pred_policy_logits_k,
            jnp.array(-1e9, dtype=pred_policy_logits_k.dtype),
        )
        masked_target_policy = target_policy_k * mask_k
        target_policy_norm = masked_target_policy / (
            masked_target_policy.sum(-1, keepdims=True) + 1e-9
        )

        # Categorical targets over the value support
        target_value_prob = self.support.encode(target_value_k)

        loss_pi = optax.softmax_cross_entropy(
            masked_policy_logits,
            target_policy_norm,
        ).mean()

        loss_v = optax.softmax_cross_entropy(
            pred_value_logits_k,
            target_value_prob,
        ).mean()

        total_loss = loss_pi + 0.5 * loss_v

        return total_loss, (
            pred_state_new,
            jax.lax.stop_gradient(loss_v),
            jax.lax.stop_gradient(loss_pi),
        )

    def _step_env(
        self, key: chex.PRNGKey, env_states: chex.ArrayTree, actions: jax.Array
    ) -> tuple[chex.ArrayTree, jax.Array, jax.Array]:
        env_states, steps = jax.vmap(self.env.step)(env_states, actions)
        reward = jax.vmap(lambda x: x.reward)(steps)
        terminal = jax.vmap(lambda x: x.last())(steps)

        subkeys = jax.random.split(key, self.config.actor_batch_size)
        reset_states, _ = jax.vmap(self.env.reset)(subkeys)

        env_states = jax.vmap(
            lambda t, reset_state, env_state: jax.lax.cond(
                t,
                lambda x: x[0],
                lambda x: x[1],
                (reset_state, env_state),
            )
        )(terminal, reset_states, env_states)

        return env_states, reward, terminal

    def _train_from_batch(self, state: TrainState) -> TrainState:
        key, subkey = jax.random.split(state.key)
        batch = self.buffer.sample(state.buffer_state, subkey)

        (loss, (net_state, loss_v, loss_pi)), grads = jax.value_and_grad(
            self._loss_fn, argnums=0, has_aux=True
        )(state.model.params, state.model.state, state.target_model, subkey, batch)

        updates, opt_state = self.opt.update(grads, state.opt_state, state.model.params)
        params = optax.apply_updates(state.model.params, updates)

        avg_pi = state.avg_pi_loss * self.config.avg_return_smoothing + loss_pi * (
            1 - self.config.avg_return_smoothing
        )
        avg_v = state.avg_value_loss * self.config.avg_return_smoothing + loss_v * (
            1 - self.config.avg_return_smoothing
        )
        avg = state.avg_loss * self.config.avg_return_smoothing + loss * (
            1 - self.config.avg_return_smoothing
        )

        return state._replace(
            key=key,
            model=ModelState(params=params, state=net_state),
            opt_state=opt_state,
            avg_loss=avg,
            avg_pi_loss=avg_pi,
            avg_value_loss=avg_v,
        )

    def _actor_step(self, state: TrainState) -> TrainState:
        key, search_key, step_key = jax.random.split(state.key, 3)
        obs = jax.vmap(self.obs_fn)(state.env_states)
        valid_actions = jax.vmap(self.action_mask_fn)(state.env_states)

        policy_output = self._alphazero_search(
            model=state.model,
            key=search_key,
            env_states=state.env_states,
            gumbel_scale=self.config.gumbel_scale,
        )
        env_states, reward, terminal = self._step_env(
            step_key, state.env_states, policy_output.action
        )

        experience = TimeStep(
            obs=obs[:, None, ...],
            reward=reward[:, None, ...],
            terminal=terminal[:, None, ...],
            pi=policy_output.action_weights[:, None, ...],
            action_mask=valid_actions.astype(jnp.bool_)[:, None, ...],
        )
        buffer_state = self.buffer.add(state.buffer_state, experience)

        new_return = state.episode_return + reward
        next_episode_return = jnp.where(terminal, 0, new_return)
        next_avg_return = jnp.where(
            terminal,
            new_return,
            state.avg_return,
        )
        next_num_episodes = state.num_episodes + terminal.astype(jnp.int32)

        return state._replace(
            key=key,
            buffer_state=buffer_state,
            env_states=env_states,
            episode_return=next_episode_return,
            avg_return=next_avg_return,
            num_episodes=next_num_episodes,
        )

    @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def train_step(self, state: TrainState) -> TrainState:
        def loop_fn(state: TrainState, _):
            state = self._actor_step(state)

            state = jax.lax.cond(
                self.buffer.can_sample(state.buffer_state),
                self._train_from_batch,
                lambda st: st,
                state,
            )

            return state, None

        state, _ = jax.lax.scan(loop_fn, state, None, length=self.config.eval_frequency)
        return state._replace(target_model=state.model)

    @functools.partial(jax.jit, static_argnums=(0,))
    def evaluate(self, state: TrainState, max_steps: int) -> jax.Array:
        def loop_fn(carry):
            env_states, reward_acc, done_mask, key, iter = carry
            key, subkey = jax.random.split(key)

            policy_output = self._alphazero_search(state.model, subkey, env_states)
            action = policy_output.action

            # step the envs
            next_states, steps = jax.vmap(self.env.step)(env_states, action)
            r = jax.vmap(lambda ts: ts.reward)(steps)
            done = jax.vmap(lambda ts: ts.last())(steps)

            # accumulate only for unfinished envs
            reward_acc = jnp.where(done_mask, reward_acc, reward_acc + r)
            done_mask = jnp.logical_or(done_mask, done)

            return next_states, reward_acc, done_mask, key, iter + 1

        key, subkey = jax.random.split(state.key)
        reset_keys = jax.random.split(subkey, self.config.actor_batch_size)
        env_states, _ = jax.vmap(self.env.reset)(reset_keys)

        reward_acc = jnp.zeros(self.config.actor_batch_size)
        done_mask = jnp.zeros(self.config.actor_batch_size, dtype=jnp.bool_)

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

        pbar = tqdm(range(num_steps // self.config.eval_frequency), leave=True)
        for _ in pbar:
            state = self.train_step(state)

            valid_returns = state.avg_return[state.num_episodes > 0]
            avg_return = jnp.mean(valid_returns) if valid_returns.size > 0 else 0.0
            avg_pi_loss = jnp.mean(state.avg_pi_loss)
            avg_value_loss = jnp.mean(state.avg_value_loss)

            returns.append(avg_return)
            time_step += self.config.eval_frequency
            steps.append(time_step)
            ev = self.evaluate(state, self.config.max_eval_steps)

            pbar.write(
                f"Step {time_step:06d} | Avg Return: {avg_return:.3f} | Eval: {ev:.3f} | "
                f"Pi Loss: {avg_pi_loss:.3f} | Value Loss: {avg_value_loss:.3f}",
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
