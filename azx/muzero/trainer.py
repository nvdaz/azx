import dataclasses
import functools
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

from .agent import Config, ModelNetState, ModelParams, ModelState, MuZero


@dataclasses.dataclass
class TrainConfig(Config):
    batch_size: int
    eval_frequency: int
    n_step: int
    unroll_steps: int
    max_eval_steps: int
    avg_return_smoothing: float
    checkpoint_frequency: int
    max_length_buffer: int
    min_length_buffer: int


class TrainState(NamedTuple):
    model: ModelState
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
    action: jax.Array


class MuZeroTrainer(MuZero):
    def __init__(
        self,
        env: Environment,
        config: TrainConfig,
        representation_fn: Callable[[jax.Array], jax.Array],
        dynamics_fn: Callable[[jax.Array, jax.Array], jax.Array],
        prediction_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
        obs_fn: Callable[[chex.ArrayTree], jax.Array],
        action_mask_fn: Callable[[chex.ArrayTree], jax.Array],
        opt: optax.GradientTransformation,
    ):
        super().__init__(config, representation_fn, dynamics_fn, prediction_fn)
        self.env = env
        self.obs_fn = obs_fn
        self.opt = opt
        self.config = config
        self.action_mask_fn = action_mask_fn
        self.train_checkpointer = ocp.StandardCheckpointer()
        self.buffer = fbx.make_trajectory_buffer(
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

        experience = TimeStep(
            obs=obs[0],
            reward=jnp.zeros((), dtype=jnp.float32),
            terminal=jnp.zeros((), dtype=jnp.bool_),
            pi=jnp.zeros((self.env.action_spec.num_values,), dtype=jnp.float32),
            action_mask=jnp.zeros((self.env.action_spec.num_values,), dtype=jnp.bool_),
            action=jnp.zeros((), dtype=jnp.int32),
        )
        buffer_state = self.buffer.init(experience)

        key, subkey = jax.random.split(key)
        rep_params, rep_state = self.rep_net.init(subkey, obs)

        key, subkey = jax.random.split(key)
        dummy_latent, _ = self.rep_net.apply(rep_params, rep_state, subkey, obs)
        dummy_action = jnp.zeros((1,), dtype=jnp.int32)

        key, subkey = jax.random.split(key)
        dyn_params, dyn_state = self.dyn_net.init(subkey, dummy_latent, dummy_action)

        key, subkey = jax.random.split(key)
        pred_params, pred_state = self.pred_net.init(subkey, dummy_latent)

        params = ModelParams(rep=rep_params, dyn=dyn_params, pred=pred_params)
        net_state = ModelNetState(rep=rep_state, dyn=dyn_state, pred=pred_state)
        model = ModelState(params=params, state=net_state)

        opt_state = self.opt.init(params)

        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, self.config.batch_size)
        env_states, _ = jax.vmap(self.env.reset, in_axes=(0,))(subkeys)

        return TrainState(
            model=model,
            buffer_state=buffer_state,
            env_states=env_states,
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
        params: ModelParams,
        net_state: ModelNetState,
        rng: jax.Array,
        batch: TrajectoryBufferSample,
    ):
        K = self.config.unroll_steps
        T = self.config.n_step + self.config.unroll_steps
        n = self.config.n_step

        keys = jax.random.split(rng, T + 1)
        rep_key, step_keys = keys[0], keys[1:]

        obs = batch.experience.obs
        actions = batch.experience.action
        rewards = batch.experience.reward
        discounts = self.config.discount * (1.0 - batch.experience.terminal)
        target_pi = batch.experience.pi
        action_mask = batch.experience.action_mask

        s0, rep_state_new = self.rep_net.apply(
            params.rep, net_state.rep, rep_key, obs[:, 0]
        )

        def step_fn(carry, t):
            s, pred_st, dyn_st, v_acc, log_acc, r_acc = carry
            key = step_keys[t]

            (logits, v), pred_st = self.pred_net.apply(params.pred, pred_st, key, s)
            s_scaled = 0.5 * s + 0.5 * jax.lax.stop_gradient(s)  # multiply grads by 0.5

            (s_next, r_pred, _), dyn_st = self.dyn_net.apply(
                params.dyn, dyn_st, key, s_scaled, actions[:, t]
            )

            v_acc = v_acc.at[:, t].set(v)
            r_acc = r_acc.at[:, t].set(r_pred)
            log_acc = log_acc.at[:, t, :].set(logits)

            return (s_next, pred_st, dyn_st, v_acc, log_acc, r_acc), None

        B = self.config.batch_size
        A = self.env.action_spec.num_values
        v_preds = jnp.zeros((B, T), dtype=jnp.float32)
        r_preds = jnp.zeros((B, T), dtype=jnp.float32)
        logits_all = jnp.zeros((B, T, A), dtype=jnp.float32)

        (_, pred_state_new, dyn_state_new, v_preds, logits_all, r_preds), _ = (
            jax.lax.scan(
                step_fn,
                (s0, net_state.pred, net_state.dyn, v_preds, logits_all, r_preds),
                jnp.arange(T),
            )
        )

        value_seq = jnp.zeros_like(rewards)
        value_seq = value_seq.at[:, n : n + K].set(v_preds[:, n : n + K])

        nstep_fn = functools.partial(
            rlax.n_step_bootstrapped_returns, n=n, stop_target_gradients=True
        )
        z_targets = jax.vmap(nstep_fn, in_axes=(0, 0, 0), out_axes=0)(
            rewards, discounts, value_seq
        )

        z_t = z_targets[:, :K]
        v_t = v_preds[:, :K]
        r_t = r_preds[:, :K]
        r_true = rewards[:, :K]
        logits_t = logits_all[:, :K]
        pi_t = target_pi[:, :K]
        mask_t = action_mask[:, :K]

        masked_logits = jnp.where(
            mask_t, logits_t, jnp.array(1e-9, dtype=logits_t.dtype)
        )
        masked_targets = pi_t * mask_t
        masked_logits = logits_t
        # masked_targets = pi_t
        target_pi_norm = masked_targets / (masked_targets.sum(-1, keepdims=True) + 1e-9)

        loss_pi = optax.softmax_cross_entropy(masked_logits, target_pi_norm).mean()
        loss_v = optax.squared_error(v_t, z_t).mean()
        loss_r = optax.squared_error(r_t, r_true).mean()

        total_loss = loss_pi + loss_v + loss_r

        new_net_state = ModelNetState(
            rep=rep_state_new, pred=pred_state_new, dyn=dyn_state_new
        )

        return total_loss, (
            new_net_state,
            jax.lax.stop_gradient(loss_r),
            jax.lax.stop_gradient(loss_v),
            jax.lax.stop_gradient(loss_pi),
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

    def _compute_gradients(
        self,
        model: ModelState,
        key: jax.Array,
        batch: TrajectoryBufferSample,
    ):
        (loss, (net_state, loss_r, loss_v, loss_pi)), grads = jax.value_and_grad(
            self._loss_fn, argnums=0, has_aux=True
        )(model.params, model.state, key, batch)

        return (loss, (net_state, loss_r, loss_v, loss_pi)), grads

    def _apply_updates(
        self, params0: ModelState, opt_state: optax.OptState, grads: optax.Updates
    ) -> tuple[hk.MutableParams, optax.OptState]:
        updates, opt_state = self.opt.update(grads, opt_state, params0)
        params = optax.apply_updates(params0, updates)
        return params, opt_state

    def _train_from_batch(self, state: TrainState) -> TrainState:
        key, subkey = jax.random.split(state.key)
        batch = self.buffer.sample(state.buffer_state, subkey)

        (loss, (net_state, loss_r, loss_v, loss_pi)), grads = self._compute_gradients(
            state.model, subkey, batch
        )

        params, opt_state = self._apply_updates(
            state.model.params, state.opt_state, grads
        )

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

    @functools.partial(jax.jit, static_argnums=(0,))
    def train_step(self, state: TrainState) -> TrainState:
        batch_obs = jax.vmap(self.obs_fn, in_axes=(0,))

        def loop_fn(state: TrainState, _):
            key, search_key, step_key = jax.random.split(state.key, 3)
            obs = jax.vmap(self.obs_fn)(state.env_states)
            policy_output = self._muzero_search(state.model, search_key, obs)
            raw_counts = policy_output.action_weights  # shape: [batch, num_actions]
            count_sums = jnp.sum(raw_counts, axis=-1, keepdims=True)
            count_sums = jnp.maximum(count_sums, 1.0)
            pi_target = raw_counts / count_sums

            obs = batch_obs(state.env_states)
            action_mask = jax.vmap(self.action_mask_fn)(state.env_states)
            env_states, reward, terminal = self._step_env(
                step_key, state.env_states, policy_output.action
            )

            experience = TimeStep(
                obs=obs.astype(jnp.float32)[:, None, ...],
                reward=reward[:, None, ...],
                terminal=terminal[:, None, ...],
                pi=pi_target[:, None, ...],
                action_mask=action_mask.astype(jnp.bool_)[:, None, ...],
                action=policy_output.action[:, None],
            )
            buffer_state = self.buffer.add(state.buffer_state, experience)

            new_return = state.episode_return + reward

            next_episode_return = jnp.where(terminal, 0, new_return)
            next_avg_return = jnp.where(
                terminal,
                state.avg_return * self.config.avg_return_smoothing
                + new_return * (1 - self.config.avg_return_smoothing),
                state.avg_return,
            )
            next_num_episodes = state.num_episodes + terminal.astype(jnp.int32)

            state = state._replace(
                key=key,
                buffer_state=buffer_state,
                env_states=env_states,
                episode_return=next_episode_return,
                avg_return=next_avg_return,
                num_episodes=next_num_episodes,
            )

            state = jax.lax.cond(
                self.buffer.can_sample(state.buffer_state),
                self._train_from_batch,
                lambda st: st,
                state,
            )

            return state, None

        state, _ = jax.lax.scan(loop_fn, state, None, self.config.eval_frequency)
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def evaluate(self, state: TrainState, max_steps: int) -> jax.Array:
        batch_reset = jax.vmap(self.env.reset, in_axes=(0,))
        batch_step = jax.vmap(self.env.step, in_axes=(0, 0))

        def single_action(env_state, key):
            return self.predict(state.model, key, self.obs_fn(env_state))

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
                f"Step {time_step:06d} | Avg Return: {avg_return:.3f} | Eval: {ev:.3f} | "
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
