import dataclasses
import functools
from pathlib import Path
from typing import NamedTuple

import chex
import flashbax as fbx
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flashbax.buffers.trajectory_buffer import (
    TrajectoryBufferSample,
    TrajectoryBufferState,
)

from azx.alphazero.agent import AgentState, AlphaZero
from azx.core.env import init_buffer, reset_envs
from azx.core.losses import (
    nstep_value_targets,
    policy_cross_entropy,
    support_cross_entropy,
    trajectory_alive_mask,
)
from azx.core.rollout import EvalStats, collect_rollout_step, evaluate_rollout
from azx.core.training import learn_loop


@dataclasses.dataclass
class TrainConfig:
    actor_batch_size: int
    train_batch_size: int
    gumbel_scale: float
    eval_frequency: int
    n_step: int
    unroll_steps: int
    max_eval_steps: int
    checkpoint_frequency: int
    max_length_buffer: int
    min_length_buffer: int
    value_loss_weight: float


class TrainState(NamedTuple):
    model: AgentState
    env_states: chex.ArrayTree
    buffer_state: TrajectoryBufferState
    opt_state: optax.OptState
    episode_return: jax.Array
    key: jax.Array
    eval_episode_return: chex.ArrayTree
    eval_avg_return: chex.ArrayTree


class TrainStats(NamedTuple):
    loss: jax.Array = jnp.array(jnp.nan)
    policy_loss: jax.Array = jnp.array(jnp.nan)
    value_loss: jax.Array = jnp.array(jnp.nan)
    episode_return: jax.Array = jnp.array(jnp.nan)


class AlphaZeroTrainer:
    def __init__(
        self,
        agent: AlphaZero,
        config: TrainConfig,
        opt: optax.GradientTransformation,
    ):
        self.agent = agent
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
        reset_key, agent_key = jax.random.split(key)
        agent_state = self.agent.init(agent_key)

        buffer_state = init_buffer(self.agent.adapter, self.buffer)
        env_states = reset_envs(
            self.agent.adapter, self.config.actor_batch_size, reset_key
        )
        opt_state = self.opt.init(agent_state.params)

        return TrainState(
            model=agent_state,
            env_states=env_states,
            buffer_state=buffer_state,
            opt_state=opt_state,
            episode_return=jnp.zeros(self.config.actor_batch_size),
            eval_avg_return=jnp.zeros(self.config.actor_batch_size),
            eval_episode_return=jnp.zeros(self.config.actor_batch_size),
            key=key,
        )

    def _loss_fn(
        self,
        params: hk.MutableParams,
        net_state: hk.MutableState,
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
        target_pi = exp.policy
        action_mask = exp.action_mask
        target_mcts_value = exp.value

        discounts = self.agent.config.discount * (1.0 - terminals)
        batch_size = obs.shape[0]

        rng, step_rng = jax.random.split(rng)
        step_keys = jax.random.split(step_rng, total_steps)

        support_size = self.agent.support.size
        num_actions = self.agent.adapter.env.action_spec.num_values

        def step_fn(carry, t):
            pred_state, v_logits, p_logits = carry
            key = step_keys[t]

            obs_t = obs[:, t]
            (policy_logits_t, value_logits_t), pred_state = self.agent.network.apply(
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

        target_z = nstep_value_targets(rewards, discounts, target_mcts_value, n_step)

        # Slice the first K steps
        target_value_k = target_z[:, :unroll_steps]  # (B, K)
        pred_value_logits_k = v_logits_unroll[:, :unroll_steps]  # (B, K, S)

        pred_policy_logits_k = p_logits_unroll[:, :unroll_steps]  # (B, K, A)
        target_policy_k = target_pi[:, :unroll_steps]  # (B, K, A)
        mask_k = action_mask[:, :unroll_steps]  # (B, K, A)

        alive_mask, alive_sum = trajectory_alive_mask(terminals, unroll_steps)

        loss_pi = policy_cross_entropy(
            pred_policy_logits_k,
            target_policy_k,
            mask_k,
            alive_mask,
            alive_sum,
        )

        loss_v = support_cross_entropy(
            pred_value_logits_k,
            target_value_k,
            self.agent.support,
            alive_mask,
            alive_sum,
        )

        total_loss = loss_pi + self.config.value_loss_weight * loss_v

        return total_loss, (
            pred_state_new,
            jax.lax.stop_gradient(loss_v),
            jax.lax.stop_gradient(loss_pi),
        )

    def _train_from_batch(self, state: TrainState) -> tuple[TrainState, TrainStats]:
        key, subkey = jax.random.split(state.key)
        batch = self.buffer.sample(state.buffer_state, subkey)

        (loss, (net_state, loss_v, loss_pi)), grads = jax.value_and_grad(
            self._loss_fn, argnums=0, has_aux=True
        )(state.model.params, state.model.state, subkey, batch)

        updates, opt_state = self.opt.update(grads, state.opt_state, state.model.params)
        params = optax.apply_updates(state.model.params, updates)

        stats = TrainStats(loss=loss, policy_loss=loss_pi, value_loss=loss_v)

        return state._replace(
            key=key,
            model=AgentState(params=params, state=net_state),
            opt_state=opt_state,
        ), stats

    def _actor_step(self, state: TrainState) -> tuple[TrainState, jax.Array]:
        key, rollout_key = jax.random.split(state.key)
        rollout = collect_rollout_step(
            self.agent.adapter,
            state.env_states,
            lambda search_key, env_states: self.agent.search(
                model=state.model,
                key=search_key,
                env_states=env_states,
                gumbel_scale=self.config.gumbel_scale,
            ),
            rollout_key,
        )
        buffer_state = self.buffer.add(state.buffer_state, rollout.experience)

        new_return = state.episode_return + rollout.reward
        next_episode_return = jnp.where(rollout.terminal, 0, new_return)

        state = state._replace(
            key=key,
            buffer_state=buffer_state,
            env_states=rollout.env_states,
            episode_return=next_episode_return,
        )

        episode_return = jnp.where(rollout.terminal, new_return, jnp.nan)
        return state, episode_return

    @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def train_step(self, state: TrainState) -> tuple[TrainState, TrainStats]:
        def loop_fn(state: TrainState, _):
            state, episode_return = self._actor_step(state)

            state, stats = jax.lax.cond(
                self.buffer.can_sample(state.buffer_state),
                self._train_from_batch,
                lambda st: (st, TrainStats()),
                state,
            )
            stats = stats._replace(episode_return=episode_return)

            return state, stats

        state, stats_series = jax.lax.scan(
            loop_fn, state, None, length=self.config.eval_frequency
        )

        stats = jax.tree_util.tree_map(jnp.nanmean, stats_series)
        return state, stats

    @functools.partial(jax.jit, static_argnums=(0,))
    def evaluate(self, state: TrainState) -> EvalStats:
        return evaluate_rollout(
            self.agent.adapter,
            self.config.actor_batch_size,
            self.config.max_eval_steps,
            state.key,
            lambda search_key, env_states: self.agent.search(
                model=state.model,
                key=search_key,
                env_states=env_states,
            ),
        )

    def learn(
        self, state: TrainState, num_steps: int, checkpoints_dir: Path
    ) -> TrainState:
        return learn_loop(
            state=state,
            eval_frequency=self.config.eval_frequency,
            checkpoint_frequency=self.config.checkpoint_frequency,
            num_steps=num_steps,
            checkpoints_dir=checkpoints_dir,
            checkpointer=self.train_checkpointer,
            train_step=self.train_step,
            evaluate=self.evaluate,
            build_train_metrics=lambda stats: {
                "loss/total": stats.loss,
                "loss/policy": stats.policy_loss,
                "loss/value": stats.value_loss,
                "train/avg_return": stats.episode_return,
            },
            build_eval_metrics=lambda ev: {"eval/avg_return": ev.episode_return},
            run_config={
                "algorithm": "alphazero",
                "agent_config": dataclasses.asdict(self.agent.config),
                "train_config": dataclasses.asdict(self.config),
            },
        )

    def save_checkpoint(self, state: TrainState, filename: str, directory: Path):
        self.train_checkpointer.save(directory / filename, state)

    def restore_checkpoint(self, filename: str, directory: Path):
        state = self.init(jax.random.PRNGKey(0))  # create dummy state
        self.train_checkpointer.restore(directory / filename, state)
        return state
