import dataclasses
import functools
from pathlib import Path
from typing import Generic, NamedTuple, TypeVar

import chex
import flashbax as fbx
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flashbax.buffers.trajectory_buffer import (
    TrajectoryBuffer,
    TrajectoryBufferSample,
    TrajectoryBufferState,
)
from mctx import PolicyOutput

from azx.core.env import EnvironmentAdapter, TimeStep, init_buffer, reset_envs
from azx.core.losses import (
    nstep_value_targets,
    policy_cross_entropy,
    support_cross_entropy,
    trajectory_alive_mask,
)
from azx.core.rollout import EvalStats, collect_rollout_step, evaluate_rollout
from azx.core.training import learn_loop
from azx.muzero.agent import ModelNetState, ModelParams, ModelState, MuZero


class _UnrollOutputs(NamedTuple):
    pred_state: hk.MutableState
    dyn_state: hk.MutableState
    v_logits: jax.Array
    p_logits: jax.Array
    r_logits: jax.Array
    h_states: jax.Array


class _Targets(NamedTuple):
    value_targets: jax.Array
    reward_targets: jax.Array
    policy_targets: jax.Array
    value_logits: jax.Array
    reward_logits: jax.Array
    policy_logits: jax.Array
    action_mask: jax.Array


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
    consistency_loss_weight: float
    value_loss_weight: float


class TrainState(NamedTuple):
    model: ModelState
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
    reward_loss: jax.Array = jnp.array(jnp.nan)
    consistency_loss: jax.Array = jnp.array(jnp.nan)
    episode_return: jax.Array = jnp.array(jnp.nan)


TState = TypeVar("TState")


class MuZeroTrainer(Generic[TState]):
    def __init__(
        self,
        agent: MuZero,
        adapter: EnvironmentAdapter,
        config: TrainConfig,
        opt: optax.GradientTransformation,
    ):
        self.agent = agent
        self.adapter = adapter
        self.opt = opt
        self.config = config
        self.train_checkpointer = ocp.StandardCheckpointer()
        self.buffer: TrajectoryBuffer[
            TimeStep, TrajectoryBufferState, TrajectoryBufferSample
        ] = fbx.make_trajectory_buffer(
            max_length_time_axis=config.max_length_buffer,
            min_length_time_axis=config.min_length_buffer,
            sample_batch_size=config.train_batch_size,
            add_batch_size=config.actor_batch_size,
            sample_sequence_length=config.n_step + config.unroll_steps + 1,
            period=1,
        )

    def init(self, key: chex.PRNGKey) -> TrainState:
        buffer_state = init_buffer(self.adapter, self.buffer)

        reset_key, model_key, env_key = jax.random.split(key, 3)
        state, _ = self.adapter.env.reset(reset_key)
        obs = self.adapter.obs_fn(state)

        model = self.agent.init(model_key, obs)

        opt_state = self.opt.init(model.params)

        env_states = reset_envs(self.adapter, self.config.actor_batch_size, env_key)

        return TrainState(
            model=model,
            buffer_state=buffer_state,
            env_states=env_states,
            opt_state=opt_state,
            episode_return=jnp.zeros(self.config.actor_batch_size),
            eval_avg_return=jnp.zeros(self.config.actor_batch_size),
            eval_episode_return=jnp.zeros(self.config.actor_batch_size),
            key=key,
        )

    def _loss_fn(
        self,
        params: ModelParams,
        net_state: ModelNetState,
        rng: jax.Array,
        batch: TrajectoryBufferSample[TimeStep],
    ):
        num_predictions = self.config.unroll_steps + 1
        exp = batch.experience

        rng, compute_key = jax.random.split(rng)
        root_state, rep_state = self.agent.rep_net.apply(
            params.rep, net_state.rep, compute_key, exp.obs[:, 0]
        )

        rng, step_key = jax.random.split(rng)
        unroll = self._unroll_predictions(
            params,
            net_state,
            root_state,
            exp,
            step_key,
        )
        discounts = self.agent.config.discount * (1.0 - exp.terminal)
        target_z = nstep_value_targets(
            exp.reward, discounts, exp.value, self.config.n_step
        )
        targets = _Targets(
            value_targets=target_z[:, :num_predictions],
            reward_targets=exp.reward[:, :num_predictions],
            policy_targets=exp.policy[:, :num_predictions],
            value_logits=unroll.v_logits[:, :num_predictions],
            reward_logits=unroll.r_logits[:, :num_predictions],
            policy_logits=unroll.p_logits[:, :num_predictions],
            action_mask=exp.action_mask[:, :num_predictions],
        )

        rng, compute_key = jax.random.split(rng)
        loss_pi, loss_v, loss_r, loss_consistency, rep_state = self._compute_losses(
            params,
            rep_state,
            exp,
            targets,
            unroll.h_states,
            compute_key,
        )

        total_loss = (
            loss_pi
            + self.config.value_loss_weight * loss_v
            + loss_r
            + self.config.consistency_loss_weight * loss_consistency
        )

        new_net_state = ModelNetState(
            rep=rep_state,
            pred=unroll.pred_state,
            dyn=unroll.dyn_state,
        )

        return total_loss, (
            new_net_state,
            jax.lax.stop_gradient(loss_r),
            jax.lax.stop_gradient(loss_v),
            jax.lax.stop_gradient(loss_pi),
            jax.lax.stop_gradient(loss_consistency),
        )

    def _unroll_predictions(
        self,
        params: ModelParams,
        net_state: ModelNetState,
        root_state: jax.Array,
        exp: TimeStep,
        key: jax.Array,
    ) -> _UnrollOutputs:
        num_predictions = self.config.unroll_steps + 1
        actions = exp.action[:, :num_predictions]
        step_keys = jax.random.split(key, num_predictions)
        terminals = exp.terminal[:, :num_predictions]
        done_cumulative = jnp.cumsum(terminals.astype(jnp.int32), axis=1) > 0
        done_before_all = jnp.concatenate(
            [
                jnp.zeros_like(done_cumulative[:, :1], dtype=jnp.bool_),
                done_cumulative[:, :-1],
            ],
            axis=1,
        )

        def step_fn(carry, t):
            hidden_state, pred_state, dyn_state = carry
            key = step_keys[t]
            pred_key, dyn_key = jax.random.split(key)

            (policy_logit_t, value_logit_t), pred_state = self.agent.pred_net.apply(
                params.pred, pred_state, pred_key, hidden_state
            )

            (next_state, reward_logit_t), dyn_state = self.agent.dyn_net.apply(
                params.dyn, dyn_state, dyn_key, hidden_state, actions[:, t]
            )

            done_before_t = done_before_all[:, t]
            done_mask = jnp.expand_dims(
                done_before_t.astype(bool), axis=tuple(range(1, hidden_state.ndim))
            )
            next_state = jnp.where(done_mask, hidden_state, next_state)
            next_state = 0.5 * next_state + 0.5 * jax.lax.stop_gradient(next_state)

            return (next_state, pred_state, dyn_state), (
                value_logit_t,
                policy_logit_t,
                reward_logit_t,
                next_state,
            )

        (_, pred_state_new, dyn_state_new), outputs = jax.lax.scan(
            step_fn,
            (root_state, net_state.pred, net_state.dyn),
            jnp.arange(num_predictions),
        )
        v_logits_unroll, p_logits_unroll, r_logits_unroll, h_logits_unroll = outputs
        v_logits_unroll = jnp.swapaxes(v_logits_unroll, 0, 1)
        p_logits_unroll = jnp.swapaxes(p_logits_unroll, 0, 1)
        r_logits_unroll = jnp.swapaxes(r_logits_unroll, 0, 1)
        h_logits_unroll = jnp.swapaxes(h_logits_unroll, 0, 1)

        return _UnrollOutputs(
            pred_state=pred_state_new,
            dyn_state=dyn_state_new,
            v_logits=v_logits_unroll,
            p_logits=p_logits_unroll,
            r_logits=r_logits_unroll,
            h_states=h_logits_unroll,
        )

    def _compute_losses(
        self,
        params: ModelParams,
        rep_state_init: hk.MutableState,
        exp: TimeStep,
        targets: _Targets,
        predicted_states: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, hk.MutableState]:
        unroll_steps = self.config.unroll_steps
        alive_mask, alive_sum = trajectory_alive_mask(exp.terminal, unroll_steps + 1)
        rep_keys = jax.random.split(key, unroll_steps)

        unroll_lengths = alive_mask.sum(axis=1, keepdims=True)
        step_scale = jnp.ones_like(alive_mask)
        step_scale = step_scale.at[:, 1:].set(unroll_lengths)

        loss_pi = policy_cross_entropy(
            targets.policy_logits,
            targets.policy_targets,
            targets.action_mask,
            alive_mask,
            alive_sum,
            per_step_scale=step_scale,
        )
        loss_v = support_cross_entropy(
            targets.value_logits,
            targets.value_targets,
            self.agent.support,
            alive_mask,
            alive_sum,
            per_step_scale=step_scale,
        )
        loss_r = support_cross_entropy(
            targets.reward_logits,
            targets.reward_targets,
            self.agent.support,
            alive_mask,
            alive_sum,
            per_step_scale=step_scale,
        )
        loss_consistency, rep_state_final = self._consistency_loss(
            params,
            rep_state_init,
            exp,
            predicted_states,
            rep_keys,
            unroll_steps,
            alive_mask,
        )
        return loss_pi, loss_v, loss_r, loss_consistency, rep_state_final

    def _consistency_loss(
        self,
        params: ModelParams,
        rep_state_init: hk.MutableState,
        exp: TimeStep,
        predicted_states: jax.Array,
        rep_keys: jax.Array,
        unroll_steps: int,
        alive_mask: jax.Array,
    ) -> tuple[jax.Array, hk.MutableState]:
        obs_next = exp.obs[:, 1 : unroll_steps + 1]
        obs_next_t = jnp.swapaxes(obs_next, 0, 1)

        def rep_step(rep_state, inputs):
            obs_t, key = inputs
            latent, rep_state = self.agent.rep_net.apply(
                params.rep, rep_state, key, obs_t
            )
            return rep_state, latent

        rep_state_final, true_states = jax.lax.scan(
            rep_step, rep_state_init, (obs_next_t, rep_keys)
        )
        true_states = jnp.swapaxes(true_states, 0, 1)

        pred_states = predicted_states[:, :unroll_steps]
        diff = pred_states - jax.lax.stop_gradient(true_states)
        diff = jnp.square(diff)
        mse = diff.mean(axis=tuple(range(2, diff.ndim)))
        step_mask = alive_mask[:, :unroll_steps] * (
            1.0 - exp.terminal[:, :unroll_steps].astype(jnp.float32)
        )
        denom = step_mask.sum() + 1e-9
        loss_consistency = (mse * step_mask).sum() / denom
        return loss_consistency, rep_state_final

    def search(
        self,
        model: ModelState,
        key: chex.PRNGKey,
        env_states: chex.ArrayTree,
        deterministic: bool = True,
    ) -> PolicyOutput:
        obs = jax.vmap(self.adapter.obs_fn)(env_states)
        valid_actions = jax.vmap(self.adapter.action_mask_fn)(env_states)
        gumbel_scale = 0.0 if deterministic else self.config.gumbel_scale
        return self.agent.search(model, key, obs, valid_actions, gumbel_scale)

    def _train_from_batch(self, state: TrainState) -> tuple[TrainState, TrainStats]:
        key, subkey = jax.random.split(state.key)
        batch = self.buffer.sample(state.buffer_state, subkey)

        (loss, (net_state, loss_r, loss_v, loss_pi, loss_consistency)), grads = (
            jax.value_and_grad(self._loss_fn, argnums=0, has_aux=True)
        )(
            state.model.params,
            state.model.state,
            key,
            batch,
        )

        updates, opt_state = self.opt.update(grads, state.opt_state, state.model.params)
        params = optax.apply_updates(state.model.params, updates)
        chex.assert_trees_all_equal_shapes_and_dtypes(state.model.params, params)

        stats = TrainStats(
            loss=loss,
            policy_loss=loss_pi,
            value_loss=loss_v,
            reward_loss=loss_r,
            consistency_loss=loss_consistency,
        )

        return state._replace(
            key=key,
            model=ModelState(params=params, state=net_state),
            opt_state=opt_state,
        ), stats

    def _actor_step(self, state: TrainState) -> tuple[TrainState, jax.Array]:
        key, rollout_key = jax.random.split(state.key)
        rollout = collect_rollout_step(
            self.adapter,
            state.env_states,
            lambda search_key, env_states: self.search(
                model=state.model,
                key=search_key,
                env_states=env_states,
                deterministic=False,
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

    @functools.partial(jax.jit, static_argnums=(0,))
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
            loop_fn, state, None, self.config.eval_frequency
        )
        stats = jax.tree_util.tree_map(jnp.nanmean, stats_series)
        return state, stats

    @functools.partial(jax.jit, static_argnums=(0,))
    def evaluate(self, state: TrainState) -> EvalStats:
        return evaluate_rollout(
            self.adapter,
            self.config.actor_batch_size,
            self.config.max_eval_steps,
            state.key,
            lambda search_key, env_states: self.search(
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
            num_steps=num_steps,
            eval_frequency=self.config.eval_frequency,
            checkpoint_frequency=self.config.checkpoint_frequency,
            checkpoints_dir=checkpoints_dir,
            checkpointer=self.train_checkpointer,
            train_step=self.train_step,
            evaluate=self.evaluate,
            build_train_metrics=lambda stats: {
                "loss/total": stats.loss,
                "loss/policy": stats.policy_loss,
                "loss/value": stats.value_loss,
                "loss/reward": stats.reward_loss,
                "loss/consistency": stats.consistency_loss,
                "train/avg_return": stats.episode_return,
            },
            build_eval_metrics=lambda ev: {"eval/avg_return": ev.episode_return},
            run_config={
                "algorithm": "muzero",
                "agent_config": dataclasses.asdict(self.agent.config),
                "train_config": dataclasses.asdict(self.config),
            },
        )

    def save_checkpoint(self, state: TrainState, filename: str, directory: Path):
        self.train_checkpointer.save(directory / filename, state)

    def restore_checkpoint(self, filename: str, directory: Path):
        state = self.init(jax.random.PRNGKey(0))  # create dummy state
        state = self.train_checkpointer.restore(
            directory / filename, state, strict=True
        )
        return state
