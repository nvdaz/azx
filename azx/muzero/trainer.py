import dataclasses
import functools
from pathlib import Path
from typing import Callable, Literal, NamedTuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import optax
import orbax.checkpoint as ocp
from jumanji.env import Environment

from .agent import Config, ModelNetState, ModelParams, ModelState, MuZero


@dataclasses.dataclass
class TrainConfig(Config):
    batch_size: int
    eval_frequency: int
    max_eval_steps: int
    avg_return_smoothing: float
    value_target: Literal["maxq", "nodev"]
    dirichlet_alpha: float
    dirichlet_mix: float
    checkpoint_frequency: int


class TrainState(NamedTuple):
    model: ModelState
    env_states: chex.Array
    opt_state: optax.OptState
    avg_return: chex.Array
    avg_loss: chex.Array
    avg_pi_loss: chex.Array
    avg_value_loss: chex.Array
    episode_return: chex.Array
    num_episodes: chex.Array
    key: chex.Array
    eval_episode_return: chex.ArrayTree
    eval_avg_return: chex.ArrayTree


class MuZeroTrainer(MuZero):
    def __init__(
        self,
        env: Environment,
        config: TrainConfig,
        representation_fn: Callable[[chex.Array], chex.Array],
        dynamics_fn: Callable[[chex.Array, chex.Array], chex.Array],
        prediction_fn: Callable[[chex.Array], tuple[chex.Array, chex.Array]],
        obs_fn: Callable[[chex.ArrayTree], chex.Array],
        action_mask_fn: Callable[[chex.ArrayTree], chex.Array],
        opt: optax.GradientTransformation,
    ):
        super().__init__(config, representation_fn, dynamics_fn, prediction_fn)
        self.env = env
        self.obs_fn = obs_fn
        self.opt = opt
        self.config = config
        self.action_mask_fn = action_mask_fn
        self.train_checkpointer = ocp.StandardCheckpointer()

    def init(self, key: chex.PRNGKey) -> TrainState:
        key, subkey = jax.random.split(key)
        state, _ = self.env.reset(subkey)

        obs = self.obs_fn(state)[None, ...]

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
        key: chex.Array,
        pi_target: chex.Array,
        value_target: chex.Array,
        reward_target: chex.Array,
        terminal_target: chex.Array,
        actions: chex.Array,
        obs: chex.Array,
    ) -> tuple[chex.Array, tuple[ModelNetState, chex.Array, chex.Array]]:
        key, subkey = jax.random.split(key)
        latent, rep_state = self.rep_net.apply(params.rep, net_state.rep, subkey, obs)
        key, subkey = jax.random.split(key)
        (_, rewards, terminals), dyn_state = self.dyn_net.apply(
            params.dyn, net_state.dyn, subkey, latent, actions
        )
        (pi_logits, value), pred_state = self.pred_net.apply(
            params.pred, net_state.pred, key, latent
        )

        pi_loss = optax.softmax_cross_entropy(pi_logits, pi_target).mean()
        value_loss = optax.l2_loss(value, value_target).mean()
        reward_loss = optax.l2_loss(rewards, reward_target).mean()
        terminal_loss = optax.l2_loss(terminals, terminal_target).mean()

        return pi_loss + value_loss + reward_loss + terminal_loss, (
            ModelNetState(rep=rep_state, dyn=dyn_state, pred=pred_state),
            jax.lax.stop_gradient(pi_loss),
            jax.lax.stop_gradient(value_loss),
        )

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
                t,
                lambda x: x[0],
                lambda x: x[1],
                (reset_state, env_state),
            )
        )(terminal, reset_states, env_states)

        state = state._replace(key=key, env_states=env_states)

        return state, reward, terminal

    def _muzero_search(self, state: TrainState) -> tuple[TrainState, mctx.PolicyOutput]:
        batch_obs = jax.vmap(self.obs_fn, in_axes=(0,))

        env_obs = batch_obs(state.env_states)
        key, subkey = jax.random.split(state.key)
        latent, _ = self.rep_net.apply(
            state.model.params.rep, state.model.state.rep, subkey, env_obs
        )

        key, subkey = jax.random.split(key)
        (pi_logits, value), _ = self.pred_net.apply(
            state.model.params.pred, state.model.state.pred, subkey, latent
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
            latent=latent,
            pi_logits=noisy_logits,
            value=value,
        )

        return state._replace(key=key), policy_output

    def _compute_gradients(
        self,
        model: ModelState,
        key: chex.Array,
        search_policy: chex.Array,
        search_value: chex.Array,
        obs: chex.Array,
        actions: chex.Array,
        reward_targets: chex.Array,
        terminal_targets: chex.Array,
    ) -> tuple[chex.Array, chex.Array]:
        (loss, (net_state, pi_loss, value_loss)), grads = jax.value_and_grad(
            self._loss_fn, argnums=0, has_aux=True
        )(
            model.params,
            model.state,
            key,
            search_policy,
            search_value,
            reward_targets,
            terminal_targets,
            actions,
            obs,
        )

        return (loss, (net_state, pi_loss, value_loss)), grads

    def _apply_updates(
        self, state: TrainState, grads: optax.Updates, net_state: hk.MutableState
    ) -> TrainState:
        updates, opt_state = self.opt.update(grads, state.opt_state, state.model.params)
        params = optax.apply_updates(state.model.params, updates)

        return state._replace(
            model=ModelState(params=params, state=net_state), opt_state=opt_state
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def train_step(self, state: TrainState) -> TrainState:
        batch_obs = jax.vmap(self.obs_fn, in_axes=(0,))

        def loop_fn(state: TrainState, _):
            state, policy_output = self._muzero_search(state)
            raw_counts = policy_output.action_weights  # shape: [batch, num_actions]
            count_sums = jnp.sum(raw_counts, axis=-1, keepdims=True)
            count_sums = jnp.maximum(count_sums, 1.0)
            pi_target = raw_counts / count_sums

            if self.config.value_target == "maxq":
                ROOT = policy_output.search_tree.ROOT_INDEX
                root_indices = jnp.full((self.config.batch_size,), ROOT)
                qvalues = policy_output.search_tree.qvalues(root_indices)
                v_target = qvalues[
                    jnp.arange(self.config.batch_size), policy_output.action
                ]
            elif self.config.value_target == "nodev":
                ROOT = policy_output.search_tree.ROOT_INDEX
                v_target = policy_output.search_tree.node_values[:, ROOT]
            else:
                raise ValueError("Unknown value target.")

            env_obs = batch_obs(state.env_states)
            state, reward, terminals = self._step_env(state, policy_output.action)

            key, subkey = jax.random.split(state.key)
            (loss, (net_state, pi_loss, value_loss)), grads = self._compute_gradients(
                state.model,
                subkey,
                pi_target,
                v_target,
                env_obs,
                policy_output.action,
                reward,
                terminals,
            )

            state = self._apply_updates(state, grads, net_state)

            new_return = state.episode_return + reward

            next_episode_return = jnp.where(terminals, 0, new_return)
            next_avg_return = jnp.where(
                terminals,
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
            next_num_episodes = state.num_episodes + terminals.astype(jnp.int32)

            state = state._replace(
                key=key,
                episode_return=next_episode_return,
                avg_return=next_avg_return,
                num_episodes=next_num_episodes,
                avg_pi_loss=next_avg_pi_loss,
                avg_value_loss=next_avg_value_loss,
            )

            return state, None

        state, _ = jax.lax.scan(loop_fn, state, None, self.config.eval_frequency)
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def evaluate(self, state: TrainState, max_steps: int) -> chex.Array:
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
        state = TrainState(
            env_states=None,  # type: ignore
            model=ModelState(
                params=ModelParams(rep=None, dyn=None, pred=None),  # type: ignore
                state=ModelNetState(rep=None, dyn=None, pred=None),  # type: ignore
            ),
            opt_state=None,  # type: ignore
            avg_return=None,  # type: ignore
            avg_loss=None,  # type: ignore
            avg_pi_loss=None,  # type: ignore
            avg_value_loss=None,  # type: ignore
            episode_return=None,  # type: ignore
            num_episodes=None,  # type: ignore
            eval_avg_return=None,  # type: ignore
            eval_episode_return=None,  # type: ignore
            key=None,  # type: ignore
        )  # type: ignore

        self.train_checkpointer.restore(directory / filename, state)
        return state
