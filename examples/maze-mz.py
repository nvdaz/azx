from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jumanji.environments.routing.maze.env import Maze
from jumanji.environments.routing.maze.generator import RandomGenerator
from jumanji.environments.routing.maze.types import State

from azx.core.env import EnvironmentAdapter
from azx.muzero.agent import Config, MuZero
from azx.muzero.trainer import MuZeroTrainer, TrainConfig

config = Config(
    discount=0.99,
    num_simulations=5,
    use_mixed_value=True,
    value_scale=1.0,
    support_min=-1,
    support_max=1,
    support_eps=0.001,
)

train_config = TrainConfig(
    actor_batch_size=128,
    train_batch_size=64,
    n_step=5,
    unroll_steps=4,
    eval_frequency=50,
    max_eval_steps=100,
    checkpoint_frequency=100000,
    gumbel_scale=0.5,
    max_length_buffer=1024,
    min_length_buffer=64,
    consistency_loss_weight=1.0,
    value_loss_weight=0.25,
)


def flatten_observation(obs: State) -> jnp.ndarray:
    rows, cols = obs.walls.shape

    agent = jnp.array(obs.agent_position, dtype=jnp.float32)
    goal = jnp.array(obs.target_position, dtype=jnp.float32)
    norm = jnp.array([rows - 1, cols - 1], dtype=jnp.float32)
    agent_n = agent / norm
    goal_n = goal / norm

    delta = (goal - agent) / norm
    manhattan = jnp.abs(goal - agent).sum() / (rows + cols - 2 + 1e-6)

    walls_flat = jnp.ravel(obs.walls).astype(jnp.float32)

    feats = [
        agent_n,
        goal_n,
        delta,
        jnp.array([manhattan]),
        walls_flat,
    ]
    return jnp.concatenate(feats, axis=0)


class RepresentationModel(hk.Module):
    def __init__(self, latent_dim: int, name=None):
        super().__init__(name=name)
        self.latent_dim = latent_dim

    def __call__(self, x):
        x = x.astype(jnp.float32)
        for width in (128, 128):
            x = hk.Linear(width)(x)
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            x = jax.nn.silu(x)

        return hk.Linear(self.latent_dim)(x)  # (B, L)


class DynamicsModel(hk.Module):
    def __init__(self, latent_dim: int, action_dim: int, name=None):
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.action_dim = action_dim

    def __call__(self, latent, action):
        latent = latent.astype(jnp.float32)
        action = action.astype(jnp.int32)
        if latent.ndim == 1:
            action = jnp.squeeze(action, axis=0)
        action_oh = jax.nn.one_hot(action, self.action_dim)
        x = jnp.concatenate([latent, action_oh], axis=-1)  # (B, L + A)
        for width in (128, 128):
            x = hk.Linear(width)(x)
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            x = jax.nn.silu(x)

        next_latent = hk.Linear(64)(x)
        next_latent = jax.nn.silu(next_latent)
        next_latent = hk.Linear(self.latent_dim)(next_latent)  # (B, L)

        reward = hk.Linear(64)(x)
        reward = jax.nn.silu(reward)
        reward = hk.Linear(3)(reward)

        return next_latent, reward


class PredictionModel(hk.Module):
    def __init__(self, action_dim: int, name=None):
        super().__init__(name=name)
        self.action_dim = action_dim

    def __call__(self, x):
        for width in (64, 64):
            x = hk.Linear(width)(x)
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            x = jax.nn.silu(x)

        v = hk.Linear(128)(x)
        value = hk.Linear(3)(v)

        pi_logits = hk.Linear(128)(x)
        pi_logits = hk.Linear(self.action_dim)(pi_logits)  # (B, A)
        return pi_logits, value


env = Maze(RandomGenerator(5, 5))
action_dim = env.action_spec.num_values
latent_dim = 64


def action_mask_fn(state):
    return state.action_mask


adapter: EnvironmentAdapter[State] = EnvironmentAdapter(
    env=env,
    obs_fn=flatten_observation,
    action_mask_fn=action_mask_fn,
)

agent = MuZero(
    config=config,
    representation_fn=lambda x: RepresentationModel(latent_dim)(x),
    dynamics_fn=lambda x1, x2: DynamicsModel(latent_dim, action_dim)(x1, x2),
    prediction_fn=lambda x: PredictionModel(action_dim)(x),
)


trainer = MuZeroTrainer(
    agent=agent,
    adapter=adapter,
    config=train_config,
    opt=optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-4)),
)


key = jax.random.PRNGKey(0)
state = trainer.init(key)


state = trainer.learn(
    state=state,
    num_steps=100000,
    checkpoints_dir=Path("./checkpoints"),
)
