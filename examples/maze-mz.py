import pathlib

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jumanji.environments.routing.maze.env import Maze
from jumanji.environments.routing.maze.generator import RandomGenerator
from jumanji.environments.routing.maze.types import State

from azx.muzero.trainer import MuZeroTrainer, TrainConfig

config = TrainConfig(
    discount=0.99,
    use_mixed_value=True,
    value_scale=1.0,
    batch_size=32,
    n_step=8,
    unroll_steps=4,
    avg_return_smoothing=0.99,
    num_simulations=50,
    eval_frequency=1000,
    max_eval_steps=1000,
    dirichlet_alpha=0.3,
    dirichlet_mix=0,
    checkpoint_frequency=100000,
    gumbel_scale=0.5,
    max_length_buffer=64,
    min_length_buffer=8,
)


def flatten_observation(obs: State) -> jnp.ndarray:
    rows, cols = obs.walls.shape

    # Normalize agent position and target position to [0,1]
    agent_pos = jnp.array(obs.agent_position, dtype=jnp.float32) / jnp.array(
        [rows - 1, cols - 1], dtype=jnp.float32
    )
    target_pos = jnp.array(obs.target_position, dtype=jnp.float32) / jnp.array(
        [rows - 1, cols - 1], dtype=jnp.float32
    )

    walls_flat = jnp.ravel(obs.walls).astype(jnp.float32)
    action_mask = jnp.array(obs.action_mask, dtype=jnp.float32)

    return jnp.concatenate([agent_pos, target_pos, walls_flat, action_mask])


class RepresentationModel(hk.Module):
    def __init__(self, latent_dim: int, name=None):
        super().__init__(name=name)
        self.latent_dim = latent_dim

    def __call__(self, x):
        x = x.astype(jnp.float32)  # (B, F)
        x = hk.nets.MLP([128, 128])(x)  # (B, 128)
        return hk.Linear(latent_dim)(x)  # (B, L)


class DynamicsModel(hk.Module):
    def __init__(self, latent_dim: int, action_dim: int, name=None):
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.action_dim = action_dim

    def __call__(self, latent, action):
        latent = latent.astype(jnp.float32)
        action = action.astype(jnp.int32)
        action_oh = jax.nn.one_hot(action, self.action_dim)
        x = jnp.concatenate([latent, action_oh], axis=-1)  # (B, L + A)
        x = hk.nets.MLP([128, 128])(x)  # (B, 128)

        next_latent = hk.Linear(self.latent_dim)(x)  # (B, L)
        reward = jnp.squeeze(hk.Linear(1)(x), -1)  # (B,)
        terminal = jnp.squeeze(hk.Linear(1)(x), -1)  # (B,)

        return next_latent, reward, terminal


class PredictionModel(hk.Module):
    def __init__(self, action_dim: int, name=None):
        super().__init__(name=name)
        self.action_dim = action_dim

    def __call__(self, x):
        x = hk.nets.MLP([128, 128])(x)  # (B, 128)
        pi_logits = hk.Linear(self.action_dim)(x)  # (B, A)
        value = jnp.squeeze(hk.Linear(1)(x), -1)  # (B)
        return pi_logits, value


env = Maze(RandomGenerator(5, 5))
action_dim = env.action_spec.num_values
latent_dim = 64


def action_mask_fn(state):
    return state.action_mask


trainer = MuZeroTrainer(
    env=env,
    config=config,
    representation_fn=lambda x: RepresentationModel(latent_dim)(x),
    dynamics_fn=lambda x1, x2: DynamicsModel(latent_dim, action_dim)(x1, x2),
    prediction_fn=lambda x: PredictionModel(action_dim)(x),
    obs_fn=flatten_observation,
    action_mask_fn=action_mask_fn,
    opt=optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(3e-4)),
)


key = jax.random.PRNGKey(0)
state = trainer.init(key)

checkpoints_dir = pathlib.Path("./checkpoints")
checkpoints_dir.mkdir(exist_ok=True)

state, returns, steps = trainer.learn(
    state=state,
    num_steps=100000,
    checkpoints_dir="./checkpoints",
)
