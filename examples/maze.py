import pathlib

import haiku as hk
import chex
import jax
import jax.numpy as jnp
import optax
from jumanji.environments.routing.maze.env import Maze
from jumanji.environments.routing.maze.generator import RandomGenerator
from jumanji.environments.routing.maze.types import State

from azx.alphazero.trainer import AlphaZeroTrainer, TrainConfig


config = TrainConfig(
    discount=0.99,
    use_mixed_value=True,
    value_scale=1.0,
    value_target="maxq",
    batch_size=32,
    avg_return_smoothing=0.99,
    num_simulations=5,
    eval_frequency=200,
    dirichlet_alpha=0.3,
    dirichlet_mix=0,
    checkpoint_frequency=1000,
    gumbel_scale=0.5,
)

ACTIVATIONS = {
    "relu": jax.nn.relu,
}


def make_network_fn(action_dim: int):
    def init():
        def net_fn(obs: jnp.ndarray):
            chex.assert_shape(obs, (None, None))
            # dense layer
            x = hk.Linear(64)(obs)
            # apply activation function
            x = jax.nn.relu(x)
            x = hk.Linear(64)(x)
            x = jax.nn.relu(x)
            pi_logits = hk.Linear(action_dim)(x)
            value = hk.Linear(1)(x)
            return pi_logits, jnp.squeeze(value, -1)

        return net_fn

    return init


class MLPBackbone(hk.Module):
    def __init__(self, num_actions: int, name=None):
        super().__init__(name=name)
        self.num_actions = num_actions

    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = hk.nets.MLP([128, 128])(x)  # [B, F] -> [B, 128]
        pi_logits = hk.Linear(self.num_actions)(x)  # [B, A]
        value = hk.Linear(1)(x)  # [B, 1]
        value = jnp.squeeze(value, -1)  # [B]
        return pi_logits, value


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


env = Maze(RandomGenerator(5, 5))
action_dim = env.action_spec.num_values


def action_mask_fn(state):
    return state.action_mask


trainer = AlphaZeroTrainer(
    env=env,
    config=config,
    network_fn=lambda obs: MLPBackbone(action_dim)(obs),
    action_mask_fn=action_mask_fn,
    obs_fn=flatten_observation,
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
