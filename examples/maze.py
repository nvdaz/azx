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


def make_network_fn(action_dim: int):
    def init():
        def net_fn(obs: jnp.ndarray):
            chex.assert_shape(obs, (None, None))
            x = hk.Linear(64)(obs)
            x = jax.nn.relu(x)
            x = hk.Linear(64)(x)
            x = jax.nn.relu(x)
            pi_logits = hk.Linear(action_dim)(x)
            value = hk.Linear(1)(x)
            return pi_logits, jnp.squeeze(value, -1)

        return net_fn
    return init


def flatten_observation(obs: State) -> jnp.ndarray:
    # Derive maze dimensions from the walls array
    rows, cols = obs.walls.shape  # each is an integer

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
    network_fn=make_network_fn(action_dim),
    action_mask_fn=action_mask_fn,
    obs_fn=flatten_observation,
    opt=optax.adamw(3e-4),
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
