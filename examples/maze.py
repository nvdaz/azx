import pathlib

import haiku as hk
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
    batch_size=32,
    n_step=8,
    unroll_steps=4,
    avg_return_smoothing=0.99,
    num_simulations=200,
    eval_frequency=100,
    max_eval_steps=100,
    dirichlet_alpha=0.3,
    dirichlet_mix=0.25,
    checkpoint_frequency=100000,
    gumbel_scale=0.5,
    max_length_buffer=64,
    min_length_buffer=24,
)


class MLP(hk.Module):
    def __init__(self, num_actions: int, name=None):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.act = jax.nn.silu
        self.head_init = hk.initializers.VarianceScaling(0.01)

    def __call__(self, x):
        x = x.astype(jnp.float32)
        # three blocks with LayerNorm for stable scales
        for width in (256, 256, 256):
            x = hk.Linear(width)(x)
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            x = self.act(x)

        # policy head (logits)
        pi_logits = hk.Linear(self.num_actions, w_init=self.head_init)(x)  # [B, A]

        v = hk.Linear(128)(x)
        v = self.act(v)
        v = hk.Linear(1, w_init=self.head_init)(v)
        value = jnp.tanh(v[..., 0])

        return pi_logits, value


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


env = Maze(RandomGenerator(5, 5))
action_dim = env.action_spec.num_values


def action_mask_fn(state):
    return state.action_mask


trainer = AlphaZeroTrainer(
    env=env,
    config=config,
    network_fn=lambda obs: MLP(action_dim)(obs),
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
