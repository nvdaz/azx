from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jumanji.environments.routing.maze.env import Maze
from jumanji.environments.routing.maze.generator import RandomGenerator
from jumanji.environments.routing.maze.types import State

from azx.alphazero.agent import AlphaZero, Config
from azx.alphazero.trainer import AlphaZeroTrainer, TrainConfig
from azx.core.env import EnvironmentAdapter

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
    value_loss_weight=0.5,
)


class MLP(hk.Module):
    def __init__(self, num_actions: int, name=None):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.act = jax.nn.silu
        self.head_init = hk.initializers.VarianceScaling(0.01)

    def __call__(self, x):
        x = x.astype(jnp.float32)
        for width in (128, 128, 128):
            x = hk.Linear(width)(x)
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            x = self.act(x)

        # policy head (logits)
        pi_logits = hk.Linear(self.num_actions, w_init=self.head_init)(x)  # (B, A)

        v = hk.Linear(128)(x)
        v = self.act(v)
        v = hk.Linear(3)(v)

        return pi_logits, v


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


adapter: EnvironmentAdapter[State] = EnvironmentAdapter(
    env=env,
    obs_fn=flatten_observation,
    action_mask_fn=action_mask_fn,
)

agent = AlphaZero(
    adapter=adapter,
    config=config,
    network_fn=lambda obs: MLP(action_dim)(obs),
)

trainer = AlphaZeroTrainer(
    agent=agent,
    config=train_config,
    opt=optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-4)),
)

key = jax.random.PRNGKey(0)
state = trainer.init(key)

state = trainer.learn(
    state=state, num_steps=100000, checkpoints_dir=Path("./checkpoints")
)
