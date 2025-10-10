import pathlib

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jumanji.environments.routing.maze.env import Maze
from jumanji.environments.routing.maze.generator import Generator, RandomGenerator
from jumanji.environments.routing.maze.types import Position, State

from azx.muzero.trainer import MuZeroTrainer, TrainConfig


def flatten_observation(obs: State) -> jnp.ndarray:
    agent_pos = jnp.array(obs.agent_position)
    target_pos = jnp.array(obs.target_position)

    walls_flat = jnp.ravel(obs.walls)
    action_mask = obs.action_mask
    step = jnp.array([obs.step_count])

    return jnp.concatenate(
        [agent_pos, target_pos, walls_flat, action_mask, step]
    ).astype(jnp.float32)


def make_representation_fn(latent_dim: int):
    def representation_fn(obs: jnp.ndarray) -> jnp.ndarray:
        x = obs
        for _ in range(3):
            x = hk.Linear(128)(x)
            x = jax.nn.relu(x)
        return hk.Linear(latent_dim)(x)

    return representation_fn


def make_dynamics_fn(latent_dim: int, action_dim: int):
    def dynamics_fn(
        latent: jnp.ndarray, action: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        a_oh = jax.nn.one_hot(action, action_dim)[0]
        x = jnp.concatenate([latent, a_oh], axis=-1)
        for _ in range(3):
            x = hk.Linear(32)(x)
            x = jax.nn.relu(x)
        next_latent = hk.Linear(latent_dim)(x)
        reward = jnp.squeeze(hk.Linear(1)(x), -1)
        terminal = jnp.squeeze(hk.Linear(1)(x), -1)
        return next_latent, reward, terminal

    return dynamics_fn


def make_prediction_fn(action_dim: int):
    def prediction_fn(latent: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = latent
        for _ in range(3):
            x = hk.Linear(32)(x)
            x = jax.nn.relu(x)
        pi_logits = hk.Linear(action_dim)(x)
        value = jnp.squeeze(hk.Linear(1)(x), -1)
        return pi_logits, value

    return prediction_fn


class ToyGenerator(Generator):
    """Generate a hardcoded 5x5 toy maze."""

    def __init__(self) -> None:
        super(ToyGenerator, self).__init__(num_rows=5, num_cols=5)

    def __call__(self, key: chex.PRNGKey) -> State:
        walls = jnp.ones((self.num_rows, self.num_cols), bool)
        walls = walls.at[0, :].set((False, True, True, False, False))
        walls = walls.at[1, :].set((False, True, True, True, True))
        walls = walls.at[2, :].set((False, True, False, False, False))
        walls = walls.at[3, :].set((False, True, True, True, True))
        walls = walls.at[4, :].set((False, False, False, False, False))

        agent_position = Position(row=0, col=0)
        target_position = Position(row=4, col=1)

        # Build the state.
        return State(
            agent_position=agent_position,
            target_position=target_position,
            walls=walls,
            action_mask=None,  # Action mask will be computed by the environment.
            key=key,
            step_count=jnp.array(0, jnp.int32),
        )


if __name__ == "__main__":
    # --- environment setup ---
    # env = Maze(RandomGenerator(5, 5))
    env = Maze(ToyGenerator())
    action_dim = env.action_spec.num_values

    # --- config ---
    key = jax.random.PRNGKey(0)
    config = TrainConfig(
        discount=0.99,
        num_simulations=20,
        use_mixed_value=False,
        value_scale=1.0,
        batch_size=32,
        eval_frequency=100,
        avg_return_smoothing=0.99,
        dirichlet_alpha=0.3,
        dirichlet_mix=0.25,
        checkpoint_frequency=100,
        gumbel_scale=0.0,
        value_target="maxq",
    )

    latent_dim = 64

    rep_fn = make_representation_fn(latent_dim)
    dyn_fn = make_dynamics_fn(latent_dim, action_dim)
    pred_fn = make_prediction_fn(action_dim)

    # --- trainer init ---
    trainer = MuZeroTrainer(
        env=env,
        config=config,
        representation_fn=rep_fn,
        dynamics_fn=dyn_fn,
        prediction_fn=pred_fn,
        obs_fn=flatten_observation,
        opt=optax.adamw(1e-3),
    )

    state = trainer.init(key)

    # --- run learning ---
    checkpoints_dir = pathlib.Path("./muzero_checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)

    # with jax.disable_jit():
    final_state = trainer.learn(
        state=state,
        num_steps=100_000,
        checkpoints_dir=str(checkpoints_dir),
    )

    print("MuZero training complete!")
