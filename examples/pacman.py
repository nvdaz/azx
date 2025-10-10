import pathlib
from typing import Callable

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax

from jumanji.environments.routing.pac_man.env import PacMan
from jumanji.environments.routing.pac_man.types import State

from azx.alphazero.trainer import AlphaZeroTrainer, TrainConfig


def make_observation_fn(initial_state: State):
    H, W = initial_state.grid.shape

    def get_observation(state: State) -> jnp.ndarray:
        obs = []

        walls = (state.grid == 1).astype(jnp.float32)
        obs.append(walls)

        pellet_grid = jnp.zeros((H, W), dtype=jnp.float32)
        pellet_grid = pellet_grid.at[
            state.pellet_locations[:, 1], state.pellet_locations[:, 0]
        ].set(1.0)
        obs.append(pellet_grid)

        power_up_grid = jnp.zeros((H, W), dtype=jnp.float32)
        power_up_grid = power_up_grid.at[
            state.power_up_locations[:, 1], state.power_up_locations[:, 0]
        ].set(1.0)
        obs.append(power_up_grid)

        player_grid = jnp.zeros((H, W), dtype=jnp.float32)
        player_grid = player_grid.at[
            state.player_locations.x, state.player_locations.y
        ].set(1.0)
        obs.append(player_grid)

        for ghost in state.ghost_locations:
            ghost_grid = jnp.zeros((H, W), dtype=jnp.float32)
            ghost_grid = ghost_grid.at[ghost[1], ghost[0]].set(1.0)
            obs.append(ghost_grid)

        frightened_layer = jnp.full(
            (H, W), state.frightened_state_time > 0, dtype=jnp.float32
        )
        obs.append(frightened_layer)

        ghost_eaten_layer = jnp.full(
            (H, W), jnp.any(state.ghost_eaten), dtype=jnp.float32
        )
        obs.append(ghost_eaten_layer)

        return jnp.stack(obs, axis=-1)  # (H, W, C)

    return get_observation

def get_action_mask(state: State) -> jnp.ndarray:
    return PacMan._compute_action_mask(None, state)

config = TrainConfig(
    discount=0.99,
    use_mixed_value=True,
    value_scale=1.0,
    value_target="maxq",
    batch_size=32,
    avg_return_smoothing=0.99,
    num_simulations=5,
    eval_frequency=100,
    dirichlet_alpha=0.3,
    dirichlet_mix=0.25,
    checkpoint_frequency=1000,
    gumbel_scale=0.5,
)


def make_cnn_policy_value(action_dim: int) -> Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    def net_fn(obs: jnp.ndarray):  # obs: (H, W, C) per sample (no batch dim)
        chex.assert_shape(obs, (31, 28, 10))
        x = obs.astype(jnp.float32)
        x = hk.Conv2D(32, kernel_shape=3, padding="SAME")(x); x = jax.nn.relu(x)
        x = hk.Conv2D(64, kernel_shape=3, padding="SAME")(x); x = jax.nn.relu(x)
        x = jnp.mean(x, axis=(0, 1))
        x = hk.Linear(128)(x); x = jax.nn.relu(x)
        pi_logits = hk.Linear(action_dim)(x)    # (action_dim,)
        value = hk.Linear(1)(x)                 # (1,)

        chex.assert_shape(pi_logits, (action_dim,))
        chex.assert_shape(value, (1,))

        return pi_logits, jnp.squeeze(value, -1)  # value -> ()
    return net_fn

def main(num_steps: int = 100_000):
    env = PacMan()
    action_dim = env.action_spec.num_values

    key = jax.random.PRNGKey(0)
    initial_state, _ = env.reset(key)

    obs_fn = make_observation_fn(initial_state)
    network_fn = make_cnn_policy_value(action_dim)

    trainer = AlphaZeroTrainer(
        env=env,
        config=config,
        network_fn=network_fn,
        obs_fn=obs_fn,
        action_mask_fn=get_action_mask,
        opt=optax.adamw(3e-4),
    )

    train_state = trainer.init(key)

    checkpoints_dir = pathlib.Path("./checkpoints_pacman")
    checkpoints_dir.mkdir(exist_ok=True)

    train_state, avg_returns, steps = trainer.learn(
        state=train_state,
        num_steps=num_steps,
        checkpoints_dir=str(checkpoints_dir),
    )

    print("Running one evaluation episode…")
    key, subkey = jax.random.split(key)
    env_state, ti = env.reset(subkey)
    done = False
    total_rew = 0.0
    t = 0
    traj = [env_state]

    while not done:
        action = trainer.predict(train_state, env_state)
        env_state, ti = env.step(env_state, action)
        done = ti.last()
        total_rew += float(ti.reward)
        t += 1
        traj.append(env_state)

    print(f"Episode length: {t}, total reward: {total_rew}")

if __name__ == "__main__":
    main(num_steps=50_000)
