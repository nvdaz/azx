# Azx: AlphaZero/MuZero in JAX

Azx is a JAX implementation of AlphaZero- and MuZero-style algorithms.


## Quick Start

Run an example end-to-end:

```bash
python examples/maze.py
```

Or the MuZero variant:

```bash
python examples/maze-mz.py
```

## Basic Usage

There are four main components to set up an Azx training run:

1. An environment adapter (`EnvironmentAdapter`)
2. A network definition
3. An agent (`AlphaZero` or `MuZero`)
4. A trainer (`AlphaZeroTrainer` or `MuZeroTrainer`)

### 1) Environment Adapter

The adapter wraps a Jumanji-compatible environments and provides standard interfaces for observations and action masks.

```python
from azx.core.env import EnvironmentAdapter

adapter = EnvironmentAdapter(
    env=env,
    obs_fn=flatten_observation,
    action_mask_fn=action_mask_fn,
)
```

Where:

- `obs_fn(state) -> jax.Array`: converts env state to a model observation.
- `action_mask_fn(state) -> jax.Array`: boolean mask of valid actions.

### 2) Networks

AlphaZero expects a single network:

```python
def policy_value(obs):
    return (policy_logits, value_logits)
```

MuZero expects three networks:

```python
def representation(obs):
    return latent

def dynamics(latent, action):
    return (next_latent, reward_logits)

def prediction(latent):
    return (policy_logits, value_logits)
```

Logits are over a discrete support for value/reward (see configs below).

### 3) Agents

The agent is what is trained. It implements the inference logic including MCTS.

AlphaZero:

```python
from azx.alphazero.agent import AlphaZero, Config as AZConfig

agent = AlphaZero(
    adapter=adapter,
    config=Config(...),
    network_fn=policy_value_net,
)
```

MuZero:

```python
from azx.muzero.agent import MuZero, Config as MZConfig

agent = MuZero(
    config=Config(...),
    representation_fn=representation,
    dynamics_fn=dynamics,
    prediction_fn=prediction,
)
```

### 4) Trainers

Trainers are responsible for optimizing the provided agent via reinforcement learning.

AlphaZero:

```python
from azx.alphazero.trainer import AlphaZeroTrainer, TrainConfig as AZTrain

trainer = AlphaZeroTrainer(
    agent=agent,
    config=TrainConfig(...),
    opt=optax.adamw(1e-4),
)
```

MuZero:

```python
from azx.muzero.trainer import MuZeroTrainer, TrainConfig as MZTrain

trainer = MuZeroTrainer(
    agent=agent,
    adapter=adapter,
    config=TrainConfig(...),
    opt=optax.adamw(1e-4),
)
```

Train:

```python
state = trainer.init(jax.random.PRNGKey(0))
state = trainer.learn(state, num_steps=100000, checkpoints_dir=Path("./checkpoints"))
```

## Configuration

### Agent configs (AlphaZero/MuZero)

- `discount`: discount factor for value targets.
- `num_simulations`: MCTS simulations per move.
- `use_mixed_value`, `value_scale`: MCTS q-transform settings.
- `support_min`, `support_max`, `support_eps`: discrete support for value/reward
  logits. All values use square root scaling.

### Trainer configs (AlphaZero)

- `actor_batch_size`: number of parallel envs for rollouts.
- `train_batch_size`: batch size sampled from the trajectory buffer.
- `n_step`: n-step return length.
- `unroll_steps`: number of steps used for loss computation.
- `eval_frequency`: number of actor steps between eval+log events.
- `max_eval_steps`: cap on stepsf or eval rollouts.
- `checkpoint_frequency`: save interval (in env steps).
- `gumbel_scale`: exploration temperature for MCTS.
- `max_length_buffer`, `min_length_buffer`: trajectory buffer size controls.
- `value_loss_weight`: weight for value loss.

### Trainer configs (MuZero)

All AlphaZero fields, plus:

- `consistency_loss_weight`: weight for representation consistency.

