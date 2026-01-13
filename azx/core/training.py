from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

import jax
from aim import Run
from orbax.checkpoint import StandardCheckpointer
from tqdm import tqdm

TState = TypeVar("TState")
TStats = TypeVar("TStats")
TEval = TypeVar("TEval")


def _format_metrics(title: str, metrics: Mapping[str, Any]) -> str:
    items = ", ".join(f"{key}={val:.3f}" for key, val in metrics.items())
    return f"{title}: {items}" if items else f"{title}: <empty>"


def learn_loop(
    *,
    state: TState,
    eval_frequency: int,
    checkpoint_frequency: int,
    num_steps: int,
    checkpoints_dir: Path,
    checkpointer: StandardCheckpointer,
    train_step: Callable[[TState], tuple[TState, TStats]],
    evaluate: Callable[[TState], TEval],
    build_train_metrics: Callable[[TStats], Mapping[str, Any]],
    build_eval_metrics: Callable[[TEval], Mapping[str, Any]],
    run_config: Mapping[str, Any],
) -> TState:
    run = Run()
    for key, value in run_config.items():
        run[key] = value

    checkpoints_dir = checkpoints_dir.resolve()
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    if any(checkpoints_dir.iterdir()):
        raise ValueError(f"Checkpoints directory {checkpoints_dir} is not empty.")

    time_step = 0

    pbar = tqdm(range(num_steps // eval_frequency), leave=True)
    for _ in pbar:
        state, stats = train_step(state)
        time_step += eval_frequency
        ev = evaluate(state)

        train_metrics = jax.device_get(build_train_metrics(stats))
        run.track(train_metrics, step=time_step, context={"subset": "train"})

        eval_metrics = jax.device_get(build_eval_metrics(ev))
        run.track(eval_metrics, step=time_step, context={"subset": "eval"})

        pbar.write(
            f"Step {time_step:06d} | "
            f"{_format_metrics('train', train_metrics)} | "
            f"{_format_metrics('eval', eval_metrics)}"
        )

        if time_step % checkpoint_frequency == 0:
            checkpointer.save(checkpoints_dir / f"checkpoint-{time_step}", state)

    checkpointer.save(checkpoints_dir / "checkpoint-final", state)
    checkpointer.wait_until_finished()

    return state
