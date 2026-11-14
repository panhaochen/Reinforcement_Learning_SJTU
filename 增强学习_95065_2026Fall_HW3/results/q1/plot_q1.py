"""Make the figures and scalar table for HW3 Basic Q-Learning."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNS = {
    "CartPole lr=0.0003": "hw3_dqn_dqn_lr0003_CartPole-v1_s64_l2_d0.99_seed1_*",
    "CartPole baseline": "hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_*",
    "CartPole lr=0.003": "hw3_dqn_dqn_lr003_CartPole-v1_s64_l2_d0.99_seed1_*",
    "CartPole lr=0.05": "hw3_dqn_dqn_lr005_CartPole-v1_s64_l2_d0.99_seed1_*",
    "LunarLander seed 1": "hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_seed1_*",
    "LunarLander seed 2": "hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_seed2_*",
    "LunarLander seed 3": "hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_seed3_*",
    "LunarLander Double Q seed 1": "hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_doubleq_seed1_*",
    "LunarLander Double Q seed 2": "hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_doubleq_seed2_*",
    "LunarLander Double Q seed 3": "hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_doubleq_seed3_*",
    "MsPacman Double Q": "hw3_dqn_dqn_tuned_MsPacmanNoFrameskip-v0_d0.99_tu2000_lr0.0001_doubleq_clip10.0_seed1_*",
}
METRICS = ("eval_return", "train_return", "q_values", "critic_loss", "target_values")


def load_runs():
    runs = {}
    rows = []
    for label, pattern in RUNS.items():
        matches = sorted((ROOT / "data").glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one log for {label}, got {matches}")
        event = EventAccumulator(str(matches[0]))
        event.Reload()
        runs[label] = {}
        for metric in METRICS:
            values = event.Scalars(metric)
            runs[label][metric] = (
                np.array([point.step for point in values]),
                np.array([point.value for point in values]),
            )
            rows.extend((label, metric, point.step, point.value) for point in values)
    with (OUT / "scalars.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("experiment", "metric", "environment_steps", "value"))
        writer.writerows(rows)
    return runs


def save_figure(name):
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.png", dpi=180)
    plt.savefig(OUT / f"{name}.pdf")
    plt.close()


def main():
    runs = load_runs()

    plt.figure(figsize=(7, 4))
    steps, values = runs["CartPole baseline"]["eval_return"]
    plt.plot(steps, values, label="DQN, learning rate 0.001")
    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation return")
    plt.title("CartPole-v1: DQN")
    plt.grid(alpha=0.25)
    plt.legend()
    save_figure("cartpole_eval")

    plt.figure(figsize=(7, 4))
    for seed in (1, 2, 3):
        steps, values = runs[f"LunarLander seed {seed}"]["eval_return"]
        plt.plot(steps, values, label=f"Seed {seed}")
    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation return")
    plt.title("LunarLander-v2: vanilla DQN")
    plt.grid(alpha=0.25)
    plt.legend()
    save_figure("lunarlander_three_seeds")

    plt.figure(figsize=(7, 4))
    for seed in (1, 2, 3):
        for prefix, color in (("LunarLander", "tab:blue"),
                              ("LunarLander Double Q", "tab:red")):
            steps, values = runs[f"{prefix} seed {seed}"]["eval_return"]
            plt.plot(steps, values, color=color, alpha=0.45, linewidth=1,
                     label=("DQN" if prefix == "LunarLander" else "Double Q")
                     if seed == 1 else None)
    for prefix, color in (("LunarLander", "tab:blue"),
                          ("LunarLander Double Q", "tab:red")):
        curves = [runs[f"{prefix} seed {seed}"]["eval_return"][1]
                  for seed in (1, 2, 3)]
        steps = runs[f"{prefix} seed 1"]["eval_return"][0]
        plt.plot(steps, np.mean(curves, axis=0), color=color, linewidth=2.2)
    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation return")
    plt.title("LunarLander-v2: DQN vs Double Q")
    plt.grid(alpha=0.25)
    plt.legend()
    save_figure("lunarlander_doubleq_comparison")

    plt.figure(figsize=(7, 4))
    steps, values = runs["MsPacman Double Q"]["train_return"]
    plt.plot(steps, values, color="tab:blue", alpha=0.12, linewidth=0.6)
    if len(values) >= 100:
        cumulative = np.cumsum(np.insert(values, 0, 0.0))
        smooth = (cumulative[100:] - cumulative[:-100]) / 100
        plt.plot(steps[99:], smooth, color="tab:blue",
                 label="Train return (100-episode mean)")
    eval_steps, eval_values = runs["MsPacman Double Q"]["eval_return"]
    plt.plot(eval_steps, eval_values, color="tab:orange", linewidth=1.8,
             label="Eval return (10-episode mean)")
    plt.xlabel("Environment steps")
    plt.ylabel("Return")
    plt.title("MsPacman: train and evaluation return")
    plt.grid(alpha=0.25)
    plt.legend()
    save_figure("mspacman_train_eval")

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for label in ("CartPole baseline", "CartPole lr=0.05"):
        for axis, metric in zip(axes, ("q_values", "critic_loss")):
            steps, values = runs[label][metric]
            axis.plot(steps, values, label=label)
            axis.set_ylabel("Mean predicted Q" if metric == "q_values" else "Critic MSE")
            axis.grid(alpha=0.25)
    axes[1].set_xlabel("Environment steps")
    axes[1].set_yscale("symlog", linthresh=1)
    axes[0].legend()
    fig.suptitle("CartPole-v1: learning rate comparison")
    save_figure("cartpole_learning_rate")

    plt.figure(figsize=(7, 4))
    for label in (
        "CartPole lr=0.0003",
        "CartPole baseline",
        "CartPole lr=0.003",
        "CartPole lr=0.05",
    ):
        steps, values = runs[label]["eval_return"]
        plt.plot(steps, values, label=label.replace("CartPole baseline", "CartPole lr=0.001"))
    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation return")
    plt.title("CartPole-v1: learning rate sensitivity")
    plt.grid(alpha=0.25)
    plt.legend()
    save_figure("cartpole_four_learning_rates")

    for label, metrics in runs.items():
        steps, values = metrics["eval_return"]
        print(f"{label}: final eval={values[-1]:.1f}, peak eval={values.max():.1f}, steps={steps[-1]}")


if __name__ == "__main__":
    main()
