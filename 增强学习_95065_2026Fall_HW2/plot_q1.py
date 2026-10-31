"""Plot the four CartPole runs against training environment steps."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parent
RUNS = (
    ("cartpole_vanilla", "Vanilla", "#d55e00"),
    ("cartpole_rtg", "Reward-to-go", "#0072b2"),
    ("cartpole_rtg_na", "Reward-to-go + norm.", "#009e73"),
    ("cartpole_rtg_na_lb", "Reward-to-go + norm., batch 4000", "#cc79a7"),
)


def load_run(name):
    matches = list((ROOT / "data").glob(f"q2_pg_{name}_CartPole-v0_*"))
    if len(matches) != 1:
        raise ValueError(f"Expected one log for {name}; found {len(matches)}")
    event = EventAccumulator(str(matches[0]))
    event.Reload()
    steps = event.Scalars("Train_EnvstepsSoFar")
    returns = event.Scalars("Eval_AverageReturn")
    if len(steps) != len(returns) or any(a.step != b.step for a, b in zip(steps, returns)):
        raise ValueError(f"Misaligned scalars in {matches[0]}")
    return [value.value for value in steps], [value.value for value in returns]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for name, label, color in RUNS:
        steps, returns = load_run(name)
        for ax in axes:
            ax.plot(steps, returns, label=label, color=color, linewidth=1.6)

    for ax, title, xmax in zip(
        axes, ("First 110k steps", "Full training"), (110_000, 410_000)
    ):
        ax.set_title(title)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 205)
        ax.set_xlabel("Training environment steps (Train_EnvstepsSoFar)")
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    axes[0].set_ylabel("Evaluation average return")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.07, right=0.98, wspace=0.11)
    output = ROOT / "image" / "cartpole_learning_curves.png"
    fig.savefig(output, dpi=240)
    print(output)


if __name__ == "__main__":
    main()
