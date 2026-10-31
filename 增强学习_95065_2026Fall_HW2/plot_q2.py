"""Plot HalfCheetah evaluation return and baseline loss by environment steps."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parent


def load_scalars(run_name, tag):
    matches = list((ROOT / "data").glob(f"q2_pg_{run_name}_HalfCheetah-v4_*"))
    if len(matches) != 1:
        raise ValueError(f"Expected one log for {run_name}; found {len(matches)}")
    event = EventAccumulator(str(matches[0]))
    event.Reload()
    steps = event.Scalars("Train_EnvstepsSoFar")
    values = event.Scalars(tag)
    if len(steps) != len(values) or any(a.step != b.step for a, b in zip(steps, values)):
        raise ValueError(f"Misaligned {tag} series in {matches[0]}")
    return [value.value for value in steps], [value.value for value in values]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    steps, returns = load_scalars("cheetah", "Eval_AverageReturn")
    axes[0].plot(steps, returns, label="No baseline", color="#d55e00", linewidth=1.6)
    steps, returns = load_scalars("cheetah_baseline", "Eval_AverageReturn")
    axes[0].plot(steps, returns, label="Baseline, 5 steps", color="#0072b2", linewidth=1.6)
    steps, returns = load_scalars("cheetah_baseline_bgs1", "Eval_AverageReturn")
    axes[0].plot(steps, returns, label="Baseline, 1 step", color="#cc79a7", linewidth=1.6)
    axes[0].axhline(300, color="#555555", linestyle="--", linewidth=1, label="Return 300")
    axes[0].set_ylabel("Evaluation average return")
    axes[0].set_title("Policy performance")
    axes[0].legend(loc="lower right", frameon=False)

    steps, losses = load_scalars("cheetah_baseline", "Baseline_Loss")
    axes[1].plot(steps, losses, label="5 steps", color="#009e73", linewidth=1.6)
    steps, losses = load_scalars("cheetah_baseline_bgs1", "Baseline_Loss")
    axes[1].plot(steps, losses, label="1 step", color="#cc79a7", linewidth=1.6)
    axes[1].set_ylabel("Baseline mean squared error")
    axes[1].set_title("Neural baseline loss")
    axes[1].legend(loc="upper right", frameon=False)

    for ax in axes:
        ax.set_xlim(0, 500_000)
        ax.set_xlabel("Training environment steps (Train_EnvstepsSoFar)")
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.19, wspace=0.26)
    output = ROOT / "image" / "halfcheetah_baseline.png"
    fig.savefig(output, dpi=240)
    print(output)


if __name__ == "__main__":
    main()
