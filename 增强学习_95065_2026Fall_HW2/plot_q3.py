"""Compare LunarLander GAE lambda runs by training environment steps."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parent
LAMBDAS = ("0", "0.95", "0.98", "0.99", "1")
COLORS = ("#d55e00", "#0072b2", "#009e73", "#cc79a7", "#e69f00")


def load_run(value):
    matches = list((ROOT / "data").glob(f"q2_pg_lunar_lander_lambda{value}_LunarLander-v2_*"))
    if len(matches) != 1:
        raise ValueError(f"Expected one log for lambda={value}; found {len(matches)}")
    event = EventAccumulator(str(matches[0]))
    event.Reload()
    x = event.Scalars("Train_EnvstepsSoFar")
    y = event.Scalars("Eval_AverageReturn")
    if len(x) != 300 or len(y) != 300 or any(a.step != b.step for a, b in zip(x, y)):
        raise ValueError(f"Incomplete or misaligned log: {matches[0]}")
    return np.array([v.value for v in x]), np.array([v.value for v in y])


def moving_average(values, radius=4):
    return np.array([
        values[max(0, i - radius):min(len(values), i + radius + 1)].mean()
        for i in range(len(values))
    ])


def main():
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for value, color in zip(LAMBDAS, COLORS):
        steps, returns = load_run(value)
        ax.plot(steps, returns, color=color, linewidth=0.7, alpha=0.16)
        ax.plot(steps, moving_average(returns), label=rf"$\lambda={value}$", color=color, linewidth=1.9)
    ax.axhline(180, color="#555555", linestyle="--", linewidth=1, label="Return 180")
    ax.set_xlabel("Training environment steps (Train_EnvstepsSoFar)")
    ax.set_ylabel("Evaluation average return")
    ax.set_title("LunarLander-v2: generalized advantage estimation")
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    ax.legend(ncol=3, loc="lower right", frameon=False)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.17)
    output = ROOT / "image" / "lunar_gae_lambda.png"
    fig.savefig(output, dpi=240)
    print(output)


if __name__ == "__main__":
    main()
