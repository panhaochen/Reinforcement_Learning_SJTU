"""Plot five-seed InvertedPendulum returns against actual training steps."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parent
GROUPS = {
    "default": ("Default: batch 5000, lr 0.005", "#0072b2", 100),
    "tune_b500_lr01": ("Tuned: batch 500, lr 0.01", "#d55e00", 200),
}


def load_group(name, expected_count):
    runs = []
    for seed in range(1, 6):
        matches = list((ROOT / "data").glob(
            f"q2_pg_pendulum_{name}_s{seed}_InvertedPendulum-v4_*"
        ))
        if len(matches) != 1:
            raise ValueError(f"Expected one log for {name}, seed {seed}: {matches}")
        event = EventAccumulator(str(matches[0]))
        event.Reload()
        x_events = event.Scalars("Train_EnvstepsSoFar")
        y_events = event.Scalars("Eval_AverageReturn")
        if (len(x_events) != expected_count or len(y_events) != expected_count
                or any(a.step != b.step for a, b in zip(x_events, y_events))):
            raise ValueError(f"Incomplete or misaligned log: {matches[0]}")
        x = np.array([e.value for e in x_events], dtype=np.int64)
        y = np.array([e.value for e in y_events])
        runs.append((x, y))

    left = max(x[0] for x, _ in runs)
    right = min(x[-1] for x, _ in runs)
    grid = np.unique(np.concatenate([x[(x >= left) & (x <= right)] for x, _ in runs]))
    means = np.mean(
        [y[np.searchsorted(x, grid, side="right") - 1] for x, y in runs], axis=0
    )
    hits = np.flatnonzero(np.isclose(means, 1000))
    first_hit = int(grid[hits[0]]) if hits.size else None
    return grid, means, first_hit


def main():
    data = {}
    for name, (label, color, count) in GROUPS.items():
        data[name] = load_group(name, count)
        print(f"{name}: first five-seed mean 1000 at {data[name][2]:,} steps")

    fig, (zoom, full) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax in (zoom, full):
        for name, (label, color, _) in GROUPS.items():
            steps, returns, first_hit = data[name]
            ax.step(steps, returns, where="post", label=label, color=color, lw=1.6)
            if first_hit is not None:
                ax.scatter([first_hit], [1000], color=color, s=32, zorder=5)
        ax.axhline(1000, color="#555555", ls="--", lw=0.8)
        ax.set_ylim(-20, 1050)
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
        ax.set_xlabel("Training environment steps")
    zoom.set_xlim(0, 150000)
    zoom.set_title("Early learning (0–150k steps)")
    zoom.set_ylabel("Mean evaluation return (5 seeds)")
    full.set_xlim(0, 525000)
    full.set_title("Complete recorded runs")
    full.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    output = ROOT / "image" / "pendulum_tuning.png"
    fig.savefig(output, dpi=250)
    print(output)


if __name__ == "__main__":
    main()
