"""Plot the required SAC evaluation curves from local TensorBoard logs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.rcParams["font.family"] = "Songti SC"
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent


def load(prefix, tag="eval_return"):
    candidates = []
    for directory in DATA.glob(f"hw3_sac_{prefix}*"):
        events = EventAccumulator(str(directory), size_guidance={"scalars": 0})
        events.Reload()
        if tag in events.Tags()["scalars"]:
            values = events.Scalars(tag)
            candidates.append((values[-1].step, directory, values))
    if not candidates:
        raise FileNotFoundError(f"No {tag} data for {prefix}")
    _, directory, values = max(candidates, key=lambda item: item[0])
    return directory, [(v.step, v.value) for v in values]


def plot_series(series, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name, points in series.items():
        ax.plot([x for x, _ in points], [y for _, y in points], label=name, linewidth=1.7)
    ax.set(title=title, xlabel="环境步数", ylabel=ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / filename)
    plt.close(fig)


def main():
    configs = {
        "REINFORCE-1": "reinforce1_HalfCheetah-v4",
        "REINFORCE-10": "reinforce10_HalfCheetah-v4",
        "重参数化": "reparametrize_HalfCheetah-v4_reparametrize_s128_l3",
    }
    summary = {}
    curves = {}
    for name, prefix in configs.items():
        directory, points = load(prefix)
        curves[name] = points
        summary[name] = {
            "log": str(directory.relative_to(ROOT)),
            "last_step": points[-1][0],
            "last_eval_return": points[-1][1],
            "best_eval_return": max(value for _, value in points),
        }
    plot_series(curves, "HalfCheetah-v4：策略梯度估计器", "平均评估回报", "halfcheetah_eval.pdf")
    plot_series(
        {name: curves[name] for name in ("REINFORCE-1", "REINFORCE-10")},
        "HalfCheetah-v4：REINFORCE 采样数",
        "平均评估回报",
        "halfcheetah_reinforce.pdf",
    )

    directory, points = load("sac_humanoid_Humanoid-v4")
    summary["Humanoid"] = {
        "log": str(directory.relative_to(ROOT)),
        "last_step": points[-1][0],
        "last_eval_return": points[-1][1],
        "best_eval_return": max(value for _, value in points),
    }
    plot_series({"SAC（截断双 Q）": points}, "Humanoid-v4：SAC", "平均评估回报", "humanoid_eval.pdf")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
