"""Summarize the bonus runs, preserving exact log provenance and selection rule."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent))
from plot_sac import ROOT, load, plot_series


def read_events(directory, tag):
    events = EventAccumulator(str(directory), size_guidance={"scalars": 0})
    events.Reload()
    return [(e.step, e.value) for e in events.Scalars(tag)]


def main():
    prefixes = {"Single Q": "sac_hopper_singlecritic_Hopper-v4",
                "Double Q": "sac_hopper_doubleq_Hopper-v4",
                "Clipped double Q": "sac_hopper_clipq_Hopper-v4"}
    summaries, returns, qs = {}, {}, {}
    for name, prefix in prefixes.items():
        directory, points = load(prefix)
        qpoints = read_events(directory, "q_values")
        returns[name], qs[name] = points, qpoints
        summaries[name] = dict(log=str(directory.relative_to(ROOT)), last_step=points[-1][0],
                               last_return=points[-1][1], best_return=max(v for _, v in points),
                               last5_return_mean=float(np.mean([v for _, v in points[-5:]])),
                               last_q=qpoints[-1][1],
                               last5_q_mean=float(np.mean([v for _, v in qpoints[-5:]])))
    plot_series(returns, "Hopper-v4：目标估值策略", "平均评估回报", "bonus_hopper_return.pdf")
    plot_series(qs, "Hopper-v4：回放批次中的 Q 估值", "Q 值均值", "bonus_hopper_q.pdf")
    best = max(summaries, key=lambda name: summaries[name]["last5_return_mean"])
    # Selection uses late training performance, never the single largest peak.
    result = dict(selection_rule="Mean of the last five evaluation returns (75k--95k steps)",
                  best=best, hopper=summaries)
    pg_path = OUT / "humanoid_pg.csv"
    sac_log = OUT / "selected_humanoid_log.txt"
    if pg_path.exists() and sac_log.exists():
        pg = [(int(r["step"]), float(r["eval_return"])) for r in csv.DictReader(pg_path.open())]
        directory = ROOT / sac_log.read_text().strip()
        sac = read_events(directory, "eval_return")
        plot_series({f"SAC（{best}）": sac, "On-policy PG + GAE": pg},
                    "Humanoid-v4：离策略与同策略学习", "平均评估回报", "bonus_humanoid_comparison.pdf")
        def describe(points):
            x, y = np.array(points).T
            shared = min(490000, int(x[-1]))
            x2 = np.unique(np.r_[0, x[x <= shared], shared])
            y2 = np.interp(x2, x, y)
            return dict(last_step=points[-1][0], last_return=points[-1][1],
                        best_return=float(y.max()), mean_return_to_490k=float(np.trapz(y2, x2) / shared),
                        return_at_490k_linear=float(np.interp(490000, x, y)),
                        first_eval_at_least_500=next((int(s) for s, v in points if v >= 500), None))
        result["humanoid"] = dict(sac_log=str(directory.relative_to(ROOT)),
                                   sac=describe(sac), pg=describe(pg))
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
