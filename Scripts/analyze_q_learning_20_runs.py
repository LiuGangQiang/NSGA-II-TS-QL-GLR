"""汇总 MK06 原配置 20 次真实 Q 行为，核对 seed=42 图的代表性。"""

import csv
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WORK = Path(__file__).resolve().parent.parent
ROOT = WORK
SOURCE = WORK / "sensitivity_results" / "reward_original"
PROTOCOL = WORK / "sensitivity_results" / "protocol.json"
OUT = WORK / "supplementary_figures" / "Q学习_MK06_20次汇总"
STATES = [f"{phase}_{condition}" for phase in ("early", "mid", "late")
          for condition in ("converged", "balanced", "diverse")]
ACTIONS = [f"f{freq}_i{iterations}" for freq in (15, 20, 25)
           for iterations in (25, 30, 35)] + ["ts_none"]


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = json.loads(PROTOCOL.read_text(encoding="utf-8"))["run_seeds"]
    if len(seeds) != 20:
        raise RuntimeError("原配置协议不是 20 次运行")
    summaries, action_rows, state_rows, provenance = [], [], [], []
    matrix = np.zeros((len(seeds), len(ACTIONS)), dtype=int)
    total_states = Counter()
    for row_index, seed in enumerate(seeds):
        source = SOURCE / f"seed_{seed}.json"
        meta = json.loads(source.read_text(encoding="utf-8"))
        actions = meta["actions"]
        q_table = meta["q_table"]
        if len(actions) != 70 or len(q_table) != 90:
            raise RuntimeError(f"seed={seed} 的 Q 记录不完整")
        counts = Counter(item["action"] for item in actions[-20:])
        visits = Counter(item["state"] for item in actions)
        total_states.update(visits)
        for col_index, action in enumerate(ACTIONS):
            matrix[row_index, col_index] = counts[action]
            action_rows.append({"seed": seed, "action": action, "last20_count": counts[action]})
        for state in STATES:
            state_rows.append({"seed": seed, "state": state, "visits": visits[state]})
        summaries.append({
            "seed": seed, "decisions": len(actions), "ts_events": len(meta["ts_events"]),
            "visited_states": sum(visits[s] > 0 for s in STATES),
            "nonzero_q_entries": sum(abs(float(x["value"])) > 1e-12 for x in q_table),
            "distinct_actions_last20": sum(counts[a] > 0 for a in ACTIONS),
            "last_epsilon": float(actions[-1]["epsilon"]),
            "reward_clips": sum(bool(x["clipped"]) for x in actions),
        })
        provenance.append({"seed": seed, "path": source.relative_to(ROOT).as_posix(),
                           "sha256": sha256(source.read_bytes()).hexdigest()})

    write_csv(OUT / "逐次行为摘要.csv",
              ["seed", "decisions", "ts_events", "visited_states", "nonzero_q_entries",
               "distinct_actions_last20", "last_epsilon", "reward_clips"], summaries)
    write_csv(OUT / "末20次动作频数.csv", ["seed", "action", "last20_count"], action_rows)
    write_csv(OUT / "各状态访问次数.csv", ["seed", "state", "visits"], state_rows)
    (OUT / "来源.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=int(matrix.max()), aspect="auto")
    ax.set_xticks(range(len(ACTIONS)), ACTIONS, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(seeds)), [str(seed) for seed in seeds], fontsize=8)
    ax.set(xlabel="Action in final 20 Q decisions", ylabel="Run seed")
    fig.colorbar(image, ax=ax, label="Selection count", fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(OUT / "末20次动作频数热图.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    ax.bar(range(len(STATES)), [total_states[s] for s in STATES], color="#377eb8")
    ax.set_xticks(range(len(STATES)), STATES, rotation=35, ha="right", fontsize=8)
    ax.set(xlabel="Population state", ylabel="Visits across 20 runs")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "20次状态访问分布.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    distinct = [r["distinct_actions_last20"] for r in summaries]
    nonzero = [r["nonzero_q_entries"] for r in summaries]
    visits = [r["visited_states"] for r in summaries]
    events = [r["ts_events"] for r in summaries]
    clipped = [r["reward_clips"] for r in summaries]
    report = (
        "# MK06 原配置 20 次 Q 行为汇总\n\n"
        f"20 个种子均有 70 次动作决策，总计 {sum(events)} 次实际 TS 触发；每次 TS 触发数为 {min(events)}—{max(events)} 次。\n\n"
        f"每次访问状态数为 {min(visits)}—{max(visits)}/9，非零 Q 值为 {min(nonzero)}—{max(nonzero)}/90。"
        f"每次最后 20 个决策仍覆盖 {min(distinct)}—{max(distinct)}/10 个动作，中位数 {median(distinct):g}；"
        f"末次 ε 均为 {summaries[0]['last_epsilon']:.3f}。"
        "所以 seed=42 的广泛动作选择并非孤例；这些记录不支持宣称动作策略已经收敛为单一选择。"
        "Q 值非零说明发生了更新，本身不能证明学习改善了性能。\n\n"
        f"20 次奖励截断合计 {sum(clipped)} 次，发生于 {sum(v > 0 for v in clipped)} 个种子。"
        "每次原始日志的路径与 SHA-256 见 `来源.json`。\n"
    )
    (OUT / "说明.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
