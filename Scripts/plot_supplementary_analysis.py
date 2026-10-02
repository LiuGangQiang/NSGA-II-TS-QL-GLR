"""从修回版已保存结果生成 Q 行为、二维 Pareto 投影与清晰甘特图。"""

import csv
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from sensitivity_algorithms import revision


WORK = Path(__file__).resolve().parent.parent
ROOT = WORK
OUT = WORK / "supplementary_figures"
SEED_Q = 42
SEED_PARETO = 102  # 20 次种子序列的第 13 次，与原图 4-5 的运行编号对应。
CASES = ("MK02", "MK06", "MK09")
ALGS = ("NSGA2", "NSGA2_TS", "NSGA2_TS_QL", "NSGA2_TS_QL_GLR")
LABELS = {
    "NSGA2": "NSGA-II", "NSGA2_TS": "NSGA-II-TS",
    "NSGA2_TS_QL": "NSGA-II-TS-QL", "NSGA2_TS_QL_GLR": "NSGA-II-TS-QL-GLR",
}
COLORS = {
    "NSGA2": "#555555", "NSGA2_TS": "#377eb8",
    "NSGA2_TS_QL": "#e69f00", "NSGA2_TS_QL_GLR": "#009e73",
}
MARKERS = {"NSGA2": "o", "NSGA2_TS": "s", "NSGA2_TS_QL": "^", "NSGA2_TS_QL_GLR": "D"}
STATES = [f"{phase}_{condition}" for phase in ("early", "mid", "late")
          for condition in ("converged", "balanced", "diverse")]
ACTIONS = [f"f{freq}_i{iterations}" for freq in (15, 20, 25)
           for iterations in (25, 30, 35)] + ["ts_none"]


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def write_csv(path, fields, records):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def result_path(case, alg, seed):
    if alg == "NSGA2_TS":
        return WORK / "固定TS修正结果" / case / f"seed_{seed}.npz"
    if case == "MK06" and alg == "NSGA2_TS_QL_GLR":
        return WORK / "sensitivity_results" / "reward_original" / f"seed_{seed}.npz"
    return WORK / "主实验正式结果" / case / alg / f"seed_{seed}.npz"


def q_figures():
    out = OUT / "Q学习_MK06_seed42"
    out.mkdir(parents=True, exist_ok=True)
    source = WORK / "sensitivity_results" / "reward_original" / f"seed_{SEED_Q}.json"
    meta = json.loads(source.read_text(encoding="utf-8"))
    q = {(item["state"], item["action"]): float(item["value"]) for item in meta["q_table"]}
    actions = meta["actions"]
    if len(q) != len(STATES) * len(ACTIONS) or not actions:
        raise RuntimeError("seed=42 的 Q 表或动作日志不完整")
    counts = {(state, action): 0 for state in STATES for action in ACTIONS}
    for item in actions:
        counts[item["state"], item["action"]] += 1
    visits = {state: sum(counts[state, action] for action in ACTIONS) for state in STATES}
    matrix = np.array([[q[state, action] for action in ACTIONS] for state in STATES])
    bound = max(1e-9, np.max(np.abs(matrix)))

    fig, ax = plt.subplots(figsize=(11.2, 5.7))
    im = ax.imshow(matrix, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), aspect="auto")
    ax.set_xticks(range(len(ACTIONS)), ACTIONS, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(STATES)), [f"{state} (n={visits[state]})" for state in STATES], fontsize=8)
    ax.set_xlabel("Selected TS action")
    ax.set_ylabel("Population state")
    ax.set_title(f"MK06 Q-table, seed {SEED_Q}  (dot = observed state-action pair)", fontsize=11)
    for row, state in enumerate(STATES):
        for col, action in enumerate(ACTIONS):
            if counts[state, action]:
                ax.text(col, row, "•", ha="center", va="center", fontsize=12,
                        color="white" if abs(matrix[row, col]) > bound * 0.55 else "black")
    for row, state in enumerate(STATES):
        if visits[state] == 0:
            ax.axhspan(row - 0.5, row + 0.5, color="#bbbbbb", alpha=0.3)
    fig.colorbar(im, ax=ax, label="Q value", fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(out / "Q表热图.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    state_colors = {state: plt.get_cmap("tab10")(i) for i, state in enumerate(STATES)}
    generations = np.array([item["generation"] for item in actions])
    state_idx = np.array([STATES.index(item["state"]) for item in actions])
    action_idx = np.array([ACTIONS.index(item["action"]) for item in actions])
    colors = [state_colors[item["state"]] for item in actions]
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.25]})
    axes[0].step(generations, state_idx, where="post", color="#999999", linewidth=0.8, alpha=0.75)
    axes[0].scatter(generations, state_idx, c=colors, s=29, edgecolors="white", linewidths=0.4, zorder=3)
    axes[0].set_yticks(range(len(STATES)), STATES, fontsize=7)
    axes[0].set_ylabel("Population state")
    axes[1].scatter(generations, action_idx, c=colors, s=48, edgecolors="white", linewidths=0.55)
    axes[1].set_yticks(range(len(ACTIONS)), ACTIONS, fontsize=8)
    axes[1].set_ylabel("Selected action")
    axes[1].set_xlabel("Generation")
    axes[1].set_xlim(290, 1005)
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    used = [state for state in STATES if visits[state]]
    handles = [Patch(facecolor=state_colors[state], label=state) for state in used]
    axes[1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7,
                   title="State color", title_fontsize=8, frameon=False)
    fig.suptitle(f"MK06 state-action trajectory, seed {SEED_Q} ({len(actions)} Q decisions)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "状态动作轨迹.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    def reward_kind(item):
        hv, igd = item["hv_delta_pct"], item["igd_delta_pct"]
        if hv > 0 and igd > 0:
            return "HV and IGD improved"
        if hv > 0 and igd < 0:
            return "HV improved, IGD worsened"
        if hv < 0 and igd > 0:
            return "HV worsened, IGD improved"
        return "Other / unchanged"

    kinds = [reward_kind(item) for item in actions]
    kind_colors = {
        "HV and IGD improved": "#009e73",
        "HV improved, IGD worsened": "#e69f00",
        "HV worsened, IGD improved": "#377eb8",
        "Other / unchanged": "#777777",
    }
    fig, ax = plt.subplots(figsize=(10.6, 4.4))
    ax.axhline(0, color="black", linewidth=0.75)
    for kind, color in kind_colors.items():
        mask = np.array([x == kind for x in kinds])
        if mask.any():
            ax.scatter(generations[mask], np.array([a["reward"] for a in actions])[mask],
                       label=kind, color=color, s=37, alpha=0.82)
    clipped = [a for a in actions if a["clipped"]]
    if clipped:
        ax.scatter([a["generation"] for a in clipped], [a["reward"] for a in clipped],
                   marker="x", color="red", s=90, label="Clipped reward")
    ax.set(xlabel="Generation of Q decision", ylabel="Assigned reward", xlim=(290, 1005))
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "奖励轨迹.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    write_csv(out / "Q矩阵.csv", ["state"] + ACTIONS,
              [{"state": state, **{action: q[state, action] for action in ACTIONS}} for state in STATES])
    write_csv(out / "状态动作次数.csv", ["state", "action", "count"],
              [{"state": state, "action": action, "count": counts[state, action]}
               for state in STATES for action in ACTIONS])
    action_fields = ["generation", "end_generation", "state", "next_state", "action",
                     "reward", "raw_reward", "clipped", "hv_delta_pct", "igd_delta_pct",
                     "nd_delta", "improved_count", "ts_calls", "epsilon"]
    write_csv(out / "状态动作原始记录.csv", action_fields, actions)
    report = [
        "# MK06 Q 学习行为图说明", "",
        f"固定使用原配置第 1 次运行（seed={SEED_Q}），源日志：`{source.relative_to(ROOT)}`；SHA-256：`{sha(source)}`。",
        f"共 {len(actions)} 次动作决策、{len(meta['ts_events'])} 次实际 TS 触发、"
        f"{sum(visits.values())} 次状态访问；{sum(visits[s] > 0 for s in STATES)}/9 个状态被访问，"
        f"{sum(abs(v) > 1e-12 for v in q.values())}/90 个 Q 值非零，"
        f"{sum(a['clipped'] for a in actions)} 次奖励截断。", "",
        "热图展示全部九状态、十动作；灰行表示该状态未访问，圆点表示实际选择过的状态—动作对。",
        "状态—动作轨迹按每次 Q 决策的代数绘制，颜色表示当时状态；奖励轨迹区分 HV/IGD 变化组合。",
        f"最后 20 次决策仍覆盖 {len(set(a['action'] for a in actions[-20:]))}/10 个动作，"
        f"末次 ε={actions[-1]['epsilon']:.3f}；该条轨迹没有显示动作选择收敛为单一稳定策略。",
        "这些图描述指定种子的真实行为，也不能证明九状态划分最优。", "",
    ]
    (out / "说明.md").write_text("\n".join(report), encoding="utf-8")
    return out


def pareto_figures():
    all_outputs = []
    for case in CASES:
        out = OUT / f"Pareto投影_{case}_seed{SEED_PARETO}"
        out.mkdir(parents=True, exist_ok=True)
        points = {}
        rows = []
        sources = {}
        for alg in ALGS:
            source = result_path(case, alg, SEED_PARETO)
            with np.load(source) as data:
                F = np.asarray(data["F"], dtype=float)
            nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
            unique = np.unique(F[nd], axis=0)
            physical = np.column_stack((unique[:, 0], unique[:, 1] * 100,
                                        unique[:, 2] / 100000, unique[:, 1]))
            points[alg] = physical
            sources[alg] = {"path": str(source.relative_to(ROOT)), "sha256": sha(source),
                            "saved_rows": len(F), "unique_non_dominated_points": len(unique)}
            for index, row in enumerate(physical, 1):
                rows.append({"case": case, "seed": SEED_PARETO, "algorithm": alg,
                             "point": index, "makespan": row[0], "processing_energy": row[1],
                             "quality_loss": row[2], "processing_energy_over_100": row[3]})
        write_csv(out / "同次运行三目标点.csv",
                  ["case", "seed", "algorithm", "point", "makespan", "processing_energy",
                   "quality_loss", "processing_energy_over_100"], rows)
        specs = [
            (0, 1, "Makespan", "Processing energy", "工期_加工能耗.png"),
            (0, 2, "Makespan", "Quality loss (1 - minimum operation FPY)", "工期_质量损失.png"),
            (1, 2, "Processing energy", "Quality loss (1 - minimum operation FPY)", "加工能耗_质量损失.png"),
            (0, 3, "Makespan", "Processing energy / 100", "工期_能耗除100.png"),
            (3, 2, "Processing energy / 100", "Quality loss (1 - minimum operation FPY)",
             "能耗除100_质量损失.png"),
        ]
        for x_col, y_col, x_label, y_label, filename in specs:
            fig, ax = plt.subplots(figsize=(7.3, 5.1))
            for alg in ALGS:
                values = points[alg]
                ax.scatter(values[:, x_col], values[:, y_col], label=LABELS[alg],
                           color=COLORS[alg], marker=MARKERS[alg], s=49, alpha=0.76,
                           edgecolors="white", linewidths=0.5)
            ax.set(xlabel=x_label, ylabel=y_label)
            ax.grid(alpha=0.22)
            ax.legend(fontsize=8, frameon=False)
            fig.tight_layout()
            fig.savefig(out / filename, dpi=300, bbox_inches="tight")
            plt.close(fig)
        (out / "来源.json").write_text(json.dumps({"case": case, "seed": SEED_PARETO,
                                                    "objective_units": ["time", "processing energy", "1-min FPY"],
                                                    "scaled_projection": "processing energy / 100, matching manuscript Figure 4-5 scale",
                                                    "algorithms": sources}, ensure_ascii=False, indent=2), encoding="utf-8")
        all_outputs.append(out)
    return all_outputs


def gantt_figures():
    case = "MK06"
    seed = SEED_PARETO
    out = OUT / f"甘特图_{case}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    source = result_path(case, "NSGA2_TS_QL_GLR", seed)
    with np.load(source) as data:
        F, X = data["F"], data["X"]
    loader = revision.FJSPDataLoader(str(ROOT / f"{case}.xlsx"))
    codec = revision.FJSPCodec(loader)
    colors = plt.get_cmap("tab20", loader.n_jobs)
    output_rows = []
    summary = []
    for objective, label in enumerate(("makespan", "energy", "quality_loss")):
        index = int(np.argmin(F[:, objective]))
        makespan, energy, loss, data = codec.decode(X[index])
        expected = np.array([makespan, energy / 100, loss * 100000])
        if not np.allclose(expected, F[index], rtol=1e-10, atol=1e-7):
            raise RuntimeError(f"甘特图 {label} 的编码与目标值不一致")
        operations = data["operations"]
        if len(operations) != loader.total_operations:
            raise RuntimeError(f"甘特图 {label} 工序数量不对")
        fig, ax = plt.subplots(figsize=(13.8, max(6.5, loader.n_machines * 0.55)))
        for op in operations:
            machine = op["machine"]
            duration = op["end"] - op["start"]
            ax.barh(machine, duration, left=op["start"], height=0.68,
                    facecolor=colors(op["job"]), edgecolor="black", linewidth=0.45)
            if duration >= 3:
                ax.text(op["start"] + duration / 2, machine, f"J{op['job']+1}-O{op['op']+1}",
                        ha="center", va="center", fontsize=6.5, color="black")
            output_rows.append({"selected_objective": label, "source_row": index,
                                "job": op["job"] + 1, "operation": op["op"] + 1,
                                "machine": machine + 1, "start": op["start"], "end": op["end"],
                                "processing_time": duration, "production_power": op["power"], "FPY": op["ftt"]})
        ax.set_yticks(range(loader.n_machines), [f"M{i+1}" for i in range(loader.n_machines)], fontsize=9)
        ax.set_ylim(-0.7, loader.n_machines - 0.3)
        ax.invert_yaxis()
        ax.set_xlim(0, makespan * 1.02)
        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel("Machine", fontsize=10)
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        handles = [Patch(facecolor=colors(j), edgecolor="black", label=f"Job {j+1}")
                   for j in range(loader.n_jobs)]
        ax.legend(handles=handles, title="Jobs", fontsize=7, title_fontsize=8,
                  loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=min(8, loader.n_jobs), frameon=False)
        ax.set_title(f"MK06 GLR, seed {seed}, minimum {label}\n"
                     f"Makespan={makespan:.2f}; processing energy={energy:.2f}; quality loss={loss:.4f}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"甘特图_{label}.png", dpi=400, bbox_inches="tight")
        fig.savefig(out / f"甘特图_{label}.pdf", bbox_inches="tight",
                    metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)
        summary.append({"selected_objective": label, "source_row": index,
                        "makespan": makespan, "processing_energy": energy, "quality_loss": loss,
                        "operations": len(operations)})
    write_csv(out / "工序排程.csv",
              ["selected_objective", "source_row", "job", "operation", "machine", "start", "end",
               "processing_time", "production_power", "FPY"], output_rows)
    write_csv(out / "选中解.csv", ["selected_objective", "source_row", "makespan", "processing_energy", "quality_loss", "operations"], summary)
    (out / "来源.json").write_text(json.dumps({"case": case, "seed": seed,
                                               "source": str(source.relative_to(ROOT)), "sha256": sha(source),
                                               "case_sha256": sha(ROOT / "MK06.xlsx"),
                                               "drawn_operations": loader.total_operations}, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main():
    OUT.mkdir(exist_ok=True)
    q = q_figures()
    pareto = pareto_figures()
    gantt = gantt_figures()
    print(q)
    for p in pareto:
        print(p)
    print(gantt)


if __name__ == "__main__":
    main()
