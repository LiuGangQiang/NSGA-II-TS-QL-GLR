"""从统一离线评价生成图 4-8/4-9 候选图和表 3 核对值。"""

import csv
from hashlib import sha256
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WORK = Path(__file__).resolve().parent.parent
SOURCE = WORK / "evaluation_results" / "主实验汇总.csv"
OUT = WORK / "supplementary_figures" / "跨算例比较_统一评价"
ALGORITHMS = ["NSGA2", "NSGA2_TS", "NSGA2_TS_QL", "NSGA2_TS_QL_GLR"]
LABELS = ["NSGA-II", "NSGA-II-TS", "NSGA-II-TS-QL", "NSGA-II-TS-QL-GLR"]
COLORS = ["#666666", "#377eb8", "#e69f00", "#009e73"]
CASES = ["MK02", "MK06", "MK09"]


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, fields, data):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {(r["case"], r["group"]): r for r in rows(SOURCE)}
    normalized = []
    for case in CASES:
        baseline = summary[case, ALGORITHMS[-1]]
        hv_ref, igd_ref = float(baseline["hv_mean"]), float(baseline["igd_mean"])
        if hv_ref <= 0 or igd_ref <= 0:
            raise ValueError(f"{case} 的 GLR 基线必须为正")
        for alg in ALGORITHMS:
            item = summary[case, alg]
            normalized.append({
                "case": case, "algorithm": alg,
                "hv_mean": item["hv_mean"], "igd_mean": item["igd_mean"],
                "hv_over_glr": float(item["hv_mean"]) / hv_ref,
                "igd_excess_over_glr": (float(item["igd_mean"]) - igd_ref) / igd_ref,
            })
    write(OUT / "图4-8_4-9归一化数据.csv",
          ["case", "algorithm", "hv_mean", "igd_mean", "hv_over_glr", "igd_excess_over_glr"], normalized)

    x = np.arange(len(CASES))
    for field, ylabel, filename, baseline in (
        ("hv_over_glr", "Mean HV / GLR mean HV (higher is better)", "图4-8_归一化HV候选.png", 1.0),
        ("igd_excess_over_glr", "Relative IGD excess vs GLR (lower is better)",
         "图4-9_相对IGD候选.png", 0.0),
    ):
        fig, ax = plt.subplots(figsize=(8.7, 4.8))
        width = 0.19
        for j, (alg, label, color) in enumerate(zip(ALGORITHMS, LABELS, COLORS)):
            values = [next(float(r[field]) for r in normalized if r["case"] == case and r["algorithm"] == alg)
                      for case in CASES]
            ax.bar(x + (j - 1.5) * width, values, width=width, color=color, label=label,
                   edgecolor="white", linewidth=0.5)
        ax.axhline(baseline, color="#444444", linestyle="--", linewidth=0.8)
        if field == "igd_excess_over_glr":
            ax.scatter(x + 1.5 * width, np.zeros(len(CASES)), color=COLORS[-1], marker="D",
                       s=33, zorder=4, clip_on=False)
        ax.set_xticks(x, CASES)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(fontsize=7, ncol=2, frameon=False)
        if field == "hv_over_glr":
            ax.set_ylim(0, 1.12)
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    time_sources = {
        "NSGA2": WORK / "主实验正式结果" / "MK06" / "最终指标.csv",
        "NSGA2_TS": WORK / "固定TS修正结果" / "MK06" / "最终指标.csv",
        "NSGA2_TS_QL": WORK / "主实验正式结果" / "MK06" / "最终指标.csv",
        "NSGA2_TS_QL_GLR": WORK / "sensitivity_results" / "最终指标.csv",
    }
    table = []
    for alg in ALGORITHMS:
        item = summary["MK06", alg]
        source = time_sources[alg]
        raw = rows(source)
        timing = [float(r["elapsed_seconds"]) for r in raw
                  if r.get("algorithm", alg) == alg and r.get("config", "reward_original") == "reward_original"]
        if len(timing) != 20:
            raise RuntimeError(f"{alg} 计时条数为 {len(timing)}，预期 20")
        table.append({"algorithm": alg, "runs": 20,
                      "hv_mean": item["hv_mean"], "hv_sd": item["hv_sd"],
                      "igd_offline_mean": item["igd_mean"], "igd_offline_sd": item["igd_sd"],
                      "elapsed_seconds_mean": mean(timing), "elapsed_seconds_sd": stdev(timing),
                      "time_source": str(source.relative_to(WORK.parent))})
    write(OUT / "表3_MK06核对值.csv",
          ["algorithm", "runs", "hv_mean", "hv_sd", "igd_offline_mean", "igd_offline_sd",
           "elapsed_seconds_mean", "elapsed_seconds_sd", "time_source"], table)
    (OUT / "说明.md").write_text(
        "# 跨算例候选图与表 3 核对值\n\n"
        "图 4-8 的纵轴是各算法 20 次平均 HV 除以同算例 GLR 的 20 次平均 HV，GLR=1。\n"
        "图 4-9 的纵轴是各算法 20 次平均离线 IGD 相对同算例 GLR 的增量比例，GLR=0；"
        "值越低越好。两图均使用修回版统一评价，不能与投稿版表 6 数值混用。\n\n"
        "`表3_MK06核对值.csv` 保留完整精度，论文排版建议 HV 统一用 ×10⁶、"
        "IGD 用 4 位小数、时间用 1 位小数，并明确写均值 ± 标准差。"
        "运行墙钟时间来自分别保存的实验记录，若当时同时运行多个进程，不适合作严格计算开销比较。\n\n"
        f"统一评价主表 SHA-256：`{sha256(SOURCE.read_bytes()).hexdigest()}`。\n",
        encoding="utf-8",
    )
    print(OUT)


if __name__ == "__main__":
    main()
