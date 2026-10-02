"""绘制 MK06 修回版四算法 20 次运行的 HV/离线 IGD 均值和标准差。"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


WORK = Path(__file__).resolve().parent.parent
FOLDER = WORK / "evaluation_results" / "MK06离线逐代"
SOURCE = FOLDER / "逐代汇总.csv"
NAMES = {
    "NSGA2": "NSGA-II",
    "NSGA2_TS": "NSGA-II-TS",
    "NSGA2_TS_QL": "NSGA-II-TS-QL",
    "NSGA2_TS_QL_GLR": "NSGA-II-TS-QL-GLR",
}
COLORS = {
    "NSGA2": "#555555",
    "NSGA2_TS": "#377eb8",
    "NSGA2_TS_QL": "#e69f00",
    "NSGA2_TS_QL_GLR": "#009e73",
}


def plot(data, mean_col, sd_col, ylabel, filename):
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for alg, label in NAMES.items():
        group = data[data["algorithm"] == alg].sort_values("generation")
        if len(group) != 1000 or not (group["completed_runs"] == 20).all():
            raise RuntimeError(f"{alg} 缺少 20 次 × 1000 代记录")
        x = group["generation"].to_numpy()
        y = group[mean_col].to_numpy()
        sd = group[sd_col].to_numpy()
        color = COLORS[alg]
        ax.plot(x, y, color=color, label=label, linewidth=1.8)
        ax.fill_between(x, (y - sd).clip(min=0), y + sd, color=color, alpha=0.14, linewidth=0)
    ax.set(xlabel="Generation", ylabel=ylabel, xlim=(1, 1000))
    ax.grid(alpha=0.22, linewidth=0.6)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path = FOLDER / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(path)


def main():
    data = pd.read_csv(SOURCE)
    plot(data, "hv_mean", "hv_sd", "Hypervolume", "MK06_HV_20runs_mean_sd.png")
    plot(data, "igd_offline_mean", "igd_offline_sd", "Offline IGD", "MK06_IGD_offline_20runs_mean_sd.png")


if __name__ == "__main__":
    main()
