"""用保存的最终解统一重算离线 HV/IGD，不改动搜索时的参考前沿或原始结果。"""

import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.stats import wilcoxon

from sensitivity_algorithms import revision


WORK = Path(__file__).resolve().parent.parent
SENS = WORK / "sensitivity_results"
MAIN = WORK / "主实验正式结果"
FIXED = WORK / "固定TS修正结果"
OUT = WORK / "evaluation_results"
CASES = ("MK02", "MK06", "MK09")
ALGS = ("NSGA2", "NSGA2_TS", "NSGA2_TS_QL", "NSGA2_TS_QL_GLR")
CONFIGS = ("reward_original", "reward_equal", "reward_hv_only", "reward_igd_only", "ratio_2_3_5", "ratio_4_2_4")
SEEDS = [42 + 5 * i for i in range(20)]


def source_dir(case, name):
    if name == "NSGA2_TS":
        return FIXED / case
    if case == "MK06" and name == "NSGA2_TS_QL_GLR":
        return SENS / "reward_original"
    return MAIN / case / name


def load_run(path):
    data_path = path.with_suffix(".npz")
    if not path.exists() or not data_path.exists():
        raise RuntimeError(f"缺少正式运行：{path}")
    meta = json.loads(path.read_text(encoding="utf-8"))
    with np.load(data_path) as data:
        F = np.asarray(data["F"], dtype=float)
        curve = np.asarray(data["generation_metrics"], dtype=float)
    if F.ndim != 2 or F.shape[1] != 3 or not np.all(np.isfinite(F)):
        raise RuntimeError(f"目标数组不合法：{data_path}")
    if curve.shape not in ((1000, 4), (0, 4)):
        raise RuntimeError(f"逐代数组不合法：{data_path}")
    nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
    F = np.unique(F[nd], axis=0)
    return {"F": F, "curve": curve, "meta": meta, "source": str(data_path.resolve())}


def load_all():
    main = {}
    sensitivity = {}
    for case in CASES:
        for alg in ALGS:
            main[case, alg] = [
                load_run(source_dir(case, alg) / f"seed_{seed}.json") for seed in SEEDS
            ]
    for config in CONFIGS:
        sensitivity[config] = [
            load_run(SENS / config / f"seed_{seed}.json") for seed in SEEDS
        ]
    return main, sensitivity


def reference(case, main, sensitivity):
    fronts = [item["F"] for alg in ALGS for item in main[case, alg]]
    if case == "MK06":
        fronts.extend(item["F"] for config in CONFIGS[1:] for item in sensitivity[config])
    pool = np.unique(np.vstack(fronts), axis=0)
    nd = NonDominatedSorting().do(pool, only_non_dominated_front=True)
    front = pool[nd]
    source_ref = SENS / "reference.npz" if case == "MK06" else MAIN / case / "reference.npz"
    with np.load(source_ref) as data:
        hv_point = data["hv_point"].copy()
        ideal = data["ideal"].copy()
        nadir = data["nadir"].copy()
    if not np.all(nadir > ideal):
        raise RuntimeError(f"{case} 的原始归一化范围不合法。")
    ref_path = OUT / f"{case}_经验参考前沿.npz"
    arrays = {"front": front, "ideal": ideal, "nadir": nadir, "hv_point": hv_point}
    if ref_path.exists():
        with np.load(ref_path) as saved:
            if any(not np.array_equal(saved[key], value) for key, value in arrays.items()):
                raise RuntimeError(f"{case} 经验参考前沿与已有文件不同；不能覆盖既有评价。")
    else:
        np.savez_compressed(ref_path, **arrays)
    return front, ideal, nadir, hv_point


def evaluate_runs(case, group, items, ref):
    front, ideal, nadir, hv_point = ref
    igd_metric = revision.IGDMetric(front, ideal, nadir)
    hv_metric = Hypervolume(ref_point=hv_point)
    output = []
    for seed, item in zip(SEEDS, items):
        F = item["F"]
        if not np.all(F < hv_point):
            raise RuntimeError(f"{case}/{group} seed={seed}: HV 参考点不足。")
        output.append({
            "case": case, "group": group, "seed": seed,
            "hv": float(hv_metric.do(F)),
            "igd_offline": float(igd_metric.compute(F)),
            "pareto_count": len(F),
            "igd_online_saved": item["meta"]["igd"],
            "source_npz": item["source"],
        })
    return output


def save_csv(path, records, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def summarize(records):
    groups = list(dict.fromkeys((r["case"], r["group"]) for r in records))
    summary = []
    for case, group in groups:
        subset = [r for r in records if r["case"] == case and r["group"] == group]
        hv = np.array([r["hv"] for r in subset])
        igd = np.array([r["igd_offline"] for r in subset])
        summary.append({
            "case": case, "group": group, "runs": len(subset),
            "hv_mean": hv.mean(), "hv_sd": hv.std(ddof=0),
            "igd_mean": igd.mean(), "igd_sd": igd.std(ddof=0),
        })
    return summary


def holm(p_values):
    order = np.argsort(p_values)
    output = np.zeros(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        output[index] = running
    return output


def paired_tests(records, case, baseline, others, metric):
    data = {(r["case"], r["group"], r["seed"]): r for r in records}
    results = []
    for other in others:
        delta = np.array([
            data[case, baseline, seed][metric] - data[case, other, seed][metric]
            for seed in SEEDS
        ], dtype=float)
        p = 1.0 if np.allclose(delta, 0, atol=0, rtol=0) else float(wilcoxon(delta).pvalue)
        results.append({
            "case": case, "metric": metric, "baseline": baseline,
            "other": other, "paired_runs": len(SEEDS),
            "baseline_minus_other_mean": delta.mean(), "p_raw": p,
        })
    adjusted = holm(np.array([r["p_raw"] for r in results]))
    for row, p in zip(results, adjusted):
        row["p_holm"] = p
    return results


def convergence(main):
    records = []
    for case in CASES:
        for alg in ALGS:
            arr = np.stack([item["curve"] for item in main[case, alg]])
            if arr.shape != (20, 1000, 4):
                raise RuntimeError(f"{case}/{alg} 没有 20 次完整逐代记录。")
            for i in range(1000):
                x = arr[:, i]
                records.append({
                    "case": case, "algorithm": alg, "generation": int(x[0, 0]),
                    "runs": 20, "hv_mean": x[:, 1].mean(), "hv_sd": x[:, 1].std(ddof=0),
                    "igd_online_mean": x[:, 2].mean(), "igd_online_sd": x[:, 2].std(ddof=0),
                    "nd_mean": x[:, 3].mean(), "nd_sd": x[:, 3].std(ddof=0),
                })
    save_csv(OUT / "主实验逐代在线指标.csv", records, [
        "case", "algorithm", "generation", "runs", "hv_mean", "hv_sd",
        "igd_online_mean", "igd_online_sd", "nd_mean", "nd_sd",
    ])


def main():
    OUT.mkdir(exist_ok=True)
    data_main, data_sens = load_all()
    all_main = []
    all_sens = []
    ref_sizes = {}
    for case in CASES:
        ref = reference(case, data_main, data_sens)
        ref_sizes[case] = len(ref[0])
        for alg in ALGS:
            all_main.extend(evaluate_runs(case, alg, data_main[case, alg], ref))
        if case == "MK06":
            for config in CONFIGS:
                all_sens.extend(evaluate_runs(case, config, data_sens[config], ref))
    fields = ["case", "group", "seed", "hv", "igd_offline", "pareto_count", "igd_online_saved", "source_npz"]
    save_csv(OUT / "主实验逐次.csv", all_main, fields)
    save_csv(OUT / "敏感性逐次.csv", all_sens, fields)
    summaries_main = summarize(all_main)
    summaries_sens = summarize(all_sens)
    sfields = ["case", "group", "runs", "hv_mean", "hv_sd", "igd_mean", "igd_sd"]
    save_csv(OUT / "主实验汇总.csv", summaries_main, sfields)
    save_csv(OUT / "敏感性汇总.csv", summaries_sens, sfields)
    tests = []
    for case in CASES:
        for metric in ("hv", "igd_offline"):
            tests.extend(paired_tests(all_main, case, "NSGA2_TS_QL_GLR", ALGS[:-1], metric))
    for metric in ("hv", "igd_offline"):
        tests.extend(paired_tests(all_sens, "MK06", "reward_original", CONFIGS[1:], metric))
    save_csv(OUT / "配对检验.csv", tests, [
        "case", "metric", "baseline", "other", "paired_runs",
        "baseline_minus_other_mean", "p_raw", "p_holm",
    ])
    convergence(data_main)
    input_protocols = [SENS / "protocol.json"] + [MAIN / c / "protocol.json" for c in CASES] + [FIXED / c / "protocol.json" for c in CASES]
    manifest = {
        "method": "per-case empirical non-dominated union of final fronts; MK06 includes five alternate sensitivity configurations; IGD keeps original pre-run ideal/nadir; HV keeps original fixed point",
        "main_runs": len(all_main), "sensitivity_rows": len(all_sens),
        "reference_points": ref_sizes,
        "evaluation_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_protocol_sha256": {str(p.resolve()): sha256(p.read_bytes()).hexdigest() for p in input_protocols},
        "online_igd_note": "search-time Q rewards and saved per-generation IGD use original pre-run reference; only final IGD is recomputed here",
    }
    (OUT / "评价口径.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 修回版统一评价（独立后处理）", "",
        "本目录不修改原始实验数据。固定 TS 使用排序修正后的独立重跑结果；其余算法复用已完成的正式运行。",
        "每个算例分别合并所有最终非支配目标点，建立经验近似 Pareto 参考前沿。MK06 同时纳入敏感性替代组。",
        "最终 IGD 的参考前沿换为经验前沿，ideal/nadir 归一化范围沿用原实验；HV 沿用原实验固定参考点。",
        "搜索时 Q 奖励及逐代 IGD 使用预实验静态前沿，其定义和数值没有被本后处理改变。",
        "逐代在线 IGD 与离线最终 IGD 口径不同，画图和写表时必须标明。", "",
        "## 主实验", "",
        "| 算例 | 算法 | 20 次 HV 均值 ± SD | 20 次离线 IGD 均值 ± SD |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in summaries_main:
        lines.append(f"| {row['case']} | {row['group']} | {row['hv_mean']:.6g} ± {row['hv_sd']:.6g} | {row['igd_mean']:.6g} ± {row['igd_sd']:.6g} |")
    lines += ["", "## MK06 敏感性", "",
              "| 配置 | 20 次 HV 均值 ± SD | 20 次离线 IGD 均值 ± SD |",
              "| --- | ---: | ---: |"]
    for row in summaries_sens:
        lines.append(f"| {row['group']} | {row['hv_mean']:.6g} ± {row['hv_sd']:.6g} | {row['igd_mean']:.6g} ± {row['igd_sd']:.6g} |")
    lines += [
        "", "配对 Wilcoxon 符号秩检验和 Holm 校正见 `配对检验.csv`。"
        "Holm 分别在每个算例、每个指标的三项主实验比较内，以及每个指标的五项敏感性比较内计算。", "",
    ]
    if (OUT / "MK06离线逐代" / "逐代汇总.csv").exists():
        lines += [
            "## MK06 收敛曲线", "",
            "已对四算法各 20 次使用相同种子确定性复跑，逐次确认最终前沿与正式保存结果完全一致。",
            "`MK06离线逐代/逐代汇总.csv` 包含 1000 代 HV/离线 IGD 的均值与标准差，"
            "第 1000 代均值与上表终值一致；两张 300 dpi 图片位于同目录。", "",
        ]
    (OUT / "结果说明.md").write_text("\n".join(lines), encoding="utf-8")
    print("reference sizes:", ref_sizes)
    print("main runs:", len(all_main), "sensitivity rows:", len(all_sens))
    print(OUT / "结果说明.md")


if __name__ == "__main__":
    main()
