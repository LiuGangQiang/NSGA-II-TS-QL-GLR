"""只读核验修回版正式结果；报告写入 正式结果核验.md。"""

import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from sensitivity_algorithms import revision


WORK = Path(__file__).resolve().parent.parent
ROOT = WORK
SENS = WORK / "sensitivity_results"
MAIN = WORK / "主实验正式结果"
CASES = ("MK02", "MK06", "MK09")
ALGS = ("NSGA2", "NSGA2_TS", "NSGA2_TS_QL", "NSGA2_TS_QL_GLR")
CONFIGS = (
    "reward_original", "reward_equal", "reward_hv_only", "reward_igd_only",
    "ratio_2_3_5", "ratio_4_2_4",
)
SEEDS = [42 + 5 * i for i in range(20)]
errors = []
warnings = []
rows = []


def check(condition, message):
    if not condition:
        errors.append(message)


def digest(path):
    d = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def inspect_protocol(folder, case):
    path = folder / "protocol.json"
    if not path.exists():
        errors.append(f"缺少 {path}")
        return None
    p = json.loads(path.read_text(encoding="utf-8"))
    check(p.get("instance") == case, f"{folder}: 算例名不符")
    check(p.get("run_seeds") == SEEDS, f"{folder}: 种子列表不符")
    check(p.get("generations") == 1000 and p.get("population") == 60,
          f"{folder}: 代数或种群不符")
    check(p.get("model") == "processing_energy_only; quality_loss=1-min_operation_FPY; precedence_enforced",
          f"{folder}: 修回模型口径不符")
    for name, saved_hash in p.get("source_sha256", {}).items():
        source = ROOT / name if name.endswith(".xlsx") else WORK / name
        check(source.exists() and digest(source) == saved_hash,
              f"{folder}: 源码或算例已改变：{name}")
    return p


def read_reference(folder):
    path = folder / "reference.npz"
    if not path.exists():
        errors.append(f"缺少 {path}")
        return None
    with np.load(path) as data:
        ref = {key: data[key].copy() for key in ("front", "ideal", "nadir", "hv_point")}
    check(ref["front"].ndim == 2 and ref["front"].shape[1] == 3 and len(ref["front"]) > 0,
          f"{path}: 参考前沿为空或维数错误")
    check(all(np.all(np.isfinite(x)) for x in ref.values()), f"{path}: 含非有限数")
    check(np.all(ref["ideal"] < ref["nadir"]), f"{path}: 归一化范围不合法")
    check(np.all(ref["front"] < ref["hv_point"]), f"{path}: HV 参考点不支配参考前沿")
    return ref


def inspect_run(folder, label, seed, reference, problem, curve_expected, q_expected, glr_expected):
    data_path = folder / f"seed_{seed}.npz"
    meta_path = folder / f"seed_{seed}.json"
    if not data_path.exists() or not meta_path.exists():
        errors.append(f"缺少 {label} seed={seed} 的 npz 或 json")
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(data_path) as data:
        F = np.asarray(data["F"], dtype=float)
        X = np.asarray(data["X"], dtype=float)
        curve = np.asarray(data["generation_metrics"], dtype=float)
    check(F.ndim == 2 and F.shape[1] == 3 and len(F) > 0,
          f"{label} seed={seed}: F 形状错误")
    check(X.ndim == 2 and X.shape[0] == len(F),
          f"{label} seed={seed}: X 与 F 行数不符")
    check(np.all(np.isfinite(F)) and np.all(F >= 0),
          f"{label} seed={seed}: F 非有限或为负")
    check(np.all(np.isfinite(X)) and np.all((X >= 0) & (X <= 1)),
          f"{label} seed={seed}: X 超出编码范围")
    recalculated = np.asarray(problem.evaluate(X), dtype=float)
    check(recalculated.shape == F.shape and np.allclose(recalculated, F, rtol=1e-10, atol=1e-7),
          f"{label} seed={seed}: 决策编码 X 重算目标与 F 不符")
    check(meta.get("seed") == seed and meta.get("generations") == 1000,
          f"{label} seed={seed}: 元数据种子或代数错误")
    check(meta.get("population") == 60 and meta.get("pareto_count") == len(F),
          f"{label} seed={seed}: 种群或 Pareto 数量错误")
    check(np.all(F < reference["hv_point"]),
          f"{label} seed={seed}: HV 参考点未支配最终解")
    nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
    hv = float(Hypervolume(ref_point=reference["hv_point"]).do(F))
    igd = float(revision.IGDMetric(
        reference["front"], reference["ideal"], reference["nadir"]
    ).compute(F))
    if len(nd) != len(F):
        clean_igd = revision.IGDMetric(
            reference["front"], reference["ideal"], reference["nadir"]
        ).compute(F[nd])
        errors.append(
            f"{label} seed={seed}: 最终 F 中 {len(F)-len(nd)}/{len(F)} 个点被支配；"
            f"剔除后 IGD {igd:.8f} → {clean_igd:.8f}"
        )
    check(np.isclose(hv, meta["hv"], rtol=1e-10, atol=1e-6),
          f"{label} seed={seed}: HV 与 F 重算不符")
    check(np.isclose(igd, meta["igd"], rtol=1e-10, atol=1e-10),
          f"{label} seed={seed}: IGD 与 F 重算不符")
    expected_shape = (1000, 4) if curve_expected else (0, 4)
    check(curve.shape == expected_shape and meta.get("generation_record_count") == expected_shape[0],
          f"{label} seed={seed}: 逐代记录不完整")
    if curve_expected and curve.shape == expected_shape:
        check(np.array_equal(curve[:, 0], np.arange(1, 1001)),
              f"{label} seed={seed}: 逐代编号不连续")
        check(np.isclose(curve[-1, 1], hv, rtol=1e-10, atol=1e-6)
              and np.isclose(curve[-1, 2], igd, rtol=1e-10, atol=1e-10),
              f"{label} seed={seed}: 最后一代 HV/IGD={curve[-1, 1]:.8f}/{curve[-1, 2]:.8f}，"
              f"最终 F 重算={hv:.8f}/{igd:.8f}")
    if q_expected:
        actions = meta.get("actions", [])
        q = meta.get("q_table", [])
        visits = meta.get("visit_count", {})
        check(bool(actions) and any(abs(item["value"]) > 1e-12 for item in q)
              and sum(visits.values()) > 0,
              f"{label} seed={seed}: Q 学习记录为空")
        check(meta.get("reward_update_count") == len(actions),
              f"{label} seed={seed}: Q 更新次数与动作记录不符")
        clip_count = sum(bool(a.get("clipped")) for a in actions)
        check(meta.get("reward_clip_count") == clip_count,
              f"{label} seed={seed}: 截断次数与动作记录不符")
        if glr_expected:
            check(meta.get("sampling", {}).get("unique") == 60,
                  f"{label} seed={seed}: GLR 初始编码不足 60 个")
    return {"meta": meta, "F": F, "curve": curve}


def inspect_csv(folder, expected_rows):
    path = folder / "最终指标.csv"
    if not path.exists():
        errors.append(f"缺少 {path}")
        return
    with path.open(encoding="utf-8-sig", newline="") as stream:
        values = list(csv.DictReader(stream))
    check(len(values) == expected_rows, f"{path}: 行数 {len(values)}，应为 {expected_rows}")


def main():
    s_protocol = inspect_protocol(SENS, "MK06")
    s_ref = read_reference(SENS)
    check(s_protocol is not None and s_ref is not None, "敏感性实验协议或参考数据缺失")
    all_fronts = {case: [] for case in CASES}
    all_stats = {}
    problems = {
        case: revision.FJSPProblem(revision.FJSPDataLoader(str(ROOT / f"{case}.xlsx")))
        for case in CASES
    }
    if s_protocol and s_ref:
        for config in CONFIGS:
            data = []
            for seed in SEEDS:
                item = inspect_run(
                    SENS / config, f"MK06/{config}", seed, s_ref, problems["MK06"],
                    curve_expected=config == "reward_original", q_expected=True,
                    glr_expected=True,
                )
                if item:
                    data.append(item)
            all_stats[f"sensitivity/{config}"] = data
        inspect_csv(SENS, 120)
        path = SENS / "原配置逐代汇总.csv"
        check(path.exists(), "敏感性实验缺少原配置逐代汇总")
        rows.append(f"MK06 敏感性：六组分别 {[len(all_stats[f'sensitivity/{x}']) for x in CONFIGS]} 次。")
        rows.append(
            "MK06 敏感性各组奖励截断率均值：" +
            "，".join(
                f"{config}={np.mean([x['meta']['reward_clip_rate'] for x in all_stats[f'sensitivity/{config}']]):.2%}"
                for config in CONFIGS
            )
        )

    for case in CASES:
        folder = MAIN / case
        protocol = inspect_protocol(folder, case)
        reference = s_ref if case == "MK06" else read_reference(folder)
        if not protocol or not reference:
            continue
        if case == "MK06":
            check(Path(protocol.get("reference_path", "")) == SENS / "reference.npz",
                  "MK06 主实验未引用敏感性实验参考前沿")
        data_by_alg = {}
        for alg in ALGS:
            if case == "MK06" and alg == "NSGA2_TS_QL_GLR":
                data = all_stats.get("sensitivity/reward_original", [])
            else:
                data = []
                for seed in SEEDS:
                    item = inspect_run(
                        folder / alg, f"{case}/{alg}", seed, reference, problems[case],
                        curve_expected=True, q_expected="QL" in alg,
                        glr_expected=alg == "NSGA2_TS_QL_GLR",
                    )
                    if item:
                        data.append(item)
            data_by_alg[alg] = data
            all_fronts[case].extend(item["F"] for item in data)
        inspect_csv(folder, 80)
        curve_csv = folder / "逐代汇总.csv"
        if curve_csv.exists():
            with curve_csv.open(encoding="utf-8-sig", newline="") as stream:
                curve_rows = sum(1 for _ in csv.DictReader(stream))
            check(curve_rows == 4000, f"{case}: 逐代汇总 {curve_rows} 行，应为 4000")
        else:
            errors.append(f"{case}: 缺少逐代汇总.csv")
        rows.append(f"{case} 主实验：四算法分别 {[len(data_by_alg[x]) for x in ALGS]} 次；参考前沿 {len(reference['front'])} 点。")

        if all_fronts[case]:
            union = np.unique(np.vstack(all_fronts[case]), axis=0)
            nd = NonDominatedSorting().do(union, only_non_dominated_front=True)
            pooled = union[nd]
            ref_front = reference["front"]
            dominated = sum(np.any(np.all(pooled <= point, axis=1)
                                  & np.any(pooled < point, axis=1)) for point in ref_front)
            rows.append(
                f"{case}：20 次 × 四算法最终解合并后有 {len(pooled)} 个非支配目标点；"
                f"预实验参考点中 {dominated}/{len(ref_front)} 个被这些最终解支配。"
            )
            alt = revision.IGDMetric(pooled, reference["ideal"], reference["nadir"])
            rows.append(
                f"{case} 诊断性合并前沿 IGD 均值（仅用于检查参考点敏感性）：" +
                "，".join(
                    f"{alg}={np.mean([alt.compute(item['F']) for item in data_by_alg[alg]]):.5f}"
                    for alg in ALGS
                )
            )
            if len(ref_front) < 50 and dominated > len(ref_front) * 0.8:
                warnings.append(
                    f"{case} 的固定 IGD 参考前沿仅 {len(ref_front)} 点，且 {dominated}/{len(ref_front)} "
                    "点被完整实验的最终解支配；参考前沿较弱，IGD 排名需谨慎解释。"
                )

    report = [
        "# 修回版正式结果核验", "",
        "本报告审计最初的正式输出，保留其问题记录。固定 TS 独立重跑和离线指标修正已完成；"
        "现行数据口径见 `当前数据处理结论.md` 与 `evaluation_results/结果说明.md`。", "",
        "核验时间：以当前工作区数据和源码为准。", "",
        "## 完整性和数值", "",
    ]
    report.extend(f"- {row}" for row in rows)
    report.extend(["", f"- 检出错误：{len(errors)} 项；解释性提醒：{len(warnings)} 项。", ""])
    if errors:
        report.extend(["## 错误", ""] + [f"- {x}" for x in errors] + [""])
    if warnings:
        report.extend(["## 解释性提醒", ""] + [f"- {x}" for x in warnings] + [""])
    report.extend([
        "## 当前结论与后续处理", "",
        "- 正式运行的次数、种子、模型、代码校验、目标重算、Q 日志及奖励记录已通过核验；原始输出保留不动。",
        "- 固定 TS 实现会直接改写个体的 X/F，而没有立即重算非支配排序与拥挤度；"
        "少数运行因此在最终导出时保留了被支配点。先从保存的 X/F 重新筛出非支配前沿并计算最终指标；"
        "若论文要求修正搜索过程中同一问题，需单独修复固定 TS 后重跑其 20 次 × 三算例。",
        "- 预实验静态前沿用于运行时 Q 奖励可以保留，但它不宜直接作为最终算法优劣的唯一 IGD 依据。"
        "建议对三算例分别由四算法所有最终非支配解建立统一的事后参考前沿，"
        "重新计算各次最终 IGD、统计检验及表格；现有逐代 IGD 曲线仍对应原静态参考，不能改标签冒充新参考。",
        "- 在上述核算完成前，不应直接把当前 `阶段结果.md` 的 IGD 排名和旧固定 TS 终值写入修回稿。",
        "",
    ])
    report.extend([
        "## 核验范围", "",
        "已核对协议、源码/算例 SHA-256、种子、代数、人口、目标数组、决策编码、逐代曲线、",
        "HV/IGD 从原始 F 重算、Q 表与访问数、奖励截断、GLR 初始种群、各 CSV 行数及 MK06 基准复用。",
        "诊断性合并前沿只用于判断参考点敏感性，未替换任何正式运行的奖励或论文指标。",
        "",
    ])
    path = WORK / "正式结果核验.md"
    path.write_text("\n".join(report), encoding="utf-8")
    print(path)
    print("\n".join(rows))
    print(f"errors={len(errors)} warnings={len(warnings)}")
    for message in errors + warnings:
        print(message)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
