"""修回版 MK02/MK06/MK09 四算法对照实验。

短程检查：python run_main_experiments.py --smoke --case MK06
正式运行：python run_main_experiments.py --full --case MK06|MK02|MK09
MK06 正式运行复用 run_sensitivity_experiments.py --full 的 reward_original 20 次结果及参考前沿。
"""

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import Hypervolume
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover as SBX
from pymoo.operators.mutation.pm import PolynomialMutation as PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

from run_sensitivity_experiments import (
    FixedReference, MetricsTracker, atomic_json, build_reference, file_hash,
)
from sensitivity_algorithms import REWARDS, SensitivityGLR, revision


WORK_DIR = Path(__file__).resolve().parent
ROOT = WORK_DIR.parent
CASES = ("MK02", "MK06", "MK09")
ALGORITHMS = ("NSGA2", "NSGA2_TS", "NSGA2_TS_QL", "NSGA2_TS_QL_GLR")
SENSITIVITY_DIR = ROOT / "sensitivity_results"


def source_hashes(case):
    paths = [
        ROOT / f"{case}.xlsx",
        WORK_DIR / "main523-mk06-revision.py",
        WORK_DIR / "sensitivity_algorithms.py",
        WORK_DIR / "run_sensitivity_experiments.py",
        Path(__file__).resolve(),
    ]
    return {p.name: file_hash(p) for p in paths}


def baseline_dir(case, full, out_dir):
    if case == "MK06" and full:
        return SENSITIVITY_DIR / "reward_original"
    return out_dir / "NSGA2_TS_QL_GLR"


def check_mk06_baseline(seeds, generations):
    protocol_file = SENSITIVITY_DIR / "protocol.json"
    reference_file = SENSITIVITY_DIR / "reference.npz"
    if not protocol_file.exists() or not reference_file.exists():
        raise RuntimeError("请先运行 run_sensitivity_experiments.py --full，生成 MK06 的共同参考前沿与原配置基准。")
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    if (
        protocol.get("model") != "processing_energy_only; quality_loss=1-min_operation_FPY; precedence_enforced"
        or protocol.get("generations") != generations
        or protocol.get("run_seeds") != seeds
        or protocol.get("population") != 60
    ):
        raise RuntimeError("MK06 敏感性实验协议与主实验不一致。")
    for name in ("MK06.xlsx", "main523-mk06-revision.py", "sensitivity_algorithms.py", "run_sensitivity_experiments.py"):
        path = ROOT / name if name.endswith(".xlsx") else WORK_DIR / name
        legacy_names = {
            "sensitivity_algorithms.py": "敏感性算法.py",
            "run_sensitivity_experiments.py": "敏感性实验.py",
        }
        recorded_name = name if name in protocol["source_sha256"] else legacy_names.get(name, name)
        if protocol["source_sha256"].get(recorded_name) != file_hash(path):
            raise RuntimeError(f"MK06 基准使用的 {name} 与当前代码/算例不一致。")
    for seed in seeds:
        data_path = SENSITIVITY_DIR / "reward_original" / f"seed_{seed}.npz"
        meta_path = data_path.with_suffix(".json")
        if not data_path.exists() or not meta_path.exists():
            raise RuntimeError(f"MK06 原配置基准 seed={seed} 未完成；先续跑敏感性实验 --full。")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("generation_record_count") != generations or meta.get("config") != "reward_original":
            raise RuntimeError(f"MK06 原配置基准 seed={seed} 没有完整逐代数据。")
    return protocol_file, reference_file


def load_reference(path):
    with np.load(path) as data:
        return (
            FixedReference(data["front"], data["ideal"], data["nadir"]),
            data["hv_point"].copy(),
        )


def make_protocol(case, full, sources, reference_path):
    generations = 1000 if full else 40
    runs = 20 if full else 1
    return {
        "model": "processing_energy_only; quality_loss=1-min_operation_FPY; precedence_enforced",
        "instance": case,
        "population": 60,
        "generations": generations,
        "run_seeds": [42 + 5 * i for i in range(runs)],
        "algorithms": list(ALGORITHMS),
        "ql_start_generation": max(2, int(generations * 0.3)),
        "ql_decision_period": 10,
        "fixed_ts": {"frequency": 20, "iterations": 30, "tabu_size": 10},
        "glr_ratio": [0.3, 0.2, 0.5],
        "reward": "original",
        "reference_path": str(reference_path.resolve()),
        "reference_source": "MK06 sensitivity reward_original" if case == "MK06" and full else "independent NSGA-II pre-runs",
        "source_sha256": sources,
        "statistics": "HV and IGD use the same fixed reference within each case; population SD",
    }


def build_algorithm(name, problem, reference, hv_point, seed, protocol):
    generations = protocol["generations"]
    if name == "NSGA2":
        return NSGA2(
            pop_size=60, crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20), eliminate_duplicates=True, seed=seed,
        )
    if name == "NSGA2_TS":
        return revision.NSGA2_TS(
            pop_size=60, ts_frequency=20, ts_iterations=30,
            tabu_size=10, seed=seed,
        )
    if name == "NSGA2_TS_QL":
        sampler = FloatRandomSampling()
    elif name == "NSGA2_TS_QL_GLR":
        sampler = revision.FJSPSampling(
            problem.data_loader, gs_ratio=0.3, ls_ratio=0.2, rs_ratio=0.5,
        )
    else:
        raise ValueError(name)
    return SensitivityGLR(
        reward_config=REWARDS["original"], hv_ref_point=hv_point,
        start_gen=protocol["ql_start_generation"],
        decision_period=protocol["ql_decision_period"],
        pop_size=60, seed=seed, max_gen=generations,
        sampling=sampler, ref_front_mgr=reference,
    )


def run_one(problem, reference, hv_point, out_dir, name, seed, protocol):
    run_dir = out_dir / name
    run_dir.mkdir(exist_ok=True)
    data_file = run_dir / f"seed_{seed}.npz"
    meta_file = run_dir / f"seed_{seed}.json"
    if data_file.exists() and meta_file.exists():
        print(f"skip {name} seed={seed}", flush=True)
        return

    random.seed(seed)
    np.random.seed(seed)
    algorithm = build_algorithm(name, problem, reference, hv_point, seed, protocol)
    tracker = MetricsTracker(reference, hv_point)
    started = time.time()
    result = minimize(
        problem, algorithm, ("n_gen", protocol["generations"]),
        seed=seed, verbose=False, callback=tracker,
    )
    elapsed = time.time() - started
    front = np.asarray(result.F, dtype=float)
    if len(tracker.records) != protocol["generations"]:
        raise RuntimeError(f"{name} seed={seed}: 逐代记录不完整。")
    if np.any(front >= hv_point):
        raise RuntimeError(f"{name} seed={seed}: 最终目标超出 HV 参考点。")
    hv = float(Hypervolume(ref_point=hv_point).do(front))
    igd = float(reference.get_igd_metric().compute(front))
    actual = result.algorithm
    meta = {
        "algorithm": name,
        "seed": seed,
        "generations": protocol["generations"],
        "population": len(actual.pop),
        "elapsed_seconds": elapsed,
        "hv": hv,
        "igd": igd,
        "pareto_count": len(front),
        "generation_record_count": len(tracker.records),
    }
    if isinstance(actual, SensitivityGLR):
        meta.update({
            "sampling": actual.initialization.sampling.last_counts if name.endswith("GLR") else "random",
            "reward_clip_count": actual.reward_clip_count,
            "reward_update_count": actual.reward_update_count,
            "reward_clip_rate": actual.reward_clip_count / actual.reward_update_count,
            "ts_events": actual.ts_event_history,
            "actions": actual.ql_action_history,
            "q_table": [
                {"state": state, "action": action, "value": float(value)}
                for (state, action), value in actual.ql.q_table.items()
            ],
            "visit_count": dict(actual.ql.visit_count),
            "q_history": actual.ql.q_history,
        })
        if not actual.ql_action_history or not any(abs(x) > 1e-9 for x in actual.ql.q_table.values()):
            raise RuntimeError(f"{name} seed={seed}: Q 日志缺失。")
        if name.endswith("GLR") and meta["sampling"]["unique"] != 60:
            raise RuntimeError(f"{name} seed={seed}: GLR 初始种群不足 60。")
    temp = data_file.with_suffix(".tmp.npz")
    np.savez_compressed(
        temp, F=front, X=np.asarray(result.X),
        generation_metrics=np.asarray(tracker.records, dtype=float),
    )
    os.replace(temp, data_file)
    atomic_json(meta_file, meta)
    print(f"done {name} seed={seed} HV={hv:.4g} IGD={igd:.4g} time={elapsed:.1f}s", flush=True)


def summarize(out_dir, case, baseline, protocol):
    metrics_rows = []
    trajectory = {}
    summary = [f"# {case} 修回版主实验阶段结果", "",
               f"计划每算法 {len(protocol['run_seeds'])} 次；每次 {protocol['generations']} 代。", "",
               "| 算法 | 已完成 | HV 均值 ± 标准差 | IGD 均值 ± 标准差 |",
               "| --- | ---: | ---: | ---: |"]
    for name in ALGORITHMS:
        folder = baseline if name == "NSGA2_TS_QL_GLR" else out_dir / name
        records = []
        curves = []
        for seed in protocol["run_seeds"]:
            meta_file = folder / f"seed_{seed}.json"
            data_file = folder / f"seed_{seed}.npz"
            if not meta_file.exists() or not data_file.exists():
                continue
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            with np.load(data_file) as data:
                curve = data["generation_metrics"].copy()
            if curve.shape != (protocol["generations"], 4):
                raise RuntimeError(f"{name} seed={seed}: 逐代指标形状错误 {curve.shape}")
            records.append(meta)
            curves.append(curve)
            metrics_rows.append({
                "case": case, "algorithm": name, "seed": seed,
                "hv": meta["hv"], "igd": meta["igd"],
                "pareto_count": meta["pareto_count"],
                "elapsed_seconds": meta["elapsed_seconds"],
            })
        if records:
            hv = np.array([r["hv"] for r in records])
            igd = np.array([r["igd"] for r in records])
            summary.append(
                f"| {name} | {len(records)} | {hv.mean():.5g} ± {hv.std(ddof=0):.5g} "
                f"| {igd.mean():.5g} ± {igd.std(ddof=0):.5g} |"
            )
            trajectory[name] = np.stack(curves)
        else:
            summary.append(f"| {name} | 0 | — | — |")
    summary += ["", "未完成全部种子的结果仅供进度检查，不用于论文结论。", ""]
    (out_dir / "阶段结果.md").write_text("\n".join(summary), encoding="utf-8")
    with (out_dir / "最终指标.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case", "algorithm", "seed", "hv", "igd", "pareto_count", "elapsed_seconds"])
        writer.writeheader()
        writer.writerows(metrics_rows)
    with (out_dir / "逐代汇总.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "algorithm", "generation", "completed_runs", "hv_mean", "hv_sd", "igd_mean", "igd_sd", "nd_mean", "nd_sd"])
        for name, curves in trajectory.items():
            for index in range(curves.shape[1]):
                values = curves[:, index, :]
                writer.writerow([
                    case, name, int(values[0, 0]), len(values),
                    values[:, 1].mean(), values[:, 1].std(ddof=0),
                    values[:, 2].mean(), values[:, 2].std(ddof=0),
                    values[:, 3].mean(), values[:, 3].std(ddof=0),
                ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="每算法 1 次 × 40 代短程检查")
    mode.add_argument("--full", action="store_true", help="每算法 20 次 × 1000 代正式运行")
    parser.add_argument("--case", required=True, choices=CASES)
    args = parser.parse_args()
    case = args.case
    sources = source_hashes(case)
    if args.smoke:
        smoke_id = sha256("".join(sources.values()).encode("ascii")).hexdigest()[:8]
        out_dir = WORK_DIR / f"主实验试运行-{smoke_id}" / case
    else:
        out_dir = WORK_DIR / "主实验正式结果" / case
    out_dir.mkdir(parents=True, exist_ok=True)
    generations = 1000 if args.full else 40
    seeds = [42 + 5 * i for i in range(20 if args.full else 1)]
    if case == "MK06" and args.full:
        _, reference_path = check_mk06_baseline(seeds, generations)
        reference, hv_point = load_reference(reference_path)
    else:
        reference_path = out_dir / "reference.npz"
        data = revision.FJSPDataLoader(str(ROOT / f"{case}.xlsx"))
        problem = revision.FJSPProblem(data)
        reference, hv_point = build_reference(
            problem, out_dir,
            200 if args.full else 5,
            500 if args.full else 100,
        )
    protocol = make_protocol(case, args.full, sources, reference_path)
    protocol_path = out_dir / "protocol.json"
    if protocol_path.exists():
        saved = json.loads(protocol_path.read_text(encoding="utf-8"))
        if saved != protocol:
            raise RuntimeError("该目录已有不同的实验协议；请勿混用新旧运行结果。")
    else:
        atomic_json(protocol_path, protocol)
    if case == "MK06" and args.full:
        data = revision.FJSPDataLoader(str(ROOT / f"{case}.xlsx"))
        problem = revision.FJSPProblem(data)
    baseline = baseline_dir(case, args.full, out_dir)
    for name in ALGORITHMS:
        if case == "MK06" and args.full and name == "NSGA2_TS_QL_GLR":
            print("reuse MK06 reward_original 20 runs", flush=True)
            continue
        for seed in seeds:
            run_one(problem, reference, hv_point, out_dir, name, seed, protocol)
            summarize(out_dir, case, baseline, protocol)
    summarize(out_dir, case, baseline, protocol)


if __name__ == "__main__":
    main()
