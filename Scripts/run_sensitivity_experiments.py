"""MK06 奖励权重与 GLR 比例敏感性实验。

短程核验：python run_sensitivity_experiments.py --smoke
正式实验：python run_sensitivity_experiments.py --full
投稿脚本和原始结果始终保持不变。正式实验预计为 6 配置 × 20 种子。
"""

import argparse
import csv
from dataclasses import asdict
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
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from sensitivity_algorithms import REWARDS, SensitivityGLR, revision


WORK_DIR = Path(__file__).resolve().parent
ROOT = WORK_DIR.parent
CASE_PATH = ROOT / "MK06.xlsx"
SOURCE_FILES = [
    CASE_PATH,
    WORK_DIR / "main523-mk06-revision.py",
    WORK_DIR / "sensitivity_algorithms.py",
    Path(__file__).resolve(),
]
CONFIGS = [
    ("reward_original", "original", (0.3, 0.2, 0.5)),
    ("reward_equal", "equal", (0.3, 0.2, 0.5)),
    ("reward_hv_only", "hv_only", (0.3, 0.2, 0.5)),
    ("reward_igd_only", "igd_only", (0.3, 0.2, 0.5)),
    ("ratio_2_3_5", "original", (0.2, 0.3, 0.5)),
    ("ratio_4_2_4", "original", (0.4, 0.2, 0.4)),
]


class FixedReference:
    def __init__(self, front, ideal, nadir):
        self.front = np.asarray(front, dtype=float)
        self.ideal = np.asarray(ideal, dtype=float)
        self.nadir = np.asarray(nadir, dtype=float)

    def get_igd_metric(self):
        return revision.IGDMetric(self.front, self.ideal, self.nadir)


class MetricsTracker:
    """逐代记录固定参考口径下的指标，供 20 次收敛曲线使用。"""

    def __init__(self, fixed_reference, hv_point):
        self.hv_indicator = Hypervolume(ref_point=hv_point)
        self.igd_indicator = fixed_reference.get_igd_metric()
        self.records = []

    def __call__(self, algorithm):
        F = algorithm.pop.get("F")
        indices = NonDominatedSorting().do(F, only_non_dominated_front=True)
        front = F[indices]
        self.records.append((
            algorithm.n_gen,
            float(self.hv_indicator.do(front)),
            float(self.igd_indicator.compute(front)),
            len(front),
        ))


def file_hash(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def build_reference(problem, out_dir, pre_generations, random_count):
    """独立预实验建立六组共用的固定 IGD 前沿与 HV 参考点。"""
    ref_file = out_dir / "reference.npz"
    if ref_file.exists():
        with np.load(ref_file) as saved:
            return FixedReference(saved["front"], saved["ideal"], saved["nadir"]), saved["hv_point"].copy()

    np.random.seed(9100)
    random_points = problem.evaluate(np.random.random((random_count, problem.n_var)))
    pool = [random_points]
    for seed in (9101, 9102, 9103):
        random.seed(seed)
        np.random.seed(seed)
        algorithm = NSGA2(pop_size=60, eliminate_duplicates=True, seed=seed)
        result = minimize(problem, algorithm, ("n_gen", pre_generations), seed=seed, verbose=False)
        pool.append(np.asarray(result.F))

    all_points = np.vstack(pool)
    front_index = NonDominatedSorting().do(all_points, only_non_dominated_front=True)
    front = np.unique(all_points[front_index], axis=0)
    ideal = np.min(all_points, axis=0) * 0.9
    nadir = np.max(all_points, axis=0) * 1.1
    nadir = np.maximum(nadir, ideal + 1e-9)
    hv_point = np.max(all_points, axis=0) * 1.25
    np.savez_compressed(ref_file, front=front, ideal=ideal, nadir=nadir, hv_point=hv_point)
    return FixedReference(front, ideal, nadir), hv_point


def load_or_create_protocol(out_dir, generations, runs, pre_generations, random_count):
    sources = {p.name: file_hash(p) for p in SOURCE_FILES}
    protocol = {
        "model": "processing_energy_only; quality_loss=1-min_operation_FPY; precedence_enforced",
        "instance": "MK06",
        "population": 60,
        "generations": generations,
        "run_seeds": [42 + 5 * i for i in range(runs)],
        "ql_start_generation": max(2, int(generations * 0.3)),
        "ql_decision_period": 10,
        "pre_generations": pre_generations,
        "reference_random_count": random_count,
        "reward_definitions": {name: asdict(config) for name, config in REWARDS.items()},
        "configs": [
            {"name": name, "reward": reward, "ratio": list(ratio)}
            for name, reward, ratio in CONFIGS
        ],
        "source_sha256": sources,
        "statistics": "HV/IGD share one fixed pre-run reference across all configurations; population SD",
    }
    path = out_dir / "protocol.json"
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != protocol:
            raise RuntimeError("Output directory has a different experiment protocol; use a new directory")
    else:
        atomic_json(path, protocol)
    return protocol


def run_one(problem, fixed_reference, hv_point, out_dir, config, seed, protocol):
    name, reward_name, ratio = config
    run_dir = out_dir / name
    run_dir.mkdir(exist_ok=True)
    data_file = run_dir / f"seed_{seed}.npz"
    log_file = run_dir / f"seed_{seed}.json"
    if data_file.exists() and log_file.exists():
        print(f"skip {name} seed={seed}")
        return

    random.seed(seed)
    np.random.seed(seed)
    sampler = revision.FJSPSampling(
        problem.data_loader, gs_ratio=ratio[0], ls_ratio=ratio[1], rs_ratio=ratio[2]
    )
    algorithm = SensitivityGLR(
        reward_config=REWARDS[reward_name], hv_ref_point=hv_point,
        start_gen=protocol["ql_start_generation"],
        decision_period=protocol["ql_decision_period"],
        pop_size=60, seed=seed, max_gen=protocol["generations"],
        sampling=sampler, ref_front_mgr=fixed_reference,
    )
    tracker = MetricsTracker(fixed_reference, hv_point) if name == "reward_original" else None
    started = time.time()
    run_kwargs = {"seed": seed, "verbose": False}
    if tracker is not None:
        run_kwargs["callback"] = tracker
    result = minimize(
        problem, algorithm, ("n_gen", protocol["generations"]),
        **run_kwargs,
    )
    elapsed = time.time() - started
    actual = result.algorithm
    front = np.asarray(result.F, dtype=float)
    decisions = actual.ql_action_history
    counts = actual.initialization.sampling.last_counts
    if len(actual.pop) != 60 or counts["unique"] != 60:
        raise RuntimeError(f"Incorrect initial population for {name} seed={seed}: {counts}")
    if not decisions or not any(abs(x) > 1e-9 for x in actual.ql.q_table.values()):
        raise RuntimeError(f"Missing Q learning records for {name} seed={seed}")
    if np.any(front >= hv_point):
        raise RuntimeError("HV reference point does not dominate the final front; rebuild protocol")

    hv = float(Hypervolume(ref_point=hv_point).do(front))
    igd = float(fixed_reference.get_igd_metric().compute(front))
    meta = {
        "config": name,
        "seed": seed,
        "generations": protocol["generations"],
        "population": len(actual.pop),
        "sampling": counts,
        "elapsed_seconds": elapsed,
        "hv": hv,
        "igd": igd,
        "pareto_count": len(front),
        "generation_record_count": len(tracker.records) if tracker else 0,
        "reward_clip_count": actual.reward_clip_count,
        "reward_update_count": actual.reward_update_count,
        "reward_clip_rate": actual.reward_clip_count / actual.reward_update_count,
        "ts_events": actual.ts_event_history,
        "actions": decisions,
        "q_table": [
            {"state": state, "action": action, "value": float(value)}
            for (state, action), value in actual.ql.q_table.items()
        ],
        "visit_count": dict(actual.ql.visit_count),
        "q_history": actual.ql.q_history,
    }
    temp_data = data_file.with_suffix(".tmp.npz")
    np.savez_compressed(
        temp_data,
        F=front,
        X=np.asarray(result.X),
        generation_metrics=np.asarray(tracker.records, dtype=float) if tracker else np.empty((0, 4)),
    )
    os.replace(temp_data, data_file)
    atomic_json(log_file, meta)
    print(
        f"done {name} seed={seed} HV={hv:.4g} IGD={igd:.4g} "
        f"Q_updates={len(decisions)} TS={len(actual.ts_event_history)} "
        f"clip={meta['reward_clip_rate']:.1%}"
    )


def summarize(out_dir, protocol):
    rows = [
        "# MK06 敏感性实验阶段结果", "",
        f"模型：{protocol['model']}。",
        f"每组计划 {len(protocol['run_seeds'])} 次；六组共用原配置。",
        "结果仅在相同 `protocol.json` 与 `reference.npz` 下可直接比较。", "",
        "| 配置 | 已完成 | HV 均值 ± 标准差 | IGD 均值 ± 标准差 | 奖励截断比例 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    metric_rows = []
    original_curves = []
    for name, _reward, _ratio in CONFIGS:
        records = []
        for seed in protocol["run_seeds"]:
            path = out_dir / name / f"seed_{seed}.json"
            if path.exists():
                data_path = path.with_suffix(".npz")
                if not data_path.exists():
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                records.append(record)
                metric_rows.append({
                    "config": name, "seed": seed, "hv": record["hv"],
                    "igd": record["igd"], "pareto_count": record["pareto_count"],
                    "elapsed_seconds": record["elapsed_seconds"],
                    "reward_clip_rate": record["reward_clip_rate"],
                })
                if name == "reward_original":
                    with np.load(data_path) as data:
                        curve = data["generation_metrics"].copy()
                    if curve.shape != (protocol["generations"], 4):
                        raise RuntimeError(f"reward_original seed={seed}: 逐代指标形状错误 {curve.shape}")
                    original_curves.append(curve)
        if not records:
            rows.append(f"| {name} | 0 | — | — | — |")
            continue
        hv = np.array([record["hv"] for record in records])
        igd = np.array([record["igd"] for record in records])
        clip = np.mean([record["reward_clip_rate"] for record in records])
        rows.append(
            f"| {name} | {len(records)} | {hv.mean():.5g} ± {hv.std(ddof=0):.5g} "
            f"| {igd.mean():.5g} ± {igd.std(ddof=0):.5g} | {clip:.1%} |"
        )
    rows += ["", "未完成 20 次的组仅为阶段检查结果，不用于论文结论。", ""]
    (out_dir / "阶段结果.md").write_text("\n".join(rows), encoding="utf-8")
    with (out_dir / "最终指标.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "config", "seed", "hv", "igd", "pareto_count", "elapsed_seconds", "reward_clip_rate",
        ])
        writer.writeheader()
        writer.writerows(metric_rows)
    if original_curves:
        curves = np.stack(original_curves)
        with (out_dir / "原配置逐代汇总.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["generation", "completed_runs", "hv_mean", "hv_sd", "igd_mean", "igd_sd", "nd_mean", "nd_sd"])
            for index in range(curves.shape[1]):
                values = curves[:, index, :]
                writer.writerow([
                    int(values[0, 0]), len(values),
                    values[:, 1].mean(), values[:, 1].std(ddof=0),
                    values[:, 2].mean(), values[:, 2].std(ddof=0),
                    values[:, 3].mean(), values[:, 3].std(ddof=0),
                ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="六组各 1 次短程核验")
    mode.add_argument("--full", action="store_true", help="六组各 20 次正式运行")
    args = parser.parse_args()
    if args.smoke:
        generations, runs, pre_generations, random_count = 40, 1, 5, 100
        smoke_id = sha256(
            "".join(file_hash(path) for path in SOURCE_FILES).encode("ascii")
        ).hexdigest()[:8]
        out_dir = WORK_DIR / f"敏感性试运行-{smoke_id}"
    else:
        generations, runs, pre_generations, random_count = 1000, 20, 200, 500
        out_dir = ROOT / "sensitivity_results"
    out_dir.mkdir(exist_ok=True)
    protocol = load_or_create_protocol(
        out_dir, generations, runs, pre_generations, random_count
    )
    data = revision.FJSPDataLoader(str(CASE_PATH))
    problem = revision.FJSPProblem(data)
    fixed_reference, hv_point = build_reference(
        problem, out_dir, pre_generations, random_count
    )
    for config in CONFIGS:
        for seed in protocol["run_seeds"]:
            run_one(problem, fixed_reference, hv_point, out_dir, config, seed, protocol)
    summarize(out_dir, protocol)


if __name__ == "__main__":
    main()
