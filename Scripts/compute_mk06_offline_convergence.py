"""确定性复跑 MK06 四算法，逐代按统一评价参考前沿计算 IGD。"""

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import random

import numpy as np
from pymoo.indicators.hv import Hypervolume
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from Scripts.run_main_experiments import ALGORITHMS, build_algorithm, load_reference
from run_sensitivity_experiments import atomic_json
from sensitivity_algorithms import revision
from 固定TS修正重跑 import CorrectedFixedTS


WORK = Path(__file__).resolve().parent.parent
ROOT = WORK
OUT = WORK / "evaluation_results" / "MK06离线逐代"
SEEDS = [42 + 5 * i for i in range(20)]


def original_path(name, seed):
    if name == "NSGA2_TS":
        return WORK / "固定TS修正结果" / "MK06" / f"seed_{seed}.npz"
    if name == "NSGA2_TS_QL_GLR":
        return WORK / "sensitivity_results" / "reward_original" / f"seed_{seed}.npz"
    return WORK / "主实验正式结果" / "MK06" / name / f"seed_{seed}.npz"


class OfflineTracker:
    def __init__(self, front, ideal, nadir, hv_point):
        self.igd = revision.IGDMetric(front, ideal, nadir)
        self.hv = Hypervolume(ref_point=hv_point)
        self.records = []

    def __call__(self, algorithm):
        F = algorithm.pop.get("F")
        indices = NonDominatedSorting().do(F, only_non_dominated_front=True)
        nd = F[indices]
        self.records.append((
            algorithm.n_gen, float(self.hv.do(nd)),
            float(self.igd.compute(nd)), len(nd),
        ))


def protocol():
    main = json.loads((WORK / "主实验正式结果" / "MK06" / "protocol.json").read_text(encoding="utf-8"))
    fixed = json.loads((WORK / "固定TS修正结果" / "MK06" / "protocol.json").read_text(encoding="utf-8"))
    if main["run_seeds"] != SEEDS or main["generations"] != 1000 or fixed["run_seeds"] != SEEDS:
        raise RuntimeError("MK06 主实验与修正固定 TS 的协议不一致")
    paths = [
        WORK / "主实验正式结果" / "MK06" / "protocol.json",
        WORK / "固定TS修正结果" / "MK06" / "protocol.json",
        WORK / "evaluation_results" / "MK06_经验参考前沿.npz",
        Path(__file__).resolve(),
    ]
    value = {
        "case": "MK06", "algorithms": list(ALGORITHMS), "seeds": SEEDS,
        "generations": 1000, "method": "replay saved seeds and evaluate each generation against offline empirical reference",
        "source_sha256": {str(p.relative_to(WORK)): sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError("MK06 离线逐代结果目录的协议不同")
    else:
        atomic_json(path, value)
    return main


def run_one(name, seed, problem, main_protocol, online_ref, online_hv, offline):
    folder = OUT / name
    folder.mkdir(exist_ok=True)
    out = folder / f"seed_{seed}.npz"
    if out.exists():
        print(f"skip {name} seed={seed}", flush=True)
        return
    random.seed(seed)
    np.random.seed(seed)
    if name == "NSGA2_TS":
        algorithm = CorrectedFixedTS(
            pop_size=60, ts_frequency=20, ts_iterations=30, tabu_size=10, seed=seed,
        )
    else:
        algorithm = build_algorithm(name, problem, online_ref, online_hv, seed, main_protocol)
    tracker = OfflineTracker(*offline)
    result = minimize(problem, algorithm, ("n_gen", 1000), seed=seed, verbose=False, callback=tracker)
    if len(tracker.records) != 1000:
        raise RuntimeError(f"{name} seed={seed}: 缺少逐代记录")
    with np.load(original_path(name, seed)) as data:
        original = np.unique(data["F"], axis=0)
    replayed = np.unique(np.asarray(result.F), axis=0)
    if original.shape != replayed.shape or not np.array_equal(original, replayed):
        raise RuntimeError(f"{name} seed={seed}: 复跑最终前沿与原始运行不同，不能使用逐代曲线")
    curve = np.asarray(tracker.records, dtype=float)
    if not np.array_equal(curve[:, 0], np.arange(1, 1001)):
        raise RuntimeError(f"{name} seed={seed}: 代数不连续")
    temp = out.with_suffix(".tmp.npz")
    np.savez_compressed(temp, generation_metrics=curve)
    os.replace(temp, out)
    print(f"done {name} seed={seed} offline_IGD={curve[-1, 2]:.6g}", flush=True)


def summarize():
    path = OUT / "逐代汇总.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["algorithm", "generation", "completed_runs", "hv_mean", "hv_sd", "igd_offline_mean", "igd_offline_sd", "nd_mean", "nd_sd"])
        for name in ALGORITHMS:
            curves = []
            for seed in SEEDS:
                file = OUT / name / f"seed_{seed}.npz"
                if file.exists():
                    with np.load(file) as data:
                        curves.append(data["generation_metrics"].copy())
            if not curves:
                continue
            arr = np.stack(curves)
            for i in range(1000):
                x = arr[:, i]
                writer.writerow([name, int(x[0, 0]), len(x),
                                 x[:, 1].mean(), x[:, 1].std(ddof=0),
                                 x[:, 2].mean(), x[:, 2].std(ddof=0),
                                 x[:, 3].mean(), x[:, 3].std(ddof=0)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", required=True, choices=ALGORITHMS)
    parser.add_argument("--seed", type=int, choices=SEEDS, help="只复跑指定种子；默认全部 20 个")
    args = parser.parse_args()
    main_protocol = protocol()
    online_ref, online_hv = load_reference(WORK / "sensitivity_results" / "reference.npz")
    with np.load(WORK / "evaluation_results" / "MK06_经验参考前沿.npz") as data:
        offline = tuple(data[key].copy() for key in ("front", "ideal", "nadir", "hv_point"))
    problem = revision.FJSPProblem(revision.FJSPDataLoader(str(ROOT / "MK06.xlsx")))
    for seed in [args.seed] if args.seed is not None else SEEDS:
        run_one(args.algorithm, seed, problem, main_protocol, online_ref, online_hv, offline)


if __name__ == "__main__":
    main()
