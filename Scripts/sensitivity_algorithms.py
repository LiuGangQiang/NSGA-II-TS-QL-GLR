"""供 MK06 敏感性实验使用的可配置 QL-TS 算法。投稿脚本保持只读。"""

from dataclasses import dataclass
import importlib.util
from collections import deque
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


REVISION_PATH = Path(__file__).with_name("main523-mk06-revision.py")
_spec = importlib.util.spec_from_file_location("fj_revision", REVISION_PATH)
revision = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(revision)


@dataclass(frozen=True)
class RewardConfig:
    name: str
    hv: float
    igd: float
    nd: float
    synergy: float = 0.15
    hv_up_igd_down: float = -0.08
    hv_down_igd_up: float = -0.05
    improve_bonus: float = 2.0
    failed_ts_penalty: float = -1.5
    no_ts_factor: float = 0.5
    # 投稿版 [-2, 2] 在短程检查中使全部奖励饱和，修回版放宽截断并记录实际截断率。
    clip: float = 50.0


REWARDS = {
    "original": RewardConfig("original", 10.0, 8.0, 2.0),
    # 三项主权重之和保持为 20，仅改变相对份额。
    "equal": RewardConfig("equal", 20 / 3, 20 / 3, 20 / 3),
    # 单指标组仅保留指定指标；其它奖励项及交互项全部关闭。
    "hv_only": RewardConfig(
        "hv_only", 20.0, 0.0, 0.0,
        synergy=0, hv_up_igd_down=0, hv_down_igd_up=0,
        improve_bonus=0, failed_ts_penalty=0, no_ts_factor=1,
    ),
    "igd_only": RewardConfig(
        "igd_only", 0.0, 20.0, 0.0,
        synergy=0, hv_up_igd_down=0, hv_down_igd_up=0,
        improve_bonus=0, failed_ts_penalty=0, no_ts_factor=1,
    ),
}


class SensitivityGLR(revision.NSGA2_TS_QL_GLR):
    """固定 10 代 Q 决策间隔；所选 TS 频率按真实代数控制执行。"""

    def __init__(
        self, *, reward_config, hv_ref_point, start_gen, decision_period=10,
        pop_size=60, seed=None, max_gen=1000, sampling=None, ref_front_mgr=None,
    ):
        super().__init__(
            pop_size=pop_size, seed=seed, max_gen=max_gen,
            sampling=sampling, ref_front_mgr=ref_front_mgr,
        )
        if reward_config.name not in REWARDS:
            raise ValueError("Unknown reward configuration")
        if start_gen < 2 or start_gen >= max_gen or decision_period < 1:
            raise ValueError("Invalid QL start or decision interval")
        self.reward_config = reward_config
        self.hv_ref_point = np.asarray(hv_ref_point, dtype=float)
        self.start_gen = start_gen
        self.ql_decision_period = decision_period
        self._active = None
        self._last_ql_decision_gen = -1
        self._last_ts_gen = 0
        self.ts_event_history = []
        self.reward_clip_count = 0
        self.reward_update_count = 0

    def _population_metrics(self):
        F = self.pop.get("F")
        nd_indices = NonDominatedSorting().do(F, only_non_dominated_front=True)
        front = F[nd_indices]
        hv = float(Hypervolume(ref_point=self.hv_ref_point).do(front))
        igd = float(self.igd_metric.compute(front)) if self.igd_metric is not None else float("inf")
        return hv, igd, len(nd_indices)

    def _apply_ts(self, config):
        F = self.pop.get("F")
        nd_indices = NonDominatedSorting().do(F, only_non_dominated_front=True)[:5]
        self.tabu_size = config["tabu"]
        self.ts_iterations = config["iter"]
        improved_count = 0

        for index in nd_indices:
            x_best = self.pop[index].X.copy()
            f_best = self.pop[index].F.copy()
            for _ in range(self.ts_iterations):
                candidate = self._local_search(x_best, self.problem)
                candidate_hash = self._solution_hash(candidate)
                if candidate_hash in self.tabu_list:
                    continue
                candidate_f = self.problem.evaluate(candidate.reshape(1, -1))[0]
                if np.all(candidate_f <= f_best) and np.any(candidate_f < f_best):
                    x_best, f_best = candidate, candidate_f
                    self.tabu_list.append(candidate_hash)
                    improved_count += 1
            self.pop[index].X = x_best
            self.pop[index].F = f_best

        # TS 改写目标值后重算 rank/crowding，供下一代选择及状态判定使用。
        self.pop = self.survival.do(
            self.problem, self.pop, n_survive=len(self.pop), algorithm=self
        )
        self._last_ts_gen = self.n_gen
        self.ts_count += 1
        self._active["improved_count"] += improved_count
        self._active["ts_calls"] += 1
        self.ts_event_history.append({
            "generation": self.n_gen,
            "action": self._active["action"],
            "frequency": config["freq"],
            "iterations": config["iter"],
            "improved_count": improved_count,
        })

    def _finish_action(self, next_state, after):
        active = self._active
        if active is None:
            return
        before_hv, before_igd, before_nd = active["before"]
        after_hv, after_igd, after_nd = after
        hv_pct = (after_hv - before_hv) / (abs(before_hv) + 1e-9) * 100 if before_hv > 0 else 0.0
        igd_pct = (
            (before_igd - after_igd) / (abs(before_igd) + 1e-9) * 100
            if np.isfinite(before_igd) and before_igd > 0 else 0.0
        )
        nd_delta = after_nd - before_nd
        self._hv_delta_history.append(hv_pct)
        self._igd_delta_history.append(igd_pct)
        hv_norm = self._normalize_reward_component(hv_pct, self._hv_delta_history, 10.0)
        igd_norm = self._normalize_reward_component(igd_pct, self._igd_delta_history, 5.0)
        cfg = self.reward_config
        raw = cfg.hv * hv_norm + cfg.igd * igd_norm + cfg.nd * nd_delta
        if active["action"] == "ts_none":
            raw *= cfg.no_ts_factor
        elif active["improved_count"] > 0:
            raw += cfg.improve_bonus * active["improved_count"]
        elif active["ts_calls"] > 0:
            raw += cfg.failed_ts_penalty
        if hv_pct > 0 and igd_pct > 0:
            raw += cfg.synergy
        elif hv_pct > 0 and igd_pct < 0:
            raw += cfg.hv_up_igd_down
        elif hv_pct < 0 and igd_pct > 0:
            raw += cfg.hv_down_igd_up

        reward = float(np.clip(raw, -cfg.clip, cfg.clip))
        clipped = abs(raw) > cfg.clip
        self.reward_update_count += 1
        self.reward_clip_count += int(clipped)
        self.ql.update(active["state"], active["action"], reward, next_state)
        self.ql.decay()
        self.ql_action_history.append({
            "generation": active["generation"],
            "end_generation": self.n_gen,
            "state": active["state"],
            "next_state": next_state,
            "action": active["action"],
            "reward": reward,
            "raw_reward": float(raw),
            "clipped": bool(clipped),
            "hv_delta_pct": float(hv_pct),
            "igd_delta_pct": float(igd_pct),
            "nd_delta": int(nd_delta),
            "improved_count": int(active["improved_count"]),
            "ts_calls": int(active["ts_calls"]),
            "hv_before": float(before_hv),
            "hv_after": float(after_hv),
            "igd_before": float(before_igd),
            "igd_after": float(after_igd),
            "epsilon": float(self.ql.epsilon),
        })
        self._active = None

    def _advance(self, infills=None, **kwargs):
        # 直接执行 NSGA-II 世代更新，避免调用投稿版中有频率错误的重载。
        NSGA2._advance(self, infills=infills, **kwargs)
        if self.n_gen >= self.start_gen:
            decision_due = (
                self._active is None
                or self.n_gen - self._last_ql_decision_gen >= self.ql_decision_period
            )
            if decision_due:
                metrics = self._population_metrics()
                state = self._get_state()
                self._finish_action(state, metrics)
                if self.n_gen < self.max_gen:
                    action = self.ql.select_action(state)
                    self._active = {
                        "generation": self.n_gen,
                        "state": state,
                        "action": action,
                        "before": metrics,
                        "improved_count": 0,
                        "ts_calls": 0,
                    }
                    self._last_ql_decision_gen = self.n_gen
                    if self._last_ts_gen == 0 and action != "ts_none":
                        self._last_ts_gen = self.n_gen - self.ts_configs[action]["freq"]
                    self.tabu_list = deque(maxlen=self.ts_configs[action]["tabu"] or 1)

            if self._active is not None and self._active["action"] != "ts_none":
                config = self.ts_configs[self._active["action"]]
                if self.n_gen - self._last_ts_gen >= config["freq"]:
                    self._apply_ts(config)

            if self.n_gen >= self.max_gen and self._active is not None:
                self._finish_action(self._get_state(), self._population_metrics())

        self.ql.record_q_stats(self.n_gen)
