"""
=============================================================================
多目标FJSP优化：四算法对比实验（预构建静态参考前沿 + 归一化IGD + HV/IGD双指标QL奖励版）
NSGA-II | NSGA-II-TS | NSGA-II-TS-QL | NSGA-II-TS-QL-GLR
=============================================================================
"""
import os
import sys

import pandas as pd
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Dict
from collections import defaultdict, deque
from scipy.stats import wilcoxon

# pymoo 核心
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2, binary_tournament
from pymoo.optimize import minimize
from pymoo.indicators.hv import Hypervolume
from pymoo.util.display.display import Display
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.termination import get_termination
from pymoo.termination.collection import TerminationCollection

# 算子
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover as SBX
from pymoo.operators.mutation.pm import PolynomialMutation as PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection

g_max_gen = 200
g_start_gen = 1


# ============================== 0. 获取输出子目录名 ==============================
def get_output_dir() -> str:
    script_name = os.path.basename(sys.argv[0])
    if script_name.endswith('.py'):
        return script_name[:-3]
    return script_name


# ============================== 1. 时间解析函数 ==============================
def parse_time_limit(time_str):
    time_str = str(time_str).strip()
    if time_str.isdigit():
        return int(time_str)
    if ':' in time_str:
        return time_str
    total_seconds = 0
    i = 0
    n = len(time_str)
    while i < n:
        if time_str[i].isdigit():
            j = i
            while j < n and time_str[j].isdigit():
                j += 1
            num = int(time_str[i:j])
            i = j
            if i < n:
                unit = time_str[i]
                if unit == 's':
                    total_seconds += num
                elif unit == 'm':
                    total_seconds += num * 60
                elif unit == 'h':
                    total_seconds += num * 3600
                i += 1
            else:
                total_seconds += num
        else:
            i += 1
    return total_seconds


# ============================== 2. 指标计算函数 ==============================
def calc_igd(F: np.ndarray, ref_front: np.ndarray) -> float:
    """原始IGD（无归一化），用于非QL算法或备用场景"""
    if len(F) == 0 or len(ref_front) == 0:
        return float('inf')
    dists = []
    for r in ref_front:
        min_d = np.min(np.linalg.norm(F - r, axis=1))
        dists.append(min_d)
    return float(np.mean(dists))


def calc_sp(F: np.ndarray) -> float:
    if len(F) <= 1:
        return 0.0
    n = len(F)
    dists = []
    for i in range(n):
        other_idx = [j for j in range(n) if j != i]
        if not other_idx:
            continue
        min_d = np.min(np.linalg.norm(F[other_idx] - F[i], axis=1))
        dists.append(min_d)
    if len(dists) == 0:
        return 0.0
    d_mean = np.mean(dists)
    sp = np.sqrt(np.sum((np.array(dists) - d_mean) ** 2) / len(dists))
    return float(sp)


# ============================== 2.5 预构建静态参考前沿 + 归一化IGD ==============================
class StaticReferenceFront:
    """
    预构建静态参考前沿：合并预实验解 → 非支配筛选 → 去重 → 插值加密
    所有 IGD 计算共享同一组 ideal / nadir，确保归一化尺度统一
    【修复1】ideal/nadir 基于所有原始解（含支配解）并留 10% 余量
    """
    def __init__(self, solutions: List[np.ndarray], target_size: int = 1000, precision: float = 1e-6):
        self.precision = precision
        self.target_size = target_size
        
        # 【修复1】保存所有原始解用于确定边界（不仅仅是非支配解）
        all_arr = np.array(solutions)
        
        # 1. 提取非支配解（用于参考前沿本身）
        raw_front = self._extract_pareto_front(solutions)
        # 2. 去重
        unique_front = self._unique_by_objectives(raw_front)
        # 3. 插值加密
        self.front = self._densify(unique_front, target_size)
        
        # 4. 【修复1】ideal/nadir 基于所有原始解，并留余量
        # 防止后续算法找到更优解时超出归一化范围
        self.ideal = np.min(all_arr, axis=0) * 0.90   # 向下留 10% 余量
        self.nadir = np.max(all_arr, axis=0) * 1.10   # 向上留 10% 余量
        
        # 安全处理：确保 nadir > ideal（防止某维度所有值相同）
        mask = self.nadir <= self.ideal
        self.nadir[mask] = self.ideal[mask] + 1.0
        
        print(f"[StaticReferenceFront] 构建完成: "
              f"原始非支配解 {len(unique_front)} 个, 加密后 {len(self.front)} 个")
        print(f"  边界基于 {len(all_arr)} 个原始解（含支配解）")
        print(f"  Ideal: {self.ideal}")
        print(f"  Nadir: {self.nadir}")

    def _extract_pareto_front(self, solutions: List[np.ndarray]) -> np.ndarray:
        if len(solutions) == 0:
            return np.empty((0, 3))
        objs = np.array(solutions)
        n = len(objs)
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if is_dominated[i]:
                continue
            dominated = np.all(objs[i] <= objs, axis=1) & np.any(objs[i] < objs, axis=1)
            is_dominated[dominated] = True
        front = objs[~is_dominated]
        return front

    def _unique_by_objectives(self, front: np.ndarray) -> np.ndarray:
        if len(front) == 0:
            return front
        rounded = np.round(front / self.precision) * self.precision
        _, unique_idx = np.unique(rounded, axis=0, return_index=True)
        return front[np.sort(unique_idx)]

    def _densify(self, front: np.ndarray, target_size: int) -> np.ndarray:
        """在相邻非支配解之间线性插值，并维持非支配性"""
        if len(front) == 0:
            return front
        if len(front) >= target_size:
            idx = np.linspace(0, len(front)-1, target_size, dtype=int)
            return front[idx]
        
        current = front.copy()
        max_iter = 20000
        for _ in range(max_iter):
            if len(current) >= target_size:
                break
            if len(current) < 2:
                break
            # 按第一目标排序，找相邻欧氏距离最大的一对
            sorted_idx = np.argsort(current[:, 0])
            sorted_current = current[sorted_idx]
            diffs = np.linalg.norm(sorted_current[1:] - sorted_current[:-1], axis=1)
            if len(diffs) == 0:
                break
            max_idx = np.argmax(diffs)
            new_point = (sorted_current[max_idx] + sorted_current[max_idx + 1]) / 2.0
            current = np.vstack([current, new_point])
            # 重新筛选非支配（插值点可能被支配）
            current = self._extract_pareto_front(current.tolist())
        
        if len(current) > target_size:
            idx = np.linspace(0, len(current)-1, target_size, dtype=int)
            current = current[idx]
        return current

    def get_front(self) -> np.ndarray:
        return self.front

    def get_ideal_nadir(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.ideal, self.nadir

    def get_igd_metric(self):
        """返回绑定本参考前沿的归一化IGD计算器"""
        return IGDMetric(self.front, self.ideal, self.nadir)


class IGDMetric:
    """
    带目标空间归一化的 IGD 计算器。
    初始化时即固定参考前沿的归一化版本，避免重复计算。
    """
    def __init__(self, ref_front: np.ndarray, ideal: np.ndarray, nadir: np.ndarray):
        self.ideal = ideal
        self.nadir = nadir
        # 预先归一化参考前沿（修复：传入 ideal 和 nadir）
        self.ref_front_norm = self._normalize(ref_front, ideal, nadir)

    @staticmethod
    def _normalize(objs: np.ndarray, ideal: np.ndarray, nadir: np.ndarray) -> np.ndarray:
        return (objs - ideal) / (nadir - ideal + 1e-9)

    def compute(self, solutions: np.ndarray) -> float:
        """计算当前解集（原始尺度）到参考前沿的归一化 IGD"""
        if len(solutions) == 0:
            return 1e6
        # 使用同一组 ideal/nadir 归一化当前解集
        sol_norm = self._normalize(solutions, self.ideal, self.nadir)
        # 向量化计算：每个参考点到最近解的距离
        dists = np.linalg.norm(self.ref_front_norm[:, None, :] - sol_norm[None, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        return float(np.mean(min_dists))


# ============================== 3. 数据层（动态适配版） ==============================
class FJSPDataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.proc_times = {}
        self.production_power = {}
        self.ftt = {}
        self.machine_idle_power = {}

        self.n_jobs = 0
        self.n_machines = 0
        self.operations_per_job = []
        self.total_operations = 0

        self._load_data()

    def _load_data(self):
        df_proc = pd.read_excel(self.file_path, sheet_name='Processing times', header=0)
        df_power = pd.read_excel(self.file_path, sheet_name='Production power', header=0)
        df_ftt = pd.read_excel(self.file_path, sheet_name='FTT', header=0)
        df_other = pd.read_excel(self.file_path, sheet_name='Other power', header=0)

        machine_cols = [c for c in df_proc.columns if c.startswith('M') and c[1:].isdigit()]
        self.n_machines = len(machine_cols)

        self.n_jobs = int(df_proc['JOB'].max())
        self.operations_per_job = []
        for j in range(1, self.n_jobs + 1):
            max_op = int(df_proc[df_proc['JOB'] == j]['OPERATION'].max())
            self.operations_per_job.append(max_op)
        self.total_operations = sum(self.operations_per_job)

        for i in range(self.n_machines):
            col_name = f'M{i+1}'
            self.machine_idle_power[i] = float(df_other[col_name].iloc[0]) if col_name in df_other.columns else 1.0

        power_dict = {}
        for _, row in df_power.iterrows():
            key = (int(row['JOB']), int(row['OPERATION']))
            power_dict[key] = row

        ftt_dict = {}
        for _, row in df_ftt.iterrows():
            key = (int(row['JOB']), int(row['OPERATION']))
            ftt_dict[key] = row

        for _, row in df_proc.iterrows():
            job = int(row['JOB']) - 1
            op = int(row['OPERATION']) - 1
            key = (int(row['JOB']), int(row['OPERATION']))
            power_row = power_dict.get(key)
            ftt_row = ftt_dict.get(key)

            for m in range(self.n_machines):
                col_name = f'M{m+1}'
                if col_name not in row:
                    continue
                time_val = row[col_name]
                if pd.notna(time_val) and time_val < 999:
                    self.proc_times[(job, op, m)] = float(time_val)
                    if power_row is not None and col_name in power_row:
                        self.production_power[(job, op, m)] = float(power_row[col_name])
                    else:
                        self.production_power[(job, op, m)] = 1.0
                    if ftt_row is not None and col_name in ftt_row:
                        self.ftt[(job, op, m)] = float(ftt_row[col_name])
                    else:
                        self.ftt[(job, op, m)] = 1.0

    def get_available_machines(self, job: int, op: int) -> List[int]:
        return [m for m in range(self.n_machines) if (job, op, m) in self.proc_times]


# ============================== 4. 编解码层 ==============================
class FJSPCodec:
    def __init__(self, data: FJSPDataLoader):
        self.data = data

    def decode(self, x: np.ndarray) -> Tuple[float, float, float, Dict]:
        n_ops = self.data.total_operations
        os_section = x[:n_ops]
        ms_section = x[n_ops:]

        operations = []
        idx = 0
        for j in range(self.data.n_jobs):
            for o in range(self.data.operations_per_job[j]):
                avail = self.data.get_available_machines(j, o)
                if not avail:
                    return 1e9, 1e9, 1.0, {}

                ms_val = ms_section[idx]
                m_idx = int(np.floor(ms_val * len(avail)))
                m_idx = min(m_idx, len(avail) - 1)
                sel_m = avail[m_idx]

                operations.append({
                    'job': j, 'op': o, 'machine': sel_m,
                    'priority': os_section[idx],
                    'proc_time': self.data.proc_times[(j, o, sel_m)],
                    'power': self.data.production_power[(j, o, sel_m)],
                    'ftt': self.data.ftt[(j, o, sel_m)]
                })
                idx += 1

        # 仅从每个作业当前可执行的下一道工序中选择优先值最小者。
        # 原版全局排序可能使后序工序先于前序工序排程。
        operations_by_job = [[] for _ in range(self.data.n_jobs)]
        for operation in operations:
            operations_by_job[operation['job']].append(operation)
        next_operation = [0] * self.data.n_jobs
        precedence_ordered = []
        while len(precedence_ordered) < len(operations):
            ready = [
                operations_by_job[j][next_operation[j]]
                for j in range(self.data.n_jobs)
                if next_operation[j] < len(operations_by_job[j])
            ]
            selected = min(ready, key=lambda v: (v['priority'], v['job']))
            precedence_ordered.append(selected)
            next_operation[selected['job']] += 1
        operations = precedence_ordered

        machine_release = np.zeros(self.data.n_machines)
        job_release = defaultdict(lambda: 0.0)

        total_proc_energy = 0.0

        for op in operations:
            j, o, m = op['job'], op['op'], op['machine']
            pt = op['proc_time']
            start = max(job_release[j], machine_release[m])
            end = start + pt
            op['start'] = start
            op['end'] = end
            machine_release[m] = end
            job_release[j] = end
            total_proc_energy += pt * op['power']

        makespan = float(np.max(machine_release))

        # 按投稿稿件中的定义，仅计入加工能耗，质量由最弱工序决定。
        total_energy = total_proc_energy
        quality_loss = 1.0 - min(op['ftt'] for op in operations)

        return makespan, total_energy, quality_loss, {
            'machine_release': machine_release,
            'operations': operations
        }


# ============================== 4.5 混合初始化采样器 (GS:LS:RS = 3:2:5) ==============================
class FJSPSampling(Sampling):
    def __init__(self, data_loader, gs_ratio=0.3, ls_ratio=0.2, rs_ratio=0.5, **kwargs):
        super().__init__(**kwargs)
        if min(gs_ratio, ls_ratio, rs_ratio) < 0 or not np.isclose(gs_ratio + ls_ratio + rs_ratio, 1.0):
            raise ValueError('GLR ratios must be non-negative and sum to one')
        self.data = data_loader
        self.gs_ratio = gs_ratio
        self.ls_ratio = ls_ratio
        self.rs_ratio = rs_ratio

        self._op_info = []
        idx = 0
        for j in range(self.data.n_jobs):
            for o in range(self.data.operations_per_job[j]):
                avail = self.data.get_available_machines(j, o)
                n_avail = len(avail)

                pts = []
                energies = []
                quality_losses = []
                for m in avail:
                    pt = self.data.proc_times[(j, o, m)]
                    power = self.data.production_power.get((j, o, m), 1.0)
                    ftt = self.data.ftt.get((j, o, m), 1.0)
                    pts.append(pt)
                    energies.append(pt * power)
                    quality_losses.append(1.0 - ftt)

                def _norm(vals):
                    v_min, v_max = min(vals), max(vals)
                    if v_max - v_min < 1e-9:
                        return [0.5] * len(vals)
                    return [(v - v_min) / (v_max - v_min) for v in vals]

                pt_norm = _norm(pts)
                energy_norm = _norm(energies)
                ql_norm = _norm(quality_losses)

                gs_scores = [pt_norm[i]*0.33 + energy_norm[i]*0.33 + ql_norm[i]*0.34 for i in range(n_avail)]
                ls_scores = [pt_norm[i]*0.20 + energy_norm[i]*0.40 + ql_norm[i]*0.40 for i in range(n_avail)]

                gs_sorted_idx = sorted(range(n_avail), key=lambda i: gs_scores[i])
                ls_sorted_idx = sorted(range(n_avail), key=lambda i: ls_scores[i])

                self._op_info.append({
                    'idx': idx,
                    'job': j,
                    'op': o,
                    'avail': avail,
                    'n_avail': n_avail,
                    'min_pt': min(pts),
                    'max_pt': max(pts),
                    'avg_pt': sum(pts) / len(pts),
                    'avg_energy': sum(energies) / len(energies),
                    'avg_ql': sum(quality_losses) / len(quality_losses),
                    'gs_best_m_idx': gs_sorted_idx[0],
                    'gs_second_m_idx': gs_sorted_idx[1] if n_avail >= 2 else gs_sorted_idx[0],
                    'ls_best_m_idx': ls_sorted_idx[0],
                    'ls_second_m_idx': ls_sorted_idx[1] if n_avail >= 2 else ls_sorted_idx[0],
                    'best_pt_m_idx': pts.index(min(pts)),
                    'best_energy_m_idx': energies.index(min(energies)),
                    'best_ql_m_idx': quality_losses.index(min(quality_losses)),
                })
                idx += 1

        all_pts = [op['avg_pt'] for op in self._op_info]
        all_energies = [op['avg_energy'] for op in self._op_info]
        all_qls = [op['avg_ql'] for op in self._op_info]

        self._pt_min, self._pt_max = min(all_pts), max(all_pts)
        self._en_min, self._en_max = min(all_energies), max(all_energies)
        self._ql_min, self._ql_max = min(all_qls), max(all_qls)

    def _norm_global(self, val, v_min, v_max):
        if v_max - v_min < 1e-9:
            return 0.5
        return (val - v_min) / (v_max - v_min)

    def _do(self, problem, n_samples, **kwargs):
        n_gs = int(n_samples * self.gs_ratio)
        n_ls = int(n_samples * self.ls_ratio)
        n_rs = n_samples - n_gs - n_ls

        X = np.zeros((n_samples, problem.n_var))
        idx = 0
        seen = set()

        def add_unique(make_individual):
            nonlocal idx
            for _ in range(100):
                individual = make_individual()
                key = individual.tobytes()
                if key not in seen:
                    seen.add(key)
                    X[idx] = individual
                    idx += 1
                    return
            raise RuntimeError('Could not generate a unique GLR individual')

        for _ in range(n_gs):
            add_unique(self._init_gs)
        for _ in range(n_ls):
            add_unique(self._init_ls)
        for _ in range(n_rs):
            add_unique(lambda: self._init_rs(problem.n_var))

        np.random.shuffle(X)
        self.last_counts = {'GS': n_gs, 'LS': n_ls, 'RS': n_rs, 'unique': len(seen)}
        return X

    def _init_gs(self):
        n_ops = self.data.total_operations
        x = np.zeros(n_ops * 2)

        def _gs_priority(op):
            p_pt = self._norm_global(op['avg_pt'], self._pt_min, self._pt_max)
            p_en = self._norm_global(op['avg_energy'], self._en_min, self._en_max)
            p_ql = self._norm_global(op['avg_ql'], self._ql_min, self._ql_max)
            return p_pt * 0.4 + p_en * 0.3 + p_ql * 0.3

        # 在启发式排序附近作小幅随机扰动，使同组样本保持偏好但不完全相同。
        sorted_ops = sorted(self._op_info, key=lambda op: _gs_priority(op) + np.random.normal(0, 0.02))
        for rank, op in enumerate(sorted_ops):
            x[op['idx']] = (rank / len(sorted_ops)) * 0.35

        for op in self._op_info:
            machine_index = op['gs_best_m_idx']
            if op['n_avail'] > 1 and np.random.random() < 0.2:
                machine_index = op['gs_second_m_idx']
            x[n_ops + op['idx']] = (machine_index + np.random.uniform(0.05, 0.95)) / op['n_avail']

        return x

    def _init_ls(self):
        n_ops = self.data.total_operations
        x = np.zeros(n_ops * 2)

        def _ls_priority(op):
            p_pt = self._norm_global(op['avg_pt'], self._pt_min, self._pt_max)
            p_en = self._norm_global(op['avg_energy'], self._en_min, self._en_max)
            p_ql = self._norm_global(op['avg_ql'], self._ql_min, self._ql_max)
            return p_pt * 0.2 + p_en * 0.4 + p_ql * 0.4

        sorted_ops = sorted(self._op_info, key=lambda op: _ls_priority(op) + np.random.normal(0, 0.02))
        for rank, op in enumerate(sorted_ops):
            x[op['idx']] = (rank / len(sorted_ops)) * 0.35 + 0.35

        for op in self._op_info:
            machine_index = op['ls_best_m_idx']
            if op['n_avail'] > 1 and np.random.random() < 0.2:
                machine_index = op['ls_second_m_idx']
            x[n_ops + op['idx']] = (machine_index + np.random.uniform(0.05, 0.95)) / op['n_avail']

        return x

    def _init_rs(self, n_var):
        return np.random.random(n_var)


# ============================== 5. 问题定义 ==============================
class FJSPProblem(Problem):
    def __init__(self, data_loader: FJSPDataLoader):
        self.data_loader = data_loader
        self.codec = FJSPCodec(data_loader)
        n_var = data_loader.total_operations * 2

        super().__init__(
            n_var=n_var, n_obj=3, n_constr=0,
            xl=np.zeros(n_var), xu=np.ones(n_var), vtype=float
        )

    def _evaluate(self, x, out, *args, **kwargs):
        n_samples = x.shape[0]
        F = np.zeros((n_samples, 3))
        for i in range(n_samples):
            f0, f1, f2, _ = self.codec.decode(x[i])
            F[i, 0] = f0 
            F[i, 1] = f1 / 100.0
            F[i, 2] = f2 * 100000.0
        out["F"] = F


# ============================== 6. Q-Learning Agent ==============================
class QLearningAgent:
    def __init__(self, states, actions, alpha=0.12, gamma=0.90,
                 epsilon=0.50, epsilon_decay=0.995, epsilon_min=0.10):
        self.q_table = {(s, a): 0.0 for s in states for a in actions}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.states = states
        self.actions = actions
        self.visit_count = defaultdict(int)

        self.q_history = []

    def select_action(self, state: str) -> str:
        self.visit_count[state] += 1
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        q_vals = {a: self.q_table[(state, a)] for a in self.actions}
        max_q = max(q_vals.values())
        best = [a for a, q in q_vals.items() if abs(q - max_q) < 1e-9]
        return np.random.choice(best)

    def update(self, state, action, reward, next_state):
        q_old = self.q_table[(state, action)]
        q_next = max(self.q_table[(next_state, a)] for a in self.actions)
        self.q_table[(state, action)] = q_old + self.alpha * (reward + self.gamma * q_next - q_old)

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self) -> Dict[str, str]:
        return {s: max(self.actions, key=lambda a: self.q_table[(s, a)]) for s in self.states}

    def record_q_stats(self, gen: int):
        q_vals = np.array(list(self.q_table.values()))
        self.q_history.append({
            'generation': gen,
            'max_q': float(np.max(q_vals)),
            'mean_q': float(np.mean(q_vals)),
            'std_q': float(np.std(q_vals)),
            'min_q': float(np.min(q_vals)),
            'epsilon': float(self.epsilon),
            'nonzero_ratio': float(np.sum(np.abs(q_vals) > 1e-9) / len(q_vals))
        })

    def get_q_matrix(self) -> pd.DataFrame:
        data = []
        for s in self.states:
            row = {'state': s}
            for a in self.actions:
                row[a] = self.q_table[(s, a)]
            data.append(row)
        return pd.DataFrame(data).set_index('state')

    def get_visit_matrix(self) -> pd.DataFrame:
        data = []
        for s in self.states:
            row = {'state': s, 'visit_count': self.visit_count.get(s, 0)}
            data.append(row)
        return pd.DataFrame(data).set_index('state')


# ============================== 7. NSGA-II-TS（固定参数禁忌搜索） ==============================
class NSGA2_TS(NSGA2):
    def __init__(self, pop_size=60, ts_frequency=20, ts_iterations=30, tabu_size=10, 
                 sampling=None, **kwargs):
        self.ts_frequency = ts_frequency
        self.ts_iterations = ts_iterations
        self.tabu_size = tabu_size

        super().__init__(
            pop_size=pop_size,
            sampling=sampling if sampling is not None else FloatRandomSampling(),
            selection=TournamentSelection(func_comp=binary_tournament),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
            **kwargs
        )

        self.tabu_list = deque(maxlen=tabu_size)
        self.ts_count = 0

    def _local_search(self, x: np.ndarray, problem) -> np.ndarray:
        n_ops = problem.n_var // 2
        x_new = x.copy()
        idx1, idx2 = np.random.choice(n_ops, 2, replace=False)
        x_new[idx1], x_new[idx2] = x_new[idx2], x_new[idx1]
        ms_idx = np.random.randint(n_ops, problem.n_var)
        x_new[ms_idx] = np.clip(x_new[ms_idx] + np.random.normal(0, 0.1), 0, 1)
        return x_new

    def _solution_hash(self, x: np.ndarray) -> str:
        os_part = x[:len(x)//2]
        sorted_indices = np.argsort(os_part)
        return ''.join(f"{i:03d}" for i in sorted_indices[:20])

    def _advance(self, infills=None, **kwargs):
        super()._advance(infills=infills, **kwargs)

        if self.n_gen > 1 and self.n_gen % self.ts_frequency == 0:
            rank = self.pop.get("rank")
            if rank is None:
                return

            nd_indices = np.where(rank == 0)[0][:10]
            if len(nd_indices) == 0:
                return

            improved_count = 0

            for idx in nd_indices:
                x_parent = self.pop[idx].X.copy()
                f_parent = self.pop[idx].F.copy()

                x_best = x_parent.copy()
                f_best = f_parent.copy()

                for _ in range(self.ts_iterations):
                    x_candidate = self._local_search(x_best, self.problem)
                    hash_val = self._solution_hash(x_candidate)

                    if hash_val in self.tabu_list:
                        continue

                    f_candidate = self.problem.evaluate(x_candidate.reshape(1, -1))[0]

                    if np.all(f_candidate <= f_best) and np.any(f_candidate < f_best):
                        x_best = x_candidate
                        f_best = f_candidate
                        self.tabu_list.append(hash_val)
                        improved_count += 1

                if not np.array_equal(x_best, x_parent):
                    self.pop[idx].X = x_best
                    self.pop[idx].F = f_best

            if improved_count > 0:
                self.ts_count += 1
                print(f"[Gen {self.n_gen:03d}] TS(固定): 改进 {improved_count} 个解，禁忌表大小 {len(self.tabu_list)}")


# ============================== 8. NSGA-II-QL（QL自适应Pc/Pm，无TS） ==============================
class NSGA2_QL(NSGA2):
    def __init__(self, pop_size=60, seed=None, max_gen=1000, sampling=None):
        pc_levels = [0.7, 0.8, 0.9]
        pm_levels = [0.05, 0.1, 0.15]
        
        self.ga_configs = {}
        for pc in pc_levels:
            for pm in pm_levels:
                action_name = f"pc{pc:.2f}_pm{pm:.2f}"
                self.ga_configs[action_name] = {'pc': pc, 'pm': pm}
        
        self.ga_configs['keep'] = {'pc': None, 'pm': None}
        
        self.ql = QLearningAgent(
            states=[
                'early_converged', 'early_balanced', 'early_diverse',
                'mid_converged', 'mid_balanced', 'mid_diverse',
                'late_converged', 'late_balanced', 'late_diverse'
            ],
            actions=list(self.ga_configs.keys()),
            alpha=0.12, gamma=0.90, epsilon=0.50,
            epsilon_decay=0.995, epsilon_min=0.10
        )
        
        self.max_gen = max_gen
        
        self._cd_history = deque(maxlen=50)
        self._ratio_history = deque(maxlen=50)
        self._hv_delta_history = deque(maxlen=50)
        self._sp_delta_history = deque(maxlen=50)
        
        self._last_ql_hv = 0.0
        self._last_ql_sp = float('inf')
        self._last_ql_nd = 0
        self._first_ql = True
        
        self.ql_decision_period = 10
        self._last_ql_decision_gen = 0
        
        super().__init__(
            pop_size=pop_size,
            sampling=sampling if sampling is not None else FloatRandomSampling(),
            selection=TournamentSelection(func_comp=binary_tournament),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
            seed=seed
        )
        
        self._default_pc = self.mating.crossover.prob
        self._default_pm = self.mating.mutation.prob
        
        self._prev_state = None
        self._prev_action = None
        
        self.ql_action_history = []
        self.ql_reward_history = []

    def _get_state(self) -> str:
        if self.pop is None:
            return 'early_balanced'

        rank = self.pop.get("rank")
        crowding = self.pop.get("crowding")
        if rank is None or crowding is None:
            return 'early_balanced'

        rank0_mask = (rank == 0)
        rank0_ratio = np.sum(rank0_mask) / len(rank)

        cd_vals = crowding[rank0_mask]
        cd_vals = cd_vals[np.isfinite(cd_vals)]
        rank0_cd = float(np.mean(cd_vals)) if len(cd_vals) > 0 else 0.0

        self._ratio_history.append(rank0_ratio)
        self._cd_history.append(rank0_cd)

        progress = self.n_gen / self.max_gen if self.max_gen and self.max_gen > 0 else 0.5
        if progress < 0.33:
            stage = 'early'
        elif progress < 0.67:
            stage = 'mid'
        else:
            stage = 'late'

        if len(self._ratio_history) >= 10:
            ratio_arr = np.array(self._ratio_history)
            cd_arr = np.array(self._cd_history)

            ratio_low = np.percentile(ratio_arr, 25)
            ratio_high = np.percentile(ratio_arr, 75)
            cd_low = np.percentile(cd_arr, 25)
            cd_high = np.percentile(cd_arr, 75)

            if rank0_ratio <= ratio_low and rank0_cd <= cd_low:
                diversity = 'converged'
            elif rank0_ratio >= ratio_high or rank0_cd >= cd_high:
                diversity = 'diverse'
            else:
                diversity = 'balanced'
        else:
            if rank0_ratio < 0.20 and rank0_cd < 0.30:
                diversity = 'converged'
            elif rank0_ratio > 0.65 or rank0_cd > 1.50:
                diversity = 'diverse'
            else:
                diversity = 'balanced'

        return f"{stage}_{diversity}"

    def _normalize_reward_component(self, val: float, history: deque, fallback_scale: float = 10.0) -> float:
        if len(history) < 5:
            return float(np.tanh(val / fallback_scale))

        arr = np.array(history)
        v_min, v_max = np.min(arr), np.max(arr)
        if v_max - v_min < 1e-9:
            return 0.0

        norm = (val - v_min) / (v_max - v_min + 1e-9) * 2.0 - 1.0
        return float(np.clip(norm, -1.0, 1.0))

    def _calc_hv_of_pop(self) -> float:
        if self.pop is None:
            return 0.0
        rank = self.pop.get("rank")
        F = self.pop.get("F")
        if rank is None or F is None:
            return 0.0
        nd_mask = (rank == 0)
        if not np.any(nd_mask):
            return 0.0
        F_nd = F[nd_mask]
        ref_pt = np.max(F, axis=0) * 1.05 + 1e-6
        try:
            hv = float(Hypervolume(ref_point=ref_pt).do(F_nd))
        except:
            hv = 0.0
        return hv

    def _advance(self, infills=None, **kwargs):
        super()._advance(infills=infills, **kwargs)
        self.ql.record_q_stats(self.n_gen)

        if self.n_gen > 1 and self.n_gen >= g_start_gen:
            if (self.n_gen - self._last_ql_decision_gen) >= self.ql_decision_period:
                rank = self.pop.get("rank")
                if rank is None:
                    return
                
                state = self._get_state()
                action = self.ql.select_action(state)
                cfg = self.ga_configs[action]
                
                if action != 'keep':
                    self.mating.crossover.prob = cfg['pc']
                    self.mating.mutation.prob = cfg['pm']
                
                hv_now = self._calc_hv_of_pop()
                nd_now = int(np.sum(rank == 0))
                sp_now = calc_sp(self.pop.get("F")[rank == 0]) if np.any(rank == 0) else float('inf')
                
                if self._first_ql:
                    reward = 0.0
                    hv_delta_pct = 0.0
                    sp_delta_pct = 0.0
                    nd_delta = 0
                    improved_flag = 0
                else:
                    hv_delta_pct = (hv_now - self._last_ql_hv) / (self._last_ql_hv + 1e-9) * 100.0
                    sp_delta_pct = (self._last_ql_sp - sp_now) / (self._last_ql_sp + 1e-9) * 100.0 if self._last_ql_sp > 1e-9 else 0.0
                    nd_delta = nd_now - self._last_ql_nd
                    improved_flag = 1 if (hv_delta_pct > 0.1 or nd_delta > 0) else 0
                
                self._hv_delta_history.append(hv_delta_pct)
                self._sp_delta_history.append(sp_delta_pct)
                
                hv_norm = self._normalize_reward_component(hv_delta_pct, self._hv_delta_history, fallback_scale=10.0)
                sp_norm = self._normalize_reward_component(sp_delta_pct, self._sp_delta_history, fallback_scale=5.0)
                
                reward = hv_norm * 10.0 + sp_norm * 6.0 + nd_delta * 2.0
                
                if improved_flag > 0:
                    reward += improved_flag * 2.0
                else:
                    if action != 'keep':
                        reward -= 1.5
                
                if action == 'keep':
                    reward = hv_norm * 5.0 + sp_norm * 3.0 + nd_delta * 1.0
                
                next_state = self._get_state()
                if self._prev_state is not None:
                    self.ql.update(self._prev_state, self._prev_action, reward, next_state)
                    self.ql.decay()
                
                self._prev_state = state
                self._prev_action = action
                self._first_ql = False
                
                self._last_ql_hv = hv_now
                self._last_ql_sp = sp_now
                self._last_ql_nd = nd_now
                self._last_ql_decision_gen = self.n_gen
                
                self.ql_action_history.append({
                    'generation': self.n_gen,
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'hv_delta_pct': hv_delta_pct,
                    'sp_delta_pct': sp_delta_pct,
                    'nd_delta': nd_delta,
                    'epsilon': self.ql.epsilon,
                    'pc': self.mating.crossover.prob,
                    'pm': self.mating.mutation.prob
                })
                
                if self.n_gen % 20 == 0 or action != 'keep':
                    if action == 'keep':
                        param_str = "KEEP          "
                    else:
                        param_str = f"Pc={cfg['pc']:.2f} Pm={cfg['pm']:.2f}"
                    
                    hv_str = f"hv={hv_delta_pct:+.4f}%" if not self._first_ql else "hv=N/A"
                    sp_str = f"sp={sp_delta_pct:+.4f}%" if not self._first_ql else "sp=N/A"
                    nd_str = f"nd={nd_delta:+d}"
                    print(f"[Gen {self.n_gen:03d}] GA-QL09: state={state:18s} | "
                          f"action={action:15s} | {param_str} | "
                          f"{hv_str} | {sp_str} | {nd_str} | "
                          f"R={reward:+.2f} | eps={self.ql.epsilon:.3f}")


# ============================== 9. NSGA-II-TS-QL-GLR（27档笛卡尔积 + HV/IGD双指标QL奖励 + 预构建参考前沿） ==============================
class NSGA2_TS_QL_GLR(NSGA2):
    def __init__(self, pop_size=60, seed=None, max_gen=1000, sampling=None, ref_front_mgr=None):
        freq_levels = [25, 20, 15]
        iter_levels = [35, 30, 25]

        self.ts_configs = {}
        for f in freq_levels:
            for i in iter_levels:
                action_name = f"f{f:02d}_i{i:02d}"
                self.ts_configs[action_name] = {
                    'freq': f, 'iter': i, 'tabu': 10
                }

        self.ts_configs['ts_none'] = {'freq': 999999, 'iter': 0, 'tabu': 0}

        self.ql = QLearningAgent(
            states=[
                'early_converged', 'early_balanced', 'early_diverse',
                'mid_converged', 'mid_balanced', 'mid_diverse',
                'late_converged', 'late_balanced', 'late_diverse'
            ],
            actions=list(self.ts_configs.keys()),
            alpha=0.12, gamma=0.90, epsilon=0.50,
            epsilon_decay=0.995, epsilon_min=0.10
        )

        self.max_gen = max_gen

        self._cd_history = deque(maxlen=50)
        self._ratio_history = deque(maxlen=50)
        self._hv_delta_history = deque(maxlen=50)
        self._igd_delta_history = deque(maxlen=50)   # 替代原 _sp_delta_history

        self._prev_hv = 0.0
        self._prev_nd = 0
        self._first_ts = True

        self.ql_decision_period = 10
        self._last_ql_decision_gen = 0

        cfg = self.ts_configs['ts_none']

        super().__init__(
            pop_size=pop_size,
            sampling=sampling if sampling is not None else FloatRandomSampling(),
            selection=TournamentSelection(func_comp=binary_tournament),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
            seed=seed
        )

        self.ts_frequency = cfg['freq']
        self.ts_iterations = cfg['iter']
        self.tabu_size = cfg['tabu']
        self.tabu_list = deque(maxlen=cfg['tabu'])
        self.ts_count = 0

        self._prev_state = None
        self._prev_action = None
        self._last_ts_gen = 0

        self.ql_action_history = []
        self.ql_reward_history = []

        # === 新增：预构建静态参考前沿与归一化IGD计算器 ===
        self.ref_front_mgr = ref_front_mgr
        self.igd_metric = ref_front_mgr.get_igd_metric() if ref_front_mgr is not None else None

    def _get_state(self) -> str:
        if self.pop is None:
            return 'early_balanced'

        rank = self.pop.get("rank")
        crowding = self.pop.get("crowding")
        if rank is None or crowding is None:
            return 'early_balanced'

        rank0_mask = (rank == 0)
        rank0_ratio = np.sum(rank0_mask) / len(rank)

        cd_vals = crowding[rank0_mask]
        cd_vals = cd_vals[np.isfinite(cd_vals)]
        rank0_cd = float(np.mean(cd_vals)) if len(cd_vals) > 0 else 0.0

        self._ratio_history.append(rank0_ratio)
        self._cd_history.append(rank0_cd)

        progress = self.n_gen / self.max_gen if self.max_gen and self.max_gen > 0 else 0.5
        if progress < 0.33:
            stage = 'early'
        elif progress < 0.67:
            stage = 'mid'
        else:
            stage = 'late'

        if len(self._ratio_history) >= 10:
            ratio_arr = np.array(self._ratio_history)
            cd_arr = np.array(self._cd_history)

            ratio_low = np.percentile(ratio_arr, 25)
            ratio_high = np.percentile(ratio_arr, 75)
            cd_low = np.percentile(cd_arr, 25)
            cd_high = np.percentile(cd_arr, 75)

            if rank0_ratio <= ratio_low and rank0_cd <= cd_low:
                diversity = 'converged'
            elif rank0_ratio >= ratio_high or rank0_cd >= cd_high:
                diversity = 'diverse'
            else:
                diversity = 'balanced'
        else:
            if rank0_ratio < 0.20 and rank0_cd < 0.30:
                diversity = 'converged'
            elif rank0_ratio > 0.65 or rank0_cd > 1.50:
                diversity = 'diverse'
            else:
                diversity = 'balanced'

        return f"{stage}_{diversity}"

    def _normalize_reward_component(self, val: float, history: deque, fallback_scale: float = 10.0) -> float:
        if len(history) < 5:
            return float(np.tanh(val / fallback_scale))

        arr = np.array(history)
        v_min, v_max = np.min(arr), np.max(arr)
        if v_max - v_min < 1e-9:
            return 0.0

        norm = (val - v_min) / (v_max - v_min + 1e-9) * 2.0 - 1.0
        return float(np.clip(norm, -1.0, 1.0))

    def _calc_hv_of_pop(self) -> float:
        if self.pop is None:
            return 0.0
        rank = self.pop.get("rank")
        F = self.pop.get("F")
        if rank is None or F is None:
            return 0.0
        nd_mask = (rank == 0)
        if not np.any(nd_mask):
            return 0.0
        F_nd = F[nd_mask]
        ref_pt = np.max(F, axis=0) * 1.05 + 1e-6
        try:
            hv = float(Hypervolume(ref_point=ref_pt).do(F_nd))
        except:
            hv = 0.0
        return hv

    def _calc_igd_of_pop(self) -> float:
        """基于预构建静态参考前沿计算当前种群的归一化 IGD"""
        if self.pop is None or self.igd_metric is None:
            return float('inf')
        rank = self.pop.get("rank")
        F = self.pop.get("F")
        if rank is None or F is None:
            return float('inf')
        nd_mask = (rank == 0)
        if not np.any(nd_mask):
            return float('inf')
        F_nd = F[nd_mask]
        return self.igd_metric.compute(F_nd)

    def _local_search(self, x: np.ndarray, problem) -> np.ndarray:
        n_ops = problem.n_var // 2
        x_new = x.copy()
        idx1, idx2 = np.random.choice(n_ops, 2, replace=False)
        x_new[idx1], x_new[idx2] = x_new[idx2], x_new[idx1]
        ms_idx = np.random.randint(n_ops, problem.n_var)
        x_new[ms_idx] = np.clip(x_new[ms_idx] + np.random.normal(0, 0.05), 0, 1)
        return x_new

    def _solution_hash(self, x: np.ndarray) -> str:
        os_part = x[:len(x)//2]
        sorted_indices = np.argsort(os_part)
        return ''.join(f"{i:03d}" for i in sorted_indices[:20])

    def _advance(self, infills=None, **kwargs):
        super()._advance(infills=infills, **kwargs)

        self.ql.record_q_stats(self.n_gen)

        if self.n_gen > 1 and self.n_gen >= g_start_gen:
            if (self.n_gen - self._last_ql_decision_gen) >= self.ql_decision_period:

                rank = self.pop.get("rank")
                if rank is None:
                    return

                state = self._get_state()
                action = self.ql.select_action(state)
                cfg = self.ts_configs[action]

                self._last_ql_decision_gen = self.n_gen

                # === 修改：同时采集 HV 与 IGD（去掉 SP） ===
                hv_before = self._calc_hv_of_pop()
                igd_before = self._calc_igd_of_pop()
                nd_before = int(np.sum(rank == 0))

                improved_count = 0

                if action != 'ts_none':
                    self.ts_frequency = cfg['freq']
                    self.ts_iterations = cfg['iter']
                    self.tabu_size = cfg['tabu']
                    self.tabu_list = deque(maxlen=cfg['tabu'])

                    nd_indices = np.where(rank == 0)[0][:5]

                    for idx in nd_indices:
                        x_parent = self.pop[idx].X.copy()
                        f_parent = self.pop[idx].F.copy()
                        x_best = x_parent.copy()
                        f_best = f_parent.copy()

                        for _ in range(self.ts_iterations):
                            x_candidate = self._local_search(x_best, self.problem)
                            hash_val = self._solution_hash(x_candidate)
                            if hash_val in self.tabu_list:
                                continue
                            f_candidate = self.problem.evaluate(x_candidate.reshape(1, -1))[0]
                            if np.all(f_candidate <= f_best) and np.any(f_candidate < f_best):
                                x_best = x_candidate
                                f_best = f_candidate
                                self.tabu_list.append(hash_val)
                                improved_count += 1

                        if not np.array_equal(x_best, x_parent):
                            self.pop[idx].X = x_best
                            self.pop[idx].F = f_best

                    if improved_count > 0:
                        self.ts_count += 1
                        self._last_ts_gen = self.n_gen

                # === 修改：TS 后同时采集 HV 与 IGD ===
                hv_after = self._calc_hv_of_pop()
                igd_after = self._calc_igd_of_pop()
                rank_after = self.pop.get("rank")
                nd_after = int(np.sum(rank_after == 0)) if rank_after is not None else 0

                # === 修改：HV + IGD 双指标奖励（彻底去掉 SP） ===
                if self._first_ts or hv_before < 1e-9:
                    hv_delta_pct = 0.0
                else:
                    hv_delta_pct = (hv_after - hv_before) / (hv_before + 1e-9) * 100.0

                if self._first_ts or not np.isfinite(igd_before) or igd_before < 1e-9:
                    igd_delta_pct = 0.0
                else:
                    igd_delta_pct = (igd_before - igd_after) / (igd_before + 1e-9) * 100.0  # IGD减小为正

                nd_delta = nd_after - nd_before

                self._hv_delta_history.append(hv_delta_pct)
                self._igd_delta_history.append(igd_delta_pct)

                hv_norm = self._normalize_reward_component(hv_delta_pct, self._hv_delta_history, fallback_scale=10.0)
                igd_norm = self._normalize_reward_component(igd_delta_pct, self._igd_delta_history, fallback_scale=5.0)

                # 双指标加权奖励
                reward = hv_norm * 10.0 + igd_norm * 8.0 + nd_delta * 2.0

                if improved_count > 0:
                    reward += improved_count * 2.0
                else:
                    if action != 'ts_none':
                        reward -= 1.5

                if action == 'ts_none':
                    reward = hv_norm * 5.0 + igd_norm * 4.0 + nd_delta * 1.0

                # 协同奖励与背离惩罚
                if hv_delta_pct > 0 and igd_delta_pct > 0:
                    reward += 0.15
                elif hv_delta_pct > 0 and igd_delta_pct < 0:
                    reward -= 0.08  # HV膨胀但分布性劣化
                elif hv_delta_pct < 0 and igd_delta_pct > 0:
                    reward -= 0.05  # 收敛性劣化

                reward = float(np.clip(reward, -2.0, 2.0))

                next_state = self._get_state()
                if self._prev_state is not None:
                    self.ql.update(self._prev_state, self._prev_action, reward, next_state)
                    self.ql.decay()

                self._prev_state = state
                self._prev_action = action
                self._prev_nd = nd_after
                self._first_ts = False

                self.ql_action_history.append({
                    'generation': self.n_gen,
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'hv_delta_pct': hv_delta_pct,
                    'igd_delta_pct': igd_delta_pct,      # 替代 sp_delta_pct
                    'nd_delta': nd_delta,
                    'improved_count': improved_count,
                    'hv_before': hv_before,
                    'hv_after': hv_after,
                    'igd_before': igd_before,            # 替代 sp_before
                    'igd_after': igd_after,              # 替代 sp_after
                    'epsilon': self.ql.epsilon
                })

                if self.n_gen % 20 == 0 or action != 'ts_none' or improved_count > 0:
                    if action == 'ts_none':
                        param_str = "TS=OFF      "
                    else:
                        param_str = f"f={cfg['freq']:2d} i={cfg['iter']:2d} t={cfg['tabu']:2d}"

                    hv_str = f"hv={hv_delta_pct:+.4f}%" if not self._first_ts else "hv=N/A"
                    igd_str = f"igd={igd_delta_pct:+.4f}%" if not self._first_ts else "igd=N/A"
                    nd_str = f"nd={nd_delta:+d}"
                    print(f"[Gen {self.n_gen:03d}] TS-QL27: state={state:18s} | "
                          f"action={action:15s} | {param_str} | "
                          f"imp={improved_count} | {hv_str} | {igd_str} | {nd_str} | "
                          f"R={reward:+.2f} | eps={self.ql.epsilon:.3f}")


# ============================== 9.5 NSGA-II-TS-QL（TS-QL机制 + 随机初始化） ==============================
class NSGA2_TS_QL_Random(NSGA2_TS_QL_GLR):
    """
    与 NSGA-II-TS-QL-GLR 完全相同的 QL+TS 机制，
    但强制使用 FloatRandomSampling 随机初始化，
    用于孤立验证混合初始化 (GS/LS/RS) 的边际贡献。
    """
    def __init__(self, pop_size=60, seed=None, max_gen=1000, sampling=None, ref_front_mgr=None):
        super().__init__(pop_size=pop_size, seed=seed, max_gen=max_gen,
                         sampling=FloatRandomSampling(), ref_front_mgr=ref_front_mgr)


# ============================== 10. 每代迭代记录器 ==============================
class GenerationTracker:
    def __init__(self):
        self.records = []
        self.start_time = None
        self.F_nd_history = []

    def __call__(self, algorithm):
        if self.start_time is None:
            self.start_time = time.time()

        elapsed = time.time() - self.start_time
        pop = algorithm.pop

        if pop is None:
            return

        rank = pop.get("rank")
        F = pop.get("F")

        if rank is None or F is None:
            return

        nd_mask = (rank == 0)
        nd_count = int(np.sum(nd_mask))
        F_nd = F[nd_mask].copy() if np.any(nd_mask) else np.empty((0, 3))

        self.records.append({
            'generation': algorithm.n_gen,
            'nd_count': nd_count,
            'elapsed_time': elapsed
        })
        self.F_nd_history.append(F_nd)


# ============================== 11. 单次运行 ==============================
class SilentDisplay(Display):
    def _do(self, problem, evaluator, algorithm):
        pass


def run_single_comparison(problem, termination, seed, max_gen=1000, ref_front_mgr=None):
    """单次运行4个算法，记录每代迭代数据（使用预构建静态参考前沿）"""
    results = {}

    # 1. NSGA-II（随机初始化）
    random.seed(seed)
    np.random.seed(seed)
    tracker1 = GenerationTracker()
    t0 = time.time()
    algo1 = NSGA2(pop_size=60, crossover=SBX(prob=0.9, eta=15),
                  mutation=PM(eta=20), eliminate_duplicates=True, seed=seed)
    res1 = minimize(problem, algo1, termination, seed=seed, verbose=False, 
                    display=SilentDisplay(), callback=tracker1)
    t1 = time.time()
    results['NSGA2'] = {
        'F': res1.F, 
        'time': t1 - t0,
        'gen_records': tracker1.records,
        'F_nd_history': tracker1.F_nd_history
    }
    print(f"\n    NSGA-II      : {t1-t0:.2f}s | ND={len(res1.F)}")

    # 2. NSGA-II-TS（随机初始化）
    random.seed(seed)
    np.random.seed(seed)
    tracker2 = GenerationTracker()
    t0 = time.time()
    algo2 = NSGA2_TS(pop_size=60, ts_frequency=20, ts_iterations=30, tabu_size=10, seed=seed)
    res2 = minimize(problem, algo2, termination, seed=seed, verbose=False, 
                    display=SilentDisplay(), callback=tracker2)
    t1 = time.time()
    results['NSGA2_TS'] = {
        'F': res2.F, 
        'time': t1 - t0,
        'gen_records': tracker2.records,
        'F_nd_history': tracker2.F_nd_history
    }
    print(f"    NSGA-II-TS   : {t1-t0:.2f}s | ND={len(res2.F)}")

    # 3. NSGA-II-TS-QLandom（随机初始化，完整TS-QL机制，使用预构建参考前沿）
    random.seed(seed)
    np.random.seed(seed)
    tracker4 = GenerationTracker()
    t0 = time.time()
    algo4 = NSGA2_TS_QL_Random(pop_size=60, seed=seed, max_gen=max_gen, ref_front_mgr=ref_front_mgr)
    res4 = minimize(problem, algo4, termination, seed=seed, verbose=False, 
                    display=SilentDisplay(), callback=tracker4)
    t1 = time.time()
    results['NSGA2_TS_QL'] = {
        'F': res4.F, 
        'time': t1 - t0,
        'gen_records': tracker4.records,
        'F_nd_history': tracker4.F_nd_history
    }
    print(f"    NSGA-II-TS-QL: {t1-t0:.2f}s | ND={len(res4.F)}")

    # 4. NSGA-II-TS-QL-GLR（混合初始化，完整TS-QL机制，使用预构建参考前沿）
    random.seed(seed)
    np.random.seed(seed)
    tracker5 = GenerationTracker()
    t0 = time.time()
    ql_sampling = FJSPSampling(problem.data_loader, gs_ratio=0.3, ls_ratio=0.2, rs_ratio=0.5)
    algo5 = NSGA2_TS_QL_GLR(pop_size=60, seed=seed, max_gen=max_gen, sampling=ql_sampling, ref_front_mgr=ref_front_mgr)
    res5 = minimize(problem, algo5, termination, seed=seed, verbose=False, 
                    display=SilentDisplay(), callback=tracker5)
    t1 = time.time()

    # 甘特图数据从 NSGA-II-TS-QL-GLR (res5) 提取
    gantt_data = {}
    if hasattr(res5, 'X') and hasattr(res5, 'F') and len(res5.F) > 0:
        F_ql = res5.F
        X_ql = res5.X
        best_indices = {
            'makespan': int(np.argmin(F_ql[:, 0])),
            'energy': int(np.argmin(F_ql[:, 1])),
            'quality': int(np.argmin(F_ql[:, 2]))
        }
        for obj_name, idx in best_indices.items():
            x_best = X_ql[idx]
            f0, f1, f2, info = problem.codec.decode(x_best)
            if not info or 'operations' not in info:
                print(f"  [Warning] seed={seed} NSGA-II-TS-QL-GLR {obj_name} 最优解无法解码，跳过甘特图")
                continue
            gantt_data[obj_name] = {
                'F_scaled': F_ql[idx].copy(),
                'makespan': f0,
                'energy': f1,
                'quality': f2,
                'operations': info['operations']
            }

    results['NSGA2_TS_QL_GLR'] = {
        'F': res5.F, 
        'time': t1 - t0,
        'gen_records': tracker5.records,
        'F_nd_history': tracker5.F_nd_history,
        'gantt': gantt_data,
        'ql_q_history': algo5.ql.q_history,
        'ql_action_history': algo5.ql_action_history,
        'ql_q_table': algo5.ql.get_q_matrix(),
        'ql_visit_matrix': algo5.ql.get_visit_matrix(),
        'ql_policy': algo5.ql.get_policy()
    }
    print(f"    NSGA-II-TS-QL-GLR: {t1-t0:.2f}s | ND={len(res5.F)}")

    run_nd_solutions = []
    for alg in ['NSGA2', 'NSGA2_TS', 'NSGA2_TS_QL', 'NSGA2_TS_QL_GLR']:
        if len(results[alg]['F']) > 0:
            run_nd_solutions.append(results[alg]['F'])

    return results, run_nd_solutions


# ============================== 12. 统计运行主函数 ==============================
def run_statistical_comparison(data_path, max_gen=2000, n_runs=20, time_limit=None):
    dataset_name = os.path.basename(data_path)
    print("=" * 95)
    print(f"对比实验：4算法对比 | {n_runs}次独立运行 | 指标: HV, IGD, SP")
    print(f"NSGA-II | NSGA-II-TS | NSGA-II-TS-QL | NSGA-II-TS-QL-GLR")
    print(f"数据集: {dataset_name}  |  代数: {max_gen}  |  运行次数: {n_runs}", end="")
    if time_limit:
        print(f"  |  时间限制: {time_limit}")
    else:
        print()
    print("=" * 95)

    if time_limit:
        time_seconds = parse_time_limit(time_limit)
        termination = TerminationCollection(
            get_termination('n_gen', max_gen),
            get_termination('time', time_seconds)
        )
        print(f"[终止条件] 代数上限: {max_gen} 或 时间上限: {time_seconds}s ({time_limit})")
    else:
        termination = ('n_gen', max_gen)
        print(f"[终止条件] 仅代数上限: {max_gen}")

    data = FJSPDataLoader(data_path)
    problem = FJSPProblem(data)

    algorithms = ['NSGA2', 'NSGA2_TS', 'NSGA2_TS_QL', 'NSGA2_TS_QL_GLR'] 

    # ==================== Phase 0: 预构建静态参考前沿 ====================
    print("\n" + "=" * 95)
    print("[Phase 0] 预构建全局静态参考前沿（多算法快速预运行 + 插值加密）")
    print("=" * 95)
    preliminary_solutions = []
    
    # 【修复2】使用多种算法各跑1次，确保参考前沿覆盖更广
    pre_configs = [
        ('NSGA2', lambda s: NSGA2(pop_size=60, crossover=SBX(prob=0.9, eta=15),
                                   mutation=PM(eta=20), eliminate_duplicates=True, seed=s)),
        ('NSGA2_TS', lambda s: NSGA2_TS(pop_size=60, ts_frequency=20, ts_iterations=30, 
                                        tabu_size=10, seed=s)),
        ('NSGA2_TS_QL_GLR', lambda s: NSGA2_TS_QL_GLR(pop_size=60, seed=s, max_gen=200, 
                                               ref_front_mgr=None)),  # QL内部会生成临时前沿
    ]
    
    for name, algo_factory in pre_configs:
        seed_p = 1000 + hash(name) % 1000
        random.seed(seed_p)
        np.random.seed(seed_p)
        algo_pre = algo_factory(seed_p)
        res_pre = minimize(problem, algo_pre, ('n_gen', 200), seed=seed_p,
                           verbose=False, display=SilentDisplay())
        if res_pre.F is not None and len(res_pre.F) > 0:
            preliminary_solutions.extend(res_pre.F.tolist())
            print(f"  [Pre-run {name}] seed={seed_p} | ND={len(res_pre.F)}")
    
    ref_front_mgr = StaticReferenceFront(preliminary_solutions, target_size=1000)
    igd_metric = ref_front_mgr.get_igd_metric()
    
    # 保存参考前沿供复现
    os.makedirs("result/ref_fronts", exist_ok=True)
    np.savetxt(f"result/ref_fronts/global_ref_front_{get_output_dir()}.csv",
               ref_front_mgr.get_front(), delimiter=',')
    print(f"[Phase 0 完成] 静态参考前沿已保存")

    # ==================== Phase 1: 执行 n_runs 次独立运行 ====================
    print("\n[Phase 1] 执行 n_runs 次独立运行 ...")
    all_run_results = []

    for run in range(n_runs):
        seed = 42 + run*5
        print(f"\n[Run {run+1:02d}/{n_runs}] seed={seed} ...", end="")

        results, run_nd_solutions = run_single_comparison(
            problem, termination, seed, max_gen=max_gen, 
            ref_front_mgr=ref_front_mgr
        )
        all_run_results.append(results)
        print("  [Run Done]")

    # ==================== Phase 2: 使用预构建参考前沿 ====================
    print("\n[Phase 2] 使用预构建静态参考前沿作为统一评估标准 ...")
    ref_front = ref_front_mgr.get_front()
    print(f"  参考前沿大小: {len(ref_front)} 个非支配解 (已插值加密)")

    # ==================== Phase 2.5: 构建后验参考前沿（所有运行合并） ====================
    print("\n[Phase 2.5] 构建后验参考前沿（所有运行所有算法合并）用于最终公平评估 ...")
    all_F_posterior = []
    for run in range(n_runs):
        for alg in algorithms:
            F = all_run_results[run][alg]['F']
            if len(F) > 0:
                all_F_posterior.append(F)
    
    if all_F_posterior:
        all_F_posterior = np.vstack(all_F_posterior)
        nds = NonDominatedSorting()
        fronts = nds.do(all_F_posterior)
        posterior_ref = all_F_posterior[fronts[0]]
        
        # 【修复3】后验边界基于所有解（更宽），并留余量
        posterior_ideal = np.min(all_F_posterior, axis=0) * 0.90
        posterior_nadir = np.max(all_F_posterior, axis=0) * 1.10
        mask = posterior_nadir <= posterior_ideal
        posterior_nadir[mask] = posterior_ideal[mask] + 1.0
        
        final_igd_metric = IGDMetric(posterior_ref, posterior_ideal, posterior_nadir)
        print(f"  后验参考前沿: {len(posterior_ref)} 非支配解（基于 {len(all_F_posterior)} 个解）")
    else:
        final_igd_metric = igd_metric  # 回退
        print("  [警告] 无后验解，回退到预构建前沿")

    # ==================== Phase 3: 统一计算 HV / IGD / SP ====================
    print("\n[Phase 3] 统一计算各指标（HV/IGD/SP）及每代迭代数据 ...")
    records = {alg: {'HV': [], 'IGD': [], 'SP': [], 'Time': []} for alg in algorithms}

    generation_data_rows = []
    pareto_solutions_rows = []
    ql_data_all_runs = []  # TS-QL

    for run in range(n_runs):
        results = all_run_results[run]

        # 收集TS-QL黑盒数据
        ql_data_all_runs.append({
            'run': run + 1,
            'q_history': results['NSGA2_TS_QL_GLR'].get('ql_q_history', []),
            'action_history': results['NSGA2_TS_QL_GLR'].get('ql_action_history', []),
            'q_table': results['NSGA2_TS_QL_GLR'].get('ql_q_table', None),
            'visit_matrix': results['NSGA2_TS_QL_GLR'].get('ql_visit_matrix', None),
            'policy': results['NSGA2_TS_QL_GLR'].get('ql_policy', {})
        })

        # 该次运行所有算法所有代的并集，用于确定该Run的统一HV参考点
        all_F_this_run_all_gens = []
        for alg in algorithms:
            for F_nd in results[alg]['F_nd_history']:
                if len(F_nd) > 0:
                    all_F_this_run_all_gens.append(F_nd)

        if all_F_this_run_all_gens:
            all_F_this_run_all_gens = np.vstack(all_F_this_run_all_gens)
            ref_pt_run = np.max(all_F_this_run_all_gens, axis=0) * 1.05 + 1e-6
        else:
            ref_pt_run = np.ones(3) * 1e6

        # 最终指标计算（IGD统一使用【后验】归一化参考前沿）
        for alg in algorithms:
            F = results[alg]['F']
            hv = float(Hypervolume(ref_point=ref_pt_run).do(F))
            igd = final_igd_metric.compute(F)   # 【修复3】用后验前沿计算IGD
            sp = calc_sp(F)

            records[alg]['HV'].append(hv)
            records[alg]['IGD'].append(igd)
            records[alg]['SP'].append(sp)
            records[alg]['Time'].append(results[alg]['time'])

            for idx_sol, f_scaled in enumerate(F):
                pareto_solutions_rows.append({
                    'Run': run + 1,
                    'Algorithm': alg,
                    'Solution_Index': idx_sol,
                    'Makespan': float(f_scaled[0]),
                    'Energy': float(f_scaled[1]) * 100.0,
                    'Quality_Loss': float(f_scaled[2]) / 10000.0
                })

        # 每代迭代数据（IGD统一使用【后验】归一化参考前沿）
        for alg in algorithms:
            gen_records = results[alg]['gen_records']
            F_nd_history = results[alg]['F_nd_history']

            for i, rec in enumerate(gen_records):
                F_nd = F_nd_history[i]

                if len(F_nd) > 0:
                    hv_gen = float(Hypervolume(ref_point=ref_pt_run).do(F_nd))
                    igd_gen = final_igd_metric.compute(F_nd)   # 【修复3】用后验前沿
                else:
                    hv_gen = 0.0
                    igd_gen = float('inf')

                sp_gen = calc_sp(F_nd) if len(F_nd) > 0 else 0.0
                generation_data_rows.append({
                    'Run': run + 1,
                    'Algorithm': alg,
                    'Generation': rec['generation'],
                    'ND_Count': rec['nd_count'],
                    'HV': hv_gen,
                    'IGD': igd_gen,
                    'SP': sp_gen,
                    'Elapsed_Time(s)': rec['elapsed_time']
                })

    return records, ref_front, generation_data_rows, all_run_results, pareto_solutions_rows, ql_data_all_runs


# ============================== 13. 保存到Excel（子目录版） ==============================
def save_generation_excel(generation_data_rows, pareto_solutions_rows, records, ql_data_all_runs, ablation_results=None, subdir='', filename='fjsp_generation_data.xlsx'):
    out_dir = os.path.join('result', subdir)
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, filename)

    df_gen = pd.DataFrame(generation_data_rows)
    df_pareto = pd.DataFrame(pareto_solutions_rows)

    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df_gen.to_excel(writer, sheet_name='Generation_Data', index=False)

        if len(df_gen) > 0:
            df_avg = df_gen.groupby(['Algorithm', 'Generation']).agg({
                'ND_Count': 'mean',
                'HV': 'mean',
                'IGD': 'mean',
                'SP': 'mean',
                'Elapsed_Time(s)': 'mean'
            }).reset_index()
            df_avg.columns = ['Algorithm', 'Generation', 'Avg_ND_Count', 'Avg_HV', 'Avg_IGD', 'Avg_SP', 'Avg_Elapsed_Time(s)']
            df_avg.to_excel(writer, sheet_name='Avg_by_Generation', index=False)

        if len(df_pareto) > 0:
            df_pareto.to_excel(writer, sheet_name='Pareto_Solutions', index=False)

        summary_rows = []
        algorithms = ['NSGA2', 'NSGA2_TS', 'NSGA2_TS_QL', 'NSGA2_TS_QL_GLR']
        labels = {
            'NSGA2': 'NSGA-II',
            'NSGA2_TS': 'NSGA-II-TS',
            'NSGA2_QL': 'NSGA-II-QL',
            'NSGA2_TS_QL': 'NSGA-II-TS-QL',       
            'NSGA2_TS_QL_GLR': 'NSGA-II-TS-QL-GLR'
        }
        for alg in algorithms:
            hv_mean = np.mean(records[alg]['HV'])
            hv_std = np.std(records[alg]['HV'])
            igd_mean = np.mean(records[alg]['IGD'])
            igd_std = np.std(records[alg]['IGD'])
            time_mean = np.mean(records[alg]['Time'])
            time_std = np.std(records[alg]['Time'])
            sp_mean = np.mean(records[alg]['SP'])
            sp_std = np.std(records[alg]['SP'])

            summary_rows.append({
                'Algorithm': labels[alg],
                'HV_Mean': hv_mean,
                'HV_Std': hv_std,
                'IGD_Mean': igd_mean,
                'IGD_Std': igd_std,
                'SP_Mean': sp_mean,
                'SP_Std': sp_std,
                'Time_Mean(s)': time_mean,
                'Time_Std(s)': time_std
            })

        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_excel(writer, sheet_name='Final_Statistics', index=False)

        # TS-QL 黑盒数据
        if ql_data_all_runs:
            ql_hist_rows = []
            for run_data in ql_data_all_runs:
                run_id = run_data['run']
                for h in run_data['q_history']:
                    ql_hist_rows.append({
                        'Run': run_id,
                        'Generation': h['generation'],
                        'Max_Q': h['max_q'],
                        'Mean_Q': h['mean_q'],
                        'Std_Q': h['std_q'],
                        'Min_Q': h['min_q'],
                        'Epsilon': h['epsilon'],
                        'Nonzero_Ratio': h['nonzero_ratio']
                    })
            if ql_hist_rows:
                pd.DataFrame(ql_hist_rows).to_excel(writer, sheet_name='TSQL_Q_History', index=False)

            ql_action_rows = []
            for run_data in ql_data_all_runs:
                run_id = run_data['run']
                for a in run_data['action_history']:
                    ql_action_rows.append({
                        'Run': run_id,
                        'Generation': a['generation'],
                        'State': a['state'],
                        'Action': a['action'],
                        'Reward': a['reward'],
                        'HV_Delta_Pct': a['hv_delta_pct'],
                        'IGD_Delta_Pct': a.get('igd_delta_pct', 0.0),      # 改为IGD
                        'ND_Delta': a['nd_delta'],
                        'Improved_Count': a.get('improved_count', 0),
                        'HV_Before': a.get('hv_before', 0.0),
                        'HV_After': a.get('hv_after', 0.0),
                        'IGD_Before': a.get('igd_before', float('inf')),   # 改为IGD
                        'IGD_After': a.get('igd_after', float('inf')),     # 改为IGD
                        'Epsilon': a['epsilon']
                    })
            if ql_action_rows:
                pd.DataFrame(ql_action_rows).to_excel(writer, sheet_name='TSQL_Action_History', index=False)

            if ql_data_all_runs[0]['q_table'] is not None:
                ql_data_all_runs[0]['q_table'].to_excel(writer, sheet_name='TSQL_Q_Table_Run1')

            if ql_data_all_runs[0]['visit_matrix'] is not None:
                ql_data_all_runs[0]['visit_matrix'].to_excel(writer, sheet_name='TSQL_Visit_Run1')

            if ql_data_all_runs[0]['policy']:
                policy_rows = [{'State': s, 'Action': a} for s, a in ql_data_all_runs[0]['policy'].items()]
                pd.DataFrame(policy_rows).to_excel(writer, sheet_name='TSQL_Policy_Run1', index=False)

        # Wilcoxon 检验结果
        if ablation_results:
            wilcoxon_rows = []
            for comparison_name, result in ablation_results.items():
                wilcoxon_rows.append({
                    'Comparison': comparison_name,
                    'HV_p': result['hv_p'],
                    'IGD_p': result['igd_p'],
                    'SP_p': result['sp_p'],
                    'Significant': result['significant']
                })
            pd.DataFrame(wilcoxon_rows).to_excel(writer, sheet_name='Wilcoxon_Test', index=False)

    print(f"\n[Excel 已保存: {filepath}]")
    print(f"  - Sheet 'Generation_Data': {len(df_gen)} 行每代聚合数据")
    print(f"  - Sheet 'Avg_by_Generation': 按代数和算法聚合的平均值")
    print(f"  - Sheet 'Pareto_Solutions': {len(df_pareto)} 行最终帕累托解详细数据")
    print(f"  - Sheet 'Final_Statistics': 最终统计汇总")
    print(f"  - TS-QL Sheets: Q历史 / 动作历史(HV+IGD) / Q表 / 访问矩阵 / 策略")


# ============================== 14. 每次运行的收敛对比图（子目录版） ==============================
def plot_per_run_convergence(generation_data_rows, n_runs, subdir=''):
    out_dir = os.path.join('result', subdir)
    os.makedirs(out_dir, exist_ok=True)
    df_gen = pd.DataFrame(generation_data_rows)

    labels = {
        'NSGA2': 'NSGA-II',
        'NSGA2_TS': 'NSGA-II-TS',
        'NSGA2_QL': 'NSGA-II-QL',
        'NSGA2_TS_QL': 'NSGA-II-TS-QL',
        'NSGA2_TS_QL_GLR': 'NSGA-II-TS-QL-GLR'
    }
    colors = {
        'NSGA2': '#E74C3C',
        'NSGA2_TS': '#3498DB',
        'NSGA2_QL': '#9B59B6',
        'NSGA2_TS_QL': '#F39C12',
        'NSGA2_TS_QL_GLR': '#1ABC9C'
    }

    for run in range(1, n_runs + 1):
        df_run = df_gen[df_gen['Run'] == run]

        metrics = [
            ('ND_Count', 'Non-dominated Count'),
            ('HV', 'Hypervolume'),
            ('IGD', 'IGD (Normalized)'),
            ('SP', 'SP (Spacing)')
        ]

        for metric, ylabel in metrics:
            fig, ax = plt.subplots(figsize=(8, 5))
            for alg in ['NSGA2', 'NSGA2_TS', 'NSGA2_TS_QL', 'NSGA2_TS_QL_GLR']:
                df_alg = df_run[df_run['Algorithm'] == alg]
                if len(df_alg) > 0:
                    vals = df_alg[metric].replace([np.inf, -np.inf], np.nan).values
                    gens = df_alg['Generation'].values
                    valid = ~np.isnan(vals)
                    if np.any(valid):
                        ax.plot(gens[valid], vals[valid],
                               label=labels[alg], color=colors[alg], linewidth=1.5, marker='o', markersize=2)
            ax.set_xlabel('Generation', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'Run {run} - {ylabel} Convergence', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            suffix = metric.lower().replace('(', '').replace(')', '').replace(' ', '_')
            plt.savefig(os.path.join(out_dir, f'run_{run:02d}_{suffix}.png'), dpi=300, bbox_inches='tight')
            plt.close()

    print(f"\n[每次运行收敛图已保存到 {out_dir}/ ，共 {n_runs} 次运行 × 4 张图]")


# ============================== 15. 每次运行的帕累托前沿对比图（2D投影，子目录版） ==============================
def plot_pareto_front_per_run(all_run_results, n_runs, subdir=''):
    out_dir = os.path.join('result', subdir)
    os.makedirs(out_dir, exist_ok=True)

    labels = {
        'NSGA2': 'NSGA-II',
        'NSGA2_TS': 'NSGA-II-TS',
        'NSGA2_QL': 'NSGA-II-QL',
        'NSGA2_TS_QL': 'NSGA-II-TS-QL',
        'NSGA2_TS_QL_GLR': 'NSGA-II-TS-QL-GLR'
    }
    colors = {
        'NSGA2': '#E74C3C',
        'NSGA2_TS': '#3498DB',
        'NSGA2_QL': '#9B59B6',
        'NSGA2_TS_QL': '#F39C12',
        'NSGA2_TS_QL_GLR': '#1ABC9C'
    }

    projections = [(0, 1), (0, 2), (1, 2)]
    axis_labels = [
        ('Makespan', 'Energy/100'),
        ('Makespan', 'Quality Loss ×10000'),
        ('Energy/100', 'Quality Loss ×10000')
    ]

    for run in range(n_runs):
        results = all_run_results[run]
        seed = 42 + run * 5

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, (ax, (dim_x, dim_y), (lx, ly)) in enumerate(zip(axes, projections, axis_labels)):
            for alg in ['NSGA2', 'NSGA2_TS', 'NSGA2_TS_QL', 'NSGA2_TS_QL_GLR']:
                F = results[alg]['F']
                if len(F) > 0:
                    ax.scatter(
                        F[:, dim_x], F[:, dim_y],
                        c=colors[alg],
                        marker='o',
                        label=labels[alg],
                        s=40,
                        alpha=0.7,
                        edgecolors='black',
                        linewidth=0.5
                    )

            ax.set_xlabel(lx, fontsize=10)
            ax.set_ylabel(ly, fontsize=10)
            ax.set_title(f'{lx} vs {ly}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            ax.grid(alpha=0.3)

        fig.suptitle(
            f'Run {run+1} Pareto Front Comparison (seed={seed})',
            fontsize=13,
            fontweight='bold',
            y=1.02
        )
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'run_{run+1:02d}_pareto_front.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\n[帕累托前沿2D投影对比图已保存到 {out_dir}/ ，共 {n_runs} 张]")


# ============================== 16. 每次运行的帕累托前沿对比图（3D，子目录版） ==============================
def plot_pareto_front_3d_per_run(all_run_results, n_runs, subdir=''):
    out_dir = os.path.join('result', subdir)
    os.makedirs(out_dir, exist_ok=True)

    labels = {
        'NSGA2': 'NSGA-II',
        'NSGA2_TS': 'NSGA-II-TS',
        'NSGA2_QL': 'NSGA-II-QL',
        'NSGA2_TS_QL': 'NSGA-II-TS-QL',
        'NSGA2_TS_QL_GLR': 'NSGA-II-TS-QL-GLR'
    }
    colors = {
        'NSGA2': '#E74C3C',
        'NSGA2_TS': '#3498DB',
        'NSGA2_QL': '#9B59B6',
        'NSGA2_TS_QL': '#F39C12',
        'NSGA2_TS_QL_GLR': '#1ABC9C'
    }

    for run in range(n_runs):
        results = all_run_results[run]
        seed = 42 + run * 5

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for alg in ['NSGA2', 'NSGA2_TS', 'NSGA2_TS_QL', 'NSGA2_TS_QL_GLR']:
            F = results[alg]['F']
            if len(F) > 0:
                ax.scatter(
                    F[:, 0], F[:, 1], F[:, 2],
                    c=colors[alg],
                    marker='o',
                    label=labels[alg],
                    s=50,
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=0.5
                )

        ax.set_xlabel('Makespan', fontsize=11, labelpad=10)
        ax.set_ylabel('Energy/100', fontsize=11, labelpad=10)
        ax.set_zlabel(r'Quality Loss $\times$10000', fontsize=11, labelpad=20)
        ax.set_title(f'3D Pareto Front Comparison', fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)

        # plt.tight_layout()
        # 手动留边距，确保 z 轴标签完整显示
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
        plt.savefig(os.path.join(out_dir, f'run_{run+1:02d}_pareto_front_3d.png'), dpi=300)
        plt.close()

    print(f"\n[帕累托前沿3D对比图已保存到 {out_dir}/ ，共 {n_runs} 张]")


# ============================== 17. 甘特图绘制（动态适配版） ==============================
def plot_gantt_chart(operations, title, filename, makespan, energy, quality):
    if not operations:
        print(f"[Warning] 无工序数据，跳过甘特图: {filename}")
        return

    n_jobs = max(op['job'] for op in operations) + 1
    n_machines = max(op['machine'] for op in operations) + 1
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_jobs)))

    fig, ax = plt.subplots(figsize=(16, 8))

    for op in operations:
        job = op['job']
        machine = op['machine']
        start = op['start']
        end = op['end']
        duration = end - start

        ax.barh(machine, duration, left=start, height=0.6,
                color=colors[job % len(colors)], edgecolor='black', linewidth=0.5, alpha=0.85)

        ax.text(start + duration / 2, machine, f'J{job+1}-O{op["op"]+1}',
                ha='center', va='center', fontsize=6, color='white',
                fontweight='bold', rotation=0)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Machine', fontsize=12)
    ax.set_title(
        f'{title}\nMakespan={makespan:.2f} | Energy={energy:.2f} | Quality Loss={quality:.6f}',
        fontsize=12, fontweight='bold'
    )
    ax.set_yticks(range(n_machines))
    ax.set_yticklabels([f'M{i+1}' for i in range(n_machines)])
    ax.set_ylim(-0.5, n_machines - 0.5)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[j % len(colors)], edgecolor='black', label=f'Job {j+1}')
                       for j in range(n_jobs)]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
              title='Jobs', title_fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_gantt_per_run(all_run_results, n_runs, subdir=''):
    out_dir = os.path.join('result', subdir)
    os.makedirs(out_dir, exist_ok=True)

    obj_labels = {
        'makespan': 'Makespan-Optimal',
        'energy': 'Energy-Optimal',
        'quality': 'Quality-Optimal'
    }

    for run in range(n_runs):
        results = all_run_results[run]
        gantt = results['NSGA2_TS_QL_GLR'].get('gantt', {})

        if not gantt:
            continue

        for obj_name, data in gantt.items():
            filename = os.path.join(out_dir, f'run_{run+1:02d}_gantt_{obj_name}.png')
            plot_gantt_chart(
                operations=data['operations'],
                title=f'NSGA-II-TS-QL-GLR {obj_labels[obj_name]}',
                filename=filename,
                makespan=data['makespan'],
                energy=data['energy'],
                quality=data['quality']
            )

    print(f"\n[甘特图已保存到 {out_dir}/ ，共 {n_runs} 次运行 × 3 张 = {n_runs*3} 张]")


# ============================== 18. QL 黑盒分析图表（TS-QL版，适配HV+IGD奖励） ==============================
def plot_ql_analysis(ql_data_all_runs, n_runs, subdir=''):
    out_dir = os.path.join('result', subdir)
    os.makedirs(out_dir, exist_ok=True)

    if not ql_data_all_runs:
        print("[QL分析] 无QL数据，跳过")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for run_data in ql_data_all_runs:
        run_id = run_data['run']
        q_hist = run_data['q_history']
        if not q_hist:
            continue

        gens = [h['generation'] for h in q_hist]
        max_q = [h['max_q'] for h in q_hist]
        mean_q = [h['mean_q'] for h in q_hist]
        std_q = [h['std_q'] for h in q_hist]
        epsilon = [h['epsilon'] for h in q_hist]

        axes[0, 0].plot(gens, max_q, alpha=0.6, label=f'Run {run_id}')
        axes[0, 1].plot(gens, mean_q, alpha=0.6, label=f'Run {run_id}')
        axes[1, 0].plot(gens, std_q, alpha=0.6, label=f'Run {run_id}')
        axes[1, 1].plot(gens, epsilon, alpha=0.6, label=f'Run {run_id}')

    axes[0, 0].set_title('Max Q-value per Generation', fontweight='bold')
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Max Q')
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].set_title('Mean Q-value per Generation', fontweight='bold')
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Mean Q')
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].set_title('Std Q-value per Generation', fontweight='bold')
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Std Q')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].set_title('Epsilon Decay', fontweight='bold')
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('Epsilon')
    axes[1, 1].grid(alpha=0.3)

    if n_runs <= 10:
        for ax in axes.flat:
            ax.legend(fontsize=7, loc='best')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'tsql_q_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[TS-QL Q收敛曲线已保存: {out_dir}/tsql_q_convergence.png]")

    q_table = ql_data_all_runs[0]['q_table']
    if q_table is not None and not q_table.empty:
        fig, ax = plt.subplots(figsize=(16, 6))
        q_display = q_table.iloc[:, :min(20, len(q_table.columns))]
        im = ax.imshow(q_display.values, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)

        ax.set_xticks(range(len(q_display.columns)))
        ax.set_xticklabels(q_display.columns, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(len(q_display.index)))
        ax.set_yticklabels(q_display.index, fontsize=9)
        ax.set_title('TS-QL Q-Table Heatmap (Run 1)', fontweight='bold', fontsize=12)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Q Value', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'tsql_q_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[TS-QL Q值热力图已保存: {out_dir}/tsql_q_heatmap.png]")

    action_hist = ql_data_all_runs[0]['action_history']
    if action_hist:
        sa_count = defaultdict(int)
        for a in action_hist:
            sa_count[(a['state'], a['action'])] += 1

        states = sorted(list(set(a['state'] for a in action_hist)))
        actions = sorted(list(set(a['action'] for a in action_hist)))

        heatmap_data = np.zeros((len(states), len(actions)))
        for i, s in enumerate(states):
            for j, act in enumerate(actions):
                heatmap_data[i, j] = sa_count.get((s, act), 0)

        fig, ax = plt.subplots(figsize=(max(12, len(actions)*0.5), max(6, len(states)*0.8)))
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(range(len(actions)))
        ax.set_xticklabels(actions, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(len(states)))
        ax.set_yticklabels(states, fontsize=9)
        ax.set_title('TS-QL State-Action Visit Frequency Heatmap (Run 1)', fontweight='bold', fontsize=12)
        ax.set_xlabel('Action', fontsize=10)
        ax.set_ylabel('State', fontsize=10)

        for i in range(len(states)):
            for j in range(len(actions)):
                if heatmap_data[i, j] > 0:
                    ax.text(j, i, int(heatmap_data[i, j]), ha='center', va='center', 
                           fontsize=6, color='white' if heatmap_data[i, j] > heatmap_data.max()*0.5 else 'black')

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Visit Count', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'tsql_visit_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[TS-QL 访问热力图已保存: {out_dir}/tsql_visit_heatmap.png]")

    if action_hist:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        rewards = [a['reward'] for a in action_hist]
        hv_deltas = [a['hv_delta_pct'] for a in action_hist]

        axes[0].hist(rewards, bins=30, color='#3498DB', edgecolor='black', alpha=0.7)
        axes[0].set_title('TS-QL Reward Distribution (Run 1)', fontweight='bold')
        axes[0].set_xlabel('Reward')
        axes[0].set_ylabel('Frequency')
        axes[0].axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean={np.mean(rewards):.2f}')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].scatter(hv_deltas, rewards, alpha=0.6, c='#1ABC9C', edgecolors='black', linewidth=0.3)
        axes[1].set_title('TS-QL Reward vs HV Delta % (Run 1)', fontweight='bold')
        axes[1].set_xlabel('HV Delta %')
        axes[1].set_ylabel('Reward')
        axes[1].grid(alpha=0.3)

        if len(hv_deltas) > 1:
            z = np.polyfit(hv_deltas, rewards, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(hv_deltas), max(hv_deltas), 100)
            axes[1].plot(x_line, p(x_line), 'r--', alpha=0.8, label=f'Trend (slope={z[0]:.2f})')
            axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'tsql_reward_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[TS-QL 奖励分析图已保存: {out_dir}/tsql_reward_analysis.png]")

    print("\n" + "=" * 80)
    print("TS-QL 确定性诊断报告")
    print("=" * 80)

    for run_data in ql_data_all_runs:
        run_id = run_data['run']
        q_hist = run_data['q_history']
        action_hist = run_data['action_history']
        policy = run_data['policy']

        if not q_hist or not action_hist:
            continue

        final_eps = q_hist[-1]['epsilon']
        mid = len(q_hist) // 2
        early_mean = np.mean([h['mean_q'] for h in q_hist[:mid]]) if mid > 0 else 0
        late_mean = np.mean([h['mean_q'] for h in q_hist[mid:]]) if mid > 0 else 0
        q_convergence = abs(late_mean - early_mean)

        state_counts = defaultdict(int)
        for a in action_hist:
            state_counts[a['state']] += 1
        total_visits = sum(state_counts.values())
        state_entropy = -sum((c/total_visits) * np.log(c/total_visits + 1e-9) for c in state_counts.values())
        max_entropy = np.log(len(state_counts)) if len(state_counts) > 0 else 1
        entropy_ratio = state_entropy / max_entropy if max_entropy > 0 else 0

        action_counts = defaultdict(int)
        for a in action_hist:
            action_counts[a['action']] += 1
        top3_ratio = sum(sorted(action_counts.values(), reverse=True)[:3]) / total_visits if total_visits > 0 else 0

        rewards_all = [a['reward'] for a in action_hist]
        reward_trend = np.polyfit(range(len(rewards_all)), rewards_all, 1)[0] if len(rewards_all) > 1 else 0

        is_deterministic = final_eps < 0.15 and entropy_ratio > 0.5 and q_convergence < 10

        print(f"\n[Run {run_id}] TS-QL 诊断结果:")
        print(f"  • 最终 Epsilon: {final_eps:.4f} {'(已收敛)' if final_eps < 0.15 else '(仍在探索)'}")
        print(f"  • Q值收敛度: {q_convergence:.2f} {'(已稳定)' if q_convergence < 10 else '(仍在变化)'}")
        print(f"  • 状态熵比率: {entropy_ratio:.2%} {'(分布均衡)' if entropy_ratio > 0.5 else '(分布偏斜)'}")
        print(f"  • Top3动作占比: {top3_ratio:.1%} {'(偏好明显)' if top3_ratio > 0.6 else '(选择分散)'}")
        print(f"  • 奖励趋势斜率: {reward_trend:+.4f}/step {'(奖励上升)' if reward_trend > 0 else '(奖励下降/平稳)'}")
        print(f"  • 综合判定: {'✓ 有意义的策略学习' if is_deterministic else '✗ 策略尚未稳定或存在随机震荡'}")

    print("=" * 80)


# ============================== 19. 统计分析与可视化（4算法对比版） ==============================
def analyze_statistics(records, ref_front, subdir=''):
    print("\n" + "=" * 95)
    print(f"4算法对比实验统计结果 ({len(records['NSGA2']['HV'])} 次运行)")
    print("=" * 95)

    algorithms = ['NSGA2', 'NSGA2_TS', 'NSGA2_TS_QL', 'NSGA2_TS_QL_GLR']
    labels = {
        'NSGA2': 'NSGA-II',
        'NSGA2_TS': 'NSGA-II-TS',
        'NSGA2_QL': 'NSGA-II-QL',
        'NSGA2_TS_QL': 'NSGA-II-TS-QL',  
        'NSGA2_TS_QL_GLR': 'NSGA-II-TS-QL-GLR'
    }

    print(f"\n{'Algorithm':<18} {'HV (mean±std)':<22} {'IGD (mean±std)':<22} {'SP (mean±std)':<22} {'Time(s) (mean±std)':<22}")
    print("-" * 115)
    stats = {}
    for alg in algorithms:
        hv_mean = np.mean(records[alg]['HV'])
        hv_std = np.std(records[alg]['HV'])
        igd_mean = np.mean(records[alg]['IGD'])
        igd_std = np.std(records[alg]['IGD'])
        sp_mean = np.mean(records[alg]['SP'])
        sp_std = np.std(records[alg]['SP'])
        time_mean = np.mean(records[alg]['Time'])
        time_std = np.std(records[alg]['Time'])

        stats[alg] = {
            'HV_mean': hv_mean, 'HV_std': hv_std,
            'IGD_mean': igd_mean, 'IGD_std': igd_std,
            'SP_mean': sp_mean, 'SP_std': sp_std,
            'Time_mean': time_mean, 'Time_std': time_std
        }

        print(f"{labels[alg]:<18} {hv_mean:>9.2f}±{hv_std:<7.2f}  "
              f"{igd_mean:>9.4f}±{igd_std:<7.4f}  "
              f"{sp_mean:>9.4f}±{sp_std:<7.4f}  "
              f"{time_mean:>9.2f}±{time_std:<7.2f}")

    # Wilcoxon 秩和检验
    print("\n" + "=" * 95)
    print("Wilcoxon 秩和检验（配对样本，显著性水平 α=0.05）")
    print("=" * 95)
    print(f"{'对比组':<55} {'HV p':<12} {'IGD p':<12} {'SP p':<12} {'结论'}")
    print("-" * 110)

    comparisons = [
        ('NSGA2_TS_QL_GLR', 'NSGA2_TS_QL', 'NSGA-II-TS-QL-GLR vs NSGA-II-TS-QL (验证初始化贡献)'),   
        ('NSGA2_TS_QL_GLR', 'NSGA2_TS', 'NSGA-II-TS-QL-GLR vs NSGA-II-TS (验证TS-QL增量)'),
        ('NSGA2_TS_QL_GLR', 'NSGA2', 'NSGA-II-TS-QL-GLR vs NSGA-II (验证整体提升)'),
        ('NSGA2_TS_QL', 'NSGA2_TS', 'NSGA-II-TS-QL vs NSGA-II-TS (随机下TS-QL增量)'),    
        ('NSGA2_TS', 'NSGA2', 'NSGA-II-TS vs NSGA-II (验证TS单独价值)'),
    ]

    ablation_results = {}

    for alg1, alg2, name in comparisons:
        pvals = {}
        for metric in ['HV', 'IGD', 'SP']:
            data1 = records[alg1][metric]
            data2 = records[alg2][metric]
            try:
                stat, p = wilcoxon(data1, data2)
            except ValueError:
                p = 1.0
            pvals[metric] = p

        hv_p = pvals['HV']
        igd_p = pvals['IGD']
        sp_p = pvals['SP']

        sig = "显著差异" if (hv_p < 0.05 or igd_p < 0.05 or sp_p < 0.05) else "无显著差异"
        print(f"{name:<55} {hv_p:<12.4f} {igd_p:<12.4f} {sp_p:<12.4f} {sig}")
        
        ablation_results[name] = {
            'hv_p': hv_p, 'igd_p': igd_p, 'sp_p': sp_p, 'significant': sig
        }

    # 箱线图（仅保留 HV 和 IGD）
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    metrics = ['HV', 'IGD']
    titles = ['Hypervolume (larger is better)', 'IGD (smaller is better)']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        data_to_plot = [records[alg][metric] for alg in algorithms]
        bp = ax.boxplot(data_to_plot, tick_labels=[labels[a] for a in algorithms], patch_artist=True)

        colors = ['#E74C3C', '#3498DB', '#F39C12', '#1ABC9C']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        if metric == 'HV':
            best_idx = np.argmax([np.mean(records[alg][metric]) for alg in algorithms])
        else:
            best_idx = np.argmin([np.mean(records[alg][metric]) for alg in algorithms])
        ax.annotate('Best', xy=(best_idx + 1, np.mean(data_to_plot[best_idx])),
                   xytext=(best_idx + 1, np.mean(data_to_plot[best_idx]) * 1.05),
                   ha='center', fontsize=9, color='red',
                   arrowprops=dict(arrowstyle='->', color='red'))

    plt.tight_layout()
    out_dir = os.path.join('result', subdir)
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'fjsp_ablation_4algorithms.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[消融实验箱线图已保存: {save_path}]")
    plt.show()

    return stats, ablation_results


# ============================== 20. 主程序 ==============================
if __name__ == "__main__":
    raise SystemExit("请运行同目录的 敏感性实验.py；此文件只提供修回模型与算法类。")
    DATA_PATH = "MK06.xlsx"

    g_max_gen = 1000
    g_start_gen = max(1, g_max_gen * 3 // 10)
    N_RUNS = 20

    OUTPUT_SUBDIR = get_output_dir()
    print(f"[输出子目录: result/{OUTPUT_SUBDIR}/]")
    print(f"[对比实验设计：4算法对比]")
    print(f"  1. NSGA-II           : 基线 (随机初始化, 固定Pc/Pm)")
    print(f"  2. NSGA-II-TS          : 基线 + 固定参数TS (验证TS独立贡献)")
    print(f"  3. NSGA-II-TS-QL       : 完整TS-QL机制 + 随机初始化 (验证初始化贡献)")
    print(f"  4. NSGA-II-TS-QL-GLR     : 完整TS-QL机制 + 混合初始化 (完整算法)")
    print(f"[初始化策略: 1/2/3=随机 | 4=GS:LS:RS=3:2:5]")
    print(f"[评价指标: HV + IGD(归一化,预构建参考前沿) + SP]")
    print(f"[QL奖励: HV + IGD 双指标 (已移除SP奖励项)]")
    print(f"[修复1] 边界基于所有原始解并留10%余量")
    print(f"[修复2] 预运行使用多算法(NSGA2/TS/TS-QL)构建参考前沿")
    print(f"[修复3] 最终评估使用后验参考前沿(所有运行合并)重新计算IGD")

    records, ref_front, generation_data_rows, all_run_results, pareto_solutions_rows, ql_data_all_runs = run_statistical_comparison(
        DATA_PATH, max_gen=g_max_gen, n_runs=N_RUNS, time_limit='100s'
    )

    # 先进行统计分析，获取 Wilcoxon 检验结果
    stats, ablation_results = analyze_statistics(records, ref_front, subdir=OUTPUT_SUBDIR)
    
    # 保存到 Excel（包含 Wilcoxon 检验结果）
    save_generation_excel(generation_data_rows, pareto_solutions_rows, records, ql_data_all_runs, ablation_results, subdir=OUTPUT_SUBDIR, filename='fjsp_generation_data.xlsx')
    
    plot_per_run_convergence(generation_data_rows, n_runs=N_RUNS, subdir=OUTPUT_SUBDIR)
    plot_pareto_front_per_run(all_run_results, n_runs=N_RUNS, subdir=OUTPUT_SUBDIR)
    plot_pareto_front_3d_per_run(all_run_results, n_runs=N_RUNS, subdir=OUTPUT_SUBDIR)
    plot_gantt_per_run(all_run_results, n_runs=N_RUNS, subdir=OUTPUT_SUBDIR)
    plot_ql_analysis(ql_data_all_runs, n_runs=N_RUNS, subdir=OUTPUT_SUBDIR)
