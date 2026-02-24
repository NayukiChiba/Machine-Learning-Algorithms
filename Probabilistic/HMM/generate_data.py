"""
生成用于 HMM 的序列数据
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from pandas import DataFrame
from utils.decorate import print_func_info


def _sample_hmm(pi, A, B, n_steps: int, random_state: int = 42):
    """
    根据给定参数采样 HMM 序列

    args:
        pi: 初始状态分布 (n_states,)
        A: 状态转移矩阵 (n_states, n_states)
        B: 发射矩阵 (n_states, n_symbols)
        n_steps: 序列长度
        random_state: 随机种子
    returns:
        states, obs
    """
    rng = np.random.default_rng(random_state)
    n_states = len(pi)

    states = np.zeros(n_steps, dtype=int)
    obs = np.zeros(n_steps, dtype=int)

    # 初始状态
    states[0] = rng.choice(n_states, p=pi)
    obs[0] = rng.choice(B.shape[1], p=B[states[0]])

    # 逐步采样
    for t in range(1, n_steps):
        states[t] = rng.choice(n_states, p=A[states[t - 1]])
        obs[t] = rng.choice(B.shape[1], p=B[states[t]])

    return states, obs


@print_func_info
def generate_data(
    n_steps: int = 300,
    random_state: int = 42,
) -> DataFrame:
    """
    生成一条 HMM 序列数据

    args:
        n_steps: 序列长度
        random_state: 随机种子
    returns:
        DataFrame: 包含 time, obs, state_true 三列
    """
    # 设定一个 3 状态 HMM
    pi = np.array([0.6, 0.3, 0.1])
    A = np.array(
        [
            [0.80, 0.15, 0.05],
            [0.20, 0.60, 0.20],
            [0.10, 0.20, 0.70],
        ]
    )
    B = np.array(
        [
            [0.60, 0.30, 0.10],
            [0.20, 0.50, 0.30],
            [0.10, 0.20, 0.70],
        ]
    )

    states, obs = _sample_hmm(pi, A, B, n_steps=n_steps, random_state=random_state)

    data = DataFrame(
        {
            "time": np.arange(n_steps),
            "obs": obs,
            "state_true": states,
        }
    )
    return data


if __name__ == "__main__":
    print(generate_data().head())
