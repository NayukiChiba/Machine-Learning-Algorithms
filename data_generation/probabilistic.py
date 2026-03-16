"""
data_generation/probabilistic.py
概率与序列模型数据生成模块
统一管理 EM(GMM)和 HMM 各自适用的数据集生成函数
"""

from dataclasses import dataclass, field
import numpy as np
from pandas import DataFrame


@dataclass
class ProbabilisticData:
    """
    概率与序列模型数据生成器
    """

    # --- 共享属性 ---
    n_samples: int = 500
    random_state: int = 42

    # --- EM(GMM)专属参数 ---
    em_n_components: int = 3  # 高斯分量数量(混合成分数)
    em_means: list = field(
        default_factory=lambda: [
            [0.0, 0.0],
            [4.0, 4.0],
            [-3.0, 4.0],
        ]
    )  # 各分量均值
    em_stds: list = field(
        default_factory=lambda: [
            [0.8, 0.5],
            [0.6, 1.0],
            [1.2, 0.7],
        ]
    )  # 各分量标准差(各维度独立)
    em_weights: list = field(default_factory=lambda: [0.5, 0.3, 0.2])  # 混合权重

    # --- HMM 专属参数 ---
    hmm_n_steps: int = 300  # 序列长度
    hmm_pi: list = field(default_factory=lambda: [0.6, 0.3, 0.1])  # 初始状态分布
    hmm_A: list = field(
        default_factory=lambda: [  # 状态转移矩阵
            [0.80, 0.15, 0.05],
            [0.20, 0.60, 0.20],
            [0.10, 0.20, 0.70],
        ]
    )
    hmm_B: list = field(
        default_factory=lambda: [  # 发射矩阵(离散观测)
            [0.60, 0.30, 0.10],
            [0.20, 0.50, 0.30],
            [0.10, 0.20, 0.70],
        ]
    )

    def em(self) -> DataFrame:
        """
        手动合成的高斯混合模型数据(GMM)
        特点:3 个分量均值、权重、形状各不相同, 比 make_blobs 更能体现 GMM 的椭圆边界
        注意:true_label 仅用于训练后评估对比, EM 训练时不使用标签

        Returns:
            DataFrame: 列包含 x1, x2, true_label(所属高斯分量, 仅用于评估对比)
        """
        rng = np.random.RandomState(self.random_state)
        weights = np.array(self.em_weights)
        n_components = self.em_n_components

        # 按权重分配各分量样本数量
        counts = rng.multinomial(self.n_samples, weights)

        X_list, y_list = [], []
        for k in range(n_components):
            mean = np.array(self.em_means[k])
            std = np.array(self.em_stds[k])
            X_k = rng.randn(counts[k], 2) * std + mean
            X_list.append(X_k)
            y_list.extend([k] * counts[k])

        X = np.vstack(X_list)
        y = np.array(y_list)

        # 打乱顺序
        idx = rng.permutation(len(y))
        X, y = X[idx], y[idx]

        return DataFrame({"x1": X[:, 0], "x2": X[:, 1], "true_label": y})

    def hmm(self) -> DataFrame:
        """
        手动参数化的 HMM 离散序列数据
        特点:3 个隐状态、3 种观测符号, 状态间转移概率各不相同
               state_true 是隐状态(训练时不可见), obs 是观测序列(训练输入)
        注意:HMM 是序列模型, n_samples 对此方法无效, 使用 hmm_n_steps 控制长度

        Returns:
            DataFrame: 列包含 time(时间步), obs(观测符号), state_true(真实隐状态)
        """
        rng = np.random.default_rng(self.random_state)

        pi = np.array(self.hmm_pi)
        A = np.array(self.hmm_A)
        B = np.array(self.hmm_B)
        n_steps = self.hmm_n_steps
        n_states = len(pi)

        states = np.zeros(n_steps, dtype=int)
        obs = np.zeros(n_steps, dtype=int)

        # 初始状态采样
        states[0] = rng.choice(n_states, p=pi)
        obs[0] = rng.choice(B.shape[1], p=B[states[0]])

        # 逐步按转移矩阵和发射矩阵采样
        for t in range(1, n_steps):
            states[t] = rng.choice(n_states, p=A[states[t - 1]])
            obs[t] = rng.choice(B.shape[1], p=B[states[t]])

        return DataFrame(
            {
                "time": np.arange(n_steps),
                "obs": obs,
                "state_true": states,
            }
        )


probabilistic_data = ProbabilisticData()
em_data = probabilistic_data.em()
hmm_data = probabilistic_data.hmm()
