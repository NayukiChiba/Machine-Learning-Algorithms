"""
概率与序列数据集加载器
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pandas import DataFrame


@dataclass
class ProbabilisticDatasetFactory:
    """概率与序列数据集工厂。"""

    nSamples: int = 500
    randomState: int = 42
    nSteps: int = 300
    hmmPi: list[float] = field(default_factory=lambda: [0.6, 0.3, 0.1])
    hmmTransition: list[list[float]] = field(
        default_factory=lambda: [
            [0.80, 0.15, 0.05],
            [0.20, 0.60, 0.20],
            [0.10, 0.20, 0.70],
        ]
    )
    hmmEmission: list[list[float]] = field(
        default_factory=lambda: [
            [0.60, 0.30, 0.10],
            [0.20, 0.50, 0.30],
            [0.10, 0.20, 0.70],
        ]
    )

    def loadEmDataset(self) -> DataFrame:
        """加载 EM/GMM 数据。"""
        rng = np.random.RandomState(self.randomState)
        weights = np.array([0.5, 0.3, 0.2])
        means = np.array([[0.0, 0.0], [4.0, 4.0], [-3.0, 4.0]])
        stds = np.array([[0.8, 0.5], [0.6, 1.0], [1.2, 0.7]])
        counts = rng.multinomial(self.nSamples, weights)
        XList: list[np.ndarray] = []
        yList: list[int] = []
        for index, count in enumerate(counts):
            XItem = rng.randn(count, 2) * stds[index] + means[index]
            XList.append(XItem)
            yList.extend([index] * count)
        X = np.vstack(XList)
        y = np.asarray(yList)
        indices = rng.permutation(len(y))
        X = X[indices]
        y = y[indices]
        return DataFrame({"x1": X[:, 0], "x2": X[:, 1], "true_label": y})

    def loadHmmDataset(self) -> DataFrame:
        """加载 HMM 序列数据。"""
        rng = np.random.default_rng(self.randomState)
        pi = np.asarray(self.hmmPi)
        transition = np.asarray(self.hmmTransition)
        emission = np.asarray(self.hmmEmission)
        states = np.zeros(self.nSteps, dtype=int)
        observations = np.zeros(self.nSteps, dtype=int)
        states[0] = rng.choice(len(pi), p=pi)
        observations[0] = rng.choice(emission.shape[1], p=emission[states[0]])
        for index in range(1, self.nSteps):
            states[index] = rng.choice(len(pi), p=transition[states[index - 1]])
            observations[index] = rng.choice(
                emission.shape[1], p=emission[states[index]]
            )
        return DataFrame(
            {
                "time": np.arange(self.nSteps),
                "obs": observations,
                "state_true": states,
            }
        )
