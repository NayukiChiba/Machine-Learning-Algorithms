"""
pipelines/probabilistic/hmm.py
HMM 端到端流水线

运行方式: python -m pipelines.probabilistic.hmm
"""

import numpy as np

from data_generation import hmm_data
from model_training.probabilistic.hmm import train_model

DATASET = "hmm"
MODEL = "hmm"


def run():
    """HMM 完整流水线"""
    print("=" * 60)
    print("HMM 流水线")
    print("=" * 60)

    data = hmm_data.copy()
    obs = data["obs"].values.astype(int)
    X_obs = obs.reshape(-1, 1)
    lengths = [len(obs)]
    y_true = data["state_true"].values.astype(int)

    model = train_model(X_obs, lengths)

    # 预测隐状态
    states_pred = model.predict(X_obs, lengths)

    # 统计准确率
    accuracy = np.mean(states_pred == y_true)
    print(f"\n隐状态预测准确率: {accuracy:.4f}")
    print(f"转移矩阵:\n{model.transmat_.round(3)}")

    print(f"\n{'=' * 60}")
    print("HMM 流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
