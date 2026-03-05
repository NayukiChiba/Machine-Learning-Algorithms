"""
评估模块
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from itertools import permutations
from utils.decorate import print_func_info
from train_model import train_model
from generate_data import generate_data
from preprocess_data import preprocess_data


def _best_state_mapping(y_true, y_pred, n_states: int):
    """
    寻找最优状态映射（解决状态标签置换问题）
    """
    best_acc = 0.0
    best_map = None

    for perm in permutations(range(n_states)):
        mapped = np.array([perm[s] for s in y_pred])
        acc = (mapped == y_true).mean()
        if acc > best_acc:
            best_acc = acc
            best_map = perm

    return best_map, best_acc


@print_func_info
def evaluate_model(model, X_obs, lengths, y_true):
    """
    评估 HMM 模型

    args:
        model: HMM 模型
        X_obs: 观测序列
        lengths: 序列长度列表
        y_true: 真实隐状态
    returns:
        y_pred
    """
    # 预测隐状态序列
    y_pred = model.predict(X_obs, lengths)

    # 对齐状态标签
    n_states = len(set(y_pred))
    best_map, best_acc = _best_state_mapping(y_true, y_pred, n_states)
    print(f"最佳状态映射: {best_map}")
    print(f"对齐后准确率: {best_acc:.4f}")

    # 对数似然
    log_likelihood = model.score(X_obs, lengths)
    print(f"对数似然: {log_likelihood:.4f}")

    return y_pred


if __name__ == "__main__":
    X_obs, lengths, y_true, n_symbols = preprocess_data(generate_data())
    model = train_model(X_obs, lengths)
    evaluate_model(model, X_obs, lengths, y_true)
