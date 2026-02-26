"""
评估模块
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from utils.decorate import print_func_info
from train_model import train_model
from generate_data import generate_data
from preprocess_data import preprocess_data


@print_func_info
def evaluate_model(model, X_scaled):
    """
    评估 PCA 模型

    args:
        model: PCA 模型
        X_scaled: 标准化后的特征
    returns:
        X_pca
    """
    X_pca = model.transform(X_scaled)

    # 解释方差比
    evr = model.explained_variance_ratio_
    cum = np.cumsum(evr)
    print(f"解释方差比: {evr.round(4)}")
    print(f"累计解释方差: {cum.round(4)}")

    return X_pca


if __name__ == "__main__":
    X_scaled, scaler, X, y = preprocess_data(generate_data())
    model = train_model(X_scaled, n_components=2)
    evaluate_model(model, X_scaled)
