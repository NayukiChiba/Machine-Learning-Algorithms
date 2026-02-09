"""
使用 sklearn 自带数据集生成正则化回归数据
基于 diabetes 数据集，并添加相关特征与噪声特征，便于观察 Lasso/Ridge/ElasticNet 的差异
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from sklearn.datasets import load_diabetes
from utils.decorate import print_func_info


@print_func_info
def generate_data(
    add_noise_features: int = 8,
    add_corr_features: bool = True,
    random_state: int = 42,
):
    """
    生成用于正则化回归的数据集

    args:
        add_noise_features(int): 额外噪声特征数量
        add_corr_features(bool): 是否添加相关特征
        random_state(int): 随机种子
    returns:
        DataFrame: 包含特征与目标变量 price 的数据
    """
    rng = np.random.RandomState(random_state)

    # 1. 加载 sklearn 自带的 diabetes 数据集
    data = load_diabetes(as_frame=True)
    df = data.frame.copy()
    df = df.rename(columns={"target": "price"})

    # 2. 添加与原特征高度相关的特征（制造多重共线性）
    if add_corr_features:
        df["bmi_corr"] = df["bmi"] * 0.9 + rng.normal(scale=0.02, size=len(df))
        df["bp_corr"] = df["bp"] * 0.9 + rng.normal(scale=0.02, size=len(df))
        df["s5_corr"] = df["s5"] * 0.9 + rng.normal(scale=0.02, size=len(df))

    # 3. 添加纯噪声特征（用于观察 Lasso 的稀疏性）
    for i in range(add_noise_features):
        df[f"noise_{i + 1}"] = rng.normal(size=len(df))

    return df


if __name__ == "__main__":
    df = generate_data()
    print(df.head())
