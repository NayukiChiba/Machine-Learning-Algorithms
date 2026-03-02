"""
数据预处理模块
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pandas import DataFrame
from utils.decorate import print_func_info
from generate_data import generate_data


@print_func_info
def preprocess_data(data: DataFrame):
    """
    HMM 数据预处理

    args:
        data(DataFrame): 输入数据
    returns:
        X_obs, lengths, y_true, n_symbols
    """
    obs = data["obs"].values.astype(int)
    X_obs = obs.reshape(-1, 1)
    lengths = [len(obs)]
    y_true = data["state_true"].values.astype(int)
    n_symbols = int(obs.max() + 1)

    print(f"观测符号数: {n_symbols}")
    print(f"序列长度: {len(obs)}")

    return X_obs, lengths, y_true, n_symbols


if __name__ == "__main__":
    preprocess_data(generate_data())
