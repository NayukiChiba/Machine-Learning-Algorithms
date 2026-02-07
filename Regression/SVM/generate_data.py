"""
生成适合SVM的数据
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pandas import DataFrame
from sklearn.datasets import make_moons
from utils.decorate import print_func_info


@print_func_info
def generate_data(
    n_samples: int = 400, noise: float = 0.2, random_state: int = 42
) -> DataFrame:
    """
    生成双月牙二分类数据集

    args:
        n_samples(int): 样本数量
        noise(float): 噪声大小
        random_state(int): 随机种子
    returns:
        DataFrame: 包含 x1, x2, label 三列
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    data = DataFrame({"x1": X[:, 0], "x2": X[:, 1], "label": y})
    return data


if __name__ == "__main__":
    print(generate_data().head())
