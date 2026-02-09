"""
生成适合 SVM 的数据
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于直接导入公共工具
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
    # make_moons 生成二维特征 + 二分类标签
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    # 将 numpy 数据封装成 DataFrame，方便后续处理与可视化
    data = DataFrame(
        {
            "x1": X[:, 0],  # 第 1 维特征
            "x2": X[:, 1],  # 第 2 维特征
            "label": y,  # 类别标签（0/1）
        }
    )
    return data


if __name__ == "__main__":
    # 模块自测：查看生成的数据前几行
    print(generate_data().head())
