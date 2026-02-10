"""
生成用于 EM (GMM) 的二维数据
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pandas import DataFrame
from sklearn.datasets import make_blobs
from utils.decorate import print_func_info


@print_func_info
def generate_data(
    n_samples: int = 500,
    centers: int = 3,
    cluster_std: float = 0.9,
    random_state: int = 42,
) -> DataFrame:
    """
    生成二维高斯簇数据

    args:
        n_samples: 样本数量
        centers: 簇数量
        cluster_std: 簇内方差
        random_state: 随机种子
    returns:
        DataFrame: 包含 x1, x2 两列
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    data = DataFrame({"x1": X[:, 0], "x2": X[:, 1]})
    return data


if __name__ == "__main__":
    print(generate_data().head())
