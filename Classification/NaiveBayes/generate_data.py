"""
生成用于朴素贝叶斯分类的二维数据
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pandas import DataFrame
from sklearn.datasets import make_classification
from utils.decorate import print_func_info


@print_func_info
def generate_data(
    n_samples: int = 500,
    class_sep: float = 1.1,
    flip_y: float = 0.03,
    random_state: int = 42,
) -> DataFrame:
    """
    生成二分类数据（二维特征，方便可视化）

    args:
        n_samples: 样本数量
        class_sep: 类别间隔
        flip_y: 标签噪声比例
        random_state: 随机种子
    returns:
        DataFrame: 包含 x1, x2, label 三列
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=random_state,
    )

    data = DataFrame({"x1": X[:, 0], "x2": X[:, 1], "label": y})
    return data


if __name__ == "__main__":
    print(generate_data().head())
