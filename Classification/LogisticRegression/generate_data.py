"""
生成用于逻辑回归的分类数据
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
    n_samples: int = 600,
    n_features: int = 6,
    n_informative: int = 3,
    n_redundant: int = 1,
    class_sep: float = 1.2,
    flip_y: float = 0.03,
    random_state: int = 42,
) -> DataFrame:
    """
    生成二分类数据集

    args:
        n_samples: 样本数量
        n_features: 特征数量
        n_informative: 有效特征数量
        n_redundant: 冗余特征数量
        class_sep: 类别间隔（越大越容易分类）
        flip_y: 标签噪声比例
        random_state: 随机种子
    returns:
        DataFrame: 包含特征与标签的数据表
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=2,
        class_sep=class_sep,
        flip_y=flip_y,
        weights=None,
        random_state=random_state,
    )

    columns = [f"x{i + 1}" for i in range(n_features)]
    data = DataFrame(X, columns=columns)
    data["label"] = y
    return data


if __name__ == "__main__":
    # 模块自测：查看生成数据
    print(generate_data().head())
