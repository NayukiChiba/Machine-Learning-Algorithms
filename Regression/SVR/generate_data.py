"""
生成 SVR 使用的非线性回归数据
说明：
- 使用 sklearn 的 Friedman1 数据集，天然包含非线性关系
- 目标列统一命名为 price，和其他回归模块保持一致
"""

from pathlib import Path
import sys

# 将项目根目录加入搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from sklearn.datasets import make_friedman1
from utils.decorate import print_func_info


@print_func_info
def generate_data(
    n_samples: int = 400,
    noise: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    生成非线性回归数据

    args:
        n_samples: 样本数量
        noise: 噪声强度（越大越难拟合）
        random_state: 随机种子
    returns:
        DataFrame: 包含特征与目标 price 的数据表
    """
    # Friedman1 数据集：适合测试非线性回归
    X, y = make_friedman1(
        n_samples=n_samples,
        n_features=10,
        noise=noise,
        random_state=random_state,
    )

    # 将特征命名为 x1, x2, ...
    columns = [f"x{i + 1}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)

    # 目标列：统一用 price 表示回归目标
    df["price"] = y
    return df


if __name__ == "__main__":
    df = generate_data()
    print("数据预览：")
    print(df.head())
