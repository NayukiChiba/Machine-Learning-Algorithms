"""
生成适合决策树学习的非线性数据
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pandas import DataFrame
from sklearn.datasets import fetch_california_housing
from utils.decorate import print_func_info


@print_func_info
def generate_data() -> DataFrame:
    """
    加载房价数据集，并返回 DataFrame

    returns:
        DataFrame: 包含特征与目标变量(价格)的数据
    """
    # 加载数据集
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    # 目标变量在原始数据中叫 MedHouseVal（单位：10万美金）
    # 这里统一改成中文列名“价格”
    df = df.rename(columns={"MedHouseVal": "price"})

    return df


if __name__ == "__main__":
    print(generate_data().head())
