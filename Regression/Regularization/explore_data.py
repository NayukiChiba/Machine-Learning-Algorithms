"""
数据探索部分
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pandas import DataFrame
from utils.decorate import print_func_info


@print_func_info
def explore_data(data: DataFrame):
    """
    数据探索分析

    args:
        data(DataFrame): 数据集
    returns:
        correlation: 与目标变量的相关性排序
    """
    print("1. 数据集基本信息")
    print(f"样本数量: {len(data)}")
    print(f"特征数量: {len(data.columns) - 1}")
    print(f"特征名称: {list(data.columns[:-1])}")

    print("2. 统计描述")
    print(data.describe().round(3))

    print("3. 缺失值检查")
    missing = data.isnull().sum()
    if missing.sum() == 0:
        print("数据无缺失值")
    else:
        print(missing)

    print("4. 与目标变量 price 的相关性")
    correlation = data.corr()["price"].drop("price").sort_values(ascending=False)
    print(correlation.round(3))

    return correlation


if __name__ == "__main__":
    from generate_data import generate_data

    df = generate_data()
    explore_data(df)


if __name__ == "__main__":
    explore_data(generate_data())
