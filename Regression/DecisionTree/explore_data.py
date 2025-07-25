"""
探索数据的结构和内容
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pandas import DataFrame
from utils.decorate import print_func_info
from generate_data import generate_data


@print_func_info
def explore_data(data: DataFrame):
    """
    数据探索分析

    args:
        data(DataFrame): 数据
    """
    print("1. 数据集基本信息")
    print(f"样本数量: {len(data)}")
    print(f"特征数量: {len(data.columns) - 1}")
    print(f"特征名称: {list(data.columns[:-1])}")

    # 描述性统计
    print("2. 数据描述统计")
    print(data.describe().round(2))

    # 缺失值处理
    print("3. 缺失值检查")
    missing = data.isnull().sum()
    if missing.sum() == 0:
        print("数据无缺失值")
    else:
        print(missing)

    # 相关性
    print("4. 与目标的相关性")
    correlation = data.corr()["price"].drop("price").sort_values(ascending=False)
    print(correlation.round(3))

    return correlation


if __name__ == "__main__":
    explore_data(generate_data())
