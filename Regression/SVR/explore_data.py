"""
数据探索模块
"""

from pathlib import Path
import sys

# 加入项目根目录，便于导入公共模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pandas import DataFrame
from utils.decorate import print_func_info


@print_func_info
def explore_data(data: DataFrame):
    """
    数据探索分析

    args:
        data(DataFrame): 输入数据集
    returns:
        correlation: 与目标变量 price 的相关性排序
    """
    # 1. 基本信息
    print("1. 数据集基础信息")
    print(f"样本数量: {len(data)}")
    print(f"特征数量: {len(data.columns) - 1}")
    print(f"特征名称: {list(data.columns[:-1])}")

    # 2. 统计描述
    print("2. 统计描述")
    print(data.describe().round(3))

    # 3. 缺失值检查
    print("3. 缺失值检查")
    missing = data.isnull().sum()
    if missing.sum() == 0:
        print("无缺失值")
    else:
        print(missing)

    # 4. 与目标变量 price 的相关性
    print("4. 与目标变量 price 的相关性")
    correlation = (
        data.corr(numeric_only=True)["price"].drop("price").sort_values(ascending=False)
    )
    print(correlation.round(3))

    return correlation


if __name__ == "__main__":
    from generate_data import generate_data

    df = generate_data()
    explore_data(df)
