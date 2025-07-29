"""
探索数据结构与内容
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
    returns:
        class_ratio: 各类别占比
    """
    print("1. 数据集基本信息")
    print(f"样本数量: {len(data)}")
    print(f"特征数量: {len(data.columns) - 1}")
    print(f"特征名称: {list(data.columns[:-1])}")

    print("2. 数据描述统计")
    print(data.describe().round(3))

    print("3. 缺失值检查")
    missing = data.isnull().sum()
    if missing.sum() == 0:
        print("数据无缺失值")
    else:
        print(missing)

    print("类别分布")
    class_count = data["label"].value_counts().sort_index()
    class_ratio = (class_count / len(data)).round(3)
    for label, cnt in class_count.items():
        print(f"类别 {label}: {cnt} ({class_ratio[label] * 100:.1f}%)")

    return class_ratio


if __name__ == "__main__":
    explore_data(generate_data())
