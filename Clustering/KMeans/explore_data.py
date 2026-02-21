"""
数据探索模块
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pandas import DataFrame
from utils.decorate import print_func_info
from generate_data import generate_data


@print_func_info
def explore_data(data: DataFrame):
    """
    数据探索分析

    args:
        data(DataFrame): 输入数据
    returns:
        None
    """
    print("1. 数据集基础信息")
    print(f"样本数量: {len(data)}")
    print(f"特征数量: {len(data.columns)}")
    print(f"特征名称: {list(data.columns)}")

    print("2. 数据描述统计")
    print(data.describe().round(3))

    print("3. 缺失值检查")
    missing = data.isnull().sum()
    if missing.sum() == 0:
        print("无缺失值")
    else:
        print(missing)


if __name__ == "__main__":
    explore_data(generate_data())
