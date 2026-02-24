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
    print(f"序列长度: {len(data)}")
    print(f"字段: {list(data.columns)}")

    print("2. 观测值统计")
    print(data["obs"].value_counts().sort_index())

    print("3. 隐状态统计")
    print(data["state_true"].value_counts().sort_index())


if __name__ == "__main__":
    explore_data(generate_data())
