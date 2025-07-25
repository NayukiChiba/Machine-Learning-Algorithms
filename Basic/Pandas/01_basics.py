"""
Pandas 基础入门
对应文档: ../../docs/pandas/01-basics.md

使用方式：
    python 01_basics.py
    或
    from Basic.Pandas.01_basics import *
    demo_all()
"""

import pandas as pd
import numpy as np


def demo_series():
    """演示 Pandas Series 的创建和基本操作"""
    print("=" * 50)
    print("1. Series 数据结构")
    print("=" * 50)

    # 从列表创建
    s1 = pd.Series([1, 2, 3, 4, 5])
    print("从列表创建 Series:")
    print(s1)
    print(f"索引: {s1.index.tolist()}")
    print(f"值: {s1.values}")
    print()

    # 指定索引
    s2 = pd.Series([10, 20, 30], index=["a", "b", "c"])
    print("带自定义索引的 Series:")
    print(s2)
    print(f"访问 s2['b']: {s2['b']}")
    print()

    # 从字典创建
    data = {"apple": 100, "banana": 200, "orange": 150}
    s3 = pd.Series(data)
    print("从字典创建 Series:")
    print(s3)


def demo_dataframe():
    """演示 Pandas DataFrame 的创建和基本操作"""
    print("=" * 50)
    print("2. DataFrame 数据结构")
    print("=" * 50)

    # 从字典创建
    data = {
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35],
        "City": ["Beijing", "Shanghai", "Guangzhou"],
    }
    df = pd.DataFrame(data)
    print("从字典创建 DataFrame:")
    print(df)
    print()

    # 查看基本信息
    print(f"形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print(f"索引: {df.index.tolist()}")
    print(f"数据类型:\n{df.dtypes}")


def demo_basic_view():
    """演示数据查看的基本方法"""
    print("=" * 50)
    print("3. 基本数据查看方法")
    print("=" * 50)

    # 创建示例数据
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "A": np.random.randn(10),
            "B": np.random.randint(0, 100, 10),
            "C": [
                "cat",
                "dog",
                "bird",
                "cat",
                "dog",
                "bird",
                "cat",
                "dog",
                "bird",
                "cat",
            ],
        }
    )

    print("原始数据:")
    print(df)
    print()

    print("head(3) - 前3行:")
    print(df.head(3))
    print()

    print("tail(3) - 后3行:")
    print(df.tail(3))
    print()

    print("info() - 数据信息:")
    df.info()
    print()

    print("describe() - 统计描述:")
    print(df.describe())


def demo_attributes():
    """演示 DataFrame 的常用属性"""
    print("=" * 50)
    print("4. DataFrame 属性")
    print("=" * 50)

    df = pd.DataFrame({"A": [1, 2, 3], "B": [4.0, 5.0, 6.0], "C": ["x", "y", "z"]})

    print(f"shape: {df.shape}")
    print(f"ndim: {df.ndim}")
    print(f"size: {df.size}")
    print(f"columns: {df.columns.tolist()}")
    print(f"index: {df.index.tolist()}")
    print(f"values:\n{df.values}")
    print(f"dtypes:\n{df.dtypes}")


def demo_all():
    """运行所有演示"""
    demo_series()
    print()
    demo_dataframe()
    print()
    demo_basic_view()
    print()
    demo_attributes()


if __name__ == "__main__":
    demo_all()
