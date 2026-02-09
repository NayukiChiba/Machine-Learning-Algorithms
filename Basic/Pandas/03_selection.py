"""
数据选择与过滤
对应文档: ../../docs/pandas/03-selection.md

使用方式：
    python 03_selection.py
    或
    from Basic.Pandas.03_selection import *
    demo_all()
"""

import pandas as pd
import numpy as np


def create_sample_df():
    """创建示例 DataFrame"""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "Age": [25, 30, 35, 28, 32],
            "City": ["Beijing", "Shanghai", "Beijing", "Guangzhou", "Shanghai"],
            "Salary": [8000, 12000, 15000, 9000, 11000],
            "Score": np.random.randint(60, 100, 5),
        }
    )


def demo_column_select():
    """演示列选择"""
    print("=" * 50)
    print("1. 列选择")
    print("=" * 50)

    df = create_sample_df()
    print("原始数据:")
    print(df)
    print()

    # 单列选择
    print("选择单列 df['Name']:")
    print(df["Name"])
    print(f"类型: {type(df['Name'])}")
    print()

    # 多列选择
    print("选择多列 df[['Name', 'Age']]:")
    print(df[["Name", "Age"]])


def demo_row_select():
    """演示行选择"""
    print("=" * 50)
    print("2. 行选择")
    print("=" * 50)

    df = create_sample_df()

    # 切片选择
    print("切片选择 df[1:3]:")
    print(df[1:3])
    print()

    # head 和 tail
    print("df.head(2):")
    print(df.head(2))


def demo_loc_iloc():
    """演示 loc 和 iloc 索引器"""
    print("=" * 50)
    print("3. loc 和 iloc 索引器")
    print("=" * 50)

    df = create_sample_df()
    df.index = ["a", "b", "c", "d", "e"]  # 设置自定义索引
    print("带自定义索引的数据:")
    print(df)
    print()

    # loc - 标签索引
    print("loc['b', 'Name'] (标签索引):")
    print(df.loc["b", "Name"])
    print()

    print("loc['a':'c', ['Name', 'Age']]:")
    print(df.loc["a":"c", ["Name", "Age"]])
    print()

    # iloc - 位置索引
    print("iloc[1, 0] (位置索引):")
    print(df.iloc[1, 0])
    print()

    print("iloc[0:3, 0:2]:")
    print(df.iloc[0:3, 0:2])


def demo_filter():
    """演示条件过滤"""
    print("=" * 50)
    print("4. 条件过滤")
    print("=" * 50)

    df = create_sample_df()
    print("原始数据:")
    print(df)
    print()

    # 单条件过滤
    print("Age > 28 的数据:")
    print(df[df["Age"] > 28])
    print()

    # 多条件过滤 (AND)
    print("Age > 25 且 Salary > 10000:")
    print(df[(df["Age"] > 25) & (df["Salary"] > 10000)])
    print()

    # 多条件过滤 (OR)
    print("City 是 Beijing 或 Shanghai:")
    print(df[(df["City"] == "Beijing") | (df["City"] == "Shanghai")])
    print()

    # isin 过滤
    print("使用 isin(['Beijing', 'Shanghai']):")
    print(df[df["City"].isin(["Beijing", "Shanghai"])])


def demo_query():
    """演示 query 方法"""
    print("=" * 50)
    print("5. query 方法")
    print("=" * 50)

    df = create_sample_df()

    print("使用 query 方法:")
    print("df.query('Age > 28 and Salary > 10000'):")
    print(df.query("Age > 28 and Salary > 10000"))
    print()

    # 使用变量
    min_age = 30
    print(f"使用外部变量 min_age={min_age}:")
    print("df.query('Age >= @min_age'):")
    print(df.query("Age >= @min_age"))


def demo_all():
    """运行所有演示"""
    demo_column_select()
    print()
    demo_row_select()
    print()
    demo_loc_iloc()
    print()
    demo_filter()
    print()
    demo_query()


if __name__ == "__main__":
    demo_all()
