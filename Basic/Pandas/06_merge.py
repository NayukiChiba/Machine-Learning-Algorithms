"""
数据合并与连接
对应文档: ../../docs/pandas/06-merge.md

使用方式：
    python 06_merge.py
    或
    from Basic.Pandas.06_merge import *
    demo_all()
"""

import pandas as pd
import numpy as np


def demo_concat():
    """演示 concat 合并"""
    print("=" * 50)
    print("1. Concat 合并")
    print("=" * 50)

    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})

    print("DataFrame 1:")
    print(df1)
    print("\nDataFrame 2:")
    print(df2)
    print()

    # 纵向合并 (axis=0)
    print("纵向合并 concat([df1, df2], axis=0):")
    print(pd.concat([df1, df2], axis=0, ignore_index=True))
    print()

    # 横向合并 (axis=1)
    print("横向合并 concat([df1, df2], axis=1):")
    print(pd.concat([df1, df2], axis=1))
    print()

    # 不同列的合并
    df3 = pd.DataFrame({"A": [1, 2], "C": [5, 6]})
    print("不同列合并 (外连接):")
    print(pd.concat([df1, df3], ignore_index=True))


def demo_merge():
    """演示 merge 数据库风格连接"""
    print("=" * 50)
    print("2. Merge 连接")
    print("=" * 50)

    # 员工表
    employees = pd.DataFrame(
        {
            "emp_id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Charlie", "David"],
            "dept_id": [10, 20, 10, 30],
        }
    )

    # 部门表
    departments = pd.DataFrame(
        {"dept_id": [10, 20, 40], "dept_name": ["Sales", "IT", "HR"]}
    )

    print("员工表:")
    print(employees)
    print("\n部门表:")
    print(departments)
    print()

    # 内连接 (默认)
    print("内连接 merge(employees, departments, on='dept_id'):")
    print(pd.merge(employees, departments, on="dept_id"))
    print()

    # 左连接
    print("左连接 merge(..., how='left'):")
    print(pd.merge(employees, departments, on="dept_id", how="left"))
    print()

    # 右连接
    print("右连接 merge(..., how='right'):")
    print(pd.merge(employees, departments, on="dept_id", how="right"))
    print()

    # 外连接
    print("外连接 merge(..., how='outer'):")
    print(pd.merge(employees, departments, on="dept_id", how="outer"))


def demo_merge_on_different_keys():
    """演示不同列名的合并"""
    print("=" * 50)
    print("3. 不同列名合并")
    print("=" * 50)

    df1 = pd.DataFrame({"id": [1, 2, 3], "value1": ["a", "b", "c"]})

    df2 = pd.DataFrame({"key": [1, 2, 4], "value2": ["x", "y", "z"]})

    print("DataFrame 1 (id列):")
    print(df1)
    print("\nDataFrame 2 (key列):")
    print(df2)
    print()

    # 使用 left_on 和 right_on
    print("使用 left_on='id', right_on='key':")
    result = pd.merge(df1, df2, left_on="id", right_on="key", how="outer")
    print(result)


def demo_join():
    """演示 join 操作"""
    print("=" * 50)
    print("4. Join 操作")
    print("=" * 50)

    df1 = pd.DataFrame({"A": [1, 2, 3]}, index=["a", "b", "c"])

    df2 = pd.DataFrame({"B": [4, 5, 6]}, index=["a", "b", "d"])

    print("DataFrame 1:")
    print(df1)
    print("\nDataFrame 2:")
    print(df2)
    print()

    # 默认左连接
    print("df1.join(df2) - 左连接:")
    print(df1.join(df2))
    print()

    # 外连接
    print("df1.join(df2, how='outer') - 外连接:")
    print(df1.join(df2, how="outer"))


def demo_merge_indicator():
    """演示合并指示器"""
    print("=" * 50)
    print("5. 合并指示器")
    print("=" * 50)

    df1 = pd.DataFrame({"key": [1, 2, 3], "value": ["a", "b", "c"]})
    df2 = pd.DataFrame({"key": [2, 3, 4], "value": ["x", "y", "z"]})

    print("DataFrame 1:")
    print(df1)
    print("\nDataFrame 2:")
    print(df2)
    print()

    # 使用 indicator 参数
    result = pd.merge(
        df1, df2, on="key", how="outer", suffixes=("_left", "_right"), indicator=True
    )
    print("带指示器的外连接:")
    print(result)
    print()

    print("_merge 列值含义:")
    print("  - left_only: 只存在于左表")
    print("  - right_only: 只存在于右表")
    print("  - both: 两表都存在")


def demo_all():
    """运行所有演示"""
    demo_concat()
    print()
    demo_merge()
    print()
    demo_merge_on_different_keys()
    print()
    demo_join()
    print()
    demo_merge_indicator()


if __name__ == "__main__":
    demo_all()
