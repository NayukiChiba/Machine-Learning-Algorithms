"""
数据分组与聚合
对应文档: ../../docs/pandas/05-groupby.md

使用方式：
    python 05_groupby.py
    或
    from Basic.Pandas.05_groupby import *
    demo_all()
"""

import pandas as pd
import numpy as np


def create_sample_df():
    """创建示例 DataFrame"""
    np.random.seed(42)
    return pd.DataFrame({
        'Department': ['Sales', 'Sales', 'IT', 'IT', 'HR', 'HR'],
        'Employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
        'Salary': [8000, 9000, 12000, 11000, 7000, 7500],
        'Bonus': [1000, 1200, 1500, 1400, 800, 900],
        'Years': [3, 5, 4, 6, 2, 3]
    })


def demo_groupby():
    """演示 groupby 基本操作"""
    print("=" * 50)
    print("1. GroupBy 基本操作")
    print("=" * 50)
    
    df = create_sample_df()
    print("原始数据:")
    print(df)
    print()
    
    # 分组对象
    grouped = df.groupby('Department')
    print(f"分组对象: {type(grouped)}")
    print(f"分组数量: {grouped.ngroups}")
    print()
    
    # 遍历分组
    print("遍历分组:")
    for name, group in grouped:
        print(f"\n--- {name} ---")
        print(group)


def demo_agg():
    """演示聚合函数"""
    print("=" * 50)
    print("2. 聚合函数")
    print("=" * 50)
    
    df = create_sample_df()
    grouped = df.groupby('Department')
    
    # 单个聚合
    print("sum() - 求和:")
    print(grouped['Salary'].sum())
    print()
    
    print("mean() - 平均值:")
    print(grouped['Salary'].mean())
    print()
    
    print("count() - 计数:")
    print(grouped['Employee'].count())
    print()
    
    # 多个聚合
    print("agg(['sum', 'mean', 'max']):")
    print(grouped['Salary'].agg(['sum', 'mean', 'max']))


def demo_multi_column_agg():
    """演示多列分组聚合"""
    print("=" * 50)
    print("3. 多列聚合")
    print("=" * 50)
    
    df = create_sample_df()
    grouped = df.groupby('Department')
    
    # 对不同列使用不同聚合
    print("不同列不同聚合函数:")
    result = grouped.agg({
        'Salary': ['mean', 'sum'],
        'Bonus': 'sum',
        'Years': 'mean'
    })
    print(result)
    print()
    
    # 自定义聚合函数
    print("自定义聚合函数:")
    result = grouped['Salary'].agg(
        total=('sum'),
        average=('mean'),
        range=lambda x: x.max() - x.min()
    )
    print(result)


def demo_transform():
    """演示 transform 方法"""
    print("=" * 50)
    print("4. Transform 方法")
    print("=" * 50)
    
    df = create_sample_df()
    print("原始数据:")
    print(df)
    print()
    
    # transform 返回与原始数据相同长度的结果
    df['Dept_Mean_Salary'] = df.groupby('Department')['Salary'].transform('mean')
    print("添加部门平均工资列:")
    print(df)
    print()
    
    # 标准化 (z-score)
    df['Salary_Zscore'] = df.groupby('Department')['Salary'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    print("添加标准化分数:")
    print(df[['Department', 'Employee', 'Salary', 'Salary_Zscore']])


def demo_apply():
    """演示 apply 方法"""
    print("=" * 50)
    print("5. Apply 方法")
    print("=" * 50)
    
    df = create_sample_df()
    grouped = df.groupby('Department')
    
    # apply 可以返回任意形状的结果
    def top_employee(group):
        return group.nlargest(1, 'Salary')
    
    print("每个部门薪资最高的员工:")
    print(grouped.apply(top_employee, include_groups=False))
    print()
    
    # 自定义汇总函数
    def summary(group):
        return pd.Series({
            'count': len(group),
            'total_salary': group['Salary'].sum(),
            'avg_years': group['Years'].mean()
        })
    
    print("自定义汇总:")
    print(grouped.apply(summary, include_groups=False))


def demo_all():
    """运行所有演示"""
    demo_groupby()
    print()
    demo_agg()
    print()
    demo_multi_column_agg()
    print()
    demo_transform()
    print()
    demo_apply()


if __name__ == "__main__":
    demo_all()
