"""
数据清洗与处理
对应文档: ../../docs/pandas/04-cleaning.md

使用方式：
    python 04_cleaning.py
    或
    from Basic.Pandas.04_cleaning import *
    demo_all()
"""

import pandas as pd
import numpy as np


def demo_missing_values():
    """演示缺失值检测和处理"""
    print("=" * 50)
    print("1. 缺失值处理")
    print("=" * 50)
    
    # 创建含缺失值的数据
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': ['x', 'y', None, 'z', 'w']
    })
    print("原始数据 (含缺失值):")
    print(df)
    print()
    
    # 检测缺失值
    print("isnull() 检测:")
    print(df.isnull())
    print()
    
    print("缺失值统计:")
    print(df.isnull().sum())
    print()
    
    # 删除缺失值
    print("dropna() - 删除含缺失值的行:")
    print(df.dropna())
    print()
    
    # 填充缺失值
    print("fillna(0) - 用0填充:")
    print(df.fillna(0))
    print()
    
    # 前向填充
    print("fillna(method='ffill') - 前向填充:")
    print(df.fillna(method='ffill'))


def demo_duplicates():
    """演示重复值处理"""
    print("=" * 50)
    print("2. 重复值处理")
    print("=" * 50)
    
    df = pd.DataFrame({
        'A': [1, 1, 2, 2, 3],
        'B': ['a', 'a', 'b', 'c', 'c']
    })
    print("原始数据:")
    print(df)
    print()
    
    # 检测重复值
    print("duplicated() 检测:")
    print(df.duplicated())
    print()
    
    # 删除重复值
    print("drop_duplicates():")
    print(df.drop_duplicates())
    print()
    
    # 基于特定列去重
    print("drop_duplicates(subset=['A']):")
    print(df.drop_duplicates(subset=['A']))


def demo_type_conversion():
    """演示数据类型转换"""
    print("=" * 50)
    print("3. 数据类型转换")
    print("=" * 50)
    
    df = pd.DataFrame({
        'A': ['1', '2', '3', '4', '5'],
        'B': [1.1, 2.2, 3.3, 4.4, 5.5],
        'C': ['2023-01-01', '2023-01-02', '2023-01-03', 
              '2023-01-04', '2023-01-05']
    })
    print("原始数据:")
    print(df)
    print(f"数据类型:\n{df.dtypes}")
    print()
    
    # astype 转换
    df['A'] = df['A'].astype(int)
    df['B'] = df['B'].astype(int)
    print("转换后:")
    print(df)
    print(f"数据类型:\n{df.dtypes}")
    print()
    
    # 日期转换
    df['C'] = pd.to_datetime(df['C'])
    print("日期转换后:")
    print(df)
    print(f"C列类型: {df['C'].dtype}")


def demo_string_ops():
    """演示字符串操作"""
    print("=" * 50)
    print("4. 字符串操作")
    print("=" * 50)
    
    df = pd.DataFrame({
        'Name': ['  Alice  ', 'BOB', 'charlie', 'David Lee'],
        'Email': ['alice@example.com', 'bob@test.org', 
                  'charlie@example.com', 'david@test.org']
    })
    print("原始数据:")
    print(df)
    print()
    
    # 常用字符串方法
    print("str.strip() - 去除空格:")
    print(df['Name'].str.strip())
    print()
    
    print("str.lower() - 转小写:")
    print(df['Name'].str.lower())
    print()
    
    print("str.upper() - 转大写:")
    print(df['Name'].str.upper())
    print()
    
    print("str.contains('example') - 包含匹配:")
    print(df['Email'].str.contains('example'))
    print()
    
    print("str.split('@') - 分割:")
    print(df['Email'].str.split('@'))


def demo_replace():
    """演示值替换"""
    print("=" * 50)
    print("5. 值替换")
    print("=" * 50)
    
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': ['yes', 'no', 'yes', 'no', 'maybe']
    })
    print("原始数据:")
    print(df)
    print()
    
    # 单值替换
    print("replace(1, 100):")
    print(df['A'].replace(1, 100))
    print()
    
    # 字典替换
    print("replace({'yes': 1, 'no': 0, 'maybe': -1}):")
    print(df['B'].replace({'yes': 1, 'no': 0, 'maybe': -1}))


def demo_all():
    """运行所有演示"""
    demo_missing_values()
    print()
    demo_duplicates()
    print()
    demo_type_conversion()
    print()
    demo_string_ops()
    print()
    demo_replace()


if __name__ == "__main__":
    demo_all()
