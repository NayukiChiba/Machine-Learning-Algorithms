"""
时间序列处理
对应文档: ../../docs/pandas/07-timeseries.md

使用方式：
    python 07_timeseries.py
    或
    from Basic.Pandas.07_timeseries import *
    demo_all()
"""

import pandas as pd
import numpy as np


def demo_datetime_create():
    """演示时间序列创建"""
    print("=" * 50)
    print("1. 时间序列创建")
    print("=" * 50)
    
    # 解析日期字符串
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    print("解析日期字符串:")
    print(dates)
    print(f"类型: {type(dates)}")
    print()
    
    # date_range 创建日期范围
    print("date_range 创建连续日期:")
    dr = pd.date_range('2023-01-01', periods=5, freq='D')
    print(dr)
    print()
    
    # 不同频率
    print("不同频率示例:")
    print("  freq='H' (小时):", pd.date_range('2023-01-01', periods=3, freq='H').tolist())
    print("  freq='W' (周):", pd.date_range('2023-01-01', periods=3, freq='W').tolist())
    print("  freq='M' (月末):", pd.date_range('2023-01-01', periods=3, freq='ME').tolist())
    print("  freq='B' (工作日):", pd.date_range('2023-01-01', periods=5, freq='B').tolist())


def demo_time_index():
    """演示时间序列索引"""
    print("=" * 50)
    print("2. 时间序列索引")
    print("=" * 50)
    
    # 创建时间序列数据
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    ts = pd.Series(np.random.randn(10), index=dates)
    print("时间序列数据:")
    print(ts)
    print()
    
    # 日期属性访问
    print("日期属性访问:")
    print(f"  year: {ts.index.year.tolist()}")
    print(f"  month: {ts.index.month.tolist()}")
    print(f"  day: {ts.index.day.tolist()}")
    print(f"  dayofweek: {ts.index.dayofweek.tolist()}")


def demo_time_slice():
    """演示时间序列切片"""
    print("=" * 50)
    print("3. 时间序列切片")
    print("=" * 50)
    
    # 创建较长的时间序列
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    ts = pd.Series(np.random.randn(100), index=dates)
    
    print(f"时间序列范围: {ts.index[0]} 到 {ts.index[-1]}")
    print()
    
    # 部分字符串索引
    print("选择 2023年1月的数据 ts['2023-01']:")
    print(ts['2023-01'].head())
    print()
    
    # 范围选择
    print("选择日期范围 ts['2023-01-15':'2023-01-20']:")
    print(ts['2023-01-15':'2023-01-20'])


def demo_resample():
    """演示时间重采样"""
    print("=" * 50)
    print("4. 时间重采样")
    print("=" * 50)
    
    # 创建每日数据
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    df = pd.DataFrame({
        'value': np.random.randint(10, 100, 30),
        'sales': np.random.randint(100, 1000, 30)
    }, index=dates)
    
    print("每日数据 (前5行):")
    print(df.head())
    print()
    
    # 降采样 - 按周聚合
    print("按周聚合 resample('W').sum():")
    print(df.resample('W').sum())
    print()
    
    # 多种聚合方式
    print("按周聚合 (多种统计):")
    print(df['value'].resample('W').agg(['sum', 'mean', 'max']))


def demo_rolling():
    """演示滚动窗口操作"""
    print("=" * 50)
    print("5. 滚动窗口操作")
    print("=" * 50)
    
    # 创建时间序列
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    ts = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=dates)
    
    print("原始数据:")
    print(ts)
    print()
    
    # 移动平均
    print("3日移动平均 rolling(3).mean():")
    print(ts.rolling(3).mean())
    print()
    
    # 移动求和
    print("3日移动求和 rolling(3).sum():")
    print(ts.rolling(3).sum())
    print()
    
    # 指数加权移动平均
    print("指数加权移动平均 ewm(span=3).mean():")
    print(ts.ewm(span=3).mean())


def demo_shift():
    """演示时间偏移操作"""
    print("=" * 50)
    print("6. 时间偏移操作")
    print("=" * 50)
    
    dates = pd.date_range('2023-01-01', periods=5, freq='D')
    ts = pd.Series([10, 20, 30, 40, 50], index=dates)
    
    print("原始数据:")
    print(ts)
    print()
    
    # 向后偏移
    print("shift(1) - 向后偏移1期:")
    print(ts.shift(1))
    print()
    
    # 向前偏移
    print("shift(-1) - 向前偏移1期:")
    print(ts.shift(-1))
    print()
    
    # 计算变化率
    print("pct_change() - 百分比变化:")
    print(ts.pct_change())


def demo_all():
    """运行所有演示"""
    demo_datetime_create()
    print()
    demo_time_index()
    print()
    demo_time_slice()
    print()
    demo_resample()
    print()
    demo_rolling()
    print()
    demo_shift()


if __name__ == "__main__":
    demo_all()
