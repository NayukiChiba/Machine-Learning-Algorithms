"""
高级操作与性能优化
对应文档: ../../docs/pandas/09-advanced.md
"""

import pandas as pd
import numpy as np


def demo_pivot_table():
    """演示透视表"""
    print("=" * 50)
    print("1. 透视表")
    print("=" * 50)
    
    df = pd.DataFrame({
        'Date': ['2023-01', '2023-01', '2023-02', '2023-02'],
        'Region': ['North', 'South', 'North', 'South'],
        'Sales': [100, 150, 120, 180],
        'Quantity': [10, 15, 12, 18]
    })
    print("原始数据:")
    print(df)
    print()
    
    pivot = pd.pivot_table(df, values='Sales', index='Date', 
                           columns='Region', aggfunc='sum')
    print("透视表:")
    print(pivot)


def demo_crosstab():
    """演示交叉表"""
    print("=" * 50)
    print("2. 交叉表")
    print("=" * 50)
    
    df = pd.DataFrame({
        'Gender': ['M', 'F', 'M', 'F', 'M', 'F'],
        'City': ['Beijing', 'Shanghai', 'Beijing', 'Beijing', 'Shanghai', 'Shanghai']
    })
    print("原始数据:")
    print(df)
    print()
    
    ct = pd.crosstab(df['Gender'], df['City'])
    print("交叉表:")
    print(ct)


def demo_multi_index():
    """演示多级索引"""
    print("=" * 50)
    print("3. 多级索引")
    print("=" * 50)
    
    arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
    index = pd.MultiIndex.from_arrays(arrays, names=['Letter', 'Number'])
    df = pd.DataFrame({'Value': [10, 20, 30, 40]}, index=index)
    
    print("多级索引 DataFrame:")
    print(df)
    print()
    
    print("选择 df.loc['A']:")
    print(df.loc['A'])
    print()
    
    print("选择 df.loc[('A', 1)]:")
    print(df.loc[('A', 1)])


def demo_vectorization():
    """演示向量化操作"""
    print("=" * 50)
    print("4. 向量化操作")
    print("=" * 50)
    
    import time
    n = 100000
    df = pd.DataFrame({'A': np.random.randn(n), 'B': np.random.randn(n)})
    
    # 循环方式
    start = time.time()
    result1 = []
    for i in range(len(df)):
        result1.append(df['A'].iloc[i] + df['B'].iloc[i])
    loop_time = time.time() - start
    
    # 向量化方式
    start = time.time()
    result2 = df['A'] + df['B']
    vec_time = time.time() - start
    
    print(f"循环耗时: {loop_time:.4f}秒")
    print(f"向量化耗时: {vec_time:.4f}秒")
    print(f"向量化快了约 {loop_time/vec_time:.1f} 倍")


def demo_memory_optimization():
    """演示内存优化"""
    print("=" * 50)
    print("5. 内存优化")
    print("=" * 50)
    
    df = pd.DataFrame({
        'int_col': np.random.randint(0, 100, 10000),
        'float_col': np.random.randn(10000),
        'str_col': np.random.choice(['A', 'B', 'C'], 10000)
    })
    
    print("优化前内存使用:")
    print(df.memory_usage(deep=True))
    print()
    
    # 优化整数列
    df['int_col'] = df['int_col'].astype('int8')
    # 优化浮点列
    df['float_col'] = df['float_col'].astype('float32')
    # 使用分类类型
    df['str_col'] = df['str_col'].astype('category')
    
    print("优化后内存使用:")
    print(df.memory_usage(deep=True))


def demo_all():
    """运行所有演示"""
    demo_pivot_table()
    print()
    demo_crosstab()
    print()
    demo_multi_index()
    print()
    demo_vectorization()
    print()
    demo_memory_optimization()


if __name__ == "__main__":
    demo_all()
