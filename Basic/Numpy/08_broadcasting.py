"""
NumPy 广播机制
对应文档: ../../docs/numpy/08-broadcasting.md

使用方式：
    python 08_broadcasting.py
"""

import numpy as np


def demo_scalar_broadcast():
    """标量与数组的广播"""
    print("=" * 50)
    print("1. 标量与数组的广播")
    print("=" * 50)
    
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"数组 (3x3):\n{arr}")
    print()
    
    # 标量加法
    print(f"arr + 10:\n{arr + 10}")
    
    # 标量乘法
    print(f"arr * 2:\n{arr * 2}")


def demo_1d_2d_broadcast():
    """一维与二维数组的广播"""
    print("=" * 50)
    print("2. 一维与二维数组的广播")
    print("=" * 50)
    
    arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    arr_1d = np.array([10, 20, 30])
    
    print(f"二维数组 (3x3):\n{arr_2d}")
    print(f"一维数组 (3,): {arr_1d}")
    print()
    
    # 一维数组按行广播
    result = arr_2d + arr_1d
    print(f"arr_2d + arr_1d:\n{result}")


def demo_column_broadcast():
    """列向量的广播"""
    print("=" * 50)
    print("3. 列向量的广播")
    print("=" * 50)
    
    arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    col = np.array([[100], [200], [300]])
    
    print(f"二维数组 (3x3):\n{arr_2d}")
    print(f"列向量 (3x1):\n{col}")
    print()
    
    # 列向量按列广播
    result = arr_2d + col
    print(f"arr_2d + col:\n{result}")


def demo_broadcast_rules():
    """广播规则说明"""
    print("=" * 50)
    print("4. 广播规则")
    print("=" * 50)
    
    print("""
广播规则：
1. 从后向前比较每个维度的大小
2. 维度大小相同 或 其中一个为1 时可以广播
3. 缺失的维度视为1

示例：
  (3, 4) + (4,)    → (3, 4)  ✓
  (3, 4) + (3, 1)  → (3, 4)  ✓
  (3, 1) + (1, 4)  → (3, 4)  ✓
  (3, 4) + (3,)    → 错误！ ✗
""")
    
    # 演示成功的广播
    A = np.ones((3, 4))
    B = np.array([1, 2, 3, 4])
    print(f"A (3, 4) + B (4,) = {(A + B).shape}")
    
    A = np.ones((3, 4))
    B = np.array([[1], [2], [3]])
    print(f"A (3, 4) + B (3, 1) = {(A + B).shape}")
    
    A = np.ones((3, 1))
    B = np.ones((1, 4))
    print(f"A (3, 1) + B (1, 4) = {(A + B).shape}")


def demo_outer_product():
    """利用广播实现外积"""
    print("=" * 50)
    print("5. 利用广播实现外积")
    print("=" * 50)
    
    a = np.array([1, 2, 3])
    b = np.array([10, 20, 30, 40])
    
    print(f"a = {a}")
    print(f"b = {b}")
    print()
    
    # 使用广播实现外积
    outer = a[:, np.newaxis] * b
    print(f"外积 a[:, np.newaxis] * b:\n{outer}")
    print()
    
    # 使用 np.outer
    outer2 = np.outer(a, b)
    print(f"np.outer(a, b):\n{outer2}")


def demo_practical_example():
    """广播的实际应用"""
    print("=" * 50)
    print("6. 广播的实际应用")
    print("=" * 50)
    
    # 场景：标准化数据（减去均值，除以标准差）
    np.random.seed(42)
    data = np.random.randint(0, 100, size=(5, 3))
    print(f"原始数据 (5个样本, 3个特征):\n{data}")
    print()
    
    # 计算每列的均值和标准差
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    print(f"每列均值: {mean.round(2)}")
    print(f"每列标准差: {std.round(2)}")
    print()
    
    # 标准化（利用广播）
    normalized = (data - mean) / std
    print(f"标准化后:\n{normalized.round(2)}")
    print()
    
    # 验证
    print(f"标准化后均值: {normalized.mean(axis=0).round(10)}")
    print(f"标准化后标准差: {normalized.std(axis=0).round(2)}")


def demo_all():
    """运行所有演示"""
    demo_scalar_broadcast()
    print()
    demo_1d_2d_broadcast()
    print()
    demo_column_broadcast()
    print()
    demo_broadcast_rules()
    print()
    demo_outer_product()
    print()
    demo_practical_example()


if __name__ == "__main__":
    demo_all()
