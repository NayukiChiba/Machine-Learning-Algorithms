"""
NumPy 数组创建
对应文档: ../../docs/numpy/02-creation.md

使用方式：
    python 02_creation.py
"""

import numpy as np


def demo_from_list():
    """从列表创建数组"""
    print("=" * 50)
    print("1. 从列表创建数组 (np.array)")
    print("=" * 50)
    
    # 一维数组
    arr_1d = np.array([1, 2, 3, 4, 5])
    print(f"一维数组: {arr_1d}")
    print(f"  形状: {arr_1d.shape}, 维度: {arr_1d.ndim}")
    
    # 二维数组
    arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"二维数组:\n{arr_2d}")
    print(f"  形状: {arr_2d.shape}, 维度: {arr_2d.ndim}")
    
    # 指定数据类型
    arr_float = np.array([1, 2, 3], dtype=np.float64)
    print(f"指定float64类型: {arr_float}")


def demo_zeros_ones_eye():
    """创建特殊数组：全零、全一、单位矩阵"""
    print("=" * 50)
    print("2. 特殊数组 (zeros, ones, eye, full)")
    print("=" * 50)
    
    # 全零数组
    zeros = np.zeros((3, 4))
    print(f"3x4全零数组:\n{zeros}")
    
    # 全一数组
    ones = np.ones((2, 3))
    print(f"2x3全一数组:\n{ones}")
    
    # 单位矩阵
    eye = np.eye(3)
    print(f"3x3单位矩阵:\n{eye}")
    
    # 对角线偏移
    eye_k1 = np.eye(3, k=1)
    print(f"上对角线偏移1:\n{eye_k1}")
    
    # 填充指定值
    full = np.full((2, 2), 7)
    print(f"2x2填充7:\n{full}")


def demo_arange_linspace():
    """创建序列数组"""
    print("=" * 50)
    print("3. 序列数组 (arange, linspace)")
    print("=" * 50)
    
    # arange: 类似 Python 的 range
    arr_arange = np.arange(0, 10, 2)
    print(f"arange(0, 10, 2): {arr_arange}")
    
    # 递减序列
    arr_desc = np.arange(10, 0, -1)
    print(f"arange(10, 0, -1): {arr_desc}")
    
    # linspace: 等间距分割
    arr_linspace = np.linspace(0, 1, 5)
    print(f"linspace(0, 1, 5): {arr_linspace}")
    
    # 创建用于绘图的 x 轴
    x = np.linspace(0, 2 * np.pi, 10)
    print(f"linspace(0, 2π, 10):\n{x}")


def demo_random():
    """创建随机数组"""
    print("=" * 50)
    print("4. 随机数组 (random)")
    print("=" * 50)
    
    # 设置随机种子（保证可复现）
    np.random.seed(42)
    
    # rand: [0, 1) 均匀分布
    rand_arr = np.random.rand(2, 3)
    print(f"rand(2, 3):\n{rand_arr}")
    
    # random: 同 rand，参数用 size
    random_arr = np.random.random(size=(2, 3))
    print(f"random(size=(2, 3)):\n{random_arr}")
    
    # randint: 随机整数
    randint_arr = np.random.randint(0, 10, (3, 3))
    print(f"randint(0, 10, (3, 3)):\n{randint_arr}")
    
    # randn: 标准正态分布
    randn_arr = np.random.randn(5)
    print(f"randn(5): {randn_arr}")
    print(f"  均值: {randn_arr.mean():.4f}")
    print(f"  标准差: {randn_arr.std():.4f}")
    
    # normal: 指定均值和标准差的正态分布
    normal_arr = np.random.normal(loc=10, scale=2, size=5)
    print(f"normal(loc=10, scale=2, size=5): {normal_arr}")


def demo_seed():
    """随机种子的使用"""
    print("=" * 50)
    print("5. 随机种子演示")
    print("=" * 50)
    
    # 相同种子产生相同结果
    np.random.seed(42)
    arr1 = np.random.random((2, 2))
    
    np.random.seed(42)
    arr2 = np.random.random((2, 2))
    
    print(f"种子42生成的数组1:\n{arr1}")
    print(f"种子42生成的数组2:\n{arr2}")
    print(f"两个数组是否相同: {np.array_equal(arr1, arr2)}")


def demo_all():
    """运行所有演示"""
    demo_from_list()
    print()
    demo_zeros_ones_eye()
    print()
    demo_arange_linspace()
    print()
    demo_random()
    print()
    demo_seed()


if __name__ == "__main__":
    demo_all()
