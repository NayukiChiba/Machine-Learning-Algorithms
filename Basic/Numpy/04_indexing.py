"""
NumPy 数组索引和切片
对应文档: ../../docs/numpy/04-indexing.md

使用方式：
    python 04_indexing.py
"""

import numpy as np


def demo_basic_indexing():
    """基本索引操作"""
    print("=" * 50)
    print("1. 基本索引")
    print("=" * 50)

    # 一维数组索引
    arr_1d = np.array([10, 20, 30, 40, 50])
    print(f"一维数组: {arr_1d}")
    print(f"  arr[0] = {arr_1d[0]}")
    print(f"  arr[-1] = {arr_1d[-1]}")
    print()

    # 二维数组索引
    arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"二维数组:\n{arr_2d}")
    print(f"  arr[0, 0] = {arr_2d[0, 0]}")
    print(f"  arr[1, 2] = {arr_2d[1, 2]}")
    print(f"  arr[-1, -1] = {arr_2d[-1, -1]}")


def demo_slicing():
    """切片操作"""
    print("=" * 50)
    print("2. 切片操作 [start:stop:step]")
    print("=" * 50)

    arr = np.arange(10)
    print(f"原数组: {arr}")
    print()

    # 基本切片
    print(f"arr[2:7] = {arr[2:7]}")
    print(f"arr[:5] = {arr[:5]}")
    print(f"arr[5:] = {arr[5:]}")
    print()

    # 带步长的切片
    print(f"arr[::2] = {arr[::2]}")
    print(f"arr[1::2] = {arr[1::2]}")
    print()

    # 反向切片
    print(f"arr[::-1] = {arr[::-1]}")
    print(f"arr[::-2] = {arr[::-2]}")


def demo_2d_slicing():
    """二维数组切片"""
    print("=" * 50)
    print("3. 二维数组切片")
    print("=" * 50)

    arr = np.arange(20).reshape(4, 5)
    print(f"4x5数组:\n{arr}")
    print()

    # 行切片
    print(f"前2行 arr[:2, :]:\n{arr[:2, :]}")

    # 列切片
    print(f"第2-4列 arr[:, 1:4]:\n{arr[:, 1:4]}")

    # 子矩阵
    print(f"2-3行,2-3列 arr[1:3, 1:3]:\n{arr[1:3, 1:3]}")

    # 间隔取值
    print(f"每隔一行每隔一列 arr[::2, ::2]:\n{arr[::2, ::2]}")


def demo_boolean_indexing():
    """布尔索引"""
    print("=" * 50)
    print("4. 布尔索引")
    print("=" * 50)

    arr = np.arange(1, 11)
    print(f"原数组: {arr}")
    print()

    # 单条件筛选
    print(f"arr > 5: {arr[arr > 5]}")
    print(f"arr % 2 == 0: {arr[arr % 2 == 0]}")
    print()

    # 多条件筛选 (使用 & 和 |)
    print(f"(arr >= 3) & (arr <= 7): {arr[(arr >= 3) & (arr <= 7)]}")
    print(f"(arr < 3) | (arr > 8): {arr[(arr < 3) | (arr > 8)]}")
    print()

    # 取反
    print(f"~(arr > 5): {arr[~(arr > 5)]}")


def demo_fancy_indexing():
    """花式索引"""
    print("=" * 50)
    print("5. 花式索引")
    print("=" * 50)

    arr = np.arange(10, 20)
    print(f"原数组: {arr}")
    print()

    # 使用整数数组索引
    indices = [0, 2, 5, 8]
    print(f"索引 {indices}: {arr[indices]}")

    # 二维数组花式索引
    arr_2d = np.arange(12).reshape(3, 4)
    print(f"\n二维数组:\n{arr_2d}")

    # 选择特定行
    print(f"选择第0行和第2行:\n{arr_2d[[0, 2]]}")

    # 选择特定元素
    rows = [0, 1, 2]
    cols = [0, 1, 2]
    print(f"选择对角线元素 (0,0),(1,1),(2,2): {arr_2d[rows, cols]}")


def demo_where():
    """np.where 条件索引"""
    print("=" * 50)
    print("6. np.where 条件索引")
    print("=" * 50)

    arr = np.array([1, -2, 3, -4, 5, -6])
    print(f"原数组: {arr}")
    print()

    # 返回满足条件的索引
    positive_idx = np.where(arr > 0)
    print(f"正数的索引: {positive_idx[0]}")
    print(f"正数的值: {arr[positive_idx]}")
    print()

    # 条件替换
    result = np.where(arr > 0, arr, 0)
    print(f"负数替换为0: {result}")

    result2 = np.where(arr > 0, 1, -1)
    print(f"正数标记为1，其他为-1: {result2}")


def demo_all():
    """运行所有演示"""
    demo_basic_indexing()
    print()
    demo_slicing()
    print()
    demo_2d_slicing()
    print()
    demo_boolean_indexing()
    print()
    demo_fancy_indexing()
    print()
    demo_where()


if __name__ == "__main__":
    demo_all()
