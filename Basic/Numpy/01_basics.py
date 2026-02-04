"""
NumPy 基础入门
对应文档: ../../docs/numpy/01-basics.md

使用方式：
    python 01_basics.py
    或
    from Basic.Numpy.01_basics import *
    demo_all()
"""

import numpy as np


def demo_version_info():
    """演示如何查看 NumPy 版本信息"""
    print("=" * 50)
    print("1. NumPy 版本信息")
    print("=" * 50)

    print(f"NumPy版本: {np.__version__}")
    print()

    # 获取打印选项
    print("打印选项:")
    options = np.get_printoptions()
    print(f"  precision: {options['precision']}")
    print(f"  threshold: {options['threshold']}")
    print(f"  linewidth: {options['linewidth']}")


def demo_numpy_vs_list():
    """演示 NumPy 数组与 Python 列表的区别"""
    print("=" * 50)
    print("2. NumPy 数组 vs Python 列表")
    print("=" * 50)

    # 创建 Python 列表
    py_list = [1, 2, 3, 4, 5]
    print(f"Python列表: {py_list}")
    print(f"类型: {type(py_list)}")

    # 创建 NumPy 数组
    np_array = np.array([1, 2, 3, 4, 5])
    print(f"NumPy数组: {np_array}")
    print(f"类型: {type(np_array)}")
    print()

    # 乘法运算的区别
    print("乘法运算的区别:")
    print(f"  列表 * 2 = {py_list * 2}")  # 列表重复
    print(f"  数组 * 2 = {np_array * 2}")  # 元素级运算
    print()

    # 加法运算的区别
    print("加法运算的区别:")
    print(f"  列表 + [6] = {py_list + [6]}")  # 列表拼接
    print(f"  数组 + 6 = {np_array + 6}")  # 元素级运算


def demo_performance():
    """演示 NumPy 的性能优势"""
    print("=" * 50)
    print("3. NumPy 性能优势")
    print("=" * 50)

    import time

    size = 1000000

    # Python 列表运算
    py_list = list(range(size))
    start = time.time()
    result_list = [x * 2 for x in py_list]
    py_time = time.time() - start

    # NumPy 数组运算
    np_array = np.arange(size)
    start = time.time()
    result_np = np_array * 2
    np_time = time.time() - start

    print(f"数据规模: {size:,}")
    print(f"Python列表耗时: {py_time:.4f}秒")
    print(f"NumPy数组耗时: {np_time:.4f}秒")
    print(f"NumPy快了约 {py_time / np_time:.1f} 倍")


def demo_ndarray_basics():
    """演示 ndarray 的基本操作"""
    print("=" * 50)
    print("4. ndarray 基本操作")
    print("=" * 50)

    # 一维数组
    arr_1d = np.array([1, 2, 3, 4, 5])
    print(f"一维数组: {arr_1d}")
    print(f"  形状: {arr_1d.shape}")
    print(f"  维度: {arr_1d.ndim}")
    print()

    # 二维数组
    arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"二维数组:\n{arr_2d}")
    print(f"  形状: {arr_2d.shape}")
    print(f"  维度: {arr_2d.ndim}")
    print()

    # 三维数组
    arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(f"三维数组:\n{arr_3d}")
    print(f"  形状: {arr_3d.shape}")
    print(f"  维度: {arr_3d.ndim}")


def demo_all():
    """运行所有演示"""
    demo_version_info()
    print()
    demo_numpy_vs_list()
    print()
    demo_performance()
    print()
    demo_ndarray_basics()


if __name__ == "__main__":
    demo_all()
