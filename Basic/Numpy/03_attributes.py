"""
NumPy 数组属性
对应文档: ../../docs/numpy/03-attributes.md

使用方式：
    python 03_attributes.py
"""

import numpy as np


def demo_shape_ndim_size():
    """演示形状、维度、大小属性"""
    print("=" * 50)
    print("1. 形状、维度、大小属性")
    print("=" * 50)

    arr = np.random.random((3, 4))
    print(f"数组:\n{arr}")
    print()

    # 形状 shape
    print(f"shape (形状): {arr.shape}")
    print(f"  - 行数: {arr.shape[0]}")
    print(f"  - 列数: {arr.shape[1]}")

    # 维度 ndim
    print(f"ndim (维度数): {arr.ndim}")

    # 元素总数 size
    print(f"size (元素总数): {arr.size}")


def demo_dtype_itemsize_nbytes():
    """演示数据类型和内存属性"""
    print("=" * 50)
    print("2. 数据类型和内存属性")
    print("=" * 50)

    arr = np.random.random((3, 4))

    # 数据类型 dtype
    print(f"dtype (数据类型): {arr.dtype}")

    # 每个元素的字节大小
    print(f"itemsize (每元素字节): {arr.itemsize}")

    # 总字节数
    print(f"nbytes (总字节数): {arr.nbytes}")
    print(f"  验证: {arr.size} * {arr.itemsize} = {arr.size * arr.itemsize}")


def demo_dtypes():
    """演示常用数据类型"""
    print("=" * 50)
    print("3. 常用数据类型")
    print("=" * 50)

    print("整数类型:")
    for dtype in [np.int8, np.int16, np.int32, np.int64]:
        info = np.iinfo(dtype)
        print(f"  {dtype.__name__}: 范围 [{info.min}, {info.max}]")

    print("\n浮点类型:")
    for dtype in [np.float16, np.float32, np.float64]:
        info = np.finfo(dtype)
        print(f"  {dtype.__name__}: 精度 {info.precision} 位小数")

    print("\n其他类型:")
    print(f"  bool: True/False")
    print(f"  complex64: 复数 (2个float32)")
    print(f"  complex128: 复数 (2个float64)")


def demo_astype():
    """演示数据类型转换"""
    print("=" * 50)
    print("4. 数据类型转换 (astype)")
    print("=" * 50)

    # 创建浮点数组
    arr_float = np.array([1.5, 2.7, 3.2, 4.8])
    print(f"原数组 (float): {arr_float}")
    print(f"  dtype: {arr_float.dtype}")

    # 转换为整数
    arr_int = arr_float.astype(np.int32)
    print(f"转为 int32: {arr_int}")
    print(f"  dtype: {arr_int.dtype}")

    # 转换为字符串
    arr_str = arr_float.astype(str)
    print(f"转为 str: {arr_str}")
    print(f"  dtype: {arr_str.dtype}")


def demo_bool_array():
    """演示布尔数组"""
    print("=" * 50)
    print("5. 布尔数组")
    print("=" * 50)

    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"原数组: {arr}")

    # 通过比较运算创建布尔数组
    bool_arr = arr > 5
    print(f"arr > 5: {bool_arr}")
    print(f"  dtype: {bool_arr.dtype}")

    # 布尔数组可用于索引
    print(f"arr[arr > 5]: {arr[bool_arr]}")

    # 统计 True 的数量
    print(f"大于5的元素个数: {bool_arr.sum()}")


def demo_all():
    """运行所有演示"""
    demo_shape_ndim_size()
    print()
    demo_dtype_itemsize_nbytes()
    print()
    demo_dtypes()
    print()
    demo_astype()
    print()
    demo_bool_array()


if __name__ == "__main__":
    demo_all()
