"""
NumPy 数组变形
对应文档: ../../docs/numpy/07-reshape.md

使用方式：
    python 07_reshape.py
"""

import numpy as np


def demo_reshape():
    """reshape 变形操作"""
    print("=" * 50)
    print("1. reshape 变形")
    print("=" * 50)

    arr = np.arange(1, 13)
    print(f"原数组: {arr}")
    print(f"形状: {arr.shape}")
    print()

    # 变形为二维
    arr_2d = arr.reshape(3, 4)
    print(f"reshape(3, 4):\n{arr_2d}")

    arr_2d_alt = arr.reshape(4, 3)
    print(f"reshape(4, 3):\n{arr_2d_alt}")

    # 使用 -1 自动计算维度
    arr_auto = arr.reshape(2, -1)
    print(f"reshape(2, -1):\n{arr_auto}")

    arr_auto2 = arr.reshape(-1, 6)
    print(f"reshape(-1, 6):\n{arr_auto2}")


def demo_flatten_ravel():
    """flatten 和 ravel 展平"""
    print("=" * 50)
    print("2. flatten 和 ravel 展平")
    print("=" * 50)

    arr = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"原数组:\n{arr}")
    print()

    # flatten: 返回副本
    flat = arr.flatten()
    print(f"flatten(): {flat}")

    # ravel: 返回视图（如果可能）
    rav = arr.ravel()
    print(f"ravel(): {rav}")
    print()

    # 验证副本 vs 视图
    print("=== 副本 vs 视图 ===")
    arr_test = np.array([[1, 2, 3], [4, 5, 6]])

    flat = arr_test.flatten()
    flat[0] = 999
    print(f"修改 flatten[0]=999 后，原数组:\n{arr_test}")

    rav = arr_test.ravel()
    rav[1] = 888
    print(f"修改 ravel[1]=888 后，原数组:\n{arr_test}")


def demo_transpose():
    """转置操作"""
    print("=" * 50)
    print("3. 转置操作")
    print("=" * 50)

    # 二维数组转置
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"原数组 (2x3):\n{arr}")
    print(f"arr.T (3x2):\n{arr.T}")
    print()

    # 三维数组转置
    arr_3d = np.arange(24).reshape(2, 3, 4)
    print(f"三维数组形状: {arr_3d.shape}")
    print(f"arr_3d.T 形状: {arr_3d.T.shape}")

    # 指定轴顺序
    transposed = np.transpose(arr_3d, axes=(1, 0, 2))
    print(f"transpose(axes=(1,0,2)) 形状: {transposed.shape}")


def demo_squeeze_expand():
    """squeeze 和 expand_dims"""
    print("=" * 50)
    print("4. squeeze 和 expand_dims")
    print("=" * 50)

    # squeeze: 移除长度为1的维度
    arr = np.array([[[1, 2, 3]]])
    print(f"原数组形状: {arr.shape}")
    squeezed = np.squeeze(arr)
    print(f"squeeze 后形状: {squeezed.shape}")
    print()

    # expand_dims: 增加维度
    arr = np.array([1, 2, 3])
    print(f"原数组: {arr}, 形状: {arr.shape}")

    expanded_0 = np.expand_dims(arr, axis=0)
    print(f"expand_dims(axis=0): 形状: {expanded_0.shape}")
    print(f"  {expanded_0}")

    expanded_1 = np.expand_dims(arr, axis=1)
    print(f"expand_dims(axis=1): 形状: {expanded_1.shape}")
    print(f"  {expanded_1.T[0]}")


def demo_resize():
    """resize 调整大小"""
    print("=" * 50)
    print("5. resize 调整大小")
    print("=" * 50)

    arr = np.array([1, 2, 3, 4])
    print(f"原数组: {arr}")
    print()

    # 使用 np.resize (会重复元素)
    resized = np.resize(arr, (2, 4))
    print(f"np.resize(arr, (2, 4)):\n{resized}")

    resized2 = np.resize(arr, (3, 3))
    print(f"np.resize(arr, (3, 3)):\n{resized2}")


def demo_newaxis():
    """newaxis 增加维度"""
    print("=" * 50)
    print("6. np.newaxis 增加维度")
    print("=" * 50)

    arr = np.array([1, 2, 3, 4, 5])
    print(f"原数组: {arr}, 形状: {arr.shape}")
    print()

    # 增加行维度
    row = arr[np.newaxis, :]
    print(f"arr[np.newaxis, :] 形状: {row.shape}")
    print(f"  {row}")

    # 增加列维度
    col = arr[:, np.newaxis]
    print(f"arr[:, np.newaxis] 形状: {col.shape}")
    print(f"  {col.T[0]}")


def demo_all():
    """运行所有演示"""
    demo_reshape()
    print()
    demo_flatten_ravel()
    print()
    demo_transpose()
    print()
    demo_squeeze_expand()
    print()
    demo_resize()
    print()
    demo_newaxis()


if __name__ == "__main__":
    demo_all()
