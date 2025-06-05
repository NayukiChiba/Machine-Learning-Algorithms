"""
NumPy 数组连接和分割
对应文档: ../../docs/numpy/09-concat-split.md

使用方式：
    python 09_concat_split.py
"""

import numpy as np


def demo_concatenate():
    """concatenate 连接"""
    print("=" * 50)
    print("1. concatenate 连接")
    print("=" * 50)
    
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    print(f"A:\n{A}")
    print(f"B:\n{B}")
    print()
    
    # 沿 axis=0 连接（垂直）
    concat_0 = np.concatenate([A, B], axis=0)
    print(f"concatenate(axis=0) 垂直连接:\n{concat_0}")
    print(f"  形状: {concat_0.shape}")
    print()
    
    # 沿 axis=1 连接（水平）
    concat_1 = np.concatenate([A, B], axis=1)
    print(f"concatenate(axis=1) 水平连接:\n{concat_1}")
    print(f"  形状: {concat_1.shape}")


def demo_vstack_hstack():
    """vstack 和 hstack"""
    print("=" * 50)
    print("2. vstack 和 hstack")
    print("=" * 50)
    
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[7, 8, 9]])
    print(f"A (2x3):\n{A}")
    print(f"B (1x3):\n{B}")
    print()
    
    # vstack: 垂直堆叠
    vstacked = np.vstack([A, B])
    print(f"vstack([A, B]):\n{vstacked}")
    print(f"  形状: {vstacked.shape}")
    print()
    
    # hstack: 水平堆叠
    C = np.array([[10], [20]])
    print(f"C (2x1):\n{C}")
    hstacked = np.hstack([A, C])
    print(f"hstack([A, C]):\n{hstacked}")
    print(f"  形状: {hstacked.shape}")


def demo_stack():
    """stack 沿新轴堆叠"""
    print("=" * 50)
    print("3. stack 沿新轴堆叠")
    print("=" * 50)
    
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    print(f"A (2x2):\n{A}")
    print(f"B (2x2):\n{B}")
    print()
    
    # stack: 增加新维度
    stacked_0 = np.stack([A, B], axis=0)
    print(f"stack(axis=0) 形状: {stacked_0.shape}")
    print(f"  结果:\n{stacked_0}")
    print()
    
    stacked_2 = np.stack([A, B], axis=2)
    print(f"stack(axis=2) 形状: {stacked_2.shape}")
    print(f"  结果:\n{stacked_2}")


def demo_dstack():
    """dstack 深度堆叠"""
    print("=" * 50)
    print("4. dstack 深度堆叠")
    print("=" * 50)
    
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    print(f"A:\n{A}")
    print(f"B:\n{B}")
    print()
    
    # dstack: 沿第三个轴堆叠
    dstacked = np.dstack([A, B])
    print(f"dstack([A, B]) 形状: {dstacked.shape}")
    print(f"  结果:\n{dstacked}")


def demo_split():
    """split 分割"""
    print("=" * 50)
    print("5. split 分割")
    print("=" * 50)
    
    arr = np.arange(12).reshape(4, 3)
    print(f"原数组 (4x3):\n{arr}")
    print()
    
    # 沿 axis=0 分割成 2 份
    split_0 = np.split(arr, 2, axis=0)
    print(f"split(arr, 2, axis=0):")
    for i, part in enumerate(split_0):
        print(f"  第{i+1}部分:\n{part}")
    print()
    
    # 沿 axis=1 分割成 3 份
    split_1 = np.split(arr, 3, axis=1)
    print(f"split(arr, 3, axis=1):")
    for i, part in enumerate(split_1):
        print(f"  第{i+1}部分: {part.flatten()}")


def demo_vsplit_hsplit():
    """vsplit 和 hsplit"""
    print("=" * 50)
    print("6. vsplit 和 hsplit")
    print("=" * 50)
    
    arr = np.arange(12).reshape(4, 3)
    print(f"原数组 (4x3):\n{arr}")
    print()
    
    # vsplit: 垂直分割
    vsplit_result = np.vsplit(arr, 2)
    print("vsplit(arr, 2):")
    for i, part in enumerate(vsplit_result):
        print(f"  第{i+1}部分:\n{part}")
    print()
    
    # hsplit: 水平分割
    hsplit_result = np.hsplit(arr, 3)
    print("hsplit(arr, 3):")
    for i, part in enumerate(hsplit_result):
        print(f"  第{i+1}部分: {part.flatten()}")


def demo_array_split():
    """array_split 不均匀分割"""
    print("=" * 50)
    print("7. array_split 不均匀分割")
    print("=" * 50)
    
    arr = np.arange(10)
    print(f"原数组: {arr}")
    print()
    
    # 分成 3 份（不均匀）
    split_result = np.array_split(arr, 3)
    print("array_split(arr, 3):")
    for i, part in enumerate(split_result):
        print(f"  第{i+1}部分 (大小{len(part)}): {part}")


def demo_all():
    """运行所有演示"""
    demo_concatenate()
    print()
    demo_vstack_hstack()
    print()
    demo_stack()
    print()
    demo_dstack()
    print()
    demo_split()
    print()
    demo_vsplit_hsplit()
    print()
    demo_array_split()


if __name__ == "__main__":
    demo_all()
