"""
NumPy 数组运算
对应文档: ../../docs/numpy/05-operations.md

使用方式：
    python 05_operations.py
"""

import numpy as np


def demo_arithmetic():
    """算术运算"""
    print("=" * 50)
    print("1. 算术运算 (元素级运算)")
    print("=" * 50)
    
    a = np.array([1, 2, 3, 4])
    b = np.array([5, 6, 7, 8])
    print(f"a = {a}")
    print(f"b = {b}")
    print()
    
    print(f"a + b = {a + b}")
    print(f"a - b = {a - b}")
    print(f"a * b = {a * b}")
    print(f"a / b = {a / b}")
    print()
    
    print(f"a ** 2 = {a ** 2}")
    print(f"a ** 0.5 = {a ** 0.5}")
    print(f"a // 2 = {a // 2}")
    print(f"a % 2 = {a % 2}")


def demo_comparison():
    """比较运算"""
    print("=" * 50)
    print("2. 比较运算")
    print("=" * 50)
    
    a = np.array([1, 2, 3, 4])
    b = np.array([4, 3, 2, 1])
    print(f"a = {a}")
    print(f"b = {b}")
    print()
    
    # 元素级比较
    print(f"a == b: {a == b}")
    print(f"a != b: {a != b}")
    print(f"a > b: {a > b}")
    print(f"a < b: {a < b}")
    print()
    
    # 数组整体比较
    print(f"np.array_equal(a, b): {np.array_equal(a, b)}")
    print(f"np.any(a == b): {np.any(a == b)}")
    print(f"np.all(a != b): {np.all(a != b)}")


def demo_statistics():
    """统计运算"""
    print("=" * 50)
    print("3. 统计运算")
    print("=" * 50)
    
    np.random.seed(42)
    arr = np.random.randint(1, 100, size=10)
    print(f"随机数组: {arr}")
    print()
    
    # 基本统计
    print(f"sum (求和): {arr.sum()}")
    print(f"mean (均值): {arr.mean():.2f}")
    print(f"std (标准差): {arr.std():.2f}")
    print(f"var (方差): {arr.var():.2f}")
    print()
    
    # 极值
    print(f"min (最小值): {arr.min()}")
    print(f"max (最大值): {arr.max()}")
    print(f"argmin (最小值索引): {arr.argmin()}")
    print(f"argmax (最大值索引): {arr.argmax()}")
    print()
    
    # 累积运算
    print(f"cumsum (累积和): {arr.cumsum()}")
    print(f"cumprod (累积积，前5个): {arr[:5].cumprod()}")


def demo_axis_operations():
    """沿轴运算"""
    print("=" * 50)
    print("4. 沿轴运算 (axis 参数)")
    print("=" * 50)
    
    arr = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
    print(f"3x3数组:\n{arr}")
    print()
    
    # axis=None: 所有元素
    print(f"sum(): {arr.sum()}")
    
    # axis=0: 沿行方向 (按列计算)
    print(f"sum(axis=0) 按列求和: {arr.sum(axis=0)}")
    print(f"mean(axis=0) 按列求均值: {arr.mean(axis=0)}")
    
    # axis=1: 沿列方向 (按行计算)
    print(f"sum(axis=1) 按行求和: {arr.sum(axis=1)}")
    print(f"mean(axis=1) 按行求均值: {arr.mean(axis=1)}")


def demo_math_functions():
    """数学函数"""
    print("=" * 50)
    print("5. 数学函数")
    print("=" * 50)
    
    arr = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
    print(f"角度数组 (弧度): {arr}")
    print()
    
    # 三角函数
    print(f"sin: {np.sin(arr).round(3)}")
    print(f"cos: {np.cos(arr).round(3)}")
    print()
    
    # 指数和对数
    arr2 = np.array([1, 2, 3, 4, 5])
    print(f"原数组: {arr2}")
    print(f"exp: {np.exp(arr2).round(3)}")
    print(f"log (自然对数): {np.log(arr2).round(3)}")
    print(f"log10: {np.log10(arr2).round(3)}")
    print()
    
    # 取整函数
    arr3 = np.array([1.2, 2.5, 3.7, -1.2, -2.5])
    print(f"原数组: {arr3}")
    print(f"floor (向下取整): {np.floor(arr3)}")
    print(f"ceil (向上取整): {np.ceil(arr3)}")
    print(f"round (四舍五入): {np.round(arr3)}")
    print(f"abs (绝对值): {np.abs(arr3)}")


def demo_logical_operations():
    """逻辑运算"""
    print("=" * 50)
    print("6. 逻辑运算")
    print("=" * 50)
    
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    print(f"a = {a}")
    print(f"b = {b}")
    print()
    
    print(f"np.logical_and(a, b): {np.logical_and(a, b)}")
    print(f"np.logical_or(a, b): {np.logical_or(a, b)}")
    print(f"np.logical_not(a): {np.logical_not(a)}")
    print(f"np.logical_xor(a, b): {np.logical_xor(a, b)}")


def demo_all():
    """运行所有演示"""
    demo_arithmetic()
    print()
    demo_comparison()
    print()
    demo_statistics()
    print()
    demo_axis_operations()
    print()
    demo_math_functions()
    print()
    demo_logical_operations()


if __name__ == "__main__":
    demo_all()
