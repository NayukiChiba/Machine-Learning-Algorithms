"""
插值方法
对应文档: ../../docs/scipy/05-interpolate.md
"""

import numpy as np
from scipy import interpolate


def demo_interp1d():
    """演示一维插值"""
    print("=" * 50)
    print("1. 一维插值")
    print("=" * 50)
    
    # 已知数据点
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([0, 1, 4, 9, 16, 25])
    
    print("已知数据点:")
    print(f"  x = {x}")
    print(f"  y = {y}")
    print()
    
    # 线性插值
    f_linear = interpolate.interp1d(x, y, kind='linear')
    print(f"线性插值 f(2.5) = {f_linear(2.5):.4f}")
    
    # 三次插值
    f_cubic = interpolate.interp1d(x, y, kind='cubic')
    print(f"三次插值 f(2.5) = {f_cubic(2.5):.4f}")
    
    # 真实值
    print(f"真实值 2.5² = {2.5**2}")


def demo_spline():
    """演示样条插值"""
    print("=" * 50)
    print("2. 样条插值")
    print("=" * 50)
    
    # 已知数据点
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.sin(x)
    
    print("已知数据点 y = sin(x):")
    print(f"  x = {x}")
    print(f"  y = {np.round(y, 4)}")
    print()
    
    # 三次样条
    tck = interpolate.splrep(x, y, s=0)
    
    # 计算插值点
    x_new = np.array([0.5, 1.5, 2.5, 3.5])
    y_interp = interpolate.splev(x_new, tck)
    y_true = np.sin(x_new)
    
    print("样条插值结果:")
    for xi, yi, yt in zip(x_new, y_interp, y_true):
        print(f"  x={xi}: 插值={yi:.4f}, 真实={yt:.4f}, 误差={abs(yi-yt):.6f}")


def demo_interp2d():
    """演示二维插值"""
    print("=" * 50)
    print("3. 二维插值")
    print("=" * 50)
    
    # 创建网格数据
    x = np.arange(0, 5)
    y = np.arange(0, 5)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) + np.cos(Y)
    
    print(f"网格大小: {Z.shape}")
    print()
    
    # 使用 RegularGridInterpolator
    interp_func = interpolate.RegularGridInterpolator((x, y), Z.T)
    
    # 在新点插值
    points = np.array([[1.5, 2.5], [2.5, 3.5]])
    values = interp_func(points)
    
    print("二维插值结果:")
    for p, v in zip(points, values):
        true_v = np.sin(p[0]) + np.cos(p[1])
        print(f"  点({p[0]}, {p[1]}): 插值={v:.4f}, 真实={true_v:.4f}")


def demo_rbf():
    """演示径向基函数插值"""
    print("=" * 50)
    print("4. 径向基函数 (RBF) 插值")
    print("=" * 50)
    
    # 散点数据
    np.random.seed(42)
    x = np.random.rand(10) * 4
    y = np.random.rand(10) * 4
    z = np.sin(x) + np.cos(y)
    
    print(f"散点数量: {len(x)}")
    print()
    
    # RBF 插值
    rbf = interpolate.RBFInterpolator(
        np.column_stack([x, y]), z, kernel='thin_plate_spline'
    )
    
    # 在新点插值
    test_points = np.array([[1.0, 1.0], [2.0, 2.0]])
    values = rbf(test_points)
    
    print("RBF 插值结果:")
    for p, v in zip(test_points, values):
        true_v = np.sin(p[0]) + np.cos(p[1])
        print(f"  点({p[0]}, {p[1]}): 插值={v:.4f}, 真实={true_v:.4f}")


def demo_all():
    """运行所有演示"""
    demo_interp1d()
    print()
    demo_spline()
    print()
    demo_interp2d()
    print()
    demo_rbf()


if __name__ == "__main__":
    demo_all()
