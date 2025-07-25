"""
插值方法
对应文档: ../../docs/scipy/05-interpolate.md
"""

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


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
    f_linear = interpolate.interp1d(x, y, kind="linear")
    print(f"线性插值 f(2.5) = {f_linear(2.5):.4f}")

    # 三次插值
    f_cubic = interpolate.interp1d(x, y, kind="cubic")
    print(f"三次插值 f(2.5) = {f_cubic(2.5):.4f}")

    # 真实值
    print(f"真实值 2.5^2 = {2.5**2}")

    # === 可视化 ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 插值对比
    ax1 = axes[0]
    x_new = np.linspace(0, 5, 100)
    y_true = x_new**2

    ax1.scatter(x, y, c="red", s=100, zorder=5, label="已知数据点", edgecolors="black")
    ax1.plot(x_new, y_true, "k--", lw=2, label="真实曲线 $y = x^2$")
    ax1.plot(x_new, f_linear(x_new), "b-", lw=2, label="线性插值", alpha=0.7)
    ax1.plot(x_new, f_cubic(x_new), "g-", lw=2, label="三次插值", alpha=0.7)

    # 标记 x=2.5 处的插值
    ax1.axvline(2.5, color="gray", linestyle=":", alpha=0.5)
    ax1.plot(2.5, f_linear(2.5), "bs", markersize=10)
    ax1.plot(2.5, f_cubic(2.5), "g^", markersize=10)
    ax1.plot(2.5, 2.5**2, "k*", markersize=15)

    ax1.set_title("一维插值方法对比")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 误差分析
    ax2 = axes[1]
    error_linear = np.abs(f_linear(x_new) - y_true)
    error_cubic = np.abs(f_cubic(x_new) - y_true)

    ax2.plot(x_new, error_linear, "b-", lw=2, label="线性插值误差")
    ax2.plot(x_new, error_cubic, "g-", lw=2, label="三次插值误差")
    ax2.scatter(x, np.zeros_like(x), c="red", s=50, zorder=5, label="数据点 (误差=0)")

    ax2.set_title("插值误差分析")
    ax2.set_xlabel("x")
    ax2.set_ylabel("绝对误差")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.5)

    plt.tight_layout()
    plt.savefig("outputs/scipy/05_interp1d.png", dpi=150, bbox_inches="tight")


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
        print(f"  x={xi}: 插值={yi:.4f}, 真实={yt:.4f}, 误差={abs(yi - yt):.6f}")

    # === 可视化 ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 样条插值
    ax1 = axes[0]
    x_smooth = np.linspace(0, 5, 200)
    y_spline = interpolate.splev(x_smooth, tck)
    y_true_smooth = np.sin(x_smooth)

    ax1.scatter(x, y, c="red", s=100, zorder=5, label="已知数据点", edgecolors="black")
    ax1.plot(x_smooth, y_true_smooth, "k--", lw=2, label="真实曲线 $y = sin(x)$")
    ax1.plot(x_smooth, y_spline, "b-", lw=2, label="三次样条插值")

    # 标记插值点
    for xi, yi in zip(x_new, y_interp):
        ax1.plot(xi, yi, "g^", markersize=10)

    ax1.set_title("三次样条插值")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 导数
    ax2 = axes[1]
    y_der1 = interpolate.splev(x_smooth, tck, der=1)
    y_der2 = interpolate.splev(x_smooth, tck, der=2)

    ax2.plot(x_smooth, np.cos(x_smooth), "k--", lw=2, label="$y' = cos(x)$ (真实)")
    ax2.plot(x_smooth, y_der1, "b-", lw=2, label="样条一阶导数")
    ax2.plot(x_smooth, y_der2, "g-", lw=2, alpha=0.7, label="样条二阶导数")

    ax2.set_title("样条插值的导数")
    ax2.set_xlabel("x")
    ax2.set_ylabel("导数值")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/scipy/05_spline.png", dpi=150, bbox_inches="tight")


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

    # === 可视化 ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始网格数据
    ax1 = axes[0]
    im1 = ax1.imshow(
        Z, extent=[0, 4, 0, 4], origin="lower", cmap="viridis", aspect="auto"
    )
    ax1.scatter(
        X.flatten(), Y.flatten(), c="white", s=30, marker="s", edgecolors="black"
    )
    ax1.set_title("原始网格数据 (5×5)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    plt.colorbar(im1, ax=ax1, label="z = sin(x) + cos(y)")

    # 插值后的细网格
    ax2 = axes[1]
    x_fine = np.linspace(0, 4, 50)
    y_fine = np.linspace(0, 4, 50)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    points_fine = np.column_stack([X_fine.flatten(), Y_fine.flatten()])
    Z_fine = interp_func(points_fine).reshape(50, 50)

    im2 = ax2.imshow(
        Z_fine, extent=[0, 4, 0, 4], origin="lower", cmap="viridis", aspect="auto"
    )
    ax2.scatter(
        X.flatten(),
        Y.flatten(),
        c="white",
        s=30,
        marker="s",
        edgecolors="black",
        label="原始点",
    )
    ax2.scatter(points[:, 0], points[:, 1], c="red", s=100, marker="*", label="插值点")
    ax2.set_title("插值后的细网格 (50×50)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend()
    plt.colorbar(im2, ax=ax2)

    # 3D 曲面
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.plot_surface(X_fine, Y_fine, Z_fine, cmap="viridis", alpha=0.8)
    ax3.scatter(
        X.flatten(), Y.flatten(), Z.flatten(), c="red", s=50, label="原始数据点"
    )
    ax3.set_title("插值曲面")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")

    # 移除原来的2D ax3,替换为3D
    axes[2].remove()

    plt.tight_layout()
    plt.savefig("outputs/scipy/05_interp2d.png", dpi=150, bbox_inches="tight")


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
        np.column_stack([x, y]), z, kernel="thin_plate_spline"
    )

    # 在新点插值
    test_points = np.array([[1.0, 1.0], [2.0, 2.0]])
    values = rbf(test_points)

    print("RBF 插值结果:")
    for p, v in zip(test_points, values):
        true_v = np.sin(p[0]) + np.cos(p[1])
        print(f"  点({p[0]}, {p[1]}): 插值={v:.4f}, 真实={true_v:.4f}")

    # === 可视化 ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 散点数据
    ax1 = axes[0]
    sc = ax1.scatter(x, y, c=z, s=150, cmap="viridis", edgecolors="black", zorder=5)
    ax1.set_title("原始散点数据")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_xlim(-0.5, 4.5)
    ax1.set_ylim(-0.5, 4.5)
    plt.colorbar(sc, ax=ax1, label="z = sin(x) + cos(y)")
    ax1.grid(True, alpha=0.3)

    # RBF 插值结果
    ax2 = axes[1]
    x_grid = np.linspace(0, 4, 50)
    y_grid = np.linspace(0, 4, 50)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    points_grid = np.column_stack([X_grid.flatten(), Y_grid.flatten()])
    Z_grid = rbf(points_grid).reshape(50, 50)

    im = ax2.imshow(
        Z_grid, extent=[0, 4, 0, 4], origin="lower", cmap="viridis", aspect="auto"
    )
    ax2.scatter(x, y, c="white", s=50, edgecolors="black", label="原始点")
    ax2.scatter(
        test_points[:, 0], test_points[:, 1], c="red", s=100, marker="*", label="测试点"
    )
    ax2.set_title("RBF 插值结果 (薄板样条)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend()
    plt.colorbar(im, ax=ax2)

    plt.tight_layout()
    plt.savefig("outputs/scipy/05_rbf.png", dpi=150, bbox_inches="tight")


def demo_all():
    """运行所有演示"""
    import os

    os.makedirs("outputs/scipy", exist_ok=True)

    demo_interp1d()
    print()
    demo_spline()
    print()
    demo_interp2d()
    print()
    demo_rbf()


if __name__ == "__main__":
    demo_all()
