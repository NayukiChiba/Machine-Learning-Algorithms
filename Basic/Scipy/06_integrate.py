"""
数值积分
对应文档: ../../docs/scipy/06-integrate.md
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def demo_quad():
    """演示定积分"""
    print("=" * 50)
    print("1. 定积分")
    print("=" * 50)

    # 积分 ∫x^2 dx, 从0到1
    result1, error1 = integrate.quad(lambda x: x**2, 0, 1)
    print(f"∫₀¹ x^2 dx = {result1:.6f} (误差: {error1:.2e})")
    print(f"解析解: 1/3 = {1 / 3:.6f}")
    print()

    # 积分 ∫sin(x) dx, 从0到π
    result2, error2 = integrate.quad(np.sin, 0, np.pi)
    print(f"∫₀π sin(x) dx = {result2:.6f} (误差: {error2:.2e})")
    print(f"解析解: 2")
    print()

    # 无穷积分 ∫e^(-x^2) dx, 从-inf到+inf
    result3, error3 = integrate.quad(lambda x: np.exp(-(x**2)), -np.inf, np.inf)
    print(f"∫_{{-inf}}^{{+inf}} e^(-x^2) dx = {result3:.6f}")
    print(f"解析解: √π = {np.sqrt(np.pi):.6f}")

    # === 可视化 ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # x^2 积分
    ax1 = axes[0]
    x = np.linspace(0, 1, 100)
    y = x**2
    ax1.plot(x, y, "b-", lw=2, label="$f(x) = x^2$")
    ax1.fill_between(x, y, alpha=0.3, color="blue")
    ax1.axhline(0, color="black", lw=0.5)
    ax1.set_title(f"$\\int_0^1 x^2 dx = {result1:.4f}$")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # sin(x) 积分
    ax2 = axes[1]
    x = np.linspace(0, np.pi, 100)
    y = np.sin(x)
    ax2.plot(x, y, "r-", lw=2, label="$f(x) = sin(x)$")
    ax2.fill_between(x, y, alpha=0.3, color="red")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_title(f"$\\int_0^\\pi sin(x) dx = {result2:.4f}$")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_xticks([0, np.pi / 2, np.pi])
    ax2.set_xticklabels(["0", "π/2", "π"])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 高斯积分
    ax3 = axes[2]
    x = np.linspace(-4, 4, 200)
    y = np.exp(-(x**2))
    ax3.plot(x, y, "g-", lw=2, label="$f(x) = e^{-x^2}$")
    ax3.fill_between(x, y, alpha=0.3, color="green")
    ax3.axhline(0, color="black", lw=0.5)
    ax3.set_title(f"$\\int_{{-\\infty}}^{{\\infty}} e^{{-x^2}} dx = {result3:.4f}$")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/scipy/06_quad.png", dpi=150, bbox_inches="tight")


def demo_dblquad():
    """演示二重积分"""
    print("=" * 50)
    print("2. 二重积分")
    print("=" * 50)

    # 积分 ∫∫xy dA, 在 [0,1]×[0,2] 上
    result1, error1 = integrate.dblquad(
        lambda y, x: x * y,  # 被积函数
        0,
        1,  # x 的范围
        lambda x: 0,  # y 的下限
        lambda x: 2,  # y 的上限
    )
    print(f"∫₀¹∫₀^2 xy dy dx = {result1:.6f}")
    print(f"解析解: 1")
    print()

    # 圆形区域积分
    result2, error2 = integrate.dblquad(
        lambda y, x: 1, -1, 1, lambda x: -np.sqrt(1 - x**2), lambda x: np.sqrt(1 - x**2)
    )
    print(f"单位圆面积 = {result2:.6f}")
    print(f"解析解: π = {np.pi:.6f}")

    # === 可视化 ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 矩形区域
    ax1 = axes[0]
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = X * Y

    im1 = ax1.contourf(X, Y, Z, levels=20, cmap="viridis")
    ax1.set_title(f"$\\iint xy\\,dA = {result1:.4f}$ (矩形区域)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    plt.colorbar(im1, ax=ax1, label="f(x,y) = xy")

    # 画矩形边界
    rect = plt.Rectangle((0, 0), 1, 2, fill=False, edgecolor="red", linewidth=2)
    ax1.add_patch(rect)

    # 圆形区域
    ax2 = axes[1]
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)

    ax2.fill(x_circle, y_circle, alpha=0.5, color="blue", label=f"面积 = {result2:.4f}")
    ax2.plot(x_circle, y_circle, "b-", lw=2)
    ax2.set_aspect("equal")
    ax2.set_title(f"单位圆面积 = $\\pi$ ≈ {result2:.4f}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)

    plt.tight_layout()
    plt.savefig("outputs/scipy/06_dblquad.png", dpi=150, bbox_inches="tight")


def demo_odeint():
    """演示常微分方程求解"""
    print("=" * 50)
    print("3. 常微分方程 (ODE)")
    print("=" * 50)

    # 一阶ODE: dy/dt = -y, y(0) = 1
    # 解析解: y = e^(-t)
    print("一阶ODE: dy/dt = -y, y(0) = 1")

    def dydt(y, t):
        return -y

    t = np.linspace(0, 5, 6)
    y0 = 1
    y = integrate.odeint(dydt, y0, t)

    print("t\t数值解\t\t解析解")
    for ti, yi in zip(t, y.flatten()):
        print(f"{ti:.1f}\t{yi:.6f}\t{np.exp(-ti):.6f}")

    # === 可视化 ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 数值解 vs 解析解
    ax1 = axes[0]
    t_fine = np.linspace(0, 5, 100)
    y_fine = integrate.odeint(dydt, y0, t_fine)
    y_analytical = np.exp(-t_fine)

    ax1.plot(t_fine, y_analytical, "b-", lw=2, label="解析解: $y = e^{-t}$")
    ax1.plot(t_fine, y_fine, "r--", lw=2, label="数值解 (odeint)")
    ax1.scatter(t, y, c="red", s=80, zorder=5, edgecolors="black")

    ax1.set_title("一阶 ODE: $\\frac{dy}{dt} = -y$, $y(0) = 1$")
    ax1.set_xlabel("t")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 向量场
    ax2 = axes[1]
    t_grid = np.linspace(0, 5, 15)
    y_grid = np.linspace(0, 1.5, 10)
    T, Y_grid = np.meshgrid(t_grid, y_grid)

    # dy/dt = -y
    U = np.ones_like(T)
    V = -Y_grid

    ax2.quiver(T, Y_grid, U, V, color="gray", alpha=0.6)
    ax2.plot(t_fine, y_fine, "r-", lw=2, label="解曲线")
    ax2.set_title("向量场与解曲线")
    ax2.set_xlabel("t")
    ax2.set_ylabel("y")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 1.5)

    plt.tight_layout()
    plt.savefig("outputs/scipy/06_ode.png", dpi=150, bbox_inches="tight")


def demo_ode_system():
    """演示ODE方程组"""
    print("=" * 50)
    print("4. ODE 方程组")
    print("=" * 50)

    # Lotka-Volterra 方程 (捕食者-猎物模型)
    # dx/dt = αx - βxy
    # dy/dt = δxy - γy

    alpha, beta, delta, gamma = 1.0, 0.1, 0.075, 1.5

    def lotka_volterra(state, t):
        x, y = state
        dxdt = alpha * x - beta * x * y
        dydt = delta * x * y - gamma * y
        return [dxdt, dydt]

    t = np.linspace(0, 40, 500)
    state0 = [10, 5]  # 初始猎物和捕食者数量

    solution = integrate.odeint(lotka_volterra, state0, t)

    print("Lotka-Volterra 模型:")
    print(f"  参数: α={alpha}, β={beta}, δ={delta}, γ={gamma}")
    print(f"  初始条件: 猎物={state0[0]}, 捕食者={state0[1]}")

    # === 可视化 ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 时间演化
    ax1 = axes[0]
    ax1.plot(t, solution[:, 0], "b-", lw=2, label="猎物 (x)")
    ax1.plot(t, solution[:, 1], "r-", lw=2, label="捕食者 (y)")
    ax1.set_title("Lotka-Volterra 捕食者-猎物模型")
    ax1.set_xlabel("时间 t")
    ax1.set_ylabel("种群数量")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 相空间轨迹
    ax2 = axes[1]
    ax2.plot(solution[:, 0], solution[:, 1], "g-", lw=1.5)
    ax2.plot(state0[0], state0[1], "ro", markersize=10, label="起点")
    ax2.arrow(
        solution[100, 0],
        solution[100, 1],
        solution[101, 0] - solution[100, 0],
        solution[101, 1] - solution[100, 1],
        head_width=0.5,
        head_length=0.3,
        fc="green",
        ec="green",
    )

    ax2.set_title("相空间轨迹 (Phase Portrait)")
    ax2.set_xlabel("猎物数量 (x)")
    ax2.set_ylabel("捕食者数量 (y)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/scipy/06_lotka.png", dpi=150, bbox_inches="tight")


def demo_all():
    """运行所有演示"""
    import os

    os.makedirs("outputs/scipy", exist_ok=True)

    demo_quad()
    print()
    demo_dblquad()
    print()
    demo_odeint()
    print()
    demo_ode_system()


if __name__ == "__main__":
    demo_all()
