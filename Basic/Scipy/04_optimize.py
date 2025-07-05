"""
优化算法
对应文档: ../../docs/scipy/04-optimize.md
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def demo_curve_fit():
    """演示曲线拟合"""
    print("=" * 50)
    print("1. 曲线拟合")
    print("=" * 50)
    
    # 定义模型函数
    def model(x, a, b, c):
        return a * x**2 + b * x + c
    
    # 生成带噪声的数据
    np.random.seed(42)
    x_data = np.linspace(0, 10, 50)
    y_data = 2 * x_data**2 + 3 * x_data + 5 + np.random.normal(0, 5, 50)
    
    # 拟合
    params, covariance = optimize.curve_fit(model, x_data, y_data)
    
    print("真实参数: a=2, b=3, c=5")
    print(f"拟合参数: a={params[0]:.4f}, b={params[1]:.4f}, c={params[2]:.4f}")
    print(f"参数标准误: {np.sqrt(np.diag(covariance))}")
    
    # === 可视化 ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 拟合结果
    ax1 = axes[0]
    ax1.scatter(x_data, y_data, c='blue', s=30, alpha=0.6, label='观测数据')
    
    x_smooth = np.linspace(0, 10, 200)
    y_true = 2 * x_smooth**2 + 3 * x_smooth + 5
    y_fit = model(x_smooth, *params)
    
    ax1.plot(x_smooth, y_true, 'g--', lw=2, label='真实曲线: $y = 2x^2 + 3x + 5$')
    ax1.plot(x_smooth, y_fit, 'r-', lw=2, 
             label=f'拟合曲线: $y = {params[0]:.2f}x^2 + {params[1]:.2f}x + {params[2]:.2f}$')
    
    ax1.set_title('曲线拟合 (curve_fit)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 残差分析
    ax2 = axes[1]
    residuals = y_data - model(x_data, *params)
    ax2.scatter(x_data, residuals, c='purple', s=30, alpha=0.7)
    ax2.axhline(0, color='red', linestyle='--', lw=2)
    ax2.fill_between(x_data, -2*np.std(residuals), 2*np.std(residuals), 
                     alpha=0.2, color='gray', label='±2σ 区间')
    ax2.set_title(f'残差分析 (标准差 = {np.std(residuals):.2f})')
    ax2.set_xlabel('x')
    ax2.set_ylabel('残差')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/scipy/04_curve_fit.png', dpi=150, bbox_inches='tight')


def demo_root_finding():
    """演示求根算法"""
    print("=" * 50)
    print("2. 求根算法")
    print("=" * 50)
    
    # 定义函数 f(x) = x^2 - 4
    def f(x):
        return x**2 - 4
    
    # 使用 brentq (区间法)
    root = optimize.brentq(f, 0, 3)
    print(f"f(x) = x^2 - 4")
    print(f"  brentq 求根 [0, 3]: x = {root:.6f}")
    print(f"  验证 f({root:.6f}) = {f(root):.10f}")
    print()
    
    # 使用 fsolve (牛顿法)
    root_fsolve = optimize.fsolve(f, x0=1)[0]
    print(f"  fsolve 求根 (x0=1): x = {root_fsolve:.6f}")
    print()
    
    # 多元方程组
    print("多元方程组求解:")
    def equations(p):
        x, y = p
        return [x + y - 3, x - y - 1]
    
    solution = optimize.fsolve(equations, x0=[0, 0])
    print(f"  x + y = 3")
    print(f"  x - y = 1")
    print(f"  解: x = {solution[0]:.4f}, y = {solution[1]:.4f}")
    
    # === 可视化 ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 一元求根
    ax1 = axes[0]
    x = np.linspace(-3, 3, 200)
    ax1.plot(x, f(x), 'b-', lw=2, label='$f(x) = x^2 - 4$')
    ax1.axhline(0, color='black', linestyle='-', lw=0.5)
    ax1.axvline(0, color='black', linestyle='-', lw=0.5)
    
    # 标记根
    ax1.plot(root, 0, 'ro', markersize=12, label=f'根 x = {root:.2f}', zorder=5)
    ax1.plot(-root, 0, 'ro', markersize=12, zorder=5)
    
    # 标记求根区间
    ax1.axvspan(0, 3, alpha=0.1, color='green', label='brentq 搜索区间 [0, 3]')
    
    ax1.set_title('一元方程求根')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_ylim(-5, 10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 多元方程组
    ax2 = axes[1]
    x_range = np.linspace(-1, 4, 100)
    
    # 绘制两条直线
    ax2.plot(x_range, 3 - x_range, 'b-', lw=2, label='$x + y = 3$')
    ax2.plot(x_range, x_range - 1, 'g-', lw=2, label='$x - y = 1$')
    
    # 标记交点
    ax2.plot(solution[0], solution[1], 'ro', markersize=12, 
             label=f'解 ({solution[0]:.1f}, {solution[1]:.1f})', zorder=5)
    
    ax2.set_title('线性方程组求解')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_xlim(-1, 4)
    ax2.set_ylim(-2, 4)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('outputs/scipy/04_roots.png', dpi=150, bbox_inches='tight')


def demo_minimize():
    """演示最小化"""
    print("=" * 50)
    print("3. 最小化")
    print("=" * 50)
    
    # 一维最小化
    def f(x):
        return (x - 3)**2 + 2
    
    result1 = optimize.minimize_scalar(f)
    print("一维最小化 f(x) = (x-3)^2 + 2:")
    print(f"  最小值点: x = {result1.x:.6f}")
    print(f"  最小值: f(x) = {result1.fun:.6f}")
    print()
    
    # 多维最小化 - Rosenbrock 函数
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    # 记录优化路径
    path = []
    def callback(xk):
        path.append(xk.copy())
    
    x0 = np.array([0.0, 0.0])
    path.append(x0.copy())
    result2 = optimize.minimize(rosenbrock, x0, method='BFGS', callback=callback)
    
    print("Rosenbrock 函数最小化:")
    print(f"  最优解: {result2.x}")
    print(f"  最小值: {result2.fun:.6f}")
    print(f"  迭代次数: {result2.nit}")
    
    # === 可视化 ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 一维函数
    ax1 = axes[0]
    x = np.linspace(-2, 8, 200)
    ax1.plot(x, f(x), 'b-', lw=2, label='$f(x) = (x-3)^2 + 2$')
    ax1.plot(result1.x, result1.fun, 'ro', markersize=12, 
             label=f'最小值点 ({result1.x:.2f}, {result1.fun:.2f})', zorder=5)
    ax1.axhline(result1.fun, color='red', linestyle='--', alpha=0.5)
    ax1.axvline(result1.x, color='red', linestyle='--', alpha=0.5)
    ax1.set_title('一维函数最小化')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rosenbrock 等高线 + 优化路径
    ax2 = axes[1]
    x_range = np.linspace(-0.5, 1.5, 100)
    y_range = np.linspace(-0.5, 1.5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2
    
    # 等高线
    levels = np.logspace(0, 3, 20)
    cs = ax2.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7)
    ax2.clabel(cs, inline=True, fontsize=8, fmt='%.0f')
    
    # 优化路径
    path_arr = np.array(path)
    ax2.plot(path_arr[:, 0], path_arr[:, 1], 'r.-', lw=1.5, markersize=8, 
             label=f'BFGS 优化路径 ({len(path)} 步)')
    ax2.plot(x0[0], x0[1], 'go', markersize=12, label='起点 (0, 0)', zorder=5)
    ax2.plot(1, 1, 'r*', markersize=15, label='全局最优 (1, 1)', zorder=5)
    
    ax2.set_title('Rosenbrock 函数优化')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(-0.5, 1.5)
    
    plt.tight_layout()
    plt.savefig('outputs/scipy/04_minimize.png', dpi=150, bbox_inches='tight')


def demo_linear_programming():
    """演示线性规划"""
    print("=" * 50)
    print("4. 线性规划")
    print("=" * 50)
    
    # 问题: 最大化 z = 2x + 3y
    # 约束: x + y <= 4, x <= 2, y <= 3, x,y >= 0
    
    # linprog 求解最小化问题，所以取负
    c = [-2, -3]  # 目标函数系数
    A_ub = [[1, 1], [1, 0], [0, 1]]  # 不等式约束矩阵
    b_ub = [4, 2, 3]  # 不等式约束右侧
    
    result = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    
    print("线性规划问题:")
    print("  max z = 2x + 3y")
    print("  s.t. x + y ≤ 4, x ≤ 2, y ≤ 3, x,y ≥ 0")
    print(f"  最优解: x = {result.x[0]:.4f}, y = {result.x[1]:.4f}")
    print(f"  最大值: z = {-result.fun:.4f}")
    
    # === 可视化 ===
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 可行域
    x = np.linspace(0, 5, 300)
    
    # 约束线
    y1 = 4 - x       # x + y = 4
    y2 = np.full_like(x, 3)  # y = 3
    
    ax.plot(x, y1, 'b-', lw=2, label='$x + y = 4$')
    ax.axhline(3, color='g', lw=2, label='$y = 3$')
    ax.axvline(2, color='r', lw=2, label='$x = 2$')
    
    # 填充可行域
    vertices = np.array([[0, 0], [2, 0], [2, 2], [1, 3], [0, 3]])
    from matplotlib.patches import Polygon
    polygon = Polygon(vertices, closed=True, facecolor='lightblue', 
                      edgecolor='black', alpha=0.5, label='可行域')
    ax.add_patch(polygon)
    
    # 顶点
    for v in vertices:
        ax.plot(v[0], v[1], 'ko', markersize=8)
        z_val = 2*v[0] + 3*v[1]
        ax.annotate(f'({v[0]}, {v[1]})\nz={z_val:.0f}', v, 
                   textcoords="offset points", xytext=(10, 5), fontsize=9)
    
    # 最优点
    ax.plot(result.x[0], result.x[1], 'r*', markersize=20, 
            label=f'最优解 ({result.x[0]:.0f}, {result.x[1]:.0f}), z={-result.fun:.0f}')
    
    # 目标函数等值线
    z_levels = [3, 6, 9, 11]
    for z in z_levels:
        y_obj = (z - 2*x) / 3
        ax.plot(x, y_obj, 'gray', linestyle='--', alpha=0.5)
        ax.text(0.1, z/3, f'z={z}', fontsize=9, color='gray')
    
    ax.set_xlim(-0.5, 4)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('线性规划: 可行域与最优解')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('outputs/scipy/04_linprog.png', dpi=150, bbox_inches='tight')


def demo_all():
    """运行所有演示"""
    import os
    os.makedirs('outputs/scipy', exist_ok=True)
    
    demo_curve_fit()
    print()
    demo_root_finding()
    print()
    demo_minimize()
    print()
    demo_linear_programming()


if __name__ == "__main__":
    demo_all()
