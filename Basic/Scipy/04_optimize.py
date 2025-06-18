"""
优化算法
对应文档: ../../docs/scipy/04-optimize.md
"""

import numpy as np
from scipy import optimize


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
    print(f"f(x) = x² - 4")
    print(f"  brentq 求根 [0, 3]: x = {root:.6f}")
    print(f"  验证 f({root:.6f}) = {f(root):.10f}")
    print()
    
    # 使用 fsolve (牛顿法)
    root = optimize.fsolve(f, x0=1)[0]
    print(f"  fsolve 求根 (x0=1): x = {root:.6f}")
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


def demo_minimize():
    """演示最小化"""
    print("=" * 50)
    print("3. 最小化")
    print("=" * 50)
    
    # 一维最小化
    def f(x):
        return (x - 3)**2 + 2
    
    result = optimize.minimize_scalar(f)
    print("一维最小化 f(x) = (x-3)² + 2:")
    print(f"  最小值点: x = {result.x:.6f}")
    print(f"  最小值: f(x) = {result.fun:.6f}")
    print()
    
    # 多维最小化
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    result = optimize.minimize(rosenbrock, x0=[0, 0], method='BFGS')
    print("Rosenbrock 函数最小化:")
    print(f"  最优解: {result.x}")
    print(f"  最小值: {result.fun:.6f}")
    print(f"  迭代次数: {result.nit}")


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


def demo_all():
    """运行所有演示"""
    demo_curve_fit()
    print()
    demo_root_finding()
    print()
    demo_minimize()
    print()
    demo_linear_programming()


if __name__ == "__main__":
    demo_all()
