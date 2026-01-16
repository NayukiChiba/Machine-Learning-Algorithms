"""
数值积分
对应文档: ../../docs/scipy/06-integrate.md
"""

import numpy as np
from scipy import integrate


def demo_quad():
    """演示定积分"""
    print("=" * 50)
    print("1. 定积分")
    print("=" * 50)
    
    # 积分 ∫x² dx, 从0到1
    result, error = integrate.quad(lambda x: x**2, 0, 1)
    print(f"∫₀¹ x² dx = {result:.6f} (误差: {error:.2e})")
    print(f"解析解: 1/3 = {1/3:.6f}")
    print()
    
    # 积分 ∫sin(x) dx, 从0到π
    result, error = integrate.quad(np.sin, 0, np.pi)
    print(f"∫₀π sin(x) dx = {result:.6f} (误差: {error:.2e})")
    print(f"解析解: 2")
    print()
    
    # 无穷积分 ∫e^(-x²) dx, 从-∞到+∞
    result, error = integrate.quad(lambda x: np.exp(-x**2), -np.inf, np.inf)
    print(f"∫_{-inf}^{+inf} e^(-x²) dx = {result:.6f}")
    print(f"解析解: √π = {np.sqrt(np.pi):.6f}")


def demo_dblquad():
    """演示二重积分"""
    print("=" * 50)
    print("2. 二重积分")
    print("=" * 50)
    
    # 积分 ∫∫xy dA, 在 [0,1]×[0,2] 上
    result, error = integrate.dblquad(
        lambda y, x: x * y,  # 被积函数
        0, 1,                 # x 的范围
        lambda x: 0,          # y 的下限
        lambda x: 2           # y 的上限
    )
    print(f"∫₀¹∫₀² xy dy dx = {result:.6f}")
    print(f"解析解: 1")
    print()
    
    # 圆形区域积分
    result, error = integrate.dblquad(
        lambda y, x: 1,
        -1, 1,
        lambda x: -np.sqrt(1 - x**2),
        lambda x: np.sqrt(1 - x**2)
    )
    print(f"单位圆面积 = {result:.6f}")
    print(f"解析解: π = {np.pi:.6f}")


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
    
    t = np.linspace(0, 20, 5)
    state0 = [10, 5]  # 初始猎物和捕食者数量
    
    solution = integrate.odeint(lotka_volterra, state0, t)
    
    print("Lotka-Volterra 模型:")
    print("t\t猎物\t捕食者")
    for ti, (x, y) in zip(t, solution):
        print(f"{ti:.1f}\t{x:.2f}\t{y:.2f}")


def demo_all():
    """运行所有演示"""
    demo_quad()
    print()
    demo_dblquad()
    print()
    demo_odeint()
    print()
    demo_ode_system()


if __name__ == "__main__":
    demo_all()
