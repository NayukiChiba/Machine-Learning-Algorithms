"""
SciPy 基础入门
对应文档: ../../docs/scipy/01-basics.md
"""

import numpy as np


def demo_scipy_modules():
    """演示 SciPy 模块结构"""
    print("=" * 50)
    print("1. SciPy 模块结构")
    print("=" * 50)

    print("SciPy 主要模块:")
    print("  scipy.constants  - 物理和数学常数")
    print("  scipy.special    - 特殊函数")
    print("  scipy.integrate  - 数值积分")
    print("  scipy.optimize   - 优化算法")
    print("  scipy.interpolate- 插值")
    print("  scipy.linalg     - 线性代数")
    print("  scipy.signal     - 信号处理")
    print("  scipy.sparse     - 稀疏矩阵")
    print("  scipy.stats      - 统计分布")
    print("  scipy.spatial    - 空间数据")


def demo_constants():
    """演示物理常数"""
    print("=" * 50)
    print("2. 物理常数")
    print("=" * 50)

    from scipy import constants

    print(f"圆周率 π: {constants.pi}")
    print(f"光速 c: {constants.c} m/s")
    print(f"普朗克常数 h: {constants.h} J·s")
    print(f"玻尔兹曼常数 k: {constants.k} J/K")
    print(f"阿伏伽德罗常数 N_A: {constants.N_A}")
    print(f"电子电荷 e: {constants.e} C")
    print()

    # 单位转换
    print("单位转换示例:")
    print(f"  1 英里 = {constants.mile} 米")
    print(f"  1 英寸 = {constants.inch} 米")
    print(f"  1 磅 = {constants.pound} 千克")


def demo_special_functions():
    """演示特殊函数"""
    print("=" * 50)
    print("3. 特殊函数")
    print("=" * 50)

    from scipy import special

    # 阶乘和组合数
    print("阶乘和组合数:")
    print(f"  5! = {special.factorial(5)}")
    print(f"  C(10, 3) = {special.comb(10, 3)}")
    print(f"  P(10, 3) = {special.perm(10, 3)}")
    print()

    # 伽马函数
    print("伽马函数 Γ(x):")
    print(f"  Γ(5) = 4! = {special.gamma(5)}")
    print(f"  Γ(0.5) = √π = {special.gamma(0.5)}")
    print()

    # 贝塞尔函数
    print("贝塞尔函数:")
    print(f"  J_0(1) = {special.jv(0, 1):.6f}")
    print(f"  J_1(1) = {special.jv(1, 1):.6f}")


def demo_version():
    """演示版本信息"""
    print("=" * 50)
    print("4. 版本信息")
    print("=" * 50)

    import scipy

    print(f"SciPy 版本: {scipy.__version__}")
    print(f"NumPy 版本: {np.__version__}")


def demo_all():
    """运行所有演示"""
    demo_scipy_modules()
    print()
    demo_constants()
    print()
    demo_special_functions()
    print()
    demo_version()


if __name__ == "__main__":
    demo_all()
