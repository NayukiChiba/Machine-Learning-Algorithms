"""
Matplotlib 基础入门
对应文档: ../../docs/visualization/01-matplotlib-basics.md
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# 添加项目根目录到搜索路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_output_dir


def demo_figure_axes():
    """演示 Figure 和 Axes"""
    print("=" * 50)
    print("1. Figure 和 Axes")
    print("=" * 50)

    print("Matplotlib 图表结构:")
    print("  Figure: 整个画布容器")
    print("  Axes: 单个绑图区域")
    print("  Axis: 坐标轴")
    print()

    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x), label="sin(x)")
    ax.plot(x, np.cos(x), label="cos(x)")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Basic Plot")
    ax.legend()
    ax.grid(True)

    output_dir = get_output_dir("visualization")
    plt.tight_layout()
    plt.savefig(output_dir / "viz_01_basic.png", dpi=100)
    plt.close()
    print("图表已保存到 outputs/visualization/viz_01_basic.png")


def demo_line_styles():
    """演示线条样式"""
    print("=" * 50)
    print("2. 线条样式")
    print("=" * 50)

    x = np.linspace(0, 10, 50)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, np.sin(x), "r-", linewidth=2, label="solid")
    ax.plot(x, np.sin(x + 0.5), "g--", linewidth=2, label="dashed")
    ax.plot(x, np.sin(x + 1), "b:", linewidth=2, label="dotted")
    ax.plot(x, np.sin(x + 1.5), "m-.", linewidth=2, label="dashdot")

    ax.legend()
    ax.set_title("Line Styles")

    output_dir = get_output_dir("visualization")
    plt.tight_layout()
    plt.savefig(output_dir / "viz_01_line_styles.png", dpi=100)
    plt.close()
    print("图表已保存")

    print("\n常用线型:")
    print("  '-'  : 实线")
    print("  '--' : 虚线")
    print("  ':'  : 点线")
    print("  '-.' : 点划线")


def demo_markers():
    """演示标记符号"""
    print("=" * 50)
    print("3. 标记符号")
    print("=" * 50)

    x = np.linspace(0, 10, 10)

    fig, ax = plt.subplots(figsize=(10, 6))

    markers = ["o", "s", "^", "D", "v", "p", "*", "x"]
    for i, m in enumerate(markers):
        ax.plot(x, np.sin(x) + i * 0.5, marker=m, label=f"'{m}'", markersize=8)

    ax.legend(ncol=4)
    ax.set_title("Marker Types")

    output_dir = get_output_dir("visualization")
    plt.tight_layout()
    plt.savefig(output_dir / "viz_01_markers.png", dpi=100)
    plt.close()
    print("图表已保存")


def demo_colors():
    """演示颜色设置"""
    print("=" * 50)
    print("4. 颜色设置")
    print("=" * 50)

    print("颜色指定方式:")
    print("  单字符: 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w'")
    print("  名称: 'red', 'green', 'blue', 'orange'")
    print("  十六进制: '#FF5733'")
    print("  RGB元组: (0.1, 0.2, 0.5)")
    print("  Colormap: plt.cm.viridis")


def demo_subplots():
    """演示子图布局"""
    print("=" * 50)
    print("5. 子图布局")
    print("=" * 50)

    x = np.linspace(0, 10, 100)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(x, np.sin(x))
    axes[0, 0].set_title("sin(x)")

    axes[0, 1].plot(x, np.cos(x))
    axes[0, 1].set_title("cos(x)")

    axes[1, 0].plot(x, np.exp(-x / 5) * np.sin(x))
    axes[1, 0].set_title("Damped sine")

    axes[1, 1].plot(x, x**2)
    axes[1, 1].set_title("x²")

    output_dir = get_output_dir("visualization")
    plt.tight_layout()
    plt.savefig(output_dir / "viz_01_subplots.png", dpi=100)
    plt.close()
    print("图表已保存")


def demo_all():
    """运行所有演示"""
    demo_figure_axes()
    print()
    demo_line_styles()
    print()
    demo_markers()
    print()
    demo_colors()
    print()
    demo_subplots()


if __name__ == "__main__":
    demo_all()
