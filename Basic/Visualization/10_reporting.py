"""
专业数据可视化报告
对应文档: ../../docs/visualization/10-reporting.md
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from pathlib import Path

# 添加项目根目录到搜索路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_output_dir


def demo_professional_style(output_dir):
    """演示专业样式"""
    print("=" * 50)
    print("1. 专业样式设置")
    print("=" * 50)

    # 设置专业样式
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x), linewidth=2, label="sin(x)")
    ax.plot(x, np.cos(x), linewidth=2, label="cos(x)")

    ax.set_xlabel("X Axis", fontsize=12)
    ax.set_ylabel("Y Axis", fontsize=12)
    ax.set_title("Professional Style Chart", fontsize=14, fontweight="bold")
    ax.legend(frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(output_dir / "viz_10_professional.png", dpi=150)
    plt.close()
    print("图表已保存")


def demo_multi_panel(output_dir):
    """演示多面板布局"""
    print("=" * 50)
    print("2. 多面板布局")
    print("=" * 50)

    np.random.seed(42)

    fig = plt.figure(figsize=(14, 10))

    # 使用 GridSpec 创建复杂布局
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 大图
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.linspace(0, 10, 100)
    ax1.plot(x, np.sin(x) * np.exp(-x / 5), linewidth=2)
    ax1.set_title("Main Chart", fontweight="bold")

    # 右上小图
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(np.random.randn(500), bins=20, edgecolor="black")
    ax2.set_title("Distribution")

    # 下方三个小图
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(["A", "B", "C"], [3, 5, 2])
    ax3.set_title("Bar Chart")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(np.random.rand(50), np.random.rand(50))
    ax4.set_title("Scatter")

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.pie([30, 40, 30], labels=["X", "Y", "Z"], autopct="%1.0f%%")
    ax5.set_title("Pie")

    plt.savefig(output_dir / "viz_10_multipanel.png", dpi=150)
    plt.close()
    print("图表已保存")


def demo_export():
    """演示导出选项"""
    print("=" * 50)
    print("3. 导出选项")
    print("=" * 50)

    print("保存格式:")
    print("  plt.savefig('fig.png', dpi=300)   # PNG")
    print("  plt.savefig('fig.pdf')            # PDF (矢量)")
    print("  plt.savefig('fig.svg')            # SVG (矢量)")
    print("  plt.savefig('fig.eps')            # EPS (矢量)")
    print()

    print("常用参数:")
    print("  dpi=300          # 分辨率")
    print("  bbox_inches='tight'  # 紧凑边界")
    print("  transparent=True     # 透明背景")
    print("  facecolor='white'    # 背景颜色")


def demo_color_palettes():
    """演示配色方案"""
    print("=" * 50)
    print("4. 配色方案")
    print("=" * 50)

    print("Matplotlib 内置 colormaps:")
    print("  顺序: viridis, plasma, magma, cividis")
    print("  发散: coolwarm, RdBu, seismic")
    print("  定性: Set1, Set2, tab10, Pastel1")
    print()

    print("使用方法:")
    print("  plt.scatter(x, y, c=values, cmap='viridis')")
    print("  colors = plt.cm.Set1(np.linspace(0, 1, 10))")


def demo_all():
    """运行所有演示"""
    output_dir = get_output_dir("visualization")

    demo_professional_style(output_dir)
    print()
    demo_multi_panel(output_dir)
    print()
    demo_export()
    print()
    demo_color_palettes()


if __name__ == "__main__":
    demo_all()
