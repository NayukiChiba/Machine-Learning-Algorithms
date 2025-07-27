"""
Matplotlib 常用图表类型
对应文档: ../../docs/visualization/02-matplotlib-charts.md
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# 添加项目根目录到搜索路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_output_dir


def demo_bar():
    """演示柱状图"""
    print("=" * 50)
    print("1. 柱状图")
    print("=" * 50)

    categories = ["A", "B", "C", "D", "E"]
    values = [23, 45, 56, 78, 32]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 垂直柱状图
    axes[0].bar(categories, values, color="steelblue", edgecolor="black")
    axes[0].set_title("Vertical Bar Chart")

    # 水平柱状图
    axes[1].barh(categories, values, color="coral", edgecolor="black")
    axes[1].set_title("Horizontal Bar Chart")

    output_dir = get_output_dir("visualization")
    plt.tight_layout()
    plt.savefig(output_dir / "viz_02_bar.png", dpi=100)
    plt.close()
    print("图表已保存")


def demo_scatter():
    """演示散点图"""
    print("=" * 50)
    print("2. 散点图")
    print("=" * 50)

    np.random.seed(42)
    x = np.random.randn(100)
    y = x + np.random.randn(100) * 0.5
    colors = np.random.rand(100)
    sizes = np.abs(np.random.randn(100)) * 200

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap="viridis")
    plt.colorbar(scatter, ax=ax, label="Color Value")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Scatter Plot with Color and Size")

    output_dir = get_output_dir("visualization")
    plt.tight_layout()
    plt.savefig(output_dir / "viz_02_scatter.png", dpi=100)
    plt.close()
    print("图表已保存")


def demo_histogram():
    """演示直方图"""
    print("=" * 50)
    print("3. 直方图")
    print("=" * 50)

    np.random.seed(42)
    data = np.random.randn(1000)

    fig, ax = plt.subplots(figsize=(8, 6))
    n, bins, patches = ax.hist(data, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(
        data.mean(), color="red", linestyle="--", label=f"Mean: {data.mean():.2f}"
    )
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram")
    ax.legend()

    output_dir = get_output_dir("visualization")
    plt.tight_layout()
    plt.savefig(output_dir / "viz_02_histogram.png", dpi=100)
    plt.close()
    print("图表已保存")


def demo_pie():
    """演示饼图"""
    print("=" * 50)
    print("4. 饼图")
    print("=" * 50)

    labels = ["Product A", "Product B", "Product C", "Product D"]
    sizes = [35, 30, 20, 15]
    colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]
    explode = (0.05, 0, 0, 0)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
    )
    ax.set_title("Market Share")

    output_dir = get_output_dir("visualization")
    plt.tight_layout()
    plt.savefig(output_dir / "viz_02_pie.png", dpi=100)
    plt.close()
    print("图表已保存")


def demo_boxplot():
    """演示箱线图"""
    print("=" * 50)
    print("5. 箱线图")
    print("=" * 50)

    np.random.seed(42)
    data = [np.random.normal(0, std, 100) for std in range(1, 5)]

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(data, patch_artist=True)

    colors = ["lightblue", "lightgreen", "lightyellow", "lightcoral"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_xticklabels(["σ=1", "σ=2", "σ=3", "σ=4"])
    ax.set_ylabel("Value")
    ax.set_title("Box Plot")

    output_dir = get_output_dir("visualization")
    plt.tight_layout()
    plt.savefig(output_dir / "viz_02_boxplot.png", dpi=100)
    plt.close()
    print("图表已保存")


def demo_all():
    """运行所有演示"""
    demo_bar()
    print()
    demo_scatter()
    print()
    demo_histogram()
    print()
    demo_pie()
    print()
    demo_boxplot()


if __name__ == "__main__":
    demo_all()
