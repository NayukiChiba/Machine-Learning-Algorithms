"""
Pandas 数据可视化
对应文档: ../../docs/pandas/08-visualization.md
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# 添加项目根目录到搜索路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_output_dir


def demo_line_plot():
    """演示折线图"""
    print("=" * 50)
    print("1. 折线图")
    print("=" * 50)

    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    df = pd.DataFrame(
        {
            "Sales": np.cumsum(np.random.randn(30)) + 100,
            "Profit": np.cumsum(np.random.randn(30)) + 50,
        },
        index=dates,
    )

    print("数据 (前5行):")
    print(df.head())

    output_dir = get_output_dir("pandas")
    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(ax=ax, title="Sales and Profit Over Time")
    plt.tight_layout()
    plt.savefig(output_dir / "pandas_line_plot.png", dpi=100)
    plt.close()
    print("折线图已保存")


def demo_bar_plot():
    """演示柱状图"""
    print("=" * 50)
    print("2. 柱状图")
    print("=" * 50)

    df = pd.DataFrame(
        {"Product": ["A", "B", "C", "D"], "Sales": [150, 200, 180, 220]}
    ).set_index("Product")

    output_dir = get_output_dir("pandas")
    fig, ax = plt.subplots(figsize=(8, 5))
    df.plot(kind="bar", ax=ax, title="Product Sales")
    plt.tight_layout()
    plt.savefig(output_dir / "pandas_bar_plot.png", dpi=100)
    plt.close()
    print("柱状图已保存")


def demo_histogram():
    """演示直方图"""
    print("=" * 50)
    print("3. 直方图")
    print("=" * 50)

    data = pd.Series(np.random.randn(1000))
    output_dir = get_output_dir("pandas")
    fig, ax = plt.subplots(figsize=(8, 5))
    data.plot(kind="hist", bins=30, ax=ax, title="Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "pandas_histogram.png", dpi=100)
    plt.close()
    print("直方图已保存")


def demo_scatter():
    """演示散点图"""
    print("=" * 50)
    print("4. 散点图")
    print("=" * 50)

    df = pd.DataFrame({"x": np.random.randn(50), "y": np.random.randn(50)})
    output_dir = get_output_dir("pandas")
    fig, ax = plt.subplots(figsize=(8, 6))
    df.plot(kind="scatter", x="x", y="y", ax=ax, title="Scatter Plot")
    plt.tight_layout()
    plt.savefig(output_dir / "pandas_scatter.png", dpi=100)
    plt.close()
    print("散点图已保存")


def demo_boxplot():
    """演示箱线图"""
    print("=" * 50)
    print("5. 箱线图")
    print("=" * 50)

    df = pd.DataFrame(
        {"A": np.random.normal(50, 10, 100), "B": np.random.normal(55, 15, 100)}
    )
    output_dir = get_output_dir("pandas")
    fig, ax = plt.subplots(figsize=(8, 6))
    df.plot(kind="box", ax=ax, title="Box Plot")
    plt.tight_layout()
    plt.savefig(output_dir / "pandas_boxplot.png", dpi=100)
    plt.close()
    print("箱线图已保存")


def demo_pie():
    """演示饼图"""
    print("=" * 50)
    print("6. 饼图")
    print("=" * 50)

    data = pd.Series([35, 25, 20, 15, 5], index=["A", "B", "C", "D", "E"])
    output_dir = get_output_dir("pandas")
    fig, ax = plt.subplots(figsize=(8, 8))
    data.plot(kind="pie", ax=ax, autopct="%1.1f%%", title="Pie Chart")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "pandas_pie.png", dpi=100)
    plt.close()
    print("饼图已保存")


def demo_all():
    """运行所有演示"""
    demo_line_plot()
    demo_bar_plot()
    demo_histogram()
    demo_scatter()
    demo_boxplot()
    demo_pie()


if __name__ == "__main__":
    demo_all()
