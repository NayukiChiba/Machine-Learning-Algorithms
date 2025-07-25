"""
Pandas 数据可视化
对应文档: ../../docs/visualization/04-pandas-viz.md
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# 添加项目根目录到搜索路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_output_dir


def demo_df_plot(output_dir):
    """演示 DataFrame 绑图"""
    print("=" * 50)
    print("1. DataFrame.plot()")
    print("=" * 50)

    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    df = pd.DataFrame(
        {
            "A": np.cumsum(np.random.randn(30)),
            "B": np.cumsum(np.random.randn(30)),
            "C": np.cumsum(np.random.randn(30)),
        },
        index=dates,
    )

    # 为 area plot 创建全正数据
    df_positive = pd.DataFrame(
        {
            "A": np.abs(np.random.randn(30).cumsum()) + 10,
            "B": np.abs(np.random.randn(30).cumsum()) + 5,
            "C": np.abs(np.random.randn(30).cumsum()) + 3,
        },
        index=dates,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    df.plot(ax=axes[0, 0], title="Line Plot")
    df_positive.plot(kind="area", ax=axes[0, 1], title="Area Plot", alpha=0.5)
    df.iloc[-1].plot(kind="bar", ax=axes[1, 0], title="Bar Plot")
    df.plot(kind="box", ax=axes[1, 1], title="Box Plot")

    plt.tight_layout()
    plt.savefig(output_dir / "viz_04_df_plot.png", dpi=100)
    plt.close()
    print("图表已保存")


def demo_series_plot(output_dir):
    """演示 Series 绑图"""
    print("=" * 50)
    print("2. Series.plot()")
    print("=" * 50)

    s = pd.Series(np.random.randn(100).cumsum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    s.plot(ax=axes[0], title="Line Plot")
    s.plot(kind="hist", bins=20, ax=axes[1], title="Histogram")

    plt.tight_layout()
    plt.savefig(output_dir / "viz_04_series_plot.png", dpi=100)
    plt.close()
    print("图表已保存")


def demo_groupby_plot(output_dir):
    """演示分组绘图"""
    print("=" * 50)
    print("3. GroupBy 绘图")
    print("=" * 50)

    df = pd.DataFrame(
        {"Category": np.repeat(["A", "B", "C"], 20), "Value": np.random.randn(60)}
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    df.groupby("Category")["Value"].mean().plot(
        kind="bar", ax=ax, color=["red", "green", "blue"]
    )
    ax.set_title("Mean Value by Category")
    ax.set_ylabel("Mean")

    plt.tight_layout()
    plt.savefig(output_dir / "viz_04_groupby.png", dpi=100)
    plt.close()
    print("图表已保存")


def demo_all():
    """运行所有演示"""
    output_dir = get_output_dir("visualization")

    demo_df_plot(output_dir)
    print()
    demo_series_plot(output_dir)
    print()
    demo_groupby_plot(output_dir)


if __name__ == "__main__":
    demo_all()
