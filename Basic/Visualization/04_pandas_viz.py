"""
Pandas 数据可视化
对应文档: ../../docs/foundations/visualization/04-pandas-viz.md
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import output_dir as get_output_dir


# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def df_plot(output_dir):
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
    plt.savefig(output_dir / "04_df_plot.png", dpi=100)
    plt.close()
    print("图表已保存")


def series_plot(output_dir):
    """演示 Series 绑图"""
    print("=" * 50)
    print("2. Series.plot()")
    print("=" * 50)

    s = pd.Series(np.random.randn(100).cumsum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    s.plot(ax=axes[0], title="Line Plot")
    s.plot(kind="hist", bins=20, ax=axes[1], title="Histogram")

    plt.tight_layout()
    plt.savefig(output_dir / "04_series_plot.png", dpi=100)
    plt.close()
    print("图表已保存")


def groupby_plot(output_dir):
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
    plt.savefig(output_dir / "04_groupby.png", dpi=100)
    plt.close()
    print("图表已保存")


def run():
    """运行所有演示"""
    output_dir = get_output_dir()

    df_plot(output_dir)
    print()
    series_plot(output_dir)
    print()
    groupby_plot(output_dir)


if __name__ == "__main__":
    run()
