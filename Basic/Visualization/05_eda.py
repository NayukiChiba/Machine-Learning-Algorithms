"""
探索性数据分析可视化
对应文档: ../../docs/foundations/visualization/05-eda.md
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from . import output_dir as get_output_dir


# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def _apply_plot_fonts():
    """重新应用中文字体设置，避免被 Seaborn 主题覆盖。"""
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def distribution_analysis(output_dir):
    """演示分布分析"""
    print("=" * 50)
    print("1. 分布分析")
    print("=" * 50)

    np.random.seed(42)
    df = pd.DataFrame(
        {
            "age": np.random.normal(35, 10, 200).astype(int),
            "income": np.random.exponential(50000, 200),
            "score": np.random.beta(2, 5, 200) * 100,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, col in zip(axes, df.columns):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.axvline(df[col].mean(), color="red", linestyle="--", label="Mean")
        ax.axvline(df[col].median(), color="green", linestyle="--", label="Median")
        ax.set_title(f"{col} Distribution")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "05_distribution.png", dpi=100)
    plt.close()
    print("图表已保存")


def correlation_analysis(output_dir):
    """演示相关性分析"""
    print("=" * 50)
    print("2. 相关性分析")
    print("=" * 50)

    np.random.seed(42)
    n = 100
    x = np.random.randn(n)
    df = pd.DataFrame(
        {
            "x": x,
            "y_strong": x + np.random.randn(n) * 0.3,
            "y_weak": x + np.random.randn(n) * 2,
            "y_none": np.random.randn(n),
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 相关矩阵热力图
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=axes[0])
    axes[0].set_title("Correlation Matrix")

    # 散点图矩阵
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "05_correlation.png", dpi=100)
    plt.close()
    print("图表已保存")


def categorical_analysis(output_dir):
    """演示分类变量分析"""
    print("=" * 50)
    print("3. 分类变量分析")
    print("=" * 50)

    np.random.seed(42)
    df = pd.DataFrame(
        {
            "category": np.random.choice(["A", "B", "C", "D"], 200),
            "value": np.random.randn(200),
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 频数统计
    df["category"].value_counts().plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title("Category Counts")

    # 分类箱线图
    sns.boxplot(x="category", y="value", data=df, ax=axes[1])
    axes[1].set_title("Value by Category")

    plt.tight_layout()
    plt.savefig(output_dir / "05_categorical.png", dpi=100)
    plt.close()
    print("图表已保存")


def run():
    """运行所有演示"""
    output_dir = get_output_dir()

    sns.set_theme(style="whitegrid")
    _apply_plot_fonts()

    distribution_analysis(output_dir)
    print()
    correlation_analysis(output_dir)
    print()
    categorical_analysis(output_dir)


if __name__ == "__main__":
    run()
