"""
Seaborn 库入门
对应文档: ../../docs/visualization/03-seaborn.md
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# 添加项目根目录到搜索路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_output_dir


def demo_catplot():
    """演示分类图"""
    print("=" * 50)
    print("1. 分类图")
    print("=" * 50)

    tips = sns.load_dataset('tips')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(x='day', y='total_bill', hue='sex', data=tips, ax=axes[0])
    axes[0].set_title('Bar Plot')

    sns.boxplot(x='day', y='total_bill', hue='sex', data=tips, ax=axes[1])
    axes[1].set_title('Box Plot')

    output_dir = get_output_dir("visualization")
    plt.tight_layout()
    plt.savefig(output_dir / 'viz_03_catplot.png', dpi=100)
    plt.close()
    print("图表已保存")


def demo_distplot():
    """演示分布图"""
    print("=" * 50)
    print("2. 分布图")
    print("=" * 50)

    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(data, kde=True, ax=axes[0])
    axes[0].set_title('Histogram with KDE')

    sns.kdeplot(data, fill=True, ax=axes[1])
    axes[1].set_title('KDE Plot')

    output_dir = get_output_dir("visualization")
    plt.tight_layout()
    plt.savefig(output_dir / 'viz_03_distplot.png', dpi=100)
    plt.close()
    print("图表已保存")


def demo_regplot():
    """演示回归图"""
    print("=" * 50)
    print("3. 回归图")
    print("=" * 50)

    tips = sns.load_dataset('tips')

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(x='total_bill', y='tip', data=tips, ax=ax)
    ax.set_title('Regression Plot')

    output_dir = get_output_dir("visualization")
    plt.tight_layout()
    plt.savefig(output_dir / 'viz_03_regplot.png', dpi=100)
    plt.close()
    print("图表已保存")


def demo_heatmap():
    """演示热力图"""
    print("=" * 50)
    print("4. 热力图")
    print("=" * 50)

    np.random.seed(42)
    data = np.random.rand(10, 10)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
    ax.set_title('Heatmap')

    output_dir = get_output_dir("visualization")
    plt.tight_layout()
    plt.savefig(output_dir / 'viz_03_heatmap.png', dpi=100)
    plt.close()
    print("图表已保存")


def demo_pairplot():
    """演示配对图"""
    print("=" * 50)
    print("5. 配对图")
    print("=" * 50)

    iris = sns.load_dataset('iris')

    output_dir = get_output_dir("visualization")
    g = sns.pairplot(iris, hue='species', height=2)
    g.fig.suptitle('Pair Plot', y=1.02)

    plt.savefig(output_dir / 'viz_03_pairplot.png', dpi=100)
    plt.close()
    print("图表已保存")


def demo_all():
    """运行所有演示"""
    sns.set_theme(style='whitegrid')

    demo_catplot()
    print()
    demo_distplot()
    print()
    demo_regplot()
    print()
    demo_heatmap()
    print()
    demo_pairplot()


if __name__ == "__main__":
    demo_all()
