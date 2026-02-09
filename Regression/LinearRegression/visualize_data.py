"""
可视化一下数据
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


import os
from pandas import DataFrame
from utils.decorate import print_func_info
import seaborn as sns
from matplotlib import pyplot as plt
from config import OUTPUTS_ROOT
from generate_data import generate_data


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
LR_OUTPUTS = os.path.join(OUTPUTS_ROOT, "LinearRegression")


@print_func_info
def visualize_data(data: DataFrame):
    """
    数据可视化

    args:
        data(DataFrame): 数据
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("数据探索可视化", fontsize=16, fontweight="bold")

    # 特征分布
    features = list(data.columns[:-1])
    for i, feature in enumerate(features):
        row, col = i // 2, i % 2
        # 创建直方图
        axes[row, col].hist(
            data[feature], bins=30, color="skyblue", edgecolor="black", alpha=0.7
        )
        axes[row, col].set_xlabel("频数", fontsize=12)
        axes[row, col].set_ylabel(f"{feature}分布", fontsize=12, fontweight="bold")
        axes[row, col].grid(True, alpha=0.3)

    # 目标变量分布
    axes[1, 1].hist(data["价格"], bins=30, color="coral", edgecolor="black", alpha=0.7)
    axes[1, 1].set_xlabel("价格(万元)", fontsize=12)
    axes[1, 1].set_ylabel("频数", fontsize=12)
    axes[1, 1].set_title("价格分布", fontsize=12, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(LR_OUTPUTS, "01_data_distribution.png")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图表在{filepath}中")

    # 相关性热力图
    plt.figure(figsize=(8, 6))
    correlation_matrix = data.corr()
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("特征相关性热力图", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    filepath = os.path.join(LR_OUTPUTS, "02_correlation_heatmap.png")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图表在{filepath}中")

    # 散点图矩阵
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("特征与目标变量关系", fontsize=16, fontweight="bold")

    for i, feature in enumerate(features):
        axes[i].scatter(data[feature], data["价格"], alpha=0.6, s=30, color="steelblue")
        axes[i].set_xlabel(feature, fontsize=12)
        axes[i].set_ylabel("价格 (万元)", fontsize=12)
        axes[i].set_title(f"{feature} vs 价格", fontsize=12, fontweight="bold")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(LR_OUTPUTS, "03_Feature_Relationship.png")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"图像已保存在{filepath}中")


if __name__ == "__main__":
    visualize_data(generate_data())
