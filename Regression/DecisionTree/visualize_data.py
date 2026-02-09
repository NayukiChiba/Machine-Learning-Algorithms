"""
数据可视化
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from utils.decorate import print_func_info
from config import OUTPUTS_ROOT

DECISIONTREE_OUTPUTS = os.path.join(OUTPUTS_ROOT, "DecisionTree")

from generate_data import generate_data

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


@print_func_info
def visualize_data(data: DataFrame):
    """
    数据可视化

    args:
        data(DataFrame): 数据
    """
    os.makedirs(DECISIONTREE_OUTPUTS, exist_ok=True)

    # 特征分布
    features = list(data.columns[:-1])

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("特征分布", fontsize=16, fontweight="bold")

    for i, feature in enumerate(features):
        row, col = i // 4, i % 4
        axes[row, col].hist(data[feature], bins=30, color="skyblue")
        axes[row, col].set_title(feature)
        axes[row, col].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(DECISIONTREE_OUTPUTS, "01_feature_distribution.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")

    # 相关性热力图
    plt.figure(figsize=(9, 7))
    corr = data.corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("相关性热力图", fontsize=14, fontweight="bold")
    plt.tight_layout()
    filepath = os.path.join(DECISIONTREE_OUTPUTS, "02_correlation_heatmap.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")

    # 特征 vs 价格 散点
    selected = ["MedInc", "AveRooms", "HouseAge", "Latitude"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("特征 vs 价格", fontsize=16, fontweight="bold")

    for i, feature in enumerate(selected):
        row, col = i // 2, i % 2
        axes[row, col].scatter(
            data[feature], data["price"], alpha=0.4, s=10, color="steelblue"
        )
        axes[row, col].set_xlabel(feature)
        axes[row, col].set_ylabel("价格")
        axes[row, col].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(DECISIONTREE_OUTPUTS, "03_feature_relationship.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")


if __name__ == "__main__":
    visualize_data(generate_data())
