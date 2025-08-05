"""
数据可视化
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import math
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from utils.decorate import print_func_info
from config import OUTPUTS_ROOT

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
OUTPUT_DIR = os.path.join(OUTPUTS_ROOT, "Regularization")


@print_func_info
def visualize_data(data: DataFrame):
    """
    数据可视化

    args:
        data(DataFrame): 数据集
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 特征分布
    features = list(data.columns[:-1])
    cols = 4
    rows = math.ceil(len(features) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.5))
    fig.suptitle("特征分布", fontsize=14, fontweight="bold")

    axes = axes.flatten()
    for i, feature in enumerate(features):
        axes[i].hist(data[feature], bins=30, color="skyblue", edgecolor="black")
        axes[i].set_title(feature)
        axes[i].grid(True, alpha=0.3)

    for j in range(len(features), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "01_feature_distribution.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")

    # 2. 相关性热力图
    plt.figure(figsize=(10, 8))
    corr = data.corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("相关性热力图", fontsize=12, fontweight="bold")
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "02_correlation_heatmap.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")

    # 3. 特征 vs 目标变量
    selected = [f for f in ["bmi", "bp", "s5", "s1"] if f in features]
    if len(selected) < 4:
        selected = features[:4]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("特征 vs 目标变量 price", fontsize=12, fontweight="bold")

    for i, feature in enumerate(selected):
        r, c = i // 2, i % 2
        axes[r, c].scatter(data[feature], data["price"], alpha=0.5, s=15)
        axes[r, c].set_xlabel(feature)
        axes[r, c].set_ylabel("price")
        axes[r, c].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "03_feature_vs_target.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")


if __name__ == "__main__":
    from generate_data import generate_data

    df = generate_data()
    visualize_data(df)
