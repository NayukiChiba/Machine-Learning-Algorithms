"""
数据可视化模块
"""

import os
import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from matplotlib import pyplot as plt
import seaborn as sns
from pandas import DataFrame
from utils.decorate import print_func_info
from config import OUTPUTS_ROOT
from generate_data import generate_data

# 输出目录
LOGREG_OUTPUTS = os.path.join(OUTPUTS_ROOT, "LogisticRegression")

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


@print_func_info
def visualize_data(data: DataFrame):
    """
    数据可视化

    args:
        data(DataFrame): 输入数据
    """
    os.makedirs(LOGREG_OUTPUTS, exist_ok=True)

    # 1. 类别分布
    class_count = data["label"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(
        class_count.index.astype(str),
        class_count.values,
        color=["steelblue", "coral"],
    )
    plt.title("类别分布", fontsize=14, fontweight="bold")
    plt.xlabel("类别")
    plt.ylabel("样本数")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(LOGREG_OUTPUTS, "01_class_distribution.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")

    # 2. 特征分布
    features = list(data.columns[:-1])
    cols = 3
    rows = (len(features) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.8))
    fig.suptitle("特征分布", fontsize=14, fontweight="bold")

    axes = axes.flatten()
    for i, feature in enumerate(features):
        axes[i].hist(data[feature], bins=30, color="skyblue", edgecolor="black")
        axes[i].set_title(feature)
        axes[i].grid(True, alpha=0.3)

    for j in range(len(features), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    filepath = os.path.join(LOGREG_OUTPUTS, "02_feature_distribution.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")

    # 3. 相关性热力图
    plt.figure(figsize=(7, 6))
    corr = data.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("相关性热力图", fontsize=14, fontweight="bold")
    plt.tight_layout()
    filepath = os.path.join(LOGREG_OUTPUTS, "03_correlation_heatmap.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")


if __name__ == "__main__":
    visualize_data(generate_data())
