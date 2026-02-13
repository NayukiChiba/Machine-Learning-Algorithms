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

NB_OUTPUTS = os.path.join(OUTPUTS_ROOT, "NaiveBayes")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


@print_func_info
def visualize_data(data: DataFrame):
    """
    数据可视化

    args:
        data(DataFrame): 输入数据
    """
    os.makedirs(NB_OUTPUTS, exist_ok=True)

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
    filepath = os.path.join(NB_OUTPUTS, "01_class_distribution.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")

    # 2. 数据散点图
    plt.figure(figsize=(7, 6))
    for label, color in zip([0, 1], ["steelblue", "coral"]):
        part = data[data["label"] == label]
        plt.scatter(
            part["x1"],
            part["x2"],
            s=28,
            alpha=0.75,
            color=color,
            label=f"Class {label}",
        )
    plt.title("二维数据分布", fontsize=14, fontweight="bold")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(NB_OUTPUTS, "02_data_scatter.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")

    # 3. 相关性热力图
    plt.figure(figsize=(6, 5))
    corr = data.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("特征相关性热力图", fontsize=14, fontweight="bold")
    plt.tight_layout()
    filepath = os.path.join(NB_OUTPUTS, "03_correlation_heatmap.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")


if __name__ == "__main__":
    visualize_data(generate_data())
