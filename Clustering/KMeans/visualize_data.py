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

KMEANS_OUTPUTS = os.path.join(OUTPUTS_ROOT, "KMeans")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


@print_func_info
def visualize_data(data: DataFrame):
    """
    数据可视化

    args:
        data(DataFrame): 输入数据
    """
    os.makedirs(KMEANS_OUTPUTS, exist_ok=True)

    # 1. 数据散点图
    plt.figure(figsize=(7, 6))
    plt.scatter(data["x1"], data["x2"], s=25, alpha=0.7, color="steelblue")
    plt.title("原始数据分布", fontsize=14, fontweight="bold")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(KMEANS_OUTPUTS, "01_raw_scatter.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")

    # 2. 相关性热力图
    plt.figure(figsize=(6, 5))
    corr = data.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("特征相关性热力图", fontsize=14, fontweight="bold")
    plt.tight_layout()
    filepath = os.path.join(KMEANS_OUTPUTS, "02_correlation_heatmap.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")


if __name__ == "__main__":
    visualize_data(generate_data())
