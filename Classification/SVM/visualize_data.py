"""
数据可视化
"""

import os
import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于直接导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from matplotlib import pyplot as plt
import seaborn as sns
from pandas import DataFrame
from utils.decorate import print_func_info
from config import OUTPUTS_ROOT
from generate_data import generate_data

# 输出目录
SVC_OUTPUTS = os.path.join(OUTPUTS_ROOT, "SVC")

# 中文字体设置（避免中文乱码）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


@print_func_info
def visualize_data(data: DataFrame):
    """
    数据可视化

    args:
        data(DataFrame): 数据
    """
    # 创建输出目录
    os.makedirs(SVC_OUTPUTS, exist_ok=True)

    # 1. 类别分布柱状图
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
    filepath = os.path.join(SVC_OUTPUTS, "01_class_distribution.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")

    # 2. 特征散点图（展示双月牙结构）
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
    plt.title("双月牙数据分布", fontsize=14, fontweight="bold")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(SVC_OUTPUTS, "02_data_scatter.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")

    # 3. 相关性热力图
    plt.figure(figsize=(6, 5))
    corr = data.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("特征相关性热力图", fontsize=14, fontweight="bold")
    plt.tight_layout()
    filepath = os.path.join(SVC_OUTPUTS, "03_correlation_heatmap.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")


if __name__ == "__main__":
    # 模块自测：生成数据并进行可视化
    visualize_data(generate_data())
