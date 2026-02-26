"""
结果可视化模块
"""

import os
import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from matplotlib import pyplot as plt
from utils.decorate import print_func_info
from config import OUTPUTS_ROOT
from train_model import train_model
from generate_data import generate_data
from preprocess_data import preprocess_data
from evaluate_model import evaluate_model

PCA_OUTPUTS = os.path.join(OUTPUTS_ROOT, "PCA")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


@print_func_info
def visualize_results(X_pca, y, explained_variance_ratio):
    """
    可视化 PCA 结果

    args:
        X_pca: PCA 降维后的数据
        y: 标签（仅用于可视化）
        explained_variance_ratio: 解释方差比
    """
    os.makedirs(PCA_OUTPUTS, exist_ok=True)

    # 1. PCA 二维散点图
    plt.figure(figsize=(7, 6))
    for label, color in zip([0, 1, 2], ["steelblue", "coral", "seagreen"]):
        idx = y == label
        plt.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            s=30,
            alpha=0.75,
            color=color,
            label=f"Class {label}",
        )
    plt.title("PCA 降维结果（2D）", fontsize=14, fontweight="bold")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(PCA_OUTPUTS, "04_pca_scatter.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")

    # 2. 解释方差比例
    plt.figure(figsize=(6, 4))
    pcs = [f"PC{i + 1}" for i in range(len(explained_variance_ratio))]
    plt.bar(pcs, explained_variance_ratio, color="steelblue")
    plt.title("解释方差比", fontsize=14, fontweight="bold")
    plt.ylabel("比例")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(PCA_OUTPUTS, "05_explained_variance.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")


if __name__ == "__main__":
    X_scaled, scaler, X, y = preprocess_data(generate_data())
    model = train_model(X_scaled, n_components=2)
    X_pca = evaluate_model(model, X_scaled)
    visualize_results(X_pca, y.values, model.explained_variance_ratio_)
