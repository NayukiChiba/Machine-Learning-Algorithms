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

DBSCAN_OUTPUTS = os.path.join(OUTPUTS_ROOT, "DBSCAN")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


@print_func_info
def visualize_results(model, scaler, X_orig, labels):
    """
    可视化聚类结果

    args:
        model: 训练好的模型
        scaler: 标准化器
        X_orig: 原始特征
        labels: 聚类标签（-1 为噪声）
    """
    os.makedirs(DBSCAN_OUTPUTS, exist_ok=True)

    # 聚类结果散点图
    plt.figure(figsize=(7, 6))

    # 噪声点
    noise_mask = labels == -1
    if noise_mask.any():
        plt.scatter(
            X_orig.loc[noise_mask, "x1"],
            X_orig.loc[noise_mask, "x2"],
            c="gray",
            s=25,
            alpha=0.6,
            marker="x",
            label="噪声",
        )

    # 正常簇
    unique_labels = sorted(set(labels) - {-1})
    for k in unique_labels:
        cluster_mask = labels == k
        plt.scatter(
            X_orig.loc[cluster_mask, "x1"],
            X_orig.loc[cluster_mask, "x2"],
            s=30,
            alpha=0.8,
            label=f"簇 {k}",
        )

    plt.title("DBSCAN 聚类结果", fontsize=14, fontweight="bold")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(DBSCAN_OUTPUTS, "03_cluster_result.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")


if __name__ == "__main__":
    X_scaled, scaler, X_orig = preprocess_data(generate_data())
    model = train_model(X_scaled)
    labels = evaluate_model(model, X_scaled)
    visualize_results(model, scaler, X_orig, labels)
