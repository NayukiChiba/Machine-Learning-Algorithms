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

KMEANS_OUTPUTS = os.path.join(OUTPUTS_ROOT, "KMeans")

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
        labels: 聚类标签
    """
    os.makedirs(KMEANS_OUTPUTS, exist_ok=True)

    # 聚类结果散点图
    plt.figure(figsize=(7, 6))
    plt.scatter(
        X_orig["x1"],
        X_orig["x2"],
        c=labels,
        cmap="tab10",
        s=30,
        alpha=0.8,
    )
    centers = scaler.inverse_transform(model.cluster_centers_)
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        c="black",
        s=120,
        marker="X",
        label="聚类中心",
    )
    plt.title("KMeans 聚类结果", fontsize=14, fontweight="bold")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(KMEANS_OUTPUTS, "03_cluster_result.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")


if __name__ == "__main__":
    X_scaled, scaler, X_orig = preprocess_data(generate_data())
    model = train_model(X_scaled)
    labels = evaluate_model(model, X_scaled)
    visualize_results(model, scaler, X_orig, labels)
