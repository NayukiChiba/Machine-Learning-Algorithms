"""
结果可视化模块
"""

from pathlib import Path
import sys

# 加入项目根目录，便于导入公共模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import numpy as np
import matplotlib.pyplot as plt
from config import OUTPUTS_ROOT
from utils.decorate import print_func_info

# 设置输出目录
OUTPUT_DIR = os.path.join(OUTPUTS_ROOT, "SVR")


@print_func_info
def visualize_results(results: dict, y_test):
    """
    绘制模型结果对比图

    args:
        results(dict): 每个模型的结果字典
        y_test: 测试集真实值
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_names = list(results.keys())
    n = len(model_names)

    # 1) 真实值 vs 预测值
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    y_min = min(
        float(np.min(y_test)),
        *[float(np.min(results[m]["y_test_pred"])) for m in model_names],
    )
    y_max = max(
        float(np.max(y_test)),
        *[float(np.max(results[m]["y_test_pred"])) for m in model_names],
    )

    for ax, name in zip(axes, model_names):
        y_pred = results[name]["y_test_pred"]
        r2 = results[name]["metrics"]["test_r2"]
        ax.scatter(y_test, y_pred, alpha=0.6, s=20)
        ax.plot([y_min, y_max], [y_min, y_max], "r--", lw=2)
        ax.set_title(f"{name} | 测试 R2={r2:.3f}")
        ax.set_xlabel("真实值")
        ax.set_ylabel("预测值")
        ax.grid(alpha=0.2)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "04_actual_vs_pred.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")

    # 2) 指标对比
    test_r2 = [results[m]["metrics"]["test_r2"] for m in model_names]
    test_rmse = [results[m]["metrics"]["test_rmse"] for m in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(model_names, test_r2, color=["#4C72B0", "#55A868", "#C44E52"])
    axes[0].set_title("测试集 R2 对比")
    axes[0].set_ylabel("R2")
    axes[0].grid(alpha=0.2, axis="y")

    axes[1].bar(model_names, test_rmse, color=["#4C72B0", "#55A868", "#C44E52"])
    axes[1].set_title("测试集 RMSE 对比")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(alpha=0.2, axis="y")

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "05_metrics_comparison.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")

    # 3) 支持向量数量对比
    sv_counts = [results[m]["support_vectors"] for m in model_names]
    plt.figure(figsize=(8, 4))
    plt.bar(model_names, sv_counts, color="#8172B3")
    plt.title("支持向量数量对比")
    plt.ylabel("数量")
    plt.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "06_support_vectors.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")
