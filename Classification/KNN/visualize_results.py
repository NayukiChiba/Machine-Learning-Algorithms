"""
结果可视化模块
"""

import os
import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.metrics import confusion_matrix
from utils.decorate import print_func_info
from config import OUTPUTS_ROOT
from train_model import train_model
from generate_data import generate_data
from preprocess_data import preprocess_data
from evaluate_model import evaluate_model

KNN_OUTPUTS = os.path.join(OUTPUTS_ROOT, "KNN")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def _plot_decision_boundary(
    ax,
    model,
    scaler,
    X_data: DataFrame,
    y_data,
    title: str,
):
    """
    绘制决策边界
    """
    x1_min, x1_max = X_data["x1"].min() - 0.6, X_data["x1"].max() + 0.6
    x2_min, x2_max = X_data["x2"].min() - 0.6, X_data["x2"].max() + 0.6

    xx, yy = np.meshgrid(
        np.linspace(x1_min, x1_max, 300),
        np.linspace(x2_min, x2_max, 300),
    )

    grid = DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=["x1", "x2"])
    grid_scaled = scaler.transform(grid)
    zz = model.predict(grid_scaled).reshape(xx.shape)

    ax.contourf(xx, yy, zz, alpha=0.25, cmap="coolwarm")

    plot_data = X_data.copy()
    plot_data["label"] = y_data.values
    for label, color in zip([0, 1], ["steelblue", "coral"]):
        part = plot_data[plot_data["label"] == label]
        ax.scatter(
            part["x1"],
            part["x2"],
            s=25,
            alpha=0.8,
            color=color,
            label=f"Class {label}",
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True, alpha=0.3)


@print_func_info
def visualize_results(
    model,
    scaler,
    X_train_orig,
    X_test_orig,
    y_train,
    y_test,
    y_train_pred,
    y_test_pred,
):
    """
    可视化模型结果

    args:
        model: 训练好的模型
        scaler: 标准化器
        X_train_orig, X_test_orig: 原始特征（未标准化）
        y_train, y_test: 真实标签
        y_train_pred, y_test_pred: 预测标签
    """
    os.makedirs(KNN_OUTPUTS, exist_ok=True)

    # 1. 混淆矩阵
    train_cm = confusion_matrix(y_train, y_train_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("混淆矩阵", fontsize=16, fontweight="bold")

    sns.heatmap(train_cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
    axes[0].set_title("训练集")
    axes[0].set_xlabel("预测标签")
    axes[0].set_ylabel("真实标签")

    sns.heatmap(test_cm, annot=True, fmt="d", cmap="Oranges", cbar=False, ax=axes[1])
    axes[1].set_title("测试集")
    axes[1].set_xlabel("预测标签")
    axes[1].set_ylabel("真实标签")

    plt.tight_layout()
    filepath = os.path.join(KNN_OUTPUTS, "04_confusion_matrix.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")

    # 2. 决策边界
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("决策边界", fontsize=16, fontweight="bold")

    _plot_decision_boundary(axes[0], model, scaler, X_train_orig, y_train, "训练集")
    _plot_decision_boundary(axes[1], model, scaler, X_test_orig, y_test, "测试集")

    handles, labels = axes[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[1].legend(unique.values(), unique.keys(), loc="best")

    plt.tight_layout()
    filepath = os.path.join(KNN_OUTPUTS, "05_decision_boundary.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, X_train_orig, X_test_orig = (
        preprocess_data(generate_data())
    )
    model = train_model(X_train, y_train)
    y_train_pred, y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
    visualize_results(
        model,
        scaler,
        X_train_orig,
        X_test_orig,
        y_train,
        y_test,
        y_train_pred,
        y_test_pred,
    )
