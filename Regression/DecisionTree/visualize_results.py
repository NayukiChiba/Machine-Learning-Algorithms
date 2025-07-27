"""
结果可视化
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import plot_tree
from utils.decorate import print_func_info
from config import OUTPUTS_ROOT

DECISIONTREE_OUTPUTS = os.path.join(OUTPUTS_ROOT, "DecisionTree")

from train_model import train_model
from generate_data import generate_data
from preprocess_data import preprocess_data
from evaluate_model import evaluate_model


# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


@print_func_info
def visualize_results(y_train, y_train_pred, y_test, y_test_pred, model, feature_names):
    """
    可视化预测结果与决策树结构

    args:
        y_train, y_train_pred: 训练集真实/预测
        y_test, y_test_pred: 测试集真实/预测
        model: 训练好的模型
        feature_names: 特征名列表
    """
    os.makedirs(DECISIONTREE_OUTPUTS, exist_ok=True)

    # 预测值 vs 真实值
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("预测值 vs 真实值", fontsize=16, fontweight="bold")

    axes[0].scatter(y_train, y_train_pred, alpha=0.6, s=20, color="steelblue")
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--")
    axes[0].set_title("训练集")
    axes[0].set_xlabel("真实值")
    axes[0].set_ylabel("预测值")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y_test, y_test_pred, alpha=0.6, s=20, color="coral")
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    axes[1].set_title("测试集")
    axes[1].set_xlabel("真实值")
    axes[1].set_ylabel("预测值")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(DECISIONTREE_OUTPUTS, "04_prediction_effect.png")
    plt.savefig(filepath, dpi=1000, bbox_inches="tight")
    print(f"保存图像: {filepath}")

    # 残差分析
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("残差分析", fontsize=16, fontweight="bold")

    axes[0, 0].hist(
        train_residuals, bins=30, color="steelblue", edgecolor="black", alpha=0.7
    )
    axes[0, 0].axvline(0, color="red", linestyle="--")
    axes[0, 0].set_title("训练集残差分布")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(
        test_residuals, bins=30, color="coral", edgecolor="black", alpha=0.7
    )
    axes[0, 1].axvline(0, color="red", linestyle="--")
    axes[0, 1].set_title("测试集残差分布")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].scatter(
        y_train_pred, train_residuals, alpha=0.6, s=20, color="steelblue"
    )
    axes[1, 0].axhline(0, color="red", linestyle="--")
    axes[1, 0].set_title("训练集残差 vs 预测值")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(y_test_pred, test_residuals, alpha=0.6, s=20, color="coral")
    axes[1, 1].axhline(0, color="red", linestyle="--")
    axes[1, 1].set_title("测试集残差 vs 预测值")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(DECISIONTREE_OUTPUTS, "05_residual_analysis.png")
    plt.savefig(filepath, dpi=1000, bbox_inches="tight")
    print(f"保存图像: {filepath}")

    # 特征重要性
    importances = model.feature_importances_
    plt.figure(figsize=(7, 4))
    sns.barplot(x=importances, y=feature_names)
    plt.title("特征重要性")
    plt.xlabel("重要性")
    plt.ylabel("特征")
    plt.tight_layout()
    filepath = os.path.join(DECISIONTREE_OUTPUTS, "06_feature_importance.png")
    plt.savefig(filepath, dpi=1000, bbox_inches="tight")
    print(f"保存图像: {filepath}")

    # 决策树结构（深度大时会很复杂）
    plt.figure(figsize=(28, 14))
    plot_tree(
        model,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        fontsize=10,  # 字体变大
        max_depth=4,  # 只画前4层，避免挤在一起
    )
    plt.title("决策树结构")
    plt.tight_layout()
    filepath = os.path.join(DECISIONTREE_OUTPUTS, "07_tree_structure.png")
    plt.savefig(filepath, dpi=1000, bbox_inches="tight")
    print(f"保存图像: {filepath}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, X, y = preprocess_data(generate_data())
    model = train_model(X_train, y_train)
    y_train_pred, y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
    features = X.columns.tolist()
    visualize_results(y_train, y_train_pred, y_test, y_test_pred, model, features)
