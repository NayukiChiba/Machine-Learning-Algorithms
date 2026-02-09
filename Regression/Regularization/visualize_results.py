"""
结果可视化
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from utils.decorate import print_func_info
from config import OUTPUTS_ROOT


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
OUTPUT_DIR = os.path.join(OUTPUTS_ROOT, "Regularization")


def _get_top_features(models, feature_names, top_k: int = 12):
    coef_matrix = np.vstack([np.abs(m.coef_) for m in models.values()])
    avg_abs = coef_matrix.mean(axis=0)
    idx = np.argsort(avg_abs)[::-1][:top_k]
    return idx


@print_func_info
def visualize_results(
    y_train,
    y_train_pred,
    y_test,
    y_test_pred,
    feature_names,
    models,
):
    """
    可视化模型结果

    args:
        y_train, y_test: 真实值
        y_train_pred, y_test_pred: 各模型预测值
        feature_names(list): 特征名称
        models(dict): 模型字典
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 系数对比（选取权重最大的特征）
    top_idx = _get_top_features(models, feature_names, top_k=12)
    top_features = [feature_names[i] for i in top_idx]

    x = np.arange(len(top_features))
    width = 0.25

    plt.figure(figsize=(12, 5))
    for i, (name, model) in enumerate(models.items()):
        coef = model.coef_[top_idx]
        plt.bar(x + i * width, coef, width=width, label=name)

    plt.xticks(x + width, top_features, rotation=45)
    plt.title("特征系数对比（Top Features）")
    plt.xlabel("特征")
    plt.ylabel("系数")
    plt.legend()
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "04_coefficients.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")

    # 2. 预测值 vs 真实值（测试集）
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, name in zip(axes, models.keys()):
        pred = y_test_pred[name]
        ax.scatter(y_test, pred, s=20, alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        ax.set_title(name)
        ax.set_xlabel("真实值")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("预测值")
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "05_prediction_scatter.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")

    # 3. 残差分布（测试集）
    plt.figure(figsize=(8, 5))
    for name in models.keys():
        residuals = y_test - y_test_pred[name]
        sns.kdeplot(residuals, label=name, fill=False)
    plt.title("测试集残差分布")
    plt.xlabel("残差")
    plt.legend()
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "06_residuals.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"保存图像: {filepath}")


if __name__ == "__main__":
    from generate_data import generate_data
    from preprocess_data import preprocess_data
    from train_model import train_model
    from evaluate_model import evaluate_model

    df = generate_data()
    X_train, X_test, y_train, y_test, scaler, X_train_orig, X_test_orig = (
        preprocess_data(df)
    )
    models = train_model(X_train, y_train, feature_names=X_train_orig.columns.tolist())
    y_train_pred, y_test_pred, metrics = evaluate_model(
        models, X_train, X_test, y_train, y_test
    )
    visualize_results(
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        feature_names=X_train_orig.columns.tolist(),
        models=models,
    )
