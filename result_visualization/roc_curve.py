"""
result_visualization/roc_curve.py
ROC 曲线可视化

绘制分类模型的 ROC 曲线及 AUC 值。
支持二分类和多分类（One-vs-Rest）。

使用方式:
    from result_visualization.roc_curve import plot_roc_curve
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve as sk_roc_curve, auc
from sklearn.preprocessing import label_binarize

from config import get_model_output_dir


def plot_roc_curve(
    y_true,
    y_scores,
    class_names: list[str] | None = None,
    title: str = "ROC 曲线",
    model_name: str = "model",
    figsize: tuple = (8, 7),
):
    """
    绘制 ROC 曲线

    args:
        y_true: 真实标签
        y_scores: 预测概率（二分类时为正类概率，多分类时为各类概率矩阵）
        class_names: 类别名称
        title: 图标题
        model_name: 模型名称
        figsize: 图像尺寸
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    fig, ax = plt.subplots(figsize=figsize)

    classes = np.unique(y_true)
    n_classes = len(classes)

    if n_classes == 2:
        # 二分类
        if y_scores.ndim == 2:
            y_score_pos = y_scores[:, 1]
        else:
            y_score_pos = y_scores
        fpr, tpr, _ = sk_roc_curve(y_true, y_score_pos)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
    else:
        # 多分类 One-vs-Rest
        y_bin = label_binarize(y_true, classes=classes)
        if class_names is None:
            class_names = [str(c) for c in classes]
        colors = plt.cm.tab10.colors
        for i in range(n_classes):
            fpr, tpr, _ = sk_roc_curve(y_bin[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr,
                tpr,
                linewidth=2,
                color=colors[i % len(colors)],
                label=f"{class_names[i]} (AUC = {roc_auc:.4f})",
            )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="随机基线")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    save_dir = get_model_output_dir(model_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / "roc_curve.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"ROC 曲线已保存至: {filepath}")
    plt.close(fig)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_scores = model.predict_proba(X_test)
    plot_roc_curve(
        y_test,
        y_scores,
        class_names=["负类", "正类"],
        title="逻辑回归 ROC 曲线",
        model_name="logistic_regression",
    )
