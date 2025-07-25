"""
模型性能评估可视化
对应文档: ../../docs/visualization/08-model-evaluation.md
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
import sys
from pathlib import Path

# 添加项目根目录到搜索路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_output_dir


def demo_confusion_matrix(output_dir):
    """演示混淆矩阵"""
    print("=" * 50)
    print("1. 混淆矩阵")
    print("=" * 50)

    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Class 0", "Class 1"])
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(output_dir / "viz_08_confusion.png", dpi=100)
    plt.close()
    print("图表已保存")


def demo_roc_curve(output_dir):
    """演示 ROC 曲线"""
    print("=" * 50)
    print("2. ROC 曲线")
    print("=" * 50)

    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "r--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "viz_08_roc.png", dpi=100)
    plt.close()
    print("图表已保存")


def demo_learning_curve(output_dir):
    """演示学习曲线"""
    print("=" * 50)
    print("3. 学习曲线")
    print("=" * 50)

    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)

    clf = LogisticRegression(random_state=42)

    train_sizes, train_scores, test_scores = learning_curve(
        clf, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_mean, "o-", color="blue", label="Training")
    ax.fill_between(
        train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2
    )
    ax.plot(train_sizes, test_mean, "o-", color="green", label="Validation")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2)
    ax.set_xlabel("Training Size")
    ax.set_ylabel("Score")
    ax.set_title("Learning Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "viz_08_learning.png", dpi=100)
    plt.close()
    print("图表已保存")


def demo_all():
    """运行所有演示"""
    output_dir = get_output_dir("visualization")

    demo_confusion_matrix(output_dir)
    print()
    demo_roc_curve(output_dir)
    print()
    demo_learning_curve(output_dir)


if __name__ == "__main__":
    demo_all()
