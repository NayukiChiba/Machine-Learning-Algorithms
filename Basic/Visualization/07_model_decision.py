"""
模型决策过程可视化
对应文档: ../../docs/foundations/visualization/07-model-decision.md
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from . import output_dir as get_output_dir


# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def decision_boundary(output_dir):
    """演示决策边界可视化"""
    print("=" * 50)
    print("1. 决策边界")
    print("=" * 50)

    # 创建数据
    np.random.seed(42)
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
    )

    # 训练模型
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y)

    # 创建网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdYlBu", edgecolors="black")
    ax.set_title("Decision Boundary")

    plt.tight_layout()
    plt.savefig(output_dir / "07_boundary.png", dpi=100)
    plt.close()
    print("图表已保存")


def tree_viz(output_dir):
    """演示决策树可视化"""
    print("=" * 50)
    print("2. 决策树可视化")
    print("=" * 50)

    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=4, n_redundant=0)

    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y)

    fig, ax = plt.subplots(figsize=(15, 10))
    plot_tree(
        clf,
        ax=ax,
        filled=True,
        feature_names=["F1", "F2", "F3", "F4"],
        class_names=["Class 0", "Class 1"],
        rounded=True,
    )
    ax.set_title("Decision Tree Visualization")

    plt.tight_layout()
    plt.savefig(output_dir / "07_tree.png", dpi=100)
    plt.close()
    print("图表已保存")


def feature_importance(output_dir):
    """演示特征重要性"""
    print("=" * 50)
    print("3. 特征重要性")
    print("=" * 50)

    np.random.seed(42)
    X, y = make_classification(
        n_samples=200, n_features=10, n_redundant=3, n_informative=5
    )
    feature_names = [f"Feature_{i}" for i in range(10)]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(importances)), importances[indices], color="steelblue")
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_dir / "07_importance.png", dpi=100)
    plt.close()
    print("图表已保存")


def run():
    """运行所有演示"""
    output_dir = get_output_dir()

    decision_boundary(output_dir)
    print()
    tree_viz(output_dir)
    print()
    feature_importance(output_dir)


if __name__ == "__main__":
    run()
