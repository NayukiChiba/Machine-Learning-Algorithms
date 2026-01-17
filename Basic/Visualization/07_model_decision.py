"""
模型决策过程可视化
对应文档: ../../docs/visualization/07-model-decision.md
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier


def demo_decision_boundary():
    """演示决策边界可视化"""
    print("=" * 50)
    print("1. 决策边界")
    print("=" * 50)
    
    # 创建数据
    np.random.seed(42)
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1)
    
    # 训练模型
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y)
    
    # 创建网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    ax.set_title('Decision Boundary')
    
    plt.tight_layout()
    plt.savefig('../outputs/viz_07_boundary.png', dpi=100)
    plt.close()
    print("图表已保存")


def demo_tree_viz():
    """演示决策树可视化"""
    print("=" * 50)
    print("2. 决策树可视化")
    print("=" * 50)
    
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=4, n_redundant=0)
    
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y)
    
    fig, ax = plt.subplots(figsize=(15, 10))
    plot_tree(clf, ax=ax, filled=True, feature_names=['F1', 'F2', 'F3', 'F4'],
              class_names=['Class 0', 'Class 1'], rounded=True)
    ax.set_title('Decision Tree Visualization')
    
    plt.tight_layout()
    plt.savefig('../outputs/viz_07_tree.png', dpi=100)
    plt.close()
    print("图表已保存")


def demo_feature_importance():
    """演示特征重要性"""
    print("=" * 50)
    print("3. 特征重要性")
    print("=" * 50)
    
    np.random.seed(42)
    X, y = make_classification(n_samples=200, n_features=10, n_redundant=3,
                               n_informative=5)
    feature_names = [f'Feature_{i}' for i in range(10)]
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(importances)), importances[indices], color='steelblue')
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('../outputs/viz_07_importance.png', dpi=100)
    plt.close()
    print("图表已保存")


def demo_all():
    """运行所有演示"""
    import os
    os.makedirs('../outputs', exist_ok=True)
    
    demo_decision_boundary()
    print()
    demo_tree_viz()
    print()
    demo_feature_importance()


if __name__ == "__main__":
    demo_all()
