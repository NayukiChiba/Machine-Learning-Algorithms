"""
Scikit-learn 基础入门
对应文档: ../docs/01_basics.md

使用方式：
    from code.01_basics import *
    demo_load_datasets()
    demo_first_model()
"""

import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def demo_load_datasets():
    """演示如何加载和使用 sklearn 内置数据集"""
    print("=" * 50)
    print("1. 加载内置数据集")
    print("=" * 50)
    
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    
    print(f"特征矩阵形状: {iris.data.shape}")
    print(f"目标向量形状: {iris.target.shape}")
    print(f"特征名称: {iris.feature_names}")
    print(f"类别名称: {iris.target_names}")
    
    # 直接返回 X, y
    X, y = datasets.load_iris(return_X_y=True)
    print(f"\nreturn_X_y=True: X={X.shape}, y={y.shape}")
    
    # 返回 DataFrame 格式
    iris_df = datasets.load_iris(as_frame=True)
    print(f"\nas_frame=True:\n{iris_df.frame.head()}")
    
    # === 可视化 ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图: 散点图 - 特征可视化
    ax1 = axes[0]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for i, (name, color) in enumerate(zip(iris.target_names, colors)):
        mask = iris.target == i
        ax1.scatter(iris.data[mask, 0], iris.data[mask, 1], 
                   c=color, label=name, alpha=0.7, s=60, edgecolors='white')
    ax1.set_xlabel('花萼长度 (cm)')
    ax1.set_ylabel('花萼宽度 (cm)')
    ax1.set_title('鸢尾花数据集 - 特征分布')
    ax1.legend(title='类别')
    ax1.grid(True, alpha=0.3)
    
    # 右图: 柱状图 - 类别分布
    ax2 = axes[1]
    unique, counts = np.unique(iris.target, return_counts=True)
    bars = ax2.bar(iris.target_names, counts, color=colors, edgecolor='black', alpha=0.8)
    ax2.set_xlabel('类别')
    ax2.set_ylabel('样本数量')
    ax2.set_title('鸢尾花数据集 - 类别分布')
    for bar, count in zip(bars, counts):
        ax2.annotate(f'{count}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('outputs/sklearn/01_datasets.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return iris


def demo_generate_data():
    """演示如何生成人工数据集"""
    print("=" * 50)
    print("2. 生成人工数据集")
    print("=" * 50)
    
    from sklearn.datasets import make_classification, make_regression, make_blobs, make_moons, make_circles
    
    # 分类数据
    X_clf, y_clf = make_classification(
        n_samples=1000,      # 样本数
        n_features=20,       # 特征数
        n_informative=10,    # 有信息量的特征数
        n_redundant=5,       # 冗余特征数
        n_classes=3,         # 类别数
        random_state=42
    )
    print(f"分类数据: X={X_clf.shape}, y 各类别数量={np.bincount(y_clf)}")
    
    # 回归数据
    X_reg, y_reg = make_regression(
        n_samples=1000,
        n_features=10,
        noise=10,            # 噪声标准差
        random_state=42
    )
    print(f"回归数据: X={X_reg.shape}, y 范围=[{y_reg.min():.1f}, {y_reg.max():.1f}]")
    
    # 聚类数据
    X_blob, y_blob = make_blobs(
        n_samples=500,
        centers=4,           # 聚类中心数
        cluster_std=1.0,     # 簇的标准差
        random_state=42
    )
    print(f"聚类数据: X={X_blob.shape}, y={np.unique(y_blob)}")
    
    # === 可视化: 人工数据集 ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # blobs
    ax1 = axes[0, 0]
    scatter = ax1.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap='viridis', 
                         alpha=0.7, s=40, edgecolors='white')
    ax1.set_title('make_blobs - 聚类数据')
    ax1.set_xlabel('特征 1')
    ax1.set_ylabel('特征 2')
    ax1.grid(True, alpha=0.3)
    
    # moons
    X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)
    ax2 = axes[0, 1]
    ax2.scatter(X_moons[y_moons==0, 0], X_moons[y_moons==0, 1], c='#FF6B6B', label='类别 0', alpha=0.7, s=40)
    ax2.scatter(X_moons[y_moons==1, 0], X_moons[y_moons==1, 1], c='#4ECDC4', label='类别 1', alpha=0.7, s=40)
    ax2.set_title('make_moons - 月牙形数据')
    ax2.set_xlabel('特征 1')
    ax2.set_ylabel('特征 2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # circles
    X_circles, y_circles = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)
    ax3 = axes[1, 0]
    ax3.scatter(X_circles[y_circles==0, 0], X_circles[y_circles==0, 1], c='#FF6B6B', label='类别 0', alpha=0.7, s=40)
    ax3.scatter(X_circles[y_circles==1, 0], X_circles[y_circles==1, 1], c='#4ECDC4', label='类别 1', alpha=0.7, s=40)
    ax3.set_title('make_circles - 同心圆数据')
    ax3.set_xlabel('特征 1')
    ax3.set_ylabel('特征 2')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # regression
    X_reg_2d, y_reg_2d = make_regression(n_samples=200, n_features=1, noise=15, random_state=42)
    ax4 = axes[1, 1]
    ax4.scatter(X_reg_2d, y_reg_2d, c='#45B7D1', alpha=0.6, s=40, edgecolors='white')
    ax4.set_title('make_regression - 回归数据')
    ax4.set_xlabel('特征')
    ax4.set_ylabel('目标值')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/sklearn/01_generate_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return X_clf, y_clf


def demo_train_test_split():
    """演示数据划分"""
    print("=" * 50)
    print("3. 数据划分 (train_test_split)")
    print("=" * 50)
    
    from sklearn.model_selection import train_test_split
    
    X, y = datasets.load_iris(return_X_y=True)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,       # 测试集比例
        random_state=42,     # 随机种子（保证可复现）
        stratify=y           # 分层抽样（保持类别比例）
    )
    
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    print(f"训练集类别分布: {np.bincount(y_train)}")
    print(f"测试集类别分布: {np.bincount(y_test)}")
    
    # === 可视化: 训练/测试集划分 ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图: 散点图 - 训练集 vs 测试集
    ax1 = axes[0]
    ax1.scatter(X_train[:, 0], X_train[:, 1], c='#4ECDC4', label=f'训练集 ({len(X_train)})', 
                alpha=0.6, s=50, edgecolors='white')
    ax1.scatter(X_test[:, 0], X_test[:, 1], c='#FF6B6B', label=f'测试集 ({len(X_test)})', 
                alpha=0.8, s=80, marker='s', edgecolors='black')
    ax1.set_xlabel('花萼长度 (cm)')
    ax1.set_ylabel('花萼宽度 (cm)')
    ax1.set_title('数据划分可视化 (test_size=0.3)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图: 柱状图 - 类别分布对比
    ax2 = axes[1]
    x = np.arange(3)
    width = 0.35
    bars1 = ax2.bar(x - width/2, np.bincount(y_train), width, label='训练集', color='#4ECDC4', edgecolor='black')
    bars2 = ax2.bar(x + width/2, np.bincount(y_test), width, label='测试集', color='#FF6B6B', edgecolor='black')
    ax2.set_xlabel('类别')
    ax2.set_ylabel('样本数量')
    ax2.set_title('分层抽样 (stratify=y) 类别分布对比')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['setosa', 'versicolor', 'virginica'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('outputs/sklearn/01_train_test_split.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return X_train, X_test, y_train, y_test


def demo_first_model():
    """演示第一个机器学习模型完整流程"""
    print("=" * 50)
    print("4. 第一个模型 (KNN)")
    print("=" * 50)
    
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    
    # 加载和划分数据
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 创建模型
    knn = KNeighborsClassifier(n_neighbors=5)
    
    # 训练模型
    knn.fit(X_train, y_train)
    
    # 预测
    y_pred = knn.predict(X_test)
    
    # 评估
    print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
    print(f"score方法: {knn.score(X_test, y_test):.4f}")
    
    # === 可视化: KNN 决策边界 (使用前两个特征) ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 只使用前两个特征进行可视化
    X_2d = X[:, :2]
    X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
        X_2d, y, test_size=0.3, random_state=42, stratify=y
    )
    
    knn_2d = KNeighborsClassifier(n_neighbors=5)
    knn_2d.fit(X_train_2d, y_train_2d)
    
    # 左图: 决策边界
    ax1 = axes[0]
    h = 0.02  # 网格步长
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax1.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    iris = datasets.load_iris()
    for i, (name, color) in enumerate(zip(iris.target_names, colors)):
        mask = y_train_2d == i
        ax1.scatter(X_train_2d[mask, 0], X_train_2d[mask, 1], c=color, label=name, 
                   alpha=0.7, s=60, edgecolors='white')
    ax1.set_xlabel('花萼长度 (cm)')
    ax1.set_ylabel('花萼宽度 (cm)')
    ax1.set_title(f'KNN (k=5) 决策边界')
    ax1.legend(title='类别')
    ax1.grid(True, alpha=0.3)
    
    # 右图: 不同 k 值的准确率
    ax2 = axes[1]
    k_values = range(1, 21)
    train_scores = []
    test_scores = []
    for k in k_values:
        knn_k = KNeighborsClassifier(n_neighbors=k)
        knn_k.fit(X_train, y_train)
        train_scores.append(knn_k.score(X_train, y_train))
        test_scores.append(knn_k.score(X_test, y_test))
    
    ax2.plot(k_values, train_scores, 'o-', color='#4ECDC4', label='训练集', lw=2, markersize=6)
    ax2.plot(k_values, test_scores, 's-', color='#FF6B6B', label='测试集', lw=2, markersize=6)
    ax2.axvline(5, color='gray', linestyle='--', alpha=0.5, label='k=5')
    ax2.set_xlabel('k (邻居数量)')
    ax2.set_ylabel('准确率')
    ax2.set_title('KNN 不同 k 值的准确率')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_values)
    
    plt.tight_layout()
    plt.savefig('outputs/sklearn/01_knn.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return knn


def demo_estimator_methods():
    """演示估计器通用方法"""
    print("=" * 50)
    print("5. 估计器通用方法")
    print("=" * 50)
    
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.base import clone
    
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    # get_params() - 获取模型参数
    print("get_params():")
    params = knn.get_params()
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # set_params() - 设置模型参数
    knn.set_params(n_neighbors=3, weights='distance')
    print(f"\nset_params() 后: n_neighbors={knn.n_neighbors}, weights={knn.weights}")
    
    # predict_proba() - 预测概率
    knn.fit(X_train, y_train)  # 重新训练
    proba = knn.predict_proba(X_test[:3])
    print(f"\npredict_proba() 前3个样本:\n{proba.round(3)}")
    
    # clone() - 克隆模型
    knn_clone = clone(knn)
    print(f"\nclone():")
    print(f"  原模型已训练: {hasattr(knn, 'classes_')}")
    print(f"  克隆模型已训练: {hasattr(knn_clone, 'classes_')}")
    
    # 训练后属性（带下划线）
    print(f"\n训练后属性:")
    print(f"  classes_: {knn.classes_}")
    print(f"  n_features_in_: {knn.n_features_in_}")


def demo_all():
    """运行所有演示"""
    import os
    os.makedirs('outputs/sklearn', exist_ok=True)
    
    demo_load_datasets()
    print()
    demo_generate_data()
    print()
    demo_train_test_split()
    print()
    demo_first_model()
    print()
    demo_estimator_methods()


if __name__ == "__main__":
    demo_all()

