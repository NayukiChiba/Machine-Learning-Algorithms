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
    
    return iris


def demo_generate_data():
    """演示如何生成人工数据集"""
    print("=" * 50)
    print("2. 生成人工数据集")
    print("=" * 50)
    
    from sklearn.datasets import make_classification, make_regression, make_blobs
    
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
