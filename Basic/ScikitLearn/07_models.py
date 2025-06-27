"""
Scikit-learn 常用模型速查
对应文档: ../docs/07_models.md

使用方式：
    from code.07_models import *
    demo_linear_models()
    demo_ensemble_models()
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def demo_linear_regression():
    """线性回归模型"""
    print("=" * 50)
    print("1. 线性回归模型")
    print("=" * 50)
    
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge(alpha=1)': Ridge(alpha=1.0),
        'Lasso(alpha=0.1)': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name}: R² = {model.score(X_test, y_test):.4f}")


def demo_logistic_regression():
    """逻辑回归"""
    print("=" * 50)
    print("2. 逻辑回归")
    print("=" * 50)
    
    from sklearn.linear_model import LogisticRegression
    
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 基础用法
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    print(f"基础: 准确率 = {log_reg.score(X_test, y_test):.4f}")
    
    # 带类别权重
    log_reg_balanced = LogisticRegression(class_weight='balanced', max_iter=1000)
    log_reg_balanced.fit(X_train, y_train)
    print(f"class_weight='balanced': 准确率 = {log_reg_balanced.score(X_test, y_test):.4f}")


def demo_tree_models():
    """决策树"""
    print("=" * 50)
    print("3. 决策树")
    print("=" * 50)
    
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    dt = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion='gini',
        random_state=42
    )
    dt.fit(X_train, y_train)
    
    print(f"准确率: {dt.score(X_test, y_test):.4f}")
    print(f"特征重要性: {dt.feature_importances_.round(3)}")
    print(f"树深度: {dt.get_depth()}")


def demo_ensemble_models():
    """集成模型"""
    print("=" * 50)
    print("4. 集成模型")
    print("=" * 50)
    
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier, 
        AdaBoostClassifier, HistGradientBoostingClassifier
    )
    
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
        'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name}: {model.score(X_test, y_test):.4f}")


def demo_svm():
    """SVM"""
    print("=" * 50)
    print("5. SVM (需要标准化)")
    print("=" * 50)
    
    from sklearn.svm import SVC, SVR, LinearSVC
    
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # SVC 需要标准化
    svc = make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf'))
    svc.fit(X_train, y_train)
    print(f"SVC (rbf): {svc.score(X_test, y_test):.4f}")
    
    linear_svc = make_pipeline(StandardScaler(), LinearSVC(max_iter=10000))
    linear_svc.fit(X_train, y_train)
    print(f"LinearSVC: {linear_svc.score(X_test, y_test):.4f}")


def demo_naive_bayes():
    """朴素贝叶斯"""
    print("=" * 50)
    print("6. 朴素贝叶斯")
    print("=" * 50)
    
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
    
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    print(f"GaussianNB: {gnb.score(X_test, y_test):.4f}")


def demo_knn():
    """K近邻"""
    print("=" * 50)
    print("7. K近邻")
    print("=" * 50)
    
    from sklearn.neighbors import KNeighborsClassifier
    
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    knn.fit(X_train, y_train)
    print(f"KNN (k=5): {knn.score(X_test, y_test):.4f}")


def demo_clustering():
    """聚类算法"""
    print("=" * 50)
    print("8. 聚类算法")
    print("=" * 50)
    
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    
    X, _ = datasets.make_blobs(n_samples=300, centers=4, random_state=42)
    
    # KMeans
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels_km = kmeans.fit_predict(X)
    print(f"KMeans 轮廓系数: {silhouette_score(X, labels_km):.4f}")
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels_db = dbscan.fit_predict(X)
    n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
    print(f"DBSCAN 聚类数: {n_clusters}")


def demo_dimensionality_reduction():
    """降维"""
    print("=" * 50)
    print("9. 降维")
    print("=" * 50)
    
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    iris = datasets.load_iris()
    X = iris.data
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    print(f"PCA 解释方差比: {pca.explained_variance_ratio_.round(4)}")
    print(f"PCA 累计解释方差: {pca.explained_variance_ratio_.sum():.4f}")
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    print(f"t-SNE 输出形状: {X_tsne.shape}")


def demo_all():
    """运行所有演示"""
    demo_linear_regression()
    print()
    demo_logistic_regression()
    print()
    demo_tree_models()
    print()
    demo_ensemble_models()
    print()
    demo_svm()
    print()
    demo_naive_bayes()
    print()
    demo_knn()
    print()
    demo_clustering()
    print()
    demo_dimensionality_reduction()


if __name__ == "__main__":
    demo_all()
