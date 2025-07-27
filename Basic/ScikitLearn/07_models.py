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
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def demo_linear_regression():
    """线性回归模型"""
    print("=" * 50)
    print("1. 线性回归模型")
    print("=" * 50)

    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge (L2)": Ridge(alpha=1.0),
        "Lasso (L1)": Lasso(alpha=0.1),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    }

    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores[name] = score
        print(f"{name}: R² = {score:.4f}")

    # === 可视化: 回归模型对比 ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 模型得分对比
    ax1 = axes[0]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
    bars = ax1.bar(scores.keys(), scores.values(), color=colors, edgecolor="black")
    ax1.set_ylabel("R² 分数")
    ax1.set_title("线性回归模型对比 (糖尿病数据集)")
    ax1.set_ylim(0, 0.6)
    for bar, score in zip(bars, scores.values()):
        ax1.annotate(
            f"{score:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax1.grid(True, alpha=0.3, axis="y")
    plt.setp(ax1.get_xticklabels(), rotation=15, ha="right")

    # 右图: 正则化系数对比
    ax2 = axes[1]
    ridge = Ridge(alpha=1.0).fit(X_train, y_train)
    lasso = Lasso(alpha=0.1).fit(X_train, y_train)
    lr = LinearRegression().fit(X_train, y_train)

    x_pos = np.arange(len(diabetes.feature_names))
    width = 0.25
    ax2.bar(x_pos - width, lr.coef_, width, label="Linear", color="#FF6B6B", alpha=0.7)
    ax2.bar(x_pos, ridge.coef_, width, label="Ridge", color="#4ECDC4", alpha=0.7)
    ax2.bar(
        x_pos + width, lasso.coef_, width, label="Lasso", color="#45B7D1", alpha=0.7
    )
    ax2.set_xlabel("特征")
    ax2.set_ylabel("系数值")
    ax2.set_title("不同正则化的系数对比")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(diabetes.feature_names, rotation=45, ha="right")
    ax2.legend()
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/sklearn/07_linear_models.png", dpi=150, bbox_inches="tight")
    plt.close()


def demo_logistic_regression():
    """逻辑回归"""
    print("=" * 50)
    print("2. 逻辑回归")
    print("=" * 50)

    from sklearn.linear_model import LogisticRegression

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 基础用法
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    print(f"基础: 准确率 = {log_reg.score(X_test, y_test):.4f}")

    # 带类别权重
    log_reg_balanced = LogisticRegression(class_weight="balanced", max_iter=1000)
    log_reg_balanced.fit(X_train, y_train)
    print(
        f"class_weight='balanced': 准确率 = {log_reg_balanced.score(X_test, y_test):.4f}"
    )


def demo_tree_models():
    """决策树"""
    print("=" * 50)
    print("3. 决策树")
    print("=" * 50)

    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    dt = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="gini",
        random_state=42,
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
        RandomForestClassifier,
        GradientBoostingClassifier,
        AdaBoostClassifier,
        HistGradientBoostingClassifier,
    )

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
    }

    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores[name] = score
        print(f"{name}: {score:.4f}")

    # === 可视化: 集成模型对比 ===
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
    bars = ax.barh(
        list(scores.keys()), list(scores.values()), color=colors, edgecolor="black"
    )
    ax.set_xlabel("准确率")
    ax.set_title("集成学习模型对比 (鸢尾花数据集)")
    ax.set_xlim(0, 1.1)
    for bar, score in zip(bars, scores.values()):
        ax.annotate(
            f"{score:.3f}",
            xy=(score + 0.02, bar.get_y() + bar.get_height() / 2),
            va="center",
            fontsize=11,
            fontweight="bold",
        )
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("outputs/sklearn/07_ensemble.png", dpi=150, bbox_inches="tight")
    plt.close()


def demo_svm():
    """SVM"""
    print("=" * 50)
    print("5. SVM (需要标准化)")
    print("=" * 50)

    from sklearn.svm import SVC, SVR, LinearSVC
    from sklearn.datasets import make_moons

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # SVC 需要标准化
    svc = make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf"))
    svc.fit(X_train, y_train)
    print(f"SVC (rbf): {svc.score(X_test, y_test):.4f}")

    linear_svc = make_pipeline(StandardScaler(), LinearSVC(max_iter=10000))
    linear_svc.fit(X_train, y_train)
    print(f"LinearSVC: {linear_svc.score(X_test, y_test):.4f}")

    # === 可视化: SVM 决策边界 ===
    # 使用月牙形数据集展示
    X_moons, y_moons = make_moons(n_samples=200, noise=0.15, random_state=42)
    X_moons = StandardScaler().fit_transform(X_moons)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    kernels = ["linear", "rbf", "poly"]
    titles = ["线性核 (Linear)", "径向基核 (RBF)", "多项式核 (Poly)"]

    for ax, kernel, title in zip(axes, kernels, titles):
        svc_k = SVC(kernel=kernel, C=1.0)
        svc_k.fit(X_moons, y_moons)

        # 绘制决策边界
        h = 0.02
        x_min, x_max = X_moons[:, 0].min() - 0.5, X_moons[:, 0].max() + 0.5
        y_min, y_max = X_moons[:, 1].min() - 0.5, X_moons[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = svc_k.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        ax.scatter(
            X_moons[y_moons == 0, 0],
            X_moons[y_moons == 0, 1],
            c="#FF6B6B",
            label="类别 0",
            edgecolors="white",
            s=50,
        )
        ax.scatter(
            X_moons[y_moons == 1, 0],
            X_moons[y_moons == 1, 1],
            c="#4ECDC4",
            label="类别 1",
            edgecolors="white",
            s=50,
        )
        ax.set_title(f"{title}\n准确率: {svc_k.score(X_moons, y_moons):.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/sklearn/07_svm.png", dpi=150, bbox_inches="tight")
    plt.close()


def demo_naive_bayes():
    """朴素贝叶斯"""
    print("=" * 50)
    print("6. 朴素贝叶斯")
    print("=" * 50)

    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

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

    X, y_true = datasets.make_blobs(n_samples=300, centers=4, random_state=42)

    # KMeans
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels_km = kmeans.fit_predict(X)
    print(f"KMeans 轮廓系数: {silhouette_score(X, labels_km):.4f}")

    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels_db = dbscan.fit_predict(X)
    n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
    print(f"DBSCAN 聚类数: {n_clusters}")

    # === 可视化: 聚类对比 ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 真实标签
    ax1 = axes[0]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
    for i in range(4):
        ax1.scatter(
            X[y_true == i, 0],
            X[y_true == i, 1],
            c=colors[i],
            s=40,
            alpha=0.7,
            edgecolors="white",
        )
    ax1.set_title("真实标签")
    ax1.set_xlabel("特征 1")
    ax1.set_ylabel("特征 2")
    ax1.grid(True, alpha=0.3)

    # KMeans
    ax2 = axes[1]
    for i in range(4):
        ax2.scatter(
            X[labels_km == i, 0],
            X[labels_km == i, 1],
            c=colors[i],
            s=40,
            alpha=0.7,
            edgecolors="white",
        )
    ax2.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        c="black",
        marker="X",
        s=200,
        label="聚类中心",
    )
    ax2.set_title(f"KMeans (轮廓系数: {silhouette_score(X, labels_km):.3f})")
    ax2.set_xlabel("特征 1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # DBSCAN
    ax3 = axes[2]
    unique_labels = set(labels_db)
    for label in unique_labels:
        if label == -1:
            ax3.scatter(
                X[labels_db == -1, 0],
                X[labels_db == -1, 1],
                c="gray",
                marker="x",
                s=40,
                label="噪声",
                alpha=0.5,
            )
        else:
            ax3.scatter(
                X[labels_db == label, 0],
                X[labels_db == label, 1],
                c=colors[label % 4],
                s=40,
                alpha=0.7,
                edgecolors="white",
            )
    ax3.set_title(f"DBSCAN (聚类数: {n_clusters})")
    ax3.set_xlabel("特征 1")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/sklearn/07_clustering.png", dpi=150, bbox_inches="tight")
    plt.close()


def demo_dimensionality_reduction():
    """降维"""
    print("=" * 50)
    print("9. 降维")
    print("=" * 50)

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    print(f"PCA 解释方差比: {pca.explained_variance_ratio_.round(4)}")
    print(f"PCA 累计解释方差: {pca.explained_variance_ratio_.sum():.4f}")

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    print(f"t-SNE 输出形状: {X_tsne.shape}")

    # === 可视化: 降维对比 ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    # PCA
    ax1 = axes[0]
    for i, name in enumerate(iris.target_names):
        ax1.scatter(
            X_pca[y == i, 0],
            X_pca[y == i, 1],
            c=colors[i],
            label=name,
            s=50,
            alpha=0.7,
            edgecolors="white",
        )
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax1.set_title("PCA 降维结果")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # t-SNE
    ax2 = axes[1]
    for i, name in enumerate(iris.target_names):
        ax2.scatter(
            X_tsne[y == i, 0],
            X_tsne[y == i, 1],
            c=colors[i],
            label=name,
            s=50,
            alpha=0.7,
            edgecolors="white",
        )
    ax2.set_xlabel("t-SNE 维度 1")
    ax2.set_ylabel("t-SNE 维度 2")
    ax2.set_title("t-SNE 降维结果")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "outputs/sklearn/07_dimensionality_reduction.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


def demo_all():
    """运行所有演示"""
    import os

    os.makedirs("outputs/sklearn", exist_ok=True)

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
