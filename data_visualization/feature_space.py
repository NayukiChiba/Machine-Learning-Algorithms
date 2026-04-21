"""
data_visualization/feature_space.py
特征空间可视化模块

使用 PCA 将高维数据降至 2D/3D 后可视化:
  - 2D 投影: PCA 降至 2 维后按类别/簇着色
  - 3D 投影: PCA 降至 3 维后的三维散点图

对原本就是 2D 的数据 (如 SVC, KNN, KMeans 等) 直接绘制原始特征空间

输出目录: outputs/data_visualization/feature_space/

使用方式:
    from data_visualization.feature_space import plot_feature_spaces
    plot_feature_spaces()

或直接运行:
    python -m data_visualization.feature_space
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from sklearn.decomposition import PCA

from config import DATA_VIS_FEATURE_SPACE_DIR as OUTPUT_DIR

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


# --- 通用绘图工具 ---


def _save_fig(fig: plt.Figure, filename: str, output_name: str) -> None:
    """
    保存图表到当前模块目录

    args:
        fig(Figure): matplotlib 图表对象
        filename(str): 文件名
        output_name(str): 输出名称前缀
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / f"{output_name}_{filename}"
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {filepath}")


def _plot_2d_projection(
    data: DataFrame,
    feature_cols: list[str],
    color_col: str,
    output_name: str,
    title_suffix: str = "",
) -> None:
    """
    PCA 降至 2D 后按标签着色的散点图

    如果原始特征就是 2D，直接绘制原始特征空间

    args:
        data(DataFrame): 数据
        feature_cols(list[str]): 特征列名列表
        color_col(str): 着色列名
        output_name(str): 输出名称前缀
        title_suffix(str): 标题后缀
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    X = data[feature_cols].values
    labels = data[color_col].values

    if len(feature_cols) <= 2:
        # 原始就是 2D，直接使用
        x_plot = X[:, 0]
        y_plot = X[:, 1] if X.shape[1] > 1 else np.zeros(len(X))
        x_label = feature_cols[0]
        y_label = feature_cols[1] if len(feature_cols) > 1 else ""
        fig.suptitle(
            f"{output_name} — 原始特征空间{title_suffix}",
            fontsize=13,
            fontweight="bold",
        )
    else:
        # PCA 降至 2D
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        x_plot = X_2d[:, 0]
        y_plot = X_2d[:, 1]

        # 解释方差比
        ev1 = pca.explained_variance_ratio_[0] * 100
        ev2 = pca.explained_variance_ratio_[1] * 100
        x_label = f"PC1 ({ev1:.1f}%)"
        y_label = f"PC2 ({ev2:.1f}%)"
        fig.suptitle(
            f"{output_name} — PCA 2D 投影{title_suffix}",
            fontsize=13,
            fontweight="bold",
        )

    # 按类别着色
    unique_labels = sorted(np.unique(labels))
    colors = sns.color_palette("Set2", len(unique_labels))

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(
            x_plot[mask],
            y_plot[mask],
            s=15,
            alpha=0.6,
            color=colors[i],
            label=f"{color_col}={lbl}",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    _save_fig(fig, "01_2d_projection.png", output_name)


def _plot_3d_projection(
    data: DataFrame,
    feature_cols: list[str],
    color_col: str,
    output_name: str,
) -> None:
    """
    PCA 降至 3D 后的三维散点图

    至少需要 3 个特征才有意义

    args:
        data(DataFrame): 数据
        feature_cols(list[str]): 特征列名列表
        color_col(str): 着色列名
        output_name(str): 输出名称前缀
    """
    if len(feature_cols) < 3:
        return

    X = data[feature_cols].values
    labels = data[color_col].values

    # PCA 降至 3D
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ev = pca.explained_variance_ratio_ * 100

    unique_labels = sorted(np.unique(labels))
    colors = sns.color_palette("Set2", len(unique_labels))

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(
            X_3d[mask, 0],
            X_3d[mask, 1],
            X_3d[mask, 2],
            s=10,
            alpha=0.5,
            color=colors[i],
            label=f"{color_col}={lbl}",
        )

    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.set_zlabel(f"PC3 ({ev[2]:.1f}%)")
    ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"{output_name} — PCA 3D 投影", fontsize=13, fontweight="bold")

    fig.tight_layout()
    _save_fig(fig, "02_3d_projection.png", output_name)


# --- 按数据集类型的入口函数 ---


def _feature_space_classification(
    data: DataFrame, name: str, short_name: str, target_col: str = "label"
) -> None:
    """分类数据集的特征空间可视化"""
    feature_cols = [c for c in data.columns if c != target_col]
    print(f"数据集: {name}")

    _plot_2d_projection(data, feature_cols, target_col, short_name)
    _plot_3d_projection(data, feature_cols, target_col, short_name)


def _feature_space_clustering(
    data: DataFrame, name: str, short_name: str, label_col: str = "true_label"
) -> None:
    """聚类数据集的特征空间可视化"""
    feature_cols = [c for c in data.columns if c != label_col]
    print(f"数据集: {name}")

    _plot_2d_projection(data, feature_cols, label_col, short_name)
    _plot_3d_projection(data, feature_cols, label_col, short_name)


# --- 主入口 ---


def plot_feature_spaces() -> None:
    """
    为所有适合的数据集生成特征空间可视化

    回归任务没有离散类别标签，不适合着色散点图，跳过
    HMM 是离散序列，不适合特征空间可视化，跳过
    """
    from data_generation import (
        logistic_regression_data,
        decision_tree_classification_data,
        svc_data,
        naive_bayes_data,
        knn_data,
        random_forest_data,
        kmeans_data,
        dbscan_data,
        em_data,
        bagging_data,
        gbdt_data,
        lightgbm_data,
        pca_data,
        lda_data,
    )

    print("=" * 50)
    print("生成特征空间可视化 (PCA 2D/3D 投影)")
    print("=" * 50)

    # --- 分类 (6) ---

    _feature_space_classification(
        logistic_regression_data, "LogisticRegression", "logistic_regression"
    )
    _feature_space_classification(
        decision_tree_classification_data, "DecisionTree", "decision_tree_clf"
    )
    _feature_space_classification(svc_data, "SVC", "svc")
    _feature_space_classification(naive_bayes_data, "NaiveBayes", "naive_bayes")
    _feature_space_classification(knn_data, "KNN", "knn")
    _feature_space_classification(random_forest_data, "RandomForest", "random_forest")

    # --- 聚类 (2) ---

    _feature_space_clustering(kmeans_data, "KMeans", "kmeans")
    _feature_space_clustering(dbscan_data, "DBSCAN", "dbscan")

    # --- 集成 (4, 跳过回归的 XGBoost) ---

    _feature_space_classification(bagging_data, "Bagging", "bagging")
    _feature_space_classification(gbdt_data, "GBDT", "gbdt")
    _feature_space_classification(lightgbm_data, "LightGBM", "lightgbm")

    # --- 降维 (2) ---

    _feature_space_classification(pca_data, "PCA", "pca")
    _feature_space_classification(lda_data, "LDA", "lda")

    # --- 概率 (1, 跳过 HMM) ---

    _feature_space_clustering(em_data, "EM(GMM)", "em")

    print("=" * 50)
    print("特征空间可视化完成")
    print("=" * 50)


# --- 直接运行 ---

if __name__ == "__main__":
    plot_feature_spaces()
