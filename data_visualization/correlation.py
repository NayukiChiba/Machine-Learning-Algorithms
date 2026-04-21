"""
data_visualization/correlation.py
相关性热力图模块

为每个数据集生成相关性热力图:
  - 特征间皮尔逊相关系数热力图
  - 含目标变量的完整相关系数热力图

输出目录: outputs/data_visualization/correlation/

使用方式:
    from data_visualization.correlation import plot_correlations
    plot_correlations()

或直接运行:
    python -m data_visualization.correlation
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

from config import DATA_VIS_CORRELATION_DIR as OUTPUT_DIR

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


def _plot_heatmap(
    data: DataFrame,
    columns: list[str],
    output_name: str,
    title: str,
    filename: str,
    annot: bool = True,
) -> None:
    """
    绘制相关性热力图

    使用 seaborn heatmap, coolwarm 配色:
      - 红色: 正相关
      - 蓝色: 负相关
      - 白色: 无相关

    特征数 > 15 时关闭标注 (annot=False), 避免文字过密

    args:
        data(DataFrame): 数据
        columns(list[str]): 列名列表
        output_name(str): 输出名称前缀
        title(str): 图标题
        filename(str): 保存文件名
        annot(bool): 是否显示相关系数数值
    """
    corr = data[columns].corr(method="pearson")

    # 动态调整图表大小
    n = len(columns)
    size = max(6, n * 0.6)

    # 特征过多时关闭数字标注
    show_annot = annot and n <= 15
    fmt = ".2f" if show_annot else ""

    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    sns.heatmap(
        corr,
        annot=show_annot,
        fmt=fmt,
        cmap="coolwarm",
        center=0,
        square=True,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
        vmin=-1,
        vmax=1,
    )

    # 特征名过长时旋转
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    _save_fig(fig, filename, output_name)


# --- 按数据集类型的绘图函数 ---


def _corr_with_target(
    data: DataFrame,
    name: str,
    short_name: str,
    target_col: str,
) -> None:
    """
    绘制包含目标变量的完整相关性热力图

    args:
        data(DataFrame): 数据
        name(str): 显示名称
        short_name(str): 目录名
        target_col(str): 目标变量列名
    """
    all_cols = [c for c in data.columns if c != target_col] + [target_col]
    print(f"数据集: {name}")

    # 完整热力图 (含目标变量)
    _plot_heatmap(
        data,
        all_cols,
        short_name,
        f"{short_name} — 相关性热力图",
        "01_correlation_heatmap.png",
    )

    # 特征数 >= 3 时还画一个纯特征间的热力图
    feature_cols = [c for c in data.columns if c != target_col]
    if len(feature_cols) >= 3:
        _plot_heatmap(
            data,
            feature_cols,
            short_name,
            f"{short_name} — 特征间相关性",
            "02_feature_correlation.png",
        )


# --- 主入口 ---


def plot_correlations() -> None:
    """
    为所有 20 个数据集生成相关性热力图
    """
    from data_generation import (
        logistic_regression_data,
        decision_tree_classification_data,
        svc_data,
        naive_bayes_data,
        knn_data,
        random_forest_data,
        linear_regression_data,
        svr_data,
        decision_tree_regression_data,
        regularization_data,
        kmeans_data,
        dbscan_data,
        em_data,
        hmm_data,
        bagging_data,
        gbdt_data,
        xgboost_data,
        lightgbm_data,
        pca_data,
        lda_data,
    )

    print("=" * 50)
    print("生成相关性热力图")
    print("=" * 50)

    # --- 分类 (6) ---

    _corr_with_target(
        logistic_regression_data, "LogisticRegression", "logistic_regression", "label"
    )
    _corr_with_target(
        decision_tree_classification_data, "DecisionTree", "decision_tree_clf", "label"
    )
    _corr_with_target(svc_data, "SVC", "svc", "label")
    _corr_with_target(naive_bayes_data, "NaiveBayes", "naive_bayes", "label")
    _corr_with_target(knn_data, "KNN", "knn", "label")
    _corr_with_target(random_forest_data, "RandomForest", "random_forest", "label")

    # --- 回归 (4) ---

    _corr_with_target(
        linear_regression_data, "LinearRegression", "linear_regression", "price"
    )
    _corr_with_target(svr_data, "SVR", "svr", "price")
    _corr_with_target(
        decision_tree_regression_data,
        "DecisionTree(回归)",
        "decision_tree_reg",
        "price",
    )
    _corr_with_target(regularization_data, "Regularization", "regularization", "price")

    # --- 聚类 (2) ---

    _corr_with_target(kmeans_data, "KMeans", "kmeans", "true_label")
    _corr_with_target(dbscan_data, "DBSCAN", "dbscan", "true_label")

    # --- 集成 (4) ---

    _corr_with_target(bagging_data, "Bagging", "bagging", "label")
    _corr_with_target(gbdt_data, "GBDT", "gbdt", "label")
    _corr_with_target(xgboost_data, "XGBoost", "xgboost", "price")
    _corr_with_target(lightgbm_data, "LightGBM", "lightgbm", "label")

    # --- 降维 (2) ---

    _corr_with_target(pca_data, "PCA", "pca", "label")
    _corr_with_target(lda_data, "LDA", "lda", "label")

    # --- 概率 (2) ---

    _corr_with_target(em_data, "EM(GMM)", "em", "true_label")
    # HMM 是离散序列，相关性热力图意义不大，但仍生成
    _corr_with_target(hmm_data, "HMM", "hmm", "state_true")

    print("=" * 50)
    print("相关性热力图生成完成, 共 20 个数据集")
    print("=" * 50)


# --- 直接运行 ---

if __name__ == "__main__":
    plot_correlations()
