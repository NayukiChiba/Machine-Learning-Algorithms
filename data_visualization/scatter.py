"""
data_visualization/scatter.py
散点图模块

为每个数据集生成散点图相关的可视化:
  - 二维散点图: 按类别/簇着色
  - 散点矩阵 (Pairplot): 多特征间的两两关系

输出目录: outputs/data_visualization/scatter/

使用方式:
    from data_visualization.scatter import plot_scatters
    plot_scatters()

或直接运行:
    python -m data_visualization.scatter
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

from config import DATA_VIS_SCATTER_DIR as OUTPUT_DIR

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


def _plot_2d_scatter(
    data: DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    output_name: str,
    title: str,
    filename: str,
) -> None:
    """
    绘制二维散点图，按类别/簇着色

    args:
        data(DataFrame): 数据
        x_col(str): x 轴特征列名
        y_col(str): y 轴特征列名
        color_col(str): 着色列名 (类别/簇标签)
        output_name(str): 输出名称前缀
        title(str): 图标题
        filename(str): 保存文件名
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    classes = sorted(data[color_col].unique())
    colors = sns.color_palette("Set2", len(classes))

    for i, cls in enumerate(classes):
        subset = data[data[color_col] == cls]
        ax.scatter(
            subset[x_col],
            subset[y_col],
            s=20,
            alpha=0.6,
            color=colors[i],
            label=f"{color_col}={cls}",
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    _save_fig(fig, filename, output_name)


def _plot_pairplot(
    data: DataFrame,
    feature_cols: list[str],
    color_col: str,
    output_name: str,
    max_features: int = 6,
) -> None:
    """
    绘制散点矩阵 (Pairplot)

    当特征数超过 max_features 时，只选取前 max_features 个特征
    避免图表过大导致内存问题

    args:
        data(DataFrame): 数据
        feature_cols(list[str]): 特征列名列表
        color_col(str): 着色列名
        output_name(str): 输出名称前缀
        max_features(int): 最大特征数
    """
    # 限制特征数量，避免散点矩阵过大
    plot_cols = feature_cols[:max_features]

    # seaborn pairplot
    plot_data = data[plot_cols + [color_col]].copy()
    plot_data[color_col] = plot_data[color_col].astype(str)

    g = sns.pairplot(
        plot_data,
        hue=color_col,
        palette="Set2",
        plot_kws={"s": 15, "alpha": 0.5},
        diag_kws={"alpha": 0.5},
        height=2,
    )
    g.fig.suptitle(f"{output_name} — 散点矩阵", fontsize=13, fontweight="bold", y=1.02)

    if len(feature_cols) > max_features:
        g.fig.text(
            0.5,
            -0.02,
            f"(仅展示前 {max_features} 个特征，共 {len(feature_cols)} 个)",
            ha="center",
            fontsize=9,
            color="gray",
        )

    _save_fig(g.fig, "02_pairplot.png", output_name)


def _plot_regression_scatter(
    data: DataFrame,
    feature_cols: list[str],
    target_col: str,
    output_name: str,
    max_features: int = 8,
) -> None:
    """
    回归任务: 绘制每个特征 vs 目标变量的散点图

    每个子图是一个特征与目标变量的关系

    args:
        data(DataFrame): 数据
        feature_cols(list[str]): 特征列名列表
        target_col(str): 目标变量列名
        output_name(str): 输出名称前缀
        max_features(int): 最大特征数
    """
    plot_cols = feature_cols[:max_features]
    n = len(plot_cols)

    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    fig.suptitle(
        f"{output_name} — 特征 vs {target_col}", fontsize=13, fontweight="bold"
    )

    if n == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    for i, col in enumerate(plot_cols):
        row_idx = i // cols
        col_idx = i % cols
        ax = axes[row_idx][col_idx]

        ax.scatter(data[col], data[target_col], s=8, alpha=0.4, color="steelblue")
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel(target_col, fontsize=9)
        ax.grid(True, alpha=0.2)

    # 关闭多余的子图
    for i in range(n, rows * cols):
        row_idx = i // cols
        col_idx = i % cols
        axes[row_idx][col_idx].axis("off")

    fig.tight_layout()
    _save_fig(fig, "01_feature_vs_target.png", output_name)


def _plot_sequence_plot(data: DataFrame, output_name: str) -> None:
    """
    HMM 序列: 绘制观测和隐状态的时间序列图

    上下两个子图:
      - 上: 观测序列
      - 下: 隐状态序列 (真实)

    args:
        data(DataFrame): HMM 序列数据
        output_name(str): 输出名称前缀
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    fig.suptitle(f"{output_name} — 时间序列", fontsize=13, fontweight="bold")

    # 观测序列
    axes[0].step(
        data["time"], data["obs"], where="mid", linewidth=0.8, color="steelblue"
    )
    axes[0].set_ylabel("观测符号")
    axes[0].grid(True, alpha=0.2)

    # 隐状态序列
    axes[1].step(
        data["time"], data["state_true"], where="mid", linewidth=0.8, color="coral"
    )
    axes[1].set_ylabel("隐状态")
    axes[1].set_xlabel("时间步")
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    _save_fig(fig, "01_time_series.png", output_name)


# --- 按数据集类型的绘图函数 ---


def _scatter_classification(
    data: DataFrame, name: str, short_name: str, target_col: str = "label"
) -> None:
    """分类数据集的散点图"""
    feature_cols = [c for c in data.columns if c != target_col]
    print(f"数据集: {name}")

    # 只有 2 个特征时直接画散点图
    if len(feature_cols) == 2:
        _plot_2d_scatter(
            data,
            feature_cols[0],
            feature_cols[1],
            target_col,
            short_name,
            f"{short_name} — 散点图",
            "01_scatter.png",
        )

    # 散点矩阵
    if len(feature_cols) >= 2:
        _plot_pairplot(data, feature_cols, target_col, short_name)


def _scatter_regression(
    data: DataFrame, name: str, short_name: str, target_col: str = "price"
) -> None:
    """回归数据集的散点图"""
    feature_cols = [c for c in data.columns if c != target_col]
    print(f"数据集: {name}")

    # 特征 vs 目标
    _plot_regression_scatter(data, feature_cols, target_col, short_name)


def _scatter_clustering(
    data: DataFrame, name: str, short_name: str, label_col: str = "true_label"
) -> None:
    """聚类数据集的散点图"""
    feature_cols = [c for c in data.columns if c != label_col]
    print(f"数据集: {name}")

    if len(feature_cols) == 2:
        # 无标签原始散点
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(
            data[feature_cols[0]], data[feature_cols[1]], s=15, alpha=0.5, color="gray"
        )
        ax.set_xlabel(feature_cols[0])
        ax.set_ylabel(feature_cols[1])
        ax.set_title(f"{short_name} — 原始数据 (无标签)")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        _save_fig(fig, "01_raw_scatter.png", short_name)

        # 按真实标签着色
        _plot_2d_scatter(
            data,
            feature_cols[0],
            feature_cols[1],
            label_col,
            short_name,
            f"{short_name} — 真实标签着色",
            "02_true_label_scatter.png",
        )


# --- 主入口 ---


def plot_scatters() -> None:
    """
    为所有 20 个数据集生成散点图
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
    print("生成散点图 (散点图 + 散点矩阵)")
    print("=" * 50)

    # --- 分类 (6) ---

    _scatter_classification(
        logistic_regression_data, "LogisticRegression", "logistic_regression"
    )
    _scatter_classification(
        decision_tree_classification_data, "DecisionTree", "decision_tree_clf"
    )
    _scatter_classification(svc_data, "SVC", "svc")
    _scatter_classification(naive_bayes_data, "NaiveBayes — Iris", "naive_bayes")
    _scatter_classification(knn_data, "KNN", "knn")
    _scatter_classification(random_forest_data, "RandomForest", "random_forest")

    # --- 回归 (4) ---

    _scatter_regression(linear_regression_data, "LinearRegression", "linear_regression")
    _scatter_regression(svr_data, "SVR", "svr")
    _scatter_regression(
        decision_tree_regression_data, "DecisionTree(回归)", "decision_tree_reg"
    )
    _scatter_regression(regularization_data, "Regularization", "regularization")

    # --- 聚类 (2) ---

    _scatter_clustering(kmeans_data, "KMeans", "kmeans")
    _scatter_clustering(dbscan_data, "DBSCAN", "dbscan")

    # --- 集成 (4) ---

    _scatter_classification(bagging_data, "Bagging", "bagging")
    _scatter_classification(gbdt_data, "GBDT", "gbdt")
    _scatter_regression(xgboost_data, "XGBoost", "xgboost")
    _scatter_classification(lightgbm_data, "LightGBM", "lightgbm")

    # --- 降维 (2) ---

    _scatter_classification(pca_data, "PCA", "pca")
    _scatter_classification(lda_data, "LDA — Wine", "lda")

    # --- 概率 (2) ---

    _scatter_clustering(em_data, "EM(GMM)", "em")
    _plot_sequence_plot(hmm_data, "hmm")
    print("数据集: HMM — 离散序列")

    print("=" * 50)
    print("散点图生成完成, 共 20 个数据集")
    print("=" * 50)


# --- 直接运行 ---

if __name__ == "__main__":
    plot_scatters()
