"""
data_visualization/distribution.py
分布图模块

为每个数据集生成分布相关的可视化图表:
  - 直方图 (Histogram): 展示各特征的值域分布
  - 箱线图 (Box Plot): 展示四分位数和异常值
  - 密度图 (KDE): 展示概率密度估计
  - 类别分布柱状图: 展示目标变量各类别的样本数量

输出目录: outputs/data_visualization/distribution/

使用方式:
    from data_visualization.distribution import plot_distributions
    plot_distributions()

或直接运行:
    python -m data_visualization.distribution
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

from config import DATA_VIS_DISTRIBUTION_DIR as OUTPUT_DIR

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


# --- 通用绘图工具 ---


def _save_fig(fig: plt.Figure, filename: str, output_name: str) -> None:
    """
    保存图表到当前模块目录

    args:
        fig(Figure): matplotlib 图表对象
        filename(str): 文件名 (不含路径)
        output_name(str): 输出名称前缀
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / f"{output_name}_{filename}"
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {filepath}")


def _plot_histograms(
    data: DataFrame, feature_cols: list[str], output_name: str
) -> None:
    """
    为所有连续特征绘制直方图 + KDE 密度曲线

    每个子图包含:
      - 直方图 (30 bins, 半透明填充)
      - KDE 密度曲线 (红色叠加)

    args:
        data(DataFrame): 数据
        feature_cols(list[str]): 特征列名列表
        output_name(str): 输出名称前缀
    """
    n = len(feature_cols)
    if n == 0:
        return

    # 计算子图行列数
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    fig.suptitle(
        f"{output_name} — 特征分布 (直方图 + KDE)", fontsize=13, fontweight="bold"
    )

    # 确保 axes 总是二维数组
    if n == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    for i, col in enumerate(feature_cols):
        row_idx = i // cols
        col_idx = i % cols
        ax = axes[row_idx][col_idx]

        # 直方图 + KDE
        ax.hist(
            data[col],
            bins=30,
            color="steelblue",
            alpha=0.6,
            edgecolor="white",
            density=True,
        )
        data[col].plot.kde(ax=ax, color="coral", linewidth=1.5)

        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.grid(True, alpha=0.2)

    # 关闭多余的子图
    for i in range(n, rows * cols):
        row_idx = i // cols
        col_idx = i % cols
        axes[row_idx][col_idx].axis("off")

    fig.tight_layout()
    _save_fig(fig, "01_histogram_kde.png", output_name)


def _plot_boxplots(data: DataFrame, feature_cols: list[str], output_name: str) -> None:
    """
    为所有连续特征绘制箱线图

    箱线图可以直观展示:
      - 中位数 (箱中线)
      - 四分位距 IQR (箱体)
      - 异常值 (离群点)

    args:
        data(DataFrame): 数据
        feature_cols(list[str]): 特征列名列表
        output_name(str): 输出名称前缀
    """
    if len(feature_cols) == 0:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(feature_cols) * 0.8), 5))
    fig.suptitle(f"{output_name} — 箱线图", fontsize=13, fontweight="bold")

    # 数据可能量纲差异大，用标准化后的数据画箱线图
    plot_data = data[feature_cols]
    plot_data_norm = (plot_data - plot_data.mean()) / plot_data.std()

    ax.boxplot(
        [plot_data_norm[col].dropna().values for col in feature_cols],
        tick_labels=feature_cols,
        patch_artist=True,
        boxprops={"facecolor": "lightblue", "alpha": 0.7},
        medianprops={"color": "coral", "linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
    )

    ax.set_ylabel("标准化值")
    ax.grid(True, axis="y", alpha=0.2)

    # 特征名过长时旋转
    if any(len(col) > 8 for col in feature_cols):
        ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    _save_fig(fig, "02_boxplot.png", output_name)


def _plot_target_distribution(
    data: DataFrame, target_col: str, output_name: str, is_classification: bool = True
) -> None:
    """
    绘制目标变量分布图

    分类任务: 柱状图 (各类别样本数)
    回归任务: 直方图 + KDE (连续值分布)

    args:
        data(DataFrame): 数据
        target_col(str): 目标变量列名
        output_name(str): 输出名称前缀
        is_classification(bool): 是否为分类任务
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    if is_classification:
        # 分类: 柱状图
        counts = data[target_col].value_counts().sort_index()
        colors = sns.color_palette("Set2", len(counts))
        ax.bar(counts.index.astype(str), counts.values, color=colors)
        ax.set_xlabel("类别")
        ax.set_ylabel("样本数")
        fig.suptitle(f"{output_name} — 类别分布", fontsize=13, fontweight="bold")

        # 在柱子上方显示数量
        for i, (idx, val) in enumerate(counts.items()):
            ax.text(i, val + len(data) * 0.01, str(val), ha="center", fontsize=9)
    else:
        # 回归: 直方图 + KDE
        ax.hist(
            data[target_col],
            bins=40,
            color="steelblue",
            alpha=0.6,
            edgecolor="white",
            density=True,
        )
        data[target_col].plot.kde(ax=ax, color="coral", linewidth=1.5)
        ax.set_xlabel(target_col)
        ax.set_ylabel("密度")
        fig.suptitle(f"{output_name} — 目标变量分布", fontsize=13, fontweight="bold")

    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    _save_fig(fig, "03_target_distribution.png", output_name)


# --- 按数据集类型的绘图函数 ---


def _plot_classification_dist(
    data: DataFrame, name: str, short_name: str, target_col: str = "label"
) -> None:
    """
    分类数据集的分布图

    args:
        data(DataFrame): 分类数据集
        name(str): 显示名称
        short_name(str): 目录名 (简短英文)
        target_col(str): 目标变量列名
    """
    feature_cols = [c for c in data.columns if c != target_col]
    print(f"数据集: {name}")

    _plot_histograms(data, feature_cols, short_name)
    _plot_boxplots(data, feature_cols, short_name)
    _plot_target_distribution(data, target_col, short_name, is_classification=True)


def _plot_regression_dist(
    data: DataFrame, name: str, short_name: str, target_col: str = "price"
) -> None:
    """
    回归数据集的分布图

    args:
        data(DataFrame): 回归数据集
        name(str): 显示名称
        short_name(str): 目录名 (简短英文)
        target_col(str): 目标变量列名
    """
    feature_cols = [c for c in data.columns if c != target_col]
    print(f"数据集: {name}")

    _plot_histograms(data, feature_cols, short_name)
    _plot_boxplots(data, feature_cols, short_name)
    _plot_target_distribution(data, target_col, short_name, is_classification=False)


def _plot_clustering_dist(
    data: DataFrame, name: str, short_name: str, label_col: str = "true_label"
) -> None:
    """
    聚类数据集的分布图

    args:
        data(DataFrame): 聚类数据集
        name(str): 显示名称
        short_name(str): 目录名 (简短英文)
        label_col(str): 真实标签列名
    """
    feature_cols = [c for c in data.columns if c != label_col]
    print(f"数据集: {name}")

    _plot_histograms(data, feature_cols, short_name)
    _plot_boxplots(data, feature_cols, short_name)
    _plot_target_distribution(data, label_col, short_name, is_classification=True)


def _plot_sequence_dist(data: DataFrame, name: str, short_name: str) -> None:
    """
    序列数据集(HMM)的分布图

    绘制观测符号和隐状态的频率分布柱状图

    args:
        data(DataFrame): HMM 序列数据
        name(str): 显示名称
        short_name(str): 目录名 (简短英文)
    """
    print(f"数据集: {name}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{short_name} — 观测/隐状态分布", fontsize=13, fontweight="bold")

    # 观测符号分布
    obs_counts = data["obs"].value_counts().sort_index()
    axes[0].bar(obs_counts.index.astype(str), obs_counts.values, color="steelblue")
    axes[0].set_title("观测符号分布")
    axes[0].set_xlabel("观测符号")
    axes[0].set_ylabel("频次")
    axes[0].grid(True, alpha=0.2)

    # 隐状态分布
    state_counts = data["state_true"].value_counts().sort_index()
    axes[1].bar(state_counts.index.astype(str), state_counts.values, color="coral")
    axes[1].set_title("隐状态分布")
    axes[1].set_xlabel("隐状态")
    axes[1].set_ylabel("频次")
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    _save_fig(fig, "01_obs_state_distribution.png", short_name)


# --- 主入口 ---


def plot_distributions() -> None:
    """
    为所有 20 个数据集生成分布图

    所有图片直接输出到 outputs/data_visualization/distribution/
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
    print("生成分布图 (直方图 + 箱线图 + 目标变量分布)")
    print("=" * 50)

    # --- 分类 (6) ---

    _plot_classification_dist(
        logistic_regression_data,
        "LogisticRegression — 线性可分高维二分类",
        "logistic_regression",
    )
    _plot_classification_dist(
        decision_tree_classification_data,
        "DecisionTree — blob 多分类",
        "decision_tree_clf",
    )
    _plot_classification_dist(svc_data, "SVC — 同心圆二分类", "svc")
    _plot_classification_dist(naive_bayes_data, "NaiveBayes — Iris", "naive_bayes")
    _plot_classification_dist(knn_data, "KNN — 双月牙二分类", "knn")
    _plot_classification_dist(
        random_forest_data,
        "RandomForest — 高维多噪声三分类",
        "random_forest",
    )

    # --- 回归 (4) ---

    _plot_regression_dist(
        linear_regression_data,
        "LinearRegression — 手动合成线性房价",
        "linear_regression",
    )
    _plot_regression_dist(svr_data, "SVR — Friedman1 非线性回归", "svr")
    _plot_regression_dist(
        decision_tree_regression_data,
        "DecisionTree(回归) — California Housing",
        "decision_tree_reg",
    )
    _plot_regression_dist(
        regularization_data,
        "Regularization — 糖尿病+共线性+噪声",
        "regularization",
    )

    # --- 聚类 (2) ---

    _plot_clustering_dist(kmeans_data, "KMeans — 球形多簇", "kmeans")
    _plot_clustering_dist(dbscan_data, "DBSCAN — 双月牙非线性", "dbscan")

    # --- 集成 (4) ---

    _plot_classification_dist(bagging_data, "Bagging — 高噪声双月牙", "bagging")
    _plot_classification_dist(gbdt_data, "GBDT — 多类别中等难度", "gbdt")
    _plot_regression_dist(xgboost_data, "XGBoost — California Housing", "xgboost")
    _plot_classification_dist(lightgbm_data, "LightGBM — 高维四分类", "lightgbm")

    # --- 降维 (2) ---

    _plot_classification_dist(pca_data, "PCA — 高维低秩合成", "pca")
    _plot_classification_dist(lda_data, "LDA — Wine 数据集", "lda")

    # --- 概率 (2) ---

    _plot_clustering_dist(em_data, "EM(GMM) — 高斯混合模型", "em")
    _plot_sequence_dist(hmm_data, "HMM — 离散隐马尔可夫序列", "hmm")

    print("=" * 50)
    print("分布图生成完成, 共 20 个数据集")
    print("=" * 50)


# --- 直接运行 ---

if __name__ == "__main__":
    plot_distributions()
