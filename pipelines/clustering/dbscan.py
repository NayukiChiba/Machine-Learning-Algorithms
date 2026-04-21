"""
pipelines/clustering/dbscan.py
DBSCAN 聚类端到端流水线

运行方式: python -m pipelines.clustering.dbscan
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import get_model_output_dir
from data_exploration import (
    explore_clustering_bivariate,
    explore_clustering_multivariate,
    explore_clustering_univariate,
)
from data_generation import dbscan_data
from data_visualization import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_labeled_2d_scatter,
    plot_raw_2d_scatter,
)
from model_evaluation.clustering_metrics import evaluate_clustering_with_ground_truth
from model_training.clustering.dbscan import train_model
from result_visualization.cluster_plot import plot_clusters
from result_visualization.clustering_diagnostics import (
    plot_dbscan_eps_sweep,
    plot_dbscan_k_distance,
)

MODEL = "dbscan"
DBSCAN_EPS = 0.3
DBSCAN_MIN_SAMPLES = 5
DBSCAN_METRIC = "euclidean"


def align_cluster_labels_for_display(labels_pred, labels_true):
    """
    为“展示用途”对聚类标签做一个简单对齐

    聚类任务里的簇编号本身没有固定语义。
    比如真实标签是 {0, 1}，预测标签也可能是 {1, 0}，
    这并不代表聚类错了，只是编号刚好相反。

    为了让“结果展示图”和终端表格更直观，这里做一个多数表决映射：
    - 每个预测簇映射到它内部样本占比最高的真实标签；
    - 噪声点标签 -1 保持不变。

    注意：
    这一步只用于展示，不参与指标计算。
    ARI/NMI 等指标仍然使用原始预测标签，因为它们本身就对簇编号置换不敏感。
    """
    labels_pred = pd.Series(labels_pred)
    labels_true = pd.Series(labels_true)
    aligned = labels_pred.copy()

    for cluster_id in sorted(labels_pred.unique()):
        if cluster_id == -1:
            continue
        mask = labels_pred == cluster_id
        majority_true_label = labels_true[mask].value_counts().idxmax()
        aligned.loc[mask] = majority_true_label

    return aligned.to_numpy()


def show_data_exploration(data) -> None:
    """
    展示 DBSCAN 训练前的数据探索结果

    DBSCAN 这里使用的是双月牙非线性数据。
    这类数据非常适合做聚类教学，因为：
    1. 真实簇不是球形；
    2. KMeans 很容易失败；
    3. DBSCAN 的密度聚类思路更容易体现优势。
    """
    explore_clustering_univariate(
        data,
        dataset_name="DBSCAN",
    )
    explore_clustering_bivariate(
        data,
        dataset_name="DBSCAN",
    )
    explore_clustering_multivariate(
        data,
        dataset_name="DBSCAN",
    )


def show_data_preview(data, feature_names: list[str]) -> None:
    """
    展示 DBSCAN 训练前的数据图

    当前数据天然是二维，因此直接展示：
    1. 真实簇标签分布；
    2. 原始散点图；
    3. 相关性热力图。
    """
    save_dir = get_model_output_dir(MODEL)

    print("\n开始生成数据展示图...")
    plot_class_distribution(
        data,
        target_col="true_label",
        save_dir=save_dir,
        title="DBSCAN 数据展示：真实簇分布",
        filename="data_cluster_distribution.png",
    )
    plot_raw_2d_scatter(
        data,
        x_col=feature_names[0],
        y_col=feature_names[1],
        save_dir=save_dir,
        title="DBSCAN 数据展示：原始散点图",
        filename="data_raw_scatter.png",
    )
    plot_labeled_2d_scatter(
        data,
        x_col=feature_names[0],
        y_col=feature_names[1],
        label_col="true_label",
        save_dir=save_dir,
        title="DBSCAN 数据展示：真实标签散点图",
        filename="data_true_label_scatter.png",
    )
    plot_correlation_heatmap(
        data,
        columns=feature_names + ["true_label"],
        save_dir=save_dir,
        title="DBSCAN 数据展示：相关性热力图",
        filename="data_correlation.png",
    )
    print("数据展示图生成完成。")


def show_result_preview(X, y_true, labels_pred) -> None:
    """
    在终端展示部分样本的聚类结果

    这部分偏“结果展示”，主要帮助直接看：
    1. 某些点真实属于哪个簇；
    2. DBSCAN 最后把它分到了哪个簇；
    3. 是否被判成噪声点。
    """
    preview_size = min(10, len(X))
    preview_df = pd.DataFrame(X[:preview_size], columns=["x1", "x2"])
    preview_df["真实标签"] = y_true[:preview_size]
    preview_df["预测簇标签"] = labels_pred[:preview_size]
    preview_df["是否噪声点"] = preview_df["预测簇标签"] == -1

    print()
    print("=" * 60)
    print("DBSCAN 结果展示")
    print("=" * 60)
    print(preview_df.to_string(index=False))


def show_diagnostic_summary(diagnostic_df) -> None:
    """
    在终端展示 DBSCAN 的参数诊断摘要

    这部分不是最终聚类评估，而是回答：
    “当前 eps 设定大概处在什么位置？”

    逻辑上主要看三件事：
    1. 当前 eps 对应的簇数量；
    2. 当前 eps 对应的噪声点占比；
    3. 扫描范围内哪一个 eps 的 ARI / Silhouette 最好。
    """
    current_row = diagnostic_df.iloc[(diagnostic_df["eps"] - DBSCAN_EPS).abs().argmin()]

    best_ari_row = diagnostic_df.iloc[diagnostic_df["ari"].idxmax()]
    silhouette_candidates = diagnostic_df.dropna(subset=["silhouette"])
    best_silhouette_row = (
        silhouette_candidates.iloc[silhouette_candidates["silhouette"].idxmax()]
        if not silhouette_candidates.empty
        else None
    )

    print()
    print("=" * 60)
    print("DBSCAN 参数诊断摘要")
    print("=" * 60)
    print(f"当前 eps: {DBSCAN_EPS:.4f}")
    print(f"当前簇数量: {int(current_row['n_clusters'])}")
    print(f"当前噪声点占比: {current_row['noise_ratio']:.4f}")
    print(f"当前 ARI: {current_row['ari']:.4f}")
    print(f"当前 NMI: {current_row['nmi']:.4f}")
    if pd.notna(current_row["silhouette"]):
        print(f"当前 Silhouette: {current_row['silhouette']:.4f}")
    else:
        print("当前 Silhouette: 不可计算（簇数量不足或噪声过多）")

    print()
    print(
        f"扫描范围内 ARI 最佳 eps: {best_ari_row['eps']:.4f} "
        f"(ARI={best_ari_row['ari']:.4f})"
    )
    if best_silhouette_row is not None:
        print(
            f"扫描范围内 Silhouette 最佳 eps: {best_silhouette_row['eps']:.4f} "
            f"(Silhouette={best_silhouette_row['silhouette']:.4f})"
        )


def run():
    """
    DBSCAN 聚类完整流水线

    当前流程分成：
    1. 数据探索；
    2. 数据展示；
    3. 模型训练；
    4. 结果展示；
    5. 聚类评估。
    """
    print("=" * 60)
    print("DBSCAN 聚类流水线")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 第 1 步：读取当前算法对应的原始数据
    # ------------------------------------------------------------------
    # 这里使用的是双月牙非线性数据。
    # 这份数据非常适合 DBSCAN，因为：
    # 1. 两个簇的形状不是球形；
    # 2. 线性边界无法分开；
    # 3. DBSCAN 能按密度把弯月形簇识别出来。
    data = dbscan_data.copy()
    y_true = data["true_label"].values
    X = data.drop(columns=["true_label"])
    feature_names = list(X.columns)

    # ------------------------------------------------------------------
    # 第 2 步：数据探索
    # ------------------------------------------------------------------
    show_data_exploration(data)

    # ------------------------------------------------------------------
    # 第 3 步：数据展示
    # ------------------------------------------------------------------
    show_data_preview(data, feature_names)

    # ------------------------------------------------------------------
    # 第 4 步：预处理
    # ------------------------------------------------------------------
    # DBSCAN 依赖距离密度，因此尺度统一非常重要。
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------------------------
    # 第 5 步：参数诊断曲线
    # ------------------------------------------------------------------
    # 这里补上聚类任务中更有意义的“诊断曲线”：
    # 1. k-distance 曲线：辅助选择 eps；
    # 2. eps 扫描曲线：看 eps 改变时聚类结构和评估指标怎么变化。
    plot_dbscan_k_distance(
        X_scaled,
        min_samples=DBSCAN_MIN_SAMPLES,
        model_name=MODEL,
        current_eps=DBSCAN_EPS,
    )
    diagnostic_df = plot_dbscan_eps_sweep(
        X_scaled,
        y_true,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric=DBSCAN_METRIC,
        model_name=MODEL,
        current_eps=DBSCAN_EPS,
    )

    # ------------------------------------------------------------------
    # 第 6 步：训练
    # ------------------------------------------------------------------
    model = train_model(
        X_scaled,
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric=DBSCAN_METRIC,
    )
    labels_pred = model.labels_
    labels_pred_display = align_cluster_labels_for_display(labels_pred, y_true)

    # ------------------------------------------------------------------
    # 第 7 步：结果展示图
    # ------------------------------------------------------------------
    plot_clusters(
        X_scaled,
        labels_pred=labels_pred_display,
        labels_true=y_true,
        feature_names=[f"{name}（标准化）" for name in feature_names],
        title="DBSCAN 结果展示（对齐后预测标签）",
        model_name=MODEL,
    )

    # ------------------------------------------------------------------
    # 第 8 步：终端里的结果展示、参数诊断和模型评估
    # ------------------------------------------------------------------
    show_result_preview(X.values, y_true, labels_pred_display)
    show_diagnostic_summary(diagnostic_df)
    evaluate_clustering_with_ground_truth(
        X_scaled, labels_pred, y_true, print_report=True
    )

    print(f"\n{'=' * 60}")
    print("DBSCAN 流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
