"""
pipelines/clustering/kmeans.py
KMeans 聚类端到端流水线

运行方式: python -m pipelines.clustering.kmeans
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import get_model_output_dir
from data_exploration import (
    explore_clustering_bivariate,
    explore_clustering_multivariate,
    explore_clustering_univariate,
)
from data_generation import kmeans_data
from data_visualization import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_labeled_2d_scatter,
    plot_raw_2d_scatter,
)
from model_evaluation.clustering_metrics import evaluate_clustering_with_ground_truth
from model_training.clustering.kmeans import train_model
from result_visualization.cluster_plot import plot_clusters
from result_visualization.clustering_diagnostics import plot_kmeans_k_sweep

MODEL = "kmeans"
KMEANS_N_CLUSTERS = 4
KMEANS_INIT = "k-means++"
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 300


def align_cluster_labels_for_display(labels_pred, labels_true):
    """
    为“展示用途”对聚类标签做一个简单对齐

    聚类簇编号本身没有固定语义，因此这里只做用于展示的多数表决映射。
    注意：
    这一步只影响“结果展示”，不影响真实评估指标。
    """
    labels_pred = pd.Series(labels_pred)
    labels_true = pd.Series(labels_true)
    aligned = labels_pred.copy()

    for cluster_id in sorted(labels_pred.unique()):
        mask = labels_pred == cluster_id
        majority_true_label = labels_true[mask].value_counts().idxmax()
        aligned.loc[mask] = majority_true_label

    return aligned.to_numpy()


def show_data_exploration(data) -> None:
    """
    展示 KMeans 训练前的数据探索结果

    KMeans 当前使用的是球形多簇数据。
    这类数据很适合 KMeans，因为：
    1. 各簇近似球形；
    2. 各簇尺度相近；
    3. 真实簇中心比较清晰。
    """
    explore_clustering_univariate(
        data,
        dataset_name="KMeans",
    )
    explore_clustering_bivariate(
        data,
        dataset_name="KMeans",
    )
    explore_clustering_multivariate(
        data,
        dataset_name="KMeans",
    )


def show_data_preview(data, feature_names: list[str]) -> None:
    """
    展示 KMeans 训练前的数据图

    当前数据天然是二维，因此直接展示：
    1. 真实簇标签分布；
    2. 原始散点图；
    3. 真实标签散点图；
    4. 相关性热力图。
    """
    save_dir = get_model_output_dir(MODEL)

    print("\n开始生成数据展示图...")
    plot_class_distribution(
        data,
        target_col="true_label",
        save_dir=save_dir,
        title="KMeans 数据展示：真实簇分布",
        filename="data_cluster_distribution.png",
    )
    plot_raw_2d_scatter(
        data,
        x_col=feature_names[0],
        y_col=feature_names[1],
        save_dir=save_dir,
        title="KMeans 数据展示：原始散点图",
        filename="data_raw_scatter.png",
    )
    plot_labeled_2d_scatter(
        data,
        x_col=feature_names[0],
        y_col=feature_names[1],
        label_col="true_label",
        save_dir=save_dir,
        title="KMeans 数据展示：真实标签散点图",
        filename="data_true_label_scatter.png",
    )
    plot_correlation_heatmap(
        data,
        columns=feature_names + ["true_label"],
        save_dir=save_dir,
        title="KMeans 数据展示：相关性热力图",
        filename="data_correlation.png",
    )
    print("数据展示图生成完成。")


def show_result_preview(X, y_true, labels_pred) -> None:
    """
    在终端展示部分样本的聚类结果

    用于帮助直接看：
    1. 某些点真实属于哪个簇；
    2. KMeans 最后把它分到了哪个簇。
    """
    preview_size = min(10, len(X))
    preview_df = pd.DataFrame(X[:preview_size], columns=["x1", "x2"])
    preview_df["真实标签"] = y_true[:preview_size]
    preview_df["预测簇标签"] = labels_pred[:preview_size]

    print()
    print("=" * 60)
    print("KMeans 结果展示")
    print("=" * 60)
    print(preview_df.to_string(index=False))


def show_diagnostic_summary(diagnostic_df) -> None:
    """
    在终端展示 KMeans 的参数诊断摘要

    这里重点看三件事：
    1. 当前 k 对应的 inertia；
    2. 当前 k 对应的 ARI / NMI / Silhouette；
    3. 扫描范围内哪个 k 的指标最好。
    """
    current_row = diagnostic_df.loc[diagnostic_df["k"] == KMEANS_N_CLUSTERS].iloc[0]
    best_ari_row = diagnostic_df.iloc[diagnostic_df["ari"].idxmax()]
    best_silhouette_row = diagnostic_df.iloc[diagnostic_df["silhouette"].idxmax()]

    print()
    print("=" * 60)
    print("KMeans 参数诊断摘要")
    print("=" * 60)
    print(f"当前 k: {KMEANS_N_CLUSTERS}")
    print(f"当前 Inertia: {current_row['inertia']:.4f}")
    print(f"当前 ARI: {current_row['ari']:.4f}")
    print(f"当前 NMI: {current_row['nmi']:.4f}")
    print(f"当前 Silhouette: {current_row['silhouette']:.4f}")
    print()
    print(
        f"扫描范围内 ARI 最佳 k: {int(best_ari_row['k'])} "
        f"(ARI={best_ari_row['ari']:.4f})"
    )
    print(
        f"扫描范围内 Silhouette 最佳 k: {int(best_silhouette_row['k'])} "
        f"(Silhouette={best_silhouette_row['silhouette']:.4f})"
    )


def run():
    """
    KMeans 聚类完整流水线

    当前流程分成：
    1. 数据探索；
    2. 数据展示；
    3. 参数诊断；
    4. 模型训练；
    5. 结果展示；
    6. 聚类评估。
    """
    print("=" * 60)
    print("KMeans 聚类流水线")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 第 1 步：读取当前算法对应的原始数据
    # ------------------------------------------------------------------
    # 这里使用的是球形多簇数据。
    # 这份数据和 KMeans 的假设最契合，因此非常适合用来展示：
    # 1. KMeans 对球形簇的适应性；
    # 2. 聚类中心的意义；
    # 3. inertia / silhouette 等指标的诊断作用。
    data = kmeans_data.copy()
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
    # KMeans 基于欧氏距离，因此不同量纲的特征需要先统一尺度。
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------------------------
    # 第 5 步：参数诊断曲线
    # ------------------------------------------------------------------
    diagnostic_df = plot_kmeans_k_sweep(
        X_scaled,
        y_true,
        model_name=MODEL,
        current_k=KMEANS_N_CLUSTERS,
        init=KMEANS_INIT,
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
        random_state=42,
    )

    # ------------------------------------------------------------------
    # 第 6 步：训练
    # ------------------------------------------------------------------
    model = train_model(
        X_scaled,
        n_clusters=KMEANS_N_CLUSTERS,
        init=KMEANS_INIT,
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
        random_state=42,
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
        centers=model.cluster_centers_,
        feature_names=[f"{name}（标准化）" for name in feature_names],
        title="KMeans 结果展示（对齐后预测标签）",
        model_name=MODEL,
    )

    # ------------------------------------------------------------------
    # 第 8 步：终端里的结果展示、参数诊断和模型评估
    # ------------------------------------------------------------------
    show_result_preview(X.values, y_true, labels_pred_display)
    show_diagnostic_summary(diagnostic_df)
    evaluate_clustering_with_ground_truth(
        X_scaled,
        labels_pred,
        y_true,
        inertia=model.inertia_,
        print_report=True,
    )

    print(f"\n{'=' * 60}")
    print("KMeans 流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
