"""
pipelines/probabilistic/em.py
EM (GMM) 聚类端到端流水线

运行方式: python -m pipelines.probabilistic.em
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import get_model_output_dir
from data_exploration import (
    explore_clustering_bivariate,
    explore_clustering_multivariate,
    explore_clustering_univariate,
)
from data_generation import em_data
from data_visualization import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_labeled_2d_scatter,
    plot_raw_2d_scatter,
)
from model_evaluation.clustering_metrics import evaluate_clustering_with_ground_truth
from model_training.probabilistic.em import train_model
from result_visualization.cluster_plot import plot_clusters
from result_visualization.clustering_diagnostics import plot_gmm_component_sweep

MODEL = "gmm"
GMM_COMPONENTS = 3
GMM_COVARIANCE_TYPE = "full"
GMM_MAX_ITER = 200


def align_cluster_labels_for_display(labels_pred, labels_true):
    """
    为“展示用途”对聚类标签做多数表决映射

    聚类编号本身没有固定语义。
    为了让结果展示图和终端表格更直观，
    这里把每个预测簇映射到它内部占比最高的真实标签。

    注意：
    这一步只服务于展示，不影响 ARI/NMI 等真正的评估指标。
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
    展示 EM(GMM) 训练前的数据探索结果

    当前数据是手工合成的高斯混合数据。
    这类数据很适合用来展示：
    1. 同一类簇可以有不同形状；
    2. GMM 比 KMeans 更适合拟合椭圆形簇；
    3. EM 的“概率分配”思路比硬分配更灵活。
    """
    explore_clustering_univariate(
        data,
        dataset_name="EM(GMM)",
    )
    explore_clustering_bivariate(
        data,
        dataset_name="EM(GMM)",
    )
    explore_clustering_multivariate(
        data,
        dataset_name="EM(GMM)",
    )


def show_data_preview(data, feature_names: list[str]) -> None:
    """
    展示 EM(GMM) 训练前的数据图

    当前数据天然是二维，因此直接展示：
    1. 真实簇分布；
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
        title="EM(GMM) 数据展示：真实簇分布",
        filename="data_cluster_distribution.png",
    )
    plot_raw_2d_scatter(
        data,
        x_col=feature_names[0],
        y_col=feature_names[1],
        save_dir=save_dir,
        title="EM(GMM) 数据展示：原始散点图",
        filename="data_raw_scatter.png",
    )
    plot_labeled_2d_scatter(
        data,
        x_col=feature_names[0],
        y_col=feature_names[1],
        label_col="true_label",
        save_dir=save_dir,
        title="EM(GMM) 数据展示：真实标签散点图",
        filename="data_true_label_scatter.png",
    )
    plot_correlation_heatmap(
        data,
        columns=feature_names + ["true_label"],
        save_dir=save_dir,
        title="EM(GMM) 数据展示：相关性热力图",
        filename="data_correlation.png",
    )
    print("数据展示图生成完成。")


def show_result_preview(X, y_true, labels_pred, probabilities) -> None:
    """
    在终端展示部分样本的聚类结果

    EM(GMM) 和 KMeans/DBSCAN 的一个重要区别是：
    它不仅给出最终簇标签，还能给出“属于每个簇的概率”。
    因此这里把最大后验概率一起展示出来。
    """
    preview_size = min(10, len(X))
    preview_df = pd.DataFrame(X[:preview_size], columns=["x1", "x2"])
    preview_df["真实标签"] = y_true[:preview_size]
    preview_df["预测簇标签"] = labels_pred[:preview_size]
    preview_df["最大后验概率"] = probabilities[:preview_size].max(axis=1).round(4)

    print()
    print("=" * 60)
    print("EM(GMM) 结果展示")
    print("=" * 60)
    print(preview_df.to_string(index=False))


def show_model_evaluation(model, X_scaled, labels_pred, y_true) -> None:
    """
    在终端展示 EM(GMM) 的模型评估结果

    除了 ARI / NMI / Silhouette 这类常规聚类指标，
    这里还会输出：
    1. 混合权重
    2. 分量均值
    3. 对数似然下界
    这些都非常适合解释 GMM 的结果。
    """
    evaluate_clustering_with_ground_truth(
        X_scaled,
        labels_pred,
        y_true,
        print_report=True,
    )

    print()
    print("GMM 参数摘要")
    print("-" * 60)
    print(f"混合权重: {model.weights_.round(4).tolist()}")
    print("各分量中心:")
    for index, center in enumerate(model.means_):
        print(f"  分量 {index}: {center.round(4).tolist()}")
    print(f"对数似然下界: {model.lower_bound_:.6f}")


def show_diagnostic_summary(diagnostic_df) -> None:
    """
    在终端展示 GMM 的分量数诊断摘要

    这里重点看：
    1. 当前分量数对应的 BIC/AIC；
    2. 当前分量数对应的 ARI/NMI/Silhouette；
    3. 扫描范围内哪个分量数最好。
    """
    current_row = diagnostic_df.loc[
        diagnostic_df["n_components"] == GMM_COMPONENTS
    ].iloc[0]
    best_bic_row = diagnostic_df.iloc[diagnostic_df["bic"].idxmin()]
    best_ari_row = diagnostic_df.iloc[diagnostic_df["ari"].idxmax()]
    best_silhouette_row = diagnostic_df.iloc[diagnostic_df["silhouette"].idxmax()]

    print()
    print("=" * 60)
    print("EM(GMM) 参数诊断摘要")
    print("=" * 60)
    print(f"当前分量数: {GMM_COMPONENTS}")
    print(f"当前 BIC: {current_row['bic']:.4f}")
    print(f"当前 AIC: {current_row['aic']:.4f}")
    print(f"当前 lower bound: {current_row['lower_bound']:.4f}")
    print(f"当前 ARI: {current_row['ari']:.4f}")
    print(f"当前 NMI: {current_row['nmi']:.4f}")
    if pd.notna(current_row["silhouette"]):
        print(f"当前 Silhouette: {current_row['silhouette']:.4f}")
    else:
        print("当前 Silhouette: 不可计算")
    print()
    print(
        f"扫描范围内 BIC 最优分量数: {int(best_bic_row['n_components'])} "
        f"(BIC={best_bic_row['bic']:.4f})"
    )
    print(
        f"扫描范围内 ARI 最优分量数: {int(best_ari_row['n_components'])} "
        f"(ARI={best_ari_row['ari']:.4f})"
    )
    print(
        f"扫描范围内 Silhouette 最优分量数: {int(best_silhouette_row['n_components'])} "
        f"(Silhouette={best_silhouette_row['silhouette']:.4f})"
    )


def run():
    """
    EM (GMM) 聚类完整流水线

    当前流程分成：
    1. 数据探索；
    2. 数据展示；
    3. 模型训练；
    4. 结果展示；
    5. 模型评估。
    """
    print("=" * 60)
    print("EM (GMM) 聚类流水线")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 第 1 步：读取当前算法对应的原始数据
    # ------------------------------------------------------------------
    # 当前使用的是手工合成的高斯混合数据。
    # 这类数据非常适合展示 EM / GMM 的优势：
    # 1. 簇形状不必是球形；
    # 2. 各簇方差可以不同；
    # 3. 每个样本都能有软概率归属。
    data = em_data.copy()
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
    # GMM 基于概率密度和协方差估计，尺度统一有助于更稳定地拟合。
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------------------------
    # 第 5 步：模型评估曲线
    # ------------------------------------------------------------------
    diagnostic_df = plot_gmm_component_sweep(
        X_scaled,
        y_true,
        model_name=MODEL,
        current_components=GMM_COMPONENTS,
        covariance_type=GMM_COVARIANCE_TYPE,
        max_iter=GMM_MAX_ITER,
        random_state=42,
    )

    # ------------------------------------------------------------------
    # 第 6 步：训练
    # ------------------------------------------------------------------
    model = train_model(
        X_scaled,
        n_components=GMM_COMPONENTS,
        covariance_type=GMM_COVARIANCE_TYPE,
        max_iter=GMM_MAX_ITER,
    )
    labels_pred = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    labels_pred_display = align_cluster_labels_for_display(labels_pred, y_true)

    # ------------------------------------------------------------------
    # 第 7 步：结果展示图
    # ------------------------------------------------------------------
    plot_clusters(
        X_scaled,
        labels_pred=labels_pred_display,
        labels_true=y_true,
        feature_names=[f"{name}（标准化）" for name in feature_names],
        centers=model.means_,
        title="EM (GMM) 结果展示（对齐后预测标签）",
        model_name=MODEL,
        filename="result_display.png",
    )

    # ------------------------------------------------------------------
    # 第 8 步：终端里的结果展示、参数诊断和模型评估
    # ------------------------------------------------------------------
    show_result_preview(X.values, y_true, labels_pred_display, probabilities)
    show_diagnostic_summary(diagnostic_df)
    show_model_evaluation(model, X_scaled, labels_pred, y_true)

    print(f"\n{'=' * 60}")
    print("EM (GMM) 流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
