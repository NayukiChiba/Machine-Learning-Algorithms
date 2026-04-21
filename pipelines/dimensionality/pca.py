"""
pipelines/dimensionality/pca.py
PCA 降维端到端流水线

运行方式: python -m pipelines.dimensionality.pca
"""

from sklearn.preprocessing import StandardScaler

from config import get_model_output_dir
from data_exploration import (
    explore_classification_bivariate,
    explore_classification_multivariate,
    explore_classification_univariate,
)
from data_generation import pca_data
from data_visualization import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_feature_space_2d,
    plot_feature_space_3d,
)
from model_evaluation.dimensionality_metrics import evaluate_dimensionality
from model_training.dimensionality.pca import train_model
from result_visualization.dimensionality_plot import plot_dimensionality
from result_visualization.dimensionality_diagnostics import plot_pca_training_curve

MODEL = "pca"


def show_data_exploration(data) -> None:
    """
    展示 PCA 训练前的数据探索结果

    当前这份数据是“高维低秩合成数据”，
    做 PCA 前最值得看的就是：
    1. 特征间整体相关性；
    2. 是否存在明显冗余结构；
    3. 从统计角度看是否存在降维潜力。
    """
    explore_classification_univariate(
        data,
        dataset_name="PCA",
    )
    explore_classification_bivariate(
        data,
        dataset_name="PCA",
    )
    explore_classification_multivariate(
        data,
        dataset_name="PCA",
    )


def show_data_preview(data, feature_names: list[str]) -> None:
    """
    展示 PCA 训练前的数据图

    这里的数据本身是高维的，因此用：
    1. 类别分布图；
    2. 相关性热力图；
    3. 原始数据的 PCA 2D/3D 特征空间图。
    """
    save_dir = get_model_output_dir(MODEL)

    print("\n开始生成数据展示图...")
    plot_class_distribution(
        data,
        target_col="label",
        save_dir=save_dir,
        title="PCA 数据展示：类别分布",
        filename="data_class_distribution.png",
    )
    plot_correlation_heatmap(
        data,
        columns=feature_names + ["label"],
        save_dir=save_dir,
        title="PCA 数据展示：相关性热力图",
        filename="data_correlation.png",
    )
    plot_feature_space_2d(
        data,
        feature_cols=feature_names,
        label_col="label",
        save_dir=save_dir,
        title="PCA 数据展示：原始数据 2D 特征空间",
        filename="data_feature_space_2d.png",
    )
    plot_feature_space_3d(
        data,
        feature_cols=feature_names,
        label_col="label",
        save_dir=save_dir,
        title="PCA 数据展示：原始数据 3D 特征空间",
        filename="data_feature_space_3d.png",
    )
    print("数据展示图生成完成。")


def show_result_preview(X_transformed_2d, y) -> None:
    """
    在终端展示部分样本的 PCA 2D 结果

    这部分属于“结果展示”，目的不是看分类是否正确，
    而是直接看一些样本在主成分空间里被投影到了哪里。
    """
    preview_size = min(8, len(X_transformed_2d))
    preview_rows = []
    for index in range(preview_size):
        preview_rows.append(
            {
                "PC1": round(float(X_transformed_2d[index, 0]), 4),
                "PC2": round(float(X_transformed_2d[index, 1]), 4),
                "标签": int(y[index]),
            }
        )

    print()
    print("=" * 60)
    print("PCA 结果展示")
    print("=" * 60)
    for row in preview_rows:
        print(row)


def show_model_evaluation(metrics_2d, metrics_3d, training_curve_data) -> None:
    """
    在终端展示 PCA 的模型评估结果

    PCA 是无监督降维模型，因此这里关注的是：
    1. 解释方差比；
    2. 累计解释方差；
    3. 重建误差；
    4. 主成分数量变化时的诊断趋势。
    """
    print()
    print("=" * 60)
    print("PCA 模型评估展示")
    print("=" * 60)
    print("2D PCA:")
    print(f"  总解释方差比: {metrics_2d['total_explained_variance']:.4f}")
    print(f"  解释方差比: {metrics_2d['explained_variance_ratio'].round(4).tolist()}")
    print(
        f"  累计解释方差比: {metrics_2d['cumulative_variance_ratio'].round(4).tolist()}"
    )
    if "reconstruction_error" in metrics_2d:
        print(f"  重建误差(MSE): {metrics_2d['reconstruction_error']:.6f}")

    print("3D PCA:")
    print(f"  总解释方差比: {metrics_3d['total_explained_variance']:.4f}")
    print(f"  解释方差比: {metrics_3d['explained_variance_ratio'].round(4).tolist()}")
    print(
        f"  累计解释方差比: {metrics_3d['cumulative_variance_ratio'].round(4).tolist()}"
    )
    if "reconstruction_error" in metrics_3d:
        print(f"  重建误差(MSE): {metrics_3d['reconstruction_error']:.6f}")

    best_idx = int(training_curve_data["cumulative_variances"].argmax())
    best_n = int(training_curve_data["component_range"][best_idx])
    print()
    print("训练诊断摘要:")
    print(f"  最大扫描主成分数: {int(training_curve_data['component_range'][-1])}")
    print(
        f"  最终累计解释方差比: {training_curve_data['cumulative_variances'][-1]:.4f}"
    )
    print(f"  最终重建误差: {training_curve_data['reconstruction_errors'][-1]:.6f}")
    print(f"  扫描范围内累计解释方差最高的主成分数: {best_n}")


def run():
    """
    PCA 降维完整流水线

    当前流程分成：
    1. 数据探索；
    2. 数据展示；
    3. 模型训练与降维；
    4. 结果展示；
    5. 模型评估；
    6. 训练诊断曲线。
    """
    print("=" * 60)
    print("PCA 降维流水线")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 第 1 步：读取当前算法对应的原始数据
    # ------------------------------------------------------------------
    # 当前使用的是高维低秩合成数据：
    # 1. 表面上是 10 维；
    # 2. 但真正有信息的主方向只有少数几个；
    # 3. 非常适合用来展示 PCA 的降维能力。
    data = pca_data.copy()
    X = data.drop(columns=["label"])
    y = data["label"].values
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
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------------------------
    # 第 5 步：训练诊断曲线
    # ------------------------------------------------------------------
    training_curve_data = plot_pca_training_curve(
        X_scaled,
        model_name=MODEL,
        max_components=X_scaled.shape[1],
    )

    # ------------------------------------------------------------------
    # 第 6 步：训练模型并做降维
    # ------------------------------------------------------------------
    model = train_model(X_scaled, n_components=2)
    X_transformed = model.transform(X_scaled)
    plot_dimensionality(
        X_transformed,
        y=y,
        explained_variance_ratio=model.explained_variance_ratio_,
        title="PCA 降维 (2D)",
        model_name=MODEL,
        mode="2d",
    )

    # 3D PCA
    model_3d = train_model(X_scaled, n_components=3)
    X_3d = model_3d.transform(X_scaled)
    plot_dimensionality(
        X_3d,
        y=y,
        explained_variance_ratio=model_3d.explained_variance_ratio_,
        title="PCA 降维 (3D)",
        model_name=MODEL,
        mode="3d",
    )

    # ------------------------------------------------------------------
    # 第 7 步：结果展示
    # ------------------------------------------------------------------
    show_result_preview(X_transformed, y)

    # ------------------------------------------------------------------
    # 第 8 步：模型评估
    # ------------------------------------------------------------------
    metrics_2d = evaluate_dimensionality(
        model,
        X_original=X_scaled,
        X_transformed=X_transformed,
        print_report=False,
    )
    metrics_3d = evaluate_dimensionality(
        model_3d,
        X_original=X_scaled,
        X_transformed=X_3d,
        print_report=False,
    )
    show_model_evaluation(metrics_2d, metrics_3d, training_curve_data)

    print(f"\n{'=' * 60}")
    print("PCA 流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
