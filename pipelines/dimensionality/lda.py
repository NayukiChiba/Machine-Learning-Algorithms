"""
pipelines/dimensionality/lda.py
LDA 降维端到端流水线

运行方式: python -m pipelines.dimensionality.lda
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import get_model_output_dir
from data_exploration import (
    explore_classification_bivariate,
    explore_classification_multivariate,
    explore_classification_univariate,
)
from data_generation import lda_data
from data_visualization import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_feature_space_2d,
)
from model_evaluation.dimensionality_metrics import evaluate_dimensionality
from model_training.dimensionality.lda import train_model
from result_visualization.classification_result import plot_classification_result
from result_visualization.confusion_matrix import plot_confusion_matrix
from result_visualization.dimensionality_plot import plot_dimensionality
from result_visualization.learning_curve import plot_learning_curve
from result_visualization.roc_curve import plot_roc_curve

MODEL = "lda"


def build_learning_curve_model() -> LinearDiscriminantAnalysis:
    """
    构造与主模型参数一致的学习曲线模型

    LDA 虽然不是按 epoch 训练的模型，但仍然可以画“样本数变化 -> 训练/验证得分变化”的学习曲线。
    这类曲线更适合用来观察：
    1. 数据量增加时模型是否稳定；
    2. 当前判别方向是否容易过拟合；
    3. 训练集和验证集的差距是否合理。
    """
    # 对学习曲线来说，这里更关心“分类效果随样本量变化”，
    # 而不是固定拿 2 个判别轴做 transform。
    # 因此 n_components 不显式写死，避免交叉验证某些切分里因为类别数不足而报错。
    return LinearDiscriminantAnalysis(solver="svd")


def show_data_exploration(data) -> None:
    """
    展示 LDA 训练前的数据探索结果

    LDA 是监督式降维，因此标签信息本身就是模型的重要组成部分。
    所以在训练前的数据探索阶段，值得重点关注：
    1. 类别是否均衡；
    2. 各特征是否具备类别区分能力；
    3. 特征间是否存在明显冗余；
    4. 从统计角度看，数据是否适合被压缩到更低维。
    """
    explore_classification_univariate(
        data,
        dataset_name="LDA",
    )
    explore_classification_bivariate(
        data,
        dataset_name="LDA",
    )
    explore_classification_multivariate(
        data,
        dataset_name="LDA",
    )


def show_data_preview(data, feature_names: list[str]) -> None:
    """
    展示 LDA 训练前的数据图

    Wine 数据是高维多分类数据，因此这里采用：
    1. 类别分布图；
    2. 相关性热力图；
    3. 原始高维数据的 PCA 2D 特征空间图。
    """
    save_dir = get_model_output_dir(MODEL)

    print("\n开始生成数据展示图...")
    plot_class_distribution(
        data,
        target_col="label",
        save_dir=save_dir,
        title="LDA 数据展示：类别分布",
        filename="data_class_distribution.png",
    )
    plot_correlation_heatmap(
        data,
        columns=feature_names + ["label"],
        save_dir=save_dir,
        title="LDA 数据展示：相关性热力图",
        filename="data_correlation.png",
    )
    plot_feature_space_2d(
        data,
        feature_cols=feature_names,
        label_col="label",
        save_dir=save_dir,
        title="LDA 数据展示：原始数据 PCA 2D 特征空间",
        filename="data_feature_space_2d.png",
    )
    print("数据展示图生成完成。")


def show_result_preview(X_test_lda, y_test, y_pred) -> None:
    """
    在终端展示部分测试样本的降维结果和预测结果

    这里的“结果展示”不是只看分类对不对，
    还会顺手把样本在 LDA 空间中的投影坐标打出来，
    方便理解 LDA 是如何把类别拉开的。
    """
    preview_size = min(8, len(X_test_lda))
    preview_rows = []
    for index in range(preview_size):
        preview_rows.append(
            {
                "LD1": round(float(X_test_lda[index, 0]), 4),
                "LD2": round(float(X_test_lda[index, 1]), 4),
                "真实标签": int(y_test[index]),
                "预测标签": int(y_pred[index]),
            }
        )

    print()
    print("=" * 60)
    print("LDA 结果展示")
    print("=" * 60)
    for row in preview_rows:
        print(row)


def show_model_evaluation(
    model, X_test_s, X_test_lda, y_test, y_pred, y_scores
) -> None:
    """
    在终端展示 LDA 的模型评估结果

    这里会同时展示两类信息：
    1. 降维质量：解释方差比、累计解释方差；
    2. 分类效果：准确率、混淆矩阵、分类报告。
    """
    metrics = evaluate_dimensionality(
        model,
        X_original=X_test_s,
        X_transformed=X_test_lda,
        print_report=False,
    )
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_scores, multi_class="ovr")
    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, digits=4, zero_division=0)

    print()
    print("=" * 60)
    print("LDA 模型评估展示")
    print("=" * 60)
    print(f"测试集分类准确率: {accuracy:.4f}")
    print(f"多分类 AUC(OVR): {auc_score:.4f}")
    if "total_explained_variance" in metrics:
        print(f"总解释方差比: {metrics['total_explained_variance']:.4f}")
        print(f"解释方差比: {metrics['explained_variance_ratio'].round(4).tolist()}")
        print(
            f"累计解释方差比: {metrics['cumulative_variance_ratio'].round(4).tolist()}"
        )
    if "reconstruction_error" in metrics:
        print(f"重建误差(MSE): {metrics['reconstruction_error']:.6f}")
    print()
    print("混淆矩阵原始计数:")
    print(cm)
    print()
    print("分类报告:")
    print(report_text)


def run():
    """
    LDA 降维完整流水线

    当前流程分成：
    1. 数据探索；
    2. 数据展示；
    3. 模型训练与降维；
    4. 结果展示；
    5. 模型评估。
    """
    print("=" * 60)
    print("LDA 降维流水线")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 第 1 步：读取当前算法对应的原始数据
    # ------------------------------------------------------------------
    # 当前使用的是 Wine 真实数据集：
    # 1. 13 个化学成分特征；
    # 2. 3 个类别；
    # 3. 很适合展示监督式降维如何把类别在低维空间中拉开。
    data = lda_data.copy()
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
    # LDA 对尺度较敏感，因此先标准化后再做投影。
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------------------------
    # 第 5 步：切分训练集和测试集
    # ------------------------------------------------------------------
    # 这里额外做训练/测试切分，是因为当前不仅要展示降维结果，
    # 还要评价 LDA 作为分类器时的预测效果。
    X_train_s, X_test_s, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # ------------------------------------------------------------------
    # 第 6 步：训练模型并做降维
    # ------------------------------------------------------------------
    model = train_model(X_train_s, y_train, n_components=2)
    X_test_lda = model.transform(X_test_s)
    X_transformed_all = model.transform(X_scaled)
    y_pred = model.predict(X_test_s)
    y_scores = model.predict_proba(X_test_s)

    evr = (
        model.explained_variance_ratio_
        if hasattr(model, "explained_variance_ratio_")
        else None
    )

    # ------------------------------------------------------------------
    # 第 7 步：结果图展示
    # ------------------------------------------------------------------
    # 这里分别展示：
    # 1. 全体样本在 LDA 2D 空间中的分布；
    # 2. 测试集真实标签 vs 预测标签在 LDA 2D 空间中的对比；
    # 3. 测试集分类混淆矩阵。
    plot_dimensionality(
        X_transformed_all,
        y=y,
        explained_variance_ratio=evr,
        title="LDA 降维 (2D)",
        model_name=MODEL,
        mode="2d",
    )
    plot_classification_result(
        X_test_lda,
        y_test,
        y_pred,
        feature_names=["LD1", "LD2"],
        title="LDA 结果展示",
        model_name=MODEL,
    )
    plot_confusion_matrix(
        y_test,
        y_pred,
        title="LDA 混淆矩阵",
        model_name=MODEL,
    )
    plot_roc_curve(
        y_test,
        y_scores,
        title="LDA ROC 曲线",
        model_name=MODEL,
    )
    plot_learning_curve(
        build_learning_curve_model(),
        X_train_s,
        y_train,
        title="LDA 学习曲线",
        model_name=MODEL,
    )

    # ------------------------------------------------------------------
    # 第 8 步：终端里的结果展示和模型评估
    # ------------------------------------------------------------------
    show_result_preview(X_test_lda, y_test, y_pred)
    show_model_evaluation(model, X_test_s, X_test_lda, y_test, y_pred, y_scores)

    print(f"\n{'=' * 60}")
    print("LDA 流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
