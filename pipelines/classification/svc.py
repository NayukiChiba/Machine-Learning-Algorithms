"""
pipelines/classification/svc.py
SVC 分类端到端流水线

运行方式: python -m pipelines.classification.svc
"""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC as SVC_Model

from config import get_model_output_dir
from data_exploration import (
    explore_classification_bivariate,
    explore_classification_multivariate,
    explore_classification_univariate,
)
from data_generation import svc_data
from data_visualization import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_labeled_2d_scatter,
)
from model_training.classification.svc import train_model
from result_visualization.classification_result import plot_classification_result
from result_visualization.confusion_matrix import plot_confusion_matrix
from result_visualization.decision_boundary import plot_decision_boundary
from result_visualization.learning_curve import plot_learning_curve
from result_visualization.roc_curve import plot_roc_curve

MODEL = "svc"


def build_learning_curve_model() -> SVC_Model:
    """
    构造与主模型参数一致的学习曲线模型

    SVC 对核函数和参数组合非常敏感，
    因此学习曲线模型必须与主模型保持同样的核心配置。
    """
    return SVC_Model(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        random_state=42,
    )


def show_data_exploration(data) -> None:
    """
    展示 SVC 训练前的数据探索结果

    SVC 当前使用的是同心圆二分类数据。
    这类数据的特点是：
    1. 线性不可分；
    2. 局部几何结构很明显；
    3. 很适合展示核方法的作用。
    """
    explore_classification_univariate(
        data,
        dataset_name="SVC",
    )
    explore_classification_bivariate(
        data,
        dataset_name="SVC",
    )
    explore_classification_multivariate(
        data,
        dataset_name="SVC",
    )


def show_data_preview(data, feature_names: list[str]) -> None:
    """
    展示 SVC 训练前的数据图

    当前数据天然就是二维，因此最合适的展示方式是：
    1. 类别分布图；
    2. 原始散点图；
    3. 相关性热力图。
    """
    save_dir = get_model_output_dir(MODEL)

    print("\n开始生成数据展示图...")
    plot_class_distribution(
        data,
        target_col="label",
        save_dir=save_dir,
        title="SVC 数据展示：类别分布",
        filename="data_class_distribution.png",
    )
    plot_labeled_2d_scatter(
        data,
        x_col=feature_names[0],
        y_col=feature_names[1],
        label_col="label",
        save_dir=save_dir,
        title="SVC 数据展示：原始散点图",
        filename="data_scatter.png",
    )
    plot_correlation_heatmap(
        data,
        columns=feature_names + ["label"],
        save_dir=save_dir,
        title="SVC 数据展示：相关性热力图",
        filename="data_correlation.png",
    )
    print("数据展示图生成完成。")


def show_prediction_examples(X_test, y_test, y_pred, decision_scores) -> None:
    """
    在终端展示部分测试样本的预测结果

    对 SVC 来说，除了预测标签，还值得看 decision score。
    score 离 0 越远，表示模型判别得越“有把握”。
    """
    preview_size = min(8, len(X_test))
    preview_df = X_test.reset_index(drop=True).iloc[:preview_size].copy()
    preview_df["真实标签"] = y_test.reset_index(drop=True).iloc[:preview_size].values
    preview_df["预测标签"] = y_pred[:preview_size]
    preview_df["decision_score"] = decision_scores[:preview_size].round(4)

    print()
    print("=" * 60)
    print("SVC 结果展示")
    print("=" * 60)
    print(preview_df.to_string(index=False))


def show_model_evaluation(model, y_test, y_pred, decision_scores) -> None:
    """
    在终端展示 SVC 的模型评估结果

    对 SVC 而言，支持向量数量本身就是关键解释信息，
    因此这里会把它和准确率、AUC 一起展示出来。
    """
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, digits=4, zero_division=0)
    auc_score = roc_auc_score(y_test, decision_scores)
    support_vector_count = int(model.n_support_.sum())

    print()
    print("=" * 60)
    print("SVC 模型评估展示")
    print("=" * 60)
    print(f"测试集准确率: {accuracy:.4f}")
    print(f"二分类 AUC: {auc_score:.4f}")
    print(f"支持向量总数: {support_vector_count}")
    print(f"各类别支持向量数: {model.n_support_.tolist()}")
    print(f"核函数: {model.kernel}")
    print(f"C: {model.C}")
    print(f"gamma: {model.gamma}")
    print()
    print("混淆矩阵原始计数:")
    print(cm)
    print()
    print("分类报告:")
    print(report_text)


def run():
    """
    SVC 分类完整流水线

    当前流程分成：
    1. 数据探索；
    2. 数据展示；
    3. 训练与预测；
    4. 结果展示；
    5. 模型评估。
    """
    print("=" * 60)
    print("SVC 分类流水线")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 第 1 步：读取当前算法对应的原始数据
    # ------------------------------------------------------------------
    # SVC 这里使用的是同心圆二分类数据。
    # 它最大的特点是：
    # 1. 线性不可分；
    # 2. 但用核技巧后可以在更高维空间中分开；
    # 3. 非常适合展示 SVM / SVC 的非线性分类能力。
    data = svc_data.copy()
    X = data.drop(columns=["label"])
    y = data["label"]
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
    # SVC 对特征尺度比较敏感，因此标准化是必要步骤。
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # 第 5 步：训练与预测
    # ------------------------------------------------------------------
    model = train_model(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    decision_scores = model.decision_function(X_test_s)

    # ------------------------------------------------------------------
    # 第 6 步：结果图和评估图展示
    # ------------------------------------------------------------------
    plot_confusion_matrix(y_test, y_pred, title="SVC 混淆矩阵", model_name=MODEL)

    # SVC 默认没有 predict_proba，因此这里直接使用 decision_function 的输出绘制 ROC。
    plot_roc_curve(
        y_test,
        decision_scores,
        title="SVC ROC 曲线",
        model_name=MODEL,
    )

    # 当前数据本身已经是二维，因此这里不再做 PCA。
    # 直接用标准化后的二维空间画结果图和决策边界，
    # 才能保证看到的是最终模型真实学习到的分界。
    X_all_s = scaler.transform(X)
    plot_classification_result(
        X_test_s,
        y_test.values,
        y_pred,
        feature_names=[f"{name}（标准化）" for name in feature_names],
        title="SVC 结果展示",
        model_name=MODEL,
    )

    plot_decision_boundary(
        model,
        X_all_s,
        y.values,
        feature_names=[f"{name}（标准化）" for name in feature_names],
        title="SVC 决策边界",
        model_name=MODEL,
    )

    plot_learning_curve(
        build_learning_curve_model(),
        X_train_s,
        y_train,
        title="SVC 学习曲线",
        model_name=MODEL,
    )

    # ------------------------------------------------------------------
    # 第 7 步：终端里的结果展示和模型评估
    # ------------------------------------------------------------------
    show_prediction_examples(X_test, y_test, y_pred, decision_scores)
    show_model_evaluation(model, y_test, y_pred, decision_scores)

    print(f"\n{'=' * 60}")
    print("SVC 流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
