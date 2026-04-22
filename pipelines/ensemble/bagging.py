"""
pipelines/ensemble/bagging.py
Bagging 分类端到端流水线

运行方式: python -m pipelines.ensemble.bagging
"""

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
from data_generation import bagging_data
from data_visualization import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_labeled_2d_scatter,
)
from model_training.classification.knn import train_model as _unused  # noqa: F401
from model_training.ensemble.bagging import train_model
from result_visualization.classification_result import plot_classification_result
from result_visualization.confusion_matrix import plot_confusion_matrix
from result_visualization.decision_boundary import plot_decision_boundary
from result_visualization.learning_curve import plot_learning_curve
from result_visualization.roc_curve import plot_roc_curve
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

MODEL = "bagging"


def build_learning_curve_model() -> BaggingClassifier:
    """
    构造与主模型参数一致的学习曲线模型

    Bagging 的学习曲线如果不沿用和主模型一致的参数，
    最后得到的曲线就不能真实反映当前流水线的训练配置。
    """
    base = DecisionTreeClassifier(
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )

    try:
        return BaggingClassifier(
            estimator=base,
            n_estimators=80,
            max_samples=0.8,
            max_features=1.0,
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=1,
        )
    except TypeError:
        return BaggingClassifier(
            base_estimator=base,
            n_estimators=80,
            max_samples=0.8,
            max_features=1.0,
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=1,
        )


def show_data_exploration(data) -> None:
    """
    展示 Bagging 训练前的数据探索结果

    Bagging 当前使用的是高噪声双月牙数据。
    这份数据非常适合展示 Bagging 的核心价值：
    1. 单棵树在高噪声场景下容易过拟合；
    2. Bagging 通过并行集成多个高方差基学习器来降低方差；
    3. 因此数据探索阶段尤其值得关注噪声和局部结构。
    """
    explore_classification_univariate(
        data,
        dataset_name="Bagging",
    )
    explore_classification_bivariate(
        data,
        dataset_name="Bagging",
    )
    explore_classification_multivariate(
        data,
        dataset_name="Bagging",
    )


def show_data_preview(data, feature_names: list[str]) -> None:
    """
    展示 Bagging 训练前的数据图

    当前数据本身就是二维高噪声分类数据，
    因此最适合直接展示：
    1. 类别分布；
    2. 原始散点图；
    3. 相关性热力图。
    """
    save_dir = get_model_output_dir(MODEL)

    print("\n开始生成数据展示图...")
    plot_class_distribution(
        data,
        target_col="label",
        save_dir=save_dir,
        title="Bagging 数据展示：类别分布",
        filename="data_class_distribution.png",
    )
    plot_labeled_2d_scatter(
        data,
        x_col=feature_names[0],
        y_col=feature_names[1],
        label_col="label",
        save_dir=save_dir,
        title="Bagging 数据展示：原始散点图",
        filename="data_scatter.png",
    )
    plot_correlation_heatmap(
        data,
        columns=feature_names + ["label"],
        save_dir=save_dir,
        title="Bagging 数据展示：相关性热力图",
        filename="data_correlation.png",
    )
    print("数据展示图生成完成。")


def show_prediction_examples(X_test, y_test, y_pred, y_scores) -> None:
    """
    在终端展示部分测试样本的预测结果

    这里偏“结果展示”，帮助直接看样本层面的预测表现。
    """
    preview_size = min(8, len(X_test))
    preview_df = X_test.reset_index(drop=True).iloc[:preview_size].copy()
    preview_df["真实标签"] = y_test.reset_index(drop=True).iloc[:preview_size].values
    preview_df["预测标签"] = y_pred[:preview_size]
    preview_df["预测为正类的概率"] = y_scores[:preview_size, 1].round(4)

    print()
    print("=" * 60)
    print("Bagging 结果展示")
    print("=" * 60)
    print(preview_df.to_string(index=False))


def show_model_evaluation(model, y_test, y_pred, y_scores) -> None:
    """
    在终端展示 Bagging 的模型评估结果

    对 Bagging 来说，除了准确率和 AUC，
    OOB 得分也是很有代表性的评估信息，因此这里一并输出。
    """
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_scores[:, 1])
    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, digits=4, zero_division=0)

    print()
    print("=" * 60)
    print("Bagging 模型评估展示")
    print("=" * 60)
    print(f"测试集准确率: {accuracy:.4f}")
    print(f"二分类 AUC: {auc_score:.4f}")
    print(f"基学习器数量: {model.n_estimators}")
    print(f"样本采样比例: {model.max_samples}")
    print(f"特征采样比例: {model.max_features}")
    if hasattr(model, "oob_score_"):
        print(f"OOB 得分: {model.oob_score_:.4f}")
    print()
    print("混淆矩阵原始计数:")
    print(cm)
    print()
    print("分类报告:")
    print(report_text)


def run():
    """
    Bagging 分类完整流水线

    当前流程分成：
    1. 数据探索；
    2. 数据展示；
    3. 训练与预测；
    4. 结果展示；
    5. 模型评估。
    """
    print("=" * 60)
    print("Bagging 分类流水线")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 第 1 步：读取当前算法对应的原始数据
    # ------------------------------------------------------------------
    # 这里使用的是高噪声双月牙数据。
    # 这类数据和 Bagging 很契合，因为高噪声会使单棵树方差较大，
    # 而 Bagging 正是通过多个基学习器投票来降低方差。
    data = bagging_data.copy()
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
    # Bagging 的基学习器是决策树，理论上不强依赖标准化；
    # 但这里继续做标准化，主要是为了：
    # 1. 和仓库其他二维分类流水线保持一致；
    # 2. 决策边界展示时坐标尺度更统一。
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
    y_scores = model.predict_proba(X_test_s)

    # ------------------------------------------------------------------
    # 第 6 步：结果图和评估图展示
    # ------------------------------------------------------------------
    plot_confusion_matrix(y_test, y_pred, title="Bagging 混淆矩阵", model_name=MODEL)

    plot_roc_curve(
        y_test,
        y_scores,
        title="Bagging ROC 曲线",
        model_name=MODEL,
    )

    plot_classification_result(
        X_test_s,
        y_test.values,
        y_pred,
        feature_names=[f"{name}（标准化）" for name in feature_names],
        title="Bagging 结果展示",
        model_name=MODEL,
    )

    # 当前数据本身已经是二维，因此直接在标准化后的空间中画边界，
    # 更能直观看到 Bagging 对高噪声非线性数据的分区效果。
    X_all_s = scaler.transform(X)
    plot_decision_boundary(
        model,
        X_all_s,
        y.values,
        feature_names=[f"{name}（标准化）" for name in feature_names],
        title="Bagging 决策边界",
        model_name=MODEL,
    )

    plot_learning_curve(
        build_learning_curve_model(),
        X_train_s,
        y_train,
        title="Bagging 学习曲线",
        model_name=MODEL,
    )

    # ------------------------------------------------------------------
    # 第 7 步：终端里的结果展示和模型评估
    # ------------------------------------------------------------------
    show_prediction_examples(X_test, y_test, y_pred, y_scores)
    show_model_evaluation(model, y_test, y_pred, y_scores)

    print(f"\n{'=' * 60}")
    print("Bagging 流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
