"""
pipelines/ensemble/gbdt.py
GBDT 分类端到端流水线

运行方式: python -m pipelines.ensemble.gbdt
"""

from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
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
from data_generation import gbdt_data
from data_visualization import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_feature_space_2d,
)
from model_training.ensemble.gbdt import train_model
from result_visualization.classification_result import plot_classification_result
from result_visualization.confusion_matrix import plot_confusion_matrix
from result_visualization.decision_boundary import plot_decision_boundary
from result_visualization.roc_curve import plot_roc_curve
from result_visualization.feature_importance import plot_feature_importance
from result_visualization.learning_curve import plot_learning_curve

MODEL = "gbdt"


def build_learning_curve_model() -> GradientBoostingClassifier:
    """
    构造与主模型参数一致的学习曲线模型

    GBDT 的学习曲线如果和主模型参数不一致，
    后面的训练/验证趋势就不能真实反映当前流水线配置。
    """
    return GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        subsample=1.0,
        random_state=42,
    )


def show_data_exploration(data) -> None:
    """
    展示 GBDT 训练前的数据探索结果

    GBDT 当前使用的是中等难度的多分类数据。
    这类数据很适合展示 Boosting 的特点：
    1. 类别之间存在一定重叠；
    2. 线性模型不容易一次分清；
    3. GBDT 可以通过逐步拟合残差来修正边界。
    """
    explore_classification_univariate(
        data,
        dataset_name="GBDT",
    )
    explore_classification_bivariate(
        data,
        dataset_name="GBDT",
    )
    explore_classification_multivariate(
        data,
        dataset_name="GBDT",
    )


def show_data_preview(data, feature_names: list[str]) -> None:
    """
    展示 GBDT 训练前的数据图

    当前数据是高维多分类数据，因此这里采用：
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
        title="GBDT 数据展示：类别分布",
        filename="data_class_distribution.png",
    )
    plot_correlation_heatmap(
        data,
        columns=feature_names + ["label"],
        save_dir=save_dir,
        title="GBDT 数据展示：相关性热力图",
        filename="data_correlation.png",
    )
    plot_feature_space_2d(
        data,
        feature_cols=feature_names,
        label_col="label",
        save_dir=save_dir,
        title="GBDT 数据展示：原始数据 PCA 2D 特征空间",
        filename="data_feature_space_2d.png",
    )
    print("数据展示图生成完成。")


def show_prediction_examples(X_test, y_test, y_pred, y_scores) -> None:
    """
    在终端展示部分测试样本的预测结果

    多分类场景下，这里重点展示：
    1. 真实标签；
    2. 预测标签；
    3. 最大预测概率（置信度）。
    """
    preview_size = min(8, len(X_test))
    preview_df = X_test.reset_index(drop=True).iloc[:preview_size].copy()
    preview_df["真实标签"] = y_test.reset_index(drop=True).iloc[:preview_size].values
    preview_df["预测标签"] = y_pred[:preview_size]
    preview_df["预测置信度"] = y_scores[:preview_size].max(axis=1).round(4)

    print()
    print("=" * 60)
    print("GBDT 结果展示")
    print("=" * 60)
    print(preview_df.to_string(index=False))


def show_model_evaluation(model, y_test, y_pred, y_scores, feature_names) -> None:
    """
    在终端展示 GBDT 的模型评估结果

    除了常规的准确率和 AUC，这里还会直接输出特征重要性排序，
    因为这在树模型里是最值得看的解释信息之一。
    """
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_scores, multi_class="ovr")
    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, digits=4, zero_division=0)
    sorted_importances = sorted(
        zip(feature_names, model.feature_importances_, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )

    print()
    print("=" * 60)
    print("GBDT 模型评估展示")
    print("=" * 60)
    print(f"测试集准确率: {accuracy:.4f}")
    print(f"多分类 AUC(OVR): {auc_score:.4f}")
    print(f"弱学习器数量: {model.n_estimators}")
    print(f"学习率: {model.learning_rate}")
    print(f"子采样比例: {model.subsample}")
    print()
    print("混淆矩阵原始计数:")
    print(cm)
    print()
    print("分类报告:")
    print(report_text)
    print("按重要性排序的特征:")
    for feature_name, importance in sorted_importances:
        print(f"  {feature_name}: {importance:.6f}")


def run():
    """
    GBDT 分类完整流水线

    当前流程分成：
    1. 数据探索；
    2. 数据展示；
    3. 训练与预测；
    4. 结果展示；
    5. 模型评估。
    """
    print("=" * 60)
    print("GBDT 分类流水线")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 第 1 步：读取当前算法对应的原始数据
    # ------------------------------------------------------------------
    # 当前使用的是中等难度的多分类数据。
    # 这份数据的重点不是“简单线性可分”，
    # 而是通过多轮 boosting 逐步修正分类边界。
    data = gbdt_data.copy()
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
    # 对 GBDT 来说，标准化不是必须，但这里仍保留统一预处理流程，
    # 方便和仓库中其它分类流水线保持一致，也方便后续统一做 2D 展示。
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
    plot_confusion_matrix(y_test, y_pred, title="GBDT 混淆矩阵", model_name=MODEL)

    plot_roc_curve(y_test, y_scores, title="GBDT ROC 曲线", model_name=MODEL)

    plot_feature_importance(
        model,
        feature_names=feature_names,
        title="GBDT 特征重要性",
        model_name=MODEL,
    )

    # GBDT 当前数据是高维多分类数据，因此这里用 PCA 2D 空间做结果展示与边界展示。
    pca = PCA(n_components=2, random_state=42)
    X_all_s = scaler.transform(X)
    X_2d = pca.fit_transform(X_all_s)
    X_train_2d = pca.transform(X_train_s)
    X_test_2d = pca.transform(X_test_s)
    model_2d = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        subsample=1.0,
        random_state=42,
    )
    model_2d.fit(X_train_2d, y_train)

    plot_classification_result(
        X_test_2d,
        y_test.values,
        y_pred,
        feature_names=["PC1", "PC2"],
        title="GBDT 结果展示 (PCA 2D)",
        model_name=MODEL,
    )

    plot_decision_boundary(
        model_2d,
        X_2d,
        y.values,
        feature_names=["PC1", "PC2"],
        title="GBDT 决策边界 (PCA 2D)",
        model_name=MODEL,
    )

    plot_learning_curve(
        build_learning_curve_model(),
        X_train_s,
        y_train,
        title="GBDT 学习曲线",
        model_name=MODEL,
    )

    # ------------------------------------------------------------------
    # 第 7 步：终端里的结果展示和模型评估
    # ------------------------------------------------------------------
    show_prediction_examples(X_test, y_test, y_pred, y_scores)
    show_model_evaluation(model, y_test, y_pred, y_scores, feature_names)

    print(f"\n{'=' * 60}")
    print("GBDT 流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
