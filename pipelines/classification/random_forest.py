"""
pipelines/classification/random_forest.py
随机森林分类端到端流水线

运行方式: python -m pipelines.classification.random_forest
"""

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from config import get_model_output_dir
from data_exploration import (
    explore_classification_bivariate,
    explore_classification_multivariate,
    explore_classification_univariate,
)
from data_generation import random_forest_data
from data_visualization import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_feature_space_2d,
)
from model_training.classification.random_forest import train_model
from result_visualization.classification_result import plot_classification_result
from result_visualization.confusion_matrix import plot_confusion_matrix
from result_visualization.decision_boundary import plot_decision_boundary
from result_visualization.learning_curve import plot_learning_curve
from result_visualization.feature_importance import plot_feature_importance
from result_visualization.roc_curve import plot_roc_curve

MODEL = "random_forest"


def build_learning_curve_model() -> RandomForestClassifier:
    """
    构造与主模型参数一致的学习曲线模型

    学习曲线里的模型如果和主模型参数不一致，
    后面的评估就会失真。
    因此这里单独封装，避免训练参数和评估参数跑偏。
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=1,
    )


def show_data_exploration(data) -> None:
    """
    展示随机森林训练前的数据探索结果

    随机森林特别适合处理“高维 + 含冗余 + 含噪声”的数据。
    因此在训练前的探索阶段，值得特别关注：
    1. 类别是否均衡；
    2. 特征之间是否存在冗余关系；
    3. 哪些特征可能更有区分能力；
    4. 数据是否存在明显降维空间。
    """
    explore_classification_univariate(
        data,
        dataset_name="RandomForest",
    )
    explore_classification_bivariate(
        data,
        dataset_name="RandomForest",
    )
    explore_classification_multivariate(
        data,
        dataset_name="RandomForest",
    )


def show_data_preview(data, feature_names: list[str]) -> None:
    """
    展示随机森林训练前的数据图

    随机森林这里的数据是 10 维高维多分类数据，
    不适合直接画原始二维散点图。
    因此这里选择：
    1. 类别分布图；
    2. 相关性热力图；
    3. PCA 2D 特征空间图。
    """
    save_dir = get_model_output_dir(MODEL)

    print("\n开始生成数据展示图...")
    plot_class_distribution(
        data,
        target_col="label",
        save_dir=save_dir,
        title="随机森林数据展示：类别分布",
        filename="data_class_distribution.png",
    )
    plot_correlation_heatmap(
        data,
        columns=feature_names + ["label"],
        save_dir=save_dir,
        title="随机森林数据展示：相关性热力图",
        filename="data_correlation.png",
    )
    plot_feature_space_2d(
        data,
        feature_cols=feature_names,
        label_col="label",
        save_dir=save_dir,
        title="随机森林数据展示：PCA 2D 特征空间",
        filename="data_feature_space_2d.png",
    )
    print("数据展示图生成完成。")


def show_prediction_examples(X_test, y_test, y_pred, y_scores) -> None:
    """
    在终端展示部分测试样本的预测结果

    随机森林这里是三分类任务，
    因此结果展示里重点打印：
    1. 真实标签；
    2. 预测标签；
    3. 模型对当前样本的最大预测概率。
    """
    preview_size = min(8, len(X_test))
    preview_df = X_test.reset_index(drop=True).iloc[:preview_size].copy()
    preview_df["真实标签"] = y_test.reset_index(drop=True).iloc[:preview_size].values
    preview_df["预测标签"] = y_pred[:preview_size]
    preview_df["预测置信度"] = y_scores[:preview_size].max(axis=1).round(4)

    print()
    print("=" * 60)
    print("随机森林结果展示")
    print("=" * 60)
    print(preview_df.to_string(index=False))


def show_model_evaluation(model, y_test, y_pred, y_scores, feature_names) -> None:
    """
    在终端展示随机森林的模型评估结果

    对随机森林来说，除了准确率和 AUC 之外，
    “特征重要性”是最重要的可解释信息之一，
    因此这里会把它直接打印出来。
    """
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, digits=4, zero_division=0)
    auc_score = roc_auc_score(y_test, y_scores, multi_class="ovr")
    sorted_importances = sorted(
        zip(feature_names, model.feature_importances_, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )

    print()
    print("=" * 60)
    print("随机森林模型评估展示")
    print("=" * 60)
    print(f"测试集准确率: {accuracy:.4f}")
    print(f"多分类 AUC(OVR): {auc_score:.4f}")
    print(f"树的数量: {model.n_estimators}")
    print(f"最大深度: {model.max_depth}")
    print(f"分裂特征选择方式: {model.max_features}")
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
    随机森林分类完整流水线

    当前流程分成：
    1. 数据探索；
    2. 数据展示；
    3. 训练与预测；
    4. 结果展示；
    5. 模型评估。
    """
    print("=" * 60)
    print("随机森林分类流水线")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 第 1 步：读取当前算法对应的原始数据
    # ------------------------------------------------------------------
    # 随机森林这里使用的是：
    # 1. 10 个特征；
    # 2. 3 个类别；
    # 3. 既有有效特征，也有冗余和噪声特征的多分类数据。
    # 这非常适合体现随机森林对复杂表格数据的适应能力。
    data = random_forest_data.copy()
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
    # 第 4 步：训练与预测
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_model(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values)
    y_scores = model.predict_proba(X_test.values)

    # ------------------------------------------------------------------
    # 第 5 步：结果图和评估图展示
    # ------------------------------------------------------------------
    plot_confusion_matrix(
        y_test,
        y_pred,
        title="随机森林 混淆矩阵",
        model_name=MODEL,
    )

    plot_roc_curve(
        y_test,
        y_scores,
        title="随机森林 ROC 曲线",
        model_name=MODEL,
    )

    plot_feature_importance(
        model,
        feature_names=feature_names,
        title="随机森林 特征重要性",
        model_name=MODEL,
    )

    # 对高维随机森林数据来说，结果展示图和决策边界图都放到 PCA 2D 空间里，
    # 这样可以更直观看到：
    # 1. 测试集真实/预测标签在投影平面中的分布；
    # 2. 随机森林在二维投影空间里形成的大致分区。
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X.values)
    X_train_2d = pca.transform(X_train.values)
    X_test_2d = pca.transform(X_test.values)
    model_2d = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=1,
    )
    model_2d.fit(X_train_2d, y_train)

    plot_classification_result(
        X_test_2d,
        y_test.values,
        y_pred,
        feature_names=["PC1", "PC2"],
        title="随机森林 结果展示 (PCA 2D)",
        model_name=MODEL,
    )

    plot_decision_boundary(
        model_2d,
        X_2d,
        y.values,
        feature_names=["PC1", "PC2"],
        title="随机森林 决策边界 (PCA 2D)",
        model_name=MODEL,
    )

    plot_learning_curve(
        build_learning_curve_model(),
        X_train.values,
        y_train.values,
        title="随机森林 学习曲线",
        model_name=MODEL,
    )

    # ------------------------------------------------------------------
    # 第 6 步：终端里的结果展示和模型评估
    # ------------------------------------------------------------------
    show_prediction_examples(X_test, y_test, y_pred, y_scores)
    show_model_evaluation(model, y_test, y_pred, y_scores, feature_names)

    print(f"\n{'=' * 60}")
    print("随机森林流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
