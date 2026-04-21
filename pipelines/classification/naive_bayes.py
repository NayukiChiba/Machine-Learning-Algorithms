"""
pipelines/classification/naive_bayes.py
朴素贝叶斯分类端到端流水线

运行方式: python -m pipelines.classification.naive_bayes
"""

from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from config import get_model_output_dir
from data_exploration import (
    explore_classification_bivariate,
    explore_classification_multivariate,
    explore_classification_univariate,
)
from data_generation import naive_bayes_data
from data_visualization import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_feature_space_2d,
)
from model_training.classification.naive_bayes import train_model
from result_visualization.classification_result import plot_classification_result
from result_visualization.confusion_matrix import plot_confusion_matrix
from result_visualization.decision_boundary import plot_decision_boundary
from result_visualization.learning_curve import plot_learning_curve
from result_visualization.roc_curve import plot_roc_curve

MODEL = "naive_bayes"


def build_learning_curve_model() -> GaussianNB:
    """
    构造与主模型参数一致的学习曲线模型

    朴素贝叶斯虽然参数不多，但仍然建议把学习曲线模型单独封装。
    这样后续如果要调整 `var_smoothing`，不会出现训练和评估配置不一致的问题。
    """
    return GaussianNB(var_smoothing=1e-9)


def show_data_exploration(data) -> None:
    """
    展示朴素贝叶斯训练前的数据探索结果

    朴素贝叶斯对特征分布假设比较敏感，
    因此在训练前先看数据探索结果非常有必要。
    尤其值得关注：
    1. 类别是否均衡；
    2. 各特征在不同类别下的均值差异；
    3. 特征间是否存在明显相关性。
    """
    explore_classification_univariate(
        data,
        dataset_name="NaiveBayes",
    )
    explore_classification_bivariate(
        data,
        dataset_name="NaiveBayes",
    )
    explore_classification_multivariate(
        data,
        dataset_name="NaiveBayes",
    )


def show_data_preview(data, feature_names: list[str]) -> None:
    """
    展示朴素贝叶斯训练前的数据图

    Iris 是 4 维数据，不适合直接用原始二维散点图完整展示，
    因此这里选择更适合高维分类数据的三类图：
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
        title="朴素贝叶斯数据展示：类别分布",
        filename="data_class_distribution.png",
    )
    plot_correlation_heatmap(
        data,
        columns=feature_names + ["label"],
        save_dir=save_dir,
        title="朴素贝叶斯数据展示：相关性热力图",
        filename="data_correlation.png",
    )
    plot_feature_space_2d(
        data,
        feature_cols=feature_names,
        label_col="label",
        save_dir=save_dir,
        title="朴素贝叶斯数据展示：PCA 2D 特征空间",
        filename="data_feature_space_2d.png",
    )
    print("数据展示图生成完成。")


def show_prediction_examples(X_test, y_test, y_pred, y_scores) -> None:
    """
    在终端展示部分测试样本的预测结果

    朴素贝叶斯这里是三分类任务，
    因此“结果展示”里不再展示单一正类概率，
    而是展示每个样本的预测标签和最大预测概率（置信度）。
    """
    preview_size = min(8, len(X_test))
    preview_df = X_test.reset_index(drop=True).iloc[:preview_size].copy()
    preview_df["真实标签"] = y_test.reset_index(drop=True).iloc[:preview_size].values
    preview_df["预测标签"] = y_pred[:preview_size]
    preview_df["预测置信度"] = y_scores[:preview_size].max(axis=1).round(4)

    print()
    print("=" * 60)
    print("朴素贝叶斯结果展示")
    print("=" * 60)
    print(preview_df.to_string(index=False))


def show_model_evaluation(model, y_test, y_pred, y_scores, feature_names) -> None:
    """
    在终端展示朴素贝叶斯的模型评估结果

    除了常规的准确率、AUC、分类报告外，
    朴素贝叶斯还有一类很重要的可解释信息：
    各类别下每个特征的均值和方差估计。
    这些参数正是高斯朴素贝叶斯做概率计算的基础。
    """
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, digits=4, zero_division=0)
    auc_score = roc_auc_score(y_test, y_scores, multi_class="ovr")

    print()
    print("=" * 60)
    print("朴素贝叶斯模型评估展示")
    print("=" * 60)
    print(f"测试集准确率: {accuracy:.4f}")
    print(f"多分类 AUC(OVR): {auc_score:.4f}")
    print(f"类别先验: {model.class_prior_.round(6)}")
    print(f"方差平滑项: {model.var_smoothing}")
    print()
    print("混淆矩阵原始计数:")
    print(cm)
    print()
    print("分类报告:")
    print(report_text)
    print("各类别的特征均值估计:")
    for class_index, class_label in enumerate(model.classes_):
        print(f"  类别 {class_label}:")
        for feature_name, mean_value in zip(
            feature_names, model.theta_[class_index], strict=True
        ):
            print(f"    {feature_name}: {mean_value:.6f}")


def run():
    """
    朴素贝叶斯分类完整流水线

    当前流程和其它分类 pipeline 保持一致：
    1. 数据探索；
    2. 数据展示；
    3. 模型训练；
    4. 结果展示；
    5. 模型评估。
    """
    print("=" * 60)
    print("朴素贝叶斯分类流水线")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 第 1 步：读取当前算法对应的原始数据
    # ------------------------------------------------------------------
    # 这里使用的是 Iris 真实数据集：
    # 1. 共 4 个特征；
    # 2. 共 3 个类别；
    # 3. 是讲解高斯朴素贝叶斯的经典小数据集。
    data = naive_bayes_data.copy()
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
    # 对高斯朴素贝叶斯来说，标准化不是绝对必须，但这里保留和仓库其它分类流水线一致的输入处理方式，
    # 也便于后续做统一比较。
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
    plot_confusion_matrix(
        y_test,
        y_pred,
        title="朴素贝叶斯 混淆矩阵",
        model_name=MODEL,
    )

    plot_roc_curve(
        y_test,
        y_scores,
        title="朴素贝叶斯 ROC 曲线",
        model_name=MODEL,
    )

    # Iris 是 4 维数据，因此结果展示图和决策边界图都使用 PCA 2D 空间。
    # 这样可以在同一个二维投影空间里同时观察：
    # 1. 测试集真实/预测标签；
    # 2. 模型学到的决策边界。
    pca = PCA(n_components=2, random_state=42)
    X_all_s = scaler.transform(X)
    X_2d = pca.fit_transform(X_all_s)
    model_2d = GaussianNB()
    model_2d.fit(pca.transform(X_train_s), y_train)

    plot_classification_result(
        pca.transform(X_test_s),
        y_test.values,
        y_pred,
        feature_names=["PC1", "PC2"],
        title="朴素贝叶斯 结果展示 (PCA 2D)",
        model_name=MODEL,
    )

    plot_decision_boundary(
        model_2d,
        X_2d,
        y.values,
        feature_names=["PC1", "PC2"],
        title="朴素贝叶斯 决策边界 (PCA 2D)",
        model_name=MODEL,
    )

    plot_learning_curve(
        build_learning_curve_model(),
        X_train_s,
        y_train,
        title="朴素贝叶斯 学习曲线",
        model_name=MODEL,
    )

    # ------------------------------------------------------------------
    # 第 7 步：终端里的结果展示和模型评估
    # ------------------------------------------------------------------
    show_prediction_examples(X_test, y_test, y_pred, y_scores)
    show_model_evaluation(model, y_test, y_pred, y_scores, feature_names)

    print(f"\n{'=' * 60}")
    print("朴素贝叶斯流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
