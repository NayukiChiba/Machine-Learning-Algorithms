"""
pipelines/classification/knn.py
KNN 分类端到端流水线

运行方式: python -m pipelines.classification.knn
"""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from config import get_model_output_dir
from data_exploration import (
    explore_classification_bivariate,
    explore_classification_multivariate,
    explore_classification_univariate,
)
from data_generation import knn_data
from data_visualization import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_labeled_2d_scatter,
)
from model_training.classification.knn import train_model
from result_visualization.classification_result import plot_classification_result
from result_visualization.confusion_matrix import plot_confusion_matrix
from result_visualization.decision_boundary import plot_decision_boundary
from result_visualization.learning_curve import plot_learning_curve
from result_visualization.roc_curve import plot_roc_curve

MODEL = "knn"


def build_learning_curve_model() -> KNeighborsClassifier:
    """
    构造与主模型参数一致的学习曲线模型

    KNN 是基于距离的惰性学习模型，
    因此学习曲线的模型参数应尽量与主模型保持一致，
    避免训练结果和评估结果使用了不同的 K 值或距离配置。
    """
    return KNeighborsClassifier(
        n_neighbors=5,
        weights="uniform",
        metric="minkowski",
    )


def show_data_preview(data, feature_names: list[str]) -> None:
    """
    展示 KNN 训练前的数据图

    KNN 对数据分布非常敏感，因为它本质上依赖“局部邻域”做预测。
    所以在训练前把数据先看清楚，非常重要：
    1. 类别分布：看是否均衡；
    2. 原始散点图：看局部结构是否清晰；
    3. 相关性热力图：做辅助观察。
    """
    save_dir = get_model_output_dir(MODEL)

    print("\n开始生成数据展示图...")
    plot_class_distribution(
        data,
        target_col="label",
        save_dir=save_dir,
        title="KNN 数据展示：类别分布",
        filename="data_class_distribution.png",
    )
    plot_labeled_2d_scatter(
        data,
        x_col=feature_names[0],
        y_col=feature_names[1],
        label_col="label",
        save_dir=save_dir,
        title="KNN 数据展示：原始散点图",
        filename="data_scatter.png",
    )
    plot_correlation_heatmap(
        data,
        columns=feature_names + ["label"],
        save_dir=save_dir,
        title="KNN 数据展示：相关性热力图",
        filename="data_correlation.png",
    )
    print("数据展示图生成完成。")


def show_data_exploration(data) -> None:
    """
    展示 KNN 训练前的数据探索结果

    KNN 对局部数据结构很敏感，
    因此在训练前先做数据探索，可以帮助理解：
    1. 双月牙数据的类别分布；
    2. 特征之间是否存在统计关系；
    3. 从统计角度看，类别是否容易区分。
    """
    explore_classification_univariate(
        data,
        dataset_name="KNN",
    )
    explore_classification_bivariate(
        data,
        dataset_name="KNN",
    )
    explore_classification_multivariate(
        data,
        dataset_name="KNN",
    )


def show_prediction_examples(X_test, y_test, y_pred, y_scores) -> None:
    """
    在终端展示部分测试样本的预测结果

    这部分内容更偏“结果展示”：
    不是讲全局指标，而是直接看几个具体样本被判成了什么类别。
    对教学和排查错误都很直观。
    """
    preview_size = min(8, len(X_test))
    preview_df = X_test.reset_index(drop=True).iloc[:preview_size].copy()
    preview_df["真实标签"] = y_test.reset_index(drop=True).iloc[:preview_size].values
    preview_df["预测标签"] = y_pred[:preview_size]

    # 当前 KNN 数据是二分类，因此直接展示正类概率最直观。
    if y_scores.ndim == 2 and y_scores.shape[1] == 2:
        preview_df["预测为正类的概率"] = y_scores[:preview_size, 1].round(4)

    print()
    print("=" * 60)
    print("KNN 结果展示")
    print("=" * 60)
    print(preview_df.to_string(index=False))


def show_model_evaluation(model, y_test, y_pred, y_scores) -> None:
    """
    在终端展示 KNN 的模型评估结果

    这里展示的是“模型评估摘要”，让命令行执行结束时，
    可以直接看到这次训练的核心结论，而不需要再去翻图片。
    """
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, digits=4, zero_division=0)

    # 当前 KNN 数据是二分类，因此这里直接取正类概率计算 AUC。
    auc_score = roc_auc_score(y_test, y_scores[:, 1])

    print()
    print("=" * 60)
    print("KNN 模型评估展示")
    print("=" * 60)
    print(f"测试集准确率: {accuracy:.4f}")
    print(f"二分类 AUC: {auc_score:.4f}")
    print(f"K 值: {model.n_neighbors}")
    print(f"投票方式: {model.weights}")
    print(f"距离度量: {model.metric}")
    print()
    print("混淆矩阵原始计数:")
    print(cm)
    print()
    print("分类报告:")
    print(report_text)


def run():
    """
    KNN 分类完整流水线

    当前流程分成三段：
    1. 数据展示：先看数据再训练；
    2. 结果图展示：保存模型评估图；
    3. 终端评估展示：直接输出关键指标和预测样例。
    """
    print("=" * 60)
    print("KNN 分类流水线")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 第 1 步：读取当前算法对应的原始数据
    # ------------------------------------------------------------------
    # KNN 这里使用的是双月牙二分类数据。
    # 这种数据的最大特点是：
    # 1. 全局上不是线性可分的；
    # 2. 但局部邻域结构很清晰；
    # 3. 非常适合观察 KNN 的“局部近邻决策”特点。
    data = knn_data.copy()
    X = data.drop(columns=["label"])
    y = data["label"]
    feature_names = list(X.columns)

    # ------------------------------------------------------------------
    # 第 2 步：数据探索
    # ------------------------------------------------------------------
    # 这一层输出以统计分析为主，
    # 帮助从数值角度先理解当前双月牙数据集。
    show_data_exploration(data)

    # ------------------------------------------------------------------
    # 第 3 步：数据展示
    # ------------------------------------------------------------------
    # 这一步只负责把当前数据的原始形态展示出来，
    # 不参与训练，也不改变后面的建模流程。
    show_data_preview(data, feature_names)

    # ------------------------------------------------------------------
    # 第 4 步：预处理
    # ------------------------------------------------------------------
    # 先做分层切分，再做标准化。
    # KNN 对特征尺度特别敏感，所以标准化几乎是必须步骤。
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # 第 5 步：训练
    # ------------------------------------------------------------------
    model = train_model(X_train_s, y_train)

    # ------------------------------------------------------------------
    # 第 6 步：预测
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test_s)
    y_scores = model.predict_proba(X_test_s)

    # ------------------------------------------------------------------
    # 第 7 步：结果图展示
    # ------------------------------------------------------------------
    # 这些图分别对应不同的观察目标：
    # 1. 混淆矩阵：看分类对不对；
    # 2. ROC 曲线：看概率区分能力；
    # 3. 结果展示图：直接对比测试集真实标签和预测标签；
    # 4. 决策边界：看 KNN 如何按局部邻域分块；
    # 5. 学习曲线：看数据量变化时表现是否稳定。
    plot_confusion_matrix(y_test, y_pred, title="KNN 混淆矩阵", model_name=MODEL)

    plot_roc_curve(
        y_test,
        y_scores,
        title="KNN ROC 曲线",
        model_name=MODEL,
    )

    plot_classification_result(
        X_test.values,
        y_test.values,
        y_pred,
        feature_names=feature_names,
        title="KNN 结果展示",
        model_name=MODEL,
    )

    # 当前 KNN 数据本身就是二维，因此不需要额外做 PCA。
    # 直接在标准化后的二维特征空间中绘制决策边界，
    # 才能保证看到的是“最终主模型真实学到的边界”。
    X_all_s = scaler.transform(X)
    plot_decision_boundary(
        model,
        X_all_s,
        y.values,
        feature_names=[f"{name}（标准化）" for name in feature_names],
        title="KNN 决策边界",
        model_name=MODEL,
    )

    plot_learning_curve(
        build_learning_curve_model(),
        X_train_s,
        y_train,
        title="KNN 学习曲线",
        model_name=MODEL,
    )

    # ------------------------------------------------------------------
    # 第 8 步：终端里的结果展示和模型评估
    # ------------------------------------------------------------------
    # 这里的两段输出是互补关系：
    # 1. “结果展示”偏具体样本；
    # 2. “模型评估”偏全局统计。
    show_prediction_examples(X_test, y_test, y_pred, y_scores)
    show_model_evaluation(model, y_test, y_pred, y_scores)

    print(f"\n{'=' * 60}")
    print("KNN 流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
