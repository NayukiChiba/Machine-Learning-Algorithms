"""
pipelines/classification/logistic_regression.py
逻辑回归分类端到端流水线

运行方式: python -m pipelines.classification.logistic_regression
"""

from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import get_model_output_dir
from data_exploration import (
    explore_classification_bivariate,
    explore_classification_multivariate,
    explore_classification_univariate,
)
from data_generation import logistic_regression_data
from data_visualization import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_feature_space_2d,
)
from model_training.classification.logistic_regression import train_model
from result_visualization.classification_result import plot_classification_result
from result_visualization.confusion_matrix import plot_confusion_matrix
from result_visualization.decision_boundary import plot_decision_boundary
from result_visualization.feature_importance import plot_feature_importance
from result_visualization.learning_curve import plot_learning_curve
from result_visualization.roc_curve import plot_roc_curve

MODEL = "logistic_regression"


def build_learning_curve_model() -> LogisticRegression:
    """
    构造与主模型参数一致的学习曲线模型

    逻辑回归的学习曲线如果和主模型参数不一致，
    就会出现“训练用的是一种模型、评估曲线却是另一种模型”的问题。
    因此这里单独封装，避免后续改超参数时漏改。
    """
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )


def show_data_exploration(data) -> None:
    """
    展示逻辑回归训练前的数据探索结果

    对逻辑回归来说，训练前最值得关注的是：
    1. 类别是否均衡；
    2. 特征之间是否存在明显冗余；
    3. 各特征对类别是否有较强区分能力；
    4. 是否有较好的线性可分趋势。
    """
    explore_classification_univariate(
        data,
        dataset_name="LogisticRegression",
    )
    explore_classification_bivariate(
        data,
        dataset_name="LogisticRegression",
    )
    explore_classification_multivariate(
        data,
        dataset_name="LogisticRegression",
    )


def show_data_preview(data, feature_names: list[str]) -> None:
    """
    展示逻辑回归训练前的数据图

    逻辑回归原始数据有 6 个特征，因此不适合直接用原始二维散点图展示。
    这里改用更适合高维线性模型的数据展示方式：
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
        title="逻辑回归数据展示：类别分布",
        filename="data_class_distribution.png",
    )
    plot_correlation_heatmap(
        data,
        columns=feature_names + ["label"],
        save_dir=save_dir,
        title="逻辑回归数据展示：相关性热力图",
        filename="data_correlation.png",
    )
    plot_feature_space_2d(
        data,
        feature_cols=feature_names,
        label_col="label",
        save_dir=save_dir,
        title="逻辑回归数据展示：PCA 2D 特征空间",
        filename="data_feature_space_2d.png",
    )
    print("数据展示图生成完成。")


def show_prediction_examples(X_test, y_test, y_pred, y_scores) -> None:
    """
    在终端展示部分测试样本的预测结果

    这部分输出偏“结果展示”：
    它不是讲整体指标，而是直接展示若干样本被模型怎么判别。
    """
    preview_size = min(8, len(X_test))
    preview_df = X_test.reset_index(drop=True).iloc[:preview_size].copy()
    preview_df["真实标签"] = y_test.reset_index(drop=True).iloc[:preview_size].values
    preview_df["预测标签"] = y_pred[:preview_size]

    if y_scores.ndim == 2 and y_scores.shape[1] == 2:
        preview_df["预测为正类的概率"] = y_scores[:preview_size, 1].round(4)

    print()
    print("=" * 60)
    print("逻辑回归结果展示")
    print("=" * 60)
    print(preview_df.to_string(index=False))


def show_model_evaluation(model, y_test, y_pred, y_scores, feature_names) -> None:
    """
    在终端展示逻辑回归的模型评估结果

    除了常规的准确率、AUC、分类报告外，
    逻辑回归还应重点展示系数信息，因为系数本身就是模型解释的一部分。
    """
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, digits=4, zero_division=0)
    auc_score = roc_auc_score(y_test, y_scores[:, 1])

    # 逻辑回归是线性模型，系数的正负和大小都很有解释意义。
    coef_values = model.coef_[0]
    sorted_coef = sorted(
        zip(feature_names, coef_values, strict=True),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    print()
    print("=" * 60)
    print("逻辑回归模型评估展示")
    print("=" * 60)
    print(f"测试集准确率: {accuracy:.4f}")
    print(f"二分类 AUC: {auc_score:.4f}")
    print(f"正则化类型: {model.penalty}")
    print(f"正则强度倒数 C: {model.C}")
    print(f"优化器: {model.solver}")
    print(f"迭代次数上限: {model.max_iter}")
    print(f"截距: {model.intercept_.round(6)}")
    print()
    print("混淆矩阵原始计数:")
    print(cm)
    print()
    print("分类报告:")
    print(report_text)
    print("按绝对值排序后的系数:")
    for feature_name, coef in sorted_coef:
        direction = "正向推动预测为 1 类" if coef > 0 else "反向推动预测为 1 类"
        print(f"  {feature_name}: {coef:.6f} ({direction})")


def run():
    """
    逻辑回归分类完整流水线

    当前流程同样拆成几层：
    1. 数据探索；
    2. 数据展示；
    3. 训练与预测；
    4. 结果展示；
    5. 模型评估。
    """
    print("=" * 60)
    print("逻辑回归分类流水线")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 第 1 步：读取当前算法对应的原始数据
    # ------------------------------------------------------------------
    # 逻辑回归这里使用的是“线性可分的高维二分类数据”。
    # 它的特点是：
    # 1. 总共有 6 个特征；
    # 2. 其中既有有效特征，也有冗余和噪声特征；
    # 3. 很适合观察逻辑回归在线性可分场景下的拟合能力。
    data = logistic_regression_data.copy()
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
    # 逻辑回归对特征尺度也比较敏感，因此这里同样使用标准化。
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
        title="逻辑回归 混淆矩阵",
        model_name=MODEL,
    )

    plot_roc_curve(
        y_test,
        y_scores,
        title="逻辑回归 ROC 曲线",
        model_name=MODEL,
    )

    plot_feature_importance(
        model,
        feature_names=feature_names,
        title="逻辑回归 特征重要性",
        model_name=MODEL,
    )

    # 对于高维逻辑回归数据，结果展示图和决策边界图都基于同一个 PCA 2D 空间，
    # 这样可以保证：
    # 1. 结果展示图和边界图在同一个投影坐标系里；
    # 2. 方便对照“真实标签 / 预测标签 / 决策边界”的关系。
    pca = PCA(n_components=2, random_state=42)
    X_all_s = scaler.transform(X)
    X_2d = pca.fit_transform(X_all_s)
    model_2d = LogisticRegression(max_iter=1000, random_state=42)
    model_2d.fit(pca.transform(X_train_s), y_train)

    plot_classification_result(
        pca.transform(X_test_s),
        y_test.values,
        y_pred,
        feature_names=["PC1", "PC2"],
        title="逻辑回归 结果展示 (PCA 2D)",
        model_name=MODEL,
    )

    plot_decision_boundary(
        model_2d,
        X_2d,
        y.values,
        feature_names=["PC1", "PC2"],
        title="逻辑回归 决策边界 (PCA 2D)",
        model_name=MODEL,
    )

    plot_learning_curve(
        build_learning_curve_model(),
        X_train_s,
        y_train,
        title="逻辑回归 学习曲线",
        model_name=MODEL,
    )

    # ------------------------------------------------------------------
    # 第 7 步：终端里的结果展示和模型评估
    # ------------------------------------------------------------------
    show_prediction_examples(X_test, y_test, y_pred, y_scores)
    show_model_evaluation(model, y_test, y_pred, y_scores, feature_names)

    print(f"\n{'=' * 60}")
    print("逻辑回归流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
