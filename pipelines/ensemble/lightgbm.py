"""
pipelines/ensemble/lightgbm.py
LightGBM 分类端到端流水线

运行方式: python -m pipelines.ensemble.lightgbm
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import warnings

from config import get_model_output_dir
from data_exploration import (
    explore_classification_bivariate,
    explore_classification_multivariate,
    explore_classification_univariate,
)
from data_generation import lightgbm_data
from data_visualization import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_feature_space_2d,
)
from model_training.ensemble.lightgbm import train_model
from result_visualization.classification_result import plot_classification_result
from result_visualization.confusion_matrix import plot_confusion_matrix
from result_visualization.decision_boundary import plot_decision_boundary
from result_visualization.learning_curve import plot_learning_curve
from result_visualization.roc_curve import plot_roc_curve
from result_visualization.feature_importance import plot_feature_importance

MODEL = "lightgbm"


def build_learning_curve_model():
    """
    构造与主模型参数一致的学习曲线模型

    这里用单独函数封装，避免训练主模型和学习曲线模型参数漂移。
    """
    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:  # noqa: BLE001
        raise ImportError("未安装 lightgbm，无法构建 LightGBM 学习曲线模型。") from exc

    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def show_data_exploration(data) -> None:
    """
    展示 LightGBM 训练前的数据探索结果

    LightGBM 当前使用的是高维四分类数据。
    这里重点看：
    1. 类别是否均衡；
    2. 特征之间是否有冗余关系；
    3. 是否有明显的降维空间；
    4. 哪些特征从统计角度更可能有区分度。
    """
    explore_classification_univariate(
        data,
        dataset_name="LightGBM",
    )
    explore_classification_bivariate(
        data,
        dataset_name="LightGBM",
    )
    explore_classification_multivariate(
        data,
        dataset_name="LightGBM",
    )


def show_data_preview(data, feature_names: list[str]) -> None:
    """
    展示 LightGBM 训练前的数据图

    当前数据是 20 维四分类数据，
    因此这里采用：
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
        title="LightGBM 数据展示：类别分布",
        filename="data_class_distribution.png",
    )
    plot_correlation_heatmap(
        data,
        columns=feature_names + ["label"],
        save_dir=save_dir,
        title="LightGBM 数据展示：相关性热力图",
        filename="data_correlation.png",
    )
    plot_feature_space_2d(
        data,
        feature_cols=feature_names,
        label_col="label",
        save_dir=save_dir,
        title="LightGBM 数据展示：原始数据 PCA 2D 特征空间",
        filename="data_feature_space_2d.png",
    )
    print("数据展示图生成完成。")


def show_prediction_examples(X_test, y_test, y_pred, y_scores) -> None:
    """
    在终端展示部分测试样本的预测结果

    四分类场景下，结果展示重点放在：
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
    print("LightGBM 结果展示")
    print("=" * 60)
    print(preview_df.to_string(index=False))


def show_model_evaluation(model, y_test, y_pred, y_scores, feature_names) -> None:
    """
    在终端展示 LightGBM 的模型评估结果

    除了准确率和多分类 AUC，
    这里也会直接打印特征重要性排序，方便查看模型依赖了哪些特征。
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
    print("LightGBM 模型评估展示")
    print("=" * 60)
    print(f"测试集准确率: {accuracy:.4f}")
    print(f"多分类 AUC(OVR): {auc_score:.4f}")
    print(f"树的数量: {model.n_estimators}")
    print(f"学习率: {model.learning_rate}")
    print(f"叶子数: {model.num_leaves}")
    print(f"最大深度: {model.max_depth}")
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
    LightGBM 分类完整流水线

    当前流程分成：
    1. 数据探索；
    2. 数据展示；
    3. 训练与预测；
    4. 结果展示；
    5. 模型评估。
    """
    print("=" * 60)
    print("LightGBM 分类流水线")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 第 1 步：读取当前算法对应的原始数据
    # ------------------------------------------------------------------
    # 当前使用的是高维四分类数据。
    # 这类数据很适合 LightGBM，因为：
    # 1. 特征维度较高；
    # 2. 任务是多分类；
    # 3. LightGBM 对大规模表格数据和复杂边界都比较有优势。
    data = lightgbm_data.copy()
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
    # LightGBM 属于树模型，不依赖特征缩放。
    # 这里主模型直接使用原始 DataFrame，
    # 一方面更符合树模型习惯，另一方面也能保留稳定的特征名。
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------------------------------------------------------
    # 第 5 步：训练与预测
    # ------------------------------------------------------------------
    model = train_model(X_train, y_train)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
        )
        y_pred = model.predict(X_test)
        y_scores = model.predict_proba(X_test)

    # ------------------------------------------------------------------
    # 第 6 步：结果图和评估图展示
    # ------------------------------------------------------------------
    plot_confusion_matrix(
        y_test,
        y_pred,
        title="LightGBM 混淆矩阵",
        model_name=MODEL,
    )

    plot_roc_curve(
        y_test,
        y_scores,
        title="LightGBM ROC 曲线",
        model_name=MODEL,
    )

    plot_feature_importance(
        model,
        feature_names=feature_names,
        title="LightGBM 特征重要性",
        model_name=MODEL,
    )

    # 用 PCA 2D 空间承载结果展示和边界展示，
    # 这样高维多分类数据也能直观看图。
    from sklearn.preprocessing import StandardScaler

    pca_scaler = StandardScaler()
    X_all_s = pca_scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_all_s)
    X_train_2d = pd.DataFrame(
        pca.transform(pca_scaler.transform(X_train)),
        columns=["PC1", "PC2"],
        index=X_train.index,
    )
    X_test_2d = pd.DataFrame(
        pca.transform(pca_scaler.transform(X_test)),
        columns=["PC1", "PC2"],
        index=X_test.index,
    )
    model_2d = build_learning_curve_model()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
        )
        model_2d.fit(X_train_2d, y_train)

    plot_classification_result(
        X_test_2d.values,
        y_test.values,
        y_pred,
        feature_names=["PC1", "PC2"],
        title="LightGBM 结果展示 (PCA 2D)",
        model_name=MODEL,
    )

    plot_decision_boundary(
        model_2d,
        X_2d,
        y.values,
        feature_names=["PC1", "PC2"],
        title="LightGBM 决策边界 (PCA 2D)",
        model_name=MODEL,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
        )
        plot_learning_curve(
            build_learning_curve_model(),
            X_train,
            y_train,
            title="LightGBM 学习曲线",
            model_name=MODEL,
        )

    # ------------------------------------------------------------------
    # 第 7 步：终端里的结果展示和模型评估
    # ------------------------------------------------------------------
    show_prediction_examples(X_test, y_test, y_pred, y_scores)
    show_model_evaluation(model, y_test, y_pred, y_scores, feature_names)

    print(f"\n{'=' * 60}")
    print("LightGBM 流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
