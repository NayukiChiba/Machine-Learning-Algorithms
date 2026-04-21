"""
pipelines/classification/decision_tree.py
决策树分类端到端流水线

运行方式: python -m pipelines.classification.decision_tree
"""

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from data_generation import decision_tree_classification_data
from model_training.classification.decision_tree import train_model
from result_visualization.confusion_matrix import plot_confusion_matrix
from result_visualization.roc_curve import plot_roc_curve
from result_visualization.decision_boundary import plot_decision_boundary
from result_visualization.learning_curve import plot_learning_curve
from result_visualization.feature_importance import plot_feature_importance

MODEL = "decision_tree"


def run():
    """决策树分类完整流水线"""
    print("=" * 60)
    print("决策树分类流水线")
    print("=" * 60)

    data = decision_tree_classification_data.copy()
    X = data.drop(columns=["label"])
    y = data["label"]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_model(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values)

    plot_confusion_matrix(y_test, y_pred, title="决策树 混淆矩阵", model_name=MODEL)

    y_scores = model.predict_proba(X_test.values)
    plot_roc_curve(
        y_test,
        y_scores,
        title="决策树 ROC 曲线",
        model_name=MODEL,
    )

    plot_feature_importance(
        model,
        feature_names=feature_names,
        title="决策树 特征重要性",
        model_name=MODEL,
    )

    # 决策边界
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X.values)
    model_2d = DecisionTreeClassifier(max_depth=6, random_state=42)
    model_2d.fit(pca.transform(X_train.values), y_train.values)
    plot_decision_boundary(
        model_2d,
        X_2d,
        y.values,
        title="决策树 决策边界 (PCA 2D)",
        model_name=MODEL,
    )

    plot_learning_curve(
        DecisionTreeClassifier(max_depth=6, random_state=42),
        X_train.values,
        y_train.values,
        title="决策树 学习曲线",
        model_name=MODEL,
    )

    print(f"\n{'=' * 60}")
    print("决策树分类流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
