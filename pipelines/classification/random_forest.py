"""
pipelines/classification/random_forest.py
随机森林分类端到端流水线

运行方式: python -m pipelines.classification.random_forest
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from data_generation import random_forest_data
from model_training.classification.random_forest import train_model
from result_visualization.confusion_matrix import plot_confusion_matrix
from result_visualization.roc_curve import plot_roc_curve
from result_visualization.learning_curve import plot_learning_curve
from result_visualization.feature_importance import plot_feature_importance

MODEL = "random_forest"


def run():
    """随机森林分类完整流水线"""
    print("=" * 60)
    print("随机森林分类流水线")
    print("=" * 60)

    data = random_forest_data.copy()
    X = data.drop(columns=["label"])
    y = data["label"]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_model(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values)

    plot_confusion_matrix(
        y_test,
        y_pred,
        title="随机森林 混淆矩阵",
        model_name=MODEL,
    )

    y_scores = model.predict_proba(X_test.values)
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

    plot_learning_curve(
        RandomForestClassifier(n_estimators=100, random_state=42),
        X_train.values,
        y_train.values,
        title="随机森林 学习曲线",
        model_name=MODEL,
    )

    print(f"\n{'=' * 60}")
    print("随机森林流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
