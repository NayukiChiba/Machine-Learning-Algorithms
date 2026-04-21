"""
pipelines/classification/logistic_regression.py
逻辑回归分类端到端流水线

运行方式: python -m pipelines.classification.logistic_regression
"""

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_generation import logistic_regression_data
from model_training.classification.logistic_regression import train_model
from result_visualization.confusion_matrix import plot_confusion_matrix
from result_visualization.roc_curve import plot_roc_curve
from result_visualization.decision_boundary import plot_decision_boundary
from result_visualization.learning_curve import plot_learning_curve

MODEL = "logistic_regression"


def run():
    """逻辑回归分类完整流水线"""
    print("=" * 60)
    print("逻辑回归分类流水线")
    print("=" * 60)

    data = logistic_regression_data.copy()
    X = data.drop(columns=["label"])
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = train_model(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    plot_confusion_matrix(
        y_test,
        y_pred,
        title="逻辑回归 混淆矩阵",
        model_name=MODEL,
    )

    y_scores = model.predict_proba(X_test_s)
    plot_roc_curve(
        y_test,
        y_scores,
        title="逻辑回归 ROC 曲线",
        model_name=MODEL,
    )

    # 决策边界
    pca = PCA(n_components=2, random_state=42)
    X_all_s = scaler.transform(X)
    X_2d = pca.fit_transform(X_all_s)
    model_2d = LogisticRegression(max_iter=1000, random_state=42)
    model_2d.fit(pca.transform(X_train_s), y_train)
    plot_decision_boundary(
        model_2d,
        X_2d,
        y.values,
        title="逻辑回归 决策边界 (PCA 2D)",
        model_name=MODEL,
    )

    plot_learning_curve(
        LogisticRegression(max_iter=1000, random_state=42),
        X_train_s,
        y_train,
        title="逻辑回归 学习曲线",
        model_name=MODEL,
    )

    print(f"\n{'=' * 60}")
    print("逻辑回归流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
