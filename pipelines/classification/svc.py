"""
pipelines/classification/svc.py
SVC 分类端到端流水线

运行方式: python -m pipelines.classification.svc
"""

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC as SVC_Model

from data_generation import svc_data
from model_training.classification.svc import train_model
from result_visualization.confusion_matrix import plot_confusion_matrix
from result_visualization.decision_boundary import plot_decision_boundary
from result_visualization.learning_curve import plot_learning_curve

MODEL = "svc"


def run():
    """SVC 分类完整流水线"""
    print("=" * 60)
    print("SVC 分类流水线")
    print("=" * 60)

    data = svc_data.copy()
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

    plot_confusion_matrix(y_test, y_pred, title="SVC 混淆矩阵", model_name=MODEL)

    # 决策边界
    pca = PCA(n_components=2, random_state=42)
    X_all_s = scaler.transform(X)
    X_2d = pca.fit_transform(X_all_s)
    model_2d = SVC_Model(kernel="rbf", random_state=42)
    model_2d.fit(pca.transform(X_train_s), y_train)
    plot_decision_boundary(
        model_2d,
        X_2d,
        y.values,
        title="SVC 决策边界 (PCA 2D)",
        model_name=MODEL,
    )

    plot_learning_curve(
        SVC_Model(kernel="rbf", random_state=42),
        X_train_s,
        y_train,
        title="SVC 学习曲线",
        model_name=MODEL,
    )

    print(f"\n{'=' * 60}")
    print("SVC 流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
