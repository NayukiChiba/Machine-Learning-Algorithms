"""
pipelines/classification/knn.py
KNN 分类端到端流水线

运行方式: python -m pipelines.classification.knn
"""

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_generation import knn_data
from model_training.classification.knn import train_model
from result_visualization.confusion_matrix import plot_confusion_matrix
from result_visualization.roc_curve import plot_roc_curve
from result_visualization.decision_boundary import plot_decision_boundary
from result_visualization.learning_curve import plot_learning_curve

DATASET = "knn"
MODEL = "knn"


def run():
    """KNN 分类完整流水线"""
    print("=" * 60)
    print("KNN 分类流水线")
    print("=" * 60)

    # 1. 数据生成
    data = knn_data.copy()
    X = data.drop(columns=["label"])
    y = data["label"]

    # 2. 预处理：分层拆分 + 标准化
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 3. 训练
    model = train_model(X_train_s, y_train)

    # 4. 预测
    y_pred = model.predict(X_test_s)

    # 5. 可视化
    plot_confusion_matrix(
        y_test, y_pred, title="KNN 混淆矩阵", dataset_name=DATASET, model_name=MODEL
    )

    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test_s)
        plot_roc_curve(
            y_test,
            y_scores,
            title="KNN ROC 曲线",
            dataset_name=DATASET,
            model_name=MODEL,
        )

    # 决策边界（PCA 降至 2 维）
    pca = PCA(n_components=2, random_state=42)
    X_all_s = scaler.transform(X)
    X_2d = pca.fit_transform(X_all_s)
    from sklearn.neighbors import KNeighborsClassifier

    model_2d = KNeighborsClassifier(n_neighbors=5)
    model_2d.fit(pca.transform(X_train_s), y_train)
    plot_decision_boundary(
        model_2d, X_2d, y.values, title="KNN 决策边界 (PCA 2D)", dataset_name=DATASET
    )

    # 学习曲线
    from sklearn.neighbors import KNeighborsClassifier as KNC

    plot_learning_curve(
        KNC(n_neighbors=5),
        X_train_s,
        y_train,
        title="KNN 学习曲线",
        dataset_name=DATASET,
        model_name=MODEL,
    )

    print(f"\n{'=' * 60}")
    print("KNN 流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
