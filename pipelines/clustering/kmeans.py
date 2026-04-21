"""
pipelines/clustering/kmeans.py
KMeans 聚类端到端流水线

运行方式: python -m pipelines.clustering.kmeans
"""

from sklearn.preprocessing import StandardScaler

from data_generation import kmeans_data
from model_training.clustering.kmeans import train_model
from result_visualization.cluster_plot import plot_clusters

MODEL = "kmeans"


def run():
    """KMeans 聚类完整流水线"""
    print("=" * 60)
    print("KMeans 聚类流水线")
    print("=" * 60)

    data = kmeans_data.copy()
    y_true = data["true_label"].values
    X = data.drop(columns=["true_label"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = train_model(X_scaled)

    plot_clusters(
        X_scaled,
        labels_pred=model.labels_,
        labels_true=y_true,
        centers=model.cluster_centers_,
        title="KMeans 聚类分布",
        model_name=MODEL,
    )

    print(f"\n{'=' * 60}")
    print("KMeans 流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
