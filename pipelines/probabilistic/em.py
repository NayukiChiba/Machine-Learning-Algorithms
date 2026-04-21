"""
pipelines/probabilistic/em.py
EM (GMM) 聚类端到端流水线

运行方式: python -m pipelines.probabilistic.em
"""

from sklearn.preprocessing import StandardScaler

from data_generation import em_data
from model_training.probabilistic.em import train_model
from result_visualization.cluster_plot import plot_clusters

MODEL = "gmm"


def run():
    """EM (GMM) 聚类完整流水线"""
    print("=" * 60)
    print("EM (GMM) 聚类流水线")
    print("=" * 60)

    data = em_data.copy()
    y_true = data["true_label"].values
    X = data.drop(columns=["true_label"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = train_model(X_scaled)
    labels_pred = model.predict(X_scaled)

    plot_clusters(
        X_scaled,
        labels_pred=labels_pred,
        labels_true=y_true,
        title="EM (GMM) 聚类分布",
        model_name=MODEL,
    )

    print(f"\n{'=' * 60}")
    print("EM (GMM) 流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
