"""
pipelines/dimensionality/pca.py
PCA 降维端到端流水线

运行方式: python -m pipelines.dimensionality.pca
"""

from sklearn.preprocessing import StandardScaler

from data_generation import pca_data
from model_training.dimensionality.pca import train_model
from result_visualization.dimensionality_plot import plot_dimensionality

MODEL = "pca"


def run():
    """PCA 降维完整流水线"""
    print("=" * 60)
    print("PCA 降维流水线")
    print("=" * 60)

    data = pca_data.copy()
    X = data.drop(columns=["label"])
    y = data["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = train_model(X_scaled, n_components=2)
    X_transformed = model.transform(X_scaled)

    plot_dimensionality(
        X_transformed,
        y=y,
        explained_variance_ratio=model.explained_variance_ratio_,
        title="PCA 降维 (2D)",
        model_name=MODEL,
        mode="2d",
    )

    # 3D
    model_3d = train_model(X_scaled, n_components=3)
    X_3d = model_3d.transform(X_scaled)
    plot_dimensionality(
        X_3d,
        y=y,
        explained_variance_ratio=model_3d.explained_variance_ratio_,
        title="PCA 降维 (3D)",
        model_name=MODEL,
        mode="3d",
    )

    print(f"\n{'=' * 60}")
    print("PCA 流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
