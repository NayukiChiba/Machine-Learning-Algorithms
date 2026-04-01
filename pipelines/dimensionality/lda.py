"""
pipelines/dimensionality/lda.py
LDA 降维端到端流水线

运行方式: python -m pipelines.dimensionality.lda
"""

from sklearn.preprocessing import StandardScaler

from data_generation import lda_data
from model_training.dimensionality.lda import train_model
from result_visualization.dimensionality_plot import plot_dimensionality

DATASET = "lda"
MODEL = "lda"


def run():
    """LDA 降维完整流水线"""
    print("=" * 60)
    print("LDA 降维流水线")
    print("=" * 60)

    data = lda_data.copy()
    X = data.drop(columns=["label"])
    y = data["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = train_model(X_scaled, y, n_components=2)
    X_transformed = model.transform(X_scaled)

    evr = (
        model.explained_variance_ratio_
        if hasattr(model, "explained_variance_ratio_")
        else None
    )
    plot_dimensionality(
        X_transformed,
        y=y,
        explained_variance_ratio=evr,
        title="LDA 降维 (2D)",
        dataset_name=DATASET,
        model_name=MODEL,
        mode="2d",
    )

    print(f"\n{'=' * 60}")
    print("LDA 流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
