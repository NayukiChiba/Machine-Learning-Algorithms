"""
result_visualization/dimensionality_plot.py
降维可视化

将 PCA/LDA 变换后的坐标绘制为 2D 或 3D 散点图，
按标签着色以展示降维效果。

使用方式:
    from result_visualization.dimensionality_plot import plot_dimensionality
"""

import matplotlib.pyplot as plt

from config import RV_DIMENSIONALITY_PLOT_DIR


def plot_dimensionality(
    X_transformed,
    y=None,
    explained_variance_ratio=None,
    class_names: list[str] | None = None,
    title: str = "降维可视化",
    dataset_name: str = "default",
    model_name: str = "model",
    figsize: tuple = (10, 8),
    mode: str = "2d",
):
    """
    绘制降维后的散点图

    args:
        X_transformed: 降维后的坐标矩阵 (n_samples, 2 或 3)
        y: 标签数组（可选，用于着色）
        explained_variance_ratio: 各主成分的解释方差比（可选，用于标注轴）
        class_names: 类别名称列表
        title: 图标题
        dataset_name: 数据集名称
        model_name: 模型名称
        figsize: 图像尺寸
        mode: "2d" 或 "3d"
    """
    n_components = X_transformed.shape[1]

    if mode == "3d" and n_components >= 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            X_transformed[:, 0],
            X_transformed[:, 1],
            X_transformed[:, 2],
            c=y,
            cmap="tab10",
            edgecolors="k",
            s=30,
            alpha=0.7,
        )
        z_label = "PC3"
        if explained_variance_ratio is not None and len(explained_variance_ratio) >= 3:
            z_label = f"PC3 ({explained_variance_ratio[2]:.1%})"
        ax.set_zlabel(z_label)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        scatter = ax.scatter(
            X_transformed[:, 0],
            X_transformed[:, 1],
            c=y,
            cmap="tab10",
            edgecolors="k",
            s=30,
            alpha=0.7,
        )

    # 设置轴标签
    x_label = "PC1"
    y_label = "PC2"
    if explained_variance_ratio is not None:
        if len(explained_variance_ratio) >= 1:
            x_label = f"PC1 ({explained_variance_ratio[0]:.1%})"
        if len(explained_variance_ratio) >= 2:
            y_label = f"PC2 ({explained_variance_ratio[1]:.1%})"

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if y is not None:
        legend = ax.legend(*scatter.legend_elements(), title="类别")
        if class_names is not None:
            for i, text in enumerate(legend.get_texts()):
                if i < len(class_names):
                    text.set_text(class_names[i])

    plt.tight_layout()
    save_dir = RV_DIMENSIONALITY_PLOT_DIR / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / f"{model_name}_dim_{mode}.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"降维可视化已保存至: {filepath}")
    plt.close(fig)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.decomposition import PCA

    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        n_classes=3,
        random_state=42,
    )
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    plot_dimensionality(
        X_pca,
        y=y,
        explained_variance_ratio=pca.explained_variance_ratio_,
        title="PCA 降维 (2D)",
        dataset_name="test_pca",
        model_name="pca",
        mode="2d",
    )
