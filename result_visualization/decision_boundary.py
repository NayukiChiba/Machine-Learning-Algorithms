"""
result_visualization/decision_boundary.py
分类决策边界可视化

对二维特征空间进行网格采样，绘制分类模型的分界区域。
支持任意具有 .predict() 方法的分类器。

使用方式:
    from result_visualization.decision_boundary import plot_decision_boundary
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

from config import get_model_output_dir


def plot_decision_boundary(
    model,
    X,
    y,
    feature_names: list[str] | None = None,
    title: str = "决策边界",
    model_name: str = "model",
    resolution: float = 0.02,
    figsize: tuple = (10, 8),
):
    """
    绘制二维分类决策边界

    args:
        model: 训练好的分类器（必须有 .predict 方法）
        X: 特征矩阵（n_samples, 2）— 仅支持二维
        y: 标签数组
        feature_names: 两个特征的名称
        title: 图标题
        model_name: 模型名称
        resolution: 网格分辨率
        figsize: 图像尺寸
    """
    if X.shape[1] != 2:
        raise ValueError(
            f"决策边界绘制仅支持二维特征，当前为 {X.shape[1]} 维。"
            "请先用 PCA 或 LDA 降至 2 维后再传入。"
        )

    if feature_names is None:
        feature_names = ["Feature 1", "Feature 2"]

    # 获取坐标范围
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution),
    )

    # 预测网格点
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # 某些模型（如 LightGBM）在训练时会记录特征名。
    # 如果预测时直接传裸 ndarray，就会触发“特征名不匹配”的 warning。
    # 这里在能够确定特征名时，显式构造成 DataFrame。
    if feature_names is not None and len(feature_names) == 2:
        predict_input = pd.DataFrame(grid_points, columns=feature_names)
    else:
        predict_input = grid_points

    Z = model.predict(predict_input)
    Z = Z.reshape(xx.shape)

    # 配色
    classes = np.unique(y)
    n_classes = len(classes)
    cmap_bg = ListedColormap(plt.cm.tab10.colors[:n_classes])
    cmap_pt = ListedColormap(plt.cm.tab10.colors[:n_classes])

    fig, ax = plt.subplots(figsize=figsize)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_bg)
    scatter = ax.scatter(
        X[:, 0], X[:, 1], c=y, cmap=cmap_pt, edgecolors="k", s=30, alpha=0.8
    )
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(title)
    ax.legend(*scatter.legend_elements(), title="类别")

    plt.tight_layout()
    save_dir = get_model_output_dir(model_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / "decision_boundary.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"决策边界图已保存至: {filepath}")
    plt.close(fig)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.neighbors import KNeighborsClassifier

    X, y = make_classification(
        n_samples=300,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    plot_decision_boundary(
        model,
        X,
        y,
        title="KNN 决策边界",
        model_name="knn",
    )
