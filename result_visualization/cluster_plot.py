"""
result_visualization/cluster_plot.py
聚类分布可视化

绘制聚类结果的二维散点图，对比真实标签与预测标签。
支持 KMeans / DBSCAN / GMM 等聚类模型。

使用方式:
    from result_visualization.cluster_plot import plot_clusters
"""

import numpy as np
import matplotlib.pyplot as plt

from config import get_model_output_dir

DISCRETE_COLORS = [
    "#D81B60",  # 洋红
    "#1E88E5",  # 蓝
    "#FFC107",  # 黄
    "#004D40",  # 深青
    "#E64A19",  # 橙红
    "#6A1B9A",  # 紫
    "#2E7D32",  # 绿
    "#5D4037",  # 棕
]


def _get_label_color_map(labels) -> dict[int, str]:
    """
    为离散标签构造高对比度颜色映射

    Args:
        labels: 标签数组

    Returns:
        dict[int, str]: 标签到颜色的映射
    """
    unique_labels = sorted(set(labels) - {-1})
    color_map = {}
    for index, label in enumerate(unique_labels):
        color_map[label] = DISCRETE_COLORS[index % len(DISCRETE_COLORS)]
    return color_map


def plot_clusters(
    X,
    labels_pred,
    labels_true=None,
    centers=None,
    feature_names: list[str] | None = None,
    title: str = "聚类分布",
    model_name: str = "model",
    figsize: tuple = (12, 5),
    filename: str = "cluster_plot.png",
):
    """
    绘制聚类分布散点图

    args:
        X: 特征矩阵（n_samples, 2）— 仅支持二维
        labels_pred: 预测簇标签
        labels_true: 真实标签（可选，有则画对比图）
        centers: 聚类中心坐标（可选，如 KMeans 的 .cluster_centers_）
        feature_names: 两个特征的名称
        title: 图标题
        model_name: 模型名称
        figsize: 图像尺寸
        filename: 保存文件名
    """
    if X.shape[1] != 2:
        raise ValueError(
            f"聚类散点绘制仅支持二维特征，当前为 {X.shape[1]} 维。"
            "请先降维至 2 维后再传入。"
        )

    if feature_names is None:
        feature_names = ["Feature 1", "Feature 2"]

    labels_pred = np.asarray(labels_pred)
    has_true = labels_true is not None

    n_plots = 2 if has_true else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    # 预测标签图
    ax = axes[0]
    # DBSCAN 等算法中，-1 通常表示噪声点。
    # 这里把噪声点单独高亮，避免它们和正常簇混在一起不易识别。
    normal_mask = labels_pred != -1
    noise_mask = labels_pred == -1
    color_map_pred = _get_label_color_map(labels_pred)

    if normal_mask.any():
        for label, color in color_map_pred.items():
            label_mask = labels_pred == label
            ax.scatter(
                X[label_mask, 0],
                X[label_mask, 1],
                c=color,
                edgecolors="k",
                s=30,
                alpha=0.8,
                label=f"预测簇 {label}",
            )
    if noise_mask.any():
        ax.scatter(
            X[noise_mask, 0],
            X[noise_mask, 1],
            c="#6E6E6E",
            marker="x",
            s=60,
            linewidths=1.2,
            alpha=0.9,
            label="噪声点",
        )
    if centers is not None:
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            c="red",
            marker="X",
            s=200,
            edgecolors="k",
            linewidths=1.5,
            label="中心",
        )
    if centers is not None or noise_mask.any():
        ax.legend()
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(f"{title} — 预测标签")

    # 真实标签图
    if has_true:
        ax2 = axes[1]
        color_map_true = _get_label_color_map(labels_true)
        for label, color in color_map_true.items():
            label_mask = np.asarray(labels_true) == label
            ax2.scatter(
                X[label_mask, 0],
                X[label_mask, 1],
                c=color,
                edgecolors="k",
                s=30,
                alpha=0.8,
                label=f"真实簇 {label}",
            )
        ax2.set_xlabel(feature_names[0])
        ax2.set_ylabel(feature_names[1])
        ax2.set_title(f"{title} — 真实标签")
        ax2.legend(loc="best")

    plt.tight_layout()
    save_dir = get_model_output_dir(model_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"聚类分布图已保存至: {filepath}")
    plt.close(fig)


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans

    X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)
    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    model.fit(X)
    plot_clusters(
        X,
        labels_pred=model.labels_,
        labels_true=y_true,
        centers=model.cluster_centers_,
        title="KMeans 聚类",
        model_name="kmeans",
    )
