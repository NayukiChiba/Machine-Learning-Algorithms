"""
result_visualization/clustering_diagnostics.py
聚类诊断曲线

当前模块主要服务于无监督聚类场景下的“参数诊断”：
1. DBSCAN k-distance 曲线
2. DBSCAN eps 扫描指标曲线

和分类/回归不同，聚类通常没有标准的 loss/accuracy 训练曲线。
因此这里更适合用“参数变化 -> 聚类结构变化”的方式来诊断模型效果。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from config import get_model_output_dir
from model_evaluation.clustering_metrics import evaluate_clustering_with_ground_truth


def _save_figure(fig: plt.Figure, save_dir: Path, filename: str) -> None:
    """
    保存图表到模型输出目录

    Args:
        fig: matplotlib 图对象
        save_dir: 保存目录
        filename: 文件名
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / filename
    fig.savefig(filepath, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"聚类诊断图已保存至: {filepath}")


def plot_dbscan_k_distance(
    X,
    min_samples: int,
    model_name: str,
    current_eps: float | None = None,
    figsize: tuple = (9, 5),
) -> np.ndarray:
    """
    绘制 DBSCAN 的 k-distance 曲线

    这张图的作用是辅助选择 `eps`：
    1. 对每个样本，计算它到第 `min_samples` 个近邻的距离；
    2. 把这些距离从小到大排序；
    3. 拐点附近通常可以作为 `eps` 的参考候选值。

    Args:
        X: 特征矩阵
        min_samples: DBSCAN 的 min_samples
        model_name: 模型名称
        current_eps: 当前实际使用的 eps，可选；若传入会画参考线
        figsize: 图尺寸

    Returns:
        np.ndarray: 排序后的 k-distance 数组
    """
    X = np.asarray(X)
    save_dir = get_model_output_dir(model_name)

    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    kth_distances = np.sort(distances[:, -1])

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(np.arange(len(kth_distances)), kth_distances, color="#1E88E5", linewidth=2)
    ax.set_title("DBSCAN 诊断：k-distance 曲线")
    ax.set_xlabel("样本排序索引")
    ax.set_ylabel(f"到第 {min_samples} 个近邻的距离")
    ax.grid(True, alpha=0.25)

    if current_eps is not None:
        ax.axhline(
            current_eps,
            color="#D81B60",
            linestyle="--",
            linewidth=1.6,
            label=f"当前 eps = {current_eps:.3f}",
        )
        ax.legend(loc="best")

    fig.tight_layout()
    _save_figure(fig, save_dir, "k_distance_curve.png")
    return kth_distances


def plot_dbscan_eps_sweep(
    X,
    labels_true,
    min_samples: int,
    metric: str,
    model_name: str,
    current_eps: float,
    eps_values: np.ndarray | None = None,
    figsize: tuple = (12, 9),
) -> pd.DataFrame:
    """
    绘制 DBSCAN 的 eps 扫描评估曲线

    这张图回答的是：
    “当 eps 改变时，聚类结构和评估指标会怎么变化？”

    展示的指标包括：
    1. 簇数量
    2. 噪声点占比
    3. ARI
    4. NMI
    5. Silhouette

    Args:
        X: 特征矩阵
        labels_true: 真实标签（仅用于评估）
        min_samples: DBSCAN 的 min_samples
        metric: 距离度量
        model_name: 模型名称
        current_eps: 当前实际使用的 eps
        eps_values: 手动指定的 eps 候选值；为空则自动生成
        figsize: 图尺寸

    Returns:
        pd.DataFrame: 每个 eps 对应的评估结果
    """
    X = np.asarray(X)
    labels_true = np.asarray(labels_true)
    save_dir = get_model_output_dir(model_name)

    if eps_values is None:
        # 用 k-distance 的分位数自动生成一个相对稳妥的搜索区间。
        nn = NearestNeighbors(n_neighbors=min_samples)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        kth_distances = distances[:, -1]
        start = max(0.05, float(np.percentile(kth_distances, 10)))
        end = float(np.percentile(kth_distances, 95))
        if end <= start:
            end = start + 0.2
        eps_values = np.linspace(start, end, 16)

    # 无论自动生成还是手动传入，都把当前实际使用的 eps 放进扫描点中。
    # 这样诊断摘要里“当前 eps”对应的结果就和正式训练保持一致。
    eps_values = np.unique(np.append(np.asarray(eps_values, dtype=float), current_eps))
    eps_values.sort()

    records = []
    for eps in eps_values:
        model = DBSCAN(eps=float(eps), min_samples=min_samples, metric=metric)
        labels_pred = model.fit_predict(X)
        metrics = evaluate_clustering_with_ground_truth(
            X,
            labels_pred,
            labels_true,
            print_report=False,
        )
        record = {
            "eps": float(eps),
            "n_clusters": metrics["n_clusters"],
            "noise_ratio": metrics["noise_ratio"],
            "ari": metrics["ari"],
            "nmi": metrics["nmi"],
            "silhouette": metrics.get("silhouette", np.nan),
        }
        records.append(record)

    result_df = pd.DataFrame(records)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("DBSCAN 诊断：eps 扫描评估曲线", fontsize=14, fontweight="bold")

    # 簇数量
    axes[0, 0].plot(
        result_df["eps"],
        result_df["n_clusters"],
        marker="o",
        color="#1E88E5",
        linewidth=2,
    )
    axes[0, 0].axvline(current_eps, color="#D81B60", linestyle="--", linewidth=1.5)
    axes[0, 0].set_title("eps vs 簇数量")
    axes[0, 0].set_xlabel("eps")
    axes[0, 0].set_ylabel("簇数量")
    axes[0, 0].grid(True, alpha=0.25)

    # 噪声点占比
    axes[0, 1].plot(
        result_df["eps"],
        result_df["noise_ratio"],
        marker="o",
        color="#E64A19",
        linewidth=2,
    )
    axes[0, 1].axvline(current_eps, color="#D81B60", linestyle="--", linewidth=1.5)
    axes[0, 1].set_title("eps vs 噪声点占比")
    axes[0, 1].set_xlabel("eps")
    axes[0, 1].set_ylabel("噪声点占比")
    axes[0, 1].grid(True, alpha=0.25)

    # ARI / NMI
    axes[1, 0].plot(
        result_df["eps"],
        result_df["ari"],
        marker="o",
        color="#2E7D32",
        linewidth=2,
        label="ARI",
    )
    axes[1, 0].plot(
        result_df["eps"],
        result_df["nmi"],
        marker="s",
        color="#6A1B9A",
        linewidth=2,
        label="NMI",
    )
    axes[1, 0].axvline(current_eps, color="#D81B60", linestyle="--", linewidth=1.5)
    axes[1, 0].set_title("eps vs 外部评估指标")
    axes[1, 0].set_xlabel("eps")
    axes[1, 0].set_ylabel("得分")
    axes[1, 0].legend(loc="best")
    axes[1, 0].grid(True, alpha=0.25)

    # Silhouette
    axes[1, 1].plot(
        result_df["eps"],
        result_df["silhouette"],
        marker="o",
        color="#004D40",
        linewidth=2,
    )
    axes[1, 1].axvline(current_eps, color="#D81B60", linestyle="--", linewidth=1.5)
    axes[1, 1].set_title("eps vs Silhouette")
    axes[1, 1].set_xlabel("eps")
    axes[1, 1].set_ylabel("Silhouette")
    axes[1, 1].grid(True, alpha=0.25)

    fig.tight_layout()
    _save_figure(fig, save_dir, "eps_sweep_curve.png")
    return result_df


def plot_kmeans_k_sweep(
    X,
    labels_true,
    model_name: str,
    current_k: int,
    init: str = "k-means++",
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
    k_values: np.ndarray | None = None,
    figsize: tuple = (12, 9),
) -> pd.DataFrame:
    """
    绘制 KMeans 的 k 扫描评估曲线

    这张图回答的是：
    “当簇数 k 改变时，聚类结构和评估指标会怎么变化？”

    展示的指标包括：
    1. Inertia
    2. Silhouette
    3. ARI
    4. NMI

    Args:
        X: 特征矩阵
        labels_true: 真实标签（仅用于评估）
        model_name: 模型名称
        current_k: 当前实际使用的簇数
        init: 初始化方式
        n_init: 初始化次数
        max_iter: 最大迭代次数
        random_state: 随机种子
        k_values: 手动指定的 k 候选值；为空则自动生成
        figsize: 图尺寸

    Returns:
        pd.DataFrame: 每个 k 对应的评估结果
    """
    X = np.asarray(X)
    labels_true = np.asarray(labels_true)
    save_dir = get_model_output_dir(model_name)

    if k_values is None:
        k_values = np.arange(2, 9)

    k_values = np.unique(np.append(np.asarray(k_values, dtype=int), current_k))
    k_values.sort()

    records = []
    for k in k_values:
        model = KMeans(
            n_clusters=int(k),
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        labels_pred = model.fit_predict(X)
        metrics = evaluate_clustering_with_ground_truth(
            X,
            labels_pred,
            labels_true,
            inertia=model.inertia_,
            print_report=False,
        )
        record = {
            "k": int(k),
            "inertia": metrics["inertia"],
            "ari": metrics["ari"],
            "nmi": metrics["nmi"],
            "silhouette": metrics.get("silhouette", np.nan),
        }
        records.append(record)

    result_df = pd.DataFrame(records)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("KMeans 诊断：k 扫描评估曲线", fontsize=14, fontweight="bold")

    axes[0, 0].plot(
        result_df["k"],
        result_df["inertia"],
        marker="o",
        color="#1E88E5",
        linewidth=2,
    )
    axes[0, 0].axvline(current_k, color="#D81B60", linestyle="--", linewidth=1.5)
    axes[0, 0].set_title("k vs Inertia")
    axes[0, 0].set_xlabel("k")
    axes[0, 0].set_ylabel("Inertia")
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].plot(
        result_df["k"],
        result_df["silhouette"],
        marker="o",
        color="#004D40",
        linewidth=2,
    )
    axes[0, 1].axvline(current_k, color="#D81B60", linestyle="--", linewidth=1.5)
    axes[0, 1].set_title("k vs Silhouette")
    axes[0, 1].set_xlabel("k")
    axes[0, 1].set_ylabel("Silhouette")
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].plot(
        result_df["k"],
        result_df["ari"],
        marker="o",
        color="#2E7D32",
        linewidth=2,
        label="ARI",
    )
    axes[1, 0].plot(
        result_df["k"],
        result_df["nmi"],
        marker="s",
        color="#6A1B9A",
        linewidth=2,
        label="NMI",
    )
    axes[1, 0].axvline(current_k, color="#D81B60", linestyle="--", linewidth=1.5)
    axes[1, 0].set_title("k vs 外部评估指标")
    axes[1, 0].set_xlabel("k")
    axes[1, 0].set_ylabel("得分")
    axes[1, 0].legend(loc="best")
    axes[1, 0].grid(True, alpha=0.25)

    # 右下角给一个综合对照表格感更强的折线图：ARI 与 Inertia 同时看趋势
    axes[1, 1].plot(
        result_df["k"],
        result_df["ari"],
        marker="o",
        color="#E64A19",
        linewidth=2,
        label="ARI",
    )
    normalized_inertia = result_df["inertia"] / result_df["inertia"].max()
    axes[1, 1].plot(
        result_df["k"],
        normalized_inertia,
        marker="^",
        color="#5D4037",
        linewidth=2,
        label="Inertia(归一化)",
    )
    axes[1, 1].axvline(current_k, color="#D81B60", linestyle="--", linewidth=1.5)
    axes[1, 1].set_title("k vs 综合趋势")
    axes[1, 1].set_xlabel("k")
    axes[1, 1].set_ylabel("归一化得分")
    axes[1, 1].legend(loc="best")
    axes[1, 1].grid(True, alpha=0.25)

    fig.tight_layout()
    _save_figure(fig, save_dir, "k_sweep_curve.png")
    return result_df


def plot_gmm_component_sweep(
    X,
    labels_true,
    model_name: str,
    current_components: int,
    covariance_type: str = "full",
    max_iter: int = 200,
    random_state: int = 42,
    component_values: np.ndarray | None = None,
    figsize: tuple = (12, 9),
) -> pd.DataFrame:
    """
    绘制 GMM 的分量数扫描评估曲线

    这张图回答的是：
    “当高斯分量数改变时，模型拟合质量和聚类结果会怎么变化？”

    这里展示的指标包括：
    1. BIC / AIC
    2. ARI / NMI
    3. Silhouette
    4. 对数似然下界

    Args:
        X: 特征矩阵
        labels_true: 真实标签（仅用于评估）
        model_name: 模型名称
        current_components: 当前实际使用的分量数
        covariance_type: 协方差类型
        max_iter: 最大迭代次数
        random_state: 随机种子
        component_values: 手动指定的分量数候选值
        figsize: 图尺寸

    Returns:
        pd.DataFrame: 每个分量数对应的评估结果
    """
    X = np.asarray(X)
    labels_true = np.asarray(labels_true)
    save_dir = get_model_output_dir(model_name)

    if component_values is None:
        component_values = np.arange(1, 7)

    component_values = np.unique(
        np.append(np.asarray(component_values, dtype=int), current_components)
    )
    component_values.sort()

    records = []
    for n_components in component_values:
        model = GaussianMixture(
            n_components=int(n_components),
            covariance_type=covariance_type,
            max_iter=max_iter,
            random_state=random_state,
        )
        model.fit(X)
        labels_pred = model.predict(X)
        metrics = evaluate_clustering_with_ground_truth(
            X,
            labels_pred,
            labels_true,
            print_report=False,
        )
        record = {
            "n_components": int(n_components),
            "bic": model.bic(X),
            "aic": model.aic(X),
            "lower_bound": model.lower_bound_,
            "ari": metrics["ari"],
            "nmi": metrics["nmi"],
            "silhouette": metrics.get("silhouette", np.nan),
        }
        records.append(record)

    result_df = pd.DataFrame(records)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("EM(GMM) 诊断：分量数扫描评估曲线", fontsize=14, fontweight="bold")

    axes[0, 0].plot(
        result_df["n_components"],
        result_df["bic"],
        marker="o",
        color="#1E88E5",
        linewidth=2,
        label="BIC",
    )
    axes[0, 0].plot(
        result_df["n_components"],
        result_df["aic"],
        marker="s",
        color="#E64A19",
        linewidth=2,
        label="AIC",
    )
    axes[0, 0].axvline(
        current_components, color="#D81B60", linestyle="--", linewidth=1.5
    )
    axes[0, 0].set_title("分量数 vs BIC/AIC")
    axes[0, 0].set_xlabel("高斯分量数")
    axes[0, 0].set_ylabel("信息准则")
    axes[0, 0].legend(loc="best")
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].plot(
        result_df["n_components"],
        result_df["ari"],
        marker="o",
        color="#2E7D32",
        linewidth=2,
        label="ARI",
    )
    axes[0, 1].plot(
        result_df["n_components"],
        result_df["nmi"],
        marker="s",
        color="#6A1B9A",
        linewidth=2,
        label="NMI",
    )
    axes[0, 1].axvline(
        current_components, color="#D81B60", linestyle="--", linewidth=1.5
    )
    axes[0, 1].set_title("分量数 vs 外部评估指标")
    axes[0, 1].set_xlabel("高斯分量数")
    axes[0, 1].set_ylabel("得分")
    axes[0, 1].legend(loc="best")
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].plot(
        result_df["n_components"],
        result_df["silhouette"],
        marker="o",
        color="#004D40",
        linewidth=2,
    )
    axes[1, 0].axvline(
        current_components, color="#D81B60", linestyle="--", linewidth=1.5
    )
    axes[1, 0].set_title("分量数 vs Silhouette")
    axes[1, 0].set_xlabel("高斯分量数")
    axes[1, 0].set_ylabel("Silhouette")
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].plot(
        result_df["n_components"],
        result_df["lower_bound"],
        marker="o",
        color="#5D4037",
        linewidth=2,
    )
    axes[1, 1].axvline(
        current_components, color="#D81B60", linestyle="--", linewidth=1.5
    )
    axes[1, 1].set_title("分量数 vs 对数似然下界")
    axes[1, 1].set_xlabel("高斯分量数")
    axes[1, 1].set_ylabel("lower bound")
    axes[1, 1].grid(True, alpha=0.25)

    fig.tight_layout()
    _save_figure(fig, save_dir, "component_sweep_curve.png")
    return result_df
