"""
model_evaluation/clustering_metrics.py
聚类模型评估指标

包含: 轮廓系数 (Silhouette)、Inertia、DB 指数 (Davies-Bouldin)、CH 指数 (Calinski-Harabasz)

使用方式:
    from model_evaluation.clustering_metrics import evaluate_clustering
"""

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)


def evaluate_clustering(
    X,
    labels,
    inertia: float | None = None,
    print_report: bool = True,
) -> dict:
    """
    计算聚类模型的全套评估指标

    Args:
        X: 特征矩阵
        labels: 聚类标签
        inertia: 簇内平方和（如 KMeans 的 .inertia_，可选）
        print_report: 是否打印报告

    Returns:
        dict: 包含所有指标的字典
    """
    X = np.asarray(X)
    labels = np.asarray(labels)

    n_clusters = len(set(labels) - {-1})
    metrics = {"n_clusters": n_clusters}

    if n_clusters >= 2:
        # 排除噪声点 (label == -1) 进行评估
        mask = labels != -1
        if mask.sum() > n_clusters:
            metrics["silhouette"] = silhouette_score(X[mask], labels[mask])
            metrics["davies_bouldin"] = davies_bouldin_score(X[mask], labels[mask])
            metrics["calinski_harabasz"] = calinski_harabasz_score(
                X[mask], labels[mask]
            )

    if inertia is not None:
        metrics["inertia"] = inertia

    if print_report:
        print("=" * 60)
        print("聚类评估报告")
        print("=" * 60)
        print(f"  簇数量:                    {metrics['n_clusters']}")
        if "silhouette" in metrics:
            print(f"  轮廓系数 (Silhouette):     {metrics['silhouette']:.4f}")
            print(f"  DB 指数 (Davies-Bouldin):  {metrics['davies_bouldin']:.4f}")
            print(f"  CH 指数 (Calinski-Harabasz): {metrics['calinski_harabasz']:.2f}")
        if "inertia" in metrics:
            print(f"  Inertia (簇内平方和):      {metrics['inertia']:.4f}")

    return metrics


def evaluate_clustering_with_ground_truth(
    X,
    labels_pred,
    labels_true,
    inertia: float | None = None,
    print_report: bool = True,
) -> dict:
    """
    计算带真实标签参照的聚类评估指标

    适用于像 DBSCAN / KMeans / GMM 这类：
    1. 训练时不使用标签；
    2. 但我们手里有 synthetic data 的真实标签可用于事后评估。

    Args:
        X: 特征矩阵
        labels_pred: 预测簇标签
        labels_true: 真实标签
        print_report: 是否打印报告

    Returns:
        dict: 聚类评估结果
    """
    X = np.asarray(X)
    labels_pred = np.asarray(labels_pred)
    labels_true = np.asarray(labels_true)

    n_clusters = len(set(labels_pred) - {-1})
    n_noise = int((labels_pred == -1).sum())
    noise_ratio = n_noise / len(labels_pred)

    metrics = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": noise_ratio,
        "ari": adjusted_rand_score(labels_true, labels_pred),
        "nmi": normalized_mutual_info_score(labels_true, labels_pred),
        "homogeneity": homogeneity_score(labels_true, labels_pred),
        "completeness": completeness_score(labels_true, labels_pred),
        "v_measure": v_measure_score(labels_true, labels_pred),
    }

    if inertia is not None:
        metrics["inertia"] = inertia

    if n_clusters >= 2:
        mask = labels_pred != -1
        if mask.sum() > n_clusters:
            metrics["silhouette"] = silhouette_score(X[mask], labels_pred[mask])
            metrics["davies_bouldin"] = davies_bouldin_score(X[mask], labels_pred[mask])
            metrics["calinski_harabasz"] = calinski_harabasz_score(
                X[mask], labels_pred[mask]
            )

    if print_report:
        print("=" * 60)
        print("聚类模型评估展示")
        print("=" * 60)
        print(f"簇数量: {metrics['n_clusters']}")
        print(f"噪声点数量: {metrics['n_noise']}")
        print(f"噪声点占比: {metrics['noise_ratio']:.4f}")
        print(f"ARI: {metrics['ari']:.4f}")
        print(f"NMI: {metrics['nmi']:.4f}")
        print(f"Homogeneity: {metrics['homogeneity']:.4f}")
        print(f"Completeness: {metrics['completeness']:.4f}")
        print(f"V-measure: {metrics['v_measure']:.4f}")
        if "inertia" in metrics:
            print(f"Inertia: {metrics['inertia']:.4f}")
        if "silhouette" in metrics:
            print(f"Silhouette: {metrics['silhouette']:.4f}")
            print(f"Davies-Bouldin: {metrics['davies_bouldin']:.4f}")
            print(f"Calinski-Harabasz: {metrics['calinski_harabasz']:.2f}")

    return metrics


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans

    X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    model.fit(X)
    evaluate_clustering(X, model.labels_, inertia=model.inertia_)
