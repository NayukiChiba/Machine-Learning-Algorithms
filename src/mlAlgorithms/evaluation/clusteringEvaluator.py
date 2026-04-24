"""
聚类评估器
"""

from __future__ import annotations

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


def evaluateClustering(
    X, labels, inertia: float | None = None, printReport: bool = True
) -> dict:
    """评估无监督聚类。"""
    X = np.asarray(X)
    labels = np.asarray(labels)
    nClusters = len(set(labels) - {-1})
    metrics = {"n_clusters": nClusters}
    if nClusters >= 2:
        mask = labels != -1
        if mask.sum() > nClusters:
            metrics["silhouette"] = silhouette_score(X[mask], labels[mask])
            metrics["davies_bouldin"] = davies_bouldin_score(X[mask], labels[mask])
            metrics["calinski_harabasz"] = calinski_harabasz_score(
                X[mask], labels[mask]
            )
    if inertia is not None:
        metrics["inertia"] = inertia
    if printReport:
        print("=" * 60)
        print("聚类评估摘要")
        print("=" * 60)
        print(f"簇数量: {metrics['n_clusters']}")
    return metrics


def evaluateClusteringWithGroundTruth(
    X, labelsPred, labelsTrue, inertia: float | None = None, printReport: bool = True
) -> dict:
    """评估带真实标签的聚类任务。"""
    X = np.asarray(X)
    labelsPred = np.asarray(labelsPred)
    labelsTrue = np.asarray(labelsTrue)
    nClusters = len(set(labelsPred) - {-1})
    nNoise = int((labelsPred == -1).sum())
    metrics = {
        "n_clusters": nClusters,
        "n_noise": nNoise,
        "noise_ratio": nNoise / len(labelsPred),
        "ari": adjusted_rand_score(labelsTrue, labelsPred),
        "nmi": normalized_mutual_info_score(labelsTrue, labelsPred),
        "homogeneity": homogeneity_score(labelsTrue, labelsPred),
        "completeness": completeness_score(labelsTrue, labelsPred),
        "v_measure": v_measure_score(labelsTrue, labelsPred),
    }
    if inertia is not None:
        metrics["inertia"] = inertia
    if nClusters >= 2:
        mask = labelsPred != -1
        if mask.sum() > nClusters:
            metrics["silhouette"] = silhouette_score(X[mask], labelsPred[mask])
            metrics["davies_bouldin"] = davies_bouldin_score(X[mask], labelsPred[mask])
            metrics["calinski_harabasz"] = calinski_harabasz_score(
                X[mask], labelsPred[mask]
            )
    if printReport:
        print("=" * 60)
        print("聚类评估摘要")
        print("=" * 60)
        print(f"ARI: {metrics['ari']:.4f}")
        print(f"NMI: {metrics['nmi']:.4f}")
        print(f"噪声点占比: {metrics['noise_ratio']:.4f}")
    return metrics
