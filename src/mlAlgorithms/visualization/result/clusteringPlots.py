"""
聚类结果与诊断可视化
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from mlAlgorithms.evaluation.clusteringEvaluator import (
    evaluateClusteringWithGroundTruth,
)
from mlAlgorithms.visualization.figureSaver import saveFigure


def plotClusters(
    X,
    labelsPred,
    labelsTrue,
    featureNames: list[str],
    outputDir: Path,
    title: str,
    centers=None,
    filename: str = "cluster_plot.png",
) -> Path:
    """绘制聚类结果图。"""
    X = np.asarray(X)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    axes[0].scatter(X[:, 0], X[:, 1], c=labelsTrue, s=22, alpha=0.75, cmap="viridis")
    axes[0].set_title("真实标签")
    axes[1].scatter(X[:, 0], X[:, 1], c=labelsPred, s=22, alpha=0.75, cmap="viridis")
    if centers is not None:
        centers = np.asarray(centers)
        axes[1].scatter(centers[:, 0], centers[:, 1], c="red", s=140, marker="X")
    axes[1].set_title("预测簇")
    for axis in axes:
        axis.set_xlabel(featureNames[0])
        axis.set_ylabel(featureNames[1])
        axis.grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    return saveFigure(fig, outputDir, filename)


def plotKmeansSweep(
    X,
    labelsTrue,
    outputDir: Path,
    currentK: int = 4,
    filename: str = "k_sweep_curve.png",
) -> tuple[Path, pd.DataFrame]:
    """绘制 KMeans k 扫描曲线。"""
    records = []
    for kValue in np.arange(2, 9):
        model = KMeans(
            n_clusters=int(kValue),
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42,
        )
        labelsPred = model.fit_predict(X)
        metrics = evaluateClusteringWithGroundTruth(
            X, labelsPred, labelsTrue, inertia=model.inertia_, printReport=False
        )
        records.append(
            {
                "k": int(kValue),
                "inertia": metrics["inertia"],
                "ari": metrics["ari"],
                "nmi": metrics["nmi"],
                "silhouette": metrics.get("silhouette", np.nan),
            }
        )
    result = pd.DataFrame(records)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].plot(result["k"], result["inertia"], marker="o")
    axes[0, 1].plot(result["k"], result["silhouette"], marker="o")
    axes[1, 0].plot(result["k"], result["ari"], marker="o", label="ARI")
    axes[1, 0].plot(result["k"], result["nmi"], marker="s", label="NMI")
    normalizedInertia = result["inertia"] / result["inertia"].max()
    axes[1, 1].plot(result["k"], result["ari"], marker="o", label="ARI")
    axes[1, 1].plot(result["k"], normalizedInertia, marker="^", label="Inertia(归一化)")
    for axis in axes.ravel():
        axis.axvline(currentK, color="#D81B60", linestyle="--", linewidth=1.2)
        axis.grid(True, alpha=0.25)
    axes[1, 0].legend(loc="best")
    axes[1, 1].legend(loc="best")
    fig.suptitle("KMeans 诊断：k 扫描评估曲线")
    fig.tight_layout()
    path = saveFigure(fig, outputDir, filename)
    return path, result


def plotDbscanKDistance(
    X,
    outputDir: Path,
    minSamples: int = 5,
    currentEps: float | None = None,
    filename: str = "k_distance_curve.png",
) -> tuple[Path, np.ndarray]:
    """绘制 DBSCAN k-distance 曲线。"""
    neighbors = NearestNeighbors(n_neighbors=minSamples)
    neighbors.fit(X)
    distances, _ = neighbors.kneighbors(X)
    kthDistances = np.sort(distances[:, -1])
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(np.arange(len(kthDistances)), kthDistances, color="#1E88E5")
    if currentEps is not None:
        axis.axhline(
            currentEps, color="#D81B60", linestyle="--", label=f"eps={currentEps:.3f}"
        )
        axis.legend(loc="best")
    axis.set_title("DBSCAN 诊断：k-distance 曲线")
    axis.set_xlabel("样本索引")
    axis.set_ylabel(f"第 {minSamples} 个近邻距离")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    path = saveFigure(fig, outputDir, filename)
    return path, kthDistances


def plotDbscanEpsSweep(
    X,
    labelsTrue,
    outputDir: Path,
    currentEps: float = 0.3,
    minSamples: int = 5,
    filename: str = "eps_sweep_curve.png",
) -> tuple[Path, pd.DataFrame]:
    """绘制 DBSCAN eps 扫描曲线。"""
    epsValues = np.linspace(0.05, 1.0, 16)
    records = []
    for epsValue in epsValues:
        model = DBSCAN(eps=float(epsValue), min_samples=minSamples, metric="euclidean")
        labelsPred = model.fit_predict(X)
        metrics = evaluateClusteringWithGroundTruth(
            X, labelsPred, labelsTrue, printReport=False
        )
        records.append(
            {
                "eps": float(epsValue),
                "n_clusters": metrics["n_clusters"],
                "noise_ratio": metrics["noise_ratio"],
                "ari": metrics["ari"],
                "nmi": metrics["nmi"],
                "silhouette": metrics.get("silhouette", np.nan),
            }
        )
    result = pd.DataFrame(records)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].plot(result["eps"], result["n_clusters"], marker="o")
    axes[0, 1].plot(result["eps"], result["noise_ratio"], marker="o")
    axes[1, 0].plot(result["eps"], result["ari"], marker="o", label="ARI")
    axes[1, 0].plot(result["eps"], result["nmi"], marker="s", label="NMI")
    axes[1, 1].plot(result["eps"], result["silhouette"], marker="o")
    for axis in axes.ravel():
        axis.axvline(currentEps, color="#D81B60", linestyle="--", linewidth=1.2)
        axis.grid(True, alpha=0.25)
    axes[1, 0].legend(loc="best")
    fig.suptitle("DBSCAN 诊断：eps 扫描评估曲线")
    fig.tight_layout()
    path = saveFigure(fig, outputDir, filename)
    return path, result


def plotGmmComponentSweep(
    X,
    labelsTrue,
    outputDir: Path,
    currentComponents: int = 3,
    filename: str = "component_sweep_curve.png",
) -> tuple[Path, pd.DataFrame]:
    """绘制 GMM 分量扫描曲线。"""
    records = []
    for componentCount in np.arange(1, 7):
        model = GaussianMixture(
            n_components=int(componentCount),
            covariance_type="full",
            max_iter=200,
            random_state=42,
        )
        model.fit(X)
        labelsPred = model.predict(X)
        metrics = evaluateClusteringWithGroundTruth(
            X, labelsPred, labelsTrue, printReport=False
        )
        records.append(
            {
                "n_components": int(componentCount),
                "bic": model.bic(X),
                "aic": model.aic(X),
                "lower_bound": model.lower_bound_,
                "ari": metrics["ari"],
                "nmi": metrics["nmi"],
                "silhouette": metrics.get("silhouette", np.nan),
            }
        )
    result = pd.DataFrame(records)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].plot(result["n_components"], result["bic"], marker="o", label="BIC")
    axes[0, 0].plot(result["n_components"], result["aic"], marker="s", label="AIC")
    axes[0, 1].plot(result["n_components"], result["ari"], marker="o", label="ARI")
    axes[0, 1].plot(result["n_components"], result["nmi"], marker="s", label="NMI")
    axes[1, 0].plot(result["n_components"], result["silhouette"], marker="o")
    axes[1, 1].plot(result["n_components"], result["lower_bound"], marker="o")
    for axis in axes.ravel():
        axis.axvline(currentComponents, color="#D81B60", linestyle="--", linewidth=1.2)
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(loc="best")
    axes[0, 1].legend(loc="best")
    fig.suptitle("EM(GMM) 诊断：分量数扫描曲线")
    fig.tight_layout()
    path = saveFigure(fig, outputDir, filename)
    return path, result
