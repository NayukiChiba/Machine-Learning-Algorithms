"""
训练前数据可视化
"""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from sklearn.decomposition import PCA

from mlAlgorithms.visualization.figureSaver import saveFigure

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def plotClassDistribution(
    data: DataFrame, targetColumn: str, outputDir: Path, title: str, filename: str
) -> Path:
    """绘制类别分布图。"""
    counts = data[targetColumn].value_counts().sort_index()
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(counts.index.astype(str), counts.values, color="#1E88E5")
    axis.set_title(title)
    axis.set_xlabel(targetColumn)
    axis.set_ylabel("样本数")
    axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return saveFigure(fig, outputDir, filename)


def plotCorrelationHeatmap(
    data: DataFrame,
    columns: list[str],
    outputDir: Path,
    title: str,
    filename: str,
    annot: bool = True,
) -> Path:
    """绘制相关性热力图。"""
    corr = data[columns].corr(method="pearson")
    width = max(10, len(columns) * 1.1)
    height = max(8, len(columns) * 0.9)
    showAnnot = annot and len(columns) <= 15
    fig, axis = plt.subplots(figsize=(width, height))
    sns.heatmap(
        corr,
        annot=showAnnot,
        fmt=".2f" if showAnnot else "",
        cmap="coolwarm",
        center=0,
        square=True,
        ax=axis,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
        vmin=-1,
        vmax=1,
    )
    labels = [fill(column.replace("_", " "), width=18) for column in columns]
    axis.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor")
    axis.set_yticklabels(labels, rotation=0, va="center")
    axis.set_title(title)
    fig.subplots_adjust(left=0.26, bottom=0.27, right=0.95, top=0.92)
    return saveFigure(fig, outputDir, filename)


def plotRaw2dScatter(
    data: DataFrame,
    xColumn: str,
    yColumn: str,
    outputDir: Path,
    title: str,
    filename: str,
) -> Path:
    """绘制原始二维散点图。"""
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(data[xColumn], data[yColumn], s=18, alpha=0.65, color="#1E88E5")
    axis.set_title(title)
    axis.set_xlabel(xColumn)
    axis.set_ylabel(yColumn)
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    return saveFigure(fig, outputDir, filename)


def plotLabeled2dScatter(
    data: DataFrame,
    xColumn: str,
    yColumn: str,
    labelColumn: str,
    outputDir: Path,
    title: str,
    filename: str,
) -> Path:
    """绘制带标签的二维散点图。"""
    fig, axis = plt.subplots(figsize=(7, 5))
    scatter = axis.scatter(
        data[xColumn],
        data[yColumn],
        c=data[labelColumn],
        s=20,
        alpha=0.7,
        cmap="viridis",
    )
    axis.set_title(title)
    axis.set_xlabel(xColumn)
    axis.set_ylabel(yColumn)
    axis.grid(True, alpha=0.25)
    legend = axis.legend(*scatter.legend_elements(), title=labelColumn, loc="best")
    axis.add_artist(legend)
    fig.tight_layout()
    return saveFigure(fig, outputDir, filename)


def plotFeatureSpace2d(
    data: DataFrame,
    featureColumns: list[str],
    labelColumn: str,
    outputDir: Path,
    title: str,
    filename: str,
) -> Path:
    """绘制 2D 特征空间图。"""
    X = data[featureColumns].to_numpy()
    if X.shape[1] > 2:
        XPlot = PCA(n_components=2, random_state=42).fit_transform(X)
        xLabel, yLabel = "PC1", "PC2"
    else:
        XPlot = X
        xLabel, yLabel = featureColumns[0], featureColumns[1]
    fig, axis = plt.subplots(figsize=(7, 5))
    scatter = axis.scatter(
        XPlot[:, 0], XPlot[:, 1], c=data[labelColumn], s=20, alpha=0.7, cmap="viridis"
    )
    axis.set_title(title)
    axis.set_xlabel(xLabel)
    axis.set_ylabel(yLabel)
    axis.grid(True, alpha=0.25)
    legend = axis.legend(*scatter.legend_elements(), title=labelColumn, loc="best")
    axis.add_artist(legend)
    fig.tight_layout()
    return saveFigure(fig, outputDir, filename)


def plotFeatureSpace3d(
    data: DataFrame,
    featureColumns: list[str],
    labelColumn: str,
    outputDir: Path,
    title: str,
    filename: str,
) -> Path:
    """绘制 3D 特征空间图。"""
    X = data[featureColumns].to_numpy()
    if X.shape[1] > 3:
        XPlot = PCA(n_components=3, random_state=42).fit_transform(X)
        axisLabels = ["PC1", "PC2", "PC3"]
    else:
        XPlot = X[:, :3]
        axisLabels = featureColumns[:3]
    fig = plt.figure(figsize=(8, 6))
    axis = fig.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        XPlot[:, 0],
        XPlot[:, 1],
        XPlot[:, 2],
        c=data[labelColumn],
        s=18,
        alpha=0.7,
        cmap="viridis",
    )
    axis.set_title(title)
    axis.set_xlabel(axisLabels[0])
    axis.set_ylabel(axisLabels[1])
    axis.set_zlabel(axisLabels[2])
    axis.legend(*scatter.legend_elements(), title=labelColumn, loc="best")
    return saveFigure(fig, outputDir, filename)
