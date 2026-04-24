"""
降维结果与诊断可视化
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from mlAlgorithms.visualization.figureSaver import saveFigure


def plotDimensionality(
    XTransformed,
    y,
    explainedVarianceRatio,
    outputDir: Path,
    title: str,
    mode: str,
    filename: str,
) -> Path:
    """绘制降维结果。"""
    XTransformed = np.asarray(XTransformed)
    if mode == "3d":
        fig = plt.figure(figsize=(8, 6))
        axis = fig.add_subplot(111, projection="3d")
        axis.scatter(
            XTransformed[:, 0],
            XTransformed[:, 1],
            XTransformed[:, 2],
            c=y,
            s=20,
            alpha=0.7,
            cmap="viridis",
        )
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.set_zlabel("PC3")
    else:
        fig, axis = plt.subplots(figsize=(7, 5))
        axis.scatter(
            XTransformed[:, 0], XTransformed[:, 1], c=y, s=20, alpha=0.7, cmap="viridis"
        )
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
    axis.set_title(title)
    if explainedVarianceRatio is not None:
        axis.set_title(
            f"{title}\n解释方差比: {np.round(explainedVarianceRatio, 4).tolist()}"
        )
    if mode != "3d":
        axis.grid(True, alpha=0.25)
    return saveFigure(fig, outputDir, filename)


def plotPcaTrainingCurve(
    XScaled, outputDir: Path, maxComponents: int, filename: str = "training_curve.png"
) -> tuple[Path, dict[str, np.ndarray]]:
    """绘制 PCA 训练诊断曲线。"""
    componentRange = np.arange(1, maxComponents + 1)
    cumulativeVariances: list[float] = []
    reconstructionErrors: list[float] = []
    for componentCount in componentRange:
        model = PCA(n_components=int(componentCount), random_state=42)
        XReduced = model.fit_transform(XScaled)
        XReconstructed = model.inverse_transform(XReduced)
        cumulativeVariances.append(float(model.explained_variance_ratio_.sum()))
        reconstructionErrors.append(float(np.mean((XScaled - XReconstructed) ** 2)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(componentRange, cumulativeVariances, marker="o")
    axes[0].set_title("累计解释方差比")
    axes[0].set_xlabel("主成分数")
    axes[0].set_ylabel("累计解释方差比")
    axes[1].plot(componentRange, reconstructionErrors, marker="o", color="#D81B60")
    axes[1].set_title("重建误差")
    axes[1].set_xlabel("主成分数")
    axes[1].set_ylabel("MSE")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.suptitle("PCA 训练诊断曲线")
    fig.tight_layout()
    data = {
        "component_range": componentRange,
        "cumulative_variances": np.asarray(cumulativeVariances),
        "reconstruction_errors": np.asarray(reconstructionErrors),
    }
    path = saveFigure(fig, outputDir, filename)
    return path, data
