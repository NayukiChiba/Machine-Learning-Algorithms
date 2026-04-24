"""
回归结果可视化
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mlAlgorithms.visualization.figureSaver import saveFigure


def plotRegressionResult(
    yTrue, yPred, outputDir: Path, title: str, filename: str = "result_display.png"
) -> Path:
    """绘制回归结果图。"""
    yTrue = np.asarray(yTrue)
    yPred = np.asarray(yPred)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(yTrue, yPred, s=24, alpha=0.7, color="#1E88E5")
    lower = min(yTrue.min(), yPred.min())
    upper = max(yTrue.max(), yPred.max())
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="gray")
    axis.set_title(title)
    axis.set_xlabel("真实值")
    axis.set_ylabel("预测值")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    return saveFigure(fig, outputDir, filename)


def plotResiduals(
    yTrue, yPred, outputDir: Path, title: str, filename: str = "residual_plot.png"
) -> Path:
    """绘制残差图。"""
    yTrue = np.asarray(yTrue)
    yPred = np.asarray(yPred)
    residuals = yTrue - yPred
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(yPred, residuals, s=24, alpha=0.7, color="#D81B60")
    axis.axhline(0.0, linestyle="--", color="gray")
    axis.set_title(title)
    axis.set_xlabel("预测值")
    axis.set_ylabel("残差")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    return saveFigure(fig, outputDir, filename)
