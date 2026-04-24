"""
序列结果可视化
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mlAlgorithms.visualization.figureSaver import saveFigure


def plotHmmDataOverview(data, outputDir: Path) -> list[Path]:
    """绘制 HMM 数据预览图。"""
    artifacts: list[Path] = []
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].step(data["time"], data["obs"], where="mid", color="#1E88E5", linewidth=1.2)
    axes[0].set_ylabel("观测符号")
    axes[0].grid(True, alpha=0.25)
    axes[1].step(
        data["time"], data["state_true"], where="mid", color="#D81B60", linewidth=1.2
    )
    axes[1].set_xlabel("时间步")
    axes[1].set_ylabel("真实隐状态")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle("HMM 数据展示：观测序列与真实隐状态")
    fig.tight_layout()
    artifacts.append(saveFigure(fig, outputDir, "data_sequence.png"))

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    obsCounts = data["obs"].value_counts().sort_index()
    stateCounts = data["state_true"].value_counts().sort_index()
    axes2[0].bar(obsCounts.index.astype(str), obsCounts.values, color="#1E88E5")
    axes2[0].set_title("观测符号分布")
    axes2[1].bar(stateCounts.index.astype(str), stateCounts.values, color="#D81B60")
    axes2[1].set_title("真实隐状态分布")
    for axis in axes2:
        axis.grid(True, axis="y", alpha=0.25)
    fig2.tight_layout()
    artifacts.append(saveFigure(fig2, outputDir, "data_distribution.png"))
    return artifacts


def plotHmmResultFigure(
    data, statesPred, outputDir: Path, filename: str = "result_display.png"
) -> Path:
    """绘制 HMM 结果图。"""
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    axes[0].step(data["time"], data["obs"], where="mid", color="#1E88E5", linewidth=1.1)
    axes[0].set_title("观测序列")
    axes[1].step(
        data["time"], data["state_true"], where="mid", color="#2E7D32", linewidth=1.1
    )
    axes[1].set_title("真实隐状态")
    axes[2].step(data["time"], statesPred, where="mid", color="#D81B60", linewidth=1.1)
    axes[2].set_title("预测隐状态")
    axes[2].set_xlabel("时间步")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.suptitle("HMM 结果展示")
    fig.tight_layout()
    return saveFigure(fig, outputDir, filename)


def plotHmmEvaluationFigure(
    model, yTrue, yPred, outputDir: Path, filename: str = "evaluation_display.png"
) -> Path:
    """绘制 HMM 评估图。"""
    uniqueStates = sorted(
        np.unique(np.concatenate([np.asarray(yTrue), np.asarray(yPred)]))
    )
    confusion = np.zeros((len(uniqueStates), len(uniqueStates)), dtype=int)
    for trueState, predState in zip(yTrue, yPred, strict=True):
        confusion[int(trueState), int(predState)] += 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
    axes[0].set_title("隐状态混淆矩阵")
    sns.heatmap(
        model.transmat_, annot=True, fmt=".3f", cmap="YlOrRd", cbar=False, ax=axes[1]
    )
    axes[1].set_title("转移矩阵")
    if hasattr(model, "emissionprob_"):
        sns.heatmap(
            model.emissionprob_,
            annot=True,
            fmt=".3f",
            cmap="PuBuGn",
            cbar=False,
            ax=axes[2],
        )
        axes[2].set_title("发射矩阵")
    else:
        axes[2].axis("off")
    fig.suptitle("HMM 模型评估")
    fig.tight_layout()
    return saveFigure(fig, outputDir, filename)
