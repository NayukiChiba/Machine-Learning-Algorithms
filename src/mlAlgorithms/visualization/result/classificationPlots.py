"""
分类结果可视化
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import DataFrame, Series
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize
from sklearn.tree import export_text, plot_tree

from mlAlgorithms.visualization.figureSaver import saveFigure
from mlAlgorithms.workflows.baseRunner import prepareModelInput


def plotClassificationResult(
    XPlot,
    yTrue,
    yPred,
    featureNames: list[str],
    outputDir: Path,
    title: str,
    filename: str = "result_display.png",
) -> Path:
    """绘制真实标签与预测标签对照图。"""
    XPlot = np.asarray(XPlot)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    axes[0].scatter(XPlot[:, 0], XPlot[:, 1], c=yTrue, s=22, alpha=0.75, cmap="viridis")
    axes[0].set_title("真实标签")
    axes[1].scatter(XPlot[:, 0], XPlot[:, 1], c=yPred, s=22, alpha=0.75, cmap="viridis")
    axes[1].set_title("预测标签")
    for axis in axes:
        axis.set_xlabel(featureNames[0])
        axis.set_ylabel(featureNames[1])
        axis.grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    return saveFigure(fig, outputDir, filename)


def plotConfusionMatrix(
    yTrue, yPred, outputDir: Path, title: str, filename: str = "confusion_matrix.png"
) -> Path:
    """绘制混淆矩阵。"""
    labels = sorted(np.unique(np.concatenate([np.asarray(yTrue), np.asarray(yPred)])))
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    labelToIndex = {label: index for index, label in enumerate(labels)}
    for trueValue, predValue in zip(yTrue, yPred, strict=True):
        matrix[labelToIndex[trueValue], labelToIndex[predValue]] += 1
    fig, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
    axis.set_title(title)
    axis.set_xlabel("预测")
    axis.set_ylabel("真实")
    axis.set_xticklabels(labels)
    axis.set_yticklabels(labels, rotation=0)
    fig.tight_layout()
    return saveFigure(fig, outputDir, filename)


def plotRocCurve(
    yTrue, yScores, outputDir: Path, title: str, filename: str = "roc_curve.png"
) -> Path:
    """绘制 ROC 曲线。"""
    yTrue = np.asarray(yTrue)
    yScores = np.asarray(yScores)
    fig, axis = plt.subplots(figsize=(6, 5))
    classes = np.unique(yTrue)
    if len(classes) == 2:
        score = yScores[:, 1] if yScores.ndim == 2 else yScores
        RocCurveDisplay.from_predictions(yTrue, score, ax=axis)
    else:
        yBinarized = label_binarize(yTrue, classes=classes)
        for index, classValue in enumerate(classes):
            fpr, tpr, _ = roc_curve(yBinarized[:, index], yScores[:, index])
            axis.plot(fpr, tpr, label=f"类别 {classValue} (AUC={auc(fpr, tpr):.3f})")
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
        axis.legend(loc="lower right")
        axis.set_xlabel("False Positive Rate")
        axis.set_ylabel("True Positive Rate")
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    return saveFigure(fig, outputDir, filename)


def plotDecisionBoundary(
    model,
    X,
    y,
    featureNames: list[str],
    outputDir: Path,
    title: str,
    filename: str = "decision_boundary.png",
) -> Path:
    """绘制决策边界。"""
    X = np.asarray(X)
    y = np.asarray(y)
    xMin, xMax = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    yMin, yMax = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    gridX, gridY = np.meshgrid(
        np.linspace(xMin, xMax, 240),
        np.linspace(yMin, yMax, 240),
    )
    grid = np.c_[gridX.ravel(), gridY.ravel()]
    z = model.predict(prepareModelInput(model, grid)).reshape(gridX.shape)
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.contourf(gridX, gridY, z, cmap="Pastel1", alpha=0.7)
    axis.scatter(
        X[:, 0], X[:, 1], c=y, s=20, cmap="viridis", edgecolors="black", linewidths=0.2
    )
    axis.set_title(title)
    axis.set_xlabel(featureNames[0])
    axis.set_ylabel(featureNames[1])
    axis.grid(True, alpha=0.2)
    fig.tight_layout()
    return saveFigure(fig, outputDir, filename)


def plotFeatureImportance(
    model,
    featureNames: list[str],
    outputDir: Path,
    title: str,
    filename: str = "feature_importance.png",
) -> Path | None:
    """绘制特征重要性或系数图。"""
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_)
        if coefficients.ndim == 2:
            importances = np.mean(np.abs(coefficients), axis=0)
        else:
            importances = np.abs(coefficients)
    if importances is None:
        return None
    order = np.argsort(importances)
    fig, axis = plt.subplots(figsize=(8, max(4, len(featureNames) * 0.35)))
    axis.barh(np.asarray(featureNames)[order], importances[order], color="#1E88E5")
    axis.set_title(title)
    axis.set_xlabel("重要性")
    axis.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    return saveFigure(fig, outputDir, filename)


def plotLearningCurve(
    estimator,
    X,
    y,
    outputDir: Path,
    title: str,
    filename: str = "learning_curve.png",
    scoring: str = "accuracy",
) -> Path:
    """绘制学习曲线。"""
    # LightGBM 即使用 ndarray 训练，也会自动生成 Column_0 这类特征名。
    # 因此这里统一把学习曲线输入包装成带稳定列名的 DataFrame，
    # 保证交叉验证内部 fit / score 两侧看到的特征名一致。
    XArray = np.asarray(X)
    if XArray.ndim == 2:
        XValues = DataFrame(
            XArray,
            columns=[f"Column_{index}" for index in range(XArray.shape[1])],
        )
    else:
        XValues = XArray
    yValues = Series(np.asarray(y))
    trainSizes, trainScores, testScores = learning_curve(
        estimator,
        XValues,
        yValues,
        cv=5,
        n_jobs=1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring=scoring,
    )
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(trainSizes, trainScores.mean(axis=1), marker="o", label="训练得分")
    axis.plot(trainSizes, testScores.mean(axis=1), marker="s", label="验证得分")
    axis.set_title(title)
    axis.set_xlabel("训练样本数")
    axis.set_ylabel(scoring)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    return saveFigure(fig, outputDir, filename)


def plotTreeStructure(
    model,
    featureNames: list[str],
    classNames: list[str] | None,
    outputDir: Path,
    title: str,
    filename: str = "tree_structure.png",
) -> Path:
    """绘制树结构。"""
    fig, axis = plt.subplots(figsize=(16, 8))
    plot_tree(
        model,
        feature_names=featureNames,
        class_names=classNames,
        filled=True,
        ax=axis,
        rounded=True,
        fontsize=8,
    )
    axis.set_title(title)
    fig.tight_layout()
    return saveFigure(fig, outputDir, filename)


def formatTreeRules(model, featureNames: list[str]) -> str:
    """导出树规则文本。"""
    return export_text(model, feature_names=featureNames)
