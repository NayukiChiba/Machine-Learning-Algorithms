"""
基础工作流辅助函数
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Any

import numpy as np
from pandas import DataFrame, Series
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE, resolveOutputDir
from mlAlgorithms.analysis.sequenceAnalyzer import buildSequenceExplorationReport
from mlAlgorithms.analysis.tabularAnalyzer import (
    buildClassificationExplorationReport,
    buildClusteringExplorationReport,
    buildRegressionExplorationReport,
)
from mlAlgorithms.analysis.terminalRenderer import printExplorationReport
from mlAlgorithms.core.datasetSpec import DatasetSpec
from mlAlgorithms.core.pipelineSpec import PipelineSpec
from mlAlgorithms.core.runContext import RunContext
from mlAlgorithms.core.runResult import RunResult
from mlAlgorithms.core.taskTypes import DataKind, TaskType


def ensureOptionalDependencies(spec: PipelineSpec) -> None:
    """检查可选依赖。"""
    for dependency in spec.optionalDependencies:
        if importlib.util.find_spec(dependency) is None:
            raise ModuleNotFoundError(dependency)


def buildRunContext(spec: PipelineSpec, datasetSpec: DatasetSpec) -> RunContext:
    """构建一次运行上下文。"""
    data = datasetSpec.load()
    featureColumns = datasetSpec.resolveFeatureColumns(data)
    features = data[featureColumns] if featureColumns else None
    target = (
        data[datasetSpec.targetColumn] if datasetSpec.targetColumn is not None else None
    )
    outputDir = (
        resolveOutputDir(spec.outputKey)
        if spec.outputKey
        else resolveOutputDir("visualization")
    )
    return RunContext(
        spec=spec,
        datasetSpec=datasetSpec,
        data=data,
        features=features,
        target=target,
        outputDir=outputDir,
        randomState=RANDOM_STATE,
    )


def runAnalysis(context: RunContext) -> Any:
    """执行数据探索。"""
    if context.datasetSpec.dataKind == DataKind.SEQUENCE:
        report = buildSequenceExplorationReport(context.data, context.datasetSpec)
    elif context.datasetSpec.taskType in {
        TaskType.CLASSIFICATION,
        TaskType.DIMENSIONALITY,
    }:
        report = buildClassificationExplorationReport(context.data, context.datasetSpec)
    elif context.datasetSpec.taskType == TaskType.REGRESSION:
        report = buildRegressionExplorationReport(context.data, context.datasetSpec)
    else:
        report = buildClusteringExplorationReport(context.data, context.datasetSpec)
    context.analysisReport = report
    printExplorationReport(report)
    return report


def makeSplit(
    X,
    y,
    splitter: str | None,
    randomState: int,
) -> dict[str, Any]:
    """根据 splitter 规则划分数据。"""
    if splitter in {None, "none"}:
        return {"X_all": X, "y_all": y}
    if splitter == "stratifiedSplit":
        XTrain, XTest, yTrain, yTest = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=randomState,
            stratify=y,
        )
    else:
        XTrain, XTest, yTrain, yTest = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=randomState,
        )
    return {
        "X_train": XTrain,
        "X_test": XTest,
        "y_train": yTrain,
        "y_test": yTest,
        "X_all": X,
        "y_all": y,
    }


def applyPreprocessor(
    splitData: dict[str, Any], preprocessor: str | None
) -> dict[str, Any]:
    """根据预处理规则处理数据。"""
    if preprocessor != "standardScaler":
        splitData["X_train_processed"] = splitData.get("X_train")
        splitData["X_test_processed"] = splitData.get("X_test")
        splitData["X_all_processed"] = splitData.get("X_all")
        splitData["scaler"] = None
        return splitData

    scaler = StandardScaler()
    fitSource = splitData.get("X_train", splitData.get("X_all"))
    splitData["X_train_processed"] = scaler.fit_transform(fitSource)
    splitData["X_test_processed"] = (
        scaler.transform(splitData["X_test"])
        if splitData.get("X_test") is not None
        else None
    )
    splitData["X_all_processed"] = scaler.transform(splitData["X_all"])
    splitData["scaler"] = scaler
    return splitData


def collectScoreOutput(model, XTestProcessed):
    """收集预测分数输出。"""
    XPrepared = prepareModelInput(model, XTestProcessed)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(XPrepared)
    if hasattr(model, "decision_function"):
        return model.decision_function(XPrepared)
    return None


def prepareModelInput(model, XInput):
    """按模型训练时的列名对齐输入，避免特征名 warning。"""
    if XInput is None:
        return None
    if isinstance(XInput, DataFrame):
        return XInput
    featureNames = getattr(model, "feature_names_in_", None)
    if featureNames is None:
        return XInput
    XArray = np.asarray(XInput)
    if XArray.ndim != 2 or XArray.shape[1] != len(featureNames):
        return XInput
    return DataFrame(XArray, columns=list(featureNames))


def prepare2dProjection(
    splitData: dict[str, Any],
    yTrain,
    model,
    modelFactory,
) -> dict[str, Any]:
    """为二维展示准备投影与边界模型。"""
    XTrain = np.asarray(splitData["X_train_processed"])
    XTest = np.asarray(splitData["X_test_processed"])
    XAll = np.asarray(splitData["X_all_processed"])
    if XTrain.shape[1] == 2:
        return {
            "X_train_plot": XTrain,
            "X_test_plot": XTest,
            "X_all_plot": XAll,
            "boundary_model": model,
            "feature_names": ["x1", "x2"],
        }
    projector = PCA(n_components=2, random_state=RANDOM_STATE)
    XTrainPlot = projector.fit_transform(XTrain)
    XTestPlot = projector.transform(XTest)
    XAllPlot = projector.transform(XAll)
    boundaryModel = modelFactory() if modelFactory is not None else None
    if boundaryModel is not None:
        boundaryModel.fit(XTrainPlot, yTrain)
    return {
        "X_train_plot": XTrainPlot,
        "X_test_plot": XTestPlot,
        "X_all_plot": XAllPlot,
        "boundary_model": boundaryModel,
        "feature_names": ["PC1", "PC2"],
    }


def alignClusterLabelsForDisplay(labelsPred, labelsTrue):
    """为展示用途对齐聚类标签。"""
    labelsPred = Series(labelsPred)
    labelsTrue = Series(labelsTrue)
    aligned = labelsPred.copy()
    for clusterId in sorted(labelsPred.unique()):
        mask = labelsPred == clusterId
        majorityLabel = labelsTrue[mask].value_counts().idxmax()
        aligned.loc[mask] = majorityLabel
    return aligned.to_numpy()


def makeRunResult(
    model, predictions=None, scores=None, metrics=None, extras=None
) -> RunResult:
    """构建运行结果。"""
    return RunResult(
        model=model,
        predictions=predictions,
        scores=scores,
        metrics=metrics or {},
        extras=extras or {},
    )


def callTrainer(trainer, *args, **kwargs):
    """按训练函数签名过滤关键字参数。"""
    signature = inspect.signature(trainer)
    accepted = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return trainer(*args, **accepted)


def appendArtifact(result: RunResult, artifact: Path | None) -> None:
    """向结果中追加单个产物。"""
    if artifact is None:
        return
    result.artifacts.append(artifact)


def appendArtifacts(result: RunResult, artifacts: list[Path]) -> None:
    """向结果中追加多个产物。"""
    for artifact in artifacts:
        appendArtifact(result, artifact)
