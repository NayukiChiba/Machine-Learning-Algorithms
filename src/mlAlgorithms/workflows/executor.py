"""
工作流执行分发
"""

from __future__ import annotations

from mlAlgorithms.core.taskTypes import RunnerType
from mlAlgorithms.workflows.classificationRunner import runClassificationPipeline
from mlAlgorithms.workflows.clusteringRunner import runClusteringPipeline
from mlAlgorithms.workflows.dimensionalityRunner import runDimensionalityPipeline
from mlAlgorithms.workflows.probabilisticRunner import runProbabilisticPipeline
from mlAlgorithms.workflows.regressionRunner import runRegressionPipeline


def executePipeline(spec, datasetSpec):
    """按 runner 类型执行流水线。"""
    if spec.runnerType == RunnerType.CLASSIFICATION:
        return runClassificationPipeline(spec, datasetSpec)
    if spec.runnerType == RunnerType.REGRESSION:
        return runRegressionPipeline(spec, datasetSpec)
    if spec.runnerType == RunnerType.CLUSTERING:
        return runClusteringPipeline(spec, datasetSpec)
    if spec.runnerType == RunnerType.DIMENSIONALITY:
        return runDimensionalityPipeline(spec, datasetSpec)
    if spec.runnerType == RunnerType.PROBABILISTIC:
        return runProbabilisticPipeline(spec, datasetSpec)
    raise ValueError(f"不支持的 runnerType: {spec.runnerType}")
