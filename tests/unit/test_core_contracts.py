"""
核心契约测试
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pandas import DataFrame, Series

from mlAlgorithms.core.datasetSpec import DatasetSpec
from mlAlgorithms.core.pipelineSpec import PipelineSpec
from mlAlgorithms.core.runContext import RunContext
from mlAlgorithms.core.runResult import RunResult
from mlAlgorithms.core.taskTypes import DataKind, RunnerType, TaskType
from mlAlgorithms.workflows.baseRunner import prepareModelInput


def testDatasetSpecResolvesFeatureColumns():
    """应能解析特征列。"""
    spec = DatasetSpec(
        id="demo.dataset",
        taskType=TaskType.CLASSIFICATION,
        dataKind=DataKind.TABULAR,
        loader=lambda: DataFrame({"x1": [1], "x2": [2], "label": [0]}),
        targetColumn="label",
        featureColumns=None,
        description="demo",
    )
    data = spec.load()
    assert spec.resolveFeatureColumns(data) == ["x1", "x2"]


def testPipelineSpecStoresFixedFields():
    """流水线规格应保存固定字段。"""
    spec = PipelineSpec(
        id="classification.demo",
        taskType=TaskType.CLASSIFICATION,
        datasetId="demo.dataset",
        runnerType=RunnerType.CLASSIFICATION,
        trainer=lambda X, y: None,
        preprocessor="standardScaler",
        splitter="stratifiedSplit",
        predictor="default",
        evaluator="classification",
        analysisProfile="classification",
        outputKey="demo",
    )
    assert spec.id == "classification.demo"
    assert spec.outputKey == "demo"
    assert spec.analysisProfile == "classification"


def testRunContextAndRunResultDefaults():
    """运行上下文和结果应有稳定默认值。"""
    datasetSpec = DatasetSpec(
        id="demo.dataset",
        taskType=TaskType.REGRESSION,
        dataKind=DataKind.TABULAR,
        loader=lambda: DataFrame({"x1": [1.0], "price": [2.0]}),
        targetColumn="price",
        featureColumns=["x1"],
        description="demo",
    )
    pipelineSpec = PipelineSpec(
        id="regression.demo",
        taskType=TaskType.REGRESSION,
        datasetId="demo.dataset",
        runnerType=RunnerType.REGRESSION,
        trainer=lambda X, y: None,
        preprocessor=None,
        splitter=None,
        predictor=None,
        evaluator=None,
        analysisProfile="regression",
        outputKey="linear_regression",
    )
    context = RunContext(
        spec=pipelineSpec,
        datasetSpec=datasetSpec,
        data=DataFrame({"x1": [1.0], "price": [2.0]}),
        features=DataFrame({"x1": [1.0]}),
        target=Series([2.0]),
        outputDir=Path("outputs") / "linear_regression",
        randomState=42,
    )
    result = RunResult(model="demo")
    assert context.outputDir.name == "linear_regression"
    assert result.artifacts == []
    assert result.metrics == {}


def testPrepareModelInputUsesModelFeatureNames():
    """模型带特征名时，应把数组包装成同名 DataFrame。"""

    class DemoModel:
        feature_names_in_ = np.asarray(["x1", "x2"])

    prepared = prepareModelInput(DemoModel(), np.asarray([[1.0, 2.0], [3.0, 4.0]]))
    assert isinstance(prepared, DataFrame)
    assert list(prepared.columns) == ["x1", "x2"]
