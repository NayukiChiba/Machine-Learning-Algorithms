"""
分析器测试
"""

from __future__ import annotations

from pandas import DataFrame

from mlAlgorithms.analysis.sequenceAnalyzer import buildSequenceExplorationReport
from mlAlgorithms.analysis.tabularAnalyzer import buildClassificationExplorationReport
from mlAlgorithms.core.datasetSpec import DatasetSpec
from mlAlgorithms.core.taskTypes import DataKind, TaskType


def testTabularAnalyzerReturnsStructuredReport():
    """表格分析器应返回结构化报告。"""
    data = DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [2.0, 3.0, 4.0, 5.0],
            "label": [0, 0, 1, 1],
        }
    )
    spec = DatasetSpec(
        id="classification.demo",
        taskType=TaskType.CLASSIFICATION,
        dataKind=DataKind.TABULAR,
        loader=lambda: data.copy(),
        targetColumn="label",
        featureColumns=["x1", "x2"],
        description="demo",
    )
    report = buildClassificationExplorationReport(data, spec)
    assert report.overview.rowCount == 4
    assert "x1" in report.numericSummary
    assert report.targetSummary["nunique"] == 2


def testSequenceAnalyzerReturnsStructuredReport():
    """序列分析器应返回结构化报告。"""
    data = DataFrame(
        {
            "time": [0, 1, 2, 3],
            "obs": [0, 1, 0, 2],
            "state_true": [0, 1, 1, 2],
        }
    )
    spec = DatasetSpec(
        id="probabilistic.hmm",
        taskType=TaskType.PROBABILISTIC,
        dataKind=DataKind.SEQUENCE,
        loader=lambda: data.copy(),
        targetColumn="state_true",
        featureColumns=["obs"],
        description="demo",
    )
    report = buildSequenceExplorationReport(data, spec)
    assert report.overview.rowCount == 4
    assert report.observationSummary["0"] == 2
    assert "1->2" in report.transitionSummary
