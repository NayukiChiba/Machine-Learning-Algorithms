"""
代表性流水线集成测试
"""

from __future__ import annotations

import importlib.util

import pytest

from mlAlgorithms.catalog.datasets import DATASET_REGISTRY
from mlAlgorithms.catalog.pipelines import PIPELINE_REGISTRY
from mlAlgorithms.workflows.baseRunner import ensureOptionalDependencies
from mlAlgorithms.workflows.executor import executePipeline


@pytest.mark.parametrize(
    ("pipelineId", "metricKey"),
    [
        ("classification.logistic_regression", "accuracy"),
        ("regression.linear_regression", "r2"),
        ("clustering.kmeans", "ari"),
        ("probabilistic.hmm", "accuracy"),
    ],
)
def testRepresentativePipelinesRunEndToEnd(pipelineId: str, metricKey: str):
    """代表性流水线应可端到端执行。"""
    spec = PIPELINE_REGISTRY.get(pipelineId)
    if spec.optionalDependencies:
        missing = [
            item
            for item in spec.optionalDependencies
            if importlib.util.find_spec(item) is None
        ]
        if missing:
            pytest.skip(f"缺少可选依赖: {missing}")
    ensureOptionalDependencies(spec)
    datasetSpec = DATASET_REGISTRY.get(spec.datasetId)
    result = executePipeline(spec, datasetSpec)
    assert result.model is not None
    if pipelineId == "probabilistic.hmm":
        assert metricKey in result.metrics
    else:
        assert metricKey in result.metrics or metricKey in result.metrics.get(
            "classification", {}
        )
    assert result.artifacts
