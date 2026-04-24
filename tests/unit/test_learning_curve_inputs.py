"""
学习曲线输入规范测试
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pandas import DataFrame, Series

from mlAlgorithms.visualization.result import classificationPlots


def testPlotLearningCurveConvertsInputsToNumpy(monkeypatch):
    """学习曲线应统一接收带稳定列名的 DataFrame。"""
    captured: dict[str, object] = {}

    def fakeLearningCurve(estimator, X, y, **kwargs):
        captured["x_type"] = type(X)
        captured["y_type"] = type(y)
        captured["columns"] = list(X.columns)
        return (
            np.asarray([1, 2, 3]),
            np.asarray([[0.9], [0.92], [0.95]]),
            np.asarray([[0.8], [0.85], [0.88]]),
        )

    monkeypatch.setattr(classificationPlots, "learning_curve", fakeLearningCurve)

    class DemoEstimator:
        """测试用估计器。"""

    outputDir = Path("outputs") / "visualization"
    outputPath = outputDir / "test_learning_curve.png"
    if outputPath.exists():
        outputPath.unlink()

    classificationPlots.plotLearningCurve(
        DemoEstimator(),
        DataFrame({"x1": [1.0, 2.0, 3.0, 4.0], "x2": [2.0, 3.0, 4.0, 5.0]}),
        Series([0, 1, 0, 1]),
        outputDir,
        "测试学习曲线",
        filename="test_learning_curve.png",
    )

    assert captured["x_type"] is DataFrame
    assert captured["y_type"] is Series
    assert captured["columns"] == ["Column_0", "Column_1"]

    if outputPath.exists():
        outputPath.unlink()
