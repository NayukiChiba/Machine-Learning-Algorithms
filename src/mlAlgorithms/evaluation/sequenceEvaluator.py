"""
序列评估器
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def evaluateSequenceLabels(
    yTrue, yPred, logLikelihood: float | None = None, printReport: bool = True
) -> dict:
    """评估序列标签预测。"""
    yTrue = np.asarray(yTrue)
    yPred = np.asarray(yPred)
    metrics = {
        "accuracy": float(np.mean(yTrue == yPred)),
        "confusion_matrix": confusion_matrix(yTrue, yPred),
    }
    if logLikelihood is not None:
        metrics["log_likelihood"] = float(logLikelihood)
    if printReport:
        print("=" * 60)
        print("序列评估摘要")
        print("=" * 60)
        print(f"准确率: {metrics['accuracy']:.4f}")
    return metrics
