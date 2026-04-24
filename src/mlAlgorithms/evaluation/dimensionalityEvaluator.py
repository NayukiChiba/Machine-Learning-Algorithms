"""
降维评估器
"""

from __future__ import annotations

import numpy as np


def evaluateDimensionality(
    model, XOriginal=None, XTransformed=None, printReport: bool = True
) -> dict:
    """评估降维任务。"""
    metrics: dict[str, object] = {}
    if hasattr(model, "explained_variance_ratio_"):
        ratio = model.explained_variance_ratio_
        metrics["explained_variance_ratio"] = ratio
        metrics["cumulative_variance_ratio"] = np.cumsum(ratio)
        metrics["total_explained_variance"] = np.sum(ratio)
    if (
        XOriginal is not None
        and XTransformed is not None
        and hasattr(model, "inverse_transform")
    ):
        reconstructed = model.inverse_transform(XTransformed)
        metrics["reconstruction_error"] = float(
            np.mean((np.asarray(XOriginal) - reconstructed) ** 2)
        )
    if printReport:
        print("=" * 60)
        print("降维评估摘要")
        print("=" * 60)
        if "total_explained_variance" in metrics:
            print(f"总解释方差比: {metrics['total_explained_variance']:.4f}")
    return metrics
