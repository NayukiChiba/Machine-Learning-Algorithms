"""
回归评估器
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluateRegression(
    yTrue, yPred, nFeatures: int | None = None, printReport: bool = True
) -> dict:
    """评估回归任务。"""
    yTrue = np.asarray(yTrue).ravel()
    yPred = np.asarray(yPred).ravel()
    mse = mean_squared_error(yTrue, yPred)
    metrics = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(yTrue, yPred),
        "r2": r2_score(yTrue, yPred),
    }
    if nFeatures is not None and len(yTrue) > nFeatures + 1:
        metrics["adjusted_r2"] = 1 - (1 - metrics["r2"]) * (len(yTrue) - 1) / (
            len(yTrue) - nFeatures - 1
        )
    if printReport:
        print("=" * 60)
        print("回归评估摘要")
        print("=" * 60)
        print(f"MSE: {metrics['mse']:.6f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"R2: {metrics['r2']:.6f}")
    return metrics
