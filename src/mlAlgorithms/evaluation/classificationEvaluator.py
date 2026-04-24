"""
分类评估器
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluateClassification(
    yTrue,
    yPred,
    yScores=None,
    average: str = "weighted",
    printReport: bool = True,
) -> dict:
    """评估分类任务。"""
    yTrue = np.asarray(yTrue)
    yPred = np.asarray(yPred)
    metrics = {
        "accuracy": accuracy_score(yTrue, yPred),
        "precision": precision_score(yTrue, yPred, average=average, zero_division=0),
        "recall": recall_score(yTrue, yPred, average=average, zero_division=0),
        "f1": f1_score(yTrue, yPred, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(yTrue, yPred),
        "classification_report": classification_report(
            yTrue, yPred, digits=4, zero_division=0
        ),
    }
    if yScores is not None:
        try:
            yScores = np.asarray(yScores)
            classes = np.unique(yTrue)
            if len(classes) == 2:
                score = yScores[:, 1] if yScores.ndim == 2 else yScores
                metrics["auc"] = roc_auc_score(yTrue, score)
            else:
                metrics["auc"] = roc_auc_score(
                    yTrue, yScores, multi_class="ovr", average=average
                )
        except Exception:
            metrics["auc"] = None
    if printReport:
        print("=" * 60)
        print("分类评估摘要")
        print("=" * 60)
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        if metrics.get("auc") is not None:
            print(f"AUC: {metrics['auc']:.4f}")
    return metrics
