"""
model_evaluation/classification_metrics.py
分类模型评估指标

包含: 精度 (Accuracy)、精确率/召回率/F1、AUC、分类报告

使用方式:
    from model_evaluation.classification_metrics import evaluate_classification
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


def evaluate_classification(
    y_true,
    y_pred,
    y_scores=None,
    average: str = "weighted",
    target_names: list[str] | None = None,
    print_report: bool = True,
) -> dict:
    """
    计算分类模型的全套评估指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_scores: 预测概率（用于 AUC 计算，可选）
        average: 多分类聚合方式 ('micro', 'macro', 'weighted')
        target_names: 类别名称
        print_report: 是否打印报告

    Returns:
        dict: 包含所有指标的字典
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    # AUC
    if y_scores is not None:
        y_scores = np.asarray(y_scores)
        classes = np.unique(y_true)
        if len(classes) == 2:
            score = y_scores[:, 1] if y_scores.ndim == 2 else y_scores
            metrics["auc"] = roc_auc_score(y_true, score)
        else:
            try:
                metrics["auc"] = roc_auc_score(
                    y_true, y_scores, multi_class="ovr", average=average
                )
            except ValueError:
                metrics["auc"] = None

    if print_report:
        print("=" * 60)
        print("分类评估报告")
        print("=" * 60)
        print(f"  准确率 (Accuracy):  {metrics['accuracy']:.4f}")
        print(f"  精确率 (Precision): {metrics['precision']:.4f}")
        print(f"  召回率 (Recall):    {metrics['recall']:.4f}")
        print(f"  F1 分数:            {metrics['f1']:.4f}")
        if "auc" in metrics and metrics["auc"] is not None:
            print(f"  AUC:                {metrics['auc']:.4f}")
        print()
        print(
            classification_report(
                y_true, y_pred, target_names=target_names, zero_division=0
            )
        )

    return metrics


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)
    evaluate_classification(y_test, y_pred, y_scores=y_scores)
