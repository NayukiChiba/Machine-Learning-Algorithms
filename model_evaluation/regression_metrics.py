"""
model_evaluation/regression_metrics.py
回归模型评估指标

包含: MSE、RMSE、MAE、R²、调整 R²

使用方式:
    from model_evaluation.regression_metrics import evaluate_regression
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def evaluate_regression(
    y_true,
    y_pred,
    n_features: int | None = None,
    print_report: bool = True,
) -> dict:
    """
    计算回归模型的全套评估指标

    Args:
        y_true: 真实值
        y_pred: 预测值
        n_features: 特征数（用于计算调整 R²，可选）
        print_report: 是否打印报告

    Returns:
        dict: 包含所有指标的字典
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

    # 调整 R²
    if n_features is not None:
        n = len(y_true)
        if n > n_features + 1:
            metrics["adjusted_r2"] = 1 - (1 - metrics["r2"]) * (n - 1) / (
                n - n_features - 1
            )

    if print_report:
        print("=" * 60)
        print("回归评估报告")
        print("=" * 60)
        print(f"  MSE:         {metrics['mse']:.6f}")
        print(f"  RMSE:        {metrics['rmse']:.6f}")
        print(f"  MAE:         {metrics['mae']:.6f}")
        print(f"  R²:          {metrics['r2']:.6f}")
        if "adjusted_r2" in metrics:
            print(f"  调整 R²:     {metrics['adjusted_r2']:.6f}")

    return metrics


if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_regression(y_test, y_pred, n_features=5)
