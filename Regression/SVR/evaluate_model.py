"""
评估 SVR 模型
"""

from pathlib import Path
import sys

# 加入项目根目录，便于导入公共模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.decorate import print_func_info
from generate_data import generate_data
from preprocess_data import preprocess_data
from train_model import train_model


@print_func_info
def evaluate_model(model_name: str, model, X_train, X_test, y_train, y_test):
    """
    输出训练集与测试集的回归指标

    args:
        model_name: 模型名称
        model: 已训练模型
        X_train, X_test: 训练/测试特征
        y_train, y_test: 训练/测试标签
    returns:
        metrics, y_train_pred, y_test_pred
    """
    # 预测结果
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 计算指标
    metrics = {
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
    }

    # 打印结果
    print(f"[{model_name}]")
    print(
        f"训练集 -> R2: {metrics['train_r2']:.4f}, "
        f"RMSE: {metrics['train_rmse']:.3f}, MAE: {metrics['train_mae']:.3f}"
    )
    print(
        f"测试集 -> R2: {metrics['test_r2']:.4f}, "
        f"RMSE: {metrics['test_rmse']:.3f}, MAE: {metrics['test_mae']:.3f}"
    )

    # 简单泛化判断
    gap = metrics["train_r2"] - metrics["test_r2"]
    if gap < 0.05:
        print("泛化情况：良好")
    elif gap < 0.10:
        print("泛化情况：轻微过拟合")
    else:
        print("泛化情况：可能过拟合")

    return metrics, y_train_pred, y_test_pred


if __name__ == "__main__":
    df = generate_data()
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test = (
        preprocess_data(df)
    )
    model = train_model(X_train_scaled, y_train, kernel="rbf")
    evaluate_model("SVR-RBF", model, X_train_scaled, X_test_scaled, y_train, y_test)
