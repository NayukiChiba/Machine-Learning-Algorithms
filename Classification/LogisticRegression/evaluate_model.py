"""
模型评估模块
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from utils.decorate import print_func_info
from train_model import train_model
from generate_data import generate_data
from preprocess_data import preprocess_data


@print_func_info
def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    评估模型性能

    args:
        model: 训练好的模型
        X_train, X_test: 特征
        y_train, y_test: 标签
    returns:
        y_train_pred, y_test_pred
    """
    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 训练集指标
    train_acc = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)

    # 测试集指标
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

    # 输出训练集性能
    print("训练集性能:")
    print(f"Accuracy:  {train_acc:.4f}")
    print(f"Precision: {train_precision:.4f}")
    print(f"Recall:    {train_recall:.4f}")
    print(f"F1-Score:  {train_f1:.4f}")

    # 输出测试集性能
    print("测试集性能:")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")

    # 泛化检查
    print("泛化检查:")
    acc_diff = train_acc - test_acc
    if acc_diff < 0.03:
        print(f"模型泛化良好 (Accuracy差异: {acc_diff:.4f})")
    elif acc_diff < 0.08:
        print(f"轻微过拟合 (Accuracy差异: {acc_diff:.4f})")
    else:
        print(f"存在过拟合风险 (Accuracy差异: {acc_diff:.4f})")

    # 分类报告
    print("\n测试集分类报告:")
    print(classification_report(y_test, y_test_pred, digits=4, zero_division=0))

    return y_train_pred, y_test_pred


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, X_train_orig, X_test_orig = (
        preprocess_data(generate_data())
    )
    model = train_model(X_train, y_train)
    evaluate_model(model, X_train, X_test, y_train, y_test)
