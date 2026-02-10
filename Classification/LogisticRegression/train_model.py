"""
训练逻辑回归模型
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.linear_model import LogisticRegression
from utils.decorate import print_func_info, timeit
from utils.contextmanage import timer
from generate_data import generate_data
from preprocess_data import preprocess_data


@print_func_info
@timeit
def train_model(
    X_train,
    y_train,
    penalty: str = "l2",
    C: float = 1.0,
    solver: str = "lbfgs",
    max_iter: int = 1000,
    class_weight=None,
    random_state: int = 42,
):
    """
    训练逻辑回归分类模型

    args:
        X_train: 训练特征（标准化后）
        y_train: 训练标签
        penalty: 正则化类型（l1, l2, elasticnet, none）
        C: 正则化强度的倒数（越大正则越弱）
        solver: 优化器
        max_iter: 最大迭代次数
        class_weight: 类别权重
        random_state: 随机种子
    returns:
        model: 训练好的模型
    """
    model = LogisticRegression(
        penalty=penalty,
        C=C,
        solver=solver,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
    )

    with timer(name="模型训练耗时"):
        model.fit(X_train, y_train)

    print("模型训练完成")
    print(f"penalty: {penalty}")
    print(f"C: {C}")
    print(f"solver: {solver}")
    print(f"max_iter: {max_iter}")
    print(f"classes: {model.classes_.tolist()}")
    print(f"截距: {model.intercept_.round(4)}")
    print(f"系数: {model.coef_.round(4)}")

    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, X_train_orig, X_test_orig = (
        preprocess_data(generate_data())
    )
    train_model(X_train, y_train)
