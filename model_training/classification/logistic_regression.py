"""
model_training/classification/logistic_regression.py
逻辑回归分类模型训练

使用方式:
    from model_training.classification.logistic_regression import train_model

或直接运行:
    python -m model_training.classification.logistic_regression
"""

from sklearn.linear_model import LogisticRegression

from utils.decorate import print_func_info, timeit
from utils.contextmanage import timer


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
    from data_generation import logistic_regression_data
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = logistic_regression_data.drop(columns=["label"])
    y = logistic_regression_data["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    train_model(X_train, y_train)
