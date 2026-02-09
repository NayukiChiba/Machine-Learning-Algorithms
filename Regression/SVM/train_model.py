"""
训练SVM模型
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.svm import SVC
from utils.decorate import print_func_info, timeit
from utils.contextmanage import timer
from generate_data import generate_data
from preprocess_data import preprocess_data


@print_func_info
@timeit
def train_model(
    X_train,
    y_train,
    C: float = 1.0,
    kernel: str = "rbf",
    gamma="scale",
    random_state: int = 42,
):
    """
    训练支持向量机分类模型

    args:
        X_train: 训练特征（标准化后）
        y_train: 训练标签
        C(float): 惩罚系数
        kernel(str): 核函数类型
        gamma: 核函数系数
        random_state(int): 随机种子
    returns:
        model: 训练好的模型
    """
    model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=random_state)
    """
    kernel :
        linear:  K(X1, X2) = <X1, X2>
        rbf:     K(X1, X2) = exp(-\gamma ||X1 - X2||^2)
        poly:    K(X1, X2) = (\gamma <X1, X2> + coef0)^{dim}
        sigmoid: K(X1, X2) = tanh(\gamma <X1, X2> + coef0)
    gamma: 核参数
        scale: \gamma = frac{1}{n_features * Var(X)}
        auto: \gamma = frac{1}{n_features}
    """
    with timer(name="模型训练耗时"):
        model.fit(X_train, y_train)

    print("模型训练完成")
    print(f"支持向量总数: {model.n_support_.sum()}")
    print(f"各类别支持向量数: {model.n_support_.tolist()}")

    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, X_train_orig, X_test_orig = (
        preprocess_data(generate_data())
    )
    train_model(X_train, y_train)
