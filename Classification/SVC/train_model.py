"""
训练 SVC 模型
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于直接导入公共工具
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
    # 创建 SVC 分类器
    model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=random_state)

    # kernel 说明：
    # linear:  K(x1, x2) = <x1, x2>
    # rbf:     K(x1, x2) = exp(-gamma * ||x1 - x2||^2)
    # poly:    K(x1, x2) = (gamma * <x1, x2> + coef0)^degree
    # sigmoid: K(x1, x2) = tanh(gamma * <x1, x2> + coef0)
    # gamma 说明：
    # scale: 1 / (n_features * Var(X))
    # auto:  1 / n_features

    # 训练模型并统计耗时
    with timer(name="模型训练耗时"):
        model.fit(X_train, y_train)

    # 打印训练信息
    print("模型训练完成")
    print(f"支持向量总数: {model.n_support_.sum()}")
    print(f"各类别支持向量数: {model.n_support_.tolist()}")

    return model


if __name__ == "__main__":
    # 模块自测：训练模型
    X_train, X_test, y_train, y_test, scaler, X_train_orig, X_test_orig = (
        preprocess_data(generate_data())
    )
    train_model(X_train, y_train)
