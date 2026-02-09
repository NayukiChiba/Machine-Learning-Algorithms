"""
训练 SVR 模型
"""

from pathlib import Path
import sys

# 加入项目根目录，便于导入公共模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.svm import SVR
from utils.decorate import print_func_info
from utils.contextmanage import timer
from generate_data import generate_data
from preprocess_data import preprocess_data


@print_func_info
def train_model(
    X_train,
    y_train,
    C: float = 10.0,
    epsilon: float = 0.1,
    kernel: str = "rbf",
    gamma="scale",
    degree: int = 3,
    coef0: float = 0.0,
):
    """
    训练支持向量回归模型

    args:
        X_train: 标准化后的训练特征
        y_train: 训练标签
        C: 惩罚系数，越大越拟合训练数据
        epsilon: epsilon-不敏感区间宽度
        kernel: 核函数类型（linear / rbf / poly / sigmoid）
        gamma: 核函数系数（rbf/poly/sigmoid 常用）
        degree: 多项式核的阶数（kernel=poly 才生效）
        coef0: 多项式/ sigmoid 核的常数项
    returns:
        model: 训练好的 SVR 模型
    """
    # 创建 SVR 模型
    model = SVR(
        C=C,
        epsilon=epsilon,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
    )

    # 训练并计时
    with timer(name="SVR 训练耗时"):
        model.fit(X_train, y_train)

    # 输出训练信息
    print("模型训练完成")
    print(f"kernel: {kernel}")
    print(f"C: {C}")
    print(f"epsilon: {epsilon}")
    print(f"gamma: {gamma}")
    if kernel == "poly":
        print(f"degree: {degree}")
        print(f"coef0: {coef0}")

    # 支持向量数量
    n_sv = model.support_.shape[0]
    print(f"支持向量数量: {n_sv}")

    return model


if __name__ == "__main__":
    # 简单自测：训练 RBF 核 SVR
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test = (
        preprocess_data(generate_data())
    )
    train_model(X_train_scaled, y_train, kernel="rbf")
