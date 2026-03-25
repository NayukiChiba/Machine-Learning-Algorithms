"""
model_training/regression/svr.py
SVR 回归模型训练

使用方式:
    from model_training.regression.svr import train_model

或直接运行:
    python -m model_training.regression.svr
"""

from sklearn.svm import SVR

from utils.decorate import print_func_info
from utils.contextmanage import timer


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
        gamma: 核函数系数
        degree: 多项式核的阶数（kernel=poly 才生效）
        coef0: 多项式/sigmoid 核的常数项
    returns:
        model: 训练好的 SVR 模型
    """
    model = SVR(
        C=C,
        epsilon=epsilon,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
    )

    with timer(name="SVR 训练耗时"):
        model.fit(X_train, y_train)

    print("模型训练完成")
    print(f"kernel: {kernel}")
    print(f"C: {C}")
    print(f"epsilon: {epsilon}")
    print(f"gamma: {gamma}")
    if kernel == "poly":
        print(f"degree: {degree}")
        print(f"coef0: {coef0}")

    n_sv = model.support_.shape[0]
    print(f"支持向量数量: {n_sv}")

    return model


if __name__ == "__main__":
    from data_generation import svr_data
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = svr_data.drop(columns=["price"])
    y = svr_data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    train_model(X_train, y_train)
