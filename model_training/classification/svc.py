"""
model_training/classification/svc.py
SVC 分类模型训练

使用方式:
    from model_training.classification.svc import train_model

或直接运行:
    python -m model_training.classification.svc
"""

from sklearn.svm import SVC

from utils.decorate import print_func_info, timeit
from utils.contextmanage import timer


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

    with timer(name="模型训练耗时"):
        model.fit(X_train, y_train)

    print("模型训练完成")
    print(f"支持向量总数: {model.n_support_.sum()}")
    print(f"各类别支持向量数: {model.n_support_.tolist()}")

    return model


if __name__ == "__main__":
    from data_generation import svc_data
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = svc_data.drop(columns=["label"])
    y = svc_data["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    train_model(X_train, y_train)
