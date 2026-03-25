"""
model_training/classification/naive_bayes.py
朴素贝叶斯分类模型训练

使用方式:
    from model_training.classification.naive_bayes import train_model

或直接运行:
    python -m model_training.classification.naive_bayes
"""

from sklearn.naive_bayes import GaussianNB

from utils.decorate import print_func_info, timeit
from utils.contextmanage import timer


@print_func_info
@timeit
def train_model(
    X_train,
    y_train,
    var_smoothing: float = 1e-9,
):
    """
    训练高斯朴素贝叶斯分类模型

    args:
        X_train: 训练特征
        y_train: 训练标签
        var_smoothing: 方差平滑项
    returns:
        model: 训练好的模型
    """
    model = GaussianNB(var_smoothing=var_smoothing)

    with timer(name="模型训练耗时"):
        model.fit(X_train, y_train)

    print("模型训练完成")
    print(f"var_smoothing: {var_smoothing}")
    print(f"类别: {model.classes_.tolist()}")
    print(f"类别先验: {model.class_prior_.round(4)}")

    return model


if __name__ == "__main__":
    from data_generation import naive_bayes_data
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = naive_bayes_data.drop(columns=["label"])
    y = naive_bayes_data["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    train_model(X_train, y_train)
