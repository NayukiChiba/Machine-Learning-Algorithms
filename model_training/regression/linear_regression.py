"""
model_training/regression/linear_regression.py
线性回归模型训练

使用方式:
    from model_training.regression.linear_regression import train_model

或直接运行:
    python -m model_training.regression.linear_regression
"""

from sklearn.linear_model import LinearRegression

from utils.decorate import print_func_info


@print_func_info
def train_model(X_train, y_train, feature_names=None):
    """
    训练线性回归模型

    args:
        X_train: 训练集
        y_train: 标签
        feature_names: 特征名称列表（可选）
    returns:
        model: 训练好的模型
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("模型训练完成")
    print(f"截距(intercept): {model.intercept_:.2f}")
    print("斜率(coefficients):")

    # 处理列名
    if feature_names is not None:
        names = feature_names
    elif hasattr(X_train, "columns"):
        names = list(X_train.columns)
    else:
        names = [f"Feature_{i}" for i in range(X_train.shape[1])]

    for name, coef in zip(names, model.coef_):
        print(f"  {name}: {coef:.2f}")

    return model


if __name__ == "__main__":
    from data_generation import linear_regression_data
    from sklearn.model_selection import train_test_split

    X = linear_regression_data.drop(columns=["price"])
    y = linear_regression_data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    train_model(X_train, y_train)
