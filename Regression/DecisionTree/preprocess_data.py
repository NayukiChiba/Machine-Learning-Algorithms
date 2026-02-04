"""
数据预处理
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from utils.decorate import print_func_info
from generate_data import generate_data


@print_func_info
def preprocess_data(data: DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    划分训练集与测试集（决策树不需要标准化）

    args:
        data(DataFrame): 数据
        test_size(float): 测试集比例
        random_state(int): 随机种子
    returns:
        X_train, X_test, y_train, y_test, X, y
    """
    features = data.drop("price", axis=1)
    target = data["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )

    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"训练集占比: {(1 - test_size) * 100:.0f}%")
    print(f"测试集占比: {test_size * 100:.0f}%")

    return X_train, X_test, y_train, y_test, features, target


if __name__ == "__main__":
    preprocess_data(generate_data())
