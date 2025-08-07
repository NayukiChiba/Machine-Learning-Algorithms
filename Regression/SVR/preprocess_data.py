"""
数据预处理模块：划分 + 标准化
"""

from pathlib import Path
import sys

# 加入项目根目录，便于导入公共模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.decorate import print_func_info


@print_func_info
def preprocess_data(data: DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    训练/测试划分 + 标准化

    args:
        data(DataFrame): 原始数据
        test_size(float): 测试集比例
        random_state(int): 随机种子
    returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test
    """
    # 拆分特征与目标
    features = data.drop("price", axis=1)
    target = data["price"]

    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )

    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"训练集占比: {(1 - test_size) * 100:.0f}%")
    print(f"测试集占比: {test_size * 100:.0f}%")

    # SVR 对特征尺度敏感，必须标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("标准化完成")
    print(f"标准化前均值: {X_train.mean().values.round(3)}")
    print(f"标准化后均值: {X_train_scaled.mean(axis=0).round(3)}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test


if __name__ == "__main__":
    from generate_data import generate_data

    df = generate_data()
    preprocess_data(df)
