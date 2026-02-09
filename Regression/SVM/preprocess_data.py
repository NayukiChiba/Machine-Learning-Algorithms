"""
数据预处理
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于直接导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.decorate import print_func_info
from generate_data import generate_data


@print_func_info
def preprocess_data(data: DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    划分训练集与测试集，并进行标准化

    args:
        data(DataFrame): 数据
        test_size(float): 测试集比例
        random_state(int): 随机种子
    returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test
    """
    # 分离特征和标签
    features = data.drop("label", axis=1)
    target = data["label"]

    # 按照类别比例进行分层采样，保证训练/测试集类别分布一致
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    # 打印划分信息
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"训练集占比: {(1 - test_size) * 100:.0f}%")
    print(f"测试集占比: {test_size * 100:.0f}%")

    # 标准化：只在训练集上拟合，测试集只做变换
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 打印标准化信息
    print("特征标准化完成")
    print(f"标准化前均值: {X_train.mean().values.round(3)}")
    print(f"标准化后均值: {X_train_scaled.mean(axis=0).round(3)}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test


if __name__ == "__main__":
    # 模块自测：生成数据并完成预处理
    preprocess_data(generate_data())
