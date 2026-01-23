'''
数据的预处理和划分操作
'''
# 导入根目录为搜索路径
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pandas import DataFrame
from utils.decorate import print_func_info
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from generate_data import generate_data


@print_func_info
def preprocess_data(data:DataFrame, test_size:float=0.2, random_state:int=42):
    '''
    数据的预处理和划分

    args:
        data(DataFrame): 数据
        test_size(float): 测试集比例
        random_state(int): 随机种子
    returns:
        X_train_scaled: 归一化之后的训练集
        X_test_scaled: 归一化之后的测试集
        y_train: 原训练集的目标变量
        y_test: 测试集的目标变量
        scaler: 归一化方法
        X_train: 训练集
        X_test: 测试集
    '''
    # 分离特征和目标变量
    features = data.drop("价格", axis=1)
    price = data["价格"]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, price, test_size=test_size, random_state=random_state)

    print(f"训练集大小: {len(X_train)}个样本")
    print(f"测试集大小: {len(X_test)}个样本")
    print(f"训练集占比: {(1-test_size)*100:.0f}%")
    print(f"测试集占比: {test_size*100:.0f}%")

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    print("特征标准化完成")
    print(f"标准化前均值: {X_train.mean().values.round(2)}")
    print(f"标准化后均值: {X_train_scaled.mean(axis=0).round(2)}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test


if __name__ == "__main__":
    preprocess_data(generate_data())