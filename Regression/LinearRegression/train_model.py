'''
训练线性回归模型
'''
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


from pandas import DataFrame, Series
from typing import Union
from sklearn.linear_model import LinearRegression
from utils.decorate import print_func_info
from generate_data import generate_data
from preprocess_data import preprocess_data



@print_func_info
def train_model(X_train:DataFrame, y_train:Union[DataFrame, Series]):
    '''
    训练线性回归模型

    args:
        X_train(DataFrame): 训练集是DataFrame类型
        y_train(Union[DataFrame, Series]): 标签只有一列, 可以是DataFrame, 也可以是Series
    '''
    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"模型训练完成")
    print(f"模型参数: ")
    print(f"截距(intercept): {model.intercept_:.2f}")
    print(f"斜率(coefficients):" )
    features_names = list(X_train.columns)
    for name, coef in zip(features_names, model.coef_):
        print(f"{name}: {coef:.2f}")
    return model


if __name__ == "__main__":
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test = preprocess_data(generate_data())
    train_model(X_train=X_train, y_train=y_train)