"""
数据预处理模块
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from utils.decorate import print_func_info
from generate_data import generate_data


@print_func_info
def preprocess_data(data: DataFrame):
    """
    标准化特征（聚类对尺度敏感）

    args:
        data(DataFrame): 输入数据
    returns:
        X_scaled, scaler, X
    """
    X = data.copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("特征标准化完成")
    print(f"标准化前均值: {X.mean().values.round(3)}")
    print(f"标准化后均值: {X_scaled.mean(axis=0).round(3)}")

    return X_scaled, scaler, X


if __name__ == "__main__":
    preprocess_data(generate_data())
