"""
训练朴素贝叶斯模型
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.naive_bayes import GaussianNB
from utils.decorate import print_func_info, timeit
from utils.contextmanage import timer
from generate_data import generate_data
from preprocess_data import preprocess_data


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
    X_train, X_test, y_train, y_test, scaler, X_train_orig, X_test_orig = (
        preprocess_data(generate_data())
    )
    train_model(X_train, y_train)
