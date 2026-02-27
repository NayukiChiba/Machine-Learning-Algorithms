"""
训练 LDA 模型
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from utils.decorate import print_func_info, timeit
from utils.contextmanage import timer
from generate_data import generate_data
from preprocess_data import preprocess_data


@print_func_info
@timeit
def train_model(
    X_train,
    y_train,
    n_components: int = 2,
    solver: str = "svd",
):
    """
    训练 LDA 模型

    args:
        X_train: 标准化后的特征
        y_train: 标签
        n_components: 降维维度
        solver: 求解器（svd / lsqr / eigen）
    returns:
        model: 训练好的 LDA 模型
    """
    model = LinearDiscriminantAnalysis(
        n_components=n_components,
        solver=solver,
    )

    with timer(name="模型训练耗时"):
        model.fit(X_train, y_train)

    print("模型训练完成")
    print(f"n_components: {n_components}")
    if hasattr(model, "explained_variance_ratio_"):
        print(f"解释方差比: {model.explained_variance_ratio_.round(4)}")
        print(f"累计解释方差: {model.explained_variance_ratio_.sum():.4f}")

    return model


if __name__ == "__main__":
    X_scaled, scaler, X, y = preprocess_data(generate_data())
    train_model(X_scaled, y, n_components=2)
