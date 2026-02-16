"""
训练 EM (Gaussian Mixture Model) 模型
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.mixture import GaussianMixture
from utils.decorate import print_func_info, timeit
from utils.contextmanage import timer
from generate_data import generate_data
from preprocess_data import preprocess_data


@print_func_info
@timeit
def train_model(
    X_train,
    n_components: int = 3,
    covariance_type: str = "full",
    max_iter: int = 200,
    random_state: int = 42,
):
    """
    训练 GMM 模型（EM 算法）

    args:
        X_train: 标准化后的特征
        n_components: 高斯分量数
        covariance_type: 协方差类型
        max_iter: 最大迭代次数
        random_state: 随机种子
    returns:
        model: 训练好的模型
    """
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        random_state=random_state,
    )

    with timer(name="模型训练耗时"):
        model.fit(X_train)

    print("模型训练完成")
    print(f"n_components: {n_components}")
    print(f"covariance_type: {covariance_type}")
    print(f"log-likelihood: {model.lower_bound_:.4f}")

    return model


if __name__ == "__main__":
    X_scaled, scaler, X = preprocess_data(generate_data())
    train_model(X_scaled)
