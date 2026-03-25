"""
model_training/probabilistic/em.py
EM (Gaussian Mixture Model) 模型训练

使用方式:
    from model_training.probabilistic.em import train_model

或直接运行:
    python -m model_training.probabilistic.em
"""

from sklearn.mixture import GaussianMixture

from utils.decorate import print_func_info, timeit
from utils.contextmanage import timer


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
    from data_generation import em_data
    from sklearn.preprocessing import StandardScaler

    X = em_data.drop(columns=["true_label"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    train_model(X_scaled)
