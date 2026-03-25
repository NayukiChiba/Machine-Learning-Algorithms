"""
model_training/dimensionality/pca.py
PCA 降维模型训练

使用方式:
    from model_training.dimensionality.pca import train_model

或直接运行:
    python -m model_training.dimensionality.pca
"""

from sklearn.decomposition import PCA

from utils.decorate import print_func_info, timeit
from utils.contextmanage import timer


@print_func_info
@timeit
def train_model(
    X_train,
    n_components: int = 2,
    svd_solver: str = "auto",
    random_state: int = 42,
):
    """
    训练 PCA 模型

    args:
        X_train: 标准化后的特征
        n_components: 主成分数量
        svd_solver: 求解器
        random_state: 随机种子
    returns:
        model: 训练好的 PCA 模型
    """
    model = PCA(
        n_components=n_components,
        svd_solver=svd_solver,
        random_state=random_state,
    )

    with timer(name="模型训练耗时"):
        model.fit(X_train)

    print("模型训练完成")
    print(f"n_components: {n_components}")
    print(f"解释方差比: {model.explained_variance_ratio_.round(4)}")
    print(f"累计解释方差: {model.explained_variance_ratio_.sum():.4f}")

    return model


if __name__ == "__main__":
    from data_generation import pca_data
    from sklearn.preprocessing import StandardScaler

    X = pca_data.drop(columns=["label"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    train_model(X_scaled)
