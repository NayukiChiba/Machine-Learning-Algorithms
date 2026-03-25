"""
model_training/dimensionality/lda.py
LDA 降维模型训练

使用方式:
    from model_training.dimensionality.lda import train_model

或直接运行:
    python -m model_training.dimensionality.lda
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from utils.decorate import print_func_info, timeit
from utils.contextmanage import timer


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
    from data_generation import lda_data
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = lda_data.drop(columns=["label"])
    y = lda_data["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    train_model(X_train, y_train, n_components=2)
