"""
model_training/classification/knn.py
KNN 分类模型训练

使用方式:
    from model_training.classification.knn import train_model

或直接运行:
    python -m model_training.classification.knn
"""

from sklearn.neighbors import KNeighborsClassifier

from utils.decorate import print_func_info, timeit
from utils.contextmanage import timer


@print_func_info
@timeit
def train_model(
    X_train,
    y_train,
    n_neighbors: int = 5,
    weights: str = "uniform",
    metric: str = "minkowski",
):
    """
    训练 KNN 分类模型

    args:
        X_train: 训练特征（标准化后）
        y_train: 训练标签
        n_neighbors: 近邻数量 K
        weights: 投票权重（uniform / distance）
        metric: 距离度量
    returns:
        model: 训练好的模型
    """
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
    )

    with timer(name="模型训练耗时"):
        model.fit(X_train, y_train)

    print("模型训练完成")
    print(f"K: {n_neighbors}")
    print(f"weights: {weights}")
    print(f"metric: {metric}")

    return model


if __name__ == "__main__":
    from data_generation import knn_data
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = knn_data.drop(columns=["label"])
    y = knn_data["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    train_model(X_train, y_train)
