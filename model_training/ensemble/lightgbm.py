"""
model_training/ensemble/lightgbm.py
LightGBM 分类模型训练

使用方式:
    from model_training.ensemble.lightgbm import train_model

或直接运行:
    python -m model_training.ensemble.lightgbm
"""

from utils.decorate import print_func_info, timeit
from utils.contextmanage import timer

try:
    from lightgbm import LGBMClassifier
except Exception as e:  # noqa: BLE001
    LGBMClassifier = None
    _IMPORT_ERROR = e


@print_func_info
@timeit
def train_model(
    X_train,
    y_train,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = -1,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    random_state: int = 42,
):
    """
    训练 LightGBM 分类模型

    args:
        X_train: 训练特征
        y_train: 训练标签
        n_estimators: 弱学习器数量
        learning_rate: 学习率
        num_leaves: 叶子数
        max_depth: 最大深度（-1 不限制）
        subsample: 行采样比例
        colsample_bytree: 列采样比例
        random_state: 随机种子
    returns:
        model: 训练好的模型
    """
    if LGBMClassifier is None:
        raise ImportError(
            "未安装 lightgbm，请先安装后再运行该模块。"
        ) from _IMPORT_ERROR

    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )

    with timer(name="模型训练耗时"):
        model.fit(X_train, y_train)

    print("模型训练完成")
    print(f"n_estimators: {n_estimators}")
    print(f"learning_rate: {learning_rate}")
    print(f"num_leaves: {num_leaves}")
    print(f"max_depth: {max_depth}")
    print(f"subsample: {subsample}")
    print(f"colsample_bytree: {colsample_bytree}")

    return model


if __name__ == "__main__":
    from data_generation import lightgbm_data
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = lightgbm_data.drop(columns=["label"])
    y = lightgbm_data["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    train_model(X_train, y_train)
