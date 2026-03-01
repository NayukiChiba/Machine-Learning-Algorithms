"""
训练 Bagging 分类模型
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from utils.decorate import print_func_info, timeit
from utils.contextmanage import timer
from generate_data import generate_data
from preprocess_data import preprocess_data


@print_func_info
@timeit
def train_model(
    X_train,
    y_train,
    n_estimators: int = 80,
    max_samples: float = 0.8,
    max_features: float = 1.0,
    bootstrap: bool = True,
    oob_score: bool = True,
    random_state: int = 42,
):
    """
    训练 Bagging 分类模型

    args:
        X_train: 训练特征
        y_train: 训练标签
        n_estimators: 基学习器数量
        max_samples: 采样比例
        max_features: 特征采样比例
        bootstrap: 是否有放回采样
        oob_score: 是否启用 OOB 估计
        random_state: 随机种子
    returns:
        model: 训练好的模型
    """
    base = DecisionTreeClassifier(
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
    )

    # 兼容不同 sklearn 版本
    try:
        model = BaggingClassifier(
            estimator=base,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            oob_score=oob_score,
            random_state=random_state,
            n_jobs=-1,
        )
    except TypeError:
        model = BaggingClassifier(
            base_estimator=base,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            oob_score=oob_score,
            random_state=random_state,
            n_jobs=-1,
        )

    with timer(name="模型训练耗时"):
        model.fit(X_train, y_train)

    print("模型训练完成")
    print(f"n_estimators: {n_estimators}")
    print(f"max_samples: {max_samples}")
    print(f"max_features: {max_features}")
    print(f"bootstrap: {bootstrap}")
    if oob_score:
        print(f"OOB 得分: {model.oob_score_:.4f}")

    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, X_train_orig, X_test_orig = (
        preprocess_data(generate_data())
    )
    train_model(X_train, y_train)
