"""
训练决策树分类模型
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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
    max_depth: int = 6,
    min_samples_split: int = 4,
    min_samples_leaf: int = 2,
    criterion: str = "gini",
    random_state: int = 42,
):
    """
    训练决策树分类模型

    args:
        X_train: 训练特征（可为标准化后）
        y_train: 训练标签
        max_depth: 最大深度
        min_samples_split: 内部节点再划分所需最小样本数
        min_samples_leaf: 叶子节点最小样本数
        criterion: 划分标准（gini / entropy / log_loss）
        random_state: 随机种子
    returns:
        model: 训练好的模型
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=random_state,
    )

    with timer(name="模型训练耗时"):
        model.fit(X_train, y_train)

    print("模型训练完成")
    print(f"最大深度: {model.get_depth()}")
    print(f"叶子节点数: {model.get_n_leaves()}")
    print(f"criterion: {criterion}")

    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, X_train_orig, X_test_orig = (
        preprocess_data(generate_data())
    )
    train_model(X_train, y_train)
