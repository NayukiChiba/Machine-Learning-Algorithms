'''
训练决策树模型
'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.tree import DecisionTreeRegressor
from utils.decorate import print_func_info, timeit
from utils.contextmanage import timer
from generate_data import generate_data
from preprocess_data import preprocess_data


@print_func_info
@timeit
def train_model(X_train, y_train,
                max_depth:int=6,
                min_samples_split:int=6,
                min_samples_leaf:int=3,
                random_state:int=42):
    '''
    训练决策树回归模型

    args:
        X_train: 训练特征
        y_train: 训练标签
        max_depth(int): 最大深度
        min_samples_split(int): 继续划分的最小样本数
        min_samples_leaf(int): 叶子节点最小样本数
        random_state(int): 随机种子
    returns:
        model: 训练好的模型
    '''
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    with timer(name="模型训练耗时"):
        model.fit(X_train, y_train)
    
    print("模型训练完成")
    print(f"树深度:{model.get_depth()}")
    print(f"叶子节点数: {model.get_n_leaves()}")

    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, X, y = preprocess_data(generate_data())
    train_model(X_train=X_train, y_train=y_train)