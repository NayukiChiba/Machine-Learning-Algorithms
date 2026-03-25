"""
model_training/probabilistic/hmm.py
HMM 模型训练

使用方式:
    from model_training.probabilistic.hmm import train_model

或直接运行:
    python -m model_training.probabilistic.hmm
"""

from utils.decorate import print_func_info, timeit
from utils.contextmanage import timer

try:
    from hmmlearn.hmm import CategoricalHMM
except Exception:  # noqa: BLE001
    CategoricalHMM = None

try:
    from hmmlearn.hmm import MultinomialHMM
except Exception:  # noqa: BLE001
    MultinomialHMM = None


@print_func_info
@timeit
def train_model(
    X_obs,
    lengths,
    n_components: int = 3,
    n_iter: int = 100,
    tol: float = 1e-3,
    random_state: int = 42,
):
    """
    训练 HMM 模型

    args:
        X_obs: 观测序列 (n_samples, 1)
        lengths: 序列长度列表
        n_components: 隐状态数
        n_iter: 最大迭代次数
        tol: 收敛阈值
        random_state: 随机种子
    returns:
        model: 训练好的 HMM 模型
    """
    if CategoricalHMM is None and MultinomialHMM is None:
        raise ImportError("未安装 hmmlearn，请先安装后再运行该模块。")

    if CategoricalHMM is not None:
        model = CategoricalHMM(
            n_components=n_components,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
        )
    else:
        model = MultinomialHMM(
            n_components=n_components,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
        )

    with timer(name="模型训练耗时"):
        model.fit(X_obs, lengths)

    print("模型训练完成")
    print(f"n_components: {n_components}")
    print(f"n_iter: {n_iter}")
    print(f"tol: {tol}")

    return model


if __name__ == "__main__":
    from data_generation import hmm_data

    obs = hmm_data["obs"].values.reshape(-1, 1)
    lengths = [len(hmm_data)]
    train_model(obs, lengths)
