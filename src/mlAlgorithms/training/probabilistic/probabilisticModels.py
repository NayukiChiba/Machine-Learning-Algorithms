"""
概率任务训练器
"""

from __future__ import annotations

from sklearn.mixture import GaussianMixture

try:
    from hmmlearn.hmm import CategoricalHMM
except Exception:  # noqa: BLE001
    CategoricalHMM = None

try:
    from hmmlearn.hmm import MultinomialHMM
except Exception:  # noqa: BLE001
    MultinomialHMM = None


def trainGaussianMixtureModel(XTrain, randomState: int = 42):
    """训练 GMM。"""
    model = GaussianMixture(
        n_components=3,
        covariance_type="full",
        max_iter=200,
        random_state=randomState,
    )
    model.fit(XTrain)
    return model


def trainHmmModel(XObs, lengths, randomState: int = 42):
    """训练 HMM。"""
    if CategoricalHMM is None and MultinomialHMM is None:
        raise ImportError("未安装 hmmlearn，请先安装后再运行。")
    if CategoricalHMM is not None:
        model = CategoricalHMM(
            n_components=3, n_iter=100, tol=1e-3, random_state=randomState
        )
    else:
        model = MultinomialHMM(
            n_components=3, n_iter=100, tol=1e-3, random_state=randomState
        )
    model.fit(XObs, lengths)
    return model
