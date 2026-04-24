"""
降维任务训练器
"""

from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def trainPcaModel(XTrain, nComponents: int, randomState: int = 42):
    """训练 PCA。"""
    model = PCA(n_components=nComponents, svd_solver="auto", random_state=randomState)
    model.fit(XTrain)
    return model


def trainLdaModel(XTrain, yTrain, nComponents: int = 2):
    """训练 LDA。"""
    model = LinearDiscriminantAnalysis(n_components=nComponents, solver="svd")
    model.fit(XTrain, yTrain)
    return model
