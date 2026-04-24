"""
聚类任务训练器
"""

from __future__ import annotations

from sklearn.cluster import DBSCAN, KMeans


def trainKmeansModel(XTrain, randomState: int = 42):
    """训练 KMeans。"""
    model = KMeans(
        n_clusters=4,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=randomState,
    )
    model.fit(XTrain)
    return model


def trainDbscanModel(XTrain):
    """训练 DBSCAN。"""
    model = DBSCAN(eps=0.3, min_samples=5, metric="euclidean")
    model.fit(XTrain)
    return model
