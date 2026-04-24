"""
分类任务训练器
"""

from __future__ import annotations

from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from lightgbm import LGBMClassifier
except Exception as importError:  # noqa: BLE001
    LGBMClassifier = None
    LIGHTGBM_IMPORT_ERROR = importError


def trainLogisticRegression(XTrain, yTrain, randomState: int = 42):
    """训练逻辑回归。"""
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=randomState,
    )
    model.fit(XTrain, yTrain)
    return model


def trainDecisionTreeClassifier(XTrain, yTrain, randomState: int = 42):
    """训练决策树分类器。"""
    model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        criterion="gini",
        random_state=randomState,
    )
    model.fit(XTrain, yTrain)
    return model


def trainSvcClassifier(XTrain, yTrain, randomState: int = 42):
    """训练 SVC。"""
    model = SVC(
        C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=randomState
    )
    model.fit(XTrain, yTrain)
    return model


def trainNaiveBayesClassifier(XTrain, yTrain):
    """训练高斯朴素贝叶斯。"""
    model = GaussianNB(var_smoothing=1e-9)
    model.fit(XTrain, yTrain)
    return model


def trainKnnClassifier(XTrain, yTrain):
    """训练 KNN。"""
    model = KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="minkowski")
    model.fit(XTrain, yTrain)
    return model


def trainRandomForestClassifier(XTrain, yTrain, randomState: int = 42):
    """训练随机森林。"""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=randomState,
        n_jobs=1,
    )
    model.fit(XTrain, yTrain)
    return model


def trainBaggingClassifier(XTrain, yTrain, randomState: int = 42):
    """训练 Bagging。"""
    baseEstimator = DecisionTreeClassifier(random_state=randomState)
    try:
        model = BaggingClassifier(
            estimator=baseEstimator,
            n_estimators=80,
            max_samples=0.8,
            max_features=1.0,
            bootstrap=True,
            oob_score=True,
            random_state=randomState,
            n_jobs=1,
        )
    except TypeError:
        model = BaggingClassifier(
            base_estimator=baseEstimator,
            n_estimators=80,
            max_samples=0.8,
            max_features=1.0,
            bootstrap=True,
            oob_score=True,
            random_state=randomState,
            n_jobs=1,
        )
    model.fit(XTrain, yTrain)
    return model


def trainGbdtClassifier(XTrain, yTrain, randomState: int = 42):
    """训练 GBDT。"""
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        subsample=1.0,
        random_state=randomState,
    )
    model.fit(XTrain, yTrain)
    return model


def trainLightgbmClassifier(XTrain, yTrain, randomState: int = 42):
    """训练 LightGBM。"""
    if LGBMClassifier is None:
        raise ImportError(
            "未安装 lightgbm，请先安装后再运行。"
        ) from LIGHTGBM_IMPORT_ERROR
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=randomState,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(XTrain, yTrain)
    return model
