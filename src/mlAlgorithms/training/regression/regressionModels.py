"""
回归任务训练器
"""

from __future__ import annotations

from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
except Exception as importError:  # noqa: BLE001
    XGBRegressor = None
    XGBOOST_IMPORT_ERROR = importError


def trainLinearRegressionModel(XTrain, yTrain):
    """训练线性回归。"""
    model = LinearRegression()
    model.fit(XTrain, yTrain)
    return model


def trainSvrRegressionModel(XTrain, yTrain):
    """训练 SVR。"""
    model = SVR(C=10.0, epsilon=0.1, kernel="rbf", gamma="scale")
    model.fit(XTrain, yTrain)
    return model


def trainDecisionTreeRegressionModel(XTrain, yTrain, randomState: int = 42):
    """训练决策树回归。"""
    model = DecisionTreeRegressor(
        max_depth=6,
        min_samples_split=6,
        min_samples_leaf=3,
        random_state=randomState,
    )
    model.fit(XTrain, yTrain)
    return model


def trainRegularizationModels(XTrain, yTrain, randomState: int = 42):
    """训练正则化模型组。"""
    models = {
        "lasso": Lasso(alpha=0.15, max_iter=10000, random_state=randomState),
        "ridge": Ridge(alpha=2.0, random_state=randomState),
        "elasticnet": ElasticNet(
            alpha=0.2, l1_ratio=0.5, max_iter=10000, random_state=randomState
        ),
    }
    for model in models.values():
        model.fit(XTrain, yTrain)
    return models


def trainXgboostRegressionModel(XTrain, yTrain, randomState: int = 42):
    """训练 XGBoost。"""
    if XGBRegressor is None:
        raise ImportError(
            "未安装 xgboost，请先安装后再运行。"
        ) from XGBOOST_IMPORT_ERROR
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=randomState,
        n_jobs=-1,
    )
    model.fit(XTrain, yTrain)
    return model
