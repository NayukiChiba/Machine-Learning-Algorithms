"""
pipelines/ensemble/xgboost.py
XGBoost 回归端到端流水线

运行方式: python -m pipelines.ensemble.xgboost
"""

from sklearn.model_selection import train_test_split

from data_generation import xgboost_data
from model_training.ensemble.xgboost import train_model
from result_visualization.residual_plot import plot_residuals
from result_visualization.feature_importance import plot_feature_importance

MODEL = "xgboost"


def run():
    """XGBoost 回归完整流水线"""
    print("=" * 60)
    print("XGBoost 回归流水线")
    print("=" * 60)

    data = xgboost_data.copy()
    X = data.drop(columns=["price"])
    y = data["price"]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    plot_residuals(y_test, y_pred, title="XGBoost 残差分析", model_name=MODEL)
    plot_feature_importance(
        model,
        feature_names=feature_names,
        title="XGBoost 特征重要性",
        model_name=MODEL,
    )

    print(f"\n{'=' * 60}")
    print("XGBoost 流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
