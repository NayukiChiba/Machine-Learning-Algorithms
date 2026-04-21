"""
pipelines/regression/decision_tree.py
决策树回归端到端流水线

运行方式: python -m pipelines.regression.decision_tree
"""

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from data_generation import decision_tree_regression_data
from model_training.regression.decision_tree import train_model
from result_visualization.residual_plot import plot_residuals
from result_visualization.learning_curve import plot_learning_curve
from result_visualization.feature_importance import plot_feature_importance

MODEL = "decision_tree"


def run():
    """决策树回归完整流水线"""
    print("=" * 60)
    print("决策树回归流水线")
    print("=" * 60)

    data = decision_tree_regression_data.copy()
    X = data.drop(columns=["price"])
    y = data["price"]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values)

    plot_residuals(
        y_test,
        y_pred,
        title="决策树回归 残差分析",
        model_name=MODEL,
    )
    plot_feature_importance(
        model,
        feature_names=feature_names,
        title="决策树回归 特征重要性",
        model_name=MODEL,
    )
    plot_learning_curve(
        DecisionTreeRegressor(max_depth=6, random_state=42),
        X_train.values,
        y_train.values,
        scoring="r2",
        title="决策树回归 学习曲线",
        model_name=MODEL,
    )

    print(f"\n{'=' * 60}")
    print("决策树回归流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
