"""
pipelines/regression/linear_regression.py
线性回归端到端流水线

运行方式: python -m pipelines.regression.linear_regression
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from data_generation import linear_regression_data
from model_training.regression.linear_regression import train_model
from result_visualization.residual_plot import plot_residuals
from result_visualization.learning_curve import plot_learning_curve

MODEL = "linear_regression"


def run():
    """线性回归完整流水线"""
    print("=" * 60)
    print("线性回归流水线")
    print("=" * 60)

    data = linear_regression_data.copy()
    X = data.drop(columns=["price"])
    y = data["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    plot_residuals(
        y_test,
        y_pred,
        title="线性回归 残差分析",
        model_name=MODEL,
    )
    plot_learning_curve(
        LinearRegression(),
        X_train,
        y_train,
        scoring="r2",
        title="线性回归 学习曲线",
        model_name=MODEL,
    )

    print(f"\n{'=' * 60}")
    print("线性回归流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
