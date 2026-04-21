"""
pipelines/regression/regularization.py
正则化回归端到端流水线 (Lasso / Ridge / ElasticNet)

运行方式: python -m pipelines.regression.regularization
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_generation import regularization_data
from model_training.regression.regularization import train_model
from result_visualization.residual_plot import plot_residuals


def run():
    """正则化回归完整流水线"""
    print("=" * 60)
    print("正则化回归流水线 (Lasso / Ridge / ElasticNet)")
    print("=" * 60)

    data = regularization_data.copy()
    X = data.drop(columns=["price"])
    y = data["price"]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = train_model(X_train_s, y_train, feature_names=feature_names)

    for name, model in models.items():
        y_pred = model.predict(X_test_s)
        plot_residuals(
            y_test,
            y_pred,
            title=f"{name} 残差分析",
            model_name=name.lower(),
        )

    print(f"\n{'=' * 60}")
    print("正则化回归流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
