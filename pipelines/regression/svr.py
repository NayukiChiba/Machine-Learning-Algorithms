"""
pipelines/regression/svr.py
SVR 回归端到端流水线

运行方式: python -m pipelines.regression.svr
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from data_generation import svr_data
from model_training.regression.svr import train_model
from result_visualization.residual_plot import plot_residuals
from result_visualization.learning_curve import plot_learning_curve

DATASET = "svr"
MODEL = "svr"


def run():
    """SVR 回归完整流水线"""
    print("=" * 60)
    print("SVR 回归流水线")
    print("=" * 60)

    data = svr_data.copy()
    X = data.drop(columns=["price"])
    y = data["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = train_model(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    plot_residuals(
        y_test, y_pred, title="SVR 残差分析", dataset_name=DATASET, model_name=MODEL
    )
    plot_learning_curve(
        SVR(kernel="rbf", C=10.0),
        X_train_s,
        y_train,
        scoring="r2",
        title="SVR 学习曲线",
        dataset_name=DATASET,
        model_name=MODEL,
    )

    print(f"\n{'=' * 60}")
    print("SVR 流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
