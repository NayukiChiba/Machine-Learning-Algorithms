"""
pipelines/regression/svr.py
SVR 回归端到端流水线

运行方式: python -m pipelines.regression.svr
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np

from config import get_model_output_dir
from data_exploration import (
    explore_regression_bivariate,
    explore_regression_multivariate,
    explore_regression_univariate,
)
from data_generation import svr_data
from data_visualization import plot_correlation_heatmap
from model_evaluation.regression_metrics import evaluate_regression
from model_training.regression.svr import train_model
from result_visualization.learning_curve import plot_learning_curve
from result_visualization.regression_result import plot_regression_result
from result_visualization.residual_plot import plot_residuals

MODEL = "svr"


def show_data_exploration(data) -> None:
    """
    展示 SVR 回归训练前的数据探索结果

    当前使用的是 Friedman1 非线性合成数据集（x1~x10，前 5 个特征有效）。
    """
    explore_regression_univariate(
        data,
        dataset_name="SVR",
    )
    explore_regression_bivariate(
        data,
        dataset_name="SVR",
    )
    explore_regression_multivariate(
        data,
        dataset_name="SVR",
    )


def show_data_preview(data, feature_names: list[str]) -> None:
    """
    展示 SVR 回归训练前的数据图

    使用相关性热力图 + 前 6 个特征与目标变量的散点网格。
    """
    save_dir = get_model_output_dir(MODEL)

    print("\n开始生成数据展示图...")
    plot_correlation_heatmap(
        data,
        columns=feature_names + ["price"],
        save_dir=save_dir,
        title="SVR 数据展示：相关性热力图",
        filename="data_correlation.png",
    )

    plot_cols = feature_names[:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("SVR 数据展示：特征与目标关系", fontsize=14, fontweight="bold")
    axes = axes.flatten()
    for axis, column in zip(axes, plot_cols, strict=True):
        axis.scatter(data[column], data["price"], s=14, alpha=0.4, color="#1E88E5")
        axis.set_xlabel(column)
        axis.set_ylabel("price")
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    scatter_path = save_dir / "data_feature_vs_price.png"
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"数据展示图已保存至: {scatter_path}")
    print("数据展示图生成完成。")


def show_result_preview(y_test, y_pred) -> None:
    """
    在终端展示部分预测结果（真实值/预测值/残差）
    """
    preview_size = min(8, len(y_test))
    y_true_values = np.asarray(y_test)[:preview_size]
    y_pred_values = np.asarray(y_pred)[:preview_size]

    print()
    print("=" * 60)
    print("SVR 结果展示")
    print("=" * 60)
    for y_true_value, y_pred_value in zip(y_true_values, y_pred_values, strict=True):
        row = {
            "真实值": round(float(y_true_value), 4),
            "预测值": round(float(y_pred_value), 4),
            "残差": round(float(y_true_value - y_pred_value), 4),
        }
        print(row)


def show_model_evaluation(y_test, y_pred, n_features: int) -> None:
    """
    在终端展示 SVR 回归的模型评估结果
    """
    metrics = evaluate_regression(
        y_test,
        y_pred,
        n_features=n_features,
        print_report=False,
    )

    print()
    print("=" * 60)
    print("SVR 模型评估展示")
    print("=" * 60)
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"R2: {metrics['r2']:.6f}")
    if "adjusted_r2" in metrics:
        print(f"调整R2: {metrics['adjusted_r2']:.6f}")


def run():
    """
    SVR 回归完整流水线

    流程：
    1. 数据探索；
    2. 数据展示；
    3. 训练与预测；
    4. 结果图展示；
    5. 终端结果预览与模型评估。
    """
    print("=" * 60)
    print("SVR 回归流水线")
    print("=" * 60)

    # 第 1 步：读取数据
    data = svr_data.copy()
    X = data.drop(columns=["price"])
    y = data["price"]
    feature_names = list(X.columns)

    # 第 2 步：数据探索
    show_data_exploration(data)

    # 第 3 步：数据展示
    show_data_preview(data, feature_names)

    # 第 4 步：训练与预测
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = train_model(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    # 第 5 步：结果图展示
    # 注意：SVR（RBF 核）无 feature_importances_ / coef_，因此不画特征重要性图
    plot_regression_result(
        y_test,
        y_pred,
        title="SVR 结果展示",
        model_name=MODEL,
    )
    plot_residuals(
        y_test,
        y_pred,
        title="SVR 残差分析",
        model_name=MODEL,
    )
    plot_learning_curve(
        SVR(kernel="rbf", C=10.0),
        X_train_s,
        y_train,
        scoring="r2",
        title="SVR 学习曲线",
        model_name=MODEL,
    )

    # 第 6 步：终端结果预览与模型评估
    show_result_preview(y_test, y_pred)
    show_model_evaluation(y_test, y_pred, n_features=len(feature_names))

    print(f"\n{'=' * 60}")
    print("SVR 流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
