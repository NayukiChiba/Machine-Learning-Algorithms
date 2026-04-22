"""
pipelines/regression/regularization.py
正则化回归端到端流水线 (Lasso / Ridge / ElasticNet)

每个模型独立完整地走一遍 6 步流程（数据探索 → 数据展示 → 训练预测
→ 结果图 → 终端预览 → 模型评估），输出到各自的 outputs/<name> 目录。

运行方式: python -m pipelines.regression.regularization
"""

from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

from config import get_model_output_dir
from data_exploration import (
    explore_regression_bivariate,
    explore_regression_multivariate,
    explore_regression_univariate,
)
from data_generation import regularization_data
from data_visualization import plot_correlation_heatmap
from model_evaluation.regression_metrics import evaluate_regression
from model_training.regression.regularization import train_model
from result_visualization.learning_curve import plot_learning_curve
from result_visualization.regression_result import plot_regression_result
from result_visualization.residual_plot import plot_residuals


def show_data_exploration(data, name: str) -> None:
    """
    展示某个正则化模型训练前的数据探索结果

    Args:
        data: 完整数据集
        name: 模型名（Lasso / Ridge / ElasticNet）
    """
    explore_regression_univariate(
        data,
        dataset_name=f"{name} 回归",
    )
    explore_regression_bivariate(
        data,
        dataset_name=f"{name} 回归",
    )
    explore_regression_multivariate(
        data,
        dataset_name=f"{name} 回归",
    )


def show_data_preview(data, feature_names: list[str], name: str) -> None:
    """
    展示某个正则化模型训练前的数据图

    使用相关性热力图 + 前 6 个特征与目标变量的散点网格。
    """
    save_dir = get_model_output_dir(name.lower())

    print(f"\n开始生成 {name} 数据展示图...")
    plot_correlation_heatmap(
        data,
        columns=feature_names + ["price"],
        save_dir=save_dir,
        title=f"{name} 数据展示：相关性热力图",
        filename="data_correlation.png",
    )

    plot_cols = feature_names[:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"{name} 数据展示：特征与目标关系", fontsize=14, fontweight="bold")
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


def plot_coefficients(model, feature_names: list[str], name: str) -> None:
    """
    绘制单个正则化模型的系数条形图

    线性模型有 .coef_ 但没有 .feature_importances_，因此自绘条形图。
    """
    save_dir = get_model_output_dir(name.lower())
    coef = np.asarray(model.coef_).ravel()

    # 按系数绝对值降序排列，更直观
    order = np.argsort(np.abs(coef))[::-1]
    sorted_names = [feature_names[i] for i in order]
    sorted_coef = coef[order]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(feature_names))))
    colors = ["#1E88E5" if c >= 0 else "#E64A19" for c in sorted_coef]
    ax.barh(sorted_names, sorted_coef, color=colors, edgecolor="none", alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("系数")
    ax.set_title(f"{name} 系数（按绝对值降序）")
    ax.grid(True, axis="x", alpha=0.25)
    ax.invert_yaxis()
    fig.tight_layout()

    filepath = save_dir / "coefficients.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"系数条形图已保存至: {filepath}")


def show_result_preview(y_test, y_pred, name: str) -> None:
    """
    在终端展示部分预测结果（真实值/预测值/残差）
    """
    preview_size = min(8, len(y_test))
    y_true_values = np.asarray(y_test)[:preview_size]
    y_pred_values = np.asarray(y_pred)[:preview_size]

    print()
    print("=" * 60)
    print(f"{name} 结果展示")
    print("=" * 60)
    for y_true_value, y_pred_value in zip(y_true_values, y_pred_values, strict=True):
        row = {
            "真实值": round(float(y_true_value), 4),
            "预测值": round(float(y_pred_value), 4),
            "残差": round(float(y_true_value - y_pred_value), 4),
        }
        print(row)


def show_model_evaluation(y_test, y_pred, n_features: int, name: str) -> None:
    """
    在终端展示某个正则化模型的评估结果
    """
    metrics = evaluate_regression(
        y_test,
        y_pred,
        n_features=n_features,
        print_report=False,
    )

    print()
    print("=" * 60)
    print(f"{name} 模型评估展示")
    print("=" * 60)
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"R2: {metrics['r2']:.6f}")
    if "adjusted_r2" in metrics:
        print(f"调整R2: {metrics['adjusted_r2']:.6f}")


def _build_learning_curve_estimator(name: str, model):
    """
    根据已训练模型的超参构造一个未训练的同型估计器，供学习曲线使用
    """
    if name == "Lasso":
        return Lasso(alpha=model.alpha, max_iter=10000, random_state=42)
    if name == "Ridge":
        return Ridge(alpha=model.alpha, random_state=42)
    if name == "ElasticNet":
        return ElasticNet(
            alpha=model.alpha,
            l1_ratio=model.l1_ratio,
            max_iter=10000,
            random_state=42,
        )
    raise ValueError(f"未知正则化模型: {name}")


def run():
    """
    正则化回归完整流水线 (Lasso / Ridge / ElasticNet)

    流程：
    1. 共享数据预处理（划分 + 标准化）；
    2. 一次性训练三个模型；
    3. 对每个模型独立完整地走 6 步：
       数据探索 → 数据展示 → 训练预测（已完成） → 结果图 → 终端预览 → 模型评估。
    """
    print("=" * 60)
    print("正则化回归流水线 (Lasso / Ridge / ElasticNet)")
    print("=" * 60)

    # 第 1 步：读取数据
    data = regularization_data.copy()
    X = data.drop(columns=["price"])
    y = data["price"]
    feature_names = list(X.columns)

    # 数据预处理：训练集/测试集划分 + 标准化（三个模型共用）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 第 2 步：一次性训练三个模型
    models = train_model(X_train_s, y_train, feature_names=feature_names)

    # 第 3 步：对每个模型独立完整地走完整流程
    for name, model in models.items():
        print()
        print("#" * 60)
        print(f"# {name} 完整流水线")
        print("#" * 60)

        # 数据探索（每个模型独立打印一次，dataset_name 区分）
        show_data_exploration(data, name)

        # 数据展示（独立目录）
        show_data_preview(data, feature_names, name)

        # 预测
        y_pred = model.predict(X_test_s)

        # 结果图
        plot_regression_result(
            y_test,
            y_pred,
            title=f"{name} 结果展示",
            model_name=name.lower(),
        )
        plot_residuals(
            y_test,
            y_pred,
            title=f"{name} 残差分析",
            model_name=name.lower(),
        )
        plot_coefficients(model, feature_names, name)
        plot_learning_curve(
            _build_learning_curve_estimator(name, model),
            X_train_s,
            y_train,
            scoring="r2",
            title=f"{name} 学习曲线",
            model_name=name.lower(),
        )

        # 终端结果预览与模型评估
        show_result_preview(y_test, y_pred, name)
        show_model_evaluation(y_test, y_pred, n_features=len(feature_names), name=name)

    print(f"\n{'=' * 60}")
    print("正则化回归流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
