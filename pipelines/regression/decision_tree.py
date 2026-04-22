"""
pipelines/regression/decision_tree.py
决策树回归端到端流水线

运行方式: python -m pipelines.regression.decision_tree
"""

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np

from config import get_model_output_dir
from data_exploration import (
    explore_regression_bivariate,
    explore_regression_multivariate,
    explore_regression_univariate,
)
from data_generation import decision_tree_regression_data
from data_visualization import plot_correlation_heatmap
from model_evaluation.regression_metrics import evaluate_regression
from model_training.regression.decision_tree import train_model
from result_visualization.feature_importance import plot_feature_importance
from result_visualization.learning_curve import plot_learning_curve
from result_visualization.regression_result import plot_regression_result
from result_visualization.residual_plot import plot_residuals
from result_visualization.tree_structure import get_tree_rules, plot_tree_structure

MODEL = "decision_tree_regression"

# 限制终端打印的规则行数，避免刷屏
TREE_RULES_PREVIEW_LINES = 40


def show_data_exploration(data) -> None:
    """
    展示决策树回归训练前的数据探索结果

    当前使用的是 California Housing 真实数据集（8 个特征）。
    """
    explore_regression_univariate(
        data,
        dataset_name="决策树回归",
    )
    explore_regression_bivariate(
        data,
        dataset_name="决策树回归",
    )
    explore_regression_multivariate(
        data,
        dataset_name="决策树回归",
    )


def show_data_preview(data, feature_names: list[str]) -> None:
    """
    展示决策树回归训练前的数据图

    使用相关性热力图 + 前 6 个特征与目标变量的散点网格。
    """
    save_dir = get_model_output_dir(MODEL)

    print("\n开始生成数据展示图...")
    plot_correlation_heatmap(
        data,
        columns=feature_names + ["price"],
        save_dir=save_dir,
        title="决策树回归 数据展示：相关性热力图",
        filename="data_correlation.png",
    )

    plot_cols = feature_names[:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("决策树回归 数据展示：特征与房价关系", fontsize=14, fontweight="bold")
    axes = axes.flatten()
    for axis, column in zip(axes, plot_cols, strict=True):
        axis.scatter(data[column], data["price"], s=8, alpha=0.35, color="#1E88E5")
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
    print("决策树回归 结果展示")
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
    在终端展示决策树回归的模型评估结果
    """
    metrics = evaluate_regression(
        y_test,
        y_pred,
        n_features=n_features,
        print_report=False,
    )

    print()
    print("=" * 60)
    print("决策树回归 模型评估展示")
    print("=" * 60)
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"R2: {metrics['r2']:.6f}")
    if "adjusted_r2" in metrics:
        print(f"调整R2: {metrics['adjusted_r2']:.6f}")


def show_tree_rules_preview(model, feature_names: list[str]) -> None:
    """
    在终端预览决策树的文本规则（截取前若干行避免刷屏）
    """
    rules = get_tree_rules(model, feature_names=feature_names)
    lines = rules.splitlines()
    preview = lines[:TREE_RULES_PREVIEW_LINES]

    print()
    print("=" * 60)
    print(f"决策树规则预览（前 {len(preview)}/{len(lines)} 行）")
    print("=" * 60)
    for line in preview:
        print(line)
    if len(lines) > len(preview):
        print(f"... 省略剩余 {len(lines) - len(preview)} 行")


def run():
    """
    决策树回归完整流水线

    流程：
    1. 数据探索；
    2. 数据展示；
    3. 训练与预测；
    4. 结果图展示（含树结构）；
    5. 终端结果预览、模型评估、规则预览。
    """
    print("=" * 60)
    print("决策树回归流水线")
    print("=" * 60)

    # 第 1 步：读取数据
    data = decision_tree_regression_data.copy()
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

    model = train_model(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values)

    # 第 5 步：结果图展示
    plot_regression_result(
        y_test,
        y_pred,
        title="决策树回归 结果展示",
        model_name=MODEL,
    )
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
    plot_tree_structure(
        model,
        feature_names=feature_names,
        title="决策树回归 结构图",
        model_name=MODEL,
    )

    # 第 6 步：终端结果预览、模型评估、规则预览
    show_result_preview(y_test, y_pred)
    show_model_evaluation(y_test, y_pred, n_features=len(feature_names))
    show_tree_rules_preview(model, feature_names=feature_names)

    print(f"\n{'=' * 60}")
    print("决策树回归流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
