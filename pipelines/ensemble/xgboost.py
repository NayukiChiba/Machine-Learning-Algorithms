"""
pipelines/ensemble/xgboost.py
XGBoost 回归端到端流水线

运行方式: python -m pipelines.ensemble.xgboost
"""

from sklearn.model_selection import learning_curve, train_test_split
import matplotlib.pyplot as plt
import numpy as np

from config import get_model_output_dir
from data_exploration import (
    explore_regression_bivariate,
    explore_regression_multivariate,
    explore_regression_univariate,
)
from data_generation import xgboost_data
from data_visualization import plot_correlation_heatmap
from model_evaluation.regression_metrics import evaluate_regression
from model_training.ensemble.xgboost import train_model
from result_visualization.residual_plot import plot_residuals
from result_visualization.feature_importance import plot_feature_importance
from result_visualization.regression_result import plot_regression_result

MODEL = "xgboost"


def show_data_exploration(data) -> None:
    """
    展示 XGBoost 回归训练前的数据探索结果

    当前使用的是 California Housing 真实数据集。
    回归任务里重点看：
    1. 目标变量分布；
    2. 特征与目标变量的相关性；
    3. 特征之间的共线性和降维潜力。
    """
    explore_regression_univariate(
        data,
        dataset_name="XGBoost",
    )
    explore_regression_bivariate(
        data,
        dataset_name="XGBoost",
    )
    explore_regression_multivariate(
        data,
        dataset_name="XGBoost",
    )


def show_data_preview(data, feature_names: list[str]) -> None:
    """
    展示 XGBoost 回归训练前的数据图

    这里用两个最有代表性的图：
    1. 相关性热力图
    2. 前几个特征与目标变量的散点关系图
    """
    save_dir = get_model_output_dir(MODEL)

    print("\n开始生成数据展示图...")
    plot_correlation_heatmap(
        data,
        columns=feature_names + ["price"],
        save_dir=save_dir,
        title="XGBoost 数据展示：相关性热力图",
        filename="data_correlation.png",
    )

    plot_cols = feature_names[:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("XGBoost 数据展示：特征与房价关系", fontsize=14, fontweight="bold")
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
    在终端展示部分预测结果

    回归任务的结果展示重点是：
    1. 真实值；
    2. 预测值；
    3. 残差。
    """
    preview_size = min(8, len(y_test))
    preview_rows = []
    y_true_values = y_test.to_numpy()[:preview_size]
    y_pred_values = np.asarray(y_pred)[:preview_size]
    for y_true_value, y_pred_value in zip(y_true_values, y_pred_values, strict=True):
        preview_rows.append(
            {
                "真实值": round(float(y_true_value), 4),
                "预测值": round(float(y_pred_value), 4),
                "残差": round(float(y_true_value - y_pred_value), 4),
            }
        )

    print()
    print("=" * 60)
    print("XGBoost 结果展示")
    print("=" * 60)
    for row in preview_rows:
        print(row)


def show_model_evaluation(y_test, y_pred, n_features: int) -> None:
    """
    在终端展示 XGBoost 回归的模型评估结果
    """
    metrics = evaluate_regression(
        y_test,
        y_pred,
        n_features=n_features,
        print_report=False,
    )

    print()
    print("=" * 60)
    print("XGBoost 模型评估展示")
    print("=" * 60)
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"R2: {metrics['r2']:.6f}")
    if "adjusted_r2" in metrics:
        print(f"调整R2: {metrics['adjusted_r2']:.6f}")


def plot_learning_curve_for_xgboost(X_train, y_train) -> None:
    """
    绘制 XGBoost 回归学习曲线
    """
    try:
        from xgboost import XGBRegressor
    except Exception as exc:  # noqa: BLE001
        raise ImportError("未安装 xgboost，无法构建学习曲线模型。") from exc

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
        random_state=42,
        n_jobs=-1,
    )

    train_sizes, train_scores, valid_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=1,
    )

    train_mean = train_scores.mean(axis=1)
    valid_mean = valid_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    valid_std = valid_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(train_sizes, train_mean, "o-", color="#1E88E5", label="训练得分")
    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color="#1E88E5",
    )
    ax.plot(train_sizes, valid_mean, "o-", color="#E64A19", label="验证得分")
    ax.fill_between(
        train_sizes,
        valid_mean - valid_std,
        valid_mean + valid_std,
        alpha=0.15,
        color="#E64A19",
    )
    ax.set_xlabel("训练样本数")
    ax.set_ylabel("R²")
    ax.set_title("XGBoost 学习曲线")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    save_dir = get_model_output_dir(MODEL)
    filepath = save_dir / "learning_curve.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"学习曲线已保存至: {filepath}")


def run():
    """
    XGBoost 回归完整流水线

    当前流程分成：
    1. 数据探索；
    2. 数据展示；
    3. 训练与预测；
    4. 结果展示；
    5. 模型评估。
    """
    print("=" * 60)
    print("XGBoost 回归流水线")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 第 1 步：读取当前算法对应的原始数据
    # ------------------------------------------------------------------
    # 当前使用的是 California Housing 真实数据集。
    data = xgboost_data.copy()
    X = data.drop(columns=["price"])
    y = data["price"]
    feature_names = list(X.columns)

    # ------------------------------------------------------------------
    # 第 2 步：数据探索
    # ------------------------------------------------------------------
    show_data_exploration(data)

    # ------------------------------------------------------------------
    # 第 3 步：数据展示
    # ------------------------------------------------------------------
    show_data_preview(data, feature_names)

    # ------------------------------------------------------------------
    # 第 4 步：训练与预测
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    # ------------------------------------------------------------------
    # 第 5 步：结果图展示
    # ------------------------------------------------------------------
    plot_regression_result(
        y_test,
        y_pred,
        title="XGBoost 结果展示",
        model_name=MODEL,
    )
    plot_residuals(y_test, y_pred, title="XGBoost 残差分析", model_name=MODEL)
    plot_feature_importance(
        model,
        feature_names=feature_names,
        title="XGBoost 特征重要性",
        model_name=MODEL,
    )
    plot_learning_curve_for_xgboost(X_train, y_train)

    # ------------------------------------------------------------------
    # 第 6 步：终端里的结果展示和模型评估
    # ------------------------------------------------------------------
    show_result_preview(y_test, y_pred)
    show_model_evaluation(y_test, y_pred, n_features=len(feature_names))

    print(f"\n{'=' * 60}")
    print("XGBoost 流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
