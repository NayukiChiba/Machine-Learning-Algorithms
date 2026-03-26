"""
result_visualization/residual_plot.py
回归残差可视化

绘制预测值 vs 真实值散点图、残差分布图。
适用于任何回归模型。

使用方式:
    from result_visualization.residual_plot import plot_residuals
"""

import numpy as np
import matplotlib.pyplot as plt

from config import RV_RESIDUAL_PLOT_DIR


def plot_residuals(
    y_true,
    y_pred,
    title: str = "残差分析",
    dataset_name: str = "default",
    model_name: str = "model",
    figsize: tuple = (14, 5),
):
    """
    绘制回归残差分析图（预测 vs 真实 + 残差分布）

    args:
        y_true: 真实值数组
        y_pred: 预测值数组
        title: 图标题
        dataset_name: 数据集名称
        model_name: 模型名称
        figsize: 图像尺寸
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 子图 1: 预测值 vs 真实值
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5, s=30)
    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    ax1.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=1.5, label="y = x")
    ax1.set_xlabel("真实值")
    ax1.set_ylabel("预测值")
    ax1.set_title(f"{title} — 预测 vs 真实")
    ax1.legend()

    # 子图 2: 残差分布
    ax2 = axes[1]
    ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidth=0.5, s=30)
    ax2.axhline(y=0, color="r", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("预测值")
    ax2.set_ylabel("残差 (真实 - 预测)")
    ax2.set_title(f"{title} — 残差分布")

    plt.tight_layout()
    save_dir = RV_RESIDUAL_PLOT_DIR / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / f"{model_name}_residual.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"残差图已保存至: {filepath}")
    plt.close(fig)


if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    X, y = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plot_residuals(
        y_test,
        y_pred,
        title="线性回归残差",
        dataset_name="test_lr",
        model_name="linear_regression",
    )
