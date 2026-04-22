"""
result_visualization/regression_result.py
回归结果展示图

用于展示测试集上真实值与预测值的对应关系。
"""

import matplotlib.pyplot as plt
import numpy as np

from config import get_model_output_dir


def plot_regression_result(
    y_true,
    y_pred,
    title: str = "回归结果展示",
    model_name: str = "model",
    figsize: tuple = (7, 6),
) -> None:
    """
    绘制真实值 vs 预测值散点图

    Args:
        y_true: 真实值
        y_pred: 预测值
        title: 图标题
        model_name: 模型名称
        figsize: 图尺寸
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        y_true,
        y_pred,
        s=18,
        alpha=0.45,
        color="#1E88E5",
        edgecolors="none",
    )

    lower = min(float(y_true.min()), float(y_pred.min()))
    upper = max(float(y_true.max()), float(y_pred.max()))
    ax.plot(
        [lower, upper], [lower, upper], color="#D81B60", linewidth=2, linestyle="--"
    )

    ax.set_xlabel("真实值")
    ax.set_ylabel("预测值")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    save_dir = get_model_output_dir(model_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / "result_display.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"回归结果展示图已保存至: {filepath}")
