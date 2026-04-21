"""
result_visualization/classification_result.py
分类结果展示图

用于展示测试集上“真实标签”和“预测标签”的对比效果。
"""

import matplotlib.pyplot as plt
import numpy as np

from config import get_model_output_dir


def plot_classification_result(
    X_test,
    y_true,
    y_pred,
    feature_names: list[str] | None = None,
    title: str = "分类结果展示",
    model_name: str = "model",
    figsize: tuple = (12, 5),
) -> None:
    """
    绘制分类结果展示图

    左图展示测试集真实标签，右图展示测试集预测标签。
    这样可以很直观地看出模型到底把哪些区域分对了、分错了。

    Args:
        X_test: 测试集特征矩阵，仅支持二维
        y_true: 测试集真实标签
        y_pred: 测试集预测标签
        feature_names: 特征名称列表
        title: 总标题
        model_name: 模型名称
        figsize: 图尺寸
    """
    if X_test.shape[1] != 2:
        raise ValueError("分类结果展示图仅支持二维特征。")

    if feature_names is None:
        feature_names = ["Feature 1", "Feature 2"]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    colors = plt.cm.tab10.colors

    # 左图：真实标签
    for index, cls in enumerate(unique_classes):
        mask = y_true == cls
        axes[0].scatter(
            X_test[mask, 0],
            X_test[mask, 1],
            s=32,
            alpha=0.85,
            color=colors[index % len(colors)],
            edgecolors="black",
            linewidths=0.3,
            label=f"类别 {cls}",
        )
    axes[0].set_title("测试集真实标签")
    axes[0].set_xlabel(feature_names[0])
    axes[0].set_ylabel(feature_names[1])
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    # 右图：预测标签
    for index, cls in enumerate(unique_classes):
        mask = y_pred == cls
        axes[1].scatter(
            X_test[mask, 0],
            X_test[mask, 1],
            s=32,
            alpha=0.85,
            color=colors[index % len(colors)],
            edgecolors="black",
            linewidths=0.3,
            label=f"类别 {cls}",
        )
    axes[1].set_title("测试集预测标签")
    axes[1].set_xlabel(feature_names[0])
    axes[1].grid(True, alpha=0.25)

    # 用红圈额外标出误分类样本，结果展示会更直接
    error_mask = y_true != y_pred
    if np.any(error_mask):
        axes[1].scatter(
            X_test[error_mask, 0],
            X_test[error_mask, 1],
            s=90,
            facecolors="none",
            edgecolors="red",
            linewidths=1.5,
            label="误分类样本",
        )
        axes[1].legend(loc="best")

    plt.tight_layout()
    save_dir = get_model_output_dir(model_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / "result_display.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"分类结果展示图已保存至: {filepath}")
