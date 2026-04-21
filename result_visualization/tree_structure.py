"""
result_visualization/tree_structure.py
决策树结构可视化

保存决策树结构图，并提供文本规则导出能力。
"""

import matplotlib.pyplot as plt
from sklearn.tree import export_text, plot_tree

from config import get_model_output_dir


def plot_tree_structure(
    model,
    feature_names: list[str],
    class_names: list[str] | None = None,
    title: str = "决策树结构图",
    model_name: str = "model",
    figsize: tuple = (18, 10),
):
    """
    绘制并保存决策树结构图

    Args:
        model: 已训练的决策树模型
        feature_names: 特征名称列表
        class_names: 类别名称列表
        title: 图标题
        model_name: 模型名称
        figsize: 图像尺寸
    """
    save_dir = get_model_output_dir(model_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=True,
        proportion=True,
        ax=ax,
    )
    ax.set_title(title)

    plt.tight_layout()
    filepath = save_dir / "tree_structure.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"决策树结构图已保存至: {filepath}")


def get_tree_rules(model, feature_names: list[str]) -> str:
    """
    获取文本形式的决策树规则

    Args:
        model: 已训练的决策树模型
        feature_names: 特征名称列表

    Returns:
        str: 文本规则
    """
    return export_text(model, feature_names=feature_names)
