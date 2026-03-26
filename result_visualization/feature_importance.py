"""
result_visualization/feature_importance.py
特征重要性可视化

绘制基于树模型的 feature_importances_ 或线性模型 coef_ 的水平柱状图。

使用方式:
    from result_visualization.feature_importance import plot_feature_importance
"""

import numpy as np
import matplotlib.pyplot as plt

from config import RV_FEATURE_IMPORTANCE_DIR


def plot_feature_importance(
    model,
    feature_names: list[str] | None = None,
    top_n: int | None = None,
    title: str = "特征重要性",
    dataset_name: str = "default",
    model_name: str = "model",
    figsize: tuple = (10, 7),
):
    """
    绘制特征重要性水平柱状图

    args:
        model: 训练好的模型（需有 feature_importances_ 或 coef_ 属性）
        feature_names: 特征名称列表
        top_n: 只展示前 N 个最重要的特征（None 则展示全部）
        title: 图标题
        dataset_name: 数据集名称
        model_name: 模型名称
        figsize: 图像尺寸
    """
    # 获取重要性
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            importances = np.abs(coef).mean(axis=0)
        else:
            importances = np.abs(coef)
    else:
        raise AttributeError("模型没有 feature_importances_ 或 coef_ 属性")

    n_features = len(importances)
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]

    # 排序
    indices = np.argsort(importances)[::-1]
    if top_n is not None:
        indices = indices[:top_n]

    sorted_names = [feature_names[i] for i in indices]
    sorted_values = importances[indices]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_values, align="center", color="tab:blue", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()  # 最重要的在最上方
    ax.set_xlabel("重要性")
    ax.set_title(title)

    plt.tight_layout()
    save_dir = RV_FEATURE_IMPORTANCE_DIR / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / f"{model_name}_feature_importance.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"特征重要性图已保存至: {filepath}")
    plt.close(fig)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    feature_names = [f"特征_{i}" for i in range(10)]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    plot_feature_importance(
        model,
        feature_names=feature_names,
        top_n=8,
        title="随机森林 特征重要性",
        dataset_name="test_rf",
        model_name="random_forest",
    )
