"""
result_visualization/dimensionality_diagnostics.py
降维模型诊断曲线

当前模块主要用于 PCA 这类降维模型的训练诊断。
这里的“训练曲线”不是神经网络那种 epoch-loss 曲线，
而是更适合降维任务的诊断图：
1. 主成分数 vs 累计解释方差
2. 主成分数 vs 重建误差
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from config import get_model_output_dir


def plot_pca_training_curve(
    X_scaled,
    model_name: str,
    max_components: int | None = None,
    figsize: tuple = (12, 5),
) -> dict:
    """
    绘制 PCA 的训练诊断曲线

    Args:
        X_scaled: 标准化后的特征矩阵
        model_name: 模型名称
        max_components: 最大扫描主成分数，默认取全部特征维度
        figsize: 图尺寸

    Returns:
        dict: 包含主成分范围、累计解释方差、重建误差
    """
    X_scaled = np.asarray(X_scaled)
    n_features = X_scaled.shape[1]
    if max_components is None:
        max_components = n_features
    max_components = min(max_components, n_features)

    component_range = np.arange(1, max_components + 1)
    cumulative_variances = []
    reconstruction_errors = []

    for n_components in component_range:
        model = PCA(n_components=n_components, random_state=42)
        X_transformed = model.fit_transform(X_scaled)
        X_reconstructed = model.inverse_transform(X_transformed)

        cumulative_variances.append(model.explained_variance_ratio_.sum())
        reconstruction_errors.append(np.mean((X_scaled - X_reconstructed) ** 2))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("PCA 训练诊断曲线", fontsize=14, fontweight="bold")

    axes[0].plot(
        component_range,
        cumulative_variances,
        marker="o",
        color="#1E88E5",
        linewidth=2,
    )
    axes[0].set_title("主成分数 vs 累计解释方差")
    axes[0].set_xlabel("主成分数")
    axes[0].set_ylabel("累计解释方差比")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(
        component_range,
        reconstruction_errors,
        marker="o",
        color="#D81B60",
        linewidth=2,
    )
    axes[1].set_title("主成分数 vs 重建误差")
    axes[1].set_xlabel("主成分数")
    axes[1].set_ylabel("重建误差 (MSE)")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    save_dir = get_model_output_dir(model_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / "training_curve.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"训练诊断曲线已保存至: {filepath}")

    return {
        "component_range": component_range,
        "cumulative_variances": np.asarray(cumulative_variances),
        "reconstruction_errors": np.asarray(reconstruction_errors),
    }
