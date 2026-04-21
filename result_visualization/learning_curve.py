"""
result_visualization/learning_curve.py
学习曲线可视化

绘制模型随训练集规模变化的训练得分与验证得分曲线，
用于判断欠拟合/过拟合。

使用方式:
    from result_visualization.learning_curve import plot_learning_curve
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve as sk_learning_curve

from config import get_model_output_dir


def plot_learning_curve(
    model,
    X,
    y,
    cv: int = 5,
    scoring: str = "accuracy",
    train_sizes=None,
    title: str = "学习曲线",
    model_name: str = "model",
    figsize: tuple = (10, 7),
    n_jobs: int = 1,
):
    """
    绘制学习曲线

    args:
        model: 分类器或回归器实例（未训练的克隆会自动创建）
        X: 完整特征矩阵
        y: 完整标签
        cv: 交叉验证折数
        scoring: 评分指标
        train_sizes: 训练集大小比例数组
        title: 图标题
        model_name: 模型名称
        figsize: 图像尺寸
        n_jobs: 并行数，默认 1 以避免 Windows 环境下的进程权限问题
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    train_sizes_abs, train_scores, val_scores = sk_learning_curve(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        train_sizes=train_sizes,
        n_jobs=n_jobs,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=figsize)

    # 训练得分
    ax.plot(train_sizes_abs, train_mean, "o-", color="tab:blue", label="训练得分")
    ax.fill_between(
        train_sizes_abs,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color="tab:blue",
    )

    # 验证得分
    ax.plot(train_sizes_abs, val_mean, "o-", color="tab:orange", label="验证得分")
    ax.fill_between(
        train_sizes_abs,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.15,
        color="tab:orange",
    )

    ax.set_xlabel("训练样本数")
    ax.set_ylabel(scoring)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_dir = get_model_output_dir(model_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / "learning_curve.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"学习曲线已保存至: {filepath}")
    plt.close(fig)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.neighbors import KNeighborsClassifier

    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    model = KNeighborsClassifier(n_neighbors=5)
    plot_learning_curve(
        model,
        X,
        y,
        title="KNN 学习曲线",
        model_name="knn",
    )
