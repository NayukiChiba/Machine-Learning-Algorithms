"""
result_visualization/confusion_matrix.py
混淆矩阵可视化

绘制分类模型的混淆矩阵热力图，支持二分类和多分类。

使用方式:
    from result_visualization.confusion_matrix import plot_confusion_matrix
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from config import RV_CONFUSION_MATRIX_DIR


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: list[str] | None = None,
    normalize: bool = False,
    title: str = "混淆矩阵",
    dataset_name: str = "default",
    model_name: str = "model",
    figsize: tuple = (8, 7),
    cmap: str = "Blues",
):
    """
    绘制混淆矩阵热力图

    args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        normalize: 是否归一化（按行显示百分比）
        title: 图标题
        dataset_name: 数据集名称
        model_name: 模型名称
        figsize: 图像尺寸
        cmap: 颜色映射
    """
    cm = sk_confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="真实标签",
        xlabel="预测标签",
        title=title,
    )

    # 旋转 x 轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 在矩阵中填入数字
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    save_dir = RV_CONFUSION_MATRIX_DIR / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / f"{model_name}_confusion_matrix.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"混淆矩阵已保存至: {filepath}")
    plt.close(fig)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plot_confusion_matrix(
        y_test,
        y_pred,
        class_names=["负类", "正类"],
        title="逻辑回归 混淆矩阵",
        dataset_name="test_lr",
        model_name="logistic_regression",
    )
