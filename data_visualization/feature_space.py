"""
data_visualization/feature_space.py
特征空间可视化模块

提供可复用的单数据集特征空间可视化函数。
当前模块不再负责“批量生成所有数据集的图片”。
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from sklearn.decomposition import PCA

from config import DATA_VIS_FEATURE_SPACE_DIR as OUTPUT_DIR

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


# --- 通用绘图工具 ---


def _save_fig(fig: plt.Figure, filename: str, output_name: str) -> None:
    """
    保存图表到当前模块目录

    args:
        fig(Figure): matplotlib 图表对象
        filename(str): 文件名
        output_name(str): 输出名称前缀
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / f"{output_name}_{filename}"
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {filepath}")


def _save_single_dataset_fig(fig: plt.Figure, save_dir: Path, filename: str) -> None:
    """
    保存单数据集展示图到指定目录

    Args:
        fig: matplotlib 图表对象
        save_dir: 保存目录
        filename: 文件名
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / filename
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"数据展示图已保存至: {filepath}")


def _plot_2d_projection(
    data: DataFrame,
    feature_cols: list[str],
    color_col: str,
    output_name: str,
    title_suffix: str = "",
) -> None:
    """
    PCA 降至 2D 后按标签着色的散点图

    如果原始特征就是 2D，直接绘制原始特征空间

    args:
        data(DataFrame): 数据
        feature_cols(list[str]): 特征列名列表
        color_col(str): 着色列名
        output_name(str): 输出名称前缀
        title_suffix(str): 标题后缀
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    X = data[feature_cols].values
    labels = data[color_col].values

    if len(feature_cols) <= 2:
        # 原始就是 2D，直接使用
        x_plot = X[:, 0]
        y_plot = X[:, 1] if X.shape[1] > 1 else np.zeros(len(X))
        x_label = feature_cols[0]
        y_label = feature_cols[1] if len(feature_cols) > 1 else ""
        fig.suptitle(
            f"{output_name} — 原始特征空间{title_suffix}",
            fontsize=13,
            fontweight="bold",
        )
    else:
        # PCA 降至 2D
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        x_plot = X_2d[:, 0]
        y_plot = X_2d[:, 1]

        # 解释方差比
        ev1 = pca.explained_variance_ratio_[0] * 100
        ev2 = pca.explained_variance_ratio_[1] * 100
        x_label = f"PC1 ({ev1:.1f}%)"
        y_label = f"PC2 ({ev2:.1f}%)"
        fig.suptitle(
            f"{output_name} — PCA 2D 投影{title_suffix}",
            fontsize=13,
            fontweight="bold",
        )

    # 按类别着色
    unique_labels = sorted(np.unique(labels))
    colors = sns.color_palette("Set2", len(unique_labels))

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(
            x_plot[mask],
            y_plot[mask],
            s=15,
            alpha=0.6,
            color=colors[i],
            label=f"{color_col}={lbl}",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    _save_fig(fig, "01_2d_projection.png", output_name)


def _plot_3d_projection(
    data: DataFrame,
    feature_cols: list[str],
    color_col: str,
    output_name: str,
) -> None:
    """
    PCA 降至 3D 后的三维散点图

    至少需要 3 个特征才有意义

    args:
        data(DataFrame): 数据
        feature_cols(list[str]): 特征列名列表
        color_col(str): 着色列名
        output_name(str): 输出名称前缀
    """
    if len(feature_cols) < 3:
        return

    X = data[feature_cols].values
    labels = data[color_col].values

    # PCA 降至 3D
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ev = pca.explained_variance_ratio_ * 100

    unique_labels = sorted(np.unique(labels))
    colors = sns.color_palette("Set2", len(unique_labels))

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(
            X_3d[mask, 0],
            X_3d[mask, 1],
            X_3d[mask, 2],
            s=10,
            alpha=0.5,
            color=colors[i],
            label=f"{color_col}={lbl}",
        )

    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.set_zlabel(f"PC3 ({ev[2]:.1f}%)")
    ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"{output_name} — PCA 3D 投影", fontsize=13, fontweight="bold")

    fig.tight_layout()
    _save_fig(fig, "02_3d_projection.png", output_name)


def plot_feature_space_2d(
    data: DataFrame,
    feature_cols: list[str],
    label_col: str,
    save_dir: Path,
    title: str = "特征空间 2D 投影",
    filename: str = "data_feature_space_2d.png",
) -> None:
    """
    为单个数据集绘制 2D 特征空间图

    当原始特征已经是 2D 时，直接绘制原始特征空间；
    当原始特征维度大于 2 时，自动使用 PCA 压到 2D 再绘制。

    Args:
        data: 数据集
        feature_cols: 特征列名
        label_col: 标签列名
        save_dir: 保存目录
        title: 图标题
        filename: 保存文件名
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    X = data[feature_cols].values
    labels = data[label_col].values

    if len(feature_cols) <= 2:
        x_plot = X[:, 0]
        y_plot = X[:, 1] if X.shape[1] > 1 else np.zeros(len(X))
        x_label = feature_cols[0]
        y_label = feature_cols[1] if len(feature_cols) > 1 else ""
    else:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        x_plot = X_2d[:, 0]
        y_plot = X_2d[:, 1]
        ev1 = pca.explained_variance_ratio_[0] * 100
        ev2 = pca.explained_variance_ratio_[1] * 100
        x_label = f"PC1 ({ev1:.1f}%)"
        y_label = f"PC2 ({ev2:.1f}%)"

    unique_labels = sorted(np.unique(labels))
    colors = sns.color_palette("Set2", len(unique_labels))

    for color, label in zip(colors, unique_labels, strict=True):
        mask = labels == label
        ax.scatter(
            x_plot[mask],
            y_plot[mask],
            s=16,
            alpha=0.65,
            color=color,
            label=f"{label_col}={label}",
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    _save_single_dataset_fig(fig, save_dir, filename)


def plot_feature_space_3d(
    data: DataFrame,
    feature_cols: list[str],
    label_col: str,
    save_dir: Path,
    title: str = "特征空间 3D 投影",
    filename: str = "data_feature_space_3d.png",
) -> None:
    """
    为单个数据集绘制 3D 特征空间图

    原始特征数不足 3 时，函数会直接返回，不强行生成无意义图。

    Args:
        data: 数据集
        feature_cols: 特征列名
        label_col: 标签列名
        save_dir: 保存目录
        title: 图标题
        filename: 保存文件名
    """
    if len(feature_cols) < 3:
        return

    X = data[feature_cols].values
    labels = data[label_col].values
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_ * 100

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    unique_labels = sorted(np.unique(labels))
    colors = sns.color_palette("Set2", len(unique_labels))

    for color, label in zip(colors, unique_labels, strict=True):
        mask = labels == label
        ax.scatter(
            X_3d[mask, 0],
            X_3d[mask, 1],
            X_3d[mask, 2],
            s=10,
            alpha=0.55,
            color=color,
            label=f"{label_col}={label}",
        )

    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    ax.set_zlabel(f"PC3 ({explained[2]:.1f}%)")
    ax.legend(fontsize=7, loc="best")

    fig.tight_layout()
    _save_single_dataset_fig(fig, save_dir, filename)
