"""
data_visualization/scatter.py
散点图模块

提供可复用的单数据集散点图函数。
当前模块不再负责“批量生成所有数据集的图片”。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

from config import DATA_VIS_SCATTER_DIR as OUTPUT_DIR

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 使用高对比度离散色板，避免类别点云颜色过淡、过接近。
DISCRETE_COLORS = [
    "#D81B60",  # 洋红
    "#1E88E5",  # 蓝
    "#FFC107",  # 黄
    "#004D40",  # 深青
    "#E64A19",  # 橙红
    "#6A1B9A",  # 紫
    "#2E7D32",  # 绿
    "#5D4037",  # 棕
]


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


def _get_discrete_colors(n_colors: int) -> list[str]:
    """
    返回指定数量的高对比度离散颜色

    Args:
        n_colors: 需要的颜色数量

    Returns:
        list[str]: 十六进制颜色列表
    """
    colors = []
    for index in range(n_colors):
        colors.append(DISCRETE_COLORS[index % len(DISCRETE_COLORS)])
    return colors


def _plot_2d_scatter(
    data: DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    output_name: str,
    title: str,
    filename: str,
) -> None:
    """
    绘制二维散点图，按类别/簇着色

    args:
        data(DataFrame): 数据
        x_col(str): x 轴特征列名
        y_col(str): y 轴特征列名
        color_col(str): 着色列名 (类别/簇标签)
        output_name(str): 输出名称前缀
        title(str): 图标题
        filename(str): 保存文件名
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    classes = sorted(data[color_col].unique())
    colors = sns.color_palette("Set2", len(classes))

    for i, cls in enumerate(classes):
        subset = data[data[color_col] == cls]
        ax.scatter(
            subset[x_col],
            subset[y_col],
            s=20,
            alpha=0.6,
            color=colors[i],
            label=f"{color_col}={cls}",
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    _save_fig(fig, filename, output_name)


def _plot_pairplot(
    data: DataFrame,
    feature_cols: list[str],
    color_col: str,
    output_name: str,
    max_features: int = 6,
) -> None:
    """
    绘制散点矩阵 (Pairplot)

    当特征数超过 max_features 时，只选取前 max_features 个特征
    避免图表过大导致内存问题

    args:
        data(DataFrame): 数据
        feature_cols(list[str]): 特征列名列表
        color_col(str): 着色列名
        output_name(str): 输出名称前缀
        max_features(int): 最大特征数
    """
    # 限制特征数量，避免散点矩阵过大
    plot_cols = feature_cols[:max_features]

    # seaborn pairplot
    plot_data = data[plot_cols + [color_col]].copy()
    plot_data[color_col] = plot_data[color_col].astype(str)

    g = sns.pairplot(
        plot_data,
        hue=color_col,
        palette="Set2",
        plot_kws={"s": 15, "alpha": 0.5},
        diag_kws={"alpha": 0.5},
        height=2,
    )
    g.fig.suptitle(f"{output_name} — 散点矩阵", fontsize=13, fontweight="bold", y=1.02)

    if len(feature_cols) > max_features:
        g.fig.text(
            0.5,
            -0.02,
            f"(仅展示前 {max_features} 个特征，共 {len(feature_cols)} 个)",
            ha="center",
            fontsize=9,
            color="gray",
        )

    _save_fig(g.fig, "02_pairplot.png", output_name)


def _plot_regression_scatter(
    data: DataFrame,
    feature_cols: list[str],
    target_col: str,
    output_name: str,
    max_features: int = 8,
) -> None:
    """
    回归任务: 绘制每个特征 vs 目标变量的散点图

    每个子图是一个特征与目标变量的关系

    args:
        data(DataFrame): 数据
        feature_cols(list[str]): 特征列名列表
        target_col(str): 目标变量列名
        output_name(str): 输出名称前缀
        max_features(int): 最大特征数
    """
    plot_cols = feature_cols[:max_features]
    n = len(plot_cols)

    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    fig.suptitle(
        f"{output_name} — 特征 vs {target_col}", fontsize=13, fontweight="bold"
    )

    if n == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    for i, col in enumerate(plot_cols):
        row_idx = i // cols
        col_idx = i % cols
        ax = axes[row_idx][col_idx]

        ax.scatter(data[col], data[target_col], s=8, alpha=0.4, color="steelblue")
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel(target_col, fontsize=9)
        ax.grid(True, alpha=0.2)

    # 关闭多余的子图
    for i in range(n, rows * cols):
        row_idx = i // cols
        col_idx = i % cols
        axes[row_idx][col_idx].axis("off")

    fig.tight_layout()
    _save_fig(fig, "01_feature_vs_target.png", output_name)


def _plot_sequence_plot(data: DataFrame, output_name: str) -> None:
    """
    HMM 序列: 绘制观测和隐状态的时间序列图

    上下两个子图:
      - 上: 观测序列
      - 下: 隐状态序列 (真实)

    args:
        data(DataFrame): HMM 序列数据
        output_name(str): 输出名称前缀
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    fig.suptitle(f"{output_name} — 时间序列", fontsize=13, fontweight="bold")

    # 观测序列
    axes[0].step(
        data["time"], data["obs"], where="mid", linewidth=0.8, color="steelblue"
    )
    axes[0].set_ylabel("观测符号")
    axes[0].grid(True, alpha=0.2)

    # 隐状态序列
    axes[1].step(
        data["time"], data["state_true"], where="mid", linewidth=0.8, color="coral"
    )
    axes[1].set_ylabel("隐状态")
    axes[1].set_xlabel("时间步")
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    _save_fig(fig, "01_time_series.png", output_name)


def plot_labeled_2d_scatter(
    data: DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    save_dir: Path,
    title: str = "原始散点图",
    filename: str = "data_scatter.png",
) -> None:
    """
    为单个数据集绘制二维带标签散点图

    Args:
        data: 数据集
        x_col: 横轴特征列
        y_col: 纵轴特征列
        label_col: 标签列
        save_dir: 保存目录
        title: 图标题
        filename: 保存文件名
    """
    classes = sorted(data[label_col].unique())
    colors = _get_discrete_colors(len(classes))

    fig, ax = plt.subplots(figsize=(8, 6))
    for color, cls in zip(colors, classes, strict=True):
        subset = data[data[label_col] == cls]
        ax.scatter(
            subset[x_col],
            subset[y_col],
            s=28,
            alpha=0.75,
            color=color,
            edgecolors="black",
            linewidths=0.3,
            label=f"类别 {cls}",
        )

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    _save_single_dataset_fig(fig, save_dir, filename)


def plot_raw_2d_scatter(
    data: DataFrame,
    x_col: str,
    y_col: str,
    save_dir: Path,
    title: str = "原始散点图",
    filename: str = "data_raw_scatter.png",
) -> None:
    """
    为单个数据集绘制不带标签着色的二维原始散点图

    这个函数主要服务于无监督学习场景。
    在聚类任务里，训练阶段本来就看不到真实标签，
    因此先看一张“纯原始数据散点图”会更符合任务语义。

    Args:
        data: 数据集
        x_col: 横轴特征列
        y_col: 纵轴特征列
        save_dir: 保存目录
        title: 图标题
        filename: 保存文件名
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        data[x_col],
        data[y_col],
        s=28,
        alpha=0.7,
        color="#37474F",
        edgecolors="black",
        linewidths=0.3,
    )
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    _save_single_dataset_fig(fig, save_dir, filename)
