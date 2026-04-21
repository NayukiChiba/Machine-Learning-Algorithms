"""
data_visualization/correlation.py
相关性热力图模块

提供可复用的单数据集相关性热力图函数。
当前模块不再负责“批量生成所有数据集的图片”。
"""

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

from config import DATA_VIS_CORRELATION_DIR as OUTPUT_DIR

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


def _wrap_labels(columns: list[str], width: int = 18) -> list[str]:
    """
    对长坐标标签做自动换行

    这里不截断字段名，而是尽量保留完整含义。
    对于像 Wine 这种长特征名数据集，完整显示更利于阅读。

    Args:
        columns: 原始列名列表
        width: 每行最大字符数

    Returns:
        list[str]: 换行后的标签
    """
    labels = []
    for column in columns:
        normalized = column.replace("_", " ")
        labels.append(fill(normalized, width=width))
    return labels


def _plot_heatmap(
    data: DataFrame,
    columns: list[str],
    output_name: str,
    title: str,
    filename: str,
    annot: bool = True,
) -> None:
    """
    绘制相关性热力图

    使用 seaborn heatmap, coolwarm 配色:
      - 红色: 正相关
      - 蓝色: 负相关
      - 白色: 无相关

    特征数 > 15 时关闭标注 (annot=False), 避免文字过密

    args:
        data(DataFrame): 数据
        columns(list[str]): 列名列表
        output_name(str): 输出名称前缀
        title(str): 图标题
        filename(str): 保存文件名
        annot(bool): 是否显示相关系数数值
    """
    corr = data[columns].corr(method="pearson")

    # 把整体画布和单元格都放大，避免长标签把图挤得发乱。
    n = len(columns)
    max_label_length = max(len(column) for column in columns)
    width = max(10, n * 1.15, max_label_length * 0.55)
    height = max(8, n * 0.95)

    # 特征过多时关闭数字标注
    show_annot = annot and n <= 15
    fmt = ".2f" if show_annot else ""

    fig, ax = plt.subplots(figsize=(width, height))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    sns.heatmap(
        corr,
        annot=show_annot,
        fmt=fmt,
        cmap="coolwarm",
        center=0,
        square=True,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
        vmin=-1,
        vmax=1,
    )

    wrapped_labels = _wrap_labels(columns, width=18)
    ax.set_xticklabels(wrapped_labels, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(wrapped_labels, rotation=0, va="center")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)

    fig.subplots_adjust(left=0.26, bottom=0.27, right=0.95, top=0.92)
    _save_fig(fig, filename, output_name)


def plot_correlation_heatmap(
    data: DataFrame,
    columns: list[str],
    save_dir: Path,
    title: str = "相关性热力图",
    filename: str = "data_correlation.png",
    annot: bool = True,
) -> None:
    """
    为单个数据集绘制相关性热力图

    Args:
        data: 数据集
        columns: 参与相关系数计算的列名列表
        save_dir: 保存目录
        title: 图标题
        filename: 保存文件名
        annot: 是否显示数值标注
    """
    corr = data[columns].corr(method="pearson")
    n = len(columns)
    max_label_length = max(len(column) for column in columns)
    width = max(10, n * 1.15, max_label_length * 0.55)
    height = max(8, n * 0.95)
    show_annot = annot and n <= 15
    fmt = ".2f" if show_annot else ""

    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        corr,
        annot=show_annot,
        fmt=fmt,
        cmap="coolwarm",
        center=0,
        square=True,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
        vmin=-1,
        vmax=1,
    )
    ax.set_title(title)
    wrapped_labels = _wrap_labels(columns, width=18)
    ax.set_xticklabels(wrapped_labels, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(wrapped_labels, rotation=0, va="center")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)

    fig.subplots_adjust(left=0.26, bottom=0.27, right=0.95, top=0.92)
    _save_single_dataset_fig(fig, save_dir, filename)
