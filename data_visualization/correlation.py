"""
data_visualization/correlation.py
相关性热力图模块

提供可复用的单数据集相关性热力图函数。
当前模块不再负责“批量生成所有数据集的图片”。
"""

from pathlib import Path

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

    # 动态调整图表大小
    n = len(columns)
    size = max(6, n * 0.6)

    # 特征过多时关闭数字标注
    show_annot = annot and n <= 15
    fmt = ".2f" if show_annot else ""

    fig, ax = plt.subplots(figsize=(size, size * 0.85))
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

    # 特征名过长时旋转
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
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
    size = max(6, n * 0.6)
    show_annot = annot and n <= 15
    fmt = ".2f" if show_annot else ""

    fig, ax = plt.subplots(figsize=(size, size * 0.85))
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
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    _save_single_dataset_fig(fig, save_dir, filename)
