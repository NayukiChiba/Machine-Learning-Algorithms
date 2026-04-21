"""
data_visualization/distribution.py
分布图模块

提供可复用的单数据集分布可视化函数。
当前模块不再负责“批量生成所有数据集的图片”，
而是只提供给 pipeline 或其它调用方按需复用。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

from config import DATA_VIS_DISTRIBUTION_DIR as OUTPUT_DIR

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


# --- 通用绘图工具 ---


def _save_fig(fig: plt.Figure, filename: str, output_name: str) -> None:
    """
    保存图表到当前模块目录

    args:
        fig(Figure): matplotlib 图表对象
        filename(str): 文件名 (不含路径)
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

    这套保存逻辑服务于 pipeline 场景。
    和当前文件里那组“批量生成所有数据集图”的逻辑不同，
    这里不再使用统一的 data_visualization 输出目录，而是由调用方显式传入保存目录。

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


def _plot_histograms(
    data: DataFrame, feature_cols: list[str], output_name: str
) -> None:
    """
    为所有连续特征绘制直方图 + KDE 密度曲线

    每个子图包含:
      - 直方图 (30 bins, 半透明填充)
      - KDE 密度曲线 (红色叠加)

    args:
        data(DataFrame): 数据
        feature_cols(list[str]): 特征列名列表
        output_name(str): 输出名称前缀
    """
    n = len(feature_cols)
    if n == 0:
        return

    # 计算子图行列数
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    fig.suptitle(
        f"{output_name} — 特征分布 (直方图 + KDE)", fontsize=13, fontweight="bold"
    )

    # 确保 axes 总是二维数组
    if n == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    for i, col in enumerate(feature_cols):
        row_idx = i // cols
        col_idx = i % cols
        ax = axes[row_idx][col_idx]

        # 直方图 + KDE
        ax.hist(
            data[col],
            bins=30,
            color="steelblue",
            alpha=0.6,
            edgecolor="white",
            density=True,
        )
        data[col].plot.kde(ax=ax, color="coral", linewidth=1.5)

        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.grid(True, alpha=0.2)

    # 关闭多余的子图
    for i in range(n, rows * cols):
        row_idx = i // cols
        col_idx = i % cols
        axes[row_idx][col_idx].axis("off")

    fig.tight_layout()
    _save_fig(fig, "01_histogram_kde.png", output_name)


def _plot_boxplots(data: DataFrame, feature_cols: list[str], output_name: str) -> None:
    """
    为所有连续特征绘制箱线图

    箱线图可以直观展示:
      - 中位数 (箱中线)
      - 四分位距 IQR (箱体)
      - 异常值 (离群点)

    args:
        data(DataFrame): 数据
        feature_cols(list[str]): 特征列名列表
        output_name(str): 输出名称前缀
    """
    if len(feature_cols) == 0:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(feature_cols) * 0.8), 5))
    fig.suptitle(f"{output_name} — 箱线图", fontsize=13, fontweight="bold")

    # 数据可能量纲差异大，用标准化后的数据画箱线图
    plot_data = data[feature_cols]
    plot_data_norm = (plot_data - plot_data.mean()) / plot_data.std()

    ax.boxplot(
        [plot_data_norm[col].dropna().values for col in feature_cols],
        tick_labels=feature_cols,
        patch_artist=True,
        boxprops={"facecolor": "lightblue", "alpha": 0.7},
        medianprops={"color": "coral", "linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
    )

    ax.set_ylabel("标准化值")
    ax.grid(True, axis="y", alpha=0.2)

    # 特征名过长时旋转
    if any(len(col) > 8 for col in feature_cols):
        ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    _save_fig(fig, "02_boxplot.png", output_name)


def _plot_target_distribution(
    data: DataFrame, target_col: str, output_name: str, is_classification: bool = True
) -> None:
    """
    绘制目标变量分布图

    分类任务: 柱状图 (各类别样本数)
    回归任务: 直方图 + KDE (连续值分布)

    args:
        data(DataFrame): 数据
        target_col(str): 目标变量列名
        output_name(str): 输出名称前缀
        is_classification(bool): 是否为分类任务
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    if is_classification:
        # 分类: 柱状图
        counts = data[target_col].value_counts().sort_index()
        colors = sns.color_palette("Set2", len(counts))
        ax.bar(counts.index.astype(str), counts.values, color=colors)
        ax.set_xlabel("类别")
        ax.set_ylabel("样本数")
        fig.suptitle(f"{output_name} — 类别分布", fontsize=13, fontweight="bold")

        # 在柱子上方显示数量
        for i, (idx, val) in enumerate(counts.items()):
            ax.text(i, val + len(data) * 0.01, str(val), ha="center", fontsize=9)
    else:
        # 回归: 直方图 + KDE
        ax.hist(
            data[target_col],
            bins=40,
            color="steelblue",
            alpha=0.6,
            edgecolor="white",
            density=True,
        )
        data[target_col].plot.kde(ax=ax, color="coral", linewidth=1.5)
        ax.set_xlabel(target_col)
        ax.set_ylabel("密度")
        fig.suptitle(f"{output_name} — 目标变量分布", fontsize=13, fontweight="bold")

    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    _save_fig(fig, "03_target_distribution.png", output_name)


def plot_class_distribution(
    data: DataFrame,
    target_col: str,
    save_dir: Path,
    title: str = "类别分布",
    filename: str = "data_class_distribution.png",
) -> None:
    """
    为单个数据集绘制类别分布图

    这是给 pipeline 直接调用的公共函数。
    它和 `_plot_target_distribution(...)` 的区别在于：
    1. 它只处理“当前一个数据集”；
    2. 输出目录由调用方控制；
    3. 文件名不再带数据集前缀，便于直接放到 outputs/<model_name>/ 中。

    Args:
        data: 数据集
        target_col: 标签列名
        save_dir: 保存目录
        title: 图标题
        filename: 保存文件名
    """
    counts = data[target_col].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = sns.color_palette("Set2", len(counts))
    ax.bar(counts.index.astype(str), counts.values, color=colors)
    ax.set_title(title)
    ax.set_xlabel("类别")
    ax.set_ylabel("样本数")
    ax.grid(True, axis="y", alpha=0.25)

    for idx, value in enumerate(counts.values):
        ax.text(idx, value + max(counts.values) * 0.02, str(value), ha="center")

    fig.tight_layout()
    _save_single_dataset_fig(fig, save_dir, filename)
