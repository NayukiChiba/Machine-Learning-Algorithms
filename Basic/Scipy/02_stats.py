"""
统计分布与描述统计
对应文档: ../../docs/scipy/02-stats.md
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def demo_distributions():
    """演示概率分布"""
    print("=" * 50)
    print("1. 概率分布")
    print("=" * 50)

    # 正态分布
    print("正态分布 (μ=0, σ=1):")
    norm = stats.norm(loc=0, scale=1)
    print(f"  PDF at x=0: {norm.pdf(0):.4f}")
    print(f"  CDF at x=0: {norm.cdf(0):.4f}")
    print(f"  PPF at p=0.95: {norm.ppf(0.95):.4f}")
    print(f"  随机样本: {norm.rvs(size=5)}")
    print()

    # 二项分布
    print("二项分布 (n=10, p=0.5):")
    binom = stats.binom(n=10, p=0.5)
    print(f"  P(X=5): {binom.pmf(5):.4f}")
    print(f"  P(X≤5): {binom.cdf(5):.4f}")
    print(f"  期望值: {binom.mean()}")
    print(f"  方差: {binom.var()}")
    print()

    # 泊松分布
    print("泊松分布 (λ=3):")
    poisson = stats.poisson(mu=3)
    print(f"  P(X=3): {poisson.pmf(3):.4f}")
    print(f"  期望值: {poisson.mean()}")

    # === 可视化 ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 正态分布 PDF
    x = np.linspace(-4, 4, 100)
    ax1 = axes[0, 0]
    ax1.plot(x, norm.pdf(x), "b-", lw=2, label="PDF")
    ax1.fill_between(x, norm.pdf(x), alpha=0.3)
    ax1.axvline(0, color="r", linestyle="--", label=f"均值 μ=0")
    ax1.set_title("正态分布 N(0, 1) - 概率密度函数")
    ax1.set_xlabel("x")
    ax1.set_ylabel("密度")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 正态分布 CDF
    ax2 = axes[0, 1]
    ax2.plot(x, norm.cdf(x), "g-", lw=2, label="CDF")
    ax2.axhline(0.5, color="r", linestyle="--", alpha=0.5)
    ax2.axvline(0, color="r", linestyle="--", alpha=0.5)
    ax2.set_title("正态分布 N(0, 1) - 累积分布函数")
    ax2.set_xlabel("x")
    ax2.set_ylabel("概率")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 二项分布 PMF
    ax3 = axes[1, 0]
    k = np.arange(0, 11)
    ax3.bar(k, binom.pmf(k), color="steelblue", alpha=0.7, edgecolor="black")
    ax3.axvline(
        binom.mean(), color="r", linestyle="--", lw=2, label=f"期望值 = {binom.mean()}"
    )
    ax3.set_title("二项分布 B(10, 0.5) - 概率质量函数")
    ax3.set_xlabel("k (成功次数)")
    ax3.set_ylabel("概率 P(X=k)")
    ax3.set_xticks(k)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    # 泊松分布 PMF
    ax4 = axes[1, 1]
    k = np.arange(0, 12)
    ax4.bar(k, poisson.pmf(k), color="coral", alpha=0.7, edgecolor="black")
    ax4.axvline(
        poisson.mean(),
        color="r",
        linestyle="--",
        lw=2,
        label=f"期望值 λ = {poisson.mean()}",
    )
    ax4.set_title("泊松分布 Poisson(λ=3) - 概率质量函数")
    ax4.set_xlabel("k (事件发生次数)")
    ax4.set_ylabel("概率 P(X=k)")
    ax4.set_xticks(k)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("outputs/scipy/02_distributions.png", dpi=150, bbox_inches="tight")


def demo_descriptive_stats():
    """演示描述性统计"""
    print("=" * 50)
    print("2. 描述性统计")
    print("=" * 50)

    np.random.seed(42)
    data = np.random.normal(100, 15, 100)

    print(f"数据样本: {data[:5]}")
    print()

    # 集中趋势
    print("集中趋势:")
    print(f"  均值: {np.mean(data):.2f}")
    print(f"  中位数: {np.median(data):.2f}")
    print(f"  众数: {stats.mode(data.astype(int), keepdims=True)[0][0]}")
    print()

    # 离散程度
    print("离散程度:")
    print(f"  方差: {np.var(data):.2f}")
    print(f"  标准差: {np.std(data):.2f}")
    print(f"  变异系数: {stats.variation(data):.4f}")
    print()

    # 分布形态
    print("分布形态:")
    print(f"  偏度: {stats.skew(data):.4f}")
    print(f"  峰度: {stats.kurtosis(data):.4f}")

    # === 可视化 ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 直方图 + 正态拟合
    ax1 = axes[0]
    n, bins, patches = ax1.hist(
        data,
        bins=15,
        density=True,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        label="数据分布",
    )

    # 拟合正态分布
    mu, std = stats.norm.fit(data)
    x = np.linspace(data.min(), data.max(), 100)
    ax1.plot(
        x,
        stats.norm.pdf(x, mu, std),
        "r-",
        lw=2,
        label=f"正态拟合 N({mu:.1f}, {std:.1f}^2)",
    )

    # 标记均值和中位数
    ax1.axvline(
        np.mean(data),
        color="green",
        linestyle="--",
        lw=2,
        label=f"均值 = {np.mean(data):.1f}",
    )
    ax1.axvline(
        np.median(data),
        color="orange",
        linestyle=":",
        lw=2,
        label=f"中位数 = {np.median(data):.1f}",
    )

    ax1.set_title("数据分布直方图与正态拟合")
    ax1.set_xlabel("数值")
    ax1.set_ylabel("密度")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 箱线图
    ax2 = axes[1]
    bp = ax2.boxplot(data, vert=True, patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    bp["boxes"][0].set_alpha(0.7)

    # 添加统计信息
    q1, median, q3 = np.percentile(data, [25, 50, 75])
    stats_text = (
        f"Q1 = {q1:.1f}\n中位数 = {median:.1f}\nQ3 = {q3:.1f}\nIQR = {q3 - q1:.1f}"
    )
    ax2.text(
        1.3,
        np.mean(data),
        stats_text,
        fontsize=10,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax2.set_title("箱线图 (Box Plot)")
    ax2.set_ylabel("数值")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("outputs/scipy/02_descriptive.png", dpi=150, bbox_inches="tight")


def demo_percentiles():
    """演示百分位数"""
    print("=" * 50)
    print("3. 百分位数")
    print("=" * 50)

    np.random.seed(42)
    data = np.random.normal(100, 15, 100)

    print("百分位数:")
    for p in [25, 50, 75, 90, 95]:
        print(f"  P{p}: {np.percentile(data, p):.2f}")
    print()

    # 四分位距
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    print(f"四分位距 (IQR): {iqr:.2f}")

    # === 可视化 ===
    fig, ax = plt.subplots(figsize=(10, 6))

    # 排序数据
    sorted_data = np.sort(data)
    percentiles = np.arange(1, 101)

    ax.plot(sorted_data, percentiles, "b-", lw=2)
    ax.fill_betweenx(percentiles, sorted_data.min(), sorted_data, alpha=0.3)

    # 标记关键百分位数
    key_p = [25, 50, 75, 90, 95]
    for p in key_p:
        val = np.percentile(data, p)
        ax.axhline(p, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(val, color="gray", linestyle=":", alpha=0.5)
        ax.plot(val, p, "ro", markersize=8)
        ax.annotate(
            f"P{p}={val:.1f}",
            (val, p),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=9,
        )

    ax.set_title("百分位数分布曲线")
    ax.set_xlabel("数值")
    ax.set_ylabel("百分位数(%)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/scipy/02_percentiles.png", dpi=150, bbox_inches="tight")


def demo_all():
    """运行所有演示"""
    import os

    os.makedirs("outputs/scipy", exist_ok=True)

    demo_distributions()
    print()
    demo_descriptive_stats()
    print()
    demo_percentiles()


if __name__ == "__main__":
    demo_all()
