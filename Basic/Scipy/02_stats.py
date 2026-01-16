"""
统计分布与描述统计
对应文档: ../../docs/scipy/02-stats.md
"""

import numpy as np
from scipy import stats


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


def demo_all():
    """运行所有演示"""
    demo_distributions()
    print()
    demo_descriptive_stats()
    print()
    demo_percentiles()


if __name__ == "__main__":
    demo_all()
