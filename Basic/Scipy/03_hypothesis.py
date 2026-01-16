"""
假设检验
对应文档: ../../docs/scipy/03-hypothesis.md
"""

import numpy as np
from scipy import stats


def demo_ttest():
    """演示 t 检验"""
    print("=" * 50)
    print("1. t 检验")
    print("=" * 50)
    
    np.random.seed(42)
    
    # 单样本 t 检验
    print("单样本 t 检验:")
    sample = np.random.normal(105, 15, 30)
    t_stat, p_value = stats.ttest_1samp(sample, 100)
    print(f"  样本均值: {sample.mean():.2f}")
    print(f"  H0: μ = 100")
    print(f"  t统计量: {t_stat:.4f}")
    print(f"  p值: {p_value:.4f}")
    print()
    
    # 独立样本 t 检验
    print("独立样本 t 检验:")
    group1 = np.random.normal(100, 15, 30)
    group2 = np.random.normal(110, 15, 30)
    t_stat, p_value = stats.ttest_ind(group1, group2)
    print(f"  组1均值: {group1.mean():.2f}")
    print(f"  组2均值: {group2.mean():.2f}")
    print(f"  t统计量: {t_stat:.4f}")
    print(f"  p值: {p_value:.4f}")
    print()
    
    # 配对 t 检验
    print("配对 t 检验:")
    before = np.random.normal(100, 10, 20)
    after = before + np.random.normal(5, 3, 20)
    t_stat, p_value = stats.ttest_rel(before, after)
    print(f"  前测均值: {before.mean():.2f}")
    print(f"  后测均值: {after.mean():.2f}")
    print(f"  t统计量: {t_stat:.4f}")
    print(f"  p值: {p_value:.4f}")


def demo_chi2():
    """演示卡方检验"""
    print("=" * 50)
    print("2. 卡方检验")
    print("=" * 50)
    
    # 卡方拟合优度检验
    print("卡方拟合优度检验:")
    observed = np.array([45, 35, 20])  # 观察频数
    expected = np.array([40, 40, 20])  # 期望频数
    chi2, p_value = stats.chisquare(observed, f_exp=expected)
    print(f"  观察值: {observed}")
    print(f"  期望值: {expected}")
    print(f"  χ²统计量: {chi2:.4f}")
    print(f"  p值: {p_value:.4f}")
    print()
    
    # 卡方独立性检验
    print("卡方独立性检验 (列联表):")
    contingency_table = np.array([[30, 20], [25, 25]])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"  列联表:\n{contingency_table}")
    print(f"  χ²统计量: {chi2:.4f}")
    print(f"  p值: {p_value:.4f}")
    print(f"  自由度: {dof}")


def demo_anova():
    """演示方差分析"""
    print("=" * 50)
    print("3. 方差分析 (ANOVA)")
    print("=" * 50)
    
    np.random.seed(42)
    
    # 单因素方差分析
    group1 = np.random.normal(100, 10, 20)
    group2 = np.random.normal(105, 10, 20)
    group3 = np.random.normal(110, 10, 20)
    
    f_stat, p_value = stats.f_oneway(group1, group2, group3)
    
    print("单因素方差分析:")
    print(f"  组1均值: {group1.mean():.2f}")
    print(f"  组2均值: {group2.mean():.2f}")
    print(f"  组3均值: {group3.mean():.2f}")
    print(f"  F统计量: {f_stat:.4f}")
    print(f"  p值: {p_value:.4f}")


def demo_nonparametric():
    """演示非参数检验"""
    print("=" * 50)
    print("4. 非参数检验")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Mann-Whitney U 检验
    print("Mann-Whitney U 检验:")
    group1 = np.random.normal(100, 15, 20)
    group2 = np.random.normal(110, 15, 20)
    stat, p_value = stats.mannwhitneyu(group1, group2)
    print(f"  U统计量: {stat:.4f}")
    print(f"  p值: {p_value:.4f}")
    print()
    
    # Wilcoxon 符号秩检验
    print("Wilcoxon 符号秩检验:")
    before = np.random.normal(100, 10, 20)
    after = before + np.random.normal(5, 3, 20)
    stat, p_value = stats.wilcoxon(before, after)
    print(f"  统计量: {stat:.4f}")
    print(f"  p值: {p_value:.4f}")


def demo_all():
    """运行所有演示"""
    demo_ttest()
    print()
    demo_chi2()
    print()
    demo_anova()
    print()
    demo_nonparametric()


if __name__ == "__main__":
    demo_all()
