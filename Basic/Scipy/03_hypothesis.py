"""
假设检验
对应文档: ../../docs/scipy/03-hypothesis.md
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


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
    t_stat_ind, p_value_ind = stats.ttest_ind(group1, group2)
    print(f"  组1均值: {group1.mean():.2f}")
    print(f"  组2均值: {group2.mean():.2f}")
    print(f"  t统计量: {t_stat_ind:.4f}")
    print(f"  p值: {p_value_ind:.4f}")
    print()
    
    # 配对 t 检验
    print("配对 t 检验:")
    before = np.random.normal(100, 10, 20)
    after = before + np.random.normal(5, 3, 20)
    t_stat_paired, p_value_paired = stats.ttest_rel(before, after)
    print(f"  前测均值: {before.mean():.2f}")
    print(f"  后测均值: {after.mean():.2f}")
    print(f"  t统计量: {t_stat_paired:.4f}")
    print(f"  p值: {p_value_paired:.4f}")
    
    # === 可视化 ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 单样本 t 检验 - t 分布
    ax1 = axes[0, 0]
    df = len(sample) - 1
    x = np.linspace(-5, 5, 200)
    t_dist = stats.t(df=df)
    
    ax1.plot(x, t_dist.pdf(x), 'b-', lw=2, label=f't分布 (df={df})')
    ax1.fill_between(x, t_dist.pdf(x), alpha=0.2)
    
    # 标记拒绝域 (α=0.05, 双尾)
    alpha = 0.05
    t_crit = t_dist.ppf(1 - alpha/2)
    ax1.fill_between(x[x >= t_crit], t_dist.pdf(x[x >= t_crit]), color='red', alpha=0.4, label=f'拒绝域 (α={alpha})')
    ax1.fill_between(x[x <= -t_crit], t_dist.pdf(x[x <= -t_crit]), color='red', alpha=0.4)
    
    # 标记 t 统计量
    ax1.axvline(t_stat, color='green', linestyle='--', lw=2, label=f't统计量 = {t_stat:.2f}')
    ax1.axvline(-t_crit, color='red', linestyle=':', alpha=0.7)
    ax1.axvline(t_crit, color='red', linestyle=':', alpha=0.7)
    
    ax1.set_title('单样本 t 检验: t 分布与检验统计量')
    ax1.set_xlabel('t 值')
    ax1.set_ylabel('密度')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 独立样本 t 检验 - 两组数据对比
    ax2 = axes[0, 1]
    positions = [1, 2]
    bp = ax2.boxplot([group1, group2], positions=positions, widths=0.6, patch_artist=True)
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.scatter(np.ones(len(group1)) + np.random.normal(0, 0.05, len(group1)), 
                group1, alpha=0.5, s=30, color='blue')
    ax2.scatter(2*np.ones(len(group2)) + np.random.normal(0, 0.05, len(group2)), 
                group2, alpha=0.5, s=30, color='red')
    
    ax2.set_xticks(positions)
    ax2.set_xticklabels(['组1 (μ=100)', '组2 (μ=110)'])
    ax2.set_title(f'独立样本 t 检验\nt={t_stat_ind:.2f}, p={p_value_ind:.4f}')
    ax2.set_ylabel('数值')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 配对 t 检验 - 前后对比
    ax3 = axes[1, 0]
    for i in range(len(before)):
        ax3.plot([1, 2], [before[i], after[i]], 'gray', alpha=0.5, linewidth=0.8)
    ax3.scatter(np.ones(len(before)), before, c='blue', s=50, label='前测', zorder=5)
    ax3.scatter(2*np.ones(len(after)), after, c='red', s=50, label='后测', zorder=5)
    
    ax3.set_xticks([1, 2])
    ax3.set_xticklabels(['前测', '后测'])
    ax3.set_title(f'配对 t 检验: 前后测试对比\nt={t_stat_paired:.2f}, p={p_value_paired:.4f}')
    ax3.set_ylabel('数值')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # p 值解释
    ax4 = axes[1, 1]
    tests = ['单样本 t', '独立样本 t', '配对 t']
    p_values = [p_value, p_value_ind, p_value_paired]
    colors = ['green' if p < 0.05 else 'gray' for p in p_values]
    
    bars = ax4.barh(tests, p_values, color=colors, alpha=0.7, edgecolor='black')
    ax4.axvline(0.05, color='red', linestyle='--', lw=2, label='α = 0.05')
    ax4.set_xlabel('p 值')
    ax4.set_title('假设检验 p 值对比')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='x')
    
    for bar, p in zip(bars, p_values):
        ax4.text(p + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{p:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('outputs/scipy/03_ttest.png', dpi=150, bbox_inches='tight')


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
    print(f"  χ^2统计量: {chi2:.4f}")
    print(f"  p值: {p_value:.4f}")
    print()
    
    # 卡方独立性检验
    print("卡方独立性检验 (列联表):")
    contingency_table = np.array([[30, 20], [25, 25]])
    chi2_ind, p_value_ind, dof, expected_ind = stats.chi2_contingency(contingency_table)
    print(f"  列联表:\n{contingency_table}")
    print(f"  χ^2统计量: {chi2_ind:.4f}")
    print(f"  p值: {p_value_ind:.4f}")
    print(f"  自由度: {dof}")
    
    # === 可视化 ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 拟合优度 - 观察 vs 期望
    ax1 = axes[0]
    categories = ['类别A', '类别B', '类别C']
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, observed, width, label='观察频数', color='steelblue', alpha=0.7)
    ax1.bar(x + width/2, expected, width, label='期望频数', color='coral', alpha=0.7)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.set_ylabel('频数')
    ax1.set_title(f'卡方拟合优度检验\nχ^2={chi2:.2f}, p={p_value:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 列联表热力图
    ax2 = axes[1]
    im = ax2.imshow(contingency_table, cmap='Blues', aspect='auto')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['变量B=0', '变量B=1'])
    ax2.set_yticklabels(['变量A=0', '变量A=1'])
    
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, contingency_table[i, j], ha='center', va='center', 
                    fontsize=16, fontweight='bold')
    
    ax2.set_title(f'列联表\nχ^2={chi2_ind:.2f}, p={p_value_ind:.4f}')
    plt.colorbar(im, ax=ax2, label='频数')
    
    # 卡方分布
    ax3 = axes[2]
    x = np.linspace(0, 15, 200)
    for df in [1, 2, 3, 5]:
        ax3.plot(x, stats.chi2.pdf(x, df), lw=2, label=f'df={df}')
    
    ax3.axvline(chi2, color='red', linestyle='--', lw=2, label=f'χ^2统计量={chi2:.2f}')
    ax3.set_title('卡方分布 (不同自由度)')
    ax3.set_xlabel('χ^2 值')
    ax3.set_ylabel('密度')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 15)
    
    plt.tight_layout()
    plt.savefig('outputs/scipy/03_chi2.png', dpi=150, bbox_inches='tight')


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
    
    # === 可视化 ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 箱线图
    ax1 = axes[0]
    data_groups = [group1, group2, group3]
    bp = ax1.boxplot(data_groups, patch_artist=True, tick_labels=['组1\n(μ=100)', '组2\n(μ=105)', '组3\n(μ=110)'])
    
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 添加散点
    for i, group in enumerate(data_groups, 1):
        x = np.random.normal(i, 0.04, len(group))
        ax1.scatter(x, group, alpha=0.5, s=30)
    
    # 添加均值连线
    means = [g.mean() for g in data_groups]
    ax1.plot([1, 2, 3], means, 'ko-', markersize=8, linewidth=2, label='组均值')
    
    ax1.set_title(f'单因素方差分析 (ANOVA)\nF={f_stat:.2f}, p={p_value:.4f}')
    ax1.set_ylabel('数值')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # F 分布
    ax2 = axes[1]
    df1 = 2  # 组间自由度 = k - 1
    df2 = 57  # 组内自由度 = N - k
    x = np.linspace(0, 8, 200)
    
    ax2.plot(x, stats.f.pdf(x, df1, df2), 'b-', lw=2, label=f'F分布 (df1={df1}, df2={df2})')
    ax2.fill_between(x, stats.f.pdf(x, df1, df2), alpha=0.2)
    
    # 拒绝域
    f_crit = stats.f.ppf(0.95, df1, df2)
    ax2.fill_between(x[x >= f_crit], stats.f.pdf(x[x >= f_crit], df1, df2), 
                     color='red', alpha=0.4, label=f'拒绝域 (α=0.05)')
    
    ax2.axvline(f_stat, color='green', linestyle='--', lw=2, label=f'F统计量 = {f_stat:.2f}')
    
    ax2.set_title('F 分布与检验统计量')
    ax2.set_xlabel('F 值')
    ax2.set_ylabel('密度')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 8)
    
    plt.tight_layout()
    plt.savefig('outputs/scipy/03_anova.png', dpi=150, bbox_inches='tight')


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
    stat_w, p_value_w = stats.wilcoxon(before, after)
    print(f"  统计量: {stat_w:.4f}")
    print(f"  p值: {p_value_w:.4f}")
    
    # === 可视化 ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mann-Whitney U
    ax1 = axes[0]
    ax1.hist(group1, bins=10, alpha=0.6, label='组1', color='blue', edgecolor='black')
    ax1.hist(group2, bins=10, alpha=0.6, label='组2', color='red', edgecolor='black')
    ax1.axvline(np.median(group1), color='blue', linestyle='--', lw=2)
    ax1.axvline(np.median(group2), color='red', linestyle='--', lw=2)
    ax1.set_title(f'Mann-Whitney U 检验\nU={stat:.0f}, p={p_value:.4f}')
    ax1.set_xlabel('数值')
    ax1.set_ylabel('频数')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Wilcoxon
    ax2 = axes[1]
    diff = after - before
    colors = ['green' if d > 0 else 'red' for d in diff]
    ax2.bar(range(len(diff)), diff, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(0, color='black', linestyle='-', lw=1)
    ax2.set_title(f'Wilcoxon 符号秩检验\n差值分布 (后测 - 前测)\nW={stat_w:.0f}, p={p_value_w:.4f}')
    ax2.set_xlabel('样本编号')
    ax2.set_ylabel('差值')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('outputs/scipy/03_nonparam.png', dpi=150, bbox_inches='tight')


def demo_all():
    """运行所有演示"""
    import os
    os.makedirs('outputs/scipy', exist_ok=True)
    
    demo_ttest()
    print()
    demo_chi2()
    print()
    demo_anova()
    print()
    demo_nonparametric()


if __name__ == "__main__":
    demo_all()
