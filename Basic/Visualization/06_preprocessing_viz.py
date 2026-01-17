"""
数据预处理可视化
对应文档: ../../docs/visualization/06-preprocessing-viz.md
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def demo_missing_viz():
    """演示缺失值可视化"""
    print("=" * 50)
    print("1. 缺失值可视化")
    print("=" * 50)
    
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
    
    # 随机添加缺失值
    for col in df.columns:
        mask = np.random.rand(len(df)) < 0.1
        df.loc[mask, col] = np.nan
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 缺失值热力图
    sns.heatmap(df.isnull(), cbar=True, ax=axes[0], cmap='YlOrRd')
    axes[0].set_title('Missing Values Heatmap')
    
    # 缺失值统计
    missing_pct = df.isnull().mean() * 100
    missing_pct.plot(kind='bar', ax=axes[1], color='coral')
    axes[1].set_title('Missing Values Percentage')
    axes[1].set_ylabel('%')
    
    plt.tight_layout()
    plt.savefig('../outputs/viz_06_missing.png', dpi=100)
    plt.close()
    print("图表已保存")


def demo_outlier_viz():
    """演示异常值可视化"""
    print("=" * 50)
    print("2. 异常值可视化")
    print("=" * 50)
    
    np.random.seed(42)
    data = np.random.randn(100)
    data = np.append(data, [5, -5, 6, -6])  # 添加异常值
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 箱线图
    axes[0].boxplot(data)
    axes[0].set_title('Box Plot (Outliers)')
    
    # 直方图 + IQR 标记
    axes[1].hist(data, bins=20, edgecolor='black', alpha=0.7)
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    axes[1].axvline(lower, color='red', linestyle='--', label=f'Lower: {lower:.2f}')
    axes[1].axvline(upper, color='red', linestyle='--', label=f'Upper: {upper:.2f}')
    axes[1].set_title('Histogram with IQR Bounds')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('../outputs/viz_06_outlier.png', dpi=100)
    plt.close()
    print("图表已保存")


def demo_transform_viz():
    """演示特征变换可视化"""
    print("=" * 50)
    print("3. 特征变换可视化")
    print("=" * 50)
    
    np.random.seed(42)
    data = np.random.exponential(5, 1000)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 原始分布
    axes[0, 0].hist(data, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Original Distribution')
    
    # 对数变换
    log_data = np.log1p(data)
    axes[0, 1].hist(log_data, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_title('Log Transform')
    
    # 平方根变换
    sqrt_data = np.sqrt(data)
    axes[1, 0].hist(sqrt_data, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].set_title('Square Root Transform')
    
    # 标准化
    std_data = (data - data.mean()) / data.std()
    axes[1, 1].hist(std_data, bins=30, edgecolor='black', alpha=0.7, color='purple')
    axes[1, 1].set_title('Standardization')
    
    plt.tight_layout()
    plt.savefig('../outputs/viz_06_transform.png', dpi=100)
    plt.close()
    print("图表已保存")


def demo_all():
    """运行所有演示"""
    import os
    os.makedirs('../outputs', exist_ok=True)
    
    demo_missing_viz()
    print()
    demo_outlier_viz()
    print()
    demo_transform_viz()


if __name__ == "__main__":
    demo_all()
