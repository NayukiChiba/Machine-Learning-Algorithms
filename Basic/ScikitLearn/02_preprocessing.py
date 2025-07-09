"""
Scikit-learn 数据预处理
对应文档: ../docs/02_preprocessing.md

使用方式：
    from code.02_preprocessing import *
    demo_scalers()
    demo_encoders()
"""

import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def demo_scalers():
    """演示各种缩放器的用法和区别"""
    print("=" * 50)
    print("1. 数据缩放器对比")
    print("=" * 50)
    
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
    
    # 创建含异常值的数据
    np.random.seed(42)
    X = np.random.randn(100, 2) * 10 + 50
    X[0] = [200, 200]  # 异常值
    
    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler(),
        'MaxAbsScaler': MaxAbsScaler()
    }
    
    # === 可视化: 缩放器对比 ===
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始数据
    ax_orig = axes[0, 0]
    ax_orig.scatter(X[:, 0], X[:, 1], c='#45B7D1', alpha=0.7, s=40, edgecolors='white')
    ax_orig.scatter(X[0, 0], X[0, 1], c='red', s=150, marker='*', label='异常值', edgecolors='black')
    ax_orig.set_title('原始数据')
    ax_orig.set_xlabel('特征 1')
    ax_orig.set_ylabel('特征 2')
    ax_orig.legend()
    ax_orig.grid(True, alpha=0.3)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    for idx, ((name, scaler), color) in enumerate(zip(scalers.items(), colors)):
        X_scaled = scaler.fit_transform(X)
        ax = axes.flatten()[idx + 1]
        ax.scatter(X_scaled[1:, 0], X_scaled[1:, 1], c=color, alpha=0.7, s=40, edgecolors='white')
        ax.scatter(X_scaled[0, 0], X_scaled[0, 1], c='red', s=150, marker='*', edgecolors='black')
        ax.set_title(f'{name}\n范围: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]')
        ax.set_xlabel('特征 1')
        ax.set_ylabel('特征 2')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        print(f"{name}:")
        print(f"  范围: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
        print(f"  均值: {X_scaled.mean():.4f}, 标准差: {X_scaled.std():.4f}")
    
    # 最后一个子图: 公式说明
    ax_formula = axes[1, 2]
    ax_formula.axis('off')
    formula_text = """
缩放公式说明:

StandardScaler:
  z = (x - mean) / std

MinMaxScaler:
  x' = (x - min) / (max - min)

RobustScaler:
  x' = (x - median) / IQR

MaxAbsScaler:
  x' = x / max(|x|)
"""
    ax_formula.text(0.1, 0.5, formula_text, fontsize=12, family='monospace',
                   verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('outputs/sklearn/02_scalers.png', dpi=150, bbox_inches='tight')
    plt.close()


def demo_standard_scaler():
    """StandardScaler 详细用法"""
    print("=" * 50)
    print("2. StandardScaler 详解")
    print("=" * 50)
    
    from sklearn.preprocessing import StandardScaler
    
    X = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])
    
    scaler = StandardScaler(
        copy=True,        # 是否复制数据
        with_mean=True,   # 是否中心化
        with_std=True     # 是否缩放
    )
    
    X_scaled = scaler.fit_transform(X)
    
    print(f"原始数据:\n{X}")
    print(f"\n缩放后:\n{X_scaled.round(3)}")
    print(f"\n学到的均值 (mean_): {scaler.mean_}")
    print(f"学到的标准差 (scale_): {scaler.scale_}")
    
    # 逆变换
    X_back = scaler.inverse_transform(X_scaled)
    print(f"\n逆变换后:\n{X_back}")


def demo_minmax_scaler():
    """MinMaxScaler 详细用法"""
    print("=" * 50)
    print("3. MinMaxScaler 详解")
    print("=" * 50)
    
    from sklearn.preprocessing import MinMaxScaler
    
    X = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])
    
    # 缩放到 [0, 1]
    scaler1 = MinMaxScaler(feature_range=(0, 1))
    X_scaled1 = scaler1.fit_transform(X)
    print(f"feature_range=(0,1):\n{X_scaled1.round(3)}")
    
    # 缩放到 [-1, 1]
    scaler2 = MinMaxScaler(feature_range=(-1, 1))
    X_scaled2 = scaler2.fit_transform(X)
    print(f"\nfeature_range=(-1,1):\n{X_scaled2.round(3)}")


def demo_power_transformer():
    """PowerTransformer 幂变换"""
    print("=" * 50)
    print("4. PowerTransformer 幂变换")
    print("=" * 50)
    
    from sklearn.preprocessing import PowerTransformer
    from scipy import stats
    
    # 创建偏态数据
    np.random.seed(42)
    X_skewed = np.random.exponential(scale=2, size=(500, 1))
    
    # Yeo-Johnson（支持负数）
    pt_yj = PowerTransformer(method='yeo-johnson')
    X_yj = pt_yj.fit_transform(X_skewed)
    
    # Box-Cox（只支持正数）
    pt_bc = PowerTransformer(method='box-cox')
    X_bc = pt_bc.fit_transform(X_skewed)
    
    print(f"原始数据: 均值={X_skewed.mean():.2f}, 标准差={X_skewed.std():.2f}")
    print(f"Yeo-Johnson后: 均值={X_yj.mean():.4f}, 标准差={X_yj.std():.4f}")
    print(f"Yeo-Johnson lambda: {pt_yj.lambdas_}")
    print(f"Box-Cox lambda: {pt_bc.lambdas_}")
    
    # === 可视化: 幂变换效果 ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    ax1 = axes[0]
    ax1.hist(X_skewed, bins=30, color='#FF6B6B', edgecolor='black', alpha=0.7)
    ax1.axvline(X_skewed.mean(), color='black', linestyle='--', lw=2, label=f'均值={X_skewed.mean():.2f}')
    ax1.set_title(f'原始数据 (偏度={stats.skew(X_skewed)[0]:.2f})')
    ax1.set_xlabel('值')
    ax1.set_ylabel('频数')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.hist(X_yj, bins=30, color='#4ECDC4', edgecolor='black', alpha=0.7)
    ax2.axvline(X_yj.mean(), color='black', linestyle='--', lw=2, label=f'均值={X_yj.mean():.2f}')
    ax2.set_title(f'Yeo-Johnson 变换 (偏度={stats.skew(X_yj)[0]:.2f})')
    ax2.set_xlabel('值')
    ax2.set_ylabel('频数')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    ax3.hist(X_bc, bins=30, color='#45B7D1', edgecolor='black', alpha=0.7)
    ax3.axvline(X_bc.mean(), color='black', linestyle='--', lw=2, label=f'均值={X_bc.mean():.2f}')
    ax3.set_title(f'Box-Cox 变换 (偏度={stats.skew(X_bc)[0]:.2f})')
    ax3.set_xlabel('值')
    ax3.set_ylabel('频数')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/sklearn/02_power_transform.png', dpi=150, bbox_inches='tight')
    plt.close()


def demo_encoders():
    """类别编码器"""
    print("=" * 50)
    print("5. 类别编码")
    print("=" * 50)
    
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
    
    colors = np.array([['红'], ['绿'], ['蓝'], ['红'], ['绿']])
    
    # LabelEncoder - 单列
    le = LabelEncoder()
    colors_le = le.fit_transform(colors.ravel())
    print(f"LabelEncoder:")
    print(f"  输入: {colors.ravel()}")
    print(f"  编码: {colors_le}")
    print(f"  类别: {le.classes_}")
    
    # OneHotEncoder
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    colors_ohe = ohe.fit_transform(colors)
    print(f"\nOneHotEncoder:")
    print(f"  特征名: {ohe.get_feature_names_out()}")
    print(f"  编码:\n{colors_ohe}")
    
    # === 可视化: 编码对比 ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # LabelEncoder 可视化
    ax1 = axes[0]
    categories = ['红', '绿', '蓝']
    encoded = [0, 1, 2]
    bars = ax1.barh(categories, encoded, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], edgecolor='black')
    ax1.set_xlabel('编码值')
    ax1.set_title('LabelEncoder 编码')
    for bar, val in zip(bars, encoded):
        ax1.annotate(f'{val}', xy=(val + 0.1, bar.get_y() + bar.get_height()/2),
                    va='center', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 3)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # OneHotEncoder 可视化
    ax2 = axes[1]
    onehot_data = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
    im = ax2.imshow(onehot_data, cmap='Blues', aspect='auto')
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['x0_红', 'x0_绿', 'x0_蓝'])
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['红', '绿', '蓝'])
    ax2.set_title('OneHotEncoder 编码')
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, onehot_data[i, j], ha='center', va='center', 
                    color='white' if onehot_data[i, j] == 1 else 'black', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/sklearn/02_encoding.png', dpi=150, bbox_inches='tight')
    plt.close()


def demo_imputers():
    """缺失值处理"""
    print("=" * 50)
    print("6. 缺失值处理")
    print("=" * 50)
    
    from sklearn.impute import SimpleImputer, KNNImputer
    
    X = np.array([
        [1, 2, np.nan],
        [3, np.nan, 6],
        [7, 8, 9],
        [np.nan, 5, 3]
    ])
    
    print(f"原始数据:\n{X}")
    
    # 均值填充
    imp_mean = SimpleImputer(strategy='mean')
    print(f"\nmean填充:\n{imp_mean.fit_transform(X).round(2)}")
    
    # 中位数填充
    imp_median = SimpleImputer(strategy='median')
    print(f"\nmedian填充:\n{imp_median.fit_transform(X)}")
    
    # 常数填充
    imp_const = SimpleImputer(strategy='constant', fill_value=0)
    print(f"\nconstant=0填充:\n{imp_const.fit_transform(X)}")
    
    # KNN填充
    imp_knn = KNNImputer(n_neighbors=2)
    print(f"\nKNN填充:\n{imp_knn.fit_transform(X).round(2)}")


def demo_column_transformer():
    """ColumnTransformer 组合预处理"""
    print("=" * 50)
    print("7. ColumnTransformer 组合预处理")
    print("=" * 50)
    
    from sklearn.compose import ColumnTransformer, make_column_selector as selector
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    
    # 混合类型数据
    df = pd.DataFrame({
        '年龄': [25, 30, np.nan, 40, 35],
        '收入': [50000, 60000, 55000, np.nan, 70000],
        '性别': ['男', '女', '男', '女', '男'],
        '城市': ['北京', '上海', '北京', '广州', '上海']
    })
    
    print(f"原始数据:\n{df}")
    
    # 数值列处理
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 类别列处理
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])
    
    # 组合
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, selector(dtype_include='number')),
        ('cat', categorical_transformer, selector(dtype_include='object'))
    ])
    
    X_processed = preprocessor.fit_transform(df)
    print(f"\n处理后形状: {X_processed.shape}")
    print(f"特征名称: {preprocessor.get_feature_names_out()}")


def demo_all():
    """运行所有演示"""
    import os
    os.makedirs('outputs/sklearn', exist_ok=True)
    
    demo_scalers()
    print()
    demo_standard_scaler()
    print()
    demo_minmax_scaler()
    print()
    demo_power_transformer()
    print()
    demo_encoders()
    print()
    demo_imputers()
    print()
    demo_column_transformer()


if __name__ == "__main__":
    demo_all()

