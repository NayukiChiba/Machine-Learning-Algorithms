'''
结果可视化
'''
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


from pandas import DataFrame, Series
from typing import Union
from utils.decorate import print_func_info
from matplotlib import pyplot as plt
plt.rcParams["font.sans-serif"] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
from config import OUTPUTS_ROOT
LR_OUTPUTS = os.path.join(OUTPUTS_ROOT, "LinearRegression")


from train_model import train_model
from generate_data import generate_data
from preprocess_data import preprocess_data
from evaluate_model import evaluate_model

@print_func_info
def visualize_results(y_train:Union[DataFrame, Series], 
                        y_train_pred:Union[DataFrame, Series], 
                        y_test:Union[DataFrame, Series], 
                        y_test_pred:Union[DataFrame, Series], 
                        X_test_original:DataFrame, feature_names:list):
    """
    可视化模型结果
    
    参数:
        y_train(Union[DataFrame, Series]), y_train_pred(Union[DataFrame, Series]): 训练集真实值和预测值
        y_test(Union[DataFrame, Series]), y_test_pred(Union[DataFrame, Series]): 测试集真实值和预测值
        X_test_original(DataFrame): 原始测试特征
        feature_names(list): 特征名称
    """
    
    # 1. 预测值 vs 真实值
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('预测值 vs 真实值', fontsize=16, fontweight='bold')
    
    # 训练集
    axes[0].scatter(y_train, y_train_pred, alpha=0.6, s=30, color='steelblue', label='训练集')
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
                 'r--', lw=2, label='理想预测线')
    axes[0].set_xlabel('真实价格 (万元)', fontsize=12)
    axes[0].set_ylabel('预测价格 (万元)', fontsize=12)
    axes[0].set_title('训练集', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 测试集
    axes[1].scatter(y_test, y_test_pred, alpha=0.6, s=30, color='coral', label='测试集')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 'r--', lw=2, label='理想预测线')
    axes[1].set_xlabel('真实价格 (万元)', fontsize=12)
    axes[1].set_ylabel('预测价格 (万元)', fontsize=12)
    axes[1].set_title('测试集', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(LR_OUTPUTS, "04_Prediction_effect.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n保存图表: {filepath}")
    
    # 2. 残差分析
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('残差分析', fontsize=16, fontweight='bold')
    
    # 残差分布
    axes[0, 0].hist(train_residuals, bins=30, color='steelblue', 
                    edgecolor='black', alpha=0.7, label='训练集')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('残差', fontsize=12)
    axes[0, 0].set_ylabel('频数', fontsize=12)
    axes[0, 0].set_title('训练集残差分布', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(test_residuals, bins=30, color='coral', 
                    edgecolor='black', alpha=0.7, label='测试集')
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('残差', fontsize=12)
    axes[0, 1].set_ylabel('频数', fontsize=12)
    axes[0, 1].set_title('测试集残差分布', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 残差 vs 预测值
    axes[1, 0].scatter(y_train_pred, train_residuals, alpha=0.6, s=30, color='steelblue')
    axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('预测价格 (万元)', fontsize=12)
    axes[1, 0].set_ylabel('残差', fontsize=12)
    axes[1, 0].set_title('训练集残差图', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(y_test_pred, test_residuals, alpha=0.6, s=30, color='coral')
    axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('预测价格 (万元)', fontsize=12)
    axes[1, 1].set_ylabel('残差', fontsize=12)
    axes[1, 1].set_title('测试集残差图', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(LR_OUTPUTS, "05_Residual_analysis.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"保存图表: {filepath}")

    # 3. 单特征回归线
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('单特征回归效果', fontsize=16, fontweight='bold')
    
    for i, feature in enumerate(feature_names):
        axes[i].scatter(X_test_original.iloc[:, i], y_test, 
                       alpha=0.6, s=30, color='lightblue', label='真实值')
        axes[i].scatter(X_test_original.iloc[:, i], y_test_pred, 
                       alpha=0.6, s=30, color='coral', label='预测值')
        axes[i].set_xlabel(feature, fontsize=12)
        axes[i].set_ylabel('价格 (万元)', fontsize=12)
        axes[i].set_title(f'{feature}的回归效果', fontsize=12, fontweight='bold')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(LR_OUTPUTS, "06_Single_feature_regression.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"保存图表: {filepath}")


if __name__ == "__main__":
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test = preprocess_data(generate_data())
    model = train_model(X_train=X_train, y_train=y_train)
    feature_names = X_train.columns
    y_train_pred, y_test_pred= evaluate_model(model=model , X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    visualize_results(y_train=y_train, y_train_pred=y_train_pred, 
                        y_test=y_test, y_test_pred=y_test_pred, 
                        X_test_original=X_test, feature_names=feature_names)