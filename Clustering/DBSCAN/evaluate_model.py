"""
聚类评估模块
"""

import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入公共工具
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.metrics import silhouette_score
from utils.decorate import print_func_info
from train_model import train_model
from generate_data import generate_data
from preprocess_data import preprocess_data


@print_func_info
def evaluate_model(model, X_scaled):
    """
    评估聚类效果

    args:
        model: 训练好的模型
        X_scaled: 标准化后的特征
    returns:
        labels
    """
    labels = model.labels_

    # 轮廓系数（去掉噪声后再计算）
    mask = labels != -1
    unique_labels = set(labels[mask])
    if len(unique_labels) > 1:
        score = silhouette_score(X_scaled[mask], labels[mask])
        print(f"轮廓系数: {score:.4f}")
    else:
        print("轮廓系数无法计算（有效簇不足）")

    return labels


if __name__ == "__main__":
    X_scaled, scaler, X = preprocess_data(generate_data())
    model = train_model(X_scaled)
    evaluate_model(model, X_scaled)
