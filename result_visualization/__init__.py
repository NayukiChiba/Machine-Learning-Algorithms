"""
result_visualization 结果可视化子包

提供模型训练后的结果可视化工具，包括:
- decision_boundary: 分类决策边界
- residual_plot: 回归残差图
- cluster_plot: 聚类分布图
- confusion_matrix: 混淆矩阵
- roc_curve: ROC 曲线
- learning_curve: 学习曲线
- feature_importance: 特征重要性
- dimensionality_plot: 降维可视化
"""

import matplotlib.pyplot as plt

# 全局设置 matplotlib 中文字体，避免中文标签乱码
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
