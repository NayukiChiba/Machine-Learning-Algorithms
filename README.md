# Machine-Learning-Algorithms

机器学习算法学习与实现的总结库。下面是算法分类清单，你可以按这个分类建文件夹。

## 分类

### 分类算法

- 逻辑回归
- 决策树
- 支持向量机（SVC）
- 朴素贝叶斯
- KNN
- 随机森林

### 回归算法

- 线性回归
- Ridge / Lasso
- 决策树回归
- 支持向量回归（SVR）

### 聚类算法

- K-Means
- DBSCAN

### 降维算法

- PCA
- LDA

### 集成学习

- Bagging
- GBDT
- LightGBM

### 概率与序列模型

- EM
- HMM

```
Machine-Learning-Algorithms/
│
├── data_generation/               # 阶段 1：生成数据
│   ├── __init__.py
│   ├── classification.py          # 所有分类算法的数据生成
│   ├── regression.py              # 所有回归算法的数据生成
│   ├── clustering.py              # 所有聚类算法的数据生成
│   ├── ensemble.py                # 集成学习的数据生成
│   ├── dimensionality.py          # 降维算法的数据生成
│   └── probabilistic.py           # 概率模型的数据生成
│
├── data_exploration/              # 阶段 2：数据探索
│   ├── __init__.py
│   ├── univariate.py              # 单变量分析（分布、统计摘要）
│   ├── bivariate.py               # 双变量分析（相关性、散点矩阵）
│   └── multivariate.py            # 多变量分析（热力图、PCA 前分析）
│
├── data_processing/               # 阶段 3：数据处理
│   ├── __init__.py
│   ├── splitter.py                # 数据集划分（train/test/val）
│   ├── scaler.py                  # 标准化 / 归一化
│   ├── encoder.py                 # 编码（标签编码、One-Hot）
│   └── pipeline.py                # sklearn Pipeline 组合封装
│
├── data_visualization/            # 阶段 4：数据可视化（原始数据）
│   ├── __init__.py
│   ├── distribution.py            # 分布图（直方图、箱线图、密度图）
│   ├── scatter.py                 # 散点图、散点矩阵
│   ├── correlation.py             # 相关性热力图
│   └── feature_space.py           # 特征空间可视化（2D/3D）
│
├── model_training/                # 阶段 5：训练模型
│   ├── __init__.py
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── knn.py                 # KNN 训练
│   │   ├── svc.py                 # SVC 训练
│   │   ├── logistic_regression.py
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   └── naive_bayes.py
│   ├── regression/
│   │   ├── __init__.py
│   │   ├── linear_regression.py
│   │   ├── decision_tree.py
│   │   ├── svr.py
│   │   └── regularization.py      # Ridge / Lasso / ElasticNet
│   ├── clustering/
│   │   ├── __init__.py
│   │   ├── kmeans.py
│   │   └── dbscan.py
│   ├── ensemble/
│   │   ├── __init__.py
│   │   ├── bagging.py
│   │   ├── gbdt.py
│   │   ├── lightgbm.py
│   │   └── xgboost.py
│   ├── dimensionality/
│   │   ├── __init__.py
│   │   ├── pca.py
│   │   └── lda.py
│   └── probabilistic/
│       ├── __init__.py
│       ├── em.py
│       └── hmm.py
│
├── result_visualization/          # 阶段 6：结果可视化
│   ├── __init__.py
│   ├── decision_boundary.py       # 决策边界（分类）
│   ├── residual_plot.py           # 残差图（回归）
│   ├── cluster_plot.py            # 聚类分布图
│   ├── confusion_matrix.py        # 混淆矩阵
│   ├── roc_curve.py               # ROC 曲线
│   ├── learning_curve.py          # 学习曲线
│   ├── feature_importance.py      # 特征重要性（集成学习）
│   └── dimensionality_plot.py     # 降维后散点图
│
├── model_evaluation/              # 阶段 7：模型评估
│   ├── __init__.py
│   ├── classification_metrics.py  # 精度、F1、AUC、混淆矩阵
│   ├── regression_metrics.py      # MSE、RMSE、MAE、R²
│   ├── clustering_metrics.py      # 轮廓系数、Inertia、DB指数
│   └── dimensionality_metrics.py  # 解释方差比、重建误差
│
├── pipelines/                     # 流水线入口（串联以上各阶段）
│   ├── __init__.py
│   ├── classification/
│   │   ├── knn.py                 # KNN 端到端流程
│   │   ├── svc.py
│   │   ├── logistic_regression.py
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   └── naive_bayes.py
│   ├── regression/
│   │   ├── linear_regression.py
│   │   ├── decision_tree.py
│   │   ├── svr.py
│   │   └── regularization.py
│   ├── clustering/
│   │   ├── kmeans.py
│   │   └── dbscan.py
│   ├── ensemble/
│   │   ├── bagging.py
│   │   ├── gbdt.py
│   │   ├── lightgbm.py
│   │   └── xgboost.py
│   ├── dimensionality/
│   │   ├── pca.py
│   │   └── lda.py
│   └── probabilistic/
│       ├── em.py
│       └── hmm.py
│
├── Basic/                         # 工具库学习模块（保持不动）
│   ├── Numpy/
│   ├── Pandas/
│   ├── ScikitLearn/
│   ├── Scipy/
│   └── Visualization/
│
├── utils/                         # 通用工具
│   ├── __init__.py
│   ├── contextmanage.py           # 已有
│   ├── decorate.py                # 已有
│   ├── plot_style.py              # 统一绘图风格（颜色、字体、保存）
│   └── report.py                  # 统一报告打印格式
│
├── outputs/                       # 统一输出（按算法路径）
│   ├── classification/knn/
│   ├── regression/linear/
│   └── ...
│
├── config.py                      # 项目配置
├── requirements.txt
└── README.md

```
