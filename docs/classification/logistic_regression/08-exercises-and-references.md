---
title: LogisticRegression 逻辑回归分类 — 练习与参考文献
outline: deep
---

# 练习与参考文献

## 本章目标

1. 用练习题帮助读者检查自己是否真正理解当前 Logistic Regression 实现。
2. 给出继续深入阅读逻辑回归与相关数据集工具的可靠入口。

## 自检题

1. 为什么 `pipelines/classification/logistic_regression.py` 要先做训练/测试切分，再做标准化？如果在切分前标准化会有什么问题？
2. 为什么当前 `make_classification(n_features=6, n_informative=3, class_sep=1.2)` 数据适合逻辑回归的线性边界假设？`n_informative=3` 在 6 个特征中有什么教学意义？
3. 当前 `train_model(...)` 中的 `penalty`、`C`、`solver`、`max_iter` 分别控制什么？`C` 与正则化强度 $\lambda$ 的关系是什么？
4. 为什么 `model.coef_` 与 `model.intercept_` 对理解逻辑回归很重要？标准化后 $w_j$ 的正负和大小分别代表什么？
5. 为什么 ROC 曲线这里使用 `predict_proba(...)` 而不是 `predict(...)`？逻辑回归的 Sigmoid 概率输出与 KNN 的邻域频率概率输出有什么不同？
6. 为什么决策边界图里需要额外训练一个 `model_2d`？它在什么特征空间上训练？逻辑回归的 PCA 边界通常长什么样？
7. `n_iter_` 属性的含义是什么？如果它等于 `max_iter=1000`，意味着什么？

## 练习方向

### 1. 改动 C

- 把 `C=1.0` 改成 `0.01`、`0.1`、`10.0`、`100.0`、`1000.0`
- 观察变化：
  - `coef_` 的绝对值大小——$C$ 越小（正则越强），系数越收缩趋近 0
  - $\vert w_j\vert$ 的分布——强正则下只有最重要的特征保留较大系数
  - 混淆矩阵中各类别正确/错误分布
  - 学习曲线中训练得分与验证得分的差距——强正则时两者接近（欠拟合），弱正则时训练得分远高于验证得分（过拟合）
- 核心理解：$C$ 是正则化强度的倒数，$\lambda = 1/C$

### 2. 改动 penalty

- 尝试 `penalty='l2'`、`penalty='l1'`（需切换 solver 为 `'saga'` 或 `'liblinear'`）、`penalty=None`
- 对比变化：
  - L1 正则化的系数稀疏性——部分 $w_j$ 会被压缩为 0，自动做特征选择
  - L2 正则化的系数均匀收缩——所有 $w_j$ 变小但非零
  - 无正则化时系数的量级——通常最大，但也最容易过拟合
- 理解 L1 和 L2 的数学公式差异：$\|\mathbf{w}\|_1 = \sum \vert w_j\vert$（稀疏）vs $\|\mathbf{w}\|_2^2 = \sum w_j^2$（均匀）

### 3. 去掉标准化

- 暂时去掉 `StandardScaler()`，直接用 `X_train`、`X_test`
- 对比变化：
  - `coef_` 的值——各特征的系数不可直接比较
  - 训练收敛情况——可能收到 `ConvergenceWarning`
- 体会：标准化不仅影响系数可比性，还影响梯度优化的收敛速度和稳定性

### 4. 观察 coef_ 与特征序号的关系

- 逻辑回归数据中 `n_informative=3`、`n_redundant=1`，即前 3 个特征真正有用，第 4 个是冗余线性组合，后 2 个是随机噪声
- 观察 `coef_` 的 6 个值——看看模型是否自动赋予前几个特征更大的系数绝对值
- 对比强正则（$C=0.01$）和弱正则（$C=100$）下系数分布的差异

### 5. 与 KNN、决策树、SVC 对比

- 对照阅读 `docs/classification/knn/`、`docs/classification/decision_tree/`、`docs/classification/svc/`
- 比较要点：
  - 决策边界的性质：逻辑回归是全局线性超平面 $\mathbf{w}^T\mathbf{x} + b = 0$，KNN 是局部非参数边界，决策树是轴对齐分段边界，SVC 是最大间隔 + 核变换边界
  - 概率输出的来源：逻辑回归是连续的 Sigmoid 映射，KNN 是离散的邻域频率
  - 是否需要标准化：逻辑回归需要（梯度优化 + 系数可比），KNN 需要（距离度量），决策树不需要（阈值切分）
  - 可解释性：逻辑回归有 `coef_`（方向+强度），决策树有特征重要性（贡献度），KNN 没有显式特征解释

## 参考文献

| # | 文献 | 说明 |
|---|---|---|
| 1 | scikit-learn 官方文档：`LogisticRegression` | 完整构造器参数列表、属性与方法说明 |
| 2 | scikit-learn 官方文档：`make_classification` | 高维分类数据生成器的参数与使用说明 |
| 3 | scikit-learn 用户指南：Linear Models | 逻辑回归的数学原理、优化器选择与正则化策略详细讲解 |
| 4 | Hastie, T., Tibshirani, R., and Friedman, J. (2009). *The Elements of Statistical Learning*. | 第 4 章：Linear Methods for Classification，涵盖逻辑回归、LDA、线性可分性的完整数学推导 |

- scikit-learn `LogisticRegression`：https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- scikit-learn `make_classification`：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
- scikit-learn 用户指南 Linear Models：https://scikit-learn.org/stable/modules/linear_model.html

## 小结

- 这一章的重点不是新增概念，而是把前面章节学到的内容重新落到源码和实验现象上。
- 如果能独立解释以下问题，说明已经掌握了当前 Logistic Regression 分册的核心内容：
  - 标准化必须在切分后执行（防止数据泄露），且对逻辑回归有三大好处（收敛稳定、正则均匀、系数可比）
  - 线性打分 → Sigmoid 概率 → 交叉熵优化的完整数学链
  - `C` 是 $\lambda$ 的倒数（$C$ 越大正则越弱）——这是最容易写反的核心概念
  - `coef_` 的正负和绝对值反映特征对正类概率的影响方向和强度
  - 逻辑回归的 Sigmoid 概率输出是连续的，ROC 曲线平滑——与 KNN 的离散邻域频率本质不同
  - `model`（6 维空间）、`model_2d`（PCA 空间）和学习曲线实例的职责差异
