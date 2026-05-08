---
title: SVC 支持向量分类 — 练习与参考文献
outline: deep
---

# 练习与参考文献

## 本章目标

1. 用练习题帮助读者检查自己是否真正理解当前 SVC 实现。
2. 给出继续深入阅读支持向量机与核方法的可靠入口。

## 自检题

1. 为什么 `pipelines/classification/svc.py` 要先做训练/测试切分，再做标准化？如果在切分前标准化会有什么问题？
2. 为什么当前 `make_circles(noise=0.1, factor=0.5)` 同心圆数据必须依赖 RBF 核而非线性核？RBF 核 $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma\|\mathbf{x} - \mathbf{z}\|^2)$ 与线性核 $\mathbf{x}^T \mathbf{z}$ 在几何上有何本质差异？
3. 当前 `train_model(...)` 中的 `C`、`kernel`、`gamma` 分别控制什么？$C$ 与软间隔目标函数 $C\sum\xi_i$ 的关系是什么？`gamma='scale'` 时 $\gamma = 1/(d \cdot X.var())$ 的实际意义是什么？
4. 为什么 `model.n_support_` 对理解 SVC 很重要？它与 KKT 条件中的 $\alpha_i > 0$ 有什么关系？支持向量多说明什么？少说明什么？
5. 为什么当前 SVC 流水线没有 ROC 曲线评估？启用 ROC 需要做什么额外配置？代价是什么？
6. 为什么决策边界图里需要额外训练一个 `model_2d`？它在什么特征空间上训练？主模型与 `model_2d` 的核函数配置有何异同？
7. 为什么 SVC 的标准化不是可选的优化手段而是硬性要求？RBF 核中的 $\|\mathbf{x} - \mathbf{z}\|^2$ 如何被特征量纲影响？

## 练习方向

### 1. 改动 $C$

- 把 `C=1.0` 改成 `0.01`、`0.1`、`1.0`、`10.0`、`100.0`
- 观察变化：
  - `n_support_` 的数量——$C$ 越小（间隔越宽），支持向量通常越多
  - 混淆矩阵中内外圈的正确率变化
  - 决策边界的弯曲程度——$C$ 小时边界更平滑但可能过于简单，$C$ 大时边界更精细但可能过拟合噪声
  - 学习曲线中训练得分与验证得分的差距——$C$ 大时容易过拟合（两者差距大）
- 核心理解：$C$ 越大 $\approx$ 正则越弱——与逻辑回归中 $C = 1/\lambda$ 的关系一致

### 2. 改动 `gamma`

- 把 `gamma='scale'` 改成 `0.01`、`0.1`、`1.0`、`10.0`、`'auto'`
- 观察变化：
  - RBF 核边界的弯曲精细程度——$\gamma$ 越大边界越"崎岖"，单个支持向量影响范围越小
  - 模型的过拟合/欠拟合倾向——$\gamma$ 过大时每个支持向量只影响周围很小的区域
  - 支持向量数量的变化
- 核心理解：$C$ 和 $\gamma$ 的联合效应——两者共同决定模型复杂度：$C \uparrow + \gamma \uparrow$ 最容易过拟合

### 3. 改用线性核

- 把 `kernel='rbf'` 改为 `kernel='linear'`
- 对比变化：
  - 决策边界的形状——从环形曲线变为一条直线，无法分离内外圈
  - 混淆矩阵——大量误分类集中在边界线上
  - 准确性上限——线性核对同心圆数据的理论最佳准确率约为 50%（一条直线最多切中一半的点）
- 核心理解：核函数不是"更复杂就更强"，而是必须匹配数据形状——这是 SVC 最核心的教学启示

### 4. 去掉标准化

- 暂时去掉 `StandardScaler()`，直接用 `X_train`、`X_test` 训练
- 对比变化：
  - 决策边界的形状——RBF 核的距离计算失真，边界可能完全偏离正确的环形结构
  - 混淆矩阵中的误分类大幅增加
  - `gamma='scale'` 计算出的 $\gamma$ 值因量纲而变化
- 体会：标准化不是锦上添花——对 RBF 核 SVC 而言，它是核函数几何意义正确的前提

### 5. 观察支持向量数量与数据噪声的关系

- 修改 `make_circles(noise=...)` 的 `noise` 参数（`0`、`0.05`、`0.1`、`0.2`）
- 观察 `n_support_` 的变化趋势——噪声越大，两类样本越纠缠，支持向量通常越多
- 核心理解：支持向量数量间接反映了数据的线性不可分程度——支持向量越多，说明两类越"纠结"

## 参考文献

| # | 文献 | 说明 |
|---|---|---|
| 1 | scikit-learn 官方文档：`SVC` | 完整构造器参数列表（`C`、`kernel`、`gamma`、`degree`、`probability` 等）、属性（`support_vectors_`、`n_support_`、`dual_coef_`、`intercept_`）与方法说明 |
| 2 | scikit-learn 官方文档：`make_circles` | 同心圆数据生成器的 `n_samples`、`noise`、`factor` 等参数说明 |
| 3 | scikit-learn 用户指南：SVM | C-SVC 的完整数学推导、核函数对比、多分类策略（OvO/OvR）与实用调参指南 |
| 4 | Cortes, C. and Vapnik, V. (1995). *Support-Vector Networks*. Machine Learning, 20, 273-297. | SVM 的原始论文——最大间隔思想、软间隔引入和核技巧的源头 |

- scikit-learn `SVC`：https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- scikit-learn `make_circles`：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html
- scikit-learn 用户指南 SVM：https://scikit-learn.org/stable/modules/svm.html

## 小结

- 这一章的重点不是新增概念，而是把前面章节学到的内容重新落到源码和实验现象上。
- 如果能独立解释以下问题，说明已经掌握了当前 SVC 分册的核心内容：
  - 标准化必须在切分后执行（防止数据泄露），且对 RBF 核是硬性要求（距离计算不能被量纲绑架）
  - 最大间隔 $\min\frac{1}{2}\|\mathbf{w}\|^2$ → 软间隔 $+C\sum\xi_i$ → 对偶 + 内积 → RBF 核 → 支持向量 → 决策函数 $f(\mathbf{x})$ 的完整数学链
  - $C$ 越大正则越弱（与逻辑回归一致），$\gamma$ 越大核窗口越窄——两者联合决定复杂度
  - 线性核对同心圆数据必然失败——这是 SVC 分册最有教学启示的实验对比
  - `n_support_` 是 SVC 独有的教学窗口——通过它可以看到模型依赖了多少关键样本
  - SVC 默认 `probability=False`——当前流水线无概率输出、无 ROC 曲线，这是与其他分类分册的核心差异
  - `model`（原始 2 维标准化空间）、`model_2d`（PCA 空间）和学习曲线实例的职责边界
