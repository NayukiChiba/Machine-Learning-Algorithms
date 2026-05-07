---
title: GaussianNB 高斯朴素贝叶斯 — 练习与参考文献
outline: deep
---

# 练习与参考文献

## 本章目标

1. 用练习题帮助读者检查自己是否真正理解当前 Naive Bayes 实现。
2. 给出继续深入阅读高斯朴素贝叶斯与相关数据集工具的可靠入口。

## 自检题

1. 为什么 `pipelines/classification/naive_bayes.py` 要先做训练/测试切分，再做标准化？如果在切分前标准化会有什么问题？
2. 为什么当前 iris 连续特征数据适合 `GaussianNB` 而非文本分类常见的 `MultinomialNB`？`GaussianNB` 的高斯似然 $\mathcal{N}(\mu_{kj}, \sigma_{kj}^2)$ 与另外两种朴素贝叶斯变体的似然建模有什么本质区别？
3. 当前 `train_model(...)` 中的 `var_smoothing` 控制什么？实际计算中 $\sigma_{kj}^2 + \epsilon \cdot \sigma_{\max}^2$ 里的 $\sigma_{\max}^2$ 是什么？为什么方差接近 0 时会引发数值问题？
4. 为什么 `model.class_prior_`、`model.theta_` 和 `model.var_` 对理解 GaussianNB 很重要？它们分别对应贝叶斯公式中的哪些项？
5. 为什么 ROC 曲线这里使用 `predict_proba(...)` 而不是 `predict(...)`？GaussianNB 的连续后验概率与 KNN 的离散邻域频率概率输出有什么不同？
6. 为什么决策边界图里需要额外训练一个 `model_2d`？它在什么特征空间上训练？GaussianNB 的 PCA 二维边界为什么可能呈现曲线而非直线？
7. GaussianNB 的 `fit()` 为什么是所有分类模型中最快的之一？它不依赖迭代优化的数学原因是什么？

## 练习方向

### 1. 改动 `var_smoothing`

- 把 `var_smoothing=1e-9` 改成 `1e-12`、`1e-9`、`1e-6`、`1e-3`
- 观察变化：
  - `model.epsilon_` 的实际值——即 $\epsilon \cdot \sigma_{\max}^2$ 的量级
  - `model.var_` 中各类别各特征方差的变化——平滑越小，方差越接近原始样本方差
  - 混淆矩阵和 ROC 曲线的变化——极端平滑值可能导致概率估计失准
- 核心理解：`var_smoothing` 在数值稳定性和模型精度之间的权衡——$\epsilon$ 太大过度平滑，$\epsilon$ 太小可能数值崩溃

### 2. 观察 `theta_` 与 `var_` 的类别间差异

- 在训练完成后打印 `model.theta_`（形状 $(3, 4)$）和 `model.var_`（形状 $(3, 4)$）
- 对比三类鸢尾花在 4 个特征上的均值差异——例如 Setosa 与另外两类的花瓣长度均值差距最大
- 对比各类别各特征的方差——方差较小的特征（如 Setosa 的花瓣长度）表示该类在该特征上更集中
- 核心理解：$\mu_{kj}$ 的类间差异越大、$\sigma_{kj}^2$ 越小，该特征对该类别的区分力越强

### 3. 去掉标准化

- 暂时去掉 `StandardScaler()`，直接用 `X_train`、`X_test` 训练和预测
- 对比变化：
  - `theta_` 和 `var_` 的值——特征量纲差异直接反映在方差数值上
  - PCA 决策边界图和混淆矩阵的变化
- 体会：虽然 GaussianNB 不像逻辑回归那样依赖标准化做梯度优化，但标准化使方差估计更稳定，并影响 PCA 可视化的主导方向

### 4. 观察 `predict_proba` 与 `predict` 的关系

- 对同一测试样本同时输出 `y_pred`（硬分类）和 `y_scores`（各类别概率）
- 验证：`y_pred[i]` 是否总是等于 `np.argmax(y_scores[i])`
- 观察三类概率的分布——正确预测样本的概率是否接近 1？错误预测样本的概率是否较均匀（如 $[0.1, 0.45, 0.45]$）？
- 核心理解：MAP 决策 $\hat{y} = \arg\max_c P(c \vert \mathbf{x})$ 等价于在 `predict_proba` 的输出行上取 `argmax`

### 5. 与逻辑回归、KNN、决策树对比

- 对照阅读 `docs/classification/logistic_regression/`、`docs/classification/knn/`、`docs/classification/decision_tree/`
- 比较要点：
  - 建模方式：GaussianNB 是生成式（对 $P(\mathbf{x} \vert Y)$ 建模），逻辑回归是判别式（对 $P(Y \vert \mathbf{x})$ 直接建模），KNN 是非参数（无显式 $P(\mathbf{x} \vert Y)$ 建模），决策树是判别式（递归划分）
  - 训练方式：GaussianNB 是统计量扫描（$\mu_{kj}$、$\sigma_{kj}^2$ 一步到位），逻辑回归是迭代优化（`lbfgs`），KNN 无训练（仅建索引），决策树是递归贪心搜索
  - 是否需要标准化：GaussianNB 不强制（但利于可视化和方差比较），逻辑回归必须（梯度收敛），KNN 必须（距离度量），决策树不需要（阈值切分）
  - 可解释性：GaussianNB 有 `theta_`/`var_`（各类别特征分布）、`class_prior_`（先验），逻辑回归有 `coef_`（线性权重），决策树有 `feature_importances_`（贡献度），KNN 没有显式特征重要性
  - 概率输出性质：GaussianNB 是连续的高斯后验（平滑 ROC），KNN 是离散的邻域频率（阶梯状 ROC）

## 参考文献

| # | 文献 | 说明 |
|---|---|---|
| 1 | scikit-learn 官方文档：`GaussianNB` | 完整构造器参数（`priors`、`var_smoothing`）、属性（`class_prior_`、`theta_`、`var_`、`epsilon_`）与方法说明 |
| 2 | scikit-learn 官方文档：`load_iris` | iris 数据集的来源、特征含义与类别说明 |
| 3 | scikit-learn 用户指南：Naive Bayes | GaussianNB、MultinomialNB、BernoulliNB、ComplementNB 的完整数学推导与使用场景对比 |
| 4 | Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. | 第 3 章：生成式分类模型；第 4 章：高斯判别分析（GDA）与朴素贝叶斯的数学关系 |

- scikit-learn `GaussianNB`：https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
- scikit-learn `load_iris`：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
- scikit-learn 用户指南 Naive Bayes：https://scikit-learn.org/stable/modules/naive_bayes.html

## 小结

- 这一章的重点不是新增概念，而是把前面章节学到的内容重新落到源码和实验现象上。
- 如果能独立解释以下问题，说明已经掌握了当前 Naive Bayes 分册的核心内容：
  - 标准化必须在切分后执行（防止数据泄露），GaussianNB 保留标准化主要是为了方差稳定性和 PCA 可视化
  - 贝叶斯公式 $P(Y \vert \mathbf{x}) \propto P(\mathbf{x} \vert Y) P(Y)$ → 条件独立 $\prod P(x_j \vert Y)$ → 高斯似然 $\mathcal{N}(\mu_{kj}, \sigma_{kj}^2)$ → MAP 决策（对数形式）的完整数学链
  - `var_smoothing` 是 $\sigma_{kj}^2$ 的数值保护——方差近零时 $\frac{1}{\sqrt{2\pi \sigma^2}} \to \infty$
  - `theta_` 和 `var_` 反映各类别各特征的分布特征，类间均值差异大且方差小的特征是区分力最强的特征
  - GaussianNB 的概率输出是连续的贝叶斯后验概率，ROC 曲线平滑——与 KNN 的离散邻域频率本质不同
  - `model`（4 维空间）、`model_2d`（PCA 空间）和学习曲线实例的职责差异
  - GaussianNB 的所有参数（先验、均值、方差）都是解析解——不涉及迭代优化，这是它训练极快的根本原因
