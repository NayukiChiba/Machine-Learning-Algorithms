---
title: Bagging 与随机森林 — 数学原理
outline: deep
---

# 数学原理

> 对应代码：`model_training/ensemble/bagging.py`
>  
> 相关对象：`BaggingClassifier`、`DecisionTreeClassifier`、`train_model(...)`

## 本章目标

1. 理解 Bagging 为什么通过 Bootstrap 重采样和集成平均来降低方差。
2. 理解 OOB 样本为什么会自然出现，以及它在工程上有什么意义。
3. 把这些统计直觉和当前源码中的 `n_estimators`、`max_samples`、`bootstrap`、`oob_score` 对应起来。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| Bootstrap 采样 | 重采样机制 | 为每个基学习器构造不同训练子样本 |
| Bagging | 集成方法 | 通过并行平均或投票降低模型方差 |
| OOB | 袋外样本 | 用于额外参考评估 |
| 方差缩减公式 | 统计结果 | 说明集成平均为什么更稳定 |
| `oob_score_` | 源码属性 | 当前实现中的袋外参考得分 |

## 1. Bagging（Bootstrap Aggregating）

### 核心思想

Bagging 通过自举采样（Bootstrap）构建多个独立的基学习器，再通过投票（分类）或平均（回归）进行集成，以降低方差。

### 理解重点

- 当前源码中的 `BaggingClassifier(...)`，本质上就是在执行这种“重采样 + 并行训练 + 集成输出”的过程。
- 它和 GBDT / XGBoost / LightGBM 的串行纠错逻辑完全不同。
- 当前 Bagging 分册的数学主轴应该始终围绕“降方差”展开。

## 2. Bootstrap 采样

从大小为 $N$ 的训练集中有放回地抽取 $N$ 个样本，形成一个 Bootstrap 样本集。每个样本被抽中的概率为：

$$
P(\text{被选中}) = 1 - \left(1 - \frac{1}{N}\right)^N \xrightarrow{N \to \infty} 1 - \frac{1}{e} \approx 0.632
$$

约 36.8% 的样本未被选中，这些样本称为 OOB（Out-Of-Bag）样本。

### 理解重点

- Bootstrap 让每棵基学习器看到的训练数据都略有不同，这是 Bagging 能产生“模型多样性”的关键。
- 同时，也正因为有放回采样，总会有一部分样本没被某棵树看到，这才自然产生 OOB 样本。
- 当前源码里的 `bootstrap=True` 和 `oob_score=True`，正是这一数学与工程联系的直接体现。

## 3. 方差缩减

假设 $T$ 个基学习器 $h_1, \dots, h_T$ 的预测方差均为 $\sigma^2$，两两相关系数为 $\rho$，集成后方差为：

$$
\text{Var}\left[\frac{1}{T}\sum_{t=1}^T h_t\right] = \rho\sigma^2 + \frac{1-\rho}{T}\sigma^2
$$

- 第一项 $\rho\sigma^2$ 不可消除（受相关性限制）
- 第二项随 $T$ 增大而趋近于零

### 理解重点

- 这条公式说明 Bagging 的好处不是“让每棵树更聪明”，而是“让整体结果更稳定”。
- 只要基学习器之间不是完全相关，增加集成数量通常就能降低整体波动。
- 当前源码中的 `n_estimators=80`，正是对这个“多模型平均”思想的工程映射。

## 4. OOB 样本为什么有用

### 理解重点

- OOB 样本没有参与某棵基学习器的训练，因此可以拿来作为该学习器的额外参考评估。
- 把所有基学习器对应的 OOB 预测汇总起来，就能得到一个不依赖额外验证集的参考得分。
- 这也是为什么当前训练日志中的 `OOB 得分` 对 Bagging 特别有代表性。

## 5. 随机森林的改进方向

### 改进：特征随机化

在 Bagging 的基础上，随机森林在每次节点分裂时，只从随机抽取的 $m$ 个特征中选择最优分裂特征，进一步降低基学习器的相关性 $\rho$。

推荐值：

| 任务 | $m$ 的推荐值 |
|------|-------------|
| 分类 | $m = \lfloor \sqrt{d} \rfloor$ |
| 回归 | $m = \lfloor d/3 \rfloor$ |

### 特征重要性

基于不纯度（MDI）：特征 $j$ 在所有树中被用于分裂时带来的不纯度减少之和。

$$
\text{Imp}(j) = \sum_{t=1}^{T} \sum_{\text{node } v \text{ splits on } j} \Delta \text{Gini}(v)
$$

### 理解重点

- 当前文档标题沿用了“Bagging 与随机森林”，因此这里保留随机森林的理论延伸是合理的。
- 但要特别注意：当前仓库本分册的工程实现主体是 `BaggingClassifier`，而不是 `RandomForestClassifier`。
- 所以这一节更适合作为“Bagging 的自然扩展理解”，而不能误写成当前代码已经实现了随机森林流程。

## 6. 数学原理如何映射到当前源码

### 理解重点

- Bootstrap 重采样在工程里对应 `bootstrap=True`。
- 集成数量在工程里对应 `n_estimators`。
- 每棵基学习器看到多少样本和多少特征，在工程里分别对应 `max_samples` 和 `max_features`。
- OOB 样本带来的额外参考估计，在工程里对应 `oob_score=True` 和 `model.oob_score_`。

## 常见坑

1. 把 Bagging 的收益误解成“降低偏差”，而忽略它的主要目标其实是降低方差。
2. 只记住 Bootstrap 会重采样，却不理解 OOB 为什么会自然出现。
3. 把随机森林的特征随机化机制误写成当前 Bagging 实现默认就已经这样做了。

## 小结

- Bagging 的数学核心，是通过 Bootstrap 重采样构造多个不同训练子集，再通过并行集成降低模型输出方差。
- OOB 样本是这种重采样机制自然产生的副产品，也是 Bagging 很有代表性的工程估计手段。
- 当前源码中的 `n_estimators`、`max_samples`、`bootstrap`、`oob_score`，正是这些统计思想在工程层面的直接映射。
