---
title: HMM — 总览
outline: deep
---

# HMM

## 本章目标

1. 明确本分册对应的 HMM 源码入口与运行方式——注意这是序列模型，与 EM 的独立样本聚类、集成学习的分类/回归任务有根本差异。
2. 理解当前 HMM 文档各章节分别负责解释什么内容。
3. 建立从马尔可夫假设、Forward/Viterbi/Baum-Welch 三大算法到隐状态预测的整体阅读路线。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/probabilistic.py` | `ProbabilisticData.hmm()` 手动参数化生成离散观测序列与真实隐状态序列 |
| 数据导出 | `data_generation/__init__.py` | 导出 `hmm_data` |
| 训练封装 | `model_training/probabilistic/hmm.py` | `train_model(...)` 封装 `hmmlearn` 的 HMM 训练——含 `CategoricalHMM` / `MultinomialHMM` 双备份 |
| 端到端流水线 | `pipelines/probabilistic/hmm.py` | 完成观测序列整理、训练、Viterbi 解码和终端评估输出 |

## 默认配置速览（来自源码）

| 项目 | 当前实现 |
|---|---|
| 训练模型 | `CategoricalHMM(n_components=3, n_iter=100, tol=1e-3, random_state=42)`——若不可用则回退到 `MultinomialHMM` |
| 数据来源 | 手动参数化生成的单条离散观测序列——$\pi=[0.6,0.3,0.1]$，$A$ 和 $B$ 为 $3 \times 3$ 矩阵，300 时间步 |
| 数据形态 | 离散观测符号 $\{0, 1, 2\}$——无需标准化 |
| 训练方式 | 全量序列直接训练，无 train/test split |
| 评估方式 | 终端打印隐状态预测准确率 + 学习到的转移矩阵 |

## HMM 与本仓库其他算法的定位对比

| 配置项 | KMeans | EM (GMM) | HMM |
|---|---|---|---|
| 任务类型 | 聚类 | 聚类 | **序列状态推断** |
| 算法范式 | 质心迭代 | 概率生成模型 | **概率图模型 + 动态规划** |
| 数据特性 | i.i.d. | i.i.d. | **时序依赖（马尔可夫链）** |
| 赋值方式 | 硬赋值 | 软赋值 | **Viterbi 全局解码路径** |
| 训练输入 | `fit(X)` | `fit(X)` | **`fit(X, lengths)`——序列数据** |
| 标准化 | 有 | 有 | **无（离散观测）** |
| 核心输出 | `labels_`、`cluster_centers_` | `predict_proba()`、`means_`、`covariances_` | **`predict()`（Viterbi）、`transmat_`、`emissionprob_`** |
| 可视化 | 聚类图 | 双面板聚类图 | **无（终端文本）** |

## 阅读路线

1. [数学原理](/probabilistic/hmm/01-mathematics)
2. [数据构成](/probabilistic/hmm/02-data)
3. [思路与直觉](/probabilistic/hmm/03-intuition)
4. [模型构建](/probabilistic/hmm/04-model)
5. [训练与预测](/probabilistic/hmm/05-training-and-prediction)
6. [评估与诊断](/probabilistic/hmm/06-evaluation)
7. [工程实现](/probabilistic/hmm/07-implementation)
8. [练习与参考文献](/probabilistic/hmm/08-exercises-and-references)

## 如何运行

### 示例代码

```bash
python -m pipelines.probabilistic.hmm
```

### 理解重点

- 这个命令会运行 HMM 的 Baum-Welch 训练——学习一个 3 状态离散 HMM 的转移矩阵和发射矩阵。
- 当前流程是**序列状态推断**——`state_true` 仅在评估时用于对比真实隐状态路径，**不参与模型训练**。
- HMM 的输出是 Viterbi 全局解码路径——与分类器的单点标签预测、聚类的逐点 argmax 有本质区别。

## 先修

- [库生态总览](/foundations/overview)
- [NumPy 基础与数组概念](/foundations/numpy/01-basics)
- [EM 与 GMM](/probabilistic/em/)
- [项目架构](/appendix/)

## 小结

- 本分册严格对应当前仓库中的 HMM 源码实现——数据生成、Baum-Welch 训练、Viterbi 解码构成完整序列建模流水线。
- HMM 的核心特点：马尔可夫假设 + 离散观测 + Forward/Backward/Viterbi 动态规划 + Baum-Welch EM 学习——与 EM 的 i.i.d. 软聚类形成根本差异。
- 当前使用手动参数化生成的 3 状态 300 步离散序列 + `CategoricalHMM`，是展示序列状态推断能力最经典的教学配置。
