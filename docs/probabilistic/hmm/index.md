---
title: HMM — 总览
outline: deep
---

# HMM

> 对应代码：`pipelines/probabilistic/hmm.py`、`model_training/probabilistic/hmm.py`
>  
> 运行方式：`python -m pipelines.probabilistic.hmm`

## 本章目标

1. 明确本分册对应的 HMM 源码入口与运行方式。
2. 理解当前 HMM 文档各章节分别负责解释什么内容。
3. 建立从数据、模型、训练到隐状态预测的整体阅读路线。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/probabilistic.py` | `ProbabilisticData.hmm()` 生成离散观测序列与真实隐状态序列 |
| 数据导出 | `data_generation/__init__.py` | 导出 `hmm_data` |
| 训练封装 | `model_training/probabilistic/hmm.py` | `train_model(...)` 封装 `hmmlearn` 的 HMM 训练 |
| 端到端流水线 | `pipelines/probabilistic/hmm.py` | 完成观测序列整理、训练、隐状态预测和控制台评估输出 |

## 默认配置速览（来自源码）

| 项目 | 当前实现 |
|---|---|
| 训练模型 | `CategoricalHMM(...)` 优先，若不可用则回退到 `MultinomialHMM(...)` |
| 默认参数 | `n_components=3, n_iter=100, tol=1e-3, random_state=42` |
| 数据形态 | 单条离散观测序列 `obs` + 对比隐状态 `state_true` |
| 训练方式 | 全量序列直接训练，无 train/test split |
| 评估方式 | 控制台打印隐状态预测准确率 + 转移矩阵 |

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

- 这个命令会串起当前 HMM 分册中最核心的工程流程。
- 运行后会训练一个离散 HMM，输出隐状态预测准确率和学习得到的转移矩阵。
- 当前实现重点在于展示“观测序列驱动的隐状态建模”，而不是图形化可视化流程。

## 先修

- [库生态总览](/foundations/overview)
- [NumPy 基础与数组概念](/foundations/numpy/01-basics)
- [术语表](/appendix/glossary)
- [概率模型总览](/probabilistic/em/)

## 小结

- 本分册严格对应当前仓库中的 HMM 源码实现。
- 阅读时建议始终把文档内容与 `pipelines/probabilistic/hmm.py` 和 `model_training/probabilistic/hmm.py` 对照起来看。
- 如果已经熟悉整体入口，可以直接从“模型构建”或“训练与预测”章节开始阅读。
