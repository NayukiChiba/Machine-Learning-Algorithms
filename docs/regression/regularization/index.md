---
title: 正则化回归 — 概述
outline: deep
---

# 正则化回归

## 本章目标

1. 理解正则化回归在回归分册中的定位——线性回归的升级版，通过 L1/L2 惩罚约束系数。
2. 了解正则化回归的三模型架构（Lasso / Ridge / ElasticNet）及其工程结构。
3. 明确正则化回归与线性回归、决策树回归、SVR 的关键差异。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `trainRegularizationModels(...)` | 函数 | 一次性训练 Lasso、Ridge、ElasticNet 三个模型，返回 dict |
| `StandardScaler` | 预处理 | 对 21 维特征做 Z-score 标准化——正则化回归强制要求 |
| `Lasso(alpha=0.15)` | 类 | L1 正则化——将不相关特征的系数驱动到精确零 |
| `Ridge(alpha=2.0)` | 类 | L2 正则化——将系数向零收缩但不精确为零 |
| `ElasticNet(alpha=0.2, l1_ratio=0.5)` | 类 | L1 + L2 混合——兼具稀疏性和收缩能力 |
| 近零系数计数 | 诊断 | `np.sum(np.abs(coef) < 1e-3)`——衡量 Lasso/ElasticNet 的稀疏效果 |

## 正则化回归与本仓库其他回归模型的定位对比

| 维度 | 线性回归 | 正则化回归 | 决策树回归 | SVR |
|---|---|---|---|---|
| 核心能力 | 无偏线性拟合 | **带约束的线性拟合——控制系数大小** | 非线性分段常数 | 核化非线性 |
| 特征选择 | 无 | **Lasso 可精确清零** | 隐式（分裂选择） | 无 |
| 共线性处理 | 系数不稳定 | **显式——L2 收缩 + L1 稀疏** | 鲁棒 | 核隐式处理 |
| 模型复杂度控制 | 固定（特征数） | **惩罚力度 α + l1_ratio** | max_depth / min_samples_split | C + ε + γ |
| 需要标准化 | 否（当前） | **是——惩罚项对尺度敏感** | 否 | **是** |
| 训练方式 | SVD 闭式解 | **坐标下降 / 闭式解** | CART 贪心递归 | SMO 类凸优化 |
| 输出模型数 | 1 | **3（Lasso + Ridge + ElasticNet）** | 1 | 1 |
| 教学定位 | 回归起点 | **线性回归的升级——约束与选择的艺术** | 非线性对比 | 核方法对比 |

## 文件导航

| 文件 | 内容 | 核心问题 |
|---|---|---|
| [01-mathematics](01-mathematics.md) | L1/L2 罚项的数学形式、坐标下降、近端梯度 | L1 惩罚为什么产生稀疏解？ |
| [02-data](02-data.md) | 糖尿病数据 + 共线列 + 噪声列——三层特征结构 | 为什么正则化回归数据需要共线和噪声特征？ |
| [03-intuition](03-intuition.md) | 收缩直觉、稀疏性直觉、L1 vs L2 几何直觉 | 为什么 L1 收缩到零而 L2 只收缩不归零？ |
| [04-model](04-model.md) | `trainRegularizationModels` 三模型字典构建 | Lasso/Ridge/ElasticNet 的超参数各控制什么？ |
| [05-training-and-prediction](05-training-and-prediction.md) | 标准化 → 训练三个模型 → 分别预测 | 为什么正则化必须先标准化？ |
| [06-evaluation](06-evaluation.md) | 系数打印 + 近零计数 + 多模型对比 | 三个模型的系数稀疏性如何对比？ |
| [07-implementation](07-implementation.md) | PipelineSpec 多模型配置 + 运行器多模型分支 | 运行器如何处理 `multiModel=True`？ |
| [08-exercises-and-references](08-exercises-and-references.md) | 自检问题 + 动手练习 + 参考文献 | 调 α 观察系数归零过程 |

## 学习路线

1. **先看线性回归**：正则化回归是线性回归的约束版本——理解 OLS 后再看约束才有意义。
2. **理解标准化必要性**：正则化回归是本仓库回归分册中第一个**强制标准化**的模型——这是它与线性回归和决策树回归最关键的工程差异。
3. **三模型并列对比**：Lasso（L1）→ 稀疏、Ridge（L2）→ 收缩、ElasticNet→ 混合——始终以对比视角理解三者。
4. **关注近零系数**：`np.sum(np.abs(coef) < 1e-3)` 是正则化回归独有的诊断指标——其他回归模型没有"系数清零"这个概念。
