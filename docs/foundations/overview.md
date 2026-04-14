---
title: 库生态总览
outline: deep
---

# 库生态总览

本页说明 Python 机器学习常见库的分工、与仓库 [`Basic/`](https://github.com/NayukiChiba/Machine-Learning-Algorithms/tree/main/Basic) 教程的对应关系，以及阅读顺序建议。

## 各库角色

| 库 | 作用 | 分册入口 |
|----|------|----------|
| NumPy | 同质数组、向量化、基础线代 | [01 基础](/foundations/numpy/01-basics) |
| pandas | 表格、索引、清洗、时间序列 | [Pandas 01](/foundations/pandas/01-basics) |
| SciPy | 分布、检验、优化、稀疏、空间结构 | [SciPy 01](/foundations/scipy/01-basics) |
| Matplotlib / Seaborn | 绑图与统计图 | [Matplotlib 基础](/foundations/visualization/01-matplotlib-basics) |
| scikit-learn | 估计器接口、流水线、搜索、指标 | [sklearn 入门](/foundations/sklearn/01-basics) |

## 依赖版本说明

文档中的 API 签名表由本地 `inspect` 生成时，与当前环境版本一致。请以各库 **官方稳定文档** 为准核对：

- [NumPy](https://numpy.org/doc/stable/)
- [pandas](https://pandas.pydata.org/docs/)
- [SciPy](https://docs.scipy.org/doc/scipy/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/stable/)
- [Seaborn](https://seaborn.pydata.org/)

仓库根目录 [requirements.txt](https://github.com/NayukiChiba/Machine-Learning-Algorithms/blob/main/requirements.txt) 列出项目直接依赖；`xgboost` 等若未列出需自行安装。

## 建议阅读顺序

1. [符号与记号](/appendix/notation)、[术语表](/appendix/glossary)
2. NumPy → Pandas →（按需 SciPy）→ sklearn 预处理与 Pipeline
3. 各 [算法分册](/classification/knn/)（从总览进入）

## 算法文档结构

每个算法独立目录，含：数学原理、数据、思路、模型、训练、评估、实现、练习与文献。从任意算法 [总览](/classification/knn/) 页可看到完整阅读路线。
