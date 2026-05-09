---
title: 项目架构 — 概览
outline: deep
---

# 项目架构

## 本章目标

1. 理解项目的整体目录结构和模块分层。
2. 建立从源码到文档的全局导航。

本项目是一个机器学习算法教学代码库，覆盖分类、回归、聚类、降维、概率模型和集成学习六大任务类型。每个算法配有 9 篇文档，代码遵循**声明式流水线架构**。

**核心哲学**：显式优于隐式，教学清晰度优于工程复用度。所有步骤（加载、切分、预处理、训练、评估、可视化）在 Runner 中显式编排。

---

## 1. 目录结构

```
Machine-Learning-Algorithms/
├── main.py                          # CLI 统一入口
├── config.py                        # 路径常量与配置
├── src/
│   └── mlAlgorithms/
│       ├── core/                    # 核心抽象
│       │   ├── pipelineSpec.py      #   PipelineSpec —— 流水线声明
│       │   ├── datasetSpec.py       #   DatasetSpec —— 数据集声明
│       │   ├── registry.py          #   Registry —— 简单注册表
│       │   ├── runContext.py        #   RunContext —— 运行时上下文
│       │   ├── runResult.py         #   RunResult —— 运行结果
│       │   ├── taskTypes.py         #   枚举：TaskType / RunnerType / DataKind
│       │   └── artifactManager.py   #   产物路径管理
│       ├── catalog/                 # 注册表（声明式配置聚合）
│       │   ├── pipelines.py         #   PIPELINE_REGISTRY —— 所有流水线声明
│       │   └── datasets.py          #   DATASET_REGISTRY —— 所有数据集声明
│       ├── datasets/                # 数据层
│       │   ├── datasetCatalog.py    #   统一构建所有 DatasetSpec
│       │   ├── tabular/             #   表格数据工厂
│       │   └── sequence/            #   序列数据工厂（HMM）
│       ├── training/                # 训练层
│       │   ├── classification/      #   分类模型
│       │   ├── regression/          #   回归模型
│       │   ├── clustering/          #   聚类模型
│       │   ├── dimensionality/      #   降维模型
│       │   └── probabilistic/       #   概率模型（GMM / HMM）
│       ├── workflows/               # 运行器层
│       │   ├── executor.py          #   按 RunnerType 分发
│       │   ├── baseRunner.py        #   共享辅助函数
│       │   ├── classificationRunner.py
│       │   ├── regressionRunner.py
│       │   ├── clusteringRunner.py
│       │   ├── dimensionalityRunner.py
│       │   └── probabilisticRunner.py
│       ├── evaluation/              # 评估层
│       ├── visualization/           # 可视化层
│       │   ├── data/                #   训练前数据可视化
│       │   └── result/              #   训练后结果可视化
│       └── analysis/                # 数据探索层
├── docs/                            # 文档
│   ├── appendix/                    #   项目架构（本目录）
│   ├── foundations/                 #   基础库教程
│   ├── classification/              #   分类算法文档
│   ├── regression/                  #   回归算法文档
│   ├── clustering/                  #   聚类算法文档
│   ├── ensemble/                    #   集成学习文档
│   ├── dimensionality/              #   降维算法文档
│   └── probabilistic/               #   概率模型文档
└── outputs/                         # 运行产物（图像/报告）
```

---

## 2. 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                       main.py (CLI)                         │
│          list / run / suite / analyze                       │
└─────────────────────┬───────────────────────────────────────┘
                      │ 查找
          ┌───────────┴───────────┐
          │   PIPELINE_REGISTRY   │  ← PipelineSpec × 20
          │   DATASET_REGISTRY    │  ← DatasetSpec × 20
          └───────────┬───────────┘
                      │ 分发
          ┌───────────┴───────────┐
          │     executor.py       │  ← 按 RunnerType 路由
          └───────────┬───────────┘
                      │
     ┌──────┬──────┬──┴───┬──────┬──────────┐
     │      │      │      │      │          │
  Class.  Regr.  Clust.  Dim.  Prob.    (Runner)
     │      │      │      │      │
     └──────┴──────┴──────┴──────┴──────────┘
                      │
          ┌───────────┴───────────┐
          │  buildRunContext()     │  ← 数据加载 + 探索
          │  makeSplit()           │  ← 切分
          │  applyPreprocessor()   │  ← 标准化
          │  callTrainer()         │  ← 训练
          │  evaluate()            │  ← 评估 + 打印
          │  plot*()               │  ← 可视化
          └───────────────────────┘
```

---

## 3. 文档导航

| 文件 | 内容 |
|---|---|
| [01-核心抽象](01-core-abstractions.md) | `PipelineSpec`、`DatasetSpec`、`Registry`、`RunContext`、`RunResult`、枚举类型 |
| [02-模块分层](02-module-layers.md) | 数据层、训练层、流水线注册层、运行器层、评估层、可视化层 |
| [03-CLI 与流水线](03-cli-and-pipelines.md) | CLI 命令用法、全部 20 条流水线速览表 |
| [04-扩展指南](04-extending.md) | 如何新增算法、PipelineSpec 字段填写指引、关键设计决策 |
