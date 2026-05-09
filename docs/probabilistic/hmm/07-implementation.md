---
title: HMM — 工程实现
outline: deep
---

# 工程实现

## 本章目标

1. 理解 HMM 流水线的模块分层——数据生成层、模型训练层、流水线编排层（无可视化层）。
2. 理清 `run()` 内部的函数调用链——HMM 是本仓库最简流水线，无标准化/切分/可视化。
3. 理解 HMM 与 GMM（EM）在工程实现上的关键差异——序列数据、离散观测、双备份依赖。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `ProbabilisticData.hmm()` | 方法 | 手动参数化的 HMM 序列数据——含真实隐状态和观测 |
| `train_model(...)` | 函数 | 构建并训练 `CategoricalHMM`——含双备份可选依赖检查 |
| `run()` | 函数 | 序列模型流水线编排——4 步完成数据整形、训练、Viterbi 解码和评估 |
| `model.predict(X_obs, lengths)` | 方法 | Viterbi 解码——全局最优隐状态路径 |
| `model.score(X_obs, lengths)` | 方法 | Forward 算法——对数似然（可用于 diagnostic） |

## 1. 模块分层总览

### 参数速览

| 层 | 文件 | 职责 | 输出 |
|---|---|---|---|
| 数据生成层 | `data_generation/probabilistic.py` | 手动参数化生成 HMM 序列数据并导出 `hmm_data` | 全局 `DataFrame`（300 行 × 3 列） |
| 模型训练层 | `model_training/probabilistic/hmm.py` | 封装 `CategoricalHMM` 训练——含双备份可选依赖处理 | `CategoricalHMM` 模型对象 |
| 流水线编排层 | `pipelines/probabilistic/hmm.py` | 串联数据整形、训练、Viterbi 解码和准确率评估——端到端入口 | 终端日志 + 准确率 + 转移矩阵 |
| 可视化层 | **无** | HMM 序列数据不适合散点图/矩阵图——终端文本输出已足够 | — |

### 理解重点

- HMM 是本仓库唯一**没有可视化层**的流水线——序列推断的评估更适合用终端文本（准确率 + 转移矩阵）。
- 训练层有双备份依赖处理——`CategoricalHMM`（hmmlearn 0.3+）/ `MultinomialHMM`（旧版），任一可用即可。
- 与 GMM 共享 `data_generation/probabilistic.py`——两者均为概率模型，数据生成器统一管理。

## 2. `run()` 内部的函数调用链

### 参数速览

| 序号 | 调用 | 输入 | 输出 | 目的 |
|---|---|---|---|---|
| 1 | `hmm_data.copy()` | — | `DataFrame`，形状 `(300, 3)` | 避免修改全局变量 |
| 2 | `data["obs"].values.astype(int)` | `DataFrame` | `ndarray`，`(300,)` | 提取离散观测序列 |
| 3 | `obs.reshape(-1, 1)` | `ndarray` | `ndarray`，`(300, 1)` | 整形为 hmmlearn 要求的列向量 |
| 4 | `[len(obs)]` | — | `list[int]` | 序列长度——单条 300 步 |
| 5 | `data["state_true"].values.astype(int)` | `DataFrame` | `ndarray`，`(300,)` | 提取真实隐状态——仅用于评估 |
| 6 | `train_model(X_obs, lengths)` | `(ndarray, list)` | `CategoricalHMM` | Baum-Welch 训练 |
| 7 | `model.predict(X_obs, lengths)` | `(ndarray, list)` | `ndarray`，`(300,)` | Viterbi 解码 |
| 8 | `np.mean(states_pred == y_true)` | `(ndarray, ndarray)` | `float` | 逐步准确率计算 |
| 9 | `model.transmat_.round(3)` | 属性访问 | — | 打印学习到的转移矩阵 |

### 理解重点

- 步骤 3（`reshape`）是 hmmlearn 接口特有的数据整形——观测必须为列向量。
- 步骤 6 内部触发可选依赖检查——如果 `hmmlearn` 未安装会抛出 `ImportError`。
- 与 GMM 流水线的关键差异：无 `StandardScaler`（离散观测无需缩放）、无 `plot_clusters`（序列不可散点图化）。

## 3. 数据依赖关系

```
hmm_data (全局 DataFrame)
    │
    ├─→ obs = data["obs"].values ──→ reshape(-1, 1) ──→ X_obs ──┐
    ├─→ lengths = [len(obs)] ────────────────────────────────────┤
    ├─→ y_true = data["state_true"].values ─────────────────────┐│
    │                                                             ││
    │   train_model(X_obs, lengths) ──→ model                    ││
    │      │                                                      ││
    │      └─→ model.predict(X_obs, lengths) ──→ states_pred ──┐ ││
    │                                                            │ ││
    │   accuracy = np.mean(states_pred == y_true) ←─────────────┘ ││
    │   print(model.transmat_) ←──────────────────────────────────┘│
    │                                                               │
    └───────────────────────────────────────────────────────────────┘
```

### 理解重点

- `y_true` 仅用于评估——不经过训练模块，直接与 `states_pred` 对比。
- `lengths` 与 `X_obs` 同时传入 `train_model` 和 `model.predict`——HMM 必须知道序列边界。
- 这是本仓库最简单的数据依赖图——无标准化分支、无可视化分支、无切分分支。

## 4. 输出一览

### 参数速览

| 输出项 | 路径/位置 | 格式 | 说明 |
|---|---|---|---|
| 隐状态准确率 | 标准输出 | 文本 `float` | Viterbi 路径与真实状态的逐步匹配率 |
| 转移矩阵 | 标准输出 | 文本 `ndarray` | 学习到的 3×3 转移矩阵（行和为 1） |
| 终端日志 | 标准输出 | 文本 | 训练超参数 + 运行耗时 |

### 示例代码

```bash
python -m pipelines.probabilistic.hmm
```

### 输出

```text
============================================================
HMM 流水线
============================================================
模型训练完成
n_components: 3
n_iter: 100
tol: 0.001
模型训练耗时: 0.08s

隐状态预测准确率: 0.8933
转移矩阵:
[[0.782  0.176  0.042 ]
 [0.215  0.582  0.203 ]
 [0.118  0.223  0.659 ]]

============================================================
HMM 流水线完成！
============================================================
```

### 理解重点

- HMM **无任何文件输出**——所有评估结果以终端文本呈现，是本仓库唯一纯终端输出的流水线。
- 训练耗时极短（~0.08s）——300 步 × 3 状态，Baum-Welch 在此规模上收敛很快。
- 转移矩阵保留 3 位小数——足够直观对比学习结果与真实参数的差异。

## 5. 训练层细节：与 GMM 的对比

| 工程维度 | GMM (EM) | HMM |
|---|---|---|
| 模型类 | `GaussianMixture` | **`CategoricalHMM` / `MultinomialHMM`（双备份）** |
| 依赖 | sklearn 内置 | **`pip install hmmlearn`（可选依赖）** |
| 训练输入 | `fit(X)`——独立样本矩阵 | **`fit(X, lengths)`——序列列向量 + 长度列表** |
| 算法 | EM（E 步: 逐点后验, M 步: 加权更新） | **Baum-Welch（E 步: Forward-Backward, M 步: 计数重估）** |
| 预测 | `predict(X)` + `predict_proba(X)` | **`predict(X, lengths)`（Viterbi）——无 `predict_proba`** |
| 模型属性 | `means_`、`covariances_`、`weights_` | **`transmat_`、`emissionprob_`、`startprob_`** |
| 标准化 | 有 | **无**（离散观测） |
| 可视化 | 聚类图 | **无**（终端文本） |
| 装饰器 | `@print_func_info` + `@timeit` + `timer` | `@print_func_info` + `@timeit` + `timer`——相同 |

### 理解重点

- HMM 的训练层设计是四个概率模型中最特殊的——输入不是 `(X, y)` 或 `(X)`，而是 `(X, lengths)`。
- 双备份依赖是可选的极限容错——无论用户装的是新版还是旧版 hmmlearn，都能正常运行。
- 无 `predict_proba` 但有 `score`（Forward 算法给出对数概率）——两者的评估用途不同。

## 阅读顺序

1. `data_generation/probabilistic.py` — 了解 `hmm()` 的 HMM 序列数据生成逻辑
2. `model_training/probabilistic/hmm.py` — 理解 `CategoricalHMM` 的构建、双备份依赖和 Baum-Welch 训练
3. `pipelines/probabilistic/hmm.py` — 看清序列流水线的端到端流程和终端评估

## 常见坑

1. 在不含 `hmmlearn` 的环境中直接 `from model_training.probabilistic.hmm import train_model`——会抛出 `ImportError`，需先 `pip install hmmlearn`。
2. 忘记将观测 `reshape(-1, 1)`——hmmlearn 的 `fit` 要求观测为 `(n_steps, 1)` 形状。
3. 把 `astype(int)` 漏掉——hmmlearn 可能将 float 观测处理为连续值，触发错误的模型行为。
4. 混淆 `CategoricalHMM` 和 `MultinomialHMM` 的参数——两者基本相同，但类名和包路径不同。

## 小结

- HMM 工程实现遵循三层架构（无可视化层）：数据生成层 → 模型训练层 → 流水线编排层。
- `run()` 是本仓库最简编排函数——4 步核心操作完成数据整形、训练、Viterbi 解码和评估，所有输出均为终端文本。
- 与 GMM 的四个关键工程差异：（1）序列输入 + lengths；（2）离散观测无需标准化；（3）双备份可选依赖；（4）无可视化层（纯终端评估）。
