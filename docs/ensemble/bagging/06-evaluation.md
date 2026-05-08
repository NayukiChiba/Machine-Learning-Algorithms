---
title: Bagging 集成学习 — 评估与诊断
outline: deep
---

# 评估与诊断

## 本章目标

1. 理解 Bagging 的三项评估输出——混淆矩阵（硬标签）、ROC 曲线（软概率）、OOB 得分（训练内）。
2. 理解 OOB 得分作为泛化估计的内在价值——无需额外划分验证集。
3. 明确当前代码中评估的实现范围——什么叫"已实现"、什么叫"未实现"、以及为什么。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `plot_confusion_matrix(...)` | 函数 | 在测试集上绘制混淆矩阵——评估 80 棵树投票后的硬分类准确率 |
| `plot_roc_curve(...)` | 函数 | 绘制 ROC 曲线——评估概率输出的排序能力 |
| `model.oob_score_` | 属性 | OOB 得分——训练过程中"免费"获得的泛化能力估计 |
| `accuracy` | 概念 | 混淆矩阵对角元素之和 / 总样本数——当前通过混淆矩阵可视化间接呈现 |
| AUC | 概念 | 面积 Under ROC Curve——当前通过 ROC 曲线可视化间接呈现 |

## 1. 混淆矩阵：硬标签评估

混淆矩阵评估的是 `model.predict()` 的输出——80 棵树硬投票的最终分类结果。

### 参数速览

适用 API：`plot_confusion_matrix(y_test, y_pred, title="Bagging 混淆矩阵", dataset_name=DATASET, model_name=MODEL)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_test` | `Series`，形状 `(100,)` | 测试集真实标签 | `y_test` |
| `y_pred` | `ndarray`，形状 `(100,)` | 模型硬投票预测结果 | `model.predict(X_test_s)` |
| `title` | `str` | 图表标题 | `"Bagging 混淆矩阵"` |
| `dataset_name` | `str` | 数据集名称——用于输出路径和标识 | `"bagging"` |
| `model_name` | `str` | 模型名称——用于输出路径区分 | `"bagging"` |
| 输出 | PNG 文件 | 混淆矩阵热力图 | `outputs/bagging/confusion_matrix.png` |

### 示例代码

```python
y_pred = model.predict(X_test_s)

plot_confusion_matrix(
    y_test, y_pred,
    title="Bagging 混淆矩阵",
    dataset_name="bagging",
    model_name="bagging",
)
```

### 理解重点

- 混淆矩阵的四个格子分别对应：
  - 真阳性（TP）——正确预测为类别 1
  - 真阴性（TN）——正确预测为类别 0
  - 假阳性（FP）——错误预测为类别 1
  - 假阴性（FN）——错误预测为类别 0
- 当前高噪声双月牙数据（`noise=0.35`）下——Bagging 的混淆矩阵相比单棵决策树应有明显改善：对角元素更"粗"、非对角元素更"细"。
- 混淆矩阵始终生成——不依赖 `predict_proba`，这是硬投票评估的底线。

## 2. ROC 曲线：软概率评估

ROC 曲线评估的是 `model.predict_proba()` 的输出——80 棵树概率平均后，在不同阈值下的 TPR/FPR 分布。

### 参数速览

适用 API：`plot_roc_curve(y_test, y_scores, title="Bagging ROC 曲线", dataset_name=DATASET, model_name=MODEL)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_test` | `Series`，形状 `(100,)` | 测试集真实标签 | `y_test` |
| `y_scores` | `ndarray`，形状 `(100, 2)` | 模型概率输出——每行两个类别的预测概率 | `model.predict_proba(X_test_s)` |
| `title` | `str` | 图表标题 | `"Bagging ROC 曲线"` |
| `dataset_name` | `str` | 数据集名称 | `"bagging"` |
| `model_name` | `str` | 模型名称 | `"bagging"` |
| 输出 | PNG 文件 | ROC 曲线图 | `outputs/bagging/roc_curve.png` |

### 示例代码

```python
if hasattr(model, "predict_proba"):
    y_scores = model.predict_proba(X_test_s)
    plot_roc_curve(
        y_test, y_scores,
        title="Bagging ROC 曲线",
        dataset_name="bagging",
        model_name="bagging",
    )
```

### 理解重点

- ROC 曲线是条件输出——只有 `predict_proba` 可用时才生成。`BaggingClassifier` 始终支持概率输出（只要基学习器支持），条件判断是防御性工程习惯。
- AUC（Area Under Curve）越接近 1.0，模型的排序能力越强——当前高噪声场景下，Bagging 的 AUC 通常显著高于单棵决策树。
- 概率输出比硬标签更有信息量——混淆矩阵只看最终分类是否正确，ROC 曲线看模型对每个预测的置信度。

## 3. OOB 得分：训练内的泛化估计

OOB 得分是 Bagging 独有的诊断工具——不需要额外划分验证集，在训练过程中"免费"获得。

### 参数速览

| 属性名 | 类型 | 取值范围 | 获取条件 | 含义 |
|---|---|---|---|---|
| `model.oob_score_` | `float` | $[0, 1]$ | `oob_score=True` | 用未参与训练的样本估计的泛化准确率——OOB Score = 1 - OOB Error |

### 示例代码

```python
# 训练时打印（4 位小数）
if oob_score:
    print(f"OOB 得分: {model.oob_score_:.4f}")

# 典型输出
# OOB 得分: 0.8975
```

### 理解重点

- OOB 得分与测试集准确率通常接近——如果差距很大（如 OOB 得分远高于测试集准确率），说明数据分布可能有问题或测试集存在偏差。
- OOB 估计等价于对每个样本做一次留出验证——对每个样本，只用没"见过"它的基学习器做预测。
- 当前源码打印到 4 位小数——提供足够精度用于模型对比。
- OOB 得分只存在于终端日志中——不会生成单独的图表或文件。

## 4. 当前代码已实现 vs 未实现的评估内容

### 已实现

| 评估项 | 输出形式 | 触发条件 |
|---|---|---|
| 混淆矩阵 | PNG 热力图（`outputs/bagging/confusion_matrix.png`） | 始终 |
| ROC 曲线 | PNG 曲线图（`outputs/bagging/roc_curve.png`） | `hasattr(model, "predict_proba")` |
| OOB 得分 | 终端打印（4 位小数） | `oob_score=True` |
| 训练超参数日志 | 终端打印（n_estimators、max_samples 等） | `train_model(...)` 调用 |

### 未实现（以及原因）

| 未实现的评估项 | 原因 |
|---|---|
| 测试集准确率（硬数字打印） | 当前通过混淆矩阵可视化间接呈现——可直接读出对角元素占比 |
| 精确率/召回率/F1 分数 | 教学型代码保持最小范围——混淆矩阵 + ROC 已覆盖二分类评估的核心 |
| 学习曲线（n_estimators 递增 vs 性能） | 需要额外训练开销——教学型流水线保持轻量 |
| 与单棵决策树的性能对比 | 流水线中未训练单棵决策树——但在直觉章节已做定性对比 |
| 基学习器多样性分析 | 属于深度分析——超出了教学型流水线的范围 |
| 决策边界可视化 | 虽然 `predict` 可支持，但当前流水线选择混淆矩阵 + ROC 作为评估重点 |

### 理解重点

- 当前评估策略是"可视化优先 + OOB 补足"——两项图表（混淆矩阵 + ROC）覆盖二分类的核心诊断维度，OOB 得分提供训练内的泛化参考。
- 未实现并非"做不到"——而是教学型流水线有意保持轻量，聚焦于最核心的评估维度。
- 如果需要精确率/召回率/F1，可以在此基础上添加 `sklearn.metrics.classification_report` 一行代码。

## 5. Bagging vs 单模型分类器评估对比

| 评估维度 | Bagging | 单模型分类器（如 SVC、逻辑回归） |
|---|---|---|
| 硬分类评估 | 混淆矩阵——80 棵树投票结果 | 混淆矩阵——单个模型决策边界结果 |
| 概率评估 | ROC 曲线——80 棵树概率取平均 | ROC 曲线——单个模型的概率输出 |
| 训练内诊断 | **OOB 得分**——免费且独有 | 无——单模型没有 Bootstrap 采样 |
| 基学习器诊断 | `estimators_`——可逐个检查 80 棵树 | 无——只有一个模型 |
| 方差诊断 | OOB 得分与测试集准确率的差距反映方差大小 | 无法直接诊断方差 |

### 理解重点

- OOB 得分和 `estimators_` 是 Bagging 独有的诊断优势——单模型分类器没有。
- 混淆矩阵和 ROC 曲线是所有二分类器的共性评估——但 Bagging 的输出来自 80 棵树的聚合，本质上是"委员会"的集体表现。

## 常见坑

1. 只看 OOB 得分不看测试集表现——OOB 得分是泛化能力的参考，但最终评估应以测试集为准。
2. 忽略 OOB 得分与测试集准确率的差距——差距过大说明数据分布或切分方式有问题。
3. 认为混淆矩阵够好了就不需要 ROC 曲线——混淆矩阵只看一个阈值点，ROC 曲线看所有阈值下的表现，信息更全面。
4. 把"未实现"当成"做不到"——大部分"未实现"的评估指标都是一行代码的事。

## 小结

- Bagging 有三项评估输出：混淆矩阵（硬标签）→ ROC 曲线（软概率）→ OOB 得分（训练内免费估计）——三者从不同角度描述模型质量。
- OOB 得分是 Bagging 独有的诊断工具——无需额外划分验证集，利用 Bootstrap 采样天然产生的"没被抽中"样本做无偏估计。
- 当前评估策略是"可视化优先 + OOB 补足"——聚焦于最核心的诊断维度，保持教学型代码的轻量性。
