---
title: 决策树回归 — 评估与诊断
outline: deep
---

# 评估与诊断

> 对应代码：`pipelines/regression/decision_tree.py`、`result_visualization/residual_plot.py`、`result_visualization/feature_importance.py`、`result_visualization/learning_curve.py`
>  
> 相关对象：`plot_residuals(...)`、`plot_feature_importance(...)`、`plot_learning_curve(...)`

## 本章目标

1. 明确当前仓库实际使用了哪些评估手段，而不是泛泛讨论所有回归指标。
2. 理解残差图、特征重要性图和学习曲线分别能帮助我们诊断什么。
3. 明确当前实现没有做哪些数值指标输出，以免误读源码能力边界。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `plot_residuals(...)` | 函数 | 生成预测-真实图和残差分布图 |
| `plot_feature_importance(...)` | 函数 | 生成特征重要性柱状图 |
| `plot_learning_curve(...)` | 函数 | 生成训练得分和验证得分曲线 |
| `feature_importances_` | 属性 | 树模型中各特征的重要性来源 |
| `scoring='r2'` | 参数 | 当前学习曲线使用的评分指标 |

## 1. 当前实现真正做了什么评估

### 参数速览（本节）

适用评估手段（本节）：

1. 残差图
2. 特征重要性图
3. 学习曲线

| 评估方式 | 来源 | 用途 |
|---|---|---|
| 残差图 | `plot_residuals(...)` | 观察测试集上的整体拟合情况与误差分布 |
| 特征重要性图 | `plot_feature_importance(...)` | 观察哪些特征在树分裂中更重要 |
| 学习曲线 | `plot_learning_curve(...)` | 观察训练样本数变化时训练/验证得分走势 |

### 理解重点

- 当前 decision tree 流水线没有显式打印 `MSE`、`MAE`、`RMSE`、`R^2` 等数值指标。
- 这并不表示这些指标不重要，而是说明本仓库当前实现更强调图像化诊断和结构性观察。
- 因此阅读这一分册时，不能把“指标表格”想成已经在源码里实现的内容。

## 2. 残差图是怎么生成的

### 参数速览（本节）

适用函数：`plot_residuals(y_true, y_pred, title='残差分析', dataset_name='default', model_name='model', figsize=(14, 5))`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `y_true` | `y_test` | 测试集真实值 |
| `y_pred` | 模型预测值 | 测试集预测结果 |
| `dataset_name` | `"decision_tree_reg"` | 输出目录名 |
| `model_name` | `"decision_tree"` | 输出文件名前缀 |
| `figsize` | `(14, 5)` | 图像尺寸 |

### 示例代码

```python
residuals = y_true - y_pred

ax1.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5, s=30)
ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidth=0.5, s=30)
ax2.axhline(y=0, color="r", linestyle="--", linewidth=1.5)
```

### 理解重点

- 第一张子图看“预测值是否靠近真实值对角线”，适合粗看拟合效果。
- 第二张子图看“残差是否围绕 0 随机分布”，适合发现系统偏差、异常点和可能的局部拟合不足。
- 对树模型来说，如果残差图表现很碎片化，也可能是在提示树结构过于复杂或局部划分不够稳定。

## 3. 特征重要性图是怎么生成的

### 参数速览（本节）

适用函数：`plot_feature_importance(model, feature_names=None, top_n=None, title='特征重要性', dataset_name='default', model_name='model', figsize=(10, 7))`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `model` | 训练好的决策树模型 | 提供 `feature_importances_` |
| `feature_names` | `list(X.columns)` | 为每个重要性值提供真实列名 |
| `top_n` | `None` | 当前实现展示全部特征 |
| `dataset_name` | `"decision_tree_reg"` | 输出目录名 |
| `model_name` | `"decision_tree"` | 输出文件名前缀 |

### 示例代码

```python
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
```

### 理解重点

- 对当前决策树分册来说，特征重要性图是理解模型行为的关键补充信息。
- 它反映的是特征对树分裂贡献的相对大小，而不是线性模型里的系数。
- 因此“重要性高”并不等于“该特征与房价线性正相关”，两者含义完全不同。

## 4. 学习曲线是怎么生成的

### 参数速览（本节）

适用函数：`plot_learning_curve(model, X, y, cv=5, scoring='accuracy', train_sizes=None, ...)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `model` | `DecisionTreeRegressor(max_depth=6, random_state=42)` | 一个新的未训练模型实例 |
| `X` | `X_train.values` | 使用训练集特征 |
| `y` | `y_train.values` | 使用训练集标签 |
| `cv` | 默认 `5` | 交叉验证折数 |
| `scoring` | `"r2"` | 当前分册使用的评分指标 |
| `train_sizes` | 默认 `0.1` 到 `1.0` 共 10 个点 | 训练样本比例 |

### 示例代码

```python
plot_learning_curve(
    DecisionTreeRegressor(max_depth=6, random_state=42),
    X_train.values,
    y_train.values,
    scoring="r2",
    title="决策树回归 学习曲线",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 学习曲线内部会在不同训练样本规模下重复训练和验证，不是直接复用已经训练好的 `model`。
- 当前分册使用的是 `R^2` 作为学习曲线评分指标，但这个分数不会单独打印成文本表。
- 对树模型来说，学习曲线常常比单张残差图更容易暴露过拟合趋势。

## 5. 看残差图时重点观察什么

### 参数速览（本节）

适用观察点（本节）：

1. 对角线贴合程度
2. 残差围绕 0 的分布
3. 离群点与局部结构

| 现象 | 可能含义 |
|---|---|
| 点大多贴近 `y=x` | 整体预测较稳定 |
| 残差围绕 0 随机散开 | 没有明显系统偏差 |
| 残差整体偏正或偏负 | 模型存在系统性高估或低估 |
| 局部出现成簇误差 | 某些区域分裂效果可能不足 |
| 少量极端散点 | 可能有异常样本或局部难例 |

### 理解重点

- 决策树回归虽然能处理非线性，但并不意味着残差一定理想。
- 如果某些局部区域的样本始终预测不好，残差图常会出现明显聚簇或偏移现象。
- 这也是为什么本分册不能只看树结构和重要性，还要看预测误差分布。

## 6. 看特征重要性图和学习曲线时重点观察什么

### 参数速览（本节）

适用观察点（本节）：

1. 重要性是否集中在少数特征
2. 训练/验证曲线是否分裂过大
3. 两条曲线是否逐渐收敛

| 现象 | 可能含义 |
|---|---|
| 少数特征重要性非常高 | 树主要依赖这些特征做分裂 |
| 训练得分高、验证得分低 | 可能过拟合 |
| 两条曲线都较低 | 可能欠拟合 |
| 两条曲线逐渐收敛 | 模型随样本数增加趋于稳定 |

### 理解重点

- 特征重要性图帮助你理解“模型主要看什么”，但不能替代误差评估。
- 学习曲线帮助你理解“模型复杂度和样本量是否匹配”。
- 只有把“重要性 + 学习曲线 + 残差图”三条线索结合起来，才能更完整地读懂当前树模型。

## 7. 当前实现没有做什么

### 参数速览（本节）

当前源码未包含的内容：

1. 显式数值指标打印
2. 自动调参
3. 树结构可视化导出

| 未实现项 | 当前状态 |
|---|---|
| `MSE` / `MAE` / `RMSE` / `R^2` 打印 | 未在流水线中出现 |
| 超参数搜索 | 未使用 `GridSearchCV` 等 API |
| 树结构图导出 | 未调用 `plot_tree` 或导出 `.dot` |

### 理解重点

- 评估章节必须以源码为准，不能把“决策树常见分析手段”写成“当前仓库已经实现”。
- 当前实现的评估重点是残差图、特征重要性图和学习曲线，而不是文本化指标面板。
- 如果后续扩展这部分，最自然的方向是补数值指标或树结构图，而不是替换现有三类图像。

## 常见坑

1. 只看特征重要性图，不看残差图和学习曲线，误把“重要性高”当成“预测一定好”。
2. 只看残差图，不看学习曲线，错过对过拟合趋势的观察。
3. 误以为当前流水线已经输出了完整的 `R^2`、`MSE` 指标表，实际源码并没有这一步。

## 小结

- 当前决策树回归的评估主线由三部分组成：残差图、特征重要性图和学习曲线。
- 残差图负责看误差分布，特征重要性图负责看模型主要依赖哪些特征，学习曲线负责看复杂度与泛化趋势。
- 只有把这三条线索一起看，才能更完整地理解当前实现的表现。
