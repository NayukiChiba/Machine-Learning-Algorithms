---
title: SVR 支持向量回归 — 评估与诊断
outline: deep
---

# 评估与诊断

> 对应代码：`pipelines/regression/svr.py`、`result_visualization/residual_plot.py`、`result_visualization/learning_curve.py`
>  
> 运行方式：`python -m pipelines.regression.svr`

## 本章目标

1. 明确本仓库 SVR 当前实际使用的评估方式。
2. 理解残差图和学习曲线各自能回答什么问题。
3. 找到与当前源码一致的诊断和调参入口。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `plot_residuals(...)` | 函数 | 绘制预测 vs 真实散点图与残差分布图 |
| `plot_learning_curve(...)` | 函数 | 绘制训练得分与验证得分随样本数变化的曲线 |
| `scoring='r2'` | 参数 | 指定学习曲线使用回归 `R^2` 评分 |
| `y_pred = model.predict(X_test_s)` | 表达式 | 生成用于评估的预测结果 |

## 1. 当前仓库的评估入口

本仓库当前 SVR 流水线没有单独打印 MAE、RMSE 之类的数值指标，而是通过两类图形来做诊断：

1. 残差分析图：`plot_residuals(...)`
2. 学习曲线：`plot_learning_curve(...)`

### 示例代码

```python
y_pred = model.predict(X_test_s)

plot_residuals(
    y_test, y_pred, title="SVR 残差分析", dataset_name=DATASET, model_name=MODEL
)

plot_learning_curve(
    SVR(kernel="rbf", C=10.0),
    X_train_s,
    y_train,
    scoring="r2",
    title="SVR 学习曲线",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 这里的评估重点是“看图判断模型状态”，不是直接输出一个分数就结束。
- 残差图更关注误差结构，学习曲线更关注数据规模与泛化趋势。

## 2. 残差图：`plot_residuals(...)`

### 参数速览（本节）

适用函数：`plot_residuals(y_true, y_pred, title='SVR 残差分析', dataset_name='svr', model_name='svr', figsize=(14, 5))`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `y_true` | `y_test` | 测试集真实值 |
| `y_pred` | `model.predict(X_test_s)` | 测试集预测值 |
| `title` | `'SVR 残差分析'` | 图标题前缀 |
| `dataset_name` | `'svr'` | 输出子目录名 |
| `model_name` | `'svr'` | 输出文件名前缀 |
| `figsize` | `(14, 5)` | 图像尺寸 |
| 返回值 | `None` | 保存图像并在控制台打印保存路径 |

### 示例代码

```python
plot_residuals(
    y_test,
    y_pred,
    title="SVR 残差分析",
    dataset_name="svr",
    model_name="svr",
)
```

### 结果输出（示例）

```text
残差图已保存至: <残差图路径>
```

### 理解重点

- 该函数会生成两个子图：预测值 vs 真实值、预测值 vs 残差。
- 如果点云明显偏离 `y=x` 对角线，说明整体拟合不足。
- 如果残差分布出现弯曲趋势或方差随预测值变化，通常说明当前参数或核函数仍需调整。

## 3. 学习曲线：`plot_learning_curve(...)`

### 参数速览（本节）

适用函数：`plot_learning_curve(model, X, y, cv=5, scoring='r2', title='SVR 学习曲线', dataset_name='svr', model_name='svr')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `model` | `SVR(kernel='rbf', C=10.0)` | 用于生成学习曲线的未训练模型 |
| `X` | `X_train_s` | 训练特征 |
| `y` | `y_train` | 训练标签 |
| `cv` | 默认 `5` | 交叉验证折数 |
| `scoring` | `'r2'` | 回归评分指标 |
| `title` | `'SVR 学习曲线'` | 图标题 |
| `dataset_name` | `'svr'` | 输出子目录名 |
| `model_name` | `'svr'` | 输出文件名前缀 |
| 返回值 | `None` | 保存图像并在控制台打印保存路径 |

### 示例代码

```python
plot_learning_curve(
    SVR(kernel="rbf", C=10.0),
    X_train_s,
    y_train,
    scoring="r2",
    title="SVR 学习曲线",
    dataset_name="svr",
    model_name="svr",
)
```

### 结果输出（示例）

```text
学习曲线已保存至: <学习曲线路径>
```

### 理解重点

- 学习曲线使用的是未训练的 `SVR(kernel='rbf', C=10.0)`，而不是前面已经训练好的模型对象。
- `scoring='r2'` 明确说明这里是在看回归拟合质量，而不是分类准确率。
- 训练曲线和验证曲线差距很大时，往往提示过拟合；两条曲线都低时，往往提示欠拟合。

## 4. 当前实现中的直接调参入口

当前代码中最直接的调参位置有两个：

1. `train_model(...)` 中的 `C`、`epsilon`、`kernel`、`gamma`
2. `plot_learning_curve(...)` 里重新创建的 `SVR(kernel='rbf', C=10.0)`

### 示例代码

```python
model = train_model(X_train_s, y_train, C=20.0, epsilon=0.2, gamma="scale")

plot_learning_curve(
    SVR(kernel="rbf", C=20.0),
    X_train_s,
    y_train,
    scoring="r2",
    title="SVR 学习曲线",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 如果训练模型和学习曲线里的模型参数不一致，图上的诊断就不能完全对应实际训练结果。
- 因此调参时要尽量同步修改这两个位置。

## 常见坑

1. 只看某一张图，不结合残差图和学习曲线一起判断。
2. 修改了 `train_model(...)` 的参数，却忘记同步修改学习曲线里的 `SVR(...)`。
3. 误以为当前仓库已经输出了完整数值指标报告，实际上默认实现主要是图形诊断。

## 小结

- 当前 SVR 评估策略强调“图形诊断优先”。
- 残差图帮助观察误差结构，学习曲线帮助判断欠拟合与过拟合。
- 如果后续要补充 MAE、RMSE、`R^2` 数值打印，最自然的扩展点就在当前流水线预测完成之后。
