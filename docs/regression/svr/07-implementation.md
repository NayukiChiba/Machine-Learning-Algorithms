---
title: SVR 支持向量回归 — 工程实现
outline: deep
---

# 工程实现

> 对应代码：`pipelines/regression/svr.py`、`model_training/regression/svr.py`
>  
> 运行方式：`python -m pipelines.regression.svr`

## 本章目标

1. 从工程角度看清 SVR 流水线的完整调用链。
2. 明确当前实现依赖了哪些模块、会产生哪些输出。
3. 理解本仓库 SVR 代码采用的“薄封装 + 可视化诊断”结构。

## 本仓库路径速览

| 组件 | 路径 | 作用 |
|---|---|---|
| 端到端流水线 | `pipelines/regression/svr.py` | 组织数据、训练、预测和可视化 |
| 训练封装 | `model_training/regression/svr.py` | 封装 `SVR` 的训练过程 |
| 数据入口 | `data_generation/__init__.py` | 导出 `svr_data` |
| 数据生成 | `data_generation/regression.py` | 定义 `RegressionData.svr()` |
| 残差可视化 | `result_visualization/residual_plot.py` | 绘制残差分析图 |
| 学习曲线可视化 | `result_visualization/learning_curve.py` | 绘制学习曲线 |

## 1. 流水线调用链

`run()` 是当前 SVR 工程实现的主入口，调用链严格对应源码：

1. `data = svr_data.copy()`
2. `X = data.drop(columns=['price'])`，`y = data['price']`
3. `train_test_split(..., test_size=0.2, random_state=42)`
4. `StandardScaler().fit_transform(X_train)` 与 `transform(X_test)`
5. `model = train_model(X_train_s, y_train)`
6. `y_pred = model.predict(X_test_s)`
7. `plot_residuals(...)`
8. `plot_learning_curve(SVR(kernel='rbf', C=10.0), ..., scoring='r2')`

### 示例代码

```python
def run():
    data = svr_data.copy()
    X = data.drop(columns=["price"])
    y = data["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = train_model(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
```

### 理解重点

- 整条流水线很短，说明当前实现刻意保持了教学和调试友好性。
- 训练逻辑集中在 `train_model(...)`，而 `run()` 负责把数据、模型和可视化串起来。

## 2. 关键函数参数

### 参数速览（本节）

适用函数（分项）：

1. `run()`
2. `train_model(X_train, y_train, C=10.0, epsilon=0.1, kernel='rbf', gamma='scale', degree=3, coef0=0.0)`
3. `SVR(kernel='rbf', C=10.0)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回值（`run()`） | `None` | 执行完整 SVR 回归流程并打印日志 |
| `X_train` | `X_train_s` | 传给训练函数的标准化训练特征 |
| `y_train` | `y_train` | 训练标签 |
| `C` | `10.0` | 惩罚系数 |
| `epsilon` | `0.1` | 不敏感区间 |
| `kernel` | `'rbf'` | 默认核函数 |
| `gamma` | `'scale'` | 默认核函数系数 |
| `degree` | `3` | 多项式核阶数 |
| `coef0` | `0.0` | 多项式 / sigmoid 核常数项 |
| `SVR(kernel='rbf', C=10.0)` | 学习曲线专用模型 | 用于 `plot_learning_curve(...)` 的未训练估计器 |

### 理解重点

- `run()` 本身不接收参数，因此当前工程入口是固定配置。
- 如果想把参数做成可配置入口，最自然的扩展点就是 `run()` 和 `train_model(...)` 之间。

## 3. 运行方式与控制台输出

### 示例代码

```bash
python -m pipelines.regression.svr
```

### 结果输出（关键）

```text
============================================================
SVR 回归流水线
============================================================
...
模型训练完成
kernel: rbf
C: 10.0
epsilon: 0.1
gamma: scale
支持向量数量: <n_sv>
残差图已保存至: <残差图路径>
学习曲线已保存至: <学习曲线路径>
============================================================
SVR 流水线完成！
============================================================
```

### 理解重点

- 控制台信息既包含流程开始/结束标识，也包含训练阶段的关键参数打印。
- 训练耗时由 `timer(name='SVR 训练耗时')` 输出，便于观察不同参数下的训练成本。

## 4. 输出产物

运行结束后，当前实现会生成两类图像文件。

### 参数速览（本节）

适用路径表达式（分项）：

1. `outputs/<model_name>/residual_plot.png`
2. `outputs/<model_name>/learning_curve.png`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `dataset_name` | `'svr'` | 输出目录名 |
| `model_name` | `'svr'` | 输出文件名前缀 |
| 输出文件 1 | `svr_residual.png` | 残差分析图 |
| 输出文件 2 | `svr_learning_curve.png` | 学习曲线图 |

### 理解重点

- 图像输出目录由配置常量控制，SVR 流水线只负责按 `dataset_name` 和 `model_name` 组织文件名。
- 因此这个实现天然适合在多个模型之间复用可视化模块。

## 5. 依赖库与模块关系

当前 SVR 工程实现直接依赖以下对象：

1. `sklearn.svm.SVR`
2. `sklearn.model_selection.train_test_split`
3. `sklearn.preprocessing.StandardScaler`
4. `data_generation.svr_data`
5. `result_visualization.residual_plot.plot_residuals`
6. `result_visualization.learning_curve.plot_learning_curve`

### 理解重点

- 当前结构把“数据生成”“训练”“可视化”明确分层，便于替换某一个模块而不影响其他部分。
- 这也是为什么文档里可以按“数据构成 / 模型构建 / 训练预测 / 评估 / 工程实现”拆分章节。

## 常见坑

1. 误以为 `run()` 会自动接受外部参数，实际上当前入口是固定配置。
2. 忽略学习曲线里重新创建了一个新的 `SVR(...)` 实例，而不是复用前面训练好的模型。
3. 只关注训练结果，不关注图像输出路径是否正确生成。

## 小结

- 本仓库 SVR 工程实现采用的是“薄封装 + 明确日志 + 图形诊断”的结构。
- `pipelines/regression/svr.py` 负责编排流程，`model_training/regression/svr.py` 负责训练，两个可视化模块负责结果解释。
- 这种结构非常适合教学、调试和后续逐步扩展。
