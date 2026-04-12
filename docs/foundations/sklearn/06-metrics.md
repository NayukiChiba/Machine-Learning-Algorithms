---
title: 指标
outline: deep
---

# 指标

> 对应脚本：`Basic/ScikitLearn/06_metrics.py`
> 运行方式：`python Basic/ScikitLearn/06_metrics.py`（仓库根目录）

## 本章目标

1. 掌握分类任务的核心指标：准确率、精确率、召回率、F1。
2. 理解混淆矩阵与分类报告的阅读方式。
3. 学会使用 ROC/PR 曲线与 AUC 评价概率输出质量。
4. 掌握多分类指标中的 `average` 与 `multi_class` 参数。
5. 学会回归指标与自定义评分函数的实践写法。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `accuracy_score(...)` | 分类准确率 | `demo_classification_metrics` |
| `precision_score(...)` / `recall_score(...)` / `f1_score(...)` | 分类核心指标 | `demo_classification_metrics` |
| `confusion_matrix(...)` | 生成混淆矩阵 | `demo_confusion_matrix` |
| `classification_report(...)` | 输出分类统计摘要 | `demo_confusion_matrix` |
| `roc_curve(...)` / `roc_auc_score(...)` | ROC 曲线与 AUC | `demo_roc_auc` |
| `precision_recall_curve(...)` / `auc(...)` | PR 曲线与面积 | `demo_roc_auc` |
| `r2_score(...)` / `mean_squared_error(...)` / `mean_absolute_error(...)` | 回归指标 | `demo_regression_metrics` |
| `make_scorer(func)` | 自定义评分器 | `demo_custom_scorer` |
| `*Display.from_estimator(...)` | 指标可视化工具类 | `demo_display_tools` |

## 1. 分类指标基础

### 方法重点

- 准确率是全局正确比例，但在类别不平衡下可能失真。
- 精确率关注“预测为正”的可靠性，召回率关注“真实为正”的覆盖率。
- F1 综合平衡精确率与召回率，适合综合评估。

### 参数速览（本节）

1. `accuracy_score(y_true, y_pred)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `y_true` | 测试集真实标签 | 真值向量 |
| `y_pred` | `clf.predict(X_test)` | 预测标签 |
| 返回值 | `float` | 准确率 |

2. `precision_score(y_true, y_pred)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `y_true` | 测试集真实标签 | 真值向量 |
| `y_pred` | `clf.predict(X_test)` | 预测标签 |
| 返回值 | `float` | 精确率 |

3. `recall_score(y_true, y_pred)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `y_true` | 测试集真实标签 | 真值向量 |
| `y_pred` | `clf.predict(X_test)` | 预测标签 |
| 返回值 | `float` | 召回率 |

4. `f1_score(y_true, y_pred)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `y_true` | 测试集真实标签 | 真值向量 |
| `y_pred` | `clf.predict(X_test)` | 预测标签 |
| 返回值 | `float` | F1 分数 |

### 示例代码

```python
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

cancer = datasets.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
	cancer.data, cancer.target, test_size=0.3, random_state=42
)

clf = LogisticRegression(max_iter=10000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
```

### 结果输出（示例）

```text
准确率 (accuracy): 0.9708
----------------
精确率 (precision): 0.9630
----------------
召回率 (recall): 0.9811
----------------
F1分数: 0.9720
```

### 理解重点

- 高准确率不代表业务可用，需结合误判类型分析。
- 医疗、风控等高风险场景通常优先保证召回率。
- 指标冲突时要回到业务代价函数做决策。

## 2. 混淆矩阵与分类报告

### 方法重点

- 混淆矩阵直接给出 TN、FP、FN、TP，最利于误差归因。
- 分类报告整合 precision/recall/F1/support，便于类别级比较。
- 对二分类任务，FP 与 FN 往往对应不同业务风险。

### 参数速览（本节）

1. `confusion_matrix(y_true, y_pred)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `y_true` | 测试集真实标签 | 真值向量 |
| `y_pred` | 模型预测标签 | 预测向量 |
| 返回值 | `ndarray(2,2)` | 二分类混淆矩阵 |

2. `classification_report(y_true, y_pred, target_names=['恶性', '良性'])`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `y_true` | 测试集真实标签 | 真值向量 |
| `y_pred` | 模型预测标签 | 预测向量 |
| `target_names` | `['恶性', '良性']` | 报告中类别显示名称 |
| 返回值 | `str` | 文本统计报告 |

### 示例代码

```python
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred, target_names=["恶性", "良性"]))
```

### 结果输出（示例）

```text
混淆矩阵:
[[ 60   3]
 [  2 106]]
----------------
TN=60, FP=3
----------------
FN=2, TP=106
----------------
分类报告:
			  precision    recall  f1-score   support
		  恶性       0.97      0.95      0.96        63
		  良性       0.97      0.98      0.97       108
```

### 理解重点

- 混淆矩阵是阈值调优与误判成本分析的起点。
- 观察 support 可避免被样本量差异误导。
- 模型上线前应把业务关注类别单独做阈值评估。

## 3. ROC/PR 与 AUC

### 方法重点

- ROC 关注 TPR 与 FPR 的权衡，PR 更适合类别不平衡场景。
- AUC 是曲线面积摘要，便于模型快速对比。
- 这类指标依赖概率输出，通常使用 `predict_proba`。

### 参数速览（本节）

1. `roc_curve(y_true, y_score)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `y_true` | 测试集真实标签 | 真值向量 |
| `y_score` | `clf.predict_proba(X_test)[:, 1]` | 正类概率 |
| 返回值 | `fpr, tpr, thresholds` | ROC 曲线坐标 |

2. `roc_auc_score(y_true, y_score)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `y_true` | 测试集真实标签 | 真值向量 |
| `y_score` | `clf.predict_proba(X_test)[:, 1]` | 正类概率 |
| 返回值 | `float` | ROC AUC 值 |

3. `precision_recall_curve(y_true, y_score)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `y_true` | 测试集真实标签 | 真值向量 |
| `y_score` | `clf.predict_proba(X_test)[:, 1]` | 正类概率 |
| 返回值 | `precision, recall, thresholds` | PR 曲线坐标 |

4. `auc(x, y)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `x` / `y` | `recall` / `precision` | 曲线横纵坐标 |
| 返回值 | `float` | 曲线面积 |

### 示例代码

```python
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc

y_proba = clf.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)
fpr, tpr, _ = roc_curve(y_test, y_proba)
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)

print(roc_auc, pr_auc)
```

### 结果输出（示例）

```text
ROC AUC: 0.9978
----------------
FPR 范围: [0.00, 1.00]
----------------
TPR 范围: [0.00, 1.00]
----------------
PR AUC: 0.9981
```

### 理解重点

- ROC AUC 高通常说明排序能力强，但阈值仍需业务化设定。
- 正负样本极不平衡时，PR 曲线更有参考价值。
- 概率未校准时，曲线仍可用但阈值解释要谨慎。

## 4. 多分类指标

### 方法重点

- 多分类 F1 的 `average` 方式会影响结论。
- `micro` 强调总体样本，`macro` 强调类别公平，`weighted` 兼顾样本量。
- 多分类 AUC 常用 `ovr` 与 `ovo` 两种策略。

### 参数速览（本节）

1. `f1_score(y_true, y_pred, average='micro'|'macro'|'weighted')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `y_true` / `y_pred` | 多分类真值与预测 | 输入标签 |
| `average` | `'micro'`、`'macro'`、`'weighted'` | 多分类 F1 聚合方式 |
| 返回值 | `float` | 聚合后的 F1 |

2. `roc_auc_score(y_true, y_proba, multi_class='ovr'|'ovo')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `y_true` | 多分类真值 | 输入标签 |
| `y_proba` | `clf.predict_proba(X_test)` | 每类概率矩阵 |
| `multi_class` | `'ovr'`、`'ovo'` | 多分类 AUC 策略 |
| 返回值 | `float` | 聚合后的 AUC |

### 示例代码

```python
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
	iris.data, iris.target, test_size=0.3, random_state=42
)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

for avg in ["micro", "macro", "weighted"]:
	print(avg, f1_score(y_test, y_pred, average=avg))

for strategy in ["ovr", "ovo"]:
	print(strategy, roc_auc_score(y_test, y_proba, multi_class=strategy))
```

### 结果输出（示例）

```text
F1 不同 average:
  micro: 1.0000
  macro: 1.0000
  weighted: 1.0000
----------------
多分类 ROC AUC:
  ovr: 1.0000
  ovo: 1.0000
```

### 理解重点

- 类别不均衡时应优先关注 `macro` 或类别级报告。
- `ovr` 与 `ovo` 的差异在类别增多时更明显。
- 多分类概率质量可继续用校准曲线补充验证。

## 5. 回归指标

### 方法重点

- 回归指标关注误差大小与解释能力两个维度。
- `R²` 衡量解释比例，`MSE/RMSE/MAE` 衡量误差幅度。
- 业务上常用 MAE（可解释）与 RMSE（对大误差更敏感）组合。

### 参数速览（本节）

1. `r2_score(y_true, y_pred)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `y_true` | 测试集真实值 | 真值向量 |
| `y_pred` | 回归预测值 | 预测向量 |
| 返回值 | `float` | 拟合优度 |

2. `mean_squared_error(y_true, y_pred)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `y_true` | 测试集真实值 | 真值向量 |
| `y_pred` | 回归预测值 | 预测向量 |
| 返回值 | `float` | 均方误差 |

3. `mean_absolute_error(y_true, y_pred)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `y_true` | 测试集真实值 | 真值向量 |
| `y_pred` | 回归预测值 | 预测向量 |
| 返回值 | `float` | 平均绝对误差 |

### 示例代码

```python
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
	diabetes.data, diabetes.target, test_size=0.3, random_state=42
)

reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)

print(r2_score(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
print(mse, np.sqrt(mse), mean_absolute_error(y_test, y_pred))
```

### 结果输出（示例）

```text
R^2: 0.4773
----------------
MSE: 2821.7509
----------------
RMSE: 53.1202
----------------
MAE: 42.7941
```

### 理解重点

- R² 可比较解释力，但不代表误差绝对可接受。
- RMSE 对异常误差更敏感，适合强调大偏差风险的场景。
- 指标应与业务单位和容忍阈值一起解释。

## 6. 自定义评分函数

### 方法重点

- `make_scorer` 可把业务目标映射为可优化的评分函数。
- 自定义评分可直接接入交叉验证与搜索器。
- 适合将多指标加权成单一决策目标。

### 参数速览（本节）

1. `make_scorer(custom_score)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 自定义函数 | `0.7 * precision + 0.3 * recall` | 业务加权目标 |
| 返回值 | `callable` | sklearn 可识别的评分器 |

2. `cross_val_score(estimator, X, y, cv=5, scoring=custom_scorer)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `cv` | `5` | 五折交叉验证 |
| `scoring` | `custom_scorer` | 自定义评分器 |
| 返回值 | `ndarray` | 每折自定义得分 |

### 示例代码

```python
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import cross_val_score

cancer = datasets.load_breast_cancer()
clf = LogisticRegression(max_iter=10000)

def custom_score(y_true, y_pred):
	p = precision_score(y_true, y_pred)
	r = recall_score(y_true, y_pred)
	return 0.7 * p + 0.3 * r

scorer = make_scorer(custom_score)
scores = cross_val_score(clf, cancer.data, cancer.target, cv=5, scoring=scorer)
print(scores)
print(scores.mean())
```

### 结果输出（示例）

```text
自定义评分各折: [0.9762 0.9861 0.9732 0.9815 0.9780]
----------------
平均: 0.9790
```

### 理解重点

- 自定义指标应先做单元测试，确保方向与数值正确。
- 若指标不可导或不稳定，搜索过程会更噪声化。
- 评分函数最好与线上 KPI 口径保持一致。

## 7. Display 可视化工具

### 方法重点

- sklearn 提供多种 Display 类快速生成标准评估图。
- 工具类可减少手写绘图代码，统一风格。
- 适合评估报告与 notebook 讲解场景。

### 参数速览（本节）

1. `ConfusionMatrixDisplay.from_estimator(model, X, y)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `model` | `LogisticRegression` | 已训练模型 |
| `X` / `y` | 测试数据与标签 | 绘图输入 |
| 返回值 | `Display` 对象 | 混淆矩阵可视化对象 |

2. `ConfusionMatrixDisplay.from_predictions(y_true, y_pred)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `y_true` / `y_pred` | 真值与预测 | 绘图输入 |
| 返回值 | `Display` 对象 | 混淆矩阵可视化对象 |

3. `RocCurveDisplay.from_estimator(model, X, y)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `model` | 已训练模型 | 具备概率输出能力 |
| `X` / `y` | 测试数据与标签 | 绘图输入 |
| 返回值 | `Display` 对象 | ROC 曲线对象 |

4. `PrecisionRecallDisplay.from_estimator(model, X, y)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `model` | 已训练模型 | 具备概率输出能力 |
| `X` / `y` | 测试数据与标签 | 绘图输入 |
| 返回值 | `Display` 对象 | PR 曲线对象 |

5. `DecisionBoundaryDisplay.from_estimator(model, X)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `model` | 已训练模型 | 分类器 |
| `X` | 特征矩阵 | 绘图输入 |
| 返回值 | `Display` 对象 | 决策边界可视化对象 |

### 示例代码

```python
print("可用的 Display 类:")
print("  - ConfusionMatrixDisplay.from_estimator(model, X, y)")
print("  - ConfusionMatrixDisplay.from_predictions(y_true, y_pred)")
print("  - RocCurveDisplay.from_estimator(model, X, y)")
print("  - RocCurveDisplay.from_predictions(y_true, y_proba)")
print("  - PrecisionRecallDisplay.from_estimator(model, X, y)")
print("  - DecisionBoundaryDisplay.from_estimator(model, X)")
```

### 结果输出（示例）

```text
可用的 Display 类:
  - ConfusionMatrixDisplay.from_estimator(model, X, y)
  - ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
  - RocCurveDisplay.from_estimator(model, X, y)
  - RocCurveDisplay.from_predictions(y_true, y_proba)
  - PrecisionRecallDisplay.from_estimator(model, X, y)
  - DecisionBoundaryDisplay.from_estimator(model, X)
```

### 理解重点

- Display 类降低了评估可视化门槛，适合快速对比实验。
- 工具类默认图形足够标准，优先使用再做样式增强。
- 图形结论应与数值指标共同解读，避免视觉偏差。

## 常见坑

1. 直接用类别预测做 ROC/PR，导致曲线信息失真。
2. 多分类任务只看总体准确率，忽略少数类别表现。
3. 自定义评分函数方向写反，搜索结果被误导。

## 小结

- 指标体系应围绕业务目标构建，不应只追求单一高分。
- 建议建立固定评估模板：分类指标 + 混淆矩阵 + 曲线 + 业务加权分。
- 上游调参与交叉验证流程可参考 [模型选择](/foundations/sklearn/05-model-selection)。
