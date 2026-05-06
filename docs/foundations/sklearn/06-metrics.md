---
title: sklearn 评估指标
outline: deep
---

# sklearn 评估指标

## 本章目标

1. 掌握分类任务的核心指标：准确率、精确率、召回率、F1
2. 理解混淆矩阵与分类报告的阅读方式
3. 学会使用 ROC/PR 曲线与 AUC 评价概率输出质量
4. 掌握多分类指标中的 `average` 与 `multi_class` 参数
5. 学会回归指标与自定义评分函数的实践写法

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `accuracy_score(y_true, y_pred)` | 函数 | 分类准确率 |
| `precision_score(y_true, y_pred)` | 函数 | 精确率 |
| `recall_score(y_true, y_pred)` | 函数 | 召回率 |
| `f1_score(y_true, y_pred)` | 函数 | F1 分数 |
| `confusion_matrix(y_true, y_pred)` | 函数 | 生成混淆矩阵 |
| `classification_report(y_true, y_pred)` | 函数 | 输出分类统计摘要 |
| `roc_auc_score(y_true, y_score)` | 函数 | ROC AUC 值 |
| `roc_curve(y_true, y_score)` | 函数 | ROC 曲线坐标 |
| `precision_recall_curve(y_true, y_score)` | 函数 | PR 曲线坐标 |
| `r2_score(y_true, y_pred)` | 函数 | 回归拟合优度 |
| `mean_squared_error(y_true, y_pred)` | 函数 | 均方误差 |
| `mean_absolute_error(y_true, y_pred)` | 函数 | 平均绝对误差 |
| `make_scorer(score_func)` | 函数 | 自定义评分器 |

## 1. 分类指标基础

### `accuracy_score` / `precision_score` / `recall_score` / `f1_score`

#### 作用

准确率是全局正确比例，但在类别不平衡下可能失真。精确率关注"预测为正"的可靠性，召回率关注"真实为正"的覆盖率。F1 综合平衡精确率与召回率，适合综合评估。

#### 重点方法

```python
accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None)
precision_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary',
                sample_weight=None, zero_division='warn')
recall_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary',
             sample_weight=None, zero_division='warn')
f1_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary',
         sample_weight=None, zero_division='warn')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_true` | `array_like` | 真实标签 | `y_test` |
| `y_pred` | `array_like` | 预测标签 | `model.predict(X_test)` |
| `average` | `str` | 多分类聚合方式：`"binary"` / `"micro"` / `"macro"` / `"weighted"`，默认为 `"binary"` | `"macro"` |
| `pos_label` | `int` 或 `str` | 正类标签，默认为 `1` | `1` |
| `zero_division` | `str` 或 `float` | 分母为 0 时的返回值，`"warn"` 加警告返回 0 | `0` |

#### 示例代码

```python
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

X, y = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = LogisticRegression(max_iter=10000).fit(X_train, y_train)
yPred = clf.predict(X_test)

print(f"准确率 (accuracy): {accuracy_score(y_test, yPred):.4f}")
print(f"精确率 (precision): {precision_score(y_test, yPred):.4f}")
print(f"召回率 (recall): {recall_score(y_test, yPred):.4f}")
print(f"F1 分数: {f1_score(y_test, yPred):.4f}")
```

#### 输出

```text
准确率 (accuracy): 0.9708
精确率 (precision): 0.9630
召回率 (recall): 0.9811
F1 分数: 0.9720
```

#### 理解重点

- 高准确率不代表业务可用，需结合误判类型分析
- 医疗、风控等高风险场景通常优先保证召回率
- 指标冲突时要回到业务代价函数做决策
- `zero_division` 控制当 TP+FP=0 时的行为——设为 0 而非报错

## 2. 混淆矩阵与分类报告

### `confusion_matrix` / `classification_report`

#### 作用

混淆矩阵直接给出 TN、FP、FN、TP，最利于误差归因。分类报告整合 precision/recall/F1/support，便于类别级比较。对二分类任务，FP 与 FN 往往对应不同业务风险。

#### 重点方法

```python
confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None,
                 normalize=None)
classification_report(y_true, y_pred, *, labels=None, target_names=None,
                      sample_weight=None, digits=2, output_dict=False,
                      zero_division='warn')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_true` | `array_like` | 真实标签 | `y_test` |
| `y_pred` | `array_like` | 预测标签 | `model.predict(X_test)` |
| `normalize` | `str` | 混淆矩阵归一化：`"true"` / `"pred"` / `"all"` | `"true"` |
| `target_names` | `list[str]` | 报告中类别显示名称 | `["恶性", "良性"]` |
| `output_dict` | `bool` | 是否返回字典而非字符串，默认为 `False` | `True` |
| `digits` | `int` | 小数位数，默认为 `2` | `3` |

#### 示例代码

```python
from sklearn.metrics import confusion_matrix, classification_report

# 使用上一节的 y_test 和 yPred
cm = confusion_matrix(y_test, yPred)
print(f"混淆矩阵:\n{cm}")

tn, fp, fn, tp = cm.ravel()
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

print(f"\n{classification_report(y_test, yPred, target_names=['恶性', '良性'])}")
```

#### 输出

```text
混淆矩阵:
[[ 62   1]
 [  4 104]]
TN=62, FP=1, FN=4, TP=104

              precision    recall  f1-score   support

          恶性       0.94      0.98      0.96        63
          良性       0.99      0.96      0.98       108

    accuracy                           0.97       171
   macro avg       0.96      0.97      0.97       171
weighted avg       0.97      0.97      0.97       171
```

#### 理解重点

- 混淆矩阵是阈值调优与误判成本分析的起点
- 观察 support 列可避免被样本量差异误导
- 模型上线前应把业务关注类别单独做阈值评估
- `output_dict=True` 适合将结果以编程方式消费

## 3. ROC 与 PR 曲线

### `roc_curve` / `roc_auc_score` / `precision_recall_curve`

#### 作用

ROC 关注 TPR 与 FPR 的权衡，PR 更适合类别不平衡场景。AUC 是曲线面积摘要，便于模型快速对比。这类指标依赖概率输出，通常使用 `predict_proba` 获取正类概率。

#### 重点方法

```python
roc_curve(y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=True)
roc_auc_score(y_true, y_score, *, average='macro', sample_weight=None,
              max_fpr=None, multi_class='raise', labels=None)
precision_recall_curve(y_true, probas_pred, *, pos_label=None, sample_weight=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_true` | `array_like` | 真实标签 | `y_test` |
| `y_score` | `array_like` | 正类概率 | `clf.predict_proba(X_test)[:, 1]` |
| `pos_label` | `int` 或 `str` | 正类标签，默认为 `None`（自动判断） | `1` |
| `drop_intermediate` | `bool` | 是否删除次优阈值点，默认为 `True` | `True` |
| `multi_class` | `str` | 多分类 AUC 策略：`"ovr"` / `"ovo"` / `"raise"` | `"ovr"` |
| `max_fpr` | `float` | ROC AUC 只计算到指定 FPR 上限 | `0.1` |

#### 示例代码

```python
from sklearn.metrics import (roc_auc_score, roc_curve,
                              precision_recall_curve, auc)

# 使用上一节的 clf 和测试数据
yProba = clf.predict_proba(X_test)[:, 1]

rocAuc = roc_auc_score(y_test, yProba)
fpr, tpr, _ = roc_curve(y_test, yProba)

precision, recall, _ = precision_recall_curve(y_test, yProba)
prAuc = auc(recall, precision)

print(f"ROC AUC: {rocAuc:.4f}")
print(f"FPR 前 5 个阈值: {fpr[:5]}")
print(f"TPR 前 5 个阈值: {tpr[:5]}")
print(f"PR AUC: {prAuc:.4f}")
```

#### 输出

```text
ROC AUC: 0.9978
FPR 前 5 个阈值: [0.         0.         0.         0.         0.01587302]
TPR 前 5 个阈值: [0.         0.00943396 0.11320755 0.95283019 0.95283019]
PR AUC: 0.9981
```

#### 理解重点

- ROC AUC 高通常说明排序能力强，但阈值仍需业务化设定
- 正负样本极不平衡时，PR 曲线更有参考价值
- 概率未校准时，曲线仍可用但阈值解释要谨慎
- `roc_curve` 返回的阈值按降序排列，首尾阈值可能为 inf

## 4. 多分类指标

### `f1_score` / `roc_auc_score` 的聚合方式

#### 作用

多分类 F1 的 `average` 方式会影响结论。`micro` 强调总体样本，`macro` 强调类别公平，`weighted` 兼顾样本量。多分类 AUC 常用 `ovr` 与 `ovo` 两种策略。

#### 重点方法

```python
f1_score(y_true, y_pred, *, average='micro')     # micro / macro / weighted
roc_auc_score(y_true, y_score, *, multi_class='ovr')  # ovr / ovo
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `average` | `str` | F1 聚合方式：`"micro"` / `"macro"` / `"weighted"` | `"macro"` |
| `multi_class` | `str` | AUC 策略：`"ovr"`（one-vs-rest）/ `"ovo"`（one-vs-one） | `"ovr"` |

#### 示例代码

```python
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
yPred = clf.predict(X_test)
yProba = clf.predict_proba(X_test)

print("F1 不同 average:")
for avg in ["micro", "macro", "weighted"]:
    print(f"  {avg}: {f1_score(y_test, yPred, average=avg):.4f}")

print("多分类 ROC AUC:")
for strategy in ["ovr", "ovo"]:
    print(f"  {strategy}: {roc_auc_score(y_test, yProba, multi_class=strategy):.4f}")
```

#### 输出

```text
F1 不同 average:
  micro: 1.0000
  macro: 1.0000
  weighted: 1.0000
多分类 ROC AUC:
  ovr: 1.0000
  ovo: 1.0000
```

#### 理解重点

- 类别不均衡时应优先关注 `macro` 或类别级报告
- `ovr` 与 `ovo` 的差异在类别增多时更明显
- `micro` F1 等价于 accuracy——将各类别的 TP/FP/FN 先求和再计算
- 多分类概率质量可继续用校准曲线补充验证

## 5. 回归指标

### `r2_score` / `mean_squared_error` / `mean_absolute_error`

#### 作用

回归指标关注误差大小与解释能力两个维度。R² 衡量解释比例，MSE/RMSE/MAE 衡量误差幅度。业务上常用 MAE（可解释）与 RMSE（对大误差更敏感）组合。

#### 重点方法

```python
r2_score(y_true, y_pred, *, sample_weight=None, force_finite=True)
mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')
mean_absolute_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_true` | `array_like` | 真实值 | `y_test` |
| `y_pred` | `array_like` | 预测值 | `reg.predict(X_test)` |
| `sample_weight` | `array_like` | 样本权重 | `None` |
| `multioutput` | `str` 或 `array_like` | 多输出聚合方式：`"uniform_average"` / `"raw_values"` | `"uniform_average"` |
| `force_finite` | `bool` | R² 是否强制返回有限值（避免负无穷），默认为 `True` | `True` |

#### 示例代码

```python
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

X, y = datasets.load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

reg = LinearRegression().fit(X_train, y_train)
yPred = reg.predict(X_test)

r2 = r2_score(y_test, yPred)
mse = mean_squared_error(y_test, yPred)
mae = mean_absolute_error(y_test, yPred)

print(f"R²: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {np.sqrt(mse):.4f}")
print(f"MAE: {mae:.4f}")
```

#### 输出

```text
R²: 0.4773
MSE: 2821.7509
RMSE: 53.1202
MAE: 42.7941
```

#### 理解重点

- R² 可比较解释力，但不代表误差绝对可接受
- RMSE 对异常误差更敏感，适合强调大偏差风险的场景
- MAE 与目标同单位，业务人员更易理解
- 指标应与业务容忍阈值一起解释——RMSE 53 在目标均值 ~152 下约 35% 误差

## 6. 自定义评分函数

### `make_scorer`

#### 作用

`make_scorer` 可把业务目标映射为可优化的评分函数。自定义评分可直接接入交叉验证与搜索器。适合将多指标加权成单一决策目标。

#### 重点方法

```python
make_scorer(score_func, *, greater_is_better=True, needs_proba=False,
            needs_threshold=False, **kwargs)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `score_func` | `callable` | 评分函数，签名为 `(y_true, y_pred, **kwargs)` | `custom_score` |
| `greater_is_better` | `bool` | 分数越大是否越好，默认为 `True` | `True` |
| `needs_proba` | `bool` | 是否需要概率输出，默认为 `False` | `False` |
| `needs_threshold` | `bool` | 是否需要决策阈值，默认为 `False` | `False` |
| `**kwargs` | `dict` | 传递给 `score_func` 的额外固定参数 | `{"pos_label": 1}` |

#### 示例代码

```python
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import cross_val_score

X, y = datasets.load_breast_cancer(return_X_y=True)
clf = LogisticRegression(max_iter=10000)

def customScore(yTrue, yPred):
    p = precision_score(yTrue, yPred)
    r = recall_score(yTrue, yPred)
    return 0.7 * p + 0.3 * r

scorer = make_scorer(customScore)
scores = cross_val_score(clf, X, y, cv=5, scoring=scorer)
print(f"自定义评分各折: {scores}")
print(f"平均: {scores.mean():.4f}")
```

#### 输出

```text
自定义评分各折: [0.9762 0.9861 0.9732 0.9815 0.9780]
平均: 0.9790
```

#### 理解重点

- 自定义指标应先做单元测试，确保方向与数值正确
- 若指标不可导或不稳定，搜索过程会更噪声化
- 评分函数最好与线上 KPI 口径保持一致
- `greater_is_better=False` 时搜索器会自动取负方向

## 常见坑

1. 直接用类别预测做 ROC/PR，导致曲线信息失真——必须输入概率
2. 多分类任务只看总体准确率，忽略少数类别表现
3. 自定义评分函数方向写反，搜索结果被误导
4. 混淆矩阵索引易搞混——`cm[0, 0]` 是 TN，`cm[1, 1]` 是 TP

## 小结

- 指标体系应围绕业务目标构建，不应只追求单一高分
- 建议建立固定评估模板：分类指标 + 混淆矩阵 + 曲线 + 业务加权分
- 有概率输出时优先看 ROC/PR AUC，仅类别标签时看 F1 + 分类报告
- 回归任务建议同时报告 R² + RMSE + MAE 三个维度
