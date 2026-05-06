---
title: sklearn 模型选择
outline: deep
---

# sklearn 模型选择

## 本章目标

1. 理解交叉验证分数的统计意义与波动范围
2. 掌握 `cross_val_score` 与 `cross_validate` 的使用边界
3. 学会选择合适的划分器（KFold、StratifiedKFold、TimeSeriesSplit）
4. 掌握网格搜索与随机搜索的参数设计思路
5. 能用学习曲线与验证曲线诊断欠拟合和过拟合

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `cross_val_score(estimator, X, y)` | 函数 | 单指标交叉验证，返回每折分数 |
| `cross_validate(estimator, X, y)` | 函数 | 多指标交叉验证，返回训练与测试分数 |
| `KFold(n_splits)` | 构造器 | 基础 K 折划分 |
| `StratifiedKFold(n_splits)` | 构造器 | 分层 K 折，保持类别比例 |
| `TimeSeriesSplit(n_splits)` | 构造器 | 时间序列逐步扩窗划分 |
| `GridSearchCV(estimator, param_grid)` | 构造器 | 穷举参数组合搜索 |
| `RandomizedSearchCV(estimator, param_distributions)` | 构造器 | 随机采样参数搜索 |
| `learning_curve(estimator, X, y)` | 函数 | 训练集规模-性能曲线 |
| `validation_curve(estimator, X, y, ...)` | 函数 | 单参数-性能曲线 |

## 1. 交叉验证

### `cross_val_score`

#### 作用

`cross_val_score` 返回每折分数数组，适合快速评估模型稳定性。分数均值反映整体性能，标准差反映稳定性。推荐将分数与模型复杂度一起解读。

#### 重点方法

```python
cross_val_score(estimator, X, y=None, *, groups=None, scoring=None, cv=None,
                n_jobs=None, verbose=0, params=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `estimator` | 待评估模型 | `make_pipeline(StandardScaler(), SVC())` |
| `X` | `array_like` | 特征矩阵 | `iris.data` |
| `y` | `array_like` | 标签向量 | `iris.target` |
| `scoring` | `str` 或 `callable` | 评估指标，默认为 `None`（用估计器默认） | `"accuracy"` |
| `cv` | `int` 或 `splitter` | 交叉验证折数，默认为 `None`（5 折） | `5` |
| `n_jobs` | `int` 或 `None` | 并行数，`-1` 使用全部核心 | `-1` |

#### 示例代码

```python
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = datasets.load_iris(return_X_y=True)
model = make_pipeline(StandardScaler(), SVC())
scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

print(f"各折得分: {scores}")
print(f"平均: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

#### 输出

```text
各折得分: [0.96666667 0.96666667 0.93333333 0.96666667 1.        ]
平均: 0.9667 (+/- 0.0422)
```

#### 理解重点

- 单次 train/test 切分容易偶然偏高或偏低，交叉验证更稳健
- 平均值高但方差大时，模型泛化稳定性仍需警惕
- 不同指标会改变结论，应与任务目标对齐
- 返回分数与 `scoring` 参数绑定——"accuracy" 越高越好，"neg_mean_squared_error" 越低越好

### `cross_validate`

#### 作用

`cross_validate` 能同时返回多指标结果与训练分数。训练分数与测试分数差距可辅助判断过拟合。返回字典结构便于后续可视化和日志记录。

#### 重点方法

```python
cross_validate(estimator, X, y=None, *, groups=None, scoring=None, cv=None,
               n_jobs=None, verbose=0, params=None,
               return_train_score=False, return_estimator=False)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `estimator` | 待评估模型 | `make_pipeline(StandardScaler(), SVC())` |
| `X` | `array_like` | 特征矩阵 | `iris.data` |
| `y` | `array_like` | 标签向量 | `iris.target` |
| `scoring` | `str`、`list[str]` 或 `dict` | 评估指标（支持多个），默认为 `None` | `["accuracy", "f1_macro"]` |
| `cv` | `int` 或 `splitter` | 交叉验证折数，默认为 `None`（5 折） | `5` |
| `return_train_score` | `bool` | 是否返回训练集分数，默认为 `False` | `True` |
| `return_estimator` | `bool` | 是否返回每折训练的估计器，默认为 `False` | `True` |
| `n_jobs` | `int` 或 `None` | 并行数，默认为 `None` | `-1` |

#### 示例代码

```python
from sklearn import datasets
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = datasets.load_iris(return_X_y=True)
model = make_pipeline(StandardScaler(), SVC())

cvResults = cross_validate(
    model, X, y, cv=5,
    scoring=["accuracy", "f1_macro"],
    return_train_score=True,
)

print(f"返回的键: {list(cvResults.keys())}")
print(f"测试准确率: {cvResults['test_accuracy'].mean():.4f}")
print(f"训练准确率: {cvResults['train_accuracy'].mean():.4f}")
print(f"测试 F1: {cvResults['test_f1_macro'].mean():.4f}")
```

#### 输出

```text
返回的键: ['fit_time', 'score_time', 'test_accuracy', 'train_accuracy', 'test_f1_macro', 'train_f1_macro']
测试准确率: 0.9667
训练准确率: 0.9833
测试 F1: 0.9664
```

#### 理解重点

- 多指标结果能避免"单指标最优但业务不优"的问题
- 训练分数显著高于测试分数时，优先检查过拟合
- 这类结果适合沉淀到实验追踪系统
- `return_estimator=True` 可取出每折模型做进一步分析

## 2. 划分策略

### `KFold` / `StratifiedKFold` / `TimeSeriesSplit`

#### 作用

划分器选择必须匹配数据分布与任务类型。分类任务默认优先 `StratifiedKFold` 保持类别比例。时间序列不能随机打乱，需使用 `TimeSeriesSplit`。错误划分策略会比模型选择本身造成更大偏差。

#### 重点方法

```python
KFold(n_splits=5, *, shuffle=False, random_state=None)
StratifiedKFold(n_splits=5, *, shuffle=False, random_state=None)
TimeSeriesSplit(n_splits=5, *, max_train_size=None)
# split(X, y=None) → 迭代器，产出 (train_indices, test_indices)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_splits` | `int` | 划分折数 | `5` |
| `shuffle` | `bool` | 划分前是否打乱样本（KFold/StratifiedKFold），默认为 `False` | `True` |
| `random_state` | `int` | 随机种子，保证可复现 | `42` |
| `max_train_size` | `int` 或 `None` | TimeSeriesSplit：训练集最大容量限制 | `100` |

#### 示例代码

```python
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

X = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

kf = KFold(n_splits=3, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
tscv = TimeSeriesSplit(n_splits=3)

print("KFold (shuffle=True):")
for i, (tr, te) in enumerate(kf.split(X)):
    print(f"  Fold {i}: train={tr}, test={te}")

print("\nStratifiedKFold 类别分布:")
for i, (tr, te) in enumerate(skf.split(X, y)):
    print(f"  Fold {i}: train={np.bincount(y[tr])}, test={np.bincount(y[te])}")

print("\nTimeSeriesSplit:")
for i, (tr, te) in enumerate(tscv.split(X)):
    print(f"  Fold {i}: train={tr}, test={te}")
```

#### 输出

```text
KFold (shuffle=True):
  Fold 0: train=[2 3 4 6 7 9], test=[0 1 5 8]
  Fold 1: train=[0 1 4 5 7 8 9], test=[2 3 6]
  Fold 2: train=[0 1 2 3 5 6 8], test=[4 7 9]

StratifiedKFold 类别分布:
  Fold 0: train=[3 3], test=[2 2]
  Fold 1: train=[3 3], test=[2 2]
  Fold 2: train=[4 4], test=[1 1]

TimeSeriesSplit:
  Fold 0: train=[0 1 2 3], test=[4 5]
  Fold 1: train=[0 1 2 3 4 5], test=[6 7]
  Fold 2: train=[0 1 2 3 4 5 6 7], test=[8 9]
```

#### 理解重点

- 分类任务默认优先 `StratifiedKFold`——保持每折类别比例一致
- 时间序列任务应严格遵守时间先后关系——不能用随机 K 折
- 类别不平衡时不分层会导致评估结果波动异常
- `TimeSeriesSplit` 训练集逐步扩大——模拟真实时间预测场景

## 3. GridSearchCV

### `GridSearchCV`

#### 作用

网格搜索会遍历所有参数组合，结果稳定但计算成本高。参数空间应先由经验收敛，否则会出现组合爆炸。常与 Pipeline 结合以统一预处理与调参。

#### 重点方法

```python
GridSearchCV(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True,
             cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=nan)
# fit(X, y) → best_params_ / best_score_ / best_estimator_ / predict(X)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `estimator` | 待调参模型或流水线 | `make_pipeline(StandardScaler(), SVC())` |
| `param_grid` | `dict` 或 `list[dict]` | 参数网格，键用 `step__param` 格式 | `{"svc__C": [0.1, 1, 10]}` |
| `scoring` | `str`、`callable`、`list` 或 `dict` | 评估指标，默认为 `None` | `"accuracy"` |
| `cv` | `int` 或 `splitter` | 交叉验证折数，默认为 `None`（5 折） | `5` |
| `n_jobs` | `int` 或 `None` | 并行数，`-1` 使用全部核心 | `-1` |
| `refit` | `bool` | 是否用最优参数在全量数据重训，默认为 `True` | `True` |
| `verbose` | `int` | 日志详细度，默认为 `0` | `1` |
| `pre_dispatch` | `int` 或 `str` | 并行任务预分配数，默认为 `"2*n_jobs"` | `"2*n_jobs"` |

训练后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `best_params_` | `dict` | 最优参数组合 |
| `best_score_` | `float` | 最优参数对应的交叉验证平均分 |
| `best_estimator_` | `estimator` | 用最优参数在全量数据训练的模型 |
| `cv_results_` | `dict` | 完整搜索结果 |

#### 示例代码

```python
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = datasets.load_iris(return_X_y=True)
pipe = make_pipeline(StandardScaler(), SVC())

paramGrid = {
    "svc__C": [0.1, 1, 10],
    "svc__kernel": ["linear", "rbf"],
}

grid = GridSearchCV(pipe, paramGrid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X, y)
print(f"最佳参数: {grid.best_params_}")
print(f"最佳得分: {grid.best_score_:.4f}")
print(f"候选数: {len(grid.cv_results_['params'])}")
```

#### 输出

```text
最佳参数: {'svc__C': 1, 'svc__kernel': 'linear'}
最佳得分: 0.9733
候选数: 6
```

#### 理解重点

- GridSearchCV 更像"精细扫描"，前提是搜索区间合理
- 参数边界选择不当会浪费大量算力且结果无效
- 大规模任务可先随机搜索粗定位，再网格精调
- `cv_results_` 包含所有候选参数的时间和分值——可用于绘制热力图

## 4. RandomizedSearchCV

### `RandomizedSearchCV`

#### 作用

随机搜索通过概率分布采样参数，成本可控。在高维参数空间里，常比小网格更高效。常与 `loguniform` 分布配合搜索正实数超参数。

#### 重点方法

```python
RandomizedSearchCV(estimator, param_distributions, *, n_iter=10, scoring=None,
                   n_jobs=None, refit=True, cv=None, verbose=0,
                   random_state=None)
# fit(X, y) → best_params_ / best_score_ / best_estimator_
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `estimator` | 待调参模型或流水线 | `make_pipeline(StandardScaler(), SVC())` |
| `param_distributions` | `dict` | 采样空间，值可为分布对象或列表 | `{"svc__C": loguniform(0.01, 100)}` |
| `n_iter` | `int` | 采样次数，默认为 `10` | `20` |
| `scoring` | `str` 或 `callable` | 评估指标，默认为 `None` | `"accuracy"` |
| `cv` | `int` 或 `splitter` | 交叉验证折数，默认为 `None`（5 折） | `5` |
| `n_jobs` | `int` 或 `None` | 并行数，`-1` 使用全部核心 | `-1` |
| `random_state` | `int` | 采样可复现种子 | `42` |
| `refit` | `bool` | 是否最优参数重训，默认为 `True` | `True` |

#### 示例代码

```python
from scipy.stats import loguniform
from sklearn import datasets
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = datasets.load_iris(return_X_y=True)
pipe = make_pipeline(StandardScaler(), SVC())

paramDist = {
    "svc__C": loguniform(0.01, 100),
    "svc__gamma": loguniform(0.001, 10),
    "svc__kernel": ["rbf", "linear"],
}

search = RandomizedSearchCV(pipe, paramDist, n_iter=20, cv=5,
                            scoring="accuracy", random_state=42, n_jobs=-1)
search.fit(X, y)
print(f"最佳参数: {search.best_params_}")
print(f"最佳得分: {search.best_score_:.4f}")
```

#### 输出

```text
最佳参数: {'svc__C': 2.7323713729500725, 'svc__gamma': 0.013895024894200025, 'svc__kernel': 'rbf'}
最佳得分: 0.9800
```

#### 理解重点

- 采样分布比搜索算法本身更关键，应根据参数尺度设计
- `n_iter` 不是越大越好，应与预算和收益平衡
- 随机搜索结果可作为网格搜索的初始范围参考
- 连续参数用 `loguniform`/`uniform`，离散参数用列表

## 5. 学习曲线与验证曲线

### `learning_curve`

#### 作用

学习曲线观察训练样本量变化对性能的影响。训练分数高、验证分数低通常提示过拟合。两条曲线都低且接近，通常提示欠拟合或特征不足。

#### 重点方法

```python
learning_curve(estimator, X, y, *, groups=None, train_sizes=None, cv=None,
               scoring=None, exploit_incremental_learning=False,
               n_jobs=None, shuffle=True, random_state=None)
# 返回：train_sizes_abs, train_scores, test_scores
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `estimator` | 待评估模型 | `make_pipeline(StandardScaler(), SVC())` |
| `X` | `array_like` | 特征矩阵 | `iris.data` |
| `y` | `array_like` | 标签向量 | `iris.target` |
| `train_sizes` | `array_like` | 训练样本比例或绝对数量 | `np.linspace(0.3, 1.0, 5)` |
| `cv` | `int` 或 `splitter` | 交叉验证划分器，默认为 `None`（5 折） | `StratifiedKFold(3, shuffle=True, random_state=42)` |
| `scoring` | `str` 或 `callable` | 评估指标，默认为 `None` | `"accuracy"` |
| `shuffle` | `bool` | 训练子集是否打乱，默认为 `True` | `True` |
| `random_state` | `int` | 随机种子 | `42` |
| `n_jobs` | `int` 或 `None` | 并行数，默认为 `None` | `-1` |

#### 示例代码

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = datasets.load_iris(return_X_y=True)
model = make_pipeline(StandardScaler(), SVC())
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

trainSizes, trainScores, testScores = learning_curve(
    model, X, y, cv=cv,
    train_sizes=np.linspace(0.3, 1.0, 5),
    scoring="accuracy", shuffle=True, random_state=42,
)

print(f"训练集大小: {trainSizes}")
print(f"训练得分: {trainScores.mean(axis=1).round(3)}")
print(f"测试得分: {testScores.mean(axis=1).round(3)}")
```

#### 输出

```text
训练集大小: [ 36  49  63  76  90]
训练得分: [1.    0.986 0.974 0.969 0.972]
测试得分: [0.919 0.953 0.967 0.967 0.967]
```

#### 理解重点

- 随样本量增加，训练分数略降、验证分数上升是常见健康趋势
- 若两条曲线始终有大间隙，优先考虑正则化与特征简化
- 学习曲线是判定"继续收集数据是否有价值"的重要依据
- 训练分数和测试分数都低——特征不足或模型太简单

### `validation_curve`

#### 作用

验证曲线用于观察单个超参数变化对性能的影响。常用于确定参数大致有效区间，再进入细化搜索。训练曲线和验证曲线同时看，能识别过拟合拐点。

#### 重点方法

```python
validation_curve(estimator, X, y, *, param_name, param_range, groups=None,
                 cv=None, scoring=None, n_jobs=None, verbose=0)
# 返回：train_scores, test_scores
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `estimator` | 待评估模型 | `make_pipeline(StandardScaler(), SVC())` |
| `X` | `array_like` | 特征矩阵 | `iris.data` |
| `y` | `array_like` | 标签向量 | `iris.target` |
| `param_name` | `str` | 要扫描的参数名 | `"svc__C"` |
| `param_range` | `array_like` | 参数候选序列 | `np.logspace(-3, 2, 5)` |
| `cv` | `int` 或 `splitter` | 交叉验证折数，默认为 `None`（5 折） | `5` |
| `scoring` | `str` 或 `callable` | 评估指标，默认为 `None` | `"accuracy"` |

#### 示例代码

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = datasets.load_iris(return_X_y=True)
paramRange = np.logspace(-3, 2, 5)

trainScores, testScores = validation_curve(
    make_pipeline(StandardScaler(), SVC()),
    X, y,
    param_name="svc__C",
    param_range=paramRange,
    cv=5, scoring="accuracy",
)

print(f"C 值: {paramRange}")
print(f"测试得分: {testScores.mean(axis=1).round(3)}")
```

#### 输出

```text
C 值: [1.000e-03 1.778e-02 3.162e-01 5.623e+00 1.000e+02]
测试得分: [0.347 0.840 0.953 0.973 0.967]
```

#### 理解重点

- 参数过小通常欠拟合，参数过大可能过拟合
- 验证曲线能帮你发现"性能平台区"，降低调参敏感性
- 与网格搜索相比，验证曲线更偏诊断与解释
- 训练和测试分差最大处即为过拟合起始点

## 常见坑

1. 把时间序列数据用随机 K 折，导致评估严重乐观
2. 在极大参数空间直接网格搜索，计算成本不可控
3. 只看平均分不看标准差，忽略模型稳定性风险
4. `cross_val_score` 默认不返回训练分数——无法判断过拟合

## 小结

- 模型选择的核心不是"找到最高分"，而是"找到稳定可部署的方案"
- 推荐流程：先交叉验证基线 → 随机搜索粗调 → 网格精调
- 划分策略需匹配数据特征：分类用分层，时间序列按序划分
- 学习曲线和验证曲线是诊断工具——在调参前先确认方向
