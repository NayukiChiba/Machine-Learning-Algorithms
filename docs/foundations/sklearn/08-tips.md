---
title: 技巧
outline: deep
---

# 技巧

> 对应脚本：`Basic/ScikitLearn/08_tips.py`
> 运行方式：`python Basic/ScikitLearn/08_tips.py`（仓库根目录）

## 本章目标

1. 掌握模型克隆与参数管理的工程化写法。
2. 学会处理类别不平衡相关的权重配置。
3. 了解如何编写自定义 Transformer 以接入 Pipeline。
4. 掌握模型持久化、配置管理与版本检查实践。
5. 学会快速检索 sklearn 可用估计器以提升研发效率。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `clone(estimator)` | 克隆模型参数，不复制训练状态 | `demo_clone` |
| `get_params()` / `set_params(...)` | 获取与修改模型参数 | `demo_get_set_params` |
| `LogisticRegression(class_weight='balanced')` | 处理类别不平衡 | `demo_class_weight` |
| `compute_class_weight(...)` / `compute_sample_weight(...)` | 显式计算类别/样本权重 | `demo_compute_class_weight` |
| `BaseEstimator` + `TransformerMixin` | 构建自定义转换器 | `demo_custom_transformer` |
| `joblib.dump(...)` / `joblib.load(...)` | 模型保存与加载 | `demo_model_persistence` |
| `set_config(...)` / `get_config()` | 全局配置管理 | `demo_sklearn_config` |
| `sklearn.__version__` + `version.parse(...)` | 版本能力检查 | `demo_version_check` |
| `all_estimators(type_filter=...)` | 查看可用估计器列表 | `demo_all_estimators` |

## 1. clone 克隆模型

### 方法重点

- `clone` 复制超参数配置，但不会复制拟合状态。
- 适合在交叉验证或实验分支中复用模型配置。
- 避免在不同实验间复用同一已训练对象造成污染。

### 参数速览（本节）

1. `clone(estimator, safe=True)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `estimator` | 已训练 `RandomForestClassifier` | 待克隆模型 |
| `safe` | 默认 `True` | 仅允许 sklearn 估计器对象 |
| 返回值 | 与输入同类型模型对象 | 参数相同，训练状态为空 |

### 示例代码

```python
from sklearn import datasets
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
	iris.data, iris.target, test_size=0.3, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_clone = clone(rf)

print(hasattr(rf, "estimators_"))
print(hasattr(rf_clone, "estimators_"))
print(rf.get_params()["n_estimators"] == rf_clone.get_params()["n_estimators"])
```

### 结果输出（示例）

```text
原模型已训练: True
----------------
克隆模型已训练: False
----------------
参数相同: True
```

### 理解重点

- clone 是“复制配置”，不是“复制权重”。
- 用于多实验并行时，可降低对象共享导致的副作用。
- 与 [模型选择](/foundations/sklearn/05-model-selection) 的评估流程天然契合。

## 2. get_params 与 set_params

### 方法重点

- `get_params` 提供统一参数字典，便于日志、配置化、追踪。
- `set_params` 可动态更新模型超参数。
- 该接口也是网格搜索和随机搜索的底层依赖。

### 参数速览（本节）

1. `estimator.get_params(deep=True)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `deep` | 默认 `True` | 是否递归获取子估计器参数 |
| 返回值 | `dict` | 模型参数字典 |

2. `estimator.set_params(**params)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_estimators` | `100 -> 50` | 随机森林树数量变更 |
| `max_depth` | `None -> 5` | 树深度约束 |
| 返回值 | `estimator` | 修改后的模型对象 |

### 示例代码

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
print(rf.get_params()["n_estimators"])
print(rf.get_params()["max_depth"])

rf.set_params(n_estimators=50, max_depth=5)
print(rf.get_params()["n_estimators"])
print(rf.get_params()["max_depth"])
```

### 结果输出（示例）

```text
get_params()['n_estimators']: 100
----------------
get_params()['max_depth']: None
----------------
修改后:
  n_estimators: 50
  max_depth: 5
```

### 理解重点

- 参数接口是自动化实验系统的关键入口。
- 推荐将关键参数记录到实验日志，便于复现和回滚。
- 对 Pipeline 对象也适用同样机制（配合双下划线）。

## 3. class_weight 处理类别不平衡

### 方法重点

- 类别不平衡时，模型会偏向多数类。
- `class_weight='balanced'` 根据类别频率自动调整损失权重。
- 实际效果应以少数类指标（如 F1、Recall）验证。

### 参数速览（本节）

1. `LogisticRegression(max_iter=1000)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `max_iter` | `1000` | 迭代上限 |
| 返回值（少数类 F1） | `float` | 无类别权重基线结果 |

2. `LogisticRegression(class_weight='balanced', max_iter=1000)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `class_weight` | `'balanced'` | 自动类别权重 |
| `max_iter` | `1000` | 迭代上限 |
| 返回值（少数类 F1） | `float` | 加权后的少数类指标 |

### 示例代码

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
clf_bal = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X_train, y_train)

print(classification_report(y_test, clf.predict(X_test), output_dict=True)["1"]["f1-score"])
print(classification_report(y_test, clf_bal.predict(X_test), output_dict=True)["1"]["f1-score"])
```

### 结果输出（示例）

```text
类别分布: [630  70]
----------------
无权重 - 少数类F1: 0.612
----------------
balanced - 少数类F1: 0.711
```

### 理解重点

- 准确率可能上升但少数类表现下降，需警惕指标幻觉。
- 权重策略能提升召回，但可能牺牲精确率。
- 业务上应提前定义错判代价，再决定权重方案。

## 4. compute_class_weight

### 方法重点

- 手动计算权重可获得更透明的类别补偿机制。
- `compute_sample_weight` 可生成逐样本权重供模型训练使用。
- 常用于定制化损失或自定义训练流程。

### 参数速览（本节）

1. `compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `class_weight` | `'balanced'` | 平衡策略 |
| `classes` | `np.unique(y)` | 类别集合 |
| `y` | 类别标签数组 | 标签向量 |
| 返回值 | `ndarray` | 每类权重 |

2. `compute_sample_weight(class_weight='balanced', y=y)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `class_weight` | `'balanced'` | 平衡策略 |
| `y` | 类别标签数组 | 标签向量 |
| 返回值 | `ndarray` | 每个样本权重 |

### 示例代码

```python
import numpy as np
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
sample_weights = compute_sample_weight("balanced", y)

print(class_weights)
print(np.unique(sample_weights))
```

### 结果输出（示例）

```text
类别分布: [8 2]
----------------
类别权重: {0: 0.625, 1: 2.5}
----------------
样本权重 (唯一值): [0.625 2.5]
```

### 理解重点

- 类权重本质是重写经验风险最小化中的样本贡献。
- 手动权重适合“类别比率已知且有业务代价”的场景。
- 权重过大可能导致训练不稳定，需配合验证集检查。

## 5. 自定义 Transformer

### 方法重点

- 继承 `BaseEstimator` 与 `TransformerMixin` 可无缝接入 Pipeline。
- 只要实现 `fit` 与 `transform`，即可构建可复用处理器。
- 自定义转换器是把业务规则工程化的关键手段。

### 参数速览（本节）

1. `class LogTransformer(BaseEstimator, TransformerMixin)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `offset` | `1` | 对数变换偏移量，避免对 0 取对数 |

2. `fit(self, X, y=None)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X` / `y` | 训练输入 | `y` 默认可选 |
| 返回值 | `self` | sklearn 约定 |

3. `transform(self, X)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X` | 特征矩阵 | 待变换输入 |
| 返回值 | `ndarray` | 变换后特征 |

### 示例代码

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransformer(BaseEstimator, TransformerMixin):
	def __init__(self, offset=1):
		self.offset = offset

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		return np.log(X + self.offset)

X = np.array([[1, 10], [100, 1000]])
print(LogTransformer(offset=1).fit_transform(X))
```

### 结果输出（示例）

```text
原始: [[1, 10], [100, 1000]]
----------------
变换后: [[0.693, 2.398], [4.615, 6.909]]
```

### 理解重点

- 自定义转换器应保持无副作用和确定性输出。
- 建议给转换器写单测，验证边界值与缺失值行为。
- 复杂逻辑拆分为多个小转换器更易维护。

## 6. 模型持久化

### 方法重点

- `joblib` 是 sklearn 模型持久化的常用方案。
- 压缩可减少体积，但会增加读写开销。
- 加载后的预测一致性必须验证。

### 参数速览（本节）

1. `joblib.dump(model, path)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `path` | 临时目录下 `model.joblib` | 模型文件路径 |
| 返回值 | 保存文件列表 | 持久化结果 |

2. `joblib.load(path)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `path` | 模型文件路径 | 反序列化输入 |
| 返回值 | 模型对象 | 反序列化后的模型 |

3. `joblib.dump(model, path, compress=3)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `path` | 压缩文件路径 | 模型文件路径 |
| `compress` | `3` | 压缩等级 |
| 返回值 | 保存文件列表 | 压缩持久化结果 |

### 示例代码

```python
import joblib
import os
from tempfile import mkdtemp
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
	iris.data, iris.target, test_size=0.3, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
temp_dir = mkdtemp()
path = os.path.join(temp_dir, "model.joblib")
path_compressed = os.path.join(temp_dir, "model_compressed.joblib")

joblib.dump(rf, path)
rf_loaded = joblib.load(path)
joblib.dump(rf, path_compressed, compress=3)

print((rf_loaded.predict(X_test) == rf.predict(X_test)).run())
```

### 结果输出（示例）

```text
保存大小: 182.4 KB
----------------
加载后预测一致: True
----------------
压缩后大小: 96.7 KB
```

### 理解重点

- 模型文件应与训练代码版本、依赖版本一起管理。
- 线上加载前要做一致性回归测试。
- 生产环境优先使用稳定路径与权限管理，不用临时目录。

## 7. sklearn 全局配置

### 方法重点

- `set_config` 可修改 sklearn 的全局行为配置。
- `transform_output='pandas'` 在数据分析阶段更友好。
- 配置变更应及时恢复，避免影响其他流程。

### 参数速览（本节）

1. `get_config()`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回值 | `dict` | 当前全局配置字典 |

2. `set_config(transform_output='pandas')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `transform_output` | `'pandas'` | 变换器输出为 DataFrame |
| 返回值 | `None` | 修改全局配置 |

3. `set_config(transform_output='default')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `transform_output` | `'default'` | 恢复默认输出格式 |
| 返回值 | `None` | 修改全局配置 |

### 示例代码

```python
from sklearn import get_config, set_config

print(get_config())
set_config(transform_output="pandas")
print("设置 transform_output='pandas'")
set_config(transform_output="default")
print("恢复 transform_output='default'")
```

### 结果输出（示例）

```text
当前配置: {...}
----------------
设置 transform_output='pandas'
----------------
恢复 transform_output='default'
```

### 理解重点

- 全局配置适合实验和分析，不宜在库代码中隐式修改。
- 多人协作时建议显式记录配置变更。
- 配置差异可能导致同一代码输出格式不一致。

## 8. 版本检查

### 方法重点

- 不同 sklearn 版本 API 可用性不同，需显式校验。
- 使用 `packaging.version` 比字符串比较更可靠。
- 版本门控能避免线上环境 API 不匹配。

### 参数速览（本节）

1. `sklearn.__version__`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回值 | 版本字符串 | 当前 sklearn 版本 |

2. `version.parse(sklearn.__version__) >= version.parse('1.2')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 版本阈值 | `1.0`、`1.2` | 判断特性是否可用 |
| 返回值 | `bool` | 是否满足版本条件 |

### 示例代码

```python
import sklearn
from packaging import version

print(sklearn.__version__)
print(version.parse(sklearn.__version__) >= version.parse("1.0"))
print(version.parse(sklearn.__version__) >= version.parse("1.2"))
```

### 结果输出（示例）

```text
sklearn 版本: 1.6.1
----------------
✓ 版本 >= 1.0
----------------
✓ 版本 >= 1.2, 支持 set_output API
```

### 理解重点

- 版本门控应成为工具脚本与部署脚本的标准步骤。
- 当文档示例依赖新特性时，必须标注最低版本要求。
- 建议将关键依赖版本固定在项目配置中。

## 9. all_estimators 快速检索

### 方法重点

- `all_estimators` 可快速查看当前环境可用估计器。
- 适合做模型候选池构建与自动化实验初始化。
- `type_filter` 可按任务类型筛选。

### 参数速览（本节）

1. `all_estimators(type_filter='classifier')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `type_filter` | `'classifier'` | 仅返回分类器 |
| 返回值 | `list[(name, class)]` | 估计器名称与类对象列表 |

2. `all_estimators(type_filter='regressor')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `type_filter` | `'regressor'` | 仅返回回归器 |
| 返回值 | `list[(name, class)]` | 估计器名称与类对象列表 |

3. `all_estimators(type_filter='transformer')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `type_filter` | `'transformer'` | 仅返回转换器 |
| 返回值 | `list[(name, class)]` | 估计器名称与类对象列表 |

### 示例代码

```python
from sklearn.utils import all_estimators

classifiers = all_estimators(type_filter="classifier")
regressors = all_estimators(type_filter="regressor")
transformers = all_estimators(type_filter="transformer")

print(len(classifiers), len(regressors), len(transformers))
print([name for name, _ in classifiers[:5]])
```

### 结果输出（示例）

```text
分类器数量: 49
----------------
回归器数量: 55
----------------
转换器数量: 93
----------------
分类器前5个: ['AdaBoostClassifier', 'BaggingClassifier', 'BernoulliNB', 'CalibratedClassifierCV', 'CategoricalNB']
```

### 理解重点

- 该工具可用于快速探索，但不替代模型选择流程。
- 不同版本中估计器数量会变化，应结合版本信息解读。
- 可与自动化评估框架结合构建候选模型库。

## 常见坑

1. 把 clone 误认为深拷贝训练状态，导致实验误判。
2. 忽略版本差异直接调用新 API，引发环境兼容问题。
3. 持久化后不做预测一致性校验，埋下线上风险。

## 小结

- 技巧章节的核心是把“能跑”升级为“可维护、可复现、可部署”。
- 推荐将权重策略、版本校验、持久化检查纳入项目模板。
- 组合实践可与 [Pipeline](/foundations/sklearn/04-pipeline) 和 [模型](/foundations/sklearn/07-models) 章节联动使用。
