---
title: sklearn 工程技巧
outline: deep
---

# sklearn 工程技巧

## 本章目标

1. 掌握模型克隆与参数管理的工程化写法
2. 学会处理类别不平衡相关的权重配置
3. 了解如何编写自定义 Transformer 以接入 Pipeline
4. 掌握模型持久化、配置管理与版本检查实践
5. 学会快速检索 sklearn 可用估计器以提升研发效率

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `clone(estimator)` | 函数 | 克隆模型参数，不复制训练状态 |
| `estimator.get_params()` | 方法 | 获取全部超参数字典 |
| `estimator.set_params(**p)` | 方法 | 动态修改超参数 |
| `class_weight='balanced'` | 参数 | 自动按类别频率加权 |
| `compute_class_weight('balanced', classes, y)` | 函数 | 显式计算类别权重 |
| `BaseEstimator` + `TransformerMixin` | 基类 | 构建自定义转换器 |
| `joblib.dump(model, path)` | 函数 | 模型持久化保存 |
| `joblib.load(path)` | 函数 | 模型反序列化加载 |
| `set_config(...)` / `get_config()` | 函数 | sklearn 全局配置 |
| `all_estimators(type_filter)` | 函数 | 检索可用估计器列表 |

## 1. 模型克隆

### `clone`

#### 作用

`clone` 复制超参数配置，但不会复制拟合状态。适合在交叉验证或实验分支中复用模型配置。避免在不同实验间复用同一已训练对象造成污染。

#### 重点方法

```python
clone(estimator, *, safe=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `estimator` | 待克隆的估计器对象 | `RandomForestClassifier()` |
| `safe` | `bool` | 仅允许 sklearn 估计器对象，默认为 `True` | `True` |

#### 示例代码

```python
from sklearn import datasets
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rfClone = clone(rf)

print(f"原模型已训练: {hasattr(rf, 'estimators_')}")
print(f"克隆模型已训练: {hasattr(rfClone, 'estimators_')}")
print(f"参数相同: {rf.get_params()['n_estimators'] == rfClone.get_params()['n_estimators']}")
```

#### 输出

```text
原模型已训练: True
克隆模型已训练: False
参数相同: True
```

#### 理解重点

- clone 是"复制配置"，不是"复制权重"
- 用于多实验并行时，可降低对象共享导致的副作用
- 与交叉验证评估流程天然契合——每折需要干净的模型
- `clone` 内部调用 `get_params` 获取配置，因此依赖正确的 `__init__`

## 2. get_params 与 set_params

### `get_params` / `set_params`

#### 作用

`get_params` 提供统一参数字典，便于日志、配置化、追踪。`set_params` 可动态更新模型超参数。该接口也是网格搜索和随机搜索的底层依赖。

#### 重点方法

```python
estimator.get_params(deep=True)
estimator.set_params(**params)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `deep` | `bool` | 是否递归获取子估计器参数，默认为 `True` | `True` |
| `**params` | `dict` | 要修改的参数名与值 | `n_estimators=50, max_depth=5` |

#### 示例代码

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
params = rf.get_params()
print(f"n_estimators: {params['n_estimators']}")
print(f"max_depth: {params['max_depth']}")

rf.set_params(n_estimators=50, max_depth=5)
print(f"修改后 n_estimators: {rf.get_params()['n_estimators']}")
print(f"修改后 max_depth: {rf.get_params()['max_depth']}")
```

#### 输出

```text
n_estimators: 100
max_depth: None
修改后 n_estimators: 50
修改后 max_depth: 5
```

#### 理解重点

- 参数接口是自动化实验系统的关键入口
- 推荐将关键参数记录到实验日志，便于复现和回滚
- 对 Pipeline 对象也适用同样机制——配合双下划线语法
- 自定义估计器必须将 `__init__` 参数以同名属性存储才能被正确读取

## 3. 类别权重处理

### `class_weight` / `compute_class_weight` / `compute_sample_weight`

#### 作用

类别不平衡时，模型会偏向多数类。`class_weight='balanced'` 根据类别频率自动调整损失权重。手动计算权重可获得更透明的类别补偿机制。

#### 重点方法

```python
LogisticRegression(class_weight='balanced', max_iter=1000)
compute_class_weight(class_weight='balanced', *, classes=None, y)
compute_sample_weight(class_weight='balanced', *, y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `class_weight` | `str` 或 `dict` | `"balanced"` 自动 / 或手动 `{0: 0.6, 1: 2.5}` | `"balanced"` |
| `classes` | `array_like` | `compute_class_weight`：类别集合 | `np.unique(y)` |
| `y` | `array_like` | 标签向量 | `y` |

#### 示例代码

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
clfBal = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X_train, y_train)

print(f"类别分布: {np.bincount(y_train)}")
print(f"无权重 少数类F1: {classification_report(y_test, clf.predict(X_test), output_dict=True, zero_division=0)['1']['f1-score']:.3f}")
print(f"balanced 少数类F1: {classification_report(y_test, clfBal.predict(X_test), output_dict=True, zero_division=0)['1']['f1-score']:.3f}")

classWeights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
print(f"计算得类别权重: {classWeights}")
```

#### 输出

```text
类别分布: [630  70]
无权重 少数类F1: 0.606
balanced 少数类F1: 0.740
计算得类别权重: [0.55555556 3.36842105]
```

#### 理解重点

- 准确率可能上升但少数类表现下降，需警惕指标幻觉
- 权重策略能提升召回，但可能牺牲精确率
- 业务上应提前定义错判代价，再决定权重方案
- 权重过大可能导致训练不稳定，需配合验证集检查

## 4. 自定义 Transformer

### `BaseEstimator` + `TransformerMixin`

#### 作用

继承 `BaseEstimator` 与 `TransformerMixin` 可无缝接入 Pipeline。只要实现 `fit` 与 `transform`，即可构建可复用处理器。自定义转换器是把业务规则工程化的关键手段。

#### 重点方法

```python
class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, ...):    # 初始化参数
    def fit(self, X, y=None):   # 返回 self
    def transform(self, X):     # 返回变换后 ndarray
```

#### 示例代码

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransformer(BaseEstimator, TransformerMixin):
    """对数变换器：对每个特征做 log(x + offset)"""
    def __init__(self, offset=1):
        self.offset = offset

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log(X + self.offset)

X = np.array([[1, 10], [100, 1000]])
lt = LogTransformer(offset=1)
print(f"原始:\n{X}")
print(f"\n变换后:\n{lt.fit_transform(X)}")
```

#### 输出

```text
原始:
[[   1   10]
 [ 100 1000]]

变换后:
[[0.69314718 2.39789527]
 [4.61512052 6.90875478]]
```

#### 理解重点

- 自定义转换器应保持无副作用和确定性输出
- 建议给转换器写单测，验证边界值与缺失值行为
- 复杂逻辑拆分为多个小转换器更易维护
- `TransformerMixin` 自动提供 `fit_transform` 方法

## 5. 模型持久化

### `joblib.dump` / `joblib.load`

#### 作用

`joblib` 是 sklearn 模型持久化的常用方案，对 numpy 数组序列化做了优化。压缩可减少体积，但会增加读写开销。加载后的预测一致性必须验证。

#### 重点方法

```python
joblib.dump(value, filename, compress=0, protocol=None)
joblib.load(filename, mmap_mode=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `value` | `any` | 待序列化对象（模型、Pipeline 等） | `rf` |
| `filename` | `str` | 文件路径 | `"./model.joblib"` |
| `compress` | `int` | 压缩等级：`0` / `3`，默认为 `0` | `3` |
| `mmap_mode` | `str` 或 `None` | 加载时内存映射模式 | `None` |

#### 示例代码

```python
import joblib
import os
import numpy as np
from tempfile import mkdtemp
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
tmpDir = mkdtemp()
path = os.path.join(tmpDir, "model.joblib")

joblib.dump(rf, path)
rfLoaded = joblib.load(path)

print(f"预测一致: {np.array_equal(rfLoaded.predict(X_test), rf.predict(X_test))}")
print(f"文件大小: {os.path.getsize(path) / 1024:.1f} KB")
```

#### 输出

```text
预测一致: True
文件大小: 182.4 KB
```

#### 理解重点

- 模型文件应与训练代码版本、依赖版本一起管理
- 线上加载前要做一致性回归测试
- 生产环境优先使用稳定路径与权限管理，不用临时目录
- `compress=3` 可将体积压缩到原始的 50%-60%

## 6. 全局配置

### `set_config` / `get_config`

#### 作用

`set_config` 可修改 sklearn 的全局行为配置。`transform_output='pandas'` 在数据分析阶段更友好。配置变更应及时恢复，避免影响其他流程。

#### 重点方法

```python
sklearn.get_config()
sklearn.set_config(transform_output='pandas')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `transform_output` | `str` | 变换器输出格式：`"default"` / `"pandas"`，默认为 `"default"` | `"pandas"` |
| `assume_finite` | `bool` | 是否假设数据已有限，默认为 `False` | `True` |
| `working_memory` | `int` | 算法内存上限（MB），默认为 `1024` | `2048` |
| `enable_metadata_routing` | `bool` | 启用元数据路由（1.4+），默认为 `False` | `True` |

#### 示例代码

```python
from sklearn import get_config, set_config

print(f"当前 transform_output: {get_config()['transform_output']}")

set_config(transform_output="pandas")
print(f"设置后: {get_config()['transform_output']}")

set_config(transform_output="default")
print(f"恢复后: {get_config()['transform_output']}")
```

#### 输出

```text
当前 transform_output: default
设置后: pandas
恢复后: default
```

#### 理解重点

- 全局配置适合实验和分析，不宜在库代码中隐式修改
- 多人协作时建议显式记录配置变更
- 配置差异可能导致同一代码输出格式不一致
- `transform_output='pandas'` 让所有 `transform` 输出 DataFrame——在 EDA 阶段非常实用

## 7. 版本检查

### `sklearn.__version__` + `packaging.version`

#### 作用

不同 sklearn 版本 API 可用性不同，需显式校验。使用 `packaging.version` 比字符串比较更可靠。版本门控能避免线上环境 API 不匹配。

#### 重点方法

```python
import sklearn; sklearn.__version__
from packaging import version
version.parse(sklearn.__version__) >= version.parse('1.2')
```

#### 示例代码

```python
import sklearn
from packaging import version

print(f"sklearn 版本: {sklearn.__version__}")
print(f">= 1.0: {version.parse(sklearn.__version__) >= version.parse('1.0')}")
print(f">= 1.2 (set_output API): {version.parse(sklearn.__version__) >= version.parse('1.2')}")
print(f">= 1.6: {version.parse(sklearn.__version__) >= version.parse('1.6')}")
```

#### 输出

```text
sklearn 版本: 1.6.1
>= 1.0: True
>= 1.2 (set_output API): True
>= 1.6: True
```

#### 理解重点

- 版本门控应成为工具脚本与部署脚本的标准步骤
- 当文档示例依赖新特性时，必须标注最低版本要求
- 建议将关键依赖版本固定在项目配置中
- `packaging.version.parse` 正确处理 `1.10 > 1.2` 的语义化比较

## 8. all_estimators 快速检索

### `all_estimators`

#### 作用

`all_estimators` 可快速查看当前环境可用估计器。适合做模型候选池构建与自动化实验初始化。`type_filter` 可按任务类型筛选。

#### 重点方法

```python
all_estimators(type_filter=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `type_filter` | `str` 或 `None` | 筛选类型：`"classifier"` / `"regressor"` / `"transformer"` / `"cluster"` | `"classifier"` |

#### 示例代码

```python
from sklearn.utils import all_estimators

classifiers = all_estimators(type_filter="classifier")
regressors = all_estimators(type_filter="regressor")
transformers = all_estimators(type_filter="transformer")

print(f"分类器: {len(classifiers)} 个")
print(f"回归器: {len(regressors)} 个")
print(f"转换器: {len(transformers)} 个")
print(f"\n分类器前 5: {[name for name, _ in classifiers[:5]]}")
```

#### 输出

```text
分类器: 49 个
回归器: 55 个
转换器: 93 个

分类器前 5: ['AdaBoostClassifier', 'BaggingClassifier', 'BernoulliNB', 'CalibratedClassifierCV', 'CategoricalNB']
```

#### 理解重点

- 该工具可用于快速探索，但不替代模型选择流程
- 不同版本中估计器数量会变化，应结合版本信息解读
- 可与自动化评估框架结合构建候选模型库
- 返回的是类对象（不是实例）——需实例化后才能使用

## 常见坑

1. 把 clone 误认为深拷贝训练状态，导致实验误判
2. 忽略版本差异直接调用新 API，引发环境兼容问题
3. 持久化后不做预测一致性校验，埋下线上风险
4. 自定义 Transformer 未将 `__init__` 参数存为同名属性——`get_params` 失效

## 小结

- 技巧章节的核心是把"能跑"升级为"可维护、可复现、可部署"
- 推荐将权重策略、版本校验、持久化检查纳入项目模板
- clone + get_params/set_params 是自动化实验的基础设施
- 自定义 Transformer + Pipeline 可将业务规则工程化
- 模型文件、版本、配置三者应统一管理
