---
title: sklearn Pipeline
outline: deep
---

# sklearn Pipeline

## 本章目标

1. 掌握 `Pipeline` 与 `make_pipeline` 的构建方式与差异
2. 学会访问流水线步骤、读取与修改步骤参数
3. 理解双下划线参数命名规则在调参中的作用
4. 掌握 `ColumnTransformer` 处理混合类型特征的标准写法
5. 学会把目标变量变换纳入回归流程

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `Pipeline(steps)` | 构造器 | 显式命名步骤构建流水线 |
| `make_pipeline(*steps)` | 函数 | 自动命名步骤快速构建 |
| `pipe.named_steps` | 属性 | 按名称访问步骤对象 |
| `pipe.set_params(**params)` | 方法 | 修改子步骤参数（双下划线规则） |
| `ColumnTransformer(transformers)` | 构造器 | 按列类型组合预处理 |
| `TransformedTargetRegressor(regressor)` | 构造器 | 目标变量变换回归 |

## 1. Pipeline 基础

### `Pipeline` / `make_pipeline`

#### 作用

流水线把预处理和模型封装成一个可复用对象，避免训练与推理逻辑分叉。`Pipeline` 需要手动命名步骤；`make_pipeline` 自动命名，写法更短。统一对象后可整体调用 `fit`、`predict`、`score`。

#### 重点方法

```python
Pipeline(steps, *, memory=None, verbose=False)
make_pipeline(*steps, memory=None, verbose=False)
# 核心方法：fit(X, y) → predict(X) → score(X, y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `steps` | `list[tuple[str, estimator]]` | 步骤名与变换器/估计器列表 | `[("scaler", StandardScaler()), ("svm", SVC())]` |
| `*steps` | `estimator` | make_pipeline：按顺序传入变换器/估计器 | `StandardScaler(), SVC()` |
| `memory` | `str` 或 `None` | 缓存路径，None 不缓存，默认为 `None` | `"./cache"` |
| `verbose` | `bool` | 是否输出步骤耗时，默认为 `False` | `True` |

训练后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `steps` | `list[tuple[str, estimator]]` | 步骤名与已训练对象列表 |
| `named_steps` | `utils.Bunch` | 按名称访问步骤对象 |
| `classes_` | `ndarray` | 最终估计器的类别标签 |

#### 示例代码

```python
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2)),
    ("svm", SVC()),
])
pipe.fit(X_train, y_train)
print(f"Pipeline 准确率: {pipe.score(X_test, y_test):.4f}")

pipeAuto = make_pipeline(StandardScaler(), PCA(n_components=2), SVC())
pipeAuto.fit(X_train, y_train)
print(f"make_pipeline 准确率: {pipeAuto.score(X_test, y_test):.4f}")
print(f"自动命名步骤: {[name for name, _ in pipeAuto.steps]}")
```

#### 输出

```text
Pipeline 准确率: 0.9556
make_pipeline 准确率: 0.9556
自动命名步骤: ['standardscaler', 'pca', 'svc']
```

#### 理解重点

- 只要对象实现 sklearn 接口，就能被纳入同一流水线
- 步骤命名不是装饰，而是后续调参与调试的锚点
- 训练完成的流水线可整体持久化，部署更稳定
- 最后一步可以是分类器、回归器或任何估计器

## 2. 访问 Pipeline 步骤

### `pipe.steps` / `named_steps`

#### 作用

可以用 `steps`、`named_steps`、整数索引多种方式访问组件。`named_steps` 适合生产代码，稳定且可读。步骤对象可直接拿出来检查参数或属性。

#### 重点方法

```python
pipe.steps          # → list[tuple[str, estimator]]
pipe.named_steps    # → Bunch（属性式访问）
pipe[index]         # → 第 index 个步骤对象
pipe[-1]            # → 最后一个步骤（通常是预估器）
```

#### 示例代码

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2)),
    ("svm", SVC()),
])

print(f"步骤列表: {pipe.steps}")
print(f"named_steps['pca']: {pipe.named_steps['pca']}")
print(f"pipe[0]: {pipe[0]}")
print(f"pipe[-1]: {pipe[-1]}")
```

#### 输出

```text
步骤列表: [('scaler', StandardScaler()), ('pca', PCA(n_components=2)), ('svm', SVC())]
named_steps['pca']: PCA(n_components=2)
pipe[0]: StandardScaler()
pipe[-1]: SVC()
```

#### 理解重点

- 大多数调试问题都能通过检查 `named_steps` 快速定位
- 步骤顺序直接影响输入输出维度和模型表现
- 推荐在文档和代码中保持统一步骤命名规范
- 索引访问适合循环遍历步骤做诊断

## 3. Pipeline 参数设置

### `set_params`

#### 作用

子步骤参数通过 `步骤名__参数名` 写法进行设置。该规则同样适用于网格搜索与随机搜索。`set_params` 返回对象自身，支持链式调用。

#### 重点方法

```python
pipe.set_params(**params)      # 修改步骤参数，返回 self
pipe.get_params(deep=True)     # 获取所有可调参数字典
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `**params` | `dict` | 子步骤参数，格式 `step__param=value` | `pca__n_components=3` |
| `deep` | `bool` | 是否递归获取嵌套参数，默认为 `True` | `True` |

#### 示例代码

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2)),
    ("svm", SVC(C=1.0)),
])

print(f"修改前: PCA n_components={pipe.named_steps['pca'].n_components}, "
      f"SVM C={pipe.named_steps['svm'].C}")

pipe.set_params(pca__n_components=3, svm__C=10)
print(f"修改后: PCA n_components={pipe.named_steps['pca'].n_components}, "
      f"SVM C={pipe.named_steps['svm'].C}")

# 查看完整参数
print(f"pca__n_components: {pipe.get_params()['pca__n_components']}")
```

#### 输出

```text
修改前: PCA n_components=2, SVM C=1.0
修改后: PCA n_components=3, SVM C=10
pca__n_components: 3
```

#### 理解重点

- 双下划线规则是 sklearn 组合对象调参的核心约定
- 复杂流水线里，参数命名准确性直接决定调参是否生效
- `get_params()` 返回的键名即为 `set_params` 接受的参数名
- 在网格搜索中大规模使用该规则（见下节）

## 4. Pipeline + GridSearchCV

### `GridSearchCV` 联合调参

#### 作用

将预处理与模型打包后调参，可避免预处理阶段数据泄露。参数网格按 `步骤名__参数名` 书写。网格搜索会自动在交叉验证中重复完整流水线，保证评估无偏。

#### 重点方法

```python
GridSearchCV(estimator, param_grid, *, scoring=None, n_jobs=None, cv=None, verbose=0)
# fit(X, y) → best_params_ / best_score_ / predict(X)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `estimator` | 待调参的流水线或模型 | `make_pipeline(StandardScaler(), SVC())` |
| `param_grid` | `dict` 或 `list[dict]` | 搜索参数网格，键用 `step__param` 格式 | `{"svc__C": [0.1, 1, 10]}` |
| `scoring` | `str` 或 `callable` | 评估指标，默认为 `None`（用估计器默认） | `"accuracy"` |
| `cv` | `int` 或 `splitter` | 交叉验证折数，默认为 `None`（5 折） | `5` |
| `n_jobs` | `int` 或 `None` | 并行数，`-1` 使用全部核心，默认为 `None` | `-1` |
| `verbose` | `int` | 日志详细度，默认为 `0` | `1` |

训练后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `best_params_` | `dict` | 最优参数组合 |
| `best_score_` | `float` | 最优参数对应的交叉验证平均分 |
| `best_estimator_` | `estimator` | 用最优参数在全量数据训练的模型 |
| `cv_results_` | `dict` | 完整搜索结果（各参数组合的分值、时间） |

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

grid = GridSearchCV(pipe, paramGrid, cv=5, scoring="accuracy")
grid.fit(X, y)
print(f"最佳参数: {grid.best_params_}")
print(f"最佳得分: {grid.best_score_:.4f}")
print(f"最佳模型: {grid.best_estimator_}")
```

#### 输出

```text
最佳参数: {'svc__C': 1, 'svc__kernel': 'linear'}
最佳得分: 0.9733
最佳模型: Pipeline(steps=[('standardscaler', StandardScaler()), ('svc', SVC(C=1, kernel='linear'))])
```

#### 理解重点

- 将预处理写进流水线后，调参与训练流程天然一致——每折内部做 `fit_transform`，外部做 `transform`
- 网格搜索成本高，参数空间要先做工程化收敛
- 实战可先随机搜索粗定位，再网格精搜
- `cv_results_` 包含所有候选参数的分值，可用于进一步分析

## 5. 跳过步骤

### `set_params(step='passthrough')`

#### 作用

可用 `passthrough` 暂时禁用某步骤，便于做消融实验。消融结果可帮助判断该步骤是否真正贡献性能。对比实验要保证其他设置不变，避免混杂结论。

#### 重点方法

```python
pipe.set_params(step_name='passthrough')
# 设置后该步骤不执行任何变换，输出 = 输入
```

#### 示例代码

```python
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2)),
    ("svm", SVC()),
])
pipe.fit(X_train, y_train)
print(f"含 PCA: {pipe.score(X_test, y_test):.4f}")

pipe.set_params(pca="passthrough")
pipe.fit(X_train, y_train)
print(f"跳过 PCA: {pipe.score(X_test, y_test):.4f}")
print(f"当前 pca 步骤: {pipe.named_steps['pca']}")
```

#### 输出

```text
含 PCA: 0.9556
跳过 PCA: 0.9778
当前 pca 步骤: passthrough
```

#### 理解重点

- `passthrough` 是快速做 A/B 对比的高效工具
- 某步骤"可删"不代表永远不需要，取决于任务与数据规模
- 该技巧也适用于特征选择、标准化等模块
- 消融实验应成为建模流程的标准步骤

## 6. ColumnTransformer 混合类型处理

### `ColumnTransformer`

#### 作用

数值列和类别列应分开处理，再统一拼接。`ColumnTransformer` 将多条子流水线拼接为统一特征空间——这是生产级预处理的核心模式，后续可直接接入模型。

#### 重点方法

```python
ColumnTransformer(transformers, *, remainder='drop', sparse_threshold=0.3,
                 n_jobs=None, verbose=False, verbose_feature_names_out=True)
# fit(X) → transform(X) / fit_transform(X) → get_feature_names_out()
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `transformers` | `list[tuple[str, estimator, columns]]` | 列分组与对应处理器列表 | `[("num", numPipe, [0, 1])]` |
| `remainder` | `str` 或 `estimator` | 未指定列的处理：`"drop"` 丢弃 / `"passthrough"` 保留，默认为 `"drop"` | `"passthrough"` |
| `sparse_threshold` | `float` | 稀疏阈值，低于此比例输出稀疏矩阵，默认为 `0.3` | `0.3` |
| `n_jobs` | `int` 或 `None` | 并行数，默认为 `None` | `None` |
| `verbose` | `bool` | 是否输出耗时信息，默认为 `False` | `True` |
| `verbose_feature_names_out` | `bool` | 特征名是否加前缀，默认为 `True` | `True` |

快捷列选择器：

```python
from sklearn.compose import make_column_selector
make_column_selector(dtype_include='number')   # 按类型自动选择列
make_column_selector(dtype_include='object')   # 选字符串列
make_column_selector(pattern='.*')             # 按列名正则匹配
```

#### 示例代码

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.DataFrame({
    "年龄": [25, 30, np.nan, 40, 35],
    "收入": [50000, 60000, 55000, np.nan, 70000],
    "学历": ["本科", "硕士", "本科", "博士", "硕士"],
})
y = [0, 1, 0, 1, 0]

numPipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

catPipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numPipe, selector(dtype_include="number")),
    ("cat", catPipe, selector(dtype_include="object")),
])

fullPipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

fullPipe.fit(df, y)
print(f"特征名: {fullPipe.named_steps['preprocessor'].get_feature_names_out()}")
print(f"预测: {fullPipe.predict(df)}")
```

#### 输出

```text
特征名: ['num__年龄' 'num__收入' 'cat__学历_博士' 'cat__学历_本科' 'cat__学历_硕士']
预测: [0 1 0 1 0]
```

#### 理解重点

- 列级流水线能把预处理逻辑完全纳入模型训练过程——减少线上线下不一致
- 该模式可直接嵌入 `Pipeline` 做联合调参与部署
- `make_column_selector` 按 dtype 自动选列——避免手动列名硬编码
- 当类别空间较大时应关注 One-Hot 维度膨胀问题
- 训练后要检查输出特征名，确保下游解释与监控一致

## 7. TransformedTargetRegressor

### `TransformedTargetRegressor`

#### 作用

对目标变量做变换（如对数）可缓解长尾分布问题。训练在变换空间进行，预测自动做逆变换返回原尺度。常见于金额、流量等偏态回归任务。

#### 重点方法

```python
TransformedTargetRegressor(regressor=None, *, transformer=None, func=None,
                           inverse_func=None, check_inverse=True)
# fit(X, y) → predict(X) → score(X, y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `regressor` | `estimator` | 基础回归器，默认为 `None`（需指定） | `LinearRegression()` |
| `transformer` | `estimator` | 变换器（需实现 `transform`/`inverse_transform`） | `PowerTransformer(method="box-cox")` |
| `func` | `callable` 或 `None` | 目标变量前向变换函数 | `np.log1p` |
| `inverse_func` | `callable` 或 `None` | 预测值逆变换函数 | `np.expm1` |
| `check_inverse` | `bool` | 是否检查逆变换一致性，默认为 `True` | `True` |

#### 示例代码

```python
import numpy as np
from sklearn import datasets
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X, y = datasets.load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

lr = LinearRegression().fit(X_train, y_train)
print(f"普通回归 R²: {lr.score(X_test, y_test):.4f}")

ttr = TransformedTargetRegressor(
    regressor=LinearRegression(),
    func=np.log1p,
    inverse_func=np.expm1,
)
ttr.fit(X_train, y_train)
print(f"对数目标变换 R²: {ttr.score(X_test, y_test):.4f}")
```

#### 输出

```text
普通回归 R²: 0.4773
对数目标变换 R²: 0.4314
```

#### 理解重点

- 是否使用目标变换应通过验证集结果决定，不是固定加分项
- 变换后指标变化要结合业务误差定义解读
- 若目标存在 0 或负值，要确认变换函数可用性（`yeo-johnson` 可处理非正数）
- `func`/`inverse_func` 与 `transformer` 二选一即可

## 常见坑

1. 先全量标准化再切分训练测试，导致数据泄露——流水线也不能解决外部 fit 的问题
2. 参数名漏写步骤前缀，导致 `set_params` 或网格搜索未生效
3. `ColumnTransformer` 的 `remainder='drop'` 会静默丢弃未指定列——建议显式设置
4. 用 `make_pipeline` 时忘记步骤名由类名决定——调参时需查看 `steps` 确认名称
5. `TransformedTargetRegressor` 的 `check_inverse` 为 `True` 时可能触发逆变换验证失败

## 小结

- Pipeline 是 sklearn 工程化落地的基础组件——推荐把预处理、特征构造、模型训练统一封装
- 双下划线参数命名是贯通 Pipeline 调参的唯一规则——`get_params` 返回的键名即网格搜索键名
- `ColumnTransformer` + `Pipeline` 是生产级预处理的标准模式——按列类型分而治之
- `passthrough` 与 `TransformedTargetRegressor` 是流程实验和优化的重要辅助工具
