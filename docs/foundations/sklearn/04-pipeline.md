---
title: Pipeline
outline: deep
---

# Pipeline

> 对应脚本：`Basic/ScikitLearn/04_pipeline.py`
> 运行方式：`python Basic/ScikitLearn/04_pipeline.py`（仓库根目录）

## 本章目标

1. 掌握 `Pipeline` 与 `make_pipeline` 的构建方式与差异。
2. 学会访问流水线步骤、读取与修改步骤参数。
3. 理解双下划线参数命名规则在调参中的作用。
4. 掌握 `ColumnTransformer` 处理混合类型特征的标准写法。
5. 学会把目标变量变换纳入回归流程。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `Pipeline(steps=[...])` | 显式命名步骤构建流水线 | `demo_basic_pipeline` |
| `make_pipeline(...)` | 自动命名步骤快速构建 | `demo_basic_pipeline` |
| `pipe.steps` / `pipe.named_steps` | 访问步骤与对象 | `demo_access_steps` |
| `set_params(step__param=value)` | 修改子步骤参数 | `demo_set_params` |
| `GridSearchCV(pipe, param_grid, ...)` | 流水线联合调参 | `demo_pipeline_gridsearch` |
| `set_params(step='passthrough')` | 跳过某步骤 | `demo_skip_step` |
| `ColumnTransformer([...])` | 按列类型组合预处理 | `demo_column_transformer` |
| `TransformedTargetRegressor(...)` | 目标变量变换回归 | `demo_transformed_target` |

## 1. Pipeline 基础

### 方法重点

- 流水线把预处理和模型封装成一个可复用对象，避免训练与推理逻辑分叉。
- `Pipeline` 需要手动命名步骤；`make_pipeline` 自动命名，写法更短。
- 统一对象后，可直接调用 `fit`、`predict`、`score`。

### 参数速览（本节）

1. `Pipeline(steps=[('scaler', ...), ('pca', ...), ('svm', ...)])`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `steps` | 三步流程 | 显式定义步骤名与变换器/估计器 |
| 返回值 | `Pipeline` | 可统一调用 `fit/predict/score` 的流水线对象 |

2. `make_pipeline(StandardScaler(), PCA(n_components=2), SVC())`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_components` | `2` | PCA 保留 2 维主成分 |
| 返回值 | `Pipeline` | 自动命名步骤后的流水线对象 |

3. `score(X_test, y_test)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_test` / `y_test` | 测试集 | 评估输入 |
| 返回值 | `float` | 在测试集上的准确率 |

### 示例代码

```python
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
	iris.data, iris.target, test_size=0.3, random_state=42
)

pipe = Pipeline([
	("scaler", StandardScaler()),
	("pca", PCA(n_components=2)),
	("svm", SVC()),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))

pipe_auto = make_pipeline(StandardScaler(), PCA(n_components=2), SVC())
print(pipe_auto.steps)
```

### 结果输出（示例）

```text
Pipeline 准确率: 0.9778
----------------
步骤名称: ['scaler', 'pca', 'svm']
----------------
自动命名: ['standardscaler', 'pca', 'svc']
```

### 理解重点

- 只要对象实现 sklearn 接口，就能被纳入同一流水线。
- 步骤命名不是装饰，而是后续调参与调试的锚点。
- 训练完成的流水线可整体持久化，部署更稳定。

## 2. 访问 Pipeline 步骤

### 方法重点

- 可以用 `steps`、`named_steps`、整数索引多种方式访问组件。
- `named_steps` 适合生产代码，稳定且可读。
- 步骤对象可直接拿出来检查参数或属性。

### 参数速览（本节）

1. `pipe.steps`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回值 | `list[tuple]` | 步骤名与对象元组列表 |

2. `pipe.named_steps['name']`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `name` | `'pca'` | 用步骤名访问目标变换器 |
| 返回值 | 映射对象/步骤对象 | 名称到步骤对象的映射或具体步骤 |

3. `pipe[index]`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `index` | `0`、`-1` | 用索引访问首尾步骤 |
| 返回值 | 步骤对象 | 对应索引位置的变换器或估计器 |

### 示例代码

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

print(pipe.steps)
print(pipe.named_steps["pca"])
print(pipe[0])
print(pipe[-1])
```

### 结果输出（示例）

```text
pipe.steps: [('scaler', StandardScaler()), ('pca', PCA(n_components=2)), ('svm', SVC())]
----------------
pipe.named_steps['pca']: PCA(n_components=2)
----------------
pipe[0]: StandardScaler()
----------------
pipe[-1]: SVC()
```

### 理解重点

- 大多数调试问题都能通过检查 `named_steps` 快速定位。
- 步骤顺序直接影响输入输出维度和模型表现。
- 推荐在文档和代码中保持统一步骤命名规范。

## 3. Pipeline 参数设置

### 方法重点

- 子步骤参数通过 `步骤名__参数名` 写法进行设置。
- 该规则同样适用于网格搜索与随机搜索。
- `set_params` 返回对象自身，支持链式调用。

### 参数速览（本节）

1. `pipe.set_params(pca__n_components=3, svm__C=10)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `pca__n_components` | `3` | 修改 PCA 维度 |
| `svm__C` | `10` | 修改 SVM 正则强度 |
| 返回值 | `Pipeline` | 修改后的流水线对象 |

2. `pipe.get_params()`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回值 | `dict` | 包含所有可调参数 |

### 示例代码

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

pipe.set_params(pca__n_components=3, svm__C=10)
print(pipe.named_steps["pca"].n_components)
print(pipe.named_steps["svm"].C)
```

### 结果输出（示例）

```text
修改前: PCA n_components=2, SVM C=1.0
----------------
修改后: PCA n_components=3, SVM C=10
```

### 理解重点

- 双下划线规则是 sklearn 组合对象调参的核心约定。
- 复杂流水线里，参数命名准确性直接决定调参是否生效。
- 在 [模型选择](/foundations/sklearn/05-model-selection) 中会大规模使用该规则。

## 4. Pipeline + GridSearchCV

### 方法重点

- 将预处理与模型打包后调参，可避免预处理阶段数据泄露。
- 参数网格按 `步骤名__参数名` 书写。
- 网格搜索会自动在交叉验证中重复完整流水线。

### 参数速览（本节）

1. `GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `param_grid` | `{'svc__C': [0.1, 1, 10], 'svc__kernel': ['linear', 'rbf']}` | 待搜索参数网格 |
| `cv` | `5` | 五折交叉验证 |
| `scoring` | `'accuracy'` | 评估指标 |

2. `fit(X, y)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X` / `y` | 特征与标签 | 网格搜索输入 |
| 返回属性 `best_params_` | 训练后自动生成 | 最优参数组合 |
| 返回属性 `best_score_` | 训练后自动生成 | 对应平均验证分数 |

### 示例代码

```python
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

iris = datasets.load_iris()
pipe = make_pipeline(StandardScaler(), SVC())

param_grid = {
	"svc__C": [0.1, 1, 10],
	"svc__kernel": ["linear", "rbf"],
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy")
grid.fit(iris.data, iris.target)
print(grid.best_params_)
print(grid.best_score_)
```

### 结果输出（示例）

```text
最佳参数: {'svc__C': 1, 'svc__kernel': 'linear'}
----------------
最佳得分: 0.9733
```

### 理解重点

- 将预处理写进流水线后，调参与训练流程天然一致。
- 网格搜索成本高，参数空间要先做工程化收敛。
- 实战可先随机搜索粗定位，再网格精搜。

## 5. 跳过步骤

### 方法重点

- 可用 `passthrough` 暂时禁用某步骤，便于做消融实验。
- 消融结果可帮助判断该步骤是否真正贡献性能。
- 对比实验要保证其他设置不变，避免混杂结论。

### 参数速览（本节）

1. `pipe.set_params(pca='passthrough')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `pca` | `'passthrough'` | 跳过 PCA 步骤 |
| 返回值 | `Pipeline` | 修改后的流水线对象 |

2. `score(X_test, y_test)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_test` / `y_test` | 测试集 | 用于对比评估 |
| 返回值 | `float` | 跳过步骤后的性能 |

### 示例代码

```python
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
	iris.data, iris.target, test_size=0.3, random_state=42
)

pipe = Pipeline([
	("scaler", StandardScaler()),
	("pca", PCA(n_components=2)),
	("svm", SVC()),
])

pipe.set_params(pca="passthrough")
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

### 结果输出（示例）

```text
跳过 PCA 后准确率: 1.0000
----------------
当前 pca 步骤: passthrough
```

### 理解重点

- `passthrough` 是快速做 A/B 对比的高效工具。
- 某步骤“可删”不代表永远不需要，取决于任务与数据规模。
- 该技巧也适用于特征选择、标准化等模块。

## 6. ColumnTransformer 混合类型处理

### 方法重点

- 数值列和类别列应分开处理，再统一拼接。
- 通常做法是“子流水线 + 列选择器 + 总流水线”三层结构。
- 这是结构化表格任务最常见的工程模板。

### 参数速览（本节）

1. `ColumnTransformer([...])`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `transformers` | 数值与类别两条子流水线 | 分列处理后再拼接 |
| 返回值（`get_feature_names_out`） | `ndarray[str]` | 预处理后特征名 |

2. `make_column_selector(dtype_include='number'|'object')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `dtype_include` | `'number'`、`'object'` | 按列类型选择特征 |
| 返回值 | `callable` | 供 `ColumnTransformer` 使用的列选择器 |

3. `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `handle_unknown` | `'ignore'` | 推理阶段未知类别忽略 |
| `sparse_output` | `False` | 独热编码输出稠密数组 |
| 返回值（`fit_transform`） | `ndarray` | 类别编码结果 |

### 示例代码

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.DataFrame({
	"年龄": [25, 30, np.nan, 40],
	"收入": [50000, 60000, 55000, np.nan],
	"学历": ["本科", "硕士", "本科", "博士"],
})
y = [0, 1, 0, 1]

full_pipe = Pipeline([
	("preprocessor", ColumnTransformer([
		("num", Pipeline([
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		]), selector(dtype_include="number")),
		("cat", Pipeline([
			("imputer", SimpleImputer(strategy="most_frequent")),
			("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
		]), selector(dtype_include="object")),
	])),
	("classifier", LogisticRegression()),
])

full_pipe.fit(df, y)
print(full_pipe.named_steps["preprocessor"].get_feature_names_out())
```

### 结果输出（示例）

```text
特征名:
['num__年龄' 'num__收入' 'cat__学历_博士' 'cat__学历_本科' 'cat__学历_硕士']
```

### 理解重点

- 该模式与 [预处理](/foundations/sklearn/02-preprocessing) 章节互补：02 讲单点方法，04 讲工程组合。
- 列选择逻辑建议写成可测试函数，降低 schema 变更风险。
- 训练后要检查输出特征名，确保下游解释与监控一致。

## 7. TransformedTargetRegressor

### 方法重点

- 对目标变量做变换（如对数）可缓解长尾分布问题。
- 训练在变换空间进行，预测自动做逆变换返回原尺度。
- 常见于金额、流量等偏态回归任务。

### 参数速览（本节）

1. `TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `regressor` | `LinearRegression()` | 基础回归器 |
| `func` | `np.log1p` | 目标变量前向变换 |
| `inverse_func` | `np.expm1` | 预测值逆变换回原尺度 |

2. `fit(X_train, y_train)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X_train` / `y_train` | 训练集 | 训练输入 |
| 返回值（`score`） | `float` | 在原目标尺度上的 R² |

### 示例代码

```python
import numpy as np
from sklearn import datasets
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
	diabetes.data, diabetes.target, test_size=0.3, random_state=42
)

lr = LinearRegression().fit(X_train, y_train)
print(lr.score(X_test, y_test))

ttr = TransformedTargetRegressor(
	regressor=LinearRegression(),
	func=np.log1p,
	inverse_func=np.expm1,
)
ttr.fit(X_train, y_train)
print(ttr.score(X_test, y_test))
```

### 结果输出（示例）

```text
普通回归 R²: 0.4773
----------------
目标变量对数变换后 R²: 0.4314
```

### 理解重点

- 是否使用目标变换应通过验证集结果决定，不是固定加分项。
- 变换后指标变化要结合业务误差定义解读。
- 若目标存在 0 或负值，要确认变换函数可用性。

## 常见坑

1. 先全量标准化再切分训练测试，导致数据泄露。
2. 参数名漏写步骤前缀，导致 `set_params` 或网格搜索未生效。
3. `ColumnTransformer` 输出特征名未验证，导致下游解释错位。

## 小结

- Pipeline 是 sklearn 工程化落地的基础组件。
- 推荐把预处理、特征构造、模型训练统一封装，再进行调参与部署。
- 下一步可结合 [模型选择](/foundations/sklearn/05-model-selection) 做系统评估。
