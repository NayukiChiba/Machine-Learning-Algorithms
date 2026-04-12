---
title: 特征
outline: deep
---

# 特征

> 对应脚本：`Basic/ScikitLearn/03_feature_engineering.py`
> 运行方式：`python Basic/ScikitLearn/03_feature_engineering.py`（仓库根目录）

## 本章目标

1. 掌握文本、字典、数值三类常见特征构造方式。
2. 理解 `TfidfVectorizer`、`PolynomialFeatures` 的参数影响。
3. 学会使用过滤法、包裹法、模型法进行特征选择。
4. 明确高维特征场景下的稀疏表示与维度控制策略。
5. 建立特征工程与模型训练、调参之间的衔接思路。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `CountVectorizer()` | 将文本转为词频特征 | `demo_count_vectorizer` |
| `TfidfVectorizer(...)` | 将文本转为 TF-IDF 特征 | `demo_tfidf_vectorizer` |
| `DictVectorizer(sparse=False)` | 将字典样本展开为特征矩阵 | `demo_dict_vectorizer` |
| `PolynomialFeatures(...)` | 生成多项式与交互项特征 | `demo_polynomial_features` |
| `VarianceThreshold(threshold)` | 过滤低方差特征 | `demo_variance_threshold` |
| `SelectKBest(f_classif, k=2)` | 基于统计检验筛选特征 | `demo_select_k_best` |
| `RFE(estimator, ...)` | 递归消除特征 | `demo_rfe` |
| `SelectFromModel(estimator, ...)` | 基于模型重要性筛选特征 | `demo_select_from_model` |

## 1. CountVectorizer 词频统计

### 方法重点

- 词袋模型仅统计出现次数，不考虑词序与语义。
- 返回稀疏矩阵，适合高维文本特征表达。
- 词汇表由训练语料自动学习，推理阶段应复用同一向量器。

### 参数速览（本节）

1. `CountVectorizer()`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `lowercase` | 默认 `True` | 统一小写化 |
| `token_pattern` | 默认 `(?u)\b\w\w+\b` | 分词正则，至少两个字符 |
| `ngram_range` | 默认 `(1, 1)` | 一元词袋 |

2. `fit_transform(corpus)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `corpus` | 文本列表 | 输入文档集合 |
| 返回值 | `csr_matrix` | 文档-词项稀疏矩阵 |

3. `get_feature_names_out()`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回值 | `ndarray[str]` | 词汇表 |

### 示例代码

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
	"This is the first document.",
	"This document is the second document.",
	"And this is the third one.",
]

cv = CountVectorizer()
X = cv.fit_transform(corpus)

print(cv.get_feature_names_out())
print(X.toarray())
```

### 结果输出（示例）

```text
词汇表: ['and' 'document' 'first' 'is' 'one' 'second' 'the' 'third' 'this']
----------------
词频矩阵:
[[0 1 1 1 0 0 1 0 1]
 [0 2 0 1 0 1 1 0 1]
 [1 0 0 1 1 0 1 1 1]]
----------------
稀疏矩阵类型: <class 'scipy.sparse._csr.csr_matrix'>
```

### 理解重点

- Count 特征易解释，但无法表达词义接近关系。
- 对文本分类基线非常有效，常与线性模型搭配。
- 停用词、ngram 与最小词频阈值是最常见调参点。

## 2. TfidfVectorizer

### 方法重点

- TF-IDF 在词频基础上引入逆文档频率，削弱高频泛化词影响。
- `max_df`、`min_df`、`ngram_range` 会直接改变特征空间大小。
- 文本规模上升后，建议先限制 `max_features` 控制维度。

### 参数速览（本节）

1. `TfidfVectorizer()`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `max_features` | 默认 `None` | 不限制特征数 |
| `ngram_range` | 默认 `(1, 1)` | 使用 unigram |
| `stop_words` | 默认 `None` | 不启用停用词过滤 |

2. `TfidfVectorizer(max_features=100, min_df=1, max_df=0.9, ngram_range=(1, 2), stop_words='english')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `max_features` | `100` | 限制最大词特征数 |
| `min_df` | `1` | 至少出现在 1 篇文档 |
| `max_df` | `0.9` | 忽略出现在 90% 以上文档的词 |
| `ngram_range` | `(1, 2)` | 使用 unigram 与 bigram |
| `stop_words` | `'english'` | 英文停用词过滤 |

### 示例代码

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
	"This is the first document.",
	"This document is the second document.",
	"And this is the third one.",
]

tfidf = TfidfVectorizer(max_features=100, min_df=1, max_df=0.9, ngram_range=(1, 2), stop_words="english")
X = tfidf.fit_transform(corpus)

print(X.shape)
print(tfidf.get_feature_names_out())
```

### 结果输出（示例）

```text
基础用法 - 特征数: 9
----------------
高级用法 - 特征数: 8
----------------
词汇表: ['document' 'document second' 'second' 'third' 'third one' 'document']
```

### 理解重点

- TF-IDF 更关注区分度高的词，不等于语义建模。
- 大多数场景可从 unigram 开始，再试 `(1, 2)`。
- 中文文本需要自定义分词与停用词策略。

## 3. DictVectorizer

### 方法重点

- 适合结构化字典输入，自动展开类别字段与数值字段。
- 对类别键做独热展开，对数值键保留原值。
- 可通过 `inverse_transform` 回看特征与原字典映射关系。

### 参数速览（本节）

1. `DictVectorizer(sparse=False)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `sparse` | `False` | 返回稠密矩阵，便于教学展示 |

2. `fit_transform(data)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `data` | `list[dict]` | 字典样本集合 |
| 返回值 | `ndarray` | 特征矩阵 |

3. `get_feature_names_out()`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回值 | `ndarray[str]` | 展开后的特征名 |

4. `inverse_transform(X)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X` | 特征矩阵 | 向量化后的输入 |
| 返回值 | `list[dict]` | 逆映射结果 |

### 示例代码

```python
from sklearn.feature_extraction import DictVectorizer

data = [
	{"city": "北京", "temperature": 20},
	{"city": "上海", "temperature": 25},
	{"city": "北京", "temperature": 18},
]

dv = DictVectorizer(sparse=False)
X = dv.fit_transform(data)

print(dv.get_feature_names_out())
print(X)
print(dv.inverse_transform(X)[0])
```

### 结果输出（示例）

```text
特征名: ['city=上海' 'city=北京' 'temperature']
----------------
特征矩阵:
[[0. 1. 20.]
 [1. 0. 25.]
 [0. 1. 18.]]
----------------
逆变换: {'city=北京': 1.0, 'temperature': 20.0}
```

### 理解重点

- DictVectorizer 常用于日志特征、规则特征、浅层推荐特征工程。
- 产物特征名可追踪，利于可解释性。
- 类别字段较多时，建议结合频次阈值做后续裁剪。

## 4. PolynomialFeatures

### 方法重点

- 多项式扩展可让线性模型拟合非线性关系。
- 维度增长很快，`degree` 不宜盲目增大。
- `interaction_only=True` 可只保留交互项，降低维度膨胀。

### 参数速览（本节）

1. `PolynomialFeatures(degree=2, include_bias=True)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `degree` | `2` | 最高多项式阶数 |
| `include_bias` | `True` | 包含常数项 1 |
| `interaction_only` | 默认 `False` | 同时包含平方项和交互项 |

2. `PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `degree` | `2` | 最高多项式阶数 |
| `interaction_only` | `True` | 只保留交互项 |
| `include_bias` | `False` | 不包含常数项 |

3. `get_feature_names_out()`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回值 | `ndarray[str]` | 生成后的特征表达式 |

### 示例代码

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

X = np.array([[1, 2], [3, 4]])

poly2 = PolynomialFeatures(degree=2, include_bias=True)
X_poly2 = poly2.fit_transform(X)

poly_inter = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_inter = poly_inter.fit_transform(X)

print(poly2.get_feature_names_out())
print(X_poly2)
print(poly_inter.get_feature_names_out())
print(X_inter)
```

### 结果输出（示例）

```text
degree=2 特征: ['1' 'x0' 'x1' 'x0^2' 'x0 x1' 'x1^2']
----------------
interaction_only=True 特征: ['x0' 'x1' 'x0 x1']
```

### 理解重点

- 多项式特征是经典有效方法，但非常依赖正则化。
- 若模型已具备强非线性能力（如树模型），未必需要此扩展。
- 建议与 [模型选择](/foundations/sklearn/05-model-selection) 联动验证收益。

## 5. VarianceThreshold

### 方法重点

- 过滤低方差特征是最快速的特征筛选基线。
- 常量特征信息量几乎为零，应优先剔除。
- 该方法不依赖标签，属于无监督过滤。

### 参数速览（本节）

1. `VarianceThreshold(threshold=0)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `threshold` | `0` | 去除方差为 0 的特征 |

2. `fit_transform(X)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X` | 特征矩阵 | 输入样本 |
| 返回值 | `ndarray` | 过滤后的特征矩阵 |

3. `variances_`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回值 | 训练后自动生成 | 每列方差 |

### 示例代码

```python
import numpy as np
from sklearn.feature_selection import VarianceThreshold

X = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])

selector = VarianceThreshold(threshold=0)
X_filtered = selector.fit_transform(X)

print(selector.variances_)
print(X.shape, X_filtered.shape)
```

### 结果输出（示例）

```text
原始形状: (4, 3)
----------------
各特征方差: [0.25 0.25 0.  ]
----------------
过滤后形状: (4, 2)
```

### 理解重点

- 这是特征筛选的“第一刀”，通常放在流程最前面。
- 方差阈值不能反映与标签的相关性。
- 常与后续监督式选择方法结合使用。

## 6. SelectKBest

### 方法重点

- `SelectKBest` 根据统计检验分数保留前 K 个特征。
- 分类任务中常用 `f_classif`，回归任务可替换为 `f_regression`。
- 简单高效，适合特征数量较多时快速压缩维度。

### 参数速览（本节）

1. `SelectKBest(score_func=f_classif, k=2)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `score_func` | `f_classif` | 评分函数，基于方差分析 F 值 |
| `k` | `2` | 保留的特征数量 |

2. `fit_transform(X, y)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X` / `y` | 特征与标签 | 监督式特征筛选输入 |
| 返回值 | `ndarray` | 筛选后的特征矩阵 |

3. `scores_` 与 `get_support()`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回属性 `scores_` | 训练后自动生成 | 各特征评分 |
| 返回值（`get_support`） | `ndarray[bool]` | 被选中特征掩码 |

### 示例代码

```python
import numpy as np
from sklearn import datasets
from sklearn.feature_selection import SelectKBest, f_classif

iris = datasets.load_iris()
X, y = iris.data, iris.target

selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X, y)

print(selector.scores_.round(2))
print(np.array(iris.feature_names)[selector.get_support()])
```

### 结果输出（示例）

```text
原始特征数: 4
----------------
选择后特征数: 2
----------------
各特征得分: [119.26 49.16 1180.16 960.01]
----------------
选中的特征: ['petal length (cm)' 'petal width (cm)']
```

### 理解重点

- 统计分数高不代表对所有模型都最优。
- 建议把 `k` 作为超参数，用交叉验证选择。
- 若特征强相关，单变量检验可能忽略联合信息。

## 7. RFE 递归特征消除

### 方法重点

- RFE 反复训练模型并移除最弱特征，直到目标特征数。
- 依赖基学习器权重或重要性，因此模型选择很关键。
- 相比过滤法更精细，但计算成本更高。

### 参数速览（本节）

1. `RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=2, step=1)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `estimator` | `LogisticRegression(max_iter=1000)` | 用于评估特征重要性的基模型 |
| `n_features_to_select` | `2` | 最终保留特征数 |
| `step` | `1` | 每轮移除的特征数 |

2. `fit_transform(X, y)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X` / `y` | 特征与标签 | 输入数据 |
| 返回值 | `ndarray` | 递归筛选后的特征矩阵 |

3. `ranking_` 与 `support_`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回属性 `ranking_` | 训练后自动生成 | 特征排名，1 表示选中 |
| 返回属性 `support_` | 训练后自动生成 | 是否选中掩码 |

### 示例代码

```python
import numpy as np
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X, y = iris.data, iris.target

rfe = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=2, step=1)
X_rfe = rfe.fit_transform(X, y)

print(rfe.ranking_)
print(np.array(iris.feature_names)[rfe.support_])
```

### 结果输出（示例）

```text
特征排名: [3 2 1 1]
----------------
选中特征: ['petal length (cm)' 'petal width (cm)']
```

### 理解重点

- RFE 的结果依赖基模型，换模型可能得到不同子集。
- 维度很高时可先过滤再 RFE，降低计算开销。
- 对性能敏感任务可结合 [模型选择](/foundations/sklearn/05-model-selection) 验证稳定性。

## 8. SelectFromModel

### 方法重点

- 基于模型内置的重要性进行筛选，属于嵌入式方法。
- 与 RFE 相比，训练次数更少，通常更高效。
- 本例使用 `RandomForestClassifier`，与树模型特征重要性天然匹配。

### 参数速览（本节）

1. `RandomForestClassifier(n_estimators=100, random_state=42)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_estimators` | `100` | 随机森林树数量 |
| `random_state` | `42` | 控制可复现 |

2. `SelectFromModel(estimator, threshold='median')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `estimator` | `RandomForestClassifier(...)` | 提供特征重要性的基模型 |
| `threshold` | `'median'` | 重要性高于中位数的特征保留 |
| 返回值（`get_support`） | `ndarray[bool]` | 选中特征掩码 |

3. `estimator_.feature_importances_`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回属性 | 训练后自动生成 | 模型特征重要性 |

### 示例代码

```python
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

iris = datasets.load_iris()
X, y = iris.data, iris.target

rf = RandomForestClassifier(n_estimators=100, random_state=42)
sfm = SelectFromModel(rf, threshold="median")
X_sfm = sfm.fit_transform(X, y)

print(sfm.estimator_.feature_importances_.round(3))
print(sfm.threshold_)
print(np.array(iris.feature_names)[sfm.get_support()])
```

### 结果输出（示例）

```text
特征重要性: [0.102 0.023 0.436 0.439]
----------------
阈值: 0.269
----------------
选中特征: ['petal length (cm)' 'petal width (cm)']
```

### 理解重点

- 该方法对树模型与线性模型都适用，但重要性定义不同。
- 阈值可调，是控制维度与性能平衡的重要旋钮。
- 模型型特征选择与 [模型](/foundations/sklearn/07-models) 章节联系最紧密。

## 常见坑

1. 文本向量化后维度过大却不做限制，导致训练和推理成本骤增。
2. 在训练前先查看全量数据选择特征，造成数据泄露。
3. 把某一模型的特征选择结果直接迁移到完全不同模型而不复验。

## 小结

- 特征工程的本质是表达能力与泛化能力的平衡。
- 先用稳健基线方法构造与筛选，再通过交叉验证量化收益。
- 推荐将特征工程步骤放入流水线，以便和模型调参一体化管理。
