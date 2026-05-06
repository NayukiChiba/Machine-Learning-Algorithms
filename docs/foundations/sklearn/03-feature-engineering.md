---
title: sklearn 特征工程
outline: deep
---

# sklearn 特征工程

## 本章目标

1. 掌握文本、字典、数值三类常见特征构造方式
2. 理解 `TfidfVectorizer`、`PolynomialFeatures` 的参数影响
3. 学会使用过滤法、包裹法、模型法进行特征选择
4. 明确高维特征场景下的稀疏表示与维度控制策略

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `CountVectorizer()` | 构造器 | 将文本转为词频特征（词袋模型） |
| `TfidfVectorizer(...)` | 构造器 | 将文本转为 TF-IDF 特征 |
| `DictVectorizer(sparse=False)` | 构造器 | 将字典样本展开为特征矩阵 |
| `PolynomialFeatures(degree)` | 构造器 | 生成多项式与交互项特征 |
| `VarianceThreshold(threshold)` | 构造器 | 过滤低方差特征（无监督） |
| `SelectKBest(score_func, k)` | 构造器 | 基于统计检验筛选特征（过滤法） |
| `RFE(estimator, n_features_to_select)` | 构造器 | 递归特征消除（包裹法） |
| `SelectFromModel(estimator)` | 构造器 | 基于模型重要性筛选（嵌入式） |

## 1. 文本特征提取

### `CountVectorizer` / `TfidfVectorizer`

#### 作用

`CountVectorizer` 将文本转为词频特征（词袋模型），返回稀疏矩阵。`TfidfVectorizer` 在词频基础上引入逆文档频率——削弱高频泛化词影响。`max_df`、`min_df`、`ngram_range` 控制特征空间大小。

#### 重点方法

```python
CountVectorizer(*, lowercase=True, ngram_range=(1, 1), stop_words=None)
TfidfVectorizer(*, max_features=None, ngram_range=(1, 1), stop_words=None,
                max_df=1.0, min_df=1)
# fit(corpus) → transform(corpus) → get_feature_names_out()
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `max_features` | `int` 或 `None` | 限制最大词特征数，默认为 `None` | `1000` |
| `ngram_range` | `tuple[int, int]` | n-gram 范围，默认为 `(1, 1)` | `(1, 2)` |
| `stop_words` | `str`、`list` 或 `None` | 停用词过滤，`'english'` 使用英文停用词表 | `'english'` |
| `max_df` | `float` 或 `int` | TfidfVectorizer：忽略出现频率过高的词，默认为 `1.0` | `0.9` |
| `min_df` | `float` 或 `int` | TfidfVectorizer：忽略出现频率过低的词，默认为 `1` | `2` |
| `lowercase` | `bool` | 是否统一小写化，默认为 `True` | `True` |

#### 示例代码

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
]

cv = CountVectorizer()
Xc = cv.fit_transform(corpus)
print(f"CountVectorizer 词汇表: {cv.get_feature_names_out()}")
print(f"词频矩阵:\n{Xc.toarray()}")

tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
Xt = tfidf.fit_transform(corpus)
print(f"\nTfidfVectorizer (stop_words+bigram): {Xt.shape}")
print(f"词汇表: {tfidf.get_feature_names_out()}")
```

#### 输出

```text
CountVectorizer 词汇表: ['and' 'document' 'first' 'is' 'one' 'second' 'the' 'third' 'this']
词频矩阵:
[[0 1 1 1 0 0 1 0 1]
 [0 2 0 1 0 1 1 0 1]
 [1 0 0 1 1 0 1 1 1]]

TfidfVectorizer (stop_words+bigram): (3, 8)
词汇表: ['document' 'document second' 'second' 'third' 'third one' ...]
```

#### 理解重点

- Count 特征易解释——但无法表达词义接近关系，对文本分类基线非常有效
- TF-IDF 更关注区分度高的词——不等于语义建模
- 中文文本需自定义分词与停用词策略
- 返回稀疏矩阵（`csr_matrix`）——适合高维文本特征

## 2. 字典特征展开

### `DictVectorizer`

#### 作用

将结构化字典输入自动展开为特征矩阵——对类别键做独热展开，对数值键保留原值。通过 `inverse_transform` 可回看特征与原字典的映射关系。

#### 重点方法

```python
DictVectorizer(*, sparse=True)
# fit(data) → transform(data) / fit_transform(data)
# get_feature_names_out() / inverse_transform(X)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `sparse` | `bool` | `True` 返回稀疏矩阵 / `False` 返回稠密数组，默认为 `True` | `False` |

#### 示例代码

```python
from sklearn.feature_extraction import DictVectorizer

data = [
    {"city": "北京", "temperature": 20},
    {"city": "上海", "temperature": 25},
    {"city": "北京", "temperature": 18},
]

dv = DictVectorizer(sparse=False)
X = dv.fit_transform(data)

print(f"特征名: {dv.get_feature_names_out()}")
print(f"特征矩阵:\n{X}")
print(f"逆变换: {dv.inverse_transform(X)[0]}")
```

#### 输出

```text
特征名: ['city=上海' 'city=北京' 'temperature']
特征矩阵:
[[0. 1. 20.]
 [1. 0. 25.]
 [0. 1. 18.]]
逆变换: {'city=北京': 1.0, 'temperature': 20.0}
```

#### 理解重点

- DictVectorizer 常用于日志特征、规则特征、浅层推荐特征工程
- 产物特征名可追踪——利于可解释性
- 类别字段较多时建议结合频次阈值做后续裁剪

## 3. 多项式特征扩展

### `PolynomialFeatures`

#### 作用

生成多项式与交互项特征——可让线性模型拟合非线性关系。维度增长很快（$O(d^{degree})$），`interaction_only=True` 只保留交互项降低膨胀。

#### 重点方法

```python
PolynomialFeatures(degree=2, *, interaction_only=False, include_bias=True)
# fit(X) → transform(X) / get_feature_names_out()
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `degree` | `int` | 最高多项式阶数，默认为 `2` | `3` |
| `interaction_only` | `bool` | `True` 只保留交互项（不含平方项），默认为 `False` | `True` |
| `include_bias` | `bool` | `True` 包含常数项 1，默认为 `True` | `False` |

#### 示例代码

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

X = np.array([[1, 2], [3, 4]])

poly2 = PolynomialFeatures(degree=2, include_bias=True)
print(f"degree=2 特征名: {poly2.fit_transform(X).shape}")

polyInter = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
print(f"interaction_only 特征名: {polyInter.fit(X).get_feature_names_out()}")
```

#### 输出

```text
degree=2 特征: ['1' 'x0' 'x1' 'x0^2' 'x0 x1' 'x1^2'] → 6 维
interaction_only 特征: ['x0' 'x1' 'x0 x1'] → 3 维
```

#### 理解重点

- 多项式特征是经典有效方法——但非常依赖正则化
- 若模型已具备强非线性能力（树模型）——未必需要此扩展
- `degree` 不宜盲目增大——维度呈组合爆炸增长

## 4. 过滤法特征选择

### `VarianceThreshold` / `SelectKBest`

#### 作用

**过滤法**在模型训练前独立筛选特征。`VarianceThreshold` 去掉方差为 0 的常量特征（无监督）。`SelectKBest` 根据统计检验分数保留前 K 个特征（监督式，分类用 `f_classif`，回归用 `f_regression`）。

#### 重点方法

```python
VarianceThreshold(threshold=0.0)
SelectKBest(score_func=f_classif, *, k=10)
# fit(X[, y]) → transform(X) / get_support()
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `threshold` | `float` | VarianceThreshold：方差阈值，低于此值的特征被移除 | `0.1` |
| `score_func` | `callable` | SelectKBest：评分函数，`f_classif`（分类）或 `f_regression`（回归） | `f_classif` |
| `k` | `int` | SelectKBest：保留的特征数量 | `2` |

训练后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `variances_` | `ndarray` | VarianceThreshold：每列方差 |
| `scores_` | `ndarray` | SelectKBest：各特征评分 |
| `get_support()` | `ndarray[bool]` | 被选中特征的布尔掩码 |

#### 示例代码

```python
import numpy as np
from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

# VarianceThreshold：去除常量特征
X = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
vt = VarianceThreshold(threshold=0)
Xvt = vt.fit_transform(X)
print(f"方差过滤: {X.shape} → {Xvt.shape}, 方差={vt.variances_}")

# SelectKBest：保留得分最高的 K 个特征
iris = datasets.load_iris()
skb = SelectKBest(score_func=f_classif, k=2)
Xskb = skb.fit_transform(iris.data, iris.target)
print(f"SelectKBest: {iris.data.shape} → {Xskb.shape}")
print(f"得分: {skb.scores_.round(1)}")
print(f"选中特征: {np.array(iris.feature_names)[skb.get_support()]}")
```

#### 输出

```text
方差过滤: (4, 3) → (4, 2), 方差=[0.25 0.25 0.  ]
SelectKBest: (150, 4) → (150, 2)
得分: [ 119.3   49.2 1180.2  960. ]
选中特征: ['petal length (cm)' 'petal width (cm)']
```

#### 理解重点

- 方差过滤是特征筛选的"第一刀"——放在流程最前面
- 单变量统计分数高不代表对所有模型最优——忽略特征联合效应
- 建议把 `k` 作为超参数用交叉验证选择

## 5. 包裹法与嵌入式特征选择

### `RFE` / `SelectFromModel`

#### 作用

**RFE**（包裹法）反复训练模型并移除最弱特征——依赖基学习器的权重或重要性。**SelectFromModel**（嵌入式）基于模型内置重要性一次性筛选——比 RFE 更高效。两者都适用于分类和回归。

#### 重点方法

```python
RFE(estimator, *, n_features_to_select=None, step=1)
SelectFromModel(estimator, *, threshold=None)
# fit(X, y) → transform(X) → get_support()
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `estimator` | 提供特征权重的基模型 | `LogisticRegression(max_iter=1000)` |
| `n_features_to_select` | `int` 或 `None` | RFE：最终保留特征数；`None` 保留一半 | `2` |
| `step` | `int` 或 `float` | RFE：每轮移除的特征数或比例，默认为 `1` | `1` |
| `threshold` | `str` 或 `float` | SelectFromModel：`'median'` / `'mean'` 或数值阈值 | `'median'` |

训练后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `ranking_` | `ndarray` | RFE：特征排名，1 表示最终选中 |
| `support_` | `ndarray[bool]` | 选中掩码 |
| `estimator_.feature_importances_` | `ndarray` | SelectFromModel：基模型的特征重要性 |

#### 示例代码

```python
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel

iris = datasets.load_iris()
X, y = iris.data, iris.target

# RFE
rfe = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=2)
rfe.fit(X, y)
print(f"RFE ranking: {rfe.ranking_}, 选中: {np.array(iris.feature_names)[rfe.support_]}")

# SelectFromModel
rf = RandomForestClassifier(n_estimators=100, random_state=42)
sfm = SelectFromModel(rf, threshold="median")
sfm.fit(X, y)
print(f"SelectFromModel 重要性: {sfm.estimator_.feature_importances_.round(3)}")
print(f"选中: {np.array(iris.feature_names)[sfm.get_support()]}")
```

#### 输出

```text
RFE ranking: [3 2 1 1], 选中: ['petal length (cm)' 'petal width (cm)']
SelectFromModel 重要性: [0.102 0.023 0.436 0.439], 选中: ['petal length (cm)' 'petal width (cm)']
```

#### 理解重点

- RFE 结果依赖基模型——换模型可能得到不同子集
- `SelectFromModel` 对树模型与线性模型都适用——但重要性定义不同
- 维度很高时可先过滤再 RFE/SFM——降低计算开销
- 阈值是可调旋钮——控制维度与性能的平衡

## 常见坑

1. 文本向量化后维度过大却不限制 `max_features`——导致训练和推理成本骤增
2. 在训练前先查看全量数据选择特征——造成数据泄露
3. 把某一模型的特征选择结果直接迁移到完全不同模型而不复验

## 小结

- 特征工程的本质是表达能力与泛化能力的平衡
- 先用稳健基线方法构造与筛选——再通过交叉验证量化收益
- 推荐将特征工程步骤放入 Pipeline——以便和模型调参一体化管理
- 文本用 `CountVectorizer`/`TfidfVectorizer`，类别用 `DictVectorizer`/`OneHotEncoder`，数值用 `PolynomialFeatures`
