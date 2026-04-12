---
title: 模型
outline: deep
---

# 模型

> 对应脚本：`Basic/ScikitLearn/07_models.py`
> 运行方式：`python Basic/ScikitLearn/07_models.py`（仓库根目录）

## 本章目标

1. 建立 sklearn 常见模型族的整体认知与使用边界。
2. 掌握线性、树、集成、核方法等模型的核心参数。
3. 理解模型效果对数据缩放、特征分布的依赖关系。
4. 学会用统一方式对比不同模型表现。
5. 明确分类、聚类、降维模型在流程中的角色差异。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `LinearRegression()` / `Ridge()` / `Lasso()` / `ElasticNet()` | 线性回归与正则化 | `demo_linear_regression` |
| `LogisticRegression(...)` | 逻辑回归分类 | `demo_logistic_regression` |
| `DecisionTreeClassifier(...)` | 决策树分类 | `demo_tree_models` |
| `RandomForestClassifier(...)` / `GradientBoostingClassifier(...)` / `AdaBoostClassifier(...)` / `HistGradientBoostingClassifier(...)` | 集成分类模型 | `demo_ensemble_models` |
| `SVC(...)` / `LinearSVC(...)` | 支持向量机 | `demo_svm` |
| `GaussianNB()` | 朴素贝叶斯 | `demo_naive_bayes` |
| `KNeighborsClassifier(n_neighbors=5)` | K 近邻分类 | `demo_knn` |
| `KMeans(...)` / `DBSCAN(...)` | 聚类模型 | `demo_clustering` |
| `PCA(...)` / `TSNE(...)` | 降维模型 | `demo_dimensionality_reduction` |

## 1. 线性回归模型

### 方法重点

- 线性回归族可作为回归任务的强基线与可解释基线。
- Ridge 抑制系数震荡，Lasso 提供稀疏特征选择能力。
- ElasticNet 兼顾 L1 与 L2，适合特征相关性较强场景。

### 参数速览（本节）

1. `LinearRegression()`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回值（`score`） | `float` | 回归任务 R² |

2. `Ridge(alpha=1.0)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `alpha` | `1.0` | L2 正则化强度 |
| 返回值（`score`） | `float` | 回归任务 R² |

3. `Lasso(alpha=0.1)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `alpha` | `0.1` | L1 正则化强度 |
| 返回值（`score`） | `float` | 回归任务 R² |

4. `ElasticNet(alpha=0.1, l1_ratio=0.5)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `alpha` | `0.1` | 正则化强度 |
| `l1_ratio` | `0.5` | L1 在混合正则中的占比 |
| 返回值（`score`） | `float` | 回归任务 R² |

### 示例代码

```python
from sklearn import datasets
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
	diabetes.data, diabetes.target, test_size=0.3, random_state=42
)

models = {
	"LinearRegression": LinearRegression(),
	"Ridge": Ridge(alpha=1.0),
	"Lasso": Lasso(alpha=0.1),
	"ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
}

for name, model in models.items():
	model.fit(X_train, y_train)
	print(name, model.score(X_test, y_test))
```

### 结果输出（示例）

```text
LinearRegression: R² = 0.4773
----------------
Ridge (L2): R² = 0.4791
----------------
Lasso (L1): R² = 0.4770
----------------
ElasticNet: R² = 0.4432
```

### 理解重点

- 正则化不是必然提分，而是控制方差与可解释性的手段。
- 线性模型对特征尺度与共线性较敏感。
- 系数分布可作为特征重要性的初步参考。

## 2. 逻辑回归

### 方法重点

- 逻辑回归是分类任务最强基线之一，稳定、可解释、可校准。
- `class_weight='balanced'` 可缓解类别不平衡。
- 多分类默认使用 one-vs-rest 或 multinomial 方案。

### 参数速览（本节）

1. `LogisticRegression(max_iter=1000)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `max_iter` | `1000` | 最大迭代次数 |
| 返回值（`score`） | `float` | 分类准确率 |

2. `LogisticRegression(class_weight='balanced', max_iter=1000)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `class_weight` | `'balanced'` | 自动按类别频率加权 |
| `max_iter` | `1000` | 最大迭代次数 |
| 返回值（`score`） | `float` | 分类准确率 |

### 示例代码

```python
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
	iris.data, iris.target, test_size=0.3, random_state=42
)

lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
lr_bal = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X_train, y_train)

print(lr.score(X_test, y_test))
print(lr_bal.score(X_test, y_test))
```

### 结果输出（示例）

```text
基础: 准确率 = 1.0000
----------------
class_weight='balanced': 准确率 = 1.0000
```

### 理解重点

- 逻辑回归在中小规模任务上常作为上线首选模型。
- 类别不平衡下，建议配合 [指标](/foundations/sklearn/06-metrics) 章节的召回与 F1 联合评估。
- 若线性边界不足，再考虑核方法或树模型。

## 3. 决策树

### 方法重点

- 决策树可捕捉非线性与特征交互，且无需标准化。
- 易过拟合，通常需控制树深和叶子样本量。
- 可直接输出特征重要性与树结构深度。

### 参数速览（本节）

1. `DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=1, criterion='gini', random_state=42)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `max_depth` | `5` | 限制树深，抑制过拟合 |
| `min_samples_split` | `2` | 内部节点最小划分样本数 |
| `min_samples_leaf` | `1` | 叶子节点最小样本数 |
| `criterion` | `'gini'` | 划分纯度指标 |
| 返回属性 `feature_importances_` | 训练后自动生成 | 特征重要性 |

### 示例代码

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
	iris.data, iris.target, test_size=0.3, random_state=42
)

dt = DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=1, criterion="gini", random_state=42)
dt.fit(X_train, y_train)

print(dt.score(X_test, y_test))
print(dt.feature_importances_)
print(dt.get_depth())
```

### 结果输出（示例）

```text
准确率: 1.0000
----------------
特征重要性: [0.    0.    0.557 0.443]
----------------
树深度: 5
```

### 理解重点

- 决策树解释性强，但单树稳定性较弱。
- 小数据集表现常很好，大数据集更推荐集成方法。
- 重要性排序可反哺特征工程步骤。

## 4. 集成模型

### 方法重点

- 集成模型通过组合弱学习器获得更稳定的泛化能力。
- 随机森林偏并行 bagging，梯度提升偏串行 boosting。
- 同类任务中常可获得比单模型更鲁棒的结果。

### 参数速览（本节）

1. `RandomForestClassifier(n_estimators=100, random_state=42)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_estimators` | `100` | 基学习器数量 |
| `random_state` | `42` | 可复现 |
| 返回值（`score`） | `float` | 分类准确率 |

2. `GradientBoostingClassifier(n_estimators=100, random_state=42)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_estimators` | `100` | 基学习器数量 |
| `random_state` | `42` | 可复现 |
| 返回值（`score`） | `float` | 分类准确率 |

3. `AdaBoostClassifier(n_estimators=50, random_state=42)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_estimators` | `50` | 基学习器数量 |
| `random_state` | `42` | 可复现 |
| 返回值（`score`） | `float` | 分类准确率 |

4. `HistGradientBoostingClassifier(random_state=42)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `random_state` | `42` | 可复现 |
| 返回值（`score`） | `float` | 分类准确率 |

### 示例代码

```python
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
	iris.data, iris.target, test_size=0.3, random_state=42
)

models = {
	"RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
	"GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
	"AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
	"HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
}

for name, model in models.items():
	model.fit(X_train, y_train)
	print(name, model.score(X_test, y_test))
```

### 结果输出（示例）

```text
RandomForest: 1.0000
----------------
GradientBoosting: 1.0000
----------------
AdaBoost: 0.9778
----------------
HistGradientBoosting: 1.0000
```

### 理解重点

- 集成模型通常性能更优，但训练和解释成本更高。
- 参数调优建议结合 [模型选择](/foundations/sklearn/05-model-selection) 系统进行。
- 小样本下 boosting 更易过拟合，需关注验证曲线。

## 5. SVM

### 方法重点

- SVM 对特征尺度敏感，通常必须先标准化。
- 核函数选择决定决策边界形状与复杂度。
- 线性不可分问题可用 RBF 或多项式核处理。

### 参数速览（本节）

1. `SVC(C=1.0, kernel='rbf')`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `C` | `1.0` | 误分类惩罚系数 |
| `kernel` | `'rbf'`、`'linear'`、`'poly'` | 核函数类型 |
| 返回值（`score`） | `float` | 分类准确率 |

2. `LinearSVC(max_iter=10000)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `max_iter` | `10000` | 最大迭代次数 |
| 返回值（`score`） | `float` | 分类准确率 |

3. `make_pipeline(StandardScaler(), model)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `StandardScaler()` | 标准化步骤 | SVM 前置缩放 |
| `model` | `SVC` 或 `LinearSVC` | 分类器主体 |
| 返回值 | `Pipeline` | 可直接 `fit/score` 的流水线 |

### 示例代码

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
	iris.data, iris.target, test_size=0.3, random_state=42
)

svc = make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf"))
lsvc = make_pipeline(StandardScaler(), LinearSVC(max_iter=10000))

svc.fit(X_train, y_train)
lsvc.fit(X_train, y_train)
print(svc.score(X_test, y_test))
print(lsvc.score(X_test, y_test))
```

### 结果输出（示例）

```text
SVC (rbf): 1.0000
----------------
LinearSVC: 0.9778
```

### 理解重点

- 先做标准化再训练 SVM 几乎是默认最佳实践。
- RBF 核在非线性任务中常见，但需调节 `C` 与 `gamma`。
- 边界更复杂不一定更好，需用验证曲线判断。

## 6. 朴素贝叶斯

### 方法重点

- `GaussianNB` 训练速度快、参数少，是高效分类基线。
- 假设特征条件独立，在现实中常不严格成立。
- 对小数据集和快速原型非常友好。

### 参数速览（本节）

1. `GaussianNB()`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| 返回值（`score`） | `float` | 分类准确率 |

### 示例代码

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
	iris.data, iris.target, test_size=0.3, random_state=42
)

gnb = GaussianNB().fit(X_train, y_train)
print(gnb.score(X_test, y_test))
```

### 结果输出（示例）

```text
GaussianNB: 0.9778
```

### 理解重点

- 朴素贝叶斯常用于“先跑通流程”的第一版模型。
- 若准确率不够，可逐步切换到更复杂模型。
- 模型虽简单，但在文本分类等场景仍常有竞争力。

## 7. K 近邻

### 方法重点

- KNN 基于邻域投票，直观但推理成本随样本数上升。
- 对特征尺度敏感，应先标准化。
- `k` 的选择影响偏差-方差平衡。

### 参数速览（本节）

1. `KNeighborsClassifier(n_neighbors=5)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_neighbors` | `5` | 近邻个数 |
| 返回值（`score`） | `float` | 分类准确率 |

2. `make_pipeline(StandardScaler(), KNeighborsClassifier(...))`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `StandardScaler()` | 标准化步骤 | 距离计算前统一量纲 |
| `KNeighborsClassifier(...)` | KNN 模型 | 负责分类预测 |
| 返回值 | `Pipeline` | 可直接 `fit/score` 的流水线 |

### 示例代码

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
	iris.data, iris.target, test_size=0.3, random_state=42
)

knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
```

### 结果输出（示例）

```text
KNN (k=5): 1.0000
```

### 理解重点

- KNN 是距离模型，标准化优先级高。
- 大规模数据上推理慢，常需近似检索或改用其他模型。
- `k` 可通过验证曲线快速定位合理范围。

## 8. 聚类模型

### 方法重点

- KMeans 需要预设簇数，DBSCAN 自动识别簇并标记噪声。
- 不同算法适配不同数据分布和噪声水平。
- 聚类评价常使用轮廓系数等无监督指标。

### 参数速览（本节）

1. `KMeans(n_clusters=4, random_state=42, n_init=10)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_clusters` | `4` | 聚类数 |
| `n_init` | `10` | 随机初始化次数 |
| `random_state` | `42` | 可复现 |

2. `DBSCAN(eps=0.5, min_samples=5)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `eps` | `0.5` | 邻域半径 |
| `min_samples` | `5` | 核心点最小样本数 |

3. `silhouette_score(X, labels)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `X` / `labels` | 样本与聚类标签 | 评估输入 |
| 返回值 | `float` | 轮廓系数 |

### 示例代码

```python
from sklearn import datasets
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

X, y_true = datasets.make_blobs(n_samples=300, centers=4, random_state=42)

labels_km = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X)
labels_db = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)

print(silhouette_score(X, labels_km))
print(len(set(labels_db)) - (1 if -1 in labels_db else 0))
```

### 结果输出（示例）

```text
KMeans 轮廓系数: 0.7916
----------------
DBSCAN 聚类数: 4
```

### 理解重点

- KMeans 对球形簇更友好，DBSCAN 对噪声更稳健。
- DBSCAN 的 `eps` 和 `min_samples` 对结果非常敏感。
- 聚类结果应结合业务可解释性检验，不只看内部指标。

## 9. 降维模型

### 方法重点

- PCA 是线性降维，强调最大方差方向。
- t-SNE 是非线性嵌入，更偏可视化而非特征工程。
- 降维后应检查类别可分性和信息保留程度。

### 参数速览（本节）

1. `PCA(n_components=2)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_components` | `2` | 目标降维维度 |
| 返回属性 `explained_variance_ratio_` | PCA 训练后生成 | 各主成分解释方差比 |
| 返回值（`fit_transform`） | `ndarray` | 降维后的样本表示 |

2. `TSNE(n_components=2, random_state=42)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `n_components` | `2` | 目标嵌入维度 |
| `random_state` | `42` | 可复现控制 |
| 返回值（`fit_transform`） | `ndarray` | 降维后的样本表示 |

### 示例代码

```python
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

iris = datasets.load_iris()
X, y = iris.data, iris.target

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

print(pca.explained_variance_ratio_)
print(X_tsne.shape)
```

### 结果输出（示例）

```text
PCA 解释方差比: [0.9246 0.0531]
----------------
PCA 累计解释方差: 0.9777
----------------
t-SNE 输出形状: (150, 2)
```

### 理解重点

- PCA 可用于降噪和压缩，t-SNE 更适合可视化探索。
- t-SNE 的空间距离不宜直接做定量解释。
- 降维后建模时应验证性能是否受损。

## 常见坑

1. 忘记对 SVM、KNN 做标准化，导致性能异常波动。
2. 只比较准确率，不结合训练成本与可解释性。
3. 将 t-SNE 结果直接作为下游生产特征而不做稳定性验证。

## 小结

- 模型选择不应只看分数，还要考虑成本、稳定性与解释性。
- 推荐先建立线性和树模型基线，再逐步引入复杂模型。
- 评估细节可配合 [指标](/foundations/sklearn/06-metrics) 与 [模型选择](/foundations/sklearn/05-model-selection) 章节联动。
