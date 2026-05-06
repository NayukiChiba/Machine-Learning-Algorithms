---
title: sklearn 常用模型
outline: deep
---

# sklearn 常用模型

## 本章目标

1. 建立 sklearn 常见模型族的整体认知与使用边界
2. 掌握线性、树、集成、核方法等模型的核心参数
3. 理解模型效果对数据缩放、特征分布的依赖关系
4. 学会用统一方式对比不同模型表现
5. 明确分类、聚类、降维模型在流程中的角色差异

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `LinearRegression()` | 构造器 | 普通最小二乘线性回归 |
| `Ridge(alpha)` / `Lasso(alpha)` / `ElasticNet(alpha)` | 构造器 | 带正则化的线性回归 |
| `LogisticRegression()` | 构造器 | 逻辑回归分类 |
| `DecisionTreeClassifier()` | 构造器 | 决策树分类 |
| `RandomForestClassifier()` | 构造器 | 随机森林集成分类 |
| `GradientBoostingClassifier()` | 构造器 | 梯度提升集成分类 |
| `SVC()` / `LinearSVC()` | 构造器 | 支持向量机 |
| `GaussianNB()` | 构造器 | 高斯朴素贝叶斯 |
| `KNeighborsClassifier(n_neighbors)` | 构造器 | K 近邻分类 |
| `KMeans(n_clusters)` / `DBSCAN()` | 构造器 | 聚类模型 |
| `PCA(n_components)` / `TSNE()` | 构造器 | 降维模型 |

## 1. 线性回归模型

### `LinearRegression` / `Ridge` / `Lasso` / `ElasticNet`

#### 作用

线性回归族可作为回归任务的强基线与可解释基线。Ridge 通过 L2 惩罚抑制系数震荡，Lasso 通过 L1 惩罚提供稀疏特征选择能力。ElasticNet 兼顾 L1 与 L2，适合特征相关性较强场景。

#### 重点方法

```python
LinearRegression(*, fit_intercept=True, copy_X=True, n_jobs=None, positive=False)
Ridge(alpha=1.0, *, fit_intercept=True, solver='auto')
Lasso(alpha=1.0, *, fit_intercept=True, max_iter=1000)
ElasticNet(alpha=1.0, *, l1_ratio=0.5, fit_intercept=True, max_iter=1000)
# 核心方法：fit(X, y) → predict(X) → score(X, y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `fit_intercept` | `bool` | 是否拟合截距项，默认为 `True` | `True` |
| `alpha` | `float` | 正则化强度，默认为 `1.0` | `0.1` |
| `l1_ratio` | `float` | ElasticNet：L1 在混合正则中的占比，`0` = Ridge，`1` = Lasso，默认为 `0.5` | `0.7` |
| `max_iter` | `int` | Lasso/ElasticNet：最大迭代次数，默认为 `1000` | `5000` |
| `solver` | `str` | Ridge：求解器，`"auto"` / `"svd"` / `"cholesky"` 等 | `"auto"` |

训练后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `coef_` | `ndarray` | 各特征系数 |
| `intercept_` | `float` | 截距项 |

#### 示例代码

```python
from sklearn import datasets
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split

X, y = datasets.load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge (L2)": Ridge(alpha=1.0),
    "Lasso (L1)": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name}: R² = {model.score(X_test, y_test):.4f}")
```

#### 输出

```text
LinearRegression: R² = 0.4773
Ridge (L2): R² = 0.4791
Lasso (L1): R² = 0.4770
ElasticNet: R² = 0.4432
```

#### 理解重点

- 正则化不是必然提分，而是控制方差与可解释性的手段
- 线性模型对特征尺度与共线性较敏感——建议先标准化
- Lasso 可将部分系数压到 0——天然具备特征选择功能
- 系数分布可作为特征重要性的初步参考

## 2. 逻辑回归

### `LogisticRegression`

#### 作用

逻辑回归是分类任务最强基线之一，稳定、可解释、可校准。`class_weight='balanced'` 可缓解类别不平衡。多分类默认使用 one-vs-rest 或 multinomial 方案。

#### 重点方法

```python
LogisticRegression(penalty='l2', *, C=1.0, fit_intercept=True, max_iter=100,
                   multi_class='auto', class_weight=None, solver='lbfgs')
# fit(X, y) → predict(X) → predict_proba(X) → score(X, y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `penalty` | `str` | 正则类型，`"l2"` / `"l1"` / `"elasticnet"` / `None`，默认为 `"l2"` | `"l2"` |
| `C` | `float` | 正则强度的倒数（越小正则越强），默认为 `1.0` | `0.5` |
| `max_iter` | `int` | 最大迭代次数，默认为 `100` | `1000` |
| `class_weight` | `str` 或 `dict` | `"balanced"` 自动按类别频率加权 | `"balanced"` |
| `multi_class` | `str` | 多分类策略：`"auto"` / `"ovr"` / `"multinomial"` | `"multinomial"` |
| `solver` | `str` | 优化器：`"lbfgs"` / `"liblinear"` / `"saga"` 等 | `"lbfgs"` |

#### 示例代码

```python
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
lrBal = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X_train, y_train)

print(f"基础: 准确率 = {lr.score(X_test, y_test):.4f}")
print(f"balanced: 准确率 = {lrBal.score(X_test, y_test):.4f}")
print(f"预测概率 (前 3):\n{lr.predict_proba(X_test[:3])}")
```

#### 输出

```text
基础: 准确率 = 1.0000
balanced: 准确率 = 1.0000
预测概率 (前 3):
[[2.72513336e-04 2.51077806e-01 7.48649681e-01]
 [7.80283689e-01 2.18481740e-01 1.23457120e-03]
 [8.10557788e-01 1.88417402e-01 1.02481011e-03]]
```

#### 理解重点

- 逻辑回归在中小规模任务上常作为上线首选模型
- 类别不平衡下，建议配合 F1 与 Recall 联合评估
- 若线性边界不足，再考虑核方法或树模型
- `predict_proba` 输出的概率可配合校准方法进一步精调

## 3. 决策树

### `DecisionTreeClassifier`

#### 作用

决策树可捕捉非线性与特征交互，且无需标准化。易过拟合，通常需控制树深和叶子样本量。可直接输出特征重要性与树结构深度。

#### 重点方法

```python
DecisionTreeClassifier(*, criterion='gini', splitter='best', max_depth=None,
                       min_samples_split=2, min_samples_leaf=1,
                       max_features=None, random_state=None)
# fit(X, y) → predict(X) → predict_proba(X) → score(X, y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `criterion` | `str` | 划分纯度指标：`"gini"` / `"entropy"`，默认为 `"gini"` | `"gini"` |
| `max_depth` | `int` 或 `None` | 树深度上限，`None` 不限制，默认为 `None` | `5` |
| `min_samples_split` | `int` 或 `float` | 内部节点最小划分样本数，默认为 `2` | `10` |
| `min_samples_leaf` | `int` 或 `float` | 叶子节点最小样本数，默认为 `1` | `5` |
| `max_features` | `int`、`str` 或 `None` | 每次划分考虑的候选特征数，默认为 `None`（全用） | `"sqrt"` |
| `random_state` | `int` | 随机种子 | `42` |

训练后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `feature_importances_` | `ndarray` | 特征重要性（Gini importance） |
| `n_features_in_` | `int` | 训练特征数 |
| `get_depth()` | `int` | 树实际深度 |

#### 示例代码

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

dt = DecisionTreeClassifier(max_depth=5, min_samples_split=2,
                            min_samples_leaf=1, criterion="gini", random_state=42)
dt.fit(X_train, y_train)

print(f"准确率: {dt.score(X_test, y_test):.4f}")
print(f"特征重要性: {dt.feature_importances_}")
print(f"树深度: {dt.get_depth()}")
```

#### 输出

```text
准确率: 1.0000
特征重要性: [0.         0.         0.56624158 0.43375842]
树深度: 4
```

#### 理解重点

- 决策树解释性强，但单树稳定性较弱
- 小数据集表现常很好，大数据集更推荐集成方法
- 重要性排序可反哺特征工程步骤
- `min_samples_leaf` 是最有效的过拟合控制参数之一

## 4. 集成模型

### `RandomForestClassifier` / `GradientBoostingClassifier` / `AdaBoostClassifier` / `HistGradientBoostingClassifier`

#### 作用

集成模型通过组合弱学习器获得更稳定的泛化能力。随机森林偏并行 bagging，梯度提升偏串行 boosting。同类任务中常可获得比单模型更鲁棒的结果。

#### 重点方法

```python
RandomForestClassifier(n_estimators=100, *, criterion='gini', max_depth=None,
                       min_samples_split=2, min_samples_leaf=1,
                       max_features='sqrt', bootstrap=True, random_state=None)
GradientBoostingClassifier(*, loss='log_loss', learning_rate=0.1,
                           n_estimators=100, max_depth=3, random_state=None)
AdaBoostClassifier(estimator=None, *, n_estimators=50, learning_rate=1.0,
                   random_state=None)
HistGradientBoostingClassifier(loss='log_loss', *, learning_rate=0.1,
                               max_iter=100, max_depth=None, random_state=None)
# 核心方法：fit(X, y) → predict(X) → predict_proba(X) → score(X, y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_estimators` | `int` | 基学习器数量（Hist 用 `max_iter`），默认为 `100` | `100` |
| `learning_rate` | `float` | Boosting 学习率，控制每棵树贡献，默认为 `0.1` | `0.05` |
| `max_depth` | `int` 或 `None` | 树深度上限，`None` 不限制，默认为 `None`（RF）/ `3`（GB） | `5` |
| `max_features` | `str` | 每次划分候选特征数：`"sqrt"` / `"log2"` / `None` | `"sqrt"` |
| `bootstrap` | `bool` | RF：是否自助采样，默认为 `True` | `True` |
| `random_state` | `int` | 随机种子 | `42` |

训练后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `feature_importances_` | `ndarray` | 特征重要性 |
| `estimators_` | `list` | 所有基学习器列表 |

#### 示例代码

```python
from sklearn import datasets
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, HistGradientBoostingClassifier)
from sklearn.model_selection import train_test_split

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name}: {model.score(X_test, y_test):.4f}")
```

#### 输出

```text
RandomForest: 1.0000
GradientBoosting: 1.0000
AdaBoost: 0.9778
HistGradientBoosting: 0.9778
```

#### 理解重点

- 集成模型通常性能更优，但训练和解释成本更高
- 小样本下 boosting 更易过拟合，需关注验证曲线
- 随机森林是大多数分类任务的强基线——参数不敏感
- `HistGradientBoostingClassifier` 比传统 GBDT 快一个数量级，支持缺失值

## 5. SVM

### `SVC` / `LinearSVC`

#### 作用

SVM 对特征尺度敏感，通常必须先标准化。核函数选择决定决策边界形状与复杂度。线性不可分问题可用 RBF 或多项式核处理。

#### 重点方法

```python
SVC(*, C=1.0, kernel='rbf', gamma='scale', degree=3, probability=False,
    class_weight=None, random_state=None)
LinearSVC(penalty='l2', loss='squared_hinge', *, C=1.0, max_iter=1000,
          random_state=None)
# fit(X, y) → predict(X) → score(X, y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `C` | `float` | 误分类惩罚系数，越小正则越强，默认为 `1.0` | `10` |
| `kernel` | `str` | 核函数：`"linear"` / `"rbf"` / `"poly"` / `"sigmoid"`，默认为 `"rbf"` | `"rbf"` |
| `gamma` | `str` 或 `float` | 核系数，`"scale"` = 1/(n_features*Var)，`"auto"` = 1/n_features | `"scale"` |
| `degree` | `int` | 多项式核阶数，默认为 `3` | `3` |
| `probability` | `bool` | 是否输出概率（需额外训练），默认为 `False` | `True` |
| `max_iter` | `int` | LinearSVC 最大迭代次数，默认为 `1000` | `10000` |

#### 示例代码

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

svc = make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf"))
lsvc = make_pipeline(StandardScaler(), LinearSVC(max_iter=10000))

svc.fit(X_train, y_train)
lsvc.fit(X_train, y_train)
print(f"SVC (rbf): {svc.score(X_test, y_test):.4f}")
print(f"LinearSVC: {lsvc.score(X_test, y_test):.4f}")
```

#### 输出

```text
SVC (rbf): 0.9778
LinearSVC: 0.9778
```

#### 理解重点

- 先做标准化再训练 SVM 几乎是默认最佳实践
- RBF 核在非线性任务中常见，但需调节 `C` 与 `gamma`
- 边界更复杂不一定更好，需用验证曲线判断
- `LinearSVC` 比 `SVC(kernel='linear')` 更快，但不支持概率输出

## 6. 朴素贝叶斯

### `GaussianNB`

#### 作用

`GaussianNB` 训练速度快、参数少，是高效分类基线。假设特征条件独立，在现实中常不严格成立。对小数据集和快速原型非常友好。

#### 重点方法

```python
GaussianNB(*, priors=None, var_smoothing=1e-09)
# fit(X, y) → predict(X) → predict_proba(X) → score(X, y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `priors` | `array_like` 或 `None` | 类别先验概率，`None` 从数据估计 | `[0.3, 0.7]` |
| `var_smoothing` | `float` | 方差平滑项，防止除零，默认为 `1e-09` | `1e-08` |

#### 示例代码

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

gnb = GaussianNB().fit(X_train, y_train)
print(f"GaussianNB: {gnb.score(X_test, y_test):.4f}")
```

#### 输出

```text
GaussianNB: 0.9778
```

#### 理解重点

- 朴素贝叶斯常用于"先跑通流程"的第一版模型
- 若准确率不够，可逐步切换到更复杂模型
- 模型虽简单，但在文本分类等场景仍常有竞争力
- 特征条件独立假设在连续数据上常不成立——但当数据量小时偏差可接受

## 7. K 近邻

### `KNeighborsClassifier`

#### 作用

KNN 基于邻域投票，直观但推理成本随样本数上升。对特征尺度敏感，应先标准化。`k` 的选择影响偏差-方差平衡。

#### 重点方法

```python
KNeighborsClassifier(n_neighbors=5, *, weights='uniform', algorithm='auto',
                     leaf_size=30, p=2, metric='minkowski', n_jobs=None)
# fit(X, y) → predict(X) → predict_proba(X) → score(X, y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_neighbors` | `int` | 近邻个数 k，默认为 `5` | `3` |
| `weights` | `str` | `"uniform"` 等权投票 / `"distance"` 距离加权，默认为 `"uniform"` | `"distance"` |
| `algorithm` | `str` | 近邻搜索算法：`"auto"` / `"ball_tree"` / `"kd_tree"` / `"brute"` | `"auto"` |
| `p` | `int` | Minkowski 距离的 p 值，`2` = 欧氏距离，默认为 `2` | `2` |
| `metric` | `str` | 距离度量方式，默认为 `"minkowski"` | `"euclidean"` |

#### 示例代码

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
knn.fit(X_train, y_train)
print(f"KNN (k=5): {knn.score(X_test, y_test):.4f}")
```

#### 输出

```text
KNN (k=5): 0.9778
```

#### 理解重点

- KNN 是距离模型，标准化优先级高
- 大规模数据上推理慢，常需近似检索或改用其他模型
- `k` 可通过验证曲线快速定位合理范围
- `weights='distance'` 可让近邻贡献更大——对小样本更友好

## 8. 聚类模型

### `KMeans` / `DBSCAN`

#### 作用

KMeans 需要预设簇数，最小化簇内平方和。DBSCAN 基于密度自动识别簇并标记噪声点。不同算法适配不同数据分布和噪声水平。

#### 重点方法

```python
KMeans(n_clusters=8, *, init='k-means++', n_init='auto', max_iter=300,
       random_state=None, algorithm='lloyd')
DBSCAN(eps=0.5, *, min_samples=5, metric='euclidean', algorithm='auto',
       n_jobs=None)
# fit(X) → predict(X) / fit_predict(X)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_clusters` | `int` | KMeans：聚类数，默认为 `8` | `4` |
| `init` | `str` | KMeans：初始化方法，`"k-means++"` / `"random"` | `"k-means++"` |
| `n_init` | `int` 或 `str` | KMeans：随机初始化次数，`"auto"` 自动选择，默认为 `"auto"` | `10` |
| `max_iter` | `int` | KMeans：单次运行最大迭代，默认为 `300` | `300` |
| `eps` | `float` | DBSCAN：邻域半径，默认为 `0.5` | `0.5` |
| `min_samples` | `int` | DBSCAN：核心点最小邻域样本数，默认为 `5` | `10` |
| `random_state` | `int` | 随机种子 | `42` |

训练后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `labels_` | `ndarray` | 每个样本的聚类标签（-1 = 噪声） |
| `cluster_centers_` | `ndarray` | KMeans：簇中心坐标 |

#### 示例代码

```python
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

X, yTrue = datasets.make_blobs(n_samples=300, centers=4, random_state=42)

labelsKm = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X)
labelsDb = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)

nClustersDb = len(set(labelsDb)) - (1 if -1 in labelsDb else 0)
print(f"KMeans 轮廓系数: {silhouette_score(X, labelsKm):.4f}")
print(f"DBSCAN 聚类数: {nClustersDb}")
print(f"DBSCAN 噪声点数: {np.sum(labelsDb == -1)}")
```

#### 输出

```text
KMeans 轮廓系数: 0.7916
DBSCAN 聚类数: 4
DBSCAN 噪声点数: 0
```

#### 理解重点

- KMeans 对球形簇更友好，DBSCAN 对噪声和任意形状更稳健
- DBSCAN 的 `eps` 和 `min_samples` 对结果非常敏感——建议用 k-distance 图确定 `eps`
- 聚类结果应结合业务可解释性检验，不只看内部指标
- `n_clusters` 可通过肘部法或轮廓系数辅助确定

## 9. 降维模型

### `PCA` / `TSNE`

#### 作用

PCA 是线性降维，强调最大方差方向，适合特征压缩与去噪。t-SNE 是非线性嵌入，更偏可视化探索而非特征工程。降维后应检查类别可分性和信息保留程度。

#### 重点方法

```python
PCA(n_components=None, *, copy=True, whiten=False, svd_solver='auto',
    random_state=None)
TSNE(n_components=2, *, perplexity=30.0, learning_rate='auto',
     n_iter=1000, random_state=None)
# fit(X) → transform(X) / fit_transform(X)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_components` | `int` 或 `float` | 目标维度，`None` 保留全量，float=解释方差比例 | `2` / `0.95` |
| `whiten` | `bool` | PCA：白化变换，输出各分量方差为 1，默认为 `False` | `False` |
| `svd_solver` | `str` | PCA：SVD 求解器，`"auto"` 自动选择 | `"auto"` |
| `perplexity` | `float` | TSNE：平衡局部与全局的困惑度，默认为 `30.0` | `30.0` |
| `learning_rate` | `float` 或 `str` | TSNE：学习率，`"auto"` 自动，默认为 `"auto"` | `"auto"` |
| `n_iter` | `int` | TSNE：优化迭代次数，默认为 `1000` | `1000` |
| `random_state` | `int` | 随机种子 | `42` |

训练后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `explained_variance_ratio_` | `ndarray` | PCA：各主成分解释方差比 |

#### 示例代码

```python
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

X, y = datasets.load_iris(return_X_y=True)

pca = PCA(n_components=2)
Xpca = pca.fit_transform(X)
print(f"PCA 解释方差比: {pca.explained_variance_ratio_}")
print(f"PCA 累计解释方差: {pca.explained_variance_ratio_.sum():.4f}")

tsne = TSNE(n_components=2, random_state=42)
Xtsne = tsne.fit_transform(X)
print(f"t-SNE 输出形状: {Xtsne.shape}")
```

#### 输出

```text
PCA 解释方差比: [0.92461872 0.05306648]
PCA 累计解释方差: 0.9777
t-SNE 输出形状: (150, 2)
```

#### 理解重点

- PCA 可用于降噪和压缩，t-SNE 更适合可视化探索
- t-SNE 的空间距离不宜直接做定量解释——只适合辅助观察
- 降维后建模时应验证性能是否受损
- PCA 的 `n_components=0.95` 可自动保留 95% 解释方差的维度数

## 常见坑

1. 忘记对 SVM、KNN 做标准化，导致性能异常波动
2. 只比较准确率，不结合训练成本与可解释性
3. 将 t-SNE 结果直接作为下游生产特征而不做稳定性验证
4. 决策树 `max_depth=None` 在小数据上几乎必然过拟合
5. KMeans 的 `n_clusters` 靠猜——应结合肘部法和业务分群

## 小结

- 模型选择不应只看分数，还要考虑成本、稳定性与解释性
- 推荐先建立线性和树模型基线，再逐步引入复杂模型
- 距离模型（SVM、KNN）必须标准化；树模型无需
- 集成模型是多数表格任务的最强通用方案
- 降维和聚类用于辅助分析，不直接用于预测任务
