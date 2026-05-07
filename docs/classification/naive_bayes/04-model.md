---
title: GaussianNB 高斯朴素贝叶斯 — 模型构建
outline: deep
---

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `GaussianNB`。
2. 理解 `GaussianNB` 的构造器参数 `priors` 和 `var_smoothing` 的数学含义。
3. 看清训练完成后最重要的模型属性及其与数学公式的对应关系。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `GaussianNB` 模型，打印训练日志 |
| `GaussianNB(...)` | 类 | scikit-learn 提供的高斯朴素贝叶斯分类器——对连续特征的每个类别每个特征拟合 $\mathcal{N}(\mu_{kj}, \sigma_{kj}^2)$ |
| `model.fit(X_train, y_train)` | 方法 | 在训练数据上统计类别先验和特征高斯参数——纯统计计算，无迭代优化 |
| `model.classes_` | 属性 | 模型识别到的类别标签数组 |
| `model.class_prior_` | 属性 | 各类别先验概率 $P(Y=c_k)$ |
| `model.theta_` | 属性 | 各类别各特征的均值 $\mu_{kj}$ |
| `model.var_` | 属性 | 各类别各特征的方差 $\sigma_{kj}^2$（平滑后） |

## 1. `train_model(...)` 的函数签名

### 参数速览

适用函数：`train_model(X_train, y_train, var_smoothing=1e-9)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like` | 标准化后的训练特征矩阵，形状 $(120, 4)$，传入 `GaussianNB.fit()` | `X_train_s` |
| `y_train` | `array_like` | 训练标签向量，形状 $(120,)$，取值 $y_i \in \{0, 1, 2\}$ | `y_train` |
| `var_smoothing` | `float` | 方差平滑项 $\epsilon$。实际计算 $\sigma_{kj}^2 + \epsilon \cdot \sigma_{\max}^2$，防止 $\sigma_{kj}^2 \to 0$ 数值崩溃。默认 `1e-9` | `1e-9`、`1e-8` |
| 返回值 | `GaussianNB` | 已完成 `fit()` 的模型对象，含 `classes_`、`class_prior_`、`theta_`、`var_` 等属性 | — |

### 示例代码

```python
from model_training.classification.naive_bayes import train_model

model = train_model(X_train_s, y_train)
```

### 理解重点

- 当前入口很直接：只负责构建一个 `GaussianNB` 并 `fit`，没有变体对比或超参数搜索。
- 所有默认超参数都写在函数签名里，阅读成本低，适合作为源码入门。
- `train_model(...)` 是对 `sklearn.naive_bayes.GaussianNB` 的薄封装——算法本体在 sklearn，本仓库负责组织日志和工程流程。

## 2. `GaussianNB` 构造器参数

### 参数速览

适用 API：`GaussianNB(priors=None, var_smoothing=1e-9)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `priors` | `array_like` 或 `None` | 类别的先验概率 $P(Y=c_k)$。`None` 时从训练数据估计：$P(Y=c_k) = n_k / N$。可手动传入数组覆盖数据估计 | `None`、`[0.3, 0.3, 0.4]` |
| `var_smoothing` | `float` | 方差平滑项 $\epsilon$。最终方差为 $\sigma_{kj}^2 + \epsilon \cdot \sigma_{\max}^2$，其中 $\sigma_{\max}^2$ 是所有特征所有类别中最大的方差。防止 $\sigma_{kj}^2 \approx 0$ 时 $\frac{1}{\sqrt{2\pi \sigma^2}} \to \infty$ 导数炸 | `1e-9`、`1e-8`、`1e-7` |

### 示例代码

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB(var_smoothing=1e-9)
model.fit(X_train_s, y_train)
```

### 理解重点

- GaussianNB 的参数极简——一共两个：`priors`（先验）和 `var_smoothing`（数值保护）。这反映了朴素贝叶斯"参数少、假设强"的特点。
- `priors` 默认为 `None` 时从训练数据按频率估计，对 iris 均衡数据来说三个类别的先验约为 $[0.33, 0.33, 0.33]$。
- `var_smoothing` 是当前分册最重要的超参数——它直接关联到方差为零时的数值稳定性问题。
- GaussianNB 的 `fit()` 不涉及迭代优化——它只是扫描数据统计均值和方差。这与逻辑回归的 `lbfgs` 迭代和决策树的递归分裂形成鲜明对比。

## 3. 训练完成后的关键属性

### 参数速览

| 属性名 | 类型 | 数学含义 | 说明 |
|---|---|---|---|
| `classes_` | `ndarray`，形状 `(n_classes,)` | $\{c_1, c_2, c_3\}$ | 模型识别到的类别标签列表，iris 中为 `[0, 1, 2]` |
| `class_prior_` | `ndarray`，形状 `(n_classes,)` | $P(Y=c_k) = n_k / N$ | 各类别的先验概率 |
| `class_count_` | `ndarray`，形状 `(n_classes,)` | $n_k$ | 训练集中各类别的样本数 |
| `theta_` | `ndarray`，形状 `(n_classes, n_features)` | $\mu_{kj}$ | 各类别各特征的均值，对应高斯分布的位置参数 |
| `var_` | `ndarray`，形状 `(n_classes, n_features)` | $\sigma_{kj}^2$（平滑后） | 各类别各特征的方差，对应高斯分布的尺度参数——已应用 `var_smoothing` |
| `epsilon_` | `float` | $\epsilon \cdot \sigma_{\max}^2$ | `var_smoothing` 对应的实际平滑绝对值 |

### 示例代码

```python
print(f"类别: {model.classes_.tolist()}")
print(f"类别先验: {model.class_prior_.round(4)}")
print(f"各类别样本数: {model.class_count_}")
print(f"均值(theta_):\n{model.theta_}")
print(f"方差(var_):\n{model.var_}")
```

### 理解重点

- `theta_` 和 `var_` 是 GaussianNB 最核心的两个训练产出——它们就是各类别下各特征高斯分布的参数。
- `theta_` 形状 $(3, 4)$ 意味着 3 个类别 × 4 个特征 = 12 个均值；`var_` 同样有 12 个方差——模型一共只估计 24 个数字，训练极快。
- `class_prior_` 把"先验概率"这一理论概念直接映射为可观察的数值，是理解生成式分类思路的入口。
- `epsilon_` 提供了方差平滑的实际量级，对于理解 `var_smoothing` 是否真正生效有参考价值。

## 4. 训练阶段的工程封装

除了 `GaussianNB(...).fit(...)` 之外，`train_model(...)` 还做了几层工程包装：

### 参数速览

| 输出项 | 作用 |
|---|---|
| `@print_func_info` 标题 | 在终端中定位训练入口 |
| `@timeit` 训练耗时 | 观察 `fit()` 的执行时间——对 GaussianNB 通常是毫秒级 |
| `var_smoothing` 日志 | 确认当前平滑参数配置 |
| `类别` 日志 | 确认多分类类别集合 |
| `类别先验` 日志 | 观察各类别基础比例，对应 $P(Y=c_k)$ |

### 理解重点

- 当前封装强调的是教学型可读性——通过装饰器打印函数信息和耗时，通过 `print` 输出关键属性。
- 这一层把"构建模型""训练模型""打印结果"收在一个函数里，方便流水线和文档复用。
- 从工程角度看，这样的拆分让 `pipelines/classification/naive_bayes.py` 保持简洁——编排层不需要关心日志打印细节。

## 常见坑

1. 误以为当前实现使用的是所有朴素贝叶斯的通用封装——`train_model` 明确构建 `GaussianNB`，不是 `MultinomialNB` 或 `BernoulliNB`。
2. 只知道 `predict(...)`，却忽略 `theta_`、`var_`、`class_prior_` 才是理解概率分类本质的关键属性。
3. 忘记当前 `X_train` 应该是标准化后的特征——虽然 GaussianNB 不像逻辑回归那样对尺度敏感，但标准化影响方差估计的稳定性和 PCA 可视化。
4. 把训练函数和后续评估逻辑混在一起理解——`train_model` 只负责训练主模型，不负责混淆矩阵、ROC 等诊断。

## 小结

- `train_model(...)` 是本仓库 Naive Bayes 的核心训练入口，是对 `GaussianNB` 的薄封装。
- `GaussianNB` 只有两个构造器参数（`priors` 和 `var_smoothing`），属于参数最少的分类模型之一。
- 训练完成后的关键属性：`theta_`（均值 $\mu_{kj}$）、`var_`（方差 $\sigma_{kj}^2$）、`class_prior_`（先验概率）——全部是解析计算，无迭代优化。
