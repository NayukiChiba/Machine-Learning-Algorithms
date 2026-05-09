---
title: 正则化回归 — 数据构成
outline: deep
---

# 数据构成

## 本章目标

1. 明确正则化回归数据的来源——`loadRegularizationDataset()` 在 diabetes 基础上构造三层特征。
2. 理解三层特征结构（原始医学 → 共线特征 → 纯噪声）的设计意图。
3. 理解标准化在数据层的执行边界——`StandardScaler` 由流水线层而非数据层执行。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `loadRegularizationDataset()` | 方法 | 加载 diabetes 并追加共线特征和纯噪声特征——返回 `(442, 22)` DataFrame |
| `load_diabetes(as_frame=True)` | 函数 | scikit-learn 提供的糖尿病回归数据集——10 个医学特征 |
| `bmi_corr` / `bp_corr` / `s5_corr` | 列名 | 人为制造多重共线性的相关特征——与原始列相关系数约 0.9 |
| `noise_1` ~ `noise_8` | 列名 | 纯随机噪声特征——用于观察 L1 稀疏化效果 |
| `price` | 列名 | 回归目标列——由 diabetes 的 `target` 重命名而来 |

## 1. 数据入口：`loadRegularizationDataset()`

### 参数速览

适用函数：`loadRegularizationDataset()`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `randomState` | `int` | 随机种子——保证共线噪声和纯噪声的可复现性 | `42` |
| 返回值 | `DataFrame` | 形状 `(442, 22)`——10 原始 + 3 共线 + 8 噪声 + 1 标签 | — |

### 示例代码

```python
def loadRegularizationDataset(self) -> DataFrame:
    rng = np.random.RandomState(self.randomState)
    data = load_diabetes(as_frame=True).frame.copy().rename(
        columns={"target": "price"}
    )
    # 追加共线特征
    data["bmi_corr"] = data["bmi"] * 0.9 + rng.normal(scale=0.02, size=len(data))
    data["bp_corr"] = data["bp"] * 0.9 + rng.normal(scale=0.02, size=len(data))
    data["s5_corr"] = data["s5"] * 0.9 + rng.normal(scale=0.02, size=len(data))
    # 追加纯噪声特征
    for index in range(8):
        data[f"noise_{index + 1}"] = rng.normal(size=len(data))
    return data
```

### 理解重点

- 基础数据来自 scikit-learn 的真实糖尿病数据集——10 个标准化后的医学特征，442 个样本。
- 原始列名 `target` 被重命名为 `price`——保持与仓库其他回归分册的标签列名一致。
- 共线特征通过 `原始值 × 0.9 + 微小噪声` 构造——与原始列的相关系数约 0.9，刻意制造多重共线性。

## 2. 三层特征结构

### 参数速览

| 特征层 | 列名 | 数量 | 设计意图 |
|---|---|---|---|
| 原始医学特征 | `age`, `sex`, `bmi`, `bp`, `s1`~`s6` | 10 | 提供真实回归信号——来自真实医学数据 |
| 共线特征 | `bmi_corr`, `bp_corr`, `s5_corr` | 3 | 人为制造多重共线性——观察 Ridge 的收缩 vs Lasso 的筛选 |
| 纯噪声特征 | `noise_1` ~ `noise_8` | 8 | 完全无预测能力的随机列——观察 Lasso 是否将其系数压到零 |
| 标签列 | `price` | 1 | 糖尿病病情进展量化指标 |

### 理解重点

- 三层结构是刻意设计的——每一层测试正则化的不同能力：共线层测试稳定性，噪声层测试稀疏性。
- `bmi_corr` 与 `bmi` 高度相关（r ≈ 0.9）——OLS 会在这两个特征之间难以分配系数，而 Ridge 会均匀分摊，Lasso 可能只保留一个。
- `noise_*` 理论上不应有任何非零系数——观察 Lasso 的 `near_zero` 计数可以直接验证 L1 的稀疏化效果。
- 与线性回归的合成数据不同——正则化回归使用真实数据 + 人工干扰，更接近实际应用场景。

## 3. 特征切分与标准化边界

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `test_size` | `float` | 测试集占比 | `0.2` |
| `random_state` | `int` | 切分随机种子 | `42` |
| 训练集形状 | — | `X_train`: `(353, 21)`, `y_train`: `(353,)` | — |
| 测试集形状 | — | `X_test`: `(89, 21)`, `y_test`: `(89,)` | — |
| `StandardScaler` | 预处理 | 仅在 `X_train` 上 `fit_transform`，`X_test` 仅 `transform` | — |

### 示例代码

```python
X = data.drop(columns=["price"])
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

### 理解重点

- 标准化不在数据层执行——数据层只返回原始 DataFrame，标准化由流水线（运行器层）负责。
- 正则化回归**必须标准化**——L1/L2 惩罚对系数量级敏感，未标准化的特征会导致惩罚不均匀。
- 这与线性回归和决策树回归形成关键差异——它们不需要标准化，正则化回归是回归分册中首个强制标准化的模型。
- `StandardScaler` 仅在训练集上 `fit`——测试集使用训练集的均值和标准差做 `transform`，避免数据泄露。

## 4. 数据特征总览

### 参数速览

| 属性 | 值 |
|---|---|
| 样本总数 | 442 |
| 特征总数 | 21（10 原始 + 3 共线 + 8 噪声） |
| 训练样本数 | 353（80%） |
| 测试样本数 | 89（20%） |
| 标签列名 | `price` |
| 是否有缺失值 | 否——diabetes 数据集已预处理 |
| 数据来源 | `sklearn.datasets.load_diabetes` |

### 理解重点

- 442 样本 vs 21 特征——样本量远大于特征数，但共线和噪声特征的存在使得纯 OLS 仍然不稳定。
- 与线性回归的 200 样本 / 3 特征相比——正则化回归的数据规模和复杂度明显更高。
- 与决策树回归的 20640 样本 / 8 特征相比——正则化回归样本量较小但特征结构更复杂（共线 + 噪声层）。

## 5. 数据设计意图：与线性回归/决策树回归的对比

| 数据维度 | 线性回归 | 决策树回归 | 正则化回归 |
|---|---|---|---|
| 数据来源 | 手工合成 | 真实数据（California Housing） | **真实数据 + 人工干扰（diabetes + 共线 + 噪声）** |
| 样本量 | 200 | 20640 | **442** |
| 特征数 | 3 | 8 | **21（10 + 3 + 8）** |
| 特征关系 | 完全独立 | 自然相关 | **刻意构造共线 + 纯噪声** |
| 标签 | `price = 2×面积 + 10×房间 - 3×房龄 + ε` | 加州房价中位数 | **糖尿病病情进展（真实医学指标）** |
| 标准化 | 否 | 否 | **是——强制要求** |
| 设计意图 | 透明验证 OLS 恢复精度 | 非线性树结构演示 | **观察 L1 稀疏化 + L2 收缩 + 共线性处理** |

## 常见坑

1. 忘记 diabetes 的 `target` 已被重命名为 `price`——拆分标签时写错列名。
2. 只关注原始 10 个医学特征——忽略 `bmi_corr`/`bp_corr`/`s5_corr` 和 `noise_*` 是正则化行为展示的核心。
3. 在切分前对全量数据做标准化——造成测试集信息泄露到训练过程。
4. 认为"数据层已标准化"——标准化实际在运行器层执行，数据层返回的是原始值。

## 小结

- 正则化回归数据由三层构成：10 个真实医学特征 + 3 个人工共线特征 + 8 个纯噪声特征 = 21 维。
- 三层结构各有设计意图：原始特征提供信号，共线特征测试收缩/筛选，噪声特征验证稀疏化。
- 标准化是正则化回归的强制预处理——与线性回归和决策树回归形成关键工程差异。
- 数据量（442 样本 / 21 特征）介于线性回归和决策树回归之间——复杂度适中，适合观察正则化行为。
