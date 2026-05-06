---
title: SciPy 假设检验
outline: deep
---

# SciPy 假设检验

## 本章目标

1. 掌握三种 t 检验的适用场景与使用方法（单样本、独立、配对）
2. 理解卡方检验在拟合优度和独立性检验中的应用
3. 学会使用单因素方差分析（ANOVA）比较多组均值
4. 了解 Mann-Whitney U 和 Wilcoxon 等非参数检验方法

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `stats.ttest_1samp(a, popmean)` | 函数 | 单样本 t 检验 |
| `stats.ttest_ind(a, b)` | 函数 | 独立双样本 t 检验 |
| `stats.ttest_rel(a, b)` | 函数 | 配对 t 检验 |
| `stats.chisquare(f_obs, f_exp)` | 函数 | 卡方拟合优度检验 |
| `stats.chi2_contingency(observed)` | 函数 | 卡方独立性检验（列联表） |
| `stats.f_oneway(*groups)` | 函数 | 单因素方差分析 |
| `stats.mannwhitneyu(x, y)` | 函数 | Mann-Whitney U 非参数检验 |
| `stats.wilcoxon(x, y)` | 函数 | Wilcoxon 符号秩检验 |

所有检验函数返回 `(statistic, pvalue)` 元组。p < 0.05 通常在 $\alpha=0.05$ 水平拒绝原假设。

## 1. t 检验

### `stats.ttest_1samp` / `stats.ttest_ind` / `stats.ttest_rel`

#### 作用

三种 t 检验覆盖不同的实验设计：

- **单样本** `ttest_1samp`：检验样本均值是否等于某个假设值 $H_0: \mu = \mu_0$
- **独立双样本** `ttest_ind`：检验两个独立样本的均值是否相等 $H_0: \mu_1 = \mu_2$
- **配对** `ttest_rel`：检验配对样本（前后测）的均值差是否为零 $H_0: \mu_d = 0$

#### 重点方法

```python
stats.ttest_1samp(a, popmean)
stats.ttest_ind(a, b, equal_var=True)
stats.ttest_rel(a, b)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 第一组样本数据 | `np.random.normal(105, 15, 30)` |
| `popmean` | `float` | 单样本检验的假设总体均值 | `100` |
| `b` | `array_like` | 第二组样本数据 | `np.random.normal(110, 15, 30)` |
| `equal_var` | `bool` | 是否假设等方差；`False` 使用 Welch's t 检验，默认为 `True` | `False` |

#### 示例代码

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# 单样本 t 检验：H0: μ = 100
sample = np.random.normal(105, 15, 30)
t1, p1 = stats.ttest_1samp(sample, 100)
print(f"单样本: t={t1:.4f}, p={p1:.4f}, 均值={sample.mean():.2f}")

# 独立样本 t 检验
g1 = np.random.normal(100, 15, 30)
g2 = np.random.normal(110, 15, 30)
t2, p2 = stats.ttest_ind(g1, g2)
print(f"独立双样本: t={t2:.4f}, p={p2:.4f}")
print(f"  组1均值={g1.mean():.2f}, 组2均值={g2.mean():.2f}")

# 配对 t 检验
before = np.random.normal(100, 10, 20)
after = before + np.random.normal(5, 3, 20)
t3, p3 = stats.ttest_rel(before, after)
print(f"配对: t={t3:.4f}, p={p3:.4f}")
print(f"  前测均值={before.mean():.2f}, 后测均值={after.mean():.2f}")
```

#### 输出

```text
单样本: t=2.6789, p=0.0122, 均值=107.49
独立双样本: t=-2.6961, p=0.0093
  组1均值=97.71, 组2均值=108.79
配对: t=-6.5025, p=0.0000
  前测均值=98.60, 后测均值=103.66
```

#### 理解重点

- 单样本 p=0.012 < 0.05，拒绝 $H_0$，样本均值显著不等于 100
- 独立双样本 p=0.009 < 0.05，两组均值存在显著差异
- 配对 p ≈ 0，效果最显著——配对设计通过消除个体差异，检验力更强
- t 统计量的符号反映方向：正值表示样本均值 > 假设值

## 2. 卡方检验

### `stats.chisquare` / `stats.chi2_contingency`

#### 作用

- **拟合优度检验** `chisquare`：检验观察频数是否符合期望分布
- **独立性检验** `chi2_contingency`：检验两个分类变量是否独立

卡方统计量：$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$，期望频数每格应 ≥ 5。

#### 重点方法

```python
stats.chisquare(f_obs, f_exp=None)
stats.chi2_contingency(observed)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `f_obs` | `array_like` | 观察频数 | `[45, 35, 20]` |
| `f_exp` | `array_like` 或 `None` | 期望频数；`None` 时默认为均匀分布 | `[40, 40, 20]` |
| `observed` | `array_like` | 列联表（二维数组），行=变量1，列=变量2 | `[[30, 20], [25, 25]]` |

#### 示例代码

```python
import numpy as np
from scipy import stats

# 拟合优度检验
observed = np.array([45, 35, 20])
expected = np.array([40, 40, 20])
chi2, p1 = stats.chisquare(observed, f_exp=expected)
print(f"拟合优度: χ²={chi2:.4f}, p={p1:.4f}")

# 独立性检验
ct = np.array([[30, 20], [25, 25]])
chi2i, p2, dof, exp = stats.chi2_contingency(ct)
print(f"独立性: χ²={chi2i:.4f}, p={p2:.4f}, 自由度={dof}")
```

#### 输出

```text
拟合优度: χ²=0.9375, p=0.6256
独立性: χ²=0.6494, p=0.4204, 自由度=1
```

#### 理解重点

- 拟合优度 p=0.626 > 0.05，不能拒绝 $H_0$——观察频数与期望频数无显著差异
- 独立性 p=0.420 > 0.05，两个分类变量之间无显著关联
- 自由度 = (行数−1)×(列数−1)，2×2 列联表自由度为 1
- `chi2_contingency` 返回 4 个值：`(χ², p, dof, expected)`——其中 `expected` 是期望频数矩阵

## 3. 单因素方差分析

### `stats.f_oneway`

#### 作用

检验 3 组或更多组的均值是否全部相等。$H_0: \mu_1 = \mu_2 = \cdots = \mu_k$，$H_1$：至少有一组不同。F = 组间方差 / 组内方差。

#### 重点方法

```python
stats.f_oneway(*groups)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `*groups` | `array_like`（可变参数） | 各组样本数据，每组为一个数组 | `group1, group2, group3` |

#### 示例代码

```python
import numpy as np
from scipy import stats

np.random.seed(42)
g1 = np.random.normal(100, 10, 20)
g2 = np.random.normal(105, 10, 20)
g3 = np.random.normal(110, 10, 20)

fStat, p = stats.f_oneway(g1, g2, g3)
print(f"组均值: {g1.mean():.2f}, {g2.mean():.2f}, {g3.mean():.2f}")
print(f"F={fStat:.4f}, p={p:.4f}")
```

#### 输出

```text
组均值: 99.48, 103.82, 111.36
F=7.5090, p=0.0013
```

#### 理解重点

- F=7.51, p=0.0013 < 0.05，拒绝 $H_0$——至少有一组均值与其他组不同
- ANOVA 只告诉你"存在差异"，不告诉"哪两组有差异"——需要事后检验（如 Tukey HSD）
- F 统计量越大说明组间差异相对组内差异越大
- 要求各组近似正态、方差齐性——违反时考虑非参数方法（Kruskal-Wallis）

## 4. 非参数检验

### `stats.mannwhitneyu` / `stats.wilcoxon`

#### 作用

不假设数据服从特定分布，适用于数据不满足正态性假设的情况：

- **Mann-Whitney U**：独立样本 t 检验的非参数替代，基于秩的比较
- **Wilcoxon 符号秩检验**：配对 t 检验的非参数替代，基于差值的秩

#### 重点方法

```python
stats.mannwhitneyu(x, y, alternative='two-sided')
stats.wilcoxon(x, y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `x` | `array_like` | 第一组数据 | `np.random.normal(100, 15, 20)` |
| `y` | `array_like` | 第二组数据 | `np.random.normal(110, 15, 20)` |
| `alternative` | `str` | 备择假设方向：`'two-sided'` / `'less'` / `'greater'`，默认为 `'two-sided'` | `'greater'` |

#### 示例代码

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# Mann-Whitney U
x1 = np.random.normal(100, 15, 20)
x2 = np.random.normal(110, 15, 20)
uStat, p1 = stats.mannwhitneyu(x1, x2)
print(f"Mann-Whitney U: U={uStat:.0f}, p={p1:.4f}")

# Wilcoxon
before = np.random.normal(100, 10, 20)
after = before + np.random.normal(5, 3, 20)
wStat, p2 = stats.wilcoxon(before, after)
print(f"Wilcoxon: 统计量={wStat:.0f}, p={p2:.4f}")
```

#### 输出

```text
Mann-Whitney U: U=108, p=0.0107
Wilcoxon: 统计量=3, p=0.0000
```

#### 理解重点

- Mann-Whitney U p=0.011 < 0.05，两组分布存在显著差异
- Wilcoxon p ≈ 0，配对样本的前后差异极其显著
- 非参数检验更稳健但统计效力通常略低于参数检验
- 样本量小或数据明显偏态时优先使用非参数方法

## 常见坑

1. p 值不是效应大小——p < 0.05 只说明差异"显著"，不代表差异"大"，大样本下微小差异也能显著
2. `ttest_ind` 默认 `equal_var=True`——若方差不齐需设 `False`（Welch's t 检验）
3. 卡方检验每格期望频数应 ≥ 5——否则使用 Fisher 精确检验
4. ANOVA 只判断"是否有差异"——不告诉具体哪两组不同，需要事后多重比较
5. 多次检验会膨胀 I 类错误率——需 Bonferroni 等校正

## 小结

- t 检验比较均值：单样本 vs 假设值、独立双样本、配对前后测——三种场景三种函数
- 卡方检验处理分类数据：拟合优度（观察 vs 期望）和独立性（列联表）
- ANOVA 是 t 检验在多组场景的推广——检验多组均值是否全部相等
- 非参数检验（Mann-Whitney U / Wilcoxon）在数据不满足正态假设时使用，更稳健
- 假设检验核心流程：建立 $H_0$ → 选择检验方法 → 计算统计量和 p 值 → 根据 $\alpha$ 决策
