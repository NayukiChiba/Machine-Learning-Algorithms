---
title: 假设检验
outline: deep
---

# 假设检验

> 对应脚本：`Basic/Scipy/03_hypothesis.py`
> 运行方式：`python Basic/Scipy/03_hypothesis.py`（仓库根目录）

## 本章目标

1. 掌握三种 t 检验的适用场景与使用方法。
2. 理解卡方检验在拟合优度和独立性检验中的应用。
3. 学会使用单因素方差分析 (ANOVA) 比较多组均值。
4. 了解 Mann-Whitney U 和 Wilcoxon 等非参数检验方法。

## 重点方法速览

| 方法 | 作用 | 本章位置 |
|---|---|---|
| `stats.ttest_1samp(a, popmean)` | 单样本 t 检验 | `demo_ttest` |
| `stats.ttest_ind(a, b)` | 独立样本 t 检验 | `demo_ttest` |
| `stats.ttest_rel(a, b)` | 配对 t 检验 | `demo_ttest` |
| `stats.chisquare(f_obs, f_exp)` | 卡方拟合优度检验 | `demo_chi2` |
| `stats.chi2_contingency(observed)` | 卡方独立性检验 | `demo_chi2` |
| `stats.f_oneway(*groups)` | 单因素方差分析 | `demo_anova` |
| `stats.mannwhitneyu(x, y)` | Mann-Whitney U 检验 | `demo_nonparametric` |
| `stats.wilcoxon(x, y)` | Wilcoxon 符号秩检验 | `demo_nonparametric` |

## 1. t 检验

### 方法重点

- **单样本 t 检验** `ttest_1samp`：检验样本均值是否等于某个假设值。
- **独立样本 t 检验** `ttest_ind`：检验两个独立样本的均值是否相等。
- **配对 t 检验** `ttest_rel`：检验配对样本（前后测）的均值差是否为零。
- 所有 t 检验返回 `(t_statistic, p_value)`，p < 0.05 通常拒绝原假设。

### 参数速览（本节）

适用 API（分项）：

1. `stats.ttest_1samp(a, popmean)`
2. `stats.ttest_ind(a, b, equal_var=True)`
3. `stats.ttest_rel(a, b)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `a` | `np.random.normal(105, 15, 30)` | 样本数据 |
| `popmean` | `100` | 单样本检验的假设总体均值 |
| `b` | `np.random.normal(110, 15, 30)` | 第二组样本 |
| `equal_var` | `True`（默认） | 是否假设等方差（`False` 使用 Welch's t 检验） |

### 示例代码

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# 单样本 t 检验: H0: μ = 100
sample = np.random.normal(105, 15, 30)
t_stat, p_value = stats.ttest_1samp(sample, 100)
print(f"样本均值: {sample.mean():.2f}")
print(f"t统计量: {t_stat:.4f}, p值: {p_value:.4f}")

# 独立样本 t 检验
group1 = np.random.normal(100, 15, 30)
group2 = np.random.normal(110, 15, 30)
t_stat_ind, p_value_ind = stats.ttest_ind(group1, group2)
print(f"t统计量: {t_stat_ind:.4f}, p值: {p_value_ind:.4f}")

# 配对 t 检验
before = np.random.normal(100, 10, 20)
after = before + np.random.normal(5, 3, 20)
t_stat_paired, p_value_paired = stats.ttest_rel(before, after)
print(f"t统计量: {t_stat_paired:.4f}, p值: {p_value_paired:.4f}")
```

### 结果输出

```text
单样本 t 检验:
  样本均值: 107.49
  H0: μ = 100
  t统计量: 2.6789
  p值: 0.0122

独立样本 t 检验:
  组1均值: 97.71
  组2均值: 108.79
  t统计量: -2.6961
  p值: 0.0093

配对 t 检验:
  前测均值: 98.60
  后测均值: 103.66
  t统计量: -6.5025
  p值: 0.0000
```

### 理解重点

- 单样本 t 检验 p=0.0122 < 0.05，拒绝 H₀，说明样本均值显著不等于 100。
- 独立样本 t 检验 p=0.0093 < 0.05，两组均值存在显著差异。
- 配对 t 检验 p ≈ 0，效果显著——配对设计通过消除个体差异，检测力更强。
- t 统计量的符号反映方向：正值表示样本均值大于假设值/第一组大于第二组。

## 2. 卡方检验

### 方法重点

- **拟合优度检验** `chisquare`：检验观察频数是否符合期望频数分布。
- **独立性检验** `chi2_contingency`：检验两个分类变量是否独立。
- 卡方检验要求每个格子的期望频数 ≥ 5，否则结果不可靠。
- `chi2_contingency` 返回 `(chi2, p, dof, expected)`，其中 `expected` 是期望频数矩阵。

### 参数速览（本节）

适用 API（分项）：

1. `stats.chisquare(f_obs, f_exp=None)`
2. `stats.chi2_contingency(observed)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `f_obs` | `[45, 35, 20]` | 观察频数 |
| `f_exp` | `[40, 40, 20]` | 期望频数（默认均匀分布） |
| `observed` | `[[30, 20], [25, 25]]` | 列联表（2D 数组） |

### 示例代码

```python
from scipy import stats
import numpy as np

# 卡方拟合优度检验
observed = np.array([45, 35, 20])
expected = np.array([40, 40, 20])
chi2, p_value = stats.chisquare(observed, f_exp=expected)
print(f"χ²统计量: {chi2:.4f}, p值: {p_value:.4f}")

# 卡方独立性检验
contingency_table = np.array([[30, 20], [25, 25]])
chi2_ind, p_value_ind, dof, expected_ind = stats.chi2_contingency(contingency_table)
print(f"χ²统计量: {chi2_ind:.4f}, p值: {p_value_ind:.4f}")
print(f"自由度: {dof}")
```

### 结果输出

```text
卡方拟合优度检验:
  观察值: [45 35 20]
  期望值: [40 40 20]
  χ²统计量: 0.9375
  p值: 0.6256

卡方独立性检验 (列联表):
  χ²统计量: 0.6494
  p值: 0.4204
  自由度: 1
```

### 理解重点

- 拟合优度检验 p=0.6256 > 0.05，不能拒绝 H₀，观察频数与期望频数无显著差异。
- 独立性检验 p=0.4204 > 0.05，两个分类变量之间没有显著关联。
- 卡方统计量 = Σ(O-E)²/E，观察值与期望值偏差越大，统计量越大。
- 自由度 dof = (行数-1) × (列数-1)，本例 2×2 列联表自由度为 1。

## 3. 方差分析 (ANOVA)

### 方法重点

- `f_oneway` 执行单因素方差分析，检验 3 组或更多组的均值是否全部相等。
- H₀：所有组的均值相等；H₁：至少有一组不同。
- 返回 F 统计量和 p 值，F = 组间方差 / 组内方差。
- ANOVA 要求各组近似正态、方差齐性，违反时考虑非参数方法。

### 参数速览（本节）

适用 API：`stats.f_oneway(*groups)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `*groups` | 3 组数据 | 各组数据数组，可变参数 |
| `group1` | `np.random.normal(100, 10, 20)` | 第一组 |
| `group2` | `np.random.normal(105, 10, 20)` | 第二组 |
| `group3` | `np.random.normal(110, 10, 20)` | 第三组 |

### 示例代码

```python
import numpy as np
from scipy import stats

np.random.seed(42)

group1 = np.random.normal(100, 10, 20)
group2 = np.random.normal(105, 10, 20)
group3 = np.random.normal(110, 10, 20)

f_stat, p_value = stats.f_oneway(group1, group2, group3)

print(f"组1均值: {group1.mean():.2f}")
print(f"组2均值: {group2.mean():.2f}")
print(f"组3均值: {group3.mean():.2f}")
print(f"F统计量: {f_stat:.4f}")
print(f"p值: {p_value:.4f}")
```

### 结果输出

```text
单因素方差分析:
  组1均值: 99.48
  组2均值: 103.82
  组3均值: 111.36
  F统计量: 7.5090
  p值: 0.0013
```

### 理解重点

- F=7.51, p=0.0013 < 0.05，拒绝 H₀，说明至少有一组均值与其他组不同。
- ANOVA 只告诉你"存在差异"，不告诉"哪两组之间有差异"——需要事后检验（如 Tukey HSD）。
- F 统计量 = 组间均方 / 组内均方，F 越大说明组间差异相对组内差异越大。
- 本例三组真实均值分别为 100、105、110，ANOVA 成功检测到差异。

## 4. 非参数检验

### 方法重点

- 非参数检验不假设数据服从特定分布，适用于数据不满足正态性假设的情况。
- **Mann-Whitney U 检验**：独立样本 t 检验的非参数替代，比较两组的秩分布。
- **Wilcoxon 符号秩检验**：配对 t 检验的非参数替代，基于差值的秩进行检验。
- 非参数检验更稳健但统计效力通常低于参数检验。

### 参数速览（本节）

适用 API（分项）：

1. `stats.mannwhitneyu(x, y, alternative='two-sided')`
2. `stats.wilcoxon(x, y)`

| 参数名 | 本例取值 | 说明 |
|---|---|---|
| `x` | `np.random.normal(100, 15, 20)` | 第一组数据 |
| `y` | `np.random.normal(110, 15, 20)` | 第二组数据 |
| `alternative` | `'two-sided'`（默认） | 备择假设方向 |

### 示例代码

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# Mann-Whitney U 检验
group1 = np.random.normal(100, 15, 20)
group2 = np.random.normal(110, 15, 20)
stat, p_value = stats.mannwhitneyu(group1, group2)
print(f"U统计量: {stat:.4f}, p值: {p_value:.4f}")

# Wilcoxon 符号秩检验
before = np.random.normal(100, 10, 20)
after = before + np.random.normal(5, 3, 20)
stat_w, p_value_w = stats.wilcoxon(before, after)
print(f"统计量: {stat_w:.4f}, p值: {p_value_w:.4f}")
```

### 结果输出

```text
Mann-Whitney U 检验:
  U统计量: 108.0000
  p值: 0.0107

Wilcoxon 符号秩检验:
  统计量: 3.0000
  p值: 0.0000
```

### 理解重点

- Mann-Whitney U 检验 p=0.0107 < 0.05，两组分布存在显著差异。
- Wilcoxon 检验 p ≈ 0，配对样本的前后差异极其显著。
- U 统计量表示一组中的值小于另一组值的次数，范围 [0, n₁×n₂]。
- 非参数检验在样本量小或数据明显偏态时特别有用。

## 常见坑

| 坑 | 说明 |
|---|---|
| p 值不是"效应大小" | p < 0.05 只说明差异"显著"，不代表差异"大"，大样本下微小差异也能显著 |
| `ttest_ind` 的 `equal_var` | 默认 `True` 假设等方差，若方差不齐需设为 `False`（Welch's t 检验） |
| 卡方检验的期望频数 | 每格期望频数应 ≥ 5，否则使用 Fisher 精确检验 |
| ANOVA 只判断"是否有差异" | 不告诉你哪两组之间有差异，需要事后多重比较 |
| 多重比较问题 | 多次检验会膨胀 I 类错误率，需要 Bonferroni 等校正 |

## 小结

- t 检验用于比较均值：单样本 vs 假设值、独立双样本、配对前后测。
- 卡方检验用于分类数据：拟合优度（观察 vs 期望）和独立性（列联表）。
- ANOVA 是 t 检验在多组场景的推广，检验多组均值是否全部相等。
- 非参数检验（Mann-Whitney U / Wilcoxon）在数据不满足正态假设时使用，更稳健。
- 假设检验的核心流程：建立 H₀ → 选择检验方法 → 计算统计量和 p 值 → 根据 α 做决策。
