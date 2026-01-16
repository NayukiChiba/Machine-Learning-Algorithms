# 假设检验

> 对应代码: [03_hypothesis.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Scipy/03_hypothesis.py)

## t 检验

```python
from scipy import stats

# 单样本 t 检验
t_stat, p_value = stats.ttest_1samp(sample, 100)

# 独立样本 t 检验
t_stat, p_value = stats.ttest_ind(group1, group2)

# 配对 t 检验
t_stat, p_value = stats.ttest_rel(before, after)
```

## 卡方检验

```python
# 拟合优度检验
chi2, p_value = stats.chisquare(observed, f_exp=expected)

# 独立性检验 (列联表)
chi2, p_value, dof, expected = stats.chi2_contingency(table)
```

## 方差分析

```python
# 单因素 ANOVA
f_stat, p_value = stats.f_oneway(group1, group2, group3)
```

## 非参数检验

```python
# Mann-Whitney U 检验
stat, p_value = stats.mannwhitneyu(group1, group2)

# Wilcoxon 符号秩检验
stat, p_value = stats.wilcoxon(before, after)
```

## 练习

```bash
python Basic/Scipy/03_hypothesis.py
```
