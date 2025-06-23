# Seaborn 库入门

> 对应代码: [03_seaborn.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Visualization/03_seaborn.py)

## 基本设置

```python
import seaborn as sns
sns.set_theme(style='whitegrid')
```

## 分类图

```python
sns.barplot(x='category', y='value', hue='group', data=df)
sns.boxplot(x='category', y='value', data=df)
sns.violinplot(x='category', y='value', data=df)
```

## 分布图

```python
sns.histplot(data, kde=True)
sns.kdeplot(data, fill=True)
```

## 回归图

```python
sns.regplot(x='x', y='y', data=df)
```

## 热力图

```python
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
```

## 配对图

```python
sns.pairplot(df, hue='category')
```

## 练习

```bash
python Basic/Visualization/03_seaborn.py
```
