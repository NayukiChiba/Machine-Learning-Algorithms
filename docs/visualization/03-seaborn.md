# Seaborn 库入门

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

![03_catplot](https://nayukichiba.github.io/Machine-Learning-Algorithms/outputs/visualization/03_catplot.png)

## 分布图

```python
sns.histplot(data, kde=True)
sns.kdeplot(data, fill=True)
```

![03_distplot](https://nayukichiba.github.io/Machine-Learning-Algorithms/outputs/visualization/03_distplot.png)

## 回归图

```python
sns.regplot(x='x', y='y', data=df)
```

![03_regplot](https://nayukichiba.github.io/Machine-Learning-Algorithms/outputs/visualization/03_regplot.png)

## 热力图

```python
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
```

![03_heatmap](https://nayukichiba.github.io/Machine-Learning-Algorithms/outputs/visualization/03_heatmap.png)

## 配对图

```python
sns.pairplot(df, hue='category')
```

![03_pairplot](https://nayukichiba.github.io/Machine-Learning-Algorithms/outputs/visualization/03_pairplot.png)

## 练习

```bash
python Basic/Visualization/03_seaborn.py
```
