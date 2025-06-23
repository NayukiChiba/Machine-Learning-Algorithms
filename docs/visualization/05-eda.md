# 探索性数据分析可视化

> 对应代码: [05_eda.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Visualization/05_eda.py)

## 分布分析

```python
# 直方图 + KDE
sns.histplot(df['column'], kde=True)

# 标记均值和中位数
ax.axvline(df['column'].mean(), color='red', label='Mean')
ax.axvline(df['column'].median(), color='green', label='Median')
```

## 相关性分析

```python
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
```

## 分类变量分析

```python
# 频数统计
df['category'].value_counts().plot(kind='bar')

# 分类箱线图
sns.boxplot(x='category', y='value', data=df)
```

## 练习

```bash
python Basic/Visualization/05_eda.py
```
