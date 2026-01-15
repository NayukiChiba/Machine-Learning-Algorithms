# 高级操作与性能优化

> 对应代码: [09_advanced.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Pandas/09_advanced.py)

## 学习目标

- 掌握透视表和交叉表
- 理解多级索引操作
- 学会性能优化技巧

## 透视表

```python
pd.pivot_table(
    df,
    values='Sales',
    index='Date',
    columns='Region',
    aggfunc='sum'
)
```

## 交叉表

```python
pd.crosstab(df['Gender'], df['City'])
```

## 多级索引

```python
# 创建
arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
index = pd.MultiIndex.from_arrays(arrays, names=['Letter', 'Number'])

# 访问
df.loc['A']          # 选择第一级
df.loc[('A', 1)]     # 选择具体组合
```

## 向量化操作

避免循环，使用向量化操作：

```python
# 慢
for i in range(len(df)):
    result.append(df['A'].iloc[i] + df['B'].iloc[i])

# 快 (向量化)
result = df['A'] + df['B']
```

## 内存优化

```python
# 优化数据类型
df['int_col'] = df['int_col'].astype('int8')
df['float_col'] = df['float_col'].astype('float32')
df['str_col'] = df['str_col'].astype('category')

# 查看内存使用
df.memory_usage(deep=True)
```

## 练习

```bash
python Basic/Pandas/09_advanced.py
```
