# 数据选择与过滤

> 对应代码: [03_selection.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Pandas/03_selection.py)

## 学习目标

- 掌握列选择和行选择
- 理解 loc 和 iloc 的区别
- 学会条件过滤

## 列选择

```python
df['Name']           # 选择单列 (返回 Series)
df[['Name', 'Age']]  # 选择多列 (返回 DataFrame)
```

## 行选择

```python
df[0:3]       # 切片选择
df.head(3)    # 前3行
df.tail(3)    # 后3行
```

## loc 和 iloc

| 方法   | 索引类型 | 示例                  |
| ------ | -------- | --------------------- |
| `loc`  | 标签索引 | `df.loc['a', 'Name']` |
| `iloc` | 位置索引 | `df.iloc[0, 1]`       |

```python
df.loc['b', 'Name']           # 标签索引
df.loc['a':'c', ['Name', 'Age']]

df.iloc[1, 0]                 # 位置索引
df.iloc[0:3, 0:2]
```

## 条件过滤

```python
df[df['Age'] > 28]                              # 单条件
df[(df['Age'] > 25) & (df['Salary'] > 10000)]  # 多条件 AND
df[(df['City'] == 'A') | (df['City'] == 'B')]  # 多条件 OR
df[df['City'].isin(['Beijing', 'Shanghai'])]   # isin
df.query('Age > 28 and Salary > 10000')        # query 方法
```

## 练习

```bash
python Basic/Pandas/03_selection.py
```
