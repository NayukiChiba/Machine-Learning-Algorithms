# 数据分组与聚合

> 对应代码: [05_groupby.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Pandas/05_groupby.py)

## 学习目标

- 掌握 groupby 分组操作
- 学会使用聚合函数
- 理解 transform 和 apply 的区别

## GroupBy 基本操作

```python
grouped = df.groupby('Department')
grouped.ngroups        # 分组数量
for name, group in grouped:
    print(name, group)
```

## 聚合函数

```python
grouped['Salary'].sum()
grouped['Salary'].mean()
grouped['Salary'].agg(['sum', 'mean', 'max'])
```

## 多列不同聚合

```python
grouped.agg({
    'Salary': ['mean', 'sum'],
    'Bonus': 'sum',
    'Years': 'mean'
})
```

## Transform

返回与原数据相同长度的结果：

```python
df['Dept_Mean'] = df.groupby('Dept')['Salary'].transform('mean')
```

## Apply

可以返回任意形状的结果：

```python
def top_employee(group):
    return group.nlargest(1, 'Salary')

grouped.apply(top_employee)
```

## 练习

```bash
python Basic/Pandas/05_groupby.py
```
