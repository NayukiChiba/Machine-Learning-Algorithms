# 数据探索（explore_data.py）

这一模块帮助你快速理解数据的规模、分布和相关性。

---

## 1. 样本结构

```python
print(f"样本数量: {len(data)}")
print(f"特征数量: {len(data.columns) - 1}")
print(f"特征名称: {list(data.columns[:-1])}")
```

作用：
- 了解数据量
- 确认特征列和目标列是否完整

---

## 2. 描述统计

```python
print(data.describe().round(2))
```

输出包括：均值、标准差、最大最小值、分位数。

用途：
- 判断特征的范围
- 发现异常值

---

## 3. 缺失值检查

```python
missing = data.isnull().sum()
```

California Housing 数据集默认无缺失值，但这是标准流程。

---

## 4. 相关性分析

```python
correlation = data.corr()["price"].drop("price").sort_values(ascending=False)
```

- 用皮尔逊相关系数观察线性相关性
- 相关性只是参考，不代表因果

---

## 5. 小结

- 这一步是建模前的“体检”
- 帮助判断数据是否适合当前模型
- 相关性高的特征通常更重要
