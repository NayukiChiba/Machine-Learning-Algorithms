# Pandas 基础入门

> 对应代码: [01_basics.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Pandas/01_basics.py)

## 学习目标

- 了解 Pandas 的核心数据结构
- 掌握 Series 和 DataFrame 的创建
- 学会基本的数据查看方法

## 核心数据结构

| 数据结构      | 维度 | 说明                 |
| ------------- | ---- | -------------------- |
| **Series**    | 一维 | 带索引的一维数组     |
| **DataFrame** | 二维 | 带行列索引的二维表格 |

## Series

```python
import pandas as pd

# 从列表创建
s = pd.Series([1, 2, 3, 4, 5])

# 指定索引
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])

# 从字典创建
s = pd.Series({'apple': 100, 'banana': 200})
```

## DataFrame

```python
# 从字典创建
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['Beijing', 'Shanghai', 'Guangzhou']
}
df = pd.DataFrame(data)
```

## 基本数据查看

| 方法            | 说明          |
| --------------- | ------------- |
| `df.head(n)`    | 查看前 n 行   |
| `df.tail(n)`    | 查看后 n 行   |
| `df.info()`     | 数据基本信息  |
| `df.describe()` | 统计描述      |
| `df.shape`      | 形状 (行, 列) |
| `df.dtypes`     | 各列数据类型  |

## 练习

```bash
python Basic/Pandas/01_basics.py
```
