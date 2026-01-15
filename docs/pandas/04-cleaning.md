# 数据清洗与处理

> 对应代码: [04_cleaning.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Pandas/04_cleaning.py)

## 学习目标

- 掌握缺失值检测和处理
- 学会去除重复值
- 了解数据类型转换和字符串操作

## 缺失值处理

```python
df.isnull()           # 检测缺失值
df.isnull().sum()     # 统计每列缺失数量
df.dropna()           # 删除含缺失值的行
df.fillna(0)          # 用0填充
df.fillna(method='ffill')  # 前向填充
```

## 重复值处理

```python
df.duplicated()       # 检测重复值
df.drop_duplicates()  # 删除重复值
df.drop_duplicates(subset=['A'])  # 基于特定列去重
```

## 数据类型转换

```python
df['A'] = df['A'].astype(int)
df['Date'] = pd.to_datetime(df['Date'])
```

## 字符串操作

```python
df['Name'].str.strip()      # 去除空格
df['Name'].str.lower()      # 转小写
df['Name'].str.upper()      # 转大写
df['Email'].str.contains('example')  # 包含匹配
df['Email'].str.split('@')  # 分割
```

## 值替换

```python
df['A'].replace(1, 100)
df['B'].replace({'yes': 1, 'no': 0})
```

## 练习

```bash
python Basic/Pandas/04_cleaning.py
```
