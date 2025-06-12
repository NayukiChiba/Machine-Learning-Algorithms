# 数据合并与连接

> 对应代码: [06_merge.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Pandas/06_merge.py)

## 学习目标

- 掌握 concat 纵向和横向合并
- 理解 merge 的各种连接方式
- 学会使用 join 操作

## Concat 合并

```python
pd.concat([df1, df2], axis=0)  # 纵向合并
pd.concat([df1, df2], axis=1)  # 横向合并
pd.concat([df1, df2], ignore_index=True)
```

## Merge 连接

| 连接方式 | 说明          |
| -------- | ------------- |
| `inner`  | 内连接 (交集) |
| `left`   | 左连接        |
| `right`  | 右连接        |
| `outer`  | 外连接 (并集) |

```python
pd.merge(df1, df2, on='key')
pd.merge(df1, df2, on='key', how='left')
pd.merge(df1, df2, left_on='id', right_on='key')
```

## Join 操作

基于索引的连接：

```python
df1.join(df2)
df1.join(df2, how='outer')
```

## 合并指示器

```python
pd.merge(df1, df2, on='key', how='outer', indicator=True)
# _merge 列: left_only, right_only, both
```

## 练习

```bash
python Basic/Pandas/06_merge.py
```
