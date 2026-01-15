# 时间序列处理

> 对应代码: [07_timeseries.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Pandas/07_timeseries.py)

## 学习目标

- 掌握时间序列的创建和索引
- 学会时间重采样
- 理解滚动窗口操作

## 创建时间序列

```python
pd.to_datetime(['2023-01-01', '2023-01-02'])
pd.date_range('2023-01-01', periods=10, freq='D')
```

常用频率：

| 频率 | 说明   |
| ---- | ------ |
| `D`  | 天     |
| `H`  | 小时   |
| `W`  | 周     |
| `ME` | 月末   |
| `B`  | 工作日 |

## 时间索引属性

```python
ts.index.year
ts.index.month
ts.index.day
ts.index.dayofweek
```

## 时间切片

```python
ts['2023-01']              # 选择整个月
ts['2023-01-15':'2023-01-20']  # 范围选择
```

## 重采样

```python
df.resample('W').sum()   # 按周聚合
df.resample('M').mean()  # 按月平均
```

## 滚动窗口

```python
ts.rolling(3).mean()    # 3期移动平均
ts.rolling(3).sum()     # 3期移动求和
ts.ewm(span=3).mean()   # 指数加权移动平均
```

## 时间偏移

```python
ts.shift(1)       # 向后偏移1期
ts.shift(-1)      # 向前偏移1期
ts.pct_change()   # 百分比变化
```

## 练习

```bash
python Basic/Pandas/07_timeseries.py
```
