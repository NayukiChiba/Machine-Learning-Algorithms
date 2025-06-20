# Pandas 数据可视化

> 对应代码: [04_pandas_viz.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Visualization/04_pandas_viz.py)

## DataFrame.plot()

```python
df.plot()                    # 折线图
df.plot(kind='bar')          # 柱状图
df.plot(kind='area')         # 面积图
df.plot(kind='box')          # 箱线图
```

## kind 参数

| 值        | 图表类型   |
| --------- | ---------- |
| `line`    | 折线图     |
| `bar`     | 柱状图     |
| `barh`    | 水平柱状图 |
| `hist`    | 直方图     |
| `box`     | 箱线图     |
| `area`    | 面积图     |
| `scatter` | 散点图     |
| `pie`     | 饼图       |

## 分组绘图

```python
df.groupby('category')['value'].mean().plot(kind='bar')
```

## 练习

```bash
python Basic/Visualization/04_pandas_viz.py
```
