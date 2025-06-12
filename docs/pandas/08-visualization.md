# Pandas 数据可视化

> 对应代码: [08_visualization.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Pandas/08_visualization.py)

## 学习目标

- 掌握 df.plot() 基本绑图
- 了解不同图表类型
- 学会自定义图表样式

## 基本绑图

```python
df.plot()                    # 默认折线图
df.plot(kind='bar')          # 柱状图
df.plot(kind='scatter', x='A', y='B')  # 散点图
```

## 图表类型

| kind 参数 | 图表类型   |
| --------- | ---------- |
| `line`    | 折线图     |
| `bar`     | 柱状图     |
| `barh`    | 水平柱状图 |
| `hist`    | 直方图     |
| `box`     | 箱线图     |
| `scatter` | 散点图     |
| `pie`     | 饼图       |

## 常用参数

```python
df.plot(
    figsize=(10, 6),    # 图表大小
    title='Title',      # 标题
    xlabel='X Label',   # X轴标签
    ylabel='Y Label',   # Y轴标签
    legend=True,        # 显示图例
    grid=True           # 显示网格
)
```

## 多子图

```python
df.plot(subplots=True, layout=(2, 2))
```

## 练习

```bash
python Basic/Pandas/08_visualization.py
```
