# 交互式可视化

> 对应代码: [09_interactive.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Visualization/09_interactive.py)

## Plotly Express

```python
import plotly.express as px

# 散点图
fig = px.scatter(df, x='x', y='y', color='category')

# 折线图
fig = px.line(df, x='date', y='value')

# 柱状图
fig = px.bar(df, x='category', y='value')

# 3D 散点图
fig = px.scatter_3d(df, x='x', y='y', z='z')

fig.show()
```

## 保存图表

```python
fig.write_html('chart.html')
fig.write_image('chart.png')
```

## 自定义布局

```python
fig.update_layout(
    title='Title',
    xaxis_title='X',
    yaxis_title='Y',
    template='plotly_dark'
)
```

## 练习

```bash
python Basic/Visualization/09_interactive.py
```
