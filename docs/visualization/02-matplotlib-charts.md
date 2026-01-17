# Matplotlib 常用图表类型

> 对应代码: [02_matplotlib_charts.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Visualization/02_matplotlib_charts.py)

## 柱状图

```python
ax.bar(x, y)       # 垂直
ax.barh(x, y)      # 水平
```

## 散点图

```python
ax.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
```

## 直方图

```python
ax.hist(data, bins=30, edgecolor='black')
```

## 饼图

```python
ax.pie(sizes, labels=labels, autopct='%1.1f%%')
```

## 箱线图

```python
ax.boxplot(data)
```

## 练习

```bash
python Basic/Visualization/02_matplotlib_charts.py
```
