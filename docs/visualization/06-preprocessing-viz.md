# 数据预处理可视化

> 对应代码: [06_preprocessing_viz.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Visualization/06_preprocessing_viz.py)

## 缺失值可视化

```python
# 缺失值热力图
sns.heatmap(df.isnull(), cbar=True, cmap='YlOrRd')

# 缺失比例
(df.isnull().mean() * 100).plot(kind='bar')
```

## 异常值可视化

```python
# 箱线图检测
ax.boxplot(data)

# IQR 边界
q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
```

## 特征变换

```python
# 对数变换
np.log1p(data)

# 标准化
(data - data.mean()) / data.std()
```

## 练习

```bash
python Basic/Visualization/06_preprocessing_viz.py
```
