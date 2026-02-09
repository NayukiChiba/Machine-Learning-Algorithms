# 数据可视化（visualize_data.py）

本模块绘制类别分布、散点图与相关性热力图，帮助你直观看到数据结构。

---

## 1. 输出目录

图像保存到：

```
outputs/SVM/
```

对应文件名：

- `01_class_distribution.png`
- `02_data_scatter.png`
- `03_correlation_heatmap.png`

---

## 2. 类别分布柱状图

```python
class_count = data["label"].value_counts().sort_index()
plt.bar(class_count.index.astype(str), class_count.values, color=["steelblue", "coral"])
```

作用：

- 检查类别是否平衡
- 直观了解样本数量

---

## 3. 双月牙散点图

```python
for label, color in zip([0, 1], ["steelblue", "coral"]):
    part = data[data["label"] == label]
    plt.scatter(part["x1"], part["x2"], ...)
```

作用：

- 观察两个类别的空间分布
- 判断是否需要非线性模型

---

## 4. 相关性热力图

```python
corr = data.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
```

作用：

- 查看特征之间的线性关系
- 理解特征与标签的简单相关性

---

## 5. 小结

- 可视化可以帮助判断数据结构是否适合 SVM
- 双月牙数据明显非线性，非常适合 RBF 核展示
