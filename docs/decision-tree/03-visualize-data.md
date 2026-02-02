# 数据可视化（visualize_data.py）

这一模块绘制特征分布、相关性热力图和特征与价格的散点关系。

---

## 1. 输出目录

图像保存到：

```
outputs/DecisionTree/
```

对应文件名：
- `01_feature_distribution.png`
- `02_correlation_heatmap.png`
- `03_feature_relationship.png`

---

## 2. 特征分布图

```python
axes[row, col].hist(data[feature], bins=30, color='skyblue')
```

作用：
- 查看每个特征的取值范围和分布
- 判断是否存在偏态或异常值

![01_feature_distribution](images/decision_tree/01_feature_distribution.png)

---

## 3. 相关性热力图

```python
corr = data.corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
```

作用：
- 观察特征之间以及与 `price` 的线性相关程度

![02_correlation_heatmap](images/decision_tree/02_correlation_heatmap.png)

---

## 4. 特征 vs 价格散点图

```python
selected = ["MedInc", "AveRooms", "HouseAge", "Latitude"]
```

只绘制几个代表性特征，避免图像过于拥挤。

![03_feature_relationship](images/decision_tree/03_feature_relationship.png)

---

## 5. 中文字体设置

```python
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
```

如果你的电脑没有这些字体，可以换成其他中文字体。

---

## 6. 小结

- 可视化帮助判断数据分布是否合理
- 决策树可以处理非线性关系，但可视化仍然很有价值
