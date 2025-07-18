# 数据可视化（visualize_data.py）

这一模块把数据的分布和关系“画出来”，让你直观看到线性趋势。

---

## 1. 为什么要可视化

- 直方图看分布（是否集中 / 偏斜 / 是否异常）
- 散点图看关系（是否近似线性）
- 相关性热力图看整体相关强弱

---

## 2. 输出位置

代码中通过 `config.py` 获取输出目录：

```python
from config import OUTPUTS_ROOT
LR_OUTPUTS = os.path.join(OUTPUTS_ROOT, "LinearRegression")
```

所有图像都会保存到：

```
outputs/LinearRegression/
```

---

## 3. 图 1：特征分布直方图

```python
axes[row, col].hist(data[feature], bins=30, color='skyblue')
```

作用：
- 判断特征范围是否集中
- 观察是否存在极端值

![01_data_distribution](https://img.yumeko.site/file/articles/LinearRegression/01_data_distribution.png)

---

## 4. 图 2：相关性热力图

```python
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
```

作用：
- 查看特征与目标之间的线性相关
- 相关系数越接近 1 或 -1，线性关系越明显

![02_correlation_heatmap](https://img.yumeko.site/file/articles/LinearRegression/02_correlation_heatmap.png)

---

## 5. 图 3：特征 vs 目标散点图

```python
axes[i].scatter(data[feature], data["价格"], alpha=0.6)
```

作用：
- 直接观察线性趋势
- 如果散点大致呈一条“斜线”，线性回归效果通常不错

![03_feature_relationship](https://img.yumeko.site/file/articles/LinearRegression/03_Feature_Relationship.png)

---

## 6. 中文字体设置

为了避免中文乱码，代码设置了字体：

```python
plt.rcParams["font.sans-serif"] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
```

如果你的电脑没有这些字体，可以替换为其他中文字体。

---

## 7. 小结

- 可视化帮你判断：线性回归是否适合当前数据
- 这一步不改变数据，只提供理解与判断
- 输出图像会被后续报告或笔记使用
