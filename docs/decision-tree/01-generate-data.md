# 加载数据集（generate_data.py）

本模块负责加载 **California Housing** 经典房价数据集，并返回 `DataFrame`。

---

## 1. 数据集简介

California Housing 是经典的回归数据集，特点：

- 特征全部是数值型
- 目标变量是房价中位数
- 适合回归模型学习

常见特征包括：

- `MedInc`：居民收入中位数
- `HouseAge`：房屋年龄中位数
- `AveRooms`：平均房间数
- `AveBedrms`：平均卧室数
- `Population`：区域人口
- `AveOccup`：平均居住人数
- `Latitude` / `Longitude`：地理位置

---

## 2. 代码结构

```python
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame
```

- `as_frame=True` 会直接返回 `pandas.DataFrame`
- 第一次加载可能需要联网下载数据

---

## 3. 目标变量改名

原始数据集中，目标列名为 `MedHouseVal`（单位 10 万美元）。

为了统一项目风格，代码改名为：

```python
df = df.rename(columns={"MedHouseVal": "price"})
```

---

## 4. 返回结果

```python
return df
```

输出的 `DataFrame` 将包含：
- 8 个特征列
- 1 个目标列 `price`

---

## 5. 小结

- 本模块只负责“加载 + 统一列名”
- 数据的清洗和划分在后续模块完成
