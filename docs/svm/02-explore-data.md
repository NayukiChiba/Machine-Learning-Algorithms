# 数据探索（explore_data.py）

该模块用于快速了解数据的结构、统计特征以及类别分布。

---

## 1. 探索内容

- 样本数量与特征数量
- 描述性统计信息
- 缺失值检查
- 类别分布占比

---

## 2. 核心函数

```python
@print_func_info
def explore_data(data: DataFrame):
    ...
```

---

## 3. 输出示例

```text
样本数量: 400
特征数量: 2
特征名称: ['x1', 'x2']
```

---

## 4. 类别分布

```python
class_count = data["label"].value_counts().sort_index()
class_ratio = (class_count / len(data)).round(3)
```

- 检查是否存在类别不平衡
- 为后续模型评估提供背景信息

---

## 5. 小结

- 这个模块不改变数据，只输出统计信息
- 如果发现缺失值或异常分布，应先处理再训练
