# 生成模拟数据（generate_data.py）

本模块负责**造数据**，让你在没有真实数据集时也能完整跑通线性回归流程。

---

## 1. 核心目标

- 构造 3 个特征：**面积、房间数、房龄**
- 设置一个“真实线性关系”
- 添加噪声，模拟真实世界数据的不确定性
- 返回 `pandas.DataFrame` 供后续步骤使用

---

## 2. 代码结构

```python
from utils.decorate import print_func_info

@print_func_info
def generate_data(n_samples=200, noise=10, random_state=42) -> DataFrame:
    ...
```

- 使用装饰器 `print_func_info`：调用函数时会打印提示，方便学习与调试。

---

## 3. 关键参数

- `n_samples`：样本数量
- `noise`：噪声强度（越大，数据越“乱”）
- `random_state`：随机种子，保证结果可复现

---

## 4. 生成特征

```python
area = np.random.uniform(low=20, high=80, size=n_samples)
num  = np.random.uniform(low=1,  high=5,  size=n_samples)
age  = np.random.uniform(low=1,  high=20, size=n_samples)
```

解释：
- `np.random.uniform` 生成均匀分布随机数
- 这里的范围模拟现实场景（比如房屋面积 20~80）

---

## 5. 生成目标变量（价格）

核心公式：

$$
\text{价格} = 2\cdot\text{面积} + 10\cdot\text{房间数} - 3\cdot\text{房龄} + \epsilon + 50
$$

```python
price = 2*area + 10*num - 3*age + np.random.normal(0, noise, n_samples) + 50
```

解释：
- **线性关系**：面积和房间数对价格正向影响，房龄负向影响
- **噪声项 $\epsilon$**：让数据更接近真实场景
- **+50**：让价格整体上移（类似“基准价”）

---

## 6. 输出 DataFrame

```python
data = DataFrame({
    "面积": area,
    "房间数": num,
    "房龄": age,
    "价格": price
})
```

输出表格结构：

| 面积 | 房间数 | 房龄 | 价格 |
|------|--------|------|------|
| 45.2 | 3.0    | 12.1 |  ... |

---

## 7. 你可以尝试的修改

1) **改变噪声**
- `noise` 变大，拟合会更困难

2) **改系数**
- 修改 `2*area + 10*num - 3*age`，理解权重的意义

3) **增加非线性项（练习）**
- 例如加入 `area**2` 或 `num*age`

---

## 8. 小结

- 这个模块为你提供了可控的“实验数据”。
- 你知道了“真实关系”，所以更容易判断模型是否学对了。
- 后续的探索、训练、评估全部基于这里的数据。
