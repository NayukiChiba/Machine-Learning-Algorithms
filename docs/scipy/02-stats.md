# 统计分布与描述统计

> 对应代码: [02_stats.py](file:///d:/Nayey/Code/NayukiChiba/Machine-Learning-Algorithms/Basic/Scipy/02_stats.py)

## 概率分布

```python
from scipy import stats

# 正态分布
norm = stats.norm(loc=0, scale=1)
norm.pdf(0)       # 概率密度函数
norm.cdf(0)       # 累积分布函数
norm.ppf(0.95)    # 分位数函数
norm.rvs(size=5)  # 随机样本

# 二项分布
binom = stats.binom(n=10, p=0.5)
binom.pmf(5)      # 概率质量函数

# 泊松分布
poisson = stats.poisson(mu=3)
```

## 描述性统计

```python
import numpy as np
from scipy import stats

data = np.random.normal(100, 15, 100)

np.mean(data)           # 均值
np.median(data)         # 中位数
np.std(data)            # 标准差
stats.skew(data)        # 偏度
stats.kurtosis(data)    # 峰度
np.percentile(data, 75) # 百分位数
```

## 练习

```bash
python Basic/Scipy/02_stats.py
```
