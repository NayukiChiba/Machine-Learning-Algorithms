---
title: HMM — 练习与参考文献
outline: deep
---

# 练习与参考文献

## 本章目标

1. 通过自检问题确认对 HMM 核心概念的理解程度。
2. 通过动手练习在代码层面验证和探索 HMM 的行为。
3. 提供扩展阅读的参考文献入口。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 自检问题 | 诊断 | 确认对 Forward/Viterbi/Baum-Welch、马尔可夫性、HMM vs GMM 等核心概念的理解 |
| 动手练习 | 实践 | 修改参数和模型配置观察 HMM 行为——建立序列模型直觉 |
| 参考文献 | 入口 | 提供 HMM 经典教材和 Rabiner 教程 |

## 1. 自检问题

1. HMM 的三个基本问题是什么？Forward、Viterbi、Baum-Welch 分别解决哪个问题？

2. Forward 算法和 Viterbi 算法都是动态规划——两者的递推公式有何本质区别？一个求和，一个取最大，这反映了什么不同的目标？

3. 为什么 Viterbi 解码优于逐步 $\arg\max_i P(s_t = i \mid O, \lambda)$？给出一个简单例子说明后者可能产生非法状态转移。

4. Baum-Welch 与 GMM 的 EM 在 E 步上的本质区别是什么？为什么 HMM 的 E 步需要 Forward-Backward 而非逐点后验？

5. 马尔可夫性假设 $P(s_t \mid s_1, \dots, s_{t-1}) = P(s_t \mid s_{t-1})$ 在什么实际场景下会被违反？一阶 HMM 如何被扩展来处理更高阶的依赖？

6. 转移矩阵 $A$ 的行和为 1，对角元素通常较大——这反映了隐状态的什么特性？如果对角元素都接近 0.33（3 状态），说明什么？

7. HMM 和 GMM 都是概率生成模型——它们在数据结构、独立性假设、隐变量含义上有哪些根本差异？HMM 可以视为什么结构的概率模型？

## 2. 动手练习

### 练习 1：改变隐状态数 `n_components`

将 `n_components` 分别设为 `2`、`3`、`4`、`5`，观察准确率和转移矩阵的变化。

```python
model = train_model(X_obs, lengths, n_components=2)
```

回答：`n_components=2` 时 HMM 如何将 3 个真实状态"合并"为 2 个？准确率是否显著下降？`n_components=5` 时是否出现了"多余"的状态？

### 练习 2：改变序列长度

修改 `data_generation/probabilistic.py` 中的 `hmm_n_steps`（分别设为 `50`、`100`、`300`、`1000`），观察准确率的变化。

```python
# 在 ProbabilisticData 中
hmm_n_steps: int = 50  # 试试 50, 100, 300, 1000
```

回答：序列越短，Baum-Welch 估计的转移矩阵越不稳定——具体多短时准确率开始显著下降？

### 练习 3：改变转移矩阵的惯性

修改 `hmm_A` 的对角线值（如将 `0.8` 分别改为 `0.5` 和 `0.95`），观察转移矩阵的学习精度。

```python
# 低惯性
hmm_A: list = [[0.50, 0.30, 0.20], [0.30, 0.40, 0.30], [0.20, 0.30, 0.50]]
# 高惯性
hmm_A: list = [[0.95, 0.03, 0.02], [0.03, 0.94, 0.03], [0.03, 0.03, 0.94]]
```

回答：高惯性（状态几乎不跳变）的转移矩阵是否更容易被 HMM 恢复？为什么？

### 练习 4：对比 `predict`（Viterbi）与逐点 argmax

手动实现逐点 argmax 解码（不使用 Viterbi），对比两者的准确率。

```python
# 计算后验概率（需要自己实现 Forward-Backward 或使用 model.score_samples）
# 逐点 argmax: ŝ_t = argmax_i γ_t(i)
```

回答：有没有发现逐点 argmax 产生了"不可能的转移"（状态 0 → 2 等）？哪种方法的准确率更高？

### 练习 5：使用 Forward 得分评估模型质量

比较不同 `n_components` 下训练模型的 `score`（Forward 对数概率）。

```python
for k in [2, 3, 4, 5]:
    model = train_model(X_obs, lengths, n_components=k)
    log_prob = model.score(X_obs, lengths)
    print(f"K={k}: log P(O|λ) = {log_prob:.2f}")
```

回答：对数概率随 $K$ 增大是否单调递增（总是偏好更多参数）？是否可以用 BIC 来平衡拟合和复杂度？

## 3. 参考文献

| 序号 | 文献 | 说明 |
|---|---|---|
| 1 | Rabiner, L. R. (1989). *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition*. Proceedings of the IEEE, 77(2), 257-286. | HMM 最经典的入门教程——Forward/Viterbi/Baum-Welch 的完整推导和语音识别应用 |
| 2 | Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 13. | 教材——HMM 的概率图模型视角和变分推断推广 |
| 3 | hmmlearn 官方文档 — [CategoricalHMM](https://hmmlearn.readthedocs.io/en/latest/api.html#categoricalhmm) | hmmlearn 的 API 参考——参数、方法和使用示例 |
| 4 | Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. Chapter 17. | 教材——HMM 的马尔可夫链理论、卡尔曼滤波推广和状态空间模型 |

## 常见坑

1. 在真实数据（非合成）上期待完美的隐状态准确率——真实数据的 HMM 假设（马尔可夫 + 离散观测）常被违反。
2. 把 HMM 的 `score` 当成"准确率"——`score` 返回对数概率（负值绝对值越小越好），不是匹配比例。
3. 以为 `n_components` 越大越好——过度设定状态数会导致每个状态下观测极少，转移矩阵估计不稳定。
4. 忘记 hmmlearn 的列向量要求——`reshape(-1, 1)` 是必需的数据整形步骤。

## 小结

- 7 个自检问题覆盖 HMM 的核心概念：三个基本问题、Forward vs Viterbi、Viterbi vs 逐点 argmax、Baum-Welch vs EM、马尔可夫性、转移矩阵解读、HMM vs GMM。
- 5 个动手练习从不同角度探索 HMM 的行为——改变状态数、序列长度、转移惯性、对比解码方式、使用 Forward 得分做模型选择。
- 4 篇参考文献覆盖经典入门（Rabiner 1989）、教材和官方文档——构成完整的 HMM 学习路线。
