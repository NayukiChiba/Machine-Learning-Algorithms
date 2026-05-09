---
title: EM 与 GMM — 练习与参考文献
outline: deep
---

# 练习与参考文献

## 本章目标

1. 通过自检问题确认对 EM 算法核心概念的理解程度。
2. 通过动手练习在代码层面验证和探索 GMM 的行为。
3. 提供扩展阅读的参考文献入口。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 自检问题 | 诊断 | 确认对 E 步/M 步、软赋值、协方差类型、EM vs KMeans 等核心概念的理解 |
| 动手练习 | 实践 | 修改超参数观察 GMM 行为变化——建立参数-效果的直觉 |
| 参考文献 | 入口 | 提供 EM 算法原始论文、教材章节和 scikit-learn 官方文档 |

## 1. 自检问题

1. EM 算法的 E 步和 M 步分别完成什么任务？为什么说 EM 的每次迭代都保证了数据对数似然单调不减？

2. GMM 的软赋值（后验责任 $\gamma_{ik}$）与 KMeans 的硬赋值（最近质心归属）有何本质区别？在什么场景下软赋值的优势最为明显？

3. `covariance_type` 的四个选项（`full`、`tied`、`diag`、`spherical`）分别对应什么约束？为什么当前数据使用 `full` 而非 `spherical`？

4. `lower_bound_` 的含义是什么？为什么它始终为负数？如果连续两次训练的 `lower_bound_` 差异很大，可能是什么原因？

5. GMM 的混合权重 $\pi_k$、均值 $\boldsymbol{\mu}_k$、协方差 $\boldsymbol{\Sigma}_k$ 三者在 M 步的更新公式是什么？为什么每个都使用了责任加权？

6. 如果 `n_components` 被设为不等于 3 的值（如 2 或 5），GMM 的聚类结果会怎样？如何用 BIC 来辅助选择最优 $K$？

7. EM 和 KMeans 在初始化、参数更新方式、收敛条件上有哪些本质差异？KMeans 可以视为 GMM 的哪种特殊情况？

## 2. 动手练习

### 练习 1：改变协方差类型

将 `covariance_type` 分别设为 `"full"`、`"tied"`、`"diag"`、`"spherical"`，观察聚类分布图的变化。

```python
model = train_model(X_scaled, covariance_type="spherical")
```

回答：`spherical` 下 GMM 的椭圆建模能力完全丢失——聚类分布图是否退化为与 KMeans 类似？`tied`（所有分量共享协方差）的簇形状与 `full` 有何差异？

### 练习 2：错误设定分量数 `n_components`

将 `n_components` 分别设为 `2`、`3`、`4`、`5`、`7`，观察聚类分布图和 `lower_bound_` 的变化。

```python
model = train_model(X_scaled, n_components=2)
```

回答：`n_components=2` 时 EM 如何将 3 个真实分量"合并"为 2 个簇？`n_components=7` 时是否出现了多余的空分量？`lower_bound_` 随 $K$ 增大是否单调递增？

### 练习 3：改变随机种子观察局部最优

将 `random_state` 分别设为 `0`、`1`、`42`、`99`、`123`，观察聚类分布图的变化。

```python
model = train_model(X_scaled, random_state=0)
```

回答：不同的随机种子是否导致显著不同的聚类结果？哪些种子下 EM 收敛到了"不好的"局部最优？如何通过增加 `n_init` 来缓解此问题？

### 练习 4：分析软归属

提取 `predict_proba` 的输出，找到不确定性最高的样本。

```python
probas = model.predict_proba(X_scaled)
uncertainty = 1 - probas.max(axis=1)
top_uncertain = np.argsort(uncertainty)[-10:]  # 最不确定的 10 个点
```

回答：高不确定性样本在二维空间中的位置在哪？它们是否位于两个真实分量之间的"重叠带"？

### 练习 5：对比 GMM 与 KMeans 的聚类差异

在相同数据上分别训练 KMeans（`KMeans(n_clusters=3)`）和 GMM（`GaussianMixture(n_components=3, covariance_type="full")`），对比聚类分布图。

```python
from sklearn.cluster import KMeans

model_km = KMeans(n_clusters=3, random_state=42)
model_km.fit(X_scaled)
labels_km = model_km.predict(X_scaled)
```

回答：在非球形分量（如分量 1 的 $x_1$ 标准差 0.8 vs $x_2$ 标准差 0.5）的区域，KMeans 的硬边界是否显得"不合理"？GMM 是否成功捕捉了椭圆的各向异性？

## 3. 参考文献

| 序号 | 文献 | 说明 |
|---|---|---|
| 1 | Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). *Maximum Likelihood from Incomplete Data via the EM Algorithm*. Journal of the Royal Statistical Society, Series B, 39(1), 1-38. | EM 算法的原始论文——E 步/M 步形式化和收敛性证明 |
| 2 | Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 9. | 标准教材——GMM 和 EM 算法的完整推导和实例 |
| 3 | scikit-learn 官方文档 — [GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) | API 参考——全部参数、属性和方法的详细说明 |
| 4 | Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. Chapter 11. | 概率视角教材——EM 算法作为变分推断的特例，含 BIC/AIC 模型选择 |

## 常见坑

1. 把 `n_components` 设得过大——过多的分量会导致某些分量的协方差矩阵退化为奇异矩阵（`reg_covar` 是最后防线）。
2. 在不做标准化的数据上跑 EM——不同特征尺度会导致协方差矩阵被大尺度特征主导。
3. 认为 `lower_bound_` 可以跨数据集比较——对数似然依赖于数据规模和维度，不同数据集间的值不可直接对比。
4. 忽略 EM 的局部最优特性——单次训练的结果可能只是一个局部最优解，`n_init > 1` 可降低此风险。

## 小结

- 7 个自检问题覆盖 EM 算法的核心概念：E 步/M 步、软赋值、协方差类型、`lower_bound_`、参数更新公式、$K$ 选择、与 KMeans 对比。
- 5 个动手练习从不同角度探索 GMM 的行为——改变协方差类型、错误设定分量数、测试随机种子、分析软归属、对比 KMeans。
- 4 篇参考文献从原始论文（Dempster et al. 1977）→ 教材（Bishop + Murphy）→ 官方文档构成完整的阅读路线。
