---
title: HMM — 练习与参考文献
outline: deep
---

# 练习与参考文献

> 对应代码：`data_generation/probabilistic.py`、`model_training/probabilistic/hmm.py`、`pipelines/probabilistic/hmm.py`
>  
> 相关对象：`ProbabilisticData.hmm()`、`train_model(...)`

## 本章目标

1. 用练习把前面章节里的数据、模型、训练和评估方法串起来。
2. 引导读者直接修改当前源码中的关键参数，观察 HMM 行为变化。
3. 给出与当前分册强相关的参考资料，便于继续深入。

## 自检题

1. 当前仓库为什么要用手工参数化的离散序列数据来讲 HMM？
2. 在本项目里，`obs`、`state_true`、`lengths` 分别起什么作用？
3. `n_components`、`n_iter`、`tol` 分别在控制什么？
4. 当前流水线为什么能计算隐状态预测准确率，但这并不代表所有真实应用都能直接这样评估？
5. 转移矩阵 `transmat_` 和逐时间步准确率分别更适合回答什么问题？

## 动手练习

### 1. 修改 `n_components`

修改 `model_training/probabilistic/hmm.py` 中的默认参数：

```python
n_components: int = 3
```

观察重点：

- 当隐状态数改成 `2` 或 `4` 后，准确率是否变化。
- 学到的转移矩阵结构是否明显变化。
- 预测隐状态路径是否更容易混淆或被过分细分。

### 2. 修改 `n_iter` 或 `tol`

修改以下默认参数：

```python
n_iter: int = 100
tol: float = 1e-3
```

观察重点：

- 更严格或更宽松的收敛设置是否影响训练结果。
- 控制台训练耗时是否变化。
- 预测隐状态准确率和转移矩阵是否明显变化。

### 3. 修改数据生成的转移矩阵或发射矩阵

修改 `data_generation/probabilistic.py` 中以下参数之一：

```python
hmm_A
hmm_B
```

观察重点：

- 状态更稳定或更容易切换时，学习到的 `transmat_` 会怎样变化。
- 发射矩阵区分度降低时，隐状态解码是否更困难。
- 当前准确率是否会明显下降。

### 4. 修改序列长度 `hmm_n_steps`

修改以下默认参数：

```python
hmm_n_steps: int = 300
```

观察重点：

- 序列更短时，训练得到的转移矩阵是否更不稳定。
- 序列更长时，预测隐状态准确率是否更稳定。
- 由此理解当前实现为什么把整条序列长度单独建模成参数。

### 5. 补一个更多结构输出

在 `pipelines/probabilistic/hmm.py` 中增加发射矩阵或模型得分打印。

观察重点：

- 发射矩阵是否和数据生成时的观测模式大致一致。
- 结构性输出是否比单个准确率更有解释价值。
- 由此区分“路径对得上”和“模型内部参数学得合理”这两件事。

## 阅读建议

1. 先运行一次默认源码，记录准确率和转移矩阵。
2. 每次只改一个参数，例如只改 `n_components` 或只改 `hmm_A`，避免多个变量同时变化。
3. 观察时优先对比三条线索：数据生成参数变化、预测隐状态准确率变化、转移矩阵变化。

## 参考文献

- hmmlearn 文档：`https://hmmlearn.readthedocs.io/`
- Rabiner, *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition*.
- Bishop, *Pattern Recognition and Machine Learning*, Hidden Markov Models.
- Murphy, *Machine Learning: A Probabilistic Perspective*, Hidden Markov Models.
- Jurafsky, Martin, *Speech and Language Processing*, HMM 相关章节。

## 小结

- 这部分练习最重要的目标，不是死记前向、后向和 Viterbi 公式，而是亲手观察状态数、转移结构和序列长度如何影响解码结果。
- 当前源码已经提供了非常透明的离散序列生成过程和直接的控制台结构输出，因此很适合做基础 HMM 实验。
- 把这些练习做完，再回头看数学原理、模型构建和评估章节，理解通常会更扎实。
