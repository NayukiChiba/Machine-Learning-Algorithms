---
title: SVR 支持向量回归 — 练习与参考文献
outline: deep
---

# 练习与参考文献

> 对应代码：`model_training/regression/svr.py`、`pipelines/regression/svr.py`
>  
> 运行方式：`python -m pipelines.regression.svr`

## 本章目标

1. 用练习把前面各章的源码理解串起来。
2. 通过改参数和观察图像建立更稳定的调参感觉。
3. 给出继续深入学习 SVR 的参考资料与代码入口。

## 自检题

1. 阅读 `model_training/regression/svr.py`，解释 `C` 与 `epsilon` 在当前默认配置下的作用分工。
2. 阅读 `pipelines/regression/svr.py`，说明为什么标准化必须在 `train_test_split(...)` 之后进行。
3. 运行 `python -m pipelines.regression.svr`，观察控制台中的“支持向量数量”，并解释它与模型复杂度的关系。
4. 将 `kernel` 从 `rbf` 改为 `linear`，比较残差图差异并说明原因。
5. 保持 `kernel='rbf'`，将 `epsilon` 改为 `0.3`，比较学习曲线变化并给出结论。

## 实操练习 1：参数敏感性实验

### 目标

只改 `train_model(...)` 入参，观察残差图和学习曲线变化。

### 建议配置

1. `C=1.0, epsilon=0.1, kernel='rbf'`
2. `C=10.0, epsilon=0.1, kernel='rbf'`
3. `C=50.0, epsilon=0.1, kernel='rbf'`

### 记录项

1. 支持向量数量
2. 残差分布是否更集中
3. 学习曲线中训练/验证间隔是否缩小

## 实操练习 2：核函数对比实验

### 目标

比较 `linear` 与 `rbf` 在 Friedman1 数据上的拟合差异。

### 建议步骤

1. 保持其他参数不变，仅切换 `kernel`
2. 分别保存残差图和学习曲线
3. 对比两组图并给出结论

## 实操练习 3：`epsilon` 对平滑度的影响

### 目标

观察 `epsilon` 变化后模型对小误差的容忍度如何变化。

### 建议配置

1. `epsilon=0.05`
2. `epsilon=0.1`
3. `epsilon=0.3`

### 记录项

1. 支持向量数量是否变化
2. 残差图是否更分散或更平滑
3. 学习曲线验证得分是否出现明显变化

## 参考文献

1. Smola, A. J., & Schölkopf, B. (2004). A tutorial on support vector regression.
2. scikit-learn 官方文档：
   1. [SVR API](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
   2. [SVM 回归用户指南](https://scikit-learn.org/stable/modules/svm.html#svm-regression)
   3. [learning_curve API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html)

## 代码参考清单

1. `data_generation/regression.py`（`RegressionData.svr()`）
2. `data_generation/__init__.py`（`svr_data`）
3. `model_training/regression/svr.py`（`train_model(...)`）
4. `pipelines/regression/svr.py`（`run()`）
5. `result_visualization/residual_plot.py`（`plot_residuals(...)`）
6. `result_visualization/learning_curve.py`（`plot_learning_curve(...)`）

## 小结

- 前面的章节解决“理解源码”，本章更强调“亲自修改并观察变化”。
- 对 SVR 来说，最有效的学习方式通常不是死记公式，而是把参数变化和图形诊断联系起来。
