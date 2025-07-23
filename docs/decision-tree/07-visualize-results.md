# 结果可视化（visualize_results.py）

这一模块绘制预测效果、残差分析、特征重要性和树结构。

---

## 1. 输出文件

图像保存到：

```
outputs/DecisionTree/
```

对应文件名：
- `04_prediction_effect.png`
- `05_residual_analysis.png`
- `06_feature_importance.png`
- `07_tree_structure.png`

---

## 2. 预测值 vs 真实值

- 越接近对角线越好
- 可以直观看到整体拟合效果

![04_prediction_effect](images/decision_tree/04_prediction_effect.png)

---

## 3. 残差分析

残差定义：

$$
\text{残差} = y - \hat{y}
$$

理想残差应该：
- 分布在 0 附近
- 无明显结构

![05_residual_analysis](images/decision_tree/05_residual_analysis.png)

---

## 4. 特征重要性

```python
importances = model.feature_importances_
```

- 数值越大，说明该特征在分裂时贡献越多
- 仅代表“模型使用频率”，不等价因果关系

![06_feature_importance](images/decision_tree/06_feature_importance.png)

---

## 5. 决策树结构可视化

```python
plot_tree(
    model,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=4
)
```

建议：
- 树太深时，图会挤在一起
- 可以通过 `max_depth` 只画前几层
- 增大 `figsize` 和 `fontsize` 提高清晰度

![07_tree_structure](images/decision_tree/07_tree_structure.png)

---

## 6. 小结

- 可视化能帮助判断“预测是否合理”
- 特征重要性和树结构更利于解释模型
