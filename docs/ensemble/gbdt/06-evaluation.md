---
title: GBDT 梯度提升树 — 评估与诊断
outline: deep
---

# 评估与诊断

## 本章目标

1. 理解 GBDT 的四项评估输出——混淆矩阵、ROC 曲线、特征重要性、学习曲线，各自回答什么问题。
2. 理解特征重要性的计算原理和解读方法——它是 GBDT 独有的"自动特征选择"工具。
3. 理解学习曲线的诊断价值——判断模型是欠拟合、恰好还是过拟合。
4. 明确当前代码中评估的实现范围。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `plot_confusion_matrix(...)` | 函数 | 在测试集上绘制混淆矩阵——评估多分类硬标签准确率 |
| `plot_roc_curve(...)` | 函数 | 绘制多分类 ROC 曲线（one-vs-rest）——评估概率输出的排序能力 |
| `plot_feature_importance(...)` | 函数 | 绘制特征重要性柱状图——展示 8 个特征对分类的贡献排序 |
| `plot_learning_curve(...)` | 函数 | 绘制学习曲线——训练集/测试集准确率随训练样本数变化 |
| `model.feature_importances_` | 属性 | 8 个特征的重要性值——基于 200 棵树中的分裂增益 |
| `model.train_score_` | 属性 | 每轮迭代的训练得分——可观察损失下降趋势 |

## 1. 混淆矩阵：多分类硬标签评估

混淆矩阵评估的是 `model.predict()` 的输出——200 棵树加权累加后 softmax 取最大的分类结果。

### 参数速览

适用 API：`plot_confusion_matrix(y_test, y_pred, title="GBDT 混淆矩阵", dataset_name=DATASET, model_name=MODEL)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_test` | `Series`，形状 `(100,)` | 测试集真实标签 $\{0, 1, 2\}$ | `y_test` |
| `y_pred` | `ndarray`，形状 `(100,)` | 模型预测结果 | `model.predict(X_test_s)` |
| `title` | `str` | 图表标题 | `"GBDT 混淆矩阵"` |
| `dataset_name` | `str` | 数据集名称——用于输出路径 | `"gbdt"` |
| `model_name` | `str` | 模型名称——用于输出路径 | `"gbdt"` |
| 输出 | PNG 文件 | 3×3 混淆矩阵热力图 | `outputs/gbdt/confusion_matrix.png` |

### 理解重点

- 三分类混淆矩阵是 3×3 的表格——对角线是正确分类的样本，非对角线是错误分类。
- 与 Bagging 的二分类混淆矩阵（2×2）不同——三分类有更多错分组合，诊断信息更丰富。
- 对于 `class_sep=0.7` 的中等难度数据——对角线元素应该明显亮于非对角线元素，但可能不如 Bagging 的高噪声双月牙那样极端。

## 2. ROC 曲线：多分类概率评估

ROC 曲线评估的是 `model.predict_proba()` 的输出——多分类使用 one-vs-rest 策略，为每个类别单独绘制一条 ROC 曲线。

### 参数速览

适用 API：`plot_roc_curve(y_test, y_scores, title="GBDT ROC 曲线", dataset_name=DATASET, model_name=MODEL)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_test` | `Series`，形状 `(100,)` | 测试集真实标签 | `y_test` |
| `y_scores` | `ndarray`，形状 `(100, 3)` | softmax 概率输出——每行三类概率 | `model.predict_proba(X_test_s)` |
| `title` | `str` | 图表标题 | `"GBDT ROC 曲线"` |
| `dataset_name` | `str` | 数据集名称 | `"gbdt"` |
| `model_name` | `str` | 模型名称 | `"gbdt"` |
| 输出 | PNG 文件 | 含 3 条曲线的 ROC 图（每个类别一条 + 微平均/宏平均） | `outputs/gbdt/roc_curve.png` |

### 理解重点

- 多分类 ROC 为每个类别绘制一条曲线——可以看到模型在哪个类别上区分能力最强/最弱。
- 当前流水线直接调用 `predict_proba`，没有条件检查——`GradientBoostingClassifier` 始终支持概率输出。
- 与 Bagging 的二分类 ROC（只有一条曲线）不同——GBDT 的 ROC 图展示三条曲线的对比。

## 3. 特征重要性：谁在真正做决策

特征重要性是 GBDT 独有的诊断工具——基于 200 棵树中每个特征的分裂增益，自动排序特征贡献。

### 参数速览

适用 API：`plot_feature_importance(model, feature_names=feature_names, title="GBDT 特征重要性", dataset_name=DATASET, model_name=MODEL)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `model` | `GradientBoostingClassifier` | 已训练的 GBDT 模型——从中提取 `feature_importances_` | `model` |
| `feature_names` | `list[str]` | 8 个特征的名称列表 | `['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']` |
| `title` | `str` | 图表标题 | `"GBDT 特征重要性"` |
| `dataset_name` | `str` | 数据集名称 | `"gbdt"` |
| `model_name` | `str` | 模型名称 | `"gbdt"` |
| 输出 | PNG 文件 | 水平或垂直柱状图——特征按重要性降序排列 | `outputs/gbdt/feature_importance.png` |

### 理解重点

- 好的模型应让 `x1`~`x4`（有效特征）的重要性显著高于 `x5`~`x8`（冗余和噪声特征）——这是数据设计的"标准答案"。
- 特征重要性来自 200 棵树的累积——不是单棵树的随机判断，比 Bagging 的特征重要性更稳定。
- 特征重要性是 GBDT 的"自动特征选择"——如果某些噪声特征重要性异常高，说明模型可能过拟合。
- Bagging 没有此项评估——因为 Bagging 的基学习器是并行随机采样，特征重要性不如 GBDT 稳定。

## 4. 学习曲线：诊断偏差-方差状态

学习曲线展示模型性能随训练样本数增加的变化——是诊断欠拟合/过拟合的标准工具。

### 参数速览

适用 API：`plot_learning_curve(GradientBoostingClassifier(n_estimators=100, random_state=42), X_train_s, y_train, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `GradientBoostingClassifier` | 新实例化的 GBDT 模型（`n_estimators=100`，注意不同于主模型的 200） | `GradientBoostingClassifier(n_estimators=100, random_state=42)` |
| `X` | `ndarray`，形状 `(400, 8)` | 训练特征 | `X_train_s` |
| `y` | `Series`，形状 `(400,)` | 训练标签 | `y_train` |
| `title` | `str` | 图表标题 | `"GBDT 学习曲线"` |
| `dataset_name` | `str` | 数据集名称 | `"gbdt"` |
| `model_name` | `str` | 模型名称 | `"gbdt"` |
| 输出 | PNG 文件 | 训练集和交叉验证集准确率 vs 训练样本数 | `outputs/gbdt/learning_curve.png` |

### 理解重点

- 学习曲线使用交叉验证——在不同大小的训练子集上评估模型性能。
- 训练集准确率高 + 交叉验证准确率低 = **过拟合**（高方差）——两条曲线之间有大的间隙。
- 训练集准确率低 + 交叉验证准确率也低 = **欠拟合**（高偏差）——两条曲线都低且接近。
- 两条曲线收敛且都高 = **拟合良好**——GBDT 期望达到的状态。
- 注意学习曲线使用的 `n_estimators=100`（不同于主模型的 200）——以减少计算开销。

## 5. 当前代码已实现 vs 未实现的评估内容

### 已实现

| 评估项 | 输出形式 | 触发条件 |
|---|---|---|
| 混淆矩阵 | PNG 热力图（`outputs/gbdt/confusion_matrix.png`） | 始终 |
| ROC 曲线 | PNG 曲线图（`outputs/gbdt/roc_curve.png`） | 始终 |
| 特征重要性 | PNG 柱状图（`outputs/gbdt/feature_importance.png`） | 始终 |
| 学习曲线 | PNG 曲线图（`outputs/gbdt/learning_curve.png`） | 始终 |
| 训练超参数日志 | 终端打印（n_estimators、learning_rate、max_depth、subsample） | `train_model(...)` 调用 |

### 未实现（以及原因）

| 未实现的评估项 | 原因 |
|---|---|
| 准确率/精确率/召回率/F1 硬数字打印 | 教学型代码通过混淆矩阵可视化间接呈现 |
| 每轮迭代的训练损失曲线 | 需要从 `train_score_` 提取——但当前用学习曲线覆盖了类似需求 |
| 早停（early stopping） | 需要额外划分验证集——增加教学复杂度 |
| 与 Bagging 的性能定量对比 | 两个算法的数据和任务不同，直接对比不公平 |
| 特征交互效应分析 | 属于深度分析——超出教学型流水线范围 |
| SHAP 值 / 部分依赖图 | 需要额外依赖（shap 库）——当前保持最小依赖原则 |

### 理解重点

- GBDT 的评估体系比 Bagging 更丰富——四项图表输出覆盖了分类性能（混淆矩阵 + ROC）、特征诊断（特征重要性）和模型健康度（学习曲线）三个维度。
- "未实现"并非"做不到"——教学型流水线选择最具代表性的评估项，保持轻量。

## 6. GBDT vs Bagging 评估对比

| 评估维度 | Bagging | GBDT |
|---|---|---|
| 硬分类评估 | 混淆矩阵（2×2 二分类） | 混淆矩阵（3×3 三分类） |
| 概率评估 | ROC 曲线（1 条，二分类） | ROC 曲线（每类 1 条 + 平均，三分类） |
| 特征诊断 | 无 | **特征重要性**——基于分裂增益 |
| 模型健康度 | 无 | **学习曲线**——训练/验证准确率变化 |
| 训练内诊断 | **OOB 得分**——Bagging 独有 | `train_score_`——每轮迭代训练得分 |
| 基学习器诊断 | `estimators_`——80 棵分类树 | `estimators_`——600 棵回归树 |
| 评估项数量 | 3（混淆矩阵 + ROC + OOB） | 4（混淆矩阵 + ROC + 特征重要性 + 学习曲线） |

### 理解重点

- Bagging 有 OOB（免费泛化估计）但无特征重要性——因为并行随机采样的特征重要性不够稳定。
- GBDT 有特征重要性和学习曲线但无 OOB——因为串行训练没有"天然留出"的样本。
- 两者是互补的评估体系——各有各的优势诊断工具。

## 常见坑

1. 只看特征重要性柱状图的高度就下结论——高度受特征尺度影响，标准化后可比性更好。
2. 忽略学习曲线中两条曲线之间的间隙——间隙越大，过拟合越严重。
3. 把特征重要性当成因果关系——高重要性只说明"该特征被用于分裂很多次"，不说明因果关系。
4. 忽略 `n_estimators=100`（学习曲线）与 `n_estimators=200`（主模型）的差异——学习曲线的 GBDT 配置与主模型不同。

## 小结

- GBDT 有四项评估输出：混淆矩阵（多分类硬标签）→ ROC 曲线（one-vs-rest 软概率）→ 特征重要性（自动特征选择）→ 学习曲线（过拟合/欠拟合诊断）——四者从分类性能、特征贡献、模型健康度三个维度完整描述模型质量。
- 特征重要性是 GBDT 独有的诊断优势——基于 200 棵树的分裂增益累积，比 Bagging 的并行随机采样更稳定。
- 学习曲线是偏差-方差状态的标准诊断工具——两条曲线（训练集/验证集）的位置和间隙直接反映欠拟合或过拟合。
