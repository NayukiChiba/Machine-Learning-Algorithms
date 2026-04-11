# Machine-Learning-Algorithms

机器学习算法学习与实现仓库：配套可运行代码（数据生成、训练、评估与可视化），原理与推导见 VitePress 文档站点。

**在线文档（GitHub Pages）：** [https://nayukichiba.github.io/Machine-Learning-Algorithms/](https://nayukichiba.github.io/Machine-Learning-Algorithms/)

## 环境

- Python 3.10+（建议；需与当前 `scikit-learn` 版本兼容）
- 安装依赖：

```bash
pip install -r requirements.txt
```

- **XGBoost**：集成学习中的 XGBoost 脚本依赖包 `xgboost`，未写入 `requirements.txt` 时需自行安装：

```bash
pip install xgboost
```

- **文档本地预览**（可选）：进入 `docs` 目录后执行：

```bash
npm ci
npm run docs:dev
```

- 仓库含 `pre-commit` 与 `ruff` 相关配置；若参与开发，可在安装依赖后按需启用 `pre-commit install`。

## 运行代码

在**仓库根目录**下执行（保证当前工作目录为项目根，以便 `data_generation`、`model_training` 等包可被解析）。各流水线模块的 docstring 中一般写有推荐命令，例如 KNN 分类端到端：

```bash
python -m pipelines.classification.knn
```

其它算法可类推，例如 `python -m pipelines.regression.linear_regression`、`python -m pipelines.clustering.kmeans` 等（具体以对应文件内说明为准）。

运行产生的图表等默认写入 `outputs/`（路径由 [`config.py`](config.py) 中的 `OUTPUTS_ROOT` 及各子目录常量约定）。

## 仓库结构

```
Machine-Learning-Algorithms/
├── Basic/                    # 基础库练习：Numpy / Pandas / Scikit-learn / Scipy / Visualization
├── config.py                 # 输出目录等全局配置
├── data_exploration/         # 数据探索（单变量 / 双变量 / 多变量）
├── data_generation/          # 示例或合成数据生成
├── data_visualization/       # 原始数据可视化（分布、散点、相关性、特征空间等）
├── docs/                     # VitePress 文档（数学原理与算法说明，构建后部署到 GitHub Pages）
├── model_evaluation/       # 分类 / 回归 / 聚类 / 降维等指标
├── model_training/           # 按任务划分的训练脚本
│   ├── classification/
│   ├── regression/
│   ├── clustering/
│   ├── ensemble/
│   ├── dimensionality/
│   └── probabilistic/
├── outputs/                  # 运行输出（按模块约定子路径）
├── pipelines/                # 端到端流水线入口（串联数据 → 训练 → 可视化等）
│   ├── classification/
│   ├── regression/
│   ├── clustering/
│   ├── ensemble/
│   ├── dimensionality/
│   └── probabilistic/
├── result_visualization/     # 结果可视化（决策边界、ROC、混淆矩阵、残差、聚类等）
├── utils/                    # 通用工具（上下文管理、装饰器等）
├── requirements.txt
├── LICENSE                   # Apache License 2.0
└── README.md
```

说明：若你希望补充「独立的数据预处理包」等目录，可在后续迭代中新增；当前仓库**不包含**名为 `data_processing/` 的顶层包。

## 算法与主题清单

文档与代码覆盖下列方向（与 `docs` 站点导航一致；集成学习代码中含 **XGBoost**）。

### 分类

- 逻辑回归、决策树、SVC、朴素贝叶斯、KNN、随机森林

### 回归

- 线性回归、Ridge / Lasso（及正则化相关）、决策树回归、SVR

### 聚类

- K-Means、DBSCAN

### 降维

- PCA、LDA

### 集成学习

- Bagging、GBDT、LightGBM、XGBoost

### 概率与序列模型

- EM、HMM

## 许可

本项目以 [Apache License 2.0](LICENSE) 授权。
