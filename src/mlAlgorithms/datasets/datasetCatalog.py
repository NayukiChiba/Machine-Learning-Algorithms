"""
数据集目录构建工具
"""

from __future__ import annotations

from mlAlgorithms.core.datasetSpec import DatasetSpec
from mlAlgorithms.core.taskTypes import DataKind, TaskType
from mlAlgorithms.datasets.sequence.probabilisticDatasets import (
    ProbabilisticDatasetFactory,
)
from mlAlgorithms.datasets.tabular.classificationDatasets import (
    ClassificationDatasetFactory,
)
from mlAlgorithms.datasets.tabular.clusteringDatasets import ClusteringDatasetFactory
from mlAlgorithms.datasets.tabular.dimensionalityDatasets import (
    DimensionalityDatasetFactory,
)
from mlAlgorithms.datasets.tabular.ensembleDatasets import EnsembleDatasetFactory
from mlAlgorithms.datasets.tabular.regressionDatasets import RegressionDatasetFactory


def buildDatasetSpecs() -> list[DatasetSpec]:
    """构建全部数据集规格。"""
    classification = ClassificationDatasetFactory()
    regression = RegressionDatasetFactory()
    clustering = ClusteringDatasetFactory()
    ensemble = EnsembleDatasetFactory()
    dimensionality = DimensionalityDatasetFactory()
    probabilistic = ProbabilisticDatasetFactory()

    return [
        DatasetSpec(
            "classification.logistic_regression",
            TaskType.CLASSIFICATION,
            DataKind.TABULAR,
            classification.loadLogisticRegressionDataset,
            "label",
            None,
            "线性可分高维二分类",
        ),
        DatasetSpec(
            "classification.decision_tree",
            TaskType.CLASSIFICATION,
            DataKind.TABULAR,
            classification.loadDecisionTreeClassificationDataset,
            "label",
            None,
            "blob 多分类",
        ),
        DatasetSpec(
            "classification.svc",
            TaskType.CLASSIFICATION,
            DataKind.TABULAR,
            classification.loadSvcDataset,
            "label",
            None,
            "同心圆二分类",
        ),
        DatasetSpec(
            "classification.naive_bayes",
            TaskType.CLASSIFICATION,
            DataKind.TABULAR,
            classification.loadNaiveBayesDataset,
            "label",
            None,
            "Iris 真实数据",
        ),
        DatasetSpec(
            "classification.knn",
            TaskType.CLASSIFICATION,
            DataKind.TABULAR,
            classification.loadKnnDataset,
            "label",
            None,
            "双月牙二分类",
        ),
        DatasetSpec(
            "classification.random_forest",
            TaskType.CLASSIFICATION,
            DataKind.TABULAR,
            classification.loadRandomForestDataset,
            "label",
            None,
            "高维多分类",
        ),
        DatasetSpec(
            "regression.linear_regression",
            TaskType.REGRESSION,
            DataKind.TABULAR,
            regression.loadLinearRegressionDataset,
            "price",
            None,
            "线性房价数据",
        ),
        DatasetSpec(
            "regression.svr",
            TaskType.REGRESSION,
            DataKind.TABULAR,
            regression.loadSvrDataset,
            "price",
            None,
            "Friedman1 非线性回归",
        ),
        DatasetSpec(
            "regression.decision_tree",
            TaskType.REGRESSION,
            DataKind.TABULAR,
            regression.loadDecisionTreeRegressionDataset,
            "price",
            None,
            "California Housing",
        ),
        DatasetSpec(
            "regression.regularization",
            TaskType.REGRESSION,
            DataKind.TABULAR,
            regression.loadRegularizationDataset,
            "price",
            None,
            "糖尿病共线性回归",
        ),
        DatasetSpec(
            "clustering.kmeans",
            TaskType.CLUSTERING,
            DataKind.TABULAR,
            clustering.loadKmeansDataset,
            "true_label",
            None,
            "球形多簇",
        ),
        DatasetSpec(
            "clustering.dbscan",
            TaskType.CLUSTERING,
            DataKind.TABULAR,
            clustering.loadDbscanDataset,
            "true_label",
            None,
            "双月牙非线性",
        ),
        DatasetSpec(
            "ensemble.bagging",
            TaskType.CLASSIFICATION,
            DataKind.TABULAR,
            ensemble.loadBaggingDataset,
            "label",
            None,
            "高噪声双月牙二分类",
        ),
        DatasetSpec(
            "ensemble.gbdt",
            TaskType.CLASSIFICATION,
            DataKind.TABULAR,
            ensemble.loadGbdtDataset,
            "label",
            None,
            "中等难度多分类",
        ),
        DatasetSpec(
            "ensemble.xgboost",
            TaskType.REGRESSION,
            DataKind.TABULAR,
            ensemble.loadXgboostDataset,
            "price",
            None,
            "XGBoost 回归",
        ),
        DatasetSpec(
            "ensemble.lightgbm",
            TaskType.CLASSIFICATION,
            DataKind.TABULAR,
            ensemble.loadLightgbmDataset,
            "label",
            None,
            "高维四分类",
        ),
        DatasetSpec(
            "dimensionality.pca",
            TaskType.DIMENSIONALITY,
            DataKind.TABULAR,
            dimensionality.loadPcaDataset,
            "label",
            None,
            "高维低秩合成数据",
        ),
        DatasetSpec(
            "dimensionality.lda",
            TaskType.DIMENSIONALITY,
            DataKind.TABULAR,
            dimensionality.loadLdaDataset,
            "label",
            None,
            "Wine 真实数据",
        ),
        DatasetSpec(
            "probabilistic.em",
            TaskType.PROBABILISTIC,
            DataKind.TABULAR,
            probabilistic.loadEmDataset,
            "true_label",
            None,
            "GMM 混合数据",
        ),
        DatasetSpec(
            "probabilistic.hmm",
            TaskType.PROBABILISTIC,
            DataKind.SEQUENCE,
            probabilistic.loadHmmDataset,
            "state_true",
            ["obs"],
            "离散 HMM 序列",
        ),
    ]
