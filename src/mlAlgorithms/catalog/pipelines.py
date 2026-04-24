"""
流水线注册表
"""

from __future__ import annotations


from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

try:
    from lightgbm import LGBMClassifier
except Exception:  # noqa: BLE001
    LGBMClassifier = None

from mlAlgorithms.core.pipelineSpec import PipelineSpec
from mlAlgorithms.core.registry import Registry
from mlAlgorithms.core.taskTypes import RunnerType, TaskType
from mlAlgorithms.training.classification.classificationModels import (
    trainBaggingClassifier,
    trainDecisionTreeClassifier,
    trainGbdtClassifier,
    trainKnnClassifier,
    trainLightgbmClassifier,
    trainLogisticRegression,
    trainNaiveBayesClassifier,
    trainRandomForestClassifier,
    trainSvcClassifier,
)
from mlAlgorithms.training.clustering.clusteringModels import (
    trainDbscanModel,
    trainKmeansModel,
)
from mlAlgorithms.training.dimensionality.dimensionalityModels import (
    trainLdaModel,
    trainPcaModel,
)
from mlAlgorithms.training.probabilistic.probabilisticModels import (
    trainGaussianMixtureModel,
    trainHmmModel,
)
from mlAlgorithms.training.regression.regressionModels import (
    trainDecisionTreeRegressionModel,
    trainLinearRegressionModel,
    trainRegularizationModels,
    trainSvrRegressionModel,
    trainXgboostRegressionModel,
)


def _buildVisualModelFactory(pipelineId: str):
    """构建二维展示用模型工厂。"""
    mapping = {
        "classification.logistic_regression": lambda: LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "classification.decision_tree": lambda: DecisionTreeClassifier(
            max_depth=6, min_samples_split=4, min_samples_leaf=2, random_state=42
        ),
        "classification.svc": lambda: SVC(
            kernel="rbf", gamma="scale", probability=True, random_state=42
        ),
        "classification.naive_bayes": lambda: GaussianNB(),
        "classification.knn": lambda: KNeighborsClassifier(n_neighbors=5),
        "classification.random_forest": lambda: RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=1
        ),
        "ensemble.bagging": lambda: BaggingClassifier(
            DecisionTreeClassifier(random_state=42), n_estimators=30, random_state=42
        ),
        "ensemble.gbdt": lambda: GradientBoostingClassifier(random_state=42),
    }
    if LGBMClassifier is not None:
        mapping["ensemble.lightgbm"] = lambda: LGBMClassifier(
            n_estimators=80, random_state=42, verbosity=-1
        )
    return mapping.get(pipelineId)


def _buildLearningCurveFactory(pipelineId: str):
    """构建学习曲线模型工厂。"""
    mapping = {
        "classification.logistic_regression": lambda: LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "classification.decision_tree": lambda: DecisionTreeClassifier(
            max_depth=6, min_samples_split=4, min_samples_leaf=2, random_state=42
        ),
        "classification.svc": lambda: SVC(
            kernel="rbf", gamma="scale", probability=True, random_state=42
        ),
        "ensemble.bagging": lambda: BaggingClassifier(
            DecisionTreeClassifier(random_state=42), n_estimators=30, random_state=42
        ),
        "ensemble.gbdt": lambda: GradientBoostingClassifier(random_state=42),
        "ensemble.lightgbm": (
            lambda: LGBMClassifier(n_estimators=80, random_state=42, verbosity=-1)
        )
        if LGBMClassifier is not None
        else None,
        "regression.linear_regression": lambda: LinearRegression(),
        "regression.svr": lambda: SVR(C=10.0, epsilon=0.1, kernel="rbf", gamma="scale"),
        "regression.decision_tree": lambda: DecisionTreeRegressor(
            max_depth=6, min_samples_split=6, min_samples_leaf=3, random_state=42
        ),
    }
    return mapping.get(pipelineId)


PIPELINE_REGISTRY = Registry()

for pipelineSpec in [
    PipelineSpec(
        "classification.logistic_regression",
        TaskType.CLASSIFICATION,
        "classification.logistic_regression",
        RunnerType.CLASSIFICATION,
        trainLogisticRegression,
        "standardScaler",
        "stratifiedSplit",
        "default",
        "classification",
        "classification",
        ["classDistribution", "correlationHeatmap", "featureSpace2d"],
        [
            "confusionMatrix",
            "rocCurve",
            "featureImportance",
            "classificationResult",
            "decisionBoundary",
        ],
        ["learningCurve"],
        "logistic_regression",
        metadata={
            "visualModelFactory": _buildVisualModelFactory(
                "classification.logistic_regression"
            ),
            "learningCurveEstimatorFactory": _buildLearningCurveFactory(
                "classification.logistic_regression"
            ),
        },
    ),
    PipelineSpec(
        "classification.decision_tree",
        TaskType.CLASSIFICATION,
        "classification.decision_tree",
        RunnerType.CLASSIFICATION,
        trainDecisionTreeClassifier,
        None,
        "stratifiedSplit",
        "default",
        "classification",
        "classification",
        ["classDistribution", "labeledScatter2d", "correlationHeatmap"],
        ["confusionMatrix", "rocCurve", "featureImportance", "decisionBoundary"],
        ["learningCurve", "treeStructure"],
        "decision_tree",
        metadata={
            "visualModelFactory": _buildVisualModelFactory(
                "classification.decision_tree"
            ),
            "learningCurveEstimatorFactory": _buildLearningCurveFactory(
                "classification.decision_tree"
            ),
        },
    ),
    PipelineSpec(
        "classification.svc",
        TaskType.CLASSIFICATION,
        "classification.svc",
        RunnerType.CLASSIFICATION,
        trainSvcClassifier,
        "standardScaler",
        "stratifiedSplit",
        "default",
        "classification",
        "classification",
        ["classDistribution", "labeledScatter2d", "correlationHeatmap"],
        ["confusionMatrix", "rocCurve", "classificationResult", "decisionBoundary"],
        ["learningCurve"],
        "svc",
        metadata={
            "visualModelFactory": _buildVisualModelFactory("classification.svc"),
            "learningCurveEstimatorFactory": _buildLearningCurveFactory(
                "classification.svc"
            ),
        },
    ),
    PipelineSpec(
        "classification.naive_bayes",
        TaskType.CLASSIFICATION,
        "classification.naive_bayes",
        RunnerType.CLASSIFICATION,
        trainNaiveBayesClassifier,
        "standardScaler",
        "stratifiedSplit",
        "default",
        "classification",
        "classification",
        ["classDistribution", "featureSpace2d", "correlationHeatmap"],
        ["confusionMatrix", "rocCurve", "classificationResult", "decisionBoundary"],
        [],
        "naive_bayes",
        metadata={
            "visualModelFactory": _buildVisualModelFactory("classification.naive_bayes")
        },
    ),
    PipelineSpec(
        "classification.knn",
        TaskType.CLASSIFICATION,
        "classification.knn",
        RunnerType.CLASSIFICATION,
        trainKnnClassifier,
        "standardScaler",
        "stratifiedSplit",
        "default",
        "classification",
        "classification",
        ["classDistribution", "labeledScatter2d", "correlationHeatmap"],
        ["confusionMatrix", "rocCurve", "classificationResult", "decisionBoundary"],
        [],
        "knn",
        metadata={"visualModelFactory": _buildVisualModelFactory("classification.knn")},
    ),
    PipelineSpec(
        "classification.random_forest",
        TaskType.CLASSIFICATION,
        "classification.random_forest",
        RunnerType.CLASSIFICATION,
        trainRandomForestClassifier,
        None,
        "stratifiedSplit",
        "default",
        "classification",
        "classification",
        ["classDistribution", "featureSpace2d", "correlationHeatmap"],
        [
            "confusionMatrix",
            "rocCurve",
            "featureImportance",
            "classificationResult",
            "decisionBoundary",
        ],
        [],
        "random_forest",
        metadata={
            "visualModelFactory": _buildVisualModelFactory(
                "classification.random_forest"
            )
        },
    ),
    PipelineSpec(
        "regression.linear_regression",
        TaskType.REGRESSION,
        "regression.linear_regression",
        RunnerType.REGRESSION,
        trainLinearRegressionModel,
        None,
        "randomSplit",
        "default",
        "regression",
        "regression",
        ["correlationHeatmap", "featureTargetScatter"],
        ["featureImportance"],
        ["learningCurve"],
        "linear_regression",
        metadata={
            "learningCurveEstimatorFactory": _buildLearningCurveFactory(
                "regression.linear_regression"
            )
        },
    ),
    PipelineSpec(
        "regression.svr",
        TaskType.REGRESSION,
        "regression.svr",
        RunnerType.REGRESSION,
        trainSvrRegressionModel,
        "standardScaler",
        "randomSplit",
        "default",
        "regression",
        "regression",
        ["correlationHeatmap", "featureTargetScatter"],
        [],
        ["learningCurve"],
        "svr",
        metadata={
            "learningCurveEstimatorFactory": _buildLearningCurveFactory(
                "regression.svr"
            )
        },
    ),
    PipelineSpec(
        "regression.decision_tree",
        TaskType.REGRESSION,
        "regression.decision_tree",
        RunnerType.REGRESSION,
        trainDecisionTreeRegressionModel,
        None,
        "randomSplit",
        "default",
        "regression",
        "regression",
        ["correlationHeatmap", "featureTargetScatter"],
        ["featureImportance"],
        ["learningCurve", "treeStructure"],
        "decision_tree_regression",
        metadata={
            "learningCurveEstimatorFactory": _buildLearningCurveFactory(
                "regression.decision_tree"
            )
        },
    ),
    PipelineSpec(
        "regression.regularization",
        TaskType.REGRESSION,
        "regression.regularization",
        RunnerType.REGRESSION,
        trainRegularizationModels,
        "standardScaler",
        "randomSplit",
        "default",
        "regression",
        "regression",
        ["correlationHeatmap", "featureTargetScatter"],
        ["featureImportance"],
        [],
        "ridge",
        metadata={"multiModel": True},
    ),
    PipelineSpec(
        "clustering.kmeans",
        TaskType.CLUSTERING,
        "clustering.kmeans",
        RunnerType.CLUSTERING,
        trainKmeansModel,
        "standardScaler",
        None,
        "default",
        "clustering",
        "clustering",
        ["classDistribution", "rawScatter2d", "labeledScatter2d", "correlationHeatmap"],
        [],
        ["kmeansSweep"],
        "kmeans",
    ),
    PipelineSpec(
        "clustering.dbscan",
        TaskType.CLUSTERING,
        "clustering.dbscan",
        RunnerType.CLUSTERING,
        trainDbscanModel,
        "standardScaler",
        None,
        "default",
        "clustering",
        "clustering",
        ["classDistribution", "rawScatter2d", "labeledScatter2d", "correlationHeatmap"],
        [],
        ["dbscanKDistance", "dbscanEpsSweep"],
        "dbscan",
    ),
    PipelineSpec(
        "ensemble.bagging",
        TaskType.CLASSIFICATION,
        "ensemble.bagging",
        RunnerType.CLASSIFICATION,
        trainBaggingClassifier,
        "standardScaler",
        "stratifiedSplit",
        "default",
        "classification",
        "classification",
        ["classDistribution", "labeledScatter2d", "correlationHeatmap"],
        ["confusionMatrix", "rocCurve", "classificationResult", "decisionBoundary"],
        ["learningCurve"],
        "bagging",
        metadata={
            "visualModelFactory": _buildVisualModelFactory("ensemble.bagging"),
            "learningCurveEstimatorFactory": _buildLearningCurveFactory(
                "ensemble.bagging"
            ),
        },
    ),
    PipelineSpec(
        "ensemble.gbdt",
        TaskType.CLASSIFICATION,
        "ensemble.gbdt",
        RunnerType.CLASSIFICATION,
        trainGbdtClassifier,
        "standardScaler",
        "stratifiedSplit",
        "default",
        "classification",
        "classification",
        ["classDistribution", "featureSpace2d", "correlationHeatmap"],
        [
            "confusionMatrix",
            "rocCurve",
            "featureImportance",
            "classificationResult",
            "decisionBoundary",
        ],
        ["learningCurve"],
        "gbdt",
        metadata={
            "visualModelFactory": _buildVisualModelFactory("ensemble.gbdt"),
            "learningCurveEstimatorFactory": _buildLearningCurveFactory(
                "ensemble.gbdt"
            ),
        },
    ),
    PipelineSpec(
        "ensemble.xgboost",
        TaskType.REGRESSION,
        "ensemble.xgboost",
        RunnerType.REGRESSION,
        trainXgboostRegressionModel,
        None,
        "randomSplit",
        "default",
        "regression",
        "regression",
        ["correlationHeatmap", "featureTargetScatter"],
        ["featureImportance"],
        [],
        "xgboost",
        optionalDependencies=("xgboost",),
    ),
    PipelineSpec(
        "ensemble.lightgbm",
        TaskType.CLASSIFICATION,
        "ensemble.lightgbm",
        RunnerType.CLASSIFICATION,
        trainLightgbmClassifier,
        "standardScaler",
        "stratifiedSplit",
        "default",
        "classification",
        "classification",
        ["classDistribution", "featureSpace2d", "correlationHeatmap"],
        [
            "confusionMatrix",
            "rocCurve",
            "featureImportance",
            "classificationResult",
            "decisionBoundary",
        ],
        ["learningCurve"],
        "lightgbm",
        optionalDependencies=("lightgbm",),
        metadata={
            "visualModelFactory": _buildVisualModelFactory("ensemble.lightgbm"),
            "learningCurveEstimatorFactory": _buildLearningCurveFactory(
                "ensemble.lightgbm"
            ),
        },
    ),
    PipelineSpec(
        "dimensionality.pca",
        TaskType.DIMENSIONALITY,
        "dimensionality.pca",
        RunnerType.DIMENSIONALITY,
        trainPcaModel,
        "standardScaler",
        None,
        "transformOnly",
        "dimensionality",
        "classification",
        ["classDistribution", "correlationHeatmap", "featureSpace2d", "featureSpace3d"],
        [],
        ["pcaTrainingCurve"],
        "pca",
    ),
    PipelineSpec(
        "dimensionality.lda",
        TaskType.DIMENSIONALITY,
        "dimensionality.lda",
        RunnerType.DIMENSIONALITY,
        trainLdaModel,
        "standardScaler",
        "stratifiedSplit",
        "ldaClassifier",
        "classification",
        "classification",
        ["classDistribution", "correlationHeatmap", "featureSpace2d"],
        ["confusionMatrix", "rocCurve", "classificationResult"],
        ["learningCurve"],
        "lda",
    ),
    PipelineSpec(
        "probabilistic.em",
        TaskType.PROBABILISTIC,
        "probabilistic.em",
        RunnerType.PROBABILISTIC,
        trainGaussianMixtureModel,
        "standardScaler",
        None,
        "gmmPredictor",
        "clustering",
        "clustering",
        ["classDistribution", "rawScatter2d", "labeledScatter2d", "correlationHeatmap"],
        [],
        ["gmmComponentSweep"],
        "gmm",
    ),
    PipelineSpec(
        "probabilistic.hmm",
        TaskType.PROBABILISTIC,
        "probabilistic.hmm",
        RunnerType.PROBABILISTIC,
        trainHmmModel,
        None,
        None,
        "hmmPredictor",
        "sequence",
        "sequence",
        [],
        [],
        [],
        "hmm",
        optionalDependencies=("hmmlearn",),
    ),
]:
    PIPELINE_REGISTRY.register(pipelineSpec.id, pipelineSpec)
