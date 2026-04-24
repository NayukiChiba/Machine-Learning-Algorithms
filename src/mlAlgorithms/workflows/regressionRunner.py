"""
回归工作流
"""

from __future__ import annotations

from config import resolveOutputDir
from mlAlgorithms.evaluation.regressionEvaluator import evaluateRegression
from mlAlgorithms.visualization.data.dataPlots import plotCorrelationHeatmap
from mlAlgorithms.visualization.result.classificationPlots import (
    plotFeatureImportance,
    plotLearningCurve,
    plotTreeStructure,
)
from mlAlgorithms.visualization.result.regressionPlots import (
    plotRegressionResult,
    plotResiduals,
)
from mlAlgorithms.workflows.baseRunner import (
    appendArtifact,
    applyPreprocessor,
    buildRunContext,
    callTrainer,
    makeRunResult,
    makeSplit,
    prepareModelInput,
    runAnalysis,
)


def _plotFeatureTargetScatter(data, featureColumns, outputDir, filename):
    """绘制特征与目标关系图。"""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1, len(featureColumns), figsize=(5 * len(featureColumns), 4)
    )
    if len(featureColumns) == 1:
        axes = [axes]
    for axis, column in zip(axes, featureColumns, strict=True):
        axis.scatter(data[column], data["price"], s=18, alpha=0.45, color="#1E88E5")
        axis.set_xlabel(column)
        axis.set_ylabel("price")
        axis.grid(True, alpha=0.25)
    figure.suptitle("特征与目标关系")
    figure.tight_layout()
    from mlAlgorithms.visualization.figureSaver import saveFigure

    return saveFigure(figure, outputDir, filename)


def runRegressionPipeline(spec, datasetSpec):
    """执行回归流水线。"""
    context = buildRunContext(spec, datasetSpec)
    runAnalysis(context)
    featureColumns = datasetSpec.resolveFeatureColumns(context.data)
    result = makeRunResult(None)
    if "correlationHeatmap" in spec.dataPlots:
        appendArtifact(
            result,
            plotCorrelationHeatmap(
                context.data,
                featureColumns + [datasetSpec.targetColumn],
                context.outputDir,
                f"{spec.id} 相关性热力图",
                "data_correlation.png",
            ),
        )
    if "featureTargetScatter" in spec.dataPlots:
        appendArtifact(
            result,
            _plotFeatureTargetScatter(
                context.data,
                featureColumns,
                context.outputDir,
                "data_feature_vs_price.png",
            ),
        )

    splitData = makeSplit(
        context.features, context.target, spec.splitter, context.randomState
    )
    splitData = applyPreprocessor(splitData, spec.preprocessor)

    if spec.metadata.get("multiModel"):
        models = callTrainer(
            spec.trainer,
            splitData["X_train_processed"],
            splitData["y_train"],
            randomState=context.randomState,
        )
        result.model = models
        result.metrics = {}
        for modelName, model in models.items():
            outputDir = resolveOutputDir(modelName)
            yPred = model.predict(
                prepareModelInput(model, splitData["X_test_processed"])
            )
            metrics = evaluateRegression(
                splitData["y_test"],
                yPred,
                nFeatures=len(featureColumns),
                printReport=True,
            )
            result.metrics[modelName] = metrics
            appendArtifact(
                result,
                plotRegressionResult(
                    splitData["y_test"], yPred, outputDir, f"{modelName} 结果展示"
                ),
            )
            appendArtifact(
                result,
                plotResiduals(
                    splitData["y_test"], yPred, outputDir, f"{modelName} 残差图"
                ),
            )
            appendArtifact(
                result,
                plotFeatureImportance(
                    model,
                    featureColumns,
                    outputDir,
                    f"{modelName} 系数图",
                    filename="coefficients.png",
                ),
            )
        return result

    model = callTrainer(
        spec.trainer,
        splitData["X_train_processed"],
        splitData["y_train"],
        randomState=context.randomState,
    )
    yPred = model.predict(prepareModelInput(model, splitData["X_test_processed"]))
    metrics = evaluateRegression(
        splitData["y_test"], yPred, nFeatures=len(featureColumns), printReport=True
    )
    result.model = model
    result.predictions = yPred
    result.metrics = metrics
    appendArtifact(
        result,
        plotRegressionResult(
            splitData["y_test"], yPred, context.outputDir, f"{spec.id} 结果展示"
        ),
    )
    appendArtifact(
        result,
        plotResiduals(
            splitData["y_test"], yPred, context.outputDir, f"{spec.id} 残差图"
        ),
    )
    if "featureImportance" in spec.resultPlots:
        appendArtifact(
            result,
            plotFeatureImportance(
                model, featureColumns, context.outputDir, f"{spec.id} 特征重要性"
            ),
        )
    if "treeStructure" in spec.diagnostics:
        appendArtifact(
            result,
            plotTreeStructure(
                model, featureColumns, None, context.outputDir, f"{spec.id} 树结构"
            ),
        )
    if (
        "learningCurve" in spec.diagnostics
        and spec.metadata.get("learningCurveEstimatorFactory") is not None
    ):
        estimator = spec.metadata["learningCurveEstimatorFactory"]()
        appendArtifact(
            result,
            plotLearningCurve(
                estimator,
                splitData["X_train_processed"],
                splitData["y_train"],
                context.outputDir,
                f"{spec.id} 学习曲线",
                scoring="r2",
            ),
        )
    return result
