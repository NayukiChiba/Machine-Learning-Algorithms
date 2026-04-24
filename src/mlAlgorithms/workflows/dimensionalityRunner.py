"""
降维工作流
"""

from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from mlAlgorithms.evaluation.classificationEvaluator import evaluateClassification
from mlAlgorithms.evaluation.dimensionalityEvaluator import evaluateDimensionality
from mlAlgorithms.visualization.data.dataPlots import (
    plotClassDistribution,
    plotCorrelationHeatmap,
    plotFeatureSpace2d,
    plotFeatureSpace3d,
)
from mlAlgorithms.visualization.result.classificationPlots import (
    plotClassificationResult,
    plotConfusionMatrix,
    plotLearningCurve,
    plotRocCurve,
)
from mlAlgorithms.visualization.result.dimensionalityPlots import (
    plotDimensionality,
    plotPcaTrainingCurve,
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


def runDimensionalityPipeline(spec, datasetSpec):
    """执行降维流水线。"""
    context = buildRunContext(spec, datasetSpec)
    runAnalysis(context)
    featureColumns = datasetSpec.resolveFeatureColumns(context.data)
    result = makeRunResult(None)
    if "classDistribution" in spec.dataPlots:
        appendArtifact(
            result,
            plotClassDistribution(
                context.data,
                datasetSpec.targetColumn,
                context.outputDir,
                f"{spec.id} 类别分布",
                "data_class_distribution.png",
            ),
        )
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
    if "featureSpace2d" in spec.dataPlots:
        appendArtifact(
            result,
            plotFeatureSpace2d(
                context.data,
                featureColumns,
                datasetSpec.targetColumn,
                context.outputDir,
                f"{spec.id} 2D 特征空间",
                "data_feature_space_2d.png",
            ),
        )
    if "featureSpace3d" in spec.dataPlots:
        appendArtifact(
            result,
            plotFeatureSpace3d(
                context.data,
                featureColumns,
                datasetSpec.targetColumn,
                context.outputDir,
                f"{spec.id} 3D 特征空间",
                "data_feature_space_3d.png",
            ),
        )

    if spec.id == "dimensionality.pca":
        splitData = {
            "X_all": context.features,
            "y_all": context.target,
            "X_train": context.features,
            "y_train": context.target,
        }
        splitData = applyPreprocessor(splitData, spec.preprocessor)
        if "pcaTrainingCurve" in spec.diagnostics:
            path, curveData = plotPcaTrainingCurve(
                splitData["X_all_processed"],
                context.outputDir,
                maxComponents=splitData["X_all_processed"].shape[1],
            )
            appendArtifact(result, path)
            result.extras["training_curve"] = curveData
        model2d = callTrainer(
            spec.trainer,
            splitData["X_all_processed"],
            nComponents=2,
            randomState=context.randomState,
        )
        X2d = model2d.transform(splitData["X_all_processed"])
        metrics2d = evaluateDimensionality(
            model2d,
            XOriginal=splitData["X_all_processed"],
            XTransformed=X2d,
            printReport=True,
        )
        appendArtifact(
            result,
            plotDimensionality(
                X2d,
                context.target,
                model2d.explained_variance_ratio_,
                context.outputDir,
                "PCA 降维 (2D)",
                "2d",
                "dimensionality_2d.png",
            ),
        )
        model3d = callTrainer(
            spec.trainer,
            splitData["X_all_processed"],
            nComponents=3,
            randomState=context.randomState,
        )
        X3d = model3d.transform(splitData["X_all_processed"])
        metrics3d = evaluateDimensionality(
            model3d,
            XOriginal=splitData["X_all_processed"],
            XTransformed=X3d,
            printReport=True,
        )
        appendArtifact(
            result,
            plotDimensionality(
                X3d,
                context.target,
                model3d.explained_variance_ratio_,
                context.outputDir,
                "PCA 降维 (3D)",
                "3d",
                "dimensionality_3d.png",
            ),
        )
        result.model = {"pca_2d": model2d, "pca_3d": model3d}
        result.metrics = {"pca_2d": metrics2d, "pca_3d": metrics3d}
        return result

    splitData = makeSplit(
        context.features, context.target, spec.splitter, context.randomState
    )
    splitData = applyPreprocessor(splitData, spec.preprocessor)
    model = callTrainer(
        spec.trainer,
        splitData["X_train_processed"],
        splitData["y_train"],
        nComponents=2,
        randomState=context.randomState,
    )
    XTestPrepared = prepareModelInput(model, splitData["X_test_processed"])
    yPred = model.predict(XTestPrepared)
    yScores = model.predict_proba(XTestPrepared)
    XTest2d = model.transform(splitData["X_test_processed"])
    XAll2d = model.transform(splitData["X_all_processed"])
    dimMetrics = evaluateDimensionality(
        model,
        XOriginal=splitData["X_test_processed"],
        XTransformed=XTest2d,
        printReport=True,
    )
    clsMetrics = evaluateClassification(
        splitData["y_test"], yPred, yScores=yScores, printReport=True
    )
    result.model = model
    result.predictions = yPred
    result.scores = yScores
    result.metrics = {"dimensionality": dimMetrics, "classification": clsMetrics}
    appendArtifact(
        result,
        plotDimensionality(
            XAll2d,
            context.target,
            getattr(model, "explained_variance_ratio_", None),
            context.outputDir,
            "LDA 降维 (2D)",
            "2d",
            "dimensionality_2d.png",
        ),
    )
    appendArtifact(
        result,
        plotClassificationResult(
            XTest2d,
            splitData["y_test"],
            yPred,
            ["LD1", "LD2"],
            context.outputDir,
            "LDA 结果展示",
        ),
    )
    appendArtifact(
        result,
        plotConfusionMatrix(
            splitData["y_test"], yPred, context.outputDir, "LDA 混淆矩阵"
        ),
    )
    appendArtifact(
        result,
        plotRocCurve(splitData["y_test"], yScores, context.outputDir, "LDA ROC 曲线"),
    )
    if "learningCurve" in spec.diagnostics:
        appendArtifact(
            result,
            plotLearningCurve(
                LinearDiscriminantAnalysis(solver="svd"),
                splitData["X_train_processed"],
                splitData["y_train"],
                context.outputDir,
                "LDA 学习曲线",
            ),
        )
    return result
