"""
分类工作流
"""

from __future__ import annotations

from mlAlgorithms.evaluation.classificationEvaluator import evaluateClassification
from mlAlgorithms.visualization.data.dataPlots import (
    plotClassDistribution,
    plotCorrelationHeatmap,
    plotFeatureSpace2d,
    plotLabeled2dScatter,
)
from mlAlgorithms.visualization.result.classificationPlots import (
    formatTreeRules,
    plotClassificationResult,
    plotConfusionMatrix,
    plotDecisionBoundary,
    plotFeatureImportance,
    plotLearningCurve,
    plotRocCurve,
    plotTreeStructure,
)
from mlAlgorithms.workflows.baseRunner import (
    appendArtifact,
    applyPreprocessor,
    buildRunContext,
    callTrainer,
    collectScoreOutput,
    makeRunResult,
    makeSplit,
    prepareModelInput,
    prepare2dProjection,
    runAnalysis,
)


def runClassificationPipeline(spec, datasetSpec):
    """执行分类流水线。"""
    context = buildRunContext(spec, datasetSpec)
    runAnalysis(context)
    featureColumns = datasetSpec.resolveFeatureColumns(context.data)
    if "classDistribution" in spec.dataPlots:
        appendArtifact(
            result := makeRunResult(None),
            plotClassDistribution(
                context.data,
                datasetSpec.targetColumn,
                context.outputDir,
                f"{spec.id} 类别分布",
                "data_class_distribution.png",
            ),
        )
        context.extras["seed_result"] = result
    if "correlationHeatmap" in spec.dataPlots:
        appendArtifact(
            context.extras["seed_result"],
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
            context.extras["seed_result"],
            plotFeatureSpace2d(
                context.data,
                featureColumns,
                datasetSpec.targetColumn,
                context.outputDir,
                f"{spec.id} 特征空间",
                "data_feature_space_2d.png",
            ),
        )
    if "labeledScatter2d" in spec.dataPlots and len(featureColumns) >= 2:
        appendArtifact(
            context.extras["seed_result"],
            plotLabeled2dScatter(
                context.data,
                featureColumns[0],
                featureColumns[1],
                datasetSpec.targetColumn,
                context.outputDir,
                f"{spec.id} 原始散点",
                "data_scatter.png",
            ),
        )

    splitData = makeSplit(
        context.features, context.target, spec.splitter, context.randomState
    )
    splitData = applyPreprocessor(splitData, spec.preprocessor)
    model = callTrainer(
        spec.trainer,
        splitData["X_train_processed"],
        splitData["y_train"],
        randomState=context.randomState,
    )
    yPred = model.predict(prepareModelInput(model, splitData["X_test_processed"]))
    yScores = collectScoreOutput(model, splitData["X_test_processed"])
    metrics = evaluateClassification(
        splitData["y_test"], yPred, yScores=yScores, printReport=True
    )
    result = context.extras.pop("seed_result", makeRunResult(model))
    result.model = model
    result.predictions = yPred
    result.scores = yScores
    result.metrics = metrics

    if "confusionMatrix" in spec.resultPlots:
        appendArtifact(
            result,
            plotConfusionMatrix(
                splitData["y_test"], yPred, context.outputDir, f"{spec.id} 混淆矩阵"
            ),
        )
    if "rocCurve" in spec.resultPlots and yScores is not None:
        appendArtifact(
            result,
            plotRocCurve(
                splitData["y_test"], yScores, context.outputDir, f"{spec.id} ROC 曲线"
            ),
        )
    if "featureImportance" in spec.resultPlots:
        appendArtifact(
            result,
            plotFeatureImportance(
                model, featureColumns, context.outputDir, f"{spec.id} 特征重要性"
            ),
        )
    projection = prepare2dProjection(
        splitData,
        splitData["y_train"],
        model,
        spec.metadata.get("visualModelFactory"),
    )
    if "classificationResult" in spec.resultPlots:
        appendArtifact(
            result,
            plotClassificationResult(
                projection["X_test_plot"],
                splitData["y_test"],
                yPred,
                projection["feature_names"],
                context.outputDir,
                f"{spec.id} 结果展示",
            ),
        )
    if (
        "decisionBoundary" in spec.resultPlots
        and projection["boundary_model"] is not None
    ):
        appendArtifact(
            result,
            plotDecisionBoundary(
                projection["boundary_model"],
                projection["X_all_plot"],
                splitData["y_all"],
                projection["feature_names"],
                context.outputDir,
                f"{spec.id} 决策边界",
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
            ),
        )
    if "treeStructure" in spec.diagnostics:
        classNames = [str(label) for label in sorted(context.target.unique())]
        appendArtifact(
            result,
            plotTreeStructure(
                model,
                featureColumns,
                classNames,
                context.outputDir,
                f"{spec.id} 树结构",
            ),
        )
        print(formatTreeRules(model, featureColumns))
    return result
