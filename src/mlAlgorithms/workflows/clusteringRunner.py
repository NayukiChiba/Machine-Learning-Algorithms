"""
聚类工作流
"""

from __future__ import annotations

from mlAlgorithms.evaluation.clusteringEvaluator import (
    evaluateClusteringWithGroundTruth,
)
from mlAlgorithms.visualization.data.dataPlots import (
    plotClassDistribution,
    plotCorrelationHeatmap,
    plotLabeled2dScatter,
    plotRaw2dScatter,
)
from mlAlgorithms.visualization.result.clusteringPlots import (
    plotClusters,
    plotDbscanEpsSweep,
    plotDbscanKDistance,
    plotKmeansSweep,
)
from mlAlgorithms.workflows.baseRunner import (
    alignClusterLabelsForDisplay,
    appendArtifact,
    applyPreprocessor,
    buildRunContext,
    callTrainer,
    makeRunResult,
    runAnalysis,
)


def runClusteringPipeline(spec, datasetSpec):
    """执行聚类流水线。"""
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
                f"{spec.id} 真实簇分布",
                "data_cluster_distribution.png",
            ),
        )
    if "rawScatter2d" in spec.dataPlots:
        appendArtifact(
            result,
            plotRaw2dScatter(
                context.data,
                featureColumns[0],
                featureColumns[1],
                context.outputDir,
                f"{spec.id} 原始散点",
                "data_raw_scatter.png",
            ),
        )
    if "labeledScatter2d" in spec.dataPlots:
        appendArtifact(
            result,
            plotLabeled2dScatter(
                context.data,
                featureColumns[0],
                featureColumns[1],
                datasetSpec.targetColumn,
                context.outputDir,
                f"{spec.id} 真实标签散点",
                "data_true_label_scatter.png",
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

    splitData = {
        "X_all": context.features,
        "y_all": context.target,
        "X_train": context.features,
        "y_train": context.target,
    }
    splitData = applyPreprocessor(splitData, spec.preprocessor)
    if "kmeansSweep" in spec.diagnostics:
        path, table = plotKmeansSweep(
            splitData["X_all_processed"], context.target, context.outputDir, currentK=4
        )
        appendArtifact(result, path)
        result.extras["diagnostic_table"] = table
    if "dbscanKDistance" in spec.diagnostics:
        path, distances = plotDbscanKDistance(
            splitData["X_all_processed"],
            context.outputDir,
            minSamples=5,
            currentEps=0.3,
        )
        appendArtifact(result, path)
        result.extras["k_distance"] = distances
    if "dbscanEpsSweep" in spec.diagnostics:
        path, table = plotDbscanEpsSweep(
            splitData["X_all_processed"],
            context.target,
            context.outputDir,
            currentEps=0.3,
            minSamples=5,
        )
        appendArtifact(result, path)
        result.extras["diagnostic_table"] = table

    model = callTrainer(
        spec.trainer,
        splitData["X_all_processed"],
        randomState=context.randomState,
    )
    labelsPred = (
        model.labels_
        if hasattr(model, "labels_")
        else model.fit_predict(splitData["X_all_processed"])
    )
    labelsDisplay = alignClusterLabelsForDisplay(labelsPred, context.target)
    metrics = evaluateClusteringWithGroundTruth(
        splitData["X_all_processed"],
        labelsPred,
        context.target,
        inertia=getattr(model, "inertia_", None),
        printReport=True,
    )
    result.model = model
    result.predictions = labelsPred
    result.metrics = metrics
    appendArtifact(
        result,
        plotClusters(
            splitData["X_all_processed"],
            labelsDisplay,
            context.target,
            [f"{column}（标准化）" for column in featureColumns],
            context.outputDir,
            f"{spec.id} 聚类结果",
            centers=getattr(model, "cluster_centers_", None),
        ),
    )
    return result
