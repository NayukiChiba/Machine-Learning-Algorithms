"""
概率工作流
"""

from __future__ import annotations

from mlAlgorithms.evaluation.clusteringEvaluator import (
    evaluateClusteringWithGroundTruth,
)
from mlAlgorithms.evaluation.sequenceEvaluator import evaluateSequenceLabels
from mlAlgorithms.visualization.data.dataPlots import (
    plotClassDistribution,
    plotCorrelationHeatmap,
    plotLabeled2dScatter,
    plotRaw2dScatter,
)
from mlAlgorithms.visualization.result.clusteringPlots import (
    plotClusters,
    plotGmmComponentSweep,
)
from mlAlgorithms.visualization.result.sequencePlots import (
    plotHmmDataOverview,
    plotHmmEvaluationFigure,
    plotHmmResultFigure,
)
from mlAlgorithms.workflows.baseRunner import (
    alignClusterLabelsForDisplay,
    appendArtifact,
    appendArtifacts,
    applyPreprocessor,
    buildRunContext,
    callTrainer,
    makeRunResult,
    runAnalysis,
)


def runProbabilisticPipeline(spec, datasetSpec):
    """执行概率任务流水线。"""
    context = buildRunContext(spec, datasetSpec)
    runAnalysis(context)

    if datasetSpec.dataKind.value == "sequence":
        result = makeRunResult(None)
        appendArtifacts(result, plotHmmDataOverview(context.data, context.outputDir))
        observations = context.data["obs"].to_numpy().reshape(-1, 1)
        lengths = [len(observations)]
        model = callTrainer(
            spec.trainer,
            observations,
            lengths,
            randomState=context.randomState,
        )
        statesPred = model.predict(observations, lengths)
        metrics = evaluateSequenceLabels(
            context.data["state_true"].to_numpy(),
            statesPred,
            logLikelihood=model.score(observations, lengths),
            printReport=True,
        )
        result.model = model
        result.predictions = statesPred
        result.metrics = metrics
        appendArtifact(
            result, plotHmmResultFigure(context.data, statesPred, context.outputDir)
        )
        appendArtifact(
            result,
            plotHmmEvaluationFigure(
                model,
                context.data["state_true"].to_numpy(),
                statesPred,
                context.outputDir,
            ),
        )
        return result

    featureColumns = datasetSpec.resolveFeatureColumns(context.data)
    result = makeRunResult(None)
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
        "X_train": context.features,
        "y_all": context.target,
        "y_train": context.target,
    }
    splitData = applyPreprocessor(splitData, spec.preprocessor)
    if "gmmComponentSweep" in spec.diagnostics:
        path, table = plotGmmComponentSweep(
            splitData["X_all_processed"],
            context.target,
            context.outputDir,
            currentComponents=3,
        )
        appendArtifact(result, path)
        result.extras["diagnostic_table"] = table
    model = callTrainer(
        spec.trainer,
        splitData["X_all_processed"],
        randomState=context.randomState,
    )
    labelsPred = model.predict(splitData["X_all_processed"])
    probabilities = model.predict_proba(splitData["X_all_processed"])
    labelsDisplay = alignClusterLabelsForDisplay(labelsPred, context.target)
    metrics = evaluateClusteringWithGroundTruth(
        splitData["X_all_processed"], labelsPred, context.target, printReport=True
    )
    result.model = model
    result.predictions = labelsPred
    result.scores = probabilities
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
            centers=model.means_,
            filename="result_display.png",
        ),
    )
    return result
