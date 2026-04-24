"""
表格数据分析器
"""

from __future__ import annotations

import math

import numpy as np
from pandas import DataFrame

from mlAlgorithms.analysis.reportModels import DatasetOverview, TabularExplorationReport
from mlAlgorithms.core.datasetSpec import DatasetSpec
from mlAlgorithms.core.taskTypes import TaskType


def _buildNumericSummary(
    data: DataFrame, featureColumns: list[str]
) -> dict[str, dict[str, float]]:
    """构建数值统计摘要。"""
    summary: dict[str, dict[str, float]] = {}
    for column in featureColumns:
        series = data[column]
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = int(((series < lower) | (series > upper)).sum())
        summary[column] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "q1": q1,
            "q3": q3,
            "iqr": float(iqr),
            "skew": float(series.skew()),
            "kurt": float(series.kurt()),
            "outliers": outliers,
        }
    return summary


def _buildTargetSummary(
    data: DataFrame, targetColumn: str | None, taskType: TaskType
) -> dict[str, object]:
    """构建目标变量摘要。"""
    if targetColumn is None:
        return {}
    series = data[targetColumn]
    if taskType in {
        TaskType.CLASSIFICATION,
        TaskType.CLUSTERING,
        TaskType.PROBABILISTIC,
        TaskType.DIMENSIONALITY,
    }:
        counts = series.value_counts().sort_index()
        return {
            "nunique": int(series.nunique()),
            "distribution": {str(index): int(value) for index, value in counts.items()},
        }
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def _buildCorrelationSummary(
    data: DataFrame, featureColumns: list[str]
) -> dict[str, object]:
    """构建相关性摘要。"""
    if len(featureColumns) < 2:
        return {"pair_count": 0, "top_pairs": []}
    corr = data[featureColumns].corr(method="pearson")
    pairs: list[tuple[str, str, float]] = []
    for leftIndex, leftColumn in enumerate(featureColumns):
        for rightColumn in featureColumns[leftIndex + 1 :]:
            pairs.append(
                (leftColumn, rightColumn, float(corr.loc[leftColumn, rightColumn]))
            )
    pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    return {
        "pair_count": len(pairs),
        "top_pairs": pairs[: min(10, len(pairs))],
        "average_abs_correlation": float(np.mean([abs(item[2]) for item in pairs]))
        if pairs
        else 0.0,
    }


def _buildRelationSummary(
    data: DataFrame,
    featureColumns: list[str],
    targetColumn: str | None,
    taskType: TaskType,
) -> dict[str, object]:
    """构建特征与目标的关系摘要。"""
    if targetColumn is None:
        return {}
    if taskType == TaskType.REGRESSION:
        results = []
        for column in featureColumns:
            results.append(
                (
                    column,
                    float(data[column].corr(data[targetColumn], method="pearson")),
                    float(data[column].corr(data[targetColumn], method="spearman")),
                )
            )
        results.sort(key=lambda item: abs(item[1]), reverse=True)
        return {"feature_target_correlation": results}
    groupMeans: dict[str, dict[str, float]] = {}
    for column in featureColumns:
        grouped = data.groupby(targetColumn)[column].mean().sort_index()
        groupMeans[column] = {
            str(index): float(value) for index, value in grouped.items()
        }
    return {"group_means": groupMeans}


def _buildVifSummary(data: DataFrame, featureColumns: list[str]) -> dict[str, float]:
    """构建 VIF 摘要。"""
    if len(featureColumns) < 2:
        return {}
    X = data[featureColumns].to_numpy(dtype=float)
    XMean = X.mean(axis=0)
    XStd = X.std(axis=0)
    XStd[XStd == 0] = 1.0
    XNorm = (X - XMean) / XStd
    result: dict[str, float] = {}
    for index, column in enumerate(featureColumns):
        target = XNorm[:, index]
        others = np.delete(XNorm, index, axis=1)
        XDesign = np.column_stack([np.ones(len(target)), others])
        beta, *_ = np.linalg.lstsq(XDesign, target, rcond=None)
        predicted = XDesign @ beta
        ssResidual = np.sum((target - predicted) ** 2)
        ssTotal = np.sum((target - target.mean()) ** 2)
        rSquared = 1 - ssResidual / ssTotal if ssTotal > 0 else 0.0
        vif = math.inf if rSquared >= 0.999999 else 1 / (1 - rSquared)
        result[column] = float(vif)
    return result


def _buildPcaPotentialSummary(
    data: DataFrame, featureColumns: list[str]
) -> dict[str, object]:
    """构建 PCA 潜力摘要。"""
    if len(featureColumns) < 2:
        return {}
    X = data[featureColumns].to_numpy(dtype=float)
    X = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
    _, singularValues, _ = np.linalg.svd(X, full_matrices=False)
    variances = singularValues**2 / max(len(X) - 1, 1)
    ratio = variances / variances.sum()
    cumulative = np.cumsum(ratio)
    dims90 = int(np.searchsorted(cumulative, 0.90)) + 1
    return {
        "explained_variance_ratio": ratio[: min(10, len(ratio))].round(4).tolist(),
        "dims_for_90_percent": dims90,
    }


def _buildFisherSummary(
    data: DataFrame, featureColumns: list[str], targetColumn: str | None
) -> dict[str, float]:
    """构建 Fisher 比摘要。"""
    if targetColumn is None:
        return {}
    classes = sorted(data[targetColumn].unique())
    total = len(data)
    results: dict[str, float] = {}
    for column in featureColumns:
        overallMean = float(data[column].mean())
        between = 0.0
        within = 0.0
        for classValue in classes:
            subset = data[data[targetColumn] == classValue][column]
            between += len(subset) * (float(subset.mean()) - overallMean) ** 2
            within += len(subset) * float(subset.var())
        between /= max(total, 1)
        within /= max(total, 1)
        results[column] = float(math.inf if within == 0 else between / within)
    return results


def buildClassificationExplorationReport(
    data: DataFrame, datasetSpec: DatasetSpec
) -> TabularExplorationReport:
    """构建分类探索报告。"""
    return _buildTabularReport(data, datasetSpec, TaskType.CLASSIFICATION)


def buildRegressionExplorationReport(
    data: DataFrame, datasetSpec: DatasetSpec
) -> TabularExplorationReport:
    """构建回归探索报告。"""
    return _buildTabularReport(data, datasetSpec, TaskType.REGRESSION)


def buildClusteringExplorationReport(
    data: DataFrame, datasetSpec: DatasetSpec
) -> TabularExplorationReport:
    """构建聚类探索报告。"""
    return _buildTabularReport(data, datasetSpec, TaskType.CLUSTERING)


def _buildTabularReport(
    data: DataFrame, datasetSpec: DatasetSpec, taskType: TaskType
) -> TabularExplorationReport:
    """构建统一表格探索报告。"""
    featureColumns = datasetSpec.resolveFeatureColumns(data)
    targetColumn = datasetSpec.targetColumn
    overview = DatasetOverview(
        datasetId=datasetSpec.id,
        rowCount=len(data),
        featureCount=len(featureColumns),
        featureColumns=featureColumns,
        targetColumn=targetColumn,
        missingCount=int(data.isnull().sum().sum()),
        description=datasetSpec.description,
    )
    warnings: list[str] = []
    if overview.missingCount > 0:
        warnings.append(f"存在缺失值: {overview.missingCount}")
    return TabularExplorationReport(
        reportType=taskType.value,
        overview=overview,
        numericSummary=_buildNumericSummary(data, featureColumns),
        targetSummary=_buildTargetSummary(data, targetColumn, taskType),
        correlationSummary=_buildCorrelationSummary(data, featureColumns),
        relationSummary=_buildRelationSummary(
            data, featureColumns, targetColumn, taskType
        ),
        multivariateSummary={
            "vif": _buildVifSummary(data, featureColumns),
            "pca_potential": _buildPcaPotentialSummary(data, featureColumns),
            "fisher_ratio": _buildFisherSummary(data, featureColumns, targetColumn),
        },
        warnings=warnings,
    )
