"""
序列数据分析器
"""

from __future__ import annotations

import numpy as np
from pandas import DataFrame

from mlAlgorithms.analysis.reportModels import (
    DatasetOverview,
    SequenceExplorationReport,
)
from mlAlgorithms.core.datasetSpec import DatasetSpec


def buildSequenceExplorationReport(
    data: DataFrame, datasetSpec: DatasetSpec
) -> SequenceExplorationReport:
    """构建序列探索报告。"""
    observations = data["obs"].value_counts().sort_index()
    states = data["state_true"].value_counts().sort_index()
    transitionCounts: dict[tuple[int, int], int] = {}
    stateValues = data["state_true"].to_numpy()
    obsValues = data["obs"].to_numpy()
    for index in range(len(stateValues) - 1):
        pair = (int(stateValues[index]), int(stateValues[index + 1]))
        transitionCounts[pair] = transitionCounts.get(pair, 0) + 1

    durationSummary: dict[str, dict[str, float]] = {}
    currentState = int(stateValues[0])
    currentLength = 1
    durations: dict[int, list[int]] = {}
    for index in range(1, len(stateValues)):
        stateValue = int(stateValues[index])
        if stateValue == currentState:
            currentLength += 1
            continue
        durations.setdefault(currentState, []).append(currentLength)
        currentState = stateValue
        currentLength = 1
    durations.setdefault(currentState, []).append(currentLength)
    for stateValue, items in durations.items():
        durationSummary[str(stateValue)] = {
            "segments": float(len(items)),
            "mean_duration": float(np.mean(items)),
            "max_duration": float(max(items)),
        }

    bigramCounts: dict[tuple[int, int], int] = {}
    for index in range(len(obsValues) - 1):
        pair = (int(obsValues[index]), int(obsValues[index + 1]))
        bigramCounts[pair] = bigramCounts.get(pair, 0) + 1

    warnings: list[str] = []
    if not (data["time"].diff().dropna() == 1).all():
        warnings.append("时间步存在间断")

    return SequenceExplorationReport(
        reportType="sequence",
        overview=DatasetOverview(
            datasetId=datasetSpec.id,
            rowCount=len(data),
            featureCount=1,
            featureColumns=["obs"],
            targetColumn=datasetSpec.targetColumn,
            missingCount=int(data.isnull().sum().sum()),
            description=datasetSpec.description,
        ),
        observationSummary={
            str(index): int(value) for index, value in observations.items()
        },
        stateSummary={str(index): int(value) for index, value in states.items()},
        transitionSummary={
            f"{left}->{right}": count
            for (left, right), count in sorted(transitionCounts.items())
        },
        durationSummary=durationSummary,
        bigramSummary={
            f"{left}->{right}": count
            for (left, right), count in sorted(bigramCounts.items())
        },
        warnings=warnings,
    )
