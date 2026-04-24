"""
统一命令行入口

支持命令：
1. `python main.py list`
2. `python main.py run <pipelineId>`
3. `python main.py suite <groupName>`
4. `python main.py analyze <pipelineId>`
"""

from __future__ import annotations

import argparse
import traceback

from src.mlAlgorithms.analysis.sequenceAnalyzer import buildSequenceExplorationReport
from src.mlAlgorithms.analysis.tabularAnalyzer import (
    buildClassificationExplorationReport,
    buildClusteringExplorationReport,
    buildRegressionExplorationReport,
)
from src.mlAlgorithms.analysis.terminalRenderer import printExplorationReport
from src.mlAlgorithms.catalog.datasets import DATASET_REGISTRY
from src.mlAlgorithms.catalog.pipelines import PIPELINE_REGISTRY
from src.mlAlgorithms.core.taskTypes import DataKind, TaskType
from src.mlAlgorithms.workflows.baseRunner import ensureOptionalDependencies
from src.mlAlgorithms.workflows.executor import executePipeline


def _buildParser() -> argparse.ArgumentParser:
    """构建命令行解析器。"""
    parser = argparse.ArgumentParser(description="机器学习算法统一入口")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="列出所有流水线")

    runParser = subparsers.add_parser("run", help="运行单个流水线")
    runParser.add_argument(
        "pipelineId", help="流水线 ID，例如 classification.logistic_regression"
    )

    suiteParser = subparsers.add_parser("suite", help="运行一个流水线组")
    suiteParser.add_argument(
        "groupName", help="组名，例如 all、classification、regression、ensemble"
    )

    analyzeParser = subparsers.add_parser("analyze", help="只做数据探索")
    analyzeParser.add_argument("pipelineId", help="流水线 ID")
    return parser


def _groupPipelineIds(groupName: str) -> list[str]:
    """根据组名筛选流水线。"""
    if groupName == "all":
        return sorted(PIPELINE_REGISTRY.keys())
    return sorted(
        [
            itemId
            for itemId in PIPELINE_REGISTRY.keys()
            if itemId.split(".")[0] == groupName
        ]
    )


def _printPipelineList() -> None:
    """打印所有流水线。"""
    print("=" * 72)
    print("可用流水线")
    print("=" * 72)
    for pipelineId in sorted(PIPELINE_REGISTRY.keys()):
        spec = PIPELINE_REGISTRY.get(pipelineId)
        datasetSpec = DATASET_REGISTRY.get(spec.datasetId)
        print(f"- {pipelineId}")
        print(f"  数据集: {datasetSpec.id}")
        print(f"  任务类型: {spec.taskType}")
        print(f"  输出目录: outputs/{spec.outputKey}")


def _analyzePipeline(pipelineId: str) -> None:
    """只执行数据探索。"""
    spec = PIPELINE_REGISTRY.get(pipelineId)
    datasetSpec = DATASET_REGISTRY.get(spec.datasetId)
    data = datasetSpec.load()
    if datasetSpec.dataKind == DataKind.SEQUENCE:
        report = buildSequenceExplorationReport(data, datasetSpec)
    elif datasetSpec.taskType in {TaskType.CLASSIFICATION, TaskType.DIMENSIONALITY}:
        report = buildClassificationExplorationReport(data, datasetSpec)
    elif datasetSpec.taskType == TaskType.REGRESSION:
        report = buildRegressionExplorationReport(data, datasetSpec)
    else:
        report = buildClusteringExplorationReport(data, datasetSpec)
    printExplorationReport(report)


def _runPipeline(pipelineId: str) -> int:
    """运行单个流水线。"""
    spec = PIPELINE_REGISTRY.get(pipelineId)
    datasetSpec = DATASET_REGISTRY.get(spec.datasetId)
    try:
        ensureOptionalDependencies(spec)
        executePipeline(spec, datasetSpec)
    except ModuleNotFoundError as error:
        missingName = error.name or str(error)
        if missingName in spec.optionalDependencies:
            print(f"[SKIP] {pipelineId}: 缺少可选依赖 `{missingName}`")
            return 0
        print(f"[FAIL] {pipelineId}: {error}")
        print(traceback.format_exc())
        return 1
    except Exception as error:  # noqa: BLE001
        print(f"[FAIL] {pipelineId}: {error}")
        print(traceback.format_exc())
        return 1
    print(f"[OK] {pipelineId}: 执行完成")
    return 0


def _runSuite(groupName: str) -> int:
    """运行一组流水线。"""
    pipelineIds = _groupPipelineIds(groupName)
    if not pipelineIds:
        print(f"未找到组 `{groupName}` 对应的流水线")
        return 1
    failures = 0
    for index, pipelineId in enumerate(pipelineIds, start=1):
        print()
        print("-" * 72)
        print(f"[{index}/{len(pipelineIds)}] {pipelineId}")
        print("-" * 72)
        failures += _runPipeline(pipelineId)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    """程序入口。"""
    args = _buildParser().parse_args(argv)
    if args.command == "list":
        _printPipelineList()
        return 0
    if args.command == "run":
        return _runPipeline(args.pipelineId)
    if args.command == "suite":
        return _runSuite(args.groupName)
    if args.command == "analyze":
        _analyzePipeline(args.pipelineId)
        return 0
    raise ValueError(f"未知命令: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
