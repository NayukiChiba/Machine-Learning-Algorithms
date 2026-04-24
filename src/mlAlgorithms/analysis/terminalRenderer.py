"""
终端报告渲染器
"""

from __future__ import annotations

from mlAlgorithms.analysis.reportModels import (
    SequenceExplorationReport,
    TabularExplorationReport,
)


def printExplorationReport(
    report: TabularExplorationReport | SequenceExplorationReport,
) -> None:
    """统一打印探索报告。"""
    print("=" * 72)
    print(f"数据探索报告: {report.overview.datasetId}")
    print("=" * 72)
    print(f"描述: {report.overview.description}")
    print(f"样本数: {report.overview.rowCount}")
    print(f"特征数: {report.overview.featureCount}")
    print(f"特征列: {report.overview.featureColumns}")
    print(f"目标列: {report.overview.targetColumn}")
    print(f"缺失值总数: {report.overview.missingCount}")
    if report.warnings:
        print("警告:")
        for item in report.warnings:
            print(f"- {item}")

    if isinstance(report, TabularExplorationReport):
        print("--- 目标摘要 ---")
        for key, value in report.targetSummary.items():
            print(f"{key}: {value}")
        print("--- 相关性摘要 ---")
        print(f"特征对数量: {report.correlationSummary.get('pair_count', 0)}")
        if report.correlationSummary.get("top_pairs"):
            print(f"最强相关特征对: {report.correlationSummary['top_pairs'][0]}")
        print("--- 多变量摘要 ---")
        vif = report.multivariateSummary.get("vif", {})
        if vif:
            topColumn = max(vif, key=vif.get)
            print(f"最高 VIF 特征: {topColumn} -> {vif[topColumn]:.4f}")
        pcaPotential = report.multivariateSummary.get("pca_potential", {})
        if pcaPotential:
            print(f"达到 90% 累计方差所需维度: {pcaPotential['dims_for_90_percent']}")
    else:
        print("--- 观测分布 ---")
        print(report.observationSummary)
        print("--- 隐状态分布 ---")
        print(report.stateSummary)
        print("--- 转移摘要 ---")
        preview = list(report.transitionSummary.items())[:5]
        for key, value in preview:
            print(f"{key}: {value}")
