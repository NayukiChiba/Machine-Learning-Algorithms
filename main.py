"""
全量流水线执行入口

功能：
1. 顺序执行所有模型流水线
2. 单个模型失败时不中断后续执行
3. 输出成功、跳过、失败汇总

使用方法：
    python main.py
"""

from __future__ import annotations

import importlib
import time
import traceback
from typing import Any


PIPELINES: list[tuple[str, str]] = [
    ("逻辑回归分类", "pipelines.classification.logistic_regression"),
    ("决策树分类", "pipelines.classification.decision_tree"),
    ("SVC 分类", "pipelines.classification.svc"),
    ("朴素贝叶斯分类", "pipelines.classification.naive_bayes"),
    ("KNN 分类", "pipelines.classification.knn"),
    ("随机森林分类", "pipelines.classification.random_forest"),
    ("线性回归", "pipelines.regression.linear_regression"),
    ("SVR 回归", "pipelines.regression.svr"),
    ("决策树回归", "pipelines.regression.decision_tree"),
    ("正则化回归", "pipelines.regression.regularization"),
    ("KMeans 聚类", "pipelines.clustering.kmeans"),
    ("DBSCAN 聚类", "pipelines.clustering.dbscan"),
    ("PCA 降维", "pipelines.dimensionality.pca"),
    ("LDA 降维", "pipelines.dimensionality.lda"),
    ("Bagging 集成", "pipelines.ensemble.bagging"),
    ("GBDT 集成", "pipelines.ensemble.gbdt"),
    ("LightGBM 集成", "pipelines.ensemble.lightgbm"),
    ("XGBoost 集成", "pipelines.ensemble.xgboost"),
    ("EM 概率模型", "pipelines.probabilistic.em"),
    ("HMM 概率模型", "pipelines.probabilistic.hmm"),
]

# 当前环境可能缺失，但不应阻塞全量运行的可选依赖
OPTIONAL_DEPENDENCIES = {"xgboost", "lightgbm", "hmmlearn"}


def _run_pipeline(module_path: str) -> None:
    """导入并执行单个流水线模块。"""
    module = importlib.import_module(module_path)
    run_func = getattr(module, "run", None)
    if not callable(run_func):
        raise AttributeError(f"模块 {module_path} 未提供可调用的 run()")
    run_func()


def _format_seconds(seconds: float) -> str:
    """格式化耗时。"""
    return f"{seconds:.2f}s"


def _get_missing_optional_dependency(exc: BaseException) -> str | None:
    """从异常及其因果链中提取缺失的可选依赖名。"""
    candidates: list[BaseException | None] = [exc, getattr(exc, "__cause__", None)]

    for item in candidates:
        if isinstance(item, ModuleNotFoundError) and item.name in OPTIONAL_DEPENDENCIES:
            return item.name

    message = str(exc).lower()
    for dep_name in OPTIONAL_DEPENDENCIES:
        if dep_name in message:
            return dep_name

    return None


def main() -> int:
    """执行所有流水线，并输出汇总结果。"""
    total = len(PIPELINES)
    start_total = time.perf_counter()
    results: list[dict[str, Any]] = []

    print("=" * 72)
    print("机器学习算法全量流水线执行")
    print("=" * 72)
    print(f"总任务数: {total}")

    for index, (display_name, module_path) in enumerate(PIPELINES, start=1):
        print()
        print("-" * 72)
        print(f"[{index}/{total}] 开始执行: {display_name}")
        print(f"模块路径: {module_path}")
        print("-" * 72)

        start_one = time.perf_counter()
        try:
            _run_pipeline(module_path)
        except ModuleNotFoundError as exc:
            elapsed = time.perf_counter() - start_one
            missing_dep = _get_missing_optional_dependency(exc)
            if missing_dep is not None:
                message = f"缺少可选依赖 `{missing_dep}`，已跳过"
                print(f"[SKIP] {display_name}: {message}")
                results.append(
                    {
                        "name": display_name,
                        "module": module_path,
                        "status": "skipped",
                        "elapsed": elapsed,
                        "message": message,
                    }
                )
                continue

            message = f"模块导入失败: {exc}"
            print(f"[FAIL] {display_name}: {message}")
            print(traceback.format_exc())
            results.append(
                {
                    "name": display_name,
                    "module": module_path,
                    "status": "failed",
                    "elapsed": elapsed,
                    "message": message,
                }
            )
            continue
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - start_one
            missing_dep = _get_missing_optional_dependency(exc)
            if missing_dep is not None:
                message = f"缺少可选依赖 `{missing_dep}`，已跳过"
                print(f"[SKIP] {display_name}: {message}")
                results.append(
                    {
                        "name": display_name,
                        "module": module_path,
                        "status": "skipped",
                        "elapsed": elapsed,
                        "message": message,
                    }
                )
                continue

            message = f"执行失败: {exc}"
            print(f"[FAIL] {display_name}: {message}")
            print(traceback.format_exc())
            results.append(
                {
                    "name": display_name,
                    "module": module_path,
                    "status": "failed",
                    "elapsed": elapsed,
                    "message": message,
                }
            )
            continue

        elapsed = time.perf_counter() - start_one
        print(f"[OK] {display_name}: 执行完成，耗时 {_format_seconds(elapsed)}")
        results.append(
            {
                "name": display_name,
                "module": module_path,
                "status": "success",
                "elapsed": elapsed,
                "message": "",
            }
        )

    total_elapsed = time.perf_counter() - start_total
    success_count = sum(item["status"] == "success" for item in results)
    skipped_count = sum(item["status"] == "skipped" for item in results)
    failed_count = sum(item["status"] == "failed" for item in results)

    print()
    print("=" * 72)
    print("执行汇总")
    print("=" * 72)
    print(f"成功: {success_count}")
    print(f"跳过: {skipped_count}")
    print(f"失败: {failed_count}")
    print(f"总耗时: {_format_seconds(total_elapsed)}")

    if skipped_count:
        print()
        print("跳过任务：")
        for item in results:
            if item["status"] == "skipped":
                print(f"- {item['name']} ({item['module']}): {item['message']}")

    if failed_count:
        print()
        print("失败任务：")
        for item in results:
            if item["status"] == "failed":
                print(f"- {item['name']} ({item['module']}): {item['message']}")

    print()
    print("各任务耗时：")
    for item in results:
        print(
            f"- {item['name']}: {item['status']} ({_format_seconds(item['elapsed'])})"
        )

    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
