"""
数据集注册表
"""

from __future__ import annotations

from mlAlgorithms.core.registry import Registry
from mlAlgorithms.datasets.datasetCatalog import buildDatasetSpecs


DATASET_REGISTRY = Registry()
for datasetSpec in buildDatasetSpecs():
    DATASET_REGISTRY.register(datasetSpec.id, datasetSpec)
