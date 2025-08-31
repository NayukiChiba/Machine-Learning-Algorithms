# Clustering/KMeans/main.py
import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，便于导入同级模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from generate_data import generate_data
from explore_data import explore_data
from visualize_data import visualize_data
from preprocess_data import preprocess_data
from train_model import train_model
from evaluate_model import evaluate_model
from visualize_results import visualize_results

print("\n" + "=" * 60)
print("KMeans 聚类学习项目：二维聚类数据")
print("=" * 60)

# 1. 生成数据
print("\n正在生成数据...")
df = generate_data(n_samples=400, centers=4, cluster_std=0.8, random_state=42)

# 2. 数据探索
explore_data(df)

# 3. 数据可视化
print("\n" + "=" * 60)
print("数据可视化")
print("=" * 60)
visualize_data(df)

# 4. 数据预处理（标准化）
X_scaled, scaler, X_orig = preprocess_data(df)

# 5. 模型训练
model = train_model(
    X_scaled,
    n_clusters=4,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=42,
)

# 6. 模型评估
labels = evaluate_model(model, X_scaled)

# 7. 结果可视化
visualize_results(model, scaler, X_orig, labels)
