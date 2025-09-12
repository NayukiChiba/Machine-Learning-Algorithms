# Dimensionality/LDA/main.py
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
print("LDA 降维学习项目：Iris 数据集")
print("=" * 60)

# 1. 生成数据
print("\n正在生成数据...")
df = generate_data()

# 2. 数据探索
explore_data(df)

# 3. 数据可视化
print("\n" + "=" * 60)
print("数据可视化")
print("=" * 60)
visualize_data(df)

# 4. 数据预处理（标准化）
X_scaled, scaler, X, y = preprocess_data(df)

# 5. 模型训练
model = train_model(
    X_scaled,
    y,
    n_components=2,
    solver="svd",
)

# 6. 模型评估
X_lda = evaluate_model(model, X_scaled)

# 7. 结果可视化
evr = (
    model.explained_variance_ratio_
    if hasattr(model, "explained_variance_ratio_")
    else None
)
visualize_results(X_lda, y.values, evr)
