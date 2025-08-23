# Classification/KNN/main.py
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
from compare_k import compare_k_values

print("\n" + "=" * 60)
print("KNN 分类学习项目：双月牙二分类数据")
print("=" * 60)

# 1. 生成数据
print("\n正在生成数据...")
df = generate_data(n_samples=400, noise=0.25, random_state=42)

# 2. 数据探索
class_ratio = explore_data(df)

# 3. 数据可视化
print("\n" + "=" * 60)
print("数据可视化")
print("=" * 60)
visualize_data(df)

# 4. 数据预处理（划分 + 标准化）
X_train, X_test, y_train, y_test, scaler, X_train_orig, X_test_orig = preprocess_data(
    df, test_size=0.2, random_state=42
)

# 5. 模型训练
model = train_model(
    X_train,
    y_train,
    n_neighbors=5,
    weights="uniform",
    metric="minkowski",
)

# 6. 模型评估
y_train_pred, y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test)

# 7. 结果可视化
visualize_results(
    model,
    scaler,
    X_train_orig,
    X_test_orig,
    y_train,
    y_test,
    y_train_pred,
    y_test_pred,
)

# 8. K 值对比曲线
compare_k_values(k_min=1, k_max=15, weights="uniform", metric="minkowski")
