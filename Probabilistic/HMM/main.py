# Probabilistic/HMM/main.py
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
print("HMM 算法学习项目：隐马尔可夫模型")
print("=" * 60)

# 1. 生成数据
print("\n正在生成数据...")
data = generate_data(n_steps=300, random_state=42)

# 2. 数据探索
explore_data(data)

# 3. 数据可视化
print("\n" + "=" * 60)
print("数据可视化")
print("=" * 60)
visualize_data(data)

# 4. 数据预处理
X_obs, lengths, y_true, n_symbols = preprocess_data(data)

# 5. 模型训练
model = train_model(
    X_obs,
    lengths,
    n_components=3,
    n_iter=100,
    tol=1e-3,
    random_state=42,
)

# 6. 模型评估
y_pred = evaluate_model(model, X_obs, lengths, y_true)

# 7. 结果可视化
visualize_results(data, y_pred)
