"""
pipelines/probabilistic/hmm.py
HMM 端到端流水线

运行方式: python -m pipelines.probabilistic.hmm
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import get_model_output_dir
from data_exploration.bivariate import _print_sequence_transition
from data_exploration.univariate import _print_discrete_distribution
from data_generation import hmm_data
from result_visualization.clustering_diagnostics import _save_figure
from model_training.probabilistic.hmm import train_model

MODEL = "hmm"


def show_data_exploration(data) -> None:
    """
    展示 HMM 训练前的数据探索结果

    HMM 是序列模型，因此这里关注的不是普通特征分布，
    而是：
    1. 时间步是否连续；
    2. 观测符号分布；
    3. 真实隐状态分布；
    4. 隐状态转移结构。
    """
    print("=" * 60)
    print("HMM：数据探索")
    print("=" * 60)
    print(f"序列长度: {len(data)}")
    print(f"字段: {list(data.columns)}")
    print(f"时间步范围: {data['time'].min()} ~ {data['time'].max()}")

    is_continuous = (data["time"].diff().dropna() == 1).all()
    print(f"时间步是否连续: {'是' if is_continuous else '否'}")

    print("--- 观测符号分布 ---")
    _print_discrete_distribution(data, "obs", "obs")
    print("--- 真实隐状态分布 ---")
    _print_discrete_distribution(data, "state_true", "state_true")
    print("--- 隐状态转移统计 ---")
    _print_sequence_transition(data)


def show_data_preview(data) -> None:
    """
    展示 HMM 训练前的数据图

    HMM 最适合的展示图不是散点图，而是时间序列图和分布图。
    """
    save_dir = get_model_output_dir(MODEL)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n开始生成数据展示图...")

    # 图 1：观测序列与真实隐状态的时间序列图
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle("HMM 数据展示：观测序列与真实隐状态", fontsize=14, fontweight="bold")

    axes[0].step(data["time"], data["obs"], where="mid", color="#1E88E5", linewidth=1.2)
    axes[0].set_ylabel("观测符号")
    axes[0].grid(True, alpha=0.25)

    axes[1].step(
        data["time"],
        data["state_true"],
        where="mid",
        color="#D81B60",
        linewidth=1.2,
    )
    axes[1].set_xlabel("时间步")
    axes[1].set_ylabel("真实隐状态")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    _save_figure(fig, save_dir, "data_sequence.png")

    # 图 2：观测符号与真实隐状态分布
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    fig2.suptitle("HMM 数据展示：分布统计", fontsize=14, fontweight="bold")

    obs_counts = data["obs"].value_counts().sort_index()
    axes2[0].bar(obs_counts.index.astype(str), obs_counts.values, color="#1E88E5")
    axes2[0].set_title("观测符号分布")
    axes2[0].set_xlabel("obs")
    axes2[0].set_ylabel("频次")
    axes2[0].grid(True, axis="y", alpha=0.25)

    state_counts = data["state_true"].value_counts().sort_index()
    axes2[1].bar(state_counts.index.astype(str), state_counts.values, color="#D81B60")
    axes2[1].set_title("真实隐状态分布")
    axes2[1].set_xlabel("state_true")
    axes2[1].set_ylabel("频次")
    axes2[1].grid(True, axis="y", alpha=0.25)

    fig2.tight_layout()
    _save_figure(fig2, save_dir, "data_distribution.png")

    print("数据展示图生成完成。")


def show_result_preview(data, states_pred) -> None:
    """
    在终端展示部分序列结果

    这里直接看前若干个时间步的：
    1. 观测符号
    2. 真实隐状态
    3. 预测隐状态
    """
    preview_size = min(15, len(data))
    preview_df = data.head(preview_size).copy()
    preview_df["state_pred"] = states_pred[:preview_size]

    print()
    print("=" * 60)
    print("HMM 结果展示")
    print("=" * 60)
    print(preview_df.to_string(index=False))


def show_result_figure(data, states_pred) -> None:
    """
    生成 HMM 的结果展示图

    这张图直接把“观测序列 / 真实隐状态 / 预测隐状态”放在同一张图里，
    用于回答最直观的问题：
    “模型到底把整条序列预测成了什么样？”
    """
    save_dir = get_model_output_dir(MODEL)

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    fig.suptitle("HMM 结果展示", fontsize=14, fontweight="bold")

    axes[0].step(
        data["time"],
        data["obs"],
        where="mid",
        color="#1E88E5",
        linewidth=1.1,
    )
    axes[0].set_ylabel("观测符号")
    axes[0].set_title("观测序列")
    axes[0].grid(True, alpha=0.25)

    axes[1].step(
        data["time"],
        data["state_true"],
        where="mid",
        color="#2E7D32",
        linewidth=1.1,
    )
    axes[1].set_ylabel("真实隐状态")
    axes[1].set_title("真实隐状态序列")
    axes[1].grid(True, alpha=0.25)

    axes[2].step(
        data["time"],
        states_pred,
        where="mid",
        color="#D81B60",
        linewidth=1.1,
    )
    mismatch_mask = states_pred != data["state_true"].to_numpy()
    if mismatch_mask.any():
        axes[2].scatter(
            data["time"][mismatch_mask],
            states_pred[mismatch_mask],
            color="#FFC107",
            edgecolors="black",
            linewidths=0.3,
            s=25,
            zorder=3,
            label="预测错误点",
        )
        axes[2].legend(loc="best")
    axes[2].set_xlabel("时间步")
    axes[2].set_ylabel("预测隐状态")
    axes[2].set_title("预测隐状态序列")
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    _save_figure(fig, save_dir, "result_display.png")


def show_model_evaluation(model, X_obs, lengths, y_true, states_pred) -> None:
    """
    在终端展示 HMM 的模型评估结果

    这里重点展示：
    1. 隐状态预测准确率
    2. 转移矩阵
    3. 发射矩阵
    4. 对数似然
    """
    accuracy = np.mean(states_pred == y_true)
    log_likelihood = model.score(X_obs, lengths)

    print()
    print("=" * 60)
    print("HMM 模型评估展示")
    print("=" * 60)
    print(f"隐状态预测准确率: {accuracy:.4f}")
    print(f"对数似然: {log_likelihood:.4f}")
    print("转移矩阵:")
    print(np.round(model.transmat_, 4))

    if hasattr(model, "emissionprob_"):
        print("发射矩阵:")
        print(np.round(model.emissionprob_, 4))


def show_model_evaluation_figure(model, y_true, states_pred) -> None:
    """
    生成 HMM 的模型评估图

    这张图包含三部分：
    1. 真实隐状态 vs 预测隐状态的混淆矩阵；
    2. 学到的转移矩阵；
    3. 学到的发射矩阵。
    """
    save_dir = get_model_output_dir(MODEL)

    unique_states = sorted(np.unique(np.concatenate([y_true, states_pred])))
    confusion = np.zeros((len(unique_states), len(unique_states)), dtype=int)
    for true_state, pred_state in zip(y_true, states_pred, strict=True):
        confusion[true_state, pred_state] += 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle("HMM 模型评估", fontsize=14, fontweight="bold")

    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axes[0],
    )
    axes[0].set_title("隐状态混淆矩阵")
    axes[0].set_xlabel("预测隐状态")
    axes[0].set_ylabel("真实隐状态")

    sns.heatmap(
        model.transmat_,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        cbar=False,
        ax=axes[1],
    )
    axes[1].set_title("转移矩阵")
    axes[1].set_xlabel("下一状态")
    axes[1].set_ylabel("当前状态")

    if hasattr(model, "emissionprob_"):
        sns.heatmap(
            model.emissionprob_,
            annot=True,
            fmt=".3f",
            cmap="PuBuGn",
            cbar=False,
            ax=axes[2],
        )
        axes[2].set_title("发射矩阵")
        axes[2].set_xlabel("观测符号")
        axes[2].set_ylabel("隐状态")
    else:
        axes[2].axis("off")
        axes[2].set_title("无发射矩阵可视化")

    fig.tight_layout()
    _save_figure(fig, save_dir, "evaluation_display.png")


def run():
    """
    HMM 完整流水线

    当前流程分成：
    1. 数据探索；
    2. 数据展示；
    3. 模型训练；
    4. 结果展示；
    5. 模型评估。
    """
    print("=" * 60)
    print("HMM 流水线")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 第 1 步：读取当前算法对应的原始数据
    # ------------------------------------------------------------------
    # HMM 当前使用的是离散观测序列数据。
    # 这里的重点不是普通表格特征，而是“时间步上的状态演化”。
    data = hmm_data.copy()
    obs = data["obs"].values.astype(int)
    X_obs = obs.reshape(-1, 1)
    lengths = [len(obs)]
    y_true = data["state_true"].values.astype(int)

    # ------------------------------------------------------------------
    # 第 2 步：数据探索
    # ------------------------------------------------------------------
    show_data_exploration(data)

    # ------------------------------------------------------------------
    # 第 3 步：数据展示
    # ------------------------------------------------------------------
    show_data_preview(data)

    # ------------------------------------------------------------------
    # 第 4 步：训练
    # ------------------------------------------------------------------
    model = train_model(X_obs, lengths)

    # ------------------------------------------------------------------
    # 第 5 步：预测隐状态
    # ------------------------------------------------------------------
    states_pred = model.predict(X_obs, lengths)

    # ------------------------------------------------------------------
    # 第 6 步：结果展示
    # ------------------------------------------------------------------
    show_result_preview(data, states_pred)
    show_result_figure(data, states_pred)

    # ------------------------------------------------------------------
    # 第 7 步：模型评估
    # ------------------------------------------------------------------
    show_model_evaluation(model, X_obs, lengths, y_true, states_pred)
    show_model_evaluation_figure(model, y_true, states_pred)

    print(f"\n{'=' * 60}")
    print("HMM 流水线完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
