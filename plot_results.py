"""
plot_results.py  —  Generate paper-quality figures from CSV results
Output: results/figures/fig2_main_comparison.png
                        fig3_ablation.png
                        fig4_scalability.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs("results/figures", exist_ok=True)

COLORS = ["#e74c3c","#e67e22","#f1c40f","#2ecc71","#3498db","#9b59b6"]
AGENTS = ["GBA","SRA","GAPAF-E","GaaS-Agent","HITL-Agent","GUARD"]

def fig2_main():
    df = pd.read_csv("results/main_results.csv", index_col=0)
    metrics = ["AUROC","F1","AuditCompleteness","ProvenanceCoverage"]
    labels  = ["AUROC","F1 Score","Audit Completeness","Provenance Coverage"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Fig. 2  Main Comparison on CWRU Dataset", fontsize=14, fontweight="bold", y=1.01)
    for ax, m, lab in zip(axes.flat, metrics, labels):
        vals = [df.loc[a, m] if a in df.index else 0 for a in AGENTS]
        bars = ax.bar(AGENTS, vals, color=COLORS, edgecolor="white", linewidth=0.8)
        ax.set_title(lab, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Score"); ax.tick_params(axis="x", rotation=20)
        ax.axhline(0.9, ls="--", lw=0.8, color="gray", alpha=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.01,
                    f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
        bars[-1].set_edgecolor("#c0392b"); bars[-1].set_linewidth(2.5)
    plt.tight_layout()
    plt.savefig("results/figures/fig2_main_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(); print("✓ fig2_main_comparison.png")

def fig3_ablation():
    df = pd.read_csv("results/ablation_results.csv")
    configs = df["Config"].tolist()
    metrics = ["AUROC","F1","AuditCompleteness","AutonomyRate"]
    labels  = ["AUROC","F1 Score","Audit Completeness","Autonomy Rate"]
    colors  = ["#3498db" if c == "GUARD-Full" else "#95a5a6" for c in configs]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Fig. 3  Ablation Study", fontsize=14, fontweight="bold", y=1.01)
    for ax, m, lab in zip(axes.flat, metrics, labels):
        vals = df[m].values
        bars = ax.bar(configs, vals, color=colors, edgecolor="white")
        ax.set_title(lab, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.08); ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=25)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.01,
                    f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig("results/figures/fig3_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(); print("✓ fig3_ablation.png")

def fig4_scalability():
    df = pd.read_csv("results/scalability_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Fig. 4  Scalability Analysis", fontsize=14, fontweight="bold")
    ax = axes[0]
    ax.plot(df["BatchSize"], df["Latency_ms_per_sample"],
            marker="o", color="#9b59b6", lw=2)
    ax.set_xlabel("Batch Size"); ax.set_ylabel("Latency (ms/sample)")
    ax.set_title("Inference Latency vs Batch Size")
    ax2 = axes[1]
    ax2.plot(df["BatchSize"], df["AuditCompleteness"],
             marker="s", color="#27ae60", lw=2)
    ax2.set_xlabel("Batch Size"); ax2.set_ylabel("Audit Completeness")
    ax2.set_title("Audit Completeness vs Batch Size")
    ax2.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig("results/figures/fig4_scalability.png", dpi=150, bbox_inches="tight")
    plt.close(); print("✓ fig4_scalability.png")

if __name__ == "__main__":
    try: fig2_main()
    except Exception as e: print(f"fig2 error: {e}")
    try: fig3_ablation()
    except Exception as e: print(f"fig3 error: {e}")
    try: fig4_scalability()
    except Exception as e: print(f"fig4 error: {e}")
    print("All figures saved to results/figures/")
