"""
Function to create a 2x2 grid plot for anomaly detection benchmark results.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def create_benchmark_plot(csv_path: str, output_path: str = None):
    """
    Create a 2x2 grid plot showing F1, Precision, Recall, and FPR vs Contamination Rate.

    Args:
        csv_path: Path to the CSV file with benchmark results
        output_path: Optional path to save the plot. If None, uses same directory as CSV

    Returns:
        None (saves plot to file)
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Set up the figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    # Define color palette
    colors = plt.cm.tab10(range(len(df["model"].unique())))

    # Plot 1: F1 Score vs Contamination Rate (top-left)
    ax1 = axes[0, 0]
    for i, model in enumerate(df["model"].unique()):
        model_data = df[df["model"] == model]
        ax1.plot(
            model_data["contamination"],
            model_data["f1_mean"],
            marker="o",
            label=model,
            linewidth=2,
            color=colors[i],
            markersize=4,
        )
    ax1.set_xlabel("Contamination Rate", fontsize=9)
    ax1.set_ylabel("F1 Score (Mean/Macro)", fontsize=9)
    ax1.set_title("F1 Score vs Contamination Rate", fontsize=10, fontweight="bold")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    # Add label
    ax1.text(
        0.02,
        0.02,
        "a)",
        transform=ax1.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    # Plot 2: Precision vs Contamination Rate (top-right)
    ax2 = axes[0, 1]
    for i, model in enumerate(df["model"].unique()):
        model_data = df[df["model"] == model]
        ax2.plot(
            model_data["contamination"],
            model_data["precision_mean"],
            marker="o",
            label=model,
            linewidth=2,
            color=colors[i],
            markersize=4,
        )
    ax2.set_xlabel("Contamination Rate", fontsize=9)
    ax2.set_ylabel("Precision", fontsize=9)
    ax2.set_title("Precision vs Contamination Rate", fontsize=10, fontweight="bold")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    # Add label
    ax2.text(
        0.02,
        0.02,
        "b)",
        transform=ax2.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    # Plot 3: Recall vs Contamination Rate (bottom-left)
    ax3 = axes[1, 0]
    for i, model in enumerate(df["model"].unique()):
        model_data = df[df["model"] == model]
        ax3.plot(
            model_data["contamination"],
            model_data["recall_mean"],
            marker="o",
            label=model,
            linewidth=2,
            color=colors[i],
            markersize=4,
        )
    ax3.set_xlabel("Contamination Rate", fontsize=9)
    ax3.set_ylabel("Recall", fontsize=9)
    ax3.set_title("Recall vs Contamination Rate", fontsize=10, fontweight="bold")
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)
    # Add label
    ax3.text(
        0.02,
        0.02,
        "c)",
        transform=ax3.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    # Plot 4: Recall vs Precision (bottom-right)
    ax4 = axes[1, 1]
    for i, model in enumerate(df["model"].unique()):
        model_data = df[df["model"] == model]
        ax4.plot(
            model_data["recall_mean"],
            model_data["precision_mean"],
            marker="o",
            label=model,
            linewidth=2,
            color=colors[i],
            markersize=4,
        )
    ax4.set_xlabel("Recall", fontsize=9)
    ax4.set_ylabel("Precision", fontsize=9)
    ax4.set_title("Precision vs Recall", fontsize=10, fontweight="bold")
    ax4.legend(loc="best", fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1.05)
    ax4.set_ylim(0, 1.05)
    # Add label
    ax4.text(
        0.02,
        0.02,
        "d)",
        transform=ax4.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    # Adjust layout
    plt.tight_layout()

    # Save plot
    if output_path is None:
        output_path = Path(csv_path).parent / "benchmark_2x2_plot.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.close()
