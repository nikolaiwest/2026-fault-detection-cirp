"""
Function to load and plot confusion matrices from stage directories.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path


def plot_stage_confusion_matrices(main_dir, output_path=None):
    """
    Load confusion matrices from stage1 and stage2 directories and plot them side by side.

    Directory structure expected:
        main_dir/
            stage1/
                fold_1/metrics.json
                fold_2/metrics.json
                ...
                fold_5/metrics.json
            stage2/
                fold_1/metrics.json
                fold_2/metrics.json
                ...
                fold_5/metrics.json

    Each metrics.json contains a "confusion_matrix" field with format: [[TN, FP], [FN, TP]]

    Args:
        main_dir: Path to the main directory containing stage1 and stage2 folders
        output_path: Optional path to save the plot. If None, saves to main_dir/confusion_matrices.png

    Returns:
        None (saves plot to file)
    """
    main_dir = Path(main_dir)

    # Define stages
    stages = ["stage1", "stage2"]
    stage_cms = {}

    # Load confusion matrices for each stage
    for stage in stages:
        stage_dir = main_dir / stage

        if not stage_dir.exists():
            raise FileNotFoundError(f"Stage directory not found: {stage_dir}")

        # Initialize accumulator for confusion matrix
        cm_sum = np.zeros((2, 2), dtype=int)
        n_folds = 0

        # Load all fold metrics
        for fold_num in range(1, 6):  # fold_1 to fold_5
            fold_dir = stage_dir / f"fold_{fold_num}"
            metrics_file = fold_dir / "metrics.json"

            if not metrics_file.exists():
                print(f"Warning: Metrics file not found: {metrics_file}")
                continue

            # Load JSON
            with open(metrics_file, "r") as f:
                metrics = json.load(f)

            # Extract confusion matrix and add to sum
            cm = np.array(metrics["confusion_matrix"])
            cm_sum += cm
            n_folds += 1

        if n_folds == 0:
            raise ValueError(f"No valid folds found in {stage_dir}")

        # Store aggregated confusion matrix
        stage_cms[stage] = cm_sum
        print(f"✓ Loaded {stage}: {n_folds} folds, Total samples: {cm_sum.sum()}")

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Plot Stage 1
    ax1 = axes[0]
    cm1 = stage_cms["stage1"]
    sns.heatmap(
        cm1,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=["OK (0)", "NOK (1)"],
        yticklabels=["OK (0)", "NOK (1)"],
        ax=ax1,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax1.set_title(
        "Confusion Matrix after Stage 1", fontsize=10, fontweight="bold", pad=5
    )
    ax1.set_ylabel("True Label", fontsize=9)
    ax1.set_xlabel("Predicted Label", fontsize=9)

    # Add label and metrics
    ax1.text(
        -0.1,
        1.05,
        "a)",
        transform=ax1.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # Calculate and display metrics for Stage 1
    tn1, fp1, fn1, tp1 = cm1.ravel()
    precision1 = tp1 / (tp1 + fp1) if (tp1 + fp1) > 0 else 0
    recall1 = tp1 / (tp1 + fn1) if (tp1 + fn1) > 0 else 0
    f1_1 = (
        2 * (precision1 * recall1) / (precision1 + recall1)
        if (precision1 + recall1) > 0
        else 0
    )

    # Plot Stage 2
    ax2 = axes[1]
    cm2 = stage_cms["stage2"]
    sns.heatmap(
        cm2,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=["OK (0)", "NOK (1)"],
        yticklabels=["OK (0)", "NOK (1)"],
        ax=ax2,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax2.set_title(
        "Confusion Matrix after Stage 2", fontsize=10, fontweight="bold", pad=5
    )
    ax2.set_ylabel("True Label", fontsize=9)
    ax2.set_xlabel("Predicted Label", fontsize=9)

    # Add label
    ax2.text(
        -0.1,
        1.05,
        "b)",
        transform=ax2.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # Calculate and display metrics for Stage 2
    tn2, fp2, fn2, tp2 = cm2.ravel()
    precision2 = tp2 / (tp2 + fp2) if (tp2 + fp2) > 0 else 0
    recall2 = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0
    f1_2 = (
        2 * (precision2 * recall2) / (precision2 + recall2)
        if (precision2 + recall2) > 0
        else 0
    )

    # Adjust layout
    plt.tight_layout()

    # Save plot
    if output_path is None:
        output_path = main_dir / "confusion_matrices.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Confusion matrices plot saved to: {output_path}")
    plt.close()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nStage 1:")
    print(f"  Precision: {precision1:.3f}")
    print(f"  Recall:    {recall1:.3f}")
    print(f"  F1-Score:  {f1_1:.3f}")
    print(f"  Total:     {cm1.sum()} samples")

    print(f"\nStage 2:")
    print(f"  Precision: {precision2:.3f}")
    print(f"  Recall:    {recall2:.3f}")
    print(f"  F1-Score:  {f1_2:.3f}")
    print(f"  Total:     {cm2.sum()} samples")
