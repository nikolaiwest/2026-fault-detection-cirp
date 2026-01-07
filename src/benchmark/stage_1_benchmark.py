"""
Benchmark script for Stage 1 model comparison.

Evaluates all available Stage 1 anomaly detection models using cross-validation
with different contamination rates and saves results as CSV + plots.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data import load_class_config, load_pipeline_config, run_data_pipeline
from src.methodology.cross_validation import prepare_cv_folds
from src.methodology.stage1 import run_stage1
from src.models.stage_1 import STAGE1_MODELS

# Contamination rates to test
CONTAMINATION_RATES = np.arange(0.0025, 0.0525, 0.0025).tolist()

# Output directory
OUTPUT_DIR = Path("results") / "stage1_benchmark"


def run_stage1_benchmark(config_name: str = "default-top5.yml"):
    """
    Benchmark all Stage 1 models across different contamination rates.

    Saves results to CSV and generates comparison plots.

    Args:
        config_name: Name of YAML config file in configs/ directory

    Returns:
        None (saves results to OUTPUT_DIR)
    """
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAGE 1 MODEL BENCHMARK")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")

    # Load config and data
    config = load_pipeline_config(config_name)

    x_values, y_true, label_mapping = run_data_pipeline(
        force_reload=config.data.force_reload,
        keep_exceptions=config.data.keep_exceptions,
        classes_to_keep=load_class_config(config.data.classes_to_keep),
        paa_segments=config.data.paa_segments,
    )

    # Prepare CV folds
    cv_folds = prepare_cv_folds(
        x_values=x_values,
        y_values=y_true,
        n_splits=config.cross_validation.n_splits,
        target_nok_per_fold=config.cross_validation.target_nok_per_fold,
        target_ok_per_fold=config.cross_validation.target_ok_per_fold,
        random_state=config.cross_validation.random_state,
    )

    # Store results for DataFrame
    results_list = []

    # Iterate over contamination rates
    for contamination in CONTAMINATION_RATES:
        print(f"\n{'='*80}")
        print(f"CONTAMINATION RATE: {contamination:.2%}")
        print(f"{'='*80}")

        # Iterate over all available models
        for model_name in STAGE1_MODELS.keys():

            # if model_name in ["auto_encoder", "one_class_svm"]:  # slooow
            #    continue

            print(f"\nModel: {model_name}...", end=" ")

            fold_results = []
            fold_times = []

            # Run model on each fold
            for fold_num, (x_fold, y_fold) in enumerate(cv_folds, 1):
                # Start timing
                start_time = time.time()

                # Run Stage 1
                y_anomalies, anomaly_scores = run_stage1(
                    x_values=x_fold,
                    y_values=y_fold,
                    model_name=model_name,
                    contamination=contamination,
                    random_state=config.stage1.random_state,
                )

                # End timing
                elapsed_time = time.time() - start_time
                fold_times.append(elapsed_time)

                # Convert to binary
                y_true_binary = (y_fold > 0).astype(int)

                # Calculate metrics
                metrics = {
                    "precision": precision_score(
                        y_true_binary, y_anomalies, zero_division=0
                    ),
                    "recall": recall_score(y_true_binary, y_anomalies, zero_division=0),
                    "f1": f1_score(y_true_binary, y_anomalies, zero_division=0),
                    "accuracy": accuracy_score(y_true_binary, y_anomalies),
                }

                fold_results.append(metrics)

            # Aggregate across folds
            avg_metrics = {
                "model": model_name,
                "contamination": contamination,
                "precision_mean": np.mean([r["precision"] for r in fold_results]),
                "precision_std": np.std([r["precision"] for r in fold_results]),
                "recall_mean": np.mean([r["recall"] for r in fold_results]),
                "recall_std": np.std([r["recall"] for r in fold_results]),
                "f1_mean": np.mean([r["f1"] for r in fold_results]),
                "f1_std": np.std([r["f1"] for r in fold_results]),
                "accuracy_mean": np.mean([r["accuracy"] for r in fold_results]),
                "accuracy_std": np.std([r["accuracy"] for r in fold_results]),
                "time_mean": np.mean(fold_times),
                "time_std": np.std(fold_times),
                "time_total": np.sum(fold_times),
            }

            results_list.append(avg_metrics)
            print(
                f"F1={avg_metrics['f1_mean']:.3f}, Time={avg_metrics['time_mean']:.3f}s"
            )

    # Create DataFrame
    df = pd.DataFrame(results_list)

    # Save to CSV
    csv_path = OUTPUT_DIR / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")

    # Generate plots
    _generate_plots(df)

    # Print summary
    _print_summary(df)


def _generate_plots(df: pd.DataFrame):
    """Generate comparison plots using only matplotlib."""

    # Color palette
    colors = plt.cm.tab10(range(len(df["model"].unique())))

    # 1. F1-Score by Contamination (line plot)
    plt.figure(figsize=(9, 7))
    for i, model in enumerate(df["model"].unique()):
        model_data = df[df["model"] == model]
        plt.plot(
            model_data["contamination"],
            model_data["f1_mean"],
            marker="o",
            label=model,
            linewidth=2,
            color=colors[i],
        )
        plt.fill_between(
            model_data["contamination"],
            model_data["f1_mean"] - model_data["f1_std"],
            model_data["f1_mean"] + model_data["f1_std"],
            alpha=0.2,
            color=colors[i],
        )

    plt.xlabel("Contamination Rate", fontsize=12)
    plt.ylabel("F1-Score", fontsize=12)
    # plt.title("Model Performance vs Contamination Rate", fontsize=14, fontweight="bold")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "f1_by_contamination.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Plot saved: f1_by_contamination.png")

    # 2. Precision by Contamination (NEW)
    plt.figure(figsize=(9, 7))
    for i, model in enumerate(df["model"].unique()):
        model_data = df[df["model"] == model]
        plt.plot(
            model_data["contamination"],
            model_data["precision_mean"],
            marker="o",
            label=model,
            linewidth=2,
            color=colors[i],
        )
        plt.fill_between(
            model_data["contamination"],
            model_data["precision_mean"] - model_data["precision_std"],
            model_data["precision_mean"] + model_data["precision_std"],
            alpha=0.2,
            color=colors[i],
        )

    plt.xlabel("Contamination Rate", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision vs Contamination Rate", fontsize=14, fontweight="bold")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "precision_by_contamination.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"✓ Plot saved: precision_by_contamination.png")

    # 3. Recall by Contamination (NEW)
    plt.figure(figsize=(12, 6))
    for i, model in enumerate(df["model"].unique()):
        model_data = df[df["model"] == model]
        plt.plot(
            model_data["contamination"],
            model_data["recall_mean"],
            marker="o",
            label=model,
            linewidth=2,
            color=colors[i],
        )
        plt.fill_between(
            model_data["contamination"],
            model_data["recall_mean"] - model_data["recall_std"],
            model_data["recall_mean"] + model_data["recall_std"],
            alpha=0.2,
            color=colors[i],
        )

    plt.xlabel("Contamination Rate", fontsize=12)
    plt.ylabel("Recall", fontsize=12)
    plt.title("Recall vs Contamination Rate", fontsize=14, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "recall_by_contamination.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"✓ Plot saved: recall_by_contamination.png")

    # 4. Precision-Recall Trade-off
    plt.figure(figsize=(12, 6))
    for i, model in enumerate(df["model"].unique()):
        model_data = df[df["model"] == model]
        plt.plot(
            model_data["recall_mean"],
            model_data["precision_mean"],
            marker="o",
            label=model,
            linewidth=2,
            color=colors[i],
        )

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Trade-off", fontsize=14, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "precision_recall.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Plot saved: precision_recall.png")

    # 5. Heatmap of F1-scores
    pivot_df = df.pivot(index="model", columns="contamination", values="f1_mean")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot_df.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_xticklabels([f"{c:.0%}" for c in pivot_df.columns])
    ax.set_yticklabels(pivot_df.index)

    # Add text annotations
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            text = ax.text(
                j,
                i,
                f"{pivot_df.values[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

    ax.set_xlabel("Contamination Rate", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    ax.set_title(
        "F1-Score Heatmap: Models vs Contamination", fontsize=14, fontweight="bold"
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("F1-Score", fontsize=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "f1_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Plot saved: f1_heatmap.png")

    # 6. Best configuration per model (bar plot)
    best_per_model = df.loc[df.groupby("model")["f1_mean"].idxmax()]

    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = range(len(best_per_model))
    bars = ax.bar(
        x_pos,
        best_per_model["f1_mean"],
        yerr=best_per_model["f1_std"],
        capsize=5,
        color=colors[: len(best_per_model)],
        alpha=0.8,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(best_per_model["model"], rotation=45, ha="right")
    ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_title(
        "Best F1-Score per Model (optimal contamination)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")

    # Add contamination labels on bars
    for i, (idx, row) in enumerate(best_per_model.iterrows()):
        ax.text(
            i,
            row["f1_mean"] + 0.02,
            f"{row['contamination']:.0%}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "best_per_model.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Plot saved: best_per_model.png")

    # 7. Runtime comparison
    avg_time_per_model = df.groupby("model")["time_mean"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = range(len(avg_time_per_model))
    bars = ax.barh(
        x_pos,
        avg_time_per_model.values,
        color=colors[: len(avg_time_per_model)],
        alpha=0.8,
    )

    ax.set_yticks(x_pos)
    ax.set_yticklabels(avg_time_per_model.index)
    ax.set_xlabel("Average Runtime per Fold (seconds)", fontsize=12)
    ax.set_title("Model Runtime Comparison", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (model, time_val) in enumerate(avg_time_per_model.items()):
        ax.text(time_val + 0.01, i, f"{time_val:.3f}s", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "runtime_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Plot saved: runtime_comparison.png")

    # 8. F1 vs Runtime scatter
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, model in enumerate(df["model"].unique()):
        model_data = df[df["model"] == model]
        ax.scatter(
            model_data["time_mean"],
            model_data["f1_mean"],
            s=100,
            color=colors[i],
            label=model,
            alpha=0.7,
        )

        # Add contamination labels
        for _, row in model_data.iterrows():
            ax.annotate(
                f"{row['contamination']:.0%}",
                (row["time_mean"], row["f1_mean"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

    ax.set_xlabel("Runtime per Fold (seconds)", fontsize=12)
    ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_title("Performance vs Runtime Trade-off", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "f1_vs_runtime.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Plot saved: f1_vs_runtime.png")


def _print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Best overall configuration
    best_idx = df["f1_mean"].idxmax()
    best = df.loc[best_idx]

    print("\nBest Configuration (by F1-Score):")
    print(f"  Model: {best['model']}")
    print(f"  Contamination: {best['contamination']:.2%}")
    print(f"  F1-Score: {best['f1_mean']:.3f} ±{best['f1_std']:.3f}")
    print(f"  Precision: {best['precision_mean']:.3f} ±{best['precision_std']:.3f}")
    print(f"  Recall: {best['recall_mean']:.3f} ±{best['recall_std']:.3f}")
    print(f"  Runtime: {best['time_mean']:.3f}s ±{best['time_std']:.3f}s")

    # Fastest model
    fastest_idx = df["time_mean"].idxmin()
    fastest = df.loc[fastest_idx]

    print("\nFastest Configuration:")
    print(f"  Model: {fastest['model']}")
    print(f"  Contamination: {fastest['contamination']:.2%}")
    print(f"  F1-Score: {fastest['f1_mean']:.3f}")
    print(f"  Runtime: {fastest['time_mean']:.3f}s")

    # Best per model
    print("\nBest Configuration per Model:")
    best_per_model = df.loc[df.groupby("model")["f1_mean"].idxmax()]
    for _, row in best_per_model.iterrows():
        print(
            f"  {row['model']:<20} @ {row['contamination']:.2%}: F1={row['f1_mean']:.3f}, Time={row['time_mean']:.3f}s"
        )
