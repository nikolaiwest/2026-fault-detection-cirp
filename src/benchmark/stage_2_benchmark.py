"""
Benchmark script for Stage 2 model comparison.

Evaluates all available Stage 2 clustering models using cross-validation
with simulated Stage 1 output (controllable false positive rate).
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.data import load_class_config, load_pipeline_config, run_data_pipeline
from src.methodology.cross_validation import prepare_cv_folds
from src.methodology.stage2 import run_stage2
from src.models.stage_2 import STAGE2_MODELS

# Parameters to test
N_CLUSTERS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
OK_THRESHOLDS = [0]
METRICS = ["msm"]
FALSE_POSITIVE_RATES = [0.01]  # Simulated Stage 1 FP rates
MODELS = [
    # "sklearn_kmeans",
    "sklearn_dbscan",
    # "sklearn_birch",
    # "sklearn_agglomerative",
    "sktime_kmeans",
    # "sktime_kmedoids",
    # "sktime_dbscan",
    # "sktime_kshapes",
]
TARGET_OKS = [25]  # Number of OK references to sample


# Output directory
OUTPUT_DIR = Path("results") / "stage2_benchmark"


def run_stage2_benchmark(config_name: str = "default-top5.yml"):
    """
    Benchmark all Stage 2 models across different configurations.
    Uses simulated Stage 1 output with controllable false positive rates.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("STAGE 2 MODEL BENCHMARK")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Simulated Stage 1 FP rates: {FALSE_POSITIVE_RATES}")

    # Load config and data
    config = load_pipeline_config(config_name)
    x_values, y_values, label_mapping = run_data_pipeline(
        force_reload=config.data.force_reload,
        keep_exceptions=config.data.keep_exceptions,
        classes_to_keep=load_class_config(config.data.classes_to_keep),
        paa_segments=config.data.paa_segments,
    )

    # Prepare CV folds
    cv_folds = prepare_cv_folds(
        x_values=x_values,
        y_values=y_values,
        n_splits=config.cross_validation.n_splits,
        target_nok_per_fold=config.cross_validation.target_nok_per_fold,
        target_ok_per_fold=config.cross_validation.target_ok_per_fold,
        random_state=config.cross_validation.random_state,
    )

    # Store results
    results_list = []

    total = (
        len(MODELS)
        * len(TARGET_OKS)
        * len(METRICS)
        * len(N_CLUSTERS)
        * len(OK_THRESHOLDS)
        * len(FALSE_POSITIVE_RATES)
    )
    print(f"\nTotal configurations: {total}")

    idx = 0
    for model_name in MODELS:
        for target_ok in TARGET_OKS:
            for metric in METRICS:
                for n_cluster in N_CLUSTERS:
                    for ok_threshold in OK_THRESHOLDS:
                        for fp_rate in FALSE_POSITIVE_RATES:
                            idx += 1
                            print(
                                f"\n[{idx}/{total}] {model_name} | ok={target_ok} | metric={metric} | "
                                f"clusters={n_cluster} | threshold={ok_threshold} | FP={fp_rate:.1%}"
                            )

                            fold_metrics = []

                            # Run on each CV fold
                            for fold_num, (x_fold, y_fold) in enumerate(cv_folds, 1):
                                # Simulate Stage 1
                                y_anomalies = _simulate_stage1_with_fp(
                                    y_values=y_fold,
                                    false_positive_rate=fp_rate,
                                    random_state=42 + fold_num,
                                )

                                # Run Stage 2
                                start_time = time.time()
                                stage2_predictions = run_stage2(
                                    x_values=x_fold,
                                    y_anomalies=y_anomalies,
                                    y_true=y_fold,
                                    label_mapping=label_mapping,
                                    model_name=model_name,
                                    target_ok_to_sample=target_ok,
                                    metric=metric,
                                    ok_reference_threshold=ok_threshold,
                                    n_clusters=n_cluster,
                                    random_state=42,
                                )
                                elapsed = time.time() - start_time

                                # Evaluate: Reconstruct clustering indices to match stage2_predictions
                                nok_indices = np.where(y_anomalies == 1)[0]
                                ok_indices = np.where(y_anomalies == 0)[0]
                                n_pred = len(stage2_predictions)

                                if n_pred > 0:
                                    # Reconstruct the exact OK sampling that stage2 did
                                    n_ok_sample = min(target_ok, len(ok_indices))
                                    np.random.seed(42)  # Match stage2 random_state
                                    ok_sample_indices = np.random.choice(
                                        ok_indices, size=n_ok_sample, replace=False
                                    )
                                    clustering_indices = np.concatenate(
                                        [nok_indices, ok_sample_indices]
                                    )

                                    # Verify size match
                                    if len(clustering_indices) == n_pred:
                                        y_binary = (
                                            y_fold[clustering_indices] > 0
                                        ).astype(int)

                                        prec = precision_score(
                                            y_binary,
                                            stage2_predictions,
                                            zero_division=0,
                                        )
                                        rec = recall_score(
                                            y_binary,
                                            stage2_predictions,
                                            zero_division=0,
                                        )
                                        f1 = f1_score(
                                            y_binary,
                                            stage2_predictions,
                                            zero_division=0,
                                        )

                                        fold_metrics.append(
                                            {
                                                "precision": prec,
                                                "recall": rec,
                                                "f1": f1,
                                                "time": elapsed,
                                            }
                                        )
                                    else:
                                        print(
                                            f"    Warning: Size mismatch - predictions={n_pred}, expected={len(clustering_indices)}"
                                        )

                            # Aggregate
                            if fold_metrics:
                                results_list.append(
                                    {
                                        "model": model_name,
                                        "target_ok": target_ok,
                                        "metric": metric,
                                        "n_clusters": n_cluster,
                                        "ok_threshold": ok_threshold,
                                        "fp_rate": fp_rate,
                                        "f1_mean": np.mean(
                                            [m["f1"] for m in fold_metrics]
                                        ),
                                        "f1_std": np.std(
                                            [m["f1"] for m in fold_metrics]
                                        ),
                                        "precision_mean": np.mean(
                                            [m["precision"] for m in fold_metrics]
                                        ),
                                        "precision_std": np.std(
                                            [m["precision"] for m in fold_metrics]
                                        ),
                                        "recall_mean": np.mean(
                                            [m["recall"] for m in fold_metrics]
                                        ),
                                        "recall_std": np.std(
                                            [m["recall"] for m in fold_metrics]
                                        ),
                                        "avg_time": np.mean(
                                            [m["time"] for m in fold_metrics]
                                        ),
                                    }
                                )

                                print(
                                    f"  F1={results_list[-1]['f1_mean']:.3f}, Time={results_list[-1]['avg_time']:.2f}s"
                                )
                            else:
                                print(f"  No valid predictions")

    # Save results
    df = pd.DataFrame(results_list)
    csv_path = OUTPUT_DIR / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")

    # Generate plots
    _generate_plots(df)


def _simulate_stage1_with_fp(
    y_values: np.ndarray, false_positive_rate: float, random_state: int = 42
) -> np.ndarray:
    """
    Simulate Stage 1 output with controllable false positive rate.

    Assumes perfect recall (all true NOKs detected), then adds false positives
    by randomly flagging OK samples as anomalies.

    Args:
        y_values: Ground truth labels
        false_positive_rate: Fraction of OK samples to flag as anomalies
        random_state: Random seed for reproducibility

    Returns:
        Simulated Stage 1 predictions (0=OK, 1=NOK)
    """
    rng = np.random.RandomState(random_state)

    # Start with perfect recall: all true NOKs detected
    y_anomalies = (y_values > 0).astype(int)

    # Inject false positives
    ok_mask = y_values == 0
    ok_indices = np.where(ok_mask)[0]
    n_fps = int(len(ok_indices) * false_positive_rate)

    if n_fps > 0:
        fp_indices = rng.choice(ok_indices, size=n_fps, replace=False)
        y_anomalies[fp_indices] = 1

    return y_anomalies


def _generate_plots(df: pd.DataFrame):
    """Generate comparison plots for Stage 2 benchmark."""

    # Check if DataFrame is empty or missing required columns
    if df.empty or "model" not in df.columns:
        print("⚠ No results to plot (empty DataFrame or missing data)")
        return

    colors = plt.cm.tab10(range(len(df["model"].unique())))

    # 1. Average Time by Model
    plt.figure(figsize=(12, 6))
    avg_time = df.groupby("model")["avg_time"].mean().sort_values()

    bars = plt.barh(
        range(len(avg_time)), avg_time.values, color=colors[: len(avg_time)]
    )
    plt.yticks(range(len(avg_time)), avg_time.index)
    plt.xlabel("Average Runtime (seconds)", fontsize=12)
    plt.title("Stage 2 Runtime by Model", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="x")

    for i, val in enumerate(avg_time.values):
        plt.text(val + 0.1, i, f"{val:.2f}s", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "runtime_by_model.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Plot saved: runtime_by_model.png")

    # 2. Runtime by Number of Clusters
    plt.figure(figsize=(12, 6))
    for i, model in enumerate(df["model"].unique()):
        model_data = df[df["model"] == model]
        cluster_avg = model_data.groupby("n_clusters")["avg_time"].mean()
        plt.plot(
            cluster_avg.index,
            cluster_avg.values,
            marker="o",
            label=model,
            linewidth=2,
            color=colors[i],
        )

    plt.xlabel("Number of Clusters", fontsize=12)
    plt.ylabel("Average Runtime (seconds)", fontsize=12)
    plt.title("Runtime vs Number of Clusters", fontsize=14, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "runtime_vs_clusters.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Plot saved: runtime_vs_clusters.png")

    # 3. Runtime by Metric
    plt.figure(figsize=(12, 6))
    for i, model in enumerate(df["model"].unique()):
        model_data = df[df["model"] == model]
        metric_avg = model_data.groupby("metric")["avg_time"].mean()
        x_pos = range(len(metric_avg))
        plt.bar(
            [x + i * 0.2 for x in x_pos],
            metric_avg.values,
            width=0.2,
            label=model,
            color=colors[i],
            alpha=0.8,
        )

    plt.xticks([x + 0.2 for x in range(len(metric_avg))], metric_avg.index)
    plt.ylabel("Average Runtime (seconds)", fontsize=12)
    plt.xlabel("Distance Metric", fontsize=12)
    plt.title("Runtime by Distance Metric", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "runtime_by_metric.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Plot saved: runtime_by_metric.png")

    # 4. Runtime by False Positive Rate
    plt.figure(figsize=(12, 6))
    for i, model in enumerate(df["model"].unique()):
        model_data = df[df["model"] == model]
        fp_avg = model_data.groupby("fp_rate")["avg_time"].mean()
        plt.plot(
            [f"{x:.1%}" for x in fp_avg.index],
            fp_avg.values,
            marker="o",
            label=model,
            linewidth=2,
            color=colors[i],
        )

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("Average Runtime (seconds)", fontsize=12)
    plt.title("Runtime vs False Positive Rate", fontsize=14, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "runtime_vs_fp_rate.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Plot saved: runtime_vs_fp_rate.png")

    # 6. Parameter Interaction: Clusters vs OK Threshold
    fig, axes = plt.subplots(1, len(df["model"].unique()), figsize=(18, 5))
    if len(df["model"].unique()) == 1:
        axes = [axes]

    for idx, (ax, model) in enumerate(zip(axes, df["model"].unique())):
        model_data = df[df["model"] == model]
        pivot = (
            model_data.groupby(["n_clusters", "ok_threshold"])["avg_time"]
            .mean()
            .unstack()
        )

        im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("OK Threshold", fontsize=10)
        ax.set_ylabel("N Clusters", fontsize=10)
        ax.set_title(f"{model}", fontsize=12, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Runtime (s)", fontsize=9)

    plt.suptitle(
        "Runtime: Clusters vs OK Threshold by Model", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "interaction_clusters_threshold.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("✓ Plot saved: interaction_clusters_threshold.png")

    # 7. F1 Score by Number of Clusters
    plt.figure(figsize=(12, 6))
    for i, model in enumerate(df["model"].unique()):
        model_data = df[df["model"] == model]
        cluster_avg = model_data.groupby("n_clusters")["f1_mean"].mean()
        cluster_std = model_data.groupby("n_clusters")["f1_std"].mean()

        plt.plot(
            cluster_avg.index,
            cluster_avg.values,
            marker="o",
            label=model,
            linewidth=2,
            color=colors[i],
        )
        plt.fill_between(
            cluster_avg.index,
            cluster_avg.values - cluster_std.values,
            cluster_avg.values + cluster_std.values,
            alpha=0.2,
            color=colors[i],
        )

    plt.xlabel("Number of Clusters", fontsize=12)
    plt.ylabel("F1-Score", fontsize=12)
    plt.title("F1-Score vs Number of Clusters", fontsize=14, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "f1_vs_clusters.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Plot saved: f1_vs_clusters.png")

    # 8. F1 Score by OK Threshold
    plt.figure(figsize=(12, 6))
    for i, model in enumerate(df["model"].unique()):
        model_data = df[df["model"] == model]
        threshold_avg = model_data.groupby("ok_threshold")["f1_mean"].mean()
        threshold_std = model_data.groupby("ok_threshold")["f1_std"].mean()

        plt.plot(
            threshold_avg.index,
            threshold_avg.values,
            marker="o",
            label=model,
            linewidth=2,
            color=colors[i],
        )
        plt.fill_between(
            threshold_avg.index,
            threshold_avg.values - threshold_std.values,
            threshold_avg.values + threshold_std.values,
            alpha=0.2,
            color=colors[i],
        )

    plt.xlabel("OK Reference Threshold", fontsize=12)
    plt.ylabel("F1-Score", fontsize=12)
    plt.title("F1-Score vs OK Reference Threshold", fontsize=14, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "f1_vs_threshold.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Plot saved: f1_vs_threshold.png")

    # 9. F1 Score by False Positive Rate
    plt.figure(figsize=(12, 6))
    for i, model in enumerate(df["model"].unique()):
        model_data = df[df["model"] == model]
        fp_avg = model_data.groupby("fp_rate")["f1_mean"].mean()
        fp_std = model_data.groupby("fp_rate")["f1_std"].mean()

        plt.plot(
            fp_avg.index,
            fp_avg.values,
            marker="o",
            label=model,
            linewidth=2,
            color=colors[i],
        )
        plt.fill_between(
            fp_avg.index,
            fp_avg.values - fp_std.values,
            fp_avg.values + fp_std.values,
            alpha=0.2,
            color=colors[i],
        )

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("F1-Score", fontsize=12)
    plt.title("F1-Score vs False Positive Rate", fontsize=14, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    # Format x-axis as percentages
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "f1_vs_fp_rate.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Plot saved: f1_vs_fp_rate.png")


if __name__ == "__main__":
    run_stage2_benchmark()
