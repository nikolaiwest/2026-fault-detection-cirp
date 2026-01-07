"""
Benchmark script for unsupervised anomaly detection models (Stage 1).

Evaluates eight PyOD models using cross-validation to detect anomalies without labels.
Tests different contamination rates to find optimal thresholds.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from sklearn.metrics import f1_score, precision_score, recall_score

from src.data import load_class_config, load_pipeline_config, run_data_pipeline
from src.methodology.cross_validation import prepare_cv_folds

# Contamination rates to test
CONTAMINATION_RATES = np.arange(0.0025, 0.0525, 0.0025).tolist()

# Output directory
OUTPUT_DIR = Path("results") / "semi_supervised_classification"


def get_pyod_models(contamination: float):
    """
    Get dictionary of PyOD models with specified contamination rate.

    Args:
        contamination: Expected proportion of anomalies in the dataset

    Returns:
        Dictionary of model_name -> model_instance
    """
    return {
        "IsolationForest": IForest(contamination=contamination, random_state=42),
        "ECOD": ECOD(contamination=contamination),
        "HBOS": HBOS(contamination=contamination),
        "COPOD": COPOD(contamination=contamination),
        "LOF": LOF(contamination=contamination),
        "KNN": KNN(contamination=contamination),
        "AutoEncoder": AutoEncoder(contamination=contamination),
        "OC-SVM": OCSVM(contamination=contamination),
    }


def run_semi_supervised_classification_benchmark(
    config_name: str = "default-top5.yml",
    test_contamination_rates: bool = True,
):
    """
    Benchmark unsupervised anomaly detection models using cross-validation.

    Args:
        config_name: Name of YAML config file in configs/ directory
        test_contamination_rates: If True, test multiple contamination rates

    Returns:
        None (saves results to OUTPUT_DIR)
    """
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ANOMALY DETECTION BENCHMARK (UNSUPERVISED)")
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

    # Convert to binary labels for evaluation (0=normal, 1=anomaly)
    y_binary = (y_true > 0).astype(int)

    # Prepare CV folds
    cv_folds = prepare_cv_folds(
        x_values=x_values,
        y_values=y_true,
        n_splits=config.cross_validation.n_splits,
        target_nok_per_fold=config.cross_validation.target_nok_per_fold,
        target_ok_per_fold=config.cross_validation.target_ok_per_fold,
        random_state=config.cross_validation.random_state,
    )

    print(f"\nData shape: {x_values.shape}")
    print(f"Number of folds: {len(cv_folds)}")
    print(f"Normal samples: {(y_binary == 0).sum()}")
    print(f"Anomaly samples: {(y_binary == 1).sum()}")
    print(f"Actual contamination: {y_binary.mean():.4f}")

    # Determine contamination rates to test
    if test_contamination_rates:
        contamination_rates = CONTAMINATION_RATES
        print(f"Testing {len(contamination_rates)} contamination rates")
    else:
        # Use single contamination rate from config or default
        contamination_rates = [0.01]
        print(f"Using single contamination rate: {contamination_rates[0]}")

    # Store results
    results_list = []

    # Iterate over contamination rates
    for contamination in contamination_rates:
        print(f"\n{'='*80}")
        print(f"CONTAMINATION RATE: {contamination:.4f} ({contamination*100:.2f}%)")
        print(f"{'='*80}")

        # Get models for this contamination rate
        models = get_pyod_models(contamination)

        # Iterate over all models
        for model_name, model_template in models.items():
            print(f"\n  Model: {model_name}...", end=" ")

            fold_results = []
            fold_times = []

            # Run model on each fold
            for fold_num, (x_fold, y_fold) in enumerate(cv_folds, 1):
                # Convert to binary for this fold
                y_fold_binary = (y_fold > 0).astype(int)

                # Split fold into train/test (80/20)
                n_samples = len(x_fold)
                n_train = int(0.8 * n_samples)

                # Create indices
                indices = np.arange(n_samples)
                np.random.seed(42 + fold_num)  # Different seed per fold
                np.random.shuffle(indices)

                train_idx = indices[:n_train]
                test_idx = indices[n_train:]

                X_train = x_fold[train_idx]
                X_test = x_fold[test_idx]
                y_test = y_fold_binary[test_idx]

                try:
                    # Start timing
                    start_time = time.time()

                    # Clone model (PyOD models can be reused, but safer to clone)
                    from sklearn.base import clone

                    model = clone(model_template)

                    # Train (unsupervised - no labels used!)
                    model.fit(X_train)

                    # Predict on test set
                    y_pred = model.predict(X_test)  # 0=normal, 1=anomaly

                    # End timing
                    elapsed_time = time.time() - start_time
                    fold_times.append(elapsed_time)

                    # Calculate metrics
                    metrics = {
                        "precision": precision_score(y_test, y_pred, zero_division=0),
                        "recall": recall_score(y_test, y_pred, zero_division=0),
                        "f1": f1_score(
                            y_test, y_pred, zero_division=0, average="macro"
                        ),
                    }

                    # Also calculate false positive rate
                    if (y_test == 0).sum() > 0:
                        fpr = (y_pred[y_test == 0] == 1).sum() / (y_test == 0).sum()
                    else:
                        fpr = 0.0

                    metrics["fpr"] = fpr

                    fold_results.append(metrics)

                except Exception as e:
                    print(f"FAILED (fold {fold_num}): {e}", end=" ")
                    # Append NaN results for failed fold
                    fold_results.append(
                        {
                            "precision": np.nan,
                            "recall": np.nan,
                            "f1": np.nan,
                            "fpr": np.nan,
                        }
                    )
                    fold_times.append(np.nan)

            # Aggregate across folds
            avg_metrics = {
                "model": model_name,
                "contamination": contamination,
                "precision_mean": np.nanmean([r["precision"] for r in fold_results]),
                "precision_std": np.nanstd([r["precision"] for r in fold_results]),
                "recall_mean": np.nanmean([r["recall"] for r in fold_results]),
                "recall_std": np.nanstd([r["recall"] for r in fold_results]),
                "f1_mean": np.nanmean([r["f1"] for r in fold_results]),
                "f1_std": np.nanstd([r["f1"] for r in fold_results]),
                "fpr_mean": np.nanmean([r["fpr"] for r in fold_results]),
                "fpr_std": np.nanstd([r["fpr"] for r in fold_results]),
                "time_mean": np.nanmean(fold_times),
                "time_std": np.nanstd(fold_times),
                "time_total": np.nansum(fold_times),
            }

            results_list.append(avg_metrics)

            print(
                f"F1={avg_metrics['f1_mean']:.3f} ± {avg_metrics['f1_std']:.3f}, "
                f"FPR={avg_metrics['fpr_mean']:.3f}, "
                f"Time={avg_metrics['time_mean']:.2f}s"
            )

    # Create DataFrame
    df = pd.DataFrame(results_list)

    # Save to CSV
    csv_path = OUTPUT_DIR / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")

    # Generate plots
    _generate_plots(df, test_contamination_rates)

    # Print summary
    _print_summary(df, test_contamination_rates)

    return df


def _generate_plots(df: pd.DataFrame, test_contamination_rates: bool):
    """Generate comparison plots."""

    if test_contamination_rates:
        # Plot F1 vs contamination rate for each model
        fig, axes = plt.subplots(2, 2, figsize=(9, 9))

        models = df["model"].unique()

        # Plot 1: F1 vs Contamination
        ax = axes[0, 0]
        for model in models:
            model_data = df[df["model"] == model]
            ax.plot(
                model_data["contamination"],
                model_data["f1_mean"],
                marker="o",
                label=model,
            )
            # ax.fill_between(
            #    model_data["contamination"],
            #    model_data["f1_mean"] - model_data["f1_std"],
            #    model_data["f1_mean"] + model_data["f1_std"],
            #    alpha=0.2,
            # )
        ax.set_xlabel("Contamination Rate")
        ax.set_ylabel("F1-Score (macro avg.)")
        ax.set_title("F1 Score vs Contamination Rate")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1])

        # Plot 2: Precision vs Contamination
        ax = axes[0, 1]
        for model in models:
            model_data = df[df["model"] == model]
            ax.plot(
                model_data["contamination"],
                model_data["precision_mean"],
                marker="o",
                label=model,
            )
        ax.set_xlabel("Contamination Rate")
        ax.set_ylabel("Precision")
        ax.set_title("Precision vs Contamination Rate")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1])

        # Plot 3: Recall vs Contamination
        ax = axes[1, 0]
        for model in models:
            model_data = df[df["model"] == model]
            ax.plot(
                model_data["contamination"],
                model_data["recall_mean"],
                marker="o",
                label=model,
            )
        ax.set_xlabel("Contamination Rate")
        ax.set_ylabel("Recall")
        ax.set_title("Recall vs Contamination Rate")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1])

        # Plot 4: FPR vs Contamination
        ax = axes[1, 1]
        for model in models:
            model_data = df[df["model"] == model]
            ax.plot(
                model_data["contamination"],
                model_data["fpr_mean"],
                marker="o",
                label=model,
            )
        ax.set_xlabel("Contamination Rate")
        ax.set_ylabel("False Positive Rate")
        ax.set_title("FPR vs Contamination Rate")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plot_path = OUTPUT_DIR / "benchmark_contamination_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"✓ Contamination analysis plot saved to: {plot_path}")
        plt.close()

    # Bar chart comparison at best contamination rate for each model
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))

    # Find best contamination rate for each model (by F1)
    best_results = []
    for model in df["model"].unique():
        model_data = df[df["model"] == model]
        best_idx = model_data["f1_mean"].idxmax()
        best_results.append(model_data.loc[best_idx])

    df_best = pd.DataFrame(best_results)
    models = df_best["model"].values
    x_pos = np.arange(len(models))

    # Plot 1: F1 Score
    ax = axes[0, 0]
    ax.bar(x_pos, df_best["f1_mean"], yerr=df_best["f1_std"], capsize=5, alpha=0.7)
    ax.set_xlabel("Model")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score Comparison (at Best Contamination)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 1])

    # Plot 2: Precision
    ax = axes[0, 1]
    ax.bar(
        x_pos,
        df_best["precision_mean"],
        yerr=df_best["precision_std"],
        capsize=5,
        alpha=0.7,
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Precision")
    ax.set_title("Precision Comparison")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 1])

    # Plot 3: Recall
    ax = axes[1, 0]
    ax.bar(
        x_pos, df_best["recall_mean"], yerr=df_best["recall_std"], capsize=5, alpha=0.7
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Recall")
    ax.set_title("Recall Comparison")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 1])

    # Plot 4: FPR
    ax = axes[1, 1]
    ax.bar(
        x_pos,
        df_best["fpr_mean"],
        yerr=df_best["fpr_std"],
        capsize=5,
        alpha=0.7,
        color="red",
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("False Positive Rate")
    ax.set_title("FPR Comparison (Lower is Better)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "benchmark_model_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"✓ Model comparison plot saved to: {plot_path}")
    plt.close()


def _print_summary(df: pd.DataFrame, test_contamination_rates: bool):
    """Print summary statistics."""

    print("\n" + "=" * 80)
    print("SUMMARY - ANOMALY DETECTION BENCHMARK")
    print("=" * 80)

    if test_contamination_rates:
        print("\nBest F1 Score for Each Model (across all contamination rates):")
        print("-" * 80)
        print(
            f"{'Model':<18} | Best F1       | Best Cont. | Precision     | Recall        | FPR"
        )
        print("-" * 80)

        for model in df["model"].unique():
            model_data = df[df["model"] == model]
            best_idx = model_data["f1_mean"].idxmax()
            best = model_data.loc[best_idx]

            print(
                f"{model:<18} | "
                f"{best['f1_mean']:.3f} ± {best['f1_std']:.3f} | "
                f"{best['contamination']:.4f}     | "
                f"{best['precision_mean']:.3f} ± {best['precision_std']:.3f} | "
                f"{best['recall_mean']:.3f} ± {best['recall_std']:.3f} | "
                f"{best['fpr_mean']:.3f}"
            )

        # Overall best
        best_overall_idx = df["f1_mean"].idxmax()
        best_overall = df.loc[best_overall_idx]

        print("\n" + "=" * 80)
        print(f" Best Overall Configuration:")
        print(f"   Model: {best_overall['model']}")
        print(f"   Contamination: {best_overall['contamination']:.4f}")
        print(
            f"   F1 Score: {best_overall['f1_mean']:.3f} ± {best_overall['f1_std']:.3f}"
        )
        print(f"   Precision: {best_overall['precision_mean']:.3f}")
        print(f"   Recall: {best_overall['recall_mean']:.3f}")
        print(f"   FPR: {best_overall['fpr_mean']:.3f}")
        print(f"   Training Time: {best_overall['time_mean']:.2f}s per fold")
        print("=" * 80)

    else:
        # Single contamination rate - just sort by F1
        df_sorted = df.sort_values("f1_mean", ascending=False)

        print(
            f"\n{'Model':<18} | F1 Score      | Precision     | Recall        | FPR       | Time (s)"
        )
        print("-" * 80)

        for _, row in df_sorted.iterrows():
            print(
                f"{row['model']:<18} | "
                f"{row['f1_mean']:.3f} ± {row['f1_std']:.3f} | "
                f"{row['precision_mean']:.3f} ± {row['precision_std']:.3f} | "
                f"{row['recall_mean']:.3f} ± {row['recall_std']:.3f} | "
                f"{row['fpr_mean']:.3f}     | "
                f"{row['time_mean']:.2f}"
            )

        best_model = df_sorted.iloc[0]
        print("\n" + "=" * 80)
        print(f"🏆 Best Model: {best_model['model']}")
        print(f"   F1 Score: {best_model['f1_mean']:.3f} ± {best_model['f1_std']:.3f}")
        print(f"   Precision: {best_model['precision_mean']:.3f}")
        print(f"   Recall: {best_model['recall_mean']:.3f}")
        print(f"   FPR: {best_model['fpr_mean']:.3f}")
        print(f"   Training Time: {best_model['time_mean']:.2f}s per fold")
        print("=" * 80)

    # Comparison to supervised benchmark (if available)
    supervised_path = (
        Path("results") / "supervised_benchmark" / "benchmark_results_binary.csv"
    )
    if supervised_path.exists():
        print("\n" + "=" * 80)
        print("COMPARISON: Unsupervised (Anomaly Detection) vs Supervised (Binary)")
        print("=" * 80)

        df_supervised = pd.read_csv(supervised_path)
        best_supervised = df_supervised.loc[df_supervised["f1_mean"].idxmax()]

        best_unsupervised_idx = df["f1_mean"].idxmax()
        best_unsupervised = df.loc[best_unsupervised_idx]

        print(
            f"\nBest Supervised:   {best_supervised['classifier']:<18} F1 = {best_supervised['f1_mean']:.3f}"
        )
        print(
            f"Best Unsupervised: {best_unsupervised['model']:<18} F1 = {best_unsupervised['f1_mean']:.3f}"
        )
        print(
            f"\nGap: {best_supervised['f1_mean'] - best_unsupervised['f1_mean']:.3f} F1 points"
        )

        gap_percentage = (
            (best_supervised["f1_mean"] - best_unsupervised["f1_mean"])
            / best_supervised["f1_mean"]
            * 100
        )
        print(f"Relative gap: {gap_percentage:.1f}% lower than supervised")
        print("=" * 80)
